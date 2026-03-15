from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_project_paths(project_root: Path) -> Dict[str, Path]:
    data_dir = project_root / 'data'
    outputs_dir = project_root / 'outputs'
    models_dir = project_root / 'models'
    return {
        'data_dir': data_dir,
        'outputs_dir': outputs_dir,
        'models_dir': models_dir,
        'train_hybrid': outputs_dir / 'train_hybrid_with_transformer_features.csv',
        'meta': outputs_dir / 'hybrid_feature_meta.json',
        'no_hotel_context_cfg': outputs_dir / 'no_hotel_context_grid' / 'best_no_hotel_context_config.json',
        'no_hotel_id_cfg': outputs_dir / 'no_hotel_id_grid' / 'best_no_hotel_id_config.json',
        'full_model': models_dir / 'catboost_hybrid_room_match.cbm',
        'no_hotel_context_model': models_dir / 'catboost_hybrid_no_hotel_context.cbm',
        'no_hotel_id_model': models_dir / 'catboost_hybrid_no_hotel_id.cbm',
    }


def load_meta(meta_path: Path) -> Tuple[List[str], List[str]]:
    meta = read_json(meta_path)
    return meta['feature_cols_hybrid'], meta['cat_features_hybrid']


def remove_hotel_context(features: List[str]) -> List[str]:
    kept = []
    for col in features:
        c = col.lower()
        if c == 'hotel_id':
            continue
        if c.startswith('hotel_'):
            continue
        if '_vs_hotel' in c:
            continue
        kept.append(col)
    return kept


def remove_only_hotel_id(features: List[str]) -> List[str]:
    return [c for c in features if c.lower() != 'hotel_id']


def derive_variant_features(variant: str, feature_cols_hybrid: List[str], cat_features_hybrid: List[str]) -> Tuple[List[str], List[str], List[str]]:
    if variant == 'full':
        feat = list(feature_cols_hybrid)
    elif variant == 'no_hotel_id':
        feat = remove_only_hotel_id(feature_cols_hybrid)
    elif variant == 'no_hotel_context':
        feat = remove_hotel_context(feature_cols_hybrid)
    else:
        raise ValueError(f'Unknown variant: {variant}')

    removed = [c for c in feature_cols_hybrid if c not in feat]
    cat = [c for c in cat_features_hybrid if c in feat]
    return feat, cat, removed


def normalize_object_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == 'object':
            out[col] = out[col].fillna('unknown').astype(str)
    return out


def load_catboost_params_for_full(paths: Dict[str, Path], random_seed: int) -> Dict:
    # Fallback to project defaults from train_catboost_hybrid.py
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'PRAUC',
        'iterations': 2500,
        'learning_rate': 0.1,
        'depth': 13,
        'l2_leaf_reg': 10,
        'random_strength': 1.0,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'auto_class_weights': 'Balanced',
        'random_seed': random_seed,
        'early_stopping_rounds': 100,
        'verbose': 200,
    }

    # If a saved model exists, try to reuse its params (best effort)
    model_path = paths['full_model']
    if model_path.exists():
        try:
            model = CatBoostClassifier()
            model.load_model(str(model_path))
            all_params = model.get_all_params()
            # Keep only parameters accepted for constructor and useful here
            for key in ['loss_function', 'eval_metric', 'iterations', 'learning_rate', 'depth', 'l2_leaf_reg',
                        'random_strength', 'bootstrap_type', 'subsample', 'auto_class_weights']:
                if key in all_params:
                    params[key] = all_params[key]
        except Exception as e:
            log(f'Не удалось прочитать параметры full model из cbm, использую project defaults: {e}')

    return params


def load_catboost_params_for_variant(paths: Dict[str, Path], variant: str, random_seed: int) -> Dict:
    if variant == 'full':
        return load_catboost_params_for_full(paths, random_seed=random_seed)

    # Reasonable fallbacks aligned with previous scripts
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'PRAUC',
        'iterations': 5000,
        'learning_rate': 0.03,
        'depth': 10,
        'l2_leaf_reg': 20,
        'random_strength': 0.5,
        'bootstrap_type': 'Bernoulli',
        'subsample': 0.8,
        'auto_class_weights': 'Balanced',
        'random_seed': random_seed,
        'early_stopping_rounds': 200,
        'verbose': 200,
        'metric_period': 100,
    }

    cfg_path = paths['no_hotel_context_cfg'] if variant == 'no_hotel_context' else paths['no_hotel_id_cfg']
    if cfg_path.exists():
        try:
            best = read_json(cfg_path)
            for key in ['depth', 'learning_rate', 'l2_leaf_reg', 'random_strength']:
                if key in best:
                    params[key] = best[key]
        except Exception as e:
            log(f'Не удалось прочитать конфиг {cfg_path}, использую fallback: {e}')

    return params


def fit_predict_variant(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    cat_features: List[str],
    params: Dict,
) -> Tuple[CatBoostClassifier, np.ndarray]:
    X_train = normalize_object_cols(train_df[feature_cols])
    y_train = train_df['target'].astype(int).values
    X_valid = normalize_object_cols(valid_df[feature_cols])

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, valid_df['target'].astype(int).values, cat_features=cat_features)

    if torch.cuda.is_available():
        params = params.copy()
        params['task_type'] = 'GPU'
        params['devices'] = '0'
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    proba = model.predict_proba(valid_pool)[:, 1]
    return model, proba


def threshold_at_precision_floor(y_true: np.ndarray, y_proba: np.ndarray, precision_floor: float = 0.95) -> Dict:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision_cut = precision[:-1]
    recall_cut = recall[:-1]
    info = {'threshold': None, 'precision': None, 'recall': None}
    mask = precision_cut >= precision_floor
    if mask.any():
        idx = np.argmax(recall_cut[mask])
        info = {
            'threshold': float(thresholds[mask][idx]),
            'precision': float(precision_cut[mask][idx]),
            'recall': float(recall_cut[mask][idx]),
        }
    return info


def evaluate_predictions(y_true: np.ndarray, y_proba: np.ndarray, precision_floor: float = 0.95) -> Dict:
    return {
        'pr_auc': float(average_precision_score(y_true, y_proba)),
        'roc_auc': float(roc_auc_score(y_true, y_proba)),
        'logloss': float(log_loss(y_true, y_proba, labels=[0, 1])),
        'threshold_info': threshold_at_precision_floor(y_true, y_proba, precision_floor),
    }


def plot_pr_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray, float, Dict]], out_path: Path) -> None:
    plt.figure(figsize=(8.5, 6.5))
    for label, (precision, recall, pr_auc, thr_info) in curves.items():
        plt.plot(recall, precision, linewidth=2, label=f'{label} | PR-AUC={pr_auc:.5f}')
        if thr_info.get('threshold') is not None:
            plt.scatter([thr_info['recall']], [thr_info['precision']], s=70)
            plt.annotate(
                f"{label}: R@P95={thr_info['recall']:.3f}",
                (thr_info['recall'], thr_info['precision']),
                textcoords='offset points',
                xytext=(6, -12),
                fontsize=9,
            )
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR-кривые на одном random 20% holdout публичных данных')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare PR-curves on the same random 20% holdout for full vs no-hotel variant.')
    parser.add_argument('--project_root', type=str, default='.')
    parser.add_argument('--variant_b', type=str, default='no_hotel_context', choices=['no_hotel_id', 'no_hotel_context'])
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--precision_floor', type=float, default=0.95)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    paths = get_project_paths(project_root)
    out_dir = paths['outputs_dir'] / 'holdout_pr_compare'
    ensure_dir(out_dir)

    if not paths['train_hybrid'].exists():
        raise FileNotFoundError(f'Не найден {paths["train_hybrid"]}')
    if not paths['meta'].exists():
        raise FileNotFoundError(f'Не найден {paths["meta"]}')

    log('Читаю hybrid train...')
    train_hybrid_df = pd.read_csv(paths['train_hybrid'], low_memory=False)
    feature_cols_hybrid, cat_features_hybrid = load_meta(paths['meta'])

    if 'target' not in train_hybrid_df.columns:
        raise ValueError('В train_hybrid_with_transformer_features.csv нет колонки target')

    log(f'Train hybrid shape: {train_hybrid_df.shape}')

    train_df, valid_df = train_test_split(
        train_hybrid_df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=train_hybrid_df['target'].astype(int),
    )
    log(f'Holdout split: train={train_df.shape}, valid={valid_df.shape}')

    variants = ['full', args.variant_b]
    all_metrics = {}
    curves = {}
    prediction_df = valid_df[['target']].copy().reset_index(drop=True)

    for variant in variants:
        log(f'Готовлю variant={variant}')
        feat_cols, cat_cols, removed = derive_variant_features(variant, feature_cols_hybrid, cat_features_hybrid)
        log(f'{variant}: features={len(feat_cols)}, removed={removed}')
        params = load_catboost_params_for_variant(paths, variant, random_seed=args.random_seed)
        log(f'{variant}: params={params}')
        model, valid_proba = fit_predict_variant(train_df, valid_df, feat_cols, cat_cols, params)

        y_true = valid_df['target'].astype(int).values
        metrics = evaluate_predictions(y_true, valid_proba, precision_floor=args.precision_floor)
        metrics['variant'] = variant
        metrics['n_features'] = len(feat_cols)
        metrics['removed_features'] = removed
        all_metrics[variant] = metrics

        precision, recall, _ = precision_recall_curve(y_true, valid_proba)
        curves[variant] = (precision, recall, metrics['pr_auc'], metrics['threshold_info'])
        prediction_df[f'proba_{variant}'] = valid_proba

        model_out = out_dir / f'holdout_model_{variant}.cbm'
        model.save_model(str(model_out))
        log(f'{variant}: model saved to {model_out}')
        log(f"{variant}: PR-AUC={metrics['pr_auc']:.6f}; R@P95={metrics['threshold_info']['recall']}")

    png_path = out_dir / f'pr_curve_full_vs_{args.variant_b}.png'
    plot_pr_curves(curves, png_path)

    metrics_out = out_dir / f'holdout_metrics_full_vs_{args.variant_b}.json'
    with open(metrics_out, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)

    pred_out = out_dir / f'holdout_predictions_full_vs_{args.variant_b}.csv'
    prediction_df.to_csv(pred_out, index=False)

    summary_rows = []
    for variant, metrics in all_metrics.items():
        summary_rows.append({
            'variant': variant,
            'pr_auc': metrics['pr_auc'],
            'roc_auc': metrics['roc_auc'],
            'logloss': metrics['logloss'],
            'threshold_at_precision_floor': metrics['threshold_info']['threshold'],
            'precision_at_threshold': metrics['threshold_info']['precision'],
            'recall_at_precision_floor': metrics['threshold_info']['recall'],
            'n_features': metrics['n_features'],
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / f'holdout_metrics_full_vs_{args.variant_b}.csv', index=False)

    log(f'PR-кривая сохранена: {png_path}')
    log(f'Метрики сохранены: {metrics_out}')
    log(f'Предсказания сохранены: {pred_out}')
    log('Готово.')


if __name__ == '__main__':
    main()
