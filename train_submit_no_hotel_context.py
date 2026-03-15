from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False


# =========================
# Logging
# =========================
def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


# =========================
# Data classes
# =========================
@dataclass
class CVResult:
    config_id: int
    depth: int
    learning_rate: float
    l2_leaf_reg: float
    mean_pr_auc: float
    std_pr_auc: float
    mean_roc_auc: float
    mean_logloss: float
    mean_best_iteration: float
    cv_type: str


# =========================
# Metrics helpers
# =========================
def threshold_info_at_precision_floor(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    precision_floor: float = 0.95,
) -> Dict[str, float | None]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision_cut = precision[:-1]
    recall_cut = recall[:-1]

    mask = precision_cut >= precision_floor
    if not mask.any():
        return {
            "threshold": None,
            "precision": None,
            "recall": None,
        }

    filtered_thresholds = thresholds[mask]
    filtered_precision = precision_cut[mask]
    filtered_recall = recall_cut[mask]
    best_idx = int(np.argmax(filtered_recall))

    return {
        "threshold": float(filtered_thresholds[best_idx]),
        "precision": float(filtered_precision[best_idx]),
        "recall": float(filtered_recall[best_idx]),
    }


# =========================
# Feature selection
# =========================
def is_hotel_aware_feature(col: str) -> bool:
    """
    Полностью убираем hotel context:
    - hotel_id
    - все признаки, начинающиеся с hotel_
    - все признаки со сравнением относительно отеля: *_vs_hotel
    - все признаки, где явно фигурирует hotel в названии

    Смысл: получить страховочную модель, максимально независимую
    от памяти по конкретным hotel_id.
    """
    if col == "hotel_id":
        return True
    if col.startswith("hotel_"):
        return True
    if "_vs_hotel" in col:
        return True
    if col.endswith("_vs_hotel"):
        return True
    # Дополнительная защита на случай новых naming patterns
    if "hotel" in col:
        return True
    return False


def build_no_hotel_context_feature_set(
    feature_cols_hybrid: Sequence[str],
    cat_features_hybrid: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    removed = []
    keep_features = []

    for col in feature_cols_hybrid:
        if is_hotel_aware_feature(col):
            removed.append(col)
        else:
            keep_features.append(col)

    keep_cat_features = [c for c in cat_features_hybrid if not is_hotel_aware_feature(c)]
    return keep_features, keep_cat_features, removed


# =========================
# CV core
# =========================
def make_model_params(
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    iterations: int,
    early_stopping_rounds: int,
    metric_period: int,
    random_seed: int,
) -> Dict:
    params = {
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "random_strength": 0.5,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "auto_class_weights": "Balanced",
        "random_seed": random_seed,
        "early_stopping_rounds": early_stopping_rounds,
        "verbose": False,
        "metric_period": metric_period,
    }

    if torch.cuda.is_available():
        params["task_type"] = "GPU"
        params["devices"] = "0"
    return params


def fit_one_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    params: Dict,
) -> Dict:
    X_train = train_df[list(feature_cols)].copy()
    y_train = train_df["target"].astype(int).to_numpy()

    X_valid = valid_df[list(feature_cols)].copy()
    y_valid = valid_df["target"].astype(int).to_numpy()

    train_pool = Pool(X_train, y_train, cat_features=list(cat_features))
    valid_pool = Pool(X_valid, y_valid, cat_features=list(cat_features))

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

    valid_proba = model.predict_proba(valid_pool)[:, 1]
    best_iteration = int(model.get_best_iteration())
    if best_iteration <= 0:
        best_iteration = int(model.tree_count_)

    return {
        "pr_auc": float(average_precision_score(y_valid, valid_proba)),
        "roc_auc": float(roc_auc_score(y_valid, valid_proba)),
        "logloss": float(log_loss(y_valid, valid_proba, labels=[0, 1])),
        "best_iteration": best_iteration,
        "y_true": y_valid,
        "y_proba": valid_proba,
    }


def evaluate_config_cv(
    df: pd.DataFrame,
    groups: pd.Series,
    feature_cols: Sequence[str],
    cat_features: Sequence[str],
    depth: int,
    learning_rate: float,
    l2_leaf_reg: float,
    iterations: int,
    early_stopping_rounds: int,
    metric_period: int,
    n_splits: int,
    random_seed: int,
    grouped: bool,
) -> Tuple[CVResult, pd.DataFrame]:
    params = make_model_params(
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        iterations=iterations,
        early_stopping_rounds=early_stopping_rounds,
        metric_period=metric_period,
        random_seed=random_seed,
    )

    y = df["target"].astype(int).to_numpy()

    if grouped:
        if not HAS_STRATIFIED_GROUP_KFOLD:
            raise ImportError(
                "В этой версии sklearn нет StratifiedGroupKFold. Обнови sklearn или запусти без grouped CV."
            )
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df, y, groups=groups)
        cv_type = "stratified_group_kfold"
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df, y)
        cv_type = "stratified_kfold"

    fold_rows = []
    pr_aucs, roc_aucs, loglosses, best_iters = [], [], [], []

    for fold_id, (train_idx, valid_idx) in enumerate(split_iter, start=1):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        valid_fold = df.iloc[valid_idx].reset_index(drop=True)

        fold_out = fit_one_fold(
            train_df=train_fold,
            valid_df=valid_fold,
            feature_cols=feature_cols,
            cat_features=cat_features,
            params=params,
        )

        pr_aucs.append(fold_out["pr_auc"])
        roc_aucs.append(fold_out["roc_auc"])
        loglosses.append(fold_out["logloss"])
        best_iters.append(fold_out["best_iteration"])

        threshold_info = threshold_info_at_precision_floor(
            fold_out["y_true"],
            fold_out["y_proba"],
            precision_floor=0.95,
        )

        fold_rows.append({
            "fold": fold_id,
            "depth": depth,
            "learning_rate": learning_rate,
            "l2_leaf_reg": l2_leaf_reg,
            "cv_type": cv_type,
            "pr_auc": fold_out["pr_auc"],
            "roc_auc": fold_out["roc_auc"],
            "logloss": fold_out["logloss"],
            "best_iteration": fold_out["best_iteration"],
            "threshold": threshold_info["threshold"],
            "precision_at_floor": threshold_info["precision"],
            "recall_at_floor": threshold_info["recall"],
        })

    result = CVResult(
        config_id=-1,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        mean_pr_auc=float(np.mean(pr_aucs)),
        std_pr_auc=float(np.std(pr_aucs)),
        mean_roc_auc=float(np.mean(roc_aucs)),
        mean_logloss=float(np.mean(loglosses)),
        mean_best_iteration=float(np.mean(best_iters)),
        cv_type=cv_type,
    )

    return result, pd.DataFrame(fold_rows)


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid search + grouped CV + финальный сабмит для no_hotel_context страховочной модели"
    )
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--n_splits", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--early_stopping_rounds", type=int, default=200)
    parser.add_argument("--metric_period", type=int, default=100)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--depths", nargs="+", type=int, default=[8, 10, 12])
    parser.add_argument("--learning_rates", nargs="+", type=float, default=[0.03, 0.05])
    parser.add_argument("--l2_leaf_regs", nargs="+", type=float, default=[3.0, 10.0, 20.0])
    args = parser.parse_args()

    project_root = Path(args.project_root)
    data_dir = project_root / "data"
    outputs_dir = project_root / "outputs"
    models_dir = project_root / "models"
    out_dir = outputs_dir / "no_hotel_context_grid"

    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_hybrid_path = outputs_dir / "train_hybrid_with_transformer_features.csv"
    test_hybrid_path = outputs_dir / "test_hybrid_with_transformer_features.csv"
    meta_path = outputs_dir / "hybrid_feature_meta.json"

    if not train_hybrid_path.exists():
        raise FileNotFoundError(f"Не найден {train_hybrid_path}. Сначала запусти make_transformer_features.py")
    if not test_hybrid_path.exists():
        raise FileNotFoundError(f"Не найден {test_hybrid_path}. Сначала запусти make_transformer_features.py")
    if not meta_path.exists():
        raise FileNotFoundError(f"Не найден {meta_path}. Сначала запусти make_transformer_features.py")

    log("Читаю hybrid features...")
    train_df = pd.read_csv(train_hybrid_path)
    test_df = pd.read_csv(test_hybrid_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_cols_hybrid = meta["feature_cols_hybrid"]
    cat_features_hybrid = meta["cat_features_hybrid"]

    if "hotel_id" not in train_df.columns:
        raise ValueError("В train_hybrid_with_transformer_features.csv нет hotel_id. Нельзя сделать grouped CV.")

    feature_cols, cat_features, removed = build_no_hotel_context_feature_set(
        feature_cols_hybrid=feature_cols_hybrid,
        cat_features_hybrid=cat_features_hybrid,
    )

    log(f"Train shape: {train_df.shape}")
    log(f"Test shape: {test_df.shape}")
    log(f"Original feature count: {len(feature_cols_hybrid)}")
    log(f"no_hotel_context feature count: {len(feature_cols)}")
    log(f"Removed hotel-related features ({len(removed)}): {removed}")
    log(f"Cat features: {cat_features}")
    log(f"GPU available: {torch.cuda.is_available()}")

    groups = train_df["hotel_id"].astype(str)

    grid = list(itertools.product(args.depths, args.learning_rates, args.l2_leaf_regs))
    log(f"Grid size: {len(grid)} configs")

    # Stage 1: fast random CV on no_hotel_context
    stage1_rows = []
    for config_id, (depth, lr, l2) in enumerate(grid, start=1):
        log(f"[Stage 1 / {config_id}/{len(grid)}] depth={depth}, lr={lr}, l2={l2}")
        res, _ = evaluate_config_cv(
            df=train_df,
            groups=groups,
            feature_cols=feature_cols,
            cat_features=cat_features,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2,
            iterations=args.iterations,
            early_stopping_rounds=args.early_stopping_rounds,
            metric_period=args.metric_period,
            n_splits=args.n_splits,
            random_seed=args.random_seed,
            grouped=False,
        )
        row = asdict(res)
        row["config_id"] = config_id
        stage1_rows.append(row)

    stage1_df = pd.DataFrame(stage1_rows).sort_values(
        ["mean_pr_auc", "mean_roc_auc"], ascending=[False, False]
    )
    stage1_path = out_dir / "stage1_no_hotel_context_random_cv_grid.csv"
    stage1_df.to_csv(stage1_path, index=False)
    log(f"Stage 1 saved to: {stage1_path}")

    top_df = stage1_df.head(args.top_k).copy()
    log("Top configs after Stage 1:")
    print(top_df[["depth", "learning_rate", "l2_leaf_reg", "mean_pr_auc", "mean_best_iteration"]].to_string(index=False), flush=True)

    # Stage 2: grouped CV on top configs
    stage2_summary_rows = []
    stage2_fold_rows = []
    for _, row in top_df.iterrows():
        depth = int(row["depth"])
        lr = float(row["learning_rate"])
        l2 = float(row["l2_leaf_reg"])
        config_id = int(row["config_id"])

        log(f"[Stage 2] grouped CV for config_id={config_id}: depth={depth}, lr={lr}, l2={l2}")
        res_group, folds_group = evaluate_config_cv(
            df=train_df,
            groups=groups,
            feature_cols=feature_cols,
            cat_features=cat_features,
            depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2,
            iterations=args.iterations,
            early_stopping_rounds=args.early_stopping_rounds,
            metric_period=args.metric_period,
            n_splits=args.n_splits,
            random_seed=args.random_seed,
            grouped=True,
        )

        random_row = stage1_df.loc[stage1_df["config_id"] == config_id].iloc[0]
        summary_row = {
            "config_id": config_id,
            "depth": depth,
            "learning_rate": lr,
            "l2_leaf_reg": l2,
            "random_cv_pr_auc": float(random_row["mean_pr_auc"]),
            "grouped_cv_pr_auc": float(res_group.mean_pr_auc),
            "random_cv_roc_auc": float(random_row["mean_roc_auc"]),
            "grouped_cv_roc_auc": float(res_group.mean_roc_auc),
            "random_cv_logloss": float(random_row["mean_logloss"]),
            "grouped_cv_logloss": float(res_group.mean_logloss),
            "gap_random_minus_grouped": float(random_row["mean_pr_auc"] - res_group.mean_pr_auc),
            "grouped_mean_best_iteration": float(res_group.mean_best_iteration),
        }
        stage2_summary_rows.append(summary_row)

        folds_group.insert(0, "config_id", config_id)
        stage2_fold_rows.append(folds_group)

    stage2_summary_df = pd.DataFrame(stage2_summary_rows).sort_values(
        ["grouped_cv_pr_auc", "gap_random_minus_grouped"], ascending=[False, True]
    )
    stage2_folds_df = pd.concat(stage2_fold_rows, ignore_index=True) if stage2_fold_rows else pd.DataFrame()

    stage2_summary_path = out_dir / "stage2_no_hotel_context_grouped_cv_summary.csv"
    stage2_folds_path = out_dir / "stage2_no_hotel_context_grouped_cv_folds.csv"
    stage2_summary_df.to_csv(stage2_summary_path, index=False)
    stage2_folds_df.to_csv(stage2_folds_path, index=False)

    log(f"Stage 2 summary saved to: {stage2_summary_path}")
    log(f"Stage 2 folds saved to: {stage2_folds_path}")

    best = stage2_summary_df.iloc[0]
    best_config = {
        "depth": int(best["depth"]),
        "learning_rate": float(best["learning_rate"]),
        "l2_leaf_reg": float(best["l2_leaf_reg"]),
        "random_strength": 0.5,
        "random_cv_pr_auc": float(best["random_cv_pr_auc"]),
        "grouped_cv_pr_auc": float(best["grouped_cv_pr_auc"]),
        "gap_random_minus_grouped": float(best["gap_random_minus_grouped"]),
        "grouped_mean_best_iteration": int(round(best["grouped_mean_best_iteration"])),
    }

    best_config_path = out_dir / "best_no_hotel_context_config.json"
    best_config_path.write_text(json.dumps(best_config, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Best config saved to: {best_config_path}")
    log(f"Chosen config: {best_config}")

    # Final fit on full train with chosen grouped-CV config
    final_iterations = max(300, int(best_config["grouped_mean_best_iteration"]))
    final_params = make_model_params(
        depth=best_config["depth"],
        learning_rate=best_config["learning_rate"],
        l2_leaf_reg=best_config["l2_leaf_reg"],
        iterations=final_iterations,
        early_stopping_rounds=args.early_stopping_rounds,
        metric_period=args.metric_period,
        random_seed=args.random_seed,
    )
    final_params.pop("early_stopping_rounds", None)
    final_params["verbose"] = 200
    final_params["eval_metric"] = "AUC"

    log("Обучаю финальную no_hotel_context модель на всём train...")
    full_pool = Pool(
        train_df[feature_cols].copy(),
        train_df["target"].astype(int).to_numpy(),
        cat_features=cat_features,
    )

    final_model = CatBoostClassifier(**final_params)
    final_model.fit(full_pool)

    model_path = models_dir / "catboost_hybrid_no_hotel_context.cbm"
    final_model.save_model(model_path)
    log(f"Model saved to: {model_path}")

    # Submission
    missing_test_cols = [c for c in feature_cols if c not in test_df.columns]
    if missing_test_cols:
        raise ValueError("В test_hybrid_with_transformer_features.csv отсутствуют нужные признаки: " + ", ".join(missing_test_cols))

    test_pool = Pool(test_df[feature_cols].copy(), cat_features=cat_features)
    test_proba = final_model.predict_proba(test_pool)[:, 1]
    test_proba = np.nan_to_num(test_proba, nan=0.5, posinf=1.0, neginf=0.0)
    test_proba = np.clip(test_proba, 0.0, 1.0)

    if "row_id" in test_df.columns:
        row_id = test_df["row_id"].astype(int)
    else:
        test_candidates = [
            data_dir / "new_submission_sample (3).csv",
            data_dir / "new_submission_sample.csv",
            data_dir / "submission_sample.csv",
        ]
        raw_test = None
        for candidate in test_candidates:
            if candidate.exists():
                raw_test = pd.read_csv(candidate)
                break
        if raw_test is None:
            raise FileNotFoundError("Не удалось найти test/template файл с row_id")
        if "Unnamed: 0" in raw_test.columns and "row_id" not in raw_test.columns:
            raw_test = raw_test.rename(columns={"Unnamed: 0": "row_id"})
        if "row_id" not in raw_test.columns:
            raise ValueError("В найденном test/template файле нет row_id")
        if len(raw_test) != len(test_df):
            raise ValueError(f"row_id length mismatch: raw={len(raw_test)}, test={len(test_df)}")
        row_id = raw_test["row_id"].astype(int)

    submission = pd.DataFrame({
        "row_id": row_id,
        "target": test_proba,
    })

    assert list(submission.columns) == ["row_id", "target"]
    assert submission["row_id"].isna().sum() == 0
    assert submission["target"].isna().sum() == 0
    assert np.isfinite(submission["target"]).all()

    submission_path = outputs_dir / "submission_no_hotel_context.csv"
    submission.to_csv(submission_path, index=False)
    log(f"Submission saved to: {submission_path}")

    run_meta = {
        "mode": "no_hotel_context",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "removed_features": removed,
        "kept_feature_count": int(len(feature_cols)),
        "kept_cat_features": cat_features,
        "best_config": best_config,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    run_meta_path = out_dir / "run_meta_no_hotel_context.json"
    run_meta_path.write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Run meta saved to: {run_meta_path}")

    log("Готово.")


if __name__ == "__main__":
    main()
