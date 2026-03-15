from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

TRAIN_HYBRID_PATH = OUTPUTS_DIR / "train_hybrid_with_transformer_features.csv"
TEST_HYBRID_PATH = OUTPUTS_DIR / "test_hybrid_with_transformer_features.csv"
META_PATH = OUTPUTS_DIR / "hybrid_feature_meta.json"

MODEL_PATH = MODELS_DIR / "catboost_hybrid_room_match.cbm"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance_hybrid.csv"
VALID_PREDICTIONS_PATH = OUTPUTS_DIR / "valid_predictions_hybrid.csv"
METRICS_PATH = OUTPUTS_DIR / "hybrid_metrics.json"
SUBMISSION_PATH = OUTPUTS_DIR / "submission_hybrid.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.2
PRECISION_FLOOR = 0.95

if not TRAIN_HYBRID_PATH.exists():
    raise FileNotFoundError(
        f"Не найден {TRAIN_HYBRID_PATH}. Сначала запусти make_transformer_features.py"
    )

if not TEST_HYBRID_PATH.exists():
    raise FileNotFoundError(
        f"Не найден {TEST_HYBRID_PATH}. Сначала запусти make_transformer_features.py"
    )

if not META_PATH.exists():
    raise FileNotFoundError(
        f"Не найден {META_PATH}. Сначала запусти make_transformer_features.py"
    )

log("Читаю hybrid train/test...")
train_hybrid_df = pd.read_csv(TRAIN_HYBRID_PATH)
test_hybrid_df = pd.read_csv(TEST_HYBRID_PATH)
meta = json.loads(META_PATH.read_text(encoding="utf-8"))

feature_cols_hybrid = meta["feature_cols_hybrid"]
cat_features_hybrid = meta["cat_features_hybrid"]

log(f"Train hybrid shape: {train_hybrid_df.shape}")
log(f"Test hybrid shape: {test_hybrid_df.shape}")
log(f"Feature count: {len(feature_cols_hybrid)}")
log(f"Cat features: {cat_features_hybrid}")

X = train_hybrid_df[feature_cols_hybrid].copy()
y = train_hybrid_df["target"].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y,
)

log(f"Train split shape: {X_train.shape}")
log(f"Valid split shape: {X_valid.shape}")

train_pool = Pool(
    data=X_train,
    label=y_train,
    cat_features=cat_features_hybrid,
)

valid_pool = Pool(
    data=X_valid,
    label=y_valid,
    cat_features=cat_features_hybrid,
)

catboost_params = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 14,
    "l2_leaf_reg": 3, # 10
    "random_strength": 0.1, # 1
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "auto_class_weights": "Balanced",
    "random_seed": RANDOM_SEED,
    "early_stopping_rounds": 100,
    "verbose": 200,
}

if torch.cuda.is_available():
    catboost_params["task_type"] = "GPU"
    catboost_params["devices"] = "0"
    log("CatBoost будет учиться на GPU")
else:
    log("CatBoost будет учиться на CPU")

log("Запускаю обучение CatBoost...")
model = CatBoostClassifier(**catboost_params)

model.fit(
    train_pool,
    eval_set=valid_pool,
    use_best_model=True,
)

log("Считаю валидационные предсказания...")
valid_proba = model.predict_proba(valid_pool)[:, 1]
pr_auc = average_precision_score(y_valid, valid_proba)

log(f"Validation PR-AUC: {pr_auc:.6f}")

precision, recall, thresholds = precision_recall_curve(y_valid, valid_proba)
precision_cut = precision[:-1]
recall_cut = recall[:-1]

mask = precision_cut >= PRECISION_FLOOR

threshold_info = {
    "threshold": None,
    "precision": None,
    "recall": None,
}

if mask.any():
    valid_thresholds = thresholds[mask]
    valid_precisions = precision_cut[mask]
    valid_recalls = recall_cut[mask]

    best_idx = np.argmax(valid_recalls)

    threshold_info = {
        "threshold": float(valid_thresholds[best_idx]),
        "precision": float(valid_precisions[best_idx]),
        "recall": float(valid_recalls[best_idx]),
    }

    log(f"Best threshold with precision >= {PRECISION_FLOOR:.2f}: {threshold_info['threshold']:.6f}")
    log(f"Precision: {threshold_info['precision']:.6f}")
    log(f"Recall: {threshold_info['recall']:.6f}")
else:
    log(f"На валидации не найден порог с precision >= {PRECISION_FLOOR:.2f}")

log("Считаю feature importance...")
feature_importance = pd.DataFrame(
    {
        "feature": feature_cols_hybrid,
        "importance": model.get_feature_importance(train_pool),
    }
).sort_values("importance", ascending=False)

log("Top-20 feature importance:")
print(feature_importance.head(20).to_string(index=False), flush=True)

feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

valid_predictions = X_valid.copy()
valid_predictions["target"] = y_valid.to_numpy()
valid_predictions["prediction_proba"] = valid_proba
valid_predictions.to_csv(VALID_PREDICTIONS_PATH, index=False)

best_iteration = int(model.get_best_iteration())
if best_iteration <= 0:
    best_iteration = int(model.tree_count_)

log(f"Best iteration: {best_iteration}")

production_params = catboost_params.copy()
production_params["iterations"] = best_iteration
production_params.pop("early_stopping_rounds", None)

full_pool = Pool(
    data=X,
    label=y,
    cat_features=cat_features_hybrid,
)

log("Переобучаю production-модель на всём train...")
production_model = CatBoostClassifier(**production_params)
production_model.fit(full_pool)
production_model.save_model(MODEL_PATH)

log("Делаю предсказания на test...")
X_test = test_hybrid_df[feature_cols_hybrid].copy()
test_pool = Pool(
    data=X_test,
    cat_features=cat_features_hybrid,
)

test_proba = production_model.predict_proba(test_pool)[:, 1]
test_proba = np.nan_to_num(test_proba, nan=0.5, posinf=1.0, neginf=0.0)
test_proba = np.clip(test_proba, 0.0, 1.0)

if "row_id" in test_hybrid_df.columns:
    row_id = test_hybrid_df["row_id"].astype(int)
else:
    submission_template_path = DATA_DIR / "submission_sample.csv"
    submission_template = pd.read_csv(submission_template_path)

    if "row_id" not in submission_template.columns:
        raise ValueError("В submission_sample.csv нет колонки row_id")

    if len(submission_template) != len(test_hybrid_df):
        raise ValueError(
            "Число строк в submission_sample.csv не совпадает с test. "
            f"template={len(submission_template)}, test={len(test_hybrid_df)}"
        )

    row_id = submission_template["row_id"].astype(int)

submission = pd.DataFrame(
    {
        "row_id": row_id,
        "target": test_proba,
    }
)

assert list(submission.columns) == ["row_id", "target"]
assert submission["row_id"].isna().sum() == 0
assert submission["target"].isna().sum() == 0
assert np.isfinite(submission["target"]).all()

submission.to_csv(SUBMISSION_PATH, index=False)

metrics = {
    "transformer_model_name": meta["model_name"],
    "transformer_oof_pr_auc": meta["oof_pr_auc"],
    "catboost_validation_pr_auc": float(pr_auc),
    "best_iteration": best_iteration,
    "precision_floor": PRECISION_FLOOR,
    "threshold_info": threshold_info,
    "feature_cols_hybrid": feature_cols_hybrid,
    "cat_features_hybrid": cat_features_hybrid,
}

METRICS_PATH.write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

log(f"Production model saved to: {MODEL_PATH}")
log(f"Feature importance saved to: {FEATURE_IMPORTANCE_PATH}")
log(f"Validation predictions saved to: {VALID_PREDICTIONS_PATH}")
log(f"Metrics saved to: {METRICS_PATH}")
log(f"Submission saved to: {SUBMISSION_PATH}")
log("Preview submission:")
print(submission.head().to_string(index=False), flush=True)
