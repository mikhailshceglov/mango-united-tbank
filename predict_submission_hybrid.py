from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

TEST_CANDIDATES = [
    DATA_DIR / "new_submission_sample (3).csv",
    DATA_DIR / "new_submission_sample.csv",
]

TRAIN_HYBRID_PATH = OUTPUTS_DIR / "train_hybrid_with_transformer_features.csv"
TEST_HYBRID_PATH = OUTPUTS_DIR / "test_hybrid_with_transformer_features.csv"
META_PATH = OUTPUTS_DIR / "hybrid_feature_meta.json"

MODEL_PATH = MODELS_DIR / "catboost_hybrid_room_match.cbm"
SUBMISSION_TEMPLATE_PATH = DATA_DIR / "submission_sample.csv"
SUBMISSION_PATH = OUTPUTS_DIR / "submission_hybrid.csv"

EXPECTED_ROWS_FOR_NEW_TEST = 11000

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Не найдена модель {MODEL_PATH}. Сначала запусти train_catboost_hybrid.py"
    )

if not TEST_HYBRID_PATH.exists():
    raise FileNotFoundError(
        f"Не найден {TEST_HYBRID_PATH}. Сначала запусти make_transformer_features.py"
    )

if not META_PATH.exists():
    raise FileNotFoundError(
        f"Не найден {META_PATH}. Сначала запусти make_transformer_features.py"
    )

log("Ищу исходный test-файл...")
test_path = None
for candidate in TEST_CANDIDATES:
    if candidate.exists():
        test_path = candidate
        break

if test_path is None:
    raise FileNotFoundError(
        "Не найден тестовый файл. Проверял: "
        + ", ".join(str(x) for x in TEST_CANDIDATES)
    )

log(f"Использую test file: {test_path}")
log("Читаю hybrid features...")
test_hybrid_df = pd.read_csv(TEST_HYBRID_PATH)

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
feature_cols_hybrid = meta["feature_cols_hybrid"]
cat_features_hybrid = meta["cat_features_hybrid"]

log(f"Test hybrid shape: {test_hybrid_df.shape}")
log(f"Feature count: {len(feature_cols_hybrid)}")
log(f"Cat features: {cat_features_hybrid}")

missing_features = [col for col in feature_cols_hybrid if col not in test_hybrid_df.columns]
if missing_features:
    raise ValueError(
        "В test_hybrid_with_transformer_features.csv отсутствуют нужные признаки: "
        + ", ".join(missing_features)
    )

log("Загружаю CatBoost hybrid model...")
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

X_test = test_hybrid_df[feature_cols_hybrid].copy()
test_pool = Pool(
    data=X_test,
    cat_features=cat_features_hybrid,
)

log("Считаю предсказания...")
test_proba = model.predict_proba(test_pool)[:, 1]
test_proba = np.nan_to_num(test_proba, nan=0.5, posinf=1.0, neginf=0.0)
test_proba = np.clip(test_proba, 0.0, 1.0)

row_id = None

if "row_id" in test_hybrid_df.columns:
    row_id = test_hybrid_df["row_id"].astype(int)
    log("row_id взят из test_hybrid_with_transformer_features.csv")
else:
    log("row_id не найден в hybrid test, читаю сырой test-файл...")
    raw_test = pd.read_csv(test_path)

    if "Unnamed: 0" in raw_test.columns and "row_id" not in raw_test.columns:
        raw_test = raw_test.rename(columns={"Unnamed: 0": "row_id"})

    if "row_id" in raw_test.columns:
        row_id = raw_test["row_id"].astype(int)
        log("row_id взят из сырого test-файла")
    else:
        log("В сыром test row_id нет, пробую взять из submission_sample.csv...")
        if not SUBMISSION_TEMPLATE_PATH.exists():
            raise FileNotFoundError(
                "В test нет row_id и не найден data/submission_sample.csv"
            )

        submission_template = pd.read_csv(SUBMISSION_TEMPLATE_PATH)

        if "row_id" not in submission_template.columns:
            raise ValueError("В submission_sample.csv нет колонки row_id")

        if len(submission_template) != len(test_hybrid_df):
            raise ValueError(
                "Число строк в submission_sample.csv не совпадает с test. "
                f"template={len(submission_template)}, test={len(test_hybrid_df)}. "
                "Скорее всего, у тебя старый или неправильный тестовый CSV. "
                "Для валидного сабмита нужен новый тестовый файл с row_id."
            )

        row_id = submission_template["row_id"].astype(int)
        log("row_id взят из submission_sample.csv")

if len(row_id) != len(test_hybrid_df):
    raise ValueError(
        f"Длина row_id не совпадает с test: row_id={len(row_id)}, test={len(test_hybrid_df)}"
    )

submission = pd.DataFrame(
    {
        "row_id": row_id,
        "target": test_proba,
    }
)

log(f"submission shape: {submission.shape}")
print(submission.head().to_string(index=False), flush=True)

assert list(submission.columns) == ["row_id", "target"]
assert submission["row_id"].isna().sum() == 0
assert submission["target"].isna().sum() == 0
assert np.isfinite(submission["target"]).all()

if "row_id" in submission.columns and len(submission) == EXPECTED_ROWS_FOR_NEW_TEST:
    log(f"Похоже, используется новый test на {EXPECTED_ROWS_FOR_NEW_TEST} строк")
else:
    log(
        "Внимание: размер submission не равен ожидаемым 11000 строкам. "
        "Такой файл может быть невалиден для актуального leaderboard."
    )

submission.to_csv(SUBMISSION_PATH, index=False)
log(f"Submission saved to: {SUBMISSION_PATH}")
