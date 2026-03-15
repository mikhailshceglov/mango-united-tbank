from pathlib import Path
from datetime import datetime
import gc
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from room_feature_engineering import (
    build_feature_dataset,
    get_categorical_features,
    get_feature_columns,
)


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


class EncodedDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            item["labels"] = int(self.labels[idx])
        return item


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def predict_scores(model, loader, device, use_amp, amp_dtype):
    model.eval()

    probs_all = []
    logits_all = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        raw_logit = outputs.logits[:, 1] - outputs.logits[:, 0]

        probs_all.append(probs.detach().float().cpu().numpy())
        logits_all.append(raw_logit.detach().float().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    logits_all = np.concatenate(logits_all)
    return probs_all, logits_all


@torch.no_grad()
def encode_embeddings(model, loader, device, use_amp, amp_dtype):
    model.eval()

    all_embeddings = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.autocast(
            device_type="cuda",
            dtype=amp_dtype,
            enabled=use_amp,
        ):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        pooled = mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled = F.normalize(pooled, p=2, dim=1)

        all_embeddings.append(pooled.detach().float().cpu().numpy().astype(np.float32))

    return np.concatenate(all_embeddings, axis=0)


PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "public_dataset.csv"
TEST_CANDIDATES = [
    DATA_DIR / "new_submission_sample (3).csv",
    DATA_DIR / "new_submission_sample.csv",
]

TRAIN_HYBRID_PATH = OUTPUTS_DIR / "train_hybrid_with_transformer_features.csv"
TEST_HYBRID_PATH = OUTPUTS_DIR / "test_hybrid_with_transformer_features.csv"
META_PATH = OUTPUTS_DIR / "hybrid_feature_meta.json"

MODEL_NAME = "cointegrated/rubert-tiny2"
EMBEDDING_MODEL_NAME = MODEL_NAME

N_SPLITS = 5
MAX_LEN = 48
EPOCHS = 3

TRAIN_BATCH_SIZE = 256
VALID_BATCH_SIZE = 512
EMBED_BATCH_SIZE = 512

LR = 1.5e-5
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

EMBEDDING_PCA_DIM = 64

WORKERS = min(4, os.cpu_count() or 1)
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = WORKERS > 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

use_amp = torch.cuda.is_available()
use_bf16 = False
if torch.cuda.is_available():
    major, _ = torch.cuda.get_device_capability(0)
    use_bf16 = major >= 8

amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
scaler_enabled = torch.cuda.is_available() and not use_bf16

log(f"Device: {device}")
log(f"Classifier model: {MODEL_NAME}")
log(f"Frozen embedding model: {EMBEDDING_MODEL_NAME}")
log(f"AMP enabled: {use_amp}, dtype: {amp_dtype}")
log(f"TRAIN_BATCH_SIZE={TRAIN_BATCH_SIZE}, VALID_BATCH_SIZE={VALID_BATCH_SIZE}, EMBED_BATCH_SIZE={EMBED_BATCH_SIZE}")
log(f"PCA components for frozen embeddings: {EMBEDDING_PCA_DIM}")

log("Поиск тестового файла...")
test_path = None
for candidate in TEST_CANDIDATES:
    if candidate.exists():
        test_path = candidate
        break

if test_path is None:
    raise FileNotFoundError(
        f"Не найден тестовый файл. Проверял: {[str(x) for x in TEST_CANDIDATES]}"
    )

log(f"Train path: {TRAIN_PATH}")
log(f"Test path: {test_path}")

train_raw = pd.read_csv(TRAIN_PATH)
test_raw = pd.read_csv(test_path)

if "Unnamed: 0" in test_raw.columns and "row_id" not in test_raw.columns:
    log("Найдена колонка Unnamed: 0, переименовываю в row_id")
    test_raw = test_raw.rename(columns={"Unnamed: 0": "row_id"})

train_raw["supplier_room_name"] = train_raw["supplier_room_name"].fillna("").astype(str)
test_raw["supplier_room_name"] = test_raw["supplier_room_name"].fillna("").astype(str)

log("Строю базовые ручные признаки...")
train_hybrid_df = build_feature_dataset(train_raw)
test_hybrid_df = build_feature_dataset(test_raw)

base_feature_cols = get_feature_columns()
base_cat_features = get_categorical_features()

texts = train_hybrid_df["supplier_room_name"].tolist()
y = train_hybrid_df["target"].astype(int).to_numpy()
test_texts = test_hybrid_df["supplier_room_name"].tolist()

log(f"Train rows: {len(train_hybrid_df)}")
log(f"Test rows: {len(test_hybrid_df)}")
log(f"Base feature count: {len(base_feature_cols)}")

log("Загружаю tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# =========================================================
# 1. OOF transformer features без утечки
# =========================================================
oof_proba = np.zeros(len(train_hybrid_df), dtype=np.float32)
oof_logit = np.zeros(len(train_hybrid_df), dtype=np.float32)

test_proba_mean = np.zeros(len(test_hybrid_df), dtype=np.float32)
test_logit_mean = np.zeros(len(test_hybrid_df), dtype=np.float32)

skf = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

for fold, (train_idx, valid_idx) in enumerate(skf.split(texts, y), start=1):
    log(f"========== CLASSIFIER FOLD {fold}/{N_SPLITS} ==========")
    log(f"Fold train size: {len(train_idx)}, valid size: {len(valid_idx)}")

    X_train_fold = [texts[i] for i in train_idx]
    y_train_fold = y[train_idx]

    X_valid_fold = [texts[i] for i in valid_idx]
    y_valid_fold = y[valid_idx]

    log("Tokenization for classifier...")
    train_enc = tokenizer(
        X_train_fold,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    valid_enc = tokenizer(
        X_valid_fold,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )
    test_enc = tokenizer(
        test_texts,
        truncation=True,
        max_length=MAX_LEN,
        padding=False,
    )

    train_ds = EncodedDataset(train_enc, y_train_fold)
    valid_ds = EncodedDataset(valid_enc, y_valid_fold)
    test_ds = EncodedDataset(test_enc, None)

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=data_collator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=data_collator,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        collate_fn=data_collator,
    )

    log("Загружаю transformer classifier...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    neg_count = int((y_train_fold == 0).sum())
    pos_count = int((y_train_fold == 1).sum())
    class_weights = torch.tensor(
        [1.0, neg_count / max(pos_count, 1)],
        dtype=torch.float32,
        device=device,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    for epoch in range(EPOCHS):
        model.train()
        losses = []

        log(f"Fold {fold}: epoch {epoch + 1}/{EPOCHS} started")
        train_bar = tqdm(
            train_loader,
            desc=f"classifier fold {fold} epoch {epoch + 1}/{EPOCHS}",
            leave=False,
        )

        for batch in train_bar:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = F.cross_entropy(
                    outputs.logits,
                    labels,
                    weight=class_weights,
                )

            if scaler_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()

            losses.append(loss.item())
            train_bar.set_postfix(loss=float(np.mean(losses[-20:])))

        log(f"Fold {fold}: epoch {epoch + 1}/{EPOCHS} mean loss = {float(np.mean(losses)):.6f}")

    log(f"Fold {fold}: предсказания на valid...")
    fold_valid_proba, fold_valid_logit = predict_scores(model, valid_loader, device, use_amp, amp_dtype)

    log(f"Fold {fold}: предсказания на test...")
    fold_test_proba, fold_test_logit = predict_scores(model, test_loader, device, use_amp, amp_dtype)

    oof_proba[valid_idx] = fold_valid_proba
    oof_logit[valid_idx] = fold_valid_logit

    test_proba_mean += fold_test_proba / N_SPLITS
    test_logit_mean += fold_test_logit / N_SPLITS

    fold_pr_auc = average_precision_score(y_valid_fold, fold_valid_proba)
    log(f"Fold {fold}: PR-AUC = {fold_pr_auc:.6f}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        log(f"Fold {fold}: peak GPU memory allocated = {peak_gb:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

overall_oof_pr_auc = average_precision_score(y, oof_proba)
log(f"OOF transformer PR-AUC: {overall_oof_pr_auc:.6f}")

log("Добавляю OOF transformer features...")
train_hybrid_df["trf_proba"] = oof_proba
train_hybrid_df["trf_logit"] = oof_logit
train_hybrid_df["trf_margin"] = np.abs(oof_proba - 0.5)
train_hybrid_df["trf_entropy"] = binary_entropy(oof_proba)

test_hybrid_df["trf_proba"] = test_proba_mean
test_hybrid_df["trf_logit"] = test_logit_mean
test_hybrid_df["trf_margin"] = np.abs(test_proba_mean - 0.5)
test_hybrid_df["trf_entropy"] = binary_entropy(test_proba_mean)

transformer_feature_cols = [
    "trf_proba",
    "trf_logit",
    "trf_margin",
    "trf_entropy",
]

# =========================================================
# 2. Frozen embeddings + PCA 32
# =========================================================
log("========== FROZEN EMBEDDINGS ==========")
log("Готовлю токенизацию для frozen embeddings...")

all_train_enc = tokenizer(
    texts,
    truncation=True,
    max_length=MAX_LEN,
    padding=False,
)
all_test_enc = tokenizer(
    test_texts,
    truncation=True,
    max_length=MAX_LEN,
    padding=False,
)

all_train_ds = EncodedDataset(all_train_enc, None)
all_test_ds = EncodedDataset(all_test_enc, None)

all_train_loader = DataLoader(
    all_train_ds,
    batch_size=EMBED_BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=data_collator,
)
all_test_loader = DataLoader(
    all_test_ds,
    batch_size=EMBED_BATCH_SIZE,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=PERSISTENT_WORKERS,
    collate_fn=data_collator,
)

log("Загружаю frozen encoder...")
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME).to(device)

log("Считаю frozen embeddings для train...")
train_embeddings = encode_embeddings(
    embedding_model,
    all_train_loader,
    device,
    use_amp,
    amp_dtype,
)

log("Считаю frozen embeddings для test...")
test_embeddings = encode_embeddings(
    embedding_model,
    all_test_loader,
    device,
    use_amp,
    amp_dtype,
)

if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    log(f"Frozen embedding stage: peak GPU memory allocated = {peak_gb:.2f} GB")
    torch.cuda.reset_peak_memory_stats()

del embedding_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

log(f"Raw frozen train embeddings shape: {train_embeddings.shape}")
log(f"Raw frozen test embeddings shape: {test_embeddings.shape}")

log(f"Fit PCA({EMBEDDING_PCA_DIM}) on frozen train embeddings...")
pca = PCA(
    n_components=EMBEDDING_PCA_DIM,
    random_state=RANDOM_SEED,
)

train_embeddings_pca = pca.fit_transform(train_embeddings).astype(np.float32)
test_embeddings_pca = pca.transform(test_embeddings).astype(np.float32)

explained = float(np.sum(pca.explained_variance_ratio_))
log(f"Total explained variance by PCA({EMBEDDING_PCA_DIM}): {explained:.6f}")

embedding_feature_cols = []
for i in range(EMBEDDING_PCA_DIM):
    col = f"emb_pca_{i:02d}"
    embedding_feature_cols.append(col)
    train_hybrid_df[col] = train_embeddings_pca[:, i]
    test_hybrid_df[col] = test_embeddings_pca[:, i]

# =========================================================
# 3. Сборка итоговых признаков
# =========================================================
feature_cols_hybrid = base_feature_cols + transformer_feature_cols + embedding_feature_cols
cat_features_hybrid = base_cat_features.copy()

log(f"Итоговое число признаков: {len(feature_cols_hybrid)}")
log(f"Из них embedding PCA features: {len(embedding_feature_cols)}")

log(f"Сохраняю train hybrid features -> {TRAIN_HYBRID_PATH}")
train_hybrid_df.to_csv(TRAIN_HYBRID_PATH, index=False)

log(f"Сохраняю test hybrid features -> {TEST_HYBRID_PATH}")
test_hybrid_df.to_csv(TEST_HYBRID_PATH, index=False)

meta = {
    "model_name": MODEL_NAME,
    "embedding_model_name": EMBEDDING_MODEL_NAME,
    "train_path": str(TRAIN_PATH),
    "test_path": str(test_path),
    "oof_pr_auc": float(overall_oof_pr_auc),
    "feature_cols_hybrid": feature_cols_hybrid,
    "cat_features_hybrid": cat_features_hybrid,
    "base_feature_cols": base_feature_cols,
    "base_cat_features": base_cat_features,
    "transformer_feature_cols": transformer_feature_cols,
    "embedding_feature_cols": embedding_feature_cols,
    "embedding_pca_dim": EMBEDDING_PCA_DIM,
    "embedding_pca_explained_variance_sum": explained,
    "n_splits": N_SPLITS,
    "max_len": MAX_LEN,
    "epochs": EPOCHS,
    "train_batch_size": TRAIN_BATCH_SIZE,
    "valid_batch_size": VALID_BATCH_SIZE,
    "embed_batch_size": EMBED_BATCH_SIZE,
    "random_seed": RANDOM_SEED,
}

log(f"Сохраняю meta -> {META_PATH}")
META_PATH.write_text(
    json.dumps(meta, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

log("Готово.")
