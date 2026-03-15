from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split


try:
    from room_feature_engineering import normalize_text
except Exception:
    def normalize_text(text: str) -> str:
        text = str(text).lower().replace("ё", "е")
        text = re.sub(r"[|;/,(){}\[\]<>]+", " ", text)
        text = re.sub(r"[-–—_]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret supplier_room_name with TF-IDF + LogisticRegression")
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--precision_floor", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--min_df", type=int, default=3)
    parser.add_argument("--max_features", type=int, default=120000)
    parser.add_argument("--use_group_split", action="store_true", help="Split by hotel_id for tougher generalization check")
    parser.add_argument("--sample_local_explanations", type=int, default=5)
    parser.add_argument("--top_hotels", type=int, default=20)
    return parser.parse_args()


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


TOKEN_RE = re.compile(r"[a-zа-я0-9]+")


def tokenize_norm(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def build_text_views(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["supplier_room_name"] = out["supplier_room_name"].fillna("").astype(str)
    out["hotel_id"] = out["hotel_id"].fillna("unknown").astype(str)
    out["text_raw"] = out["supplier_room_name"]
    out["text_norm"] = out["supplier_room_name"].map(normalize_text)
    out["text_hotel_prefixed"] = "hotel_" + out["hotel_id"].astype(str) + " " + out["text_norm"]
    return out


def split_data(df: pd.DataFrame, test_size: float, random_state: int, use_group_split: bool):
    y = df["target"].astype(int)
    if use_group_split:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, valid_idx = next(splitter.split(df, y, groups=df["hotel_id"].astype(str)))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        valid_df = df.iloc[valid_idx].reset_index(drop=True)
    else:
        train_df, valid_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        train_df = train_df.reset_index(drop=True)
        valid_df = valid_df.reset_index(drop=True)
    return train_df, valid_df


def make_vectorizer(min_df: int, max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        lowercase=False,
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.995,
        sublinear_tf=True,
        strip_accents=None,
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b",
    )


def train_word_model(train_df: pd.DataFrame, valid_df: pd.DataFrame, min_df: int, max_features: int):
    vectorizer = make_vectorizer(min_df=min_df, max_features=max_features)
    X_train = vectorizer.fit_transform(train_df["text_norm"])
    X_valid = vectorizer.transform(valid_df["text_norm"])

    model = LogisticRegression(
        C=4.0,
        max_iter=4000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )
    model.fit(X_train, train_df["target"].astype(int))
    valid_proba = model.predict_proba(X_valid)[:, 1]
    return vectorizer, model, X_train, X_valid, valid_proba


def metric_bundle(y_true: np.ndarray, y_proba: np.ndarray, precision_floor: float) -> dict:
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ll = log_loss(y_true, y_proba, labels=[0, 1])
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision_cut = precision[:-1]
    recall_cut = recall[:-1]

    threshold_info = {"threshold": None, "precision": None, "recall": None}
    mask = precision_cut >= precision_floor
    if mask.any():
        idx = np.argmax(recall_cut[mask])
        threshold_info = {
            "threshold": float(thresholds[mask][idx]),
            "precision": float(precision_cut[mask][idx]),
            "recall": float(recall_cut[mask][idx]),
        }

    best_f1 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        cur_f1 = f1_score(y_true, pred, zero_division=0)
        if cur_f1 > best_f1:
            best_f1 = float(cur_f1)
            best_thr = float(thr)

    pred_default = (y_proba >= 0.5).astype(int)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "logloss": float(ll),
        "precision_floor": float(precision_floor),
        "threshold_info": threshold_info,
        "best_f1": float(best_f1),
        "best_f1_threshold": float(best_thr),
        "confusion_matrix_at_0_5": confusion_matrix(y_true, pred_default).tolist(),
    }


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, precision_floor: float, out_path: Path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC = {pr_auc:.5f}")
    plt.axhline(precision_floor, linestyle="--", linewidth=1, label=f"precision = {precision_floor:.2f}")

    precision_cut = precision[:-1]
    recall_cut = recall[:-1]
    mask = precision_cut >= precision_floor
    if mask.any():
        idx = np.argmax(recall_cut[mask])
        best_r = recall_cut[mask][idx]
        best_p = precision_cut[mask][idx]
        plt.scatter([best_r], [best_p], s=60, label=f"working point: R={best_r:.3f}, P={best_p:.3f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve: TF-IDF + LogisticRegression")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def coefficient_report(vectorizer: TfidfVectorizer, model: LogisticRegression, train_df: pd.DataFrame, top_k: int):
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    order_pos = np.argsort(coef)[::-1]
    order_neg = np.argsort(coef)

    train_pos_texts = train_df.loc[train_df["target"] == 1, "text_norm"].tolist()
    train_neg_texts = train_df.loc[train_df["target"] == 0, "text_norm"].tolist()

    def doc_freq(texts: list[str]) -> Counter:
        cnt = Counter()
        for text in texts:
            toks = set(TOKEN_RE.findall(text))
            for t in toks:
                cnt[t] += 1
        return cnt

    pos_df = doc_freq(train_pos_texts)
    neg_df = doc_freq(train_neg_texts)
    n_pos = max(len(train_pos_texts), 1)
    n_neg = max(len(train_neg_texts), 1)

    rows = []
    for idx in np.concatenate([order_pos[:top_k], order_neg[:top_k]]):
        token = feature_names[idx]
        rows.append({
            "token_or_ngram": token,
            "coef": float(coef[idx]),
            "direction": "towards_class_1" if coef[idx] > 0 else "towards_class_0",
            "abs_coef": float(abs(coef[idx])),
            "doc_freq_pos": int(pos_df.get(token, 0)),
            "doc_freq_neg": int(neg_df.get(token, 0)),
            "doc_rate_pos": float(pos_df.get(token, 0) / n_pos),
            "doc_rate_neg": float(neg_df.get(token, 0) / n_neg),
        })

    report = pd.DataFrame(rows).sort_values("abs_coef", ascending=False)
    top_pos = report[report["coef"] > 0].sort_values("coef", ascending=False).head(top_k)
    top_neg = report[report["coef"] < 0].sort_values("coef", ascending=True).head(top_k)
    return report, top_pos, top_neg


def plot_top_coefficients(top_pos: pd.DataFrame, top_neg: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 10))

    pos = top_pos.iloc[::-1]
    axes[0].barh(pos["token_or_ngram"], pos["coef"])
    axes[0].set_title("Top tokens towards class 1")
    axes[0].set_xlabel("LogReg coefficient")

    neg = top_neg.iloc[::-1]
    axes[1].barh(neg["token_or_ngram"], neg["coef"])
    axes[1].set_title("Top tokens towards class 0")
    axes[1].set_xlabel("LogReg coefficient")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def build_error_table(valid_df: pd.DataFrame, y_proba: np.ndarray, threshold: float) -> pd.DataFrame:
    out = valid_df[["hotel_id", "supplier_room_name", "text_norm", "target"]].copy()
    out["prediction_proba"] = y_proba
    out["pred_label"] = (y_proba >= threshold).astype(int)
    out["error_type"] = "correct"
    out.loc[(out["target"] == 1) & (out["pred_label"] == 0), "error_type"] = "FN"
    out.loc[(out["target"] == 0) & (out["pred_label"] == 1), "error_type"] = "FP"
    out.loc[(out["target"] == 1) & (out["pred_label"] == 1), "error_type"] = "TP"
    out.loc[(out["target"] == 0) & (out["pred_label"] == 0), "error_type"] = "TN"
    return out


def ngram_enrichment(texts_a: Iterable[str], texts_b: Iterable[str], top_k: int, ngram_range=(1, 2)) -> pd.DataFrame:
    # Smoothed log-odds style ratio over document frequencies.
    def iter_ngrams(text: str):
        toks = TOKEN_RE.findall(text)
        grams = set()
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(toks) - n + 1):
                grams.add(" ".join(toks[i:i+n]))
        return grams

    cnt_a = Counter()
    cnt_b = Counter()
    docs_a = 0
    docs_b = 0

    for txt in texts_a:
        docs_a += 1
        for g in iter_ngrams(txt):
            cnt_a[g] += 1
    for txt in texts_b:
        docs_b += 1
        for g in iter_ngrams(txt):
            cnt_b[g] += 1

    docs_a = max(docs_a, 1)
    docs_b = max(docs_b, 1)
    vocab = set(cnt_a) | set(cnt_b)

    rows = []
    for g in vocab:
        a = cnt_a.get(g, 0)
        b = cnt_b.get(g, 0)
        # ignore super rare grams
        if a + b < 3:
            continue
        score = math.log((a + 1) / (docs_a + 2)) - math.log((b + 1) / (docs_b + 2))
        rows.append({
            "token_or_ngram": g,
            "docs_group_a": int(a),
            "docs_group_b": int(b),
            "rate_group_a": float(a / docs_a),
            "rate_group_b": float(b / docs_b),
            "log_ratio": float(score),
        })

    if not rows:
        return pd.DataFrame(columns=["token_or_ngram", "docs_group_a", "docs_group_b", "rate_group_a", "rate_group_b", "log_ratio"])

    df = pd.DataFrame(rows).sort_values("log_ratio", ascending=False)
    return df.head(top_k)


def local_explanations(vectorizer: TfidfVectorizer, model: LogisticRegression, df: pd.DataFrame, top_n_rows: int, top_k_tokens: int = 10) -> pd.DataFrame:
    rows = []
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    selected = pd.concat([
        df[df["error_type"] == "FN"].sort_values("prediction_proba").head(top_n_rows),
        df[df["error_type"] == "FP"].sort_values("prediction_proba", ascending=False).head(top_n_rows),
        df[df["error_type"] == "TP"].sort_values("prediction_proba", ascending=False).head(top_n_rows),
        df[df["error_type"] == "TN"].sort_values("prediction_proba").head(top_n_rows),
    ], axis=0).drop_duplicates()

    X = vectorizer.transform(selected["text_norm"])
    for row_idx, (_, rec) in enumerate(selected.iterrows()):
        vec = X[row_idx]
        if sparse.issparse(vec):
            nz_idx = vec.indices
            nz_val = vec.data
        else:
            nz_idx = np.flatnonzero(vec)
            nz_val = vec[nz_idx]

        contrib = nz_val * coef[nz_idx]
        order = np.argsort(np.abs(contrib))[::-1][:top_k_tokens]
        top_items = []
        for pos in order:
            feat_idx = nz_idx[pos]
            top_items.append({
                "token_or_ngram": feature_names[feat_idx],
                "tfidf_value": float(nz_val[pos]),
                "coef": float(coef[feat_idx]),
                "contribution": float(contrib[pos]),
            })

        rows.append({
            "hotel_id": rec["hotel_id"],
            "supplier_room_name": rec["supplier_room_name"],
            "target": int(rec["target"]),
            "prediction_proba": float(rec["prediction_proba"]),
            "pred_label": int(rec["pred_label"]),
            "error_type": rec["error_type"],
            "top_token_contributions_json": json.dumps(top_items, ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def per_hotel_profile(df: pd.DataFrame, top_hotels: int) -> pd.DataFrame:
    work = df.copy()
    work["has_view_word"] = work["text_norm"].str.contains(r"\b(view|вид|sea|city|garden|mountain|pool)\b", regex=True)
    work["has_bed_word"] = work["text_norm"].str.contains(r"\b(king|queen|double|single|twin|bed|кровать|кроват)\b", regex=True)
    work["has_capacity_word"] = work["text_norm"].str.contains(r"\b(adults?|guests?|persons?|people|местн|чел|bedrooms?|спальн)\b", regex=True)
    work["has_noise_word"] = work["text_norm"].str.contains(r"\b(room only|breakfast|non refundable|refundable|package|promo|lounge access|advance purchase)\b", regex=True)
    work["has_ambiguity_word"] = work["text_norm"].str.contains(r"\b(double or twin|assigned on arrival|subject to availability|run of house)\b", regex=True)
    work["token_count"] = work["text_norm"].map(lambda x: len(TOKEN_RE.findall(x)))

    prof = work.groupby("hotel_id").agg(
        rows=("hotel_id", "size"),
        target_rate=("target", "mean"),
        avg_token_count=("token_count", "mean"),
        unique_room_names=("text_norm", "nunique"),
        share_view=("has_view_word", "mean"),
        share_bed=("has_bed_word", "mean"),
        share_capacity=("has_capacity_word", "mean"),
        share_noise=("has_noise_word", "mean"),
        share_ambiguity=("has_ambiguity_word", "mean"),
    ).reset_index().sort_values("rows", ascending=False).head(top_hotels)
    return prof


def per_hotel_token_differences(df: pd.DataFrame, hotel_ids: list[str], top_k: int) -> pd.DataFrame:
    rows = []
    for hotel_id in hotel_ids:
        sub = df[df["hotel_id"].astype(str) == str(hotel_id)]
        pos = sub[sub["target"] == 1]["text_norm"].tolist()
        neg = sub[sub["target"] == 0]["text_norm"].tolist()
        if len(pos) < 5 or len(neg) < 5:
            continue
        top_pos = ngram_enrichment(pos, neg, top_k=top_k, ngram_range=(1, 2))
        for _, rec in top_pos.iterrows():
            rows.append({
                "hotel_id": hotel_id,
                "direction": "towards_class_1_inside_hotel",
                **rec.to_dict(),
            })
        top_neg = ngram_enrichment(neg, pos, top_k=top_k, ngram_range=(1, 2))
        for _, rec in top_neg.iterrows():
            rows.append({
                "hotel_id": hotel_id,
                "direction": "towards_class_0_inside_hotel",
                **rec.to_dict(),
            })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    data_dir = project_root / "data"
    outputs_dir = Path(args.output_dir).resolve() if args.output_dir else project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path).resolve() if args.train_path else data_dir / "public_dataset.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Не найден train csv: {train_path}")

    df = pd.read_csv(train_path)
    if not {"hotel_id", "supplier_room_name", "target"}.issubset(df.columns):
        raise ValueError("Ожидаются колонки hotel_id, supplier_room_name, target")

    df = build_text_views(df)
    train_df, valid_df = split_data(df, args.test_size, args.random_state, args.use_group_split)

    vectorizer, model, X_train, X_valid, valid_proba = train_word_model(
        train_df=train_df,
        valid_df=valid_df,
        min_df=args.min_df,
        max_features=args.max_features,
    )

    metrics = metric_bundle(valid_df["target"].to_numpy(), valid_proba, args.precision_floor)
    threshold = metrics["threshold_info"]["threshold"]
    if threshold is None:
        threshold = 0.5

    error_df = build_error_table(valid_df, valid_proba, threshold=threshold)

    coeff_all, top_pos, top_neg = coefficient_report(vectorizer, model, train_df, top_k=args.top_k)
    local_df = local_explanations(vectorizer, model, error_df, top_n_rows=args.sample_local_explanations, top_k_tokens=12)
    hotel_profile_df = per_hotel_profile(df, top_hotels=args.top_hotels)
    hotel_token_df = per_hotel_token_differences(df, hotel_profile_df["hotel_id"].astype(str).tolist(), top_k=min(args.top_k, 15))

    # Error-focused token analyses
    fn_vs_tp = ngram_enrichment(
        error_df.loc[error_df["error_type"] == "FN", "text_norm"].tolist(),
        error_df.loc[error_df["error_type"] == "TP", "text_norm"].tolist(),
        top_k=args.top_k,
        ngram_range=(1, 2),
    )
    fp_vs_tn = ngram_enrichment(
        error_df.loc[error_df["error_type"] == "FP", "text_norm"].tolist(),
        error_df.loc[error_df["error_type"] == "TN", "text_norm"].tolist(),
        top_k=args.top_k,
        ngram_range=(1, 2),
    )

    # Plots
    plot_pr_curve(valid_df["target"].to_numpy(), valid_proba, args.precision_floor, outputs_dir / "tfidf_logreg_pr_curve.png")
    plot_top_coefficients(top_pos, top_neg, outputs_dir / "tfidf_logreg_top_tokens.png")

    # Save tables
    valid_export = error_df.copy()
    valid_export.to_csv(outputs_dir / "tfidf_logreg_valid_predictions.csv", index=False)
    coeff_all.to_csv(outputs_dir / "tfidf_logreg_token_coefficients_full.csv", index=False)
    top_pos.to_csv(outputs_dir / "tfidf_logreg_top_positive_tokens.csv", index=False)
    top_neg.to_csv(outputs_dir / "tfidf_logreg_top_negative_tokens.csv", index=False)
    fn_vs_tp.to_csv(outputs_dir / "tfidf_logreg_fn_vs_tp_tokens.csv", index=False)
    fp_vs_tn.to_csv(outputs_dir / "tfidf_logreg_fp_vs_tn_tokens.csv", index=False)
    local_df.to_csv(outputs_dir / "tfidf_logreg_local_explanations.csv", index=False)
    hotel_profile_df.to_csv(outputs_dir / "tfidf_logreg_hotel_profiles.csv", index=False)
    hotel_token_df.to_csv(outputs_dir / "tfidf_logreg_hotel_token_differences.csv", index=False)

    report = {
        "train_path": str(train_path),
        "use_group_split": bool(args.use_group_split),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "vocab_size": int(len(vectorizer.get_feature_names_out())),
        **metrics,
    }
    (outputs_dir / "tfidf_logreg_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved outputs to: {outputs_dir}")


if __name__ == "__main__":
    main()
