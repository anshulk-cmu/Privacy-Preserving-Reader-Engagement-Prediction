"""
05b_lstm_blind_test.py — Blind Test Set: LSTM Engagement + Re-identification
==============================================================================
Evaluates the trained LSTM on the 10K-user blind test set (zero overlap
with 50K training/validation users). Same methodology as 03b for MLP.

Three experiments:
  1. Engagement prediction — does the LSTM generalize to unseen users?
  2. Within-blind re-identification — can the LSTM fingerprint unseen users?
  3. Cross-dataset re-identification — negative control (gallery=50K, probe=blind)

Usage:
    conda activate privacy
    python src/05b_lstm_blind_test.py
"""

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BENCHMARK_SRC = PROJECT_ROOT / "ebnerd-benchmark" / "src"
sys.path.insert(0, str(BENCHMARK_SRC))

from ebrec.utils._constants import (
    DEFAULT_USER_COL,
    DEFAULT_HISTORY_READ_TIME_COL,
    DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL,
)
from models.lstm_model import LSTMEngagementModel
from models.train import get_device
from models.attack import build_gallery_probe_split, run_reidentification_attack, run_random_baseline

BLIND_DIR = PROJECT_ROOT / "data" / "ebnerd_blind_test"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LSTM_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "lstm"
MLP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "mlp_baseline"

HISTORY_SEQ_LEN = 50
ENGAGEMENT_SCROLL_THRESHOLD = 50
ENGAGEMENT_READ_TIME_THRESHOLD = 30


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def preprocess_blind_split(split, rt_scaler, agg_scaler, articles_df):
    """Preprocess one split of the blind test set using saved scalers."""
    beh_path = BLIND_DIR / split / "behaviors.parquet"
    hist_path = BLIND_DIR / split / "history.parquet"

    df = pl.read_parquet(beh_path)
    n_raw = len(df)
    df_labeled = df.filter(
        pl.col("next_read_time").is_not_null()
        & pl.col("next_scroll_percentage").is_not_null()
    ).with_columns(
        ((pl.col("next_scroll_percentage") > ENGAGEMENT_SCROLL_THRESHOLD)
         & (pl.col("next_read_time") > ENGAGEMENT_READ_TIME_THRESHOLD)
        ).cast(pl.Int8).alias("engaged")
    )
    n_labeled = len(df_labeled)
    n_pos = df_labeled.filter(pl.col("engaged") == 1).shape[0]
    print(f"  {split}: {n_raw:,} raw -> {n_labeled:,} labelable ({100*n_labeled/n_raw:.1f}%)")
    print(f"    Engaged: {n_pos:,} ({100*n_pos/n_labeled:.1f}%)")

    labels = df_labeled["engaged"].to_numpy().astype(np.float32)
    user_ids = df_labeled[DEFAULT_USER_COL].to_numpy()

    df_hist = pl.read_parquet(hist_path)
    hist_map = {}
    for row in df_hist.iter_rows(named=True):
        uid = row[DEFAULT_USER_COL]
        rt_list = row[DEFAULT_HISTORY_READ_TIME_COL]
        sp_list = row[DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL]
        hist_map[uid] = (rt_list, sp_list)

    n = len(df_labeled)
    sequences = np.zeros((n, HISTORY_SEQ_LEN, 2), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int32)
    uid_list = df_labeled[DEFAULT_USER_COL].to_list()

    for i, uid in enumerate(uid_list):
        if uid not in hist_map:
            continue
        rt_list, sp_list = hist_map[uid]
        hist_len = len(rt_list)
        actual_len = min(hist_len, HISTORY_SEQ_LEN)
        lengths[i] = actual_len
        rt_tail = rt_list[-actual_len:]
        sp_tail = sp_list[-actual_len:]
        for j in range(actual_len):
            sequences[i, j, 0] = rt_tail[j] if rt_tail[j] is not None else 0.0
            sequences[i, j, 1] = sp_tail[j] if sp_tail[j] is not None else 0.0

    sequences[:, :, 0] = np.log1p(sequences[:, :, 0])
    for i in range(n):
        l = lengths[i]
        if l > 0:
            sequences[i, :l, 0] = rt_scaler.transform(
                sequences[i, :l, 0].reshape(-1, 1)).flatten()
    sequences[:, :, 1] = sequences[:, :, 1] / 100.0

    agg = np.zeros((n, 27), dtype=np.float32)
    for i, uid in enumerate(uid_list):
        if uid not in hist_map:
            continue
        rt_list_raw, sp_list_raw = hist_map[uid]
        rt = np.array([v if v is not None else 0.0 for v in rt_list_raw], dtype=np.float32)
        sp = np.array([v if v is not None else 0.0 for v in sp_list_raw], dtype=np.float32)
        hist_len = len(rt)
        agg[i, 0] = rt.mean()
        agg[i, 1] = rt.std() if hist_len > 1 else 0.0
        agg[i, 2] = np.median(rt)
        agg[i, 3] = np.percentile(rt, 10) if hist_len >= 10 else rt.min()
        agg[i, 4] = np.percentile(rt, 90) if hist_len >= 10 else rt.max()
        agg[i, 5] = sp.mean()
        agg[i, 6] = sp.std() if hist_len > 1 else 0.0
        agg[i, 7] = np.median(sp)
        agg[i, 8] = np.percentile(sp, 10) if hist_len >= 10 else sp.min()
        agg[i, 9] = np.percentile(sp, 90) if hist_len >= 10 else sp.max()
        last5_rt = rt[-5:] if hist_len >= 5 else np.pad(rt, (5 - hist_len, 0))
        agg[i, 10:15] = last5_rt[-5:]
        last5_sp = sp[-5:] if hist_len >= 5 else np.pad(sp, (5 - hist_len, 0))
        agg[i, 15:20] = last5_sp[-5:]
        agg[i, 20] = hist_len
        engaged_mask = (rt > ENGAGEMENT_READ_TIME_THRESHOLD) & (sp > ENGAGEMENT_SCROLL_THRESHOLD)
        agg[i, 21] = engaged_mask.mean()
        agg[i, 22] = (sp > 80).mean()
        agg[i, 23] = (rt > 60).mean()
        if hist_len >= 5 and rt.mean() > 0:
            agg[i, 24] = np.clip(last5_rt.mean() / rt.mean(), 0.1, 10.0)
        else:
            agg[i, 24] = 1.0
        if hist_len >= 5 and sp.mean() > 0:
            agg[i, 25] = np.clip(last5_sp.mean() / sp.mean(), 0.1, 10.0)
        else:
            agg[i, 25] = 1.0
        if hist_len > 2 and rt.std() > 0 and sp.std() > 0:
            agg[i, 26] = np.corrcoef(rt, sp)[0, 1]
            if np.isnan(agg[i, 26]):
                agg[i, 26] = 0.0
        else:
            agg[i, 26] = 0.0

    agg = np.nan_to_num(agg, nan=0.0, posinf=10.0, neginf=-10.0)
    agg = agg_scaler.transform(agg).astype(np.float32)

    categories = sorted(articles_df["category"].unique().to_list())
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    article_types = sorted(articles_df["article_type"].unique().to_list())
    type_to_idx = {t: i for i, t in enumerate(article_types)}

    article_map = {}
    for row in articles_df.iter_rows(named=True):
        aid = row["article_id"]
        body_len = len(row["body"]) if row["body"] else 0
        title_len = len(row["title"]) if row["title"] else 0
        subtitle_len = len(row["subtitle"]) if row["subtitle"] else 0
        article_map[aid] = (
            cat_to_idx.get(row["category"], 0),
            type_to_idx.get(row["article_type"], 0),
            1.0 if row["premium"] else 0.0,
            row["sentiment_score"] if row["sentiment_score"] is not None else 0.5,
            np.log1p(body_len), np.log1p(title_len), np.log1p(subtitle_len),
        )

    article_feats = np.zeros((n, 7), dtype=np.float32)
    clicked_lists = df_labeled["article_ids_clicked"].to_list()
    for i, clicked in enumerate(clicked_lists):
        if clicked and len(clicked) > 0:
            first_aid = clicked[0]
            if first_aid in article_map:
                article_feats[i] = article_map[first_aid]

    device_col = df_labeled["device_type"].cast(pl.Float32).to_numpy()
    subscriber = df_labeled["is_subscriber"].cast(pl.Float32).to_numpy()
    sso = df_labeled["is_sso_user"].cast(pl.Float32).to_numpy()
    context = np.stack([device_col, subscriber, sso], axis=1).astype(np.float32)

    print(f"    Samples: {n:,}, Pos rate: {labels.mean():.4f}, Unique users: {len(np.unique(user_ids)):,}")

    return {
        "history_seq": sequences, "history_lengths": lengths,
        "agg_features": agg, "article_features": article_feats,
        "context_features": context, "labels": labels, "user_ids": user_ids,
    }


def evaluate_engagement(model, data, device, split_name):
    """Run LSTM on preprocessed data and compute engagement metrics."""
    model.eval()
    n = len(data["labels"])
    all_logits = []
    batch_size = 512

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            article = data["article_features"][start:end]
            batch = {
                "history_seq": torch.from_numpy(data["history_seq"][start:end]).to(device),
                "history_length": torch.from_numpy(data["history_lengths"][start:end].astype(np.int64)).to(device),
                "agg_features": torch.from_numpy(data["agg_features"][start:end]).to(device),
                "article_cat": torch.from_numpy(article[:, 0].astype(np.int64)).to(device),
                "article_type": torch.from_numpy(article[:, 1].astype(np.int64)).to(device),
                "article_cont": torch.from_numpy(article[:, 2:].astype(np.float32)).to(device),
                "context": torch.from_numpy(data["context_features"][start:end]).to(device),
            }
            logits = model(batch)
            all_logits.append(logits.cpu().numpy())

    logits = np.concatenate(all_logits)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.float32)
    labels = data["labels"]

    metrics = {
        "auc": float(roc_auc_score(labels, probs)),
        "f1": float(f1_score(labels, preds)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }
    print(f"\n  {split_name} Engagement Metrics:")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")
    return metrics


def extract_blind_representations(model, data, device):
    """Extract 64-dim representations from the LSTM for blind test data."""
    model.eval()
    n = len(data["labels"])
    all_reprs = []
    batch_size = 512

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            article = data["article_features"][start:end]
            batch = {
                "history_seq": torch.from_numpy(data["history_seq"][start:end]).to(device),
                "history_length": torch.from_numpy(data["history_lengths"][start:end].astype(np.int64)).to(device),
                "agg_features": torch.from_numpy(data["agg_features"][start:end]).to(device),
                "article_cat": torch.from_numpy(article[:, 0].astype(np.int64)).to(device),
                "article_type": torch.from_numpy(article[:, 1].astype(np.int64)).to(device),
                "article_cont": torch.from_numpy(article[:, 2:].astype(np.float32)).to(device),
                "context": torch.from_numpy(data["context_features"][start:end]).to(device),
            }
            reprs = model.get_representation(batch)
            all_reprs.append(reprs.cpu().numpy())
    return np.concatenate(all_reprs)


def main():
    t_start = time.time()

    print("=" * 70)
    print("  Blind Test Evaluation — LSTM")
    print("  10K users, ZERO overlap with training data")
    print("=" * 70)

    section("1. LOADING SCALERS AND MODEL")
    with open(PROCESSED_DIR / "rt_scaler.pkl", "rb") as f:
        rt_scaler = pickle.load(f)
    with open(PROCESSED_DIR / "agg_scaler.pkl", "rb") as f:
        agg_scaler = pickle.load(f)
    print("  Loaded rt_scaler.pkl and agg_scaler.pkl (fitted on 50K train)")

    metadata = json.load(open(PROCESSED_DIR / "metadata.json"))
    device = get_device()

    model = LSTMEngagementModel(
        n_categories=metadata["n_categories"],
        n_article_types=metadata["n_article_types"],
        article_cont_dim=metadata.get("article_cont_dim", 5),
        agg_dim=metadata.get("agg_feature_dim", 27),
        lstm_dropout=0.15,
    )
    checkpoint = torch.load(LSTM_OUTPUT_DIR / "checkpoint.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded LSTM checkpoint (epoch {checkpoint.get('epoch', '?')})")
    print(f"  Device: {device}")

    articles_df = pl.read_parquet(BLIND_DIR / "articles.parquet")

    section("2. PREPROCESSING BLIND TEST DATA")
    t0 = time.time()
    blind_val = preprocess_blind_split("validation", rt_scaler, agg_scaler, articles_df)
    blind_train = preprocess_blind_split("train", rt_scaler, agg_scaler, articles_df)
    print(f"\n  Preprocessing time: {time.time()-t0:.1f}s")

    blind_all_user_ids = np.concatenate([blind_train["user_ids"], blind_val["user_ids"]])
    n_unique_blind = len(np.unique(blind_all_user_ids))
    print(f"  Combined: {len(blind_all_user_ids):,} impressions, {n_unique_blind:,} unique users")

    section("3. ENGAGEMENT PREDICTION ON BLIND TEST")
    val_metrics = evaluate_engagement(model, blind_val, device, "Blind Val")
    train_metrics = evaluate_engagement(model, blind_train, device, "Blind Train")

    # Load MLP blind test results for comparison
    mlp_blind_path = MLP_OUTPUT_DIR / "blind_test_results.json"
    mlp_blind = {}
    if mlp_blind_path.exists():
        with open(mlp_blind_path) as f:
            mlp_blind = json.load(f)

    print(f"\n  Comparison (Blind Val):")
    print(f"    {'Metric':12s}  {'LSTM':>10s}  {'MLP':>10s}  {'Delta':>10s}")
    print(f"    {'-'*48}")
    mlp_val = mlp_blind.get("engagement", {}).get("blind_val", {})
    for k in ["auc", "f1", "accuracy", "precision", "recall"]:
        lstm_v = val_metrics[k]
        mlp_v = mlp_val.get(k, 0)
        print(f"    {k:12s}  {lstm_v:10.4f}  {mlp_v:10.4f}  {lstm_v-mlp_v:+10.4f}")

    section("4. EXTRACTING BLIND TEST REPRESENTATIONS")
    blind_val_reprs = extract_blind_representations(model, blind_val, device)
    blind_train_reprs = extract_blind_representations(model, blind_train, device)
    blind_all_reprs = np.concatenate([blind_train_reprs, blind_val_reprs])
    blind_all_labels = np.concatenate([blind_train["labels"], blind_val["labels"]])

    print(f"  Combined: {blind_all_reprs.shape}")
    print(f"  Repr mean: {blind_all_reprs.mean():.4f}, std: {blind_all_reprs.std():.4f}")

    section("5. WITHIN-BLIND RE-IDENTIFICATION")
    split = build_gallery_probe_split(
        blind_all_user_ids, blind_all_reprs, blind_all_labels,
        min_impressions=4, gallery_fraction=0.5, seed=42,
    )
    print(f"  Gallery: {split['n_gallery_users']:,} users, Probe: {split['n_probe_impressions']:,}")
    baseline = run_random_baseline(split["n_gallery_users"], split["n_probe_impressions"])
    print(f"  Random baseline Top-1: {baseline['top_k_accuracy']['1']*100:.4f}%")

    within_results = {}
    for metric_name in ["cosine", "euclidean"]:
        print(f"\n  Attack ({metric_name}):")
        results = run_reidentification_attack(
            split["gallery_profiles"], split["gallery_user_ids"],
            split["probe_reprs"], split["probe_user_ids"],
            metric=metric_name, top_k_values=(1, 5, 10, 20), batch_size=2000,
        )
        within_results[metric_name] = results
        top1 = results["top_k_accuracy"]["1"] * 100
        lift = results["top_k_accuracy"]["1"] / baseline["top_k_accuracy"]["1"]
        print(f"    Top-1: {top1:.2f}% ({lift:.0f}x above random)")
        print(f"    Top-5: {results['top_k_accuracy']['5']*100:.2f}%")
        print(f"    MRR: {results['mrr']:.4f}, Median rank: {results['median_rank']:.0f} / {split['n_gallery_users']:,}")

    section("6. CROSS-DATASET RE-IDENTIFICATION (negative control)")
    repr_data = np.load(LSTM_OUTPUT_DIR / "representations.npz")
    val50k_user_ids = repr_data["val_user_ids"]
    val50k_reprs = repr_data["val_representations"]

    user_to_reprs = defaultdict(list)
    for i, uid in enumerate(val50k_user_ids):
        user_to_reprs[uid].append(val50k_reprs[i])
    gallery_profiles = np.array([np.mean(r, axis=0) for r in user_to_reprs.values()])
    gallery_uids = np.array(list(user_to_reprs.keys()))
    n_gallery = len(gallery_uids)

    cross_baseline = run_random_baseline(n_gallery, len(blind_all_reprs))
    print(f"  50K Gallery: {n_gallery:,}, Blind Probes: {len(blind_all_reprs):,}")

    cross_results = run_reidentification_attack(
        gallery_profiles, gallery_uids, blind_all_reprs, blind_all_user_ids,
        metric="cosine", top_k_values=(1, 5, 10, 20), batch_size=2000,
    )
    print(f"  Cross-dataset Top-1: {cross_results['top_k_accuracy']['1']*100:.4f}%")
    print(f"  Mean rank: {cross_results['mean_rank']:.1f} / {n_gallery:,}")

    section("7. SAVING RESULTS")
    blind_results = {
        "engagement": {"blind_val": val_metrics, "blind_train": train_metrics},
        "within_blind_reid": {m: {k: v for k, v in r.items() if not k.startswith("_")} for m, r in within_results.items()},
        "cross_dataset_reid": {
            "n_gallery": n_gallery, "n_probes": len(blind_all_reprs),
            "top_k_accuracy": {k: float(v) for k, v in cross_results["top_k_accuracy"].items()},
            "mrr": float(cross_results["mrr"]),
            "mean_rank": float(cross_results["mean_rank"]),
            "median_rank": float(cross_results["median_rank"]),
        },
        "baseline": baseline,
    }
    results_path = LSTM_OUTPUT_DIR / "blind_test_results.json"
    with open(results_path, "w") as f:
        json.dump(blind_results, f, indent=2)
    print(f"  Saved: {results_path}")

    elapsed = time.time() - t_start
    best_within = within_results.get("cosine", within_results.get("euclidean"))
    within_top1 = best_within["top_k_accuracy"]["1"] * 100
    within_lift = best_within["top_k_accuracy"]["1"] / baseline["top_k_accuracy"]["1"]

    print(f"\n{'='*70}")
    print(f"  LSTM BLIND TEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Blind val AUC: {val_metrics['auc']:.4f}")
    print(f"  Within-blind Top-1: {within_top1:.2f}% ({within_lift:.0f}x)")
    print(f"  Cross-dataset Top-1: {cross_results['top_k_accuracy']['1']*100:.4f}%")


if __name__ == "__main__":
    main()
