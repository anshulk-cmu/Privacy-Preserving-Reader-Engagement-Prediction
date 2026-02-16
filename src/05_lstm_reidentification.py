"""
05_lstm_reidentification.py — Re-identification Test for LSTM + MLP Comparison
================================================================================
Tests whether the LSTM model's learned representations can identify individual
users, then generates comprehensive comparison plots against the MLP baseline.

Methodology (same as 03_reidentification_test.py):
  1. Load LSTM validation representations
  2. Gallery/probe split (50/50 for users with 4+ impressions)
  3. Nearest-neighbor attack with euclidean and cosine distance
  4. Compare LSTM vs MLP vs random baseline

Additional outputs:
  - lstm_vs_mlp_comparison.png: the key 2x2 comparison plot
  - reidentification_{metric}.png: per-metric 2x3 detail plots
  - reidentification_comparison.png: metric comparison bar chart
  - reidentification_results.json: all numerical results

Usage:
    source .venv/bin/activate
    python src/05_lstm_reidentification.py
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.attack import (
    build_gallery_probe_split,
    run_reidentification_attack,
    run_random_baseline,
)


def plot_attack_results(results: dict, baseline: dict, output_dir: Path,
                        metric_name: str, model_name: str = "LSTM"):
    """Generate comprehensive re-identification attack visualization (2x3 grid)."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"User Re-identification Attack — {model_name} ({metric_name} distance)\n"
        f"{results['n_gallery_users']:,} gallery users, {results['n_probes']:,} probe impressions",
        fontsize=14, fontweight="bold",
    )

    ranks = results["_ranks"]
    per_user_acc = results["_per_user_acc"]
    per_user_values = list(per_user_acc.values())

    # 1. Top-K accuracy bar chart
    ax = axes[0, 0]
    ks = sorted(results["top_k_accuracy"].keys(), key=int)
    attack_vals = [results["top_k_accuracy"][k] * 100 for k in ks]
    baseline_vals = [baseline["top_k_accuracy"][k] * 100 for k in ks]
    x = np.arange(len(ks))
    w = 0.35
    bars1 = ax.bar(x - w/2, attack_vals, w, label="Attack", color="crimson", alpha=0.8)
    bars2 = ax.bar(x + w/2, baseline_vals, w, label="Random", color="gray", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Re-identification Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, attack_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Rank CDF
    ax = axes[0, 1]
    sorted_ranks = np.sort(ranks)
    cdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)
    ax.plot(sorted_ranks, cdf * 100, "b-", linewidth=2)
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=np.median(ranks), color="red", linestyle="--", alpha=0.7,
               label=f"Median rank = {np.median(ranks):.0f}")
    ax.set_xlabel("Rank of Correct User")
    ax.set_ylabel("Cumulative % of Probes")
    ax.set_title("Rank Distribution (CDF)")
    ax.set_xlim(0, min(500, results["n_gallery_users"]))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Rank histogram (top 100)
    ax = axes[0, 2]
    top100_ranks = ranks[ranks <= 100]
    ax.hist(top100_ranks, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(x=1, color="red", linestyle="-", linewidth=2, label="Rank 1 (exact match)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    ax.set_title(f"Rank Distribution (top 100 of {results['n_gallery_users']:,})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Per-user accuracy distribution
    ax = axes[1, 0]
    ax.hist(per_user_values, bins=50, color="darkorange", alpha=0.7, edgecolor="white")
    ax.axvline(x=np.mean(per_user_values), color="red", linestyle="--",
               label=f"Mean = {np.mean(per_user_values)*100:.1f}%")
    ax.axvline(x=np.median(per_user_values), color="blue", linestyle="--",
               label=f"Median = {np.median(per_user_values)*100:.1f}%")
    ax.set_xlabel("Per-User Top-1 Accuracy")
    ax.set_ylabel("Number of Users")
    ax.set_title("Per-User Re-identification Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Accuracy vs impressions
    ax = axes[1, 1]
    user_n_impressions = {}
    for uid, stats in results.get("_user_stats", {}).items():
        if uid in per_user_acc:
            user_n_impressions[uid] = stats["n_total"]

    if user_n_impressions:
        n_imps = [user_n_impressions[uid] for uid in per_user_acc.keys() if uid in user_n_impressions]
        accs = [per_user_acc[uid] for uid in per_user_acc.keys() if uid in user_n_impressions]
        bins = [4, 6, 10, 15, 20, 30, 50, 100, 200]
        bin_labels = []
        bin_accs = []
        for i in range(len(bins) - 1):
            mask = [(bins[i] <= n < bins[i+1]) for n in n_imps]
            if sum(mask) > 0:
                bin_labels.append(f"{bins[i]}-{bins[i+1]-1}")
                bin_accs.append(np.mean([a for a, m in zip(accs, mask) if m]))
        ax.bar(range(len(bin_labels)), [a * 100 for a in bin_accs], color="teal", alpha=0.7)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45)
        ax.set_xlabel("Impressions per User")
        ax.set_ylabel("Mean Top-1 Accuracy (%)")
        ax.set_title("Re-id Accuracy vs User Activity")
        ax.grid(True, alpha=0.3, axis="y")

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    summary_text = (
        f"RE-IDENTIFICATION RESULTS ({model_name})\n"
        f"{'='*40}\n\n"
        f"Gallery users:     {results['n_gallery_users']:,}\n"
        f"Probe impressions: {results['n_probes']:,}\n"
        f"Distance metric:   {metric_name}\n\n"
        f"Top-1 Accuracy:    {results['top_k_accuracy']['1']*100:.2f}%\n"
        f"Top-5 Accuracy:    {results['top_k_accuracy']['5']*100:.2f}%\n"
        f"Top-10 Accuracy:   {results['top_k_accuracy']['10']*100:.2f}%\n"
        f"Top-20 Accuracy:   {results['top_k_accuracy']['20']*100:.2f}%\n\n"
        f"MRR:               {results['mrr']:.4f}\n"
        f"Mean Rank:         {results['mean_rank']:.1f}\n"
        f"Median Rank:       {results['median_rank']:.0f}\n\n"
        f"Per-user accuracy:\n"
        f"  Mean:  {results['per_user_accuracy_mean']*100:.1f}%\n"
        f"  Median:{results['per_user_accuracy_median']*100:.1f}%\n"
        f"  100% identifiable: {results['users_with_100pct_reid']:,}\n"
        f"  0% identifiable:   {results['users_with_0pct_reid']:,}\n\n"
        f"Random baseline Top-1:\n"
        f"  {baseline['top_k_accuracy']['1']*100:.4f}%\n"
        f"  ({results['top_k_accuracy']['1'] / max(baseline['top_k_accuracy']['1'], 1e-10):.0f}x "
        f"above random)"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / f"reidentification_{metric_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved reidentification_{metric_name}.png")


def plot_lstm_vs_mlp_comparison(
    lstm_results: dict,
    mlp_results: dict,
    baseline: dict,
    lstm_metrics: dict,
    mlp_metrics: dict,
    lstm_ranks: np.ndarray,
    mlp_ranks: np.ndarray,
    output_dir: Path,
):
    """Generate the key LSTM vs MLP comparison plot (2x2 grid)."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "LSTM vs MLP: Engagement Prediction & Re-identification Comparison",
        fontsize=15, fontweight="bold",
    )

    # 1. Top-K accuracy: LSTM vs MLP vs Random
    ax = axes[0, 0]
    ks = ["1", "5", "10", "20"]
    lstm_vals = [lstm_results["top_k_accuracy"][k] * 100 for k in ks]
    mlp_vals = [mlp_results["top_k_accuracy"][k] * 100 for k in ks]
    rand_vals = [baseline["top_k_accuracy"][k] * 100 for k in ks]

    x = np.arange(len(ks))
    w = 0.25
    bars_lstm = ax.bar(x - w, lstm_vals, w, label="LSTM", color="crimson", alpha=0.85)
    bars_mlp = ax.bar(x, mlp_vals, w, label="MLP", color="steelblue", alpha=0.85)
    ax.bar(x + w, rand_vals, w, label="Random", color="gray", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Re-identification Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars_lstm, lstm_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for bar, val in zip(bars_mlp, mlp_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=8)

    # 2. Engagement metrics comparison
    ax = axes[0, 1]
    metrics_names = ["AUC", "F1", "Accuracy", "Precision", "Recall", "Avg Prec"]
    metrics_keys = ["auc", "f1", "accuracy", "precision", "recall", "avg_precision"]
    lstm_metric_vals = [lstm_metrics.get(k, 0) for k in metrics_keys]
    mlp_metric_vals = [mlp_metrics.get(k, 0) for k in metrics_keys]

    x = np.arange(len(metrics_names))
    w = 0.35
    ax.bar(x - w/2, lstm_metric_vals, w, label="LSTM", color="crimson", alpha=0.85)
    ax.bar(x + w/2, mlp_metric_vals, w, label="MLP", color="steelblue", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Engagement Prediction Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1)

    # 3. Rank CDF overlay
    ax = axes[1, 0]
    for ranks, label, color in [
        (lstm_ranks, "LSTM", "crimson"),
        (mlp_ranks, "MLP", "steelblue"),
    ]:
        sorted_r = np.sort(ranks)
        cdf = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        ax.plot(sorted_r, cdf * 100, "-", linewidth=2, color=color, label=label)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Rank of Correct User")
    ax.set_ylabel("Cumulative % of Probes")
    ax.set_title("Rank Distribution (CDF) — Lower is Better")
    ax.set_xlim(0, min(1000, lstm_results["n_gallery_users"]))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Summary comparison table
    ax = axes[1, 1]
    ax.axis("off")

    lstm_top1 = lstm_results["top_k_accuracy"]["1"]
    mlp_top1 = mlp_results["top_k_accuracy"]["1"]
    random_top1 = baseline["top_k_accuracy"]["1"]
    lstm_lift = lstm_top1 / random_top1 if random_top1 > 0 else 0
    mlp_lift = mlp_top1 / random_top1 if random_top1 > 0 else 0

    summary = (
        f"COMPARISON SUMMARY\n"
        f"{'='*45}\n\n"
        f"{'Metric':<25s} {'LSTM':>10s} {'MLP':>10s}\n"
        f"{'-'*45}\n"
        f"{'Engagement AUC':<25s} {lstm_metrics.get('auc', 0):>10.4f} {mlp_metrics.get('auc', 0):>10.4f}\n"
        f"{'Engagement F1':<25s} {lstm_metrics.get('f1', 0):>10.4f} {mlp_metrics.get('f1', 0):>10.4f}\n"
        f"{'Engagement Accuracy':<25s} {lstm_metrics.get('accuracy', 0):>10.4f} {mlp_metrics.get('accuracy', 0):>10.4f}\n"
        f"{'-'*45}\n"
        f"{'Re-id Top-1':<25s} {lstm_top1*100:>9.2f}% {mlp_top1*100:>9.2f}%\n"
        f"{'Re-id Top-5':<25s} {lstm_results['top_k_accuracy']['5']*100:>9.2f}% {mlp_results['top_k_accuracy']['5']*100:>9.2f}%\n"
        f"{'Re-id Top-10':<25s} {lstm_results['top_k_accuracy']['10']*100:>9.2f}% {mlp_results['top_k_accuracy']['10']*100:>9.2f}%\n"
        f"{'Re-id Top-20':<25s} {lstm_results['top_k_accuracy']['20']*100:>9.2f}% {mlp_results['top_k_accuracy']['20']*100:>9.2f}%\n"
        f"{'MRR':<25s} {lstm_results['mrr']:>10.4f} {mlp_results['mrr']:>10.4f}\n"
        f"{'Median Rank':<25s} {lstm_results['median_rank']:>10.0f} {mlp_results['median_rank']:>10.0f}\n"
        f"{'Lift over Random':<25s} {lstm_lift:>9.0f}x {mlp_lift:>9.0f}x\n"
        f"{'-'*45}\n"
        f"{'Parameters':<25s} {'~1.0M':>10s} {'~207K':>10s}\n"
        f"{'Input':<25s} {'Sequences':>10s} {'Aggregates':>10s}\n"
    )
    ax.text(0.02, 0.95, summary, transform=ax.transAxes,
            fontsize=9.5, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "lstm_vs_mlp_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved lstm_vs_mlp_comparison.png")


def main():
    LSTM_DIR = PROJECT_ROOT / "outputs" / "models" / "lstm"
    MLP_DIR = PROJECT_ROOT / "outputs" / "models" / "mlp_baseline"
    LSTM_REPR = LSTM_DIR / "representations.npz"
    MLP_REPR = MLP_DIR / "representations.npz"

    print("=" * 70)
    print("  Re-identification Test — LSTM + MLP Comparison")
    print("=" * 70)

    # ---- Load LSTM representations ----
    print("\n  Loading LSTM validation representations...")
    lstm_data = np.load(LSTM_REPR)
    lstm_user_ids = lstm_data["val_user_ids"]
    lstm_reprs = lstm_data["val_representations"]
    lstm_labels = lstm_data["val_labels"]
    print(f"    {len(lstm_user_ids):,} impressions, {len(np.unique(lstm_user_ids)):,} unique users")
    print(f"    Representation dim: {lstm_reprs.shape[1]}")

    # ---- Load MLP representations for comparison ----
    print("\n  Loading MLP validation representations...")
    mlp_data = np.load(MLP_REPR)
    mlp_user_ids = mlp_data["val_user_ids"]
    mlp_reprs = mlp_data["val_representations"]
    print(f"    {len(mlp_user_ids):,} impressions")

    # ---- Build gallery/probe split (LSTM) ----
    print("\n  Building gallery/probe split (LSTM)...")
    lstm_split = build_gallery_probe_split(
        lstm_user_ids, lstm_reprs, lstm_labels,
        min_impressions=4, gallery_fraction=0.5, seed=42,
    )
    print(f"    Gallery: {lstm_split['n_gallery_users']:,} users")
    print(f"    Probe:   {lstm_split['n_probe_impressions']:,} impressions")

    # ---- Build gallery/probe split (MLP) — same seed for fair comparison ----
    print("\n  Building gallery/probe split (MLP)...")
    mlp_split = build_gallery_probe_split(
        mlp_user_ids, mlp_reprs, mlp_data["val_labels"],
        min_impressions=4, gallery_fraction=0.5, seed=42,
    )

    # ---- Random baseline ----
    baseline = run_random_baseline(
        lstm_split["n_gallery_users"], lstm_split["n_probe_impressions"],
    )
    print(f"\n    Random baseline Top-1: {baseline['top_k_accuracy']['1']*100:.4f}%")

    all_lstm_results = {}
    all_mlp_results = {}

    # ---- LSTM: Euclidean attack ----
    print(f"\n  {'='*50}")
    print(f"  LSTM Attack 1: Euclidean Distance")
    print(f"  {'='*50}")
    start = time.time()
    lstm_eucl = run_reidentification_attack(
        lstm_split["gallery_profiles"], lstm_split["gallery_user_ids"],
        lstm_split["probe_reprs"], lstm_split["probe_user_ids"],
        metric="euclidean", top_k_values=(1, 5, 10, 20), batch_size=2000,
    )
    lstm_eucl["_user_stats"] = lstm_split["user_stats"]
    elapsed = time.time() - start
    print(f"\n    Results ({elapsed:.1f}s):")
    print(f"      Top-1:  {lstm_eucl['top_k_accuracy']['1']*100:.2f}%")
    print(f"      Top-5:  {lstm_eucl['top_k_accuracy']['5']*100:.2f}%")
    print(f"      MRR:    {lstm_eucl['mrr']:.4f}")
    print(f"      Median rank: {lstm_eucl['median_rank']:.0f}")
    all_lstm_results["euclidean"] = lstm_eucl

    # ---- LSTM: Cosine attack ----
    print(f"\n  {'='*50}")
    print(f"  LSTM Attack 2: Cosine Distance")
    print(f"  {'='*50}")
    start = time.time()
    lstm_cos = run_reidentification_attack(
        lstm_split["gallery_profiles"], lstm_split["gallery_user_ids"],
        lstm_split["probe_reprs"], lstm_split["probe_user_ids"],
        metric="cosine", top_k_values=(1, 5, 10, 20), batch_size=2000,
    )
    lstm_cos["_user_stats"] = lstm_split["user_stats"]
    elapsed = time.time() - start
    print(f"\n    Results ({elapsed:.1f}s):")
    print(f"      Top-1:  {lstm_cos['top_k_accuracy']['1']*100:.2f}%")
    print(f"      Top-5:  {lstm_cos['top_k_accuracy']['5']*100:.2f}%")
    print(f"      MRR:    {lstm_cos['mrr']:.4f}")
    print(f"      Median rank: {lstm_cos['median_rank']:.0f}")
    all_lstm_results["cosine"] = lstm_cos

    # ---- MLP: Run both attacks for comparison ----
    print(f"\n  {'='*50}")
    print(f"  MLP Attacks (for comparison)")
    print(f"  {'='*50}")
    for metric_name in ["euclidean", "cosine"]:
        mlp_res = run_reidentification_attack(
            mlp_split["gallery_profiles"], mlp_split["gallery_user_ids"],
            mlp_split["probe_reprs"], mlp_split["probe_user_ids"],
            metric=metric_name, top_k_values=(1, 5, 10, 20), batch_size=2000,
        )
        mlp_res["_user_stats"] = mlp_split["user_stats"]
        all_mlp_results[metric_name] = mlp_res
        print(f"    MLP {metric_name} Top-1: {mlp_res['top_k_accuracy']['1']*100:.2f}%")

    # ---- Generate per-metric plots for LSTM ----
    print("\n  Generating LSTM attack plots...")
    plot_attack_results(lstm_eucl, baseline, LSTM_DIR, "euclidean", "LSTM")
    plot_attack_results(lstm_cos, baseline, LSTM_DIR, "cosine", "LSTM")

    # ---- LSTM metric comparison (euclidean vs cosine) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = ["1", "5", "10", "20"]
    eucl_vals = [lstm_eucl["top_k_accuracy"][k] * 100 for k in ks]
    cos_vals = [lstm_cos["top_k_accuracy"][k] * 100 for k in ks]
    rand_vals = [baseline["top_k_accuracy"][k] * 100 for k in ks]
    x = np.arange(len(ks))
    w = 0.25
    ax.bar(x - w, eucl_vals, w, label="Euclidean", color="crimson", alpha=0.8)
    ax.bar(x, cos_vals, w, label="Cosine", color="steelblue", alpha=0.8)
    ax.bar(x + w, rand_vals, w, label="Random", color="gray", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("LSTM Re-identification: Euclidean vs Cosine vs Random")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(LSTM_DIR / "reidentification_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved reidentification_comparison.png")

    # ---- LSTM vs MLP comparison plot ----
    print("\n  Generating LSTM vs MLP comparison plot...")

    # Use cosine for both (typically best)
    best_lstm = lstm_cos if lstm_cos["top_k_accuracy"]["1"] >= lstm_eucl["top_k_accuracy"]["1"] else lstm_eucl
    best_mlp = all_mlp_results["cosine"] if all_mlp_results["cosine"]["top_k_accuracy"]["1"] >= all_mlp_results["euclidean"]["top_k_accuracy"]["1"] else all_mlp_results["euclidean"]

    # Load engagement metrics
    lstm_eng_metrics = {}
    mlp_eng_metrics = {}
    lstm_metrics_path = LSTM_DIR / "metrics.json"
    mlp_metrics_path = MLP_DIR / "metrics.json"
    if lstm_metrics_path.exists():
        with open(lstm_metrics_path) as f:
            lstm_eng_metrics = json.load(f).get("final_val_metrics", {})
    if mlp_metrics_path.exists():
        with open(mlp_metrics_path) as f:
            mlp_eng_metrics = json.load(f).get("final_val_metrics", {})

    plot_lstm_vs_mlp_comparison(
        best_lstm, best_mlp, baseline,
        lstm_eng_metrics, mlp_eng_metrics,
        best_lstm["_ranks"], best_mlp["_ranks"],
        LSTM_DIR,
    )

    # ---- Save results ----
    save_results = {"lstm": {}, "mlp": {}}
    for metric_name, res in all_lstm_results.items():
        save_results["lstm"][metric_name] = {k: v for k, v in res.items() if not k.startswith("_")}
    for metric_name, res in all_mlp_results.items():
        save_results["mlp"][metric_name] = {k: v for k, v in res.items() if not k.startswith("_")}
    save_results["baseline"] = baseline
    save_results["split_info"] = {
        "n_gallery_users": lstm_split["n_gallery_users"],
        "n_probe_impressions": lstm_split["n_probe_impressions"],
        "min_impressions": 4,
        "gallery_fraction": 0.5,
    }

    with open(LSTM_DIR / "reidentification_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"    Saved reidentification_results.json")

    # ---- Final summary ----
    lstm_top1 = best_lstm["top_k_accuracy"]["1"]
    mlp_top1 = best_mlp["top_k_accuracy"]["1"]
    random_top1 = baseline["top_k_accuracy"]["1"]
    lstm_lift = lstm_top1 / random_top1 if random_top1 > 0 else float("inf")
    mlp_lift = mlp_top1 / random_top1 if random_top1 > 0 else float("inf")

    print(f"\n{'='*70}")
    print(f"  RE-IDENTIFICATION COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<25s} {'LSTM':>10s} {'MLP':>10s} {'Improvement':>12s}")
    print(f"  {'-'*57}")
    print(f"  {'Engagement AUC':<25s} {lstm_eng_metrics.get('auc', 0):>10.4f} {mlp_eng_metrics.get('auc', 0):>10.4f} {lstm_eng_metrics.get('auc', 0) - mlp_eng_metrics.get('auc', 0):>+12.4f}")
    print(f"  {'Re-id Top-1':<25s} {lstm_top1*100:>9.2f}% {mlp_top1*100:>9.2f}% {(lstm_top1-mlp_top1)*100:>+11.2f}%")
    print(f"  {'Re-id Top-5':<25s} {best_lstm['top_k_accuracy']['5']*100:>9.2f}% {best_mlp['top_k_accuracy']['5']*100:>9.2f}%")
    print(f"  {'Lift over Random':<25s} {lstm_lift:>9.0f}x {mlp_lift:>9.0f}x")
    print(f"  {'MRR':<25s} {best_lstm['mrr']:>10.4f} {best_mlp['mrr']:>10.4f}")
    print(f"  {'Median Rank':<25s} {best_lstm['median_rank']:>10.0f} {best_mlp['median_rank']:>10.0f}")

    print(f"\n  Privacy implication:")
    if lstm_top1 > mlp_top1:
        print(f"    LSTM re-identification is {lstm_top1/max(mlp_top1, 1e-10):.1f}x stronger than MLP.")
        print(f"    Temporal behavioral sequences create dramatically more")
        print(f"    distinctive fingerprints than aggregate statistics alone.")
    else:
        print(f"    LSTM and MLP show comparable re-identification rates.")
        print(f"    Aggregate statistics may capture most of the identifying signal.")


if __name__ == "__main__":
    main()
