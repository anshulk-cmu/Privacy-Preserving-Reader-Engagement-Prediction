"""
03_reidentification_test.py — Blind User Re-identification Test
================================================================
Tests whether the MLP model's learned representations can identify
individual users from their reading behavior. This is the core
privacy risk demonstration.

Methodology:
  1. Load validation set representations (model never trained on val data)
  2. For users with 4+ impressions, split 50/50 into gallery/probe
  3. Gallery: averaged into one profile vector per user (their "fingerprint")
  4. Probe: individual impressions tested against ALL gallery fingerprints
  5. Attack: nearest-neighbor lookup — can we find the right user?
  6. Both euclidean and cosine distance metrics tested
  7. Random baseline computed for comparison

This is a blind test: the model saw none of these val impressions during
training. If re-identification succeeds, it proves the model has learned
user-distinctive behavioral patterns purely from reading statistics.

Usage:
    conda activate privacy
    python src/03_reidentification_test.py
"""

import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

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


def plot_attack_results(results: dict, baseline: dict, output_dir: Path, metric_name: str):
    """Generate comprehensive re-identification attack visualization."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"User Re-identification Attack — MLP Baseline ({metric_name} distance)\n"
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
    bars2 = ax.bar(x + w/2, baseline_vals, w, label="Random Baseline", color="gray", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Re-identification Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars1, attack_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 2. Rank distribution (CDF)
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

    # 3. Rank histogram (zoomed to top 100)
    ax = axes[0, 2]
    top100_ranks = ranks[ranks <= 100]
    ax.hist(top100_ranks, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(x=1, color="red", linestyle="-", linewidth=2, label=f"Rank 1 (exact match)")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    ax.set_title(f"Rank Distribution (top 100 of {results['n_gallery_users']:,})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Per-user re-identification accuracy distribution
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

    # 5. Re-identification accuracy vs number of impressions
    ax = axes[1, 1]
    user_n_impressions = {}
    for uid, stats in results.get("_user_stats", {}).items():
        if uid in per_user_acc:
            user_n_impressions[uid] = stats["n_total"]

    if user_n_impressions:
        n_imps = [user_n_impressions[uid] for uid in per_user_acc.keys() if uid in user_n_impressions]
        accs = [per_user_acc[uid] for uid in per_user_acc.keys() if uid in user_n_impressions]

        # Bin by impression count
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
        f"RE-IDENTIFICATION RESULTS\n"
        f"{'='*35}\n\n"
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
        f"  ({results['top_k_accuracy']['1'] / baseline['top_k_accuracy']['1']:.0f}x "
        f"above random)"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / f"reidentification_{metric_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved reidentification_{metric_name}.png")


def main():
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "mlp_baseline"
    REPR_PATH = OUTPUT_DIR / "representations.npz"

    print("=" * 70)
    print("  Blind User Re-identification Test — MLP Baseline")
    print("=" * 70)

    # ---- Load representations ----
    print("\n  Loading validation set representations...")
    data = np.load(REPR_PATH)
    val_user_ids = data["val_user_ids"]
    val_reprs = data["val_representations"]
    val_labels = data["val_labels"]
    print(f"    {len(val_user_ids):,} impressions, {len(np.unique(val_user_ids)):,} unique users")
    print(f"    Representation dim: {val_reprs.shape[1]}")

    # ---- Build gallery/probe split ----
    print("\n  Building gallery/probe split...")
    split = build_gallery_probe_split(
        val_user_ids, val_reprs, val_labels,
        min_impressions=4,
        gallery_fraction=0.5,
        seed=42,
    )
    print(f"    Gallery: {split['n_gallery_users']:,} users (one profile each)")
    print(f"    Probe:   {split['n_probe_impressions']:,} impressions")
    print(f"    Users dropped (<4 impressions): {len(np.unique(val_user_ids)) - split['n_gallery_users']:,}")

    # ---- Random baseline ----
    baseline = run_random_baseline(
        split["n_gallery_users"], split["n_probe_impressions"],
    )
    print(f"\n    Random baseline Top-1: {baseline['top_k_accuracy']['1']*100:.4f}%")

    all_results = {}

    # ---- Run attack with euclidean distance ----
    print(f"\n  {'='*50}")
    print(f"  Attack 1: Euclidean Distance")
    print(f"  {'='*50}")
    start = time.time()
    results_eucl = run_reidentification_attack(
        split["gallery_profiles"],
        split["gallery_user_ids"],
        split["probe_reprs"],
        split["probe_user_ids"],
        metric="euclidean",
        top_k_values=(1, 5, 10, 20),
        batch_size=2000,
    )
    results_eucl["_user_stats"] = split["user_stats"]
    elapsed = time.time() - start
    print(f"\n    Euclidean results ({elapsed:.1f}s):")
    print(f"      Top-1:  {results_eucl['top_k_accuracy']['1']*100:.2f}%")
    print(f"      Top-5:  {results_eucl['top_k_accuracy']['5']*100:.2f}%")
    print(f"      Top-10: {results_eucl['top_k_accuracy']['10']*100:.2f}%")
    print(f"      Top-20: {results_eucl['top_k_accuracy']['20']*100:.2f}%")
    print(f"      MRR:    {results_eucl['mrr']:.4f}")
    print(f"      Mean rank:   {results_eucl['mean_rank']:.1f} / {split['n_gallery_users']:,}")
    print(f"      Median rank: {results_eucl['median_rank']:.0f}")
    all_results["euclidean"] = results_eucl

    # ---- Run attack with cosine distance ----
    print(f"\n  {'='*50}")
    print(f"  Attack 2: Cosine Distance")
    print(f"  {'='*50}")
    start = time.time()
    results_cos = run_reidentification_attack(
        split["gallery_profiles"],
        split["gallery_user_ids"],
        split["probe_reprs"],
        split["probe_user_ids"],
        metric="cosine",
        top_k_values=(1, 5, 10, 20),
        batch_size=2000,
    )
    results_cos["_user_stats"] = split["user_stats"]
    elapsed = time.time() - start
    print(f"\n    Cosine results ({elapsed:.1f}s):")
    print(f"      Top-1:  {results_cos['top_k_accuracy']['1']*100:.2f}%")
    print(f"      Top-5:  {results_cos['top_k_accuracy']['5']*100:.2f}%")
    print(f"      Top-10: {results_cos['top_k_accuracy']['10']*100:.2f}%")
    print(f"      Top-20: {results_cos['top_k_accuracy']['20']*100:.2f}%")
    print(f"      MRR:    {results_cos['mrr']:.4f}")
    print(f"      Mean rank:   {results_cos['mean_rank']:.1f} / {split['n_gallery_users']:,}")
    print(f"      Median rank: {results_cos['median_rank']:.0f}")
    all_results["cosine"] = results_cos

    # ---- Generate plots ----
    print("\n  Generating plots...")
    plot_attack_results(results_eucl, baseline, OUTPUT_DIR, "euclidean")
    plot_attack_results(results_cos, baseline, OUTPUT_DIR, "cosine")

    # ---- Comparison plot ----
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = ["1", "5", "10", "20"]
    eucl_vals = [results_eucl["top_k_accuracy"][k] * 100 for k in ks]
    cos_vals = [results_cos["top_k_accuracy"][k] * 100 for k in ks]
    rand_vals = [baseline["top_k_accuracy"][k] * 100 for k in ks]

    x = np.arange(len(ks))
    w = 0.25
    ax.bar(x - w, eucl_vals, w, label="Euclidean", color="crimson", alpha=0.8)
    ax.bar(x, cos_vals, w, label="Cosine", color="steelblue", alpha=0.8)
    ax.bar(x + w, rand_vals, w, label="Random", color="gray", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Re-identification: Euclidean vs Cosine vs Random")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for bars, vals in [(ax.patches[:4], eucl_vals), (ax.patches[4:8], cos_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "reidentification_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved reidentification_comparison.png")

    # ---- Save results (strip numpy arrays for JSON) ----
    save_results = {}
    for metric_name, res in all_results.items():
        save_results[metric_name] = {
            k: v for k, v in res.items() if not k.startswith("_")
        }
    save_results["baseline"] = baseline
    save_results["split_info"] = {
        "n_gallery_users": split["n_gallery_users"],
        "n_probe_impressions": split["n_probe_impressions"],
        "min_impressions": 4,
        "gallery_fraction": 0.5,
    }

    with open(OUTPUT_DIR / "reidentification_results.json", "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"    Saved reidentification_results.json")

    # ---- Final summary ----
    best_metric = "euclidean" if results_eucl["top_k_accuracy"]["1"] >= results_cos["top_k_accuracy"]["1"] else "cosine"
    best = all_results[best_metric]
    random_top1 = baseline["top_k_accuracy"]["1"]
    attack_top1 = best["top_k_accuracy"]["1"]
    lift = attack_top1 / random_top1 if random_top1 > 0 else float("inf")

    print(f"\n{'='*70}")
    print(f"  RE-IDENTIFICATION TEST COMPLETE")
    print(f"{'='*70}")
    print(f"  Best metric:     {best_metric}")
    print(f"  Top-1 accuracy:  {attack_top1*100:.2f}% (random: {random_top1*100:.4f}%)")
    print(f"  Lift over random: {lift:.0f}x")
    print(f"  Top-5 accuracy:  {best['top_k_accuracy']['5']*100:.2f}%")
    print(f"  Top-10 accuracy: {best['top_k_accuracy']['10']*100:.2f}%")
    print(f"  MRR:             {best['mrr']:.4f}")
    print(f"  Median rank:     {best['median_rank']:.0f} / {split['n_gallery_users']:,}")
    print(f"\n  Privacy implication:")
    if attack_top1 > 0.01:
        print(f"    The MLP model's representations enable user re-identification")
        print(f"    at {lift:.0f}x above random chance. This demonstrates that even")
        print(f"    aggregate reading statistics create distinctive behavioral")
        print(f"    fingerprints that constitute a real privacy risk.")
    else:
        print(f"    Re-identification rate is very low — aggregate features")
        print(f"    alone may not create strong behavioral fingerprints.")


if __name__ == "__main__":
    main()
