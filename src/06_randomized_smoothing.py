"""
06_randomized_smoothing.py — Privacy-Utility Tradeoff via Randomized Smoothing
================================================================================
Sweeps Gaussian noise levels on learned 64-dim representations, measuring:
  - Utility: How much engagement prediction degrades (AUC, F1)
  - Privacy: How much re-identification weakens (Top-K, MRR, lift)
  - Certification: Formal radius guarantees (Cohen et al., 2019)

Runs on both MLP and LSTM models, producing the project's core
privacy defense results and comparison plots.

Outputs (to outputs/models/smoothing/):
  - smoothing_results.json: All numerical results
  - comparison/privacy_utility_tradeoff.png: Main deliverable
  - comparison/reid_decay.png: Re-id accuracy vs sigma
  - comparison/auc_degradation.png: AUC vs sigma
  - comparison/certification_coverage.png: Certified fraction vs sigma
  - comparison/smoothing_summary.png: 2x2 summary grid
  - {model}/certified_radii.png: Radius distributions per model
  - {model}/recommended_sigma_detail.png: Detail at recommended sigma

Usage:
    source .venv/bin/activate
    python src/06_randomized_smoothing.py
    python src/06_randomized_smoothing.py --mlp-only    # if LSTM not yet trained
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.smoothing import (
    extract_head_weights,
    analytical_smoothed_predict,
    compute_certified_radii,
    compute_nn_distances,
    noisy_reidentification,
    monte_carlo_certify,
    run_sigma_sweep,
)
from models.attack import run_random_baseline

# ======================================================================
# Configuration
# ======================================================================

SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_REID_TRIALS = 10
REID_METRIC = "cosine"
MC_N_SAMPLES = 1000
MC_ALPHA = 0.001
MC_MAX_POINTS = 3000

MLP_DIR = PROJECT_ROOT / "outputs" / "models" / "mlp_baseline"
LSTM_DIR = PROJECT_ROOT / "outputs" / "models" / "lstm"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "smoothing"


# ======================================================================
# Visualization functions
# ======================================================================

def plot_privacy_utility_tradeoff(all_results: dict, baseline: dict, output_dir: Path):
    """Plot 1: The main deliverable — AUC vs Re-id lift at each sigma."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    random_top1 = baseline["top_k_accuracy"]["1"]

    for model_name, color, marker in [("mlp", "steelblue", "o"), ("lstm", "crimson", "s")]:
        if model_name not in all_results:
            continue
        results = all_results[model_name]["sweep"]
        sigmas = [r["sigma"] for r in results]
        aucs = [r["utility"]["auc"] for r in results]
        lifts = [r["privacy"]["top_k_accuracy"]["1"] / random_top1
                 if random_top1 > 0 else 0 for r in results]

        label_upper = model_name.upper()
        ax1.plot(sigmas, aucs, f"-{marker}", color=color, linewidth=2,
                 markersize=7, label=f"{label_upper} AUC", alpha=0.9)
        ax2.plot(sigmas, lifts, f"--{marker}", color=color, linewidth=2,
                 markersize=7, label=f"{label_upper} Re-id Lift", alpha=0.6)

    ax1.set_xlabel("Noise Level (σ)", fontsize=12)
    ax1.set_ylabel("Engagement AUC (↑ better)", fontsize=12, color="black")
    ax2.set_ylabel("Re-identification Lift over Random (↓ safer)", fontsize=12, color="gray")

    ax1.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3, label="Random AUC")
    ax2.axhline(y=1.0, color="green", linestyle=":", alpha=0.4, label="Random re-id")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title("Privacy-Utility Tradeoff: Randomized Smoothing\n"
                   "Adding Gaussian noise to 64-dim representations",
                   fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / "privacy_utility_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved privacy_utility_tradeoff.png")


def plot_reid_decay(all_results: dict, baseline: dict, output_dir: Path):
    """Plot 2: Re-identification accuracy decay as sigma increases."""
    fig, ax = plt.subplots(figsize=(9, 5))

    random_top1 = baseline["top_k_accuracy"]["1"]

    for model_name, color, marker in [("mlp", "steelblue", "o"), ("lstm", "crimson", "s")]:
        if model_name not in all_results:
            continue
        results = all_results[model_name]["sweep"]
        sigmas = [r["sigma"] for r in results]
        label_upper = model_name.upper()

        for k_str, ls in [("1", "-"), ("5", "--"), ("10", ":")]:
            vals = [r["privacy"]["top_k_accuracy"][k_str] * 100 for r in results]
            ax.plot(sigmas, vals, f"{ls}{marker}", color=color, linewidth=2,
                    markersize=5, label=f"{label_upper} Top-{k_str}", alpha=0.8)

    # Random baselines
    for k_str, ls in [("1", "-"), ("5", "--"), ("10", ":")]:
        rv = baseline["top_k_accuracy"][k_str] * 100
        ax.axhline(y=rv, color="gray", linestyle=ls, alpha=0.3)

    ax.set_xlabel("Noise Level (σ)", fontsize=12)
    ax.set_ylabel("Re-identification Accuracy (%)", fontsize=12)
    ax.set_title("Re-identification Accuracy Decay Under Noise", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    fig.savefig(output_dir / "reid_decay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved reid_decay.png")


def plot_auc_degradation(all_results: dict, output_dir: Path):
    """Plot 3: AUC degradation as sigma increases."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, color, marker in [("mlp", "steelblue", "o"), ("lstm", "crimson", "s")]:
        if model_name not in all_results:
            continue
        results = all_results[model_name]["sweep"]
        sigmas = [r["sigma"] for r in results]
        aucs = [r["utility"]["auc"] for r in results]
        label_upper = model_name.upper()

        ax.plot(sigmas, aucs, f"-{marker}", color=color, linewidth=2.5,
                markersize=8, label=f"{label_upper}", alpha=0.9)

        # Annotate sigma=0 baseline
        ax.annotate(f"{aucs[0]:.4f}", xy=(sigmas[0], aucs[0]),
                    xytext=(sigmas[0] + 0.1, aucs[0] + 0.005),
                    fontsize=9, color=color, fontweight="bold")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.axhline(y=0.6, color="orange", linestyle="--", alpha=0.4, label="Usability floor (0.6)")

    ax.set_xlabel("Noise Level (σ)", fontsize=12)
    ax.set_ylabel("Engagement AUC-ROC", fontsize=12)
    ax.set_title("Engagement Prediction Degradation Under Noise", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "auc_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved auc_degradation.png")


def plot_certified_radii(results: list, nn_info: dict, model_name: str,
                         representations: np.ndarray, w: np.ndarray, b: float,
                         output_dir: Path):
    """Plot 4: Certified radius distributions at selected sigma values."""
    selected_sigmas = [s for s in [0.1, 0.5, 1.0, 2.0] if any(r["sigma"] == s for r in results)]

    n_plots = len(selected_sigmas)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
    if n_plots == 1:
        axes = [axes]

    # Use euclidean NN distances for comparison with L2 certified radii
    median_nn_l2 = nn_info.get("euclidean", {}).get("median", nn_info.get("median", 0))
    label_upper = model_name.upper()

    for ax, sigma in zip(axes, selected_sigmas):
        result = next(r for r in results if r["sigma"] == sigma)
        cert = result["certification"]

        # Recompute radii for histogram
        radii_data = compute_certified_radii(representations, w, b, sigma)
        radii = radii_data["radii"]

        ax.hist(radii, bins=50, color="teal", alpha=0.7, edgecolor="white", density=True)
        if median_nn_l2 > 0:
            ax.axvline(x=median_nn_l2, color="red", linestyle="--", linewidth=2,
                       label=f"Median NN (L2)={median_nn_l2:.2f}")
        ax.axvline(x=cert["median_radius"], color="orange", linestyle="-", linewidth=2,
                   label=f"Median R={cert['median_radius']:.2f}")

        ax.set_xlabel("Certified Radius (L2)")
        ax.set_ylabel("Density")
        ax.set_title(f"σ = {sigma}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Certified Radius Distribution — {label_upper}\n"
                 f"Red dashed = median L2 NN distance between users",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "certified_radii.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved certified_radii.png ({label_upper})")


def plot_certification_coverage(all_results: dict, nn_info_all: dict, output_dir: Path):
    """Plot 5: Fraction of samples certified above NN distance threshold."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model_name, color, marker in [("mlp", "steelblue", "o"), ("lstm", "crimson", "s")]:
        if model_name not in all_results:
            continue
        results = all_results[model_name]["sweep"]
        sigmas = [r["sigma"] for r in results]
        label_upper = model_name.upper()

        frac_cert = [r["certification"]["frac_certified"] * 100 for r in results]
        frac_1 = [r["certification"]["frac_above_1"] * 100 for r in results]
        frac_2 = [r["certification"]["frac_above_2"] * 100 for r in results]

        ax.plot(sigmas, frac_cert, f"-{marker}", color=color, linewidth=2,
                markersize=6, label=f"{label_upper} R > 0", alpha=0.9)
        ax.plot(sigmas, frac_1, f"--{marker}", color=color, linewidth=1.5,
                markersize=5, label=f"{label_upper} R > 1.0", alpha=0.7)
        ax.plot(sigmas, frac_2, f":{marker}", color=color, linewidth=1.5,
                markersize=5, label=f"{label_upper} R > 2.0", alpha=0.6)

    ax.set_xlabel("Noise Level (σ)", fontsize=12)
    ax.set_ylabel("Fraction of Samples Certified (%)", fontsize=12)
    ax.set_title("Certification Coverage vs Noise Level", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "certification_coverage.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved certification_coverage.png")


def plot_smoothing_summary(all_results: dict, baseline: dict, nn_info_all: dict,
                           output_dir: Path):
    """Plot 6: Comprehensive 2x2 summary grid."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Randomized Smoothing: Privacy Defense Summary",
                 fontsize=15, fontweight="bold")

    random_top1 = baseline["top_k_accuracy"]["1"]

    # Panel 1: Privacy-utility tradeoff
    ax = axes[0, 0]
    for model_name, color, marker in [("mlp", "steelblue", "o"), ("lstm", "crimson", "s")]:
        if model_name not in all_results:
            continue
        results = all_results[model_name]["sweep"]
        aucs = [r["utility"]["auc"] for r in results]
        lifts = [r["privacy"]["top_k_accuracy"]["1"] / random_top1
                 if random_top1 > 0 else 0 for r in results]
        label_upper = model_name.upper()
        ax.scatter(aucs, lifts, c=color, marker=marker, s=60, label=label_upper, zorder=3)
        for i, r in enumerate(results):
            if r["sigma"] in [0.0, 0.1, 0.5, 1.0, 2.0]:
                ax.annotate(f"σ={r['sigma']}", (aucs[i], lifts[i]),
                            fontsize=7, xytext=(5, 5),
                            textcoords="offset points", color=color)
        ax.plot(aucs, lifts, "-", color=color, alpha=0.4, linewidth=1)

    ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.4)
    ax.set_xlabel("Engagement AUC")
    ax.set_ylabel("Re-id Lift over Random (log)")
    ax.set_title("Privacy-Utility Pareto Frontier")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Top-K at σ=0 vs recommended σ (grouped bar)
    ax = axes[0, 1]
    bar_data = {}
    for model_name in ["mlp", "lstm"]:
        if model_name not in all_results:
            continue
        results = all_results[model_name]["sweep"]
        clean = next((r for r in results if r["sigma"] == 0.0), results[0])
        # Find sigma where lift first drops below 5x
        recommended = None
        for r in results:
            lift = r["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0
            if lift < 5.0 and r["sigma"] > 0:
                recommended = r
                break
        if recommended is None:
            recommended = results[-1]
        bar_data[model_name] = {"clean": clean, "noisy": recommended}

    if bar_data:
        ks = ["1", "5", "10", "20"]
        x = np.arange(len(ks))
        w = 0.18
        offset = 0
        for model_name, color in [("mlp", "steelblue"), ("lstm", "crimson")]:
            if model_name not in bar_data:
                continue
            label_upper = model_name.upper()
            clean_vals = [bar_data[model_name]["clean"]["privacy"]["top_k_accuracy"][k] * 100
                          for k in ks]
            noisy_vals = [bar_data[model_name]["noisy"]["privacy"]["top_k_accuracy"][k] * 100
                          for k in ks]
            noisy_sigma = bar_data[model_name]["noisy"]["sigma"]
            ax.bar(x + offset, clean_vals, w, label=f"{label_upper} σ=0", color=color, alpha=0.85)
            ax.bar(x + offset + w, noisy_vals, w,
                   label=f"{label_upper} σ={noisy_sigma}", color=color, alpha=0.4)
            offset += 2 * w + 0.05

        ax.set_xticks(x + offset / 2 - w / 2)
        ax.set_xticklabels([f"Top-{k}" for k in ks])
        ax.set_ylabel("Re-id Accuracy (%)")
        ax.set_title("Re-identification: Clean vs Smoothed")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: AUC degradation
    ax = axes[1, 0]
    for model_name, color, marker in [("mlp", "steelblue", "o"), ("lstm", "crimson", "s")]:
        if model_name not in all_results:
            continue
        results = all_results[model_name]["sweep"]
        sigmas = [r["sigma"] for r in results]
        aucs = [r["utility"]["auc"] for r in results]
        label_upper = model_name.upper()
        ax.plot(sigmas, aucs, f"-{marker}", color=color, linewidth=2, markersize=6,
                label=label_upper)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("σ")
    ax.set_ylabel("AUC")
    ax.set_title("Engagement Prediction vs Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Summary text
    ax = axes[1, 1]
    ax.axis("off")

    lines = ["RANDOMIZED SMOOTHING SUMMARY", "=" * 45, ""]
    for model_name in ["mlp", "lstm"]:
        if model_name not in all_results:
            continue
        label_upper = model_name.upper()
        results = all_results[model_name]["sweep"]
        clean = next((r for r in results if r["sigma"] == 0.0), results[0])
        nn_median = nn_info_all.get(model_name, {}).get("median", 0)

        lines.append(f"{label_upper}:")
        lines.append(f"  Clean AUC:       {clean['utility']['auc']:.4f}")
        lines.append(f"  Clean Re-id Top1:{clean['privacy']['top_k_accuracy']['1']*100:.2f}%")
        lines.append(f"  Clean Lift:      "
                      f"{clean['privacy']['top_k_accuracy']['1']/random_top1:.0f}x")
        nn_median_l2 = nn_info_all.get(model_name, {}).get("euclidean", {}).get("median", 0)
        lines.append(f"  NN median (cos):  {nn_median:.4f}")
        lines.append(f"  NN median (L2):   {nn_median_l2:.4f}")
        lines.append("")

        # Find crossover points
        for r in results:
            lift = r["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0
            if lift < 2.0 and r["sigma"] > 0:
                lines.append(f"  σ_privacy (lift<2x): {r['sigma']}")
                lines.append(f"    AUC at this σ:     {r['utility']['auc']:.4f}")
                break

        for r in results:
            if r["utility"]["auc"] < 0.6 and r["sigma"] > 0:
                lines.append(f"  σ_utility (AUC<0.6): {r['sigma']}")
                break
        lines.append("")

    summary = "\n".join(lines)
    ax.text(0.02, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "smoothing_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved smoothing_summary.png")


def plot_recommended_detail(results: list, baseline: dict, nn_info: dict,
                            model_name: str, output_dir: Path):
    """Plot 7: Detailed view at the recommended sigma."""
    random_top1 = baseline["top_k_accuracy"]["1"]
    label_upper = model_name.upper()

    # Find recommended sigma: first sigma where lift < 5x
    recommended = None
    for r in results:
        if r["sigma"] == 0:
            continue
        lift = r["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0
        if lift < 5.0:
            recommended = r
            break
    if recommended is None:
        recommended = results[-1]

    clean = next((r for r in results if r["sigma"] == 0.0), results[0])
    rec_sigma = recommended["sigma"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{label_upper}: Recommended σ = {rec_sigma} — Detailed Analysis",
                 fontsize=14, fontweight="bold")

    # 1. AUC trajectory
    ax = axes[0, 0]
    sigmas = [r["sigma"] for r in results]
    aucs = [r["utility"]["auc"] for r in results]
    ax.plot(sigmas, aucs, "-o", color="steelblue", linewidth=2, markersize=6)
    ax.axvline(x=rec_sigma, color="red", linestyle="--", alpha=0.7,
               label=f"Recommended σ={rec_sigma}")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("σ")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Re-id Top-K comparison (clean vs recommended)
    ax = axes[0, 1]
    ks = ["1", "5", "10", "20"]
    clean_vals = [clean["privacy"]["top_k_accuracy"][k] * 100 for k in ks]
    noisy_vals = [recommended["privacy"]["top_k_accuracy"][k] * 100 for k in ks]
    rand_vals = [baseline["top_k_accuracy"][k] * 100 for k in ks]
    x = np.arange(len(ks))
    w = 0.25
    ax.bar(x - w, clean_vals, w, label="Clean (σ=0)", color="crimson", alpha=0.85)
    ax.bar(x, noisy_vals, w, label=f"Smoothed (σ={rec_sigma})", color="orange", alpha=0.85)
    ax.bar(x + w, rand_vals, w, label="Random", color="gray", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Top-{k}" for k in ks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Re-identification: Clean vs Smoothed")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Lift trajectory
    ax = axes[0, 2]
    lifts = [r["privacy"]["top_k_accuracy"]["1"] / random_top1
             if random_top1 > 0 else 0 for r in results]
    ax.plot(sigmas, lifts, "-s", color="crimson", linewidth=2, markersize=6)
    ax.axvline(x=rec_sigma, color="red", linestyle="--", alpha=0.7)
    ax.axhline(y=1.0, color="green", linestyle=":", alpha=0.4, label="Random (1x)")
    ax.set_xlabel("σ")
    ax.set_ylabel("Lift over Random")
    ax.set_title("Re-identification Lift Decay")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Certification at recommended sigma
    ax = axes[1, 0]
    cert = recommended["certification"]
    metrics_list = [
        ("Mean R", cert["mean_radius"]),
        ("Median R", cert["median_radius"]),
        ("Max R", cert["max_radius"]),
        ("Certified %", cert["frac_certified"] * 100),
        ("R > 1.0 %", cert["frac_above_1"] * 100),
        ("R > 2.0 %", cert["frac_above_2"] * 100),
    ]
    names = [m[0] for m in metrics_list]
    vals = [m[1] for m in metrics_list]
    bars = ax.barh(names, vals, color="teal", alpha=0.7)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.set_title(f"Certification at σ={rec_sigma}")
    ax.grid(True, alpha=0.3, axis="x")

    # 5. All metrics comparison
    ax = axes[1, 1]
    metrics_names = ["AUC", "F1", "Accuracy", "Precision", "Recall"]
    metrics_keys = ["auc", "f1", "accuracy", "precision", "recall"]
    clean_m = [clean["utility"][k] for k in metrics_keys]
    noisy_m = [recommended["utility"][k] for k in metrics_keys]
    x = np.arange(len(metrics_names))
    w = 0.35
    ax.bar(x - w/2, clean_m, w, label="Clean", color="steelblue", alpha=0.85)
    ax.bar(x + w/2, noisy_m, w, label=f"σ={rec_sigma}", color="orange", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Score")
    ax.set_title("Utility Metrics: Clean vs Smoothed")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Summary text
    ax = axes[1, 2]
    ax.axis("off")
    clean_lift = clean["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0
    noisy_lift = recommended["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0

    summary = (
        f"RECOMMENDED σ = {rec_sigma} — {label_upper}\n"
        f"{'='*42}\n\n"
        f"{'Metric':<22s} {'Clean':>8s} {'Smooth':>8s} {'Change':>8s}\n"
        f"{'-'*42}\n"
        f"{'AUC':<22s} {clean['utility']['auc']:>8.4f} {recommended['utility']['auc']:>8.4f} "
        f"{recommended['utility']['auc']-clean['utility']['auc']:>+8.4f}\n"
        f"{'F1':<22s} {clean['utility']['f1']:>8.4f} {recommended['utility']['f1']:>8.4f} "
        f"{recommended['utility']['f1']-clean['utility']['f1']:>+8.4f}\n"
        f"{'Re-id Top-1':<22s} {clean['privacy']['top_k_accuracy']['1']*100:>7.2f}% "
        f"{recommended['privacy']['top_k_accuracy']['1']*100:>7.2f}% "
        f"{(recommended['privacy']['top_k_accuracy']['1']-clean['privacy']['top_k_accuracy']['1'])*100:>+7.2f}%\n"
        f"{'Lift':<22s} {clean_lift:>7.0f}x {noisy_lift:>7.0f}x\n"
        f"{'MRR':<22s} {clean['privacy']['mrr']:>8.4f} {recommended['privacy']['mrr']:>8.4f}\n"
        f"{'-'*42}\n"
        f"{'Certified R (mean)':<22s} {cert['mean_radius']:>8.4f}\n"
        f"{'Certified R (median)':<22s} {cert['median_radius']:>8.4f}\n"
        f"{'NN median dist':<22s} {nn_info.get('median', 0):>8.4f}\n\n"
        f"Privacy reduction:\n"
        f"  Re-id lift dropped {clean_lift:.0f}x → {noisy_lift:.1f}x\n"
        f"  ({(1 - noisy_lift/clean_lift)*100:.1f}% reduction)\n"
        f"AUC cost:\n"
        f"  {clean['utility']['auc']:.4f} → {recommended['utility']['auc']:.4f}\n"
        f"  ({(clean['utility']['auc']-recommended['utility']['auc'])*100:.2f}% absolute drop)"
    )
    ax.text(0.02, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "recommended_sigma_detail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved recommended_sigma_detail.png ({label_upper})")


# ======================================================================
# Main
# ======================================================================

def run_model(model_name: str, model_dir: Path, output_subdir: Path) -> dict:
    """Run the full sigma sweep for one model."""
    label_upper = model_name.upper()
    repr_path = model_dir / "representations.npz"
    ckpt_path = model_dir / "checkpoint.pt"

    if not repr_path.exists():
        print(f"  [SKIP] {label_upper}: {repr_path} not found")
        return {}

    if not ckpt_path.exists():
        print(f"  [SKIP] {label_upper}: {ckpt_path} not found")
        return {}

    print(f"\n{'='*60}")
    print(f"  {label_upper} — Randomized Smoothing Sweep")
    print(f"{'='*60}")

    # Load representations
    print(f"\n  Loading {label_upper} representations...")
    data = np.load(repr_path)
    val_reprs = data["val_representations"]
    val_labels = data["val_labels"]
    val_user_ids = data["val_user_ids"]
    print(f"    {len(val_user_ids):,} impressions, dim={val_reprs.shape[1]}")

    # Extract head weights
    print(f"\n  Extracting classification head weights...")
    w, b = extract_head_weights(ckpt_path)

    # Compute NN distances (for sigma calibration)
    print(f"\n  Computing nearest-neighbor distances...")
    nn_info = compute_nn_distances(val_user_ids, val_reprs, metric=REID_METRIC)

    # Run sigma sweep
    print(f"\n  Running sigma sweep ({len(SIGMA_VALUES)} values)...")
    sweep = run_sigma_sweep(
        val_reprs, val_labels, val_user_ids, w, b,
        sigma_values=SIGMA_VALUES,
        reid_metric=REID_METRIC,
        n_reid_trials=N_REID_TRIALS,
        verbose=True,
    )

    # MC certification at a few sigmas (verification)
    print(f"\n  Running Monte Carlo certification (verification)...")
    mc_results = {}
    for sigma in [0.5, 1.0]:
        if sigma in SIGMA_VALUES:
            mc = monte_carlo_certify(
                val_reprs, val_labels, w, b, sigma,
                n_samples=MC_N_SAMPLES, alpha=MC_ALPHA,
                max_points=MC_MAX_POINTS,
            )
            mc_results[str(sigma)] = {k: v for k, v in mc.items()
                                       if not isinstance(v, np.ndarray)}
            print(f"    σ={sigma}: MC mean R={mc['mc_mean_radius']:.4f}, "
                  f"abstain={mc['mc_abstain_rate']*100:.1f}%")

    # Save per-model results
    output_subdir.mkdir(parents=True, exist_ok=True)
    model_results = {
        "model": model_name,
        "nn_distances": {k: v for k, v in nn_info.items()
                         if not isinstance(v, np.ndarray)},
        "sweep": [{k: v for k, v in r.items()} for r in sweep],
        "mc_verification": mc_results,
        "config": {
            "sigma_values": SIGMA_VALUES,
            "n_reid_trials": N_REID_TRIALS,
            "reid_metric": REID_METRIC,
            "mc_n_samples": MC_N_SAMPLES,
            "mc_alpha": MC_ALPHA,
        },
        # Raw numpy data for plotting (not serialized to JSON)
        "_raw": {
            "representations": val_reprs,
            "w": w,
            "b": b,
        },
    }

    return model_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp-only", action="store_true",
                        help="Run only on MLP (skip LSTM if not ready)")
    parser.add_argument("--lstm-only", action="store_true",
                        help="Run only on LSTM")
    args = parser.parse_args()

    print("=" * 70)
    print("  Phase 4: Randomized Smoothing — Privacy-Utility Tradeoff")
    print("=" * 70)
    print(f"\n  Sigma values: {SIGMA_VALUES}")
    print(f"  Re-id trials per sigma: {N_REID_TRIALS}")
    print(f"  Re-id metric: {REID_METRIC}")

    all_results = {}
    nn_info_all = {}

    # Create output dirs
    comparison_dir = OUTPUT_DIR / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Run MLP
    if not args.lstm_only:
        mlp_out = OUTPUT_DIR / "mlp"
        mlp_results = run_model("mlp", MLP_DIR, mlp_out)
        if mlp_results:
            all_results["mlp"] = mlp_results
            nn_info_all["mlp"] = mlp_results["nn_distances"]

    # Run LSTM
    if not args.mlp_only:
        lstm_out = OUTPUT_DIR / "lstm"
        lstm_results = run_model("lstm", LSTM_DIR, lstm_out)
        if lstm_results:
            all_results["lstm"] = lstm_results
            nn_info_all["lstm"] = lstm_results["nn_distances"]

    if not all_results:
        print("\n  No models found. Exiting.")
        return

    # Random baseline (same for both models — same gallery size)
    first_model = list(all_results.values())[0]
    n_gallery = first_model["sweep"][0]["privacy"]["n_gallery_users"]
    n_probes = first_model["sweep"][0]["privacy"]["n_probes"]
    baseline = run_random_baseline(n_gallery, n_probes)

    # Generate comparison plots
    print(f"\n{'='*60}")
    print(f"  Generating Plots")
    print(f"{'='*60}")

    print("\n  Comparison plots...")
    plot_privacy_utility_tradeoff(all_results, baseline, comparison_dir)
    plot_reid_decay(all_results, baseline, comparison_dir)
    plot_auc_degradation(all_results, comparison_dir)
    plot_certification_coverage(all_results, nn_info_all, comparison_dir)
    plot_smoothing_summary(all_results, baseline, nn_info_all, comparison_dir)

    # Per-model plots
    for model_name in all_results:
        model_dir = OUTPUT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        nn_info = nn_info_all.get(model_name, {})
        raw = all_results[model_name].get("_raw", {})
        print(f"\n  {model_name.upper()} plots...")
        plot_certified_radii(
            all_results[model_name]["sweep"], nn_info,
            model_name, raw.get("representations", np.array([])),
            raw.get("w", np.array([])), raw.get("b", 0.0),
            model_dir,
        )
        plot_recommended_detail(all_results[model_name]["sweep"], baseline,
                                nn_info, model_name, model_dir)

    # Save all results to JSON (exclude _raw numpy data)
    save_results = {}
    for model_name, res in all_results.items():
        model_save = {
            "nn_distances": res["nn_distances"],
            "sweep": [],
            "mc_verification": res.get("mc_verification", {}),
            "config": res["config"],
        }
        for r in res["sweep"]:
            r_clean = {
                "sigma": r["sigma"],
                "utility": r["utility"],
                "certification": r["certification"],
                "privacy": {k: v for k, v in r["privacy"].items()
                           if not isinstance(v, np.ndarray)},
            }
            model_save["sweep"].append(r_clean)
        save_results[model_name] = model_save

    save_results["baseline"] = baseline
    save_results["config"] = {
        "sigma_values": SIGMA_VALUES,
        "n_reid_trials": N_REID_TRIALS,
        "reid_metric": REID_METRIC,
    }

    results_path = OUTPUT_DIR / "smoothing_results.json"
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n    Saved {results_path}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  RANDOMIZED SMOOTHING COMPLETE")
    print(f"{'='*70}")

    random_top1 = baseline["top_k_accuracy"]["1"]
    for model_name in all_results:
        label_upper = model_name.upper()
        results = all_results[model_name]["sweep"]
        clean = next((r for r in results if r["sigma"] == 0.0), results[0])
        clean_lift = clean["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0

        print(f"\n  {label_upper}:")
        print(f"    Clean: AUC={clean['utility']['auc']:.4f}, "
              f"Re-id Lift={clean_lift:.0f}x")

        # Find crossover
        for r in results:
            if r["sigma"] == 0:
                continue
            lift = r["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0
            if lift < 5.0:
                print(f"    σ={r['sigma']}: AUC={r['utility']['auc']:.4f}, "
                      f"Re-id Lift={lift:.1f}x (privacy viable)")
                break

        for r in results:
            if r["sigma"] == 0:
                continue
            lift = r["privacy"]["top_k_accuracy"]["1"] / random_top1 if random_top1 > 0 else 0
            if lift < 2.0:
                print(f"    σ={r['sigma']}: AUC={r['utility']['auc']:.4f}, "
                      f"Re-id Lift={lift:.1f}x (near-random privacy)")
                break

    print(f"\n  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
