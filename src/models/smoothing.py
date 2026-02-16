"""
smoothing.py — Randomized Smoothing for Privacy Protection
============================================================
Implements representation-level Gaussian noise injection with:
  1. Analytical smoothed prediction (exact for linear classification head)
  2. Certified robustness radii (Cohen et al., ICML 2019)
  3. Monte Carlo smoothed prediction with Clopper-Pearson bounds
  4. Noisy re-identification attack evaluation
  5. Nearest-neighbor distance computation for sigma calibration

Mathematical framework:
  Given a 64-dim representation r and a linear head f(r) = sigmoid(w·r + b),
  adding Gaussian noise ε ~ N(0, σ²I) gives:

    P[engaged | r, σ] = Φ((w·r + b) / (σ·‖w‖))

  where Φ is the standard normal CDF. This is exact — no Monte Carlo needed.

  The certified radius (L2) within which the prediction is stable:
    R = σ · Φ⁻¹(p_A)    where p_A = max(p, 1-p)

  Privacy interpretation:
    Expected noise displacement: E[‖ε‖] ≈ σ·√d ≈ 8σ  (for d=64)
    If this exceeds nearest-neighbor distances in representation space,
    user fingerprints overlap and re-identification fails.

Reference:
  Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing"
  ICML 2019. https://proceedings.mlr.press/v97/cohen19c.html
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import norm, beta as beta_dist
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    average_precision_score,
)
from sklearn.metrics import pairwise_distances


# ---------------------------------------------------------------------------
# 1. Head weight extraction
# ---------------------------------------------------------------------------

def extract_head_weights(checkpoint_path: Path) -> tuple:
    """
    Load a trained checkpoint and extract classification head weights.

    The head is Linear(64, 1) for both MLP and LSTM models:
        logit = w · r + b

    Args:
        checkpoint_path: Path to checkpoint.pt

    Returns:
        (w, b) where w is (64,) numpy array and b is scalar
    """
    import torch
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if "head.weight" in state:
        w = state["head.weight"].squeeze().numpy()   # (64,)
        b = state["head.bias"].squeeze().item()       # scalar
    elif "model_state_dict" in state:
        sd = state["model_state_dict"]
        w = sd["head.weight"].squeeze().numpy()
        b = sd["head.bias"].squeeze().item()
    else:
        raise KeyError(
            f"Cannot find head weights in checkpoint. "
            f"Available keys: {list(state.keys())[:20]}"
        )

    print(f"    Head weights: w shape={w.shape}, ‖w‖={np.linalg.norm(w):.4f}, b={b:.4f}")
    return w, b


# ---------------------------------------------------------------------------
# 2. Analytical smoothed prediction (exact for linear head)
# ---------------------------------------------------------------------------

def analytical_smoothed_predict(
    representations: np.ndarray,
    labels: np.ndarray,
    w: np.ndarray,
    b: float,
    sigma: float,
) -> dict:
    """
    Compute exact smoothed engagement probabilities for a linear head.

    Given r ∈ R^64 and noise ε ~ N(0, σ²I):
        P[sigmoid(w·(r+ε) + b) > 0.5] = P[w·r + b + w·ε > 0]
        = Φ((w·r + b) / (σ·‖w‖))

    At σ=0, this reduces to sigmoid(w·r + b) > 0.5 (standard prediction).

    Args:
        representations: (N, 64) representation vectors
        labels: (N,) ground-truth binary labels
        w: (64,) classification head weight
        b: scalar classification head bias
        sigma: Gaussian noise standard deviation

    Returns:
        Dict with smoothed AUC, F1, accuracy, precision, recall,
        avg_precision, and raw smoothed probabilities.
    """
    logits = representations @ w + b  # (N,)

    if sigma <= 0:
        probs = 1.0 / (1.0 + np.exp(-logits))  # standard sigmoid
    else:
        noise_std = sigma * np.linalg.norm(w)
        probs = norm.cdf(logits / noise_std)

    preds = (probs > 0.5).astype(int)
    labels_int = labels.astype(int)

    metrics = {
        "auc": float(roc_auc_score(labels_int, probs)),
        "f1": float(f1_score(labels_int, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels_int, preds)),
        "precision": float(precision_score(labels_int, preds, zero_division=0)),
        "recall": float(recall_score(labels_int, preds, zero_division=0)),
        "avg_precision": float(average_precision_score(labels_int, probs)),
        "pos_rate_pred": float(preds.mean()),
        "mean_prob": float(probs.mean()),
        "_probs": probs,
    }

    return metrics


# ---------------------------------------------------------------------------
# 3. Certified radius computation
# ---------------------------------------------------------------------------

def compute_certified_radii(
    representations: np.ndarray,
    w: np.ndarray,
    b: float,
    sigma: float,
) -> dict:
    """
    Compute the certified L2 radius for each sample.

    For binary classification with a linear head:
        p = Φ((w·r + b) / (σ·‖w‖))
        p_A = max(p, 1-p)
        R = σ · Φ⁻¹(p_A)

    The certified radius R guarantees that g(r + δ) = g(r)
    for all ‖δ‖₂ ≤ R, where g is the smoothed classifier.

    Args:
        representations: (N, 64)
        w: (64,) head weight
        b: scalar head bias
        sigma: noise level

    Returns:
        Dict with per-sample radii, summary statistics, and
        fraction of samples certified at various thresholds.
    """
    if sigma <= 0:
        n = len(representations)
        return {
            "radii": np.zeros(n),
            "mean_radius": 0.0,
            "median_radius": 0.0,
            "max_radius": 0.0,
            "frac_certified": 0.0,
            "frac_above_1": 0.0,
            "frac_above_2": 0.0,
            "frac_above_5": 0.0,
        }

    logits = representations @ w + b
    noise_std = sigma * np.linalg.norm(w)

    p = norm.cdf(logits / noise_std)        # P(engaged)
    p_A = np.maximum(p, 1.0 - p)            # majority class probability

    radii = sigma * norm.ppf(p_A)            # certified L2 radius
    radii = np.maximum(radii, 0.0)           # clip numerical artifacts

    return {
        "radii": radii,
        "mean_radius": float(radii.mean()),
        "median_radius": float(np.median(radii)),
        "max_radius": float(radii.max()),
        "std_radius": float(radii.std()),
        "frac_certified": float((radii > 0).mean()),
        "frac_above_1": float((radii > 1.0).mean()),
        "frac_above_2": float((radii > 2.0).mean()),
        "frac_above_5": float((radii > 5.0).mean()),
    }


# ---------------------------------------------------------------------------
# 4. Nearest-neighbor distances in representation space
# ---------------------------------------------------------------------------

def compute_nn_distances(
    user_ids: np.ndarray,
    representations: np.ndarray,
    metric: str = "cosine",
    max_users: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Compute nearest-neighbor distances between user profiles.

    Builds per-user mean representations, then computes pairwise
    distances. Returns the distribution of 1-NN distances — the
    critical reference for calibrating sigma.

    Args:
        user_ids: (N,) user IDs
        representations: (N, 64)
        metric: distance metric for comparison
        max_users: subsample if too many users (memory)
        seed: random seed for subsampling

    Returns:
        Dict with NN distance statistics and the full distance array.
    """
    user_to_indices = defaultdict(list)
    for i, uid in enumerate(user_ids):
        user_to_indices[uid].append(i)

    # Build per-user mean profiles
    uids = []
    profiles = []
    for uid, indices in user_to_indices.items():
        if len(indices) < 2:
            continue
        profile = representations[indices].mean(axis=0)
        profiles.append(profile)
        uids.append(uid)

    profiles = np.array(profiles)
    n_users = len(profiles)

    # Subsample if too many users for pairwise computation
    if n_users > max_users:
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_users, max_users, replace=False)
        profiles = profiles[idx]
        n_users = max_users
        print(f"    Subsampled to {n_users:,} users for NN distance computation")

    print(f"    Computing pairwise {metric} distances for {n_users:,} user profiles...")
    start = time.time()
    dists = pairwise_distances(profiles, metric=metric)

    # Set diagonal to infinity so a user isn't its own nearest neighbor
    np.fill_diagonal(dists, np.inf)

    nn_dists = dists.min(axis=1)  # nearest-neighbor distance per user
    elapsed = time.time() - start

    print(f"    Done in {elapsed:.1f}s")
    print(f"    NN distances: mean={nn_dists.mean():.4f}, "
          f"median={np.median(nn_dists):.4f}, "
          f"min={nn_dists.min():.4f}, max={nn_dists.max():.4f}")

    return {
        "nn_distances": nn_dists,
        "mean": float(nn_dists.mean()),
        "median": float(np.median(nn_dists)),
        "std": float(nn_dists.std()),
        "min": float(nn_dists.min()),
        "max": float(nn_dists.max()),
        "p05": float(np.percentile(nn_dists, 5)),
        "p25": float(np.percentile(nn_dists, 25)),
        "p75": float(np.percentile(nn_dists, 75)),
        "p95": float(np.percentile(nn_dists, 95)),
        "n_users": n_users,
        "metric": metric,
    }


# ---------------------------------------------------------------------------
# 5. Noisy re-identification attack
# ---------------------------------------------------------------------------

def noisy_reidentification(
    user_ids: np.ndarray,
    representations: np.ndarray,
    labels: np.ndarray,
    sigma: float,
    metric: str = "cosine",
    n_trials: int = 10,
    min_impressions: int = 4,
    gallery_fraction: float = 0.5,
    top_k_values: tuple = (1, 5, 10, 20),
    batch_size: int = 2000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run re-identification attack on noise-perturbed representations.

    Averages over n_trials random noise draws for stable estimates.
    At sigma=0, this reproduces the clean (no-noise) attack exactly.

    The noise is added BEFORE the gallery/probe split, simulating
    a deployment where representations are noised before storage/release.

    Args:
        user_ids: (N,) user IDs
        representations: (N, 64)
        labels: (N,) engagement labels
        sigma: Gaussian noise std (0 = no noise)
        metric: distance metric for attack
        n_trials: number of noise draws to average over
        min_impressions: minimum impressions for gallery inclusion
        gallery_fraction: fraction of impressions for gallery
        top_k_values: which top-K accuracies to compute
        batch_size: probe batch size for pairwise distances
        seed: base random seed
        verbose: print progress

    Returns:
        Dict with averaged attack metrics across trials.
    """
    from models.attack import build_gallery_probe_split, run_reidentification_attack

    if sigma <= 0:
        n_trials = 1  # no noise = deterministic

    all_trial_results = []
    rng = np.random.RandomState(seed)

    for trial in range(n_trials):
        if sigma > 0:
            noise = rng.normal(0, sigma, representations.shape)
            noisy_reprs = representations + noise
        else:
            noisy_reprs = representations

        split = build_gallery_probe_split(
            user_ids, noisy_reprs, labels,
            min_impressions=min_impressions,
            gallery_fraction=gallery_fraction,
            seed=42,
        )

        result = run_reidentification_attack(
            split["gallery_profiles"],
            split["gallery_user_ids"],
            split["probe_reprs"],
            split["probe_user_ids"],
            metric=metric,
            top_k_values=top_k_values,
            batch_size=batch_size,
        )

        trial_metrics = {
            "top_k_accuracy": result["top_k_accuracy"],
            "mrr": result["mrr"],
            "mean_rank": result["mean_rank"],
            "median_rank": result["median_rank"],
            "per_user_accuracy_mean": result["per_user_accuracy_mean"],
            "n_gallery_users": result["n_gallery_users"],
            "n_probes": result["n_probes"],
        }
        all_trial_results.append(trial_metrics)

        if verbose and (trial == 0 or (trial + 1) % 5 == 0):
            t1 = trial_metrics["top_k_accuracy"]["1"]
            print(f"      Trial {trial+1}/{n_trials}: Top-1={t1*100:.2f}%")

    # Average across trials
    avg = {}
    for key in ["mrr", "mean_rank", "median_rank", "per_user_accuracy_mean"]:
        avg[key] = float(np.mean([t[key] for t in all_trial_results]))

    avg["top_k_accuracy"] = {}
    for k_str in all_trial_results[0]["top_k_accuracy"]:
        vals = [t["top_k_accuracy"][k_str] for t in all_trial_results]
        avg["top_k_accuracy"][k_str] = float(np.mean(vals))
        avg[f"top_k_{k_str}_std"] = float(np.std(vals))

    avg["n_gallery_users"] = all_trial_results[0]["n_gallery_users"]
    avg["n_probes"] = all_trial_results[0]["n_probes"]
    avg["n_trials"] = n_trials
    avg["sigma"] = sigma

    return avg


# ---------------------------------------------------------------------------
# 6. Monte Carlo smoothed prediction with Clopper-Pearson bounds
# ---------------------------------------------------------------------------

def _clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """Lower bound of Clopper-Pearson confidence interval for k successes in n trials."""
    if k == 0:
        return 0.0
    return float(beta_dist.ppf(alpha / 2, k, n - k + 1))


def monte_carlo_certify(
    representations: np.ndarray,
    labels: np.ndarray,
    w: np.ndarray,
    b: float,
    sigma: float,
    n_samples: int = 1000,
    alpha: float = 0.001,
    seed: int = 42,
    max_points: int = 5000,
) -> dict:
    """
    Monte Carlo certification with Clopper-Pearson confidence bounds.

    For each sample, draws n_samples noise vectors, evaluates the
    base classifier, and computes the certified radius using a
    one-sided binomial confidence interval.

    This is slower than the analytical method but provides rigorous
    confidence-bounded certification and serves as a verification.

    Args:
        representations: (N, 64)
        labels: (N,) ground-truth labels
        w: (64,) head weight
        b: scalar head bias
        sigma: noise level
        n_samples: Monte Carlo samples per point
        alpha: confidence level for Clopper-Pearson (1-alpha confidence)
        seed: random seed
        max_points: subsample if dataset is large (MC is expensive)

    Returns:
        Dict with MC certified radii and comparison with analytical.
    """
    rng = np.random.RandomState(seed)
    N = len(representations)

    if N > max_points:
        idx = rng.choice(N, max_points, replace=False)
        representations = representations[idx]
        labels = labels[idx]
        N = max_points

    print(f"    MC certification: {N:,} points × {n_samples:,} samples, σ={sigma}")

    mc_radii = np.zeros(N)
    mc_predictions = np.zeros(N, dtype=int)
    mc_abstain = np.zeros(N, dtype=bool)

    start = time.time()
    for i in range(N):
        r = representations[i]  # (64,)
        noise = rng.normal(0, sigma, (n_samples, len(r)))
        noisy_logits = (r + noise) @ w + b     # (n_samples,)
        noisy_preds = (noisy_logits > 0).astype(int)

        count_1 = noisy_preds.sum()
        count_0 = n_samples - count_1

        if count_1 > count_0:
            mc_predictions[i] = 1
            count_A = count_1
        else:
            mc_predictions[i] = 0
            count_A = count_0

        # Clopper-Pearson lower bound on p_A
        p_A_lower = _clopper_pearson_lower(count_A, n_samples, alpha)

        if p_A_lower > 0.5:
            mc_radii[i] = sigma * norm.ppf(p_A_lower)
        else:
            mc_radii[i] = 0.0
            mc_abstain[i] = True

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(f"      {i+1:,}/{N:,} done ({elapsed:.1f}s)")

    elapsed = time.time() - start

    mc_preds_labels = labels.astype(int)
    mc_acc = float((mc_predictions == mc_preds_labels).mean())
    mc_correct_radii = mc_radii[mc_predictions == mc_preds_labels]

    return {
        "mc_radii": mc_radii,
        "mc_predictions": mc_predictions,
        "mc_abstain_rate": float(mc_abstain.mean()),
        "mc_accuracy": mc_acc,
        "mc_mean_radius": float(mc_radii.mean()),
        "mc_median_radius": float(np.median(mc_radii)),
        "mc_frac_certified": float((mc_radii > 0).mean()),
        "mc_mean_radius_correct": float(mc_correct_radii.mean()) if len(mc_correct_radii) > 0 else 0.0,
        "n_points": N,
        "n_samples": n_samples,
        "alpha": alpha,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# 7. Sigma sweep orchestrator
# ---------------------------------------------------------------------------

def run_sigma_sweep(
    representations: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    w: np.ndarray,
    b: float,
    sigma_values: list,
    reid_metric: str = "cosine",
    n_reid_trials: int = 10,
    verbose: bool = True,
) -> list:
    """
    Run the full sigma sweep for one model.

    For each sigma, computes:
      - Utility: analytical smoothed AUC, F1, accuracy
      - Privacy: noisy re-identification Top-K, MRR, lift
      - Certification: per-sample certified radii

    Args:
        representations: (N, 64) val representations
        labels: (N,) val labels
        user_ids: (N,) val user IDs
        w: (64,) head weights
        b: scalar head bias
        sigma_values: list of sigma values to sweep
        reid_metric: distance metric for re-id attack
        n_reid_trials: noise draws for re-id averaging
        verbose: print progress

    Returns:
        List of result dicts, one per sigma.
    """
    results = []

    for i, sigma in enumerate(sigma_values):
        if verbose:
            print(f"\n  --- σ = {sigma} ({i+1}/{len(sigma_values)}) ---")

        # Utility
        if verbose:
            print(f"    Computing smoothed predictions...")
        utility = analytical_smoothed_predict(representations, labels, w, b, sigma)
        utility_clean = {k: v for k, v in utility.items() if not k.startswith("_")}

        if verbose:
            print(f"    AUC={utility['auc']:.4f}, F1={utility['f1']:.4f}, "
                  f"Acc={utility['accuracy']:.4f}")

        # Certification
        if verbose:
            print(f"    Computing certified radii...")
        cert = compute_certified_radii(representations, w, b, sigma)
        cert_clean = {k: v for k, v in cert.items() if k != "radii"}

        if verbose:
            print(f"    Mean R={cert['mean_radius']:.4f}, "
                  f"Median R={cert['median_radius']:.4f}, "
                  f"Certified={cert['frac_certified']*100:.1f}%")

        # Privacy (re-identification)
        if verbose:
            print(f"    Running noisy re-identification ({n_reid_trials} trials)...")
        privacy = noisy_reidentification(
            user_ids, representations, labels, sigma,
            metric=reid_metric, n_trials=n_reid_trials,
            verbose=verbose,
        )

        if verbose:
            t1 = privacy["top_k_accuracy"]["1"]
            print(f"    Re-id Top-1={t1*100:.3f}%, MRR={privacy['mrr']:.4f}")

        results.append({
            "sigma": sigma,
            "utility": utility_clean,
            "certification": cert_clean,
            "privacy": privacy,
        })

    return results
