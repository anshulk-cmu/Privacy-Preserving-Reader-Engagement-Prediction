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
from models.attack import _gpu_pairwise_distances


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
    p_A = np.clip(p_A, 0.5, 1.0 - 1e-12)    # avoid norm.ppf(0) or norm.ppf(1) = ±inf

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

    Computes distances in both the requested metric AND euclidean,
    since certified radii (Cohen et al.) are L2 while re-identification
    may use cosine distance. Both are needed for sigma calibration.

    Args:
        user_ids: (N,) user IDs
        representations: (N, 64)
        metric: primary distance metric for re-identification comparison
        max_users: subsample if too many users (memory)
        seed: random seed for subsampling

    Returns:
        Dict with NN distance statistics for both metric and euclidean.
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

    def _compute_for_metric(m: str) -> dict:
        start = time.time()
        dists = _gpu_pairwise_distances(profiles, metric=m)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)
        elapsed = time.time() - start
        print(f"    {m}: mean={nn_dists.mean():.4f}, "
              f"median={np.median(nn_dists):.4f}, "
              f"min={nn_dists.min():.4f}, max={nn_dists.max():.4f} ({elapsed:.1f}s)")
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
        }

    print(f"    Computing pairwise distances for {n_users:,} user profiles...")

    # Primary metric (for re-identification comparison)
    primary = _compute_for_metric(metric)

    # Euclidean (for certified radius comparison — radii are L2)
    if metric != "euclidean":
        euclidean = _compute_for_metric("euclidean")
    else:
        euclidean = primary

    return {
        # Primary metric results (top-level for backward compatibility)
        **primary,
        "n_users": n_users,
        "metric": metric,
        # Euclidean results (for certified radius comparison)
        "euclidean": {k: v for k, v in euclidean.items() if k != "nn_distances"},
        "euclidean_nn_distances": euclidean["nn_distances"],
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
            p_A_lower = min(p_A_lower, 1.0 - 1e-12)  # avoid norm.ppf(1) = +inf
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
# 7. Monte Carlo utility evaluation (deployment-realistic noise injection)
# ---------------------------------------------------------------------------

def monte_carlo_utility(
    representations: np.ndarray,
    labels: np.ndarray,
    w: np.ndarray,
    b: float,
    sigma: float,
    n_trials: int = 50,
    seed: int = 42,
) -> dict:
    """
    Measure deployment-realistic utility under actual Gaussian noise injection.

    For each trial, draws fresh noise ε ~ N(0, σ²I_d), computes
    noisy_logits = (r + ε) @ w + b, and evaluates classification metrics.
    This is Scenario B from the revised evaluation — single-draw noise.

    Unlike analytical_smoothed_predict(), which preserves AUC by construction
    (monotonic transform tautology), this measures genuine degradation because
    each sample's noise draw independently perturbs its logit.

    Args:
        representations: (N, 64) representation vectors
        labels: (N,) ground-truth binary labels
        w: (64,) classification head weight
        b: scalar classification head bias
        sigma: Gaussian noise standard deviation (0 = clean)
        n_trials: number of independent noise draws to average over
        seed: random seed

    Returns:
        Dict with per-trial and aggregated metrics:
        auc_mean, auc_std, auc_median, auc_p5, auc_p95,
        f1_mean, f1_std, accuracy_mean, accuracy_std,
        precision_mean, recall_mean, avg_precision_mean, n_trials.
    """
    if sigma <= 0:
        # No noise — deterministic, return clean metrics as single trial
        logits = representations @ w + b
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        labels_int = labels.astype(int)
        return {
            "auc_mean": float(roc_auc_score(labels_int, probs)),
            "auc_std": 0.0,
            "auc_median": float(roc_auc_score(labels_int, probs)),
            "auc_p5": float(roc_auc_score(labels_int, probs)),
            "auc_p95": float(roc_auc_score(labels_int, probs)),
            "f1_mean": float(f1_score(labels_int, preds, zero_division=0)),
            "f1_std": 0.0,
            "accuracy_mean": float(accuracy_score(labels_int, preds)),
            "accuracy_std": 0.0,
            "precision_mean": float(precision_score(labels_int, preds, zero_division=0)),
            "precision_std": 0.0,
            "recall_mean": float(recall_score(labels_int, preds, zero_division=0)),
            "recall_std": 0.0,
            "avg_precision_mean": float(average_precision_score(labels_int, probs)),
            "avg_precision_std": 0.0,
            "n_trials": 1,
        }

    rng = np.random.RandomState(seed)
    labels_int = labels.astype(int)
    N, d = representations.shape
    w_norm = np.linalg.norm(w)

    trial_aucs = []
    trial_f1s = []
    trial_accs = []
    trial_precs = []
    trial_recs = []
    trial_aps = []

    for _ in range(n_trials):
        noise = rng.normal(0, sigma, (N, d))
        noisy_logits = (representations + noise) @ w + b
        noisy_probs = 1.0 / (1.0 + np.exp(-noisy_logits))
        noisy_preds = (noisy_probs > 0.5).astype(int)

        trial_aucs.append(float(roc_auc_score(labels_int, noisy_probs)))
        trial_f1s.append(float(f1_score(labels_int, noisy_preds, zero_division=0)))
        trial_accs.append(float(accuracy_score(labels_int, noisy_preds)))
        trial_precs.append(float(precision_score(labels_int, noisy_preds, zero_division=0)))
        trial_recs.append(float(recall_score(labels_int, noisy_preds, zero_division=0)))
        trial_aps.append(float(average_precision_score(labels_int, noisy_probs)))

    aucs = np.array(trial_aucs)
    f1s = np.array(trial_f1s)
    accs = np.array(trial_accs)
    precs = np.array(trial_precs)
    recs = np.array(trial_recs)
    aps = np.array(trial_aps)

    return {
        "auc_mean": float(aucs.mean()),
        "auc_std": float(aucs.std()),
        "auc_median": float(np.median(aucs)),
        "auc_p5": float(np.percentile(aucs, 5)),
        "auc_p95": float(np.percentile(aucs, 95)),
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std()),
        "accuracy_mean": float(accs.mean()),
        "accuracy_std": float(accs.std()),
        "precision_mean": float(precs.mean()),
        "precision_std": float(precs.std()),
        "recall_mean": float(recs.mean()),
        "recall_std": float(recs.std()),
        "avg_precision_mean": float(aps.mean()),
        "avg_precision_std": float(aps.std()),
        "n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# 8. Logit statistics and SNR analysis
# ---------------------------------------------------------------------------

def extract_logit_statistics(
    representations: np.ndarray,
    labels: np.ndarray,
    w: np.ndarray,
    b: float,
) -> dict:
    """
    Compute logit distribution statistics for SNR analysis.

    The signal-to-noise ratio SNR(σ) = std(logits) / (σ·‖w‖) predicts
    how much noise affects predictions. Class-conditional statistics
    (μ₊, μ₋, σ₊, σ₋) enable the bi-Gaussian AUC approximation.

    Args:
        representations: (N, 64)
        labels: (N,) binary labels
        w: (64,) head weight
        b: scalar head bias

    Returns:
        Dict with ‖w‖, b, overall and class-conditional logit statistics.
    """
    logits = representations @ w + b
    labels_int = labels.astype(int)
    w_norm = float(np.linalg.norm(w))

    pos_mask = labels_int == 1
    neg_mask = labels_int == 0
    logits_pos = logits[pos_mask]
    logits_neg = logits[neg_mask]

    return {
        "w_norm": w_norm,
        "b": float(b),
        "logit_mean": float(logits.mean()),
        "logit_std": float(logits.std()),
        "logit_min": float(logits.min()),
        "logit_max": float(logits.max()),
        "logit_median": float(np.median(logits)),
        # Class-conditional statistics
        "mu_pos": float(logits_pos.mean()),
        "mu_neg": float(logits_neg.mean()),
        "std_pos": float(logits_pos.std()),
        "std_neg": float(logits_neg.std()),
        "delta_mu": float(logits_pos.mean() - logits_neg.mean()),
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
        # Derived quantities
        "snr_at_sigma_1": float(logits.std() / w_norm) if w_norm > 0 else float("inf"),
        # Raw logits for plotting (not serialized to JSON)
        "_logits": logits,
        "_logits_pos": logits_pos,
        "_logits_neg": logits_neg,
    }


# ---------------------------------------------------------------------------
# 9. Bi-Gaussian analytical AUC prediction
# ---------------------------------------------------------------------------

def analytical_auc_prediction(
    logit_stats: dict,
    sigma_values: list,
) -> dict:
    """
    Predict MC AUC analytically using the bi-Gaussian model.

    Models the logit distributions for positive and negative classes as
    Gaussian, then computes the expected AUC under additive noise:

        AUC(σ) ≈ Φ(Δμ / √(σ₊² + σ₋² + 2σ²‖w‖²))

    This is the standard bi-normal AUC formula with noise variance added.

    Args:
        logit_stats: output of extract_logit_statistics()
        sigma_values: list of sigma values

    Returns:
        Dict mapping sigma → predicted AUC, plus the formula parameters.
    """
    delta_mu = logit_stats["delta_mu"]
    std_pos = logit_stats["std_pos"]
    std_neg = logit_stats["std_neg"]
    w_norm = logit_stats["w_norm"]

    predicted = {}
    for sigma in sigma_values:
        noise_var = sigma * sigma * w_norm * w_norm
        denom = np.sqrt(std_pos**2 + std_neg**2 + 2 * noise_var)
        if denom > 0:
            predicted_auc = float(norm.cdf(delta_mu / denom))
        else:
            predicted_auc = float(norm.cdf(float("inf") if delta_mu > 0 else float("-inf")))
        predicted[str(sigma)] = predicted_auc

    return {
        "predicted_auc": predicted,
        "delta_mu": delta_mu,
        "std_pos": std_pos,
        "std_neg": std_neg,
        "w_norm": w_norm,
    }


# ---------------------------------------------------------------------------
# 10. Multi-draw aggregation experiment (Scenario C)
# ---------------------------------------------------------------------------

def aggregated_prediction_utility(
    representations: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    w: np.ndarray,
    b: float,
    sigma_values: list,
    m_values: list,
    n_trials: int = 20,
    n_reid_trials: int = 5,
    reid_metric: str = "cosine",
    seed: int = 42,
    verbose: bool = True,
) -> list:
    """
    Map the 2D (σ, M) tradeoff surface for multi-draw aggregation.

    For M draws per query:
      - Utility: averaged sigmoid scores → AUC improves with M
      - Privacy: averaged representations (effective noise σ/√M) → privacy degrades with M

    This exposes the fundamental utility-privacy coupling that single-draw
    evaluation misses.

    Args:
        representations: (N, 64)
        labels: (N,) binary labels
        user_ids: (N,) user IDs for re-id attack
        w: (64,) head weight
        b: scalar head bias
        sigma_values: list of sigma values to test
        m_values: list of M (number of draws) values
        n_trials: number of outer trials for utility averaging
        n_reid_trials: number of trials for re-id attack
        reid_metric: distance metric for re-id
        seed: random seed
        verbose: print progress

    Returns:
        List of dicts, one per (σ, M) pair, with utility and privacy metrics.
    """
    from models.attack import build_gallery_probe_split, run_reidentification_attack

    rng = np.random.RandomState(seed)
    labels_int = labels.astype(int)
    N, d = representations.shape
    results = []

    for sigma in sigma_values:
        for M in m_values:
            if verbose:
                print(f"    Aggregation: σ={sigma}, M={M}")

            # --- Utility: averaged scores ---
            trial_aucs = []
            for t in range(n_trials):
                scores_sum = np.zeros(N)
                for m in range(M):
                    noise = rng.normal(0, sigma, (N, d))
                    noisy_logits = (representations + noise) @ w + b
                    scores_sum += 1.0 / (1.0 + np.exp(-noisy_logits))
                avg_scores = scores_sum / M
                trial_aucs.append(float(roc_auc_score(labels_int, avg_scores)))

            auc_arr = np.array(trial_aucs)

            # --- Privacy: re-id on averaged representations ---
            # Effective noise is σ/√M
            reid_lifts = []
            reid_top1s = []

            for t in range(n_reid_trials):
                rep_sum = np.zeros_like(representations)
                for m in range(M):
                    noise = rng.normal(0, sigma, (N, d))
                    rep_sum += representations + noise
                avg_reps = rep_sum / M  # effective noise std = σ/√M

                split = build_gallery_probe_split(
                    user_ids, avg_reps, labels,
                    min_impressions=4,
                    gallery_fraction=0.5,
                    seed=42,
                )
                result = run_reidentification_attack(
                    split["gallery_profiles"],
                    split["gallery_user_ids"],
                    split["probe_reprs"],
                    split["probe_user_ids"],
                    metric=reid_metric,
                    top_k_values=(1, 5, 10),
                    batch_size=2000,
                )
                reid_top1s.append(result["top_k_accuracy"]["1"])

            results.append({
                "sigma": sigma,
                "M": M,
                "effective_sigma": sigma / np.sqrt(M),
                "auc_mean": float(auc_arr.mean()),
                "auc_std": float(auc_arr.std()),
                "reid_top1_mean": float(np.mean(reid_top1s)),
                "reid_top1_std": float(np.std(reid_top1s)),
                "n_utility_trials": n_trials,
                "n_reid_trials": n_reid_trials,
            })

            if verbose:
                print(f"      AUC={auc_arr.mean():.4f}±{auc_arr.std():.4f}, "
                      f"Re-id Top1={np.mean(reid_top1s)*100:.3f}%")

    return results


# ---------------------------------------------------------------------------
# 11. Sigma sweep orchestrator
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
    mc_utility_trials: int = 50,
    verbose: bool = True,
) -> list:
    """
    Run the full sigma sweep for one model.

    For each sigma, computes:
      - Utility (analytical): smoothed AUC — upper bound (tautologically preserved)
      - Utility (MC): deployment-realistic AUC under actual noise injection
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
        mc_utility_trials: noise draws for MC utility evaluation
        verbose: print progress

    Returns:
        List of result dicts, one per sigma.
    """
    results = []

    for i, sigma in enumerate(sigma_values):
        if verbose:
            print(f"\n  --- σ = {sigma} ({i+1}/{len(sigma_values)}) ---")

        # Analytical utility (upper bound — tautologically preserves AUC)
        if verbose:
            print(f"    Computing analytical smoothed predictions (upper bound)...")
        utility = analytical_smoothed_predict(representations, labels, w, b, sigma)
        utility_clean = {k: v for k, v in utility.items() if not k.startswith("_")}

        # MC utility (deployment-realistic — actual noise injection)
        if verbose:
            print(f"    Computing MC utility ({mc_utility_trials} trials)...")
        mc_util = monte_carlo_utility(
            representations, labels, w, b, sigma,
            n_trials=mc_utility_trials,
        )

        if verbose:
            print(f"    Analytical AUC={utility['auc']:.4f} (upper bound), "
                  f"MC AUC={mc_util['auc_mean']:.4f}±{mc_util['auc_std']:.4f}")

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
            "utility_analytical": utility_clean,
            "utility_mc": mc_util,
            # Keep "utility" key pointing to MC for backward compat in plots
            "utility": {
                "auc": mc_util["auc_mean"],
                "f1": mc_util["f1_mean"],
                "accuracy": mc_util["accuracy_mean"],
                "precision": mc_util["precision_mean"],
                "recall": mc_util["recall_mean"],
                "avg_precision": mc_util["avg_precision_mean"],
            },
            "certification": cert_clean,
            "privacy": privacy,
        })

    return results
