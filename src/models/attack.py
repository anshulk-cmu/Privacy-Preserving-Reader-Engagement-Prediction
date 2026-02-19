"""
attack.py — User Re-identification Attack Module
==================================================
Implements a nearest-neighbor re-identification attack to quantify
how well model representations can identify individual users.

Attack methodology:
  1. For each user with N impressions, split into:
     - Gallery (first half): aggregated into a single "user profile" vector
     - Probe (second half): individual impressions to test against gallery
  2. For each probe impression, find the K nearest gallery profiles
  3. If the correct user is in the top-K, the attack succeeds

Metrics:
  - Top-1 Accuracy: Exact match (probe's nearest gallery = same user)
  - Top-5 Accuracy: Correct user in top-5 nearest
  - Top-10 Accuracy: Correct user in top-10 nearest
  - MRR: Mean Reciprocal Rank (1/rank of correct user, averaged)
  - Per-user accuracy: What % of a user's probes correctly re-identify them
"""

import time
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# GPU-accelerated pairwise distance (drop-in replacement for sklearn)
# ---------------------------------------------------------------------------

def _gpu_pairwise_distances(X, Y=None, metric="cosine", batch_size=2048):
    """
    Compute pairwise distances on GPU using PyTorch.

    Drop-in replacement for sklearn.metrics.pairwise_distances.
    Supports 'cosine' and 'euclidean' metrics.

    Args:
        X: (N, D) array or tensor
        Y: (M, D) array or tensor. If None, computes X vs X.
        metric: 'cosine' or 'euclidean'
        batch_size: Process X in chunks to avoid GPU OOM.

    Returns:
        (N, M) numpy array of distances.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    if Y is None:
        Y_t = X_t
    else:
        Y_t = torch.as_tensor(Y, dtype=torch.float32, device=device)

    if metric == "cosine":
        # Normalize once upfront
        X_norm = torch.nn.functional.normalize(X_t, dim=1)
        Y_norm = torch.nn.functional.normalize(Y_t, dim=1)

        # Compute in batches to avoid OOM
        N = X_norm.shape[0]
        rows = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            # cosine distance = 1 - cosine_similarity
            sim = X_norm[start:end] @ Y_norm.T
            rows.append((1.0 - sim).cpu())
        dists = torch.cat(rows, dim=0).numpy()
    elif metric == "euclidean":
        N = X_t.shape[0]
        rows = []
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            rows.append(torch.cdist(X_t[start:end], Y_t).cpu())
        dists = torch.cat(rows, dim=0).numpy()
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'euclidean'.")

    return dists


def build_gallery_probe_split(
    user_ids: np.ndarray,
    representations: np.ndarray,
    labels: np.ndarray,
    min_impressions: int = 4,
    gallery_fraction: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Split each user's impressions into gallery and probe sets.

    Args:
        user_ids: (N,) user IDs for each impression
        representations: (N, D) model representations
        labels: (N,) engagement labels
        min_impressions: Minimum impressions needed for a user to be included
        gallery_fraction: Fraction of impressions used for gallery (rest = probe)
        seed: Random seed for reproducible splitting

    Returns:
        Dict with gallery_profiles, gallery_user_ids, probe_reprs,
        probe_user_ids, probe_labels, and statistics.
    """
    rng = np.random.RandomState(seed)

    # Group by user
    user_to_indices = defaultdict(list)
    for i, uid in enumerate(user_ids):
        user_to_indices[uid].append(i)

    gallery_profiles = []   # (n_users, D) — mean representation per user
    gallery_user_ids = []
    probe_reprs = []        # individual probe impressions
    probe_user_ids = []
    probe_labels = []
    user_stats = {}

    for uid, indices in user_to_indices.items():
        if len(indices) < min_impressions:
            continue

        indices = np.array(indices)
        rng.shuffle(indices)

        n_gallery = max(1, int(len(indices) * gallery_fraction))
        gallery_idx = indices[:n_gallery]
        probe_idx = indices[n_gallery:]

        if len(probe_idx) == 0:
            continue

        # Gallery: mean representation across gallery impressions
        gallery_profile = representations[gallery_idx].mean(axis=0)
        gallery_profiles.append(gallery_profile)
        gallery_user_ids.append(uid)

        # Probe: individual impressions
        for pi in probe_idx:
            probe_reprs.append(representations[pi])
            probe_user_ids.append(uid)
            probe_labels.append(labels[pi])

        user_stats[uid] = {
            "n_total": len(indices),
            "n_gallery": len(gallery_idx),
            "n_probe": len(probe_idx),
        }

    result = {
        "gallery_profiles": np.array(gallery_profiles),
        "gallery_user_ids": np.array(gallery_user_ids),
        "probe_reprs": np.array(probe_reprs),
        "probe_user_ids": np.array(probe_user_ids),
        "probe_labels": np.array(probe_labels),
        "user_stats": user_stats,
        "n_gallery_users": len(gallery_user_ids),
        "n_probe_impressions": len(probe_reprs),
    }

    return result


def run_reidentification_attack(
    gallery_profiles: np.ndarray,
    gallery_user_ids: np.ndarray,
    probe_reprs: np.ndarray,
    probe_user_ids: np.ndarray,
    metric: str = "euclidean",
    top_k_values: tuple = (1, 5, 10, 20),
    batch_size: int = 10000,
) -> dict:
    """
    Run the nearest-neighbor re-identification attack (GPU-accelerated).

    For each probe, finds the nearest gallery profile(s) and checks
    whether the correct user is among the top-K matches. Uses GPU for
    both pairwise distance computation and argsort, with vectorized
    rank-finding (no Python loops over probes).

    Args:
        gallery_profiles: (G, D) gallery user profiles
        gallery_user_ids: (G,) gallery user IDs
        probe_reprs: (P, D) probe impression representations
        probe_user_ids: (P,) ground-truth user IDs for probes
        metric: Distance metric ('euclidean' or 'cosine')
        top_k_values: Which top-K accuracies to compute
        batch_size: Process probes in batches for memory efficiency

    Returns:
        Dict with top-K accuracies, MRR, per-probe ranks, and per-user accuracy.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_probes = len(probe_reprs)
    n_gallery = len(gallery_profiles)
    max_k = max(top_k_values)

    # Build vectorized ground-truth gallery index for each probe
    uid_to_gallery_idx = {uid: i for i, uid in enumerate(gallery_user_ids)}
    true_gallery_indices = np.array(
        [uid_to_gallery_idx.get(uid, -1) for uid in probe_user_ids],
        dtype=np.int64,
    )

    all_ranks = np.full(n_probes, n_gallery + 1, dtype=np.int64)
    reciprocal_ranks = np.zeros(n_probes)

    print(f"    Running attack: {n_probes:,} probes vs {n_gallery:,} gallery users")
    print(f"    Distance metric: {metric}")

    start = time.time()
    for batch_start in range(0, n_probes, batch_size):
        batch_end = min(batch_start + batch_size, n_probes)
        batch_probes = probe_reprs[batch_start:batch_end]
        batch_true_idx = true_gallery_indices[batch_start:batch_end]
        bs = batch_end - batch_start

        # GPU: compute distances (bs, G)
        dists = _gpu_pairwise_distances(batch_probes, gallery_profiles, metric=metric)

        # GPU: argsort to get ranked gallery indices
        dists_t = torch.as_tensor(dists, device=device)
        sorted_indices = torch.argsort(dists_t, dim=1)  # (bs, G)

        # GPU: vectorized rank finding
        # For each probe, find where its true gallery index appears in the sorted order
        true_idx_t = torch.as_tensor(batch_true_idx, device=device).unsqueeze(1)  # (bs, 1)
        match_mask = (sorted_indices == true_idx_t)  # (bs, G) boolean
        # argmax on a boolean tensor returns the index of the first True
        rank_positions = match_mask.int().argmax(dim=1)  # (bs,)
        ranks = (rank_positions + 1).cpu().numpy()  # 1-indexed

        # Handle probes with no matching gallery user (true_idx == -1)
        valid_mask = batch_true_idx >= 0
        ranks[~valid_mask] = n_gallery + 1

        all_ranks[batch_start:batch_end] = ranks
        reciprocal_ranks[batch_start:batch_end] = np.where(valid_mask, 1.0 / ranks, 0.0)

        if (batch_end % (batch_size * 5) == 0) or batch_end == n_probes:
            elapsed = time.time() - start
            print(f"      Processed {batch_end:,}/{n_probes:,} probes ({elapsed:.1f}s)")

    # Vectorized top-K counting
    correct_at_k = {k: int(np.sum(all_ranks <= k)) for k in top_k_values}

    # Aggregate metrics
    top_k_acc = {k: correct_at_k[k] / n_probes for k in top_k_values}
    mrr = reciprocal_ranks.mean()

    # Per-user accuracy (what fraction of each user's probes are top-1 correct)
    user_correct = defaultdict(list)
    for i, uid in enumerate(probe_user_ids):
        user_correct[uid].append(1 if all_ranks[i] == 1 else 0)
    per_user_acc = {uid: np.mean(vals) for uid, vals in user_correct.items()}
    per_user_acc_values = list(per_user_acc.values())

    results = {
        "n_gallery_users": n_gallery,
        "n_probes": n_probes,
        "metric": metric,
        "top_k_accuracy": {str(k): float(v) for k, v in top_k_acc.items()},
        "mrr": float(mrr),
        "mean_rank": float(all_ranks.mean()),
        "median_rank": float(np.median(all_ranks)),
        "per_user_accuracy_mean": float(np.mean(per_user_acc_values)),
        "per_user_accuracy_std": float(np.std(per_user_acc_values)),
        "per_user_accuracy_median": float(np.median(per_user_acc_values)),
        "users_with_100pct_reid": int(sum(1 for v in per_user_acc_values if v == 1.0)),
        "users_with_0pct_reid": int(sum(1 for v in per_user_acc_values if v == 0.0)),
        "_ranks": all_ranks,
        "_per_user_acc": per_user_acc,
    }

    return results


def run_random_baseline(
    n_gallery: int,
    n_probes: int,
    top_k_values: tuple = (1, 5, 10, 20),
) -> dict:
    """
    Compute expected metrics for a random-guess baseline.
    This tells us what accuracy we'd get by chance.
    """
    return {
        "top_k_accuracy": {
            str(k): min(1.0, k / n_gallery) for k in top_k_values
        },
        "mrr": sum(1.0 / r for r in range(1, n_gallery + 1)) / n_gallery,
    }
