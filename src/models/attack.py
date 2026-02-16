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
from sklearn.metrics import pairwise_distances


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
    batch_size: int = 1000,
) -> dict:
    """
    Run the nearest-neighbor re-identification attack.

    For each probe, finds the nearest gallery profile(s) and checks
    whether the correct user is among the top-K matches.

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
    n_probes = len(probe_reprs)
    n_gallery = len(gallery_profiles)
    max_k = max(top_k_values)

    # Build user_id -> gallery_index mapping
    uid_to_gallery_idx = {uid: i for i, uid in enumerate(gallery_user_ids)}

    all_ranks = np.zeros(n_probes, dtype=np.int64)
    correct_at_k = {k: 0 for k in top_k_values}
    reciprocal_ranks = np.zeros(n_probes)

    print(f"    Running attack: {n_probes:,} probes vs {n_gallery:,} gallery users")
    print(f"    Distance metric: {metric}")

    start = time.time()
    for batch_start in range(0, n_probes, batch_size):
        batch_end = min(batch_start + batch_size, n_probes)
        batch_probes = probe_reprs[batch_start:batch_end]
        batch_uids = probe_user_ids[batch_start:batch_end]

        # Compute distances: (batch, gallery)
        dists = pairwise_distances(batch_probes, gallery_profiles, metric=metric)

        # For each probe, rank gallery users by distance
        sorted_indices = np.argsort(dists, axis=1)

        for i in range(len(batch_probes)):
            global_i = batch_start + i
            true_uid = batch_uids[i]
            true_gallery_idx = uid_to_gallery_idx.get(true_uid)

            if true_gallery_idx is None:
                all_ranks[global_i] = n_gallery + 1
                continue

            # Find rank of correct user (0-indexed)
            rank_position = np.where(sorted_indices[i] == true_gallery_idx)[0][0]
            rank = rank_position + 1  # 1-indexed
            all_ranks[global_i] = rank
            reciprocal_ranks[global_i] = 1.0 / rank

            for k in top_k_values:
                if rank <= k:
                    correct_at_k[k] += 1

        if (batch_end % (batch_size * 10) == 0) or batch_end == n_probes:
            elapsed = time.time() - start
            print(f"      Processed {batch_end:,}/{n_probes:,} probes ({elapsed:.1f}s)")

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
