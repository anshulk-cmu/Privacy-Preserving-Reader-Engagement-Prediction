"""
01b_create_blind_test.py — Create a 10K-user blind test set with zero overlap to ebnerd_50k
=============================================================================================
Privacy-Preserving Engagement Prediction Project
94-806 Privacy in the Digital Age, Carnegie Mellon University

Samples 10,000 users from ebnerd_large who:
  1. Are active in BOTH the train period (May 18-25) AND the validation period (May 25-June 1)
  2. Are NOT present in the ebnerd_50k dataset (zero overlap)

This creates a completely independent test set for:
  - Negative control: re-identification attack should get ~0% on these users (never trained on)
  - Engagement generalization: verify model performance on unseen users
  - Privacy defense calibration: measure smoothing effects on unknown vs known users

Usage:
    conda activate privacy
    python src/01b_create_blind_test.py

Requires:
    data/ebnerd_50k/    (already created)
    data/ebnerd_large/  (still available on disk)
"""

import sys
import os
from pathlib import Path
import polars as pl
import numpy as np
import shutil
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SRC = PROJECT_ROOT / "ebnerd-benchmark" / "src"
sys.path.insert(0, str(BENCHMARK_SRC))

from ebrec.utils._constants import DEFAULT_USER_COL

LARGE_DIR = PROJECT_ROOT / "data" / "ebnerd_large"
MAIN_DIR = PROJECT_ROOT / "data" / "ebnerd_50k"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ebnerd_blind_test"

TARGET_USERS = 10_000
RANDOM_SEED = 123  # Different seed from 50k creation (seed=42)


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def get_all_user_ids(data_dir: Path) -> set:
    """Get the union of user IDs from train and validation behaviors."""
    users = set()
    for split in ["train", "validation"]:
        beh_path = data_dir / split / "behaviors.parquet"
        if beh_path.exists():
            df = pl.read_parquet(beh_path, columns=[DEFAULT_USER_COL])
            users.update(df[DEFAULT_USER_COL].unique().to_list())
    return users


def main():
    t_start = time.time()

    section("1. IDENTIFYING CANDIDATE USERS")

    for d, name in [(LARGE_DIR, "ebnerd_large"), (MAIN_DIR, "ebnerd_50k")]:
        if not d.exists():
            print(f"  ERROR: {name} not found at {d}")
            sys.exit(1)

    # Get 50k users (to exclude)
    main_users = get_all_user_ids(MAIN_DIR)
    print(f"  ebnerd_50k users (to exclude): {len(main_users):,}")

    # Get ebnerd_large users per split
    large_train_users = set(
        pl.read_parquet(LARGE_DIR / "train" / "behaviors.parquet", columns=[DEFAULT_USER_COL])
        [DEFAULT_USER_COL].unique().to_list()
    )
    large_val_users = set(
        pl.read_parquet(LARGE_DIR / "validation" / "behaviors.parquet", columns=[DEFAULT_USER_COL])
        [DEFAULT_USER_COL].unique().to_list()
    )
    print(f"  ebnerd_large train users: {len(large_train_users):,}")
    print(f"  ebnerd_large val users:   {len(large_val_users):,}")

    # Candidates: in BOTH periods, NOT in 50k
    candidates = (large_train_users & large_val_users) - main_users
    print(f"  Candidates (both periods, not in 50k): {len(candidates):,}")

    section("2. SAMPLING BLIND TEST USERS")

    rng = np.random.default_rng(RANDOM_SEED)
    candidates_sorted = sorted(candidates)
    sampled = set(rng.choice(candidates_sorted, size=TARGET_USERS, replace=False).tolist())

    # Verify zero overlap
    overlap = sampled & main_users
    assert len(overlap) == 0, f"CRITICAL: {len(overlap)} users overlap with ebnerd_50k!"

    print(f"  Sampled: {len(sampled):,} users (seed={RANDOM_SEED})")
    print(f"  Overlap with ebnerd_50k: {len(overlap)} (must be 0)")

    section("3. FILTERING PARQUET FILES FROM ebnerd_large")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train").mkdir(exist_ok=True)
    (OUTPUT_DIR / "validation").mkdir(exist_ok=True)

    target_list = sorted(sampled)

    for split in ["train", "validation"]:
        print(f"\n  --- {split} split ---")

        # Filter behaviors
        beh_path = LARGE_DIR / split / "behaviors.parquet"
        print(f"  Loading {beh_path.name}...", end=" ")
        df_beh = pl.read_parquet(beh_path)
        print(f"{len(df_beh):,} rows")

        df_beh_filtered = df_beh.filter(pl.col(DEFAULT_USER_COL).is_in(target_list))
        print(f"  Filtered behaviors: {len(df_beh_filtered):,} rows")

        out_beh = OUTPUT_DIR / split / "behaviors.parquet"
        df_beh_filtered.write_parquet(out_beh)
        print(f"  Saved: {out_beh}")

        # Filter history
        hist_path = LARGE_DIR / split / "history.parquet"
        print(f"  Loading {hist_path.name}...", end=" ")
        df_hist = pl.read_parquet(hist_path)
        print(f"{len(df_hist):,} rows")

        df_hist_filtered = df_hist.filter(pl.col(DEFAULT_USER_COL).is_in(target_list))
        print(f"  Filtered history: {len(df_hist_filtered):,} rows")

        out_hist = OUTPUT_DIR / split / "history.parquet"
        df_hist_filtered.write_parquet(out_hist)
        print(f"  Saved: {out_hist}")

        del df_beh, df_beh_filtered, df_hist, df_hist_filtered

    # Copy articles (same shared file)
    print(f"\n  Copying articles.parquet...")
    articles_src = LARGE_DIR / "articles.parquet"
    articles_dst = OUTPUT_DIR / "articles.parquet"
    shutil.copy2(articles_src, articles_dst)
    print(f"  Saved: {articles_dst}")

    section("4. VERIFICATION")

    created_users = get_all_user_ids(OUTPUT_DIR)
    print(f"  Total unique users in blind test: {len(created_users):,}")

    # Re-verify zero overlap
    final_overlap = created_users & main_users
    print(f"  Overlap with ebnerd_50k: {len(final_overlap)} (MUST be 0)")
    assert len(final_overlap) == 0, "CRITICAL: Blind test overlaps with ebnerd_50k!"

    for split in ["train", "validation"]:
        df_beh = pl.read_parquet(OUTPUT_DIR / split / "behaviors.parquet")
        df_hist = pl.read_parquet(OUTPUT_DIR / split / "history.parquet", columns=[DEFAULT_USER_COL])
        n_beh_users = df_beh[DEFAULT_USER_COL].n_unique()
        n_hist_users = df_hist[DEFAULT_USER_COL].n_unique()
        print(f"\n  {split}:")
        print(f"    Behavior users: {n_beh_users:,}")
        print(f"    History users:  {n_hist_users:,}")
        print(f"    Impressions:    {len(df_beh):,}")

    # Check train-val overlap within blind test
    bt_train = set(
        pl.read_parquet(OUTPUT_DIR / "train" / "behaviors.parquet", columns=[DEFAULT_USER_COL])
        [DEFAULT_USER_COL].unique().to_list()
    )
    bt_val = set(
        pl.read_parquet(OUTPUT_DIR / "validation" / "behaviors.parquet", columns=[DEFAULT_USER_COL])
        [DEFAULT_USER_COL].unique().to_list()
    )
    bt_overlap = bt_train & bt_val
    print(f"\n  Internal train-val overlap: {len(bt_overlap):,} ({100*len(bt_overlap)/len(bt_val):.1f}% of val)")

    # Engagement signal quality
    print(f"\n  Engagement signals:")
    for split in ["train", "validation"]:
        df = pl.read_parquet(OUTPUT_DIR / split / "behaviors.parquet")
        for col in ["next_read_time", "next_scroll_percentage"]:
            if col in df.columns:
                nn = df[col].drop_nulls().len()
                print(f"    {split}/{col}: {nn:,} non-null ({100*nn/len(df):.1f}%)")

    # Engagement rate comparison
    print(f"\n  Engagement rate comparison:")
    for split in ["train", "validation"]:
        for ds_name, ds_dir in [("50k", MAIN_DIR), ("blind_test", OUTPUT_DIR)]:
            df = pl.read_parquet(ds_dir / split / "behaviors.parquet")
            labelable = df.filter(
                pl.col("next_read_time").is_not_null()
                & pl.col("next_scroll_percentage").is_not_null()
            )
            engaged = labelable.filter(
                (pl.col("next_scroll_percentage") > 50)
                & (pl.col("next_read_time") > 30)
            )
            rate = 100 * len(engaged) / max(len(labelable), 1)
            print(f"    {split}/{ds_name}: {rate:.1f}% ({len(engaged):,}/{len(labelable):,})")

    # Schema check
    print(f"\n  Schema check:")
    for split in ["train", "validation"]:
        for fname in ["behaviors.parquet", "history.parquet"]:
            main_cols = set(pl.read_parquet(MAIN_DIR / split / fname, n_rows=0).columns)
            new_cols = set(pl.read_parquet(OUTPUT_DIR / split / fname, n_rows=0).columns)
            match = main_cols == new_cols
            status = "MATCH" if match else f"DIFF"
            print(f"    {split}/{fname}: {status}")

    # Disk usage
    print(f"\n  Disk usage:")
    for split in ["train", "validation"]:
        for fname in ["behaviors.parquet", "history.parquet"]:
            size = (OUTPUT_DIR / split / fname).stat().st_size / (1024 * 1024)
            print(f"    {split}/{fname}: {size:.1f} MB")
    art_size = (OUTPUT_DIR / "articles.parquet").stat().st_size / (1024 * 1024)
    print(f"    articles.parquet: {art_size:.1f} MB")

    elapsed = time.time() - t_start

    section("DONE")
    print(f"""
  Created blind test set at: {OUTPUT_DIR}
  Total unique users: {len(created_users):,}
  Overlap with ebnerd_50k: 0 (verified)
  Random seed: {RANDOM_SEED} (different from 50k seed=42)
  Creation time: {elapsed:.1f}s

  Purpose:
    1. Negative control for re-identification (model never saw these users)
    2. Engagement prediction generalization test
    3. Privacy defense calibration on unknown users
""")


if __name__ == "__main__":
    main()
