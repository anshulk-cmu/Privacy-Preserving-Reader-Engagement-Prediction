"""
01_create_50k_dataset.py — Create a 50K-user dataset from ebnerd_small + ebnerd_large
=======================================================================================
Privacy-Preserving Engagement Prediction Project
94-806 Privacy in the Digital Age, Carnegie Mellon University

Takes all ~18,827 users from ebnerd_small and samples ~31,173 additional users
from ebnerd_large (who are NOT in small) to create a 50,000-user dataset.
Filters all parquet files from ebnerd_large to these 50K users and saves
to data/ebnerd_50k/ with the same directory structure.

The temporal split methodology is preserved exactly as in the original EB-NeRD paper:
  - Train behaviors: May 18 - May 25, 2023
  - Val behaviors:   May 25 - June 1, 2023
  - History windows: 21 days prior to each behavior window

Usage:
    conda activate privacy
    python src/01_create_50k_dataset.py

Requires:
    data/ebnerd_small/  (already downloaded)
    data/ebnerd_large/  (freshly downloaded and extracted)
"""

import sys
import os
from pathlib import Path
import polars as pl
import numpy as np
import shutil

# Windows UTF-8 console support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SRC = PROJECT_ROOT / "ebnerd-benchmark" / "src"
sys.path.insert(0, str(BENCHMARK_SRC))

from ebrec.utils._constants import DEFAULT_USER_COL

SMALL_DIR = PROJECT_ROOT / "data" / "ebnerd_small"
LARGE_DIR = PROJECT_ROOT / "data" / "ebnerd_large"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ebnerd_50k"

TARGET_USERS = 50_000
RANDOM_SEED = 42


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
    section("1. LOADING USER IDS FROM BOTH DATASETS")

    # Verify directories exist
    for d, name in [(SMALL_DIR, "ebnerd_small"), (LARGE_DIR, "ebnerd_large")]:
        if not d.exists():
            print(f"  ERROR: {name} not found at {d}")
            sys.exit(1)

    small_users = get_all_user_ids(SMALL_DIR)
    large_users = get_all_user_ids(LARGE_DIR)

    print(f"  ebnerd_small unique users: {len(small_users):,}")
    print(f"  ebnerd_large unique users: {len(large_users):,}")

    # Verify small is a subset of large
    small_in_large = small_users & large_users
    print(f"  Small users also in large:  {len(small_in_large):,} ({100*len(small_in_large)/len(small_users):.1f}%)")

    section("2. SAMPLING ADDITIONAL USERS")

    candidates = large_users - small_users
    print(f"  Candidate users (in large but not in small): {len(candidates):,}")

    needed = TARGET_USERS - len(small_users)
    print(f"  Users needed to reach {TARGET_USERS:,}: {needed:,}")

    if needed > len(candidates):
        print(f"  WARNING: Not enough candidates ({len(candidates):,}). Using all of them.")
        needed = len(candidates)

    rng = np.random.default_rng(RANDOM_SEED)
    candidates_list = sorted(candidates)  # Sort for reproducibility
    sampled_users = set(rng.choice(candidates_list, size=needed, replace=False).tolist())

    target_users = small_users | sampled_users
    print(f"  Sampled additional users: {len(sampled_users):,}")
    print(f"  Total target users:       {len(target_users):,}")

    section("3. FILTERING PARQUET FILES FROM ebnerd_large")

    # Create output directory structure
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train").mkdir(exist_ok=True)
    (OUTPUT_DIR / "validation").mkdir(exist_ok=True)

    target_list = sorted(target_users)

    for split in ["train", "validation"]:
        print(f"\n  --- {split} split ---")

        # Filter behaviors
        beh_path = LARGE_DIR / split / "behaviors.parquet"
        print(f"  Loading {beh_path.name}...", end=" ")
        df_beh = pl.read_parquet(beh_path)
        print(f"{len(df_beh):,} rows")

        df_beh_filtered = df_beh.filter(pl.col(DEFAULT_USER_COL).is_in(target_list))
        print(f"  Filtered behaviors: {len(df_beh_filtered):,} rows ({len(df_beh_filtered)/len(df_beh)*100:.1f}%)")

        out_beh = OUTPUT_DIR / split / "behaviors.parquet"
        df_beh_filtered.write_parquet(out_beh)
        print(f"  Saved: {out_beh}")

        # Filter history
        hist_path = LARGE_DIR / split / "history.parquet"
        print(f"  Loading {hist_path.name}...", end=" ")
        df_hist = pl.read_parquet(hist_path)
        print(f"{len(df_hist):,} rows")

        df_hist_filtered = df_hist.filter(pl.col(DEFAULT_USER_COL).is_in(target_list))
        print(f"  Filtered history: {len(df_hist_filtered):,} rows ({len(df_hist_filtered)/len(df_hist)*100:.1f}%)")

        out_hist = OUTPUT_DIR / split / "history.parquet"
        df_hist_filtered.write_parquet(out_hist)
        print(f"  Saved: {out_hist}")

        del df_beh, df_beh_filtered, df_hist, df_hist_filtered

    # Copy articles as-is (not user-specific)
    print(f"\n  Copying articles.parquet (shared, not user-specific)...")
    articles_src = LARGE_DIR / "articles.parquet"
    articles_dst = OUTPUT_DIR / "articles.parquet"
    shutil.copy2(articles_src, articles_dst)
    print(f"  Saved: {articles_dst}")

    section("4. VERIFICATION")

    # Verify user counts
    created_users = get_all_user_ids(OUTPUT_DIR)
    print(f"  Total unique users in ebnerd_50k: {len(created_users):,}")

    for split in ["train", "validation"]:
        df_beh = pl.read_parquet(OUTPUT_DIR / split / "behaviors.parquet")
        df_hist = pl.read_parquet(OUTPUT_DIR / split / "history.parquet", columns=[DEFAULT_USER_COL])
        n_beh_users = df_beh[DEFAULT_USER_COL].n_unique()
        n_hist_users = df_hist[DEFAULT_USER_COL].n_unique()
        print(f"\n  {split}:")
        print(f"    Behavior users: {n_beh_users:,}")
        print(f"    History users:  {n_hist_users:,}")
        print(f"    Impressions:    {len(df_beh):,}")

    # Check user overlap between splits
    train_users = set(
        pl.read_parquet(OUTPUT_DIR / "train" / "behaviors.parquet", columns=[DEFAULT_USER_COL])[DEFAULT_USER_COL]
        .unique().to_list()
    )
    val_users = set(
        pl.read_parquet(OUTPUT_DIR / "validation" / "behaviors.parquet", columns=[DEFAULT_USER_COL])[DEFAULT_USER_COL]
        .unique().to_list()
    )
    overlap = train_users & val_users
    print(f"\n  Train-val user overlap: {len(overlap):,} ({100*len(overlap)/len(val_users):.1f}% of val users)")

    # Check temporal ranges
    print(f"\n  Temporal ranges:")
    for split in ["train", "validation"]:
        df_beh = pl.read_parquet(OUTPUT_DIR / split / "behaviors.parquet")
        ts_col = [c for c in df_beh.columns if "timestamp" in c.lower() and "impression" in c.lower()]
        if ts_col:
            ts = df_beh[ts_col[0]]
            print(f"    {split} behaviors: {ts.min()} to {ts.max()}")

    # Check engagement signal availability
    print(f"\n  Engagement signals:")
    for split in ["train", "validation"]:
        df_beh = pl.read_parquet(OUTPUT_DIR / split / "behaviors.parquet")
        for col in ["next_read_time", "next_scroll_percentage", "read_time", "scroll_percentage"]:
            if col in df_beh.columns:
                non_null = df_beh[col].drop_nulls().len()
                print(f"    {split}/{col}: {non_null:,} non-null ({100*non_null/len(df_beh):.1f}%)")

    # Check schema matches
    print(f"\n  Schema check:")
    for split in ["train", "validation"]:
        for fname in ["behaviors.parquet", "history.parquet"]:
            small_cols = set(pl.read_parquet(SMALL_DIR / split / fname, n_rows=0).columns)
            new_cols = set(pl.read_parquet(OUTPUT_DIR / split / fname, n_rows=0).columns)
            match = small_cols == new_cols
            extra = new_cols - small_cols
            missing = small_cols - new_cols
            status = "MATCH" if match else f"DIFF (extra: {extra}, missing: {missing})"
            print(f"    {split}/{fname}: {status}")

    # Dataset sizes on disk
    print(f"\n  Disk usage:")
    for split in ["train", "validation"]:
        for fname in ["behaviors.parquet", "history.parquet"]:
            size = (OUTPUT_DIR / split / fname).stat().st_size / (1024 * 1024)
            print(f"    {split}/{fname}: {size:.1f} MB")
    art_size = (OUTPUT_DIR / "articles.parquet").stat().st_size / (1024 * 1024)
    print(f"    articles.parquet: {art_size:.1f} MB")

    section("DONE")
    print(f"\n  Created ebnerd_50k dataset at: {OUTPUT_DIR}")
    print(f"  Total unique users: {len(created_users):,}")
    print(f"  You can now delete ebnerd_large to free ~6GB of disk space.\n")


if __name__ == "__main__":
    main()
