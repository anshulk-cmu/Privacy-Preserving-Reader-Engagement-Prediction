"""
00_eda.py — Exploratory Data Analysis for EB-NeRD (ebnerd_50k)
=================================================================
Privacy-Preserving Engagement Prediction Project
94-806 Privacy in the Digital Age, Carnegie Mellon University

Analyzes the ebnerd_50k dataset to understand:
  1. Schema and shape of behaviors, history, and articles parquet files
  2. Distribution of read_time and scroll_percentage (our two core signals)
  3. User-level statistics (impressions per user, history lengths)
  4. Missing value analysis
  5. Engagement label feasibility (scroll > 50% AND read_time > 30s)
  6. Privacy-relevant: uniqueness of behavioral sequences

Usage:
    source .venv/bin/activate
    python src/00_eda.py

Requires ebnerd_50k dataset at data/ebnerd_50k/
"""

import sys
from pathlib import Path
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Add the benchmark repo's src to path so we can import ebrec utilities
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_SRC = PROJECT_ROOT / "ebnerd-benchmark" / "src"
sys.path.insert(0, str(BENCHMARK_SRC))

from ebrec.utils._constants import (
    DEFAULT_USER_COL,
    DEFAULT_READ_TIME_COL,
    DEFAULT_SCROLL_PERCENTAGE_COL,
    DEFAULT_IMPRESSION_ID_COL,
    DEFAULT_ARTICLE_ID_COL,
    DEFAULT_IMPRESSION_TIMESTAMP_COL,
    DEFAULT_CLICKED_ARTICLES_COL,
    DEFAULT_INVIEW_ARTICLES_COL,
    DEFAULT_SESSION_ID_COL,
    DEFAULT_HISTORY_ARTICLE_ID_COL,
    DEFAULT_HISTORY_READ_TIME_COL,
    DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL,
    DEFAULT_HISTORY_IMPRESSION_TIMESTAMP_COL,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "ebnerd_50k"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ENGAGEMENT_SCROLL_THRESHOLD = 50  # percent
ENGAGEMENT_READ_TIME_THRESHOLD = 30  # seconds


def section(title: str):
    """Print a visible section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def load_and_inspect_parquet(name: str, path: Path) -> pl.DataFrame:
    """Load a parquet file and print its schema and shape."""
    print(f"\n--- {name} ---")
    print(f"  Path: {path}")
    if not path.exists():
        print(f"  ERROR: File not found at {path}")
        print(f"  Please run 01_create_50k_dataset.py to create ebnerd_50k")
        print(f"  and extract it to {DATA_DIR}/")
        sys.exit(1)
    df = pl.read_parquet(path)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {df.columns}")
    print(f"  Dtypes:")
    for col, dtype in zip(df.columns, df.dtypes):
        print(f"    {col:35s} {str(dtype)}")
    return df


def main():
    section("1. LOADING DATA")

    # Behaviors
    df_beh_train = load_and_inspect_parquet(
        "behaviors (train)", DATA_DIR / "train" / "behaviors.parquet"
    )
    df_beh_val = load_and_inspect_parquet(
        "behaviors (validation)", DATA_DIR / "validation" / "behaviors.parquet"
    )

    # History
    df_hist_train = load_and_inspect_parquet(
        "history (train)", DATA_DIR / "train" / "history.parquet"
    )
    df_hist_val = load_and_inspect_parquet(
        "history (validation)", DATA_DIR / "validation" / "history.parquet"
    )

    # Articles
    df_articles = load_and_inspect_parquet(
        "articles", DATA_DIR / "articles.parquet"
    )

    # ===================================================================
    section("2. BEHAVIORS — read_time AND scroll_percentage DISTRIBUTIONS")
    # ===================================================================

    for split_name, df_beh in [("train", df_beh_train), ("validation", df_beh_val)]:
        print(f"\n--- {split_name} split ---")

        # Check which engagement columns exist
        has_read_time = DEFAULT_READ_TIME_COL in df_beh.columns
        has_scroll = DEFAULT_SCROLL_PERCENTAGE_COL in df_beh.columns
        has_next_read_time = "next_read_time" in df_beh.columns
        has_next_scroll = "next_scroll_percentage" in df_beh.columns

        print(f"  Has '{DEFAULT_READ_TIME_COL}': {has_read_time}")
        print(f"  Has '{DEFAULT_SCROLL_PERCENTAGE_COL}': {has_scroll}")
        print(f"  Has 'next_read_time': {has_next_read_time}")
        print(f"  Has 'next_scroll_percentage': {has_next_scroll}")

        # Use next_read_time / next_scroll_percentage if available
        # (these represent the NEXT page engagement after the impression)
        rt_col = "next_read_time" if has_next_read_time else DEFAULT_READ_TIME_COL
        sp_col = "next_scroll_percentage" if has_next_scroll else DEFAULT_SCROLL_PERCENTAGE_COL

        if rt_col in df_beh.columns:
            rt = df_beh[rt_col].drop_nulls()
            print(f"\n  {rt_col} statistics (non-null: {len(rt):,} / {len(df_beh):,}):")
            print(f"    Mean:   {rt.mean():.2f}s")
            print(f"    Median: {rt.median():.2f}s")
            print(f"    Std:    {rt.std():.2f}s")
            print(f"    Min:    {rt.min()}")
            print(f"    Max:    {rt.max()}")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                print(f"    P{p:02d}:    {rt.quantile(p/100):.1f}s")

        if sp_col in df_beh.columns:
            sp = df_beh[sp_col].drop_nulls()
            print(f"\n  {sp_col} statistics (non-null: {len(sp):,} / {len(df_beh):,}):")
            print(f"    Mean:   {sp.mean():.1f}%")
            print(f"    Median: {sp.median():.1f}%")
            print(f"    Std:    {sp.std():.1f}%")
            print(f"    Min:    {sp.min()}")
            print(f"    Max:    {sp.max()}")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            for p in percentiles:
                print(f"    P{p:02d}:    {sp.quantile(p/100):.1f}%")

    # Plot distributions for train split
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Engagement Signal Distributions (Train Split)", fontsize=14, fontweight="bold")

    for idx, (split_name, df_beh) in enumerate([("train", df_beh_train), ("validation", df_beh_val)]):
        rt_col = "next_read_time" if "next_read_time" in df_beh.columns else DEFAULT_READ_TIME_COL
        sp_col = "next_scroll_percentage" if "next_scroll_percentage" in df_beh.columns else DEFAULT_SCROLL_PERCENTAGE_COL

        if rt_col in df_beh.columns:
            rt_vals = df_beh[rt_col].drop_nulls().to_numpy()
            rt_clipped = np.clip(rt_vals, 0, 300)
            axes[0, idx].hist(rt_clipped, bins=60, color="#2196F3", alpha=0.7, edgecolor="white")
            axes[0, idx].axvline(x=ENGAGEMENT_READ_TIME_THRESHOLD, color="red", linestyle="--",
                                label=f"Threshold: {ENGAGEMENT_READ_TIME_THRESHOLD}s")
            axes[0, idx].set_title(f"Read Time — {split_name}")
            axes[0, idx].set_xlabel("Seconds (clipped at 300)")
            axes[0, idx].set_ylabel("Count")
            axes[0, idx].legend()

        if sp_col in df_beh.columns:
            sp_vals = df_beh[sp_col].drop_nulls().to_numpy()
            axes[1, idx].hist(sp_vals, bins=50, color="#4CAF50", alpha=0.7, edgecolor="white")
            axes[1, idx].axvline(x=ENGAGEMENT_SCROLL_THRESHOLD, color="red", linestyle="--",
                                label=f"Threshold: {ENGAGEMENT_SCROLL_THRESHOLD}%")
            axes[1, idx].set_title(f"Scroll Percentage — {split_name}")
            axes[1, idx].set_xlabel("Percentage")
            axes[1, idx].set_ylabel("Count")
            axes[1, idx].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_engagement_distributions.png", dpi=150)
    print(f"\n  Saved: {FIGURES_DIR / '01_engagement_distributions.png'}")
    plt.close()

    # ===================================================================
    section("3. ENGAGEMENT LABEL ANALYSIS")
    # ===================================================================

    for split_name, df_beh in [("train", df_beh_train), ("validation", df_beh_val)]:
        print(f"\n--- {split_name} split ---")

        rt_col = "next_read_time" if "next_read_time" in df_beh.columns else DEFAULT_READ_TIME_COL
        sp_col = "next_scroll_percentage" if "next_scroll_percentage" in df_beh.columns else DEFAULT_SCROLL_PERCENTAGE_COL

        if rt_col in df_beh.columns and sp_col in df_beh.columns:
            # Create engagement label
            df_labeled = df_beh.with_columns(
                (
                    (pl.col(sp_col) > ENGAGEMENT_SCROLL_THRESHOLD)
                    & (pl.col(rt_col) > ENGAGEMENT_READ_TIME_THRESHOLD)
                ).cast(pl.Int8).alias("engaged")
            )

            total = len(df_labeled)
            non_null = df_labeled.filter(
                pl.col(rt_col).is_not_null() & pl.col(sp_col).is_not_null()
            ).shape[0]
            engaged = df_labeled.filter(pl.col("engaged") == 1).shape[0]
            not_engaged = non_null - engaged
            null_count = total - non_null

            print(f"  Total impressions:    {total:,}")
            print(f"  Non-null (labelable): {non_null:,} ({100*non_null/total:.1f}%)")
            print(f"  Null (unlabelable):   {null_count:,} ({100*null_count/total:.1f}%)")
            print(f"  Engaged (label=1):    {engaged:,} ({100*engaged/non_null:.1f}% of labelable)")
            print(f"  Not engaged (label=0):{not_engaged:,} ({100*not_engaged/non_null:.1f}% of labelable)")
            print(f"  Class ratio (pos/neg): 1:{not_engaged/max(engaged,1):.1f}")

            # Also check individual conditions
            scroll_only = df_beh.filter(pl.col(sp_col) > ENGAGEMENT_SCROLL_THRESHOLD).shape[0]
            read_only = df_beh.filter(pl.col(rt_col) > ENGAGEMENT_READ_TIME_THRESHOLD).shape[0]
            print(f"\n  Scroll > {ENGAGEMENT_SCROLL_THRESHOLD}% alone:       {scroll_only:,} ({100*scroll_only/non_null:.1f}%)")
            print(f"  Read time > {ENGAGEMENT_READ_TIME_THRESHOLD}s alone:   {read_only:,} ({100*read_only/non_null:.1f}%)")
            print(f"  Both (engaged):        {engaged:,} ({100*engaged/non_null:.1f}%)")
        else:
            print(f"  Columns missing — cannot compute engagement label")

    # ===================================================================
    section("4. USER-LEVEL STATISTICS")
    # ===================================================================

    for split_name, df_beh in [("train", df_beh_train), ("validation", df_beh_val)]:
        print(f"\n--- {split_name} split ---")
        n_users = df_beh[DEFAULT_USER_COL].n_unique()
        n_impressions = len(df_beh)
        print(f"  Unique users:      {n_users:,}")
        print(f"  Total impressions: {n_impressions:,}")
        print(f"  Avg impressions/user: {n_impressions/n_users:.1f}")

        # Impressions per user distribution
        impr_per_user = df_beh.group_by(DEFAULT_USER_COL).len()
        print(f"  Impressions per user:")
        print(f"    Min:    {impr_per_user['len'].min()}")
        print(f"    Max:    {impr_per_user['len'].max()}")
        print(f"    Median: {impr_per_user['len'].median():.0f}")
        print(f"    Mean:   {impr_per_user['len'].mean():.1f}")

    # ===================================================================
    section("5. HISTORY — BEHAVIORAL SEQUENCE ANALYSIS")
    # ===================================================================

    for split_name, df_hist in [("train", df_hist_train), ("validation", df_hist_val)]:
        print(f"\n--- {split_name} split ---")
        n_users = len(df_hist)
        print(f"  Users with history: {n_users:,}")

        # History length (number of past clicked articles per user)
        if DEFAULT_HISTORY_ARTICLE_ID_COL in df_hist.columns:
            hist_lens = df_hist[DEFAULT_HISTORY_ARTICLE_ID_COL].list.len()
            print(f"  History length (# past articles):")
            print(f"    Min:    {hist_lens.min()}")
            print(f"    Max:    {hist_lens.max()}")
            print(f"    Median: {hist_lens.median():.0f}")
            print(f"    Mean:   {hist_lens.mean():.1f}")

        # Read time in history
        if DEFAULT_HISTORY_READ_TIME_COL in df_hist.columns:
            # Explode to get all individual read times
            rt_exploded = df_hist.select(DEFAULT_HISTORY_READ_TIME_COL).explode(DEFAULT_HISTORY_READ_TIME_COL)
            rt_vals = rt_exploded[DEFAULT_HISTORY_READ_TIME_COL].drop_nulls()
            print(f"\n  Historical read_time across all interactions:")
            print(f"    Count:  {len(rt_vals):,}")
            print(f"    Mean:   {rt_vals.mean():.2f}s")
            print(f"    Median: {rt_vals.median():.2f}s")
            print(f"    P10:    {rt_vals.quantile(0.10):.1f}s")
            print(f"    P90:    {rt_vals.quantile(0.90):.1f}s")

        # Scroll percentage in history
        if DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL in df_hist.columns:
            sp_exploded = df_hist.select(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL).explode(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL)
            sp_vals = sp_exploded[DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL].drop_nulls()
            print(f"\n  Historical scroll_percentage across all interactions:")
            print(f"    Count:  {len(sp_vals):,}")
            print(f"    Mean:   {sp_vals.mean():.1f}%")
            print(f"    Median: {sp_vals.median():.1f}%")
            print(f"    P10:    {sp_vals.quantile(0.10):.1f}%")
            print(f"    P90:    {sp_vals.quantile(0.90):.1f}%")

    # Plot history distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("User History Characteristics (Train)", fontsize=14, fontweight="bold")

    if DEFAULT_HISTORY_ARTICLE_ID_COL in df_hist_train.columns:
        hist_lens_np = df_hist_train[DEFAULT_HISTORY_ARTICLE_ID_COL].list.len().to_numpy()
        axes[0].hist(hist_lens_np, bins=50, color="#FF9800", alpha=0.7, edgecolor="white")
        axes[0].set_title("History Length (# articles)")
        axes[0].set_xlabel("Number of past articles")
        axes[0].set_ylabel("Number of users")

    if DEFAULT_HISTORY_READ_TIME_COL in df_hist_train.columns:
        rt_hist = df_hist_train.select(DEFAULT_HISTORY_READ_TIME_COL).explode(DEFAULT_HISTORY_READ_TIME_COL)
        rt_np = np.clip(rt_hist[DEFAULT_HISTORY_READ_TIME_COL].drop_nulls().to_numpy(), 0, 300)
        axes[1].hist(rt_np, bins=60, color="#2196F3", alpha=0.7, edgecolor="white")
        axes[1].axvline(x=ENGAGEMENT_READ_TIME_THRESHOLD, color="red", linestyle="--")
        axes[1].set_title("Historical Read Times")
        axes[1].set_xlabel("Seconds (clipped at 300)")

    if DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL in df_hist_train.columns:
        sp_hist = df_hist_train.select(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL).explode(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL)
        sp_np = sp_hist[DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL].drop_nulls().to_numpy()
        axes[2].hist(sp_np, bins=50, color="#4CAF50", alpha=0.7, edgecolor="white")
        axes[2].axvline(x=ENGAGEMENT_SCROLL_THRESHOLD, color="red", linestyle="--")
        axes[2].set_title("Historical Scroll Percentages")
        axes[2].set_xlabel("Percentage")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_history_distributions.png", dpi=150)
    print(f"\n  Saved: {FIGURES_DIR / '02_history_distributions.png'}")
    plt.close()

    # ===================================================================
    section("6. PRIVACY-RELEVANT: BEHAVIORAL UNIQUENESS")
    # ===================================================================
    print("\n  How unique are users' behavioral sequences?")
    print("  (This motivates why we need privacy protection)")

    if DEFAULT_HISTORY_READ_TIME_COL in df_hist_train.columns and \
       DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL in df_hist_train.columns:

        # Compute per-user summary stats as a "behavioral fingerprint"
        df_fingerprint = df_hist_train.select(
            DEFAULT_USER_COL,
            DEFAULT_HISTORY_READ_TIME_COL,
            DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL,
        )

        # Mean and std of read time and scroll percentage per user
        df_stats = df_fingerprint.with_columns([
            pl.col(DEFAULT_HISTORY_READ_TIME_COL).list.mean().alias("rt_mean"),
            pl.col(DEFAULT_HISTORY_READ_TIME_COL).list.eval(pl.element().std()).list.first().alias("rt_std"),
            pl.col(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL).list.mean().alias("sp_mean"),
            pl.col(DEFAULT_HISTORY_SCROLL_PERCENTAGE_COL).list.eval(pl.element().std()).list.first().alias("sp_std"),
            pl.col(DEFAULT_HISTORY_READ_TIME_COL).list.len().alias("hist_len"),
        ]).select(["rt_mean", "rt_std", "sp_mean", "sp_std", "hist_len"]).drop_nulls()

        n_users = len(df_stats)
        # Round to check for exact duplicates in the 4D fingerprint space
        df_rounded = df_stats.with_columns([
            pl.col("rt_mean").round(1),
            pl.col("rt_std").round(1),
            pl.col("sp_mean").round(1),
            pl.col("sp_std").round(1),
        ])
        n_unique = df_rounded.unique().shape[0]

        print(f"\n  Users in train history: {n_users:,}")
        print(f"  Unique 4D behavioral fingerprints")
        print(f"    (rt_mean, rt_std, sp_mean, sp_std rounded to 1dp): {n_unique:,}")
        print(f"    Uniqueness ratio: {100*n_unique/n_users:.1f}%")
        print(f"    --> {'HIGH' if n_unique/n_users > 0.8 else 'MODERATE'} re-identification risk")

        # Pairwise distance analysis on a sample
        sample_size = min(1000, n_users)
        features = df_stats.sample(n=sample_size, seed=42).to_numpy()
        from sklearn.preprocessing import StandardScaler
        features_scaled = StandardScaler().fit_transform(features)

        from scipy.spatial.distance import pdist
        distances = pdist(features_scaled, metric="euclidean")
        print(f"\n  Pairwise L2 distances (sample of {sample_size} users, standardized):")
        print(f"    Mean:   {np.mean(distances):.3f}")
        print(f"    Median: {np.median(distances):.3f}")
        print(f"    Min:    {np.min(distances):.4f}")
        print(f"    P05:    {np.percentile(distances, 5):.3f}")
        print(f"    P95:    {np.percentile(distances, 95):.3f}")
        print(f"    --> Nearest-neighbor distances indicate how easy re-identification would be")

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.hist(distances, bins=100, color="#E91E63", alpha=0.7, edgecolor="white")
        ax.set_title("Pairwise L2 Distances Between User Behavioral Fingerprints", fontsize=12, fontweight="bold")
        ax.set_xlabel("Euclidean Distance (standardized feature space)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "03_pairwise_distances.png", dpi=150)
        print(f"\n  Saved: {FIGURES_DIR / '03_pairwise_distances.png'}")
        plt.close()

    # ===================================================================
    section("7. ARTICLES OVERVIEW")
    # ===================================================================

    print(f"\n  Total articles: {len(df_articles):,}")
    print(f"  Columns: {df_articles.columns}")

    for col in ["category", "article_type", "premium"]:
        if col in df_articles.columns:
            vc = df_articles[col].value_counts().sort("count", descending=True).head(10)
            print(f"\n  Top values in '{col}':")
            for row in vc.iter_rows():
                print(f"    {row[0]}: {row[1]:,}")

    # ===================================================================
    section("8. CORRELATION: read_time vs scroll_percentage")
    # ===================================================================

    rt_col = "next_read_time" if "next_read_time" in df_beh_train.columns else DEFAULT_READ_TIME_COL
    sp_col = "next_scroll_percentage" if "next_scroll_percentage" in df_beh_train.columns else DEFAULT_SCROLL_PERCENTAGE_COL

    if rt_col in df_beh_train.columns and sp_col in df_beh_train.columns:
        df_corr = df_beh_train.select(rt_col, sp_col).drop_nulls()
        corr = np.corrcoef(df_corr[rt_col].to_numpy(), df_corr[sp_col].to_numpy())[0, 1]
        print(f"\n  Pearson correlation (train): {corr:.4f}")

        # Scatter plot (sample to avoid overplotting)
        sample = df_corr.sample(n=min(5000, len(df_corr)), seed=42)
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.scatter(
            np.clip(sample[rt_col].to_numpy(), 0, 300),
            sample[sp_col].to_numpy(),
            alpha=0.15, s=5, color="#673AB7"
        )
        ax.axvline(x=ENGAGEMENT_READ_TIME_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"RT={ENGAGEMENT_READ_TIME_THRESHOLD}s")
        ax.axhline(y=ENGAGEMENT_SCROLL_THRESHOLD, color="red", linestyle="--", alpha=0.7, label=f"SP={ENGAGEMENT_SCROLL_THRESHOLD}%")
        ax.set_xlabel("Read Time (seconds, clipped at 300)")
        ax.set_ylabel("Scroll Percentage")
        ax.set_title(f"Read Time vs Scroll Percentage (r={corr:.3f})", fontsize=12, fontweight="bold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "04_rt_vs_sp_scatter.png", dpi=150)
        print(f"  Saved: {FIGURES_DIR / '04_rt_vs_sp_scatter.png'}")
        plt.close()

    # ===================================================================
    section("9. TRAIN vs VALIDATION USER OVERLAP")
    # ===================================================================

    train_users = set(df_beh_train[DEFAULT_USER_COL].unique().to_list())
    val_users = set(df_beh_val[DEFAULT_USER_COL].unique().to_list())
    overlap = train_users & val_users
    train_only = train_users - val_users
    val_only = val_users - train_users
    all_users = train_users | val_users

    print(f"\n  Train users:      {len(train_users):,}")
    print(f"  Validation users: {len(val_users):,}")
    print(f"  Overlap:          {len(overlap):,} ({100*len(overlap)/len(val_users):.1f}% of val users)")
    print(f"  Train-only:       {len(train_only):,}")
    print(f"  Validation-only:  {len(val_only):,}")
    print(f"  Total unique:     {len(all_users):,}")
    print(f"\n  NOTE: ebnerd_50k was created by sampling 50K users from ebnerd_large,")
    print(f"  preserving the original EB-NeRD temporal split methodology.")

    # ===================================================================
    section("SUMMARY")
    # ===================================================================

    print(f"""
  Dataset: ebnerd_50k
  Train behaviors:   {len(df_beh_train):,} impressions, {df_beh_train[DEFAULT_USER_COL].n_unique():,} users
  Val behaviors:     {len(df_beh_val):,} impressions, {df_beh_val[DEFAULT_USER_COL].n_unique():,} users
  Train history:     {len(df_hist_train):,} users with past behavior sequences
  Articles:          {len(df_articles):,}

  Key signals: read_time (seconds), scroll_percentage (0-100)
  Engagement label: scroll > {ENGAGEMENT_SCROLL_THRESHOLD}% AND read_time > {ENGAGEMENT_READ_TIME_THRESHOLD}s

  Figures saved to: {FIGURES_DIR}/
    01_engagement_distributions.png
    02_history_distributions.png
    03_pairwise_distances.png
    04_rt_vs_sp_scatter.png
""")


if __name__ == "__main__":
    main()
