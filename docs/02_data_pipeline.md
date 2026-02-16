# Phase 2: Data Preprocessing Pipeline

**Privacy-Preserving Reader Engagement Prediction**
**94-806 Privacy in the Digital Age | Carnegie Mellon University**

---

## Table of Contents

1. [Overview](#1-overview)
2. [Pipeline Architecture](#2-pipeline-architecture)
3. [Input Data Inspection](#3-input-data-inspection)
4. [Feature Decisions](#4-feature-decisions)
5. [Engagement Label Creation](#5-engagement-label-creation)
6. [History Sequence Extraction](#6-history-sequence-extraction)
7. [Normalization Strategy](#7-normalization-strategy)
8. [Aggregate Feature Engineering](#8-aggregate-feature-engineering)
9. [Article and Context Features](#9-article-and-context-features)
10. [Output Formats and Shapes](#10-output-formats-and-shapes)
11. [Verification and Audit Results](#11-verification-and-audit-results)
12. [Design Decisions and Rationale](#12-design-decisions-and-rationale)

---

## 1. Overview

Phase 2 transforms the raw `ebnerd_50k` parquet files into model-ready numpy arrays and PyTorch DataLoaders. This is the bridge between the raw dataset (Phase 1) and model training (Phases 3A/3B).

**Scripts:**
- `src/data/preprocessing.py` -- Loads raw parquets, creates labels, engineers features, normalizes, saves numpy arrays and fitted scalers
- `src/data/dataset.py` -- PyTorch `EngagementDataset` class + `DataLoader` factory with custom collation

**Input:** `data/ebnerd_50k/` (raw parquet files, 291 MB)
**Output:** `data/processed/` (numpy arrays + scalers + metadata, 557 MB)

---

## 2. Pipeline Architecture

The pipeline processes each split (train, validation) through these steps:

```
behaviors.parquet + history.parquet + articles.parquet
    |
    v
[1] Filter to labelable rows (non-null next_read_time AND next_scroll_percentage)
    |
    v
[2] Create binary engagement label
    |
    v
[3] Join behaviors with history on user_id
    |
    v
[4] Extract history sequences ──> (N, 50, 2) for LSTM
    |                                    |
    v                                    v
[5] Compute aggregate features     [6] Normalize sequences
    (N, 21) for MLP                    log1p + StandardScaler for RT
    |                                  /100 for SP
    v                                  |
[7] StandardScaler (fit on train)      v
    |                              [8] Zero-pad to length 50
    v                                  |
[9] Extract article features           v
    via first clicked article_id       |
    |                                  |
    v                                  v
[10] Extract context features    ──> Save all as .npy arrays
```

Scalers are fitted on **train only** and applied to validation (no data leakage).

---

## 3. Input Data Inspection

Before building the pipeline, we inspected every column in the raw data to determine what was usable. Key findings:

### Behaviors columns

| Column | Dtype | Null Rate | Decision |
|--------|-------|-----------|----------|
| `user_id` | UInt32 | 0% | Use (join key + re-id experiments) |
| `article_id` | Int32 | **70.0%** | Skip (mostly null) |
| `article_ids_clicked` | List(Int32) | 0% (always >= 1 item) | Use first item for article lookup |
| `next_read_time` | Float32 | 2.7% | Use (label signal) |
| `next_scroll_percentage` | Float32 | 11.2% | Use (label signal) |
| `device_type` | Int8 | 0% | Use (3 values: 1, 2, 3) |
| `is_subscriber` | Boolean | 0% | Use |
| `is_sso_user` | Boolean | 0% | Use |
| `gender` | Int8 | **93.0%** | Drop |
| `age` | Int8 | **97.1%** | Drop |
| `postcode` | Int8 | **97.9%** | Drop |

### Articles columns

| Column | Dtype | Null Rate | Decision |
|--------|-------|-----------|----------|
| `category` | Int16 | 0% | Use (32 unique values, embedding) |
| `article_type` | String | 0% | Use (16 unique values, embedding) |
| `premium` | Boolean | 0% | Use |
| `sentiment_score` | Float32 | 0% | Use |
| `total_inviews` | Int32 | **85.4%** | Drop |
| `total_pageviews` | Int32 | **86.5%** | Drop |

### History columns

| Column | Dtype | Nulls | Decision |
|--------|-------|-------|----------|
| `read_time_fixed` | List(Float32) | Row-level: 0%. Within-list: 0% | Use |
| `scroll_percentage_fixed` | List(Float32) | Row-level: 0%. Within-list: ~5-10% per user | Use (fill None with 0) |
| `article_id_fixed` | List(Int32) | 0% | Not used (would need large embedding table) |

---

## 4. Feature Decisions

Based on the inspection, we selected three feature groups:

### For LSTM (sequential model)
- **History sequences**: Raw `(read_time, scroll_percentage)` pairs from the user's last 50 articles
- Shape: `(N, 50, 2)` -- a 50-step time series of 2 behavioral features per user per impression

### For MLP and LSTM (aggregate features)
- **27 handcrafted aggregate features** computed from the full history:
  - Read time: mean, std, median, P10, P90 (5 features)
  - Scroll percentage: mean, std, median, P10, P90 (5 features)
  - Last-5 read times (5 features)
  - Last-5 scroll percentages (5 features)
  - History length (1 feature)
  - Engagement rate, deep scroll rate, long read rate (3 features)
  - RT momentum, SP momentum, RT-SP correlation (3 features)

### Shared by both models
- **Article features** (7 total): category index, article type index, premium flag, sentiment score, body_len_log, title_len_log, subtitle_len_log
- **Context features**: device type, is_subscriber, is_sso_user (3 features)

---

## 5. Engagement Label Creation

**Definition:** `engaged = (next_scroll_percentage > 50%) AND (next_read_time > 30s)`

**Process:**
1. Filter behaviors to rows where BOTH `next_read_time` and `next_scroll_percentage` are non-null
2. Apply the AND condition to create a binary label (0 or 1)

**Results:**

| Metric | Train | Validation |
|--------|-------|------------|
| Raw impressions | 615,316 | 645,495 |
| Labelable (non-null) | 546,183 (88.8%) | 567,980 (88.0%) |
| Engaged (label=1) | 219,308 (40.2%) | 232,315 (40.9%) |
| Not engaged (label=0) | 326,875 (59.8%) | 335,665 (59.1%) |
| Class ratio (pos:neg) | 1:1.49 | 1:1.44 |

**Verification:** Labels were recomputed independently from raw parquets and matched the saved arrays exactly (219,308 positives, 100% agreement). Five individual rows were spot-checked including edge cases (e.g., read_time=29s correctly labeled 0 despite scroll=95%).

---

## 6. History Sequence Extraction

**Process:**
1. Join labeled behaviors with history on `user_id`
2. For each row, take the **last 50 articles** from the user's history (most recent behavior)
3. Extract `(read_time, scroll_percentage)` pairs
4. Pad shorter sequences with zeros, record actual length

**Why last 50:**
- Median user history is 92 articles, but 88.5% of samples truncate at exactly 50
- Most recent behavior is the strongest predictor of near-future engagement
- Keeps LSTM sequence length manageable (training cost scales linearly with sequence length)
- Full-history statistics are still captured in the 21 aggregate features for the MLP

**Results:**

| Metric | Train | Validation |
|--------|-------|------------|
| Output shape | (546,183, 50, 2) | (567,980, 50, 2) |
| Min actual length | 5 | 5 |
| Max actual length | 50 | 50 |
| Mean actual length | 47.2 | 46.6 |
| Rows at full length 50 | 88.5% | 87.2% |
| Rows with no history | 0 | 0 |

**Verification:** Five random rows were spot-checked by comparing the saved scroll_percentage/100 values against raw parquet data (last N items from `scroll_percentage_fixed`). All matched exactly. Padding positions were confirmed to be all zeros across 1,000 checked rows.

---

## 7. Normalization Strategy

### Read time (history channel 0)

Read time is extremely right-skewed (median 16s, max 1800s), so we apply a two-step normalization:

1. **`log1p` transform**: Compresses the long tail. Maps [0, 1800] to [0, 7.5].
2. **StandardScaler** (fitted on train non-padded values only): Centers to mean=0, std=1.

Fitted scaler parameters:
- Mean of log1p(read_time): **2.957** (corresponds to raw ~18s, close to the median)
- Std of log1p(read_time): **1.465**

Resulting distribution on train: mean = 0.009, std = 1.002 (essentially N(0,1)).

### Scroll percentage (history channel 1)

Scroll percentage is bounded [0, 100], so normalization is simple:
- Divide by 100 to map to [0, 1]
- No scaler fitting needed

Resulting distribution: mean = 0.625, range [0, 1].

### Aggregate features

All 27 aggregate features are normalized via a single StandardScaler fitted on train. Resulting train distribution: mean = 0.000, std = 1.000 per feature.

### No data leakage

Validation data uses the **train-fitted** scalers:
- Val aggregate features: mean = 0.037 (not 0), std = 0.994 (not 1) -- confirms no refitting
- Val RT history: mean = 0.011, std = 0.980 -- confirms train scaler applied

---

## 8. Aggregate Feature Engineering

27 features computed from each user's **full** history (not truncated to 50):

| # | Feature | Description |
|---|---------|-------------|
| 0 | `rt_mean` | Mean historical read time |
| 1 | `rt_std` | Std dev of historical read times |
| 2 | `rt_median` | Median historical read time |
| 3 | `rt_p10` | 10th percentile read time |
| 4 | `rt_p90` | 90th percentile read time |
| 5 | `sp_mean` | Mean historical scroll percentage |
| 6 | `sp_std` | Std dev of historical scroll percentages |
| 7 | `sp_median` | Median historical scroll percentage |
| 8 | `sp_p10` | 10th percentile scroll percentage |
| 9 | `sp_p90` | 90th percentile scroll percentage |
| 10-14 | `last5_rt_*` | Read times of last 5 articles (padded with 0 if history < 5) |
| 15-19 | `last5_sp_*` | Scroll percentages of last 5 articles (padded with 0 if history < 5) |
| 20 | `hist_len` | Number of articles in history |
| 21 | `hist_engagement_rate` | P(RT > 30s AND SP > 50%) across full history |
| 22 | `hist_deep_scroll_rate` | P(SP > 80%) across full history |
| 23 | `hist_long_read_rate` | P(RT > 60s) across full history |
| 24 | `rt_momentum` | mean(last-5 RT) / mean(all RT), clipped [0.1, 10] |
| 25 | `sp_momentum` | mean(last-5 SP) / mean(all SP), clipped [0.1, 10] |
| 26 | `rt_sp_correlation` | Pearson correlation between RT and SP sequences |

Features 0-20 are the original MLP baseline features. Features 21-26 were added for the LSTM pipeline to capture joint engagement behavior, behavioral momentum, and reading consistency.

After StandardScaler, all features have mean ~0 and std ~1 on train. Value ranges are reasonable:
- Most features range [-5, +20] -- long tails from power users with extreme reading patterns
- `sp_p90` has an unusual range [-21, 0.18] because most users have P90 scroll = 100%, compressing the distribution to the right

---

## 9. Article and Context Features

### Article features (7 per sample)

Looked up from `articles.parquet` using the **first article in `article_ids_clicked`** (since `article_id` is 70% null, but `article_ids_clicked` is always populated for labelable rows):

| Feature | Type | Range | Notes |
|---------|------|-------|-------|
| `category_idx` | Int (for embedding) | [4, 31] | 32 possible categories; not all appear in clicked articles |
| `article_type_idx` | Int (for embedding) | [1, 15] | 16 possible types; not all appear |
| `premium` | Float | {0, 1} | Binary flag |
| `sentiment_score` | Float | [0.395, 0.998] | Sentiment from NLP model |
| `body_len_log` | Float | log1p(char count) | Article body length (log-transformed) |
| `title_len_log` | Float | log1p(char count) | Article title length (log-transformed) |
| `subtitle_len_log` | Float | log1p(char count) | Article subtitle length (log-transformed) |

The first 2 features are integer indices for embedding layers. The remaining 5 are continuous features passed directly (`article_cont_dim=5`). Content length features (5-7) were added for the LSTM pipeline to capture article length effects on engagement.

Zero rows had all-zero article features (every impression had a valid clicked article in the lookup).

### Context features (3 per sample)

| Feature | Type | Values |
|---------|------|--------|
| `device_type` | Float | {1.0, 2.0, 3.0} |
| `is_subscriber` | Float | {0.0, 1.0} |
| `is_sso_user` | Float | {0.0, 1.0} |

---

## 10. Output Formats and Shapes

### Saved numpy arrays (per split)

| File | Shape (Train) | Shape (Val) | Dtype | Description |
|------|---------------|-------------|-------|-------------|
| `history_seq.npy` | (546183, 50, 2) | (567980, 50, 2) | float32 | Normalized behavioral sequences |
| `history_lengths.npy` | (546183,) | (567980,) | int32 | Actual sequence lengths |
| `agg_features.npy` | (546183, 27) | (567980, 27) | float32 | StandardScaled aggregate features |
| `article_features.npy` | (546183, 7) | (567980, 7) | float32 | Category/type idx + continuous |
| `context_features.npy` | (546183, 3) | (567980, 3) | float32 | Device, subscriber, SSO |
| `labels.npy` | (546183,) | (567980,) | float32 | Binary engagement label |
| `user_ids.npy` | (546183,) | (567980,) | uint32 | For re-identification analysis |

### Saved scalers and metadata

| File | Contents |
|------|----------|
| `rt_scaler.pkl` | Fitted StandardScaler for log1p(read_time) |
| `agg_scaler.pkl` | Fitted StandardScaler for 27 aggregate features |
| `metadata.json` | Feature dimensions, sample counts, positive rates |

### metadata.json contents

```json
{
  "n_categories": 32,
  "n_article_types": 16,
  "history_seq_len": 50,
  "n_train": 546183,
  "n_val": 567980,
  "train_pos_rate": 0.4015,
  "val_pos_rate": 0.4090,
  "agg_feature_dim": 27,
  "article_feature_dim": 7,
  "article_cont_dim": 5,
  "context_feature_dim": 3,
  "history_feature_dim": 2
}
```

### PyTorch DataLoader batch format

Each batch is a dict with these keys:

| Key | Shape | Dtype | Used by |
|-----|-------|-------|---------|
| `history_seq` | (B, 50, 2) | float32 | LSTM |
| `history_length` | (B,) | int64 | LSTM (masking) |
| `agg_features` | (B, 27) | float32 | Both |
| `article_cat` | (B,) | int64 | Both (embedding) |
| `article_type` | (B,) | int64 | Both (embedding) |
| `article_cont` | (B, 5) | float32 | Both (premium, sentiment, content lengths) |
| `context` | (B, 3) | float32 | Both |
| `label` | (B,) | float32 | Both (target) |
| `user_id` | (B,) | int64 | Privacy experiments |

Batch size: 512. At this size, train = 1,067 batches/epoch, val = 1,110 batches/epoch. Batch loading takes ~7.4ms -- well under the compute budget.

---

## 11. Verification and Audit Results

We ran 9 systematic audits on the processed data. All passed.

### Audit 1: Label correctness
- Recomputed labels independently from raw parquets
- 546,183 rows, 219,308 positive -- exact match
- 5 spot-checked rows including edge cases (rt=29s -> correctly 0)
- **Result: PASS**

### Audit 2: History sequence extraction
- 5 random rows verified against raw parquet `scroll_percentage_fixed` values
- Confirmed "last 50" extraction: values match the tail of the raw list
- Padding positions all zero across 1,000 checked rows
- No users with empty history (min length = 5)
- **Result: PASS**

### Audit 3: Read time normalization
- Non-padded values on train: mean = 0.009, std = 1.002 (essentially N(0,1))
- Scroll percentage: range [0, 1], mean = 0.625 (consistent with raw mean ~62.5%)
- All padding confirmed zero
- **Result: PASS**

### Audit 4: Aggregate feature scaling
- Train: overall mean = 0.000, std = 1.000 (StandardScaler working correctly)
- Per-feature ranges reasonable, no extreme outliers indicating data corruption
- **Result: PASS**

### Audit 5: Article feature lookup
- 5 spot-checked rows: category, article_type, premium, sentiment all match raw articles.parquet
- 0 rows with all-zero features (100% lookup success)
- **Result: PASS**

### Audit 6: Context feature extraction
- 3 spot-checked rows match raw behaviors.parquet values
- device_type has 3 values, subscriber and SSO are binary -- correct
- **Result: PASS**

### Audit 7: No train/val data leakage
- Val aggregate mean = 0.037 (not 0) -- train scaler applied, not refitted
- Val RT mean = 0.011 -- same confirmation
- **Result: PASS (no leakage)**

### Audit 8: User ID consistency
- Train: 39,517 unique users. Val: 39,657 unique users.
- Overlap: 30,051 (75.8% of val) -- matches expected ~76% from EB-NeRD temporal design
- Total unique: 49,123 (slightly < 50,000 because some users only had unlabelable impressions)
- **Result: PASS**

### Audit 9: Data cleanliness
- All 12 numpy arrays (6 per split) checked for NaN and Inf
- Zero NaN, zero Inf across all arrays
- **Result: PASS**

---

## 12. Design Decisions and Rationale

### Why truncate history to 50 (not 100 or full length)?

The median user has 92 articles in history, and 88.5% of samples truncate at 50. We chose 50 because:
- **Computational cost**: LSTM training scales linearly with sequence length. 50 vs 100 halves the per-epoch time.
- **Recency bias**: The most recent behavior is the strongest predictor. Articles from 3 weeks ago are less relevant.
- **Full information preserved**: The MLP path still gets aggregate statistics from the full history (all percentiles, mean, std). Only the LSTM's sequential view is truncated.
- **Can be revisited**: If the LSTM underperforms, increasing to 100 is a one-parameter change in `preprocessing.py`.

### Why log1p + StandardScaler for read time?

Read time is extremely right-skewed (median 16s, P99 1052s, max 1800s). Without log1p:
- Most values cluster near 0, providing almost no gradient signal
- Outliers dominate the mean and variance
- StandardScaler alone would center around ~69s (the mean), making most values strongly negative

With log1p first, the distribution becomes approximately normal, and StandardScaler produces a well-behaved N(0,1) input. This matters for both LSTM convergence and gradient flow.

### Why use first clicked article (not article_id)?

The `article_id` column in behaviors is null for 70% of rows (it represents the article being read when the impression occurred, which is often a front page with no specific article). The `article_ids_clicked` column is always populated (100% of labelable rows have at least 1 click). Using the first clicked article gives us article context for every single training sample.

### Why drop gender/age/postcode?

These demographic columns are 93-98% null. Including them would mean either:
- Imputing values for 93%+ of samples (introducing massive noise)
- Only training on the ~3-7% with data (losing 93%+ of samples)

Neither option is viable. The remaining context features (device_type, subscriber, SSO) have 0% null rate and still capture useful user-level signal.

### Why not include article_id embeddings in history?

The history contains article IDs for past articles, and we could learn per-article embeddings. We chose not to because:
- **125K unique articles** would create a large embedding table (~125K x 64 = 32M parameters) that dominates the model
- For our privacy focus, the behavioral signals (read_time, scroll_percentage) ARE the sensitive data we want to protect. Article IDs are public information.
- Keeping the LSTM input to just 2 behavioral features makes the user representation purely behavioral, which cleanly separates the privacy-sensitive signal from non-sensitive article context.

---

*Generated by `src/data/preprocessing.py` and validated by `src/data/dataset.py`.*
*Processed data saved to `data/processed/` (557 MB).*
*Runtime: ~128 seconds on Apple M3 Max.*
