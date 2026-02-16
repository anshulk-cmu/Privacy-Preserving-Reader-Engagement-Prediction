# Dataset Construction & Exploratory Data Analysis

**Privacy-Preserving Reader Engagement Prediction**
**94-806 Privacy in the Digital Age | Carnegie Mellon University**

---

## Table of Contents

1. [Dataset Source: EB-NeRD](#1-dataset-source-eb-nerd)
2. [Why We Built a Custom 50K-User Dataset](#2-why-we-built-a-custom-50k-user-dataset)
3. [How the Original EB-NeRD Splits Work](#3-how-the-original-eb-nerd-splits-work)
4. [Custom Dataset Construction](#4-custom-dataset-construction)
5. [EDA Results: Engagement Signal Distributions](#5-eda-results-engagement-signal-distributions)
6. [EDA Results: Engagement Label Analysis](#6-eda-results-engagement-label-analysis)
7. [EDA Results: User-Level Statistics](#7-eda-results-user-level-statistics)
8. [EDA Results: User History Characteristics](#8-eda-results-user-history-characteristics)
9. [EDA Results: Privacy-Relevant Behavioral Uniqueness](#9-eda-results-privacy-relevant-behavioral-uniqueness)
10. [EDA Results: Read Time vs Scroll Percentage Correlation](#10-eda-results-read-time-vs-scroll-percentage-correlation)
11. [EDA Results: Train vs Validation Consistency](#11-eda-results-train-vs-validation-consistency)
12. [Implications for Modeling and Privacy](#12-implications-for-modeling-and-privacy)

---

## 1. Dataset Source: EB-NeRD

The **Ekstra Bladet News Recommendation Dataset (EB-NeRD)** was published by Ekstra Bladet (a major Danish news outlet) and JP/Politikens Hus for the RecSys 2024 Challenge. The accompanying paper is available at [arxiv.org/abs/2410.03432](https://arxiv.org/abs/2410.03432).

The dataset contains behavioral logs from 1,103,602 "active users" (defined as those with 5-1,000 news clicks in a 3-week observation window). It is distributed in three sizes:

| Size | Unique Users | Description |
|------|-------------|-------------|
| `ebnerd_demo` | ~5,000 | Tiny subset for quick testing |
| `ebnerd_small` | ~18,800 | Small subset for development |
| `ebnerd_large` | ~975,000 | Near-complete pool |

Each size contains the same file structure and follows the same temporal splitting methodology. The smaller sizes are strict random subsets of the larger ones (we confirmed 100% containment of `ebnerd_small` users within `ebnerd_large`).

### Data Files

For each split (train, validation):

- **`behaviors.parquet`** — One row per impression event. Contains the article shown, user ID, impression timestamp, read time, scroll percentage, device type, articles in view, articles clicked, and demographic fields (SSO status, gender, postcode, age, subscriber status). Also includes `next_read_time` and `next_scroll_percentage` for the subsequent page visit.

- **`history.parquet`** — One row per user. Contains the user's past click history as list columns: article IDs clicked, read times, scroll percentages, and impression timestamps. This represents the 21-day window preceding the behavior period.

Shared across splits:

- **`articles.parquet`** — One row per article (125,541 total). Contains title, subtitle, body, category, topics, NER clusters, sentiment scores, article type, premium status, and aggregate engagement statistics.

---

## 2. Why We Built a Custom 50K-User Dataset

The `ebnerd_small` dataset contains only **18,827 unique users** across train and validation. While sufficient for initial development, this felt too small for a credible privacy experiment:

- A re-identification attack on a pool of ~19K users is less meaningful than on ~50K users. With fewer users, even a naive attack has a higher baseline success rate.
- The EB-NeRD download page advertises "~50,000 users" for the small dataset, but this count includes the separately-downloadable test set. The train+validation splits we work with contain only ~19K.
- 50K users produces ~1.26M total impressions (vs ~478K), giving substantially more training data for model convergence.

We chose **not** to use the full `ebnerd_large` (~975K users) because:
- It would far exceed our compute budget for model training and randomized smoothing certification
- The 12M+ impressions would make the re-identification attack's pairwise distance computation prohibitive
- A 50K-user "proof of concept" is the right scale for a course project

---

## 3. How the Original EB-NeRD Splits Work

The authors collected 6 weeks of user behavior logs (April 27 - June 8, 2023). The data is split **temporally** into consecutive 7-day behavior windows, each with a trailing 21-day history window:

| Window | Start Date | End Date | Duration |
|--------|-----------|----------|----------|
| Train history | April 27 | May 18 | 21 days |
| **Train behaviors** | **May 18** | **May 25** | **7 days** |
| Validation history | May 4 | May 25 | 21 days |
| **Validation behaviors** | **May 25** | **June 1** | **7 days** |

Key properties of this design:

- **Non-overlapping behavior windows**: Train and validation behaviors cover different weeks. There is no data leakage — the model never sees validation-period impressions during training.
- **Overlapping history windows**: The histories overlap by 14 days (May 4-18). This is intentional — the history simply provides context for the user's past reading habits.
- **Natural user overlap (~76%)**: Users active in both weeks appear in both splits. This is not a bug — it mirrors the real-world scenario where the same user reads news across multiple weeks.
- **User subsets preserve this structure**: `ebnerd_small` and `ebnerd_large` are random user subsets drawn from the full pool. The temporal boundaries, history construction, and file schema are identical across all sizes.

---

## 4. Custom Dataset Construction

### Method

We created `ebnerd_50k` by combining users from both available datasets:

1. **Identified all users** in `ebnerd_small` (18,827) and `ebnerd_large` (974,791)
2. **Computed candidates**: 955,964 users in large but not in small
3. **Sampled 31,173 users** from candidates (NumPy random seed 42 for reproducibility)
4. **Combined**: 18,827 + 31,173 = **50,000 target users**
5. **Filtered all parquet files** from `ebnerd_large` to only rows where `user_id` is in the target set
6. **Copied `articles.parquet`** as-is (articles are shared, not user-specific)

The script is `src/01_create_50k_dataset.py`.

### Result

| Metric | ebnerd_small | ebnerd_50k |
|--------|-------------|------------|
| Total unique users | 18,827 | **50,000** |
| Train users | 15,362 | 40,332 |
| Train impressions | 233,401 | 615,316 |
| Val users | 15,435 | 40,616 |
| Val impressions | 245,088 | 645,495 |
| Train-val user overlap | 76.2% | 76.2% |
| Train behavior period | May 18-25 | May 18-25 |
| Val behavior period | May 25 - June 1 | May 25 - June 1 |
| Disk size | 89 MB | 291 MB |

The temporal structure, schema, engagement signal availability, and user overlap ratio are preserved exactly. The only thing that changed is the number of users (and consequently, the number of impressions).

### Verification Checks (all passed)

- Schema: all 4 parquet files match `ebnerd_small` column-for-column
- Temporal ranges: train May 18 07:00 to May 25 06:59, val May 25 07:00 to June 1 06:59
- Engagement signals: `next_read_time` ~97% non-null, `next_scroll_percentage` ~88% non-null
- User overlap: 30,948 users in both splits (76.2% of validation users)

---

## 5. EDA Results: Engagement Signal Distributions

*Figure: `outputs/figures/01_engagement_distributions.png`*

### Read Time (`next_read_time`)

| Statistic | Train | Validation |
|-----------|-------|------------|
| Non-null count | 598,861 (97.3%) | 628,091 (97.3%) |
| Mean | 68.57s | 69.05s |
| Median | 21.00s | 21.00s |
| Std | 172.50s | 171.40s |
| P10 | 3.0s | 3.0s |
| P25 | 7.0s | 7.0s |
| P50 | 21.0s | 21.0s |
| P75 | 62.0s | 64.0s |
| P90 | 129.0s | 132.0s |
| P95 | 232.0s | 235.0s |
| P99 | 1,052.0s | 1,042.0s |
| Max | 1,800.0s | 1,800.0s |

**Observations:**
- The distribution is heavily **right-skewed**. The median (21s) is far below the mean (69s), pulled up by a long right tail.
- Most impressions involve very short reads: half of all reads are under 21 seconds.
- The max of 1,800s (30 minutes) appears to be a platform-imposed cap.
- The 30-second engagement threshold falls just above the median, capturing the upper ~46% of read times — a meaningful cutoff that separates casual glances from substantive reads.
- A visible spike exists at the 300s clip boundary in the plot (and the long tail up to 1,800s in the raw data).

### Scroll Percentage (`next_scroll_percentage`)

| Statistic | Train | Validation |
|-----------|-------|------------|
| Non-null count | 546,183 (88.8%) | 567,980 (88.0%) |
| Mean | 69.1% | 69.0% |
| Median | 80.0% | 79.0% |
| Std | 32.3% | 32.2% |
| P10 | 23.0% | 23.0% |
| P25 | 36.0% | 36.0% |
| P50 | 80.0% | 79.0% |
| P75 | 100.0% | 100.0% |
| P90 | 100.0% | 100.0% |
| P95 | 100.0% | 100.0% |
| P99 | 100.0% | 100.0% |

**Observations:**
- The distribution is strikingly **bimodal**: a small cluster at 0-30% and a massive spike at 100%.
- From P75 onward, every percentile is 100% — meaning at least 25% of all impressions have full scroll.
- The 100% spike likely includes short articles that fit on one screen (automatically 100% scroll) as well as genuinely engaged readers who scrolled to the bottom.
- The 50% threshold cleanly separates the low-scroll cluster from the high-scroll majority.
- The ~11-12% null rate means some impressions don't have scroll tracking data (e.g., certain device types or page layouts).

---

## 6. EDA Results: Engagement Label Analysis

We define: **engaged = (scroll_percentage > 50%) AND (read_time > 30s)**

| Metric | Train | Validation |
|--------|-------|------------|
| Total impressions | 615,316 | 645,495 |
| Labelable (non-null) | 546,183 (88.8%) | 567,980 (88.0%) |
| Unlabelable (null) | 69,133 (11.2%) | 77,515 (12.0%) |
| **Engaged (label=1)** | **219,308 (40.2%)** | **232,315 (40.9%)** |
| Not engaged (label=0) | 326,875 (59.8%) | 335,665 (59.1%) |
| Class ratio (pos:neg) | 1:1.5 | 1:1.4 |

### Individual Condition Breakdown (Train)

| Condition | Count | % of labelable |
|-----------|-------|----------------|
| Scroll > 50% alone | 352,900 | 64.6% |
| Read time > 30s alone | 252,056 | 46.1% |
| **Both (engaged)** | **219,308** | **40.2%** |

**Analysis:**
- The AND condition removes ~133K impressions that had high scroll but low read time (fast scrollers who didn't actually read) and ~33K that had long read time but low scroll (users who may have had a tab open in the background).
- The **1:1.5 class ratio is near-ideal** for binary classification. No aggressive class balancing is needed. Standard cross-entropy loss will work well.
- The ~40% positive rate is consistent across train and validation, confirming label stability.
- The 11-12% null rate is manageable — we exclude those rows from training and evaluation.

---

## 7. EDA Results: User-Level Statistics

| Metric | Train | Validation |
|--------|-------|------------|
| Unique users | 40,332 | 40,616 |
| Total impressions | 615,316 | 645,495 |
| Avg impressions/user | 15.3 | 15.9 |
| Min impressions/user | 1 | 1 |
| Max impressions/user | 214 | 195 |
| Median impressions/user | 9 | 9 |

**Observations:**
- The median user has **9 impressions** per week — about 1-2 article views per day.
- The distribution is right-skewed: most users have fewer than 15 impressions, but power users go up to ~200.
- The mean (15.3) exceeds the median (9), indicating a long tail of heavy readers.
- Users with only 1 impression in a week will have limited behavioral signal for the model. However, their 21-day history (median 92 articles) provides rich context.

---

## 8. EDA Results: User History Characteristics

*Figure: `outputs/figures/02_history_distributions.png`*

### History Length (Number of Past Articles Clicked)

| Metric | Train | Validation |
|--------|-------|------------|
| Users with history | 40,332 | 40,616 |
| Min | 5 | 5 |
| Max | 1,896 | 1,277 |
| Median | 92 | 80 |
| Mean | 158.8 | 143.1 |

**Observations:**
- The minimum of 5 articles reflects the EB-NeRD "active user" threshold — users with fewer than 5 clicks were excluded from the dataset.
- The median user clicked ~92 articles over 21 days (about 4-5 articles per day).
- The distribution is heavily right-skewed, with a long tail of power readers (up to ~1,900 articles in 21 days — roughly 90 articles per day for the most active users).
- For LSTM modeling, we'll likely truncate histories to a fixed length (e.g., 50-100 most recent articles) to manage computational cost while covering the majority of users.

### Historical Read Times (Across All Past Interactions)

| Metric | Train | Validation |
|--------|-------|------------|
| Total interactions | 6,405,882 | 5,811,531 |
| Mean | 61.01s | 61.25s |
| Median | 16.00s | 16.00s |
| P10 | 2.0s | 2.0s |
| P90 | 117.0s | 118.0s |

### Historical Scroll Percentages (Across All Past Interactions)

| Metric | Train | Validation |
|--------|-------|------------|
| Total interactions | 5,738,514 | 5,255,166 |
| Mean | 68.1% | 68.1% |
| Median | 76.0% | 77.0% |
| P10 | 22.0% | 22.0% |
| P90 | 100.0% | 100.0% |

**Key finding:** The historical distributions closely mirror the current behavior distributions. This **temporal stability of reading patterns** is critical for two reasons:
1. **For modeling**: Past behavior is a strong predictor of future behavior — the input signal (history) is informative for the output (engagement prediction).
2. **For privacy**: Stable behavioral patterns are exactly what makes re-identification attacks feasible. A user's reading fingerprint persists across time windows.

---

## 9. EDA Results: Privacy-Relevant Behavioral Uniqueness

*Figure: `outputs/figures/03_pairwise_distances.png`*

This is the most important section for motivating our privacy work.

### Behavioral Fingerprint Construction

For each user, we computed a 4-dimensional "behavioral fingerprint" from their history:
- `rt_mean`: Mean historical read time
- `rt_std`: Standard deviation of historical read times
- `sp_mean`: Mean historical scroll percentage
- `sp_std`: Standard deviation of historical scroll percentages

Plus `hist_len` (number of articles in history) for a 5D feature space.

### Uniqueness Analysis

| Metric | Value |
|--------|-------|
| Users analyzed | 40,315 |
| Unique 4D fingerprints (rounded to 1 decimal) | 40,315 |
| **Uniqueness ratio** | **100.0%** |
| Re-identification risk | **HIGH** |

**Every single user in the dataset has a unique behavioral fingerprint**, even when the four summary statistics are rounded to one decimal place. This means an attacker with access to aggregate behavioral statistics could in principle identify any user.

### Pairwise Distance Analysis

We computed Euclidean distances between all pairs of 1,000 randomly sampled users in standardized feature space:

| Metric | Value |
|--------|-------|
| Mean distance | 2.814 |
| Median distance | 2.544 |
| Min distance | 0.087 |
| P05 | 0.993 |
| P95 | 5.560 |

**Interpretation:**

- The distribution is approximately chi-distributed (expected for L2 distances in standardized multivariate space), with a mode around 1.8-2.0.
- The **minimum distance of 0.087** means some user pairs are extremely close in behavioral space — but still distinguishable.
- The **P05 at 0.993** means 5% of all user pairs are within ~1 standard deviation of each other. For a nearest-neighbor re-identification attack, what matters is whether each user's *closest* neighbor is far enough away to prevent matching.
- The bulk of the distribution (mode ~2, mean ~2.8) shows that most users are well-separated — but the left tail (distances < 1) is where re-identification becomes easy.

**Why this matters for our experiment:**
- The 100% uniqueness confirms that privacy protection is *necessary* — without it, users are trivially identifiable from their behavioral summary statistics.
- The pairwise distance distribution provides a baseline. After applying randomized smoothing, we can re-measure these distances in the *representation space* to quantify how much the smoothing degrades fingerprint distinctiveness.
- The certified radius R from randomized smoothing needs to be calibrated relative to these distances. If R is much smaller than the typical nearest-neighbor distance, the smoothing won't provide meaningful privacy. If R is too large, predictions become useless.

---

## 10. EDA Results: Read Time vs Scroll Percentage Correlation

*Figure: `outputs/figures/04_rt_vs_sp_scatter.png`*

**Pearson correlation: r = 0.172** (weak positive)

### Visible Patterns in the Scatter Plot

1. **Dense cluster at (0-10s, 0-40%)** — "Bounce" interactions where users barely read and barely scrolled. This is the largest concentration of points.

2. **Thick horizontal band at scroll = 100%** — Users who scrolled to the bottom span the entire read-time axis (1s to 300s+). Scroll = 100% alone does not indicate engagement — many of these are short articles viewed in seconds.

3. **Vertical band at read_time < 5s** — Very short reads span all scroll percentages. These are likely accidental opens or very short articles.

4. **Sparse bottom-right region** — Very few users spend a long time (>100s) without scrolling past 50%. Those who do may have a tab open in the background.

5. **The "engaged" quadrant** (above the red crosshairs at RT=30s and SP=50%) is well-populated and clearly identifies a meaningful subset of interactions.

### Why the Weak Correlation is Good News

A Pearson r of 0.172 means read time and scroll percentage capture **largely independent dimensions of engagement**. If they were highly correlated (r > 0.8), using both in the label would be redundant. At r = 0.17:
- **Scroll percentage** catches *content consumption depth* (did they see the full article?)
- **Read time** catches *time investment* (did they actually process what they saw?)

The AND combination filters out two types of false positives:
- **Quick scrollers**: High scroll, low read time — scrolled through without reading
- **Idle tabs**: High read time, low scroll — page was open but user wasn't actively reading

---

## 11. EDA Results: Train vs Validation Consistency

| Metric | Train | Validation |
|--------|-------|------------|
| Unique users | 40,332 | 40,616 |
| Overlap | 30,948 (76.2% of val) |
| Train-only users | 9,384 |
| Val-only users | 9,668 |
| **Total unique** | **50,000** |

The distributions of all key signals are nearly identical across splits:
- Read time mean: 68.57s (train) vs 69.05s (val)
- Scroll percentage mean: 69.1% (train) vs 69.0% (val)
- Engagement rate: 40.2% (train) vs 40.9% (val)
- Impressions per user: 15.3 (train) vs 15.9 (val)

This consistency confirms:
1. Our random user sampling from `ebnerd_large` produced a representative subset.
2. No meaningful distribution shift exists between the two consecutive weeks.
3. Models trained on the train split should generalize well to validation without confounding from temporal drift.

---

## 12. Implications for Modeling and Privacy

### For Engagement Prediction (MLP Baseline & LSTM)

- **Data volume is sufficient**: 546K labelable train impressions with ~40% positive rate provides ample data for LSTM models.
- **History sequences are informative**: The temporal stability of behavioral patterns (historical and current distributions match) means past behavior is a strong predictor of future engagement.
- **History truncation needed**: With history lengths ranging from 5 to 1,896, we should truncate to a fixed length (e.g., most recent 50-100 articles) for computational efficiency.
- **Two-signal input**: Read time and scroll percentage provide complementary, low-correlation (r=0.17) features — both should be used as input dimensions in behavioral sequences.

### For Privacy (Randomized Smoothing & Re-identification Attacks)

- **Re-identification risk is real and quantified**: 100% behavioral fingerprint uniqueness at coarse granularity. This is not a hypothetical risk — it's measurable.
- **Pairwise distance baseline established**: Mode ~2.0, min ~0.09 in standardized space. The randomized smoothing certified radius needs to be calibrated against this distribution.
- **Attack surface is well-defined**: With 50K users and stable behavioral fingerprints, a nearest-neighbor re-identification attack is both feasible and meaningful.
- **Privacy-utility tradeoff is measurable**: We have clear "before" metrics (uniqueness ratio, pairwise distances, attack accuracy without smoothing) that we can compare against "after" metrics (same measures with smoothing applied at various noise levels).

---

*Generated by `src/00_eda.py` on the `ebnerd_50k` dataset (50,000 users, constructed via `src/01_create_50k_dataset.py`).*
*Figures saved to `outputs/figures/`.*

*Next: [Phase 2 — Data Pipeline](02_data_pipeline.md) | [Phase 3A — MLP Baseline & Re-identification](03_mlp_baseline_analysis.md)*
