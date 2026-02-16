# Phase 2B: MLP Baseline — Model, Re-identification, and Analysis

**Privacy-Preserving Reader Engagement Prediction**
**94-806 Privacy in the Digital Age | Carnegie Mellon University**

---

## Table of Contents

1. [What We Did and Why](#1-what-we-did-and-why)
2. [Model Architecture](#2-model-architecture)
3. [Loss Function](#3-loss-function)
4. [Training Process](#4-training-process)
5. [Engagement Prediction Results](#5-engagement-prediction-results)
6. [Representation Quality](#6-representation-quality)
7. [Blind Re-identification Test](#7-blind-re-identification-test)
8. [Re-identification Results](#8-re-identification-results)
9. [Limitations of the MLP Approach](#9-limitations-of-the-mlp-approach)
10. [Generated Outputs](#10-generated-outputs)
11. [Key Takeaways](#11-key-takeaways)

---

## 1. What We Did and Why

### The Goal

We built a neural network that predicts whether a user will be "engaged" with a news article (scrolls past 50% AND reads for more than 30 seconds). But predicting engagement is not the real point — it's the vehicle. The real goal is to investigate a privacy question:

**When a model learns to predict engagement, does it also learn to identify individual users?**

If yes, that's a privacy risk. It means any company that trains an engagement model (Netflix, Spotify, news apps) is inadvertently creating a tool that could de-anonymize their users.

### Why an MLP?

An MLP (Multi-Layer Perceptron) is a straightforward neural network that takes fixed-size input and produces a prediction. It's our **baseline** — the simplest deep learning approach. We chose it because:

1. It operates on 42 pre-computed summary statistics (averages, standard deviations, percentiles of reading behavior). These are simple numbers, not sequences.
2. It establishes the floor of what's possible. If even a simple model on compressed features can fingerprint users, the privacy risk is real.
3. It trains in ~10 minutes on a laptop, making iteration fast.

The LSTM (next phase) will use raw temporal sequences and should perform much better.

### What Exactly Goes Into the Model

Each reading session is described by 42 numbers:

- **21 aggregate history features**: Statistics about the user's past reading behavior — mean, standard deviation, median, 25th/75th percentiles of read times and scroll percentages, plus the last 5 read times and scroll percentages, and the total number of articles in their history.
- **16 article embedding dimensions**: The category (e.g., "sports," "politics") and article type (e.g., "feature," "breaking news") are converted into 16-dimensional learned vectors (8 dims each in v1, 16 each in v3).
- **2 continuous article features**: Whether the article is premium (0 or 1) and its sentiment score (0.4 to 1.0).
- **3 context features**: Device type (mobile/desktop/tablet), whether the user is a subscriber, and whether they logged in via SSO.

---

## 2. Model Architecture

### Design Iterations

We went through three versions of the architecture, each fixing problems found in the previous one:

**Version 1 (120K parameters)**
- 4 layers: 256 → 256 → 128 → 64
- GELU activation, dropout 0.2-0.15
- Xavier initialization
- Standard BCEWithLogitsLoss
- Result: AUC 0.6824, but we hadn't yet checked representation quality

**Version 2 (552K parameters)**
- 7 layers: 512 → 512 → 256 → 256 → 128 → 128 → 64
- GELU activation, dropout 0.15-0.10
- Kaiming initialization
- LabelSmoothingBCE loss
- Problem: Massively over-parameterized. The first layer was 512 neurons wide for just 58 input features — that's like building a 10-lane highway for a quiet residential street. Audit revealed:
  - **6 out of 64 representation dimensions were dead** (standard deviation < 0.01). The GELU activation has a minimum value of approximately -0.170, and neurons that consistently received negative inputs were permanently clamped there.
  - **Growing overfitting gap**: Train AUC kept climbing to 0.703 while val AUC was stuck at 0.68. The gap grew from +0.003 to +0.026 over training.
  - **Representations were asymmetric**: All values were positive-skewed in the range [-0.17, +3.6] instead of being centered around zero.

**Version 3 (207K parameters) — Current**
- 6 layers: 256 → 256 → 256 → 128 → 128 → 64
- **SiLU activation** (also called Swish) instead of GELU — smoother gradient flow with no hard floor, so neurons don't die
- **No activation on the representation layer** (LayerNorm only) — allows full symmetric range instead of clamping to positive values
- 2 residual connections at same-width layer pairs
- Kaiming initialization, uniform dropout 0.2
- Fixes:
  - Dead dimensions: 6 → 1
  - Overfitting gap: +0.0125 → +0.0105
  - Representation range: [-0.17, +3.6] → [-3.4, +3.9] (symmetric, centered near 0)
  - AUC actually improved slightly: 0.6807 → 0.6817

### Final Architecture Diagram

```
Input Features (58 total)
    ├── Aggregate history stats (21 floats, StandardScaled)
    ├── Category embedding (16 dims, learned)
    ├── Article type embedding (16 dims, learned)
    ├── Article continuous [premium, sentiment] (2 floats)
    └── Context [device, subscriber, sso] (3 floats)
         │
         ▼
    Linear(58 → 256) → LayerNorm → SiLU → Dropout(0.2)
         │
         ▼
    [ResidualBlock: Linear(256 → 256) → LN → SiLU → Drop(0.2)] + skip
         │
         ▼
    Linear(256 → 256) → LayerNorm → SiLU → Dropout(0.2)
         │
         ▼
    Linear(256 → 128) → LayerNorm → SiLU → Dropout(0.2)
         │
         ▼
    [ResidualBlock: Linear(128 → 128) → LN → SiLU → Drop(0.2)] + skip
         │
         ▼
    Linear(128 → 64) → LayerNorm          ← REPRESENTATION (64-dim)
         │
         ▼
    Linear(64 → 1)                         ← LOGIT OUTPUT
```

Total: **207,361 trainable parameters**.

---

## 3. Loss Function

### The Problem

Our dataset has 40% engaged users and 60% not-engaged users. If we use a standard loss function (BCEWithLogitsLoss), the model tends to be conservative — it predicts "not engaged" too often because that's the majority class. This gives high precision but terrible recall (only catching 42% of engaged users in v1).

### What We Tried

**Attempt 1: Focal Loss (gamma=2.0)**
Focal Loss was invented for object detection where the imbalance is extreme (1 positive : 10,000 negatives). It aggressively down-weights "easy" examples to focus on "hard" ones. But our imbalance is mild (40:60), so gamma=2.0 was too aggressive — recall collapsed to 21%. The model became hyper-conservative, only predicting "engaged" when it was very confident.

**Attempt 2: LabelSmoothingBCE with pos_weight (current)**
Two techniques combined:
- **pos_weight = 1.49**: Tells the model "a positive example is worth 1.49 negative examples." This compensates for the class imbalance by upweighting the minority engaged class. Calculated as (1 - 0.40) / 0.40 = 1.49.
- **Label smoothing = 0.05**: Instead of hard labels (0 and 1), the model trains on softened labels (0.05 and 0.95). This prevents overconfident predictions and improves calibration — the model learns "nothing is 100% certain."

Result: Balanced precision (0.56) and recall (0.58), F1 = 0.57.

---

## 4. Training Process

### Setup

- **Optimizer**: AdamW (lr=0.001, weight_decay=0.0005)
- **Scheduler**: OneCycleLR with cosine annealing (ramps up learning rate for 10% of training, then gradually decreases)
- **Batch size**: 512
- **Max epochs**: 50 (with early stopping at patience=8)
- **Gradient clipping**: max norm 1.0
- **Device**: Apple M3 Max MPS (GPU)

### What Happened During Training

The model trained for 21 epochs and was early-stopped. The best model was saved at epoch 13.

- **Epochs 1-7**: Rapid improvement. Val AUC climbed from 0.672 to 0.681.
- **Epochs 7-13**: Slow refinement. Val AUC inched from 0.681 to 0.682. The model was squeezing out the last bits of signal from the aggregate features.
- **Epochs 13-21**: Plateau and mild degradation. Val AUC started drifting down while train AUC kept improving — a sign the model was starting to memorize training data rather than learning generalizable patterns.

Early stopping correctly identified epoch 13 as the best point and restored those weights.

### Training Time

~10 minutes total on Apple M3 Max (28 seconds per epoch).

---

## 5. Engagement Prediction Results

### Final Metrics (Best Model, Epoch 13)

| Metric | Train | Validation | Gap | Interpretation |
|--------|-------|------------|-----|----------------|
| AUC-ROC | 0.6922 | 0.6817 | +0.010 | Model ranks engaged users higher than non-engaged 68% of the time |
| F1 | 0.5729 | 0.5694 | +0.004 | Harmonic mean of precision and recall; balanced |
| Accuracy | 0.6460 | 0.6384 | +0.008 | 64% of predictions are correct |
| Precision | 0.5556 | 0.5551 | +0.001 | 56% of "engaged" predictions are actually engaged |
| Recall | 0.5914 | 0.5844 | +0.007 | 58% of truly engaged users are correctly identified |
| Avg Precision | 0.6039 | 0.5962 | +0.008 | Area under precision-recall curve |

### Confusion Matrix (Validation Set)

|  | Predicted: Not Engaged | Predicted: Engaged |
|---|---|---|
| **Actual: Not Engaged** | 226,851 (67.6%) | 108,814 (32.4%) |
| **Actual: Engaged** | 96,549 (41.6%) | 135,766 (58.4%) |

The model correctly identifies 58% of engaged users and correctly rejects 68% of non-engaged users. It's moderately useful but far from perfect. This is expected — we're only using 42 summary statistics, not the rich temporal reading sequences.

### What These Numbers Mean in Plain English

- **AUC 0.68**: If you picked one engaged user and one non-engaged user at random, the model would correctly rank the engaged user higher 68% of the time (50% = random, 100% = perfect).
- **The train-val gap is only +0.01**: The model is not overfitting. It generalizes almost as well to unseen data as it does to training data.
- **Precision ≈ Recall ≈ 0.56**: The model is balanced. It's not biased toward predicting one class over the other.

---

## 6. Representation Quality

### What Are Representations?

The model has 6 layers. The second-to-last layer outputs a 64-dimensional vector — a set of 64 numbers that captures everything the model has learned about that reading session. We call this the **representation** or **embedding**. It's like a compressed summary of the user's behavioral profile.

These representations are what we use for the privacy experiments. If two representations are very similar, it means the model thinks those two sessions "look alike" — potentially because they came from the same user.

### Representation Statistics

| Property | Value | Interpretation |
|----------|-------|----------------|
| Dimensions | 64 | Each session is described by 64 numbers |
| Mean | -0.009 | Centered near zero (good — symmetric) |
| Std | 0.824 | Reasonable spread across the space |
| Range | [-3.38, +3.93] | Symmetric (fixed from v2's [−0.17, +3.6]) |
| Dead dims (std < 0.01) | 1 / 64 | Almost none (fixed from v2's 6/64) |
| Unique representations | 98.0% | Model creates distinctive vectors, not collapsing |

### Key Observations

**Representations are nearly all unique** (98.0% of 568K vectors are distinct when rounded to 4 decimal places). This means the model is producing diverse, distinctive vectors — not collapsing all sessions into a few clusters.

**The representation space is symmetric and centered.** Thanks to LayerNorm on the final layer (without activation), the representations span [-3.4, +3.9] centered at zero. This is healthy for distance-based comparisons in the re-identification attack.

**29 out of 64 dimensions have low variance** (std < 0.05 across the dataset). This means the model concentrates most of its information into about 35 "active" dimensions. The remaining 29 carry minimal signal. This is normal — 64 dimensions is more capacity than needed for 42 input features. The model uses what it needs and leaves the rest near-constant.

**Class separation is weak (ratio 1.07).** The distance between an engaged user's representation and a non-engaged user's representation (inter-class) is barely larger than the distance between two engaged users (intra-class). The model is not strongly separating the two classes in representation space — consistent with AUC 0.68.

---

## 7. Blind Re-identification Test

### The Setup

This is the core privacy experiment. We tested whether the model's learned representations can be used to identify individual users — even though the model was never trained to do this.

**Why "blind"?** The model was trained on training data only. The re-identification test uses validation data exclusively. The model never saw any of these sessions during training.

### Methodology

**Step 1: Select eligible users.**
We took all validation set users who had at least 4 reading sessions (impressions). This left us with **28,361 users** out of 39,657 total. Users with fewer than 4 sessions were dropped because we need enough data to split between gallery and probe.

**Step 2: Split each user's sessions 50/50.**
For each user, we randomly split their sessions into two halves:
- **Gallery (first half)**: These sessions are averaged together to create one "behavioral profile" per user — their fingerprint. Think of it as: "This is what User #12345's reading style looks like."
- **Probe (second half)**: These are individual sessions with the user's identity removed. The question: "This anonymous session — who does it belong to?"

This gave us 28,361 gallery profiles and 280,423 probe sessions.

**Step 3: The attack.**
For each of the 280,423 anonymous probe sessions:
1. Compute the distance between its 64-dim representation and all 28,361 gallery profiles
2. Rank all gallery profiles by distance (closest first)
3. Check: is the correct user ranked #1? Top-5? Top-10?

If the correct user is ranked #1, the attack perfectly re-identified that session. We tested both **euclidean distance** (straight-line distance in 64-dim space) and **cosine distance** (angle between vectors, ignoring magnitude).

**Step 4: Compare against random baseline.**
A random guess would rank the correct user at a random position among 28,361 users. The expected Top-1 accuracy by chance is 1/28,361 = 0.0035%.

---

## 8. Re-identification Results

### Top-Level Numbers

| Metric | Cosine (Best) | Euclidean | Random Baseline | Lift |
|--------|--------------|-----------|-----------------|------|
| Top-1 Accuracy | **1.11%** | 0.93% | 0.0035% | **314x** |
| Top-5 Accuracy | **1.94%** | 1.58% | 0.018% | **109x** |
| Top-10 Accuracy | **2.45%** | 2.03% | 0.035% | **70x** |
| Top-20 Accuracy | **3.18%** | 2.67% | 0.071% | **45x** |
| MRR | **0.0168** | 0.0142 | 0.0004 | **44x** |
| Mean Rank | 3,893 | 3,891 | ~14,180 | — |
| Median Rank | 2,266 | 2,242 | ~14,180 | — |

### What These Numbers Mean

**Top-1 = 1.11%**: Out of 280,423 anonymous sessions, 3,100 were correctly linked back to the exact right user among 28,361 candidates. That's 314 times better than guessing randomly.

**Median Rank = 2,266**: For a typical anonymous session, the correct user is ranked around position 2,266 out of 28,361. That means the attacker can narrow the suspect pool to the **top 8%** of users, eliminating 92% of candidates.

**Cosine beats Euclidean by ~20%**: Cosine distance measures the *direction* of the representation (what pattern the user has), while euclidean measures both direction and *magnitude* (how large the vector is). Since LayerNorm forces all representations to have similar magnitude (~6.5), magnitude carries almost no information — so cosine, which ignores magnitude, performs better.

### Per-User Breakdown

- **34 users were re-identified 100% of the time** — every single one of their anonymous sessions was correctly matched. These users have extremely consistent, distinctive reading patterns.
- **26,350 users (93%) were never re-identified** — the attack failed on all their sessions. These users have reading patterns that overlap heavily with other users.
- **Users with fewer sessions (4-5) were easier to re-identify (~4% accuracy)** than users with many sessions (100+, ~0.5% accuracy). This is counterintuitive but makes sense: users with few sessions have more consistent profiles (less noise in the gallery average), while users with many sessions have more varied behavior that makes the gallery average less representative of any single session.

### The Privacy Argument

The attack numbers may seem small (1.11%), but the point is not whether the attack is devastatingly effective. The point is:

1. **It shouldn't work at all.** The model was trained only to predict engagement. It was never given user IDs as a training signal. Yet its internal representations are distinctive enough to re-identify users at 314x above chance.

2. **This is the weakest possible version.** We used only 42 summary statistics, not raw temporal sequences. We used the simplest model (MLP), not a sequence model. The LSTM with full behavioral sequences should produce dramatically more distinctive representations.

3. **The attack narrows the suspect pool by 92%.** Even when the exact match fails, the correct user is typically in the top 8% of candidates. Combined with other side information (location, time of day, device), this could enable full de-anonymization.

4. **This is an unintentional byproduct.** No one asked the model to learn user fingerprints. It happened automatically because predicting engagement requires understanding individual reading patterns — and individual reading patterns are inherently distinctive.

---

## 9. Limitations of the MLP Approach

### Why AUC Is Capped at ~0.68

The MLP sees only 42 aggregate features — pre-computed summary statistics like mean read time, scroll standard deviation, and percentiles. These **compress away temporal information**:

- **Temporal trends are invisible**: A user whose reading time is *increasing* over the week looks identical to one whose reading time is *decreasing*, if their averages are the same.
- **Session patterns are lost**: Binge-reading sessions (10 articles in an hour) vs. sparse reading (1 article per day) produce similar averages but have very different engagement implications.
- **Recency effects are limited**: Only the last 5 read times/scroll percentages are preserved. The ordering and gaps between sessions are discarded.

This is an inherent **information bottleneck**. No architectural change (deeper layers, transformers, fancy loss functions) can recover information that was lost during feature engineering. We confirmed this experimentally: going from 120K to 552K parameters didn't improve AUC at all.

### Why Re-identification Is "Only" 1.11%

The same information bottleneck limits re-identification. The MLP's representations encode compressed summaries, not raw behavioral sequences. Two users with similar average reading times, similar scroll depths, and similar recent behavior will have nearly identical representations — even if their moment-to-moment patterns are completely different.

The LSTM (Phase 2C) will operate on the raw sequences of (read_time, scroll_percentage) for the last 50 articles. This should capture:
- Sequential ordering and temporal dependencies
- Burstiness and session-level patterns
- Fine-grained behavioral signatures that summary statistics wash out

We expect the LSTM to significantly improve both engagement prediction (AUC 0.72-0.78) and re-identification rates, making the privacy risk more dramatic and the case for randomized smoothing more compelling.

### Low-Variance Dimensions

29 of 64 representation dimensions have very low variance across the dataset (std < 0.05). These dimensions carry almost no discriminative information. The effective representation dimensionality is closer to ~35. This is not a bug — it simply means 64 dimensions is more capacity than the model needs for 42 input features.

### The Engagement Prediction Is Moderate, Not Great

AUC 0.68 means the model is only moderately better than random at predicting engagement. For a production system, this wouldn't be useful. But for our privacy research, it's sufficient:

1. The model has learned *something* about user behavior — enough to create semi-distinctive representations.
2. The fact that even a moderate model creates a privacy risk strengthens the argument: you don't need a great engagement predictor to inadvertently fingerprint users.
3. Better models (LSTM) will make both the predictions and the fingerprints stronger, escalating the demonstrated risk.

---

## 10. Generated Outputs

All outputs are saved in `outputs/models/mlp_baseline/`:

| File | Description |
|------|-------------|
| `checkpoint.pt` | Best model weights (epoch 13, 6.6 MB) |
| `metrics.json` | Complete training history — per-epoch loss, AUC, F1, accuracy, precision, recall for both train and val, plus training config |
| `representations.npz` | 64-dim representation vectors for all 546K train + 568K val impressions, with user IDs and labels |
| `training_curves.png` | 6-panel plot: loss curves, val AUC progression, all val metrics, overfitting gap, val loss delta, precision-recall trajectory |
| `evaluation_plots.png` | 4-panel plot: ROC curve (AUC=0.6817), precision-recall curve (AP=0.5962), confusion matrix, prediction probability distribution |
| `representation_analysis.png` | 3-panel plot: t-SNE of 5K val samples, L2 norm distribution by class, per-dimension mean activation |
| `reidentification_cosine.png` | 6-panel re-identification analysis with cosine distance |
| `reidentification_euclidean.png` | 6-panel re-identification analysis with euclidean distance |
| `reidentification_comparison.png` | Side-by-side comparison of euclidean vs cosine vs random |
| `reidentification_results.json` | All numerical re-identification results (Top-K accuracies, MRR, per-user stats, baseline) |

---

## 11. Key Takeaways

1. **An engagement model inadvertently learns to fingerprint users.** Even a simple MLP on aggregate statistics creates representations that re-identify users at 314x above random chance. This is a real, measurable privacy risk.

2. **The model was never trained to identify users.** It was trained only to predict engagement (engaged vs. not engaged). User identification is an unintentional byproduct of learning behavioral patterns.

3. **The attack narrows the suspect pool to the top 8%.** Even when exact re-identification fails, the attacker eliminates 92% of candidates, which combined with other data could enable full de-anonymization.

4. **This is the weakest demonstration.** Using only summary statistics and a baseline MLP, re-identification already works. Raw temporal sequences (LSTM) should make the fingerprints dramatically more distinctive.

5. **The information bottleneck limits both engagement prediction and re-identification.** Aggregate features compress away temporal patterns. This caps AUC at ~0.68 and re-identification at ~1%. The LSTM will lift both limits.

6. **Model architecture matters less than input information.** Going from 120K to 552K parameters didn't improve AUC. The bottleneck is the 42 aggregate features, not the network capacity. We settled on 207K parameters as the efficient sweet spot.

7. **Careful loss function design matters.** Focal Loss (gamma=2) was too aggressive for our mild 40:60 imbalance. LabelSmoothingBCE with pos_weight=1.49 produced the most balanced results.

8. **Representation engineering matters for privacy experiments.** Switching from GELU to SiLU and removing the activation on the representation layer fixed dead dimensions (6→1), centered the representation space, and doubled the usable range — all important for distance-based re-identification.

---

*Next: Phase 2C — LSTM with raw temporal sequences, expected to significantly increase both engagement prediction accuracy and the demonstrated re-identification risk.*
