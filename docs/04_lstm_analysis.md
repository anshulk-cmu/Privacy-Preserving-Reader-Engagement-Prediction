# Phase 3B: LSTM Model — Beyond the MLP Baseline

**Privacy-Preserving Reader Engagement Prediction**
**94-806 Privacy in the Digital Age | Carnegie Mellon University**

---

## Table of Contents

1. [Motivation: Why Go Beyond the MLP?](#1-motivation-why-go-beyond-the-mlp)
2. [What the MLP Could Not See](#2-what-the-mlp-could-not-see)
3. [New Features for the LSTM](#3-new-features-for-the-lstm)
4. [Model Architecture](#4-model-architecture)
5. [Training Optimizations](#5-training-optimizations)
6. [Training Results](#6-training-results)
7. [Engagement Prediction: LSTM vs MLP](#7-engagement-prediction-lstm-vs-mlp)
8. [Re-identification Attack: LSTM vs MLP](#8-re-identification-attack-lstm-vs-mlp)
9. [Blind Test Set Evaluation](#9-blind-test-set-evaluation)
10. [Privacy Implications](#10-privacy-implications)
11. [Industry Context and Scope](#11-industry-context-and-scope)
12. [Generated Outputs](#12-generated-outputs)
13. [Key Takeaways](#13-key-takeaways)

---

## 1. Motivation: Why Go Beyond the MLP?

The MLP baseline (Phase 3A) demonstrated a critical finding: a simple engagement predictor trained on 27 aggregate reading statistics creates behavioral fingerprints that re-identify users at **69x above random chance**. But the MLP's feature set is deliberately limited — it sees only summary statistics of user behavior, not the raw temporal patterns.

The question driving Phase 3B is: **Does a more sophisticated model that processes raw behavioral sequences create even more distinctive fingerprints?**

If the answer is yes, it means:
- The privacy risk scales with model quality
- Industry-grade models (which are far more sophisticated than ours) pose an even greater threat
- Privacy-preserving mechanisms like randomized smoothing are not just theoretically motivated — they are practically necessary

The LSTM serves as the "enhanced model" that goes beyond the MLP baseline by processing temporal sequences, adding well-justified features, and producing richer 64-dim representations.

---

## 2. What the MLP Could Not See

The MLP analysis (Section 9 of `docs/03_mlp_baseline_analysis.md`) identified five specific information gaps. The LSTM addresses three of them:

### Gap 1: No Joint Engagement Signal (Addressed)

The MLP had `mean_read_time` and `mean_scroll_pct` as separate features but no access to the joint rate P(RT > 30s AND SP > 50%). Two users with identical means can have wildly different engagement rates depending on how their read time and scroll percentage co-vary. For example:
- User A: mean_RT=35s, mean_SP=60%, but RT and SP are anti-correlated -> engagement rate ~25%
- User B: mean_RT=35s, mean_SP=60%, RT and SP are correlated -> engagement rate ~55%

**LSTM addition**: `hist_engagement_rate`, `hist_deep_scroll_rate`, `hist_long_read_rate` — computed directly from raw history arrays.

### Gap 2: No Article Content Length (Addressed)

Article body length ranges from 0 to 47,355 characters — a 100x range that directly affects both scroll percentage (short articles are trivially scrolled past 50%) and read time (longer articles need more time to exceed 30s). The MLP had no access to article length.

**LSTM addition**: `body_len_log`, `title_len_log`, `subtitle_len_log` — log-transformed content lengths from articles.parquet.

### Gap 3: No Temporal Patterns (Addressed)

The MLP collapses all 50 history articles into static aggregates (mean, std, percentiles). It cannot distinguish a user trending toward deeper engagement from one trending toward disengagement. The temporal ordering of reading behavior carries signal.

**LSTM addition**: The BiLSTM processes the raw 50-timestep behavioral sequence directly, capturing temporal dependencies and reading momentum. Additionally, explicit `rt_momentum` and `sp_momentum` features (ratio of recent to overall behavior) and `rt_sp_correlation` (Pearson correlation between read time and scroll percentage) are added as aggregate features.

### Gaps NOT Addressed (and why)

- **No text embeddings**: Would require multilingual sentence transformers, GPU memory for encoding 125K articles, and would shift focus away from privacy analysis. The RecSys 2024 Challenge top-3 teams used full BERT embeddings to reach 0.7643 AUC — a method beyond our project's scope.
- **No collaborative filtering signals**: Would require user-item interaction matrices and graph-based methods beyond project scope.

---

## 3. New Features for the LSTM

### Feature Dimensions: MLP vs LSTM

| Feature Group | MLP | LSTM | Change |
|--------------|-----|------|--------|
| Aggregate history features | 27 | 27 | Same features |
| History sequences | — | 50 x 2 | Raw temporal input |
| Article embeddings (cat + type) | 32 | 32 | Unchanged |
| Article continuous features | 5 | 5 | Unchanged |
| Context features | 3 | 3 | Unchanged |
| **Total unique feature dims** | **67** | **67 + sequence** | **Substantially richer** |

Both models now use the same 27 aggregate features, 5 article continuous features (premium, sentiment, body/title/subtitle length), and 3 context features. The LSTM's key advantage is the additional raw temporal sequence input (50 timesteps x 2 features) processed by the BiLSTM encoder.

### Aggregate Features (27 total — same as MLP)

The 27 aggregate features include:
- Read time statistics (5): mean, std, median, p10, p90
- Scroll percentage statistics (5): mean, std, median, p10, p90
- Last-5 read times (5): rt_last1 through rt_last5
- Last-5 scroll percentages (5): sp_last1 through sp_last5
- History length (1)
- Behavioral rates (3): engagement rate, deep scroll rate, long read rate
- Momentum features (2): RT momentum, SP momentum
- Correlation (1): RT-SP Pearson correlation

### Article Features (5 continuous + 2 categorical embeddings)

| Feature | Source | Transform |
|---------|--------|-----------|
| `premium` | articles.parquet | Binary (0/1) |
| `sentiment_score` | articles.parquet | Passthrough (default 0.5) |
| `body_len_log` | articles.parquet `body` field length | log1p |
| `title_len_log` | articles.parquet `title` field length | log1p |
| `subtitle_len_log` | articles.parquet `subtitle` field length | log1p |

---

## 4. Model Architecture

### BiLSTM + Multi-Head Self-Attention

```
Input Sequence (B, 50, 2)
    |
Input Projection: Linear(2 -> 64) -> LayerNorm -> SiLU
    |
2-layer Bidirectional LSTM (hidden=128, output=256)
    |
4-Head Self-Attention Pooling (with masking for variable-length sequences)
    |
Sequence Representation (B, 256)
    |
Concatenate with Context Vector (B, 67):
  - Category embedding (16) + Type embedding (16)
  - Article continuous (5) + Context (3) + Aggregate features (27)
    |
Fusion: Linear(323 -> 256) -> LayerNorm -> SiLU -> Dropout(0.2)
    |
Residual Block: Linear(256 -> 256) -> LayerNorm -> SiLU -> Dropout(0.2) + skip
    |
Representation Layer: Linear(256 -> 64) -> LayerNorm   <- 64-dim representation
    |
Classification Head: Linear(64 -> 1)                   <- engagement logit
```

**Total parameters**: 1,025,089 (~1.03M, vs 210K for MLP — ~5x larger)

### Key Design Decisions

- **Masking instead of pack_padded_sequence**: Since ~86% of our sequences are at maximum length (50), we run the LSTM on all timesteps and use attention masking to ignore padding. This simplifies the code and gives a speedup on GPU.
- **Aggregate features in context branch**: The 27 aggregate features provide a "global user profile" alongside the temporal patterns from the LSTM. This is standard practice in industry sequence models — the aggregate context helps the model calibrate the sequential signal.
- **64-dim representation layer**: Same dimensionality as the MLP baseline, ensuring fair comparison for re-identification experiments. Both models project to the same representation space.

---

## 5. Training Optimizations

### Hyperparameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Batch size | 512 | Good GPU utilization on RTX 5070 Ti |
| Learning rate | 8e-4 | Scaling law: sqrt(512/256) x 5e-4 |
| Weight decay | 3e-4 | More parameters need more regularization |
| LR warmup (pct_start) | 0.3 | Reach peak LR by epoch ~15 with 50 max |
| Max epochs | 50 | Sufficient budget with patience 8 |
| Patience | 8 | Allow adequate exploration before stopping |
| LSTM dropout | 0.15 | Lighter regularization for 2-layer BiLSTM |
| DataLoader workers | 0 | Safe default on Windows |
| Sequence handling | Direct + masking | GPU-friendly; no pack_padded_sequence |
| Loss | LabelSmoothingBCE | pos_weight=1.49, smoothing=0.05 |
| Scheduler | OneCycleLR | pct_start=0.3 for cosine annealing |

### Execution Environment

| Component | Value |
|-----------|-------|
| Device | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB VRAM, CUDA 12.8) |
| CPU | Intel Core Ultra 9 275HX (24 cores) |
| RAM | 64 GB |
| OS | Windows 11 Home (Build 26200) |
| PyTorch | 2.10.0+cu128 |
| Python | 3.11.14 |

---

## 6. Training Results

### Convergence Trajectory

The LSTM trained for 26 epochs before early stopping (patience=8). Best model was saved at epoch 18.

| Epoch | Train Loss | Val Loss | Val AUC | Val F1 | Status |
|-------|-----------|----------|---------|--------|--------|
| 1 | 0.7887 | 0.7833 | 0.6844 | 0.5948 | * best |
| 2 | 0.7795 | 0.7805 | 0.6894 | 0.6019 | * best |
| 3 | 0.7762 | 0.7789 | 0.6908 | 0.6007 | * best |
| 4 | 0.7752 | 0.7801 | 0.6911 | 0.5752 | * best |
| 8 | 0.7727 | 0.7790 | 0.6918 | 0.6028 | * best |
| 10 | 0.7712 | 0.7779 | 0.6931 | 0.5946 | * best |
| 13 | 0.7698 | 0.7776 | 0.6944 | 0.5941 | * best |
| 15 | 0.7681 | 0.7765 | 0.6950 | 0.5992 | * best |
| 16 | 0.7671 | 0.7759 | 0.6957 | 0.5956 | * best |
| 17 | 0.7664 | 0.7772 | 0.6960 | 0.5860 | * best |
| **18** | **0.7656** | **0.7753** | **0.6975** | **0.5888** | **best (final)** |
| 26 | 0.7573 | 0.7798 | 0.6921 | 0.6003 | early stop |

Notable observations:
- The LSTM **surpassed the MLP's 0.6951 AUC at epoch 16** (with consistent improvement from epoch 1)
- Steady improvement through the OneCycleLR warmup phase (epochs 1-15)
- Peak performance at epoch 18 during the LR annealing phase
- Clear overfitting signal in later epochs (train loss continued to decrease while val loss increased)
- Training time: ~135-153s per epoch (vs ~62s for MLP), reflecting the sequential LSTM computation

### Final Model Performance (Epoch 18)

| Metric | Train | Validation | Gap |
|--------|-------|------------|-----|
| AUC-ROC | 0.7129 | 0.6975 | +0.0155 |
| F1 Score | 0.5955 | 0.5888 | +0.0067 |
| Accuracy | 0.6576 | 0.6459 | +0.0116 |
| Precision | 0.5664 | 0.5608 | +0.0056 |
| Recall | 0.6277 | 0.6198 | +0.0079 |
| Avg Precision | 0.6263 | 0.6112 | +0.0150 |

The train-val AUC gap of 0.0155 is moderate and well within acceptable range for a model with 1.03M parameters. This is comparable to the MLP's 0.016 gap, adjusted for the 5x parameter count increase.

### Representation Quality

| Metric | LSTM | MLP |
|--------|------|-----|
| Repr dim | 64 | 64 |
| Mean | 0.0061 | 0.0070 |
| Std | **0.8126** | 0.4290 |
| Min | -3.4860 | -4.3100 |
| Max | 3.4320 | 2.8500 |
| Dead dims (std < 0.01) | **0** | 0 |
| Low-var dims (std < 0.05) | 1 | 0 |

**Key observation**: The LSTM's representation spread (std 0.81) is **nearly 2x wider** than the MLP's (std 0.43). This means the LSTM creates more diverse and distinctive representations across the 64-dimensional space — a direct predictor of stronger re-identification, since more spread = more information per dimension for distinguishing users.

### Training Time

| Phase | Duration |
|-------|----------|
| Data loading | ~5s |
| Training (26 epochs) | 62.2 min |
| Representation extraction | ~30s |
| Plot generation | ~45s |
| **Total** | **63.6 min** |

Epoch time: ~135-153s (vs ~62s for MLP), reflecting the BiLSTM sequential computation over 50 timesteps.

---

## 7. Engagement Prediction: LSTM vs MLP

### 50K Validation Set

| Metric | LSTM | MLP | Delta |
|--------|------|-----|-------|
| Val AUC | **0.6975** | 0.6951 | **+0.0024** |
| Val F1 | 0.5888 | **0.5946** | -0.0058 |
| Val Accuracy | **0.6459** | 0.6378 | **+0.0081** |
| Val Precision | **0.5608** | 0.5483 | **+0.0125** |
| Val Recall | 0.6198 | **0.6495** | -0.0297 |
| Val Avg Precision | **0.6112** | 0.6102 | +0.0010 |
| Parameters | 1,025,089 | 209,665 | 4.9x larger |
| Training time | 62.2 min | 25.7 min | 2.4x slower |

### Interpretation

The LSTM achieves a **+0.24 percentage point improvement in AUC** over the MLP. The performance difference is distributed across different operating points:

- **AUC, Accuracy, and Precision improve** — the LSTM makes more accurate predictions overall and produces fewer false positives.
- **Recall trades off slightly** (-2.97 points) — the MLP was more aggressive in predicting engagement, catching slightly more true positives but at the cost of more false positives (lower precision).
- **F1 trades off slightly** (-0.58 points) — because F1 weights precision and recall equally, the LSTM's precision gain doesn't fully offset the recall loss.

The AUC is the most robust metric because it is threshold-independent and measures ranking quality. The LSTM's **0.6975** confirms it has learned additional signal from the temporal sequences that the MLP's static aggregates could not capture. However, the gain is modest, suggesting that aggregate statistics already capture most of the predictive signal for this task.

---

## 8. Re-identification Attack: LSTM vs MLP

*Re-identification methodology: identical to Phase 3A — gallery/probe split (50/50, users with 4+ impressions, seed=42), nearest-neighbor attack with both euclidean and cosine distance.*

### LSTM Re-identification Results

*28,361 gallery users, 280,423 probe impressions. Same gallery/probe split methodology as Phase 3A.*

| Metric | LSTM Cosine | LSTM Euclidean | MLP Cosine | MLP Euclidean | Random |
|--------|------------|----------------|------------|---------------|--------|
| Top-1 Accuracy | **10.43%** | 9.84% | 0.24% | 0.18% | 0.0035% |
| Top-5 Accuracy | **16.03%** | 15.19% | 0.61% | 0.50% | 0.018% |
| Top-10 Accuracy | **18.77%** | 17.86% | 0.95% | 0.81% | 0.035% |
| Top-20 Accuracy | **21.82%** | 20.80% | 1.46% | 1.28% | 0.071% |
| MRR | **0.1334** | 0.1264 | 0.0058 | 0.0049 | 0.0004 |
| Mean Rank | 3,741 | 3,795 | 4,723 | 4,633 | ~14,180 |
| Median Rank | **1,126** | 1,233 | 3,121 | 3,096 | ~14,180 |
| Lift over Random | **2,958x** | 2,791x | **69x** | 50x | 1x |
| Users 100% identifiable | 220 | 223 | 2 | 2 | 0 |
| Users 0% identifiable | 17,738 | 18,218 | 27,868 | 28,003 | ~28,361 |
| Per-user accuracy (mean) | 10.89% | 10.68% | 0.31% | 0.27% | 0.0035% |

### LSTM vs MLP: The Amplification Effect

The comparison reveals a dramatic privacy amplification:

| Comparison Metric | LSTM (cosine) | MLP (cosine) | Ratio |
|-------------------|-------------|------------|-------|
| Top-1 Accuracy | 10.43% | 0.24% | **43.1x** |
| Top-5 Accuracy | 16.03% | 0.61% | **26.3x** |
| Top-10 Accuracy | 18.77% | 0.95% | **19.8x** |
| Top-20 Accuracy | 21.82% | 1.46% | **14.9x** |
| MRR | 0.1334 | 0.0058 | **23.0x** |
| Lift over Random | 2,958x | 69x | **42.9x** |
| Users 100% re-identifiable | 220 | 2 | **110x** |
| Median Rank | 1,126 | 3,121 | **2.8x lower** |

**The key finding**: A **+0.24 percentage point** improvement in engagement AUC (0.6951 -> 0.6975) produced a **43.1x increase in re-identification accuracy**. The relationship between model quality and privacy risk is dramatically non-linear.

### Why the LSTM is So Much More Identifiable

The LSTM's 64-dim representations encode three types of signal that the MLP cannot capture:

1. **Temporal reading patterns**: The BiLSTM processes the raw sequence of 50 reading behaviors, learning patterns like "reads deeply at start of session, then skims" or "alternates between long and short reads." These temporal signatures are highly individual.

2. **Wider representation spread**: The LSTM's representation std (0.81) is nearly 2x the MLP's (0.43). More spread means more information per dimension, creating more distinctive fingerprints in the same 64-dim space.

3. **Cross-feature interactions**: The attention mechanism learns which timesteps matter most for each user. Users with distinctive attention patterns (e.g., always reading the first article deeply) create unique representation vectors.

The result: while the MLP creates somewhat distinctive fingerprints from aggregate statistics alone, the LSTM creates **dramatically** more distinctive fingerprints from temporal patterns — even though both models project to the same 64-dimensional representation space.

---

## 9. Blind Test Set Evaluation

The blind test set (10,000 users, zero overlap with the 50K training/validation data) provides the strongest validation of our findings.

### 9.1 Engagement Generalization

| Metric | LSTM (Blind Val) | LSTM (50K Val) | Delta | MLP (Blind Val) |
|--------|-----------------|---------------|-------|-----------------|
| AUC | **0.7058** | 0.6975 | **+0.0083** | 0.7030 |
| F1 | 0.5995 | 0.5888 | +0.0107 | 0.6043 |
| Accuracy | 0.6495 | 0.6459 | +0.0036 | 0.6419 |
| Precision | 0.5677 | 0.5608 | +0.0069 | 0.5559 |
| Recall | 0.6352 | 0.6198 | +0.0154 | 0.6621 |

Both models **generalize better** on blind test users (AUC improvements of +0.0083 for LSTM and +0.008 for MLP). This is expected: the blind test users are fully independent, eliminating any train/val user overlap effects. The LSTM maintains its AUC advantage over MLP on the blind set (+0.0028).

### 9.2 Within-Blind Re-identification

This answers: **Can the LSTM fingerprint users it has NEVER seen during training?**

| Metric | LSTM (Cosine) | MLP (Cosine) | Ratio | Random |
|--------|--------------|-------------|-------|--------|
| Top-1 | **9.13%** | 0.26% | **34.7x** | 0.011% |
| Top-5 | **15.77%** | 1.02% | **15.5x** | 0.054% |
| Lift | **841x** | 24x | **35.0x** | 1x |
| MRR | **0.1264** | 0.0095 | **13.3x** | 0.001 |
| Median Rank | **465** / 9,215 | 992 / 9,215 | **2.1x lower** | ~4,608 |

**Critical finding**: The LSTM achieves 841x above random on users **it never trained on**. This proves:
- Fingerprinting is **feature-level**: the raw temporal sequences are inherently identifying
- The BiLSTM's temporal encoder creates distinctive representations purely from the behavioral patterns in the input data
- This is NOT model memorization — the model has never seen these users' data

The MLP achieves 24x on the same blind users, confirming that even aggregate features carry identity signal. But the LSTM's temporal processing amplifies this by **35x**.

### 9.3 Cross-Dataset Re-identification (Negative Control)

Gallery: 39,657 users from 50K val set. Probes: 311,182 blind test impressions.

| Metric | LSTM | MLP |
|--------|------|-----|
| Top-1 | **0.0000%** | **0.0000%** |
| Mean Rank | 39,658 / 39,657 | 39,658 / 39,657 |
| Median Rank | 39,658 / 39,657 | 39,658 / 39,657 |

**Perfect negative control for both models**: when blind test users (who have no gallery entries) are probed against the 50K gallery, re-identification is exactly 0%. Every probe ranks at maximum because the correct user doesn't exist in the gallery. This confirms:
- The attack methodology is sound
- The within-blind results (841x/24x) are genuine fingerprinting, not an artifact

---

## 10. Privacy Implications

### The Non-Linear Privacy Risk

The central finding of Phase 3B is the **dramatic non-linearity** between model quality and privacy risk:

| Metric | MLP | LSTM | Ratio |
|--------|-----|------|-------|
| AUC improvement | baseline | +0.24% | marginal |
| Re-id improvement | baseline | +43.1x | massive |
| Blind re-id improvement | baseline | +35.0x | massive |

A barely perceptible improvement in prediction accuracy (+0.24% AUC) produces an enormous amplification of re-identification risk. This has profound implications:

1. **Industry models are far riskier**: Production recommendation systems achieve 0.76+ AUC — a much larger gap from our 0.70. If a +0.24% AUC gain produces 43x more identifiable representations, the privacy risk from production-grade models could be orders of magnitude worse.

2. **Temporal sequences are the key privacy risk**: The LSTM's ability to process raw reading sequences (rather than just summary statistics) is what drives the 43x amplification. Any model that processes behavioral sequences — LSTM, Transformer, GRU — would likely show similar amplification.

3. **Feature-level fingerprinting is confirmed**: The blind test (841x lift on unseen users) proves that the fingerprinting comes from the temporal features themselves, not from the model memorizing training users. This means privacy defenses must address the representation level, not just prevent memorization.

### Why This Matters

If a moderate proof-of-concept model with 1M parameters and basic behavioral features already creates identifiable fingerprints at 2,958x above random, then industry-grade models pose a far greater risk:

- **Production recommendation systems** use transformer-based content encoders, large-scale collaborative filtering, real-time session features, and ensemble stacks with hundreds of features
- **They achieve 0.76+ AUC** (RecSys 2024 Challenge top-3 averaged 0.7643)
- **Their representations are correspondingly richer** and more user-distinctive
- **Privacy-preserving mechanisms are therefore not optional** — they are a necessary design requirement

This directly motivates Phase 4 (Randomized Smoothing), which adds calibrated Gaussian noise to representations to destroy the fingerprint signal while preserving predictive accuracy.

---

## 11. Industry Context and Scope

### Why the LSTM Didn't Reach 0.71-0.74 AUC

Our initial target was 0.71-0.74 AUC based on the feature additions. The actual improvement was more modest (+0.0024 to 0.6975). This is informative:

1. **The feature additions worked** — the LSTM surpassed the MLP, confirming the temporal signal is real.
2. **The temporal signal is incremental** — on this dataset, aggregate statistics already capture most of the behaviorally predictable variance. The temporal ordering adds real value but is not transformative for prediction.
3. **The engagement label has inherent noise** — whether a user scrolls past 50% and reads for 30+ seconds depends heavily on article content, user mood, and context that no model can observe.
4. **The privacy signal is what's transformative** — while the AUC gain is modest, the re-identification amplification is massive. The temporal features add far more identity information than prediction information.

### RecSys 2024 Challenge Benchmarks

| Method | AUC | Features | Scale |
|--------|-----|----------|-------|
| Top-1 (RecSys 2024) | 0.7708 | Full text embeddings + collaborative filtering + session features | 1M+ users |
| Top-2 (RecSys 2024) | 0.7639 | Multilingual BERT + user graphs | 1M+ users |
| Top-3 (RecSys 2024) | 0.7582 | Text + behavioral + ensemble | 1M+ users |
| MIND NRMS (reference) | 0.6776 | News article text embeddings | 50K users |
| **Our LSTM** | **0.6975** | Behavioral sequences + aggregate features | 50K users |
| **Our MLP** | **0.6951** | Aggregate behavioral features only | 50K users |

Our LSTM at 0.6975 surpasses MIND NRMS (which uses text embeddings we don't have) and is well within the expected range for a behavioral-features-only model. The gap to the top RecSys solutions reflects the missing text embeddings, collaborative filtering, and engineering resources — not a deficiency in our approach.

### What This Means for Privacy

**Our project's contribution is not prediction accuracy — it is the privacy analysis.** We demonstrate:
- Even a moderate model (0.70 AUC) creates identifiable fingerprints at 2,958x above random
- Richer models (LSTM > MLP) produce dramatically more distinctive representations
- A tiny +0.24% AUC gain amplifies re-identification by 43.1x
- Industry models (0.76+ AUC) would make the privacy problem far worse
- This motivates formal privacy mechanisms (Phase 4: Randomized Smoothing)

---

## 12. Generated Outputs

All outputs are in `outputs/models/lstm/`:

| File | Description |
|------|-------------|
| `checkpoint.pt` | Best model weights (epoch 18) |
| `metrics.json` | Full per-epoch training history and configuration |
| `representations.npz` | 64-dim representations for all train + val samples |
| `training_curves.png` | Loss, AUC, metrics over 26 epochs |
| `evaluation_plots.png` | ROC curve, PR curve, confusion matrix, probability distribution |
| `representation_analysis.png` | t-SNE, norm distribution, per-dimension activation analysis |
| `reidentification_euclidean.png` | LSTM re-id attack results (euclidean distance) |
| `reidentification_cosine.png` | LSTM re-id attack results (cosine distance) |
| `reidentification_comparison.png` | Euclidean vs cosine vs random bar chart |
| `lstm_vs_mlp_comparison.png` | **Key plot**: LSTM vs MLP on engagement + re-identification |
| `reidentification_results.json` | All numerical re-identification results (LSTM + MLP) |
| `blind_test_results.json` | Blind test engagement + re-identification results |

---

## 13. Key Takeaways

1. **The LSTM surpasses the MLP baseline** (0.6975 vs 0.6951 AUC), confirming that temporal behavioral sequences add predictive value beyond static aggregates.

2. **Re-identification is 43.1x stronger with LSTM** (10.43% Top-1 vs 0.24%) — temporal sequences create dramatically more distinctive fingerprints than aggregate statistics.

3. **The AUC-to-privacy relationship is non-linear**: a +0.24% AUC improvement produces a 43.1x increase in re-identification risk. This means even small model improvements can massively worsen privacy.

4. **Blind test confirms feature-level fingerprinting** — LSTM achieves 841x above random on 10K users it never trained on. The temporal patterns in reading behavior are inherently identifying.

5. **The LSTM representation spread is 2x wider** (std 0.81 vs 0.43), explaining why the same 64-dim space carries far more identity information.

6. **Cross-dataset negative control is perfect** (0.0000% for both models) — the attack methodology is sound and re-identification cannot occur when the target user is absent from the gallery.

7. **Both models generalize better on blind test** (AUC +0.008 for both) — the blind users are fully independent, confirming the models are not overfitting to training users.

8. **Industry-scale models pose orders-of-magnitude greater risk** — our proof-of-concept LSTM with 1M params and 0.70 AUC already achieves 2,958x re-identification lift. Production systems with transformer encoders and 0.76+ AUC would amplify this dramatically.

9. **The project's contribution is the privacy analysis, not prediction quality.** Both models serve as vehicles for demonstrating that engagement prediction inherently creates re-identification risk — a risk that scales with model sophistication and especially with access to temporal behavioral sequences.

10. **This motivates Phase 4: Randomized Smoothing** — the LSTM's stronger fingerprints will require more noise to neutralize, quantifying the privacy cost of model sophistication.
