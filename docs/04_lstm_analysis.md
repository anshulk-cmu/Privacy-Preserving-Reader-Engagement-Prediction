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
9. [Privacy Implications](#9-privacy-implications)
10. [Industry Context and Scope](#10-industry-context-and-scope)
11. [Generated Outputs](#11-generated-outputs)
12. [Key Takeaways](#12-key-takeaways)

---

## 1. Motivation: Why Go Beyond the MLP?

The MLP baseline (Phase 3A) demonstrated a critical finding: a simple engagement predictor trained on aggregate reading statistics creates behavioral fingerprints that re-identify users at **314x above random chance**. But the MLP's feature set is deliberately limited — it sees only summary statistics of user behavior, not the raw temporal patterns.

The question driving Phase 3B is: **Does a more sophisticated model that processes raw behavioral sequences create even more distinctive fingerprints?**

If the answer is yes, it means:
- The privacy risk scales with model quality
- Industry-grade models (which are far more sophisticated than ours) pose an even greater threat
- Privacy-preserving mechanisms like randomized smoothing are not just theoretically motivated — they are practically necessary

The LSTM serves as the "enhanced model" that goes beyond the MLP baseline by processing temporal sequences, adding well-justified features, and producing richer 64-dim representations. The MLP is intentionally frozen at 0.6817 AUC as a fixed reference point for comparison.

---

## 2. What the MLP Could Not See

The MLP analysis (Section 9 of `docs/03_mlp_baseline_analysis.md`) identified five specific information gaps. The LSTM addresses three of them:

### Gap 1: No Joint Engagement Signal (Addressed)

The MLP had `mean_read_time` and `mean_scroll_pct` as separate features but no access to the joint rate P(RT > 30s AND SP > 50%). Two users with identical means can have wildly different engagement rates depending on how their read time and scroll percentage co-vary. For example:
- User A: mean_RT=35s, mean_SP=60%, but RT and SP are anti-correlated → engagement rate ~25%
- User B: mean_RT=35s, mean_SP=60%, RT and SP are correlated → engagement rate ~55%

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
| Aggregate history features | 21 | 27 | +6 new features |
| History sequences | — | 50 × 2 | Raw temporal input |
| Article embeddings (cat + type) | 32 | 32 | Unchanged |
| Article continuous features | 2 | 5 | +3 content lengths |
| Context features | 3 | 3 | Unchanged |
| **Total unique feature dims** | **58** | **67 + sequence** | **Substantially richer** |

### New Aggregate Features (6 additions, indices 21-26)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `hist_engagement_rate` | count(RT>30 AND SP>50) / history_length | Direct base rate — the strongest single predictor of future engagement |
| `hist_deep_scroll_rate` | count(SP>80) / history_length | Deep reading propensity |
| `hist_long_read_rate` | count(RT>60) / history_length | Sustained attention propensity |
| `rt_momentum` | mean(last-5 RT) / mean(all RT), clipped [0.1, 10] | Whether the user is reading more/less than usual |
| `sp_momentum` | mean(last-5 SP) / mean(all SP), clipped [0.1, 10] | Whether the user is scrolling more/less than usual |
| `rt_sp_correlation` | Pearson(RT sequence, SP sequence) | Reading consistency — do RT and SP co-vary? |

### New Article Features (3 additions)

| Feature | Source | Transform |
|---------|--------|-----------|
| `body_len_log` | articles.parquet `body` field length | log1p → StandardScaler |
| `title_len_log` | articles.parquet `title` field length | log1p → StandardScaler |
| `subtitle_len_log` | articles.parquet `subtitle` field length | log1p → StandardScaler |

---

## 4. Model Architecture

### BiLSTM + Multi-Head Self-Attention

```
Input Sequence (B, 50, 2)
    ↓
Input Projection: Linear(2 → 64) → LayerNorm → SiLU
    ↓
2-layer Bidirectional LSTM (hidden=128, output=256)
    ↓
4-Head Self-Attention Pooling (with masking for variable-length sequences)
    ↓
Sequence Representation (B, 256)
    ↓
Concatenate with Context Vector (B, 67):
  - Category embedding (16) + Type embedding (16)
  - Article continuous (5) + Context (3) + Aggregate features (27)
    ↓
Fusion: Linear(323 → 256) → LayerNorm → SiLU → Dropout(0.2)
    ↓
Residual Block: Linear(256 → 256) → LayerNorm → SiLU → Dropout(0.2) + skip
    ↓
Representation Layer: Linear(256 → 64) → LayerNorm   ← 64-dim representation
    ↓
Classification Head: Linear(64 → 1)                   ← engagement logit
```

**Total parameters**: 1,025,089 (~1.0M, vs 207K for MLP — 5x larger)

### Key Design Decisions

- **Masking instead of pack_padded_sequence**: The standard PyTorch `pack_padded_sequence` has severe performance issues on Apple MPS (Metal Performance Shaders). Since 88.5% of our sequences are at maximum length (50), we run the LSTM on all timesteps and use attention masking to ignore padding. This gave a ~2.5x speedup per epoch.
- **Aggregate features in context branch**: The 27 aggregate features provide a "global user profile" alongside the temporal patterns from the LSTM. This is standard practice in industry sequence models — the aggregate context helps the model calibrate the sequential signal.
- **64-dim representation layer**: Same dimensionality as the MLP baseline, ensuring fair comparison for re-identification experiments. Both models project to the same representation space.

---

## 5. Training Optimizations

### Hyperparameter Changes from Initial LSTM (v1 → v2)

| Parameter | v1 (broken) | v2 (optimized) | Justification |
|-----------|------------|----------------|---------------|
| Batch size | 256 | 512 | Fewer iterations per epoch; better GPU utilization |
| Learning rate | 5e-4 | 8e-4 | Scaling law: sqrt(512/256) × 5e-4 ≈ 7.1e-4, rounded up |
| Weight decay | 1e-4 | 3e-4 | More parameters need more regularization |
| LR warmup (pct_start) | 0.1 | 0.3 | Reach peak LR faster; match batch size increase |
| Max epochs | 60 | 30 | Tighter schedule with faster convergence |
| Patience | 10 | 7 | Faster early stopping |
| LSTM dropout | 0.3 | 0.15 | Less aggressive for 2-layer BiLSTM (over-regularized at 0.3) |
| DataLoader workers | 0 | 2 | Parallel data loading (4 workers had excessive IPC overhead on macOS) |
| Sequence handling | pack_padded_sequence | Direct + masking | MPS compatibility; 2.5x speedup |

### Training Speed

| Metric | v1 | v2 | Speedup |
|--------|----|----|---------|
| Epoch time (with train eval) | ~12 min | ~3.2 min | **3.8x** |
| Total training time | >12 hours (never converged) | ~72 min (25 epochs) | — |
| Convergence | Not reached | Epoch 18 | — |

The 3.8x speedup came from three sources:
1. Masking instead of `pack_padded_sequence` (~1.5-2x)
2. Batch size 256 → 512 (~1.5x fewer iterations)
3. DataLoader workers 0 → 2 (~1.1x)

---

## 6. Training Results

### Convergence Trajectory

The LSTM trained for 25 epochs before early stopping (patience=7). Best model was saved at epoch 18.

| Epoch | Train Loss | Val Loss | Val AUC | Val F1 | Status |
|-------|-----------|----------|---------|--------|--------|
| 1 | 0.7906 | 0.7863 | 0.6770 | 0.5864 | * best |
| 3 | 0.7814 | 0.7858 | 0.6817 | 0.5632 | * best (matched MLP) |
| 9 | 0.7764 | 0.7821 | 0.6849 | 0.5961 | * best |
| 12 | 0.7744 | 0.7834 | 0.6857 | 0.5893 | * best |
| 14 | 0.7723 | 0.7839 | 0.6869 | 0.5851 | * best |
| **18** | **0.7681** | **0.7809** | **0.6869** | **0.5947** | **★ best (final)** |
| 25 | 0.7550 | 0.7893 | 0.6794 | 0.5899 | early stop |

Notable observations:
- The LSTM **matched the MLP's 0.6817 AUC by epoch 3** (the MLP needed 13 epochs with 50 epochs max)
- Steady improvement through the OneCycleLR warmup phase (epochs 1-9)
- Peak performance at epoch 18 during the LR annealing phase
- Clear overfitting signal in later epochs (train loss continued to decrease while val loss increased)

### Final Model Performance (Epoch 18)

| Metric | Train | Validation | Gap |
|--------|-------|------------|-----|
| AUC-ROC | 0.7090 | 0.6869 | +0.022 |
| F1 Score | 0.6039 | 0.5947 | +0.009 |
| Accuracy | 0.6435 | 0.6268 | +0.017 |
| Precision | 0.5452 | 0.5350 | +0.010 |
| Recall | 0.6768 | 0.6694 | +0.007 |
| Avg Precision | 0.6219 | 0.5987 | +0.023 |

The train-val gap of 0.022 AUC is moderate (slightly larger than MLP's 0.010 gap), indicating the LSTM's 1M parameters are learning some training-set-specific patterns, but the gap is well within acceptable range for a model of this capacity.

---

## 7. Engagement Prediction: LSTM vs MLP

| Metric | LSTM (v2) | MLP (v3) | Improvement |
|--------|-----------|----------|-------------|
| Val AUC | **0.6869** | 0.6817 | **+0.0052** |
| Val F1 | **0.5947** | 0.5694 | **+0.0253** |
| Val Accuracy | 0.6268 | **0.6384** | -0.0116 |
| Val Precision | 0.5350 | **0.5551** | -0.0201 |
| Val Recall | **0.6694** | 0.5844 | **+0.0850** |
| Val Avg Precision | **0.5987** | 0.5962 | +0.0025 |
| Parameters | 1,025,089 | 207,361 | 4.9x larger |

### Interpretation

The LSTM achieves a meaningful **+0.52 percentage point improvement in AUC** over the MLP. While this may seem modest, the performance difference is concentrated in the recall direction:

- **Recall improvement of +8.5 percentage points** — the LSTM is substantially better at identifying true engagement events. This makes sense: the temporal sequence captures reading patterns (e.g., a user in a "deep reading session") that static aggregates miss.
- **Precision trades off slightly** — the LSTM predicts engagement more liberally, catching more true positives at the cost of some false positives.
- **F1 improves by +2.5 points** — the recall gain outweighs the precision loss.
- **Accuracy drops slightly** because the threshold-dependent metrics are sensitive to the probability calibration, which differs between models.

The AUC is the most robust metric because it is threshold-independent and measures ranking quality. The LSTM's 0.6869 confirms it has learned additional signal from the temporal sequences and enriched features that the MLP's static aggregates could not capture.

---

## 8. Re-identification Attack: LSTM vs MLP

*Re-identification methodology: identical to Phase 3A — gallery/probe split (50/50, users with 4+ impressions, seed=42), nearest-neighbor attack with both euclidean and cosine distance.*

### LSTM Re-identification Results

*28,361 gallery users, 280,423 probe impressions. Same gallery/probe split methodology as Phase 3A.*

| Metric | LSTM Cosine | LSTM Euclidean | MLP Cosine | MLP Euclidean | Random |
|--------|------------|----------------|------------|---------------|--------|
| Top-1 Accuracy | **10.06%** | 9.67% | 1.11% | 0.93% | 0.0035% |
| Top-5 Accuracy | **15.30%** | 14.69% | 1.94% | 1.58% | 0.018% |
| Top-10 Accuracy | **17.86%** | 17.22% | 2.45% | 2.03% | 0.035% |
| Top-20 Accuracy | **20.73%** | 20.05% | 3.18% | 2.67% | 0.071% |
| MRR | **0.1280** | 0.1231 | 0.0168 | 0.0142 | 0.0004 |
| Mean Rank | 3,632 | 3,652 | 3,893 | 3,891 | ~14,180 |
| Median Rank | **1,216** | 1,290 | 2,266 | 2,242 | ~14,180 |
| Lift over Random | **2,853x** | 2,742x | 314x | 264x | 1x |
| Users 100% identifiable | 231 | 225 | 34 | 37 | 0 |
| Users 0% identifiable | 17,440 | 17,838 | 26,350 | 26,679 | ~28,361 |
| Per-user accuracy (mean) | 11.25% | 11.09% | 1.76% | 1.65% | 0.0035% |

### LSTM vs MLP: The Amplification Effect

The comparison reveals a dramatic privacy amplification:

| Comparison Metric | LSTM (best) | MLP (best) | Ratio |
|-------------------|-------------|------------|-------|
| Top-1 Accuracy | 10.06% | 1.11% | **9.1x** |
| Top-5 Accuracy | 15.30% | 1.94% | **7.9x** |
| Top-10 Accuracy | 17.86% | 2.45% | **7.3x** |
| Top-20 Accuracy | 20.73% | 3.18% | **6.5x** |
| MRR | 0.1280 | 0.0168 | **7.6x** |
| Lift over Random | 2,853x | 314x | **9.1x** |
| Users 100% re-identifiable | 231 | 34 | **6.8x** |
| Median Rank | 1,216 | 2,266 | **1.9x lower** |

**The key finding**: A +0.52 percentage point improvement in engagement AUC (0.6817 → 0.6869) produced a **9.1x increase in re-identification lift** (314x → 2,853x). The relationship between model quality and privacy risk is dramatically non-linear.

### Why the LSTM is So Much More Identifiable

The LSTM's 64-dim representations encode three types of signal that the MLP cannot capture:

1. **Temporal reading patterns**: The BiLSTM processes the raw sequence of 50 reading behaviors, learning patterns like "reads deeply at start of session, then skims" or "alternates between long and short reads." These temporal signatures are highly individual.

2. **Behavioral momentum**: The explicit momentum features (recent behavior vs. overall average) capture whether a user is in a "deep reading" or "skimming" phase. Combined with the LSTM's sequential encoding, this creates a richer behavioral fingerprint.

3. **Cross-feature interactions**: The attention mechanism learns which timesteps matter most for each user. Users with distinctive attention patterns (e.g., always reading the first article deeply) create unique representation vectors.

The result: while the MLP creates somewhat distinctive fingerprints from aggregate statistics alone, the LSTM creates dramatically more distinctive fingerprints from temporal patterns — even though both models project to the same 64-dimensional representation space.

---

## 9. Privacy Implications

The LSTM's richer representations carry a dual consequence:

1. **Better engagement prediction** — the temporal patterns and enriched features push AUC from 0.6817 to 0.6869, with substantially better recall (+8.5 points).

2. **More distinctive fingerprints** — the 64-dim representations encode temporal behavioral patterns (reading sessions, engagement momentum, content-length preferences) in addition to the aggregate statistics the MLP learned. These temporal signatures are inherently more unique because two users might share similar average reading times but read in very different temporal patterns.

### Why This Matters

If a moderate proof-of-concept model with 1M parameters and basic behavioral features already creates identifiable fingerprints, then industry-grade models pose a far greater risk:

- **Production recommendation systems** use transformer-based content encoders, large-scale collaborative filtering, real-time session features, and ensemble stacks with hundreds of features
- **They achieve 0.76+ AUC** (RecSys 2024 Challenge top-3 averaged 0.7643)
- **Their representations are correspondingly richer** and more user-distinctive
- **Privacy-preserving mechanisms are therefore not optional** — they are a necessary design requirement

This directly motivates Phase 4 (Randomized Smoothing), which adds calibrated Gaussian noise to representations to destroy the fingerprint signal while preserving predictive accuracy.

---

## 10. Industry Context and Scope

### Why the LSTM Didn't Reach 0.71-0.74 AUC

Our initial target was 0.71-0.74 AUC based on the feature additions. The actual improvement was more modest (+0.0052 to 0.6869). This is informative:

1. **The feature additions worked** — the LSTM surpassed the MLP, confirming the information gaps were real.
2. **The temporal signal is weaker than expected** — on this dataset, aggregate statistics already capture most of the behaviorally predictable variance. The temporal ordering adds incremental value but is not transformative.
3. **The engagement label has inherent noise** — whether a user scrolls past 50% and reads for 30+ seconds depends heavily on article content, user mood, and context that no model can observe.

### RecSys 2024 Challenge Benchmarks

| Method | AUC | Features | Scale |
|--------|-----|----------|-------|
| Top-1 (RecSys 2024) | 0.7708 | Full text embeddings + collaborative filtering + session features | 1M+ users |
| Top-2 (RecSys 2024) | 0.7639 | Multilingual BERT + user graphs | 1M+ users |
| Top-3 (RecSys 2024) | 0.7582 | Text + behavioral + ensemble | 1M+ users |
| MIND NRMS (reference) | 0.6776 | News article text embeddings | 50K users |
| **Our LSTM** | **0.6869** | Behavioral sequences + aggregate features | 50K users |
| **Our MLP** | **0.6817** | Aggregate behavioral features only | 50K users |

Our LSTM at 0.6869 is competitive with MIND NRMS (which uses text embeddings we don't have) and well within the expected range for a behavioral-features-only model. The gap to the top RecSys solutions reflects the missing text embeddings, collaborative filtering, and engineering resources — not a deficiency in our approach.

### What This Means for Privacy

**Our project's contribution is not prediction accuracy — it is the privacy analysis.** We demonstrate:
- Even a moderate model (0.69 AUC) creates identifiable fingerprints
- Richer models (LSTM > MLP) produce more distinctive representations
- Industry models (0.76+ AUC) would make the privacy problem far worse
- This motivates formal privacy mechanisms (Phase 4: Randomized Smoothing)

---

## 11. Generated Outputs

All outputs are in `outputs/models/lstm/`:

| File | Description |
|------|-------------|
| `checkpoint.pt` | Best model weights (epoch 18, 12.3 MB) |
| `metrics.json` | Full per-epoch training history and configuration |
| `representations.npz` | 64-dim representations for all train + val samples |
| `training_curves.png` | Loss, AUC, metrics over 25 epochs |
| `evaluation_plots.png` | ROC curve, PR curve, confusion matrix, probability distribution |
| `representation_analysis.png` | t-SNE, norm distribution, per-dimension activation analysis |
| `reidentification_euclidean.png` | LSTM re-id attack results (euclidean distance) |
| `reidentification_cosine.png` | LSTM re-id attack results (cosine distance) |
| `reidentification_comparison.png` | Euclidean vs cosine vs random bar chart |
| `lstm_vs_mlp_comparison.png` | **Key plot**: LSTM vs MLP on engagement + re-identification |
| `reidentification_results.json` | All numerical re-identification results |

---

## 12. Key Takeaways

1. **The LSTM surpasses the MLP baseline** (0.6869 vs 0.6817 AUC), confirming that temporal behavioral sequences and enriched features add predictive value beyond static aggregates.

2. **Matched MLP in 3 epochs** — the LSTM reached the MLP's best AUC by epoch 3 (vs the MLP's 13 epochs), demonstrating that the architecture is well-suited to the data structure.

3. **Training optimizations delivered 3.8x speedup** — replacing `pack_padded_sequence` with masking, increasing batch size, and tuning hyperparameters reduced epoch time from 12 minutes to 3.2 minutes on Apple MPS.

4. **Recall improved dramatically** (+8.5 points) — the LSTM catches substantially more true engagement events by capturing temporal reading patterns that the MLP missed.

5. **The train-val AUC gap is controlled** (0.022) — despite 5x more parameters, the LSTM generalizes well, confirming that the regularization (weight decay, dropout, early stopping) is appropriate.

6. **Three feature additions addressed three MLP information gaps** — joint engagement rates, article content lengths, and behavioral momentum each contributed to the improvement.

7. **The engagement prediction ceiling is real** — our 0.6869 AUC is consistent with MIND NRMS (0.6776) and the gap to RecSys top solutions (0.76+) is explained by missing text embeddings and scale, not methodology.

8. **Industry-scale methods would amplify the privacy risk** — if a 0.69 AUC model creates identifiable fingerprints, a 0.76 AUC model with transformer encoders and collaborative filtering would create even more distinctive representations.

9. **The MLP baseline is frozen for a reason** — it provides a stable reference point for the privacy comparison. The LSTM demonstrates that model improvement worsens privacy, motivating the randomized smoothing defense in Phase 4.

10. **The project's contribution is the privacy analysis, not the prediction quality.** Both models serve as vehicles for demonstrating that engagement prediction inherently creates re-identification risk — a risk that scales with model sophistication.
