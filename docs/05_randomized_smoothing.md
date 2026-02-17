# Phase 4: Randomized Smoothing — Privacy Defense

**Privacy-Preserving Reader Engagement Prediction**
**94-806 Privacy in the Digital Age | Carnegie Mellon University**

> **Note**: This document references MLP and LSTM results from a prior execution environment.
> Both models have been retrained on the current device: MLP (AUC 0.6951, 69x lift) and LSTM (AUC 0.6975, 2,958x lift).
> All comparison numbers will be updated when this phase is re-executed on the current device.

---

## Table of Contents

1. [Motivation: Why Randomized Smoothing?](#1-motivation-why-randomized-smoothing)
2. [Threat Model and Design Choices](#2-threat-model-and-design-choices)
3. [Mathematical Framework](#3-mathematical-framework)
4. [Sigma Calibration Strategy](#4-sigma-calibration-strategy)
5. [Experimental Setup](#5-experimental-setup)
6. [Utility Results: Engagement Prediction Under Noise](#6-utility-results-engagement-prediction-under-noise)
7. [Privacy Results: Re-identification Under Noise](#7-privacy-results-re-identification-under-noise)
8. [Certified Robustness Analysis](#8-certified-robustness-analysis)
9. [Privacy-Utility Tradeoff: The Core Result](#9-privacy-utility-tradeoff-the-core-result)
10. [MLP vs LSTM: Comparative Analysis](#10-mlp-vs-lstm-comparative-analysis)
11. [Recommended Operating Point](#11-recommended-operating-point)
12. [Connection to Differential Privacy](#12-connection-to-differential-privacy)
13. [Limitations](#13-limitations)
14. [Generated Outputs](#14-generated-outputs)
15. [Key Takeaways](#15-key-takeaways)

---

## 1. Motivation: Why Randomized Smoothing?

Phases 3A and 3B established a dangerous pattern: engagement prediction models inadvertently fingerprint users. The MLP baseline re-identifies users at 314x above random chance; the LSTM amplifies this to 2,853x. The natural question is: **Can we defend against this?**

We need a defense mechanism that:

1. **Provably destroys fingerprints** — not just empirically, but with mathematical guarantees
2. **Preserves prediction accuracy** — noise that eliminates all signal is useless
3. **Is practical to deploy** — works post-training without retraining the model
4. **Quantifies the tradeoff** — tells us exactly how much utility we sacrifice for each unit of privacy

Randomized Smoothing (Cohen et al., ICML 2019) satisfies all four requirements. Originally developed for adversarial robustness, we repurpose it as a **privacy mechanism**: if a prediction is certified stable within a radius R, then perturbations within that radius (which include perturbations toward other users) cannot change the output — making the representation useless for re-identification.

### Why Not Differential Privacy Directly?

Formal differential privacy (DP) provides strong guarantees but requires either:
- **Training-time integration** (e.g., DP-SGD), which is computationally expensive and degrades model accuracy significantly at practical privacy budgets, or
- **Output perturbation**, which assumes a specific query structure

Randomized Smoothing provides **certified robustness** — a weaker but more practical guarantee that operates at inference time on already-trained models. It allows us to answer: "How much noise is needed to make re-identification infeasible, and what does it cost in prediction accuracy?"

---

## 2. Threat Model and Design Choices

### Where to Inject Noise?

Both MLP and LSTM share the same bottleneck architecture:

```
Raw Features → Encoder (MLP or LSTM) → r ∈ R^64 → Linear(64,1) → P(engaged)
                                         ↑
                                    Noise injected HERE
```

We inject Gaussian noise at the **representation level** (the 64-dimensional bottleneck), not at the input or output level, for three reasons:

1. **Re-identification operates in representation space.** The attack uses nearest-neighbor matching on 64-dim representations. Adding noise here directly counters the attack mechanism.

2. **The classification head is linear.** Since `head = Linear(64, 1)` (with sigmoid applied afterward), adding Gaussian noise to the representation yields a smoothed prediction that can be computed **analytically** — no expensive Monte Carlo sampling needed.

3. **Computational efficiency.** We work with pre-extracted `representations.npz` files. No need to re-run the encoder for each noise sample.

### Noise Distribution

We use isotropic Gaussian noise: ε ~ N(0, σ²I₆₄). This is the standard choice for Randomized Smoothing because:
- It is rotationally symmetric (no directional bias)
- It enables analytical solutions for linear classifiers
- The expected L2 displacement is E[‖ε‖₂] ≈ σ√d ≈ 8σ for d=64, providing a direct link between σ and the spatial scale of perturbation

---

## 3. Mathematical Framework

### 3.1 Analytical Smoothed Prediction

Given a representation r ∈ R^64 and classification head weights (w, b):

```
logit(r) = w · r + b
P_original(engaged | r) = σ(w · r + b)      where σ is sigmoid
```

Adding Gaussian noise ε ~ N(0, σ²I):

```
logit(r + ε) = w · (r + ε) + b = (w · r + b) + w · ε
```

Since w · ε ~ N(0, σ²‖w‖²), we have:

```
P[logit(r + ε) > 0] = P[w · r + b + w · ε > 0]
                     = P[w · ε > -(w · r + b)]
                     = Φ((w · r + b) / (σ · ‖w‖))
```

where Φ is the standard normal CDF. This is the **smoothed probability** — exact with no approximation.

**Key property:** At σ = 0, this reduces to a step function (hard classification). As σ increases, predictions compress toward 0.5, degrading discrimination but making all representations produce similar outputs.

### 3.2 Certified Radius (Cohen et al., 2019)

For binary classification, if the smoothed classifier predicts class A with probability p_A = max(p, 1-p), then the prediction is guaranteed stable within an L2 ball of radius:

```
R = σ · Φ⁻¹(p_A)
```

**Interpretation:** No perturbation of r within ‖δ‖₂ ≤ R can change the smoothed prediction. This means that even an adversary who can modify the representation cannot change the engagement prediction — and, more relevantly for privacy, two users whose representations are within distance R of each other are guaranteed to produce the same output.

**Privacy interpretation:** If the certified radius R exceeds the nearest-neighbor distance between users in representation space, then the model's output is provably identical for nearby users — making re-identification impossible within that neighborhood.

### 3.3 Nearest-Neighbor Distance as Privacy Threshold

To calibrate σ, we need a reference scale: how far apart are users in representation space?

For each user u, we compute:
- The mean representation: r̄_u = (1/|S_u|) Σᵢ r_i for all impressions i of user u
- The nearest-neighbor distance: d_NN(u) = min_{v≠u} dist(r̄_u, r̄_v)

We compute this in both metrics:
- **Cosine distance**: Used in the re-identification attack (matches Phase 3A/3B)
- **Euclidean (L2) distance**: Matches the certified radius metric (Cohen et al. guarantees are L2)

When the expected noise displacement E[‖ε‖₂] ≈ σ√64 exceeds the typical nearest-neighbor L2 distance, user fingerprints overlap and re-identification degrades to near-random.

### 3.4 Monte Carlo Certification (Verification)

As a verification of the analytical approach, we also implement Monte Carlo certification with Clopper-Pearson confidence bounds:

For each sample:
1. Draw n = 1000 noise vectors εⱼ ~ N(0, σ²I)
2. Compute n noisy predictions: ŷⱼ = 1[w · (r + εⱼ) + b > 0]
3. Count majority class: c_A = max(Σⱼ ŷⱼ, n - Σⱼ ŷⱼ)
4. Compute Clopper-Pearson lower bound: p̂_A = Beta.ppf(α/2, c_A, n - c_A + 1) at confidence 1-α
5. Certified radius: R_MC = σ · Φ⁻¹(p̂_A) if p̂_A > 0.5, else ABSTAIN

This is slower (O(N × n_samples)) but provides rigorous confidence-bounded radii. For our linear head, the analytical and MC results should agree closely.

---

## 4. Sigma Calibration Strategy

We sweep σ across a broad range to capture the full privacy-utility landscape:

```
σ ∈ {0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0}
```

### Rationale for Range

- **σ = 0.0**: No-noise baseline; should exactly reproduce Phase 3A/3B results
- **σ = 0.01 - 0.05**: Negligible noise; verifies continuity
- **σ = 0.1 - 0.5**: Moderate noise; expected "sweet spot" where re-identification drops significantly but AUC remains acceptable
- **σ = 0.75 - 1.0**: Strong noise; expected to destroy most fingerprints but may degrade AUC
- **σ = 1.5 - 3.0**: Very strong noise; expected to push re-identification to near-random at the cost of significant AUC degradation

### Crossover Points

We identify two critical sigma values:

1. **σ_privacy**: Smallest σ where re-identification lift drops below 2x (near-random)
2. **σ_utility**: Largest σ where AUC remains above 0.6 (still useful)

If σ_privacy < σ_utility, a viable "operating point" exists where both privacy and utility are acceptable.

### The Better Model = More Risk Hypothesis

We expect the LSTM to require larger σ than the MLP because:
- LSTM representations are more distinctive (2,853x vs 314x lift)
- LSTM representations may be denser (smaller NN distances)
- Destroying a richer fingerprint requires more noise

This would confirm the project's narrative: **the better the model, the harder it is to defend privacy**.

---

## 5. Experimental Setup

### Input Data

For each model (MLP and LSTM), we use the pre-extracted validation representations:

| Component | Shape | Source |
|-----------|-------|--------|
| Representations | (N_val, 64) | `representations.npz` |
| Labels | (N_val,) | Ground-truth engagement |
| User IDs | (N_val,) | For re-identification |
| Head weights | (64,) + scalar | `checkpoint.pt` |

### Per-Sigma Evaluation

For each σ, we compute three categories of metrics:

**Utility (Analytical):**
- Smoothed AUC-ROC, F1, Accuracy, Precision, Recall, Average Precision
- Computed using the analytical formula: P = Φ(logit / (σ · ‖w‖))

**Privacy (Noisy Re-identification):**
- Add noise to representations, then run the same gallery/probe attack from Phase 3A/3B
- Average over 10 independent noise draws for stable estimates
- Metrics: Top-1/5/10/20 accuracy, MRR, lift over random, mean/median rank

**Certification:**
- Per-sample certified radius R = σ · Φ⁻¹(p_A)
- Summary: mean R, median R, fraction certified (R > 0), fraction above thresholds

### Monte Carlo Verification

At σ = 0.5 and σ = 1.0, we also run Monte Carlo certification (1,000 samples, α = 0.001) on up to 3,000 points to verify that the analytical certified radii agree with the MC bounds.

### Configuration

```python
SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
N_REID_TRIALS = 10           # noise draws for stable re-id measurement
MC_N_SAMPLES = 1000          # Monte Carlo samples per point
MC_ALPHA = 0.001             # Clopper-Pearson confidence level (99.9%)
REID_METRIC = "cosine"       # matches Phase 3A/3B re-id attack metric
```

---

## 6. Utility Results: Engagement Prediction Under Noise

> **Note:** This section will be populated with actual results after running `src/06_randomized_smoothing.py`. The structure below shows the expected analysis framework.

### Expected Behavior

As σ increases:
- The smoothed probability Φ(logit / (σ · ‖w‖)) compresses all predictions toward 0.5
- AUC degrades smoothly because the ranking of predictions is preserved at small σ but eventually collapses
- At very large σ, logit / (σ · ‖w‖) → 0 for all samples, so Φ → 0.5 (random prediction)

### AUC Degradation Table

| σ | MLP AUC | LSTM AUC |
|---|---------|----------|
| 0.0 | [baseline] | [baseline] |
| 0.1 | [result] | [result] |
| 0.5 | [result] | [result] |
| 1.0 | [result] | [result] |
| 2.0 | [result] | [result] |
| 3.0 | [result] | [result] |

### Key Observations (Expected)

1. AUC should be nearly unchanged at σ ≤ 0.1 (small noise relative to signal)
2. Significant degradation expected around σ = 0.5-1.0
3. The MLP and LSTM may degrade at different rates depending on their head weight norms ‖w‖

The AUC degradation is governed by the "signal-to-noise ratio" logit / (σ · ‖w‖). Models with larger ‖w‖ are more sensitive to noise because the noise standard deviation scales with ‖w‖.

---

## 7. Privacy Results: Re-identification Under Noise

> **Note:** Results to be populated after execution.

### Noisy Re-identification Protocol

At each σ:
1. Add noise: r̃ = r + ε, where ε ~ N(0, σ²I₆₄)
2. Build gallery (per-user mean of noisy representations) and probe set
3. Run nearest-neighbor re-identification attack using cosine distance
4. Repeat 10 times with independent noise draws; average metrics

### Re-identification Decay Table

| σ | MLP Top-1 | MLP Lift | LSTM Top-1 | LSTM Lift |
|---|-----------|----------|------------|-----------|
| 0.0 | [baseline] | [baseline] | [baseline] | [baseline] |
| 0.1 | [result] | [result] | [result] | [result] |
| 0.5 | [result] | [result] | [result] | [result] |
| 1.0 | [result] | [result] | [result] | [result] |
| 2.0 | [result] | [result] | [result] | [result] |
| 3.0 | [result] | [result] | [result] | [result] |

### Key Observations (Expected)

1. Re-identification should remain high at small σ (noise doesn't change NN structure much)
2. A sharp "cliff" is expected where fingerprints start overlapping
3. The LSTM should require more noise to reach the same privacy level as the MLP (because LSTM fingerprints are 9.1x stronger)
4. At large σ, both models should converge to near-random re-identification

---

## 8. Certified Robustness Analysis

### Certified Radius Distribution

For each σ and each sample, the certified radius R = σ · Φ⁻¹(p_A) indicates how much the representation can be perturbed without changing the prediction.

**Key questions:**
1. What fraction of samples have R > 0 (non-trivially certified)?
2. How does the certified radius compare to the median L2 nearest-neighbor distance?
3. At what σ do most samples have R > d_NN (certifiably indistinguishable from their nearest neighbor)?

### Certification Coverage Table

| σ | MLP R > 0 | MLP Median R | LSTM R > 0 | LSTM Median R |
|---|-----------|-------------|------------|--------------|
| 0.1 | [result] | [result] | [result] | [result] |
| 0.5 | [result] | [result] | [result] | [result] |
| 1.0 | [result] | [result] | [result] | [result] |
| 2.0 | [result] | [result] | [result] | [result] |

### Monte Carlo Verification

At σ = 0.5 and σ = 1.0, we verify the analytical certified radii against Monte Carlo estimates with Clopper-Pearson bounds (α = 0.001, n = 1000). For a linear head, these should agree closely:

| σ | Analytical Mean R | MC Mean R | MC Abstain Rate |
|---|-------------------|-----------|-----------------|
| 0.5 | [result] | [result] | [result] |
| 1.0 | [result] | [result] | [result] |

---

## 9. Privacy-Utility Tradeoff: The Core Result

This is the main deliverable of the project — a visualization showing the Pareto frontier of achievable (AUC, Re-identification) pairs as noise varies.

### Reading the Tradeoff Curve

The ideal operating point is the **knee** of the Pareto curve: the σ where:
- Re-identification lift drops dramatically (steep descent)
- AUC degradation is still minimal (flat region)

### MLP vs LSTM on the Pareto Frontier

We expect the LSTM curve to be shifted right/upward compared to MLP:
- At the same AUC, the LSTM has higher re-identification risk
- To achieve the same privacy level, the LSTM needs more noise (and sacrifices more AUC)

This confirms the core thesis: **more capable models require stronger privacy interventions**.

---

## 10. MLP vs LSTM: Comparative Analysis

### Representation Space Geometry

The MLP and LSTM create different representation space geometries:

| Property | MLP | LSTM |
|----------|-----|------|
| Parameters | ~207K | ~1.0M |
| Representation dim | 64 | 64 |
| Clean AUC | 0.6817 | 0.6869 |
| Clean Re-id Lift | 314x | 2,853x |
| Median NN distance (cosine) | [result] | [result] |
| Median NN distance (L2) | [result] | [result] |

### Why LSTM is Harder to Defend

The LSTM's richer representations (temporal patterns, attention-weighted sequences) create more unique fingerprints. This manifests as:

1. **Smaller NN distances** (likely): Users' representations are more densely packed but more uniquely positioned
2. **Higher baseline re-identification**: 9.1x stronger than MLP
3. **Steeper privacy cost**: Need more noise to reach the same privacy level, incurring more AUC loss

### The Privacy Paradox

This reveals a fundamental tension in machine learning:
- **Better prediction** requires learning more distinctive user representations
- **More distinctive representations** enable more accurate re-identification
- **Defending privacy** requires adding noise that degrades the very features that improve prediction

The noise level σ parameterizes this tradeoff, and our experiments quantify it precisely.

---

## 11. Recommended Operating Point

> **Note:** To be calibrated after results are available.

### Criteria for the Recommended σ

We define three privacy tiers:

| Tier | Re-id Lift Threshold | Description |
|------|---------------------|-------------|
| Privacy-viable | < 5x random | Fingerprinting mostly destroyed |
| Near-random | < 2x random | Fingerprinting effectively eliminated |
| Provably safe | < 1.5x random | Statistically indistinguishable from random |

### Selection Methodology

1. Plot the re-identification lift curve for each model
2. Identify σ_privacy: smallest σ where lift < 2x (near-random threshold)
3. Identify σ_utility: largest σ where AUC > 0.6 (usability floor)
4. If σ_privacy < σ_utility: recommended σ = midpoint of [σ_privacy, σ_utility]
5. If σ_privacy > σ_utility: no viable operating point exists — privacy requires sacrificing utility below acceptable levels

### Expected Recommendation

Based on the mathematical framework:
- For the MLP (314x baseline lift), moderate σ should suffice
- For the LSTM (2,853x baseline lift), larger σ is needed, with more AUC cost
- The "gap" between MLP and LSTM operating points quantifies the privacy cost of model sophistication

---

## 12. Connection to Differential Privacy

### What We Provide vs. Formal DP

Randomized Smoothing provides **certified robustness** — a guarantee that predictions are stable within an L2 ball. This is related to, but distinct from, formal differential privacy:

| Property | Randomized Smoothing (ours) | Differential Privacy |
|----------|---------------------------|---------------------|
| Guarantee type | Prediction stability | Distributional indistinguishability |
| Formal parameter | Certified radius R (L2) | Privacy budget ε, δ |
| Applies to | Individual predictions | Query mechanisms |
| When noise is added | Inference time | Training time (DP-SGD) or query time |
| Retraining needed? | No | Yes (for DP-SGD) |
| Composability | Per-query | Composes across queries |

### Conceptual Connection

The smoothed representation can be viewed through a DP lens:

- Adding N(0, σ²I) to the representation is analogous to the Gaussian mechanism
- The certified radius R relates to the "sensitivity" of the representation function
- A larger R means more noise tolerance, analogous to a smaller ε (more private)

However, we do not claim formal (ε, δ)-differential privacy because:
1. Our noise is added to a fixed representation, not to the training process
2. Multiple queries to the same representation can leak information
3. The guarantee is per-sample, not population-level

### Why This Is Still Valuable

Even without formal DP, the certified radius provides actionable guidance:
- If R > d_NN for a user, that user's representation is provably indistinguishable from their nearest neighbor
- The σ sweep maps the full range of noise levels, showing exactly where privacy is achieved
- Practitioners can choose their operating point based on their specific risk tolerance

---

## 13. Limitations

### 1. Linear Head Assumption

Our analytical smoothing formula is exact only because the classification head is `Linear(64, 1)`. If the head had nonlinear layers, we would need Monte Carlo sampling, which is more expensive and provides only probabilistic bounds.

### 2. Representation-Level vs. Input-Level Smoothing

We add noise to the 64-dim representation, not the raw input. This means:
- An adversary who can observe the raw input (before encoding) is not defended
- The defense protects against attacks on the representation space (which is the relevant threat for re-identification)

### 3. Static Noise (No Adaptive Defense)

The noise level σ is fixed for all samples. An adaptive defense could adjust σ per-user based on their identifiability (e.g., more noise for easily identifiable users). We leave this to future work.

### 4. Single-Query Guarantee

The certified radius guarantees stability for a single query. An adversary who observes multiple noisy versions of the same representation can average out the noise. In deployment, this would require generating fresh noise for each query (or limiting query access).

### 5. Post-Hoc Defense

Randomized Smoothing is applied after training. A stronger approach would be to train the model to produce inherently privacy-preserving representations (e.g., via adversarial regularization or DP-SGD). Our approach demonstrates the tradeoff without the expense of retraining.

### 6. Binary Classification Scope

The certified radius formula R = σ · Φ⁻¹(p_A) applies to binary classification. Extension to multi-class settings (e.g., engagement level prediction) would require the general Neyman-Pearson framework.

---

## 14. Generated Outputs

### Comparison Plots (outputs/models/smoothing/comparison/)

| File | Description |
|------|-------------|
| `privacy_utility_tradeoff.png` | **Main deliverable**: AUC vs re-id lift at each σ, both models |
| `reid_decay.png` | Re-id Top-1/5/10 accuracy decay as σ increases |
| `auc_degradation.png` | AUC degradation curves with usability floor |
| `certification_coverage.png` | Fraction of samples certified at each σ |
| `smoothing_summary.png` | 2×2 grid: Pareto frontier, clean vs noisy bars, AUC curve, summary stats |

### Per-Model Plots (outputs/models/smoothing/{mlp,lstm}/)

| File | Description |
|------|-------------|
| `certified_radii.png` | Certified radius histograms at σ = {0.1, 0.5, 1.0, 2.0} with NN distance reference |
| `recommended_sigma_detail.png` | 2×3 detailed analysis at the recommended σ |

### Data Files (outputs/models/smoothing/)

| File | Description |
|------|-------------|
| `smoothing_results.json` | All numerical results (utility, privacy, certification per σ per model) |

---

## 15. Key Takeaways

> **Note:** Final takeaways will be refined after results are available. The framework below captures the expected narrative.

1. **Randomized Smoothing provides a principled defense** against re-identification by adding calibrated Gaussian noise to learned representations, with exact analytical computation for linear classification heads.

2. **The privacy-utility tradeoff is quantifiable**: each noise level σ corresponds to a specific (AUC, Re-id lift) pair, enabling practitioners to choose their operating point.

3. **Certified radii provide mathematical guarantees**: for each prediction, we can state the maximum perturbation that will not change the output — directly connecting to the nearest-neighbor distance that determines re-identification.

4. **More capable models require stronger privacy defenses**: the LSTM's richer representations (2,853x re-id lift) are expected to need more noise than the MLP (314x lift), incurring greater AUC cost — confirming that privacy risk scales non-linearly with model sophistication.

5. **Post-hoc defense is practical but limited**: Randomized Smoothing works without retraining, making it deployable on existing systems. However, it provides per-query guarantees rather than the compositional guarantees of formal differential privacy.

6. **The sigma sweep reveals the full landscape**: from zero noise (maximum utility, maximum risk) to extreme noise (random predictions, no risk), our experiments map every point on the privacy-utility frontier.

---

## References

- Cohen, J., Rosenfeld, E., & Kolter, Z. (2019). *Certified Adversarial Robustness via Randomized Smoothing*. ICML 2019. [proceedings.mlr.press/v97/cohen19c](https://proceedings.mlr.press/v97/cohen19c.html)
- Dwork, C., & Roth, A. (2014). *The Algorithmic Foundations of Differential Privacy*. Foundations and Trends in Theoretical Computer Science.
- Lecuyer, M., et al. (2019). *Certified Robustness to Adversarial Examples with Differential Privacy*. IEEE S&P 2019.

---

*This document is part of the 94-806 term project. For dataset and preprocessing details, see [docs/02_data_pipeline.md](02_data_pipeline.md). For model architectures, see [docs/03_mlp_baseline_analysis.md](03_mlp_baseline_analysis.md) and [docs/04_lstm_analysis.md](04_lstm_analysis.md).*
