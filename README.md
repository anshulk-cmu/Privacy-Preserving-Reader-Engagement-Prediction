# Privacy-Preserving Reader Engagement Prediction

**94-806 Privacy in the Digital Age | Carnegie Mellon University | Spring 2026**

## Overview

This project investigates a hidden privacy risk in engagement prediction systems. When a model is trained to predict whether a user will engage with a news article, it inadvertently learns to **fingerprint individual users** from their reading behavior — even though it was never asked to identify anyone.

We demonstrate this risk using the EB-NeRD news recommendation dataset, then explore defenses using **Randomized Smoothing** to add mathematically calibrated noise that destroys the fingerprint signal while preserving predictive accuracy.

**Key finding**: A BiLSTM model processing temporal reading sequences re-identifies users at **2,853x above random chance** among 28,361 users. Critically, upgrading from a simple MLP (314x lift) to the LSTM (+0.52% AUC) amplified re-identification risk by **9.1x** — demonstrating that **privacy risk scales non-linearly with model sophistication**.

## Dataset

We use the **EB-NeRD** (Ekstra Bladet News Recommendation Dataset), published alongside [this paper](https://arxiv.org/abs/2410.03432). The dataset contains behavioral logs from Ekstra Bladet, a major Danish news outlet, collected over 6 weeks (April 27 - June 8, 2023).

We constructed a custom **50,000-user subset** (`ebnerd_50k`) by sampling 50,000 users from `ebnerd_small` and `ebnerd_large` combined (see `src/01_create_50k_dataset.py` for details).

The dataset preserves the original **temporal split methodology** from the paper:

| Split | Period | Users | Impressions |
|-------|--------|-------|-------------|
| Train behaviors | May 18 - May 25, 2023 | 40,332 | 615,316 |
| Train history | April 27 - May 18 (21 days) | 40,332 | — |
| Validation behaviors | May 25 - June 1, 2023 | 40,616 | 645,495 |
| Validation history | May 4 - May 25 (21 days) | 40,616 | — |

User overlap between train and validation is 76.2%, which is a natural consequence of the temporal design (users active in consecutive weeks appear in both splits).

## Engagement Label

We define binary engagement as:

```
engaged = (scroll_percentage > 50%) AND (read_time > 30 seconds)
```

This produces a well-balanced label (~40% positive, class ratio 1:1.5) that captures users who both consumed the content depth-wise and invested meaningful reading time.

## Results Summary

### Engagement Prediction

| Metric | LSTM (Val) | MLP (Val) | Improvement |
|--------|-----------|-----------|-------------|
| AUC-ROC | **0.6869** | 0.6817 | +0.0052 |
| F1 Score | **0.5947** | 0.5694 | +0.0253 |
| Recall | **0.6694** | 0.5844 | +0.0850 |
| Precision | 0.5350 | **0.5551** | -0.0201 |
| Accuracy | 0.6268 | **0.6384** | -0.0116 |

The LSTM surpasses the MLP baseline across ranking metrics (AUC, F1, recall) by processing raw temporal sequences and enriched features. Both models operate well within the expected range — the RecSys 2024 Challenge top-3 teams averaged 0.7643 AUC using full text embeddings, 1M+ users, and months of engineering.

### User Re-identification Attack — The Core Finding

| Metric | LSTM (Best) | MLP (Best) | LSTM/MLP Ratio | Random |
|--------|------------|------------|----------------|--------|
| Top-1 Accuracy | **10.06%** | 1.11% | **9.1x** | 0.0035% |
| Top-5 Accuracy | **15.30%** | 1.94% | **7.9x** | 0.018% |
| Top-10 Accuracy | **17.86%** | 2.45% | **7.3x** | 0.035% |
| Top-20 Accuracy | **20.73%** | 3.18% | **6.5x** | 0.071% |
| MRR | **0.1280** | 0.0168 | **7.6x** | 0.0004 |
| Lift over random | **2,853x** | 314x | **9.1x** | 1x |
| Users 100% identifiable | 231 | 34 | 6.8x | 0 |

**Key finding**: A +0.52% AUC improvement (0.6817 → 0.6869) produced a **9.1x increase in re-identification risk** (314x → 2,853x lift). The LSTM's temporal behavioral sequences create dramatically more distinctive fingerprints than the MLP's aggregate statistics — demonstrating that **privacy risk scales non-linearly with model sophistication**.

## Project Structure

```
94806_term_project/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── docs/
│   ├── 01_eda_analysis.md           # EDA results and dataset documentation
│   ├── 02_data_pipeline.md          # Preprocessing pipeline documentation
│   ├── 03_mlp_baseline_analysis.md  # MLP model + re-identification analysis
│   ├── 04_lstm_analysis.md          # LSTM model + MLP comparison + privacy amplification
│   └── 05_randomized_smoothing.md   # Randomized smoothing: math, results, tradeoff analysis
├── data/
│   ├── ebnerd_50k/                  # Custom 50K-user dataset
│   │   ├── train/
│   │   │   ├── behaviors.parquet
│   │   │   └── history.parquet
│   │   ├── validation/
│   │   │   ├── behaviors.parquet
│   │   │   └── history.parquet
│   │   └── articles.parquet
│   └── processed/                   # Model-ready numpy arrays
│       ├── train/                   # history_seq, agg_features, labels, etc.
│       ├── val/                     # Same structure as train
│       ├── metadata.json            # Feature dimensions, sample counts
│       ├── rt_scaler.pkl            # Fitted read-time StandardScaler
│       └── agg_scaler.pkl           # Fitted aggregate-feature StandardScaler
├── src/
│   ├── 00_eda.py                    # Exploratory data analysis
│   ├── 01_create_50k_dataset.py     # Script that built ebnerd_50k
│   ├── 02_train_mlp.py              # MLP training + representation extraction
│   ├── 03_reidentification_test.py  # Blind user re-identification attack (MLP)
│   ├── 04_train_lstm.py             # LSTM training + representation extraction
│   ├── 05_lstm_reidentification.py  # Blind user re-identification attack (LSTM)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Load parquets, engineer features, normalize
│   │   └── dataset.py               # PyTorch Dataset + DataLoader
│   └── models/
│       ├── __init__.py
│       ├── mlp_baseline.py          # Deep MLP engagement model (207K params, frozen)
│       ├── lstm_model.py            # BiLSTM + Attention model (~1M params)
│       ├── train.py                 # Training infrastructure, losses, plotting
│       ├── attack.py                # Nearest-neighbor re-identification attack
│       └── smoothing.py             # Randomized smoothing: analytical + MC + re-id (Phase 4)
├── outputs/
│   ├── figures/                     # EDA plots (Phase 1)
│   └── models/
│       ├── mlp_baseline/            # MLP outputs (Phase 3A)
│       │   ├── checkpoint.pt, metrics.json, representations.npz
│       │   ├── training_curves.png, evaluation_plots.png
│       │   ├── reidentification_*.png
│       │   └── reidentification_results.json
│       ├── lstm/                    # LSTM outputs (Phase 3B)
│       │   ├── checkpoint.pt, metrics.json, representations.npz
│       │   ├── training_curves.png, evaluation_plots.png
│       │   ├── reidentification_*.png
│       │   ├── lstm_vs_mlp_comparison.png
│       │   └── reidentification_results.json
│       └── smoothing/               # Randomized smoothing outputs (Phase 4)
│           ├── smoothing_results.json
│           ├── comparison/          # Cross-model comparison plots
│           │   ├── privacy_utility_tradeoff.png  (main deliverable)
│           │   ├── reid_decay.png, auc_degradation.png
│           │   ├── certification_coverage.png
│           │   └── smoothing_summary.png
│           ├── mlp/                 # MLP-specific plots
│           │   ├── certified_radii.png
│           │   └── recommended_sigma_detail.png
│           └── lstm/                # LSTM-specific plots
│               ├── certified_radii.png
│               └── recommended_sigma_detail.png
└── ebnerd-benchmark/                # Cloned EB-NeRD benchmark repo (reference)
```

## Methodology

### Phase 1: Data & EDA (Complete)
- Constructed 50K-user dataset preserving EB-NeRD temporal splits
- Analyzed engagement signal distributions, user history characteristics, and behavioral uniqueness
- Confirmed 100% fingerprint uniqueness — strong motivation for privacy mechanisms
- Documentation: [docs/01_eda_analysis.md](docs/01_eda_analysis.md)

### Phase 2: Data Pipeline (Complete)
- Built preprocessing pipeline: label creation, history extraction, feature engineering, normalization
- 546K train / 568K val samples, 40% positive rate, no NaN/Inf, no train-to-val leakage
- History sequences (50 timesteps x 2 features), 27 aggregate features, 7 article features, 3 context features
- PyTorch Dataset and DataLoader with custom collation
- Documentation: [docs/02_data_pipeline.md](docs/02_data_pipeline.md)

### Phase 3A: MLP Baseline + Re-identification (Complete)
- 6-layer deep MLP (207K params) with SiLU activations and residual connections
- LabelSmoothingBCE loss with pos_weight for class balance
- AUC 0.6817, F1 0.57, balanced precision/recall
- 64-dim user representations extracted for privacy analysis
- Blind re-identification attack: 314x above random (1.11% Top-1 among 28K users)
- Documentation: [docs/03_mlp_baseline_analysis.md](docs/03_mlp_baseline_analysis.md)

### Phase 3B: LSTM Model + Re-identification (Complete)
- 2-layer BiLSTM (1.0M params) with multi-head self-attention pooling
- Consumes raw behavioral sequences (50 timesteps) + 27 aggregate features + article content lengths
- Addresses three MLP information gaps: joint engagement rate, article body length, behavioral momentum
- AUC 0.6869 (+0.0052 over MLP), trained in 25 epochs with 3.8x speedup over v1
- **Re-identification: 2,853x above random** (10.06% Top-1 among 28K users) — **9.1x stronger than MLP**
- A +0.52% AUC gain amplified re-identification risk by 9.1x, demonstrating non-linear privacy scaling
- Documentation: [docs/04_lstm_analysis.md](docs/04_lstm_analysis.md)

### Phase 4: Randomized Smoothing for Privacy (Code Complete — Awaiting Execution)
- Add calibrated Gaussian noise ε ~ N(0, σ²I₆₄) to 64-dim learned representations
- Analytical smoothed prediction: P = Φ(logit / (σ·‖w‖)) — exact for linear classification head
- Certified radius R = σ · Φ⁻¹(p_A) guarantees prediction stability (Cohen et al., ICML 2019)
- Monte Carlo verification with Clopper-Pearson confidence bounds (α = 0.001)
- Sweep 11 noise levels (σ = 0 to 3.0) mapping the full privacy-utility tradeoff
- Dual-metric NN distances (cosine for re-id, euclidean for L2 certification comparison)
- Compare MLP vs LSTM: quantify how much more noise the LSTM needs for equivalent privacy
- 7 visualization plots including privacy-utility Pareto frontier (main deliverable)
- Documentation: [docs/05_randomized_smoothing.md](docs/05_randomized_smoothing.md)

## Setup

### Requirements
- Python 3.11 (ARM-native on Apple Silicon)
- Apple M3 Max or equivalent (MPS acceleration supported)

### Installation

```bash
# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install benchmark utilities
pip install -e ebnerd-benchmark/
```

### Reproducing the Dataset

If you need to rebuild `ebnerd_50k` from scratch:

1. Download `ebnerd_small` and `ebnerd_large` from [recsys.eb.dk](https://recsys.eb.dk/) into `data/` (temporary; deleted after creation)
2. Extract both zip files
3. Run: `python src/01_create_50k_dataset.py`
4. Delete `ebnerd_large` to free disk space

### Running the Full Pipeline

```bash
source .venv/bin/activate

# Step 1: EDA (outputs to outputs/figures/)
python src/00_eda.py

# Step 2: Preprocess data (outputs to data/processed/)
python src/data/preprocessing.py

# Step 3: Validate dataset
python src/data/dataset.py

# Step 4: Train MLP baseline (outputs to outputs/models/mlp_baseline/)
python src/02_train_mlp.py

# Step 5: MLP re-identification attack
python src/03_reidentification_test.py

# Step 6: Train LSTM (outputs to outputs/models/lstm/)
python src/04_train_lstm.py

# Step 7: LSTM re-identification + MLP comparison
python src/05_lstm_reidentification.py

# Step 8: Randomized smoothing privacy defense (Phase 4)
python src/06_randomized_smoothing.py
```

## Key References

- **EB-NeRD Dataset**: [arxiv.org/abs/2410.03432](https://arxiv.org/abs/2410.03432)
- **Randomized Smoothing**: Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (ICML 2019)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
