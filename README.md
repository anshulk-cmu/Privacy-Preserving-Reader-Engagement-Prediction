# Privacy-Preserving Reader Engagement Prediction

**94-806 Privacy in the Digital Age | Carnegie Mellon University | Spring 2026**

## Overview

This project investigates a hidden privacy risk in engagement prediction systems. When a model is trained to predict whether a user will engage with a news article, it inadvertently learns to **fingerprint individual users** from their reading behavior — even though it was never asked to identify anyone.

We demonstrate this risk using the EB-NeRD news recommendation dataset, then explore defenses using **Randomized Smoothing** to add mathematically calibrated noise that destroys the fingerprint signal while preserving predictive accuracy.

**Key finding so far**: An MLP model trained solely to predict engagement creates behavioral representations that can re-identify users at **314x above random chance** among 28,361 users — using only aggregate reading statistics.

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

### Engagement Prediction (MLP Baseline)

| Metric | Train | Validation | Gap |
|--------|-------|------------|-----|
| AUC-ROC | 0.6922 | 0.6817 | +0.010 |
| F1 Score | 0.5729 | 0.5694 | +0.004 |
| Accuracy | 0.6460 | 0.6384 | +0.008 |
| Precision | 0.5556 | 0.5551 | +0.001 |
| Recall | 0.5914 | 0.5844 | +0.007 |

The model predicts engagement moderately well (AUC 0.68) from 42 aggregate features with minimal overfitting.

### User Re-identification Attack

| Metric | Cosine Distance | Euclidean Distance | Random Baseline |
|--------|----------------|-------------------|-----------------|
| Top-1 Accuracy | **1.11%** | 0.93% | 0.0035% |
| Top-5 Accuracy | **1.94%** | 1.58% | 0.018% |
| Top-10 Accuracy | **2.45%** | 2.03% | 0.035% |
| Top-20 Accuracy | **3.18%** | 2.67% | 0.071% |
| MRR | **0.0168** | 0.0142 | 0.0004 |
| Lift over random | **314x** | 264x | 1x |

A nearest-neighbor attack on the MLP's 64-dim representations re-identifies users at 314x above chance among 28,361 users — demonstrating that engagement models inadvertently learn user-distinctive behavioral fingerprints.

## Project Structure

```
94806_term_project/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── docs/
│   ├── 01_eda_analysis.md           # EDA results and dataset documentation
│   ├── 02_data_pipeline.md          # Preprocessing pipeline documentation
│   └── 03_mlp_baseline_analysis.md  # MLP model + re-identification analysis
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
│   ├── 03_reidentification_test.py  # Blind user re-identification attack
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Load parquets, engineer features, normalize
│   │   └── dataset.py               # PyTorch Dataset + DataLoader
│   └── models/
│       ├── __init__.py
│       ├── mlp_baseline.py          # Deep MLP engagement model (207K params)
│       ├── train.py                 # Training infrastructure, losses, plotting
│       └── attack.py                # Nearest-neighbor re-identification attack
├── outputs/
│   ├── figures/                     # EDA plots (Phase 1)
│   └── models/
│       └── mlp_baseline/            # MLP outputs (Phase 2B)
│           ├── checkpoint.pt        # Best model weights
│           ├── metrics.json         # Training history and final metrics
│           ├── representations.npz  # 64-dim representations for all samples
│           ├── training_curves.png
│           ├── evaluation_plots.png
│           ├── representation_analysis.png
│           ├── reidentification_cosine.png
│           ├── reidentification_euclidean.png
│           ├── reidentification_comparison.png
│           └── reidentification_results.json
└── ebnerd-benchmark/                # Cloned EB-NeRD benchmark repo (reference)
```

## Methodology

### Phase 1: Data & EDA (Complete)
- Constructed 50K-user dataset preserving EB-NeRD temporal splits
- Analyzed engagement signal distributions, user history characteristics, and behavioral uniqueness
- Confirmed 100% fingerprint uniqueness — strong motivation for privacy mechanisms
- Documentation: [docs/01_eda_analysis.md](docs/01_eda_analysis.md)

### Phase 2A: Data Pipeline (Complete)
- Built preprocessing pipeline: label creation, history extraction, feature engineering, normalization
- 546K train / 568K val samples, 40% positive rate, no NaN/Inf, no train-to-val leakage
- History sequences (50 timesteps x 2 features), 21 aggregate features, article + context features
- PyTorch Dataset and DataLoader with custom collation
- Documentation: [docs/02_data_pipeline.md](docs/02_data_pipeline.md)

### Phase 2B: MLP Baseline + Re-identification (Complete)
- 6-layer deep MLP (207K params) with SiLU activations and residual connections
- LabelSmoothingBCE loss with pos_weight for class balance
- AUC 0.6817, F1 0.57, balanced precision/recall
- 64-dim user representations extracted for privacy analysis
- Blind re-identification attack: 314x above random (1.11% Top-1 among 28K users)
- Documentation: [docs/03_mlp_baseline_analysis.md](docs/03_mlp_baseline_analysis.md)

### Phase 2C: LSTM Model (Next)
- 2-layer bidirectional LSTM with multi-head self-attention pooling
- Consumes raw behavioral sequences (50 timesteps of read_time + scroll_pct)
- Expected to significantly improve both engagement prediction and re-identification risk
- Richer temporal representations should demonstrate stronger privacy vulnerability

### Phase 4: Randomized Smoothing for Privacy (Planned)
- Add calibrated Gaussian noise to learned representations
- Certify a radius R within which the prediction is stable
- This certified radius provides re-identification resistance
- Sweep noise levels to map the privacy-utility tradeoff

### Phase 5: Full Privacy Evaluation (Planned)
- Re-identification attack before and after smoothing
- Compare MLP vs LSTM vulnerability
- Quantify how much noise is needed to reduce re-identification to near-random
- Measure engagement prediction degradation at each noise level

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

# Step 5: Run re-identification attack
python src/03_reidentification_test.py
```

## Key References

- **EB-NeRD Dataset**: [arxiv.org/abs/2410.03432](https://arxiv.org/abs/2410.03432)
- **Randomized Smoothing**: Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (ICML 2019)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
