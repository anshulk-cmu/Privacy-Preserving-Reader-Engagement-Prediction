# Privacy-Preserving Reader Engagement Prediction

**94-806 Privacy in the Digital Age | Carnegie Mellon University | Spring 2026**

## Overview

This project investigates a hidden privacy risk in engagement prediction systems. When a model is trained to predict whether a user will engage with a news article, it inadvertently learns to **fingerprint individual users** from their reading behavior — even though it was never asked to identify anyone.

We demonstrate this risk using the EB-NeRD news recommendation dataset, then explore defenses using **Randomized Smoothing** to add mathematically calibrated noise that destroys the fingerprint signal while preserving predictive accuracy.

**Key finding**: Even a simple MLP trained on aggregate behavioral statistics re-identifies users at **69x above random chance** among 28,361 users — despite never being trained to identify anyone. A blind test on 10,000 completely unseen users confirms that fingerprinting is **feature-level** (24x lift), not model memorization. The LSTM amplifies this to **2,958x above random** (10.43% Top-1) with only a +0.24% AUC improvement, proving that **privacy risk scales non-linearly with model sophistication**. On the blind test, the LSTM achieves 841x lift on users it never trained on.

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

### Blind Test Set (Negative Control)

We also constructed a **10,000-user blind test set** (`ebnerd_blind_test`) with **zero overlap** to the main 50K dataset (see `src/01b_create_blind_test.py`). These users were sampled from the 573,933 `ebnerd_large` users who are active in both time periods but absent from `ebnerd_50k`.

| Split | Users | Impressions | Engagement Rate |
|-------|-------|-------------|-----------------|
| Train | 10,000 | 176,798 | 40.5% |
| Validation | 10,000 | 173,534 | 41.3% |
| **Overlap with ebnerd_50k** | **0** | — | — |

This serves three purposes:
1. **Negative control for re-identification**: The model never saw these users during training, so re-identification should yield ~0% accuracy — proving the attack specifically fingerprints *known* users.
2. **Engagement generalization**: Verifies prediction performance holds on completely unseen users.
3. **Privacy defense calibration**: Measures smoothing effects on unknown vs known users.

## Engagement Label

We define binary engagement as:

```
engaged = (scroll_percentage > 50%) AND (read_time > 30 seconds)
```

This produces a well-balanced label (~40% positive, class ratio 1:1.5) that captures users who both consumed the content depth-wise and invested meaningful reading time.

## Results Summary

### Engagement Prediction

| Metric | MLP (Val) | LSTM (Val) | Interpretation |
|--------|-----------|------------|----------------|
| AUC-ROC | 0.6951 | **0.6975** | LSTM ranks engaged users slightly better |
| F1 Score | **0.5946** | 0.5888 | MLP slightly better on F1 |
| Recall | **0.6495** | 0.6198 | MLP catches more engaged users |
| Precision | 0.5483 | **0.5608** | LSTM has fewer false positives |
| Accuracy | 0.6378 | **0.6459** | LSTM is more accurate overall |

Both models use 27 aggregate behavioral features + 5 article features + 3 context features (67 dims after embeddings). The LSTM additionally processes raw behavioral sequences (50 timesteps x 2 features) through a BiLSTM + attention encoder. Both models operate well within the expected range — the RecSys 2024 Challenge top-3 averaged 0.7643 AUC using full text embeddings, 1M+ users, and months of engineering. Our behavioral-only models exceed the MIND NRMS benchmark (0.6776 AUC with text embeddings).

### User Re-identification Attack

| Metric | LSTM (Best) | MLP (Best) | Random | LSTM Lift | MLP Lift |
|--------|------------|------------|--------|-----------|----------|
| Top-1 Accuracy | **10.43%** | 0.24% | 0.0035% | **2,958x** | 69x |
| Top-5 Accuracy | **16.03%** | 0.61% | 0.018% | **907x** | 35x |
| Top-10 Accuracy | **18.77%** | 0.95% | 0.035% | **536x** | 27x |
| Top-20 Accuracy | **21.82%** | 1.46% | 0.071% | **307x** | 21x |
| MRR | **0.1334** | 0.0058 | 0.0004 | **334x** | 15x |
| Median Rank | **1,126** | 3,121 | ~14,180 | -- | -- |
| Users 100% identifiable | **220** | 2 | 0 | -- | -- |

**Key finding**: The LSTM re-identifies users at **2,958x above random** (10.43% Top-1 among 28,361 users) — **43.1x stronger** than the MLP (69x lift). A mere +0.24% AUC improvement produces 43x more identifiable representations, demonstrating the extreme non-linearity between model quality and privacy risk.

### Blind Test Validation (10K unseen users, zero overlap)

| Experiment | LSTM | MLP | Implication |
|-----------|------|-----|-------------|
| Engagement generalization | AUC **0.7058** | AUC **0.7030** | Both models generalize well to unseen users |
| Within-blind re-identification | **841x** above random (9.13% Top-1) | **24x** above random (0.26% Top-1) | LSTM fingerprinting is 35x stronger, both feature-level |
| Cross-dataset negative control | **0.0000%** Top-1 | **0.0000%** Top-1 | Attack is methodologically sound for both |

The blind test proves that both models create transferable behavioral fingerprints — the privacy risk is inherent to the feature representation, not an artifact of training exposure. The LSTM's temporal processing amplifies the fingerprinting signal by **35x** over the MLP on completely unseen users, confirming that **privacy risk scales dramatically with model sophistication**.

## Project Structure

```
Privacy-Preserving-Reader-Engagement-Prediction/
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
│   ├── ebnerd_blind_test/           # 10K-user blind test (zero overlap with 50K)
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
│   ├── 01b_create_blind_test.py     # Script that built ebnerd_blind_test (10K, zero overlap)
│   ├── 02_train_mlp.py              # MLP training + representation extraction
│   ├── 03_reidentification_test.py  # Re-identification attack on 50K val users (MLP)
│   ├── 03b_blind_test_evaluation.py # Blind test: engagement + re-id on 10K unseen users
│   ├── 04_train_lstm.py             # LSTM training + representation extraction
│   ├── 05_lstm_reidentification.py  # LSTM re-identification attack + MLP comparison
│   ├── 05b_lstm_blind_test.py       # Blind test: engagement + re-id on 10K unseen users (LSTM)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Load parquets, engineer features, normalize
│   │   └── dataset.py               # PyTorch Dataset + DataLoader
│   └── models/
│       ├── __init__.py
│       ├── mlp_baseline.py          # Deep MLP engagement model (210K params)
│       ├── lstm_model.py            # BiLSTM + Attention model (~1M params)
│       ├── train.py                 # Training infrastructure, losses, plotting
│       ├── attack.py                # GPU-accelerated nearest-neighbor re-identification attack
│       └── smoothing.py             # Randomized smoothing: analytical + MC + aggregation + SNR (Phase 4)
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
│       │   ├── reidentification_results.json
│       │   └── blind_test_results.json
│       └── smoothing/               # Randomized smoothing outputs (Phase 4)
│           ├── smoothing_results_v2.json
│           ├── comparison/          # Cross-model comparison plots
│           │   ├── privacy_utility_tradeoff.png  (main deliverable)
│           │   ├── pareto_frontier.png
│           │   ├── reid_decay.png, auc_degradation.png
│           │   ├── certification_coverage.png
│           │   └── smoothing_summary.png
│           ├── mlp/                 # MLP-specific plots
│           │   ├── certified_radii.png
│           │   ├── recommended_sigma_detail.png
│           │   ├── snr_analysis.png
│           │   └── aggregation_surface.png
│           └── lstm/                # LSTM-specific plots
│               ├── certified_radii.png
│               ├── recommended_sigma_detail.png
│               ├── snr_analysis.png
│               └── aggregation_surface.png
└── ebnerd-benchmark/                # Cloned EB-NeRD benchmark repo (reference)
```

## Methodology

### Phase 1: Data & EDA (Complete)
- Constructed 50K-user dataset preserving EB-NeRD temporal splits (seed=42 for reproducibility)
- Analyzed engagement signal distributions, user history characteristics, and behavioral uniqueness
- Confirmed 100% fingerprint uniqueness — strong motivation for privacy mechanisms
- Enhanced nearest-neighbor analysis: median 1-NN distance of 0.314, 84% of users within 0.5 std of closest neighbor
- Executed in 3.5s on 24-core CPU with Polars 1.38 (Windows 11, CUDA 12.8 verified)
- Documentation: [docs/01_eda_analysis.md](docs/01_eda_analysis.md)

### Phase 2: Data Pipeline (Complete)
- Built preprocessing pipeline: label creation, history extraction, feature engineering, normalization
- 546K train / 568K val samples, 40% positive rate, no NaN/Inf, no train-to-val leakage
- History sequences (50 timesteps x 2 features), 27 aggregate features, 7 article features, 3 context features
- PyTorch Dataset and DataLoader with custom collation (12.5 ms/batch, 1,067 batches/epoch)
- 11 verification audits all passed (labels, cleanliness, leakage, padding, batch format)
- Executed in 302s on 24-core CPU; output: 595 MB processed data
- Documentation: [docs/02_data_pipeline.md](docs/02_data_pipeline.md)

### Phase 3A: MLP Baseline + Re-identification (Complete)
- 6-layer deep MLP (210K params) with SiLU activations and residual connections
- All 27 aggregate features + 5 article continuous + 3 context (67 input dims after embeddings)
- LabelSmoothingBCE loss with pos_weight for class balance
- AUC 0.6951, F1 0.5946, recall 0.6495, precision 0.5483
- 64-dim user representations extracted for privacy analysis
- Re-identification attack: 69x above random (0.24% Top-1 among 28K users)
- Blind test (10K unseen users): AUC 0.7030, within-blind re-id 24x above random, cross-dataset 0.0%
- Proves fingerprinting is feature-level (not memorization) and model generalizes to unseen users
- Trained in 25.7 min on RTX 5070 Ti (CUDA 12.8); early stopped at epoch 24, best at epoch 16
- Documentation: [docs/03_mlp_baseline_analysis.md](docs/03_mlp_baseline_analysis.md)

### Phase 3B: LSTM Model + Re-identification (Complete)
- 2-layer BiLSTM (1.03M params) with 4-head multi-head self-attention pooling
- Consumes raw behavioral sequences (50 timesteps x 2) + 27 aggregate features + 7 article features + 3 context features
- Addresses the remaining MLP information gap: temporal patterns and sequential behavioral signatures
- AUC 0.6975, F1 0.5888, recall 0.6198, precision 0.5608
- Re-identification attack: **2,958x** above random (10.43% Top-1 among 28K users) — **43.1x stronger than MLP**
- Blind test (10K unseen users): AUC 0.7058, within-blind re-id **841x** above random, cross-dataset 0.0%
- Proves temporal sequences amplify fingerprinting by 35x over aggregate features on unseen users
- Trained in 62.2 min on RTX 5070 Ti (CUDA 12.8); early stopped at epoch 26, best at epoch 18
- Documentation: [docs/04_lstm_analysis.md](docs/04_lstm_analysis.md)

### Phase 4: Randomized Smoothing for Privacy (Code Complete — Awaiting Execution)
- Add calibrated Gaussian noise ε ~ N(0, σ²I₆₄) to 64-dim learned representations
- Analytical smoothed prediction: P = Φ(logit / (σ·‖w‖)) — exact for linear classification head
- **Tautology audit**: Analytical AUC is a monotonic transform (preserves rankings by construction) — replaced with Monte Carlo noise injection for honest utility measurement
- **Dual-evaluation framework**: Analytical AUC as upper bound + MC AUC (100-trial averaged) as deployment-realistic metric
- Certified radius R = σ · Φ⁻¹(p_A) guarantees prediction stability (Cohen et al., ICML 2019)
- Monte Carlo verification with Clopper-Pearson confidence bounds (α = 0.001)
- Sweep 11 noise levels (σ = 0 to 3.0) mapping the full privacy-utility tradeoff
- **Aggregation tradeoff**: (σ × M) surface showing utility vs privacy coupling across multi-draw scenarios
- **GPU-accelerated re-identification**: Full CUDA pipeline — pairwise distance (batched normalized matmul for cosine, `torch.cdist` for euclidean), `torch.argsort` for ranking, and vectorized broadcast rank-finding. ~15× end-to-end speedup per attack (~9s vs ~128s), enabling ~460 attacks per run in ~1.5 hours
- Dual-metric NN distances (cosine for re-id, euclidean for L2 certification comparison)
- Compare MLP vs LSTM: quantify how much more noise the LSTM needs for equivalent privacy
- 10 visualization plots including privacy-utility Pareto frontier, aggregation surface, and SNR analysis (main deliverables)
- Documentation: [docs/05_randomized_smoothing.md](docs/05_randomized_smoothing.md)

## Setup

### Requirements
- Python 3.11+
- NVIDIA GPU with CUDA 12.x (tested: RTX 5070 Ti, 12 GB VRAM)
- 16+ GB RAM recommended (64 GB used in development)

### Tested Environment (Windows 11)

| Component | Value |
|-----------|-------|
| OS | Windows 11 Home (Build 26200) |
| Python | 3.11.14 |
| PyTorch | 2.10.0+cu128 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB, CUDA 12.8) |
| CPU | Intel Core Ultra 9 275HX (24 cores) |
| Polars | 1.38.1 |
| RAM | 64 GB |

### Installation

```bash
# Create conda environment (recommended for CUDA support)
conda create -n privacy python=3.11 -y
conda activate privacy

# Install PyTorch with CUDA (Windows/Linux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install -r requirements.txt

# Install benchmark utilities
pip install -e ebnerd-benchmark/
```

### Reproducing the Dataset

If you need to rebuild `ebnerd_50k` from scratch:

1. Download `ebnerd_small` (80MB) and `ebnerd_large` (3.0GB) from [recsys.eb.dk](https://recsys.eb.dk/) into `data/`
2. Extract both zip files into `data/ebnerd_small/` and `data/ebnerd_large/`
3. Run: `python src/01_create_50k_dataset.py`
4. Delete `ebnerd_large` and the zip files to free disk space

### Running the Full Pipeline

```bash
conda activate privacy

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

# Step 5b: Blind test evaluation (engagement + re-id on 10K unseen users)
python src/03b_blind_test_evaluation.py

# Step 6: Train LSTM (outputs to outputs/models/lstm/)
python src/04_train_lstm.py

# Step 7: LSTM re-identification + MLP comparison
python src/05_lstm_reidentification.py

# Step 7b: LSTM blind test evaluation (engagement + re-id on 10K unseen users)
python src/05b_lstm_blind_test.py

# Step 8: Randomized smoothing privacy defense (Phase 4)
python src/06_randomized_smoothing.py
```

## Key References

- **EB-NeRD Dataset**: [arxiv.org/abs/2410.03432](https://arxiv.org/abs/2410.03432)
- **Randomized Smoothing**: Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (ICML 2019)
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
