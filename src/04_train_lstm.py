"""
04_train_lstm.py — Train the BiLSTM + Attention engagement predictor (v2)
==========================================================================
Loads preprocessed data, trains LSTMEngagementModel with LabelSmoothingBCE,
saves checkpoint + metrics + representations + comprehensive plots.

Key differences from MLP training:
  - Uses history_seq (B, 50, 2) sequences + agg_features (B, 27) as context
  - Batch size 512, LR 8e-4, pct_start 0.3 (faster warmup)
  - skip_train_eval=False (full train eval each epoch for monitoring)
  - Masking instead of pack_padded_sequence for MPS compatibility

v2 changes:
  - Optimized hyperparameters (batch=512, lr=8e-4, weight_decay=3e-4)
  - OneCycleLR pct_start=0.3 (reach peak LR by epoch ~3)
  - LSTM dropout 0.15, max_epochs 30, patience 7
  - num_workers=2 for parallel data loading
  - Article features expanded to 5 continuous dims (body/title/subtitle length)
  - Aggregate features (27-dim) fed to LSTM context branch

Usage:
    source .venv/bin/activate
    python src/04_train_lstm.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import get_dataloaders, load_metadata
from models.lstm_model import LSTMEngagementModel
from models.train import (
    LabelSmoothingBCE,
    fit,
    extract_representations,
    get_device,
    plot_representation_analysis,
)


def main():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "lstm"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Phase 3B: BiLSTM + Attention Training (v2 — optimized)")
    print("=" * 70)

    # ---- Load data ----
    print("\nLoading data...")
    metadata = load_metadata(PROCESSED_DIR)
    train_loader, val_loader, _ = get_dataloaders(
        PROCESSED_DIR, batch_size=512, num_workers=2, pin_memory=True,
    )
    print(f"  Train: {len(train_loader.dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Val:   {len(val_loader.dataset):,} samples, {len(val_loader):,} batches")
    print(f"  Pos rate: train={metadata['train_pos_rate']:.3f}, val={metadata['val_pos_rate']:.3f}")

    # ---- Initialize model ----
    model = LSTMEngagementModel(
        n_categories=metadata["n_categories"],
        n_article_types=metadata["n_article_types"],
        article_cont_dim=metadata.get("article_cont_dim", 5),
        agg_dim=metadata.get("agg_feature_dim", 27),
        lstm_dropout=0.15,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: LSTMEngagementModel v2 ({total_params:,} parameters)")
    print(f"  Device: {get_device()}")

    # ---- Label Smoothing BCE with pos_weight for class balance ----
    pos_rate = metadata["train_pos_rate"]
    pos_weight = (1 - pos_rate) / pos_rate
    criterion = LabelSmoothingBCE(pos_weight=pos_weight, smoothing=0.05)
    print(f"  Loss: LabelSmoothingBCE(pos_weight={pos_weight:.3f}, smoothing=0.05)")

    # ---- Train ----
    start_time = time.time()
    results = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=OUTPUT_DIR,
        criterion=criterion,
        lr=8e-4,
        weight_decay=3e-4,
        max_epochs=30,
        patience=7,
        max_grad_norm=1.0,
        pct_start=0.3,
        skip_train_eval=False,
    )
    train_time = time.time() - start_time

    # ---- Extract representations ----
    print("\n  Extracting user representations...")
    device = get_device()

    checkpoint = torch.load(OUTPUT_DIR / "checkpoint.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    val_user_ids, val_reprs, val_labels = extract_representations(model, val_loader, device)
    train_user_ids, train_reprs, train_labels = extract_representations(model, train_loader, device)

    np.savez(
        OUTPUT_DIR / "representations.npz",
        val_user_ids=val_user_ids,
        val_representations=val_reprs,
        val_labels=val_labels,
        train_user_ids=train_user_ids,
        train_representations=train_reprs,
        train_labels=train_labels,
    )
    print(f"    Val representations:   {val_reprs.shape}")
    print(f"    Train representations: {train_reprs.shape}")
    print(f"    Saved to {OUTPUT_DIR / 'representations.npz'}")

    # ---- Representation quality check ----
    unique_val_users = np.unique(val_user_ids)
    print(f"\n  Representation quality (val set):")
    print(f"    Unique users: {len(unique_val_users):,}")
    print(f"    Repr dim: {val_reprs.shape[1]}")
    print(f"    Repr mean: {val_reprs.mean():.4f}")
    print(f"    Repr std:  {val_reprs.std():.4f}")
    print(f"    Repr min:  {val_reprs.min():.4f}")
    print(f"    Repr max:  {val_reprs.max():.4f}")

    # Check for dead dimensions
    dim_stds = val_reprs.std(axis=0)
    n_dead = (dim_stds < 0.01).sum()
    n_low_var = (dim_stds < 0.05).sum()
    print(f"    Dead dims (std < 0.01): {n_dead} / {val_reprs.shape[1]}")
    print(f"    Low-var dims (std < 0.05): {n_low_var} / {val_reprs.shape[1]}")

    # ---- Representation visualization ----
    print(f"\n  Generating representation analysis plots...")
    plot_representation_analysis(
        val_user_ids, val_reprs, val_labels, OUTPUT_DIR,
        model_name="LSTMEngagementModel",
    )

    # ---- Final summary ----
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time:       {train_time:.1f}s ({train_time/60:.1f} min)")
    print(f"  Best epoch:       {results['best_epoch']}")
    print(f"  Best val AUC:     {results['best_val_auc']:.4f}")
    print(f"  Val F1:           {results['final_val_metrics']['f1']:.4f}")
    print(f"  Val accuracy:     {results['final_val_metrics']['accuracy']:.4f}")
    print(f"  Val avg prec:     {results['final_val_metrics']['avg_precision']:.4f}")

    if 'auc' in results['final_train_metrics']:
        train_val_gap = results['final_train_metrics']['auc'] - results['final_val_metrics']['auc']
        print(f"  Train-val AUC gap:{train_val_gap:+.4f}")

    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print(f"    checkpoint.pt            — best model weights")
    print(f"    metrics.json             — per-epoch training history + config")
    print(f"    representations.npz      — user representations for privacy experiments")
    print(f"    training_curves.png      — loss, AUC, metrics over epochs")
    print(f"    evaluation_plots.png     — ROC, PR, confusion matrix, prob distribution")
    print(f"    representation_analysis.png — t-SNE, norm distribution, per-dim activations")


if __name__ == "__main__":
    main()
