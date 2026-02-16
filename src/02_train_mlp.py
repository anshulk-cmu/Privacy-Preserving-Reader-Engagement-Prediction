"""
02_train_mlp.py — Train the Deep MLP engagement predictor (v3)
===============================================================
Loads preprocessed data, trains MLPEngagementModel with LabelSmoothingBCE,
saves checkpoint + metrics + representations + comprehensive plots.

Architecture: 6-layer MLP with residual connections, ~207K params.
Loss: LabelSmoothingBCE with pos_weight for class balance.
50 max epochs, patience 8, full train-set evaluation each epoch.

Usage:
    source .venv/bin/activate
    python src/02_train_mlp.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import get_dataloaders, load_metadata
from models.mlp_baseline import MLPEngagementModel
from models.train import (
    LabelSmoothingBCE,
    fit,
    extract_representations,
    get_device,
    plot_representation_analysis,
)


def main():
    PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "mlp_baseline"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Phase 3A: Deep MLP Training (v3 — Label Smoothing BCE)")
    print("=" * 70)

    # ---- Load data ----
    print("\nLoading data...")
    metadata = load_metadata(PROCESSED_DIR)
    train_loader, val_loader, _ = get_dataloaders(
        PROCESSED_DIR, batch_size=512, pin_memory=False,
    )
    print(f"  Train: {len(train_loader.dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Val:   {len(val_loader.dataset):,} samples, {len(val_loader):,} batches")
    print(f"  Pos rate: train={metadata['train_pos_rate']:.3f}, val={metadata['val_pos_rate']:.3f}")

    # ---- Initialize model ----
    model = MLPEngagementModel(
        n_categories=metadata["n_categories"],
        n_article_types=metadata["n_article_types"],
        agg_dim=metadata["agg_feature_dim"],
        article_cont_dim=metadata.get("article_cont_dim", 5),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: MLPEngagementModel v3 ({total_params:,} parameters)")
    print(f"  Device: {get_device()}")

    # ---- Label Smoothing BCE with pos_weight for class balance ----
    pos_rate = metadata["train_pos_rate"]
    pos_weight = (1 - pos_rate) / pos_rate  # ~1.49 — upweight minority positive class
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
        lr=1e-3,
        weight_decay=5e-4,
        max_epochs=50,
        patience=8,
        max_grad_norm=1.0,
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

    # ---- Representation visualization ----
    print(f"\n  Generating representation analysis plots...")
    plot_representation_analysis(
        val_user_ids, val_reprs, val_labels, OUTPUT_DIR,
        model_name="MLPEngagementModel v2",
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
    print(f"  Train-val AUC gap:{results['final_train_metrics']['auc'] - results['final_val_metrics']['auc']:+.4f}")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print(f"    checkpoint.pt            — best model weights")
    print(f"    metrics.json             — per-epoch training history + config")
    print(f"    representations.npz      — user representations for privacy experiments")
    print(f"    training_curves.png      — loss, AUC, metrics over epochs")
    print(f"    evaluation_plots.png     — ROC, PR, confusion matrix, prob distribution")
    print(f"    representation_analysis.png — t-SNE, norm distribution, per-dim activations")


if __name__ == "__main__":
    main()
