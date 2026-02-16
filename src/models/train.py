"""
train.py — Shared training infrastructure for engagement models
================================================================
Provides:
  - FocalLoss: Better loss for imbalanced binary classification
  - train_epoch(): Single training epoch
  - evaluate(): Full evaluation with AUC, F1, accuracy, precision, recall
  - fit(): Complete training loop with early stopping, LR scheduling, checkpointing
  - extract_representations(): Get user representations for privacy experiments
  - plot_training_curves(): Comprehensive training visualization
"""

import json
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification (Lin et al., 2017).

    Addresses class imbalance by down-weighting easy examples and focusing
    the loss on hard misclassified examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Balancing factor for positive class. Set to pos_rate to
               weight the minority class higher. Default 0.4 (our pos rate).
        gamma: Focusing parameter. gamma=0 recovers BCE. Higher gamma
               focuses more on hard examples. Default 1.0.
               (2.0 is standard for extreme imbalance like object detection,
                but 1.0 is better for mild imbalance like 40:60.)
    """

    def __init__(self, alpha: float = 0.4, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


class LabelSmoothingBCE(nn.Module):
    """
    BCE with label smoothing and class weighting.

    Smooths hard 0/1 targets to [smoothing, 1-smoothing] to prevent
    overconfident predictions and improve calibration.
    Applies pos_weight to upweight the minority positive class.

    Args:
        pos_weight: Weight for positive class. (1-pos_rate)/pos_rate.
        smoothing: Label smoothing factor. Default 0.05.
    """

    def __init__(self, pos_weight: float = 1.0, smoothing: float = 0.05):
        super().__init__()
        self.smoothing = smoothing
        self.register_buffer("pos_weight", torch.tensor([pos_weight]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        smoothed = targets * (1 - self.smoothing) + (1 - targets) * self.smoothing
        return F.binary_cross_entropy_with_logits(
            logits, smoothed,
            pos_weight=self.pos_weight.to(logits.device),
        )


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch dict to the target device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scheduler: Optional[OneCycleLR] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Train for one epoch.

    Returns:
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()

        logits = model(batch)
        loss = criterion(logits, batch["label"])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset.

    Returns:
        Dict with: loss, auc, f1, accuracy, precision, recall, avg_precision,
        plus raw arrays (probs, labels) for plotting.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels = []
    all_probs = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        logits = model(batch)
        loss = criterion(logits, batch["label"])

        probs = torch.sigmoid(logits).cpu().numpy()
        labels = batch["label"].cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)
        total_loss += loss.item()
        n_batches += 1

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_preds = (all_probs >= 0.5).astype(np.float32)

    metrics = {
        "loss": total_loss / n_batches,
        "auc": float(roc_auc_score(all_labels, all_probs)),
        "f1": float(f1_score(all_labels, all_preds)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        "avg_precision": float(average_precision_score(all_labels, all_probs)),
        "_probs": all_probs,
        "_labels": all_labels,
    }

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, output_dir: Path, model_name: str = "MLP"):
    """
    Generate comprehensive training visualization plots.

    Creates a 2x3 grid:
      1. Loss curve (train + val)
      2. Val AUC curve
      3. Val F1 / Accuracy / Precision / Recall curves
      4. Learning rate schedule (from val loss proxy)
      5. Train-Val gap (overfitting monitor)
      6. Val loss zoomed (last 50% of training)
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{model_name} — Training Curves", fontsize=16, fontweight="bold")

    epochs = range(1, len(history["train"]) + 1)
    train_losses = [h["loss"] for h in history["train"]]
    val_losses = [h["loss"] for h in history["val"]]
    val_aucs = [h["auc"] for h in history["val"]]
    val_f1s = [h["f1"] for h in history["val"]]
    val_accs = [h["accuracy"] for h in history["val"]]
    val_precs = [h["precision"] for h in history["val"]]
    val_recs = [h["recall"] for h in history["val"]]

    # 1. Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, "b-o", markersize=3, label="Train Loss")
    ax.plot(epochs, val_losses, "r-o", markersize=3, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Val AUC
    ax = axes[0, 1]
    ax.plot(epochs, val_aucs, "g-o", markersize=3, label="Val AUC")
    best_epoch = np.argmax(val_aucs) + 1
    best_auc = max(val_aucs)
    ax.axhline(y=best_auc, color="g", linestyle="--", alpha=0.5)
    ax.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.5)
    ax.annotate(f"Best: {best_auc:.4f} (ep {best_epoch})",
                xy=(best_epoch, best_auc), fontsize=9,
                xytext=(best_epoch + 1, best_auc - 0.005),
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Validation AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. All val metrics
    ax = axes[0, 2]
    ax.plot(epochs, val_f1s, "-o", markersize=3, label="F1")
    ax.plot(epochs, val_accs, "-s", markersize=3, label="Accuracy")
    ax.plot(epochs, val_precs, "-^", markersize=3, label="Precision")
    ax.plot(epochs, val_recs, "-d", markersize=3, label="Recall")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Train-Val AUC gap (overfitting monitor)
    ax = axes[1, 0]
    train_aucs_est = [h.get("auc", None) for h in history["train"]]
    if train_aucs_est[0] is not None:
        gaps = [t - v for t, v in zip(train_aucs_est, val_aucs)]
        ax.plot(epochs, gaps, "m-o", markersize=3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(epochs, 0, gaps, alpha=0.2, color="magenta")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train AUC - Val AUC")
    ax.set_title("Overfitting Gap")
    ax.grid(True, alpha=0.3)

    # 5. Loss delta per epoch
    ax = axes[1, 1]
    val_deltas = [val_losses[i] - val_losses[i-1] for i in range(1, len(val_losses))]
    colors = ["green" if d < 0 else "red" for d in val_deltas]
    ax.bar(range(2, len(val_losses) + 1), val_deltas, color=colors, alpha=0.7)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss Change")
    ax.set_title("Val Loss Delta (green=improving)")
    ax.grid(True, alpha=0.3)

    # 6. Precision-Recall tradeoff over epochs
    ax = axes[1, 2]
    ax.plot(val_recs, val_precs, "b-o", markersize=4)
    for i, ep in enumerate(epochs):
        if i == 0 or i == len(epochs) - 1 or i == np.argmax(val_aucs):
            ax.annotate(f"ep{ep}", (val_recs[i], val_precs[i]), fontsize=7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Recall (epoch trajectory)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved training_curves.png")


def plot_evaluation(
    val_probs: np.ndarray,
    val_labels: np.ndarray,
    output_dir: Path,
    model_name: str = "MLP",
):
    """
    Generate evaluation plots for the best model.

    Creates a 2x2 grid:
      1. ROC curve with AUC
      2. Precision-Recall curve with AP
      3. Confusion matrix
      4. Prediction probability distribution
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{model_name} — Evaluation Results", fontsize=16, fontweight="bold")

    val_preds = (val_probs >= 0.5).astype(np.float32)

    # 1. ROC curve
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(val_labels, val_probs)
    auc_score = roc_auc_score(val_labels, val_probs)
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUC = {auc_score:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # 2. Precision-Recall curve
    ax = axes[0, 1]
    prec_curve, rec_curve, _ = precision_recall_curve(val_labels, val_probs)
    ap_score = average_precision_score(val_labels, val_probs)
    ax.plot(rec_curve, prec_curve, "r-", linewidth=2, label=f"AP = {ap_score:.4f}")
    baseline_rate = val_labels.mean()
    ax.axhline(y=baseline_rate, color="gray", linestyle="--", alpha=0.5, label=f"Baseline = {baseline_rate:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 3. Confusion matrix
    ax = axes[1, 0]
    cm = confusion_matrix(val_labels, val_preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i,j]:,}\n({cm_normalized[i,j]:.1%})",
                    ha="center", va="center", fontsize=11, color=text_color)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Engaged", "Engaged"])
    ax.set_yticklabels(["Not Engaged", "Engaged"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. Prediction probability distribution
    ax = axes[1, 1]
    ax.hist(val_probs[val_labels == 0], bins=50, alpha=0.6, density=True,
            label="Not Engaged", color="blue")
    ax.hist(val_probs[val_labels == 1], bins=50, alpha=0.6, density=True,
            label="Engaged", color="red")
    ax.axvline(x=0.5, color="black", linestyle="--", alpha=0.7, label="Threshold (0.5)")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Distribution by Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "evaluation_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved evaluation_plots.png")


def plot_representation_analysis(
    user_ids: np.ndarray,
    representations: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    model_name: str = "MLP",
    n_tsne: int = 5000,
):
    """
    Visualize the learned representation space.

    Creates a 1x3 grid:
      1. t-SNE colored by engagement label
      2. Representation norm distribution
      3. Top feature activations histogram
    """
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} — Representation Analysis (64-dim)", fontsize=14, fontweight="bold")

    # Subsample for t-SNE
    n = min(n_tsne, len(representations))
    idx = np.random.RandomState(42).choice(len(representations), n, replace=False)
    sub_repr = representations[idx]
    sub_labels = labels[idx]

    # 1. t-SNE
    ax = axes[0]
    print(f"    Running t-SNE on {n:,} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(sub_repr)

    scatter = ax.scatter(emb_2d[sub_labels == 0, 0], emb_2d[sub_labels == 0, 1],
                         c="blue", alpha=0.3, s=5, label="Not Engaged")
    ax.scatter(emb_2d[sub_labels == 1, 0], emb_2d[sub_labels == 1, 1],
               c="red", alpha=0.3, s=5, label="Engaged")
    ax.set_title("t-SNE of Representations")
    ax.legend(markerscale=3)
    ax.set_xticks([])
    ax.set_yticks([])

    # 2. Representation L2 norm distribution
    ax = axes[1]
    norms = np.linalg.norm(representations, axis=1)
    norms_pos = norms[labels.flatten() == 1] if labels.ndim > 1 else norms[labels == 1]
    norms_neg = norms[labels.flatten() == 0] if labels.ndim > 1 else norms[labels == 0]
    ax.hist(norms_neg, bins=50, alpha=0.6, density=True, label="Not Engaged", color="blue")
    ax.hist(norms_pos, bins=50, alpha=0.6, density=True, label="Engaged", color="red")
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title("Representation Norm by Class")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mean activation per dimension
    ax = axes[2]
    mean_pos = representations[labels == 1].mean(axis=0)
    mean_neg = representations[labels == 0].mean(axis=0)
    x = np.arange(representations.shape[1])
    width = 0.35
    ax.bar(x - width/2, mean_pos, width, alpha=0.7, label="Engaged", color="red")
    ax.bar(x + width/2, mean_neg, width, alpha=0.7, label="Not Engaged", color="blue")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Mean Activation")
    ax.set_title("Per-Dimension Mean Activation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "representation_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved representation_analysis.png")


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    criterion: Optional[nn.Module] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 50,
    patience: int = 8,
    max_grad_norm: float = 1.0,
    device: Optional[torch.device] = None,
    pct_start: float = 0.1,
    skip_train_eval: bool = False,
) -> dict:
    """
    Full training loop with early stopping, OneCycleLR scheduling, and checkpointing.

    Args:
        model: The model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        output_dir: Where to save checkpoints and metrics
        criterion: Loss function. Defaults to FocalLoss(alpha=0.4, gamma=2.0).
        lr: Peak learning rate for OneCycleLR
        weight_decay: AdamW weight decay
        max_epochs: Maximum training epochs
        patience: Early stopping patience (epochs without val AUC improvement)
        max_grad_norm: Gradient clipping max norm
        device: Compute device (auto-detected if None)
        pct_start: Fraction of training for LR warmup (OneCycleLR). Default 0.1.
        skip_train_eval: If True, skip full train-set evaluation each epoch (saves ~33% time).

    Returns:
        Dict with best metrics and training history
    """
    if device is None:
        device = get_device()

    if criterion is None:
        criterion = FocalLoss(alpha=0.4, gamma=2.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = max_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy="cos",
    )

    # For evaluation we use the same criterion
    eval_criterion = criterion

    best_val_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = {"train": [], "val": []}

    loss_name = criterion.__class__.__name__
    print(f"\n{'='*70}")
    print(f"  Training on {device} | {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  {len(train_loader.dataset):,} train / {len(val_loader.dataset):,} val samples")
    print(f"  Loss: {loss_name} | LR: {lr} | WD: {weight_decay}")
    print(f"  Batch size: {train_loader.batch_size} | Max epochs: {max_epochs} | Patience: {patience}")
    print(f"  pct_start: {pct_start} | skip_train_eval: {skip_train_eval}")
    print(f"{'='*70}")
    print(f"\n  {'Epoch':>5s}  {'Train Loss':>10s}  {'Val Loss':>10s}  {'Val AUC':>8s}  "
          f"{'Val F1':>7s}  {'Val Acc':>7s}  {'Time':>6s}  {'Status'}")
    print(f"  {'-'*75}")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scheduler=scheduler, max_grad_norm=max_grad_norm,
        )

        # Evaluate validation set (always)
        val_metrics = evaluate(model, val_loader, eval_criterion, device)

        # Evaluate training set (optional -- expensive for large datasets)
        if skip_train_eval:
            train_record = {"loss": train_loss}
        else:
            train_metrics_full = evaluate(model, train_loader, eval_criterion, device)
            train_record = {k: v for k, v in train_metrics_full.items() if not k.startswith("_")}
            train_record["loss"] = train_loss

        val_record = {k: v for k, v in val_metrics.items() if not k.startswith("_")}

        history["train"].append(train_record)
        history["val"].append(val_record)

        elapsed = time.time() - epoch_start

        # Early stopping check
        status = ""
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_epoch = epoch
            epochs_without_improvement = 0
            status = "* best"

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_val_auc,
                "val_metrics": val_record,
            }, output_dir / "checkpoint.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                status = "early stop"

        print(f"  {epoch:5d}  {train_loss:10.4f}  {val_metrics['loss']:10.4f}  "
              f"{val_metrics['auc']:8.4f}  {val_metrics['f1']:7.4f}  "
              f"{val_metrics['accuracy']:7.4f}  {elapsed:5.1f}s  {status}")

        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best checkpoint
    checkpoint = torch.load(output_dir / "checkpoint.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\n  Loaded best model from epoch {best_epoch} (val AUC = {best_val_auc:.4f})")

    # Final evaluation on both sets
    print(f"\n  Final evaluation with best model:")
    final_train = evaluate(model, train_loader, eval_criterion, device)
    final_val = evaluate(model, val_loader, eval_criterion, device)

    print(f"    {'Metric':>12s}  {'Train':>8s}  {'Val':>8s}  {'Gap':>8s}")
    print(f"    {'-'*40}")
    for metric in ["auc", "f1", "accuracy", "precision", "recall", "avg_precision", "loss"]:
        gap = final_train[metric] - final_val[metric]
        print(f"    {metric:>12s}  {final_train[metric]:8.4f}  {final_val[metric]:8.4f}  {gap:+8.4f}")

    # Save history (strip raw arrays)
    results = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "final_train_metrics": {k: v for k, v in final_train.items() if not k.startswith("_")},
        "final_val_metrics": {k: v for k, v in final_val.items() if not k.startswith("_")},
        "history": history,
        "config": {
            "loss": loss_name,
            "lr": lr,
            "weight_decay": weight_decay,
            "max_epochs": max_epochs,
            "patience": patience,
            "pct_start": pct_start,
            "skip_train_eval": skip_train_eval,
        },
    }
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved metrics to {output_dir / 'metrics.json'}")

    # Generate plots
    print(f"\n  Generating plots...")
    plot_training_curves(history, output_dir, model_name=model.__class__.__name__)
    plot_evaluation(final_val["_probs"], final_val["_labels"], output_dir,
                    model_name=model.__class__.__name__)

    return results


@torch.no_grad()
def extract_representations(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> tuple:
    """
    Extract user representations from the model's representation layer.

    Returns:
        (user_ids, representations, labels) as numpy arrays
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    model.eval()

    all_user_ids = []
    all_reprs = []
    all_labels = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        reprs = model.get_representation(batch).cpu().numpy()
        user_ids = batch["user_id"].cpu().numpy()
        labels = batch["label"].cpu().numpy()

        all_reprs.append(reprs)
        all_user_ids.append(user_ids)
        all_labels.append(labels)

    return (
        np.concatenate(all_user_ids),
        np.concatenate(all_reprs),
        np.concatenate(all_labels),
    )
