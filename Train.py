"""
src/train.py
Defines the CNN architecture and training loop for lung/colon cancer classification.
"""

import os
import time
import copy
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils import CLASS_LABELS, CLASS_DESCRIPTIONS, CLASS_NAMES

logger = logging.getLogger(__name__)
NUM_CLASSES = len(CLASS_LABELS)


# ─────────────────────────────────────────────
#  MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class CancerCNN(nn.Module):
    """
    Transfer-learning CNN based on EfficientNet-B0 with a custom classifier head.
    EfficientNet offers excellent accuracy / parameter efficiency for medical imaging.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Freeze early layers, fine-tune last 3 blocks
        for name, param in backbone.named_parameters():
            param.requires_grad = False
        for name, param in backbone.named_parameters():
            if any(f"features.{i}" in name for i in [6, 7, 8]):
                param.requires_grad = True

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()  # remove original head

        self.backbone   = backbone
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


def get_model(device: torch.device, **kwargs) -> CancerCNN:
    model = CancerCNN(**kwargs)
    return model.to(device)


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds   = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total   = 0

    for images, labels in tqdm(loader, desc="  Val  ", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds   = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return running_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    epochs:       int  = 20,
    lr:           float = 1e-3,
    weight_decay: float = 1e-4,
    patience:     int  = 5,
    save_path:    str  = "model/lung_colon_cnn.pth",
) -> Dict:
    """
    Full training loop with:
      - Mixed-precision training (AMP)
      - CosineAnnealing LR scheduler
      - Early stopping
      - Best-model checkpoint saving
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc   = 0.0
    best_state     = None
    epochs_no_imp  = 0

    logger.info("🚀 Starting training — %d epochs | device: %s", epochs, device)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %3d/%d | Train Loss %.4f Acc %.4f | Val Loss %.4f Acc %.4f | LR %.2e | %.1fs",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            optimizer.param_groups[0]["lr"], elapsed
        )

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_state    = copy.deepcopy(model.state_dict())
            epochs_no_imp = 0
            torch.save({"model_state": best_state, "epoch": epoch, "val_acc": val_acc}, save_path)
            logger.info("   ✅ New best model saved (val_acc=%.4f)", val_acc)
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= patience:
                logger.info("   ⏹  Early stopping at epoch %d", epoch)
                break

    if best_state:
        model.load_state_dict(best_state)

    return history


# ─────────────────────────────────────────────
#  EVALUATION HELPERS
# ─────────────────────────────────────────────

@torch.no_grad()
def full_evaluation(model: nn.Module, test_loader: DataLoader, device: torch.device) -> Dict:
    """Collect predictions + probabilities on the test set."""
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        logits = model(images)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    target_names = [CLASS_DESCRIPTIONS[CLASS_NAMES[i]] for i in range(NUM_CLASSES)]
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    cm     = confusion_matrix(all_labels, all_preds)

    try:
        auc = roc_auc_score(
            np.eye(NUM_CLASSES)[all_labels], all_probs, multi_class="ovr", average="macro"
        )
    except Exception:
        auc = None

    accuracy = (all_preds == all_labels).mean()
    logger.info("Test Accuracy: %.4f | Macro AUC: %s", accuracy, f"{auc:.4f}" if auc else "N/A")

    return {
        "accuracy":   accuracy,
        "auc":        auc,
        "report":     report,
        "confusion":  cm,
        "preds":      all_preds,
        "labels":     all_labels,
        "probs":      all_probs,
    }


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=2, linestyle="--")
    axes[0].set_title("Loss por Época", fontweight="bold")
    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   linewidth=2, linestyle="--")
    axes[1].set_title("Accuracy por Época", fontweight="bold")
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[str] = None):
    labels = [CLASS_DESCRIPTIONS[CLASS_NAMES[i]] for i in range(NUM_CLASSES)]
    # Shorten labels for readability
    short = [l.split(" (")[0] for l in labels]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short, yticklabels=short, ax=ax
    )
    ax.set_title("Matriz de Confusión — Test Set", fontsize=14, fontweight="bold")
    ax.set_ylabel("Etiqueta Real"); ax.set_xlabel("Predicción")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig