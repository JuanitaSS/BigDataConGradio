"""
src/utils.py
ETL pipeline: data extraction, cleaning, transformation, and loading
for the Lung & Colon Cancer Histopathological Images dataset.
"""

import os
import shutil
import logging
import hashlib
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  CLASS LABELS
# ─────────────────────────────────────────────
CLASS_LABELS = {
    "colon_aca": 0,   # Colon Adenocarcinoma
    "colon_n":   1,   # Colon Benign Tissue
    "lung_aca":  2,   # Lung Adenocarcinoma
    "lung_n":    3,   # Lung Benign Tissue
    "lung_scc":  4,   # Lung Squamous Cell Carcinoma
}

CLASS_NAMES = {v: k for k, v in CLASS_LABELS.items()}

CLASS_DESCRIPTIONS = {
    "colon_aca": "Adenocarcinoma de Colon (Maligno)",
    "colon_n":   "Tejido Benigno de Colon",
    "lung_aca":  "Adenocarcinoma de Pulmón (Maligno)",
    "lung_n":    "Tejido Benigno de Pulmón",
    "lung_scc":  "Carcinoma de Células Escamosas de Pulmón (Maligno)",
}

MALIGNANT_CLASSES = {"colon_aca", "lung_aca", "lung_scc"}

# ─────────────────────────────────────────────
#  ETL — EXTRACT
# ─────────────────────────────────────────────

def extract_dataset(data_root: str) -> pd.DataFrame:
    """
    EXTRACT: Walk the dataset directory tree and build a raw inventory DataFrame.
    Expected structure:
        data_root/
          lung_colon_image_set/
            colon_image_sets/
              colon_aca/  colon_n/
            lung_image_sets/
              lung_aca/   lung_n/   lung_scc/
    """
    logger.info("🔍 EXTRACT — scanning dataset at: %s", data_root)
    records = []

    data_path = Path(data_root)
    if not data_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    for img_path in data_path.rglob("*.jpeg"):
        label = img_path.parent.name
        records.append({
            "path":     str(img_path),
            "filename": img_path.name,
            "label":    label,
            "organ":    "lung" if "lung" in label else "colon",
        })

    for img_path in data_path.rglob("*.jpg"):
        label = img_path.parent.name
        records.append({
            "path":     str(img_path),
            "filename": img_path.name,
            "label":    label,
            "organ":    "lung" if "lung" in label else "colon",
        })

    df = pd.DataFrame(records)
    logger.info("   Found %d images across %d classes", len(df), df["label"].nunique())
    return df


# ─────────────────────────────────────────────
#  ETL — TRANSFORM / CLEAN
# ─────────────────────────────────────────────

def _compute_hash(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def _is_valid_image(path: str, min_size: int = 32) -> Tuple[bool, str]:
    """Validate that a file is a loadable, non-corrupt, sufficiently large image."""
    try:
        img = Image.open(path)
        img.verify()          # catches corrupt headers
        img = Image.open(path)  # re-open after verify
        w, h = img.size
        if w < min_size or h < min_size:
            return False, f"too small ({w}x{h})"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def clean_dataset(df: pd.DataFrame, min_size: int = 32) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    TRANSFORM — Clean the raw inventory:
      1. Remove rows with unknown labels
      2. Validate image files (corrupt, truncated, too small)
      3. Remove exact duplicates by MD5 hash
      4. Encode numeric labels
    Returns (clean_df, removed_df).
    """
    logger.info("🧹 CLEAN — starting with %d records", len(df))
    removed_records = []

    # 1. Unknown labels
    known = set(CLASS_LABELS.keys())
    unknown_mask = ~df["label"].isin(known)
    if unknown_mask.any():
        removed_records.append(df[unknown_mask].assign(reason="unknown_label"))
        df = df[~unknown_mask].copy()
        logger.info("   Removed %d rows with unknown labels", unknown_mask.sum())

    # 2. Image validation
    validity = [_is_valid_image(p, min_size) for p in tqdm(df["path"], desc="Validating images")]
    valid_flags, reasons = zip(*validity) if validity else ([], [])
    df["_valid"]  = valid_flags
    df["_reason"] = reasons
    invalid_mask = ~df["_valid"]
    if invalid_mask.any():
        removed_records.append(df[invalid_mask].assign(reason=df.loc[invalid_mask, "_reason"]))
        df = df[df["_valid"]].copy()
        logger.info("   Removed %d invalid/corrupt images", invalid_mask.sum())
    df.drop(columns=["_valid", "_reason"], inplace=True)

    # 3. Duplicate removal
    logger.info("   Computing image hashes for deduplication…")
    df["hash"] = [_compute_hash(p) for p in tqdm(df["path"], desc="Hashing")]
    dup_mask = df.duplicated(subset="hash", keep="first")
    if dup_mask.any():
        removed_records.append(df[dup_mask].assign(reason="duplicate"))
        df = df[~dup_mask].copy()
        logger.info("   Removed %d duplicate images", dup_mask.sum())
    df.drop(columns=["hash"], inplace=True)

    # 4. Encode labels
    df["label_idx"] = df["label"].map(CLASS_LABELS)
    df["description"] = df["label"].map(CLASS_DESCRIPTIONS)
    df["is_malignant"] = df["label"].isin(MALIGNANT_CLASSES)

    removed_df = pd.concat(removed_records, ignore_index=True) if removed_records else pd.DataFrame()
    logger.info("✅ CLEAN complete — %d usable images, %d removed", len(df), len(removed_df))
    return df.reset_index(drop=True), removed_df


# ─────────────────────────────────────────────
#  ETL — LOAD (splits + DataLoaders)
# ─────────────────────────────────────────────

IMG_SIZE    = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

inference_transform = val_transform


class HistoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_idx"])


def create_dataloaders(
    df: pd.DataFrame,
    batch_size:  int = 32,
    num_workers: int = 4,
    val_size:    float = 0.15,
    test_size:   float = 0.15,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    LOAD — Split clean DataFrame into train/val/test and return DataLoaders.
    Stratified by label to preserve class balance.
    """
    logger.info("📦 LOAD — creating stratified train/val/test splits")

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label_idx"], random_state=random_state
    )
    relative_val = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val, stratify=train_val_df["label_idx"], random_state=random_state
    )

    logger.info("   Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    train_loader = DataLoader(
        HistoDataset(train_df, train_transform),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        HistoDataset(val_df, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        HistoDataset(test_df, val_transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, train_df, val_df, test_df


def run_etl_pipeline(data_root: str, batch_size: int = 32, num_workers: int = 4):
    """
    Full ETL pipeline: Extract → Clean → Load.
    Returns loaders and split DataFrames.
    """
    logger.info("=" * 60)
    logger.info(" STARTING ETL PIPELINE")
    logger.info("=" * 60)

    raw_df = extract_dataset(data_root)
    clean_df, removed_df = clean_dataset(raw_df)

    if not removed_df.empty:
        removed_path = Path(data_root).parent / "etl_removed_records.csv"
        removed_df.to_csv(removed_path, index=False)
        logger.info("   Removed records saved to: %s", removed_path)

    summary_path = Path(data_root).parent / "etl_clean_summary.csv"
    clean_df.to_csv(summary_path, index=False)
    logger.info("   Clean summary saved to: %s", summary_path)

    train_loader, val_loader, test_loader, train_df, val_df, test_df = create_dataloaders(
        clean_df, batch_size=batch_size, num_workers=num_workers
    )

    logger.info("=" * 60)
    logger.info(" ETL PIPELINE COMPLETE")
    logger.info("=" * 60)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df, clean_df


# ─────────────────────────────────────────────
#  EDA / REPORTING HELPERS
# ─────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """Bar chart of samples per class."""
    counts = df["label"].value_counts().rename(index=CLASS_DESCRIPTIONS)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.values, y=counts.index, palette="viridis", ax=ax)
    ax.set_title("Distribución de Clases — Dataset Limpio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Número de Imágenes")
    ax.set_ylabel("")
    for bar, val in zip(ax.patches, counts.values):
        ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_sample_grid(df: pd.DataFrame, n_per_class: int = 3, save_path: Optional[str] = None):
    """Grid showing sample images per class."""
    classes = sorted(df["label"].unique())
    n_cols  = n_per_class
    n_rows  = len(classes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    fig.suptitle("Muestras por Clase (Post-ETL)", fontsize=15, fontweight="bold", y=1.01)

    for row_idx, cls in enumerate(classes):
        samples = df[df["label"] == cls].sample(min(n_per_class, len(df[df["label"] == cls])))
        for col_idx, (_, sample) in enumerate(samples.iterrows()):
            ax = axes[row_idx][col_idx] if n_rows > 1 else axes[col_idx]
            img = Image.open(sample["path"]).convert("RGB")
            ax.imshow(img)
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(CLASS_DESCRIPTIONS.get(cls, cls), fontsize=8, rotation=0,
                              labelpad=80, va="center")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def generate_etl_report(clean_df: pd.DataFrame, removed_df: pd.DataFrame) -> str:
    """Return a Markdown-formatted ETL report string."""
    total_raw  = len(clean_df) + len(removed_df)
    report  = "# 📊 Reporte ETL — Lung & Colon Cancer Dataset\n\n"
    report += f"| Métrica | Valor |\n|---|---|\n"
    report += f"| Imágenes brutas encontradas | {total_raw:,} |\n"
    report += f"| Imágenes eliminadas | {len(removed_df):,} |\n"
    report += f"| Imágenes limpias | {len(clean_df):,} |\n"
    report += f"| Tasa de retención | {len(clean_df)/total_raw*100:.1f}% |\n"
    report += f"| Clases | {clean_df['label'].nunique()} |\n\n"

    report += "## Distribución por Clase\n\n"
    report += "| Clase | Descripción | N | % |\n|---|---|---|---|\n"
    for label, count in clean_df["label"].value_counts().items():
        desc = CLASS_DESCRIPTIONS.get(label, label)
        pct  = count / len(clean_df) * 100
        report += f"| `{label}` | {desc} | {count:,} | {pct:.1f}% |\n"

    if not removed_df.empty and "reason" in removed_df.columns:
        report += "\n## Razones de Eliminación\n\n"
        report += "| Razón | N |\n|---|---|\n"
        for reason, count in removed_df["reason"].value_counts().items():
            report += f"| {reason} | {count} |\n"

    return report