"""
tests/test_model.py
Unit and integration tests for the CNN model and ETL pipeline.
Run with: pytest tests/ -v
"""

import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import (
    CLASS_LABELS, CLASS_NAMES, CLASS_DESCRIPTIONS, MALIGNANT_CLASSES,
    inference_transform, HistoDataset, _is_valid_image,
)
from src.train import CancerCNN, NUM_CLASSES


# ─────────────────────────────────────────────
#  FIXTURES
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def model(device):
    return CancerCNN(num_classes=NUM_CLASSES, pretrained=False).to(device).eval()


@pytest.fixture
def dummy_image():
    """224×224 RGB PIL image filled with random noise."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def dummy_tensor(dummy_image, device):
    return inference_transform(dummy_image).unsqueeze(0).to(device)


# ─────────────────────────────────────────────
#  MODEL TESTS
# ─────────────────────────────────────────────

class TestCancerCNN:
    def test_output_shape(self, model, dummy_tensor):
        with torch.no_grad():
            output = model(dummy_tensor)
        assert output.shape == (1, NUM_CLASSES), f"Expected (1, {NUM_CLASSES}), got {output.shape}"

    def test_output_is_logits(self, model, dummy_tensor):
        """Logits can be any real value (not necessarily in [0,1])."""
        with torch.no_grad():
            output = model(dummy_tensor)
        # After softmax they must sum to 1
        probs = torch.softmax(output, dim=1)
        assert abs(probs.sum().item() - 1.0) < 1e-4

    def test_num_classes(self):
        assert NUM_CLASSES == 5

    def test_trainable_parameters(self, model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0, "Model must have trainable parameters"

    def test_batch_inference(self, model, device):
        batch = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(batch)
        assert output.shape == (4, NUM_CLASSES)


# ─────────────────────────────────────────────
#  TRANSFORM TESTS
# ─────────────────────────────────────────────

class TestTransforms:
    def test_inference_transform_shape(self, dummy_image):
        tensor = inference_transform(dummy_image)
        assert tensor.shape == (3, 224, 224)

    def test_inference_transform_normalized(self, dummy_image):
        tensor = inference_transform(dummy_image)
        # Values should be roughly in [-3, 3] after ImageNet normalization
        assert tensor.min().item() > -5
        assert tensor.max().item() < 5


# ─────────────────────────────────────────────
#  UTILS TESTS
# ─────────────────────────────────────────────

class TestUtils:
    def test_class_labels_count(self):
        assert len(CLASS_LABELS) == 5

    def test_class_names_inverse(self):
        for label, idx in CLASS_LABELS.items():
            assert CLASS_NAMES[idx] == label

    def test_malignant_classes_subset(self):
        assert MALIGNANT_CLASSES.issubset(set(CLASS_LABELS.keys()))

    def test_valid_image(self, tmp_path, dummy_image):
        p = tmp_path / "test.jpg"
        dummy_image.save(p)
        valid, reason = _is_valid_image(str(p))
        assert valid is True
        assert reason == "ok"

    def test_invalid_image_too_small(self, tmp_path):
        tiny = Image.new("RGB", (10, 10), color=(128, 0, 0))
        p = tmp_path / "tiny.jpg"
        tiny.save(p)
        valid, reason = _is_valid_image(str(p), min_size=32)
        assert valid is False
        assert "too small" in reason

    def test_invalid_image_corrupt(self, tmp_path):
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"not_an_image_at_all")
        valid, reason = _is_valid_image(str(p))
        assert valid is False

    def test_histo_dataset(self, tmp_path, dummy_image):
        import pandas as pd
        # Create a small fake dataset
        (tmp_path / "colon_aca").mkdir()
        img_path = tmp_path / "colon_aca" / "img001.jpg"
        dummy_image.save(img_path)

        df = pd.DataFrame([{
            "path": str(img_path),
            "filename": "img001.jpg",
            "label": "colon_aca",
            "organ": "colon",
            "label_idx": 0,
            "description": CLASS_DESCRIPTIONS["colon_aca"],
            "is_malignant": True,
        }])
        ds = HistoDataset(df, transform=inference_transform)
        assert len(ds) == 1
        img_t, label = ds[0]
        assert img_t.shape == (3, 224, 224)
        assert label == 0


# ─────────────────────────────────────────────
#  SMS TESTS (no actual sending)
# ─────────────────────────────────────────────

class TestSMS:
    def test_preview_malignant(self):
        from app.sms import format_sms_preview
        result = {
            "is_malignant": True,
            "description": "Adenocarcinoma de Pulmón (Maligno)",
            "confidence": 0.92,
        }
        preview = format_sms_preview(result, "Test Paciente")
        assert "MALIGNO" in preview
        assert "Test Paciente" in preview
        assert "92.0%" in preview

    def test_preview_benign(self):
        from app.sms import format_sms_preview
        result = {
            "is_malignant": False,
            "description": "Tejido Benigno de Colon",
            "confidence": 0.88,
        }
        preview = format_sms_preview(result, "Ana López")
        assert "BENIGNO" in preview
        assert "Ana López" in preview

    def test_missing_credentials(self, monkeypatch):
        from app.sms import send_diagnosis_sms
        monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
        monkeypatch.delenv("TWILIO_AUTH_TOKEN",  raising=False)
        monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)
        result = {
            "is_malignant": False,
            "description": "Tejido Benigno de Colon",
            "confidence": 0.88,
        }
        resp = send_diagnosis_sms("+5730000000", result)
        assert resp["success"] is False
        assert resp["error"] is not None