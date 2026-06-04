"""Smoke tests — verify imports and that small synthetic batches flow through models."""
from __future__ import annotations

import numpy as np
import pytest
import torch


def test_imports():
    """All top-level modules importable."""
    import src  # noqa: F401
    from src.data import SERDataModule  # noqa: F401
    from src.data.base import collate_ser_batch  # noqa: F401
    from src.data.meld import MELD_EMOTIONS, MELD_LABEL2ID  # noqa: F401
    from src.data.ravdess import RAVDESS_EMOTIONS, RAVDESS_LABEL2ID  # noqa: F401
    from src.models.fusion import build_fusion  # noqa: F401
    from src.utils.metrics import compute_classification_metrics  # noqa: F401
    from src.utils.seeding import seed_everything  # noqa: F401


def test_label_mappings():
    from src.data.meld import MELD_EMOTIONS, MELD_ID2LABEL, MELD_LABEL2ID
    from src.data.ravdess import RAVDESS_EMOTIONS, RAVDESS_ID2LABEL, RAVDESS_LABEL2ID

    assert len(RAVDESS_EMOTIONS) == 8
    assert len(MELD_EMOTIONS) == 7
    # Round-trip
    for emo, i in RAVDESS_LABEL2ID.items():
        assert RAVDESS_ID2LABEL[i] == emo
    for emo, i in MELD_LABEL2ID.items():
        assert MELD_ID2LABEL[i] == emo


def test_metrics_basic():
    from src.utils.metrics import compute_classification_metrics

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 2, 0, 0, 2]
    m = compute_classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert 0.0 <= m["accuracy"] <= 1.0
    assert 0.0 <= m["weighted_f1"] <= 1.0
    assert 0.0 <= m["unweighted_f1"] <= 1.0
    assert m["confusion_matrix"].shape == (3, 3)


def test_collate():
    from src.data.base import collate_ser_batch

    batch = [
        {"audio": np.zeros(16000, dtype=np.float32), "text": "hi", "label": 0, "utterance_id": "a"},
        {"audio": np.ones(16000, dtype=np.float32), "text": "yo", "label": 1, "utterance_id": "b"},
    ]
    out = collate_ser_batch(batch)
    assert out["audio"].shape == (2, 16000)
    assert out["label"].tolist() == [0, 1]
    assert out["text"] == ["hi", "yo"]


def test_seeding_is_deterministic():
    from src.utils.seeding import seed_everything

    seed_everything(123)
    a = torch.rand(5)
    seed_everything(123)
    b = torch.rand(5)
    assert torch.allclose(a, b)


@pytest.mark.parametrize("fusion_type", ["concat", "gated"])
def test_pooled_fusion_shapes(fusion_type):
    from src.models.fusion import build_fusion

    fusion = build_fusion(fusion_type, text_dim=128, audio_dim=256, hidden_dim=64)
    text = torch.randn(4, 128)
    audio = torch.randn(4, 256)
    out = fusion(text, audio)
    assert out.shape[0] == 4
    assert out.shape[-1] == fusion.output_dim


def test_cross_attention_fusion_shape():
    from src.models.fusion import CrossAttentionFusion

    fusion = CrossAttentionFusion(text_dim=128, audio_dim=256, hidden_dim=64, num_heads=4)
    text_seq = torch.randn(2, 10, 128)
    audio_seq = torch.randn(2, 50, 256)
    out = fusion(text_seq, audio_seq)
    assert out.shape == (2, fusion.output_dim)
