"""Lightning module wiring text/audio/multimodal models into a training loop."""
from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from transformers import get_linear_schedule_with_warmup

from src.models.audio_encoder import AudioEncoder
from src.models.fusion import build_fusion
from src.models.text_encoder import TextEncoder
from src.utils.metrics import compute_classification_metrics


class SERModel(nn.Module):
    """The actual neural network — assembled per the config."""

    def __init__(self, model_cfg: dict, data_cfg: dict):
        super().__init__()
        self.model_type = model_cfg["type"]
        self.num_classes = model_cfg["num_classes"]
        dropout = model_cfg.get("dropout", 0.1)

        if self.model_type == "text_only":
            self.text_encoder = TextEncoder(
                model_name=model_cfg["text_encoder"],
                return_sequence=False,
                freeze_encoder_layers=model_cfg.get("freeze_encoder_layers", 0),
                max_length=data_cfg.get("max_text_tokens", 128),
            )
            feat_dim = self.text_encoder.hidden_size

        elif self.model_type == "audio_only":
            self.audio_encoder = AudioEncoder(
                model_name=model_cfg["audio_encoder"],
                return_sequence=False,
                pooling=model_cfg.get("pooling", "mean"),
                freeze_feature_extractor=model_cfg.get("freeze_feature_extractor", True),
                freeze_encoder_layers=model_cfg.get("freeze_encoder_layers", 0),
                sample_rate=data_cfg.get("sample_rate", 16000),
            )
            feat_dim = self.audio_encoder.hidden_size

        elif self.model_type == "multimodal":
            fusion_type = model_cfg.get("fusion", "concat")
            # Cross-attention needs sequence-level outputs; concat/gated use pooled
            seq_out = fusion_type == "cross_attention"
            self.text_encoder = TextEncoder(
                model_name=model_cfg["text_encoder"],
                return_sequence=seq_out,
                max_length=data_cfg.get("max_text_tokens", 128),
            )
            self.audio_encoder = AudioEncoder(
                model_name=model_cfg["audio_encoder"],
                return_sequence=seq_out,
                freeze_feature_extractor=model_cfg.get("freeze_feature_extractor", True),
                sample_rate=data_cfg.get("sample_rate", 16000),
            )
            self.fusion = build_fusion(
                fusion_type=fusion_type,
                text_dim=self.text_encoder.hidden_size,
                audio_dim=self.audio_encoder.hidden_size,
                hidden_dim=model_cfg.get("fusion_hidden", 512),
                num_heads=model_cfg.get("num_attention_heads", 8),
                dropout=dropout,
            )
            feat_dim = self.fusion.output_dim

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, self.num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        if self.model_type == "text_only":
            feats = self.text_encoder(batch["text"])
        elif self.model_type == "audio_only":
            feats = self.audio_encoder(batch["audio"])
        else:
            text_feats = self.text_encoder(batch["text"])
            audio_feats = self.audio_encoder(batch["audio"])
            feats = self.fusion(text_feats, audio_feats)
        return self.classifier(feats)


class SERLightningModule(L.LightningModule):
    """Lightning wrapper: training/validation/test loops, optimizer, scheduler, metrics."""

    def __init__(self, cfg: dict[str, Any], label_names: list[str] | None = None):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.label_names = label_names or [f"class_{i}" for i in range(cfg["model"]["num_classes"])]

        self.model = SERModel(cfg["model"], cfg["data"])

        # Class weights (computed in setup if requested)
        self.register_buffer("class_weights", torch.ones(cfg["model"]["num_classes"]))
        self.use_class_weights = cfg.get("loss", {}).get("use_class_weights", False)

        # Buffers for epoch-level metric aggregation
        self._val_preds: list[int] = []
        self._val_labels: list[int] = []
        self._test_preds: list[int] = []
        self._test_labels: list[int] = []

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" and self.use_class_weights:
            train_loader = self.trainer.datamodule.train_dataloader()
            labels = []
            for batch in train_loader:
                labels.extend(batch["label"].tolist())
            classes = np.arange(self.cfg["model"]["num_classes"])
            weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
            self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self.model(batch)
        loss = self._loss(logits, batch["label"])
        acc = (logits.argmax(-1) == batch["label"]).float().mean()
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.model(batch)
        loss = self._loss(logits, batch["label"])
        preds = logits.argmax(-1)
        self._val_preds.extend(preds.cpu().tolist())
        self._val_labels.extend(batch["label"].cpu().tolist())
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        metrics = compute_classification_metrics(
            self._val_labels, self._val_preds, self.label_names
        )
        self.log("val/acc", metrics["accuracy"], prog_bar=True)
        self.log("val/wf1", metrics["weighted_f1"], prog_bar=True)
        self.log("val/uf1", metrics["unweighted_f1"], prog_bar=True)
        self._val_preds.clear()
        self._val_labels.clear()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self.model(batch)
        preds = logits.argmax(-1)
        self._test_preds.extend(preds.cpu().tolist())
        self._test_labels.extend(batch["label"].cpu().tolist())

    def on_test_epoch_end(self) -> None:
        metrics = compute_classification_metrics(
            self._test_labels, self._test_preds, self.label_names
        )
        self.log("test/acc", metrics["accuracy"])
        self.log("test/wf1", metrics["weighted_f1"])
        self.log("test/uf1", metrics["unweighted_f1"])
        # Stash for the caller (evaluate.py reads these)
        self.test_metrics = metrics

    def configure_optimizers(self) -> dict:
        opt_cfg = self.cfg["optimizer"]
        sch_cfg = self.cfg.get("scheduler", {})

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.01),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        if sch_cfg.get("name") == "linear_warmup":
            total_steps = self.trainer.estimated_stepping_batches
            warmup_steps = int(sch_cfg.get("warmup_ratio", 0.1) * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        return {"optimizer": optimizer}
