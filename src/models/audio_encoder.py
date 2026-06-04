"""Audio encoder built on self-supervised speech models (wav2vec2 / WavLM / HuBERT)."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel


class AttentionPool(nn.Module):
    """Learned attention pooling over the time dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (B, T, H)
        scores = self.attn(hidden_states).squeeze(-1)  # (B, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (hidden_states * weights).sum(dim=1)  # (B, H)


class AudioEncoder(nn.Module):
    """Wraps an SSL speech model and pools the frame-level hidden states.

    Args:
        model_name: HF model id, e.g. "facebook/wav2vec2-base", "microsoft/wavlm-base".
        return_sequence: If True, returns (B, T, H) frame embeddings; else pooled (B, H).
        pooling: "mean" | "attention" — only used when return_sequence=False.
        freeze_feature_extractor: Freeze the CNN front-end (standard practice).
        freeze_encoder_layers: Number of bottom transformer layers to freeze.
        sample_rate: Audio sample rate (must match model expectation, typically 16k).
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        return_sequence: bool = False,
        pooling: str = "mean",
        freeze_feature_extractor: bool = True,
        freeze_encoder_layers: int = 0,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.return_sequence = return_sequence
        self.pooling = pooling
        self.sample_rate = sample_rate
        self.hidden_size: int = self.model.config.hidden_size

        if freeze_feature_extractor and hasattr(self.model, "feature_extractor"):
            self.model.feature_extractor._freeze_parameters()

        if freeze_encoder_layers > 0 and hasattr(self.model, "encoder"):
            for layer in self.model.encoder.layers[:freeze_encoder_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

        if pooling == "attention":
            self.attn_pool = AttentionPool(self.hidden_size)
        else:
            self.attn_pool = None

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            audio: Pre-padded waveform tensor of shape (B, num_samples) at self.sample_rate.

        Returns:
            (B, H) pooled or (B, T, H) frame-level embeddings.
        """
        device = audio.device
        # Feature extractor expects a list of 1D arrays
        inputs = self.feature_extractor(
            [a.cpu().numpy() for a in audio],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs["input_values"].to(device)

        out = self.model(input_values)
        hidden = out.last_hidden_state  # (B, T, H)

        if self.return_sequence:
            return hidden

        if self.pooling == "attention" and self.attn_pool is not None:
            return self.attn_pool(hidden)
        return hidden.mean(dim=1)
