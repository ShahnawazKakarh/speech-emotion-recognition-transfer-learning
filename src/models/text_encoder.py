"""Text encoder built on HuggingFace transformers (RoBERTa, DeBERTa, BERT, ...)."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """Wraps a HuggingFace AutoModel and returns either pooled or token-level embeddings.

    Args:
        model_name: HF model identifier, e.g. "roberta-base", "microsoft/deberta-v3-base".
        return_sequence: If True, returns (B, T, H) token embeddings. If False, returns (B, H)
            pooled via [CLS] / mean.
        freeze_encoder_layers: Number of bottom encoder layers to freeze (0 = none).
        max_length: Tokenizer max length.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        return_sequence: bool = False,
        freeze_encoder_layers: int = 0,
        max_length: int = 128,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.return_sequence = return_sequence
        self.max_length = max_length
        self.hidden_size: int = self.model.config.hidden_size

        if freeze_encoder_layers > 0:
            self._freeze_layers(freeze_encoder_layers)

    def _freeze_layers(self, n: int) -> None:
        # Works for encoder-style transformers (BERT/RoBERTa/DeBERTa style)
        if hasattr(self.model, "embeddings"):
            for p in self.model.embeddings.parameters():
                p.requires_grad = False
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None and hasattr(encoder, "layer"):
            for layer in encoder.layer[:n]:
                for p in layer.parameters():
                    p.requires_grad = False

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, texts: list[str]) -> torch.Tensor:
        enc = self.tokenize(texts)
        enc = {k: v.to(next(self.parameters()).device) for k, v in enc.items()}
        out = self.model(**enc)

        if self.return_sequence:
            return out.last_hidden_state  # (B, T, H)

        # Pooled: use [CLS] if available, else mean-pool
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        mask = enc["attention_mask"].unsqueeze(-1).float()
        summed = (out.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
