"""Fusion modules combining text and audio embeddings."""
from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Simplest fusion: concatenate pooled text + audio embeddings, then project."""

    def __init__(self, text_dim: int, audio_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_dim = hidden_dim

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        # Both pooled: (B, H)
        return self.proj(torch.cat([text_emb, audio_emb], dim=-1))


class GatedFusion(nn.Module):
    """Learned gate that weights text vs audio contributions per sample."""

    def __init__(self, text_dim: int, audio_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        t = self.text_proj(text_emb)
        a = self.audio_proj(audio_emb)
        g = self.gate(torch.cat([t, a], dim=-1))
        fused = g * t + (1 - g) * a
        return self.dropout(fused)


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-attention between text and audio token sequences.

    Expects sequence-level inputs (B, T_text, H_text) and (B, T_audio, H_audio).
    Returns pooled fused vector (B, hidden_dim).
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.text_to_audio_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_text_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = 2 * hidden_dim

    def forward(
        self,
        text_seq: torch.Tensor,  # (B, T_text, H_text)
        audio_seq: torch.Tensor,  # (B, T_audio, H_audio)
    ) -> torch.Tensor:
        t = self.text_proj(text_seq)
        a = self.audio_proj(audio_seq)

        # Text queries attend to audio
        t_attended, _ = self.text_to_audio_attn(query=t, key=a, value=a)
        t_out = self.norm_t(t + self.dropout(t_attended))

        # Audio queries attend to text
        a_attended, _ = self.audio_to_text_attn(query=a, key=t, value=t)
        a_out = self.norm_a(a + self.dropout(a_attended))

        # Mean-pool both modalities then concat
        t_pool = t_out.mean(dim=1)
        a_pool = a_out.mean(dim=1)
        return torch.cat([t_pool, a_pool], dim=-1)


def build_fusion(
    fusion_type: str,
    text_dim: int,
    audio_dim: int,
    hidden_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.1,
) -> nn.Module:
    if fusion_type == "concat":
        return ConcatFusion(text_dim, audio_dim, hidden_dim, dropout)
    if fusion_type == "gated":
        return GatedFusion(text_dim, audio_dim, hidden_dim, dropout)
    if fusion_type == "cross_attention":
        return CrossAttentionFusion(text_dim, audio_dim, hidden_dim, num_heads, dropout)
    raise ValueError(f"Unknown fusion type: {fusion_type}")
