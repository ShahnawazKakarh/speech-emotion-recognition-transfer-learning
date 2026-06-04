"""Base dataset utilities shared across RAVDESS, MELD, IEMOCAP."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset


@dataclass
class SERSample:
    """A single SER sample: audio waveform, transcript text, label."""

    audio: np.ndarray  # shape (num_samples,) at target sample rate
    sample_rate: int
    text: str
    label: int
    utterance_id: str = ""


class BaseSERDataset(Dataset):
    """Shared logic for loading audio and clipping/padding to fixed length."""

    def __init__(
        self,
        sample_rate: int = 16000,
        max_audio_seconds: float = 8.0,
    ):
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * max_audio_seconds)

    @staticmethod
    def load_audio(path: str | Path, target_sr: int = 16000) -> np.ndarray:
        """Load audio file and resample to target_sr. Returns mono float32 array."""
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            # Lightweight resample via librosa to keep deps minimal at top level
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio

    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad with zeros or center-truncate to self.max_samples."""
        n = len(audio)
        if n >= self.max_samples:
            start = (n - self.max_samples) // 2
            return audio[start : start + self.max_samples]
        pad_total = self.max_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode="constant")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError


def collate_ser_batch(batch: list[dict]) -> dict:
    """Collate a list of SER samples into batched tensors.

    Expects each item dict to contain at minimum:
        - audio: np.ndarray of equal length (already padded)
        - label: int
    Optional:
        - text: str
        - utterance_id: str
    """
    out: dict = {}
    out["audio"] = torch.from_numpy(np.stack([b["audio"] for b in batch])).float()
    out["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    if "text" in batch[0]:
        out["text"] = [b["text"] for b in batch]
    if "utterance_id" in batch[0]:
        out["utterance_id"] = [b["utterance_id"] for b in batch]
    return out
