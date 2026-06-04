"""Whisper-based ASR for transcribing audio when ground-truth text isn't available.

Use this to generate transcripts for RAVDESS (where the actual spoken text is fixed
but you may want to validate ASR quality) or for any new .wav file the demo receives.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np


class WhisperTranscriber:
    """Thin wrapper around openai-whisper for batched transcription.

    Args:
        model_size: One of "tiny", "base", "small", "medium", "large-v3".
        language: ISO code (e.g. "en") to skip language detection.
        device: "cpu" | "cuda" | None (auto).
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str | None = "en",
        device: str | None = None,
    ):
        import whisper  # imported lazily so the package isn't required everywhere

        self.model_size = model_size
        self.language = language
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio: str | Path | np.ndarray) -> str:
        """Transcribe a single audio file or pre-loaded waveform.

        For a waveform array, must be float32, mono, 16kHz.
        """
        if isinstance(audio, (str, Path)):
            result = self.model.transcribe(str(audio), language=self.language)
        else:
            result = self.model.transcribe(audio.astype(np.float32), language=self.language)
        return result["text"].strip()

    def transcribe_batch(self, paths: list[str | Path]) -> list[str]:
        """Convenience: transcribe a list of files sequentially.

        Whisper doesn't natively support batching across files; for large jobs
        consider faster-whisper or whisperx.
        """
        return [self.transcribe(p) for p in paths]


@lru_cache(maxsize=1)
def get_default_transcriber(model_size: str = "base") -> WhisperTranscriber:
    """Cached singleton accessor to avoid reloading Whisper on every call."""
    return WhisperTranscriber(model_size=model_size)
