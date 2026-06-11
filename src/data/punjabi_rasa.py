"""Punjabi RASA Emotional Speech Dataset loader.

Source: AI4Bharat Rasa (filtered for Punjabi) released on Kaggle by Fatima Tu Zahra:
    https://www.kaggle.com/datasets/fatimatuzahra9115/punjabi-emotional-speech-dataset-rasa
License: CC BY 4.0
Total: 9,634 samples (8,672 train + 962 test), 48 kHz mono WAV, 3-15 sec each.

Class distribution (train / test):
    Angry    861 / 95     (10.0% / 9.9%)
    Happy    915 / 102    (10.6% / 10.6%)
    Sad      857 / 95     (9.9%  / 9.9%)
    Neutral  6039 / 670   (69.6% / 69.6%)

Severe class imbalance (Neutral ~7x other classes) — must use class-balanced
loss or oversampling. Class weights are computed automatically by the
DataModule from the train split.

Directory structure expected:
    data/punjabi_rasa/punjabi_emotional_speech/
        train/
            angry/audio/*.wav      + metadata.txt
            happy/audio/*.wav      + metadata.txt
            sad/audio/*.wav        + metadata.txt
            neutral/audio/*.wav    + metadata.txt
        test/
            (same structure)

Author: Muhammad Shahnawaz Khan
"""
from __future__ import annotations

from pathlib import Path

from src.data.base import BaseSERDataset, SERSample

# Punjabi RASA emotion folder names -> class indices.
# We follow alphabetical ordering for reproducibility.
PUNJABI_RASA_EMOTIONS = ["angry", "happy", "neutral", "sad"]
PUNJABI_RASA_LABEL2ID = {emo: i for i, emo in enumerate(PUNJABI_RASA_EMOTIONS)}
PUNJABI_RASA_ID2LABEL = {i: emo for emo, i in PUNJABI_RASA_LABEL2ID.items()}


class PunjabiRASADataset(BaseSERDataset):
    """Punjabi RASA emotional speech dataset (4-class).

    Args:
        data_dir: Path containing the `punjabi_emotional_speech/` directory
                  (typically `data/punjabi_rasa`).
        split:    One of {"train", "test"}.
        sample_rate: Target audio sample rate (Hz). RASA source is 48 kHz; we
                     resample to 16 kHz to match wav2vec2 / XLS-R expectations.
        max_audio_seconds: Clip / pad to this many seconds.
        emotions: Optional subset of emotions to keep.
                  e.g. ["angry","happy","sad"] removes Neutral to mitigate
                  imbalance. Default includes all 4.
        return_text: If True, populate `text` field with metadata transcript
                     (read from metadata.txt). RASA metadata is pipe-separated:
                     `filename|transcript|...`. Default False (audio-only).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        sample_rate: int = 16000,
        max_audio_seconds: float = 8.0,
        emotions: list[str] | None = None,
        return_text: bool = False,
    ):
        super().__init__(sample_rate=sample_rate, max_audio_seconds=max_audio_seconds)
        if split not in ("train", "test"):
            raise ValueError(f"split must be train or test, got {split!r}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.emotions = [e.lower() for e in (emotions or PUNJABI_RASA_EMOTIONS)]
        self.return_text = return_text

        # Locate the top-level dataset folder. Kaggle unzip may put files at
        # either `data_dir/<split>/...` or `data_dir/punjabi_emotional_speech/<split>/...`
        candidates = [
            self.data_dir / "punjabi_emotional_speech" / split,
            self.data_dir / split,
        ]
        split_dir = next((p for p in candidates if p.is_dir()), None)
        if split_dir is None:
            raise FileNotFoundError(
                f"Could not find Punjabi RASA `{split}` split under any of:\n  "
                + "\n  ".join(str(c) for c in candidates)
            )
        self.split_dir = split_dir

        # Collect (wav_path, label_str) tuples for the requested emotions.
        self.samples: list[tuple[Path, str]] = []
        self.transcripts: dict[str, str] = {}

        for emo in self.emotions:
            emo_audio_dir = split_dir / emo / "audio"
            if not emo_audio_dir.is_dir():
                # Tolerate flat layout `split/emo/*.wav` as a fallback.
                emo_audio_dir = split_dir / emo
            if not emo_audio_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing emotion directory: {split_dir / emo} (audio/ or flat)"
                )
            wavs = sorted(emo_audio_dir.glob("*.wav"))
            self.samples.extend((p, emo) for p in wavs)

            if self.return_text:
                meta_file = split_dir / emo / "metadata.txt"
                if meta_file.is_file():
                    for line in meta_file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split("|")
                        if len(parts) >= 2:
                            # `filename|transcript|...`
                            self.transcripts[parts[0].strip()] = parts[1].strip()

        if not self.samples:
            raise RuntimeError(
                f"No samples found under {split_dir} for emotions {self.emotions}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        wav_path, emo = self.samples[idx]
        audio = self.load_audio(wav_path, target_sr=self.sample_rate)
        audio = self.pad_or_truncate(audio)

        label = PUNJABI_RASA_LABEL2ID[emo]
        text = ""
        if self.return_text:
            text = self.transcripts.get(wav_path.name, "")

        return {
            "audio": audio,
            "label": label,
            "text": text,
            "utterance_id": f"punjabi_rasa/{self.split}/{emo}/{wav_path.name}",
        }

    # -------------------------------------------------------------------- #
    # Convenience properties
    # -------------------------------------------------------------------- #
    @property
    def num_classes(self) -> int:
        return len(self.emotions)

    @property
    def class_distribution(self) -> dict[str, int]:
        """Return {emotion: count} for the loaded split."""
        from collections import Counter

        return dict(Counter(emo for _, emo in self.samples))
