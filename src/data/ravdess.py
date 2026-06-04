"""RAVDESS dataset loader.

RAVDESS filenames encode metadata as 7 hyphen-separated identifiers:
    01-01-03-01-02-01-12.wav
    └─ Modality (01 = full-AV, 02 = video, 03 = audio-only)
       └─ Vocal channel (01 = speech, 02 = song)
          └─ Emotion (01=neutral, 02=calm, 03=happy, 04=sad,
                      05=angry, 06=fearful, 07=disgust, 08=surprised)
             └─ Intensity (01 = normal, 02 = strong)
                └─ Statement (01 = "Kids...", 02 = "Dogs...")
                   └─ Repetition
                      └─ Actor (01-24; odd = male, even = female)
"""
from __future__ import annotations

from pathlib import Path

from src.data.base import BaseSERDataset

# RAVDESS emotion codes → class indices (0-7)
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}
RAVDESS_LABEL2ID = {emo: i for i, emo in enumerate(RAVDESS_EMOTIONS.values())}
RAVDESS_ID2LABEL = {i: emo for emo, i in RAVDESS_LABEL2ID.items()}

# RAVDESS only has 2 fixed sentences (statement code 01 vs 02)
RAVDESS_STATEMENTS = {
    "01": "Kids are talking by the door.",
    "02": "Dogs are sitting by the door.",
}


class RAVDESSDataset(BaseSERDataset):
    """RAVDESS dataset.

    Args:
        data_dir: Path to the unpacked RAVDESS directory containing Actor_NN/ subdirs.
        file_list: Optional list of relative .wav paths to include. If None, scans data_dir.
        speech_only: If True, only include speech files (vocal_channel=01), excluding song.
        sample_rate: Target audio sample rate.
        max_audio_seconds: Clip/pad to this many seconds.
    """

    def __init__(
        self,
        data_dir: str | Path,
        file_list: list[str] | None = None,
        speech_only: bool = True,
        sample_rate: int = 16000,
        max_audio_seconds: float = 5.0,
    ):
        super().__init__(sample_rate=sample_rate, max_audio_seconds=max_audio_seconds)
        self.data_dir = Path(data_dir)

        if file_list is None:
            self.files = sorted(self.data_dir.rglob("*.wav"))
        else:
            self.files = [self.data_dir / f for f in file_list]

        if speech_only:
            # vocal_channel: "01" = speech, "02" = song
            self.files = [
                f for f in self.files if self._parse_filename(f)["vocal_channel"] == "01"
            ]

    @staticmethod
    def _parse_filename(path: Path) -> dict[str, str]:
        """Parse a RAVDESS filename like 03-01-04-01-02-01-12.wav into its 7 identifiers."""
        parts = path.stem.split("-")
        if len(parts) != 7:
            raise ValueError(f"Unexpected RAVDESS filename format: {path.name}")
        return {
            "modality": parts[0],
            "vocal_channel": parts[1],
            "emotion": parts[2],
            "intensity": parts[3],
            "statement": parts[4],
            "repetition": parts[5],
            "actor": parts[6],
        }

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        meta = self._parse_filename(path)

        emotion_name = RAVDESS_EMOTIONS[meta["emotion"]]
        label = RAVDESS_LABEL2ID[emotion_name]
        text = RAVDESS_STATEMENTS[meta["statement"]]

        audio = self.load_audio(path, target_sr=self.sample_rate)
        audio = self.pad_or_truncate(audio)

        return {
            "audio": audio,
            "text": text,
            "label": label,
            "utterance_id": path.stem,
        }
