"""MELD dataset loader (conversational emotion recognition from Friends).

MELD provides train/dev/test CSV splits with columns including:
    Dialogue_ID, Utterance_ID, Speaker, Utterance, Emotion, Sentiment, ...
and corresponding .wav audio files in the form: dia<DID>_utt<UID>.wav

Reference: https://github.com/declare-lab/MELD
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.base import BaseSERDataset

# MELD has 7 emotion categories
MELD_EMOTIONS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
MELD_LABEL2ID = {emo: i for i, emo in enumerate(MELD_EMOTIONS)}
MELD_ID2LABEL = {i: emo for emo, i in MELD_LABEL2ID.items()}


class MELDDataset(BaseSERDataset):
    """MELD dataset with optional conversational context.

    Expected layout under ``data_dir``:
        train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv
        train_audio/dia<DID>_utt<UID>.wav
        dev_audio/...
        test_audio/...

    Args:
        data_dir: Root MELD directory.
        split: One of "train", "dev", "test".
        use_context: If True, prepend up to ``context_window`` prior utterances
            from the same dialogue to the target utterance, separated by [SEP].
        context_window: How many prior utterances to include as context.
    """

    SPLIT_TO_CSV = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }
    SPLIT_TO_AUDIO_DIR = {
        "train": "train_audio",
        "dev": "dev_audio",
        "test": "test_audio",
    }

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        use_context: bool = False,
        context_window: int = 2,
        sample_rate: int = 16000,
        max_audio_seconds: float = 8.0,
    ):
        super().__init__(sample_rate=sample_rate, max_audio_seconds=max_audio_seconds)
        if split not in self.SPLIT_TO_CSV:
            raise ValueError(f"split must be one of {list(self.SPLIT_TO_CSV)}, got {split}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.use_context = use_context
        self.context_window = context_window

        csv_path = self.data_dir / self.SPLIT_TO_CSV[split]
        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["Utterance", "Emotion"])
        self.df = self.df[self.df["Emotion"].isin(MELD_EMOTIONS)].reset_index(drop=True)

        self.audio_dir = self.data_dir / self.SPLIT_TO_AUDIO_DIR[split]

        # Filter rows whose audio is missing or empty (ffmpeg sometimes produces
        # empty .wav for .mp4 files with broken moov atoms; some MELD .mp4s also
        # have no audio stream at all).
        before = len(self.df)
        def _audio_ok(row):
            p = self.audio_dir / f"dia{int(row['Dialogue_ID'])}_utt{int(row['Utterance_ID'])}.wav"
            return p.exists() and p.stat().st_size > 0
        self.df = self.df[self.df.apply(_audio_ok, axis=1)].reset_index(drop=True)
        after = len(self.df)
        if after < before:
            print(
                f"[MELD/{split}] dropped {before - after} rows with missing or empty audio "
                f"({before} → {after})"
            )

        # Pre-group by dialogue for context lookup
        self._dialogue_groups = {
            did: g.sort_values("Utterance_ID").reset_index(drop=True)
            for did, g in self.df.groupby("Dialogue_ID")
        }

    def __len__(self) -> int:
        return len(self.df)

    def _get_context_text(self, dialogue_id: int, utterance_id: int) -> str:
        group = self._dialogue_groups[dialogue_id]
        prior = group[group["Utterance_ID"] < utterance_id].tail(self.context_window)
        if prior.empty:
            return ""
        return " [SEP] ".join(
            f"{row['Speaker']}: {row['Utterance']}" for _, row in prior.iterrows()
        )

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        did = int(row["Dialogue_ID"])
        uid = int(row["Utterance_ID"])
        utterance = str(row["Utterance"])
        emotion = str(row["Emotion"])

        if self.use_context:
            context = self._get_context_text(did, uid)
            text = f"{context} [SEP] {utterance}" if context else utterance
        else:
            text = utterance

        audio_path = self.audio_dir / f"dia{did}_utt{uid}.wav"
        audio = self.load_audio(audio_path, target_sr=self.sample_rate)
        audio = self.pad_or_truncate(audio)

        return {
            "audio": audio,
            "text": text,
            "label": MELD_LABEL2ID[emotion],
            "utterance_id": f"dia{did}_utt{uid}",
        }
