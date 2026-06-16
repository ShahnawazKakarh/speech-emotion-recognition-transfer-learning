"""URDU-Dataset (Latif et al. 2018) loader.

Source: github.com/siddiquelatif/URDU-Dataset (Latif, Rana, Khalifa,
Jurdak, Epps, 2018 - "Cross Lingual Speech Emotion Recognition: Urdu
vs. Western Languages"). 400 utterances, 4 emotions (Angry, Happy,
Neutral, Sad), 38 speakers across 4 Urdu TV talk shows.

License: open access (per the authors' GitHub release).

Directory structure expected:
    data/urdu_latif/
        Angry/   SM<show>_F<utt>_A<nnn>.wav
        Happy/   SM<show>_F<utt>_H<nnn>.wav
        Neutral/ SM<show>_F<utt>_N<nnn>.wav
        Sad/     SM<show>_F<utt>_S<nnn>.wav

Class label mapping is **identical to PUNJABI_RASA_EMOTIONS** so that a
model trained on one of these two corpora can be evaluated zero-shot on
the other without any label re-mapping. This is the core requirement
for the cross-lingual transfer experiments.

Filename convention `SM<show>_F<utt>_<emo><N>.wav` exposes the TV show
(SM1..SM4) but not speaker IDs. We support a show-independent split
mode (`split_mode="show_independent"`) which holds out one show as the
test set; this is a stricter generalisation test than random splits.

Author: Muhammad Shahnawaz Khan
"""
from __future__ import annotations

import re
from pathlib import Path

from src.data.base import BaseSERDataset

# Use identical class ordering to PUNJABI_RASA so class indices align
# for cross-lingual transfer.
URDU_LATIF_EMOTIONS = ["angry", "happy", "neutral", "sad"]
URDU_LATIF_LABEL2ID = {emo: i for i, emo in enumerate(URDU_LATIF_EMOTIONS)}
URDU_LATIF_ID2LABEL = {i: emo for emo, i in URDU_LATIF_LABEL2ID.items()}

# On-disk emotion folder names use Title-case. Map to lowercase index keys.
_FOLDER_TO_LABEL = {
    "Angry": "angry",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
}

_SHOW_RE = re.compile(r"^(SM\d+)_")


class URDULatifDataset(BaseSERDataset):
    """URDU-Dataset (Latif 2018) loader.

    Args:
        data_dir: Path containing the 4 emotion sub-folders (typically
                  `data/urdu_latif`).
        split: One of {"train", "val", "test", "all"}. With
               split_mode="random" a deterministic 80/10/10 stratified
               split is used. With split_mode="show_independent" the
               held_out_show is the test set and the rest is split 90/10
               into train/val.
        split_mode: "random" or "show_independent".
        held_out_show: When split_mode="show_independent", which show
                       (e.g. "SM4") is the held-out test set. Default
                       SM4 (the largest show).
        sample_rate: Target audio sample rate (Hz). Source is mostly
                     ~22 kHz, we resample to 16 kHz for wav2vec2.
        max_audio_seconds: Clip / pad to this many seconds.
        seed: Random seed for the random split.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        split_mode: str = "random",
        held_out_show: str = "SM4",
        sample_rate: int = 16000,
        max_audio_seconds: float = 6.0,
        seed: int = 42,
    ):
        super().__init__(sample_rate=sample_rate, max_audio_seconds=max_audio_seconds)
        if split not in ("train", "val", "test", "all"):
            raise ValueError(f"split must be train/val/test/all, got {split!r}")
        if split_mode not in ("random", "show_independent"):
            raise ValueError(
                f"split_mode must be random or show_independent, got {split_mode!r}"
            )

        self.data_dir = Path(data_dir)
        self.split = split
        self.split_mode = split_mode
        self.held_out_show = held_out_show
        self.seed = seed

        # Collect (path, label_str, show) tuples across all 4 folders.
        all_items: list[tuple[Path, str, str]] = []
        for folder, emo in _FOLDER_TO_LABEL.items():
            emo_dir = self.data_dir / folder
            if not emo_dir.is_dir():
                raise FileNotFoundError(f"Missing emotion folder: {emo_dir}")
            for wav in sorted(emo_dir.glob("*.wav")):
                m = _SHOW_RE.match(wav.name)
                show = m.group(1) if m else "UNK"
                all_items.append((wav, emo, show))

        if not all_items:
            raise RuntimeError(f"No .wav files found under {self.data_dir}")

        # Compute (train, val, test) index lists according to split_mode.
        train_idx, val_idx, test_idx = self._compute_splits(all_items)

        if split == "train":
            keep = train_idx
        elif split == "val":
            keep = val_idx
        elif split == "test":
            keep = test_idx
        else:  # "all"
            keep = list(range(len(all_items)))

        self.samples: list[tuple[Path, str]] = [
            (all_items[i][0], all_items[i][1]) for i in keep
        ]
        self._shows = [all_items[i][2] for i in keep]

    def _compute_splits(
        self, items: list[tuple[Path, str, str]]
    ) -> tuple[list[int], list[int], list[int]]:
        import numpy as np
        from sklearn.model_selection import train_test_split

        n = len(items)
        labels = np.array([it[1] for it in items])
        shows = np.array([it[2] for it in items])
        all_idx = np.arange(n)

        if self.split_mode == "show_independent":
            test_idx = all_idx[shows == self.held_out_show].tolist()
            rest_idx = all_idx[shows != self.held_out_show]
            rest_labels = labels[rest_idx]
            if len(rest_idx) == 0 or len(test_idx) == 0:
                raise ValueError(
                    f"show_independent split: held_out_show={self.held_out_show!r} "
                    f"yields empty train or test set."
                )
            train_idx, val_idx = train_test_split(
                rest_idx,
                test_size=0.1,
                stratify=rest_labels,
                random_state=self.seed,
            )
            return train_idx.tolist(), val_idx.tolist(), test_idx
        else:
            # Random stratified 80/10/10.
            train_val_idx, test_idx = train_test_split(
                all_idx, test_size=0.1, stratify=labels, random_state=self.seed,
            )
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=0.111,  # 10% of 90% ≈ 10% of total
                stratify=labels[train_val_idx],
                random_state=self.seed,
            )
            return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        wav_path, emo = self.samples[idx]
        audio = self.load_audio(wav_path, target_sr=self.sample_rate)
        audio = self.pad_or_truncate(audio)
        return {
            "audio": audio,
            "label": URDU_LATIF_LABEL2ID[emo],
            "text": "",
            "utterance_id": f"urdu_latif/{wav_path.parent.name}/{wav_path.name}",
        }

    @property
    def num_classes(self) -> int:
        return len(URDU_LATIF_EMOTIONS)

    @property
    def class_distribution(self) -> dict[str, int]:
        from collections import Counter

        return dict(Counter(emo for _, emo in self.samples))
