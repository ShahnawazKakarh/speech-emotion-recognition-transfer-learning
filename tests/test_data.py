"""Tests for the data loaders that don't require the actual dataset files."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_ravdess_filename_parsing():
    from src.data.ravdess import RAVDESSDataset

    # 03-01-04-01-02-01-12.wav → modality=03, vocal_channel=01 (speech),
    # emotion=04 (sad), intensity=01, statement=02, repetition=01, actor=12
    parsed = RAVDESSDataset._parse_filename(Path("03-01-04-01-02-01-12.wav"))
    assert parsed["modality"] == "03"
    assert parsed["vocal_channel"] == "01"
    assert parsed["emotion"] == "04"
    assert parsed["statement"] == "02"
    assert parsed["actor"] == "12"


def test_ravdess_filename_invalid():
    from src.data.ravdess import RAVDESSDataset

    with pytest.raises(ValueError):
        RAVDESSDataset._parse_filename(Path("bad-name.wav"))


def test_ravdess_statements_present():
    from src.data.ravdess import RAVDESS_STATEMENTS

    assert "01" in RAVDESS_STATEMENTS
    assert "02" in RAVDESS_STATEMENTS
    assert "Kids" in RAVDESS_STATEMENTS["01"]
    assert "Dogs" in RAVDESS_STATEMENTS["02"]


def test_meld_label_coverage():
    from src.data.meld import MELD_EMOTIONS

    # MELD canonical 7-class set
    expected = {"neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"}
    assert set(MELD_EMOTIONS) == expected


def test_speaker_independent_split_disjoint():
    """Speaker-independent split: no actor appears in more than one of train/val/test."""
    from src.data.datamodule import SERDataModule
    from src.data.ravdess import RAVDESSDataset

    # Build a fake dataset with synthetic filenames so we don't need real audio
    class FakeFullDS:
        files = [
            # 4 wavs per actor for actors 1..24
            Path(f"03-01-{e:02d}-01-01-01-{a:02d}.wav")
            for a in range(1, 25)
            for e in range(1, 5)
        ]

    cfg = {
        "dataset": "ravdess",
        "data": {
            "data_dir": "data/ravdess",  # unused — overridden by fake
            "batch_size": 8,
            "split_strategy": "speaker_independent",
            "test_actors": [21, 22, 23, 24],
            "val_actors": [19, 20],
        },
    }
    dm = SERDataModule(cfg)
    train_idx, val_idx, test_idx = dm._speaker_independent_split(FakeFullDS())

    def actors_in(indices):
        return {
            int(RAVDESSDataset._parse_filename(FakeFullDS.files[i])["actor"]) for i in indices
        }

    train_actors = actors_in(train_idx)
    val_actors = actors_in(val_idx)
    test_actors = actors_in(test_idx)

    assert train_actors & val_actors == set(), "train and val actors overlap"
    assert train_actors & test_actors == set(), "train and test actors overlap"
    assert val_actors & test_actors == set(), "val and test actors overlap"
    assert test_actors == {21, 22, 23, 24}
    assert val_actors == {19, 20}
    assert train_actors == set(range(1, 19))
