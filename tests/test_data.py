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
