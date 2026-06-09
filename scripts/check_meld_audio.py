#!/usr/bin/env python3
"""Quick scan: find missing or empty .wav files referenced by MELD CSVs.

Run from repo root:
    python scripts/check_meld_audio.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/meld")
SPLITS = {
    "train": ("train_sent_emo.csv", "train_audio"),
    "dev":   ("dev_sent_emo.csv",   "dev_audio"),
    "test":  ("test_sent_emo.csv",  "test_audio"),
}

for split, (csv_name, audio_subdir) in SPLITS.items():
    df = pd.read_csv(DATA_DIR / csv_name)
    audio_dir = DATA_DIR / audio_subdir

    missing, empty, ok = 0, 0, 0
    for _, row in df.iterrows():
        did, uid = int(row["Dialogue_ID"]), int(row["Utterance_ID"])
        path = audio_dir / f"dia{did}_utt{uid}.wav"
        if not path.exists():
            missing += 1
        elif path.stat().st_size == 0:
            empty += 1
        else:
            ok += 1

    total = len(df)
    print(f"[{split:5}]  total={total:5}  ok={ok:5}  missing={missing:4}  empty={empty:4}")
