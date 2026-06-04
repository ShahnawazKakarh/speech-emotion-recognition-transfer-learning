"""PyTorch Lightning DataModule unifying RAVDESS, MELD, IEMOCAP."""
from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.base import collate_ser_batch
from src.data.meld import MELDDataset
from src.data.ravdess import RAVDESSDataset


class SERDataModule(L.LightningDataModule):
    """Wraps dataset-specific loaders behind one Lightning DataModule."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg["dataset"]
        self.data_cfg = cfg["data"]
        self.batch_size = self.data_cfg["batch_size"]
        self.num_workers = self.data_cfg.get("num_workers", 4)

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: str | None = None) -> None:
        sr = self.data_cfg.get("sample_rate", 16000)
        max_sec = self.data_cfg.get("max_audio_seconds", 8.0)

        if self.dataset_name == "ravdess":
            full = RAVDESSDataset(
                data_dir=self.data_cfg["data_dir"],
                sample_rate=sr,
                max_audio_seconds=max_sec,
            )
            n = len(full)
            indices = np.arange(n)
            # Stratified split by actor would be ideal; here we use a simple seeded split.
            train_val, test = train_test_split(
                indices,
                test_size=self.data_cfg.get("test_size", 0.2),
                random_state=self.data_cfg.get("split_seed", 42),
            )
            train, val = train_test_split(
                train_val,
                test_size=self.data_cfg.get("val_size", 0.1) / (1 - self.data_cfg.get("test_size", 0.2)),
                random_state=self.data_cfg.get("split_seed", 42),
            )
            self.train_ds = Subset(full, train.tolist())
            self.val_ds = Subset(full, val.tolist())
            self.test_ds = Subset(full, test.tolist())

        elif self.dataset_name == "meld":
            common = dict(
                data_dir=self.data_cfg["data_dir"],
                use_context=self.data_cfg.get("use_context", False),
                context_window=self.data_cfg.get("context_window", 2),
                sample_rate=sr,
                max_audio_seconds=max_sec,
            )
            self.train_ds = MELDDataset(split="train", **common)
            self.val_ds = MELDDataset(split="dev", **common)
            self.test_ds = MELDDataset(split="test", **common)

        elif self.dataset_name == "iemocap":
            raise NotImplementedError(
                "IEMOCAP requires license access — see src/data/iemocap.py"
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=collate_ser_batch,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
