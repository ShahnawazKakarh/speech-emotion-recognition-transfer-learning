"""PyTorch Lightning DataModule unifying RAVDESS, MELD, IEMOCAP."""
from __future__ import annotations

from typing import Any

import lightning as L
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.data.base import collate_ser_batch
from src.data.meld import MELDDataset
from src.data.punjabi_rasa import PunjabiRASADataset
from src.data.ravdess import RAVDESSDataset


class SERDataModule(L.LightningDataModule):
    """Wraps dataset-specific loaders behind one Lightning DataModule.

    For RAVDESS, supports two splitting strategies:
      - `random`: stratification-free 70/10/20 (same actor can appear in train and test).
      - `speaker_independent`: actor-disjoint split. This is the published RAVDESS-SER
        convention; expect ~5-10 pp lower numbers than random split.
    """

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
            split_strategy = self.data_cfg.get("split_strategy", "random")

            if split_strategy == "speaker_independent":
                train_idx, val_idx, test_idx = self._speaker_independent_split(full)
            elif split_strategy == "random":
                train_idx, val_idx, test_idx = self._random_split(len(full))
            else:
                raise ValueError(
                    f"Unknown split_strategy: {split_strategy} "
                    "(expected 'random' or 'speaker_independent')"
                )

            self.train_ds = Subset(full, train_idx)
            self.val_ds = Subset(full, val_idx)
            self.test_ds = Subset(full, test_idx)

        elif self.dataset_name == "meld":
            common = {
                "data_dir": self.data_cfg["data_dir"],
                "use_context": self.data_cfg.get("use_context", False),
                "context_window": self.data_cfg.get("context_window", 2),
                "sample_rate": sr,
                "max_audio_seconds": max_sec,
            }
            self.train_ds = MELDDataset(split="train", **common)
            self.val_ds = MELDDataset(split="dev", **common)
            self.test_ds = MELDDataset(split="test", **common)

        elif self.dataset_name == "iemocap":
            raise NotImplementedError(
                "IEMOCAP requires license access — see src/data/iemocap.py"
            )
        elif self.dataset_name == "punjabi_rasa":
            # RASA ships pre-split train/test. We carve a stratified val split
            # from train (10%) since the official release has no val partition.
            common = {
                "data_dir": self.data_cfg["data_dir"],
                "sample_rate": sr,
                "max_audio_seconds": max_sec,
                "emotions": self.data_cfg.get("emotions"),
                "return_text": self.data_cfg.get("return_text", False),
            }
            full_train = PunjabiRASADataset(split="train", **common)
            self.test_ds = PunjabiRASADataset(split="test", **common)

            # Stratified 90/10 train/val split using labels
            labels = np.array([s[1] for s in full_train.samples])
            indices = np.arange(len(full_train))
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.1,
                stratify=labels,
                random_state=42,
            )
            self.train_ds = Subset(full_train, train_idx.tolist())
            self.val_ds = Subset(full_train, val_idx.tolist())
            print(
                f"[punjabi_rasa] train={len(self.train_ds)}  val={len(self.val_ds)}  "
                f"test={len(self.test_ds)}"
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    # ---- RAVDESS split helpers -------------------------------------------------

    def _random_split(self, n: int) -> tuple[list[int], list[int], list[int]]:
        """Random 70/10/20 split (actors can overlap between sets — produces inflated metrics)."""
        indices = np.arange(n)
        test_size = self.data_cfg.get("test_size", 0.2)
        val_size = self.data_cfg.get("val_size", 0.1)
        seed = self.data_cfg.get("split_seed", 42)

        train_val, test = train_test_split(indices, test_size=test_size, random_state=seed)
        train, val = train_test_split(
            train_val, test_size=val_size / (1 - test_size), random_state=seed
        )
        return train.tolist(), val.tolist(), test.tolist()

    def _speaker_independent_split(
        self, full: RAVDESSDataset
    ) -> tuple[list[int], list[int], list[int]]:
        """Actor-disjoint split per the published RAVDESS-SER convention.

        Default split:
          - test  : actors 21, 22, 23, 24    (4 actors → ~16.7% of data)
          - val   : actors 19, 20             (2 actors → ~8.3%)
          - train : actors 1-18               (18 actors → ~75%)

        Configurable via ``data.test_actors`` and ``data.val_actors`` (lists of actor IDs as ints).
        """
        # Parse actor ID from each filename and bucket indices
        actor_to_indices: dict[int, list[int]] = {}
        for i, path in enumerate(full.files):
            actor = int(RAVDESSDataset._parse_filename(path)["actor"])
            actor_to_indices.setdefault(actor, []).append(i)

        test_actors = set(self.data_cfg.get("test_actors", [21, 22, 23, 24]))
        val_actors = set(self.data_cfg.get("val_actors", [19, 20]))
        train_actors = set(actor_to_indices.keys()) - test_actors - val_actors

        # Safety check: no overlap
        assert not (test_actors & val_actors), "test_actors and val_actors overlap"
        assert not (test_actors & train_actors), "test_actors overlap with train"
        assert not (val_actors & train_actors), "val_actors overlap with train"

        train_idx = [i for a in train_actors for i in actor_to_indices[a]]
        val_idx = [i for a in val_actors for i in actor_to_indices.get(a, [])]
        test_idx = [i for a in test_actors for i in actor_to_indices.get(a, [])]

        print(
            f"[datamodule] speaker-independent split: "
            f"train actors={sorted(train_actors)} ({len(train_idx)} samples), "
            f"val actors={sorted(val_actors)} ({len(val_idx)} samples), "
            f"test actors={sorted(test_actors)} ({len(test_idx)} samples)"
        )
        return train_idx, val_idx, test_idx

    # ---- Dataloaders -----------------------------------------------------------

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
