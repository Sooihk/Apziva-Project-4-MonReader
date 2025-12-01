from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from MonReader.config import Config
from MonReader.data.transforms import build_transforms


@dataclass
class DataModule:
    """
    Thin wrapper around ImageFolder + Stratified train/val split.
    folder structure:
        data/raw/training/
            flip/
            notflip/
        data/raw/testing/
            flip/
            notflip/
    """
    cfg: Config

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Return (train_loader, val_loader) with an 80/20 stratified split of training data.
        """
        train_tfm, eval_tfm = build_transforms(self.cfg.img_size)

        # Two copies of the same underlying samples, different transforms
        train_full = datasets.ImageFolder(str(self.cfg.train_dir), transform=train_tfm)
        val_full   = datasets.ImageFolder(str(self.cfg.train_dir), transform=eval_tfm)

        # Extract labels for stratified split
        labels = np.array([y for _, y in train_full.samples])

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=self.cfg.seed,
        )
        (train_idx, val_idx), = splitter.split(
            np.zeros(len(labels)),  # dummy X
            labels,
        )

        train_ds = Subset(train_full, train_idx)
        val_ds   = Subset(val_full, val_idx)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

        return train_loader, val_loader

    def get_test_loader(self, batch_size: Optional[int] = None) -> DataLoader:
        """
        Build a deterministic test loader from data/raw/testing.
        """
        _, eval_tfm = build_transforms(self.cfg.img_size)

        test_ds = datasets.ImageFolder(str(self.cfg.test_dir), transform=eval_tfm)

        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size or self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        return test_loader