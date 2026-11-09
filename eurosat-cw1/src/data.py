"""
src/data.py
-----------
EuroSAT dataset loading, normalisation, and augmentation.

Usage:
    from src.data import EuroSATDataset, make_loaders
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class EuroSATDataset(Dataset):
    """PyTorch Dataset wrapper for the pre-built EuroSAT numpy array.

    Parameters
    ----------
    X : np.ndarray, shape (N, 64, 64, 3), dtype uint8
        Raw pixel data for all images.
    y : np.ndarray, shape (N,), dtype int
        Class labels (0–9).
    indices : np.ndarray
        Indices into X and y for this split (train / val / test).
    mean : array-like, length 3
        Per-channel mean computed from the training split only.
    std : array-like, length 3
        Per-channel std computed from the training split only.
    transform : callable, optional
        Torchvision transform applied after normalisation (training augmentation).
    """

    def __init__(self, X, y, indices, mean, std, transform=None):
        self.X = X[indices]           # subset for this split
        self.y = y[indices]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std  = torch.tensor(std,  dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # 1. Load image, convert to float [0, 1]
        img = torch.tensor(self.X[i], dtype=torch.float32) / 255.0  # (64, 64, 3)

        # 2. Per-channel normalisation with TRAIN statistics
        img = (img - self.mean) / self.std

        # 3. HWC -> CHW for PyTorch
        img = img.permute(2, 0, 1)   # (3, 64, 64)

        # 4. Optional augmentation (training split only)
        if self.transform is not None:
            img = self.transform(img)

        return img, int(self.y[i])


# ------------------------------------------------------------------
# Augmentation transforms
# Applied to TRAINING batches only — never to val or test.
# Justified because satellite images have no fixed orientation.
# ------------------------------------------------------------------
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
])


def make_loaders(X, y, train_idx, val_idx, test_idx, mean, std, batch_size=64):
    """Create DataLoader objects for train, val and test splits.

    Parameters
    ----------
    X, y : np.ndarray
        Full dataset arrays (all 27,000 images).
    train_idx, val_idx, test_idx : np.ndarray
        Split indices from splits/split_indices.npz.
    mean, std : list[float]
        Per-channel normalisation stats from splits/preprocessing.json.
    batch_size : int
        Mini-batch size (default 64).

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    train_ds = EuroSATDataset(X, y, train_idx, mean, std, transform=TRAIN_TRANSFORM)
    val_ds   = EuroSATDataset(X, y, val_idx,   mean, std)
    test_ds  = EuroSATDataset(X, y, test_idx,  mean, std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

