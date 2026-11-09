"""
src/models.py
-------------
All model definitions for EuroSAT CW1:
  - Baseline: HOG + colour histogram feature extractor
  - PyTorch MLP (2 hidden layers)
  - PyTorch CNN (3 conv blocks)

Usage:
    from src.models import extract_features, make_mlp, make_cnn
"""

import numpy as np
import torch
import torch.nn as nn
from skimage.feature import hog


# ==============================================================================
# T3a — Baseline feature extractor: HOG + colour histogram
# ==============================================================================

def extract_features(img):
    """Extract HOG + normalised colour histogram from one image.

    Parameters
    ----------
    img : np.ndarray, shape (64, 64, 3), dtype uint8
        A single EuroSAT RGB image.

    Returns
    -------
    features : np.ndarray, shape (feature_dim,), dtype float32
        Concatenated HOG features and colour histogram (normalised).

    Design notes
    ------------
    - HOG (Histogram of Oriented Gradients): captures gradient directions
      and magnitudes — good at encoding shape/edge structure.
    - Colour histogram: captures spectral distribution per channel.
      Normalised so that relative colour proportions are preserved
      regardless of image brightness.
    - Preprocessing: grayscale for HOG (colour is captured by histogram).
      StandardScaler applied separately in the training script (fit on
      train split only).
    """
    # HOG on grayscale
    gray = img.mean(axis=2).astype(np.float32)   # (64, 64)
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # Colour histogram: 8 bins per RGB channel
    ch_parts = []
    for c in range(3):
        hist, _ = np.histogram(img[..., c], bins=8, range=(0, 256))
        ch_parts.append(hist)
    ch = np.concatenate(ch_parts).astype(np.float32)
    ch = ch / (ch.sum() + 1e-12)   # normalise to relative proportions

    return np.concatenate([hog_features, ch]).astype(np.float32)


def extract_feature_matrix(images):
    """Extract features for a batch of images.

    Parameters
    ----------
    images : np.ndarray, shape (N, 64, 64, 3), dtype uint8

    Returns
    -------
    F : np.ndarray, shape (N, feature_dim), dtype float32
    """
    return np.stack([extract_features(img) for img in images])


# ==============================================================================
# T3b — Neural Model 1: PyTorch MLP
# ==============================================================================

def make_mlp(dropout=0.3):
    """Two-hidden-layer fully connected MLP.

    Architecture
    ------------
    Flatten(12288) → Linear(512) → ReLU → Dropout → Linear(256) → ReLU → Dropout → Linear(10)

    Design choices
    --------------
    - Input 64×64×3 = 12,288 flattened pixels — treats each pixel independently.
    - ReLU activation: avoids saturating gradients (vs sigmoid/tanh).
    - Dropout: regularisation against overfitting on the larger parameter count.
    - Output: raw logits; softmax is applied inside CrossEntropyLoss.
    - Parameters: ~6.4 million — large because each pixel connected to every neuron.

    Parameters
    ----------
    dropout : float
        Dropout probability (default 0.3).

    Returns
    -------
    model : nn.Sequential
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 64 * 3, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, 10)
    )


# ==============================================================================
# T3b — Neural Model 2: PyTorch CNN
# ==============================================================================

def _conv_block(in_channels, out_channels):
    """Single CNN block: Conv3×3 → BatchNorm → ReLU → MaxPool2d.

    Design rationale
    ----------------
    - Conv3×3 with padding=1: preserves spatial size before pooling.
    - BatchNorm: stabilises training, reduces sensitivity to lr/init.
    - ReLU: fast, avoids vanishing gradients.
    - MaxPool2d(2): halves spatial dimensions, adds translation invariance.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2)
    )


def make_cnn(dropout=0.3):
    """3-block small CNN for EuroSAT 64×64 RGB classification.

    Architecture
    ------------
    Block 1: Conv(3→32)  + BN + ReLU + Pool  →  (32, 32, 32)
    Block 2: Conv(32→64) + BN + ReLU + Pool  →  (64, 16, 16)
    Block 3: Conv(64→128)+ BN + ReLU + Pool  →  (128, 8, 8)
    AdaptiveAvgPool(1)                        →  (128, 1, 1)
    Flatten                                   →  (128,)
    Dropout → Linear(128→10)

    Design choices
    --------------
    - 3 conv blocks: shallow layers learn edges, deeper layers learn textures
      and semantic structures (hierarchical feature learning).
    - AdaptiveAvgPool instead of flatten after conv: makes the model
      resolution-agnostic and dramatically cuts parameters vs full flatten.
    - Parameters: ~94,730 — far fewer than the MLP (~6.4M).
    - Dropout before the final linear layer for regularisation.

    Parameters
    ----------
    dropout : float
        Dropout probability on the classifier head (default 0.3).
        Set to 0.0 for the ablation study (cnn_ablation.yaml).

    Returns
    -------
    model : nn.Sequential
    """
    return nn.Sequential(
        _conv_block(3, 32),     # 64×64 → 32×32
        _conv_block(32, 64),    # 32×32 → 16×16
        _conv_block(64, 128),   # 16×16 → 8×8
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(128, 10)
    )


def count_parameters(model):
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

