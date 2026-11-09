"""
src/infer.py
------------
Inference pipeline for EuroSAT land-use classification.

Loads a trained CNN checkpoint and preprocessing statistics, then
predicts the land-use class and confidence for a given image file.

Usage (command line):
    python src/infer.py --image path/to/image.jpg \
                        --checkpoint checkpoints/cnn_seed0.pt \
                        --preprocessing splits/preprocessing.json

Usage (import):
    from src.infer import load_model, predict
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


# ==============================================================================
# Model definition (duplicated here so infer.py is self-contained)
# ==============================================================================

def _conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


def _make_cnn(dropout=0.3):
    return nn.Sequential(
        _conv_block(3, 32),
        _conv_block(32, 64),
        _conv_block(64, 128),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(128, 10),
    )


# ==============================================================================
# Loading utilities
# ==============================================================================

def load_model(checkpoint_path, device=None):
    """Load a trained CNN from a checkpoint file.

    Parameters
    ----------
    checkpoint_path : str — path to .pt checkpoint saved by train.py
    device          : torch.device, optional — defaults to CPU

    Returns
    -------
    model : nn.Module — model in eval() mode
    """
    if device is None:
        device = torch.device("cpu")

    model = _make_cnn(dropout=0.0)   # dropout=0 at inference (eval mode disables it anyway)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_preprocessing(json_path):
    """Load normalisation statistics and class names from preprocessing.json.

    Parameters
    ----------
    json_path : str — path to splits/preprocessing.json

    Returns
    -------
    mean    : np.ndarray, shape (3,)
    std     : np.ndarray, shape (3,)
    classes : list[str]
    """
    with open(json_path) as f:
        prep = json.load(f)
    return np.array(prep["mean"], dtype=np.float32), \
           np.array(prep["std"],  dtype=np.float32), \
           prep["classes"]


# ==============================================================================
# Inference function
# ==============================================================================

def predict(image_path, model, mean, std, classes, threshold=0.6, device=None):
    """Predict the land-use class of a single image.

    Parameters
    ----------
    image_path : str — path to image file (any PIL-readable format)
    model      : nn.Module — loaded model in eval() mode
    mean, std  : np.ndarray — per-channel normalisation stats (from train split)
    classes    : list[str] — class name list (10 EuroSAT classes)
    threshold  : float — confidence threshold below which a warning is returned.
                 Choose this value using VALIDATION data only.
    device     : torch.device, optional

    Returns
    -------
    result : dict
        Normal case    → {'prediction': str, 'confidence': float}
        Low confidence → {'warning': str, 'best_guess': str, 'confidence': float, 'note': str}
        Error          → {'error': str}

    Notes
    -----
    Softmax confidence is an imperfect out-of-distribution (OOD) detector.
    Neural networks can be confidently wrong on inputs outside the training
    distribution. Do not treat low confidence as a reliable OOD guarantee.
    """
    if device is None:
        device = torch.device("cpu")

    # 1. Open and validate image
    try:
        img = Image.open(image_path).convert("RGB").resize((64, 64))
    except Exception as e:
        return {"error": f"Cannot read image file: {e}"}

    # 2. Preprocess: normalise with train statistics
    x = np.asarray(img, dtype=np.float32) / 255.0   # [0, 1]
    x = (x - mean) / std                              # normalised

    # 3. Convert to tensor: HWC → CHW, add batch dimension
    x_tensor = torch.tensor(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    # 4. Inference
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(x_tensor), dim=1)[0]   # (10,)

    conf, k = probs.max(0)
    conf_val = conf.item()

    # 5. OOD check using confidence threshold (threshold set on val data)
    if conf_val < threshold:
        return {
            "warning"    : "Low confidence — image may be out of scope.",
            "best_guess" : classes[k],
            "confidence" : round(conf_val, 3),
            "note"       : (
                "Softmax confidence is imperfect for OOD detection. "
                "Networks can be confidently wrong on unfamiliar inputs."
            ),
        }

    return {
        "prediction" : classes[k],
        "confidence" : round(conf_val, 3),
    }


# ==============================================================================
# Command-line interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EuroSAT inference — predict land-use class for an image."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/cnn_seed0.pt",
        help="Path to the model checkpoint (.pt). Default: checkpoints/cnn_seed0.pt",
    )
    parser.add_argument(
        "--preprocessing",
        type=str,
        default="splits/preprocessing.json",
        help="Path to preprocessing.json. Default: splits/preprocessing.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for OOD warning (default 0.6, chosen on val data).",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load
    mean, std, classes = load_preprocessing(args.preprocessing)
    model = load_model(args.checkpoint, device)

    # Predict
    result = predict(args.image, model, mean, std, classes, args.threshold, device)

    # Print result
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

