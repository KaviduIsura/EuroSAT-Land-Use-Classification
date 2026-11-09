"""
src/evaluate.py
---------------
Evaluation utilities: predictions, metrics, confusion matrix, error analysis,
and stress test for all trained models.

Usage:
    from src.evaluate import get_predictions, compute_metrics, plot_confusion_matrix
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)


# ==============================================================================
# Prediction helpers
# ==============================================================================

def get_predictions(model, loader, device):
    """Run inference on a DataLoader and return true labels, predictions, and confidences.

    Parameters
    ----------
    model  : nn.Module — PyTorch model (MLP or CNN), loaded with best checkpoint
    loader : DataLoader — val or test loader (no augmentation, no shuffle)
    device : torch.device

    Returns
    -------
    y_true : np.ndarray, shape (N,) — ground truth labels
    y_pred : np.ndarray, shape (N,) — predicted class indices
    y_conf : np.ndarray, shape (N,) — softmax confidence of predicted class
    """
    model.eval()
    all_true, all_pred, all_conf = [], [], []

    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.softmax(model(xb.to(device)), dim=1)
            conf, pred = probs.max(dim=1)
            all_true.extend(yb.numpy())
            all_pred.extend(pred.cpu().numpy())
            all_conf.extend(conf.cpu().numpy())

    return np.array(all_true), np.array(all_pred), np.array(all_conf)


# ==============================================================================
# Metric computation
# ==============================================================================

def compute_metrics(y_true, y_pred, classes):
    """Compute primary and complementary metrics.

    Primary metric: macro-F1 (treats all classes equally — appropriate
    because classes are not perfectly balanced).

    Parameters
    ----------
    y_true  : np.ndarray
    y_pred  : np.ndarray
    classes : list[str] — class names

    Returns
    -------
    metrics : dict
        Keys: macro_f1, accuracy, per_class_precision, per_class_recall,
              per_class_f1
    """
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    acc      = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0
    )

    return {
        "macro_f1"            : round(float(macro_f1), 4),
        "accuracy"            : round(float(acc), 4),
        "per_class_precision" : {c: round(float(p), 4) for c, p in zip(classes, prec)},
        "per_class_recall"    : {c: round(float(r), 4) for c, r in zip(classes, rec)},
        "per_class_f1"        : {c: round(float(f), 4) for c, f in zip(classes, f1)},
    }


def print_report(y_true, y_pred, classes):
    """Print sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=classes))


# ==============================================================================
# Confusion matrix plot
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path=None):
    """Plot and optionally save a confusion matrix.

    Parameters
    ----------
    y_true    : np.ndarray
    y_pred    : np.ndarray
    classes   : list[str]
    title     : str
    save_path : str, optional — if given, saves the figure to this path

    Returns
    -------
    fig : matplotlib Figure
    """
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=13)

    # Annotate each cell
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ==============================================================================
# Error analysis
# ==============================================================================

def show_errors(X_test, y_true, y_pred, y_conf, classes, n=5, save_path=None):
    """Display misclassified test images with true label, predicted label, and confidence.

    Parameters
    ----------
    X_test    : np.ndarray, shape (N_test, 64, 64, 3), dtype uint8
                Images in the test split (unnormalised, for display).
    y_true    : np.ndarray
    y_pred    : np.ndarray
    y_conf    : np.ndarray
    classes   : list[str]
    n         : int — number of error examples to show (default 5)
    save_path : str, optional
    """
    wrong = np.where(y_true != y_pred)[0]
    sample = wrong[:n]

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for ax, i in zip(axes, sample):
        ax.imshow(X_test[i])
        ax.axis("off")
        ax.set_title(
            f"True : {classes[y_true[i]]}\n"
            f"Pred : {classes[y_pred[i]]}\n"
            f"Conf : {y_conf[i]:.2f}",
            fontsize=8,
        )
    plt.suptitle(f"Misclassified Test Examples (first {n})", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig


# ==============================================================================
# Stress test: Gaussian noise robustness
# ==============================================================================

def stress_test_noise(model, loader, device, sigmas=(0.0, 0.05, 0.10, 0.20)):
    """Evaluate model under Gaussian noise at multiple sigma levels.

    Protocol (written before running):
    - Add zero-mean Gaussian noise to normalised test tensors.
    - Sigma values: 0.0 (clean), 0.05, 0.10, 0.20.
    - Report macro-F1 at each level.
    - Limitation: tests sensitivity to synthetic noise only — does NOT
      represent real atmospheric effects (cloud, haze, sensor noise).

    Parameters
    ----------
    model   : nn.Module — trained PyTorch model
    loader  : DataLoader — test loader
    device  : torch.device
    sigmas  : tuple[float] — noise levels to test

    Returns
    -------
    results : list[dict] — each with keys 'sigma' and 'macro_f1'
    """
    model.eval()
    results = []

    for sigma in sigmas:
        all_pred, all_true = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb_noisy = xb + sigma * torch.randn_like(xb)
                pred = model(xb_noisy.to(device)).argmax(dim=1).cpu().numpy()
                all_pred.extend(pred)
                all_true.extend(yb.numpy())

        f1 = f1_score(all_true, all_pred, average="macro")
        results.append({"sigma": sigma, "macro_f1": round(float(f1), 4)})
        print(f"  sigma={sigma:.2f} | macro-F1 = {f1:.4f}")

    return results


def plot_stress_test(results, save_path=None):
    """Plot macro-F1 vs Gaussian noise sigma.

    Parameters
    ----------
    results   : list[dict] — output of stress_test_noise()
    save_path : str, optional
    """
    sigmas = [r["sigma"] for r in results]
    f1s    = [r["macro_f1"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sigmas, f1s, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xlabel("Gaussian Noise σ", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title("Robustness Stress Test — Gaussian Noise", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    return fig

