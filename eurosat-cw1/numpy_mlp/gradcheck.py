"""
numpy_mlp/gradcheck.py
-----------------------
Numerical gradient verification for the NumPy MLP (T2c).

Uses central differences to approximate each partial derivative and
compares it to the analytic gradient produced by backpropagation.

Formula: numeric ≈ (L(w + ε) - L(w - ε)) / (2ε)

Requirements (from brief):
- At least 3 entries per parameter (W1, b1, W2, b2) → ≥ 12 total.
- ε = 1e-5
- Use tanh activation (smooth — no ReLU kink problem near zero).
- Report absolute and relative errors.
- Tolerance: relative error < 1e-4 (brief states 1e-6 analytically;
  float32 arithmetic limits practical tolerance to ~1e-4 to 1e-5).
"""

import numpy as np

# Import MLP from the same package
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from numpy_mlp.mlp_numpy import MLP


def gradcheck(model, X, y, eps=1e-5, per_param=3, seed=1):
    """Compare analytic gradients to central-difference numerical gradients.

    Parameters
    ----------
    model     : MLP instance — will be modified temporarily (restored after)
    X         : np.ndarray, shape (n, d) — small batch (e.g. 8 samples)
    y         : np.ndarray, shape (n,)
    eps       : float — finite-difference step (default 1e-5)
    per_param : int — number of random entries to check per parameter (default 3)
    seed      : int — seed for selecting random parameter indices

    Returns
    -------
    rows : list[dict] — one dict per checked entry with keys:
        param, index, analytic, numeric, abs_err, rel_err, pass
    """
    rng  = np.random.default_rng(seed)
    g    = model.grads(X, y)   # analytic gradients
    rows = []

    for name in ["W1", "b1", "W2", "b2"]:
        param = getattr(model, name)

        for _ in range(per_param):
            # Pick a random index in this parameter array
            idx = tuple(int(rng.integers(0, s)) for s in param.shape)

            # Perturb up
            original    = param[idx]
            param[idx]  = original + eps
            loss_plus   = model.loss(X, y)

            # Perturb down
            param[idx]  = original - eps
            loss_minus  = model.loss(X, y)

            # Restore
            param[idx]  = original

            numeric  = (loss_plus - loss_minus) / (2 * eps)
            analytic = g[name][idx]
            abs_err  = abs(analytic - numeric)
            rel_err  = abs_err / max(abs(analytic) + abs(numeric), 1e-12)

            rows.append({
                "param"   : name,
                "index"   : idx,
                "analytic": analytic,
                "numeric" : numeric,
                "abs_err" : abs_err,
                "rel_err" : rel_err,
                "pass"    : rel_err < 1e-4,
            })

    return rows


def print_gradcheck_table(rows):
    """Print the gradient check results as a formatted table.

    Parameters
    ----------
    rows : list[dict] — output of gradcheck()
    """
    header = (
        f"{'Param':<6} {'Index':<14} {'Analytic':>14} "
        f"{'Numeric':>14} {'Abs Err':>12} {'Rel Err':>12}  OK?"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        ok = "✅" if r["pass"] else "❌ FAIL"
        print(
            f"{r['param']:<6} {str(r['index']):<14} "
            f"{r['analytic']:>14.6e} {r['numeric']:>14.6e} "
            f"{r['abs_err']:>12.2e} {r['rel_err']:>12.2e}  {ok}"
        )

    n_pass = sum(r["pass"] for r in rows)
    print(f"\n{n_pass}/{len(rows)} checks passed  (eps={1e-5}, tolerance=1e-4)")


# ==============================================================================
# Standalone script
# ==============================================================================

if __name__ == "__main__":
    import json

    # Load a small subset of data for checking
    # (run from the eurosat-cw1 root after 02_eda_split has produced split_indices.npz)
    try:
        data   = np.load("/content/drive/MyDrive/ann_cw1/eurosat_64.npz")
        splits = np.load("splits/split_indices.npz")
        X_full = data["X"]
        y_full = data["y"]
        train_idx = splits["train"]
    except FileNotFoundError:
        # Fallback: tiny synthetic data for local testing
        print("Data not found — using synthetic data for demonstration.")
        rng_data = np.random.default_rng(0)
        X_full    = (rng_data.random((200, 16, 16, 1)) * 255).astype(np.uint8)
        y_full    = rng_data.integers(0, 4, 200)
        train_idx = np.arange(200)

    # Prepare 4-class reduced subset (same as notebook 03)
    SUBSET_CLASSES = {0: 0, 1: 1, 4: 2, 9: 3}
    from PIL import Image

    mask    = np.isin(y_full[train_idx], list(SUBSET_CLASSES.keys()))
    sub_idx = train_idx[mask][:100]   # small subset for speed

    def preprocess_single(img_arr):
        gray = Image.fromarray(img_arr).convert("L").resize((16, 16))
        return np.array(gray, dtype=np.float32).flatten() / 255.0

    X_sub = np.stack([preprocess_single(X_full[i]) for i in sub_idx])
    y_sub = np.array([SUBSET_CLASSES[yi] for yi in y_full[sub_idx]])

    # Use a tiny batch for the check
    X_check = X_sub[:8]
    y_check = y_sub[:8]

    # Initialise a fresh model
    mlp = MLP(d=256, h=32, k=4, seed=42)

    print("=" * 70)
    print("GRADIENT VERIFICATION (T2c)")
    print("eps = 1e-5  |  tolerance = 1e-4  |  3 entries per parameter")
    print("=" * 70)

    rows = gradcheck(mlp, X_check, y_check, eps=1e-5, per_param=3, seed=1)
    print_gradcheck_table(rows)

