"""
numpy_mlp/mlp_numpy.py
-----------------------
One-hidden-layer MLP implemented entirely in NumPy — no autograd.

T2 requirement: manual forward pass, manual backpropagation,
                mini-batch SGD, and training loop.

You must be able to explain every line of this file in the demo.
"""

import numpy as np


class MLP:
    """One-hidden-layer MLP in pure NumPy.

    Architecture
    ------------
    Input X: (n, d)
    W1: (d, h),  b1: (h,)   — first layer weights and biases
    Hidden H: (n, h)         — tanh activations
    W2: (h, k),  b2: (k,)   — second layer weights and biases
    Output P: (n, k)         — softmax probabilities
    Loss: mean cross-entropy

    Biological analogy
    ------------------
    - Inputs X resemble dendritic signals arriving at a neuron.
    - Weights W1 and W2 resemble synaptic strengths that scale each signal.
    - The weighted sum X @ W1 + b1 resembles the membrane potential
      accumulating inside the neuron.
    - The tanh activation resembles the neuron's firing response —
      compressing the potential to a bounded output in (-1, 1).

    Two limits vs real neurons:
    1. Real neurons communicate via discrete spikes with timing (rate coding),
       not continuous real-valued activations.
    2. Backpropagation has no known biological equivalent; real learning
       mechanisms are not fully understood.

    Parameters
    ----------
    d    : int — input dimensionality (256 for 16×16 grayscale)
    h    : int — number of hidden units (e.g. 32)
    k    : int — number of output classes (4 for the reduced subset)
    seed : int — random seed for reproducibility
    """

    def __init__(self, d, h, k, seed=0):
        rng = np.random.default_rng(seed)
        # Initialise with scale = sqrt(1 / fan_in) — reduces saturating
        # tanh activations at the start of training.
        self.W1 = rng.normal(0, np.sqrt(1 / d), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, np.sqrt(1 / h), (h, k))
        self.b2 = np.zeros(k)

        # Cache populated by forward()
        self.X  = None   # input batch
        self.H  = None   # hidden activations
        self.P  = None   # softmax probabilities

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, X):
        """Compute softmax probabilities for input batch X.

        Parameters
        ----------
        X : np.ndarray, shape (n, d) — input batch (pixel values in [0,1])

        Returns
        -------
        P : np.ndarray, shape (n, k) — softmax probability distributions

        Steps
        -----
        1. Linear transform: Z1 = X @ W1 + b1       shape (n, h)
        2. Activation: H = tanh(Z1)                  shape (n, h)
        3. Linear transform: Z2 = H @ W2 + b2        shape (n, k)
        4. Numerical stability: subtract row max before exp (log-sum trick)
        5. Softmax: P = exp(Z2) / sum(exp(Z2), axis=1)
        """
        self.X = X                                        # cache for backprop
        Z1     = X @ self.W1 + self.b1
        self.H = np.tanh(Z1)                              # tanh activation
        Z2     = self.H @ self.W2 + self.b2
        Z2     = Z2 - Z2.max(axis=1, keepdims=True)       # numerical stability
        E      = np.exp(Z2)
        self.P = E / E.sum(axis=1, keepdims=True)         # softmax
        return self.P

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, X, y):
        """Mean cross-entropy loss.

        CE = -mean( log( P[i, y[i]] ) )

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,) — integer class labels

        Returns
        -------
        loss : float — mean cross-entropy over the batch
        """
        P = self.forward(X)
        # Clip with 1e-12 to avoid log(0)
        return -np.log(P[np.arange(len(y)), y] + 1e-12).mean()

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def grads(self, X, y):
        """Compute analytic gradients via backpropagation.

        Derivation
        ----------
        Let n = batch size.

        dL/dZ2  = (P - one_hot(y)) / n       — softmax + CE combined derivative
        dL/dW2  = H.T @ dZ2                   — chain rule
        dL/db2  = sum(dZ2, axis=0)
        dL/dH   = dZ2 @ W2.T                  — back through second linear layer
        dL/dZ1  = dL/dH * (1 - H^2)          — tanh derivative: d/dx tanh(x) = 1 - tanh^2(x)
        dL/dW1  = X.T @ dZ1                   — back through first linear layer
        dL/db1  = sum(dZ1, axis=0)

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
        y : np.ndarray, shape (n,) — integer labels

        Returns
        -------
        grads : dict — keys 'W1', 'b1', 'W2', 'b2'
        """
        n  = len(y)
        P  = self.forward(X).copy()

        # dL/dZ2: subtract 1 from the true-class probability (one-hot trick)
        P[np.arange(n), y] -= 1
        dZ2 = P / n

        # Second layer gradients
        dW2 = self.H.T @ dZ2
        db2 = dZ2.sum(axis=0)

        # Back through second linear layer and tanh
        dH  = dZ2 @ self.W2.T
        dZ1 = dH * (1 - self.H ** 2)   # tanh derivative

        # First layer gradients
        dW1 = self.X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ------------------------------------------------------------------
    # Parameter update (gradient descent step)
    # ------------------------------------------------------------------

    def step(self, grads, lr):
        """Apply one gradient descent step.

        Parameters
        ----------
        grads : dict — from self.grads()
        lr    : float — learning rate
        """
        for key in grads:
            setattr(self, key, getattr(self, key) - lr * grads[key])


# ==============================================================================
# Training loop
# ==============================================================================

def train(model, X, y, lr=0.1, epochs=200, batch_size=64, seed=0):
    """Mini-batch SGD training loop.

    Parameters
    ----------
    model      : MLP instance
    X          : np.ndarray, shape (n, d) — training features
    y          : np.ndarray, shape (n,) — training labels
    lr         : float — learning rate (default 0.1)
    epochs     : int — number of full passes over data (default 200)
    batch_size : int — mini-batch size (default 64)
    seed       : int — seed for shuffling order

    Returns
    -------
    loss_curve : list[float] — training loss at the end of each epoch
    """
    rng        = np.random.default_rng(seed)
    loss_curve = []

    for epoch in range(epochs):
        # Shuffle training indices each epoch
        order = rng.permutation(len(y))

        # Mini-batch updates
        for i in range(0, len(y), batch_size):
            batch = order[i: i + batch_size]
            g     = model.grads(X[batch], y[batch])
            model.step(g, lr)

        # Record full-dataset loss at end of epoch
        loss_curve.append(model.loss(X, y))

    return loss_curve

