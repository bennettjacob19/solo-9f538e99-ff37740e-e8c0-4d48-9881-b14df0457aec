"""Shared logistic-regression math (pure NumPy, float64).

Model convention
----------------
A single sample is represented as augmented vector ``[1.0, x1, x2]`` so that
the bias is folded into the weight vector ``w`` of length 3::

    z   = X_aug @ w
    p   = sigmoid(z)
    BCE = mean( -y*log(p) - (1-y)*log(1-p) )
    grad = X_aug.T @ (p - y) / n          # full-data average gradient

Update (classical momentum SGD, both buffers start at zero)::

    v <- momentum * v - lr * grad
    w <- w + v

Determinism: every operation uses float64 and fixed summation order. There is
no RNG anywhere, so the same inputs always produce bit-identical outputs.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

# Fixed hyper-parameters mandated by the task.
LEARNING_RATE: float = 0.1
MOMENTUM: float = 0.9
N_WEIGHTS: int = 3  # bias + 2 numeric features


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable element-wise logistic function (float64)."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def augment(x: np.ndarray) -> np.ndarray:
    """Prepend a constant 1 column for the bias term."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    ones = np.ones((x.shape[0], 1), dtype=np.float64)
    return np.concatenate([ones, x], axis=1)


def probabilities(x_aug: np.ndarray, w: Sequence[float]) -> np.ndarray:
    return sigmoid(x_aug @ np.asarray(w, dtype=np.float64))


def bce_loss(x_aug: np.ndarray, y: np.ndarray, w: Sequence[float]) -> float:
    """Mean binary cross-entropy over a shard or the full dataset."""
    p = probabilities(x_aug, w)
    eps = 0.0
    # Clip only to guard log(0); stable sigmoid keeps p inside (0, 1).
    p = np.clip(p, 1e-300, 1.0 - 1e-300)
    total = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    return float(total / x_aug.shape[0])


def shard_residuals(
    x_aug: np.ndarray, y: np.ndarray, w: Sequence[float]
) -> Tuple[np.ndarray, float]:
    """Per-sample prediction residuals ``p - y`` and the shard's mean loss.

    Residuals are the only shard-local quantity needed for a gradient. The
    coordinator concatenates residuals from both shards in original dataset
    row order and runs *one* matrix product, which reproduces the single
    process full-data gradient in the exact same summation order (bit
    identical, no shard-weighted re-summation).
    """
    p = probabilities(x_aug, w)
    loss = bce_loss(x_aug, y, w)
    return (p - y), float(loss)


def shard_gradient(
    x_aug: np.ndarray, y: np.ndarray, w: Sequence[float]
) -> Tuple[np.ndarray, float]:
    """Average gradient and mean loss over a shard or the full dataset."""
    n = x_aug.shape[0]
    p = probabilities(x_aug, w)
    grad = (x_aug.T @ (p - y)) / n
    loss = bce_loss(x_aug, y, w)
    return grad, float(loss)


def full_average_gradient(
    x_aug: np.ndarray, residuals: np.ndarray
) -> np.ndarray:
    """Full-data average gradient from concatenated per-sample residuals.

    ``x_aug`` is the augmented full dataset in canonical row order and
    ``residuals`` is the concatenation of worker residuals in the same order,
    so this is operation-for-operation identical to the single-process trainer.
    """
    n = x_aug.shape[0]
    return (x_aug.T @ np.asarray(residuals, dtype=np.float64)) / n


def momentum_update(
    w: Sequence[float],
    v: Sequence[float],
    grad: Sequence[float],
    lr: float = LEARNING_RATE,
    momentum: float = MOMENTUM,
) -> Tuple[np.ndarray, np.ndarray]:
    """One synchronous momentum step. Returns (new_w, new_v)."""
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    grad = np.asarray(grad, dtype=np.float64)
    new_v = momentum * v - lr * grad
    new_w = w + new_v
    return new_w, new_v


def zero_buffers() -> np.ndarray:
    """Weights and momentum both start from zero."""
    return np.zeros(N_WEIGHTS, dtype=np.float64)


def reference_train(
    x: np.ndarray,
    y: np.ndarray,
    rounds: int,
    lr: float = LEARNING_RATE,
    momentum: float = MOMENTUM,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """Single-process reference trainer over the *full* dataset.

    Used by the test-suite to cross-check the distributed two-worker
    implementation. Returns histories of weights, momentum and loss
    (loss history includes the round-0 initial loss).
    """
    x_aug = augment(x)
    w = zero_buffers()
    v = zero_buffers()
    losses = [bce_loss(x_aug, y, w)]
    ws: List[np.ndarray] = [w.copy()]
    vs: List[np.ndarray] = [v.copy()]
    for _ in range(rounds):
        grad, _ = shard_gradient(x_aug, y, w)
        w, v = momentum_update(w, v, grad, lr, momentum)
        losses.append(bce_loss(x_aug, y, w))
        ws.append(w.copy())
        vs.append(v.copy())
    return ws, vs, losses


def to_list(a) -> List[float]:
    return [float(x) for x in np.asarray(a, dtype=np.float64)]


def isnan_guard(values: Sequence[float]) -> bool:
    return any(math.isnan(float(x)) or math.isinf(float(x)) for x in values)
