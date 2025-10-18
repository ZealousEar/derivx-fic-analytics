from __future__ import annotations

import numpy as np


EPS = 1e-12


def as_array(x) -> np.ndarray:
    """Return input as float64 numpy array.

    Parameters
    ----------
    x : Any
        Scalar or array-like.

    Returns
    -------
    np.ndarray
        Float64 array view of the input.
    """

    return np.asarray(x, dtype=float)


def clamp_positive(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Clamp values to be >= eps.

    Ensures strictly positive inputs for log/ratio operations.
    """

    return np.maximum(as_array(x), eps)


def clamp_min(x: np.ndarray, min_val: float) -> np.ndarray:
    """Clamp values to be >= min_val."""

    return np.maximum(as_array(x), min_val)


def ensure_broadcastable(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Broadcast inputs to a common shape.

    Returns broadcasted views without copying when possible.
    """

    return np.broadcast_arrays(*[as_array(a) for a in arrays])


