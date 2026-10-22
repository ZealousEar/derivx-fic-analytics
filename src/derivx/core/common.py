from __future__ import annotations

import numpy as np


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


def ensure_broadcastable(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    """Broadcast inputs to a common shape.

    Returns broadcasted views without copying when possible.
    """

    return np.broadcast_arrays(*[as_array(a) for a in arrays])
