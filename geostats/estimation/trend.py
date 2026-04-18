"""
Drift (polynomial trend) builder for the RBF system.

Three modes matching Leapfrog Geo:
  - "constant":  p(x) = c₀                    → 1 extra row/column
  - "linear":    p(x) = c₀ + c₁x + c₂y + c₃z → 4 extra rows/columns
  - "none":      no polynomial                 → 0 extra rows/columns

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .config import DriftType

logger = logging.getLogger(__name__)


def drift_dimension(drift: DriftType) -> int:
    """
    Number of extra rows/columns added to the system matrix for the drift.

    Parameters
    ----------
    drift : DriftType

    Returns
    -------
    int
        0 for 'none', 1 for 'constant', 4 for 'linear'.
    """
    if drift == DriftType.NONE:
        return 0
    elif drift == DriftType.CONSTANT:
        return 1
    elif drift == DriftType.LINEAR:
        return 4
    else:
        raise ValueError(f"Unknown drift type: {drift}")


def build_polynomial_matrix(
    points: NDArray[np.float64], drift: DriftType
) -> NDArray[np.float64]:
    """
    Build the polynomial (drift) matrix P for the given data points.

    Parameters
    ----------
    points : (N, 3) ndarray
        Data point coordinates.
    drift : DriftType
        Which drift mode to use.

    Returns
    -------
    P : (N, m) ndarray
        Polynomial matrix where m = drift_dimension(drift).
        Empty (N, 0) array if drift is 'none'.
    """
    n = points.shape[0]
    m = drift_dimension(drift)

    if m == 0:
        return np.empty((n, 0), dtype=np.float64)

    if drift == DriftType.CONSTANT:
        return np.ones((n, 1), dtype=np.float64)

    if drift == DriftType.LINEAR:
        P = np.empty((n, 4), dtype=np.float64)
        P[:, 0] = 1.0
        P[:, 1] = points[:, 0]  # x
        P[:, 2] = points[:, 1]  # y
        P[:, 3] = points[:, 2]  # z
        return P

    raise ValueError(f"Unknown drift type: {drift}")


def evaluate_polynomial(
    points: NDArray[np.float64],
    coefficients: NDArray[np.float64],
    drift: DriftType,
) -> NDArray[np.float64]:
    """
    Evaluate the polynomial drift at query points.

    Parameters
    ----------
    points : (M, 3) ndarray
        Query coordinates.
    coefficients : (m,) ndarray
        Polynomial coefficients from the solved system.
    drift : DriftType

    Returns
    -------
    p(x) : (M,) ndarray
        Polynomial contribution at each query point.
    """
    m = drift_dimension(drift)

    if m == 0:
        return np.zeros(points.shape[0], dtype=np.float64)

    P = build_polynomial_matrix(points, drift)
    return P @ coefficients


def build_augmented_system(
    kernel_matrix: NDArray[np.float64],
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    drift: DriftType,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build the full augmented linear system [Φ P; Pᵀ 0] and RHS [d; 0].

    Parameters
    ----------
    kernel_matrix : (N, N) ndarray
        Φ[i,j] = φ(‖xᵢ - xⱼ‖) + nugget·δᵢⱼ  (already includes nugget).
    points : (N, 3) ndarray
        Data coordinates.
    values : (N,) ndarray
        Data values.
    drift : DriftType

    Returns
    -------
    A : (N+m, N+m) ndarray
        Augmented system matrix.
    b : (N+m,) ndarray
        Right-hand side vector.
    """
    n = kernel_matrix.shape[0]
    m = drift_dimension(drift)
    size = n + m

    A = np.zeros((size, size), dtype=np.float64)
    b = np.zeros(size, dtype=np.float64)

    # Top-left: kernel matrix
    A[:n, :n] = kernel_matrix

    if m > 0:
        P = build_polynomial_matrix(points, drift)
        # Top-right: P
        A[:n, n:] = P
        # Bottom-left: Pᵀ
        A[n:, :n] = P.T
        # Bottom-right: zeros (already initialised)

    # RHS
    b[:n] = values
    # b[n:] = 0 already

    return A, b
