"""
Grid Utilities — consistent array ordering for PyVista ImageData cell_data.

PyVista ImageData cell ordering: X varies fastest, then Y, then Z.
This matches C-order ravelling of a (nz, ny, nx) shaped array.

RULE: All 3D arrays in GeoX follow the geological convention (nz, ny, nx).
      Use array_3d_to_cell_data() to flatten them for cell_data assignment.
      NEVER call .ravel() directly on estimation output — use this utility.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def array_3d_to_cell_data(arr_3d: np.ndarray, shape_order: str = "zyx") -> np.ndarray:
    """Convert a 3D array to a flat 1D array matching PyVista ImageData cell ordering.

    Parameters
    ----------
    arr_3d : np.ndarray
        3D numpy array of estimation results.
    shape_order : str
        ``'zyx'`` if shape is (nz, ny, nx) — geological standard.
        ``'xyz'`` if shape is (nx, ny, nz) — legacy kriging output.

    Returns
    -------
    np.ndarray
        1D array matching ImageData cell ordering (X fastest, then Y, then Z).

    Raises
    ------
    ValueError
        If *arr_3d* is not 3-dimensional or *shape_order* is unknown.
    """
    if arr_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr_3d.ndim}D (shape {arr_3d.shape})")

    if shape_order == "zyx":
        # (nz, ny, nx) → C-order ravel iterates X fastest = correct for ImageData
        return arr_3d.ravel(order="C")
    elif shape_order == "xyz":
        # (nx, ny, nz) → transpose to (nz, ny, nx) then C-ravel
        return arr_3d.transpose(2, 1, 0).ravel(order="C")
    else:
        raise ValueError(f"Unknown shape_order '{shape_order}'. Use 'zyx' or 'xyz'.")


def cell_data_to_array_3d(
    flat: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    """Reverse of array_3d_to_cell_data — reshape flat cell_data to (nz, ny, nx)."""
    return flat.reshape((nz, ny, nx))


# ======================================================================
# Tolerance-aware coordinate utilities
# ======================================================================
# np.unique on float64 with tiny noise (~1e-10) returns 3991 values
# instead of 10.  These functions group values within tolerance.
# Verified in Exercise 4.1: recovers correct spacing from jittered grids.
# ======================================================================

def unique_with_tolerance(arr: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Return unique values grouped by tolerance.

    Groups adjacent values in the sorted array that differ by <= *tol*,
    returning the median of each group.

    Parameters
    ----------
    arr : np.ndarray
        1-D array of float values (e.g. X coordinates of cell centres).
    tol : float
        Maximum gap between consecutive sorted values to consider them
        the same group.  Default 1e-6 m.

    Returns
    -------
    np.ndarray
        Sorted array of unique representative values (group medians).
    """
    if len(arr) == 0:
        return np.array([], dtype=np.float64)
    sorted_arr = np.sort(arr.ravel())
    # Find split points where the gap exceeds tolerance
    gaps = np.diff(sorted_arr)
    split_idx = np.where(gaps > tol)[0] + 1
    groups = np.split(sorted_arr, split_idx)
    return np.array([np.median(g) for g in groups], dtype=np.float64)


def infer_spacing_from_coordinates(coords: np.ndarray, tol: float = 1e-6) -> float:
    """Infer uniform grid spacing from a 1-D coordinate array.

    Computes diffs of the sorted coordinates, filters out noise
    (diffs <= tol), and returns the median of the remaining diffs.

    Parameters
    ----------
    coords : np.ndarray
        1-D array of coordinate values along one axis.
    tol : float
        Diffs smaller than this are considered noise.

    Returns
    -------
    float
        Inferred spacing.  Falls back to median of all diffs if every
        diff is below tolerance (degenerate case).
    """
    sorted_c = np.sort(coords.ravel())
    diffs = np.diff(sorted_c)
    if len(diffs) == 0:
        return 1.0
    real_diffs = diffs[diffs > tol]
    if len(real_diffs) > 0:
        return float(np.median(real_diffs))
    return float(np.median(diffs)) if len(diffs) > 0 else 1.0


def detect_uniform_grid(
    positions: np.ndarray, tol: float = 1e-6
) -> "tuple[bool, dict | None]":
    """Detect whether *positions* (N, 3) form a uniform axis-aligned grid.

    Returns
    -------
    is_uniform : bool
    grid_info : dict or None
        If uniform: ``{'nx', 'ny', 'nz', 'dx', 'dy', 'dz',
        'origin': (x0, y0, z0)}``.
    """
    if positions is None or len(positions) < 2:
        return False, None

    ux = unique_with_tolerance(positions[:, 0], tol)
    uy = unique_with_tolerance(positions[:, 1], tol)
    uz = unique_with_tolerance(positions[:, 2], tol)

    nx, ny, nz = len(ux), len(uy), len(uz)
    expected = nx * ny * nz
    if expected != len(positions):
        logger.debug(
            "detect_uniform_grid: nx*ny*nz=%d != n_positions=%d",
            expected, len(positions),
        )
        return False, None

    # Check uniform spacing per axis
    def _is_uniform_axis(uniques):
        if len(uniques) < 2:
            return True, float(uniques[0]) if len(uniques) == 1 else 0.0
        diffs = np.diff(uniques)
        median_d = np.median(diffs)
        if median_d < tol:
            return False, 0.0
        return bool(np.all(np.abs(diffs - median_d) < tol * 10)), float(median_d)

    ok_x, dx = _is_uniform_axis(ux)
    ok_y, dy = _is_uniform_axis(uy)
    ok_z, dz = _is_uniform_axis(uz)

    if not (ok_x and ok_y and ok_z):
        return False, None

    # Origin = minimum coordinate minus half spacing (cell centres → cell edges)
    origin = (
        float(ux[0]) - dx / 2.0,
        float(uy[0]) - dy / 2.0,
        float(uz[0]) - dz / 2.0,
    )

    return True, {
        "nx": nx, "ny": ny, "nz": nz,
        "dx": dx, "dy": dy, "dz": dz,
        "origin": origin,
    }
