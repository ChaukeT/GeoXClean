"""
RBF Interpolation Engine (3D) - HIGH PERFORMANCE VERSION

Supports Radial Basis Function interpolation with:
- Anisotropy via metric matrix transform
- Polynomial drift (constant/linear/quadratic)
- Global/local/GPU acceleration modes
- Continuous and binary classification modes
- Basic diagnostics (MAE, RMSE, R²)

Backends: SciPy (default), ferreus_rbf (large global), CuPy (GPU local)

AUDIT NOTES:
- Added validation for minimum points per polynomial degree
- Added geometry degeneracy checks (coplanar/collinear detection)
- Added automatic fallback to safer polynomial degrees
- Improved error messages for debugging
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, Union, Dict, Any
import logging

# SciPy backend (required for most functionality)
try:
    from scipy.interpolate import RBFInterpolator as SciPyRBF
except ImportError:
    SciPyRBF = None

# Optional ferreus_rbf backend for large global solves
try:
    from ferreus_rbf import RBFInterpolator as FerreusRBF
    from ferreus_rbf.interpolant_config import InterpolantSettings, RBFKernelType
except ImportError:
    FerreusRBF = None

logger = logging.getLogger(__name__)


# =============================================================================
# AUDIT FIX: Minimum points required per polynomial degree in 3D
# degree 0 (constant): 1 term -> need >= 1 unique point
# degree 1 (linear): 4 terms (1 + x + y + z) -> need >= 4 unique points
# degree 2 (quadratic): 10 terms -> need >= 10 unique points with full rank
# =============================================================================
MIN_POINTS_FOR_DEGREE = {
    None: 1,  # No polynomial - just need 1 point
    -1: 1,    # SciPy default (kernel-dependent)
    0: 1,     # Constant
    1: 4,     # Linear (1 + 3 coords)
    2: 10,    # Quadratic (1 + 3 + 6 terms)
}

# SciPy RBFInterpolator default polynomial degrees by kernel (when degree=-1 or not specified)
# From SciPy docs: "The default value depends on kernel"
SCIPY_DEFAULT_DEGREE_BY_KERNEL = {
    "thin_plate_spline": 1,  # Linear
    "linear": 0,             # Constant
    "cubic": 1,              # Linear
    "quintic": 2,            # Quadratic
    "gaussian": 0,           # Constant (conditionally positive definite of order 0)
    "multiquadric": 1,       # Linear
    "inverse_multiquadric": 0,  # Constant
    "inverse_quadratic": 0,  # Constant
}


def _check_geometry_rank(coords: np.ndarray, degree: Optional[int]) -> Tuple[bool, str]:
    """
    Check if point geometry has sufficient rank for requested polynomial degree.
    
    This directly builds and checks the monomial matrix that SciPy will use,
    ensuring we catch degeneracy before SciPy throws a cryptic error.
    
    Returns (is_valid, message).
    """
    if degree is None or degree < 0:
        return True, "No polynomial trend - geometry check not required"
    
    n_points = len(coords)
    min_required = MIN_POINTS_FOR_DEGREE.get(degree, 10)
    
    if n_points < min_required:
        return False, (
            f"Insufficient points for degree {degree} polynomial: "
            f"have {n_points}, need >= {min_required}"
        )
    
    # AUDIT FIX: Build the actual monomial matrix and check its rank
    # This is what SciPy does internally - we check it first
    try:
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        if degree == 0:
            # Constant: just ones
            monomial_matrix = np.ones((n_points, 1))
        elif degree == 1:
            # Linear: [1, x, y, z]
            monomial_matrix = np.column_stack([np.ones(n_points), x, y, z])
        elif degree == 2:
            # Quadratic: [1, x, y, z, x², y², z², xy, xz, yz]
            monomial_matrix = np.column_stack([
                np.ones(n_points), x, y, z,
                x**2, y**2, z**2,
                x*y, x*z, y*z
            ])
        else:
            # Higher degrees - just check coordinate geometry
            monomial_matrix = np.column_stack([np.ones(n_points), x, y, z])
        
        # Check matrix rank
        expected_cols = monomial_matrix.shape[1]
        actual_rank = np.linalg.matrix_rank(monomial_matrix)
        
        if actual_rank < expected_cols:
            # Identify which dimensions have no variation
            coord_ranges = coords.max(axis=0) - coords.min(axis=0)
            flat_dims = []
            if coord_ranges[0] < 1e-10:
                flat_dims.append('X')
            if coord_ranges[1] < 1e-10:
                flat_dims.append('Y')
            if coord_ranges[2] < 1e-10:
                flat_dims.append('Z')
            
            if flat_dims:
                flat_msg = f"No variation in: {', '.join(flat_dims)}. "
            else:
                flat_msg = "Data has degenerate geometry. "
            
            return False, (
                f"Monomial matrix rank deficient ({actual_rank}/{expected_cols}). "
                f"{flat_msg}"
                f"Polynomial degree {degree} not feasible. "
                f"Try degree=None (no polynomial) or add data with more 3D spread."
            )
            
    except Exception as e:
        logger.warning(f"Geometry rank check failed: {e}")
        # Continue anyway - let SciPy provide more specific error
    
    return True, "Geometry check passed"


def _get_scipy_effective_degree(kernel: str, requested_degree: Optional[int]) -> int:
    """
    Determine what polynomial degree SciPy will actually use.
    
    When degree is not specified (None), SciPy uses a kernel-dependent default.
    """
    if requested_degree is not None:
        return requested_degree
    
    # SciPy uses kernel-dependent defaults
    kernel_lower = kernel.lower()
    return SCIPY_DEFAULT_DEGREE_BY_KERNEL.get(kernel_lower, 1)  # Default to 1 if unknown


def _auto_select_degree(coords: np.ndarray, requested_degree: Optional[int], kernel: str = "thin_plate_spline") -> Optional[int]:
    """
    Automatically select a safe polynomial degree based on data geometry.
    
    Falls back to lower degrees if requested degree is not feasible.
    Now also handles the case where requested_degree=None but SciPy uses a non-zero default.
    """
    # Determine what SciPy will actually use
    effective_degree = _get_scipy_effective_degree(kernel, requested_degree)
    
    # If effective degree is 0 or negative, no geometry check needed
    if effective_degree <= 0:
        return requested_degree
    
    # Try effective degree first
    is_valid, msg = _check_geometry_rank(coords, effective_degree)
    if is_valid:
        # If user requested None and it's valid, return None (let SciPy use default)
        # If user requested a specific degree and it's valid, return that
        return requested_degree
    
    logger.warning(f"RBF: {msg}")
    
    # Fall back to lower degrees - need to return an explicit degree now
    for fallback in [1, 0, -1]:
        if fallback >= effective_degree:
            continue
        if fallback <= 0:
            # degree 0 or -1 (let SciPy decide for CPD kernels) should always work
            logger.warning(
                f"RBF: Falling back from effective degree={effective_degree} to degree={fallback} "
                f"(original request was {requested_degree})"
            )
            return fallback
        is_valid, _ = _check_geometry_rank(coords, fallback)
        if is_valid:
            logger.warning(
                f"RBF: Falling back from effective degree={effective_degree} to degree={fallback} "
                f"(original request was {requested_degree})"
            )
            return fallback
    
    # Last resort: degree -1 (let SciPy use minimum for kernel)
    logger.warning("RBF: Falling back to degree=-1 (minimum polynomial for kernel)")
    return -1

MethodType = Literal["auto", "global", "local"]
BackendType = Literal["scipy", "ferreus", "cupy"]


@dataclass
class RBFAnisotropy:
    """
    General anisotropy as a metric transform.

    M is a 3x3 matrix applied as x' = M @ x (or x @ M^T).
    You can construct M from ranges + rotations in your variogram engine
    and pass it directly here, so the RBF sees an anisotropic metric.
    """
    metric_matrix: np.ndarray  # shape (3, 3)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=float)
        return pts @ self.metric_matrix.T


class RBFModel3D:
    """
    Scalable 3D RBF interpolator with:

    - Global / local / GPU mode
    - Optional anisotropy via metric matrix
    - Optional polynomial drift (degree 0 / 1 / 2 for SciPy backends)
    - Continuous and binary classification modes
    - Basic diagnostics (train/validation metrics)

    This is written to be embedded into GeoX, not as a demo script.
    """

    def __init__(
        self,
        coords: np.ndarray,
        values: np.ndarray,
        *,
        kernel: str = "thin_plate_spline",
        smoothing: float = 0.0,
        neighbors: Optional[int] = None,
        method: MethodType = "auto",
        use_gpu: bool = False,
        classification: bool = False,
        anisotropy: Optional[RBFAnisotropy] = None,
        trend_degree: Optional[int] = None,
        large_n_global_threshold: int = 10_000,
        local_threshold: int = 5_000,
        random_state: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        coords : (N, 3)
            XYZ coordinates of input points (original physical space).
        values : (N,) or (N, 1)
            Property values (grades) or indicators.
        kernel : str
            RBF kernel name (SciPy: 'thin_plate_spline', 'linear', 'cubic', 'gaussian', etc.).
        smoothing : float
            Smoothing factor. 0 = exact interpolation; >0 = least-squares fit.
        neighbors : int or None
            If using local interpolation, number of nearest neighbours.
            If None, defaults to 50 in local mode.
        method : 'auto' | 'global' | 'local'
            Global = single global solve; Local = neighbour-based; Auto chooses based on N.
        use_gpu : bool
            Use CuPy backend for local mode if available.
        classification : bool
            If True, values are treated as binary classes; internal representation is 0/1.
        anisotropy : RBFAnisotropy or None
            Optional metric transform. If provided, all coords and query points
            are first transformed: x' = M @ x.
        trend_degree : int or None
            Polynomial drift degree (SciPy backends only):
                None -> no explicit polynomial term (SciPy default)
                0    -> constant
                1    -> linear
                2    -> quadratic
        large_n_global_threshold : int
            Above this N and if ferreus_rbf is available, prefer ferreus global.
        local_threshold : int
            Above this N and without ferreus, prefer local mode.
        random_state : int or None
            For reproducible diagnostics splits.
        """
        if SciPyRBF is None and FerreusRBF is None:
            raise ImportError(
                "At least SciPy or ferreus_rbf must be installed to use RBFModel3D."
            )

        self.random_state = random_state
        self.kernel = kernel
        self.smoothing = smoothing
        self.neighbors = neighbors
        self.method: MethodType = method
        self.use_gpu = use_gpu
        self.classification = classification
        self.anisotropy = anisotropy
        self.large_n_global_threshold = large_n_global_threshold
        self.local_threshold = local_threshold

        # Core data
        coords = np.asarray(coords, dtype=float)
        values = np.asarray(values, dtype=float).flatten()
        if coords.shape[1] != 3:
            raise ValueError("coords must be (N, 3).")

        self.n_points = coords.shape[0]

        # Classification handling (binary only)
        if classification:
            values = self._ensure_binary(values)

        # Apply anisotropy metric to input points if provided
        # IMPORTANT: Do this BEFORE geometry check so we check transformed coords
        if self.anisotropy is not None:
            self._coords_metric = self.anisotropy.apply(coords)
        else:
            self._coords_metric = coords
        
        # AUDIT FIX: Validate and auto-select polynomial degree
        # This prevents "Singular matrix" errors from SciPy
        # CRITICAL: Check on _coords_metric (transformed), not raw coords!
        self.trend_degree = _auto_select_degree(self._coords_metric, trend_degree, kernel)
        
        if self.trend_degree != trend_degree:
            logger.info(
                f"RBF: Adjusted polynomial degree from {trend_degree} to {self.trend_degree} "
                f"based on data geometry ({self.n_points} points)"
            )

        self.coords = coords              # physical space
        self.values = values              # target values
        self.model = None                 # backend model
        self.backend: Optional[BackendType] = None
        self._fitted = False
        self._original_trend_degree = trend_degree  # Store for diagnostics

        # Decide effective method
        self._decide_method()
        
        # Fit model
        self._fit()

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _ensure_binary(self, vals: np.ndarray) -> np.ndarray:
        """Convert any two-class labels to 0/1."""
        uniq = np.unique(vals)
        if uniq.size != 2:
            raise ValueError(
                f"Classification mode expects exactly 2 classes; got {uniq}."
            )
        # Map min -> 0, max -> 1
        minv, maxv = float(uniq.min()), float(uniq.max())
        out = np.where(vals == minv, 0.0, 1.0)
        return out

    def _decide_method(self) -> None:
        """Choose interpolation method if 'auto'."""
        if self.method != "auto":
            return

        if FerreusRBF is not None and self.n_points > self.large_n_global_threshold:
            self.method = "global"  # ferreus path
        else:
            if self.n_points > self.local_threshold:
                self.method = "local"
            else:
                self.method = "global"

    def _fit(self) -> None:
        """Build the backend interpolator."""
        if self.method == "global":
            self._fit_global()
        elif self.method == "local":
            self._fit_local()
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True

    # -------------------------------------------------------------------------
    # Backend construction
    # -------------------------------------------------------------------------

    def _fit_global(self) -> None:
        """Global RBF solve.

        - If ferreus_rbf is available and N large -> ferreus
        - Else SciPy global
        """
        # Try ferreus if large and available and kernel supported
        use_ferreus = (
            FerreusRBF is not None
            and self.n_points > self.large_n_global_threshold
        )

        kernel_type = None
        if use_ferreus:
            k = self.kernel.lower()
            if k == "thin_plate_spline":
                kernel_type = RBFKernelType.ThinPlateSpline
            elif k == "linear":
                kernel_type = RBFKernelType.Linear
            elif k == "cubic":
                kernel_type = RBFKernelType.Cubic
            elif k == "spheroidal":
                kernel_type = RBFKernelType.Spheroidal

        if use_ferreus and kernel_type is not None:
            # ferreus: trend is configured through its own GlobalTrend machinery;
            # not wired here to avoid guessing the API – keep this pure RBF.
            settings = InterpolantSettings(kernel_type)
            self.model = FerreusRBF(
                self._coords_metric,
                self.values.reshape(-1, 1),
                settings,
            )
            self.backend = "ferreus"
            return

        # Fallback: SciPy global
        if SciPyRBF is None:
            raise ImportError(
                "SciPy is required for global RBF interpolation when ferreus_rbf is not used."
            )

        kwargs: Dict[str, Union[str, float, int]] = {
            "kernel": self.kernel,
            "smoothing": self.smoothing,
        }
        if self.trend_degree is not None:
            kwargs["degree"] = int(self.trend_degree)

        # AUDIT FIX: Wrap SciPy call with better error handling
        try:
            self.model = SciPyRBF(
                self._coords_metric,
                self.values,
                **kwargs,
            )
        except np.linalg.LinAlgError as e:
            error_msg = str(e)
            if "Singular matrix" in error_msg or "singular" in error_msg.lower():
                raise ValueError(
                    f"RBF interpolation failed: Data geometry is degenerate for the requested settings.\n"
                    f"Original error: {error_msg}\n\n"
                    f"Suggestions:\n"
                    f"  1. Set 'Polynomial Drift' to 'None' or 'Constant (0)'\n"
                    f"  2. Ensure data points are not coplanar (have 3D spread)\n"
                    f"  3. Check for duplicate or very close points\n"
                    f"  4. Add smoothing > 0 to regularize the interpolation"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"RBF interpolation failed during model fitting.\n"
                f"Error: {e}\n"
                f"Kernel: {self.kernel}, Degree: {self.trend_degree}, Points: {self.n_points}"
            ) from e
        
        self.backend = "scipy"

    def _fit_local(self) -> None:
        """Local neighbour-based RBF (SciPy or CuPy)."""
        k = self.neighbors if self.neighbors is not None else 50

        if self.use_gpu:
            try:
                import cupy as cp
                from cupyx.scipy.interpolate import RBFInterpolator as CuRBF
            except ImportError:
                raise ImportError(
                    "CuPy not available. Install cupy or set use_gpu=False."
                )

            y_gpu = cp.asarray(self._coords_metric, dtype=cp.float64)
            d_gpu = cp.asarray(self.values, dtype=cp.float64)

            kwargs = {
                "kernel": self.kernel,
                "smoothing": self.smoothing,
                "neighbors": k,
            }
            if self.trend_degree is not None:
                kwargs["degree"] = int(self.trend_degree)

            self.model = CuRBF(y_gpu, d_gpu, **kwargs)
            self.backend = "cupy"
            return

        # CPU local – SciPy
        if SciPyRBF is None:
            raise ImportError(
                "SciPy is required for local RBF interpolation when use_gpu=False."
            )

        kwargs = {
            "kernel": self.kernel,
            "smoothing": self.smoothing,
            "neighbors": k,
        }
        if self.trend_degree is not None:
            kwargs["degree"] = int(self.trend_degree)

        # AUDIT FIX: Wrap SciPy call with better error handling
        try:
            self.model = SciPyRBF(
                self._coords_metric,
                self.values,
                **kwargs,
            )
        except np.linalg.LinAlgError as e:
            error_msg = str(e)
            if "Singular matrix" in error_msg or "singular" in error_msg.lower():
                raise ValueError(
                    f"RBF local interpolation failed: Neighborhood geometry is degenerate.\n"
                    f"Original error: {error_msg}\n\n"
                    f"Suggestions:\n"
                    f"  1. Set 'Polynomial Drift' to 'None' or 'Constant (0)'\n"
                    f"  2. Increase 'Neighbors' to capture more 3D variation\n"
                    f"  3. Check data for clusters of coplanar points\n"
                    f"  4. Add smoothing > 0 to regularize"
                ) from e
            raise
        except Exception as e:
            raise ValueError(
                f"RBF local interpolation failed during model fitting.\n"
                f"Error: {e}\n"
                f"Kernel: {self.kernel}, Degree: {self.trend_degree}, Points: {self.n_points}, Neighbors: {k}"
            ) from e
        
        self.backend = "scipy"

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def _transform_points(self, pts: np.ndarray) -> np.ndarray:
        """Apply anisotropy metric if configured."""
        pts = np.asarray(pts, dtype=float)
        if self.anisotropy is not None:
            return self.anisotropy.apply(pts)
        return pts

    def predict(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluate the RBF at arbitrary points.

        points : (M, 3) in physical coordinates.
        Returns: (M,) continuous values (grades or probabilities).
        """
        if not self._fitted:
            raise RuntimeError("RBF model not fitted.")

        points_metric = self._transform_points(points)

        if self.backend == "ferreus":
            vals = self.model.evaluate(points_metric)  # (M, 1)
            vals = np.asarray(vals).ravel()
        elif self.backend == "cupy":
            import cupy as cp

            if isinstance(points_metric, np.ndarray):
                pts_gpu = cp.asarray(points_metric, dtype=cp.float64)
            else:
                pts_gpu = points_metric
            vals_gpu = self.model(pts_gpu)
            vals = cp.asnumpy(vals_gpu).ravel()
        else:  # SciPy
            vals = self.model(points_metric).ravel()

        return vals

    def predict_class(self, points: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Binary classification prediction.

        Returns class 0/1 based on continuous field and threshold.
        """
        if not self.classification:
            raise RuntimeError(
                "predict_class called but model was not initialised in classification mode."
            )
        cont = self.predict(points)
        return (cont >= threshold).astype(int)

    def interpolate_grid(
        self,
        x_range: Union[Tuple[float, float], np.ndarray],
        y_range: Union[Tuple[float, float], np.ndarray],
        z_range: Union[Tuple[float, float], np.ndarray],
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nz: Optional[int] = None,
        chunk_z: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate on a structured 3D grid.

        You can either pass ranges + nx/ny/nz, or explicit coordinate arrays.

        Parameters
        ----------
        x_range, y_range, z_range :
            If nx/ny/nz provided: tuples (min, max).
            If nx is None: arrays of coordinates for each axis.
        nx, ny, nz : int or None
            Number of points along each axis if using ranges.
        chunk_z : int
            Number of z-slices to evaluate per batch to control memory.

        Returns
        -------
        x_coords, y_coords, z_coords, grid_values
            grid_values has shape (nz, ny, nx).
        """
        if not self._fitted:
            raise RuntimeError("RBF model not fitted.")

        # Build coordinate arrays
        if nx is not None:
            xmin, xmax = x_range
            ymin, ymax = y_range
            zmin, zmax = z_range

            x_coords = np.linspace(xmin, xmax, nx)
            y_coords = np.linspace(ymin, ymax, ny if ny is not None else nx)
            z_coords = np.linspace(zmin, zmax, nz if nz is not None else nx)
        else:
            x_coords = np.asarray(x_range, dtype=float)
            y_coords = np.asarray(y_range, dtype=float)
            z_coords = np.asarray(z_range, dtype=float)
            nx = x_coords.size
            ny = y_coords.size
            nz = z_coords.size

        grid = np.empty((nz, ny, nx), dtype=float)

        # Chunk in Z to avoid huge query arrays
        z_indices = np.arange(len(z_coords))
        for start in range(0, len(z_indices), chunk_z):
            end = min(start + chunk_z, len(z_indices))
            idx_batch = z_indices[start:end]
            z_batch = z_coords[idx_batch]

            # Build points for this batch
            XX, YY, ZZ = np.meshgrid(
                x_coords, y_coords, z_batch, indexing="xy"
            )  # shapes (ny, nx, nbatch)
            pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

            vals = self.predict(pts)
            vals = vals.reshape(len(z_batch), ny, nx)

            grid[idx_batch, :, :] = vals

        return x_coords, y_coords, z_coords, grid

    def export_grid(self, grid_values: np.ndarray, filename: str) -> str:
        """
        Save grid to NumPy .npy file. Returns the actual file name.
        """
        fname = filename if filename.endswith(".npy") else filename + ".npy"
        np.save(fname, grid_values)
        return fname

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def diagnostics_holdout(
        self,
        test_fraction: float = 0.2,
        shuffle: bool = True,
    ) -> Dict[str, float]:
        """
        Simple train/validation diagnostics on the existing interpolator.

        This does NOT refit. It just evaluates the *current* model on a held-out
        subset of the original data and computes MAE, RMSE, R².

        For full LOOCV you'd need a separate, slower driver.
        """
        if not (0.0 < test_fraction < 1.0):
            raise ValueError("test_fraction must be in (0, 1).")

        rng = np.random.default_rng(self.random_state)
        n = self.n_points
        idx = np.arange(n)
        if shuffle:
            rng.shuffle(idx)

        split = int((1.0 - test_fraction) * n)
        test_idx = idx[split:]

        X_test = self.coords[test_idx]
        y_test = self.values[test_idx]

        y_pred = self.predict(X_test)

        mae = np.mean(np.abs(y_test - y_pred))
        rmse = float(np.sqrt(np.mean((y_test - y_pred) ** 2)))
        # R²
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return {
            "n_test": int(test_idx.size),
            "MAE": float(mae),
            "RMSE": rmse,
            "R2": float(r2),
        }


# ============================================================================
# Integration utilities for GeoX
# ============================================================================

def create_rbf_anisotropy_from_ranges(
    range_x: float,
    range_y: float,
    range_z: float,
    azimuth: float = 0.0,
    dip: float = 0.0,
    plunge: float = 0.0
) -> RBFAnisotropy:
    """
    Create RBF anisotropy from traditional variogram parameters.

    Parameters
    ----------
    range_x, range_y, range_z : float
        Anisotropic ranges in X, Y, Z directions
    azimuth, dip, plunge : float
        Rotation angles in degrees (future extension)

    Returns
    -------
    RBFAnisotropy object with metric matrix
    """
    # For now, simple diagonal scaling (no rotation)
    # Future: implement full rotation matrix from azimuth/dip/plunge
    ranges = np.array([range_x, range_y, range_z])
    M = np.diag(1.0 / ranges)  # shorter range -> stronger scaling
    return RBFAnisotropy(metric_matrix=M)


def rbf_interpolate_3d(
    coords: np.ndarray,
    values: np.ndarray,
    grid_spec: Dict[str, Any],
    anisotropy: Optional[RBFAnisotropy] = None,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function for RBF interpolation on a 3D grid.

    Parameters
    ----------
    coords : (N, 3)
        Input point coordinates
    values : (N,)
        Input point values
    grid_spec : dict
        Grid specification with keys: nx, ny, nz, xmin, ymin, zmin, xinc, yinc, zinc
    anisotropy : RBFAnisotropy, optional
        Anisotropy specification
    **kwargs
        Additional arguments passed to RBFModel3D

    Returns
    -------
    x_coords, y_coords, z_coords, grid_values
    """
    # Create RBF model
    model = RBFModel3D(
        coords=coords,
        values=values,
        anisotropy=anisotropy,
        **kwargs
    )

    # Create grid
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    nz = grid_spec["nz"]
    xmin = grid_spec["xmin"]
    ymin = grid_spec["ymin"]
    zmin = grid_spec["zmin"]
    xmax = xmin + (nx - 1) * grid_spec["xinc"]
    ymax = ymin + (ny - 1) * grid_spec["yinc"]
    zmax = zmin + (nz - 1) * grid_spec["zinc"]

    # Interpolate
    return model.interpolate_grid(
        (xmin, xmax), (ymin, ymax), (zmin, zmax),
        nx=nx, ny=ny, nz=nz
    )
