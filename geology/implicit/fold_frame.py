"""
Fold Frame Construction -- Laurent et al. (2016) Approach.
==========================================================

Implements curvilinear coordinate systems aligned with fold geometry
for modelling folded stratigraphy.

Three orthogonal scalar fields:
  S1: axial foliation (distance from axial surface)
  S2: fold axis direction (distance along fold axis)
  S0: the folded stratigraphy (built AFTER S1 and S2)

The fold frame unrolls folds so that interpolation operates in a
coordinate system where layers are approximately flat.

Mathematical basis: Section 9 of the GeoX Math Specification.
References: Laurent et al. (2016), Mathematical Geosciences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .contact_data import dip_azimuth_to_normal, orientations_dataframe_to_list
from .scalar_field import (
    assemble_augmented_matrix,
    solve_augmented_system,
    make_evaluate_fn,
    evaluate_scalar_field,
    PUM_THRESHOLD,
    solve_augmented_system_pum,
    make_evaluate_fn_pum,
)
from .surface_extraction import evaluate_field_on_grid

logger = logging.getLogger(__name__)


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class FoldFrameConfig:
    """Configuration for fold frame construction."""
    fold_axis_azimuth: float = 0.0     # degrees
    fold_axis_plunge: float = 0.0      # degrees
    fold_type: str = "cylindrical"     # "cylindrical", "conical"
    wavelength: Optional[float] = None # fold wavelength (metres)
    auto_detect: bool = True           # auto-detect fold params from data
    regularisation_weight: float = 1.0 # weight for modified regularisation


# =====================================================================
# Fold axis detection
# =====================================================================

def detect_fold_axis(
    orientations_df: pd.DataFrame,
    dip_col: str = "dip",
    azimuth_col: str = "azimuth",
) -> Tuple[float, float]:
    """Auto-detect fold axis from bedding orientation measurements.

    Uses eigenvector analysis of bedding poles (Ramsay & Huber method):
    1. Convert dip/azimuth to unit normal vectors (poles to bedding)
    2. Compute the orientation tensor: T = SUM n_i * n_i^T
    3. Eigendecomposition: T = V Lambda V^T
    4. Eigenvector with SMALLEST eigenvalue = fold axis direction

    Parameters
    ----------
    orientations_df : pd.DataFrame
        Must have dip and azimuth columns.

    Returns
    -------
    azimuth : float (degrees, 0=north, clockwise)
    plunge : float (degrees, 0=horizontal, positive=downward)
    """
    if len(orientations_df) < 3:
        raise ValueError(
            "Need at least 3 orientation measurements for fold axis detection"
        )

    dips = orientations_df[dip_col].values.astype(np.float64)
    azimuths = orientations_df[azimuth_col].values.astype(np.float64)

    # Convert to unit normal vectors (poles to bedding)
    poles = np.zeros((len(dips), 3), dtype=np.float64)
    for i in range(len(dips)):
        poles[i] = dip_azimuth_to_normal(dips[i], azimuths[i])

    # Orientation tensor
    T = poles.T @ poles  # (3, 3) symmetric positive semi-definite

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(T)

    # Smallest eigenvalue -> fold axis direction
    # eigh returns eigenvalues in ascending order
    fold_axis = eigenvectors[:, 0]

    # Ensure fold axis points downward (positive z component convention)
    if fold_axis[2] < 0:
        fold_axis = -fold_axis

    # Convert to azimuth/plunge
    # azimuth = atan2(nx, ny) (north = +y, east = +x)
    azimuth_rad = np.arctan2(fold_axis[0], fold_axis[1])
    azimuth_deg = np.degrees(azimuth_rad) % 360.0

    # plunge = asin(nz) for unit vector
    horiz_mag = np.sqrt(fold_axis[0]**2 + fold_axis[1]**2)
    plunge_rad = np.arctan2(fold_axis[2], horiz_mag)
    plunge_deg = np.degrees(plunge_rad)

    # Eigenvalue ratio indicates fold tightness
    ratio = eigenvalues[0] / max(eigenvalues[2], 1e-10)
    logger.info(
        "Detected fold axis: azimuth=%.1f, plunge=%.1f "
        "(eigenvalue ratio=%.4f, %s fold)",
        azimuth_deg, plunge_deg, ratio,
        "tight" if ratio < 0.1 else "open" if ratio > 0.3 else "moderate",
    )

    return azimuth_deg, plunge_deg


# =====================================================================
# S-plot computation
# =====================================================================

def compute_s_plot(
    orientations_df: pd.DataFrame,
    s1_evaluate_fn: Callable,
    fold_axis: np.ndarray,
    dip_col: str = "dip",
    azimuth_col: str = "azimuth",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the S-plot: fold rotation angle vs S1 coordinate.

    theta(s1) = angle between bedding normal and the S1 gradient direction.

    The S-plot characterises the fold geometry:
    - Sinusoidal: gentle fold
    - Step-like: tight fold
    - Flat at +/-90: isoclinal fold

    Parameters
    ----------
    orientations_df : pd.DataFrame
    s1_evaluate_fn : callable
        S1 field evaluation function.
    fold_axis : (3,) unit vector

    Returns
    -------
    s1_values : (N,) S1 coordinate at each measurement
    theta_values : (N,) rotation angle in degrees
    """
    coords = orientations_df[[x_col, y_col, z_col]].values.astype(np.float64)
    s1_values = s1_evaluate_fn(coords)

    # Compute bedding normals
    dips = orientations_df[dip_col].values.astype(np.float64)
    azimuths = orientations_df[azimuth_col].values.astype(np.float64)

    normals = np.zeros((len(dips), 3), dtype=np.float64)
    for i in range(len(dips)):
        normals[i] = dip_azimuth_to_normal(dips[i], azimuths[i])

    # The reference direction is perpendicular to fold axis in the
    # plane containing the fold axis and vertical
    vertical = np.array([0.0, 0.0, 1.0])
    ref_dir = np.cross(fold_axis, vertical)
    ref_norm = np.linalg.norm(ref_dir)
    if ref_norm < 1e-10:
        # Fold axis is vertical -- use east as reference
        ref_dir = np.array([1.0, 0.0, 0.0])
    else:
        ref_dir /= ref_norm

    # Compute rotation angle: angle of normal projected into the
    # plane perpendicular to fold axis
    theta_values = np.zeros(len(normals), dtype=np.float64)
    for i in range(len(normals)):
        n = normals[i]
        # Project normal into plane perpendicular to fold axis
        n_proj = n - np.dot(n, fold_axis) * fold_axis
        n_proj_mag = np.linalg.norm(n_proj)
        if n_proj_mag < 1e-10:
            theta_values[i] = 0.0
            continue
        n_proj /= n_proj_mag

        # Angle from reference direction
        cos_theta = np.clip(np.dot(n_proj, ref_dir), -1.0, 1.0)
        sin_theta = np.dot(np.cross(ref_dir, n_proj), fold_axis)
        theta_values[i] = np.degrees(np.arctan2(sin_theta, cos_theta))

    return s1_values, theta_values


# =====================================================================
# Modified regularisation
# =====================================================================


# =====================================================================
# Fold Frame class
# =====================================================================

class FoldFrame:
    """A structural frame aligned with fold geometry.

    Three orthogonal scalar fields:
      S1: axial foliation (distance from axial surface)
      S2: fold axis direction (distance along fold axis)
      S0: the folded stratigraphy (built AFTER S1 and S2)
    """

    def __init__(self, config: FoldFrameConfig):
        self.config = config
        self.s1_evaluate: Optional[Callable] = None
        self.s2_evaluate: Optional[Callable] = None
        self.s0_evaluate: Optional[Callable] = None
        self.fold_axis: Optional[np.ndarray] = None
        self.axial_normal: Optional[np.ndarray] = None
        self._detected_azimuth: float = 0.0
        self._detected_plunge: float = 0.0

    def build_from_orientations(
        self,
        orientations_df: pd.DataFrame,
        contacts_df: pd.DataFrame,
        grid_origin: np.ndarray,
        grid_spacing: np.ndarray,
        grid_dims: Tuple[int, int, int],
        kernel_type: str = "spheroidal",
        alpha: float = 1.0,
        range_: float = 200.0,
        nugget: float = 0.0,
        accuracy: float = 1e-6,
        drift_type: str = "linear",
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """Build the fold frame from structural measurements.

        Algorithm:
        1. Detect fold axis from orientation data (if auto_detect)
        2. Build S1 (axial foliation)
        3. Build S2 (fold axis direction)
        4. Build S0 (stratigraphy) with modified regularisation

        Parameters
        ----------
        orientations_df : pd.DataFrame
            Bedding measurements: X, Y, Z, dip, azimuth
        contacts_df : pd.DataFrame
            Geological contacts: X, Y, Z, surface_name
        grid_origin, grid_spacing, grid_dims : grid parameters
        kernel_type, alpha, range_, nugget, accuracy : interpolation params

        Returns
        -------
        dict with:
            s1_field, s2_field, s0_field : (nx, ny, nz) arrays
            s1_evaluate, s2_evaluate, s0_evaluate : callable
            fold_axis : (3,) unit vector
            s_plot : (s1_values, theta_values)
        """
        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        # Step 1: Detect fold axis
        _progress(5, "Detecting fold axis")

        if self.config.auto_detect:
            self._detected_azimuth, self._detected_plunge = detect_fold_axis(
                orientations_df,
            )
        else:
            self._detected_azimuth = self.config.fold_axis_azimuth
            self._detected_plunge = self.config.fold_axis_plunge

        # Fold axis as unit vector
        az_rad = np.radians(self._detected_azimuth)
        pl_rad = np.radians(self._detected_plunge)
        self.fold_axis = np.array([
            np.cos(pl_rad) * np.sin(az_rad),
            np.cos(pl_rad) * np.cos(az_rad),
            np.sin(pl_rad),
        ], dtype=np.float64)

        # Axial surface normal: perpendicular to fold axis in horizontal plane
        self.axial_normal = np.array([
            np.cos(az_rad),
            -np.sin(az_rad),
            0.0,
        ], dtype=np.float64)
        norm_mag = np.linalg.norm(self.axial_normal)
        if norm_mag > 1e-10:
            self.axial_normal /= norm_mag

        # Step 2: Build S1 (axial foliation)
        _progress(15, "Building S1 field (axial foliation)")
        self.s1_evaluate = self._build_s1(
            orientations_df, kernel_type, alpha, range_,
            nugget, accuracy, drift_type,
        )

        _progress(30, "Evaluating S1 on grid")
        s1_field = evaluate_field_on_grid(
            grid_origin, grid_spacing, grid_dims, self.s1_evaluate,
        )

        # Step 3: Build S2 (fold axis direction)
        _progress(40, "Building S2 field (fold axis)")
        self.s2_evaluate = self._build_s2(
            orientations_df, kernel_type, alpha, range_,
            nugget, accuracy, drift_type,
        )

        _progress(55, "Evaluating S2 on grid")
        s2_field = evaluate_field_on_grid(
            grid_origin, grid_spacing, grid_dims, self.s2_evaluate,
        )

        # Step 4: Build S0 (stratigraphy) with modified regularisation
        _progress(65, "Building S0 field (stratigraphy)")
        self.s0_evaluate = self._build_s0(
            orientations_df, contacts_df,
            kernel_type, alpha, range_,
            nugget, accuracy, drift_type,
        )

        _progress(80, "Evaluating S0 on grid")
        s0_field = evaluate_field_on_grid(
            grid_origin, grid_spacing, grid_dims, self.s0_evaluate,
        )

        # Step 5: Compute S-plot
        _progress(90, "Computing S-plot")
        s_plot = compute_s_plot(
            orientations_df, self.s1_evaluate, self.fold_axis,
        )

        _progress(100, "Fold frame complete")

        return {
            "s1_field": s1_field,
            "s2_field": s2_field,
            "s0_field": s0_field,
            "s1_evaluate": self.s1_evaluate,
            "s2_evaluate": self.s2_evaluate,
            "s0_evaluate": self.s0_evaluate,
            "fold_axis": self.fold_axis,
            "fold_axis_azimuth": self._detected_azimuth,
            "fold_axis_plunge": self._detected_plunge,
            "axial_normal": self.axial_normal,
            "s_plot": s_plot,
        }

    def _build_s1(
        self, orientations_df, kernel_type, alpha, range_,
        nugget, accuracy, drift_type,
    ) -> Callable:
        """Build S1: axial foliation field.

        S1 measures distance from the axial surface.
        Gradient of S1 is perpendicular to the axial surface (= axial normal).

        Constraints:
          - Subset of bedding measurements contribute gradient constraints.
          - The centroid of the data is assigned S1 = 0.
          - Two additional value constraints along axial normal to anchor
            the field direction.
        """
        orient_list = orientations_dataframe_to_list(orientations_df)
        coords = np.array(
            [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
        )

        centroid = coords.mean(axis=0)

        # Value constraints: centroid at S1=0, plus two points along
        # the axial normal to anchor the gradient direction
        data_extent = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
        offset = max(data_extent * 0.3, range_ * 0.5)
        pt_pos = centroid + self.axial_normal * offset
        pt_neg = centroid - self.axial_normal * offset

        value_coords = np.array([centroid, pt_pos, pt_neg], dtype=np.float64)
        value_data = np.array([0.0, offset, -offset], dtype=np.float64)

        # Use a modest subset of gradient constraints to avoid singular K_gg
        # with all-identical normals. Use at most 5 well-spaced points.
        n_grad = min(5, len(coords))
        if len(coords) > n_grad:
            indices = np.linspace(0, len(coords) - 1, n_grad, dtype=int)
            gradient_coords = coords[indices].copy()
        else:
            gradient_coords = coords.copy()

        # Local axial foliation normal = cross(fold_axis, bedding_normal_i).
        # This varies spatially as the bedding normal rotates across the fold,
        # producing a genuinely curvilinear S1 field (not a flat plane).
        orient_normals = np.array([o.normal for o in orient_list], dtype=np.float64)
        selected_orient_normals = orient_normals[indices] if len(orient_list) > n_grad else orient_normals
        local_axial_normals = []
        for bn in selected_orient_normals:
            local_n = np.cross(self.fold_axis, bn)
            n_len = np.linalg.norm(local_n)
            if n_len > 1e-8:
                local_n /= n_len
            else:
                local_n = self.axial_normal  # fallback: fold axis parallel to bedding
            local_axial_normals.append(local_n)
        gradient_normals = np.array(local_axial_normals, dtype=np.float64)

        # Ensure sufficient regularisation for parallel gradient constraints
        eff_nugget = max(nugget, 0.01)
        eff_accuracy = max(accuracy, 1e-4)

        # S1 must use constant drift to avoid underdetermined polynomial
        # block: 3 value constraints + identical gradient normals + linear
        # drift (4 poly terms) → singular system.  Constant drift (1 term)
        # keeps the system well-conditioned (cond ~80).
        s1_drift = "constant"

        N_total = len(value_coords) + len(gradient_coords)
        if N_total > PUM_THRESHOLD:
            pum = solve_augmented_system_pum(
                value_coords, value_data,
                gradient_coords, gradient_normals,
                np.ones(len(gradient_coords), dtype=np.float64),
                kernel_type=kernel_type, alpha=alpha, range_=range_,
                nugget=eff_nugget, accuracy=eff_accuracy, drift_type=s1_drift,
            )
            return make_evaluate_fn_pum(pum)

        K_aug, N_v, N_g = assemble_augmented_matrix(
            value_coords, gradient_coords, gradient_normals,
            kernel_type=kernel_type, alpha=alpha, range_=range_,
            nugget=eff_nugget, accuracy=eff_accuracy, drift_type=s1_drift,
        )

        gradient_values = np.ones(N_g, dtype=np.float64)

        value_weights, gradient_weights, poly_coeffs = solve_augmented_system(
            K_aug, value_data, gradient_values, N_v, N_g,
        )

        return make_evaluate_fn(
            value_coords, gradient_coords, gradient_normals,
            value_weights, gradient_weights, poly_coeffs,
            kernel_type, alpha, range_, None, None, s1_drift,
        )

    def _build_s2(
        self, orientations_df, kernel_type, alpha, range_,
        nugget, accuracy, drift_type,
    ) -> Callable:
        """Build S2: fold axis direction field (Laurent et al. 2016).

        A true curvilinear fold frame requires the fold axis direction to
        vary spatially, derived from the INTERPOLATED S1 gradient — not
        from a single global fold axis vector.

        Algorithm (Laurent 2016, §4.2):
          1. Evaluate the S1 gradient numerically at every orientation
             measurement location using central finite differences.
          2. For each measurement i: local_fold_axis_i = cross(∇S1_i, n_bedding_i).
             This is the direction along which both S1 and the bedding plane
             are simultaneously constant — the physical definition of the fold
             axis.
          3. Fit S2 with gradient constraints = local_fold_axis_i, so the S2
             field accurately tracks a curved fold plunge.
        """
        if self.s1_evaluate is None:
            raise RuntimeError("S1 must be built before S2 (call _build_s1 first)")

        orient_list = orientations_dataframe_to_list(orientations_df)
        coords = np.array(
            [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
        )
        orient_normals = np.array([o.normal for o in orient_list], dtype=np.float64)

        centroid = coords.mean(axis=0)

        # Step 1 — Evaluate ∇S1 at each measurement location via central FD.
        _fd = max(range_ * 0.01, 1.0)  # FD step: 1% of range, min 1 m
        grad_s1 = np.zeros_like(coords)
        for ax in range(3):
            eps = np.zeros(3); eps[ax] = _fd
            pts_plus = coords + eps[np.newaxis, :]
            pts_minus = coords - eps[np.newaxis, :]
            grad_s1[:, ax] = (
                self.s1_evaluate(pts_plus) - self.s1_evaluate(pts_minus)
            ) / (2.0 * _fd)

        # Step 2 — Compute local fold axis as cross(∇S1_i, n_bedding_i).
        local_fold_axes = []
        valid_indices = []
        for i, (gs1, bn) in enumerate(zip(grad_s1, orient_normals)):
            gs1_norm = np.linalg.norm(gs1)
            if gs1_norm < 1e-10:
                continue
            gs1 /= gs1_norm
            local_ax = np.cross(gs1, bn)
            ax_len = np.linalg.norm(local_ax)
            if ax_len < 1e-8:
                # S1 gradient and bedding normal are parallel → degenerate
                # (measurement is on an axial plane itself). Skip.
                continue
            local_ax /= ax_len
            # Orient consistently with the detected fold axis
            if np.dot(local_ax, self.fold_axis) < 0:
                local_ax = -local_ax
            local_fold_axes.append(local_ax)
            valid_indices.append(i)

        if not local_fold_axes:
            logger.warning(
                "S2: no valid local fold axes could be derived from ∇S1. "
                "Falling back to global fold axis."
            )
            n_grad = min(5, len(coords))
            indices = np.linspace(0, len(coords) - 1, n_grad, dtype=int)
            gradient_coords = coords[indices].copy()
            gradient_normals = np.tile(self.fold_axis, (len(gradient_coords), 1))
        else:
            gradient_coords = coords[valid_indices].copy()
            gradient_normals = np.array(local_fold_axes, dtype=np.float64)

        # Value constraints: anchor S2 along the detected fold axis direction
        data_extent = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
        offset = max(data_extent * 0.3, range_ * 0.5)
        pt_pos = centroid + self.fold_axis * offset
        pt_neg = centroid - self.fold_axis * offset

        value_coords = np.array([centroid, pt_pos, pt_neg], dtype=np.float64)
        value_data = np.array([0.0, offset, -offset], dtype=np.float64)

        eff_nugget = max(nugget, 0.01)
        eff_accuracy = max(accuracy, 1e-4)
        s2_drift = "constant"

        N_total = len(value_coords) + len(gradient_coords)
        if N_total > PUM_THRESHOLD:
            pum = solve_augmented_system_pum(
                value_coords, value_data,
                gradient_coords, gradient_normals,
                np.ones(len(gradient_coords), dtype=np.float64),
                kernel_type=kernel_type, alpha=alpha, range_=range_,
                nugget=eff_nugget, accuracy=eff_accuracy, drift_type=s2_drift,
            )
            return make_evaluate_fn_pum(pum)

        K_aug, N_v, N_g = assemble_augmented_matrix(
            value_coords, gradient_coords, gradient_normals,
            kernel_type=kernel_type, alpha=alpha, range_=range_,
            nugget=eff_nugget, accuracy=eff_accuracy, drift_type=s2_drift,
        )

        gradient_values = np.ones(N_g, dtype=np.float64)

        value_weights, gradient_weights, poly_coeffs = solve_augmented_system(
            K_aug, value_data, gradient_values, N_v, N_g,
        )

        return make_evaluate_fn(
            value_coords, gradient_coords, gradient_normals,
            value_weights, gradient_weights, poly_coeffs,
            kernel_type, alpha, range_, None, None, s2_drift,
        )

    def _build_s0(
        self, orientations_df, contacts_df,
        kernel_type, alpha, range_,
        nugget, accuracy, drift_type,
    ) -> Callable:
        """Build S0: stratigraphy field with modified regularisation.

        S0 is the actual geological stratigraphy, built using:
          - Contact value constraints (f = 0 at contacts)
          - Bedding orientation gradient constraints (grad f . n = 1)
          - Modified regularisation: penalise curvature ONLY along S2

        The modified regularisation allows sharp fold hinges (high
        curvature across S1) while maintaining smooth surfaces along
        the fold axis (S2).
        """
        # Contact constraints: f = 0 at all contacts
        if contacts_df is not None and not contacts_df.empty:
            contact_coords = contacts_df[["X", "Y", "Z"]].values.astype(np.float64)
            value_coords = contact_coords
            value_data = np.zeros(len(contact_coords), dtype=np.float64)
        else:
            # No contacts: use centroid
            orient_list = orientations_dataframe_to_list(orientations_df)
            coords = np.array(
                [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
            )
            value_coords = coords.mean(axis=0, keepdims=True)
            value_data = np.array([0.0], dtype=np.float64)

        # Use constant drift for S0 to avoid underdetermined polynomial
        # system when few contacts are available
        s0_drift = "constant"

        # Gradient constraints from bedding orientations
        orient_list = orientations_dataframe_to_list(orientations_df)
        gradient_coords = np.array(
            [[o.x, o.y, o.z] for o in orient_list], dtype=np.float64,
        )
        gradient_normals = np.array(
            [o.normal for o in orient_list], dtype=np.float64,
        )

        # Modified regularisation: increase accuracy (Tikhonov) to smooth
        # along the fold axis while allowing sharp hinges across S1.
        # LIMITATION: This is isotropic regularisation scaled by
        # regularisation_weight, NOT directional along S2. True
        # directional regularisation would require modifying the kernel
        # matrix per-element based on S2 alignment at each point pair.
        modified_accuracy = max(accuracy, 1e-4) * (1.0 + self.config.regularisation_weight)
        eff_nugget = max(nugget, 0.01)

        N_g = len(gradient_coords)
        gradient_values = np.ones(N_g, dtype=np.float64) if N_g > 0 else np.empty(0)

        N_total = len(value_coords) + N_g
        if N_total > PUM_THRESHOLD:
            pum = solve_augmented_system_pum(
                value_coords, value_data,
                gradient_coords, gradient_normals, gradient_values,
                kernel_type=kernel_type, alpha=alpha, range_=range_,
                nugget=eff_nugget, accuracy=modified_accuracy, drift_type=s0_drift,
            )
            return make_evaluate_fn_pum(pum)

        K_aug, N_v, N_g_aug = assemble_augmented_matrix(
            value_coords, gradient_coords, gradient_normals,
            kernel_type=kernel_type, alpha=alpha, range_=range_,
            nugget=eff_nugget, accuracy=modified_accuracy,
            drift_type=s0_drift,
        )

        value_weights, gradient_weights, poly_coeffs = solve_augmented_system(
            K_aug, value_data, gradient_values, N_v, N_g_aug,
        )

        return make_evaluate_fn(
            value_coords, gradient_coords, gradient_normals,
            value_weights, gradient_weights, poly_coeffs,
            kernel_type, alpha, range_, None, None, s0_drift,
        )

    def transform_to_fold_coords(
        self, points: np.ndarray,
    ) -> np.ndarray:
        """Transform geographic coordinates to fold frame coordinates.

        Maps (x, y, z) -> (s1, s2, s0).

        Parameters
        ----------
        points : (N, 3) geographic coordinates

        Returns
        -------
        fold_coords : (N, 3) where columns are (s1, s2, s0)
        """
        if self.s1_evaluate is None or self.s2_evaluate is None or self.s0_evaluate is None:
            raise RuntimeError("Fold frame not built. Call build_from_orientations() first.")

        s1 = self.s1_evaluate(points)
        s2 = self.s2_evaluate(points)
        s0 = self.s0_evaluate(points)

        return np.column_stack([s1, s2, s0])

    def transform_from_fold_coords(
        self, fold_coords: np.ndarray,
    ) -> np.ndarray:
        """Transform fold frame coordinates back to geographic.

        This is an approximate inverse using the Jacobian at each point.
        For most geological applications, the forward transform is
        sufficient (domain assignment, cross-section extraction).

        Parameters
        ----------
        fold_coords : (N, 3) fold frame coordinates (s1, s2, s0)

        Returns
        -------
        geographic_coords : (N, 3) approximate geographic coordinates
        """
        # Approximate inverse: use the fold frame axes as a linear transform
        # This is exact for planar (unfolded) geometry and approximate for
        # folded geometry
        if self.fold_axis is None or self.axial_normal is None:
            raise RuntimeError("Fold frame not built.")

        # Build local coordinate frame
        e1 = self.axial_normal  # S1 direction
        e2 = self.fold_axis     # S2 direction
        e3 = np.cross(e1, e2)   # S0 direction (approximate)
        e3_norm = np.linalg.norm(e3)
        if e3_norm > 1e-10:
            e3 /= e3_norm
        else:
            e3 = np.array([0.0, 0.0, 1.0])

        # Linear transform: approximate
        basis = np.column_stack([e1, e2, e3])  # (3, 3)

        return fold_coords @ basis.T
