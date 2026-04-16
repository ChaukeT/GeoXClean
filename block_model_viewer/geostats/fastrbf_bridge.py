"""
FastRBF Bridge — Adapts the standalone geostats.estimation engine to the
GeoX controller/panel interface.

This module translates between the GeoX conventions (DataFrames, grid_spec
dicts, PyVista meshes) and the FastRBF engine's pure NumPy API.

It also bridges variogram results from the variogram panel into
RBFConfig parameters for the FastRBF engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Import the standalone FastRBF engine
try:
    from geostats.estimation.config import RBFConfig, KernelType, DriftType
    from geostats.estimation.fastrbf_engine import FastRBFEngine, FittedRBF
    from geostats.estimation.block_estimator import BlockModelEstimator, EstimationResult
    from geostats.estimation.cross_validation import loo_cross_validation, CVResult
    from geostats.estimation.diagnostics import EstimationDiagnostics
    from geostats.estimation.audit import JORCAuditRecord
    FASTRBF_AVAILABLE = True
except ImportError:
    FASTRBF_AVAILABLE = False
    logger.warning("FastRBF engine not available (geostats package not found)")


# ── Kernel name mapping: GeoX panel names → FastRBF KernelType ───────────

_KERNEL_MAP = {
    "spheroidal": KernelType.SPHEROIDAL,
    "spherical": KernelType.SPHERICAL,
    "gaussian": KernelType.GAUSSIAN,
    "exponential": KernelType.EXPONENTIAL,
    "linear": KernelType.LINEAR,
    "cubic": KernelType.CUBIC,
    "generalised_cauchy": KernelType.GENERALISED_CAUCHY,
    "gen_cauchy": KernelType.GENERALISED_CAUCHY,
    # Aliases for convenience
    "Spheroidal (Leapfrog default)": KernelType.SPHEROIDAL,
    "Spherical": KernelType.SPHERICAL,
    "Gaussian": KernelType.GAUSSIAN,
    "Exponential": KernelType.EXPONENTIAL,
    "Linear": KernelType.LINEAR,
    "Cubic": KernelType.CUBIC,
} if FASTRBF_AVAILABLE else {}

_DRIFT_MAP = {
    "constant": DriftType.CONSTANT,
    "linear": DriftType.LINEAR,
    "none": DriftType.NONE,
    0: DriftType.CONSTANT,
    1: DriftType.LINEAR,
    None: DriftType.NONE,
} if FASTRBF_AVAILABLE else {}


def variogram_results_to_rbf_config(
    variogram_results: Dict[str, Any],
    overrides: Optional[Dict[str, Any]] = None,
) -> "RBFConfig":
    """
    Build an RBFConfig from variogram panel results.

    Extracts sill, nugget, range, anisotropy angles/ratios from the
    variogram result dict (as stored in DataRegistry).

    Parameters
    ----------
    variogram_results : dict
        From ``registry.get_variogram_results()`` — contains
        ``combined_3d_model``, ``omni_variogram``, etc.
    overrides : dict, optional
        Explicit overrides for any config field.

    Returns
    -------
    RBFConfig
    """
    if not FASTRBF_AVAILABLE:
        raise ImportError("FastRBF engine is not installed")

    # Priority: combined_3d_model > major_variogram > omni_variogram
    combined = variogram_results.get("combined_3d_model", {})
    omni = variogram_results.get("omni_variogram", {})
    source = combined if combined else omni

    if not source:
        raise ValueError(
            "No fitted variogram model found in variogram_results. "
            "Run variogram analysis first."
        )

    # Extract kernel type
    model_type_str = source.get("model_type", "spheroidal")
    kernel_type = _KERNEL_MAP.get(model_type_str, KernelType.SPHEROIDAL)

    # Extract sill and nugget
    # NOTE: "sill" key in variogram results is the TOTAL sill (C0+C1), not partial
    nugget = float(source.get("nugget", 0.0))
    raw_sill = float(source.get("sill", 1.0))
    total_sill = float(source.get("total_sill", raw_sill))
    # Ensure total_sill > nugget (partial sill must be positive)
    if total_sill <= nugget:
        total_sill = nugget + max(raw_sill, 0.01)

    # Extract range
    base_range = float(source.get("major_range", source.get("range", 100.0)))

    # Extract anisotropy
    major_range = float(source.get("major_range", base_range))
    minor_range = float(source.get("minor_range", major_range))
    vertical_range = float(source.get("vertical_range", major_range))

    # Compute ratios (relative to major)
    ratio_major = 1.0
    ratio_semi = minor_range / major_range if major_range > 0 else 1.0
    ratio_minor = vertical_range / major_range if major_range > 0 else 1.0

    # Angles
    azimuth = float(source.get("azimuth", 0.0))
    dip = float(source.get("dip", 0.0))
    pitch = float(source.get("plunge", source.get("pitch", 0.0)))

    config_dict = dict(
        kernel_type=kernel_type,
        total_sill=total_sill,
        nugget=nugget,
        base_range=major_range,
        alpha=int(source.get("alpha", 5)),
        drift=DriftType.CONSTANT,
        azimuth=azimuth % 360,
        dip=max(-90, min(90, dip)),
        pitch=max(-90, min(90, pitch)),
        ratio_major=ratio_major,
        ratio_semi=max(0.01, ratio_semi),
        ratio_minor=max(0.01, ratio_minor),
    )

    if overrides:
        config_dict.update(overrides)

    return RBFConfig(**config_dict)


def params_to_rbf_config(params: Dict[str, Any]) -> "RBFConfig":
    """
    Build an RBFConfig from RBF panel parameter dict.

    This is used when the user explicitly sets FastRBF parameters
    from the panel UI rather than importing from variogram results.
    """
    if not FASTRBF_AVAILABLE:
        raise ImportError("FastRBF engine is not installed")

    kernel_str = params.get("fastrbf_kernel", "spheroidal")
    kernel_type = _KERNEL_MAP.get(kernel_str, KernelType.SPHEROIDAL)

    drift_val = params.get("fastrbf_drift", "constant")
    drift_type = _DRIFT_MAP.get(drift_val, DriftType.CONSTANT)

    # Anisotropy from panel ranges
    aniso = params.get("anisotropy_enabled", False)
    if aniso:
        range_x = params.get("range_x", 100.0)
        range_y = params.get("range_y", 100.0)
        range_z = params.get("range_z", 100.0)
        base_range = range_x
        ratio_semi = range_y / range_x if range_x > 0 else 1.0
        ratio_minor = range_z / range_x if range_x > 0 else 1.0
    else:
        base_range = params.get("fastrbf_range", 100.0)
        ratio_semi = 1.0
        ratio_minor = 1.0

    return RBFConfig(
        kernel_type=kernel_type,
        total_sill=params.get("fastrbf_sill", 1.0),
        nugget=params.get("fastrbf_nugget", 0.0),
        base_range=base_range,
        alpha=params.get("fastrbf_alpha", 5),
        drift=drift_type,
        azimuth=params.get("azimuth", 0.0) % 360,
        dip=max(-90.0, min(90.0, params.get("dip", 0.0))),
        pitch=max(-90.0, min(90.0, params.get("plunge", 0.0))),
        ratio_major=1.0,
        ratio_semi=max(0.01, ratio_semi),
        ratio_minor=max(0.01, ratio_minor),
        search_max_samples=params.get("fastrbf_max_samples", 24),
        search_min_samples=params.get("fastrbf_min_samples", 8),
        search_max_per_octant=params.get("fastrbf_max_per_octant", 4),
        search_min_octants=params.get("fastrbf_min_octants", 2),
        discretisation_points=params.get("fastrbf_discretisation", 4),
        accuracy=params.get("fastrbf_accuracy", None),
        clip_min=params.get("fastrbf_clip_min", None),
        clip_max=params.get("fastrbf_clip_max", None),
    )


def fastrbf_interpolate_3d(
    coords: np.ndarray,
    values: np.ndarray,
    grid_spec: Dict[str, Any],
    config: "RBFConfig",
    run_cv: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Run FastRBF interpolation on a 3D grid — the main bridge function.

    Translates the GeoX grid_spec into block centroids, runs the
    FastRBF block estimator, and packages results.

    Parameters
    ----------
    coords : (N, 3) ndarray — sample coordinates
    values : (N,) ndarray — sample values
    grid_spec : dict with nx, ny, nz, xmin, ymin, zmin, xinc, yinc, zinc
    config : RBFConfig
    run_cv : bool — if True, also run LOO cross-validation
    progress_callback : callable(int, str), optional

    Returns
    -------
    dict with keys:
        grid_values, x_coords, y_coords, z_coords,
        estimation_result, cv_result, diagnostics, audit, config
    """
    if not FASTRBF_AVAILABLE:
        raise ImportError("FastRBF engine is not installed")

    # Filter NaN/Inf values (common in drillhole assays)
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).ravel()
    valid_mask = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    if not valid_mask.all():
        n_removed = int((~valid_mask).sum())
        logger.warning(
            f"FastRBF: Removing {n_removed}/{len(values)} points with NaN/Inf values"
        )
        coords = coords[valid_mask]
        values = values[valid_mask]
    if len(coords) == 0:
        raise ValueError("No valid (non-NaN) data points after filtering")

    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    nz = grid_spec["nz"]
    xmin = grid_spec["xmin"]
    ymin = grid_spec["ymin"]
    zmin = grid_spec["zmin"]
    xinc = grid_spec["xinc"]
    yinc = grid_spec["yinc"]
    zinc = grid_spec["zinc"]

    # Build coordinate arrays
    x_coords = np.arange(nx) * xinc + xmin + xinc / 2
    y_coords = np.arange(ny) * yinc + ymin + yinc / 2
    z_coords = np.arange(nz) * zinc + zmin + zinc / 2

    # Build block centroids (flattened)
    gx, gy, gz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
    centroids = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    block_sizes = np.array([xinc, yinc, zinc])

    # ── Coordinate alignment check ──────────────────────────────
    # If the grid was built from renderer bounds (local coords) but the
    # data is still in original coords (UTM), the search will find 0
    # neighbours for every block.  Detect and auto-shift the grid.
    data_center = np.mean(coords, axis=0)
    grid_center = np.array([
        (x_coords[0] + x_coords[-1]) / 2,
        (y_coords[0] + y_coords[-1]) / 2,
        (z_coords[0] + z_coords[-1]) / 2,
    ])
    offset = data_center - grid_center
    shift_mag = float(np.linalg.norm(offset))
    search_radius = config.base_range * 2.0  # max multi-pass radius
    if shift_mag > search_radius:
        logger.warning(
            "FastRBF: Grid/data coordinate mismatch detected!  "
            "Grid center=(%.1f, %.1f, %.1f), Data center=(%.1f, %.1f, %.1f), "
            "offset=%.1f m.  Auto-shifting grid to align with data.",
            grid_center[0], grid_center[1], grid_center[2],
            data_center[0], data_center[1], data_center[2], shift_mag,
        )
        # Shift grid origin so centroids align with data
        xmin += offset[0]
        ymin += offset[1]
        zmin += offset[2]
        x_coords = np.arange(nx) * xinc + xmin + xinc / 2
        y_coords = np.arange(ny) * yinc + ymin + yinc / 2
        z_coords = np.arange(nz) * zinc + zmin + zinc / 2
        gx, gy, gz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        centroids = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    if progress_callback:
        progress_callback(10, f"FastRBF: {len(centroids)} blocks, {len(values)} samples...")

    # Run block estimation
    estimator = BlockModelEstimator(config)
    est_result = estimator.estimate(centroids, block_sizes, coords, values)

    if progress_callback:
        progress_callback(70, "FastRBF: Estimation complete, building results...")

    # Reshape to 3D grid (ijk order)
    grid_values = est_result.estimated_values.reshape((nx, ny, nz))

    # Cross-validation (optional)
    cv_result = None
    if run_cv:
        if progress_callback:
            progress_callback(75, "FastRBF: Running cross-validation...")
        try:
            cv_result = loo_cross_validation(
                coords, values, config, progress_callback=progress_callback
            )
        except Exception as e:
            logger.warning("FastRBF CV failed: %s", e)

    # Diagnostics
    diag = EstimationDiagnostics()
    diagnostics = {}

    if cv_result is not None:
        slope_res = diag.slope_of_regression(cv_result.actual, cv_result.estimated)
        bias_res = diag.global_bias_check(est_result.estimated_values, float(np.mean(values)))
        diagnostics = {
            "rmse": cv_result.rmse,
            "mae": cv_result.mae,
            "r_squared": cv_result.r_squared,
            "mean_error": cv_result.mean_error,
            "correlation": cv_result.correlation,
            "slope_of_regression": slope_res.slope,
            "global_bias_percent": bias_res.bias_percent,
            "conditionally_biased": slope_res.is_conditionally_biased,
            "bias_flagged": bias_res.is_flagged,
        }
    else:
        bias_res = diag.global_bias_check(est_result.estimated_values, float(np.mean(values)))
        diagnostics = {
            "global_bias_percent": bias_res.bias_percent,
            "bias_flagged": bias_res.is_flagged,
        }

    if progress_callback:
        progress_callback(90, "FastRBF: Building audit trail...")

    # Classification counts
    classification = est_result.classification
    class_counts = {
        "Measured": int(np.sum(classification == "Measured")),
        "Indicated": int(np.sum(classification == "Indicated")),
        "Inferred": int(np.sum(classification == "Inferred")),
        "Unclassified": int(np.sum(classification == "Unclassified")),
    }

    if progress_callback:
        progress_callback(100, "FastRBF: Complete")

    return {
        "grid_values": grid_values,
        "x_coords": x_coords,
        "y_coords": y_coords,
        "z_coords": z_coords,
        "estimation_result": est_result,
        "cv_result": cv_result,
        "diagnostics": diagnostics,
        "classification": classification.reshape((nx, ny, nz)),
        "classification_counts": class_counts,
        "search_pass": est_result.search_pass.reshape((nx, ny, nz)),
        "num_samples": est_result.num_samples.reshape((nx, ny, nz)),
        "config": config,
        "n_blocks_estimated": est_result.n_blocks_estimated,
        "n_blocks_total": est_result.n_blocks_total,
    }
