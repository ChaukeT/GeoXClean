"""
Bridge between the v2 VariogramEngine and the existing GeoX variogram pipeline.

The existing system (variogram3d.run_variogram_pipeline) returns a dict with
specific keys that the variogram panel, ARBF panel, and data registry all
expect. This module provides:

1. ``run_variogram_pipeline_v2()`` — drop-in replacement that uses the new
   VariogramEngine internally but returns the same dict format.
2. Helper functions to convert between v2 dataclasses and the legacy dict
   format used by the registry.

Usage:
    In variogram3d.py, the existing run_variogram_pipeline() can delegate to
    this bridge when the v2 engine is available.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .variogram_engine_v2 import (
    DirectionSpec,
    DriftConfig,
    ExperimentalVariogram,
    NestedStructure,
    SupportMetadata,
    TransformConfig,
    VariogramEngine,
    VariogramModel3D,
    VariogramSearchConfig,
    default_directions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data hash (matches geostats.variogram_gates.compute_data_hash)
# ---------------------------------------------------------------------------

def _compute_data_hash(coords: np.ndarray, values: np.ndarray, variable: str) -> str:
    h = hashlib.sha256()
    h.update(variable.encode("utf-8"))
    h.update(coords.tobytes())
    h.update(values.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Convert ExperimentalVariogram → legacy DataFrame
# ---------------------------------------------------------------------------

def _exp_to_dataframe(exp: ExperimentalVariogram) -> pd.DataFrame:
    """Convert an ExperimentalVariogram to the legacy DataFrame format.

    Filters out lags with NaN gamma (insufficient pairs) to match
    what the legacy pipeline returns and what the plot code expects.
    """
    df = pd.DataFrame({
        "distance": exp.lag_centres,
        "gamma": exp.gamma,
        "npairs": exp.n_pairs,
    })
    # Remove rows where gamma is NaN (lag had fewer than min_pairs)
    df = df.dropna(subset=["gamma"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Convert VariogramModel3D → legacy fitted_models dict
# ---------------------------------------------------------------------------

def _model_to_fitted_dict(model: VariogramModel3D) -> Dict[str, Dict[str, float]]:
    """Convert a VariogramModel3D to the legacy fitted_models format.

    Returns a dict keyed by model_type with standardised sill semantics:
        "nugget"       — C₀ (nugget effect)
        "partial_sill" — C  (structured variance, excluding nugget)
        "sill"         — C  (alias for partial_sill, kept for backward compat)
        "total_sill"   — C₀+C (total variance reached at the sill)
        "range"        — practical range (major direction)
    """
    if not model.structures:
        return {}

    primary = max(model.structures, key=lambda s: s.sill)
    partial = primary.sill
    return {
        primary.model: {
            "model_type": primary.model,
            "nugget": model.nugget,
            "partial_sill": partial,
            "sill": partial,                   # backward-compat alias
            "range": primary.ranges[0],
            "total_sill": model.total_sill,
        }
    }


# ---------------------------------------------------------------------------
# Convert VariogramModel3D → legacy combined_3d_model dict
# ---------------------------------------------------------------------------

def _model_to_combined_3d(
    model: VariogramModel3D,
    major_range: float,
    minor_range: float,
    vert_range: float,
) -> Dict[str, Any]:
    """Build the combined_3d_model dict that the ARBF panel imports from."""
    primary = max(model.structures, key=lambda s: s.sill) if model.structures else None
    model_type = primary.model if primary else "spherical"

    structures = []
    for s in model.structures:
        # Use the externally-resolved directional ranges, not the per-structure
        # range tuple (which has unconstrained axes from the 1-D fit).
        structures.append({
            "model_type": s.model,
            "contribution": s.sill,
            "range_major": major_range,
            "range_minor": minor_range,
            "range_vertical": vert_range,
        })

    partial = sum(s.sill for s in model.structures)
    return {
        "model_type": model_type,
        "nugget": model.nugget,
        "partial_sill": partial,
        "sill": partial,                       # backward-compat alias
        "total_sill": model.total_sill,
        "major_range": major_range,
        "minor_range": minor_range,
        "vertical_range": vert_range,
        "structures": structures,
    }


# ---------------------------------------------------------------------------
# Main bridge function
# ---------------------------------------------------------------------------

def run_variogram_pipeline_v2(
    data: pd.DataFrame,
    xcol: str = "X",
    ycol: str = "Y",
    zcol: str = "Z",
    vcol: str = "Fe",
    hole_id_col: Optional[str] = None,
    from_col: Optional[str] = None,
    to_col: Optional[str] = None,
    default_azimuth: Optional[float] = None,
    default_dip: Optional[float] = None,
    z_positive_up: bool = True,
    nlag: int = 12,
    lag_distance: float = 25.0,
    lag_tolerance: Optional[float] = None,
    azimuth_tolerance: Optional[float] = None,
    dip_tolerance: Optional[float] = None,
    model_types: Optional[list] = None,
    use_sill_norm: bool = True,
    random_state: Optional[int] = 42,
    auto_lags: bool = False,
    n_structures: int = 1,
    global_nugget: Optional[float] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    sample_weights: Optional[np.ndarray] = None,
    bandwidth: Optional[float] = None,
    variable: Optional[str] = None,
    values: Optional[np.ndarray] = None,
    # v2-specific parameters
    transform_mode: str = "raw",
    drift_mode: str = "none",
    use_robust_estimator: bool = True,
    domain_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the v2 variogram engine and return results in the legacy format.

    Accepts all the same parameters as the original run_variogram_pipeline()
    plus optional v2-specific parameters for transforms and drift.
    """
    def _progress(pct: int, msg: str) -> None:
        if progress_callback:
            try:
                progress_callback(pct, msg)
            except Exception:
                pass

    _progress(2, "Variogram v2: Preparing data...")

    # ── Extract arrays ────────────────────────────────────────────────
    for cx, cy, cz in [(xcol, ycol, zcol), ("X", "Y", "Z"), ("x", "y", "z"), ("EAST", "NORTH", "RL")]:
        if cx in data.columns and cy in data.columns and cz in data.columns:
            break
    else:
        raise ValueError("Cannot find coordinate columns in data")

    clean = data.dropna(subset=[cx, cy, cz, vcol])
    coords = clean[[cx, cy, cz]].to_numpy(float)
    if values is not None:
        all_values = np.asarray(values, dtype=float).ravel()
        if len(all_values) == len(data):
            # Align by original DataFrame index so rows dropped by dropna don't
            # cause a position shift (BUG-FIX: was silently using wrong values
            # for every row after the first NaN-dropped row).
            vals = all_values[clean.index.to_numpy()]
        else:
            # Length mismatch — caller already passed a pre-filtered array;
            # fall back to positional truncation and warn.
            logger.warning(
                "run_variogram_pipeline_v2: external 'values' length (%d) does not match "
                "data length (%d). Falling back to positional truncation — values may be "
                "misaligned if data contains NaN rows.",
                len(all_values), len(data),
            )
            vals = all_values[:coords.shape[0]]
    else:
        vals = clean[vcol].to_numpy(float)

    # Domain IDs
    domain_ids = None
    if domain_column and domain_column in clean.columns:
        domain_ids = clean[domain_column].astype(str).values

    # Align external sample_weights with the same NaN-dropped rows used for
    # coords / vals, using the pandas index (same pattern as `values` above).
    weights_aligned: Optional[np.ndarray] = None
    if sample_weights is not None:
        try:
            _sw = np.asarray(sample_weights, dtype=float).ravel()
            if _sw.size == len(data):
                weights_aligned = _sw[clean.index.to_numpy()]
            elif _sw.size == coords.shape[0]:
                weights_aligned = _sw
            else:
                logger.warning(
                    "run_variogram_pipeline_v2: sample_weights length (%d) "
                    "does not match data length (%d) or clean rows (%d). "
                    "Ignoring weights.",
                    _sw.size, len(data), coords.shape[0],
                )
        except Exception as _exc:
            logger.warning(
                "run_variogram_pipeline_v2: could not align sample_weights: %s",
                _exc,
            )

    variable_name = variable or vcol

    # Compute the sample variance up front so we can pass it as a sill
    # cap to every fit call. Without this, v2's fit_nested_model had no
    # stationary-variance reference and noisy directional fits could
    # produce sills 2x+ sample variance (and inflated ranges). Preferring
    # the transformed `vals` here means the cap matches the space the
    # variogram is fitted in (NS-transformed when transform is active).
    # When declustering weights are present, use a weighted variance so
    # the cap matches the declustered variable distribution.
    try:
        if vals is None:
            sample_variance_early = None
        else:
            _finite_mask = np.isfinite(vals)
            _fvals = vals[_finite_mask]
            if _fvals.size == 0:
                sample_variance_early = None
            elif weights_aligned is not None:
                _fw = weights_aligned[_finite_mask]
                _wsum = float(np.sum(_fw))
                if _wsum > 0:
                    _wmu = float(np.sum(_fvals * _fw) / _wsum)
                    sample_variance_early = float(
                        np.sum(_fw * (_fvals - _wmu) ** 2) / _wsum
                    )
                else:
                    sample_variance_early = float(np.var(_fvals))
            else:
                sample_variance_early = float(np.var(_fvals))
    except Exception:
        sample_variance_early = None

    # ── Auto-lag computation ──────────────────────────────────────────
    # When auto_lags=True, infer lag_distance and nlag from data geometry
    # rather than using the UI spinbox values (which are often left at small
    # defaults like 5m / 18 lags that are too small for typical drill spacing).
    if auto_lags:
        try:
            xy = coords[:, :2]
            # Collapse composites to collar positions using a 50 m grid so that
            # inclined hole composites (which drift 1-2 m horizontally per
            # interval) don't masquerade as unique collars.
            grid = 50.0
            xy_snapped = np.round(xy / grid) * grid
            unique_xy = np.unique(xy_snapped, axis=0)
            if len(unique_xy) >= 2:
                try:
                    from scipy.spatial import cKDTree as _cKDTree
                    tree = _cKDTree(unique_xy)
                    dists, _ = tree.query(unique_xy, k=2)
                    drill_spacing = float(np.median(dists[:, 1]))
                except Exception:
                    extents = coords.max(axis=0) - coords.min(axis=0)
                    drill_spacing = float(np.sqrt(extents[0]**2 + extents[1]**2) / np.sqrt(len(unique_xy)))
            else:
                extents = coords.max(axis=0) - coords.min(axis=0)
                drill_spacing = float(np.sqrt(extents[0]**2 + extents[1]**2) / max(np.sqrt(len(coords)), 1))
            # Lag = half drill spacing; at least 5 m, at most 200 m
            lag_distance = float(np.clip(drill_spacing * 0.5, 5.0, 200.0))
            # n_lags: cover at least 3× the drill spacing (to see range fall-off)
            horiz_extent = float(np.sqrt(
                (coords[:, 0].max() - coords[:, 0].min())**2 +
                (coords[:, 1].max() - coords[:, 1].min())**2
            ))
            target = horiz_extent * 0.5
            nlag = int(np.clip(np.ceil(target / lag_distance), 12, 25))
            logger.info(
                "v2 auto-lag: drill_spacing=%.1fm → lag_distance=%.1fm, nlag=%d, max_range=%.1fm",
                drill_spacing, lag_distance, nlag, lag_distance * nlag,
            )
        except Exception as _e:
            logger.warning("v2 auto-lag computation failed (%s), using UI values", _e)

    # ── Build engine ──────────────────────────────────────────────────
    _progress(5, "Variogram v2: Building engine...")

    support = SupportMetadata(input_support="composite")
    engine = VariogramEngine(
        coords,
        vals,
        domain_ids=domain_ids,
        support_metadata=support,
        sample_weights=weights_aligned,
    )
    if weights_aligned is not None:
        logger.info(
            "v2 variogram engine: using declustering weights for %d samples "
            "(weight sum = %.2f, effective N = %.1f).",
            weights_aligned.size,
            float(np.sum(weights_aligned)),
            float(np.sum(weights_aligned) ** 2 / max(np.sum(weights_aligned ** 2), 1e-12)),
        )

    # Search config
    search = VariogramSearchConfig(
        lag_size=lag_distance,
        n_lags=nlag,
        min_pairs=30,
        use_robust_estimator=use_robust_estimator,
        random_seed=random_state or 42,
    )

    transform = TransformConfig(mode=transform_mode)
    drift = DriftConfig(mode=drift_mode)

    # Determine domain
    domain_name = "default"
    if domain_ids is not None:
        unique_domains = np.unique(domain_ids)
        domain_name = str(unique_domains[0])  # Use first domain for primary variogram

    # ── Directions ────────────────────────────────────────────────────
    az_tol = azimuth_tolerance or 22.5
    dip_tol = dip_tolerance or 22.5

    # Use PCA or user-specified azimuth
    if default_azimuth is not None:
        az_rad = np.deg2rad(default_azimuth)
        major_vec = np.array([np.sin(az_rad), np.cos(az_rad), 0.0])
        minor_vec = np.array([-np.cos(az_rad), np.sin(az_rad), 0.0])
    else:
        # PCA-based
        try:
            rot = engine.infer_rotation_from_pca(domain_name)
            major_vec = rot[0]
            minor_vec = rot[1]
        except Exception:
            major_vec = np.array([1.0, 0.0, 0.0])
            minor_vec = np.array([0.0, 1.0, 0.0])

    directions = [
        DirectionSpec("omni", np.array([1.0, 0.0, 0.0]), tolerance_deg=180.0, bandwidth=bandwidth),
        DirectionSpec("major", major_vec, tolerance_deg=az_tol, bandwidth=bandwidth),
        DirectionSpec("minor", minor_vec, tolerance_deg=az_tol, bandwidth=bandwidth),
        DirectionSpec("vertical", np.array([0.0, 0.0, 1.0]), tolerance_deg=dip_tol, bandwidth=bandwidth),
    ]

    # ── Compute experimental variograms ───────────────────────────────
    _progress(15, "Variogram v2: Computing experimental variograms...")

    exp_variograms: Dict[str, ExperimentalVariogram] = {}
    for d in directions:
        try:
            exp_variograms[d.name] = engine.compute_experimental_variogram(
                domain_name=domain_name,
                transform=transform,
                drift=drift,
                direction=d,
                search=search,
            )
        except ValueError as exc:
            logger.warning("Direction '%s' failed: %s", d.name, exc)

    _progress(40, "Variogram v2: Fitting models...")

    # ── Fit models ────────────────────────────────────────────────────
    mtypes = model_types or ["spherical"]
    if isinstance(mtypes, str):
        mtypes = [mtypes]

    fitted_models: Dict[str, Dict] = {}
    fitted_v2_models: Dict[str, VariogramModel3D] = {}

    rotation = np.eye(3, dtype=float)
    try:
        rotation = engine.infer_rotation_from_pca(domain_name)
    except Exception:
        pass

    for dir_name, exp in exp_variograms.items():
        _progress(40 + 10 * list(exp_variograms.keys()).index(dir_name), f"Fitting {dir_name}...")
        try:
            # Provide direction-appropriate initial ranges so the optimizer
            # starts on the correct axis for each direction:
            # - major: expect longest range (rmaj), moderate rsemi, short rmin
            # - minor: the constrained axis is rsemi; initialise it smaller
            # - vertical: the constrained axis is rmin; initialise it shortest
            # - omni: isotropic — use identity rotation so the three range
            #   parameters don't couple through PCA eigenvectors, which causes
            #   the optimizer to converge on a spuriously long range.
            max_lag = float(exp.lag_centres[-1]) if len(exp.lag_centres) > 0 else lag_distance * nlag
            if dir_name == "omni":
                fit_rotation = np.eye(3, dtype=float)
                init_ranges = (max_lag, max_lag, max_lag)
            elif dir_name == "minor":
                fit_rotation = rotation
                init_ranges = (max_lag, max_lag * 0.5, max_lag * 0.25)
            elif dir_name == "vertical":
                fit_rotation = rotation
                init_ranges = (max_lag, max_lag * 0.6, max_lag * 0.25)
            else:
                fit_rotation = rotation
                init_ranges = (max_lag, max_lag * 0.6, max_lag * 0.25)
            model = engine.fit_nested_model(
                exp,
                model_types=mtypes,
                n_structures=n_structures,
                initial_rotation=fit_rotation,
                initial_ranges=init_ranges,
                sample_variance=sample_variance_early,
                use_sill_norm=use_sill_norm,
            )
            fitted_models[dir_name] = _model_to_fitted_dict(model)
            fitted_v2_models[dir_name] = model
        except ValueError as exc:
            logger.warning("Fitting '%s' failed: %s", dir_name, exc)
            fitted_models[dir_name] = {}

    # ── Downhole variogram (within-hole pairs ONLY) ─────────────────
    # A downhole variogram MUST restrict to within-hole pairs for correct
    # nugget estimation.  The v2 engine's directional search would mix
    # inter-hole vertical pairs, inflating the short-range semivariance.
    # Use the shared helper in geostats.downhole_variogram; keeping this
    # call free of any legacy Variogram3D import is what makes v2 a
    # self-contained path (consolidation plan A3).
    downhole_df = None
    _progress(75, "Variogram v2: Computing downhole variogram (within-hole pairs)...")
    try:
        from .downhole_variogram import compute_downhole_variogram

        # Detect hole-ID column
        _hole_col = hole_id_col
        if not _hole_col or _hole_col not in data.columns:
            for candidate in ["HOLEID", "hole_id", "HoleID", "HOLE_ID", "holeid",
                              "Hole_ID", "hole", "HOLE"]:
                if candidate in data.columns:
                    _hole_col = candidate
                    break

        if _hole_col and _hole_col in clean.columns:
            _hole_arr = clean[_hole_col].astype(str).to_numpy()
            _from_arr = (
                clean[from_col].to_numpy(float)
                if from_col and from_col in clean.columns else None
            )
            _to_arr = (
                clean[to_col].to_numpy(float)
                if to_col and to_col in clean.columns else None
            )
            dh_result = compute_downhole_variogram(
                coords=coords,
                values=vals,
                hole_ids=_hole_arr,
                from_depths=_from_arr,
                to_depths=_to_arr,
                sample_weights=weights_aligned,
                n_lags=15,
            )
            if dh_result is not None and len(dh_result) > 0:
                downhole_df = dh_result
                logger.info(
                    "Downhole variogram: %d lags from within-hole pairs "
                    "(%d holes, from/to %s)",
                    len(dh_result),
                    int(np.unique(_hole_arr).size),
                    "yes" if (_from_arr is not None and _to_arr is not None) else "no",
                )
        else:
            logger.info("No hole-ID column found; skipping true downhole variogram")
    except Exception as exc:
        logger.warning("Within-hole downhole variogram failed: %s", exc, exc_info=True)

    # Fallback: vertical directional variogram if no hole data available
    if downhole_df is None or downhole_df.empty:
        try:
            dh_dir = DirectionSpec("downhole", np.array([0.0, 0.0, -1.0]), tolerance_deg=30.0)
            dh_exp = engine.compute_experimental_variogram(
                domain_name=domain_name,
                transform=transform,
                drift=drift,
                direction=dh_dir,
                search=VariogramSearchConfig(
                    lag_size=max(lag_distance * 0.5, 1.0),
                    n_lags=min(nlag, 15),
                    min_pairs=15,
                    use_robust_estimator=use_robust_estimator,
                    random_seed=random_state or 42,
                ),
            )
            downhole_df = _exp_to_dataframe(dh_exp)
            exp_variograms["downhole"] = dh_exp
            logger.info("Using vertical directional variogram as downhole fallback")
        except Exception as exc2:
            logger.warning("Vertical fallback downhole failed: %s", exc2)

    # ── Fit model to downhole variogram ─────────────────────────────
    if downhole_df is not None and not downhole_df.empty:
        try:
            dh_lags = downhole_df["distance"].values
            dh_gamma = downhole_df["gamma"].values
            dh_pairs = downhole_df["npairs"].values if "npairs" in downhole_df.columns else np.ones(len(dh_lags))
            # Build a lightweight ExperimentalVariogram for the fitter
            dh_exp_for_fit = ExperimentalVariogram(
                lag_centres=dh_lags, gamma=dh_gamma, n_pairs=dh_pairs.astype(int),
                direction_name="downhole", transform_mode="raw", drift_mode="none",
                domain_name=domain_name, support_metadata=SupportMetadata(),
                robust=False, direction_unit_vector=np.array([0.0, 0.0, -1.0]),
            )
            dh_model = engine.fit_nested_model(
                dh_exp_for_fit,
                model_types=mtypes,
                n_structures=1,
                sample_variance=sample_variance_early,
                use_sill_norm=use_sill_norm,
            )
            fitted_models["downhole"] = _model_to_fitted_dict(dh_model)
            fitted_v2_models["downhole"] = dh_model
            logger.info(
                "Downhole model fitted: nugget=%.2f, sill=%.2f, range=%.1f",
                dh_model.nugget, dh_model.structures[0].sill if dh_model.structures else 0,
                dh_model.structures[0].ranges[0] if dh_model.structures else 0,
            )
        except Exception as exc:
            logger.warning("Downhole model fitting failed: %s", exc)

    # ── Build combined 3D model ───────────────────────────────────────
    _progress(85, "Variogram v2: Building combined 3D model...")

    # Use the best available directional model for range inference
    best_model = (
        fitted_v2_models.get("major")
        or fitted_v2_models.get("omni")
        or next(iter(fitted_v2_models.values()), None)
    )
    minor_model = fitted_v2_models.get("minor")
    vert_model = fitted_v2_models.get("vertical")

    def _dominant_range_axis(direction_name: str) -> int:
        """Determine which range index (0=rmaj, 1=rsemi, 2=rmin) is actually
        constrained by a directional fit.

        The fit constructs 1-D lag vectors along the direction's unit vector,
        then the model evaluates semivariance via ``h_rot = h @ R^T`` followed
        by anisotropic scaling ``u = h_rot * (1/ranges)``.  Only the component
        of h_rot that is large drives the fit — the others are unconstrained
        and converge to noise.  We detect the active axis by projecting the
        direction unit vector into the rotation frame.
        """
        exp = exp_variograms.get(direction_name)
        model = fitted_v2_models.get(direction_name)
        if exp is None or model is None:
            return 0  # fallback: assume axis 0
        duv = exp.direction_unit_vector
        if duv is None:
            return 0
        duv_norm = duv / max(float(np.linalg.norm(duv)), 1e-12)
        h_rot = duv_norm @ model.rotation_matrix.T
        return int(np.argmax(np.abs(h_rot)))

    # H4 fix: v2's fit_nested_model does a 1D curve_fit per directional
    # variogram and stores the scalar fit_range in ``structures[0].ranges[0]``
    # alongside hardcoded ratio placeholders at indices 1 and 2 (e.g.
    # ``(fit_range, fit_range*0.5, fit_range*0.25)``). Reading ``ranges[1]``
    # or ``ranges[2]`` via ``_dominant_range_axis`` silently halves (or
    # quarters) the real fit value. Since each directional variogram is
    # fitted INDEPENDENTLY, the correct combined-3D range for each axis
    # is the scalar fit_range from that direction's own fit — always
    # ``ranges[0]``, regardless of which axis the rotation matrix aligns
    # the direction unit vector with.
    if best_model and best_model.structures:
        major_range = best_model.structures[0].ranges[0]
    else:
        major_range = lag_distance * nlag * 0.5

    if minor_model and minor_model.structures:
        minor_range = minor_model.structures[0].ranges[0]
    else:
        minor_range = major_range * 0.6

    if vert_model and vert_model.structures:
        vert_range = vert_model.structures[0].ranges[0]
    else:
        vert_range = major_range * 0.25

    # Reject unreliable directional fits before sorting. A stationary
    # variogram's fitted range cannot legitimately exceed the diameter of
    # the drilled region (≈ 2 × horizontal max_lag used in the fit), and
    # the minor axis must be ≤ the major axis by definition. Without
    # these guards a noisy direction with a wild fit gets SWAPPED into
    # the major slot below, corrupting the whole 3D model (and the
    # azimuth with it).
    def _minor_fit_unreliable() -> str:
        if not minor_model or not minor_model.structures:
            return "no structures"
        if sample_variance_early is not None and sample_variance_early > 0:
            if minor_model.total_sill > 1.3 * sample_variance_early:
                return (
                    f"total_sill {minor_model.total_sill:.3f} > 1.3x "
                    f"sample variance {sample_variance_early:.3f}"
                )
        if best_model and best_model.structures:
            if minor_range > major_range * 1.15:
                return (
                    f"minor range {minor_range:.1f}m exceeds major "
                    f"{major_range:.1f}m by > 15% (geometrically impossible)"
                )
        return ""

    _minor_reject_reason = _minor_fit_unreliable()
    if _minor_reject_reason:
        logger.warning(
            "v2 variogram: rejecting minor directional fit — %s. "
            "Falling back to min(major, omni) and keeping azimuth fixed.",
            _minor_reject_reason,
        )
        # Prefer omni as fallback (reliable, many pairs); clamp to major.
        _omni_fallback = None
        if (
            "omni" in fitted_v2_models
            and fitted_v2_models["omni"] is not None
            and fitted_v2_models["omni"].structures
        ):
            _omni_fallback = fitted_v2_models["omni"].structures[0].ranges[0]
        if _omni_fallback is not None and _omni_fallback > 0:
            minor_range = min(major_range, float(_omni_fallback))
        else:
            minor_range = major_range * 0.6
        # Do NOT swap minor↔major when minor is rejected — the azimuth
        # must stay anchored to the reliable major fit.

        # Also update the legacy-format ``fitted_models['minor']`` dict
        # so downstream code (variogram_panel._build_combined_model,
        # audit/export) doesn't see the original bad sill/range. We
        # replace the rejected entry with a clone of the omni fit (or
        # major if omni is also missing) scaled to the corrected range.
        try:
            _fallback_src = fitted_models.get("omni") or fitted_models.get("major") or {}
            if isinstance(_fallback_src, dict) and _fallback_src:
                _patched = {}
                for _mtype, _entry in _fallback_src.items():
                    if not isinstance(_entry, dict):
                        continue
                    _clone = dict(_entry)
                    _clone["range"] = float(minor_range)
                    _clone["reliability"] = "rejected_fallback"
                    _clone["reject_reason"] = _minor_reject_reason
                    _patched[_mtype] = _clone
                if _patched:
                    fitted_models["minor"] = _patched
                    logger.info(
                        "v2 variogram: patched fitted_models['minor'] from "
                        "omni/major fallback (range=%.1fm)", minor_range,
                    )
        except Exception as _exc:
            logger.debug(
                "Could not patch fitted_models['minor'] after reject: %s", _exc,
            )
    elif minor_range > major_range * 1.01:
        # Only swap when the minor fit was accepted AND genuinely longer
        # (i.e. the user mis-labelled the azimuth).
        major_range, minor_range = minor_range, major_range
        major_vec, minor_vec = minor_vec, major_vec  # swap PCA directions too

    # Final geometric guard: minor must never exceed major after all the
    # above — clamp defensively so downstream code can trust the ordering.
    if minor_range > major_range:
        minor_range = major_range

    combined_3d = {}
    if best_model:
        combined_3d = _model_to_combined_3d(best_model, major_range, minor_range, vert_range)

    # ── Compute azimuth from the actual vectors ───────────────────────
    # M1 fix: derive minor_azimuth from the real minor_vec instead of
    # assuming a 90-degree CCW rotation from major_azimuth. When the
    # major/minor vectors are swapped earlier (because minor_range
    # ended up numerically longer than major_range after fitting), the
    # stored vectors are the post-swap values — so `major + 90` can be
    # off by ±90 relative to minor_vec. Anyone reconstructing a
    # rotation matrix from the stored azimuths downstream then rotates
    # the search ellipsoid the wrong way.
    major_azimuth = float(np.rad2deg(np.arctan2(major_vec[0], major_vec[1]))) % 360.0
    minor_azimuth = float(np.rad2deg(np.arctan2(minor_vec[0], minor_vec[1]))) % 360.0
    major_dip = 0.0
    if default_dip is not None:
        major_dip = default_dip

    # ── Metadata ──────────────────────────────────────────────────────
    _progress(90, "Variogram v2: Building metadata...")

    data_hash = _compute_data_hash(coords, vals, variable_name)
    # Use the weighted sample variance computed earlier so the metadata
    # matches the sill cap used by the fitter.
    sample_var = (
        float(sample_variance_early)
        if sample_variance_early is not None else float(np.var(vals))
    )

    weak_directions = []
    for dir_name, exp in exp_variograms.items():
        valid_pairs = exp.n_pairs[exp.n_pairs > 0]
        avg_pairs = float(np.mean(valid_pairs)) if len(valid_pairs) > 0 else 0.0
        lags_with = int(len(valid_pairs))
        is_weak = avg_pairs < 30 or lags_with < 3
        # Only include genuinely weak directions — don't flag 67k pairs as weak
        if is_weak:
            weak_directions.append({
                "direction": dir_name,
                "avg_pairs_per_lag": avg_pairs,
                "total_pairs": int(np.sum(exp.n_pairs)),
                "lags_with_pairs": lags_with,
                "is_critical": lags_with < 3,
                "warning": (
                    f"WEAK: {dir_name} has {avg_pairs:.0f} avg pairs/lag "
                    f"(need \u226530 for reliable fit)"
                ),
            })

    metadata = {
        "source_data_hash": data_hash,
        "fit_timestamp": datetime.datetime.now().isoformat(),
        "source_dataset_type": "composites",
        "source_data_n_samples": int(coords.shape[0]),
        "source_data_n_rows": int(len(data)),
        "variable": variable_name,
        "sample_variance": sample_var,
        "subsampled": False,
        "subsample_size": int(coords.shape[0]),
        "subsample_seed": random_state or 42,
        "subsample_fraction": 1.0,
        "pair_cap": search.max_pairs_per_lag,
        "random_state": random_state,
        "is_deterministic": True,
        "orientation_source": "horizontal_pca_support" if default_azimuth is None else "explicit_input",
        "orientation_note": "",
        "orientation_support_points": int(coords.shape[0]),
        "dip_source": "explicit" if default_dip is not None else "assumed_horizontal",
        "weak_directions": weak_directions,
        "has_weak_directions": any(w.get("warning") for w in weak_directions),
        "engine_version": "v2",
        "transform_mode": transform_mode,
        "drift_mode": drift_mode,
        "robust_estimator": use_robust_estimator,
    }

    # ── Build legacy return dict ──────────────────────────────────────
    _progress(95, "Variogram v2: Packaging results...")

    result: Dict[str, Any] = {
        "omni_variogram": _exp_to_dataframe(exp_variograms["omni"]) if "omni" in exp_variograms else pd.DataFrame(),
        "major_variogram": _exp_to_dataframe(exp_variograms["major"]) if "major" in exp_variograms else pd.DataFrame(),
        "minor_variogram": _exp_to_dataframe(exp_variograms["minor"]) if "minor" in exp_variograms else pd.DataFrame(),
        "vertical_variogram": _exp_to_dataframe(exp_variograms["vertical"]) if "vertical" in exp_variograms else pd.DataFrame(),
        "downhole_variogram": downhole_df,
        "fitted_models": fitted_models,
        "nested_models": {},  # Legacy VariogramModel objects not generated by v2
        "combined_3d_model": combined_3d,
        "variogram_object": None,  # Not applicable for v2
        "major_azimuth": major_azimuth,
        "minor_azimuth": minor_azimuth,
        "major_dip": major_dip,
        "lag_config": {
            "auto_lags": auto_lags,
            "n_structures": n_structures,
            "global_nugget": global_nugget,
            "lag_distance": lag_distance,
            "lag_tolerance": lag_tolerance or lag_distance * 0.5,
            "max_range": lag_distance * nlag,
            "horizontal": (nlag, lag_distance, lag_distance * nlag),
            "vertical": (nlag, lag_distance, lag_distance * nlag),
            "downhole": None,
        },
        "metadata": metadata,
        # v2-specific extras (ignored by legacy consumers, useful for v2-aware code)
        "_v2_engine": engine,
        "_v2_experimental": exp_variograms,
        "_v2_fitted_models": fitted_v2_models,
        # Source data for downstream visualisations (variogram map, h-scatter)
        "_source_data": data,
        "_xcol": xcol,
        "_ycol": ycol,
        "_zcol": zcol,
        "_vcol": vcol,
    }

    _progress(100, "Variogram v2: Complete")
    return result
