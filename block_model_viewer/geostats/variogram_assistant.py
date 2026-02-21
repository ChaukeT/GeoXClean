"""
Variogram Modelling Assistant (STEP 23).

Semi-automatic variogram fitting with model selection and cross-validation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Tuple
import numpy as np
import pandas as pd
import logging

from ..utils.variogram_functions import (
    calculate_experimental_variogram,
    fit_variogram,
    get_variogram_function
)
from ..models.variogram_functions import (
    fit_nested_variogram,
    fit_variogram_model,
)
from ..models.kriging3d import get_variogram_function as get_kriging_variogram_function

logger = logging.getLogger(__name__)


@dataclass
class VariogramCandidateModel:
    """A candidate variogram model with fitted parameters and scores."""
    model_type: str  # "spherical", "exponential", "gaussian", or nested combo
    ranges: List[float]  # Range(s) for each structure
    sills: List[float]  # Sill(s) for each structure
    nugget: float
    anisotropy: Dict[str, Any] = field(default_factory=dict)
    score_sse: float = 0.0  # Sum of squared errors vs experimental
    score_cv_rmse: float = 0.0  # Cross-validation RMSE
    score_total: float = 0.0  # Composite score
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VariogramAssistantResult:
    """Result from variogram assistant analysis."""
    candidates: List[VariogramCandidateModel]
    best_model: VariogramCandidateModel
    experimental_variogram: Dict[str, Any]
    directional_variograms: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    downhole_variogram: Optional[Dict[str, Any]] = None
    directional_fits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    variogram_map: Optional[Dict[str, Any]] = None  # Variogram map data
    combined_3d_model: Optional[Dict[str, Any]] = None  # Combined 3D anisotropic model
    metadata: Dict[str, Any] = field(default_factory=dict)


def build_candidate_models(
    experimental: Dict[str, Any],
    model_families: List[str] = None,
    max_structures: int = 2
) -> List[VariogramCandidateModel]:
    """
    Build initial candidate variogram models from experimental data.
    
    Args:
        experimental: Experimental variogram dict with 'lag_distances' and 'semivariances'
        model_families: List of model types to try (default: ['spherical', 'exponential', 'gaussian'])
        max_structures: Maximum number of nested structures (default: 2)
    
    Returns:
        List of VariogramCandidateModel instances with initial parameter guesses
    """
    if model_families is None:
        model_families = ['spherical', 'exponential', 'gaussian']
    
    lag_distances = experimental.get('lag_distances', np.array([]))
    semivariances = experimental.get('semivariances', np.array([]))
    
    if len(lag_distances) == 0 or len(semivariances) == 0:
        logger.warning("No experimental variogram data provided")
        return []
    
    candidates = []
    
    # Estimate initial parameters from experimental data
    max_gamma = np.max(semivariances) if len(semivariances) > 0 else 1.0
    max_dist = np.max(lag_distances) if len(lag_distances) > 0 else 100.0
    nugget_guess = semivariances[0] if len(semivariances) > 0 else 0.0
    
    # Try single-structure models
    for model_type in model_families:
        try:
            # Fit model to experimental data
            nugget, sill, range_ = fit_variogram(
                lag_distances,
                semivariances,
                model_type=model_type
            )
            
            candidate = VariogramCandidateModel(
                model_type=model_type,
                ranges=[range_],
                sills=[sill - nugget],  # Partial sill
                nugget=nugget,
                metadata={'n_structures': 1}
            )
            candidates.append(candidate)
        except Exception as e:
            logger.warning(f"Failed to build candidate {model_type}: {e}")
    
    # Try nested structures (if max_structures > 1)
    if max_structures > 1:
        # Use actual nested fitting from experimental data
        for model_type1 in model_families:
            for model_type2 in model_families:
                if model_type1 == model_type2 and model_type1 != "spherical":
                    continue  # Skip duplicate non-spherical pairs
                try:
                    # Fit actual nested model from data
                    nested_fit = fit_nested_variogram(
                        lag_distances,
                        semivariances,
                        model_type1=model_type1,
                        model_type2=model_type2,
                        n_structures=2
                    )
                    
                    structures = nested_fit.get("structures", [])
                    if len(structures) >= 2:
                        candidate = VariogramCandidateModel(
                            model_type=f"{model_type1}+{model_type2}",
                            ranges=[s["range"] for s in structures],
                            sills=[s["contribution"] for s in structures],
                            nugget=nested_fit.get("nugget", 0.0),
                            metadata={
                                'n_structures': len(structures),
                                'structure_types': [s.get("type", model_type1) for s in structures]
                            }
                        )
                        candidates.append(candidate)
                except Exception as e:
                    logger.warning(f"Failed to build nested candidate {model_type1}+{model_type2}: {e}")
    
    return candidates


def infer_shape_hint(experimental: Dict[str, Any]) -> Optional[str]:
    """
    Lightweight shape heuristic based on the first few lags to suggest a family.

    - Steep first rise, no knee, asymptotic sill -> exponential
    - Moderate rise with a knee (slope drop) -> spherical
    - Gentle, smooth rise throughout -> gaussian
    """
    lags = experimental.get("lag_distances", np.array([]))
    gammas = experimental.get("semivariances", np.array([]))
    if lags is None or gammas is None or len(lags) < 4:
        return None

    # Use first 4 points for early-shape cues
    l = lags[:4]
    g = gammas[:4]
    if np.any(~np.isfinite(l)) or np.any(~np.isfinite(g)):
        return None

    # Normalize distances and gammas to compare slopes
    dg = np.diff(g)
    dl = np.diff(l)
    if np.any(dl == 0):
        return None
    slopes = dg / dl
    if len(slopes) < 2:
        return None

    max_gamma = float(np.nanmax(gammas)) if len(gammas) > 0 else 1.0
    max_dist = float(np.nanmax(lags)) if len(lags) > 0 else 1.0
    typical_slope = (max_gamma - gammas[0]) / max_dist if max_dist > 0 else 0.0

    s1 = slopes[0]
    s2 = slopes[1] if len(slopes) > 1 else slopes[0]
    s_max = np.nanmax(slopes) if len(slopes) else s1
    knee_score = (s1 - s2) / (abs(s1) + 1e-9)

    # Heuristics
    if abs(s1) > 2.5 * abs(typical_slope) and knee_score < 0.2:
        return "exponential"
    if knee_score > 0.35 and abs(s1) > 0.5 * abs(typical_slope):
        return "spherical"
    if abs(s1) < 0.5 * abs(typical_slope) and np.nanmax(np.abs(slopes)) < 1.5 * abs(typical_slope):
        return "gaussian"
    return None


def _directional_experimental(coords: np.ndarray, values: np.ndarray, azimuth_deg: float, dip_deg: float, cone_tol: float, n_lags: int, lag_tolerance: float, max_pairs: int = 3000) -> Dict[str, np.ndarray]:
    """
    Directional experimental variogram using cone tolerance via Variogram3D.calculate_directional
    to mirror panel behavior, with a stricter pair cap for stability.
    """
    if coords.shape[0] < 4:
        return {'lag_distances': np.array([]), 'semivariances': np.array([]), 'pair_counts': np.array([])}
    try:
        from ..models.variogram3d import Variogram3D
        # DETERMINISM: Explicit seed for reproducibility
        v3d = Variogram3D(n_lags=n_lags, lag_distance=lag_tolerance * 2, lag_tolerance=lag_tolerance, random_state=42)
        df = v3d.calculate_directional(
            coords, values, sample_weights=None,
            azimuth_deg=azimuth_deg, dip_deg=dip_deg, cone_tolerance=cone_tol,
            n_lags=n_lags, max_range=n_lags * lag_tolerance * 2,
            max_samples=None, pair_cap=max_pairs
        )
        return {
            "lag_distances": df["distance"].to_numpy() if not df.empty else np.array([]),
            "semivariances": df["gamma"].to_numpy() if not df.empty else np.array([]),
            "pair_counts": df["npairs"].to_numpy() if not df.empty else np.array([]),
        }
    except Exception:
        return {'lag_distances': np.array([]), 'semivariances': np.array([]), 'pair_counts': np.array([])}


def _dict_from_dataframe(df: Any) -> Dict[str, Any]:
    """Helper to keep variogram structures serializable."""
    if df is None:
        return {}
    return {
        "lag_distances": df.get("lag_distances", np.array([])),
        "semivariances": df.get("semivariances", np.array([])),
        "pair_counts": df.get("pair_counts", np.array([])),
    }


def _fit_directional_models(
    directional_variograms: Dict[str, Dict[str, Any]], 
    model_type: str,
    n_structures: int = 1,
    model_type2: str = "exponential"
) -> Dict[str, Dict[str, Any]]:
    """
    Fit the chosen model type to each directional experimental variogram.
    
    Args:
        directional_variograms: Dict of directional variogram data
        model_type: Primary model type
        n_structures: Number of structures (1 for single, 2+ for nested)
        model_type2: Secondary model type for nested structures
    
    Returns:
        {direction: {model_type: {model_type, nugget, sill, total_sill, range, [structures]}}}
    """
    fitted: Dict[str, Dict[str, Any]] = {}
    for dir_key, exp in directional_variograms.items():
        lags = exp.get("lag_distances", np.array([]))
        semivars = exp.get("semivariances", np.array([]))
        if lags is None or semivars is None or len(lags) < 3:
            continue
        try:
            if n_structures >= 2:
                # Fit nested model
                nested_fit = fit_nested_variogram(
                    lags, semivars,
                    model_type1=model_type,
                    model_type2=model_type2,
                    n_structures=n_structures
                )
                structures = nested_fit.get("structures", [])
                fitted[dir_key] = {
                    model_type: {
                        "model_type": model_type,
                        "nugget": float(nested_fit.get("nugget", 0)),
                        "sill": float(sum(s["contribution"] for s in structures)),
                        "total_sill": float(nested_fit.get("total_sill", 0)),
                        "range": float(structures[-1]["range"]) if structures else 0,
                        "structures": structures,
                    },
                    "nested": {
                        "nugget": float(nested_fit.get("nugget", 0)),
                        "total_sill": float(nested_fit.get("total_sill", 0)),
                        "structures": structures,
                        "range": float(structures[-1]["range"]) if structures else 0,
                    }
                }
            else:
                # Single structure fit
                nugget, sill, rng = fit_variogram(lags, semivars, model_type=model_type)
                total_sill = sill
                fitted[dir_key] = {
                    model_type: {
                        "model_type": model_type,
                        "nugget": float(nugget),
                        "sill": float(sill - nugget),
                        "total_sill": float(total_sill),
                        "range": float(rng),
                    }
                }
        except Exception as e:
            logger.warning(f"Failed to fit {dir_key}: {e}")
            continue
    return fitted


def _compute_variogram_map(
    coords: np.ndarray, 
    values: np.ndarray, 
    n_lags: int = 12,
    lag_distance: float = 25.0,
    max_samples: int = 1000
) -> Optional[Dict[str, Any]]:
    """
    Compute variogram map (semivariance vs direction and distance).
    
    Returns dict with:
        - gamma_matrix: 2D array [n_azimuths x n_lags]
        - azimuths: Array of azimuth angles (degrees)
        - distances: Array of lag distances
    """
    if len(coords) < 50:
        return None
    
    # Subsample if needed
    if len(coords) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(coords), size=max_samples, replace=False)
        coords = coords[idx]
        values = values[idx]
    
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(coords[:, :2])  # X, Y only
    except ImportError:
        return None
    
    azimuths = np.arange(0, 180, 15)
    distances = np.arange(1, n_lags + 1) * lag_distance
    max_dist = distances[-1]
    
    gamma_matrix = np.zeros((len(azimuths), len(distances)))
    pair_counts = np.zeros((len(azimuths), len(distances)))
    
    # Get all pairs and sort for deterministic ordering
    pairs = tree.query_pairs(r=max_dist, output_type='ndarray')
    if len(pairs) == 0:
        return None

    # Sort pairs lexicographically for determinism
    sort_idx = np.lexsort((pairs[:, 1], pairs[:, 0]))
    pairs = pairs[sort_idx]

    i_idx, j_idx = pairs[:, 0], pairs[:, 1]
    diffs = coords[j_idx] - coords[i_idx]
    dists = np.linalg.norm(diffs, axis=1)
    val_diffs = values[j_idx] - values[i_idx]
    gammas = 0.5 * val_diffs ** 2
    
    # Compute angles
    horiz_angles = np.rad2deg(np.arctan2(diffs[:, 0], diffs[:, 1])) % 180
    
    # Bin by direction and distance
    for i, azimuth in enumerate(azimuths):
        angle_diff = np.abs(horiz_angles - azimuth)
        angle_diff = np.minimum(angle_diff, 180 - angle_diff)
        in_dir = angle_diff <= 15
        
        for j, max_d in enumerate(distances):
            min_d = distances[j-1] if j > 0 else 0
            in_band = in_dir & (dists > min_d) & (dists <= max_d)
            
            if np.any(in_band):
                gamma_matrix[i, j] = np.mean(gammas[in_band])
                pair_counts[i, j] = np.sum(in_band)
    
    # Set low-pair cells to NaN
    gamma_matrix[pair_counts < 5] = np.nan
    
    return {
        "gamma_matrix": gamma_matrix,
        "azimuths": azimuths,
        "distances": distances,
        "pair_counts": pair_counts,
    }


def _build_combined_3d_model(
    directional_fits: Dict[str, Dict[str, Any]],
    model_type: str,
    n_structures: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Build combined 3D anisotropic model from directional fits.
    
    Same logic as VariogramAnalysisPanel._build_combined_3d_model.
    """
    # Get fits for each direction
    def get_fit(direction):
        dir_fits = directional_fits.get(direction, {})
        if n_structures >= 2 and "nested" in dir_fits:
            return dir_fits["nested"]
        return dir_fits.get(model_type, {})
    
    downhole = get_fit("downhole")
    major = get_fit("major")
    minor = get_fit("minor")
    vertical = get_fit("vertical")
    omni = get_fit("omni")
    
    # Priority: Use downhole for nugget
    vert_source = downhole if downhole else vertical
    
    if not vert_source and not major and not minor:
        if omni:
            return {
                "model_type": model_type,
                "nugget": omni.get("nugget", 0.0),
                "sill": omni.get("total_sill", 0),
                "major_range": omni.get("range", 100.0),
                "minor_range": omni.get("range", 100.0),
                "vertical_range": omni.get("range", 100.0),
                "is_isotropic": True,
            }
        return None
    
    nugget = vert_source.get("nugget", 0.0) if vert_source else 0.0
    vertical_range = vert_source.get("range", 50.0) if vert_source else 50.0
    major_range = major.get("range", 100.0) if major else 100.0
    minor_range = minor.get("range", 75.0) if minor else major_range * 0.5
    
    # Sill from directional fits
    sills = []
    for d in [major, minor, vert_source]:
        if d:
            s = d.get("total_sill", d.get("sill", 0) + d.get("nugget", 0))
            if s > 0:
                sills.append(s)
    total_sill = max(sills) if sills else nugget + 1.0
    
    # Anisotropy ratios
    aniso_minor = minor_range / major_range if major_range > 0 else 1.0
    aniso_vertical = vertical_range / major_range if major_range > 0 else 1.0
    
    # GEOSTATISTICAL FIX: Enforce proper range ordering (Primary >= Secondary >= Tertiary)
    try:
        from .anisotropy_utils import enforce_range_ordering
        
        ordered_ranges, naming_info = enforce_range_ordering(
            major_range, minor_range, vertical_range, use_descriptive_names=False
        )
        
        # Use corrected ranges for final model
        major_range = ordered_ranges["major_range"] 
        minor_range = ordered_ranges["minor_range"]
        vertical_range = ordered_ranges["vertical_range"]
        
        # Recalculate ratios with corrected ranges
        aniso_minor = minor_range / major_range if major_range > 0 else 1.0
        aniso_vertical = vertical_range / major_range if major_range > 0 else 1.0
        
        aniso_warnings = naming_info.get("warnings", [])
        
    except ImportError:
        aniso_warnings = []
    
    # Build structures
    structures = []
    if n_structures >= 2:
        omni_nested = directional_fits.get("omni", {}).get("nested", {})
        fitted_structures = omni_nested.get("structures", [])
        for fs in fitted_structures[:n_structures]:
            structures.append({
                "type": fs.get("type", model_type),
                "contribution": fs.get("contribution", (total_sill - nugget) / n_structures),
                "major_range": fs.get("range", major_range),
                "minor_range": fs.get("range", major_range) * aniso_minor,
                "vertical_range": fs.get("range", major_range) * aniso_vertical,
            })
    else:
        structures.append({
            "type": model_type,
            "contribution": total_sill - nugget,
            "major_range": major_range,
            "minor_range": minor_range,
            "vertical_range": vertical_range,
        })
    
    return {
        "model_type": model_type,
        "nugget": nugget,
        "sill": total_sill,
        "major_range": major_range,
        "minor_range": minor_range,
        "vertical_range": vertical_range,
        "anisotropy_ratio_minor": aniso_minor,
        "anisotropy_ratio_vertical": aniso_vertical,
        "anisotropy_warnings": aniso_warnings,  # Range ordering warnings
        "n_structures": n_structures,
        "structures": structures,
        "is_isotropic": False,
    }


def _downhole_variogram(coords: np.ndarray, values: np.ndarray, hole_ids: np.ndarray, n_lags: int, lag_tolerance: float) -> Dict[str, Any]:
    """Compute a simple downhole variogram by pairing samples within each hole."""
    if hole_ids is None or len(hole_ids) != len(values):
        return {}
    all_dists = []
    all_gammas = []
    df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    df["val"] = values
    df["hid"] = hole_ids
    grouped = df.groupby("hid")
    for _, grp in grouped:
        if len(grp) < 2:
            continue
        gc = grp[["X", "Y", "Z"]].to_numpy()
        gv = grp["val"].to_numpy()
        n = len(gc)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(gc[i] - gc[j]))
                g = 0.5 * float((gv[i] - gv[j]) ** 2)
                all_dists.append(d)
                all_gammas.append(g)
    if not all_dists:
        return {}
    all_dists = np.array(all_dists)
    all_gammas = np.array(all_gammas)
    max_dist = np.nanmax(all_dists)
    if max_dist <= 0:
        return {}
    # Bin into n_lags using tolerance as +/- window around centers
    lag_centers = np.linspace(0, max_dist, n_lags)
    lag_distances = []
    semivariances = []
    pair_counts = []
    for c in lag_centers:
        lower = c - lag_tolerance
        upper = c + lag_tolerance
        mask = (all_dists >= lower) & (all_dists <= upper)
        if np.any(mask):
            lag_distances.append(c)
            semivariances.append(np.nanmean(all_gammas[mask]))
            pair_counts.append(int(np.sum(mask)))
    return {
        "lag_distances": np.array(lag_distances),
        "semivariances": np.array(semivariances),
        "pair_counts": np.array(pair_counts),
    }


def evaluate_variogram_model(
    experimental: Dict[str, Any],
    model: VariogramCandidateModel
) -> Dict[str, float]:
    """
    Evaluate a variogram model against experimental data.
    
    Args:
        experimental: Experimental variogram dict
        model: VariogramCandidateModel to evaluate
    
    Returns:
        Dict with metrics: 'sse', 'rmse', 'mae', 'r2'
    """
    lag_distances = experimental.get('lag_distances', np.array([]))
    semivariances = experimental.get('semivariances', np.array([]))
    
    if len(lag_distances) == 0:
        return {'sse': np.inf, 'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
    
    # Compute model variogram values
    model_gamma = compute_model_variogram(lag_distances, model)
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(model_gamma) & np.isfinite(semivariances)
    if not np.any(valid_mask):
        return {'sse': np.inf, 'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf}
    
    model_vals = model_gamma[valid_mask]
    exp_vals = semivariances[valid_mask]
    
    # Compute metrics
    residuals = exp_vals - model_vals
    sse = np.sum(residuals ** 2)
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    
    # R-squared
    ss_tot = np.sum((exp_vals - np.mean(exp_vals)) ** 2)
    r2 = 1 - (sse / ss_tot) if ss_tot > 0 else -np.inf
    
    return {
        'sse': float(sse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def compute_model_variogram(
    distances: np.ndarray,
    model: VariogramCandidateModel
) -> np.ndarray:
    """
    Compute variogram values for given distances using a candidate model.
    
    Args:
        distances: Array of lag distances
        model: VariogramCandidateModel
    
    Returns:
        Array of variogram values
    """
    gamma = np.zeros_like(distances)
    
    # Start with nugget
    gamma[:] = model.nugget
    
    # Add each structure
    for i, (range_, sill) in enumerate(zip(model.ranges, model.sills)):
        # Determine model type for this structure
        if '+' in model.model_type:
            # Nested model: use first type for all structures
            model_type = model.model_type.split('+')[0]
        else:
            model_type = model.model_type
        
        # Get variogram function
        gamma_func = get_kriging_variogram_function(model_type)
        
        # Compute structure contribution
        structure_gamma = gamma_func(distances, range_, sill, 0.0)
        gamma += structure_gamma
    
    return gamma


def cross_validate_variogram(
    coords: np.ndarray,
    values: np.ndarray,
    model: VariogramCandidateModel,
    method: Literal["OK", "UK"] = "OK",
    n_folds: int = 5,
    random_state: Optional[int] = 42
) -> Dict[str, float]:
    """
    Perform cross-validation of a variogram model using kriging.

    DETERMINISM: This function is fully deterministic when random_state is set.
    The K-fold split uses a seeded RNG for reproducible results.

    Args:
        coords: (N, 3) data coordinates
        values: (N,) data values
        model: VariogramCandidateModel to validate
        method: Kriging method ("OK" or "UK")
        n_folds: Number of folds for cross-validation (or -1 for LOOCV)
        random_state: Random seed for reproducibility (default 42)

    Returns:
        Dict with 'rmse', 'mae', 'bias', 'correlation'
    """
    from ..models.kriging3d import ordinary_kriging_3d

    n_data = len(values)
    if n_data < 2:
        return {'rmse': np.inf, 'mae': np.inf, 'bias': 0.0, 'correlation': 0.0}

    # Create seeded RNG for deterministic fold assignment
    rng = np.random.default_rng(random_state)

    # Use LOOCV if n_folds == -1 or n_data < n_folds
    if n_folds == -1 or n_data < n_folds:
        n_folds = n_data
        use_loocv = True
    else:
        use_loocv = False
    
    # Prepare variogram parameters for kriging
    model_type = model.model_type.split('+')[0]  # Use first structure type
    total_sill = model.nugget + sum(model.sills)  # Total sill (nugget + partial sills)
    variogram_params = {
        'model_type': model_type,
        'nugget': model.nugget,
        'sill': total_sill,  # Total sill for ordinary_kriging_3d
        'range': model.ranges[0] if len(model.ranges) > 0 else 100.0
    }
    
    predictions = []
    actuals = []
    
    if use_loocv:
        # Leave-one-out cross-validation
        indices = np.arange(n_data)
    else:
        # K-fold cross-validation - use seeded RNG for determinism
        indices = rng.permutation(n_data)
        fold_size = n_data // n_folds
    
    try:
        for fold in range(n_folds):
            if use_loocv:
                test_idx = [fold]
                train_mask = np.ones(n_data, dtype=bool)
                train_mask[fold] = False
            else:
                start_idx = fold * fold_size
                end_idx = start_idx + fold_size if fold < n_folds - 1 else n_data
                test_idx = indices[start_idx:end_idx]
                train_mask = np.ones(n_data, dtype=bool)
                train_mask[test_idx] = False
            
            train_coords = coords[train_mask]
            train_values = values[train_mask]
            test_coords = coords[test_idx]
            test_values = values[test_idx]
            
            if len(train_coords) < 2:
                continue
            
            # Run kriging
            estimates, _, _ = ordinary_kriging_3d(  # Ignore variances and QA metrics for cross-validation
                train_coords,
                train_values,
                test_coords,
                variogram_params,
                n_neighbors=min(16, len(train_coords)),
                max_distance=200.0,
                model_type=model_type
            )
            
            predictions.extend(estimates.tolist())
            actuals.extend(test_values.tolist())
    
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        return {'rmse': np.inf, 'mae': np.inf, 'bias': 0.0, 'correlation': 0.0}
    
    if len(predictions) == 0:
        return {'rmse': np.inf, 'mae': np.inf, 'bias': 0.0, 'correlation': 0.0}
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Compute metrics
    residuals = actuals - predictions
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))
    bias = np.mean(residuals)
    
    # Correlation
    if np.std(predictions) > 0 and np.std(actuals) > 0:
        correlation = np.corrcoef(predictions, actuals)[0, 1]
    else:
        correlation = 0.0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'bias': float(bias),
        'correlation': float(correlation)
    }


def run_variogram_assistant(
    coords: np.ndarray,
    values: np.ndarray,
    params: Dict[str, Any]
) -> VariogramAssistantResult:
    """
    Run variogram modelling assistant to find best variogram model.
    
    Args:
        coords: (N, 3) data coordinates
        values: (N,) data values
        params: Configuration dict with:
            - n_lags: Number of lag bins (default: 15)
            - lag_tolerance: Lag tolerance factor (default: 0.5)
            - model_families: List of model types to try (default: ['spherical', 'exponential', 'gaussian'])
            - max_structures: Maximum nested structures (default: 2)
            - perform_cv: Whether to perform cross-validation (default: True)
            - cv_method: Cross-validation method ("OK" or "UK", default: "OK")
            - cv_folds: Number of CV folds (-1 for LOOCV, default: 5)
            - domain_labels: Optional (N,) array of domain labels for domain-based fitting
    
    Returns:
        VariogramAssistantResult with ranked candidates and best model
    """
    # Extract parameters
    n_lags = params.get('n_lags', 15)
    lag_distance = params.get('lag_distance', None)  # Explicit lag distance in meters
    lag_tolerance = params.get('lag_tolerance', 0.5)
    normalize = params.get('normalize', False)
    model_families = params.get('model_families', ['spherical', 'exponential', 'gaussian'])
    max_structures = params.get('max_structures', 2)
    perform_cv = params.get('perform_cv', False)
    cv_method = params.get('cv_method', 'OK')
    cv_folds = params.get('cv_folds', 5)
    domain_labels = params.get('domain_labels', None)
    hole_id_col = params.get('hole_id_col')
    hole_ids = params.get('hole_ids')  # precomputed optional

    # =========================================================================
    # AUDIT FIX (V-NEW-003): Compute hash of FULL dataset BEFORE subsampling
    # This ensures we can detect if the full dataset changes, even when
    # variogram is computed on a subsample.
    # =========================================================================
    import hashlib
    full_data_n = len(values)
    full_data_coords_hash = hashlib.sha256(
        np.ascontiguousarray(np.round(coords, 6)).tobytes()
    ).hexdigest()[:16]
    full_data_values_hash = hashlib.sha256(
        np.ascontiguousarray(np.round(values, 8)).tobytes()
    ).hexdigest()[:16]
    full_data_hash = f"{full_data_coords_hash}_{full_data_values_hash}"
    
    # Store lineage info before any subsampling
    lineage_info = {
        'full_data_hash': full_data_hash,
        'full_data_n': full_data_n,
        'subsampled': False,
        'subsample_n': None,
        'subsample_seed': None,
    }

    # Throttle dataset size for speed
    max_samples = params.get("max_samples", 5000)
    subsampled = False
    subsample_seed = None
    if len(values) > max_samples:
        # AUDIT FIX: Use deterministic seed for reproducibility
        subsample_seed = params.get("subsample_seed", 42)
        rng = np.random.default_rng(subsample_seed)
        idx = rng.choice(len(values), size=max_samples, replace=False)
        coords = coords[idx]
        values = values[idx]
        if domain_labels is not None:
            domain_labels = domain_labels[idx]
        subsampled = True
        
        # Update lineage info
        lineage_info['subsampled'] = True
        lineage_info['subsample_n'] = max_samples
        lineage_info['subsample_seed'] = subsample_seed
        logger.info(
            f"AUDIT: Variogram computed on subsample ({max_samples}/{full_data_n} samples, "
            f"seed={subsample_seed}). Full data hash: {full_data_hash}"
        )

    # Compute experimental variogram
    if domain_labels is not None:
        # Domain-based fitting: fit per domain
        unique_domains = np.unique(domain_labels)
        experimental_variograms = {}
        
        for domain in unique_domains:
            mask = domain_labels == domain
            domain_coords = coords[mask]
            domain_values = values[mask]
            
            if len(domain_values) < 2:
                continue
            
            lag_dist, semivar, pair_counts = calculate_experimental_variogram(
                domain_coords,
                domain_values,
                n_lags=n_lags,
                lag_tolerance=lag_tolerance,
                lag_distance=lag_distance,
                normalize=normalize
            )
            
            experimental_variograms[domain] = {
                'lag_distances': lag_dist,
                'semivariances': semivar,
                'pair_counts': pair_counts
            }
        
        # Use first domain's variogram for candidate building (or combine)
        if len(experimental_variograms) > 0:
            first_domain = list(experimental_variograms.keys())[0]
            experimental = experimental_variograms[first_domain]
        else:
            experimental = {'lag_distances': np.array([]), 'semivariances': np.array([])}
    else:
        # Single variogram for all data
        lag_dist, semivar, pair_counts = calculate_experimental_variogram(
            coords,
            values,
            n_lags=n_lags,
            lag_tolerance=lag_tolerance,
            lag_distance=lag_distance,
            normalize=normalize
        )
        
        experimental = {
            'lag_distances': lag_dist,
            'semivariances': semivar,
            'pair_counts': pair_counts
        }
    
    # Build candidate models
    candidates = build_candidate_models(
        experimental,
        model_families=model_families,
        max_structures=max_structures
    )

    # Shape cues (soft bias)
    shape_hint = infer_shape_hint(experimental)
    directional_hints: Dict[str, Optional[str]] = {
        'X': None,
        'Y': None,
        'Z': None
    }
    # Lightweight directional hints for professional cueing
    for axis, key in enumerate(['X', 'Y', 'Z']):
        # Use simple projections only for shape hints (not full plotting)
        try:
            proj = coords[:, axis]
            order = np.argsort(proj)
            lags = np.abs(np.diff(proj[order]))
            vals = values[order]
            gammas = 0.5 * np.square(np.diff(vals))
            dir_exp = {
                "lag_distances": lags[: min(len(lags), n_lags)],
                "semivariances": gammas[: min(len(gammas), n_lags)],
                "pair_counts": np.ones_like(lags[: min(len(lags), n_lags)]),
            }
        except Exception:
            dir_exp = {'lag_distances': np.array([]), 'semivariances': np.array([]), 'pair_counts': np.array([])}
        directional_hints[key] = infer_shape_hint(dir_exp)

    # Directional variograms for downstream plotting (heuristic)
    cone_tol = 22.5
    directional_variograms = {
        "major": _directional_experimental(coords, values, azimuth_deg=0.0, dip_deg=0.0, cone_tol=cone_tol, n_lags=n_lags, lag_tolerance=lag_tolerance),
        "minor": _directional_experimental(coords, values, azimuth_deg=90.0, dip_deg=0.0, cone_tol=cone_tol, n_lags=n_lags, lag_tolerance=lag_tolerance),
        "vertical": _directional_experimental(coords, values, azimuth_deg=0.0, dip_deg=90.0, cone_tol=cone_tol, n_lags=n_lags, lag_tolerance=lag_tolerance),
    }
    downhole_variogram = None
    if hole_ids is not None and len(hole_ids) == len(values):
        downhole_variogram = _downhole_variogram(coords, values, hole_ids, n_lags, lag_tolerance)

    # Downhole nugget suggestion: use early lags as proxy
    nugget_suggestion = None
    if len(experimental.get('semivariances', [])) > 0:
        first_vals = experimental['semivariances'][:3]
        if len(first_vals) > 0:
            nugget_suggestion = float(np.nanmin(first_vals))
    
    if len(candidates) == 0:
        logger.warning("No candidate models generated")
        # Create a default candidate
        default_candidate = VariogramCandidateModel(
            model_type='spherical',
            ranges=[100.0],
            sills=[1.0],
            nugget=0.0
        )
        return VariogramAssistantResult(
            candidates=[default_candidate],
            best_model=default_candidate,
            experimental_variogram=experimental
        )
    
    # Evaluate each candidate
    for candidate in candidates:
        # Evaluate fit to experimental data
        fit_metrics = evaluate_variogram_model(experimental, candidate)
        candidate.score_sse = fit_metrics['sse']
        candidate.metadata.update(fit_metrics)
        
        # Cross-validation if requested
        if perform_cv:
            cv_metrics = cross_validate_variogram(
                coords,
                values,
                candidate,
                method=cv_method,
                n_folds=cv_folds
            )
            candidate.score_cv_rmse = cv_metrics['rmse']
            candidate.metadata.update(cv_metrics)
        else:
            candidate.score_cv_rmse = 0.0
        
        # Composite score (lower is better)
        # Normalize SSE and CV RMSE, then combine
        max_sse = max(c.score_sse for c in candidates) if candidates else 1.0
        max_cv = max(c.score_cv_rmse for c in candidates) if candidates and perform_cv else 1.0
        
        if max_sse > 0:
            norm_sse = candidate.score_sse / max_sse
        else:
            norm_sse = 0.0
        
        if max_cv > 0:
            norm_cv = candidate.score_cv_rmse / max_cv
        else:
            norm_cv = 0.0
        
        # Weighted combination (favor CV if available)
        if perform_cv:
            candidate.score_total = 0.4 * norm_sse + 0.6 * norm_cv
        else:
            candidate.score_total = norm_sse

        # Apply a gentle bias toward the shape hint
        bias = 1.0
        if shape_hint:
            if candidate.model_type.startswith(shape_hint):
                bias = 0.9  # small reward
            elif shape_hint in candidate.model_type:
                bias = 0.93
            else:
                bias = 1.0
        candidate.score_total *= bias
        candidate.metadata["shape_bias"] = bias
    
    # Sort by total score (ascending)
    candidates.sort(key=lambda c: c.score_total)
    
    # Best model is first in sorted list
    best_model = candidates[0]
    
    # Fit directional models using the best model type
    # Use nested fitting if max_structures > 1
    n_struct_fit = max_structures if max_structures > 1 else 1
    model2_type = model_families[1] if len(model_families) > 1 else "exponential"
    
    directional_fits = _fit_directional_models(
        directional_variograms, 
        best_model.model_type.split('+')[0],  # Use primary model type
        n_structures=n_struct_fit,
        model_type2=model2_type
    )
    if downhole_variogram:
        dh_fit = _fit_directional_models(
            {"downhole": downhole_variogram}, 
            best_model.model_type.split('+')[0],
            n_structures=n_struct_fit,
            model_type2=model2_type
        )
        directional_fits.update(dh_fit)
    
    # Also add omni to directional fits
    if experimental:
        omni_fits = _fit_directional_models(
            {"omni": experimental},
            best_model.model_type.split('+')[0],
            n_structures=n_struct_fit,
            model_type2=model2_type
        )
        directional_fits.update(omni_fits)
    
    # Compute variogram map
    lag_distance = lag_tolerance * 2  # Approximate lag distance
    variogram_map = _compute_variogram_map(
        coords, values,
        n_lags=min(n_lags, 12),
        lag_distance=lag_distance,
        max_samples=1000
    )
    
    # Build combined 3D anisotropic model
    combined_3d_model = _build_combined_3d_model(
        directional_fits,
        best_model.model_type.split('+')[0],
        n_structures=n_struct_fit
    )

    # Build result
    # AUDIT FIX (V-NEW-003): Include lineage info in metadata for data tracking
    result = VariogramAssistantResult(
        candidates=candidates,
        best_model=best_model,
        experimental_variogram=experimental,
        directional_variograms=directional_variograms,
        downhole_variogram=downhole_variogram,
        variogram_map=variogram_map,
        combined_3d_model=combined_3d_model,
        directional_fits=directional_fits,
        metadata={
            'n_candidates': len(candidates),
            'perform_cv': perform_cv,
            'cv_method': cv_method,
            'domain_based': domain_labels is not None,
            'shape_hint': shape_hint,
            'directional_shape_hints': directional_hints,
            'nugget_suggestion': nugget_suggestion,
            # AUDIT FIX (V-NEW-003): Data lineage tracking
            'lineage': lineage_info,
            'full_data_hash': lineage_info['full_data_hash'],
            'full_data_n': lineage_info['full_data_n'],
            'subsampled': lineage_info['subsampled'],
            'subsample_n': lineage_info['subsample_n'],
            'subsample_seed': lineage_info['subsample_seed'],
        }
    )
    
    logger.info(f"Variogram assistant completed: {len(candidates)} candidates, best: {best_model.model_type}")
    
    return result


def run_variogram_assistant_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper for controller integration.
    
    Args:
        params: Job parameters dict with 'coords', 'values', and other config
    
    Returns:
        Result dict suitable for controller payload
        
    LINEAGE: This function expects data to have been validated and composited
    before being passed in. The AppController enforces this via 
    get_estimation_ready_data().
    """
    coords = params.get('coords')
    values = params.get('values')
    hole_ids = params.get('hole_ids')
    hole_id_col = params.get('hole_id_col')
    
    # LINEAGE CHECK: Verify data source if available
    data_df = params.get('data_df')
    if data_df is not None and hasattr(data_df, 'attrs'):
        source_type = data_df.attrs.get('source_type', 'unknown')
        lineage_gate_passed = data_df.attrs.get('lineage_gate_passed', False)
        
        if source_type == 'assays' or not lineage_gate_passed:
            logger.warning(
                "LINEAGE WARNING: Variogram analysis may be running on raw data. "
                f"Source type: {source_type}, Gate passed: {lineage_gate_passed}. "
                "For defensible variograms, use composited/declustered data."
            )

    if coords is None or values is None:
        return {'error': 'Missing coords or values'}

    # Defaults for speed unless explicitly overridden
    params.setdefault("perform_cv", False)
    params.setdefault("max_samples", 5000)

    try:
        result = run_variogram_assistant(coords, values, {**params, "hole_ids": hole_ids, "hole_id_col": hole_id_col})
        
        # Convert to dict for serialization
        candidates_dict = []
        for cand in result.candidates:
            candidates_dict.append({
                'model_type': cand.model_type,
                'ranges': cand.ranges,
                'sills': cand.sills,
                'nugget': cand.nugget,
                'anisotropy': cand.anisotropy,
                'score_sse': cand.score_sse,
                'score_cv_rmse': cand.score_cv_rmse,
                'score_total': cand.score_total,
                'metadata': cand.metadata
            })
        
        best_dict = {
            'model_type': result.best_model.model_type,
            'ranges': result.best_model.ranges,
            'sills': result.best_model.sills,
            'nugget': result.best_model.nugget,
            'anisotropy': result.best_model.anisotropy,
            'score_sse': result.best_model.score_sse,
            'score_cv_rmse': result.best_model.score_cv_rmse,
            'score_total': result.best_model.score_total,
            'metadata': result.best_model.metadata
        }
        
        # Convert variogram map to serializable format
        variogram_map_dict = None
        if result.variogram_map:
            variogram_map_dict = {
                'gamma_matrix': result.variogram_map['gamma_matrix'].tolist() if result.variogram_map.get('gamma_matrix') is not None else None,
                'azimuths': result.variogram_map['azimuths'].tolist() if result.variogram_map.get('azimuths') is not None else None,
                'distances': result.variogram_map['distances'].tolist() if result.variogram_map.get('distances') is not None else None,
            }
        
        return {
            'candidates': candidates_dict,
            'best_model': best_dict,
            'experimental_variogram': result.experimental_variogram,
            'directional_variograms': result.directional_variograms,
            'directional_fits': result.directional_fits,
            'downhole_variogram': result.downhole_variogram,
            'variogram_map': variogram_map_dict,
            'combined_3d_model': result.combined_3d_model,
            'metadata': result.metadata
        }
    
    except Exception as e:
        logger.error(f"Variogram assistant job failed: {e}", exc_info=True)
        return {'error': str(e)}

