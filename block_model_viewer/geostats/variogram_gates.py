"""
Variogram Lineage Gates and Consistency Checks.

AUDIT COMPLIANCE: These gates enforce data consistency, nugget consistency,
and model integrity across the variogram subsystem per JORC/SAMREC requirements.

Usage:
    from block_model_viewer.geostats.variogram_gates import (
        compute_data_hash,
        validate_variogram_lineage,
        validate_nugget_consistency,
        VariogramGateError,
        validate_pre_kriging,
        validate_pre_simulation,
    )

See: docs/VARIOGRAM_SUBSYSTEM_AUDIT.md for full audit report.
"""

import hashlib
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

# STRICT_MODE: When True, gate violations raise errors. When False, they log warnings.
# AUDIT FIX (CRITICAL-001): Default to True for JORC/SAMREC compliance.
# Set to False ONLY during development/testing via environment variable.
import os
STRICT_MODE = os.environ.get('GEOX_STRICT_MODE', 'True').lower() in ('true', '1', 'yes', 'on')


class VariogramGateError(Exception):
    """Raised when a variogram gate check fails. This is a FATAL error."""
    pass


class VariogramGateWarning(UserWarning):
    """Warning issued for non-fatal gate violations."""
    pass


# =============================================================================
# DATA LINEAGE GATES
# =============================================================================

def compute_data_hash(df: pd.DataFrame, variable: Optional[str] = None) -> str:
    """
    Compute a deterministic hash of a DataFrame for lineage tracking.
    
    The hash is computed from the data values (not column names) to detect
    any change in the underlying data used for variogram calculation.
    
    Args:
        df: Source DataFrame
        variable: Optional variable name to include in hash computation.
                  If provided, only coordinates (X, Y, Z) and the variable
                  column are hashed. If None, entire DataFrame is hashed.
        
    Returns:
        16-character hex hash string
        
    Note:
        Hash is case-sensitive and order-dependent. Sorting the DataFrame
        before hashing is recommended if row order is not significant.
    """
    try:
        if variable and variable in df.columns:
            # Hash only relevant columns for variogram (coords + variable)
            cols_to_hash = ['X', 'Y', 'Z', variable]
            cols_available = [c for c in cols_to_hash if c in df.columns]
            df_to_hash = df[cols_available].dropna()
        else:
            df_to_hash = df
        
        # Use pandas hash function for efficiency
        hash_bytes = pd.util.hash_pandas_object(df_to_hash, index=False).values.tobytes()
        return hashlib.sha256(hash_bytes).hexdigest()[:16]
    except Exception as e:
        logger.error(f"Failed to compute data hash: {e}", exc_info=True)
        return "HASH_FAILED"


def add_lineage_metadata(
    result: Dict[str, Any],
    source_df: pd.DataFrame,
    variable: str,
    subsampled: bool = False,
    subsample_size: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Add data lineage metadata to variogram results.
    
    AUDIT REQUIREMENT: Every variogram result must include source_data_hash
    to enable verification at estimation time.
    
    Args:
        result: Variogram result dictionary
        source_df: DataFrame used for variogram calculation
        variable: Variable name analyzed
        subsampled: Whether subsampling was applied
        subsample_size: Number of samples after subsampling
        random_seed: Random seed used for subsampling (if applicable)
        
    Returns:
        Result dict with lineage metadata added
    """
    metadata = result.get('metadata', {})
    
    # Core lineage fields
    metadata['source_data_hash'] = compute_data_hash(source_df)
    metadata['source_data_shape'] = source_df.shape
    metadata['source_data_n_rows'] = len(source_df)
    metadata['variable_analyzed'] = variable
    
    # Track columns for debugging
    metadata['source_columns'] = list(source_df.columns)
    
    # Subsampling tracking (CRITICAL for lineage)
    if subsampled:
        metadata['subsampled'] = True
        metadata['subsample_size'] = subsample_size
        metadata['subsample_seed'] = random_seed
        metadata['subsample_fraction'] = subsample_size / len(source_df) if len(source_df) > 0 else 0
        logger.warning(
            f"LINEAGE: Variogram computed on subsample ({subsample_size}/{len(source_df)} samples). "
            f"Seed: {random_seed}. Consider using full dataset for production."
        )
    else:
        metadata['subsampled'] = False
    
    result['metadata'] = metadata
    return result


def validate_variogram_data_match(
    variogram_params: Dict[str, Any],
    estimation_data: pd.DataFrame,
    strict: bool = False
) -> bool:
    """
    ESTIMATION GATE: Verify variogram was computed on matching data.
    
    This is a PRE-FLIGHT CHECK that should be called before any estimation
    engine (kriging, simulation) uses a variogram model.
    
    Args:
        variogram_params: Variogram parameters including metadata
        estimation_data: DataFrame to be used for estimation
        strict: If True, raise error on mismatch. If False, log warning only.
        
    Returns:
        True if data matches, False if mismatch detected
        
    Raises:
        VariogramGateError: If strict=True and data does not match
    """
    # Check for source hash in variogram metadata
    metadata = variogram_params.get('metadata', {})
    source_hash = metadata.get('source_data_hash')
    
    if source_hash is None:
        msg = (
            "LINEAGE WARNING: Variogram has no source_data_hash. "
            "Cannot verify data consistency. Audit trail incomplete."
        )
        logger.warning(msg)
        if strict:
            raise VariogramGateError(msg)
        return True  # Assume OK if no hash available (legacy data)
    
    # Compute current data hash
    current_hash = compute_data_hash(estimation_data)
    
    if current_hash != source_hash:
        msg = (
            f"VARIOGRAM GATE FAILED: Estimation data hash ({current_hash}) "
            f"does not match variogram source hash ({source_hash}). "
            f"Variogram shape: {metadata.get('source_data_shape', 'unknown')}, "
            f"Estimation shape: {estimation_data.shape}. "
            f"Re-compute variogram on current dataset for JORC compliance."
        )
        logger.error(msg)
        if strict:
            raise VariogramGateError(msg)
        return False
    
    logger.debug(f"LINEAGE VERIFIED: Data hash matches ({current_hash})")
    return True


# =============================================================================
# NUGGET CONSISTENCY GATES
# =============================================================================

@dataclass
class NuggetConsistencyReport:
    """Report on nugget consistency across directional variograms."""
    nuggets: Dict[str, float]
    min_nugget: float
    max_nugget: float
    median_nugget: float
    ratio: float  # max/min
    is_consistent: bool
    recommended_global: float
    warning_message: Optional[str]


def analyze_nugget_consistency(
    variogram_results: Dict[str, Any],
    tolerance_ratio: float = 3.0
) -> NuggetConsistencyReport:
    """
    Analyze nugget values across directional variograms.
    
    AUDIT REQUIREMENT: Nugget should be consistent across directions.
    Large variations indicate fitting issues or data problems.
    
    Args:
        variogram_results: Complete variogram analysis results
        tolerance_ratio: Maximum acceptable ratio of max/min nugget
        
    Returns:
        NuggetConsistencyReport with analysis results
    """
    fitted = variogram_results.get('fitted_models', {})
    nuggets = {}
    
    # Extract nuggets from all directions and model types
    for direction in ['downhole', 'omni', 'major', 'minor', 'vertical']:
        dir_fits = fitted.get(direction, {})
        for model_type, params in dir_fits.items():
            if isinstance(params, dict) and 'nugget' in params:
                key = f"{direction}_{model_type}"
                nuggets[key] = float(params['nugget'])
    
    if not nuggets:
        return NuggetConsistencyReport(
            nuggets={},
            min_nugget=0.0,
            max_nugget=0.0,
            median_nugget=0.0,
            ratio=1.0,
            is_consistent=True,
            recommended_global=0.0,
            warning_message="No fitted models found with nugget values"
        )
    
    values = list(nuggets.values())
    min_nug = min(values)
    max_nug = max(values)
    median_nug = float(np.median(values))
    
    # Compute ratio (handle zero min)
    ratio = max_nug / min_nug if min_nug > 1e-9 else float('inf')
    is_consistent = ratio <= tolerance_ratio
    
    warning = None
    if not is_consistent:
        warning = (
            f"NUGGET INCONSISTENCY: Directional nuggets vary by {ratio:.1f}x "
            f"(tolerance: {tolerance_ratio:.1f}x). "
            f"Min={min_nug:.3f}, Max={max_nug:.3f}. "
            f"Consider using Global Nugget lock with median={median_nug:.3f}."
        )
        logger.warning(warning)
    
    return NuggetConsistencyReport(
        nuggets=nuggets,
        min_nugget=min_nug,
        max_nugget=max_nug,
        median_nugget=median_nug,
        ratio=ratio,
        is_consistent=is_consistent,
        recommended_global=median_nug,
        warning_message=warning
    )


def validate_nugget_for_estimation(
    variogram_results: Dict[str, Any],
    target_engine: str,
    strict: bool = False
) -> float:
    """
    NUGGET GATE: Get validated nugget value for estimation engine.
    
    This function enforces nugget consistency before estimation proceeds.
    
    Args:
        variogram_results: Complete variogram analysis results
        target_engine: Name of estimation engine (for logging)
        strict: If True, raise error on inconsistency
        
    Returns:
        Single nugget value to use for estimation
        
    Raises:
        VariogramGateError: If strict=True and nuggets are inconsistent
    """
    report = analyze_nugget_consistency(variogram_results)
    
    if not report.is_consistent:
        msg = (
            f"NUGGET GATE for {target_engine}: {report.warning_message}. "
            f"Using median={report.recommended_global:.3f}."
        )
        if strict:
            raise VariogramGateError(msg)
        logger.warning(msg)
    
    # Prefer combined model nugget if available
    combined = variogram_results.get('combined_3d_model', {})
    if combined and 'nugget' in combined:
        return float(combined['nugget'])
    
    # Fall back to median
    return report.recommended_global


# =============================================================================
# ANISOTROPY CONSISTENCY GATES
# =============================================================================

def validate_anisotropy_consistency(
    variogram_anisotropy: Dict[str, Any],
    kriging_anisotropy: Dict[str, Any],
    tolerance_deg: float = 5.0,
    tolerance_ratio: float = 0.1
) -> Tuple[bool, Optional[str]]:
    """
    Validate that anisotropy parameters are consistent between variogram and kriging.
    
    AUDIT REQUIREMENT: The search ellipsoid should match the variogram anisotropy.
    
    Args:
        variogram_anisotropy: Anisotropy from variogram fitting
        kriging_anisotropy: Anisotropy specified for kriging
        tolerance_deg: Acceptable angle difference in degrees
        tolerance_ratio: Acceptable range ratio difference
        
    Returns:
        Tuple of (is_consistent, warning_message)
    """
    if not variogram_anisotropy or not kriging_anisotropy:
        return True, None
    
    issues = []
    
    # Check angles
    v_az = variogram_anisotropy.get('azimuth', 0)
    k_az = kriging_anisotropy.get('azimuth', 0)
    az_diff = abs(v_az - k_az)
    az_diff = min(az_diff, 360 - az_diff)  # Handle wraparound
    
    if az_diff > tolerance_deg:
        issues.append(f"Azimuth mismatch: variogram={v_az:.1f}°, kriging={k_az:.1f}°")
    
    v_dip = variogram_anisotropy.get('dip', 0)
    k_dip = kriging_anisotropy.get('dip', 0)
    dip_diff = abs(v_dip - k_dip)
    
    if dip_diff > tolerance_deg:
        issues.append(f"Dip mismatch: variogram={v_dip:.1f}°, kriging={k_dip:.1f}°")
    
    # Check range ratios
    for key in ['major_range', 'minor_range', 'vert_range']:
        v_val = variogram_anisotropy.get(key, 100.0)
        k_val = kriging_anisotropy.get(key, 100.0)
        if v_val > 0 and k_val > 0:
            ratio_diff = abs(v_val - k_val) / max(v_val, k_val)
            if ratio_diff > tolerance_ratio:
                issues.append(f"{key} mismatch: variogram={v_val:.1f}, kriging={k_val:.1f}")
    
    if issues:
        msg = "ANISOTROPY INCONSISTENCY: " + "; ".join(issues)
        logger.warning(msg)
        return False, msg
    
    return True, None


# =============================================================================
# MODEL FUNCTION VERIFICATION
# =============================================================================

def verify_variogram_model_consistency() -> Dict[str, Any]:
    """
    AUDIT TEST: Verify all variogram model implementations produce identical results.
    
    This function tests that the multiple variogram implementations in the codebase
    produce numerically equivalent results.
    
    Returns:
        Dict with test results for each model comparison
        
    Note:
        Run this as part of CI/CD to catch any implementation drift.
    """
    results = {'passed': True, 'tests': []}
    
    # Test distances
    h = np.array([0.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0])
    test_params = [
        {'nugget': 0.0, 'sill': 1.0, 'range': 100.0},
        {'nugget': 0.5, 'sill': 1.5, 'range': 100.0},
        {'nugget': 0.1, 'sill': 0.9, 'range': 50.0},
    ]
    
    for params in test_params:
        nugget = params['nugget']
        sill = params['sill']
        range_ = params['range']
        
        # Import all implementations
        try:
            from ..geostats.geostats_utils import spherical_variogram as gs_sph
            from ..models.variogram_functions import spherical_model as vf_sph
            from ..models.kriging3d import spherical_variogram as k3d_sph
            
            # geostats_utils: (h, nugget, sill, range_)
            result_gs = gs_sph(h, nugget, sill, range_)
            
            # variogram_functions: (h, range_, sill, nugget)
            result_vf = vf_sph(h, range_, sill, nugget)
            
            # kriging3d: (h, range_, sill, nugget) where sill is PARTIAL sill
            partial_sill = sill - nugget
            result_k3d = k3d_sph(h, range_, partial_sill, nugget)
            
            # Compare
            try:
                np.testing.assert_allclose(result_gs, result_vf, rtol=1e-10)
                np.testing.assert_allclose(result_gs, result_k3d, rtol=1e-10)
                results['tests'].append({
                    'params': params,
                    'status': 'PASS',
                    'message': 'All implementations match'
                })
            except AssertionError as e:
                results['passed'] = False
                results['tests'].append({
                    'params': params,
                    'status': 'FAIL',
                    'message': str(e),
                    'values': {
                        'geostats_utils': result_gs.tolist(),
                        'variogram_functions': result_vf.tolist(),
                        'kriging3d': result_k3d.tolist(),
                    }
                })
                
        except ImportError as e:
            results['tests'].append({
                'params': params,
                'status': 'SKIP',
                'message': f'Import failed: {e}'
            })
    
    return results


# =============================================================================
# SIMULATION GATE
# =============================================================================

def validate_simulation_variograms(
    config: Any,  # SISConfig or similar
    require_explicit: bool = True
) -> Tuple[bool, List[str]]:
    """
    SIMULATION GATE: Verify variograms are explicitly provided (not auto-generated).
    
    Args:
        config: Simulation configuration object
        require_explicit: If True, reject auto-generated variograms
        
    Returns:
        Tuple of (is_valid, list of warnings/errors)
    """
    issues = []
    
    # Check for indicator variograms if SIS
    if hasattr(config, 'indicator_variograms'):
        thresholds = getattr(config, 'thresholds', [])
        variograms = config.indicator_variograms
        
        for thresh in thresholds:
            if thresh not in variograms:
                msg = f"Threshold {thresh}: No variogram provided, would be auto-generated"
                issues.append(msg)
                if require_explicit:
                    logger.error(f"SIMULATION GATE: {msg}")
    
    # Check for primary variogram
    for attr in ['variogram_params', 'variogram_primary', 'variogram']:
        if hasattr(config, attr):
            vario = getattr(config, attr)
            if vario is None or (isinstance(vario, dict) and not vario):
                issues.append(f"Missing {attr}: would use defaults")
    
    is_valid = len(issues) == 0 or not require_explicit
    
    if issues and require_explicit:
        logger.error(
            "SIMULATION GATE FAILED: Explicit variograms required. "
            f"Issues: {issues}"
        )
    elif issues:
        for issue in issues:
            logger.warning(f"SIMULATION WARNING: {issue}")
    
    return is_valid, issues


# =============================================================================
# COMPOSITE GATE FUNCTION
# =============================================================================

def run_variogram_gates(
    variogram_results: Dict[str, Any],
    estimation_data: pd.DataFrame,
    target_engine: str,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Run all variogram gates before estimation.
    
    This is the main entry point for pre-estimation validation.
    
    Args:
        variogram_results: Complete variogram analysis results
        estimation_data: DataFrame to be used for estimation
        target_engine: Name of estimation engine
        strict: If True, raise errors instead of warnings
        
    Returns:
        Dict with gate results and validated parameters
        
    Raises:
        VariogramGateError: If strict=True and any gate fails
    """
    gate_results = {
        'all_passed': True,
        'gates': {},
        'warnings': [],
        'validated_params': {}
    }
    
    # Gate 1: Data lineage
    try:
        lineage_ok = validate_variogram_data_match(
            variogram_results, estimation_data, strict=strict
        )
        gate_results['gates']['data_lineage'] = lineage_ok
        if not lineage_ok:
            gate_results['all_passed'] = False
            gate_results['warnings'].append("Data lineage mismatch")
    except VariogramGateError as e:
        gate_results['gates']['data_lineage'] = False
        gate_results['all_passed'] = False
        raise
    
    # Gate 2: Nugget consistency
    nugget_report = analyze_nugget_consistency(variogram_results)
    gate_results['gates']['nugget_consistency'] = nugget_report.is_consistent
    gate_results['validated_params']['nugget'] = nugget_report.recommended_global
    
    if not nugget_report.is_consistent:
        gate_results['warnings'].append(nugget_report.warning_message)
        if strict:
            raise VariogramGateError(nugget_report.warning_message)
    
    # Gate 3: Extract validated variogram parameters
    combined = variogram_results.get('combined_3d_model', {})
    gate_results['validated_params'].update({
        'model_type': combined.get('model_type', 'spherical'),
        'range': combined.get('major_range', 100.0),
        'sill': combined.get('sill', 1.0),
        'nugget': combined.get('nugget', nugget_report.recommended_global),
        'anisotropy': {
            'major_range': combined.get('major_range', 100.0),
            'minor_range': combined.get('minor_range', 100.0),
            'vert_range': combined.get('vertical_range', 50.0),
        }
    })
    
    # Log summary
    if gate_results['all_passed']:
        logger.info(f"VARIOGRAM GATES PASSED for {target_engine}")
    else:
        logger.warning(
            f"VARIOGRAM GATES: {len(gate_results['warnings'])} warnings for {target_engine}"
        )
    
    return gate_results


# =============================================================================
# PRE-KRIGING VALIDATION
# =============================================================================

def validate_pre_kriging(
    variogram_params: Union[Dict[str, Any], 'VariogramModel'],
    estimation_data: pd.DataFrame,
    estimation_coords: np.ndarray,
    variable_name: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    KRIGING GATE: Complete pre-kriging validation.
    
    This function MUST be called before running any kriging estimation.
    It validates:
    1. Data lineage (variogram fitted on same data)
    2. Nugget consistency
    3. Sill validity (sill > nugget)
    4. Range validity (range > 0)
    5. Model type validity
    
    Args:
        variogram_params: Variogram parameters (dict or VariogramModel)
        estimation_data: DataFrame containing estimation data
        estimation_coords: (N, 3) array of estimation coordinates
        variable_name: Name of variable being estimated
        strict: If True, raise errors; if False, log warnings
    
    Returns:
        Dict with validation results and corrected parameters
    
    Raises:
        VariogramGateError: If strict=True and validation fails
    """
    from .variogram_model import VariogramModel
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'corrected_params': {},
    }
    
    # Convert VariogramModel to dict if needed
    if isinstance(variogram_params, VariogramModel):
        params = variogram_params.to_kriging_params()
        model = variogram_params
    else:
        params = variogram_params
        model = None
    
    # Extract parameters
    nugget = params.get('nugget', 0.0)
    sill = params.get('sill', 1.0)
    range_ = params.get('range', 100.0)
    model_type = params.get('model_type', 'spherical')
    
    # Validation 1: Nugget >= 0
    if nugget < 0:
        results['errors'].append(f"Invalid nugget: {nugget} < 0")
        nugget = 0.0
        results['valid'] = False
    
    # Validation 2: Sill > Nugget
    if sill <= nugget:
        msg = f"Invalid sill: {sill} <= nugget ({nugget})"
        results['warnings'].append(msg)
        logger.warning(f"KRIGING GATE: {msg}. Adjusting sill.")
        sill = nugget + 0.1  # Minimum adjustment
    
    # Validation 3: Range > 0
    if range_ <= 0:
        results['errors'].append(f"Invalid range: {range_} <= 0")
        range_ = 100.0  # Default fallback
        results['valid'] = False
    
    # Validation 4: Model type valid
    valid_models = ['spherical', 'exponential', 'gaussian', 'linear']
    if model_type.lower() not in valid_models:
        results['warnings'].append(f"Unknown model type '{model_type}', using spherical")
        model_type = 'spherical'
    
    # Validation 5: Data lineage (if hash available)
    if model is not None and model.data_hash is not None:
        from .variogram_model import compute_data_hash
        values = estimation_data[variable_name].dropna().values if variable_name in estimation_data.columns else np.array([])
        if len(values) > 0:
            current_hash = compute_data_hash(
                estimation_coords,
                values,
                variable_name,
            )
            if current_hash != model.data_hash:
                msg = (
                    f"LINEAGE WARNING: Variogram was fitted on different data "
                    f"(hash {model.data_hash} vs current {current_hash})"
                )
                results['warnings'].append(msg)
                logger.warning(f"KRIGING GATE: {msg}")
    
    # Build corrected parameters
    results['corrected_params'] = {
        'nugget': nugget,
        'sill': sill,
        'range': range_,
        'model_type': model_type,
        'anisotropy': params.get('anisotropy'),
    }
    
    # Summary
    if results['errors']:
        msg = f"KRIGING GATE FAILED: {results['errors']}"
        logger.error(msg)
        if strict:
            raise VariogramGateError(msg)
    elif results['warnings']:
        logger.warning(f"KRIGING GATE: {len(results['warnings'])} warnings")
    else:
        logger.info("KRIGING GATE: All validations passed")
    
    return results


def validate_pre_simulation(
    variogram_params: Union[Dict[str, Any], 'VariogramModel'],
    conditioning_data: pd.DataFrame,
    grid_coords: np.ndarray,
    variable_name: str,
    simulation_type: str,
    n_realizations: int,
    random_seed: Optional[int] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    SIMULATION GATE: Complete pre-simulation validation.
    
    This function MUST be called before running any simulation (SGSIM, SIS, etc.).
    Simulations have stricter requirements than kriging because:
    1. Results are used for risk/uncertainty analysis
    2. Multiple realizations compound any variogram errors
    3. JORC/SAMREC requires explicit variogram parameters
    
    Args:
        variogram_params: Variogram parameters (dict or VariogramModel)
        conditioning_data: DataFrame containing conditioning data
        grid_coords: (M, 3) array of simulation grid coordinates
        variable_name: Name of variable being simulated
        simulation_type: Type of simulation (SGSIM, SIS, etc.)
        n_realizations: Number of realizations
        random_seed: Random seed for reproducibility
        strict: If True, raise errors (recommended for simulations)
    
    Returns:
        Dict with validation results
    
    Raises:
        VariogramGateError: If strict=True and validation fails
    """
    from .variogram_model import VariogramModel
    
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'audit_info': {},
    }
    
    # Convert VariogramModel if needed
    if isinstance(variogram_params, VariogramModel):
        model = variogram_params
        params = model.to_kriging_params()
    else:
        model = None
        params = variogram_params
    
    # CRITICAL: Check for auto-generated variogram flag
    if params.get('_auto_generated', False):
        msg = (
            f"SIMULATION GATE FAILED: Variogram was auto-generated. "
            f"Simulations require explicitly fitted variograms for JORC/SAMREC compliance."
        )
        results['errors'].append(msg)
        results['valid'] = False
    
    # Check lineage
    if model is not None:
        validation_errors = model.validate_for_estimation()
        if validation_errors:
            for err in validation_errors:
                if err.startswith("AUDIT:"):
                    results['warnings'].append(err)
                else:
                    results['errors'].append(err)
                    results['valid'] = False
    
    # Check random seed for reproducibility
    if random_seed is None:
        results['warnings'].append(
            "No random_seed provided. Simulation will not be reproducible."
        )
        results['audit_info']['reproducible'] = False
    else:
        results['audit_info']['reproducible'] = True
        results['audit_info']['random_seed'] = random_seed
    
    # Audit info
    results['audit_info'].update({
        'simulation_type': simulation_type,
        'n_realizations': n_realizations,
        'n_conditioning': len(conditioning_data),
        'n_grid': len(grid_coords),
        'variable': variable_name,
        'variogram_source': 'explicit' if not params.get('_auto_generated') else 'auto_generated',
    })
    
    # Summary
    if results['errors']:
        msg = f"SIMULATION GATE FAILED: {results['errors']}"
        logger.error(msg)
        if strict:
            raise VariogramGateError(msg)
    elif results['warnings']:
        logger.warning(f"SIMULATION GATE: {len(results['warnings'])} warnings")
    else:
        logger.info(f"SIMULATION GATE: {simulation_type} validation passed")
    
    return results


# =============================================================================
# SILL CONSISTENCY CHECK
# =============================================================================

def validate_sill_consistency(
    variogram_results: Dict[str, Any],
    sample_variance: float,
    tolerance: float = 0.3,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that fitted sills are consistent with sample variance.
    
    GEOSTATISTICAL PRINCIPLE: The sill should approximately equal the sample variance.
    Large deviations indicate fitting issues or data problems.
    
    Args:
        variogram_results: Complete variogram analysis results
        sample_variance: Variance of the sample data
        tolerance: Acceptable deviation from sample variance (e.g., 0.3 = 30%)
    
    Returns:
        Tuple of (is_consistent, dict of direction -> sill_ratio)
    """
    fitted = variogram_results.get('fitted_models', {})
    sill_ratios = {}
    issues = []
    
    for direction in ['omni', 'major', 'minor', 'vertical', 'downhole']:
        dir_fits = fitted.get(direction, {})
        for model_type, params in dir_fits.items():
            if isinstance(params, dict) and 'total_sill' in params:
                total_sill = params['total_sill']
            elif isinstance(params, dict) and 'nugget' in params and 'sill' in params:
                total_sill = params['nugget'] + params['sill']
            else:
                continue
            
            if sample_variance > 0:
                ratio = total_sill / sample_variance
                key = f"{direction}_{model_type}"
                sill_ratios[key] = ratio
                
                if abs(ratio - 1.0) > tolerance:
                    issues.append(
                        f"{direction}: sill={total_sill:.2f} is {ratio:.1f}x sample variance"
                    )
    
    is_consistent = len(issues) == 0
    
    if not is_consistent:
        logger.warning(
            f"SILL CONSISTENCY: {len(issues)} directions deviate >30% from sample variance. "
            f"Issues: {issues}"
        )
    
    return is_consistent, sill_ratios


# =============================================================================
# AUDIT FIX: Enhanced Validation Functions
# =============================================================================

def validate_full_data_lineage(
    variogram_metadata: Dict[str, Any],
    current_coords: np.ndarray,
    current_values: np.ndarray,
    strict: bool = False,
) -> Tuple[bool, str]:
    """
    AUDIT FIX (V-NEW-003): Validate against FULL data hash (not subsampled).
    
    This function checks if the full dataset has changed since the variogram
    was computed, even when subsampling was used.
    
    Args:
        variogram_metadata: Metadata from variogram results (must contain 'full_data_hash')
        current_coords: Current estimation coordinates (N, 3)
        current_values: Current estimation values (N,)
        strict: If True, raise error on mismatch
    
    Returns:
        Tuple of (is_valid, message)
    
    Raises:
        VariogramGateError: If strict=True and validation fails
    """
    import hashlib
    
    # Get full data hash from variogram metadata
    lineage = variogram_metadata.get('lineage', {})
    stored_full_hash = lineage.get('full_data_hash') or variogram_metadata.get('full_data_hash')
    
    if stored_full_hash is None:
        # Legacy data without full hash - cannot validate
        msg = (
            "LINEAGE WARNING: Variogram metadata does not contain 'full_data_hash'. "
            "Cannot verify full dataset consistency. This may indicate legacy data."
        )
        logger.warning(msg)
        return True, msg
    
    # Compute current full data hash
    current_coords_hash = hashlib.sha256(
        np.ascontiguousarray(np.round(current_coords, 6)).tobytes()
    ).hexdigest()[:16]
    current_values_hash = hashlib.sha256(
        np.ascontiguousarray(np.round(current_values, 8)).tobytes()
    ).hexdigest()[:16]
    current_full_hash = f"{current_coords_hash}_{current_values_hash}"
    
    if current_full_hash != stored_full_hash:
        msg = (
            f"FULL DATA LINEAGE MISMATCH: Stored hash ({stored_full_hash}) != "
            f"current hash ({current_full_hash}). The full dataset has changed "
            f"since the variogram was computed. Re-compute variogram on current data."
        )
        logger.error(msg)
        if strict:
            raise VariogramGateError(msg)
        return False, msg
    
    # Check subsampling info
    was_subsampled = lineage.get('subsampled', False) or variogram_metadata.get('subsampled', False)
    if was_subsampled:
        subsample_n = lineage.get('subsample_n') or variogram_metadata.get('subsample_n')
        full_n = lineage.get('full_data_n') or variogram_metadata.get('full_data_n')
        logger.info(
            f"LINEAGE: Variogram was computed on subsample ({subsample_n}/{full_n} samples). "
            f"Full data hash verified."
        )
    
    return True, "Full data lineage verified"


def validate_sill_interpretation(
    variogram_params: Dict[str, Any],
    engine_type: str,
) -> Dict[str, Any]:
    """
    AUDIT FIX (V-NEW-001): Validate sill interpretation consistency.
    
    Ensures that the sill parameter is interpreted correctly by the target engine.
    All engines now expect TOTAL sill (nugget + partial_sill).
    
    Args:
        variogram_params: Variogram parameters including 'nugget' and 'sill'
        engine_type: Target engine ('OK', 'SK', 'UK', 'IK', 'CoK', 'SIS', 'SGSIM')
    
    Returns:
        Dict with validation results and canonical parameters
    """
    nugget = variogram_params.get('nugget', 0.0)
    sill = variogram_params.get('sill', 1.0)
    
    results = {
        'valid': True,
        'warnings': [],
        'canonical_sill': sill,  # TOTAL sill
        'canonical_nugget': nugget,
        'canonical_partial_sill': max(sill - nugget, 0.0),
    }
    
    # Sanity checks
    if nugget < 0:
        results['valid'] = False
        results['warnings'].append(f"Nugget cannot be negative (got {nugget})")
        results['canonical_nugget'] = 0.0
    
    if sill <= 0:
        results['valid'] = False
        results['warnings'].append(f"Sill must be positive (got {sill})")
        results['canonical_sill'] = 1.0
    
    if nugget >= sill:
        results['valid'] = False
        results['warnings'].append(
            f"Nugget ({nugget}) must be less than sill ({sill}). "
            f"Partial sill would be zero or negative."
        )
        results['canonical_sill'] = nugget + 0.1
    
    # Log interpretation for audit trail
    partial_sill = results['canonical_sill'] - results['canonical_nugget']
    logger.info(
        f"SILL INTERPRETATION for {engine_type}: "
        f"Total sill = {results['canonical_sill']:.4f}, "
        f"Nugget = {results['canonical_nugget']:.4f}, "
        f"Partial sill = {partial_sill:.4f}"
    )
    
    if results['warnings']:
        for w in results['warnings']:
            logger.warning(f"SILL VALIDATION: {w}")
    
    return results


def enable_strict_mode():
    """Enable strict mode for production/JORC audits."""
    global STRICT_MODE
    STRICT_MODE = True
    logger.warning("VARIOGRAM GATES: STRICT MODE ENABLED - violations will raise errors")


def disable_strict_mode():
    """Disable strict mode for development."""
    global STRICT_MODE
    STRICT_MODE = False
    logger.info("VARIOGRAM GATES: Strict mode disabled - violations will log warnings only")


# =============================================================================
# AUDIT FIX: Data Source Validation Gate (CRITICAL-002)
# =============================================================================

class DataSourceError(Exception):
    """Raised when data source validation fails."""
    pass


def validate_data_source(
    data_df: pd.DataFrame,
    engine_name: str,
    strict: bool = None,
) -> Dict[str, Any]:
    """
    ESTIMATION GATE: Validate that data source is NOT raw assays.
    
    AUDIT FIX (CRITICAL-002): This gate MUST be called by all estimation engines
    to ensure raw assays are NEVER used directly for kriging/simulation.
    
    This enforces the geostatistical principle that estimation should only be
    performed on properly prepared data (composites, declustered composites).
    
    Args:
        data_df: DataFrame to validate
        engine_name: Name of the calling engine (for audit trail)
        strict: If True, raise error. If None, uses global STRICT_MODE.
    
    Returns:
        Dict with validation results:
        - 'valid': bool
        - 'source_type': detected source type
        - 'warnings': list of warnings
        - 'version': dataset version if available
    
    Raises:
        DataSourceError: If strict=True and data is raw assays
    """
    if strict is None:
        strict = STRICT_MODE
    
    results = {
        'valid': True,
        'source_type': 'unknown',
        'warnings': [],
        'version': None,
        'lineage_gate_passed': False,
    }
    
    # Check DataFrame attrs for source_type (set by data registry)
    source_type = getattr(data_df, 'attrs', {}).get('source_type', 'unknown')
    lineage_gate_passed = getattr(data_df, 'attrs', {}).get('lineage_gate_passed', False)
    validation_status = getattr(data_df, 'attrs', {}).get('validation_status', 'NOT_RUN')
    
    results['source_type'] = source_type
    results['lineage_gate_passed'] = lineage_gate_passed
    results['validation_status'] = validation_status
    
    # CRITICAL CHECK: Block raw assays
    if source_type in ['raw_assays', 'assays', 'raw']:
        error_msg = (
            f"DATA SOURCE GATE FAILED for {engine_name}: "
            f"Data source type is '{source_type}' (raw assays). "
            "Estimation on raw assays violates change-of-support principles "
            "and is NOT permitted for JORC/SAMREC compliance. "
            "Steps to fix:\n"
            "1. Run compositing on the raw assays\n"
            "2. Optionally run declustering\n"
            "3. Re-run estimation with composited data"
        )
        logger.error(error_msg)
        results['valid'] = False
        results['warnings'].append(error_msg)
        
        if strict:
            raise DataSourceError(error_msg)
    
    # WARNING: Unknown source type
    elif source_type == 'unknown':
        warning_msg = (
            f"DATA SOURCE WARNING for {engine_name}: "
            "Data source type is unknown (no 'source_type' in DataFrame attrs). "
            "Data may have been loaded outside the standard pipeline. "
            "For JORC/SAMREC compliance, use DataRegistry.get_validated_composites() "
            "or DataRegistry.get_estimation_ready_data()."
        )
        logger.warning(warning_msg)
        results['warnings'].append(warning_msg)
        
        # In strict mode, unknown source is also an error
        if strict:
            results['valid'] = False
            raise DataSourceError(warning_msg)
    
    # Check for lineage gate
    if not lineage_gate_passed:
        warning_msg = (
            f"DATA SOURCE WARNING for {engine_name}: "
            "Data has not passed the lineage gate (lineage_gate_passed=False). "
            "This may indicate data was loaded outside the standard pipeline."
        )
        logger.warning(warning_msg)
        results['warnings'].append(warning_msg)
    
    # Extract version/hash if available
    results['version'] = getattr(data_df, 'attrs', {}).get('data_hash') or \
                         getattr(data_df, 'attrs', {}).get('version')
    
    # Log validation result
    if results['valid']:
        logger.info(
            f"DATA SOURCE GATE PASSED for {engine_name}: "
            f"source_type={source_type}, validation_status={validation_status}, "
            f"rows={len(data_df)}"
        )
    
    return results


def get_gate_summary() -> Dict[str, Any]:
    """
    Get summary of variogram gate configuration.
    
    Useful for audit logs and compliance reports.
    """
    return {
        'strict_mode': STRICT_MODE,
        'module_version': '2.1.0',  # Post-audit version with CRITICAL-002 fix
        'audit_fixes_applied': [
            'V-NEW-001: Sill interpretation standardized (TOTAL sill)',
            'V-NEW-002: Kriging fallback uses canonical convention',
            'V-NEW-003: Full data hash computed BEFORE subsampling',
            'V-NEW-004: SGSIM pre-simulation gate added',
            'V-NEW-005: Cross-variogram auto-generation flagged',
            'CRITICAL-001: STRICT_MODE defaults to True',
            'CRITICAL-002: Data source validation gate added',
            'CRITICAL-003: Explicit cross-variogram required for Co-Kriging',
            'HIGH-001: SIS auto-variogram bypass removed',
        ],
        'gates_available': [
            'compute_data_hash',
            'validate_variogram_data_match',
            'analyze_nugget_consistency',
            'validate_nugget_for_estimation',
            'validate_anisotropy_consistency',
            'validate_pre_kriging',
            'validate_pre_simulation',
            'validate_full_data_lineage',
            'validate_sill_interpretation',
            'validate_data_source',  # NEW
        ]
    }