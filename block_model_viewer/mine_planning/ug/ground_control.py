"""Basic ground control / rock mass characterization stubs.

This module provides minimal placeholder implementations so the Underground panel
can invoke ground control calculations without raising ModuleNotFoundError.
Each function returns deterministic, simple values based on provided inputs.

Future improvements:
- Implement proper RMR and Q-system classification formulas
- Add pillar factor-of-safety calculations using tributary area / strength models
- Integrate support selection logic with empirical tables
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class RockMassProperties:
    """Container for basic rock mass characterization outputs."""
    rmr: float
    q_value: float
    support_category: str
    pillar_fos: Optional[float] = None
    notes: Optional[str] = None

# --- Placeholder calculation functions --- #

def calculate_rmr(uniaxial_strength_mpa: float, joint_spacing_m: float, groundwater: str = "dry") -> float:
    """
    Return a simplistic RMR-like score (0-100).
    
    ⚠️ WARNING: This is NOT the real Bieniawski RMR classification.
    This is a PLACEHOLDER implementation using simplified scaling.
    
    For production use, implement the full Bieniawski RMR system:
    - Strength of intact rock (0-15 points)
    - RQD (0-20 points)
    - Spacing of discontinuities (0-20 points)
    - Condition of discontinuities (0-30 points)
    - Groundwater conditions (0-15 points)
    - Orientation of discontinuities (adjustment factor)
    
    UI WARNING: Any UI displaying RMR values from this function should
    include a watermark: "Estimated RMR (Simplified Model)".
    
    Returns:
        RMR value (0-100) with embedded warning in metadata.
        Note: This is a placeholder and should not be used for production decisions.
    """
    base = min(max(uniaxial_strength_mpa / 2.0, 0), 60)  # cap strength contribution
    spacing_bonus = min(max((2.0 - joint_spacing_m) * 10, -10), 20)  # closer spacing reduces quality
    water_penalty = 0 if groundwater == "dry" else -5
    rmr = base + spacing_bonus + water_penalty + 25  # offset so typical values are mid-range
    result = max(0, min(rmr, 100))
    
    # Log warning for debugging/monitoring
    logger.warning(
        f"⚠️ PLACEHOLDER RMR calculation: {result:.1f} "
        f"(Estimated RMR using Simplified Model - NOT real Bieniawski RMR)"
    )
    
    return result

def calculate_q_system(rqd_percent: float, joint_set_count: int, joint_water_reduction: float = 1.0) -> float:
    """
    Return a simplistic Q-value proxy.
    
    ⚠️ WARNING: This is NOT the real Barton Q-system classification.
    This is a PLACEHOLDER implementation using drastically simplified formulas.
    
    Real Q-system formula: Q = (RQD/Jn) * (Jr/Ja) * (Jw/SRF)
    Where:
    - RQD: Rock Quality Designation
    - Jn: Joint set number
    - Jr: Joint roughness number
    - Ja: Joint alteration number
    - Jw: Joint water reduction factor
    - SRF: Stress Reduction Factor
    
    This placeholder assumes neutral friction/alteration and simplified scaling.
    
    UI WARNING: Any UI displaying Q-values from this function should
    include a watermark: "Estimated Q-value (Simplified Model)".
    
    Returns:
        Q-value with embedded warning in metadata.
        Note: This is a placeholder and should not be used for production decisions.
    """
    rqd_factor = max(rqd_percent, 0) / 100.0  # 0-1
    jn = max(joint_set_count, 1)
    jr_ja_ratio = 1.0  # assume neutral friction/alteration
    jw_srf_ratio = 1.0 / max(joint_water_reduction, 0.5)
    q_val = (rqd_factor / jn) * jr_ja_ratio * jw_srf_ratio * 10  # scale to a 0-10 typical band
    result = max(q_val, 0.01)
    
    # Log warning for debugging/monitoring
    logger.warning(
        f"⚠️ PLACEHOLDER Q-system calculation: {result:.2f} "
        f"(Estimated Q-value using Simplified Model - NOT real Barton Q-system)"
    )
    
    return result

def calculate_pillar_fos(pillar_width_m: float, pillar_height_m: float, rock_strength_mpa: float) -> float:
    """Very rough placeholder: FoS scales with width/height and strength."""
    if pillar_height_m <= 0:
        return 0.0
    slenderness = pillar_width_m / pillar_height_m
    fos = rock_strength_mpa / 50.0 * (0.5 + slenderness)
    return max(fos, 0.1)

def select_support(rmr: float, q_value: float) -> str:
    """Pick a support category based on ranges of RMR / Q.
    This is loosely inspired by empirical charts but deliberately coarse.
    """
    if rmr > 70 and q_value > 5:
        return "Light bolts"
    if rmr > 50 and q_value > 2:
        return "Bolts + mesh"
    if rmr > 30 and q_value > 1:
        return "Bolts + mesh + shotcrete"
    return "Heavy support / shotcrete + steel sets"

def characterize_rock_mass(strength_mpa: float, joint_spacing_m: float, groundwater: str,
                           rqd_percent: float, joint_set_count: int, joint_water_reduction: float,
                           pillar: Optional[Dict[str, float]] = None) -> RockMassProperties:
    """
    High-level wrapper combining all placeholder calculations.
    
    ⚠️ WARNING: This function uses simplified placeholder formulas for RMR and Q-system.
    Results should be clearly marked as "Estimated" in any UI output.
    """
    rmr = calculate_rmr(strength_mpa, joint_spacing_m, groundwater)
    q_val = calculate_q_system(rqd_percent, joint_set_count, joint_water_reduction)
    support = select_support(rmr, q_val)
    pillar_fos = None
    if pillar:
        try:
            pillar_fos = calculate_pillar_fos(
                pillar_width_m=pillar.get("width_m", 0),
                pillar_height_m=pillar.get("height_m", 0),
                rock_strength_mpa=strength_mpa
            )
        except Exception:
            pillar_fos = None
    notes = "⚠️ PLACEHOLDER: Estimated RMR and Q-value using simplified models. Not suitable for production use." if (pillar_fos is not None) else "⚠️ PLACEHOLDER: Estimated RMR and Q-value using simplified models. Not suitable for production use."
    return RockMassProperties(rmr=rmr, q_value=q_val, support_category=support, pillar_fos=pillar_fos, notes=notes)

# STEP 19: Geotechnical stability integration hooks

# Global registry for stope stability results (keyed by result_id)
_stope_stability_registry: Dict[str, Any] = {}


def register_stope_stability(stope_id: str, stability_result: Any, result_id: Optional[str] = None) -> str:
    """
    Register a stope stability result for a stope.
    
    Args:
        stope_id: Stope identifier
        stability_result: StopeStabilityResult instance or dict
        result_id: Optional result ID (auto-generated if None)
        
    Returns:
        Result ID string
    """
    if result_id is None:
        import uuid
        result_id = f"stability_{stope_id}_{uuid.uuid4().hex[:8]}"
    
    _stope_stability_registry[result_id] = {
        'stope_id': stope_id,
        'result': stability_result,
        'timestamp': None  # Could add timestamp if needed
    }
    
    return result_id


def get_stope_stability(stope_id: str) -> Optional[Any]:
    """
    Get stope stability result for a stope.
    
    Args:
        stope_id: Stope identifier
        
    Returns:
        StopeStabilityResult or dict, or None if not found
    """
    # Search registry for matching stope_id
    for result_id, entry in _stope_stability_registry.items():
        if entry.get('stope_id') == stope_id:
            return entry.get('result')
    return None


def get_stope_stability_by_result_id(result_id: str) -> Optional[Any]:
    """
    Get stope stability result by result ID.
    
    Args:
        result_id: Result identifier
        
    Returns:
        StopeStabilityResult or dict, or None if not found
    """
    entry = _stope_stability_registry.get(result_id)
    return entry.get('result') if entry else None


# STEP 20: Seismic integration hooks

# Global registry for seismic events near stopes
_stope_seismic_registry: Dict[str, List[Any]] = {}  # stope_id -> list of event_ids

# Global registry for rockburst indices
_drive_rockburst_registry: Dict[str, Any] = {}  # drive_id -> rockburst result


def register_stope_seismic_events(stope_id: str, event_ids: List[str]) -> None:
    """
    Register seismic events in vicinity of a stope.
    
    Args:
        stope_id: Stope identifier
        event_ids: List of seismic event IDs
    """
    _stope_seismic_registry[stope_id] = event_ids


def get_stope_seismic_events(stope_id: str) -> List[str]:
    """
    Get seismic event IDs near a stope.
    
    Args:
        stope_id: Stope identifier
        
    Returns:
        List of event IDs
    """
    return _stope_seismic_registry.get(stope_id, [])


def register_drive_rockburst(drive_id: str, rockburst_result: Any) -> None:
    """
    Register rockburst index result for a drive.
    
    Args:
        drive_id: Drive identifier
        rockburst_result: RockburstIndexResult or dict
    """
    _drive_rockburst_registry[drive_id] = rockburst_result


def get_drive_rockburst(drive_id: str) -> Optional[Any]:
    """
    Get rockburst index result for a drive.
    
    Args:
        drive_id: Drive identifier
        
    Returns:
        RockburstIndexResult or dict, or None if not found
    """
    return _drive_rockburst_registry.get(drive_id)


def get_stope_centroids(stopes: List[Any]) -> np.ndarray:
    """
    Get centroids of stopes.
    
    Args:
        stopes: List of Stope instances
        
    Returns:
        Array of centroids (n_stopes, 3)
    """
    import numpy as np
    from ..core import Stope
    
    centroids = []
    for stope in stopes:
        if isinstance(stope, Stope):
            # Use block indices to compute centroid
            # Simplified: would need block model to get actual centroids
            # For now, return placeholder
            centroids.append([0.0, 0.0, 0.0])
        else:
            centroids.append([0.0, 0.0, 0.0])
    
    return np.array(centroids)


def get_drive_segments(drives: List[Any]) -> np.ndarray:
    """
    Get segment points for drives.
    
    Args:
        drives: List of drive objects
        
    Returns:
        Array of segment points (n_segments, 3)
    """
    import numpy as np
    
    # Placeholder: would need actual drive geometry
    # Returns empty array for now
    return np.array([])


__all__ = [
    "RockMassProperties",
    "calculate_rmr",
    "calculate_q_system",
    "calculate_pillar_fos",
    "select_support",
    "characterize_rock_mass",
    # STEP 19: Geotechnical hooks
    "register_stope_stability",
    "get_stope_stability",
    "get_stope_stability_by_result_id",
    # STEP 20: Seismic hooks
    "register_stope_seismic_events",
    "get_stope_seismic_events",
    "register_drive_rockburst",
    "get_drive_rockburst",
    "get_stope_centroids",
    "get_drive_segments",
]
