"""
Liberation Model (STEP 28)

Model mineral liberation as a function of size, texture, and mineral associations.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class LiberationCurve:
    """
    Liberation curve defining liberation fraction vs. particle size.
    
    Attributes:
        size_classes: Array of size classes (microns or mm)
        liberation_fraction: Array of liberation fractions (0-1) at each size
        mineral_name: Name of the mineral (e.g., "Fe", "Cu", "gangue")
        ore_type_code: Associated ore type code
    """
    size_classes: np.ndarray  # e.g., [150, 106, 75, 53, 38]
    liberation_fraction: np.ndarray  # fraction liberated at each size
    mineral_name: str
    ore_type_code: str
    
    def __post_init__(self):
        """Validate arrays."""
        if len(self.size_classes) != len(self.liberation_fraction):
            raise ValueError("size_classes and liberation_fraction must have same length")
        if np.any(self.liberation_fraction < 0) or np.any(self.liberation_fraction > 1):
            raise ValueError("liberation_fraction must be between 0 and 1")


@dataclass
class LiberationModelConfig:
    """
    Configuration for liberation model.
    
    Attributes:
        ore_type_code: Ore type code
        mineral_name: Mineral name
        base_curve: Base liberation curve
        variability: Optional variability parameters (e.g., CV for uncertainty)
    """
    ore_type_code: str
    mineral_name: str
    base_curve: LiberationCurve
    variability: Optional[Dict[str, Any]] = None


def predict_liberation(
    size_distribution: np.ndarray,
    config: LiberationModelConfig
) -> np.ndarray:
    """
    Predict liberation fraction per size class for given ore type + mineral.
    
    Args:
        size_distribution: Array of size class fractions (must sum to 1.0)
        config: LiberationModelConfig
        
    Returns:
        Array of liberation fractions per size class
    """
    curve = config.base_curve
    
    # Interpolate liberation curve to match size distribution
    # Simple approach: find closest size class for each distribution point
    liberation_per_size = np.zeros_like(size_distribution)
    
    for i, size_frac in enumerate(size_distribution):
        if size_frac == 0:
            continue
        
        # Find closest size class in curve
        # For now, assume size_distribution indices correspond to curve indices
        # In practice, would interpolate based on actual size values
        if i < len(curve.liberation_fraction):
            liberation_per_size[i] = curve.liberation_fraction[i]
        else:
            # Extrapolate: use last value for sizes finer than curve
            liberation_per_size[i] = curve.liberation_fraction[-1]
    
    return liberation_per_size


def build_liberation_curve_from_testwork(
    feed_sizes: np.ndarray,
    liberation_measured: np.ndarray,
    ore_type_code: str,
    mineral_name: str
) -> LiberationCurve:
    """
    Build liberation curve from testwork data.
    
    Args:
        feed_sizes: Array of feed size classes (microns)
        liberation_measured: Array of measured liberation fractions
        ore_type_code: Ore type code
        mineral_name: Mineral name
        
    Returns:
        LiberationCurve object
    """
    # Sort by size (descending - coarser first)
    sort_idx = np.argsort(feed_sizes)[::-1]
    sorted_sizes = feed_sizes[sort_idx]
    sorted_liberation = liberation_measured[sort_idx]
    
    return LiberationCurve(
        size_classes=sorted_sizes,
        liberation_fraction=sorted_liberation,
        mineral_name=mineral_name,
        ore_type_code=ore_type_code
    )

