"""
Rose Diagram - Compute histogram bins for orientation data.
"""

from typing import List, Tuple
import numpy as np

from .datasets import PlaneMeasurement, LineationMeasurement


def compute_rose_bins(
    measurements: List[PlaneMeasurement],
    n_bins: int = 36,
    use_dip_direction: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram bins for rose diagram.
    
    Args:
        measurements: List of PlaneMeasurement objects
        n_bins: Number of bins (default 36 for 10-degree bins)
        use_dip_direction: If True, use dip_direction; if False, use strike
    
    Returns:
        Tuple of (bin_centers, bin_counts)
    """
    if not measurements:
        return np.array([]), np.array([])
    
    # Extract orientations
    if use_dip_direction:
        orientations = [m.dip_direction for m in measurements]
    else:
        # Convert dip_direction to strike (strike = dip_direction - 90)
        orientations = [(m.dip_direction - 90) % 360 for m in measurements]
    
    # Create bins
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Count in bins
    counts, _ = np.histogram(orientations, bins=bin_edges)
    
    return bin_centers, counts


def compute_lineation_rose_bins(
    lineations: List[LineationMeasurement],
    n_bins: int = 36
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute histogram bins for lineation trend rose diagram.
    
    Args:
        lineations: List of LineationMeasurement objects
        n_bins: Number of bins
    
    Returns:
        Tuple of (bin_centers, bin_counts)
    """
    if not lineations:
        return np.array([]), np.array([])
    
    trends = [l.trend for l in lineations]
    
    bin_edges = np.linspace(0, 360, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    counts, _ = np.histogram(trends, bins=bin_edges)
    
    return bin_centers, counts

