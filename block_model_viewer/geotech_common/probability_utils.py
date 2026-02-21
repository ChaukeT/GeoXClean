"""
Probability Utilities - Helpers for probabilistic geotechnical calculations.
"""

from typing import List, Dict
import numpy as np
import logging

from .material_properties import GeotechMaterial

logger = logging.getLogger(__name__)


def sample_material_properties(
    material: GeotechMaterial,
    n: int,
    cov: Dict[str, float]
) -> List[GeotechMaterial]:
    """
    Sample material properties from distributions.
    
    Args:
        material: Base material properties
        n: Number of samples
        cov: Coefficient of variation dictionary (e.g., {"friction_angle": 0.1, "cohesion": 0.15})
    
    Returns:
        List of GeotechMaterial samples
    """
    samples = []
    
    # Default COV if not provided
    default_cov = {
        "friction_angle": 0.1,
        "cohesion": 0.15,
        "unit_weight": 0.05
    }
    cov = {**default_cov, **cov}
    
    # Sample friction angle (normal distribution, truncated at 0)
    phi_mean = material.friction_angle
    phi_std = phi_mean * cov.get("friction_angle", 0.1)
    phi_samples = np.maximum(0, np.random.normal(phi_mean, phi_std, n))
    
    # Sample cohesion (normal distribution, truncated at 0)
    c_mean = material.cohesion
    c_std = c_mean * cov.get("cohesion", 0.15)
    c_samples = np.maximum(0, np.random.normal(c_mean, c_std, n))
    
    # Sample unit weight (normal distribution)
    gamma_mean = material.unit_weight
    gamma_std = gamma_mean * cov.get("unit_weight", 0.05)
    gamma_samples = np.maximum(0, np.random.normal(gamma_mean, gamma_std, n))
    
    # Create samples
    for i in range(n):
        sample = GeotechMaterial(
            name=f"{material.name}_sample_{i}",
            unit_weight=float(gamma_samples[i]),
            friction_angle=float(phi_samples[i]),
            cohesion=float(c_samples[i]),
            tensile_strength=material.tensile_strength,
            hoek_brown_mb=material.hoek_brown_mb,
            hoek_brown_s=material.hoek_brown_s,
            hoek_brown_a=material.hoek_brown_a,
            water_condition=material.water_condition,
            notes=f"Sample {i} of {material.name}",
            metadata={**material.metadata, "sample_index": i}
        )
        samples.append(sample)
    
    return samples


def probability_of_failure(fos_samples: np.ndarray) -> float:
    """
    Compute probability of failure from FOS samples.
    
    Args:
        fos_samples: Array of Factor of Safety values
    
    Returns:
        Probability of failure (P(FOS < 1.0))
    """
    if len(fos_samples) == 0:
        return 0.0
    
    failures = np.sum(fos_samples < 1.0)
    return float(failures / len(fos_samples))


def compute_fos_statistics(fos_samples: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for FOS samples.
    
    Args:
        fos_samples: Array of Factor of Safety values
    
    Returns:
        Dictionary with statistics
    """
    if len(fos_samples) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p5": 0.0,
            "p95": 0.0,
            "probability_of_failure": 0.0
        }
    
    return {
        "mean": float(np.mean(fos_samples)),
        "std": float(np.std(fos_samples)),
        "min": float(np.min(fos_samples)),
        "max": float(np.max(fos_samples)),
        "p5": float(np.percentile(fos_samples, 5)),
        "p95": float(np.percentile(fos_samples, 95)),
        "probability_of_failure": probability_of_failure(fos_samples)
    }

