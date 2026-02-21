"""
Comminution Model (STEP 28)

Translate ore hardness, grind settings, and circuit type into
size distributions and energy use.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ComminutionOreProperties:
    """
    Ore properties for comminution modeling.
    
    Attributes:
        ore_type_code: Ore type code
        A: JK A parameter (kWh/t)
        b: JK b parameter (dimensionless)
        ta: JK ta parameter (dimensionless)
        work_index_bond: Bond Work Index (kWh/t)
        abrasion_index: Abrasion index
        competency_index: Competency index
    """
    ore_type_code: str
    A: Optional[float] = None  # JK A
    b: Optional[float] = None  # JK b
    ta: Optional[float] = None
    work_index_bond: Optional[float] = None
    abrasion_index: Optional[float] = None
    competency_index: Optional[float] = None


@dataclass
class ComminutionCircuitConfig:
    """
    Comminution circuit configuration.
    
    Attributes:
        circuit_type: Circuit type (e.g., "SAG-Ball", "HPGR-Ball", "Ball only")
        target_p80: Target P80 size (microns)
        f80: Feed size F80 (mm)
        specific_energy_limit: Maximum specific energy (kWh/t)
        plant_throughput_limit: Maximum throughput (t/h)
    """
    circuit_type: str
    target_p80: float  # microns
    f80: Optional[float] = None  # mm
    specific_energy_limit: Optional[float] = None  # kWh/t
    plant_throughput_limit: Optional[float] = None  # t/h


@dataclass
class ComminutionResult:
    """
    Result from comminution prediction.
    
    Attributes:
        size_classes: Array of size classes (microns)
        size_distribution: Array of size fractions (sums to 1.0)
        specific_energy: Specific energy consumption (kWh/t)
        throughput: Throughput (t/h)
    """
    size_classes: np.ndarray
    size_distribution: np.ndarray
    specific_energy: float
    throughput: float


def estimate_work_index_from_ucs(ucs_mpa: float) -> float:
    """
    Estimate Bond Work Index from Uniaxial Compressive Strength (STEP 27/28 integration).
    
    Uses empirical correlation: Wi ≈ 0.1 * UCS^0.5 (simplified)
    More accurate correlations exist but this provides a reasonable first estimate.
    
    Args:
        ucs_mpa: Uniaxial compressive strength (MPa)
        
    Returns:
        Estimated Bond Work Index (kWh/t)
    """
    # Simplified correlation - typical range: 5-25 kWh/t for most ores
    # UCS range: 20-300 MPa typical
    if ucs_mpa <= 0:
        return 10.0  # Default moderate hardness
    
    # Empirical correlation: Wi ≈ 0.1 * sqrt(UCS)
    # More sophisticated: Wi = 0.316 * UCS^0.5 (for some rock types)
    work_index = 0.1 * np.sqrt(ucs_mpa)
    
    # Clamp to reasonable range
    work_index = max(5.0, min(25.0, work_index))
    
    return work_index


def estimate_comminution_from_geotech(
    ucs_mpa: Optional[float] = None,
    rmr: Optional[float] = None,
    ore_type_code: str = "UNKNOWN"
) -> ComminutionOreProperties:
    """
    Estimate comminution properties from geotechnical data (STEP 27/28 integration).
    
    Args:
        ucs_mpa: Uniaxial compressive strength (MPa)
        rmr: Rock Mass Rating (0-100)
        ore_type_code: Ore type code
        
    Returns:
        ComminutionOreProperties with estimated values
    """
    work_index = None
    
    if ucs_mpa is not None:
        work_index = estimate_work_index_from_ucs(ucs_mpa)
    elif rmr is not None:
        # Estimate UCS from RMR (simplified correlation)
        # RMR 80+ → UCS ~150-200 MPa
        # RMR 60-80 → UCS ~80-150 MPa
        # RMR 40-60 → UCS ~40-80 MPa
        # RMR <40 → UCS ~20-40 MPa
        if rmr >= 80:
            estimated_ucs = 175.0
        elif rmr >= 60:
            estimated_ucs = 115.0
        elif rmr >= 40:
            estimated_ucs = 60.0
        else:
            estimated_ucs = 30.0
        
        work_index = estimate_work_index_from_ucs(estimated_ucs)
    
    return ComminutionOreProperties(
        ore_type_code=ore_type_code,
        work_index_bond=work_index
    )


def predict_comminution_response(
    ore_props: ComminutionOreProperties,
    circuit: ComminutionCircuitConfig
) -> ComminutionResult:
    """
    Predict comminution response for given ore properties and circuit.
    
    Args:
        ore_props: ComminutionOreProperties
        circuit: ComminutionCircuitConfig
        
    Returns:
        ComminutionResult
    """
    # Simple Bond-based approach if Work Index available
    if ore_props.work_index_bond is not None and circuit.f80 is not None:
        # Bond's law: E = Wi * (10/sqrt(P80) - 10/sqrt(F80))
        # F80 and P80 in microns
        f80_microns = circuit.f80 * 1000.0  # Convert mm to microns
        p80_microns = circuit.target_p80
        
        specific_energy = ore_props.work_index_bond * (
            10.0 / np.sqrt(p80_microns) - 10.0 / np.sqrt(f80_microns)
        )
    else:
        # Fallback: use default energy estimate
        # Typical range: 10-25 kWh/t for most ores
        specific_energy = 15.0
    
    # Apply energy limit if specified
    if circuit.specific_energy_limit is not None:
        specific_energy = min(specific_energy, circuit.specific_energy_limit)
    
    # Estimate throughput (simplified)
    if circuit.plant_throughput_limit is not None:
        throughput = circuit.plant_throughput_limit
    else:
        # Default throughput estimate
        throughput = 500.0  # t/h
    
    # Generate size distribution (simplified Rosin-Rammler approximation)
    # P80 target used to generate distribution
    size_classes = np.array([150, 106, 75, 53, 38, 25, 19, 13, 9.5, 6.7, 4.75]) * 1000  # microns
    p80 = circuit.target_p80
    
    # Rosin-Rammler parameter (n) - higher n = narrower distribution
    n = 1.2  # Typical value
    
    # Calculate cumulative passing
    cumulative_passing = 1.0 - np.exp(-(size_classes / p80) ** n)
    
    # Convert to size fractions
    size_distribution = np.diff(np.concatenate(([0.0], cumulative_passing)))
    size_distribution = np.maximum(size_distribution, 0.0)  # Ensure non-negative
    size_distribution = size_distribution / np.sum(size_distribution)  # Normalize
    
    return ComminutionResult(
        size_classes=size_classes,
        size_distribution=size_distribution,
        specific_energy=specific_energy,
        throughput=throughput
    )
