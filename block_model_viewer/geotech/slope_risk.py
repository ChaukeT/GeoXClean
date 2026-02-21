"""
Slope Risk Assessment Module.

Provides simplified slope stability risk indicators based on rock mass
properties, geometry, and environmental factors.
"""

import logging
import numpy as np
from typing import Dict, Any

from .dataclasses import SlopeRiskInput, SlopeRiskResult

logger = logging.getLogger(__name__)


def evaluate_slope(input: SlopeRiskInput) -> SlopeRiskResult:
    """
    Evaluate slope stability risk using simplified indicators.
    
    Combines:
    - Rock mass class (from RMR/Q)
    - Bench height vs recommended spacing
    - Overall slope angle vs recommended for class
    - Water/infill factors
    
    Args:
        input: SlopeRiskInput with geometry and rock mass properties
    
    Returns:
        SlopeRiskResult with risk classification
    """
    # Get rock mass properties
    rmr = input.rock_mass_properties.get('RMR', 50.0)
    q = input.rock_mass_properties.get('Q', 1.0)
    
    # Determine rock mass class
    if rmr >= 80:
        rm_class = 'Very Good'
        recommended_slope_angle = 70.0
        recommended_bench_height = 15.0
    elif rmr >= 60:
        rm_class = 'Good'
        recommended_slope_angle = 60.0
        recommended_bench_height = 12.0
    elif rmr >= 40:
        rm_class = 'Fair'
        recommended_slope_angle = 50.0
        recommended_bench_height = 10.0
    elif rmr >= 20:
        rm_class = 'Poor'
        recommended_slope_angle = 40.0
        recommended_bench_height = 8.0
    else:
        rm_class = 'Very Poor'
        recommended_slope_angle = 30.0
        recommended_bench_height = 5.0
    
    # Compute risk index (0-100, higher = more risk)
    risk_index = 0.0
    
    # Factor 1: Rock mass quality (0-40 points)
    rm_factor = (100 - rmr) / 100.0 * 40.0
    risk_index += rm_factor
    
    # Factor 2: Slope angle vs recommended (0-30 points)
    angle_excess = max(0, input.overall_slope_angle - recommended_slope_angle)
    angle_factor = min(30.0, angle_excess / 10.0 * 30.0)
    risk_index += angle_factor
    
    # Factor 3: Bench height vs recommended (0-20 points)
    height_excess = max(0, input.bench_height - recommended_bench_height)
    height_factor = min(20.0, height_excess / 5.0 * 20.0)
    risk_index += height_factor
    
    # Factor 4: Water/infill (0-10 points)
    if input.water_present:
        risk_index += 10.0
    
    # Clamp to 0-100
    risk_index = max(0.0, min(100.0, risk_index))
    
    # Determine qualitative class
    if risk_index < 25:
        qualitative_class = SlopeRiskResult.RISK_CLASSES['LOW']
        prob_failure = 0.05
        notes = f"Low risk - {rm_class} rock mass, geometry within recommended limits"
    elif risk_index < 50:
        qualitative_class = SlopeRiskResult.RISK_CLASSES['MODERATE']
        prob_failure = 0.15
        notes = f"Moderate risk - {rm_class} rock mass, some geometry concerns"
    elif risk_index < 75:
        qualitative_class = SlopeRiskResult.RISK_CLASSES['HIGH']
        prob_failure = 0.40
        notes = f"High risk - {rm_class} rock mass, geometry exceeds recommendations"
    else:
        qualitative_class = SlopeRiskResult.RISK_CLASSES['VERY_HIGH']
        prob_failure = 0.70
        notes = f"Very high risk - {rm_class} rock mass, significant geometry concerns"
    
    # Adjust probability based on structural features
    if input.structural_features.get('major_faults', False):
        prob_failure = min(0.95, prob_failure + 0.20)
        notes += "; Major faults present"
    
    if input.structural_features.get('adverse_joints', False):
        prob_failure = min(0.95, prob_failure + 0.15)
        notes += "; Adverse joint orientations"
    
    result = SlopeRiskResult(
        risk_index=risk_index,
        qualitative_class=qualitative_class,
        probability_of_failure=prob_failure,
        notes=notes
    )
    
    logger.info(f"Slope risk evaluated: Index={risk_index:.1f}, Class={qualitative_class}")
    
    return result

