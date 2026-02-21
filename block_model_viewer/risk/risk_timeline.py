"""
Risk Timeline Engine

Build time-indexed risk curves from schedule risk profiles.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from .risk_dataclasses import ScheduleRiskProfile

logger = logging.getLogger(__name__)


def build_risk_time_series(
    profile: ScheduleRiskProfile,
    metric: str = "combined_risk_score",
    use_time: bool = False
) -> List[Tuple[float, float]]:
    """
    Build time-indexed risk curve from schedule risk profile.
    
    Args:
        profile: ScheduleRiskProfile instance
        metric: Metric to plot ('combined_risk_score', 'seismic_hazard_index', etc.)
        use_time: If True, use actual datetime; if False, use period index
    
    Returns:
        List of (time_coord, metric_value) tuples
    """
    time_series = []
    
    for period in profile.periods:
        # Get time coordinate
        if use_time and period.start_time:
            # Use days since start
            if profile.periods[0].start_time:
                time_coord = (period.start_time - profile.periods[0].start_time).total_seconds() / 86400.0
            else:
                time_coord = float(period.period_index)
        else:
            time_coord = float(period.period_index)
        
        # Get metric value
        if metric == "combined_risk_score":
            value = period.combined_risk_score
        elif metric == "seismic_hazard_index":
            value = period.seismic_hazard_index
        elif metric == "rockburst_risk_index":
            value = period.rockburst_risk_index
        elif metric == "slope_risk_index":
            value = period.slope_risk_index
        else:
            value = getattr(period, metric, None)
        
        if value is not None:
            time_series.append((time_coord, float(value)))
    
    logger.info(f"Built time series: {len(time_series)} points for metric '{metric}'")
    
    return time_series


def compute_exposure_curve(
    profile: ScheduleRiskProfile,
    threshold: float,
    metric: str = "combined_risk_score"
) -> Dict[str, Any]:
    """
    Compute exposure curve statistics.
    
    Args:
        profile: ScheduleRiskProfile instance
        threshold: Risk threshold value
        metric: Metric to evaluate
    
    Returns:
        Dict with exposure statistics:
        - total_periods_above_threshold
        - pct_periods_above_threshold
        - total_tonnage_high_risk
        - pct_tonnage_high_risk
        - max_exposure_period
        - cumulative_exposure
    """
    periods_above = 0
    tonnage_above = 0.0
    total_tonnage = 0.0
    max_exposure = 0.0
    max_exposure_period = -1
    
    for period in profile.periods:
        # Get metric value
        if metric == "combined_risk_score":
            value = period.combined_risk_score
        elif metric == "seismic_hazard_index":
            value = period.seismic_hazard_index
        elif metric == "rockburst_risk_index":
            value = period.rockburst_risk_index
        elif metric == "slope_risk_index":
            value = period.slope_risk_index
        else:
            value = getattr(period, metric, None)
        
        total_tonnage += period.mined_tonnage
        
        if value is not None and value >= threshold:
            periods_above += 1
            tonnage_above += period.mined_tonnage
            
            if value > max_exposure:
                max_exposure = value
                max_exposure_period = period.period_index
    
    # Compute cumulative exposure (sum of risk × tonnage)
    cumulative_exposure = 0.0
    for period in profile.periods:
        if metric == "combined_risk_score":
            value = period.combined_risk_score
        else:
            value = getattr(period, metric, None)
        
        if value is not None:
            cumulative_exposure += value * period.mined_tonnage
    
    stats = {
        'total_periods_above_threshold': periods_above,
        'pct_periods_above_threshold': (periods_above / len(profile.periods) * 100) if profile.periods else 0.0,
        'total_tonnage_high_risk': tonnage_above,
        'pct_tonnage_high_risk': (tonnage_above / total_tonnage * 100) if total_tonnage > 0 else 0.0,
        'max_exposure_period': max_exposure_period,
        'max_exposure_value': max_exposure,
        'cumulative_exposure': cumulative_exposure,
        'threshold': threshold,
        'metric': metric
    }
    
    logger.info(f"Computed exposure curve: {periods_above}/{len(profile.periods)} periods above threshold")
    
    return stats

