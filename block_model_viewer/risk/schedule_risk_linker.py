"""
Schedule-Risk Linker Engine

Bridges mine schedules with hazard indices to create period-by-period risk profiles.
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist

from .risk_dataclasses import PeriodRisk, ScheduleRiskProfile
from ..seismic.dataclasses import HazardVolume, RockburstIndexResult
from ..geotech.dataclasses import SlopeRiskResult

logger = logging.getLogger(__name__)


def build_period_risk_profile(
    schedule: Any,
    hazard_volume: Optional[HazardVolume],
    rockburst_results: Optional[List[RockburstIndexResult]],
    slope_results: Optional[List[SlopeRiskResult]],
    params: Dict[str, Any]
) -> ScheduleRiskProfile:
    """
    Build period-by-period risk profile from schedule and hazard data.
    
    For each period:
    1. Determine which stopes/benches/panels are active
    2. Sample relevant hazard fields
    3. Aggregate to period-level metrics
    
    Args:
        schedule: Schedule object (DataFrame with PERIOD column, or list of periods)
        hazard_volume: Optional HazardVolume with seismic hazard indices
        rockburst_results: Optional list of RockburstIndexResult
        slope_results: Optional list of SlopeRiskResult
        params: Parameters dict:
            - schedule_id: Unique schedule identifier
            - aggregation_method: 'mean', 'max', 'tonnage_weighted' (default: 'mean')
            - risk_weights: Dict with weights for combined score
            - period_days: Days per period (default: 30)
            - start_date: Start date for schedule (optional)
    
    Returns:
        ScheduleRiskProfile instance
    """
    schedule_id = params.get('schedule_id', 'schedule_1')
    aggregation_method = params.get('aggregation_method', 'mean')
    period_days = params.get('period_days', 30.0)
    start_date = params.get('start_date', datetime.now())
    
    # Extract periods from schedule
    periods_data = _extract_periods_from_schedule(schedule, params)
    
    if not periods_data:
        raise ValueError("No periods found in schedule")
    
    # Build period risk list
    period_risks = []
    
    for period_idx, period_info in enumerate(periods_data):
        # Get active locations for this period
        active_locations = period_info.get('locations', [])
        active_block_ids = period_info.get('block_ids', [])
        tonnage = period_info.get('tonnage', 0.0)
        metal = period_info.get('metal', 0.0)
        
        # Compute period times
        period_start = start_date + timedelta(days=period_idx * period_days)
        period_end = period_start + timedelta(days=period_days)
        
        # Sample hazard indices
        seismic_hazard = _sample_hazard_for_period(
            active_locations, hazard_volume, aggregation_method
        )
        
        rockburst_risk = _sample_rockburst_for_period(
            active_locations, rockburst_results, aggregation_method
        )
        
        slope_risk = _sample_slope_risk_for_period(
            active_locations, slope_results, aggregation_method
        )
        
        # Sample slope FOS and failure probability (STEP 27)
        slope_fos_min, slope_failure_prob = _sample_slope_stability_for_period(
            active_locations, params.get('slope_stability_results', []), aggregation_method
        )
        
        # Create period risk
        period_risk = PeriodRisk(
            period_index=period_idx,
            start_time=period_start,
            end_time=period_end,
            mined_tonnage=tonnage,
            metal=metal,
            seismic_hazard_index=seismic_hazard,
            rockburst_risk_index=rockburst_risk,
            slope_risk_index=slope_risk,
            slope_fos_min=slope_fos_min,
            slope_failure_probability=slope_failure_prob,
            notes=f"Period {period_idx}: {len(active_locations)} active locations"
        )
        
        # Compute combined score
        risk_weights = params.get('risk_weights', {})
        period_risk.compute_combined_score(risk_weights)
        
        period_risks.append(period_risk)
    
    # Create profile
    profile = ScheduleRiskProfile(
        schedule_id=schedule_id,
        periods=period_risks,
        metadata={
            'schedule_type': params.get('schedule_type', 'unknown'),
            'n_periods': len(period_risks),
            'aggregation_method': aggregation_method
        }
    )
    
    # Compute summary stats
    profile.compute_summary_stats()
    
    logger.info(f"Built risk profile for schedule {schedule_id}: {len(period_risks)} periods")
    
    return profile


def _extract_periods_from_schedule(schedule: Any, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract period information from schedule object.
    
    Handles different schedule formats:
    - DataFrame with PERIOD column
    - List of SchedulePeriod objects
    - Dict with period data
    
    Returns:
        List of period info dicts
    """
    import pandas as pd
    periods_data = []
    
    # Handle DataFrame schedule
    if isinstance(schedule, pd.DataFrame) and 'PERIOD' in schedule.columns:
        # Group by period
        for period_idx in sorted(schedule['PERIOD'].unique()):
            period_blocks = schedule[schedule['PERIOD'] == period_idx]
            
            # Get block IDs
            block_id_col = params.get('block_id_column', 'BLOCK_ID')
            if block_id_col in period_blocks.columns:
                block_ids = period_blocks[block_id_col].tolist()
            else:
                block_ids = period_blocks.index.tolist()
            
            # Get locations (if available)
            locations = []
            if all(col in period_blocks.columns for col in ['X', 'Y', 'Z']):
                coords = period_blocks[['X', 'Y', 'Z']].values
                locations = coords.tolist()
            elif all(col in period_blocks.columns for col in ['XC', 'YC', 'ZC']):
                coords = period_blocks[['XC', 'YC', 'ZC']].values
                locations = coords.tolist()
            
            # Get tonnage and metal
            tonnage_col = params.get('tonnage_column', 'TONNAGE')
            metal_col = params.get('metal_column', None)
            
            tonnage = period_blocks[tonnage_col].sum() if tonnage_col in period_blocks.columns else 0.0
            
            metal = 0.0
            if metal_col and metal_col in period_blocks.columns:
                metal = period_blocks[metal_col].sum()
            elif 'GRADE' in period_blocks.columns and tonnage_col in period_blocks.columns:
                # Estimate metal from grade × tonnage
                metal = (period_blocks['GRADE'] * period_blocks[tonnage_col]).sum()
            
            periods_data.append({
                'period_index': int(period_idx),
                'block_ids': block_ids,
                'locations': locations,
                'tonnage': float(tonnage),
                'metal': float(metal)
            })
    
    # Handle list of SchedulePeriod objects
    elif isinstance(schedule, list):
        for period_idx, period_obj in enumerate(schedule):
            if hasattr(period_obj, 't'):
                period_idx = period_obj.t
            
            # Extract data from period object
            tonnage = getattr(period_obj, 'ore_mined', getattr(period_obj, 'tonnage', 0.0))
            metal = getattr(period_obj, 'metal', 0.0)
            
            # Get locations if available
            locations = []
            if hasattr(period_obj, 'locations'):
                locations = period_obj.locations
            elif hasattr(period_obj, 'stope_ids'):
                # Would need to look up stope locations
                pass
            
            periods_data.append({
                'period_index': period_idx,
                'block_ids': [],
                'locations': locations,
                'tonnage': float(tonnage),
                'metal': float(metal)
            })
    
    # Handle dict format
    elif isinstance(schedule, dict):
        if 'periods' in schedule:
            periods_data = schedule['periods']
        else:
            # Single period schedule
            periods_data = [schedule]
    
    return periods_data


def _sample_hazard_for_period(
    locations: List[List[float]],
    hazard_volume: Optional[HazardVolume],
    method: str = 'mean'
) -> Optional[float]:
    """
    Sample seismic hazard index for period locations.
    
    Args:
        locations: List of [x, y, z] coordinates
        hazard_volume: HazardVolume instance
        method: Aggregation method ('mean', 'max', 'min')
    
    Returns:
        Aggregated hazard index or None
    """
    if not hazard_volume or not locations:
        return None
    
    indices = []
    
    for loc in locations:
        if len(loc) >= 3:
            hazard_val = hazard_volume.get_hazard_at_point(loc[0], loc[1], loc[2])
            indices.append(hazard_val)
    
    if not indices:
        return None
    
    if method == 'mean':
        return float(np.mean(indices))
    elif method == 'max':
        return float(np.max(indices))
    elif method == 'min':
        return float(np.min(indices))
    else:
        return float(np.mean(indices))


def _sample_rockburst_for_period(
    locations: List[List[float]],
    rockburst_results: Optional[List[RockburstIndexResult]],
    method: str = 'mean'
) -> Optional[float]:
    """
    Sample rockburst risk index for period locations.
    
    Args:
        locations: List of [x, y, z] coordinates
        rockburst_results: List of RockburstIndexResult
        method: Aggregation method
    
    Returns:
        Aggregated rockburst index or None
    """
    if not rockburst_results or not locations:
        return None
    
    # Find nearest rockburst results for each location
    indices = []
    
    for loc in locations:
        if len(loc) >= 3:
            # Find nearest rockburst result
            min_dist = float('inf')
            nearest_result = None
            
            for result in rockburst_results:
                result_loc = result.location
                dist = np.sqrt(
                    (loc[0] - result_loc[0])**2 +
                    (loc[1] - result_loc[1])**2 +
                    (loc[2] - result_loc[2])**2
                )
                if dist < min_dist:
                    min_dist = dist
                    nearest_result = result
            
            if nearest_result:
                indices.append(nearest_result.index_value)
    
    if not indices:
        return None
    
    if method == 'mean':
        return float(np.mean(indices))
    elif method == 'max':
        return float(np.max(indices))
    elif method == 'min':
        return float(np.min(indices))
    else:
        return float(np.mean(indices))


def _sample_slope_risk_for_period(
    locations: List[List[float]],
    slope_results: Optional[List[SlopeRiskResult]],
    method: str = 'mean'
) -> Optional[float]:
    """
    Sample slope risk index for period locations.
    
    Args:
        locations: List of [x, y, z] coordinates
        slope_results: List of SlopeRiskResult
        method: Aggregation method
    
    Returns:
        Aggregated slope risk index or None
    """
    if not slope_results or not locations:
        return None
    
    # For slope risk, we typically have sector-based results
    # Use nearest result or average all if locations are in same sector
    indices = []
    
    for result in slope_results:
        indices.append(result.risk_index)
    
    if not indices:
        return None
    
    if method == 'mean':
        return float(np.mean(indices))
    elif method == 'max':
        return float(np.max(indices))
    elif method == 'min':
        return float(np.min(indices))
    else:
        return float(np.mean(indices))


def _sample_slope_stability_for_period(
    locations: List[List[float]],
    slope_stability_results: List[Any],
    method: str = 'mean'
) -> tuple[Optional[float], Optional[float]]:
    """
    Sample slope FOS and failure probability for period locations (STEP 27).
    
    Args:
        locations: List of [x, y, z] coordinates
        slope_stability_results: List of LEM2DResult, LEM3DResult, or ProbSlopeResult
        method: Aggregation method
    
    Returns:
        Tuple of (min_fos, failure_probability) or (None, None)
    """
    if not slope_stability_results or not locations:
        return None, None
    
    fos_values = []
    failure_probs = []
    
    for result in slope_stability_results:
        # Extract FOS
        if hasattr(result, 'fos'):
            fos_values.append(result.fos)
        elif hasattr(result, 'fos_samples'):
            # Probabilistic result
            fos_samples = result.fos_samples
            if len(fos_samples) > 0:
                fos_values.extend(fos_samples.tolist())
                if hasattr(result, 'probability_of_failure'):
                    failure_probs.append(result.probability_of_failure)
    
    if not fos_values:
        return None, None
    
    # Compute min FOS
    min_fos = float(np.min(fos_values))
    
    # Compute failure probability
    if failure_probs:
        failure_prob = float(np.mean(failure_probs)) if method == 'mean' else float(np.max(failure_probs))
    else:
        # Estimate from FOS distribution
        failures = sum(1 for fos in fos_values if fos < 1.0)
        failure_prob = failures / len(fos_values) if fos_values else 0.0
    
    return min_fos, failure_prob


def map_slope_results_to_periods(
    schedule: Any,
    slope_results: List[Any],
    params: Dict[str, Any]
) -> List[PeriodRisk]:
    """
    Map slope stability results to schedule periods (STEP 27).
    
    Light-touch integration: combines slope results into risk profiles.
    
    Args:
        schedule: Schedule object
        slope_results: List of ProbSlopeResult, LEM2DResult, or LEM3DResult
        params: Parameters dict
    
    Returns:
        List of PeriodRisk objects with slope metrics
    """
    # Use existing build_period_risk_profile with slope_stability_results parameter
    params_with_slope = {**params, 'slope_stability_results': slope_results}
    profile = build_period_risk_profile(
        schedule,
        params_with_slope.get('hazard_volume'),
        params_with_slope.get('rockburst_results'),
        params_with_slope.get('slope_results'),
        params_with_slope
    )
    
    return profile.periods

