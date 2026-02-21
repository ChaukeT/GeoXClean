"""
Production Alignment Engine (STEP 36)

Aligns NPVS schedules, haulage capacity, and reconciliation results.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AlignedPeriodMetrics:
    """
    Aligned metrics for a single period.
    
    Attributes:
        period_id: Period identifier
        index: Period index
        planned_mined_t: Planned mined tonnes
        planned_plant_t: Planned plant feed tonnes
        planned_grade_by_element: Planned grades by element
        planned_value: Planned value
        hauled_t: Effective hauled tonnes (from haulage capacity)
        haulage_utilisation: Haulage utilisation percentage
        haulage_shortfall_t: Haulage shortfall tonnes
        mined_actual_t: Actual mined tonnes (from reconciliation)
        mill_actual_t: Actual mill feed tonnes (from reconciliation)
        grade_mine_by_element: Actual mine grades by element
        grade_mill_by_element: Actual mill grades by element
        delta_mined_t: Variance in mined tonnes (actual - planned)
        delta_mill_t: Variance in mill tonnes (actual - planned)
        delta_grade_mine: Grade bias mine vs model
        delta_grade_mill: Grade bias mill vs model
    """
    period_id: str
    index: int
    
    # Plan
    planned_mined_t: float = 0.0
    planned_plant_t: float = 0.0
    planned_grade_by_element: Dict[str, float] = field(default_factory=dict)
    planned_value: float = 0.0
    
    # Haulage
    hauled_t: float = 0.0
    haulage_utilisation: float = 0.0
    haulage_shortfall_t: float = 0.0
    
    # Actual (Reconciliation)
    mined_actual_t: float = 0.0
    mill_actual_t: float = 0.0
    grade_mine_by_element: Dict[str, float] = field(default_factory=dict)
    grade_mill_by_element: Dict[str, float] = field(default_factory=dict)
    
    # Variances
    delta_mined_t: float = 0.0
    delta_mill_t: float = 0.0
    delta_grade_mine: Dict[str, float] = field(default_factory=dict)
    delta_grade_mill: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlignedDashboardResult:
    """
    Result from production alignment.
    
    Attributes:
        periods: List of AlignedPeriodMetrics
        overall_kpis: Summary KPIs
        metadata: Additional metadata
    """
    periods: List[AlignedPeriodMetrics] = field(default_factory=list)
    overall_kpis: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def align_production_data(
    schedule_result: Any,
    haulage_eval_result: Optional[Any] = None,
    recon_result: Optional[Any] = None
) -> AlignedDashboardResult:
    """
    Align production data from schedule, haulage, and reconciliation.
    
    Args:
        schedule_result: ScheduleResult from NPVS or strategic schedule
        haulage_eval_result: HaulageEvalResult (optional)
        recon_result: Reconciliation result dict (optional)
    
    Returns:
        AlignedDashboardResult with aligned metrics
    """
    from ..mine_planning.scheduling.types import ScheduleResult
    
    # Convert schedule_result if it's a dict
    if isinstance(schedule_result, dict):
        # Reconstruct ScheduleResult from dict if needed
        schedule = schedule_result
    else:
        schedule = schedule_result
    
    periods = []
    
    # Extract schedule periods
    schedule_periods = []
    if hasattr(schedule, 'periods'):
        schedule_periods = schedule.periods
    elif isinstance(schedule, dict) and 'periods' in schedule:
        schedule_periods = schedule['periods']
    
    # Extract schedule decisions
    schedule_decisions = []
    if hasattr(schedule, 'decisions'):
        schedule_decisions = schedule.decisions
    elif isinstance(schedule, dict) and 'decisions' in schedule:
        schedule_decisions = schedule['decisions']
    
    # Aggregate planned tonnes by period
    planned_by_period = {}
    for decision in schedule_decisions:
        period_id = decision.period_id if hasattr(decision, 'period_id') else decision.get('period_id', '')
        tonnes = decision.tonnes if hasattr(decision, 'tonnes') else decision.get('tonnes', 0.0)
        destination = decision.destination if hasattr(decision, 'destination') else decision.get('destination', 'plant')
        
        if period_id not in planned_by_period:
            planned_by_period[period_id] = {
                'mined_t': 0.0,
                'plant_t': 0.0,
                'value': 0.0
            }
        
        planned_by_period[period_id]['mined_t'] += tonnes
        if destination == 'plant' or 'plant' in destination.lower():
            planned_by_period[period_id]['plant_t'] += tonnes
    
    # Extract haulage metrics by period
    haulage_by_period = {}
    if haulage_eval_result:
        if hasattr(haulage_eval_result, 'period_metrics'):
            for pm in haulage_eval_result.period_metrics:
                period_id = pm.get('period_id', '')
                haulage_by_period[period_id] = {
                    'hauled_t': pm.get('effective_tonnes', pm.get('scheduled_tonnes', 0.0)),
                    'utilisation': pm.get('utilisation_pct', 0.0),
                    'shortfall_t': pm.get('shortfall_tonnes', 0.0)
                }
        elif isinstance(haulage_eval_result, dict) and 'period_metrics' in haulage_eval_result:
            for pm in haulage_eval_result['period_metrics']:
                period_id = pm.get('period_id', '')
                haulage_by_period[period_id] = {
                    'hauled_t': pm.get('effective_tonnes', pm.get('scheduled_tonnes', 0.0)),
                    'utilisation': pm.get('utilisation_pct', 0.0),
                    'shortfall_t': pm.get('shortfall_tonnes', 0.0)
                }
    
    # Extract reconciliation metrics by period
    recon_by_period = {}
    if recon_result:
        # Reconciliation structure varies; try common patterns
        if isinstance(recon_result, dict):
            # Try to find period-based reconciliation data
            if 'mined_series' in recon_result:
                mined_series = recon_result['mined_series']
                if isinstance(mined_series, dict) and 'records' in mined_series:
                    for record in mined_series['records']:
                        period_id = record.get('period_id', '')
                        if period_id not in recon_by_period:
                            recon_by_period[period_id] = {
                                'mined_t': 0.0,
                                'mill_t': 0.0,
                                'grade_mine': {},
                                'grade_mill': {}
                            }
                        recon_by_period[period_id]['mined_t'] += record.get('tonnes', 0.0)
                        grades = record.get('grades', {})
                        for element, grade in grades.items():
                            if element not in recon_by_period[period_id]['grade_mine']:
                                recon_by_period[period_id]['grade_mine'][element] = []
                            recon_by_period[period_id]['grade_mine'][element].append(grade)
            
            if 'plant_series' in recon_result:
                plant_series = recon_result['plant_series']
                if isinstance(plant_series, dict) and 'records' in plant_series:
                    for record in plant_series['records']:
                        period_id = record.get('period_id', '')
                        if period_id not in recon_by_period:
                            recon_by_period[period_id] = {
                                'mined_t': 0.0,
                                'mill_t': 0.0,
                                'grade_mine': {},
                                'grade_mill': {}
                            }
                        recon_by_period[period_id]['mill_t'] += record.get('tonnes', 0.0)
                        grades = record.get('grades', {})
                        for element, grade in grades.items():
                            if element not in recon_by_period[period_id]['grade_mill']:
                                recon_by_period[period_id]['grade_mill'][element] = []
                            recon_by_period[period_id]['grade_mill'][element].append(grade)
    
    # Build aligned periods
    aligned_periods = []
    for idx, period in enumerate(schedule_periods):
        period_id = period.id if hasattr(period, 'id') else period.get('id', f'P{idx}')
        period_index = period.index if hasattr(period, 'index') else period.get('index', idx)
        
        planned = planned_by_period.get(period_id, {})
        haulage = haulage_by_period.get(period_id, {})
        recon = recon_by_period.get(period_id, {})
        
        # Calculate average grades from reconciliation
        grade_mine = {}
        grade_mill = {}
        if recon.get('grade_mine'):
            for element, grades in recon['grade_mine'].items():
                if grades:
                    grade_mine[element] = np.mean(grades)
        if recon.get('grade_mill'):
            for element, grades in recon['grade_mill'].items():
                if grades:
                    grade_mill[element] = np.mean(grades)
        
        # Calculate variances
        planned_mined = planned.get('mined_t', 0.0)
        planned_plant = planned.get('plant_t', 0.0)
        mined_actual = recon.get('mined_t', 0.0)
        mill_actual = recon.get('mill_t', 0.0)
        
        delta_mined = mined_actual - planned_mined
        delta_mill = mill_actual - planned_plant
        
        # Calculate grade biases
        delta_grade_mine = {}
        delta_grade_mill = {}
        # Would need planned grades from schedule; simplified for now
        
        aligned_periods.append(AlignedPeriodMetrics(
            period_id=period_id,
            index=period_index,
            planned_mined_t=planned_mined,
            planned_plant_t=planned_plant,
            planned_grade_by_element={},  # Would extract from schedule if available
            planned_value=planned.get('value', 0.0),
            hauled_t=haulage.get('hauled_t', planned_mined),
            haulage_utilisation=haulage.get('utilisation', 0.0),
            haulage_shortfall_t=haulage.get('shortfall_t', 0.0),
            mined_actual_t=mined_actual,
            mill_actual_t=mill_actual,
            grade_mine_by_element=grade_mine,
            grade_mill_by_element=grade_mill,
            delta_mined_t=delta_mined,
            delta_mill_t=delta_mill,
            delta_grade_mine=delta_grade_mine,
            delta_grade_mill=delta_grade_mill
        ))
    
    # Calculate overall KPIs
    total_planned_mined = sum(p.planned_mined_t for p in aligned_periods)
    total_planned_mill = sum(p.planned_plant_t for p in aligned_periods)
    total_mined_actual = sum(p.mined_actual_t for p in aligned_periods)
    total_mill_actual = sum(p.mill_actual_t for p in aligned_periods)
    
    avg_haulage_utilisation = np.mean([p.haulage_utilisation for p in aligned_periods]) if aligned_periods else 0.0
    periods_with_shortfall = sum(1 for p in aligned_periods if p.haulage_shortfall_t > 0)
    
    overall_kpis = {
        'total_planned_mined_t': total_planned_mined,
        'total_planned_mill_t': total_planned_mill,
        'total_mined_actual_t': total_mined_actual,
        'total_mill_actual_t': total_mill_actual,
        'mined_variance_pct': ((total_mined_actual - total_planned_mined) / total_planned_mined * 100) if total_planned_mined > 0 else 0.0,
        'mill_variance_pct': ((total_mill_actual - total_planned_mill) / total_planned_mill * 100) if total_planned_mill > 0 else 0.0,
        'avg_haulage_utilisation': avg_haulage_utilisation,
        'periods_with_haulage_shortfall': periods_with_shortfall,
        'total_periods': len(aligned_periods)
    }
    
    # Extract NPV if available
    if hasattr(schedule, 'metadata') and isinstance(schedule.metadata, dict):
        npv = schedule.metadata.get('npv', 0.0)
        if npv:
            overall_kpis['npv'] = npv
    elif isinstance(schedule, dict) and 'metadata' in schedule:
        npv = schedule['metadata'].get('npv', 0.0)
        if npv:
            overall_kpis['npv'] = npv
    
    logger.info(f"Production alignment complete: {len(aligned_periods)} periods aligned")
    
    return AlignedDashboardResult(
        periods=aligned_periods,
        overall_kpis=overall_kpis,
        metadata={
            'schedule_source': 'npvs' if 'npvs' in str(type(schedule_result)).lower() else 'strategic',
            'has_haulage': haulage_eval_result is not None,
            'has_reconciliation': recon_result is not None
        }
    )

