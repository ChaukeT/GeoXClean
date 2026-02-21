"""
Risk Scenario Comparison Engine

Compare multiple schedule risk profiles side by side.
"""

import logging
from typing import Dict, Any, List

from .risk_dataclasses import ScheduleRiskProfile, RiskScenarioComparison

logger = logging.getLogger(__name__)


def compare_risk_scenarios(
    base: ScheduleRiskProfile,
    alternatives: List[ScheduleRiskProfile],
    metric: str = "combined_risk_score"
) -> RiskScenarioComparison:
    """
    Compare multiple schedule risk profiles.
    
    Computes:
    - Delta total exposure
    - Max period risk vs base
    - Period-by-period ratio
    - Risk-adjusted NPV proxy (if available)
    
    Args:
        base: Base schedule risk profile
        alternatives: List of alternative schedule profiles
        metric: Metric to compare ('combined_risk_score', etc.)
    
    Returns:
        RiskScenarioComparison instance
    """
    logger.info(f"Comparing {len(alternatives)} alternative scenarios against base")
    
    # Ensure summary stats are computed
    if not base.summary_stats:
        base.compute_summary_stats()
    
    for alt in alternatives:
        if not alt.summary_stats:
            alt.compute_summary_stats()
    
    # Create comparison object
    comparison = RiskScenarioComparison(
        base_profile=base,
        alternative_profiles=alternatives
    )
    
    # Compute comparison metrics
    comparison.compute_comparison_metrics()
    
    # Add additional metrics
    _add_risk_adjusted_metrics(comparison, metric)
    
    logger.info(f"Comparison complete: {len(comparison.metrics)} alternative scenarios analyzed")
    
    return comparison


def _add_risk_adjusted_metrics(
    comparison: RiskScenarioComparison,
    metric: str
) -> None:
    """
    Add risk-adjusted metrics to comparison.
    
    Computes simple risk-adjusted NPV proxy if period-level NPV/cashflow is available.
    """
    base = comparison.base_profile
    
    # Check if we have period-level financial data
    base_has_npv = any(
        hasattr(p, 'npv') or hasattr(p, 'cashflow') or 'npv' in p.metadata
        for p in base.periods
    )
    
    if not base_has_npv:
        return
    
    # Compute risk-adjusted metrics for each alternative
    for alt_id, alt_metrics in comparison.metrics.items():
        # Find corresponding alternative profile
        alt_profile = None
        for alt in comparison.alternative_profiles:
            if alt.schedule_id == alt_id:
                alt_profile = alt
                break
        
        if not alt_profile:
            continue
        
        # Simple risk adjustment: multiply period NPV by (1 - risk_score)
        # This is a proxy, not a true risk-adjusted NPV
        risk_adjusted_npv_base = 0.0
        risk_adjusted_npv_alt = 0.0
        
        for period_idx in range(max(len(base.periods), len(alt_profile.periods))):
            base_period = base.get_period(period_idx)
            alt_period = alt_profile.get_period(period_idx)
            
            # Get period NPV (if available)
            base_npv = 0.0
            alt_npv = 0.0
            
            if base_period:
                base_npv = getattr(base_period, 'npv', getattr(base_period, 'cashflow', 0.0))
                base_risk = base_period.combined_risk_score or 0.0
                risk_adjusted_npv_base += base_npv * (1.0 - base_risk)
            
            if alt_period:
                alt_npv = getattr(alt_period, 'npv', getattr(alt_period, 'cashflow', 0.0))
                alt_risk = alt_period.combined_risk_score or 0.0
                risk_adjusted_npv_alt += alt_npv * (1.0 - alt_risk)
        
        # Add to metrics
        alt_metrics['risk_adjusted_npv'] = {
            'base': risk_adjusted_npv_base,
            'alternative': risk_adjusted_npv_alt,
            'delta': risk_adjusted_npv_alt - risk_adjusted_npv_base
        }

