"""
Time-Evolving Hazard & Schedule-Linked Risk Module

Provides period-by-period risk analysis integrated with mine schedules,
enabling scenario comparison and risk timeline visualization.
"""

from .risk_dataclasses import (
    PeriodRisk,
    ScheduleRiskProfile,
    RiskScenarioComparison
)

from .schedule_risk_linker import build_period_risk_profile

from .risk_timeline import (
    build_risk_time_series,
    compute_exposure_curve
)

from .risk_scenarios import compare_risk_scenarios

__all__ = [
    'PeriodRisk',
    'ScheduleRiskProfile',
    'RiskScenarioComparison',
    'build_period_risk_profile',
    'build_risk_time_series',
    'compute_exposure_curve',
    'compare_risk_scenarios',
]

