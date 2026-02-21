"""
Research Mode Module
====================

Provides experiment definition, execution, metrics collection, and reporting
for reproducible geostatistical and planning research.

Also includes professional analysis tools for validation and auditing.
"""

from .experiment_definitions import (
    ExperimentParameter,
    ExperimentDefinition,
    ExperimentInstance,
    ScenarioGrid
)

from .metrics import (
    compute_metrics,
    rmse_cv,
    mae_cv,
    smoothing_index,
    nugget_to_sill_ratio,
    grade_tonnage_loss,
    npv_stat,
    irr_stat,
    risk_adjusted_npv,
    schedule_exposure,
    risk_percentile
)

from .runner import (
    ExperimentRunner,
    ExperimentRunResult
)

from .reporting import (
    experiment_to_dataframe,
    to_excel,
    to_latex_table
)

from .analysis_tools import (
    compute_kriging_summary_table,
    compute_slope_of_regression,
    leave_one_out_cross_validation,
    compute_swath_plot_data,
    compute_simulation_reproduction_stats,
    compute_uncertainty_grids
)

__all__ = [
    # Definitions
    'ExperimentParameter',
    'ExperimentDefinition',
    'ExperimentInstance',
    'ScenarioGrid',
    # Metrics
    'compute_metrics',
    'rmse_cv',
    'mae_cv',
    'smoothing_index',
    'nugget_to_sill_ratio',
    'grade_tonnage_loss',
    'npv_stat',
    'irr_stat',
    'risk_adjusted_npv',
    'schedule_exposure',
    'risk_percentile',
    # Runner
    'ExperimentRunner',
    'ExperimentRunResult',
    # Reporting
    'experiment_to_dataframe',
    'to_excel',
    'to_latex_table',
    # Analysis Tools (New)
    'compute_kriging_summary_table',
    'compute_slope_of_regression',
    'leave_one_out_cross_validation',
    'compute_swath_plot_data',
    'compute_simulation_reproduction_stats',
    'compute_uncertainty_grids'
]
