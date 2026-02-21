"""
IRR Optimization Engine for Mining Projects

This package provides tools for computing risk-adjusted Internal Rate of Return (IRR)
for mining projects using stochastic scenario simulation and MILP optimization.

Key Enhancement (2025-12):
- DynamicPitShellSelector: Enables price-responsive pit boundaries in IRR scenarios.
  For each price scenario, the algorithm selects the optimal pit shell from pre-computed
  nested shells, ensuring rational pit sizing rather than mining a fixed pit at all prices.

Audit Compliance (2025-12):
- Validation module: Classification filtering, unit enforcement, parameter validation
- Provenance module: Hashing, timestamping, reproducibility tracking
- Multiple IRR detection: Sign change analysis for cash flows
- Precedence validation: Schedule integrity checks
"""

from .npv_calc import (
    calculate_npv, 
    discount_cashflows, 
    calculate_irr,
    calculate_irr_with_metadata,
    detect_multiple_irr,
    count_sign_changes
)
from .scenario_generator import ScenarioGenerator, ScenarioConfig
from .irr_bisection import find_irr_alpha
from .milp_optimizer import MiningScheduleOptimizer
from .results_model import IRRResult, build_irr_result
from .lerchs_grossmann import LerchsGrossmann
from .config_loader import IRRConfig, PitConfig, ScheduleConfig, load_config, save_config, get_default_config
from .engine_api import run_irr, run_npv, run_pit_optimisation, run_schedule_optimisation
from .dynamic_shell_selector import (
    DynamicPitShellSelector, 
    PitShellData, 
    ShellSelectionResult,
    create_shell_selector_from_lg_result
)

# Audit Compliance Modules (2025-12)
from .validation import (
    validate_economic_params,
    validate_and_convert_units,
    apply_classification_filter,
    validate_irr_inputs,
    detect_multiple_irr_risk,
    validate_irr_result,
    EconomicParameterError,
    UnitConfig,
    PriceUnit,
    CostUnit,
    ClassificationFilterResult,
    IRRValidityResult,
    REQUIRED_ECONOMIC_PARAMS,
    DEFAULT_IRR_CLASSIFICATIONS
)
from .provenance import (
    create_provenance,
    compute_block_model_hash,
    compute_config_hash,
    compute_economic_params_hash,
    ensure_deterministic_seed,
    export_provenance_report,
    IRRProvenance,
    RandomSeedManager,
    IRR_ENGINE_VERSION
)
from .fast_scheduler import (
    FastScheduler,
    build_vertical_precedence,
    validate_schedule_precedence,
    PrecedenceViolation,
    ScheduleValidationResult
)

__all__ = [
    # NPV/IRR Calculations
    'calculate_npv',
    'discount_cashflows',
    'calculate_irr',
    'calculate_irr_with_metadata',
    'detect_multiple_irr',
    'count_sign_changes',
    
    # Scenario Generation
    'ScenarioGenerator',
    'ScenarioConfig',
    
    # Core IRR Analysis
    'find_irr_alpha',
    'MiningScheduleOptimizer',
    'IRRResult',
    'build_irr_result',
    
    # Pit Optimization
    'LerchsGrossmann',
    
    # Configuration
    'IRRConfig',
    'PitConfig',
    'ScheduleConfig',
    'load_config',
    'save_config',
    'get_default_config',
    
    # API Functions
    'run_irr',
    'run_npv',
    'run_pit_optimisation',
    'run_schedule_optimisation',
    
    # Dynamic Pit Shell Selector (2025-12)
    'DynamicPitShellSelector',
    'PitShellData',
    'ShellSelectionResult',
    'create_shell_selector_from_lg_result',
    
    # === AUDIT COMPLIANCE EXPORTS (2025-12) ===
    
    # Validation
    'validate_economic_params',
    'validate_and_convert_units',
    'apply_classification_filter',
    'validate_irr_inputs',
    'detect_multiple_irr_risk',
    'validate_irr_result',
    'EconomicParameterError',
    'UnitConfig',
    'PriceUnit',
    'CostUnit',
    'ClassificationFilterResult',
    'IRRValidityResult',
    'REQUIRED_ECONOMIC_PARAMS',
    'DEFAULT_IRR_CLASSIFICATIONS',
    
    # Provenance
    'create_provenance',
    'compute_block_model_hash',
    'compute_config_hash',
    'compute_economic_params_hash',
    'ensure_deterministic_seed',
    'export_provenance_report',
    'IRRProvenance',
    'RandomSeedManager',
    'IRR_ENGINE_VERSION',
    
    # Schedule Validation
    'FastScheduler',
    'build_vertical_precedence',
    'validate_schedule_precedence',
    'PrecedenceViolation',
    'ScheduleValidationResult'
]

