"""
Cutoff Optimiser (STEP 35)

Datamine-style cutoff optimisation for NPV maximisation.

Includes:
- Core cutoff engine for pattern generation and NPV optimization
- Geostatistical grade-tonnage analysis with proper tonnage anchoring
- Advanced analysis for JORC/SAMREC compliance:
  - SGS-based uncertainty quantification
  - Domain-wise GT analysis
  - Classification-based GT curves (Measured/Indicated/Inferred)
- Multi-period mine economics with proper DCF analysis
"""

from .cutoff_engine import (
    CutoffPattern,
    CutoffOptimiserConfig,
    CutoffOptimiserResult,
    generate_cutoff_patterns,
    evaluate_cutoff_pattern,
    optimise_cutoff_schedule
)

from .geostats_grade_tonnage import (
    GeostatsGradeTonnageEngine,
    CutoffSensitivityEngine,
    GeostatsGradeTonnageConfig,
    GradeTonnageCurve,
    GradeTonnagePoint,
    CutoffSensitivityAnalysis,
    DataMode,
    GradeWeightingMethod,
    CutoffOptimizationMethod,
    ConfidenceIntervalMethod,
    validate_grade_tonnage_data,
    export_grade_tonnage_results
)

from .gt_advanced_analysis import (
    SGSUncertaintyEngine,
    DomainGTEngine,
    ClassificationGTEngine,
    SGSUncertaintyResult,
    DomainGTResult,
    ClassificationGTResult,
    GTCurveStatistics,
    ResourceCategory,
    export_sgs_uncertainty_to_csv,
    export_domain_gt_to_csv,
    export_classification_gt_to_csv
)

from .mine_economics import (
    MineEconomicsEngine,
    MineEconomicsConfig,
    MineEconomicsResult,
    EconomicParameters,
    MineCapacity,
    CapitalExpenditure,
    TaxParameters,
    AnnualCashFlow,
    SensitivityAnalyzer,
    MiningMethod,
    ProcessingRoute,
    export_cash_flows_to_csv
)

__all__ = [
    # Core cutoff engine
    "CutoffPattern",
    "CutoffOptimiserConfig",
    "CutoffOptimiserResult",
    "generate_cutoff_patterns",
    "evaluate_cutoff_pattern",
    "optimise_cutoff_schedule",
    
    # Grade-tonnage engine
    "GeostatsGradeTonnageEngine",
    "CutoffSensitivityEngine",
    "GeostatsGradeTonnageConfig",
    "GradeTonnageCurve",
    "GradeTonnagePoint",
    "CutoffSensitivityAnalysis",
    "DataMode",
    "GradeWeightingMethod",
    "CutoffOptimizationMethod",
    "ConfidenceIntervalMethod",
    "validate_grade_tonnage_data",
    "export_grade_tonnage_results",
    
    # Advanced analysis (JORC/SAMREC)
    "SGSUncertaintyEngine",
    "DomainGTEngine",
    "ClassificationGTEngine",
    "SGSUncertaintyResult",
    "DomainGTResult",
    "ClassificationGTResult",
    "GTCurveStatistics",
    "ResourceCategory",
    "export_sgs_uncertainty_to_csv",
    "export_domain_gt_to_csv",
    "export_classification_gt_to_csv",
    
    # Mine economics (multi-period DCF)
    "MineEconomicsEngine",
    "MineEconomicsConfig",
    "MineEconomicsResult",
    "EconomicParameters",
    "MineCapacity",
    "CapitalExpenditure",
    "TaxParameters",
    "AnnualCashFlow",
    "SensitivityAnalyzer",
    "MiningMethod",
    "ProcessingRoute",
    "export_cash_flows_to_csv",
]

