"""
Geostatistical Grade-Tonnage Curves and Cut-off Sensitivity Analysis Engine.

This module provides grade-tonnage analysis with:
- Dual mode support: BLOCK_MODEL (no declustering) vs COMPOSITES (declustered)
- Proper tonnage anchoring to deposit totals
- Numba JIT optimization for performance
- Cut-off sensitivity analysis with NPV optimization
- Heuristic uncertainty bands (CV-based, not variogram-derived)

Key Features:
- Block model mode: Direct GT curves from kriged block models (no declustering)
- Composites mode: Cell-based declustering for drillhole composites
- Tonnage anchored to actual deposit totals (not scaled by weights)
- Grades properly declustered where applicable
- Economic optimization with multiple cut-off strategies
- Diagnostic grade-tonnage curves for desktop analysis

NOTE ON UNCERTAINTY:
The confidence bands shown are heuristic ±CV bands on tonnage, NOT formal 
geostatistical confidence intervals. For proper uncertainty quantification,
use multiple SGS realisations or kriging variance integration.

Author: GeoX Mining Software - Geostats Grade-Tonnage Engine
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import warnings

# Try to import scipy for proper IRR calculation
try:
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import numba for JIT optimization
try:
    from numba import jit, prange, float64, int32
    NUMBA_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Numba available - using JIT optimization for grade-tonnage calculations")
except ImportError:
    NUMBA_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Numba not available - using numpy fallbacks. Install numba for performance optimization.")

# Import geostats utilities
from ...geostats.geostats_utils import (
    spherical_variogram, exponential_variogram, gaussian_variogram,
    VARIogram_MODELS
)

# Import declustering functionality
from ...drillholes.declustering import (
    DeclusteringEngine,
    DeclusteringConfig,
    DeclusteringMethod,
    CellShape
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DataMode(str, Enum):
    """
    Data mode for grade-tonnage analysis.
    
    BLOCK_MODEL: Input is a kriged/estimated block model on a regular grid.
                 No declustering is applied (blocks are already spatially regular).
                 Tonnage = sum(block_tonnages), weights = 1.0 for all blocks.
                 
    COMPOSITES: Input is drillhole composite samples (irregularly spaced).
                Cell-based declustering is applied to correct for clustering.
                Tonnage is derived from declustered proportions × global tonnage.
    """
    BLOCK_MODEL = "block_model"
    COMPOSITES = "composites"


class CutoffOptimizationMethod(str, Enum):
    """Cut-off optimization methodologies."""
    NPV_MAXIMIZATION = "npv_maximization"
    IRR_MAXIMIZATION = "irr_maximization"
    PAYBACK_MINIMIZATION = "payback_minimization"
    RISK_ADJUSTED_NPV = "risk_adjusted_npv"
    MULTI_CRITERIA = "multi_criteria"


class ConfidenceIntervalMethod(str, Enum):
    """
    Methods for calculating uncertainty bands.
    
    NOTE: Currently only VARIANCE_ESTIMATION is implemented, which produces
    heuristic ±CV bands, NOT formal geostatistical confidence intervals.
    
    For proper geostatistical uncertainty, integrate with SGS realisations
    or kriging variance from the estimation engine.
    """
    VARIANCE_ESTIMATION = "variance_estimation"  # Heuristic CV-based bands
    BOOTSTRAP = "bootstrap"  # Not yet implemented
    KRIGING_VARIANCE = "kriging_variance"  # Not yet implemented - requires KV integration
    SEMIVARIOGRAM = "semivariogram"  # Not yet implemented

class EconomicParameter:
    """Economic parameters for cut-off optimization."""
    METAL_PRICE_PER_UNIT = 1.0  # $/unit
    MINING_COST_PER_TONNE = 0.0  # $/t
    PROCESSING_COST_PER_TONNE = 0.0  # $/t
    ADMIN_COST_PER_TONNE = 0.0  # $/t
    TRANSPORT_COST_PER_TONNE = 0.0  # $/t
    DISCOUNT_RATE = 0.1  # Annual discount rate
    RECOVERY_RATE = 1.0  # Metal recovery
    PAYBACK_PERIOD_MAX = 5.0  # Maximum payback period (years)
    MINING_LOSS = 0.0  # Mining loss factor
    DILUTION = 0.0  # Dilution factor


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class GradeTonnagePoint:
    """
    Single point on grade-tonnage curve.

    Attributes:
        cutoff_grade: Cut-off grade threshold
        tonnage: Total tonnage above cut-off (tonnes)
        avg_grade: Average grade above cut-off
        metal_quantity: Total metal quantity above cut-off
        contained_value: Gross value above cut-off ($)
        operating_costs: Operating costs for material above cut-off ($)
        net_value: Net value above cut-off ($)
        strip_ratio: Waste:ore ratio
        cv_uncertainty_band: Heuristic CV-based uncertainty band (lower, upper).
                             NOTE: This is NOT a formal statistical confidence interval.
                             For proper uncertainty, use SGS realisations.
        cv_factor: Coefficient of variation used for uncertainty band
        decluster_weight: Declustering weight applied
    """
    cutoff_grade: float
    tonnage: float
    avg_grade: float
    metal_quantity: float
    contained_value: float = 0.0
    operating_costs: float = 0.0
    net_value: float = 0.0
    strip_ratio: float = 0.0
    cv_uncertainty_band: Tuple[float, float] = (0.0, 0.0)
    cv_factor: float = 0.0
    decluster_weight: float = 1.0

@dataclass
class GradeTonnageCurve:
    """
    Complete grade-tonnage curve with statistical properties.

    Attributes:
        points: List of GradeTonnagePoint objects
        decluster_config: Declustering configuration used
        variogram_model: Variogram model parameters
        confidence_method: Method used for uncertainty quantification
        global_statistics: Overall deposit statistics
    """
    points: List[GradeTonnagePoint] = field(default_factory=list)
    decluster_config: Optional[Any] = None
    variogram_model: Optional[Dict[str, Any]] = None
    confidence_method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.VARIANCE_ESTIMATION
    global_statistics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CutoffSensitivityAnalysis:
    """
    Results of cut-off sensitivity analysis.

    Attributes:
        cutoff_range: Range of cut-offs analyzed
        npv_by_cutoff: NPV for each cut-off grade
        irr_by_cutoff: IRR for each cut-off grade
        payback_by_cutoff: Payback period for each cut-off grade
        optimal_cutoff: Optimal cut-off based on selected method
        economic_parameters: Economic parameters used
        sensitivity_curves: Sensitivity curves for key parameters
    """
    cutoff_range: np.ndarray
    npv_by_cutoff: np.ndarray
    irr_by_cutoff: np.ndarray = None
    payback_by_cutoff: np.ndarray = None
    optimal_cutoff: float = 0.0
    economic_parameters: Dict[str, Any] = field(default_factory=dict)
    sensitivity_curves: Dict[str, Any] = field(default_factory=dict)

class GradeWeightingMethod(str, Enum):
    """
    Method for weighting grades in GT curve calculation.
    
    EQUAL_WEIGHT: Equal weight per block/sample (traditional)
                  avg_grade = sum(grade_i) / n
    
    TONNAGE_WEIGHTED: Weight by block tonnage (recommended for variable-tonnage blocks)
                      avg_grade = sum(grade_i × tonnage_i) / sum(tonnage_i)
                      
    For regular grids with uniform block size, both methods give identical results.
    For irregular grids or varying density, TONNAGE_WEIGHTED is more appropriate.
    """
    EQUAL_WEIGHT = "equal_weight"
    TONNAGE_WEIGHTED = "tonnage_weighted"


@dataclass
class GeostatsGradeTonnageConfig:
    """
    Configuration for geostatistical grade-tonnage analysis.

    Attributes:
        data_mode: BLOCK_MODEL (no declustering) or COMPOSITES (declustering applied)
        grade_weighting: Method for weighting grades (EQUAL_WEIGHT or TONNAGE_WEIGHTED)
        decluster_cell_size: Cell size for declustering (m) - only used in COMPOSITES mode
        decluster_method: Declustering methodology - only used in COMPOSITES mode
        variogram_model: Variogram model type (for future kriging variance integration)
        variogram_params: Variogram parameters (nugget, sill, range)
        confidence_method: Uncertainty band method (currently heuristic CV-based only)
        min_samples_per_cell: Minimum samples per declustering cell
        max_cell_size_ratio: Maximum cell size as ratio of variogram range
        num_bootstrap_samples: Number of bootstrap samples for uncertainty
        cutoff_range: Range of cut-off grades to analyze
        economic_params: Economic parameters for optimization
        total_deposit_tonnage: Optional override for total deposit tonnage.
                               If None, computed from sum of block/sample tonnages.
    """
    data_mode: DataMode = DataMode.BLOCK_MODEL  # Default: block model input
    grade_weighting: GradeWeightingMethod = GradeWeightingMethod.TONNAGE_WEIGHTED  # Recommended
    decluster_cell_size: float = 25.0
    decluster_method: DeclusteringMethod = DeclusteringMethod.CELL_DECLUSTERING
    variogram_model: str = "spherical"
    variogram_params: Dict[str, float] = field(default_factory=lambda: {
        "nugget": 0.1, "sill": 1.0, "range": 100.0
    })
    confidence_method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.VARIANCE_ESTIMATION
    min_samples_per_cell: int = 1
    max_cell_size_ratio: float = 0.5
    num_bootstrap_samples: int = 1000
    cutoff_range: Tuple[float, float] = (0.0, 5.0)
    economic_params: Dict[str, float] = field(default_factory=lambda: {
        "metal_price": 1.0,
        "mining_cost": 0.0,
        "processing_cost": 0.0,
        "recovery": 1.0,
        "discount_rate": 0.1
    })
    total_deposit_tonnage: Optional[float] = None  # Override for anchoring


# =============================================================================
# NUMBA JIT OPTIMIZED FUNCTIONS
# =============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _calculate_gt_curve_numba(
        grades: np.ndarray,
        tonnages: np.ndarray,
        weights: np.ndarray,
        cutoff_grades: np.ndarray,
        total_tonnage_raw: float,
        use_declustering: bool,
        use_tonnage_weighted_grade: bool = True
    ) -> np.ndarray:
        """
        Numba-optimized calculation of grade-tonnage curve points.
        
        GEOSTATISTICALLY CORRECT APPROACH:
        
        For BLOCK_MODEL mode (use_declustering=False):
            - weights = 1.0 for all blocks
            - T(c) = sum(tonnage_i for g_i >= c)
            - g_bar(c) = see grade weighting option below
            
        For COMPOSITES mode (use_declustering=True):
            - weights = declustering weights (1/samples_per_cell)
            - p(c) = sum(w_i for g_i >= c) / sum(w_i) = proportion above cutoff
            - T(c) = total_tonnage_raw * p(c) = anchored tonnage
            - g_bar(c) = weight-weighted mean grade above cutoff
        
        Grade Weighting Options:
            use_tonnage_weighted_grade=True (RECOMMENDED for block models):
                g_bar(c) = sum(grade_i × tonnage_i) / sum(tonnage_i)
                Appropriate for irregular grids or varying density.
                
            use_tonnage_weighted_grade=False (equal-weight):
                g_bar(c) = sum(grade_i × weight_i) / sum(weight_i)
                Traditional method, appropriate for regular uniform grids.

        Args:
            grades: Array of grade values (n_samples,)
            tonnages: Array of tonnage values per sample/block (n_samples,)
            weights: Declustering weights (1.0 for blocks, variable for composites)
            cutoff_grades: Array of cutoff grades to evaluate (n_cutoffs,)
            total_tonnage_raw: Total deposit tonnage for anchoring
            use_declustering: True for composites mode, False for block model
            use_tonnage_weighted_grade: True for tonnage-weighted grade (recommended)

        Returns:
            Array of shape (n_cutoffs, 5) with:
            [tonnage_above, avg_grade, metal_qty, cv_factor, proportion_above]
        """
        n_cutoffs = len(cutoff_grades)
        results = np.zeros((n_cutoffs, 5), dtype=np.float64)
        
        # Pre-compute total weight for proportion calculation
        total_weight = np.sum(weights)

        for i in prange(n_cutoffs):
            cutoff = cutoff_grades[i]

            # Find samples/blocks above cutoff
            above_cutoff_mask = grades >= cutoff
            n_above = np.sum(above_cutoff_mask)

            if n_above == 0:
                continue

            # Extract values above cutoff
            weights_above = weights[above_cutoff_mask]
            grades_above = grades[above_cutoff_mask]
            tonnages_above = tonnages[above_cutoff_mask]
            
            weight_sum_above = np.sum(weights_above)
            
            if weight_sum_above <= 0:
                continue
            
            # Calculate declustered proportion above cutoff
            proportion_above = weight_sum_above / total_weight
            
            # Calculate average grade above cutoff
            # For block models: use tonnage-weighted average (recommended)
            # For composites: use decluster-weighted average
            if use_tonnage_weighted_grade and not use_declustering:
                # TONNAGE-WEIGHTED GRADE (recommended for block models)
                # g_bar = sum(grade_i × tonnage_i) / sum(tonnage_i)
                tonnage_sum_above = np.sum(tonnages_above)
                if tonnage_sum_above > 0:
                    weighted_grade = np.sum(grades_above * tonnages_above) / tonnage_sum_above
                else:
                    weighted_grade = np.sum(grades_above * weights_above) / weight_sum_above
            else:
                # EQUAL-WEIGHT or DECLUSTER-WEIGHTED GRADE
                # For composites: uses decluster weights
                # For block models with equal_weight: uses uniform weights (equal average)
                weighted_grade = np.sum(grades_above * weights_above) / weight_sum_above

            # CRITICAL: Tonnage calculation differs by mode
            if use_declustering:
                # COMPOSITES MODE: Tonnage = proportion × total deposit tonnage
                # This anchors the curve to the actual deposit
                tonnage_above = total_tonnage_raw * proportion_above
            else:
                # BLOCK MODEL MODE: Tonnage = sum of block tonnages above cutoff
                # Weights are 1.0, so this is just direct summation
                tonnage_above = np.sum(tonnages_above)

            # Metal quantity = grade × tonnage
            metal_quantity = weighted_grade * tonnage_above

            # Heuristic uncertainty: coefficient of variation of grades above cutoff
            # NOTE: This is NOT a formal confidence interval
            # Use the same weighting scheme for variance calculation
            if use_tonnage_weighted_grade and not use_declustering:
                tonnage_sum_above = np.sum(tonnages_above)
                if tonnage_sum_above > 0:
                    grade_variance = np.sum(tonnages_above * (grades_above - weighted_grade)**2) / tonnage_sum_above
                else:
                    grade_variance = np.sum(weights_above * (grades_above - weighted_grade)**2) / weight_sum_above
            else:
                grade_variance = np.sum(weights_above * (grades_above - weighted_grade)**2) / weight_sum_above
            
            cv_factor = np.sqrt(grade_variance) / weighted_grade if weighted_grade > 0 else 0.0

            results[i] = [tonnage_above, weighted_grade, metal_quantity, cv_factor, proportion_above]

        return results

    @jit(nopython=True, cache=True)
    def _calculate_npv_numba(
        cash_flows: np.ndarray,
        discount_rate: float,
        periods: np.ndarray
    ) -> float:
        """
        Numba-optimized NPV calculation.

        Args:
            cash_flows: Array of cash flows by period
            discount_rate: Annual discount rate
            periods: Array of period numbers (0-based)

        Returns:
            Net present value
        """
        npv = 0.0
        for i in range(len(cash_flows)):
            discount_factor = 1.0 / ((1.0 + discount_rate) ** periods[i])
            npv += cash_flows[i] * discount_factor
        return npv

    @jit(nopython=True, cache=True)
    def _calculate_economic_value_numba(
        grades: np.ndarray,
        tonnages: np.ndarray,
        cutoff: float,
        metal_price: float,
        mining_cost: float,
        processing_cost: float,
        recovery: float
    ) -> Tuple[float, float, float, float]:
        """
        Numba-optimized economic value calculation for single cutoff.

        Args:
            grades: Array of grade values
            tonnages: Array of tonnage values
            cutoff: Cutoff grade
            metal_price: Metal price per unit
            mining_cost: Mining cost per tonne
            processing_cost: Processing cost per tonne
            recovery: Recovery rate

        Returns:
            Tuple of (net_value, revenue, costs, tonnage_above_cutoff)
        """
        mask = grades >= cutoff
        tonnage_above = np.sum(tonnages[mask])

        if tonnage_above == 0:
            return 0.0, 0.0, 0.0, 0.0

        avg_grade = np.mean(grades[mask])
        metal_quantity = avg_grade * tonnage_above * recovery

        revenue = metal_quantity * metal_price
        costs = tonnage_above * (mining_cost + processing_cost)
        net_value = revenue - costs

        return net_value, revenue, costs, tonnage_above

else:
    # Fallback functions when numba not available
    def _calculate_gt_curve_numba(grades, tonnages, weights, cutoff_grades, total_tonnage_raw, use_declustering, use_tonnage_weighted_grade=True):
        """NumPy fallback for grade-tonnage curve calculation."""
        logger.debug("Using numpy fallback for grade-tonnage calculation")
        n_cutoffs = len(cutoff_grades)
        results = np.zeros((n_cutoffs, 5))
        
        total_weight = np.sum(weights)

        for i, cutoff in enumerate(cutoff_grades):
            mask = grades >= cutoff
            if np.sum(mask) == 0:
                continue

            weights_above = weights[mask]
            grades_above = grades[mask]
            tonnages_above = tonnages[mask]

            weight_sum_above = np.sum(weights_above)
            if weight_sum_above <= 0:
                continue
            
            # Proportion above cutoff
            proportion_above = weight_sum_above / total_weight
            
            # Calculate average grade above cutoff
            if use_tonnage_weighted_grade and not use_declustering:
                # TONNAGE-WEIGHTED GRADE (recommended for block models)
                tonnage_sum_above = np.sum(tonnages_above)
                if tonnage_sum_above > 0:
                    weighted_grade = np.sum(grades_above * tonnages_above) / tonnage_sum_above
                else:
                    weighted_grade = np.sum(grades_above * weights_above) / weight_sum_above
            else:
                # EQUAL-WEIGHT or DECLUSTER-WEIGHTED GRADE
                weighted_grade = np.sum(grades_above * weights_above) / weight_sum_above
            
            # Tonnage calculation by mode
            if use_declustering:
                tonnage_above = total_tonnage_raw * proportion_above
            else:
                tonnage_above = np.sum(tonnages_above)
            
            metal_quantity = weighted_grade * tonnage_above
            
            # CV-based heuristic uncertainty (use same weighting as grade)
            if use_tonnage_weighted_grade and not use_declustering:
                tonnage_sum_above = np.sum(tonnages_above)
                if tonnage_sum_above > 0:
                    grade_variance = np.sum(tonnages_above * (grades_above - weighted_grade)**2) / tonnage_sum_above
                else:
                    grade_variance = np.sum(weights_above * (grades_above - weighted_grade)**2) / weight_sum_above
            else:
                grade_variance = np.sum(weights_above * (grades_above - weighted_grade)**2) / weight_sum_above
            
            cv_factor = np.sqrt(grade_variance) / weighted_grade if weighted_grade > 0 else 0.0

            results[i] = [tonnage_above, weighted_grade, metal_quantity, cv_factor, proportion_above]

        return results

    def _calculate_npv_numba(cash_flows, discount_rate, periods):
        """NumPy fallback for NPV calculation."""
        discount_factors = 1.0 / ((1.0 + discount_rate) ** periods)
        return np.sum(cash_flows * discount_factors)

    def _calculate_economic_value_numba(grades, tonnages, cutoff, metal_price, mining_cost, processing_cost, recovery):
        """NumPy fallback for economic value calculation."""
        mask = grades >= cutoff
        tonnage_above = np.sum(tonnages[mask])

        if tonnage_above == 0:
            return 0.0, 0.0, 0.0, 0.0

        avg_grade = np.mean(grades[mask])
        metal_quantity = avg_grade * tonnage_above * recovery
        revenue = metal_quantity * metal_price
        costs = tonnage_above * (mining_cost + processing_cost)
        net_value = revenue - costs

        return net_value, revenue, costs, tonnage_above


# =============================================================================
# CORE GRADE-TONNAGE ENGINE
# =============================================================================

class GeostatsGradeTonnageEngine:
    """
    Grade-tonnage curve generation engine with dual-mode support.

    This engine provides:
    - BLOCK_MODEL mode: Direct GT curves from regular block models (no declustering)
    - COMPOSITES mode: Cell-based declustering for drillhole composites
    - Proper tonnage anchoring to deposit totals
    - Numba-optimized calculations for performance
    - Heuristic CV-based uncertainty bands
    - Economic value calculations
    
    IMPORTANT GEOSTATISTICAL NOTES:
    - Block models should use BLOCK_MODEL mode (declustering is inappropriate)
    - Composite samples should use COMPOSITES mode with proper declustering
    - The "confidence intervals" are heuristic ±CV bands, not formal CIs
    - For JORC/SAMREC grade, integrate with SGS realisations for proper uncertainty
    """

    def __init__(self, config: GeostatsGradeTonnageConfig = None):
        """
        Initialize the grade-tonnage engine.

        Args:
            config: Configuration object, uses defaults if None
        """
        self.config = config or GeostatsGradeTonnageConfig()
        self._decluster_engine = None
        self._variogram_model = None
        self._total_tonnage_raw = None  # Cached raw tonnage for anchoring
        self.logger = logging.getLogger(__name__)

    def calculate_grade_tonnage_curve(
        self,
        sample_data: pd.DataFrame,
        cutoff_range: Optional[np.ndarray] = None,
        element_column: str = "grade",
        tonnage_column: str = "tonnage",
        x_column: str = "x",
        y_column: str = "y",
        z_column: str = "z"
    ) -> GradeTonnageCurve:
        """
        Calculate grade-tonnage curve with proper mode handling.
        
        BLOCK_MODEL mode (default):
            - No declustering applied (blocks are spatially regular)
            - Tonnage = sum of block tonnages above cutoff
            - Grade = tonnage-weighted mean grade above cutoff
            
        COMPOSITES mode:
            - Cell-based declustering applied to composites
            - Proportion p(c) = sum(w_i for g_i >= c) / sum(w_i)
            - Tonnage = total_deposit_tonnage × p(c)
            - Grade = weight-weighted mean grade above cutoff

        Args:
            sample_data: DataFrame with sample/block data
            cutoff_range: Array of cutoff grades to evaluate
            element_column: Name of grade column
            tonnage_column: Name of tonnage column
            x_column, y_column, z_column: Coordinate columns

        Returns:
            GradeTonnageCurve with all calculated points
        """
        mode_str = self.config.data_mode.value
        self.logger.info(f"Starting grade-tonnage analysis in {mode_str} mode with {len(sample_data)} samples/blocks")

        # Validate input data
        required_columns = [element_column, tonnage_column, x_column, y_column, z_column]
        missing_cols = [col for col in required_columns if col not in sample_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Clean data
        data = sample_data.dropna(subset=required_columns).copy()

        if len(data) == 0:
            raise ValueError("No valid data after removing NaN values")

        # Set up cutoff range
        if cutoff_range is None:
            min_grade = data[element_column].min()
            max_grade = data[element_column].max()
            cutoff_range = np.linspace(max(min_grade, 0), max_grade, 50)

        # Compute raw total tonnage (before any weighting)
        # This is the anchor for all GT calculations
        self._total_tonnage_raw = self.config.total_deposit_tonnage or float(data[tonnage_column].sum())
        self.logger.info(f"Total deposit tonnage (raw): {self._total_tonnage_raw:,.0f} t")

        # Get weights based on data mode
        if self.config.data_mode == DataMode.COMPOSITES:
            self.logger.info("COMPOSITES mode: Performing cell-based declustering...")
            weights = self._perform_declustering(data, x_column, y_column, z_column)
            use_declustering = True
        else:
            self.logger.info("BLOCK_MODEL mode: Using uniform weights (no declustering)")
            weights = np.ones(len(data))
            use_declustering = False

        # Calculate grade-tonnage points with proper anchoring
        self.logger.info("Calculating grade-tonnage curve points...")
        curve_points = self._calculate_curve_points(
            data[element_column].values,
            data[tonnage_column].values,
            weights,
            cutoff_range,
            use_declustering
        )

        # Calculate global statistics (consistent with curve at cutoff 0)
        global_stats = self._calculate_global_statistics(
            data, element_column, tonnage_column, weights, use_declustering
        )

        # Create result curve
        curve = GradeTonnageCurve(
            points=curve_points,
            decluster_config=self.config,
            variogram_model=self._variogram_model,
            confidence_method=self.config.confidence_method,
            global_statistics=global_stats
        )

        self.logger.info(f"Grade-tonnage analysis complete: {len(curve_points)} points calculated")
        return curve

    def _perform_declustering(self, data: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> np.ndarray:
        """
        Perform cell-based declustering on sample data.

        Args:
            data: Sample DataFrame
            x_col, y_col, z_col: Coordinate column names

        Returns:
            Array of declustering weights
        """
        # Import required classes
        from ...drillholes.declustering import CellDefinition

        # Set up declustering configuration
        coords = data[[x_col, y_col, z_col]].values

        cell_definition = CellDefinition(
            cell_size_x=self.config.decluster_cell_size,
            cell_size_y=self.config.decluster_cell_size,
            cell_size_z=self.config.decluster_cell_size
        )

        decluster_config = DeclusteringConfig(
            method=self.config.decluster_method,
            cell_definition=cell_definition
        )

        # Initialize declustering engine
        self._decluster_engine = DeclusteringEngine(decluster_config)

        # Perform declustering
        # Create a DataFrame for the declustering engine
        decluster_df = pd.DataFrame({
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2],
            'SAMPLE_ID': range(len(coords))
        })

        result_df, summary = self._decluster_engine.compute_weights(decluster_df)

        # Debug: print available columns
        self.logger.info(f"Declustering result columns: {list(result_df.columns)}")

        # Try common weight column names
        weight_col = None
        for col in ['DECLUSTER_WEIGHT', 'WEIGHT', 'weight', 'Weight', 'declust_weight']:
            if col in result_df.columns:
                weight_col = col
                break

        if weight_col is None:
            # Default to equal weights if no weight column found
            self.logger.warning("No declustering weight column found, using equal weights")
            weights = np.ones(len(result_df))
        else:
            weights = result_df[weight_col].values

        self.logger.info(f"Declustering complete: {len(weights)} weights calculated, "
                        f"cell size: {self.config.decluster_cell_size}m")

        return weights

    def _calculate_curve_points(
        self,
        grades: np.ndarray,
        tonnages: np.ndarray,
        weights: np.ndarray,
        cutoff_grades: np.ndarray,
        use_declustering: bool
    ) -> List[GradeTonnagePoint]:
        """
        Calculate grade-tonnage points for all cutoffs with proper anchoring.

        Args:
            grades: Array of grade values
            tonnages: Array of tonnage values (per block/sample)
            weights: Weights array (1.0 for blocks, decluster weights for composites)
            cutoff_grades: Array of cutoff grades
            use_declustering: True for composites mode (proportion-based tonnage)

        Returns:
            List of GradeTonnagePoint objects
        """
        # Determine if tonnage-weighted grade should be used
        use_tonnage_weighted = (
            self.config.grade_weighting == GradeWeightingMethod.TONNAGE_WEIGHTED
        )
        
        # Use Numba-optimized calculation with proper tonnage anchoring
        results = _calculate_gt_curve_numba(
            grades.astype(np.float64),
            tonnages.astype(np.float64),
            weights.astype(np.float64),
            cutoff_grades.astype(np.float64),
            float(self._total_tonnage_raw),
            use_declustering,
            use_tonnage_weighted
        )

        points = []
        for i, cutoff in enumerate(cutoff_grades):
            tonnage, avg_grade, metal_qty, cv_factor, proportion = results[i]

            if tonnage > 0:
                # Heuristic uncertainty band (±CV on tonnage)
                # NOTE: This is NOT a formal confidence interval
                ci_lower = max(0, tonnage * (1 - cv_factor))
                ci_upper = tonnage * (1 + cv_factor)

                # Calculate economic values using the properly computed tonnage
                # Revenue = metal × price × recovery
                metal_price = self.config.economic_params["metal_price"]
                mining_cost = self.config.economic_params["mining_cost"]
                processing_cost = self.config.economic_params["processing_cost"]
                recovery = self.config.economic_params["recovery"]
                
                revenue = metal_qty * metal_price * recovery
                costs = tonnage * (mining_cost + processing_cost)
                net_value = revenue - costs

                # Average weight above cutoff (for diagnostic purposes)
                mask = grades >= cutoff
                avg_weight = np.mean(weights[mask]) if np.sum(mask) > 0 else 1.0

                point = GradeTonnagePoint(
                    cutoff_grade=cutoff,
                    tonnage=tonnage,
                    avg_grade=avg_grade,
                    metal_quantity=metal_qty,
                    contained_value=revenue,
                    operating_costs=costs,
                    net_value=net_value,
                    cv_uncertainty_band=(ci_lower, ci_upper),
                    cv_factor=cv_factor,
                    decluster_weight=avg_weight
                )
                points.append(point)

        return points

    def _calculate_global_statistics(
        self, 
        data: pd.DataFrame, 
        grade_col: str, 
        tonnage_col: str,
        weights: np.ndarray,
        use_declustering: bool
    ) -> Dict[str, Any]:
        """
        Calculate global deposit statistics consistent with curve at cutoff 0.
        
        IMPORTANT: These statistics must match the curve at cutoff=0 for consistency.
        
        For BLOCK_MODEL mode:
            - Total tonnage = sum of block tonnages
            - Mean grade = tonnage-weighted or equal-weighted based on config
            
        For COMPOSITES mode:
            - Total tonnage = anchored total (sum of block tonnages or config override)
            - Mean grade = declustered (weight-weighted) mean
        """
        grades = data[grade_col].values
        tonnages = data[tonnage_col].values

        # Use the raw total tonnage (same anchor as curve)
        total_tonnage = self._total_tonnage_raw
        
        # Calculate mean grade using appropriate weighting
        # This must match the curve calculation for consistency
        use_tonnage_weighted = (
            self.config.grade_weighting == GradeWeightingMethod.TONNAGE_WEIGHTED
            and not use_declustering
        )
        
        if use_tonnage_weighted:
            # Tonnage-weighted mean grade (recommended for block models)
            total_ton = np.sum(tonnages)
            if total_ton > 0:
                avg_grade = np.sum(grades * tonnages) / total_ton
            else:
                avg_grade = np.mean(grades)
            weighting_method = "tonnage-weighted"
        else:
            # Weight-weighted mean (decluster weights or equal weights)
            total_weight = np.sum(weights)
            if total_weight > 0:
                avg_grade = np.sum(grades * weights) / total_weight
            else:
                avg_grade = np.mean(grades)
            weighting_method = "decluster-weighted" if use_declustering else "equal-weighted"
        
        # Total metal = mean grade × total tonnage
        total_metal = avg_grade * total_tonnage

        # Grade distribution statistics
        raw_mean = float(np.mean(grades))
        weighted_mean = float(avg_grade)
        
        # Tonnage-weighted mean for reference (always compute)
        total_ton = np.sum(tonnages)
        tonnage_weighted_mean = float(np.sum(grades * tonnages) / total_ton) if total_ton > 0 else raw_mean
        
        grade_stats = {
            "mean": weighted_mean,  # Consistent with curve
            "mean_raw": raw_mean,   # Simple arithmetic mean
            "mean_tonnage_weighted": tonnage_weighted_mean,  # Tonnage-weighted for reference
            "median": float(np.median(grades)),
            "std": float(np.std(grades)),
            "min": float(np.min(grades)),
            "max": float(np.max(grades)),
            "q25": float(np.percentile(grades, 25)),
            "q75": float(np.percentile(grades, 75)),
            "cv": float(np.std(grades) / weighted_mean) if weighted_mean > 0 else 0.0,
            "iqr": float(np.percentile(grades, 75) - np.percentile(grades, 25)),
            "weighting_method": weighting_method
        }

        return {
            "total_tonnage": float(total_tonnage),
            "total_metal": float(total_metal),
            "grade_statistics": grade_stats,
            "sample_count": len(data),
            "data_mode": self.config.data_mode.value,
            "grade_weighting": self.config.grade_weighting.value,
            "declustered": use_declustering,
            "notes": (
                f"Mean grade computed using {weighting_method} method. "
                f"Mode: {self.config.data_mode.value}. "
                + ("Declustering applied to composites." if use_declustering 
                   else "No declustering (block model mode).")
            )
        }


# =============================================================================
# CUT-OFF SENSITIVITY ANALYSIS ENGINE
# =============================================================================

class CutoffSensitivityEngine:
    """
    Cut-off sensitivity analysis engine with NPV optimization.

    This engine provides:
    - NPV maximization across multiple cut-offs
    - Sensitivity analysis for economic parameters
    - IRR and payback period calculations
    - Risk-adjusted optimization
    - Multi-criteria decision analysis
    """

    def __init__(self, economic_params: Dict[str, float] = None):
        """
        Initialize the cut-off sensitivity engine.

        Args:
            economic_params: Economic parameters dictionary
        """
        self.economic_params = economic_params or {
            "metal_price": 1.0,
            "mining_cost": 0.0,
            "processing_cost": 0.0,
            "recovery": 1.0,
            "discount_rate": 0.1,
            "admin_cost": 0.0,
            "transport_cost": 0.0,
            "dilution": 0.0,
            "mining_loss": 0.0
        }
        self.logger = logging.getLogger(__name__)

    def perform_sensitivity_analysis(
        self,
        grade_tonnage_curve: GradeTonnageCurve,
        cutoff_range: np.ndarray,
        optimization_method: CutoffOptimizationMethod = CutoffOptimizationMethod.NPV_MAXIMIZATION
    ) -> CutoffSensitivityAnalysis:
        """
        Perform cut-off sensitivity analysis.

        Args:
            grade_tonnage_curve: GradeTonnageCurve object
            cutoff_range: Array of cut-off grades to analyze
            optimization_method: Optimization method to use

        Returns:
            CutoffSensitivityAnalysis results
        """
        self.logger.info(f"Performing cut-off sensitivity analysis with method: {optimization_method}")

        # Extract data from curve
        cutoffs = np.array([pt.cutoff_grade for pt in grade_tonnage_curve.points])
        tonnages = np.array([pt.tonnage for pt in grade_tonnage_curve.points])
        avg_grades = np.array([pt.avg_grade for pt in grade_tonnage_curve.points])

        # Calculate NPV for each cutoff
        npv_values = self._calculate_npv_profile(cutoffs, tonnages, avg_grades)

        # Find optimal cutoff
        optimal_idx = np.argmax(npv_values)
        optimal_cutoff = cutoffs[optimal_idx]

        # Calculate IRR and payback if requested
        irr_values = None
        payback_values = None

        if optimization_method in [CutoffOptimizationMethod.IRR_MAXIMIZATION,
                                   CutoffOptimizationMethod.MULTI_CRITERIA]:
            irr_values = self._calculate_irr_profile(cutoffs, tonnages, avg_grades)

        if optimization_method in [CutoffOptimizationMethod.PAYBACK_MINIMIZATION,
                                   CutoffOptimizationMethod.MULTI_CRITERIA]:
            payback_values = self._calculate_payback_profile(cutoffs, tonnages, avg_grades)

        # Perform sensitivity analysis on key parameters
        sensitivity_curves = self._calculate_sensitivity_curves(
            cutoffs, tonnages, avg_grades, optimal_cutoff
        )

        analysis = CutoffSensitivityAnalysis(
            cutoff_range=cutoffs,
            npv_by_cutoff=npv_values,
            irr_by_cutoff=irr_values,
            payback_by_cutoff=payback_values,
            optimal_cutoff=optimal_cutoff,
            economic_parameters=self.economic_params.copy(),
            sensitivity_curves=sensitivity_curves
        )

        self.logger.info(f"Sensitivity analysis complete. Optimal cutoff: {optimal_cutoff:.2f}")
        return analysis

    def _calculate_npv_profile(self, cutoffs: np.ndarray, tonnages: np.ndarray, avg_grades: np.ndarray) -> np.ndarray:
        """Calculate NPV for each cutoff grade."""
        npv_values = np.zeros(len(cutoffs))

        for i, (cutoff, tonnage, grade) in enumerate(zip(cutoffs, tonnages, avg_grades)):
            # Simplified NPV calculation (assuming single period)
            metal_quantity = grade * tonnage * self.economic_params["recovery"]
            revenue = metal_quantity * self.economic_params["metal_price"]

            total_cost_per_tonne = (
                self.economic_params["mining_cost"] +
                self.economic_params["processing_cost"] +
                self.economic_params["admin_cost"] +
                self.economic_params["transport_cost"]
            )
            costs = tonnage * total_cost_per_tonne

            net_value = revenue - costs
            npv_values[i] = net_value / (1 + self.economic_params["discount_rate"])

        return npv_values

    def _calculate_irr_profile(
        self,
        cutoffs: np.ndarray,
        tonnages: np.ndarray,
        avg_grades: np.ndarray,
        annual_capacity: float = 10_000_000,
        initial_capex: float = 0.0
    ) -> np.ndarray:
        """
        Calculate IRR for each cutoff grade using proper NPV=0 root finding.

        IRR is the discount rate that makes NPV = 0 for a given cash flow series.
        Uses scipy.optimize.brentq for numerical root finding when available.

        Args:
            cutoffs: Array of cutoff grades
            tonnages: Array of tonnages above each cutoff
            avg_grades: Array of average grades above each cutoff
            annual_capacity: Annual processing capacity (tonnes/year)
            initial_capex: Initial capital expenditure (for IRR calculation)

        Returns:
            Array of IRR values (as decimal, e.g., 0.15 = 15%)
        """
        irr_values = np.zeros(len(cutoffs))

        for i, (cutoff, tonnage, grade) in enumerate(zip(cutoffs, tonnages, avg_grades)):
            if tonnage <= 0:
                irr_values[i] = 0.0
                continue

            # Generate multi-period cash flows
            mine_life = max(1, int(np.ceil(tonnage / annual_capacity)))
            cash_flows = self._generate_cash_flow_series(
                tonnage, grade, mine_life, annual_capacity, initial_capex
            )

            # Calculate IRR using root finding
            irr = self._calculate_irr_from_cash_flows(cash_flows)
            irr_values[i] = irr

        return irr_values

    def _generate_cash_flow_series(
        self,
        total_tonnage: float,
        avg_grade: float,
        mine_life: int,
        annual_capacity: float,
        initial_capex: float = 0.0
    ) -> np.ndarray:
        """
        Generate annual cash flow series for IRR/NPV calculation.

        Args:
            total_tonnage: Total ore tonnage
            avg_grade: Average grade
            mine_life: Mine life in years
            annual_capacity: Annual processing capacity
            initial_capex: Initial capital (year 0 outflow)

        Returns:
            Array of cash flows starting from year 0
        """
        cash_flows = np.zeros(mine_life + 1)

        # Year 0: Initial capex (outflow)
        cash_flows[0] = -initial_capex if initial_capex > 0 else 0.0

        remaining_tonnage = total_tonnage

        total_cost_per_tonne = (
            self.economic_params.get("mining_cost", 0) +
            self.economic_params.get("processing_cost", 0) +
            self.economic_params.get("admin_cost", 0) +
            self.economic_params.get("transport_cost", 0)
        )

        for year in range(1, mine_life + 1):
            year_tonnage = min(remaining_tonnage, annual_capacity)
            remaining_tonnage -= year_tonnage

            if year_tonnage <= 0:
                break

            # Revenue
            metal_qty = year_tonnage * avg_grade * self.economic_params.get("recovery", 1.0)
            revenue = metal_qty * self.economic_params.get("metal_price", 1.0)

            # Costs
            opex = year_tonnage * total_cost_per_tonne

            # Net cash flow
            cash_flows[year] = revenue - opex

        return cash_flows

    def _calculate_irr_from_cash_flows(self, cash_flows: np.ndarray) -> float:
        """
        Calculate IRR from cash flow series using scipy brentq or Newton-Raphson fallback.

        IRR is the rate r such that: sum(CF_t / (1+r)^t) = 0

        Args:
            cash_flows: Array of cash flows (year 0 to year n)

        Returns:
            IRR as decimal (e.g., 0.15 for 15%), or 0.0 if not calculable
        """
        # Need at least one negative and one positive cash flow for valid IRR
        if not (np.any(cash_flows < 0) and np.any(cash_flows > 0)):
            # No sign change - IRR undefined or infinite
            if np.sum(cash_flows) > 0:
                return 1.0  # Very high return (cap at 100%)
            return 0.0

        def npv_at_rate(rate):
            """Calculate NPV at given discount rate."""
            periods = np.arange(len(cash_flows))
            if rate <= -1:
                return float('inf')
            return np.sum(cash_flows / ((1 + rate) ** periods))

        if SCIPY_AVAILABLE:
            try:
                # Use Brent's method to find root (IRR) between -0.99 and 10 (1000%)
                irr = brentq(npv_at_rate, -0.99, 10.0, xtol=1e-6, maxiter=100)
                return max(0.0, min(irr, 10.0))  # Cap between 0% and 1000%
            except (ValueError, RuntimeError):
                # Brentq failed - fall back to Newton-Raphson
                pass

        # Newton-Raphson fallback
        rate = 0.1  # Initial guess 10%
        for _ in range(50):
            npv = npv_at_rate(rate)

            # Numerical derivative
            delta = 0.0001
            npv_delta = npv_at_rate(rate + delta)
            derivative = (npv_delta - npv) / delta

            if abs(derivative) < 1e-10:
                break

            new_rate = rate - npv / derivative

            # Convergence check
            if abs(new_rate - rate) < 1e-6:
                return max(0.0, min(new_rate, 10.0))

            rate = max(-0.99, min(new_rate, 10.0))  # Keep in bounds

        return max(0.0, rate)

    def _calculate_payback_profile(
        self,
        cutoffs: np.ndarray,
        tonnages: np.ndarray,
        avg_grades: np.ndarray,
        annual_capacity: float = 10_000_000,
        initial_capex: float = 0.0,
        discounted: bool = True
    ) -> np.ndarray:
        """
        Calculate payback period for each cutoff grade.

        Payback period is the time required for cumulative cash flows to equal
        the initial investment.

        Args:
            cutoffs: Array of cutoff grades
            tonnages: Array of tonnages above each cutoff
            avg_grades: Array of average grades above each cutoff
            annual_capacity: Annual processing capacity (tonnes/year)
            initial_capex: Initial capital expenditure
            discounted: If True, calculate discounted payback period

        Returns:
            Array of payback periods in years (999 if never achieved)
        """
        payback_values = np.zeros(len(cutoffs))
        discount_rate = self.economic_params.get("discount_rate", 0.1)

        for i, (cutoff, tonnage, grade) in enumerate(zip(cutoffs, tonnages, avg_grades)):
            if tonnage <= 0:
                payback_values[i] = 999.0
                continue

            # Generate cash flow series
            mine_life = max(1, int(np.ceil(tonnage / annual_capacity)))
            cash_flows = self._generate_cash_flow_series(
                tonnage, grade, mine_life, annual_capacity, initial_capex
            )

            # Calculate payback period
            cumulative = 0.0
            payback = 999.0

            for year, cf in enumerate(cash_flows):
                if discounted and year > 0:
                    cf = cf / ((1 + discount_rate) ** year)

                cumulative += cf

                if cumulative >= 0 and payback == 999.0:
                    # Interpolate to find exact payback point
                    if year > 0:
                        prev_cumulative = cumulative - cf
                        if cf != 0:
                            fraction = -prev_cumulative / cf
                            payback = year - 1 + fraction
                        else:
                            payback = float(year)
                    else:
                        payback = 0.0
                    break

            payback_values[i] = payback

        return payback_values

    def _calculate_sensitivity_curves(self, cutoffs: np.ndarray, tonnages: np.ndarray,
                                    avg_grades: np.ndarray, optimal_cutoff: float) -> Dict[str, Any]:
        """Calculate sensitivity curves for key economic parameters."""
        sensitivity_params = ["metal_price", "mining_cost", "processing_cost", "recovery", "discount_rate"]
        sensitivity_curves = {}

        base_npv = self._calculate_npv_profile(np.array([optimal_cutoff]), tonnages, avg_grades)[0]

        for param in sensitivity_params:
            if param in self.economic_params:
                # Test +/- 20% variation
                variations = [-0.2, -0.1, 0.0, 0.1, 0.2]
                param_values = []
                npv_values = []

                for variation in variations:
                    # Create modified economic parameters
                    modified_params = self.economic_params.copy()
                    if param == "recovery":
                        modified_params[param] = max(0.1, min(1.0, self.economic_params[param] * (1 + variation)))
                    else:
                        modified_params[param] = max(0.0, self.economic_params[param] * (1 + variation))

                    # Calculate NPV with modified parameters
                    old_params = self.economic_params.copy()
                    self.economic_params = modified_params

                    npv = self._calculate_npv_profile(np.array([optimal_cutoff]), tonnages, avg_grades)[0]
                    param_values.append(modified_params[param])
                    npv_values.append(npv)

                    self.economic_params = old_params

                sensitivity_curves[param] = {
                    "param_values": param_values,
                    "npv_values": npv_values,
                    "base_npv": base_npv
                }

        return sensitivity_curves


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_grade_tonnage_data(data: pd.DataFrame, grade_col: str = "grade",
                               tonnage_col: str = "tonnage") -> List[str]:
    """
    Validate grade-tonnage dataset for analysis.

    Args:
        data: Input DataFrame
        grade_col: Grade column name
        tonnage_col: Tonnage column name

    Returns:
        List of validation warnings/errors
    """
    warnings = []

    # Check required columns
    required_cols = [grade_col, tonnage_col]
    for col in required_cols:
        if col not in data.columns:
            warnings.append(f"Missing required column: {col}")

    if not data.empty:
        # Check for negative values
        if (data[grade_col] < 0).any():
            warnings.append("Negative grade values found")

        if (data[tonnage_col] < 0).any():
            warnings.append("Negative tonnage values found")

        # Check for missing values
        grade_missing = data[grade_col].isna().sum()
        tonnage_missing = data[tonnage_col].isna().sum()

        if grade_missing > 0:
            warnings.append(f"{grade_missing} missing grade values")

        if tonnage_missing > 0:
            warnings.append(f"{tonnage_missing} missing tonnage values")

        # Check data ranges
        grade_range = data[grade_col].max() - data[grade_col].min()
        if grade_range == 0:
            warnings.append("All grade values are identical")

    return warnings


def export_grade_tonnage_results(curve: GradeTonnageCurve, filename: str) -> bool:
    """
    Export grade-tonnage curve results to CSV.

    Args:
        curve: GradeTonnageCurve object
        filename: Output filename

    Returns:
        True if successful
    """
    try:
        data = []
        for point in curve.points:
            data.append({
                "cutoff_grade": point.cutoff_grade,
                "tonnage": point.tonnage,
                "avg_grade": point.avg_grade,
                "metal_quantity": point.metal_quantity,
                "contained_value": point.contained_value,
                "operating_costs": point.operating_costs,
                "net_value": point.net_value,
                "strip_ratio": point.strip_ratio,
                "cv_uncertainty_lower": point.cv_uncertainty_band[0],
                "cv_uncertainty_upper": point.cv_uncertainty_band[1],
                "cv_factor": point.cv_factor,
                "avg_weight": point.decluster_weight
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        # Also export global statistics as metadata
        stats = curve.global_statistics
        if stats:
            meta_filename = filename.replace('.csv', '_metadata.csv')
            meta_data = {
                "parameter": [
                    "total_tonnage",
                    "total_metal", 
                    "mean_grade",
                    "sample_count",
                    "data_mode",
                    "declustered",
                    "note"
                ],
                "value": [
                    stats.get("total_tonnage", "N/A"),
                    stats.get("total_metal", "N/A"),
                    stats.get("grade_statistics", {}).get("mean", "N/A"),
                    stats.get("sample_count", "N/A"),
                    stats.get("data_mode", "N/A"),
                    stats.get("declustered", "N/A"),
                    "CI columns are heuristic ±CV bands, not formal confidence intervals"
                ]
            }
            meta_df = pd.DataFrame(meta_data)
            meta_df.to_csv(meta_filename, index=False)
            logger.info(f"Metadata exported to {meta_filename}")
        
        logger.info(f"Grade-tonnage results exported to {filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to export grade-tonnage results: {e}")
        return False
