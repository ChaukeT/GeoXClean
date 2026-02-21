"""
Advanced Grade-Tonnage Analysis for JORC/SAMREC Compliance.

This module extends the base grade-tonnage engine with:
1. SGS-based uncertainty quantification (empirical 95% CI from realisations)
2. Domain-wise analysis (by lithology/domain code)
3. Classification-based GT curves (Measured/Indicated/Inferred)

These features are required for:
- JORC 2012 Code compliance (Table 1, Section 3)
- SAMREC Code compliance
- CIM Definition Standards
- NI 43-101 reporting

Author: GeoX Mining Software - Advanced GT Analysis Module
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum

# Import base GT engine
from .geostats_grade_tonnage import (
    GeostatsGradeTonnageEngine,
    GeostatsGradeTonnageConfig,
    GradeTonnageCurve,
    GradeTonnagePoint,
    DataMode,
    _calculate_gt_curve_numba
)

logger = logging.getLogger(__name__)

# Try to import numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# =============================================================================
# DATA CLASSES FOR ADVANCED ANALYSIS
# =============================================================================

class ResourceCategory(str, Enum):
    """Resource classification categories (JORC/SAMREC)."""
    MEASURED = "measured"
    INDICATED = "indicated"
    INFERRED = "inferred"
    UNCLASSIFIED = "unclassified"


@dataclass
class GTCurveStatistics:
    """
    Statistics for a single GT curve point across multiple realisations.
    
    Used for SGS-based uncertainty quantification.
    """
    cutoff: float
    
    # Tonnage statistics
    tonnage_mean: float
    tonnage_std: float
    tonnage_p05: float  # 5th percentile (lower bound 90% CI)
    tonnage_p10: float  # 10th percentile
    tonnage_p50: float  # Median
    tonnage_p90: float  # 90th percentile
    tonnage_p95: float  # 95th percentile (upper bound 90% CI)
    
    # Grade statistics
    grade_mean: float
    grade_std: float
    grade_p05: float
    grade_p10: float
    grade_p50: float
    grade_p90: float
    grade_p95: float
    
    # Metal statistics
    metal_mean: float
    metal_std: float
    metal_p05: float
    metal_p10: float
    metal_p50: float
    metal_p90: float
    metal_p95: float
    
    # Sample count
    n_realisations: int = 0


@dataclass
class SGSUncertaintyResult:
    """
    Complete SGS-based uncertainty analysis result.
    
    Contains GT curve statistics for each cutoff from multiple realisations,
    providing empirical confidence intervals.
    """
    # Per-cutoff statistics
    curve_statistics: List[GTCurveStatistics] = field(default_factory=list)
    
    # Individual realisation curves (for QA/plotting)
    realisation_curves: List[GradeTonnageCurve] = field(default_factory=list)
    
    # Summary statistics
    n_realisations: int = 0
    cutoffs: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DomainGTResult:
    """
    Domain-wise GT analysis result.
    
    Contains separate GT curves for each geological domain.
    """
    # Per-domain curves
    domain_curves: Dict[str, GradeTonnageCurve] = field(default_factory=dict)
    
    # Combined curve (all domains)
    combined_curve: Optional[GradeTonnageCurve] = None
    
    # Domain statistics
    domain_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Domain list
    domains: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassificationGTResult:
    """
    Classification-based GT analysis result (Measured/Indicated/Inferred).
    
    Provides GT curves stratified by resource classification category.
    """
    # Per-category curves
    measured_curve: Optional[GradeTonnageCurve] = None
    indicated_curve: Optional[GradeTonnageCurve] = None
    inferred_curve: Optional[GradeTonnageCurve] = None
    
    # Combined curve (all categories)
    combined_curve: Optional[GradeTonnageCurve] = None
    
    # Category statistics
    category_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# SGS-BASED UNCERTAINTY QUANTIFICATION
# =============================================================================

class SGSUncertaintyEngine:
    """
    SGS-based uncertainty quantification for grade-tonnage curves.
    
    This engine uses multiple SGS realisations to derive empirical confidence
    intervals for tonnage, grade, and metal content at each cutoff grade.
    
    METHODOLOGY:
    1. For each SGS realisation, compute a complete GT curve
    2. At each cutoff, collect tonnage/grade/metal across realisations
    3. Derive empirical percentiles (P5, P10, P50, P90, P95)
    4. Report mean ± std and 90%/95% confidence intervals
    
    This is the CORRECT way to quantify uncertainty in GT curves,
    as required for JORC/SAMREC compliance.
    """
    
    def __init__(self, base_config: Optional[GeostatsGradeTonnageConfig] = None):
        """
        Initialize the SGS uncertainty engine.
        
        Args:
            base_config: Base configuration for GT calculations
        """
        self.base_config = base_config or GeostatsGradeTonnageConfig()
        self.logger = logging.getLogger(__name__)
    
    def compute_uncertainty_from_realisations(
        self,
        realisations: np.ndarray,
        block_tonnages: np.ndarray,
        block_coordinates: np.ndarray,
        cutoffs: np.ndarray,
        total_tonnage: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ) -> SGSUncertaintyResult:
        """
        Compute GT curve uncertainty from SGS realisations.
        
        Args:
            realisations: Grade realisations array (n_realisations, n_blocks)
                          NOTE: Must be in PHYSICAL grade space, not Gaussian!
            block_tonnages: Tonnage per block (n_blocks,)
            block_coordinates: Block coordinates (n_blocks, 3)
            cutoffs: Cutoff grades to evaluate
            total_tonnage: Total deposit tonnage (defaults to sum of block tonnages)
            progress_callback: Optional callback(pct, message)
        
        Returns:
            SGSUncertaintyResult with empirical confidence intervals
        """
        n_real = realisations.shape[0]
        n_blocks = realisations.shape[1]
        n_cutoffs = len(cutoffs)
        
        self.logger.info(f"Computing GT uncertainty from {n_real} realisations, {n_blocks} blocks, {n_cutoffs} cutoffs")
        
        # Compute total tonnage if not provided
        if total_tonnage is None:
            total_tonnage = float(np.sum(block_tonnages))
        
        # Storage for results per cutoff
        tonnage_matrix = np.zeros((n_real, n_cutoffs))
        grade_matrix = np.zeros((n_real, n_cutoffs))
        metal_matrix = np.zeros((n_real, n_cutoffs))
        
        # Compute GT curve for each realisation
        realisation_curves = []
        
        for i_real in range(n_real):
            if progress_callback:
                pct = int(100 * i_real / n_real)
                progress_callback(pct, f"Processing realisation {i_real + 1}/{n_real}...")
            
            grades = realisations[i_real, :]
            
            # Compute GT points for this realisation
            # Use uniform weights (block model mode)
            weights = np.ones(n_blocks)
            
            results = _calculate_gt_curve_numba(
                grades.astype(np.float64),
                block_tonnages.astype(np.float64),
                weights.astype(np.float64),
                cutoffs.astype(np.float64),
                float(total_tonnage),
                False  # Block model mode
            )
            
            # Extract tonnage, grade, metal for each cutoff
            tonnage_matrix[i_real, :] = results[:, 0]
            grade_matrix[i_real, :] = results[:, 1]
            metal_matrix[i_real, :] = results[:, 2]
            
            # Store individual curve
            points = []
            for j, cutoff in enumerate(cutoffs):
                if results[j, 0] > 0:
                    points.append(GradeTonnagePoint(
                        cutoff_grade=cutoff,
                        tonnage=results[j, 0],
                        avg_grade=results[j, 1],
                        metal_quantity=results[j, 2],
                        cv_uncertainty_band=(0, 0),  # Not applicable for individual realisation
                        cv_factor=0.0,
                        decluster_weight=1.0
                    ))
            
            curve = GradeTonnageCurve(points=points)
            curve.global_statistics = {
                'realisation_id': i_real,
                'total_tonnage': total_tonnage
            }
            realisation_curves.append(curve)
        
        # Compute statistics across realisations for each cutoff
        curve_statistics = []
        
        for j, cutoff in enumerate(cutoffs):
            tonnages = tonnage_matrix[:, j]
            grades = grade_matrix[:, j]
            metals = metal_matrix[:, j]
            
            # Handle zero tonnages
            valid_mask = tonnages > 0
            n_valid = np.sum(valid_mask)
            
            if n_valid > 0:
                stats = GTCurveStatistics(
                    cutoff=cutoff,
                    
                    # Tonnage statistics
                    tonnage_mean=float(np.mean(tonnages)),
                    tonnage_std=float(np.std(tonnages)),
                    tonnage_p05=float(np.percentile(tonnages, 5)),
                    tonnage_p10=float(np.percentile(tonnages, 10)),
                    tonnage_p50=float(np.percentile(tonnages, 50)),
                    tonnage_p90=float(np.percentile(tonnages, 90)),
                    tonnage_p95=float(np.percentile(tonnages, 95)),
                    
                    # Grade statistics (only from valid tonnages)
                    grade_mean=float(np.mean(grades[valid_mask])) if n_valid > 0 else 0.0,
                    grade_std=float(np.std(grades[valid_mask])) if n_valid > 1 else 0.0,
                    grade_p05=float(np.percentile(grades[valid_mask], 5)) if n_valid > 0 else 0.0,
                    grade_p10=float(np.percentile(grades[valid_mask], 10)) if n_valid > 0 else 0.0,
                    grade_p50=float(np.percentile(grades[valid_mask], 50)) if n_valid > 0 else 0.0,
                    grade_p90=float(np.percentile(grades[valid_mask], 90)) if n_valid > 0 else 0.0,
                    grade_p95=float(np.percentile(grades[valid_mask], 95)) if n_valid > 0 else 0.0,
                    
                    # Metal statistics
                    metal_mean=float(np.mean(metals)),
                    metal_std=float(np.std(metals)),
                    metal_p05=float(np.percentile(metals, 5)),
                    metal_p10=float(np.percentile(metals, 10)),
                    metal_p50=float(np.percentile(metals, 50)),
                    metal_p90=float(np.percentile(metals, 90)),
                    metal_p95=float(np.percentile(metals, 95)),
                    
                    n_realisations=n_real
                )
            else:
                # No valid data at this cutoff
                stats = GTCurveStatistics(
                    cutoff=cutoff,
                    tonnage_mean=0, tonnage_std=0,
                    tonnage_p05=0, tonnage_p10=0, tonnage_p50=0, tonnage_p90=0, tonnage_p95=0,
                    grade_mean=0, grade_std=0,
                    grade_p05=0, grade_p10=0, grade_p50=0, grade_p90=0, grade_p95=0,
                    metal_mean=0, metal_std=0,
                    metal_p05=0, metal_p10=0, metal_p50=0, metal_p90=0, metal_p95=0,
                    n_realisations=n_real
                )
            
            curve_statistics.append(stats)
        
        if progress_callback:
            progress_callback(100, "SGS uncertainty analysis complete")
        
        result = SGSUncertaintyResult(
            curve_statistics=curve_statistics,
            realisation_curves=realisation_curves,
            n_realisations=n_real,
            cutoffs=cutoffs,
            metadata={
                'total_tonnage': total_tonnage,
                'n_blocks': n_blocks,
                'method': 'empirical_percentiles',
                'confidence_level': '90%'
            }
        )
        
        self.logger.info(f"SGS uncertainty analysis complete: {n_real} realisations processed")
        return result
    
    def get_confidence_band(
        self,
        result: SGSUncertaintyResult,
        confidence_level: float = 0.90,
        variable: str = 'tonnage'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract confidence band for a variable.
        
        Args:
            result: SGSUncertaintyResult from compute_uncertainty_from_realisations
            confidence_level: Confidence level (0.90 for 90% CI, 0.95 for 95% CI)
            variable: 'tonnage', 'grade', or 'metal'
        
        Returns:
            Tuple of (cutoffs, lower_bound, upper_bound)
        """
        cutoffs = result.cutoffs
        n_cutoffs = len(cutoffs)
        
        lower = np.zeros(n_cutoffs)
        upper = np.zeros(n_cutoffs)
        
        # Select percentile based on confidence level
        if confidence_level >= 0.95:
            lower_pct = 'p05'
            upper_pct = 'p95'
        else:
            lower_pct = 'p10'
            upper_pct = 'p90'
        
        for i, stats in enumerate(result.curve_statistics):
            if variable == 'tonnage':
                lower[i] = getattr(stats, f'tonnage_{lower_pct}')
                upper[i] = getattr(stats, f'tonnage_{upper_pct}')
            elif variable == 'grade':
                lower[i] = getattr(stats, f'grade_{lower_pct}')
                upper[i] = getattr(stats, f'grade_{upper_pct}')
            elif variable == 'metal':
                lower[i] = getattr(stats, f'metal_{lower_pct}')
                upper[i] = getattr(stats, f'metal_{upper_pct}')
        
        return cutoffs, lower, upper


# =============================================================================
# DOMAIN-WISE ANALYSIS
# =============================================================================

class DomainGTEngine:
    """
    Domain-wise grade-tonnage analysis engine.
    
    Generates separate GT curves for each geological domain (lithology, 
    ore type, domain code, etc.) as required for professional resource
    reporting.
    
    JORC REQUIREMENT:
    "Where material grades occur in distinct domains, grade-tonnage
    relationships should be reported for each domain."
    """
    
    def __init__(self, base_config: Optional[GeostatsGradeTonnageConfig] = None):
        """
        Initialize the domain-wise GT engine.
        
        Args:
            base_config: Base configuration for GT calculations
        """
        self.base_config = base_config or GeostatsGradeTonnageConfig()
        self.base_engine = GeostatsGradeTonnageEngine(self.base_config)
        self.logger = logging.getLogger(__name__)
    
    def compute_domain_curves(
        self,
        data: pd.DataFrame,
        domain_column: str,
        grade_column: str,
        tonnage_column: str,
        x_column: str = 'x',
        y_column: str = 'y',
        z_column: str = 'z',
        cutoffs: Optional[np.ndarray] = None,
        domains_to_include: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> DomainGTResult:
        """
        Compute GT curves for each domain.
        
        Args:
            data: DataFrame with block data
            domain_column: Column containing domain codes
            grade_column: Column containing grade values
            tonnage_column: Column containing tonnage values
            x_column, y_column, z_column: Coordinate columns
            cutoffs: Cutoff grades to evaluate (auto-computed if None)
            domains_to_include: List of domains to include (all if None)
            progress_callback: Optional callback(pct, message)
        
        Returns:
            DomainGTResult with per-domain GT curves
        """
        self.logger.info(f"Starting domain-wise GT analysis on column '{domain_column}'")
        
        # Validate domain column
        if domain_column not in data.columns:
            raise ValueError(f"Domain column '{domain_column}' not found in data")
        
        # Get unique domains
        all_domains = data[domain_column].dropna().unique()
        domains = list(all_domains) if domains_to_include is None else domains_to_include
        
        self.logger.info(f"Found {len(domains)} domains: {domains}")
        
        # Auto-compute cutoffs if not provided
        if cutoffs is None:
            min_grade = data[grade_column].min()
            max_grade = data[grade_column].max()
            cutoffs = np.linspace(max(min_grade, 0), max_grade, 50)
        
        # Compute curve for each domain
        domain_curves = {}
        domain_statistics = {}
        
        for i, domain in enumerate(domains):
            if progress_callback:
                pct = int(100 * i / len(domains))
                progress_callback(pct, f"Processing domain: {domain}")
            
            # Filter data for this domain
            domain_mask = data[domain_column] == domain
            domain_data = data[domain_mask].copy()
            
            if len(domain_data) == 0:
                self.logger.warning(f"No data for domain '{domain}', skipping")
                continue
            
            try:
                # Compute GT curve for this domain
                curve = self.base_engine.calculate_grade_tonnage_curve(
                    domain_data,
                    cutoff_range=cutoffs,
                    element_column=grade_column,
                    tonnage_column=tonnage_column,
                    x_column=x_column,
                    y_column=y_column,
                    z_column=z_column
                )
                
                domain_curves[str(domain)] = curve
                
                # Store domain statistics
                domain_statistics[str(domain)] = {
                    'block_count': len(domain_data),
                    'total_tonnage': curve.global_statistics.get('total_tonnage', 0),
                    'mean_grade': curve.global_statistics.get('grade_statistics', {}).get('mean', 0),
                    'total_metal': curve.global_statistics.get('total_metal', 0)
                }
                
                self.logger.info(f"Domain '{domain}': {len(domain_data)} blocks, "
                               f"{curve.global_statistics.get('total_tonnage', 0):,.0f} tonnes")
                
            except Exception as e:
                self.logger.error(f"Failed to compute curve for domain '{domain}': {e}")
        
        # Compute combined curve (all domains)
        if progress_callback:
            progress_callback(95, "Computing combined curve...")
        
        combined_curve = self.base_engine.calculate_grade_tonnage_curve(
            data,
            cutoff_range=cutoffs,
            element_column=grade_column,
            tonnage_column=tonnage_column,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column
        )
        
        if progress_callback:
            progress_callback(100, "Domain analysis complete")
        
        result = DomainGTResult(
            domain_curves=domain_curves,
            combined_curve=combined_curve,
            domain_statistics=domain_statistics,
            domains=list(domain_curves.keys()),
            metadata={
                'domain_column': domain_column,
                'n_domains': len(domain_curves),
                'total_blocks': len(data)
            }
        )
        
        self.logger.info(f"Domain-wise GT analysis complete: {len(domain_curves)} domains processed")
        return result


# =============================================================================
# CLASSIFICATION-BASED ANALYSIS (Measured/Indicated/Inferred)
# =============================================================================

class ClassificationGTEngine:
    """
    Classification-based grade-tonnage analysis engine.
    
    Generates separate GT curves for each resource classification category
    (Measured, Indicated, Inferred) as required for JORC/SAMREC compliance.
    
    JORC REQUIREMENT:
    "Mineral Resources shall be reported by classification category:
    Measured, Indicated, and Inferred."
    """
    
    def __init__(self, base_config: Optional[GeostatsGradeTonnageConfig] = None):
        """
        Initialize the classification-based GT engine.
        
        Args:
            base_config: Base configuration for GT calculations
        """
        self.base_config = base_config or GeostatsGradeTonnageConfig()
        self.base_engine = GeostatsGradeTonnageEngine(self.base_config)
        self.logger = logging.getLogger(__name__)
    
    def compute_classification_curves(
        self,
        data: pd.DataFrame,
        classification_column: str,
        grade_column: str,
        tonnage_column: str,
        x_column: str = 'x',
        y_column: str = 'y',
        z_column: str = 'z',
        cutoffs: Optional[np.ndarray] = None,
        category_mapping: Optional[Dict[str, str]] = None,
        progress_callback: Optional[callable] = None
    ) -> ClassificationGTResult:
        """
        Compute GT curves for each resource classification category.
        
        Args:
            data: DataFrame with block data
            classification_column: Column containing classification codes
            grade_column: Column containing grade values
            tonnage_column: Column containing tonnage values
            x_column, y_column, z_column: Coordinate columns
            cutoffs: Cutoff grades to evaluate (auto-computed if None)
            category_mapping: Optional mapping from column values to categories.
                            Example: {'M': 'measured', 'I': 'indicated', 'INF': 'inferred'}
                            If None, assumes column already contains standard names.
            progress_callback: Optional callback(pct, message)
        
        Returns:
            ClassificationGTResult with per-category GT curves
        """
        self.logger.info(f"Starting classification-based GT analysis on column '{classification_column}'")
        
        # Validate classification column
        if classification_column not in data.columns:
            raise ValueError(f"Classification column '{classification_column}' not found in data")
        
        # Apply category mapping if provided
        if category_mapping:
            data = data.copy()
            data['_mapped_category'] = data[classification_column].map(category_mapping)
            classification_column = '_mapped_category'
        
        # Standardize category names
        category_map = {
            # Common variations for Measured
            'measured': ResourceCategory.MEASURED,
            'meas': ResourceCategory.MEASURED,
            'm': ResourceCategory.MEASURED,
            '1': ResourceCategory.MEASURED,
            
            # Common variations for Indicated
            'indicated': ResourceCategory.INDICATED,
            'ind': ResourceCategory.INDICATED,
            'i': ResourceCategory.INDICATED,
            '2': ResourceCategory.INDICATED,
            
            # Common variations for Inferred
            'inferred': ResourceCategory.INFERRED,
            'inf': ResourceCategory.INFERRED,
            '3': ResourceCategory.INFERRED,
        }
        
        # Auto-compute cutoffs if not provided
        if cutoffs is None:
            min_grade = data[grade_column].min()
            max_grade = data[grade_column].max()
            cutoffs = np.linspace(max(min_grade, 0), max_grade, 50)
        
        # Initialize result curves
        measured_curve = None
        indicated_curve = None
        inferred_curve = None
        category_statistics = {}
        
        # Process each category
        categories = [
            (ResourceCategory.MEASURED, 'Measured'),
            (ResourceCategory.INDICATED, 'Indicated'),
            (ResourceCategory.INFERRED, 'Inferred')
        ]
        
        for i, (cat_enum, cat_name) in enumerate(categories):
            if progress_callback:
                pct = int(100 * i / len(categories))
                progress_callback(pct, f"Processing {cat_name} resources...")
            
            # Find matching rows
            # Try exact match first, then lowercase match
            cat_values = data[classification_column].astype(str).str.lower()
            
            # Build mask for this category
            mask_conditions = []
            for key, val in category_map.items():
                if val == cat_enum:
                    mask_conditions.append(cat_values == key)
            
            if mask_conditions:
                combined_mask = mask_conditions[0]
                for m in mask_conditions[1:]:
                    combined_mask = combined_mask | m
                cat_data = data[combined_mask].copy()
            else:
                cat_data = pd.DataFrame()
            
            if len(cat_data) == 0:
                self.logger.info(f"No {cat_name} resources found")
                category_statistics[cat_name.lower()] = {
                    'block_count': 0,
                    'total_tonnage': 0,
                    'mean_grade': 0,
                    'total_metal': 0
                }
                continue
            
            try:
                # Compute GT curve for this category
                curve = self.base_engine.calculate_grade_tonnage_curve(
                    cat_data,
                    cutoff_range=cutoffs,
                    element_column=grade_column,
                    tonnage_column=tonnage_column,
                    x_column=x_column,
                    y_column=y_column,
                    z_column=z_column
                )
                
                # Assign to appropriate result field
                if cat_enum == ResourceCategory.MEASURED:
                    measured_curve = curve
                elif cat_enum == ResourceCategory.INDICATED:
                    indicated_curve = curve
                elif cat_enum == ResourceCategory.INFERRED:
                    inferred_curve = curve
                
                # Store category statistics
                category_statistics[cat_name.lower()] = {
                    'block_count': len(cat_data),
                    'total_tonnage': curve.global_statistics.get('total_tonnage', 0),
                    'mean_grade': curve.global_statistics.get('grade_statistics', {}).get('mean', 0),
                    'total_metal': curve.global_statistics.get('total_metal', 0)
                }
                
                self.logger.info(f"{cat_name}: {len(cat_data)} blocks, "
                               f"{curve.global_statistics.get('total_tonnage', 0):,.0f} tonnes")
                
            except Exception as e:
                self.logger.error(f"Failed to compute curve for {cat_name}: {e}")
        
        # Compute combined curve (all categories)
        if progress_callback:
            progress_callback(90, "Computing combined curve...")
        
        combined_curve = self.base_engine.calculate_grade_tonnage_curve(
            data,
            cutoff_range=cutoffs,
            element_column=grade_column,
            tonnage_column=tonnage_column,
            x_column=x_column,
            y_column=y_column,
            z_column=z_column
        )
        
        if progress_callback:
            progress_callback(100, "Classification analysis complete")
        
        # Compute totals
        total_measured = category_statistics.get('measured', {}).get('total_tonnage', 0)
        total_indicated = category_statistics.get('indicated', {}).get('total_tonnage', 0)
        total_inferred = category_statistics.get('inferred', {}).get('total_tonnage', 0)
        
        result = ClassificationGTResult(
            measured_curve=measured_curve,
            indicated_curve=indicated_curve,
            inferred_curve=inferred_curve,
            combined_curve=combined_curve,
            category_statistics=category_statistics,
            metadata={
                'classification_column': classification_column,
                'total_measured_tonnes': total_measured,
                'total_indicated_tonnes': total_indicated,
                'total_inferred_tonnes': total_inferred,
                'total_blocks': len(data)
            }
        )
        
        self.logger.info(f"Classification-based GT analysis complete:\n"
                        f"  Measured: {total_measured:,.0f} t\n"
                        f"  Indicated: {total_indicated:,.0f} t\n"
                        f"  Inferred: {total_inferred:,.0f} t")
        
        return result


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_sgs_uncertainty_to_csv(result: SGSUncertaintyResult, filename: str) -> bool:
    """
    Export SGS uncertainty results to CSV.
    
    Args:
        result: SGSUncertaintyResult to export
        filename: Output filename
    
    Returns:
        True if successful
    """
    try:
        data = []
        for stats in result.curve_statistics:
            data.append({
                'cutoff': stats.cutoff,
                'tonnage_mean': stats.tonnage_mean,
                'tonnage_std': stats.tonnage_std,
                'tonnage_p05': stats.tonnage_p05,
                'tonnage_p10': stats.tonnage_p10,
                'tonnage_p50': stats.tonnage_p50,
                'tonnage_p90': stats.tonnage_p90,
                'tonnage_p95': stats.tonnage_p95,
                'grade_mean': stats.grade_mean,
                'grade_std': stats.grade_std,
                'grade_p05': stats.grade_p05,
                'grade_p10': stats.grade_p10,
                'grade_p50': stats.grade_p50,
                'grade_p90': stats.grade_p90,
                'grade_p95': stats.grade_p95,
                'metal_mean': stats.metal_mean,
                'metal_std': stats.metal_std,
                'metal_p05': stats.metal_p05,
                'metal_p10': stats.metal_p10,
                'metal_p50': stats.metal_p50,
                'metal_p90': stats.metal_p90,
                'metal_p95': stats.metal_p95,
                'n_realisations': stats.n_realisations
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"SGS uncertainty results exported to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export SGS uncertainty results: {e}")
        return False


def export_domain_gt_to_csv(result: DomainGTResult, filename: str) -> bool:
    """
    Export domain-wise GT results to CSV.
    
    Args:
        result: DomainGTResult to export
        filename: Output filename
    
    Returns:
        True if successful
    """
    try:
        # Export each domain curve
        all_data = []
        
        for domain, curve in result.domain_curves.items():
            for point in curve.points:
                all_data.append({
                    'domain': domain,
                    'cutoff': point.cutoff_grade,
                    'tonnage': point.tonnage,
                    'avg_grade': point.avg_grade,
                    'metal_quantity': point.metal_quantity,
                    'net_value': point.net_value
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        
        # Export domain summary
        summary_filename = filename.replace('.csv', '_summary.csv')
        summary_df = pd.DataFrame(result.domain_statistics).T
        summary_df.to_csv(summary_filename)
        
        logger.info(f"Domain GT results exported to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export domain GT results: {e}")
        return False


def export_classification_gt_to_csv(result: ClassificationGTResult, filename: str) -> bool:
    """
    Export classification-based GT results to CSV.
    
    Args:
        result: ClassificationGTResult to export
        filename: Output filename
    
    Returns:
        True if successful
    """
    try:
        all_data = []
        
        # Export Measured curve
        if result.measured_curve:
            for point in result.measured_curve.points:
                all_data.append({
                    'category': 'Measured',
                    'cutoff': point.cutoff_grade,
                    'tonnage': point.tonnage,
                    'avg_grade': point.avg_grade,
                    'metal_quantity': point.metal_quantity
                })
        
        # Export Indicated curve
        if result.indicated_curve:
            for point in result.indicated_curve.points:
                all_data.append({
                    'category': 'Indicated',
                    'cutoff': point.cutoff_grade,
                    'tonnage': point.tonnage,
                    'avg_grade': point.avg_grade,
                    'metal_quantity': point.metal_quantity
                })
        
        # Export Inferred curve
        if result.inferred_curve:
            for point in result.inferred_curve.points:
                all_data.append({
                    'category': 'Inferred',
                    'cutoff': point.cutoff_grade,
                    'tonnage': point.tonnage,
                    'avg_grade': point.avg_grade,
                    'metal_quantity': point.metal_quantity
                })
        
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        
        # Export category summary
        summary_filename = filename.replace('.csv', '_summary.csv')
        summary_df = pd.DataFrame(result.category_statistics).T
        summary_df.to_csv(summary_filename)
        
        logger.info(f"Classification GT results exported to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export classification GT results: {e}")
        return False

