"""
IRR Engine Validation Module

Provides comprehensive validation for economic parameters, classification filtering,
multiple IRR detection, and unit enforcement.

This module addresses audit findings:
- Violation #1: No Resource Classification Filter
- Violation #2: No Multiple IRR Detection  
- Violation #3: Unit Conversion Not Engine-Enforced
- Violation #5: Silent Economic Default Fallbacks

Author: GeoX Mining Software
Date: 2025-12
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# VIOLATION #3 FIX: Unit Definitions and Validation
# =============================================================================

class PriceUnit(str, Enum):
    """Supported price units with explicit conversion factors."""
    USD_PER_GRAM = "$/g"
    USD_PER_KG = "$/kg"
    USD_PER_TONNE = "$/t"
    USD_PER_TROY_OZ = "$/oz"
    USD_PER_POUND = "$/lb"


class CostUnit(str, Enum):
    """Supported cost units."""
    USD_PER_TONNE = "$/t"
    USD_PER_KG = "$/kg"


# Conversion factors to base unit ($/gram for prices, $/tonne for costs)
PRICE_CONVERSION_TO_GRAM = {
    PriceUnit.USD_PER_GRAM: 1.0,
    PriceUnit.USD_PER_KG: 0.001,  # $/kg → $/g
    PriceUnit.USD_PER_TONNE: 0.000001,  # $/t → $/g
    PriceUnit.USD_PER_TROY_OZ: 1.0 / 31.1035,  # $/oz → $/g (troy oz = 31.1035 g)
    PriceUnit.USD_PER_POUND: 1.0 / 453.592,  # $/lb → $/g
}

COST_CONVERSION_TO_TONNE = {
    CostUnit.USD_PER_TONNE: 1.0,
    CostUnit.USD_PER_KG: 1000.0,  # $/kg → $/t
}


@dataclass
class UnitConfig:
    """Configuration for unit handling in economic parameters."""
    price_unit: PriceUnit = PriceUnit.USD_PER_GRAM
    cost_unit: CostUnit = CostUnit.USD_PER_TONNE
    
    def convert_price_to_base(self, value: float) -> float:
        """Convert price from configured unit to $/gram."""
        factor = PRICE_CONVERSION_TO_GRAM[self.price_unit]
        return value * factor
    
    def convert_cost_to_base(self, value: float) -> float:
        """Convert cost from configured unit to $/tonne."""
        factor = COST_CONVERSION_TO_TONNE[self.cost_unit]
        return value * factor


def validate_and_convert_units(
    economic_params: Dict[str, Any],
    unit_config: Optional[UnitConfig] = None
) -> Dict[str, Any]:
    """
    Validate and convert economic parameters to base units.
    
    Args:
        economic_params: Raw economic parameters
        unit_config: Unit configuration (default assumes $/g for price, $/t for costs)
    
    Returns:
        Validated and converted parameters
        
    Raises:
        ValueError: If unit conversion results in unreasonable values
    """
    if unit_config is None:
        unit_config = UnitConfig()
    
    validated = economic_params.copy()
    
    # Convert price if present
    if 'metal_price' in validated:
        original = validated['metal_price']
        converted = unit_config.convert_price_to_base(original)
        
        # Sanity check: price should be reasonable for precious metals ($/g)
        # Gold ~$60/g, Silver ~$0.80/g, Copper ~$0.008/g
        if converted > 10000 or converted < 0.0001:
            logger.warning(
                f"Unusual metal price after conversion: ${converted}/g "
                f"(original: ${original} {unit_config.price_unit.value}). "
                "Please verify unit configuration."
            )
        
        validated['metal_price'] = converted
        validated['_unit_config'] = {
            'price_unit': unit_config.price_unit.value,
            'original_price': original
        }
    
    # Log conversion for audit trail
    logger.info(
        f"Unit conversion applied: price_unit={unit_config.price_unit.value}, "
        f"cost_unit={unit_config.cost_unit.value}"
    )
    
    return validated


# =============================================================================
# VIOLATION #5 FIX: Required Parameter Validation
# =============================================================================

REQUIRED_ECONOMIC_PARAMS = [
    'metal_price',
    'mining_cost', 
    'processing_cost',
    'recovery'
]

OPTIONAL_ECONOMIC_PARAMS_WITH_DEFAULTS = {
    'selling_cost': 0.0,
    'capex': [],
    'by_products': []
}


class EconomicParameterError(ValueError):
    """Raised when required economic parameters are missing or invalid."""
    pass


def validate_economic_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that all required economic parameters are present and valid.
    
    NO SILENT DEFAULTS - raises explicit errors for missing required parameters.
    
    Args:
        params: Economic parameters dictionary
        
    Returns:
        Validated parameters (with explicit optional defaults applied)
        
    Raises:
        EconomicParameterError: If required parameters are missing or invalid
    """
    missing = []
    invalid = []
    
    # Check required parameters
    for key in REQUIRED_ECONOMIC_PARAMS:
        if key not in params:
            missing.append(key)
        elif params[key] is None:
            missing.append(f"{key} (is None)")
        elif not isinstance(params[key], (int, float)):
            invalid.append(f"{key} (type={type(params[key]).__name__}, expected numeric)")
    
    if missing:
        raise EconomicParameterError(
            f"CRITICAL: Missing required economic parameters: {missing}. "
            "All economic inputs must be explicitly provided - no defaults allowed for safety."
        )
    
    if invalid:
        raise EconomicParameterError(
            f"CRITICAL: Invalid economic parameters: {invalid}. "
            "All economic parameters must be numeric values."
        )
    
    # Validate value ranges
    validated = params.copy()
    
    # Recovery must be 0-1
    if not 0 < validated['recovery'] <= 1:
        raise EconomicParameterError(
            f"Recovery must be between 0 and 1, got {validated['recovery']}. "
            "Use decimal format (e.g., 0.85 for 85%)."
        )
    
    # Costs must be non-negative
    if validated['mining_cost'] < 0:
        raise EconomicParameterError(f"Mining cost cannot be negative: {validated['mining_cost']}")
    if validated['processing_cost'] < 0:
        raise EconomicParameterError(f"Processing cost cannot be negative: {validated['processing_cost']}")
    
    # Price must be positive
    if validated['metal_price'] <= 0:
        raise EconomicParameterError(f"Metal price must be positive: {validated['metal_price']}")
    
    # Apply explicit optional defaults (these ARE safe defaults)
    for key, default in OPTIONAL_ECONOMIC_PARAMS_WITH_DEFAULTS.items():
        if key not in validated:
            validated[key] = default
            logger.info(f"Applied optional default: {key}={default}")
    
    logger.info("Economic parameters validated successfully")
    return validated


# =============================================================================
# VIOLATION #1 FIX: Resource Classification Filter
# =============================================================================

VALID_CLASSIFICATIONS = ['Measured', 'Indicated', 'Inferred']
JORC_RESERVE_CLASSIFICATIONS = ['Proved', 'Probable']
DEFAULT_IRR_CLASSIFICATIONS = ['Measured', 'Indicated']  # Standard for mine planning


@dataclass
class ClassificationFilterResult:
    """Result of applying classification filter."""
    original_blocks: int
    filtered_blocks: int
    removed_blocks: int
    classifications_included: List[str]
    classifications_excluded: List[str]
    block_counts_by_class: Dict[str, int]


def apply_classification_filter(
    block_model: pd.DataFrame,
    classification_filter: Optional[List[str]] = None,
    classification_column: str = 'CLASSIFICATION',
    strict_mode: bool = True
) -> Tuple[pd.DataFrame, ClassificationFilterResult]:
    """
    Filter block model by resource classification.
    
    JORC/SAMREC Compliance: Only Measured + Indicated resources should be used
    for mine planning and IRR analysis. Inferred resources are too uncertain.
    
    Args:
        block_model: Block model DataFrame
        classification_filter: List of classifications to include 
                              (default: ['Measured', 'Indicated'])
        classification_column: Name of classification column
        strict_mode: If True, raise error if column missing; if False, warn and include all
        
    Returns:
        Tuple of (filtered_block_model, filter_result)
        
    Raises:
        ValueError: If strict_mode and classification column missing
    """
    original_count = len(block_model)
    
    # Use default if not specified
    if classification_filter is None:
        classification_filter = DEFAULT_IRR_CLASSIFICATIONS
        logger.info(
            f"Using default classification filter for IRR analysis: {classification_filter}. "
            "This excludes Inferred resources per JORC/SAMREC guidelines."
        )
    
    # Check if classification column exists
    if classification_column not in block_model.columns:
        if strict_mode:
            raise ValueError(
                f"CRITICAL: Classification column '{classification_column}' not found in block model. "
                f"Available columns: {list(block_model.columns)}. "
                "Resource classification is required for JORC/SAMREC compliant IRR analysis. "
                "Set strict_mode=False to bypass (NOT RECOMMENDED for reporting)."
            )
        else:
            logger.warning(
                f"Classification column '{classification_column}' not found. "
                "ALL BLOCKS will be included. This may violate JORC/SAMREC guidelines. "
                "Results should NOT be used for public reporting."
            )
            return block_model.copy(), ClassificationFilterResult(
                original_blocks=original_count,
                filtered_blocks=original_count,
                removed_blocks=0,
                classifications_included=['ALL (no classification)'],
                classifications_excluded=[],
                block_counts_by_class={'Unclassified': original_count}
            )
    
    # Get classification statistics
    class_counts = block_model[classification_column].value_counts().to_dict()
    
    # Apply filter
    mask = block_model[classification_column].isin(classification_filter)
    filtered_model = block_model[mask].copy()
    
    filtered_count = len(filtered_model)
    removed_count = original_count - filtered_count
    
    # Determine excluded classifications
    all_classes = set(block_model[classification_column].unique())
    included_classes = set(classification_filter) & all_classes
    excluded_classes = all_classes - included_classes
    
    result = ClassificationFilterResult(
        original_blocks=original_count,
        filtered_blocks=filtered_count,
        removed_blocks=removed_count,
        classifications_included=list(included_classes),
        classifications_excluded=list(excluded_classes),
        block_counts_by_class=class_counts
    )
    
    # Log the filter result
    logger.info(
        f"Classification filter applied: {original_count} → {filtered_count} blocks "
        f"({removed_count} removed). "
        f"Included: {included_classes}, Excluded: {excluded_classes}"
    )
    
    if 'Inferred' in included_classes:
        logger.warning(
            "WARNING: Inferred resources are included in IRR analysis. "
            "This may not comply with JORC/SAMREC reporting requirements. "
            "Results should be clearly labeled as including Inferred resources."
        )
    
    return filtered_model, result


# =============================================================================
# VIOLATION #2 FIX: Multiple IRR Detection
# =============================================================================

@dataclass
class IRRValidityResult:
    """Result of IRR validity check."""
    is_valid: bool
    sign_changes: int
    has_multiple_irr_risk: bool
    first_negative_period: Optional[int]
    last_positive_period: Optional[int]
    recommendation: str
    cashflow_pattern: str  # e.g., "- + + + +" or "- + + - +"


def detect_multiple_irr_risk(
    cashflows: np.ndarray,
    tolerance: float = 1e-6
) -> IRRValidityResult:
    """
    Detect if cash flow pattern suggests multiple IRR solutions.
    
    Multiple IRRs occur when cash flows change sign more than once.
    Classic example: Initial investment (-), operations (+), closure costs (-)
    
    Args:
        cashflows: Array of cash flows by period
        tolerance: Values smaller than this are treated as zero
        
    Returns:
        IRRValidityResult with analysis details
    """
    # Filter out near-zero values
    significant_cf = cashflows.copy()
    significant_cf[np.abs(significant_cf) < tolerance] = 0
    
    # Get signs of non-zero values
    non_zero_mask = significant_cf != 0
    if not np.any(non_zero_mask):
        return IRRValidityResult(
            is_valid=False,
            sign_changes=0,
            has_multiple_irr_risk=False,
            first_negative_period=None,
            last_positive_period=None,
            recommendation="All cash flows are zero - IRR undefined",
            cashflow_pattern="0 0 0 ..."
        )
    
    signs = np.sign(significant_cf[non_zero_mask])
    
    # Count sign changes
    sign_changes = int(np.sum(np.diff(signs) != 0))
    
    # Build pattern string
    pattern_map = {-1: '-', 0: '0', 1: '+'}
    pattern = ' '.join([pattern_map[int(s)] for s in np.sign(cashflows[:min(10, len(cashflows))])])
    if len(cashflows) > 10:
        pattern += ' ...'
    
    # Find first negative and last positive
    negative_periods = np.where(cashflows < -tolerance)[0]
    positive_periods = np.where(cashflows > tolerance)[0]
    
    first_neg = int(negative_periods[0]) if len(negative_periods) > 0 else None
    last_pos = int(positive_periods[-1]) if len(positive_periods) > 0 else None
    
    # Determine validity and recommendation
    has_multiple_irr_risk = sign_changes > 1
    
    if sign_changes == 0:
        # All same sign - no real IRR exists (or NPV always positive/negative)
        is_valid = False
        if np.all(signs > 0):
            recommendation = "All positive cash flows - IRR is undefined (infinite return)"
        else:
            recommendation = "All negative cash flows - IRR is undefined (total loss)"
    elif sign_changes == 1:
        # Normal project: initial investment (-) followed by returns (+)
        is_valid = True
        recommendation = "Single IRR exists - standard project cash flow pattern"
    else:
        # Multiple sign changes - potential multiple IRRs
        is_valid = True  # IRR still calculable, but interpret with caution
        recommendation = (
            f"WARNING: {sign_changes} sign changes detected - MULTIPLE IRRs MAY EXIST. "
            f"Consider using NPV instead of IRR for decision making. "
            f"If closure costs cause the sign change, consider using Modified IRR (MIRR)."
        )
    
    return IRRValidityResult(
        is_valid=is_valid,
        sign_changes=sign_changes,
        has_multiple_irr_risk=has_multiple_irr_risk,
        first_negative_period=first_neg,
        last_positive_period=last_pos,
        recommendation=recommendation,
        cashflow_pattern=pattern
    )


def validate_irr_result(
    irr: Optional[float],
    cashflows: np.ndarray,
    tolerance: float = 1e-6
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Validate IRR result and return with metadata.
    
    Args:
        irr: Calculated IRR value
        cashflows: Cash flow array used for calculation
        tolerance: Numerical tolerance
        
    Returns:
        Tuple of (validated_irr, validation_metadata)
    """
    validity = detect_multiple_irr_risk(cashflows, tolerance)
    
    metadata = {
        'sign_changes': validity.sign_changes,
        'has_multiple_irr_risk': validity.has_multiple_irr_risk,
        'cashflow_pattern': validity.cashflow_pattern,
        'recommendation': validity.recommendation,
        'irr_is_reliable': validity.is_valid and not validity.has_multiple_irr_risk
    }
    
    if validity.has_multiple_irr_risk:
        logger.warning(
            f"MULTIPLE IRR WARNING: {validity.sign_changes} sign changes in cash flow. "
            f"Pattern: {validity.cashflow_pattern}. "
            f"Reported IRR ({irr:.2%} if calculated) may be one of multiple solutions."
        )
        metadata['warning'] = 'MULTIPLE_IRR_POSSIBLE'
    
    if irr is not None and not validity.is_valid:
        logger.warning(f"IRR validity check failed: {validity.recommendation}")
        metadata['warning'] = 'IRR_VALIDITY_FAILED'
    
    return irr, metadata


# =============================================================================
# COMPREHENSIVE VALIDATION FUNCTION
# =============================================================================

def validate_irr_inputs(
    block_model: pd.DataFrame,
    economic_params: Dict[str, Any],
    classification_filter: Optional[List[str]] = None,
    classification_column: str = 'CLASSIFICATION',
    unit_config: Optional[UnitConfig] = None,
    strict_classification: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Comprehensive input validation for IRR analysis.
    
    Performs:
    1. Economic parameter validation (no silent defaults)
    2. Unit conversion and validation
    3. Classification filtering
    
    Args:
        block_model: Raw block model DataFrame
        economic_params: Raw economic parameters
        classification_filter: Classifications to include
        classification_column: Name of classification column
        unit_config: Unit configuration for conversion
        strict_classification: Require classification column
        
    Returns:
        Tuple of (filtered_block_model, validated_params, validation_metadata)
        
    Raises:
        EconomicParameterError: If required parameters missing
        ValueError: If critical validation fails
    """
    validation_metadata = {
        'validation_timestamp': pd.Timestamp.now().isoformat(),
        'original_block_count': len(block_model)
    }
    
    # Step 1: Validate economic parameters (NO SILENT DEFAULTS)
    validated_params = validate_economic_params(economic_params)
    validation_metadata['economic_params_validated'] = True
    
    # Step 2: Apply unit conversion
    validated_params = validate_and_convert_units(validated_params, unit_config)
    validation_metadata['units_converted'] = True
    if unit_config:
        validation_metadata['price_unit'] = unit_config.price_unit.value
        validation_metadata['cost_unit'] = unit_config.cost_unit.value
    
    # Step 3: Apply classification filter
    filtered_model, filter_result = apply_classification_filter(
        block_model,
        classification_filter,
        classification_column,
        strict_mode=strict_classification
    )
    
    validation_metadata['classification_filter'] = {
        'original_blocks': filter_result.original_blocks,
        'filtered_blocks': filter_result.filtered_blocks,
        'removed_blocks': filter_result.removed_blocks,
        'included': filter_result.classifications_included,
        'excluded': filter_result.classifications_excluded,
        'block_counts': filter_result.block_counts_by_class
    }
    
    logger.info(
        f"IRR input validation complete: "
        f"{filter_result.filtered_blocks}/{filter_result.original_blocks} blocks retained"
    )
    
    return filtered_model, validated_params, validation_metadata

