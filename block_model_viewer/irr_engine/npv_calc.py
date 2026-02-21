"""
NPV Calculation Module (High-Performance)

Computes Net Present Value (NPV) and IRR with Numba acceleration.

Updated 2025-12:
- Added multiple IRR detection (Violation #2 fix)
- Added cash flow validation
- Enhanced IRR result metadata
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple, Any
import logging

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)

# --- NUMBA KERNELS ---

@jit(nopython=True, cache=True)
def _numba_discount_cashflows(cashflows: np.ndarray, rate: float) -> float:
    """Fast DCF calculation."""
    npv = 0.0
    # 1 / (1+r)^t calculation optimization
    discount_factor = 1.0
    rate_factor = 1.0 / (1.0 + rate)
    
    for t in range(len(cashflows)):
        npv += cashflows[t] * discount_factor
        discount_factor *= rate_factor
    return npv

@jit(nopython=True, cache=True)
def _numba_calculate_irr(cashflows: np.ndarray, guess: float, tol: float, max_iter: int) -> float:
    """Newton-Raphson IRR calculation."""
    r = guess
    for _ in range(max_iter):
        npv = 0.0
        d_npv = 0.0
        
        factor = 1.0
        rate_inv = 1.0 / (1.0 + r)
        
        for t in range(len(cashflows)):
            # NPV term: CF / (1+r)^t
            term = cashflows[t] * factor
            npv += term
            
            # Derivative term: -t * CF / (1+r)^(t+1)
            # d_factor/dr of (1+r)^-t is -t * (1+r)^(-t-1)
            d_npv += -t * term * rate_inv
            
            factor *= rate_inv
            
        if abs(npv) < tol:
            return r
            
        if abs(d_npv) < 1e-15:
            return -999.0 # Failed to converge (derivative zero)
            
        new_r = r - npv / d_npv
        
        if abs(new_r - r) < tol:
            return new_r
            
        r = new_r
        
    return -999.0 # Failed

# --- PYTHON API ---

def discount_cashflows(cashflows: np.ndarray, discount_rate: float, time_periods: Optional[np.ndarray] = None) -> float:
    """
    Discount a series of cashflows to present value.
    
    Args:
        cashflows: Array of cashflows for each period
        discount_rate: Annual discount rate (e.g., 0.10 for 10%)
        time_periods: Optional array of time periods (defaults to 0, 1, 2, ...)
                      Note: If provided, the Numba kernel uses sequential periods
        
    Returns:
        Net Present Value (NPV)
    """
    # Ensure float array
    cfs = np.ascontiguousarray(cashflows, dtype=np.float64)
    
    # If time_periods provided and not sequential, use original method
    if time_periods is not None and not np.array_equal(time_periods, np.arange(len(cashflows))):
        discount_factors = 1.0 / np.power(1.0 + discount_rate, time_periods)
        return np.sum(cfs * discount_factors)
    
    return _numba_discount_cashflows(cfs, float(discount_rate))

def count_sign_changes(cashflows: np.ndarray, tolerance: float = 1e-6) -> int:
    """
    Count the number of sign changes in cash flow series.
    
    Multiple sign changes indicate potential multiple IRR solutions.
    
    Args:
        cashflows: Array of cash flows
        tolerance: Values smaller than this are treated as zero
        
    Returns:
        Number of sign changes (0 = no sign changes, 1 = standard project, >1 = multiple IRR risk)
    """
    # Filter out near-zero values
    significant_cf = cashflows.copy()
    significant_cf[np.abs(significant_cf) < tolerance] = 0
    
    # Get signs of non-zero values
    non_zero_mask = significant_cf != 0
    if not np.any(non_zero_mask):
        return 0
    
    signs = np.sign(significant_cf[non_zero_mask])
    return int(np.sum(np.diff(signs) != 0))


def detect_multiple_irr(
    cashflows: np.ndarray,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Detect if cash flow pattern suggests multiple IRR solutions.
    
    Args:
        cashflows: Array of cash flows by period
        tolerance: Values smaller than this are treated as zero
        
    Returns:
        Dictionary with analysis results:
        - sign_changes: Number of sign changes
        - has_multiple_irr_risk: True if >1 sign changes
        - pattern: String representation of cash flow signs
        - recommendation: Guidance for interpretation
    """
    sign_changes = count_sign_changes(cashflows, tolerance)
    
    # Build pattern string (first 10 periods)
    pattern_map = {-1: '-', 0: '0', 1: '+'}
    pattern = ' '.join([pattern_map[int(s)] for s in np.sign(cashflows[:min(10, len(cashflows))])])
    if len(cashflows) > 10:
        pattern += ' ...'
    
    has_risk = sign_changes > 1
    
    if sign_changes == 0:
        recommendation = "All cash flows have same sign - IRR may be undefined"
    elif sign_changes == 1:
        recommendation = "Standard project pattern - single IRR exists"
    else:
        recommendation = (
            f"WARNING: {sign_changes} sign changes detected - MULTIPLE IRRs MAY EXIST. "
            "Consider using NPV or Modified IRR (MIRR) instead."
        )
    
    return {
        'sign_changes': sign_changes,
        'has_multiple_irr_risk': has_risk,
        'pattern': pattern,
        'recommendation': recommendation
    }


def calculate_irr(
    cashflows: np.ndarray, 
    initial_guess: float = 0.1, 
    tolerance: float = 1e-6, 
    max_iterations: int = 100,
    check_multiple_irr: bool = True
) -> Optional[float]:
    """
    Calculate Internal Rate of Return (IRR).
    
    Args:
        cashflows: Array of cash flows (period 0 = initial investment)
        initial_guess: Starting rate for Newton-Raphson
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        check_multiple_irr: If True, log warning for multiple sign changes
        
    Returns:
        IRR as decimal (0.1 = 10%), or None if calculation fails
    """
    cfs = np.ascontiguousarray(cashflows, dtype=np.float64)
    
    # Check for multiple IRR risk
    if check_multiple_irr:
        irr_check = detect_multiple_irr(cfs, tolerance)
        if irr_check['has_multiple_irr_risk']:
            logger.warning(
                f"MULTIPLE IRR WARNING: {irr_check['sign_changes']} sign changes detected. "
                f"Cash flow pattern: {irr_check['pattern']}. "
                "The calculated IRR may be one of multiple solutions."
            )
    
    result = _numba_calculate_irr(cfs, initial_guess, tolerance, max_iterations)
    
    if result == -999.0 or result < -0.99 or result > 100.0:
        return None
    return result


def calculate_irr_with_metadata(
    cashflows: np.ndarray,
    initial_guess: float = 0.1,
    tolerance: float = 1e-6,
    max_iterations: int = 100
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Calculate IRR with comprehensive metadata for audit trail.
    
    Args:
        cashflows: Array of cash flows
        initial_guess: Starting rate
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        Tuple of (irr_value, metadata_dict)
    """
    cfs = np.ascontiguousarray(cashflows, dtype=np.float64)
    
    # Analyze cash flow pattern
    irr_analysis = detect_multiple_irr(cfs, tolerance)
    
    # Calculate IRR
    result = _numba_calculate_irr(cfs, initial_guess, tolerance, max_iterations)
    
    if result == -999.0 or result < -0.99 or result > 100.0:
        irr_value = None
        convergence_status = 'failed'
    else:
        irr_value = result
        convergence_status = 'converged'
    
    metadata = {
        'irr': irr_value,
        'convergence_status': convergence_status,
        'initial_guess': initial_guess,
        'tolerance': tolerance,
        'max_iterations': max_iterations,
        'num_periods': len(cashflows),
        'total_cashflow': float(np.sum(cashflows)),
        'net_initial_investment': float(cashflows[0]) if len(cashflows) > 0 else 0,
        **irr_analysis,
        'irr_is_reliable': (
            convergence_status == 'converged' and 
            not irr_analysis['has_multiple_irr_risk']
        )
    }
    
    if irr_analysis['has_multiple_irr_risk']:
        metadata['warning'] = 'MULTIPLE_IRR_POSSIBLE'
        logger.warning(f"IRR result may be unreliable: {irr_analysis['recommendation']}")
    
    return irr_value, metadata

def calculate_npv(
    schedule: pd.DataFrame,
    block_model: pd.DataFrame,
    economic_params: Dict,
    discount_rate: float
) -> Dict:
    """
    Calculate detailed NPV from a mining schedule.
    
    Args:
        schedule: DataFrame ['BLOCK_ID', 'PERIOD', 'MINED']
        block_model: DataFrame ['BLOCK_ID', 'TONNAGE', 'GRADE', 'VALUE']
        economic_params: Dict with costs/prices
        discount_rate: Annual rate (0.10)
    """
    # 1. Merge Schedule and Model
    # Optimization: Filter only mined blocks first
    mined_schedule = schedule[schedule['MINED'] == 1]
    
    # Fast merge using numpy if indices match, otherwise standard pandas merge
    # Assuming standard merge for safety
    cols_needed = ['BLOCK_ID', 'TONNAGE', 'GRADE']
    
    # Check if we handle by-products
    by_products = economic_params.get('by_products', [])
    for bp in by_products:
        if bp['grade_field'] in block_model.columns:
            cols_needed.append(bp['grade_field'])

    merged = pd.merge(
        mined_schedule[['BLOCK_ID', 'PERIOD']], 
        block_model[cols_needed],
        on='BLOCK_ID',
        how='left'
    )
    
    # 2. Vectorized aggregation by Period
    # Group by PERIOD and sum properties
    # This is much faster than iterating periods in Python
    agg_rules = {'TONNAGE': 'sum', 'GRADE': 'mean'} # Weighted mean needed for grade?
    
    # Correct Weighted Average Grade Calculation
    # We calculate metal content first, then aggregate
    merged['METAL_CONTENT'] = merged['TONNAGE'] * merged['GRADE']
    
    agg_dict = {
        'TONNAGE': 'sum',
        'METAL_CONTENT': 'sum'
    }
    
    # Add by-products
    for bp in by_products:
        col = bp['grade_field']
        if col in merged.columns:
            merged[f'METAL_{col}'] = merged['TONNAGE'] * merged[col]
            agg_dict[f'METAL_{col}'] = 'sum'

    periodic_data = merged.groupby('PERIOD').agg(agg_dict).sort_index()
    
    # 3. Calculate Cashflows
    max_period = int(periodic_data.index.max()) if not periodic_data.empty else 0
    cashflows = np.zeros(max_period + 1)
    
    # Map periodic data to array
    periods = periodic_data.index.values.astype(int)
    tonnage = periodic_data['TONNAGE'].values
    metal_primary = periodic_data['METAL_CONTENT'].values * economic_params['recovery']
    
    # Revenue
    revenue = metal_primary * economic_params['metal_price']
    
    # By-products revenue
    for bp in by_products:
        col = f'METAL_{bp["grade_field"]}'
        if col in periodic_data.columns:
            metal_bp = periodic_data[col].values * bp['recovery']
            rev_bp = metal_bp * bp['price']
            if 'selling_cost' in bp:
                rev_bp -= metal_bp * bp['selling_cost']
            revenue += rev_bp
            
    # Costs
    mining_cost = tonnage * economic_params['mining_cost']
    proc_cost = tonnage * economic_params['processing_cost']
    sell_cost = metal_primary * economic_params.get('selling_cost', 0.0)
    
    opex = mining_cost + proc_cost + sell_cost
    
    # CAPEX (Fixed array injection)
    capex_arr = np.zeros(max_period + 1)
    if 'capex' in economic_params:
        usr_capex = economic_params['capex']
        length = min(len(usr_capex), len(capex_arr))
        capex_arr[:length] = usr_capex[:length]
        
    # Net Cashflow
    # Be careful with index mapping
    for i, p in enumerate(periods):
        if p < len(cashflows):
            cashflows[p] = revenue[i] - opex[i]
            
    cashflows -= capex_arr
    
    # 4. Final NPV
    npv = discount_cashflows(cashflows, discount_rate)
    
    return {
        'npv': npv,
        'cashflows': cashflows,
        'revenue': np.sum(revenue),
        'operating_cost': np.sum(opex),
        'capital_cost': np.sum(capex_arr),
        'primary_revenue': np.sum(metal_primary * economic_params['metal_price']),
        'byproduct_revenue': np.sum(revenue) - np.sum(metal_primary * economic_params['metal_price']),
        'periods': max_period + 1
    }

def calculate_block_value(
    tonnage: float,
    grade: float,
    metal_price: float,
    mining_cost: float,
    processing_cost: float,
    recovery: float = 1.0
) -> float:
    """
    Calculate the economic value of a single block.
    
    Args:
        tonnage: Block tonnage (tonnes)
        grade: Metal grade (e.g., g/t, % Cu)
        metal_price: Price per unit of metal
        mining_cost: Cost per tonne to mine
        processing_cost: Cost per tonne to process
        recovery: Metallurgical recovery (0-1)
        
    Returns:
        Block value ($/block)
    """
    metal_content = tonnage * grade * recovery
    revenue = metal_content * metal_price
    cost = tonnage * (mining_cost + processing_cost)
    
    return revenue - cost

def calculate_block_value_geomet(
    grade_dict: Dict[str, np.ndarray],
    recovery_dict: Dict[str, np.ndarray],
    prices: Dict[str, float],
    costs: Dict[str, float],
    tonnage: np.ndarray
) -> np.ndarray:
    """
    Calculate block values accounting for plant recoveries per element (STEP 28).
    
    Args:
        grade_dict: Dictionary mapping element name -> grade array
        recovery_dict: Dictionary mapping element name -> recovery array (0-1)
        prices: Dictionary mapping element name -> price per unit
        costs: Dictionary with 'mining_cost' and 'processing_cost' per tonne
        tonnage: Array of block tonnages
        
    Returns:
        Array of block values ($/block)
    """
    n_blocks = len(tonnage)
    block_values = np.zeros(n_blocks)
    
    # Calculate revenue from each element
    for element, grades in grade_dict.items():
        if element not in recovery_dict or element not in prices:
            continue
        
        recoveries = recovery_dict[element]
        price = prices[element]
        
        # Metal produced = tonnage × grade × recovery
        metal_produced = tonnage * grades * recoveries
        revenue = metal_produced * price
        block_values += revenue
    
    # Calculate costs
    mining_cost = costs.get('mining_cost', 0.0)
    processing_cost = costs.get('processing_cost', 0.0)
    total_cost = tonnage * (mining_cost + processing_cost)
    
    # Net value
    block_values -= total_cost
    
    return block_values
