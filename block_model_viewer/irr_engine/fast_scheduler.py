"""
Fast Scheduler (Heuristic).

A high-performance greedy scheduler using Numpy/Numba.
Optimized for running thousands of scenarios in Monte Carlo simulations.

Updated 2025-12:
- Added precedence validation (Violation #8 fix)
- Added schedule integrity checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

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

import logging

logger = logging.getLogger(__name__)


# =============================================================================
# VIOLATION #8 FIX: Precedence Validation
# =============================================================================

@dataclass
class PrecedenceViolation:
    """Record of a single precedence violation."""
    block_id: int
    block_period: int
    predecessor_id: int
    predecessor_period: int
    violation_type: str  # 'not_mined', 'mined_later', 'same_period'


@dataclass
class ScheduleValidationResult:
    """Result of schedule validation."""
    is_valid: bool
    total_violations: int
    violations: List[PrecedenceViolation]
    blocks_checked: int
    precedence_pairs_checked: int
    warning_message: Optional[str] = None


def build_vertical_precedence(
    block_model: pd.DataFrame,
    z_column: str = 'ZC',
    block_id_column: str = 'BLOCK_ID',
    x_column: str = 'XC',
    y_column: str = 'YC',
    bench_tolerance: float = 1.0
) -> Dict[int, List[int]]:
    """
    Build vertical precedence graph from block model.
    
    In open-pit mining, a block can only be mined if all blocks directly
    above it have been mined first.
    
    Args:
        block_model: Block model DataFrame
        z_column: Column name for elevation
        block_id_column: Column name for block IDs
        x_column: Column name for X coordinate
        y_column: Column name for Y coordinate
        bench_tolerance: Tolerance for matching X/Y coordinates
        
    Returns:
        Dict mapping block_id to list of predecessor block_ids (blocks that must be mined first)
    """
    precedence = {}
    
    # Check if required columns exist
    required_cols = [z_column, block_id_column]
    if not all(col in block_model.columns for col in required_cols):
        logger.warning(
            f"Cannot build vertical precedence: missing columns. "
            f"Required: {required_cols}, Available: {list(block_model.columns)}"
        )
        return precedence
    
    # Check for coordinate columns
    has_coords = x_column in block_model.columns and y_column in block_model.columns
    
    if not has_coords:
        logger.warning(
            "No X/Y coordinates found. Building simplified vertical precedence based on Z only."
        )
        # Simplified: blocks at lower Z depend on blocks at higher Z
        sorted_blocks = block_model.sort_values(z_column, ascending=False)
        block_ids = sorted_blocks[block_id_column].tolist()
        
        for i, block_id in enumerate(block_ids):
            # All blocks above (earlier in sorted list) are predecessors
            precedence[block_id] = block_ids[:i]
        
        return precedence
    
    # Full precedence with spatial matching
    # Group blocks by bench (Z level)
    block_model = block_model.copy()
    
    # Detect bench height
    z_values = np.sort(block_model[z_column].unique())
    if len(z_values) > 1:
        bench_height = np.min(np.diff(z_values))
    else:
        bench_height = 10.0  # Default
    
    block_model['_BENCH'] = (block_model[z_column] / bench_height).astype(int)
    
    # For each block, find blocks directly above
    for idx, row in block_model.iterrows():
        block_id = row[block_id_column]
        block_x = row[x_column]
        block_y = row[y_column]
        block_bench = row['_BENCH']
        
        # Find blocks at bench + 1 (directly above) with matching X/Y
        above = block_model[
            (block_model['_BENCH'] == block_bench + 1) &
            (np.abs(block_model[x_column] - block_x) < bench_tolerance) &
            (np.abs(block_model[y_column] - block_y) < bench_tolerance)
        ]
        
        if len(above) > 0:
            precedence[block_id] = above[block_id_column].tolist()
        else:
            precedence[block_id] = []
    
    logger.info(f"Built vertical precedence graph: {len(precedence)} blocks, "
               f"{sum(len(v) for v in precedence.values())} total precedence relationships")
    
    return precedence


def validate_schedule_precedence(
    schedule: pd.DataFrame,
    precedence: Dict[int, List[int]],
    block_id_column: str = 'BLOCK_ID',
    period_column: str = 'PERIOD',
    mined_column: str = 'MINED',
    strict_mode: bool = False
) -> ScheduleValidationResult:
    """
    Validate that a schedule respects precedence constraints.
    
    For each block, verify that all its predecessors are either:
    - Mined in an earlier period, OR
    - Not required to be mined (for blocks outside the pit)
    
    Args:
        schedule: Schedule DataFrame with BLOCK_ID, PERIOD, MINED columns
        precedence: Precedence graph from build_vertical_precedence
        block_id_column: Name of block ID column
        period_column: Name of period column
        mined_column: Name of mined flag column
        strict_mode: If True, treat same-period mining as violation
        
    Returns:
        ScheduleValidationResult with validation details
    """
    violations = []
    blocks_checked = 0
    pairs_checked = 0
    
    # Build period lookup
    mined_schedule = schedule[schedule[mined_column] == 1] if mined_column in schedule.columns else schedule
    period_map = dict(zip(mined_schedule[block_id_column], mined_schedule[period_column]))
    
    # Check each block's predecessors
    for block_id, predecessors in precedence.items():
        if block_id not in period_map:
            continue  # Block not mined, skip
        
        block_period = period_map[block_id]
        blocks_checked += 1
        
        for pred_id in predecessors:
            pairs_checked += 1
            
            if pred_id not in period_map:
                # Predecessor not mined - this is a violation
                violations.append(PrecedenceViolation(
                    block_id=block_id,
                    block_period=block_period,
                    predecessor_id=pred_id,
                    predecessor_period=-1,
                    violation_type='not_mined'
                ))
            else:
                pred_period = period_map[pred_id]
                
                if pred_period > block_period:
                    # Predecessor mined AFTER the block - critical violation
                    violations.append(PrecedenceViolation(
                        block_id=block_id,
                        block_period=block_period,
                        predecessor_id=pred_id,
                        predecessor_period=pred_period,
                        violation_type='mined_later'
                    ))
                elif pred_period == block_period and strict_mode:
                    # Same period - might be okay depending on mining sequence within period
                    violations.append(PrecedenceViolation(
                        block_id=block_id,
                        block_period=block_period,
                        predecessor_id=pred_id,
                        predecessor_period=pred_period,
                        violation_type='same_period'
                    ))
    
    is_valid = len(violations) == 0
    warning = None
    
    if not is_valid:
        critical = [v for v in violations if v.violation_type == 'mined_later']
        not_mined = [v for v in violations if v.violation_type == 'not_mined']
        same_period = [v for v in violations if v.violation_type == 'same_period']
        
        warning = (
            f"Schedule precedence violations found: "
            f"{len(critical)} critical (mined later), "
            f"{len(not_mined)} predecessors not mined, "
            f"{len(same_period)} same period"
        )
        logger.warning(warning)
    else:
        logger.info(f"Schedule precedence validated: {blocks_checked} blocks, {pairs_checked} pairs checked")
    
    return ScheduleValidationResult(
        is_valid=is_valid,
        total_violations=len(violations),
        violations=violations,
        blocks_checked=blocks_checked,
        precedence_pairs_checked=pairs_checked,
        warning_message=warning
    )

@jit(nopython=True, cache=True)
def _numba_schedule_greedy_linear(
    values: np.ndarray,
    tonnages: np.ndarray,
    capacity: float,
    max_periods: int
) -> np.ndarray:
    """
    Linear bucket filling on pre-sorted arrays.
    """
    n = len(values)
    periods = np.zeros(n, dtype=np.int32)
    
    current_p = 1
    current_cap = 0.0
    
    for i in range(n):
        # Skip waste if value is very negative? 
        # For greedy, we typically mine everything in the sequence 
        # unless it's strictly below cutoff.
        # Ideally, 'block_model' passed here should already be filtered to Ultimate Pit.
        
        t = tonnages[i]
        
        if current_cap + t > capacity:
            current_p += 1
            current_cap = 0.0
            
        if current_p > max_periods:
            # Schedule full, remaining blocks unmined
            periods[i] = 0
        else:
            periods[i] = current_p
            current_cap += t
            
    return periods

class FastScheduler:
    """
    Wrapper for vectorized scheduling with optional precedence validation.
    """
    
    @staticmethod
    def schedule_greedy(
        block_model: pd.DataFrame,
        num_periods: int,
        production_capacity: float,
        sort_by_phase: bool = True,
        validate_precedence: bool = False,
        precedence_graph: Optional[Dict[int, List[int]]] = None
    ) -> pd.DataFrame:
        """
        Generate a schedule almost instantly.
        
        Args:
            block_model: Block model DataFrame with TONNAGE, VALUE columns
            num_periods: Maximum number of mining periods
            production_capacity: Maximum tonnage per period
            sort_by_phase: If True, sort by phase before value
            validate_precedence: If True, validate schedule against precedence
            precedence_graph: Optional precedence graph (will be built if not provided)
            
        Returns:
            Schedule DataFrame with BLOCK_ID, PERIOD, MINED columns
        """
        # 1. Prepare Data
        # We need a copy to sort without affecting original
        df = block_model[['TONNAGE', 'VALUE']].copy()
        
        # Add index to reconstruct order later
        df['__ORIG_IDX'] = np.arange(len(df))
        
        # Sorting Strategy
        sort_cols = []
        ascending = []
        
        if sort_by_phase and 'PHASE' in block_model.columns:
            df['PHASE'] = block_model['PHASE']
            sort_cols.append('PHASE')
            ascending.append(True) # Phase 1, then 2...
            
        # Then by Value (Highest First) -> Greedy heuristic
        sort_cols.append('VALUE')
        ascending.append(False)
        
        # Sort
        df_sorted = df.sort_values(by=sort_cols, ascending=ascending)
        
        # 2. Extract Arrays
        values = df_sorted['VALUE'].values.astype(np.float64)
        tonnages = df_sorted['TONNAGE'].values.astype(np.float64)
        
        # 3. Run Numba Kernel
        # The kernel just fills buckets linearly because we pre-sorted by priority
        periods_sorted = _numba_schedule_greedy_linear(
            values, tonnages, production_capacity, num_periods
        )
        
        # 4. Map back to original indices
        result_periods = np.zeros(len(df), dtype=np.int32)
        orig_indices = df_sorted['__ORIG_IDX'].values
        result_periods[orig_indices] = periods_sorted
        
        # 5. Create Result DataFrame
        if 'BLOCK_ID' in block_model.columns:
            block_ids = block_model['BLOCK_ID'].values
        else:
            block_ids = block_model.index.values
        
        schedule = pd.DataFrame({
            'BLOCK_ID': block_ids,
            'PERIOD': result_periods,
            'MINED': (result_periods > 0).astype(int)
        })
        
        # Optional precedence validation
        if validate_precedence:
            if precedence_graph is None:
                # Build precedence from block model
                precedence_graph = build_vertical_precedence(block_model)
            
            if precedence_graph:
                validation_result = validate_schedule_precedence(schedule, precedence_graph)
                
                if not validation_result.is_valid:
                    logger.warning(
                        f"Schedule generated with {validation_result.total_violations} "
                        f"precedence violations. This is expected for greedy scheduling - "
                        f"use MILP optimizer for strict precedence compliance."
                    )
                
                # Add validation metadata to schedule
                schedule.attrs['precedence_validation'] = {
                    'is_valid': validation_result.is_valid,
                    'total_violations': validation_result.total_violations,
                    'blocks_checked': validation_result.blocks_checked
                }
        
        return schedule
    
    @staticmethod
    def schedule_greedy_with_validation(
        block_model: pd.DataFrame,
        num_periods: int,
        production_capacity: float,
        sort_by_phase: bool = True
    ) -> Tuple[pd.DataFrame, ScheduleValidationResult]:
        """
        Generate a schedule with full precedence validation.
        
        Returns:
            Tuple of (schedule, validation_result)
        """
        # Build precedence graph
        precedence = build_vertical_precedence(block_model)
        
        # Generate schedule
        schedule = FastScheduler.schedule_greedy(
            block_model, num_periods, production_capacity, sort_by_phase
        )
        
        # Validate
        validation = validate_schedule_precedence(schedule, precedence)
        
        return schedule, validation

