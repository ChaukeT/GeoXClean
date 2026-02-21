"""
Dynamic Pit Shell Selector for Scenario-Based IRR Analysis

This module provides the missing link between pit optimization and IRR calculations.
For each price scenario, it dynamically selects the optimal pit shell from pre-computed
nested shells (Phase 2C), avoiding the computational cost of re-running Lerchs-Grossmann.

Key Insight:
-----------
When gold price drops by 20%, the optimal pit shrinks. Mining the original (larger) pit
with lower prices artificially inflates project risk. In reality, a rational operator
would simply stop mining marginal blocks earlier.

Solution:
---------
Use Nested Pit Shells generated at different Revenue Factors (RF). Each shell corresponds
to a specific price multiplier relative to the base case:
- RF = 1.0 → Ultimate Pit at base price
- RF = 0.8 → Pit that's optimal if revenue drops 20%
- RF = 0.6 → Pit that's optimal if revenue drops 40%

For each scenario with a price P_scenario, we:
1. Calculate RF = P_scenario / P_base
2. Select the shell whose RF is closest (but not exceeding) the scenario's RF
3. Mine only the blocks in that shell

Author: Mining Optimization AI
Date: 2025-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import bisect

logger = logging.getLogger(__name__)


@dataclass
class PitShellData:
    """Data for a single nested pit shell."""
    shell_id: int
    revenue_factor: float
    block_indices: np.ndarray  # Boolean mask or list of block IDs
    tonnage: float
    ore_tonnage: float
    metal_content: float
    total_value_at_rf: float  # Pit value at this revenue factor
    
    def __repr__(self):
        return (f"PitShell(id={self.shell_id}, RF={self.revenue_factor:.2f}, "
                f"tonnes={self.tonnage/1e6:.2f}Mt, value=${self.total_value_at_rf/1e6:.1f}M)")


@dataclass 
class ShellSelectionResult:
    """Result of selecting a shell for a scenario."""
    selected_shell_id: int
    revenue_factor: float
    scenario_price: float
    base_price: float
    effective_rf: float  # Actual RF for this scenario
    block_mask: np.ndarray
    tonnage: float
    interpolated: bool = False  # True if we interpolated between shells


class DynamicPitShellSelector:
    """
    Selects optimal pit shell for each Monte Carlo price scenario.
    
    This replaces the fixed-pit assumption in IRR calculations with dynamic
    pit boundaries that respond rationally to price changes.
    
    Usage:
    ------
    >>> # Pre-compute shells (done once in Phase 2C)
    >>> selector = DynamicPitShellSelector(base_price=60.0)
    >>> selector.add_shell(shell_id=1, revenue_factor=0.5, block_mask=mask_rf50, ...)
    >>> selector.add_shell(shell_id=2, revenue_factor=0.7, block_mask=mask_rf70, ...)
    >>> selector.add_shell(shell_id=3, revenue_factor=1.0, block_mask=mask_rf100, ...)
    >>> selector.finalize()
    >>> 
    >>> # In scenario loop:
    >>> for scenario in scenarios:
    >>>     result = selector.select_shell(scenario_price=48.0)
    >>>     mineable_blocks = block_model[result.block_mask]
    """
    
    def __init__(
        self,
        base_price: float,
        interpolation_mode: str = 'floor',
        min_rf: float = 0.3,
        max_rf: float = 1.5
    ):
        """
        Initialize the shell selector.
        
        Args:
            base_price: Base metal price used for shell generation (e.g., 60 $/g)
            interpolation_mode: How to handle RFs between shells:
                - 'floor': Use the shell with RF ≤ scenario RF (conservative)
                - 'nearest': Use the shell with closest RF
                - 'linear': Interpolate block values between adjacent shells
            min_rf: Minimum RF to consider (below this, use smallest shell)
            max_rf: Maximum RF to consider (above this, use ultimate pit)
        """
        self.base_price = base_price
        self.interpolation_mode = interpolation_mode
        self.min_rf = min_rf
        self.max_rf = max_rf
        
        self.shells: Dict[int, PitShellData] = {}
        self._sorted_rfs: List[float] = []
        self._rf_to_shell_id: Dict[float, int] = {}
        self._finalized = False
        
        # Cache for repeated lookups
        self._cache: Dict[float, ShellSelectionResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Initialized DynamicPitShellSelector (base_price=${base_price}, mode={interpolation_mode})")
    
    def add_shell(
        self,
        shell_id: int,
        revenue_factor: float,
        block_mask: np.ndarray,
        block_model: Optional[pd.DataFrame] = None,
        tonnage: Optional[float] = None,
        ore_tonnage: Optional[float] = None,
        metal_content: Optional[float] = None,
        total_value: Optional[float] = None
    ) -> None:
        """
        Add a pre-computed pit shell.
        
        Args:
            shell_id: Unique identifier for this shell
            revenue_factor: Revenue multiplier (0.5 = 50% of base revenue)
            block_mask: Boolean array or list of block indices in this shell
            block_model: Optional DataFrame for computing statistics
            tonnage: Total tonnage in shell (computed if not provided)
            ore_tonnage: Ore tonnage in shell
            metal_content: Total metal content (grade × tonnage)
            total_value: Total pit value at this RF
        """
        if self._finalized:
            raise RuntimeError("Cannot add shells after finalize() is called")
        
        # Convert to numpy if needed
        if isinstance(block_mask, (list, pd.Series)):
            block_mask = np.array(block_mask)
        
        # Compute statistics from block model if provided
        if block_model is not None and tonnage is None:
            shell_blocks = block_model[block_mask] if block_mask.dtype == bool else block_model.loc[block_mask]
            tonnage = shell_blocks['TONNAGE'].sum() if 'TONNAGE' in shell_blocks.columns else len(shell_blocks)
            if 'GRADE' in shell_blocks.columns:
                ore_blocks = shell_blocks[shell_blocks['GRADE'] > 0]
                ore_tonnage = ore_blocks['TONNAGE'].sum() if 'TONNAGE' in ore_blocks.columns else len(ore_blocks)
                metal_content = (ore_blocks['TONNAGE'] * ore_blocks['GRADE']).sum() if 'TONNAGE' in ore_blocks.columns else 0
            if 'VALUE' in shell_blocks.columns:
                total_value = shell_blocks['VALUE'].sum()
        
        shell = PitShellData(
            shell_id=shell_id,
            revenue_factor=revenue_factor,
            block_indices=block_mask,
            tonnage=tonnage or 0.0,
            ore_tonnage=ore_tonnage or 0.0,
            metal_content=metal_content or 0.0,
            total_value_at_rf=total_value or 0.0
        )
        
        self.shells[shell_id] = shell
        logger.debug(f"Added {shell}")
    
    def add_shells_from_dataframe(
        self,
        block_model: pd.DataFrame,
        shell_column: str = 'SHELL',
        revenue_factors: Optional[List[float]] = None
    ) -> None:
        """
        Add all shells from a block model DataFrame with SHELL column.
        
        Args:
            block_model: DataFrame with shell assignments
            shell_column: Column name containing shell IDs (1-N, 0 = not in pit)
            revenue_factors: List of RFs corresponding to shell numbers
        """
        if shell_column not in block_model.columns:
            raise ValueError(f"Column '{shell_column}' not found in block model")
        
        shell_ids = sorted([s for s in block_model[shell_column].unique() if s > 0])
        
        if revenue_factors is None:
            # Default: evenly spaced from 0.5 to 1.0
            revenue_factors = np.linspace(0.5, 1.0, len(shell_ids)).tolist()
        
        if len(revenue_factors) != len(shell_ids):
            raise ValueError(f"Number of revenue factors ({len(revenue_factors)}) must match shells ({len(shell_ids)})")
        
        for shell_id, rf in zip(shell_ids, revenue_factors):
            mask = block_model[shell_column] == shell_id
            self.add_shell(
                shell_id=shell_id,
                revenue_factor=rf,
                block_mask=mask.values,
                block_model=block_model
            )
        
        logger.info(f"Added {len(shell_ids)} shells from DataFrame")
    
    def add_shells_from_nested_result(
        self,
        nested_result: Dict[str, Any],
        block_model: pd.DataFrame
    ) -> None:
        """
        Add shells from a nested pit optimization result.
        
        Args:
            nested_result: Result dictionary from LerchsGrossmann nested shells
            block_model: Block model DataFrame
        """
        shells = nested_result.get('shells', [])
        
        for i, shell_data in enumerate(shells):
            rf = shell_data.get('factor', 0.5 + (i * 0.1))
            blocks = shell_data.get('blocks', [])
            
            # Create boolean mask
            mask = np.zeros(len(block_model), dtype=bool)
            mask[blocks] = True
            
            self.add_shell(
                shell_id=i + 1,
                revenue_factor=rf,
                block_mask=mask,
                block_model=block_model,
                tonnage=shell_data.get('tonnage')
            )
    
    def finalize(self) -> None:
        """
        Finalize the selector after all shells have been added.
        
        This sorts shells by RF and prepares lookup structures.
        Must be called before select_shell().
        """
        if not self.shells:
            raise RuntimeError("No shells added. Call add_shell() first.")
        
        # Sort RFs and create lookup
        self._sorted_rfs = sorted(self.shells.keys(), key=lambda sid: self.shells[sid].revenue_factor)
        self._sorted_rfs = [self.shells[sid].revenue_factor for sid in 
                           sorted(self.shells.keys(), key=lambda sid: self.shells[sid].revenue_factor)]
        
        # Create RF -> shell_id mapping
        for sid, shell in self.shells.items():
            self._rf_to_shell_id[shell.revenue_factor] = sid
        
        # Sort shell IDs by RF
        self._sorted_shell_ids = sorted(self.shells.keys(), key=lambda sid: self.shells[sid].revenue_factor)
        
        self._finalized = True
        
        logger.info(f"Finalized selector with {len(self.shells)} shells. "
                   f"RF range: [{min(self._sorted_rfs):.2f}, {max(self._sorted_rfs):.2f}]")
    
    def select_shell(
        self,
        scenario_price: float,
        use_cache: bool = True
    ) -> ShellSelectionResult:
        """
        Select the optimal pit shell for a given scenario price.
        
        Args:
            scenario_price: Metal price in this scenario
            use_cache: Whether to cache and reuse results for same prices
            
        Returns:
            ShellSelectionResult with selected shell data
        """
        if not self._finalized:
            raise RuntimeError("Selector not finalized. Call finalize() first.")
        
        # Calculate effective revenue factor
        effective_rf = scenario_price / self.base_price
        
        # Clamp to valid range
        effective_rf = max(self.min_rf, min(self.max_rf, effective_rf))
        
        # Round for cache key (to 3 decimal places to avoid float precision issues)
        cache_key = round(effective_rf, 3)
        
        if use_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Select shell based on interpolation mode
        if self.interpolation_mode == 'floor':
            result = self._select_floor(effective_rf, scenario_price)
        elif self.interpolation_mode == 'nearest':
            result = self._select_nearest(effective_rf, scenario_price)
        elif self.interpolation_mode == 'linear':
            result = self._select_linear(effective_rf, scenario_price)
        else:
            raise ValueError(f"Unknown interpolation mode: {self.interpolation_mode}")
        
        if use_cache:
            self._cache[cache_key] = result
        
        return result
    
    def _select_floor(self, effective_rf: float, scenario_price: float) -> ShellSelectionResult:
        """Select shell with RF ≤ effective_rf (conservative approach)."""
        # Find the largest RF that doesn't exceed effective_rf
        idx = bisect.bisect_right(self._sorted_rfs, effective_rf) - 1
        
        if idx < 0:
            # Effective RF is below all shells, use smallest
            idx = 0
        
        selected_rf = self._sorted_rfs[idx]
        shell_id = self._rf_to_shell_id[selected_rf]
        shell = self.shells[shell_id]
        
        return ShellSelectionResult(
            selected_shell_id=shell_id,
            revenue_factor=shell.revenue_factor,
            scenario_price=scenario_price,
            base_price=self.base_price,
            effective_rf=effective_rf,
            block_mask=shell.block_indices,
            tonnage=shell.tonnage,
            interpolated=False
        )
    
    def _select_nearest(self, effective_rf: float, scenario_price: float) -> ShellSelectionResult:
        """Select shell with RF closest to effective_rf."""
        # Find insertion point
        idx = bisect.bisect_left(self._sorted_rfs, effective_rf)
        
        if idx == 0:
            selected_rf = self._sorted_rfs[0]
        elif idx == len(self._sorted_rfs):
            selected_rf = self._sorted_rfs[-1]
        else:
            # Choose closer of the two
            lower_rf = self._sorted_rfs[idx - 1]
            upper_rf = self._sorted_rfs[idx]
            
            if abs(effective_rf - lower_rf) <= abs(effective_rf - upper_rf):
                selected_rf = lower_rf
            else:
                selected_rf = upper_rf
        
        shell_id = self._rf_to_shell_id[selected_rf]
        shell = self.shells[shell_id]
        
        return ShellSelectionResult(
            selected_shell_id=shell_id,
            revenue_factor=shell.revenue_factor,
            scenario_price=scenario_price,
            base_price=self.base_price,
            effective_rf=effective_rf,
            block_mask=shell.block_indices,
            tonnage=shell.tonnage,
            interpolated=False
        )
    
    def _select_linear(self, effective_rf: float, scenario_price: float) -> ShellSelectionResult:
        """
        Linearly interpolate between adjacent shells.
        
        For blocks, we use a value-weighted approach:
        - Calculate interpolated block value
        - Include blocks where interpolated value > 0
        """
        # For simplicity, fall back to floor for now
        # Full linear interpolation would require re-computing block values
        # which defeats the purpose of pre-computation
        logger.warning("Linear interpolation not fully implemented, using floor")
        return self._select_floor(effective_rf, scenario_price)
    
    def get_shell_for_scenario_batch(
        self,
        scenario_prices: np.ndarray
    ) -> List[ShellSelectionResult]:
        """
        Select shells for multiple scenarios efficiently.
        
        Args:
            scenario_prices: Array of prices for each scenario
            
        Returns:
            List of ShellSelectionResult for each scenario
        """
        return [self.select_shell(price) for price in scenario_prices]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get selector statistics including cache performance."""
        total_lookups = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_lookups if total_lookups > 0 else 0
        
        return {
            'num_shells': len(self.shells),
            'rf_range': (min(self._sorted_rfs), max(self._sorted_rfs)) if self._sorted_rfs else (0, 0),
            'base_price': self.base_price,
            'interpolation_mode': self.interpolation_mode,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'cache_hit_rate': hit_rate,
            'shells': [
                {
                    'id': s.shell_id,
                    'rf': s.revenue_factor,
                    'tonnage_Mt': s.tonnage / 1e6,
                    'value_M$': s.total_value_at_rf / 1e6
                }
                for s in sorted(self.shells.values(), key=lambda x: x.revenue_factor)
            ]
        }
    
    def clear_cache(self) -> None:
        """Clear the selection cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


def create_shell_selector_from_lg_result(
    lg_result: Dict[str, Any],
    block_model: pd.DataFrame,
    base_price: float,
    interpolation_mode: str = 'floor'
) -> DynamicPitShellSelector:
    """
    Factory function to create a shell selector from Lerchs-Grossmann nested shell result.
    
    Args:
        lg_result: Result from LerchsGrossmann nested shell optimization
        block_model: Original block model DataFrame
        base_price: Base metal price used in optimization
        interpolation_mode: Shell selection mode
        
    Returns:
        Configured DynamicPitShellSelector ready for use
    """
    selector = DynamicPitShellSelector(
        base_price=base_price,
        interpolation_mode=interpolation_mode
    )
    
    if 'shells' in lg_result:
        # Nested shells from engine_api
        selector.add_shells_from_nested_result(lg_result, block_model)
    elif 'SHELL' in block_model.columns:
        # Shell column already in block model
        selector.add_shells_from_dataframe(block_model)
    else:
        raise ValueError("Cannot create selector: no shell data found")
    
    selector.finalize()
    return selector

