"""
Pit Phase Optimization Module

Handles multi-phase pit scheduling and phase-based constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PitPhaseOptimizer:
    """
    Optimizes mining schedule across multiple pit phases (pushbacks).
    """
    
    def __init__(self, block_model: pd.DataFrame, config: Dict):
        """
        Initialize pit phase optimizer.
        
        Args:
            block_model: Block model DataFrame with XC, YC, ZC coordinates
            config: Configuration dictionary with phase parameters
        """
        self.block_model = block_model.copy()
        self.config = config
        
        self.num_phases = config.get('num_phases', 3)
        self.phase_gap = config.get('phase_gap', 30)  # meters
        self.slope_angles = config.get('slope_angles', {
            'ore': 55,
            'soil': 25,
            'weathered': 45,
            'fresh': 55
        })
        
        logger.info(f"Initialized PitPhaseOptimizer with {self.num_phases} phases")
    
    def assign_phases(self) -> pd.DataFrame:
        """
        Assign each block to a pit phase based on elevation and geometry.
        
        Returns:
            Block model with 'PHASE' column added
        """
        logger.info("Assigning blocks to pit phases...")
        
        # Get elevation field (try ZC first, fall back to ZMORIG, then DZ if needed)
        z_field = None
        if 'ZC' in self.block_model.columns and self.block_model['ZC'].max() - self.block_model['ZC'].min() > 1e-6:
            z_field = 'ZC'
        elif 'ZMORIG' in self.block_model.columns and self.block_model['ZMORIG'].max() - self.block_model['ZMORIG'].min() > 1e-6:
            logger.warning("ZC has no variation, using ZMORIG for phase assignment instead")
            z_field = 'ZMORIG'
        elif 'DZ' in self.block_model.columns:
            # Last resort: create ZC from index and DZ
            logger.warning("Creating synthetic Z coordinates from DZ field")
            self.block_model['ZC_SYNTH'] = self.block_model.index * self.block_model['DZ']
            z_field = 'ZC_SYNTH'
        
        if z_field is None:
            logger.warning("No suitable elevation field found (ZC, ZMORIG, or DZ), using default phase assignment")
            self.block_model['PHASE'] = 1
            return self.block_model
        
        z_min = self.block_model[z_field].min()
        z_max = self.block_model[z_field].max()
        z_range = z_max - z_min
        
        # DEBUG: Show actual Z values
        unique_z_count = self.block_model[z_field].nunique()
        logger.info(f"Using {z_field} for phases: min={z_min:.2f}m, max={z_max:.2f}m, range={z_range:.2f}m, unique_values={unique_z_count}")
        
        # Check for zero or very small elevation range (should not happen now)
        if z_range < 1e-6:
            logger.warning(f"Negligible elevation range ({z_range:.6f}m) even after fallback, assigning all blocks to phase 1")
            self.block_model['PHASE'] = 1
            return self.block_model
        
        # Divide elevation into phases (deeper = earlier phase)
        # Phase 1 is deepest (mined first), Phase N is shallowest (mined last)
        phase_height = z_range / self.num_phases
        
        phases = []
        for _, block in self.block_model.iterrows():
            z = block[z_field]
            # Calculate phase number (1 = deepest)
            phase_num = int((z - z_min) / phase_height) + 1
            phase_num = min(phase_num, self.num_phases)  # Cap at max phase
            phases.append(phase_num)
        
        self.block_model['PHASE'] = phases
        
        # Log phase distribution
        phase_counts = self.block_model['PHASE'].value_counts().sort_index()
        logger.info("Phase distribution:")
        for phase, count in phase_counts.items():
            logger.info(f"  Phase {phase}: {count:,} blocks")
        
        return self.block_model
    
    def get_phase_constraints(self) -> Dict[int, Dict]:
        """
        Generate constraints for each phase.
        
        Returns:
            Dictionary mapping phase number to constraint dict
        """
        constraints = {}
        
        for phase in range(1, self.num_phases + 1):
            phase_blocks = self.block_model[self.block_model['PHASE'] == phase]
            
            if len(phase_blocks) == 0:
                continue
            
            constraints[phase] = {
                'min_elevation': phase_blocks['ZC'].min() if 'ZC' in phase_blocks.columns else 0,
                'max_elevation': phase_blocks['ZC'].max() if 'ZC' in phase_blocks.columns else 0,
                'block_count': len(phase_blocks),
                'total_tonnage': phase_blocks['TONNAGE'].sum() if 'TONNAGE' in phase_blocks.columns else 0,
                'avg_grade': phase_blocks['GRADE'].mean() if 'GRADE' in phase_blocks.columns else 0
            }
        
        return constraints
    
    def create_phase_schedule(self, base_schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Modify a base schedule to respect phase constraints.
        
        Ensures that blocks in Phase N are only mined after a sufficient
        proportion of Phase N-1 has been completed.
        
        Args:
            base_schedule: Initial schedule DataFrame with BLOCK_ID and PERIOD
            
        Returns:
            Modified schedule respecting phase constraints
        """
        logger.info("Creating phase-constrained schedule...")
        
        # Merge phase information (only if PHASE column not already present)
        if 'PHASE' not in base_schedule.columns:
            schedule = base_schedule.merge(
                self.block_model[['BLOCK_ID', 'PHASE']], 
                on='BLOCK_ID', 
                how='left'
            )
            # Fill missing phases with 1
            schedule['PHASE'] = schedule['PHASE'].fillna(1).astype(int)
        else:
            # Schedule already has PHASE column (from greedy scheduler)
            schedule = base_schedule.copy()
            schedule['PHASE'] = schedule['PHASE'].fillna(1).astype(int)
        
        # Ensure phase precedence: Phase N cannot start until Phase N-1 is substantially complete
        phase_completion_threshold = 0.75  # 75% of previous phase must be mined
        
        adjusted_schedule = []
        phase_start_periods = {}
        
        for phase in range(1, self.num_phases + 1):
            phase_blocks = schedule[schedule['PHASE'] == phase].copy()
            
            if len(phase_blocks) == 0:
                continue
            
            # Determine earliest start period for this phase
            if phase == 1:
                # First phase can start in period 0
                earliest_period = 0
            else:
                # Later phases must wait for previous phase to reach threshold
                prev_phase_blocks = schedule[schedule['PHASE'] == phase - 1]
                if len(prev_phase_blocks) > 0:
                    prev_sorted = prev_phase_blocks.sort_values('PERIOD')
                    cutoff_idx = int(len(prev_sorted) * phase_completion_threshold)
                    earliest_period = prev_sorted.iloc[cutoff_idx]['PERIOD'] if cutoff_idx < len(prev_sorted) else 0
                else:
                    earliest_period = 0
            
            phase_start_periods[phase] = earliest_period
            
            # Adjust periods for this phase
            phase_blocks['PERIOD'] = phase_blocks['PERIOD'].apply(
                lambda p: max(p, earliest_period)
            )
            
            adjusted_schedule.append(phase_blocks)
        
        result = pd.concat(adjusted_schedule, ignore_index=True)
        
        # Log phase start periods
        logger.info("Phase start periods:")
        for phase, start_period in phase_start_periods.items():
            logger.info(f"  Phase {phase}: Period {start_period}")
        
        # Return ALL columns, not just the scheduling columns
        # This ensures TONNAGE, GRADE (Au), VALUE, and all other block properties are preserved
        return result
    
    def calculate_stripping_ratio(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate stripping ratio for each period.
        
        Args:
            schedule: Schedule with PHASE information
            
        Returns:
            DataFrame with period-wise stripping ratios
        """
        # Merge with block model to get ore/waste classification
        merged = schedule.merge(
            self.block_model[['BLOCK_ID', 'TONNAGE', 'GRADE']], 
            on='BLOCK_ID', 
            how='left'
        )
        
        # Classify ore vs waste (assuming grade > 0 means ore)
        merged['ORE_TONNES'] = merged.apply(
            lambda row: row['TONNAGE'] if row.get('GRADE', 0) > 0 else 0, 
            axis=1
        )
        merged['WASTE_TONNES'] = merged.apply(
            lambda row: row['TONNAGE'] if row.get('GRADE', 0) == 0 else 0, 
            axis=1
        )
        
        # Group by period
        period_summary = merged.groupby('PERIOD').agg({
            'ORE_TONNES': 'sum',
            'WASTE_TONNES': 'sum'
        }).reset_index()
        
        # Calculate stripping ratio
        period_summary['STRIPPING_RATIO'] = period_summary.apply(
            lambda row: row['WASTE_TONNES'] / row['ORE_TONNES'] if row['ORE_TONNES'] > 0 else 0,
            axis=1
        )
        
        return period_summary


def apply_slope_constraints(
    block_model: pd.DataFrame, 
    slope_angles: Dict[str, float]
) -> pd.DataFrame:
    """
    Apply pit slope angle constraints based on rock type.
    
    Args:
        block_model: Block model with XC, YC, ZC coordinates
        slope_angles: Dictionary of slope angles by rock type
        
    Returns:
        Block model with slope constraint information
    """
    logger.info("Applying slope angle constraints...")
    
    # This is a simplified implementation
    # In practice, you'd use Lerchs-Grossmann or similar algorithm
    
    # Add default rock type if not present
    if 'ROCK_TYPE' not in block_model.columns:
        # Assign rock type based on elevation (simplified)
        if 'ZC' in block_model.columns:
            z_median = block_model['ZC'].median()
            z_25 = block_model['ZC'].quantile(0.25)
            
            def assign_rock_type(z):
                if z < z_25:
                    return 'fresh'
                elif z < z_median:
                    return 'weathered'
                elif block_model[block_model['ZC'] == z].get('GRADE', 0).iloc[0] > 0:
                    return 'ore'
                else:
                    return 'soil'
            
            block_model['ROCK_TYPE'] = block_model['ZC'].apply(assign_rock_type)
        else:
            block_model['ROCK_TYPE'] = 'ore'
    
    # Assign slope angle to each block
    def get_slope_angle(rock_type):
        return slope_angles.get(rock_type, 45)  # Default 45 degrees
    
    block_model['SLOPE_ANGLE'] = block_model['ROCK_TYPE'].apply(get_slope_angle)
    
    logger.info(f"Assigned slope angles to {len(block_model)} blocks")
    
    return block_model


def generate_phase_visualization_data(
    block_model: pd.DataFrame, 
    schedule: pd.DataFrame
) -> Dict[int, List[int]]:
    """
    Generate data for visualizing pit phases in 3D.
    
    Args:
        block_model: Block model with PHASE column
        schedule: Mining schedule
        
    Returns:
        Dictionary mapping phase number to list of block IDs
    """
    merged = block_model.merge(schedule[['BLOCK_ID', 'PHASE']], on='BLOCK_ID', how='left')
    
    phase_blocks = {}
    for phase in merged['PHASE'].unique():
        if pd.notna(phase):
            phase_blocks[int(phase)] = merged[merged['PHASE'] == phase]['BLOCK_ID'].tolist()
    
    return phase_blocks





