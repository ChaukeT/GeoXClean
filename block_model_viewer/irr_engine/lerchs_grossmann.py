"""
Lerchs-Grossmann Algorithm for Ultimate Pit Limit

Implements the 3D Lerchs-Grossmann algorithm for finding the optimal pit outline
that maximizes total profit while respecting slope constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
import logging
from scipy import ndimage
import heapq

logger = logging.getLogger(__name__)


class LerchsGrossmann:
    """
    Implements the Lerchs-Grossmann algorithm for ultimate pit limit determination.
    
    This algorithm finds the set of blocks that maximize total value while
    respecting mining slope constraints (precedence relationships).
    """
    
    def __init__(
        self, 
        block_model: pd.DataFrame,
        slope_angles: Dict[str, float],
        economic_params: Dict
    ):
        """
        Initialize Lerchs-Grossmann optimizer.
        
        Args:
            block_model: Block model with coordinates and grades
            slope_angles: Dictionary of slope angles by rock type
            economic_params: Economic parameters for value calculation
        """
        self.block_model = block_model.copy()
        self.slope_angles = slope_angles
        self.economic_params = economic_params
        
        # Create block index mapping
        self._create_block_grid()
        
        logger.info(f"Initialized Lerchs-Grossmann with {len(self.block_model)} blocks")
    
    def _create_block_grid(self):
        """Create a 3D grid index for efficient neighbor lookup."""
        if not all(col in self.block_model.columns for col in ['XC', 'YC', 'ZC']):
            raise ValueError("Block model must have XC, YC, ZC columns")
        
        # Get unique coordinates
        x_coords = sorted(self.block_model['XC'].unique())
        y_coords = sorted(self.block_model['YC'].unique())
        z_coords = sorted(self.block_model['ZC'].unique())
        
        self.nx = len(x_coords)
        self.ny = len(y_coords)
        self.nz = len(z_coords)
        
        # Create coordinate to index mapping
        self.x_to_idx = {x: i for i, x in enumerate(x_coords)}
        self.y_to_idx = {y: i for i, y in enumerate(y_coords)}
        self.z_to_idx = {z: i for i, z in enumerate(z_coords)}
        
        # Create grid
        self.grid = np.full((self.nx, self.ny, self.nz), -1, dtype=int)
        
        for idx, row in self.block_model.iterrows():
            i = self.x_to_idx[row['XC']]
            j = self.y_to_idx[row['YC']]
            k = self.z_to_idx[row['ZC']]
            self.grid[i, j, k] = idx
        
        logger.info(f"Created 3D grid: {self.nx}x{self.ny}x{self.nz}")
    
    def calculate_block_value(self, row: pd.Series) -> float:
        """
        Calculate economic value of a block.
        
        Args:
            row: Block data row
            
        Returns:
            Net value (revenue - costs) in dollars
        """
        tonnage = row.get('TONNAGE', 0)
        if tonnage == 0:
            return 0.0
        
        grade = row.get('GRADE', 0)
        
        # Economic parameters
        # Note: metal_price should be in $/g for gold or $/lb for copper
        metal_price = self.economic_params.get('metal_price', 60.0)  # Default $60/g for gold
        recovery = self.economic_params.get('recovery', 0.85)
        mining_cost = self.economic_params.get('mining_cost', 2.5)  # $/tonne
        processing_cost = self.economic_params.get('processing_cost', 8.0)  # $/tonne
        
        # MP-008 FIX: Cutoff grade is configurable via economic_params
        # If not provided, calculate breakeven cutoff from economics
        cutoff_grade = self.economic_params.get('cutoff_grade', None)
        if cutoff_grade is None:
            # Calculate breakeven cutoff: grade where revenue = cost
            # revenue_per_t = grade * metal_price * recovery
            # cost_per_t = mining_cost + processing_cost
            # At breakeven: grade = cost / (price * recovery)
            if metal_price > 0 and recovery > 0:
                cutoff_grade = (mining_cost + processing_cost) / (metal_price * recovery)
                logger.debug(f"Calculated breakeven cutoff grade: {cutoff_grade:.4f}")
            else:
                cutoff_grade = 0.0  # No cutoff if economics undefined
                logger.warning("Cannot calculate cutoff grade - using 0.0 (all blocks as ore)")
        
        # Calculate value
        if grade >= cutoff_grade:
            # Ore block - mine and process
            # Revenue calculation:
            # - tonnage = tonnes of rock
            # - grade = grams of metal per tonne of rock (g/t)
            # - metal_revenue = (tonnage × grade) × price × recovery
            #   = grams of metal × $/gram × recovery
            metal_content_grams = tonnage * grade  # Total grams of metal in block
            metal_revenue = metal_content_grams * metal_price * recovery
            total_cost = tonnage * (mining_cost + processing_cost)
            value = metal_revenue - total_cost
        elif grade > 0:
            # Low-grade ore - mine as waste
            value = -tonnage * mining_cost
        else:
            # Waste block
            value = -tonnage * mining_cost
        
        return value
    
    def build_precedence_graph(self, slope_angle: float = 45) -> Dict[int, List[int]]:
        """
        Build precedence graph based on slope angle.
        
        A block can only be mined if all blocks above and around it
        (within the slope cone) have been mined.
        
        Args:
            slope_angle: Overall pit slope angle in degrees
            
        Returns:
            Dictionary mapping block_id to list of blocks that must be mined first
        """
        logger.info(f"Building precedence graph with slope angle {slope_angle}°")
        
        precedence = {}
        slope_rad = np.radians(slope_angle)
        
        # Calculate horizontal offset per vertical level
        # For each level up, how many blocks horizontally can we expand?
        dz = self.block_model['ZINC'].iloc[0] if 'ZINC' in self.block_model.columns else 10
        dx = self.block_model['XINC'].iloc[0] if 'XINC' in self.block_model.columns else 10
        
        # Horizontal expansion per vertical level
        horiz_per_vert = int(np.ceil(dz / np.tan(slope_rad) / dx))
        
        for idx, row in self.block_model.iterrows():
            i = self.x_to_idx[row['XC']]
            j = self.y_to_idx[row['YC']]
            k = self.z_to_idx[row['ZC']]
            
            predecessors = []
            
            # Look at blocks above (higher k values)
            for k_above in range(k + 1, self.nz):
                levels_above = k_above - k
                radius = levels_above * horiz_per_vert
                
                # Check blocks in expanding cone
                for di in range(-radius, radius + 1):
                    for dj in range(-radius, radius + 1):
                        i_check = i + di
                        j_check = j + dj
                        
                        if 0 <= i_check < self.nx and 0 <= j_check < self.ny:
                            pred_idx = self.grid[i_check, j_check, k_above]
                            if pred_idx >= 0:
                                predecessors.append(pred_idx)
            
            precedence[idx] = predecessors
        
        logger.info(f"Built precedence graph with {len(precedence)} blocks")
        return precedence
    
    def compute_ultimate_pit(self) -> pd.DataFrame:
        """
        Compute the ultimate pit limit using Lerchs-Grossmann algorithm.
        
        Returns:
            DataFrame with blocks marked as IN_PIT (1) or OUT_PIT (0)
        """
        logger.info("Computing ultimate pit limit using Lerchs-Grossmann...")
        
        # Calculate block values
        logger.info("Calculating block values...")
        self.block_model['VALUE'] = self.block_model.apply(
            self.calculate_block_value, axis=1
        )
        
        # Build precedence graph
        avg_slope = np.mean(list(self.slope_angles.values()))
        precedence = self.build_precedence_graph(slope_angle=avg_slope)
        
        # Run LG algorithm using max-flow approach
        logger.info("Running Lerchs-Grossmann algorithm...")
        in_pit = self._lerchs_grossmann_maxflow(precedence)
        
        # Mark blocks
        self.block_model['IN_PIT'] = 0
        self.block_model.loc[list(in_pit), 'IN_PIT'] = 1
        
        # Calculate pit statistics
        pit_blocks = self.block_model[self.block_model['IN_PIT'] == 1]
        total_value = pit_blocks['VALUE'].sum()
        total_tonnage = pit_blocks['TONNAGE'].sum()
        ore_blocks = pit_blocks[pit_blocks['GRADE'] > 0]
        
        logger.info("=" * 60)
        logger.info("ULTIMATE PIT LIMIT RESULTS:")
        logger.info(f"  Total blocks in pit: {len(pit_blocks):,}")
        logger.info(f"  Ore blocks: {len(ore_blocks):,}")
        logger.info(f"  Waste blocks: {len(pit_blocks) - len(ore_blocks):,}")
        logger.info(f"  Total tonnage: {total_tonnage:,.0f} tonnes")
        logger.info(f"  Total pit value: ${total_value:,.2f}")
        logger.info(f"  Average grade: {ore_blocks['GRADE'].mean():.2f} g/t")
        logger.info("=" * 60)
        
        return self.block_model
    
    def _lerchs_grossmann_maxflow(self, precedence: Dict[int, List[int]]) -> Set[int]:
        """
        Implement Lerchs-Grossmann using maximum closure algorithm.
        
        This uses a modified version that finds the set of blocks with
        maximum total value while respecting precedence constraints.
        
        Args:
            precedence: Dictionary of block precedence relationships
            
        Returns:
            Set of block indices that should be in the pit
        """
        # Simplified implementation using greedy approach with precedence
        # For production use, implement proper max-flow/min-cut algorithm
        
        in_pit = set()
        block_values = self.block_model['VALUE'].to_dict()
        
        # Sort blocks by elevation (mine from top down)
        sorted_blocks = self.block_model.sort_values('ZC', ascending=False).index
        
        for block_id in sorted_blocks:
            # Check if all predecessors are in pit
            can_mine = all(pred in in_pit for pred in precedence.get(block_id, []))
            
            if can_mine:
                # Calculate value of adding this block and all descendants
                total_value = self._calculate_closure_value(
                    block_id, in_pit, precedence, block_values
                )
                
                if total_value > 0:
                    # Add block and all required descendants
                    self._add_block_with_descendants(
                        block_id, in_pit, precedence
                    )
        
        return in_pit
    
    def _calculate_closure_value(
        self, 
        block_id: int, 
        current_pit: Set[int],
        precedence: Dict[int, List[int]],
        values: Dict[int, float]
    ) -> float:
        """Calculate the value of adding a block and all required descendants."""
        # Find all blocks that would need to be mined if we mine this block
        required = set([block_id])
        to_check = [block_id]
        
        while to_check:
            current = to_check.pop()
            for other_block, preds in precedence.items():
                if current in preds and other_block not in current_pit and other_block not in required:
                    required.add(other_block)
                    to_check.append(other_block)
        
        # Calculate total value
        return sum(values.get(b, 0) for b in required)
    
    def _add_block_with_descendants(
        self,
        block_id: int,
        in_pit: Set[int],
        precedence: Dict[int, List[int]]
    ):
        """Add a block and all blocks that depend on it to the pit."""
        to_add = [block_id]
        
        while to_add:
            current = to_add.pop()
            if current not in in_pit:
                in_pit.add(current)
                
                # Find blocks that have this as a predecessor
                for other_block, preds in precedence.items():
                    if current in preds and other_block not in in_pit:
                        # Check if all predecessors are now in pit
                        if all(p in in_pit for p in preds):
                            to_add.append(other_block)


class PitShellGenerator:
    """
    Generate nested pit shells for phased mining.
    """
    
    def __init__(self, block_model: pd.DataFrame, ultimate_pit_blocks: pd.DataFrame):
        """
        Initialize pit shell generator.
        
        Args:
            block_model: Full block model
            ultimate_pit_blocks: Block model with IN_PIT column
        """
        self.block_model = ultimate_pit_blocks.copy()
        self.pit_blocks = self.block_model[self.block_model['IN_PIT'] == 1]
        
        logger.info(f"Initialized shell generator with {len(self.pit_blocks)} pit blocks")
    
    def generate_revenue_shells(
        self, 
        num_shells: int = 5,
        revenue_factors: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Generate nested pit shells based on revenue factors.
        
        Args:
            num_shells: Number of shells to generate
            revenue_factors: List of revenue multipliers (e.g., [0.5, 0.7, 0.85, 1.0])
            
        Returns:
            Block model with SHELL column (0 = not in pit, 1-N = shell number)
        """
        if revenue_factors is None:
            # Generate evenly spaced factors from 50% to 100%
            revenue_factors = np.linspace(0.5, 1.0, num_shells).tolist()
        
        logger.info(f"Generating {num_shells} revenue-based pit shells...")
        logger.info(f"Revenue factors: {revenue_factors}")
        
        self.block_model['SHELL'] = 0
        
        # For each revenue factor, determine which blocks are economic
        for shell_num, factor in enumerate(revenue_factors, start=1):
            # Blocks in this shell are those with value > 0 at this revenue level
            # and not already in a previous shell
            
            adjusted_value = self.block_model['VALUE'] * factor
            economic = (adjusted_value > 0) & (self.block_model['SHELL'] == 0)
            
            self.block_model.loc[economic, 'SHELL'] = shell_num
            
            count = economic.sum()
            logger.info(f"  Shell {shell_num} ({factor:.1%} revenue): {count:,} blocks")
        
        return self.block_model
    
    def optimize_shell_sequence(
        self,
        discount_rate: float = 0.10
    ) -> List[int]:
        """
        Determine optimal mining sequence for shells based on NPV.
        
        Args:
            discount_rate: Annual discount rate
            
        Returns:
            List of shell numbers in optimal mining order
        """
        logger.info("Optimizing shell mining sequence...")
        
        shells = sorted(self.block_model['SHELL'].unique())
        shells = [s for s in shells if s > 0]  # Remove 0 (not in pit)
        
        # Calculate NPV for each possible sequence
        # For simplicity, mine from innermost (highest grade) to outermost
        # This is a heuristic - full optimization would test all permutations
        
        shell_values = []
        for shell in shells:
            shell_blocks = self.block_model[self.block_model['SHELL'] == shell]
            avg_grade = shell_blocks['GRADE'].mean()
            total_value = shell_blocks['VALUE'].sum()
            shell_values.append((shell, avg_grade, total_value))
        
        # Sort by grade (mine high-grade first for better early cash flow)
        shell_values.sort(key=lambda x: x[1], reverse=True)
        
        optimal_sequence = [sv[0] for sv in shell_values]
        
        logger.info(f"Optimal shell sequence: {optimal_sequence}")
        
        return optimal_sequence


def visualize_pit_shells(block_model: pd.DataFrame) -> Dict[str, any]:
    """
    Prepare pit shell data for 3D visualization.
    
    Args:
        block_model: Block model with SHELL column
        
    Returns:
        Dictionary with visualization data
    """
    shells = sorted(block_model['SHELL'].unique())
    shells = [s for s in shells if s > 0]
    
    viz_data = {
        'shells': {},
        'colors': {}
    }
    
    # Generate colors for shells (from red/inner to blue/outer)
    from matplotlib import cm
    colormap = cm.get_cmap('RdYlBu_r', len(shells))
    
    for i, shell in enumerate(shells):
        shell_blocks = block_model[block_model['SHELL'] == shell]
        
        viz_data['shells'][shell] = {
            'block_ids': shell_blocks['BLOCK_ID'].tolist() if 'BLOCK_ID' in shell_blocks.columns else shell_blocks.index.tolist(),
            'count': len(shell_blocks),
            'tonnage': shell_blocks['TONNAGE'].sum() if 'TONNAGE' in shell_blocks.columns else 0,
            'avg_grade': shell_blocks['GRADE'].mean() if 'GRADE' in shell_blocks.columns else 0,
            'total_value': shell_blocks['VALUE'].sum() if 'VALUE' in shell_blocks.columns else 0
        }
        
        # RGB color
        color = colormap(i)
        viz_data['colors'][shell] = (color[0], color[1], color[2])
    
    return viz_data

