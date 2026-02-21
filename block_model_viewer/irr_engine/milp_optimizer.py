"""
MILP-based Mining Schedule Optimizer

Uses Mixed-Integer Linear Programming to find the optimal mining schedule
that maximizes NPV subject to mining constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
import os

logger = logging.getLogger(__name__)

# Try to import PuLP (optimization library)
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logger.warning("PuLP not installed. MILP optimization will not be available. Install with: pip install pulp")


class MiningScheduleOptimizer:
    """
    Optimize mining schedule using MILP to maximize NPV.
    """
    
    def __init__(
        self,
        block_model: pd.DataFrame,
        num_periods: int,
        production_capacity: float,
        discount_rate: float
    ):
        """
        Initialize the optimizer.
        
        Args:
            block_model: DataFrame with columns ['BLOCK_ID', 'TONNAGE', 'VALUE', 'PREDECESSORS']
            num_periods: Number of mining periods
            production_capacity: Maximum tonnage per period
            discount_rate: Discount rate for NPV calculation
        """
        self.block_model = block_model.copy()
        self.num_periods = num_periods
        self.production_capacity = production_capacity
        self.discount_rate = discount_rate
        self.num_blocks = len(block_model)
        
        logger.info(f"Initialized MILP optimizer: {self.num_blocks} blocks, {num_periods} periods, capacity={production_capacity:.0f} t/period")
    
    def optimize(self, time_limit: int = 300, aggregation_threshold: int = 5000) -> pd.DataFrame:
        """
        Solve the MILP to find optimal mining schedule.
        
        Automatically aggregates blocks into Scheduling Units if block count exceeds threshold.
        
        Args:
            time_limit: Maximum solution time in seconds
            aggregation_threshold: Max blocks before enabling aggregation (default: 5000)
            
        Returns:
            DataFrame with columns ['BLOCK_ID', 'PERIOD', 'MINED']
        """
        if not HAS_PULP:
            logger.error("PuLP not available. Cannot optimize schedule.")
            return self._create_simple_schedule()
            
        # Check if aggregation is needed
        if self.num_blocks > aggregation_threshold:
            logger.info(f"Block count ({self.num_blocks}) exceeds threshold ({aggregation_threshold}). Using Strategic Aggregation.")
            return self._optimize_aggregated(time_limit)
        
        logger.info("Building MILP model (Block Level)...")
        
        # Create the optimization problem
        prob = pulp.LpProblem("Mining_Schedule", pulp.LpMaximize)
        
        # Decision variables: x[b, t] = 1 if block b is mined in period t
        blocks = list(self.block_model['BLOCK_ID'])
        periods = list(range(self.num_periods))
        
        x = pulp.LpVariable.dicts("mine", (blocks, periods), cat='Binary')
        
        # Discount factors
        discount_factors = {t: 1.0 / ((1 + self.discount_rate) ** t) for t in periods}
        
        # Objective: Maximize discounted value
        # Pre-calculate block values to speed up sum creation
        block_values = dict(zip(self.block_model['BLOCK_ID'], self.block_model['VALUE']))
        
        prob += pulp.lpSum([
            x[b][t] * block_values[b] * discount_factors[t]
            for b in blocks
            for t in periods
        ]), "Total_NPV"
        
        # Constraint 1: Each block mined at most once
        for b in blocks:
            prob += pulp.lpSum([x[b][t] for t in periods]) <= 1, f"Mine_Once_{b}"
        
        # Constraint 2: Production capacity per period
        block_tonnages = dict(zip(self.block_model['BLOCK_ID'], self.block_model['TONNAGE']))
        
        for t in periods:
            prob += pulp.lpSum([
                x[b][t] * block_tonnages[b]
                for b in blocks
            ]) <= self.production_capacity, f"Capacity_{t}"
        
        # Constraint 3: Precedence constraints (if predecessors defined)
        if 'PREDECESSORS' in self.block_model.columns:
            # Optimize precedence lookup
            pred_map = {}
            for idx, row in self.block_model.iterrows():
                preds = row['PREDECESSORS']
                if pd.notna(preds) and preds != '':
                    try:
                        pred_map[row['BLOCK_ID']] = [int(p.strip()) for p in str(preds).split(',') if p.strip()]
                    except ValueError:
                        pass
            
            for b, pred_list in pred_map.items():
                for t in periods:
                    for pred_id in pred_list:
                        if pred_id in blocks:
                            # Block b can only be mined in period t if predecessor was mined in t-1 or earlier
                            # Sum(x[pred, s] for s <= t) >= x[b, t]
                            # This assumes if pred is mined in t, b can be mined in t? 
                            # Usually strictly earlier for vertical, but concurrent allowed for horizontal?
                            # Standard formulation: sum(x[pred, s] for s <= t) >= sum(x[b, s] for s <= t)
                            # Which implies cumulative production of pred >= cumulative production of b
                            
                            # Simplified: If b is mined in t, pred must have been mined in 0..t
                            prob += pulp.lpSum([x[pred_id][s] for s in range(t+1)]) >= x[b][t], \
                                   f"Precedence_{b}_{pred_id}_{t}"
        
        return self._solve_and_extract(prob, x, blocks, periods, time_limit)

    def _optimize_aggregated(self, time_limit: int) -> pd.DataFrame:
        """
        Optimize schedule using aggregated Scheduling Units (Phase-Bench).
        """
        logger.info("Aggregating blocks into Scheduling Units (Phase-Bench)...")
        
        # 1. Prepare Aggregation Keys
        df = self.block_model.copy()
        
        # Ensure PHASE exists
        if 'PHASE' not in df.columns:
            df['PHASE'] = 1
            
        # Ensure Z/Bench exists (bin Z if needed)
        if 'BENCH' not in df.columns:
            if 'Z' in df.columns:
                # Auto-detect bench height if not provided
                z_vals = df['Z'].unique()
                if len(z_vals) > 1:
                    z_vals.sort()
                    min_diff = np.min(np.diff(z_vals))
                    bench_height = max(min_diff, 1.0) # avoid 0
                else:
                    bench_height = 10.0
                
                # Bin Z to benches (assuming Z is elevation at center)
                df['BENCH'] = (df['Z'] // bench_height).astype(int)
            else:
                # Fallback: Treat entire phase as one vertical unit (dangerous but works for 2D)
                df['BENCH'] = 1
        
        # 2. Aggregate
        # Group by PHASE and BENCH (descending Z usually, but Bench index is typically bottom-up or top-down.
        # Let's use Z directly for sorting.)
        if 'Z' in df.columns:
            df['Z_MEAN'] = df['Z']
        else:
            df['Z_MEAN'] = 0.0
            
        units = df.groupby(['PHASE', 'BENCH']).agg({
            'TONNAGE': 'sum',
            'VALUE': 'sum',
            'BLOCK_ID': list,  # Store list of blocks to map back
            'Z_MEAN': 'mean'
        }).reset_index()
        
        # Create Unit IDs
        units['UNIT_ID'] = range(len(units))
        
        logger.info(f"Aggregated {self.num_blocks} blocks into {len(units)} Scheduling Units.")
        
        # 3. Build Precedence for Units
        # - Vertical: Within same Phase, lower Z depends on higher Z (assuming open pit top-down)
        # - Horizontal: Phase p depends on Phase p-1 at same level (simplified pushback logic)
        
        unit_precedence = []
        
        # Sort units to help finding neighbors
        # Assuming Z increases UP. Top-down mining means Higher Z must be mined before Lower Z.
        
        # Dict for fast lookup: (phase, bench) -> unit_id
        lookup = dict(zip(zip(units['PHASE'], units['BENCH']), units['UNIT_ID']))
        
        # Find bench above and previous phase
        # We need to know the "bench above". If we binned by Z // height, bench index + 1 is above.
        
        for _, row in units.iterrows():
            u_id = row['UNIT_ID']
            phase = row['PHASE']
            bench = row['BENCH']
            
            # Vertical Precedence (Top-Down)
            # Depends on Bench + 1 (Block Above)
            bench_above = bench + 1
            if (phase, bench_above) in lookup:
                pred_id = lookup[(phase, bench_above)]
                unit_precedence.append((u_id, pred_id))
                
            # Inter-Phase Precedence (Pushback)
            # Current Phase depends on Previous Phase at same level (or higher)
            # Usually Phase i Bench k depends on Phase i-1 Bench k
            if phase > 1:
                if (phase - 1, bench) in lookup:
                    pred_id = lookup[(phase - 1, bench)]
                    unit_precedence.append((u_id, pred_id))
        
        # 4. Solve MILP on Units
        logger.info("Building Strategic MILP model...")
        prob = pulp.LpProblem("Strategic_Schedule", pulp.LpMaximize)
        
        unit_ids = list(units['UNIT_ID'])
        periods = list(range(self.num_periods))
        
        # Variables
        y = pulp.LpVariable.dicts("mine_unit", (unit_ids, periods), cat='Binary')
        # Or Continuous? Strategic schedules often allow partial bench mining (Continuous variables).
        # "Mixed Integer" -> Binary usually for blocks. For aggregated units, we might want % mined.
        # But let's stick to Binary for "Scheduling Unit" as requested, treating them as indivisible chunks 
        # to ensure precedence is respected cleanly. If units are small enough, it's fine.
        
        discount_factors = {t: 1.0 / ((1 + self.discount_rate) ** t) for t in periods}
        
        # Objective
        unit_values = dict(zip(units['UNIT_ID'], units['VALUE']))
        prob += pulp.lpSum([
            y[u][t] * unit_values[u] * discount_factors[t]
            for u in unit_ids
            for t in periods
        ]), "Total_NPV"
        
        # Constraints
        # 1. Mine once
        for u in unit_ids:
            prob += pulp.lpSum([y[u][t] for t in periods]) <= 1
            
        # 2. Capacity
        unit_tonnages = dict(zip(units['UNIT_ID'], units['TONNAGE']))
        for t in periods:
            prob += pulp.lpSum([
                y[u][t] * unit_tonnages[u]
                for u in unit_ids
            ]) <= self.production_capacity
            
        # 3. Precedence
        for child, parent in unit_precedence:
            for t in periods:
                prob += pulp.lpSum([y[parent][s] for s in range(t+1)]) >= y[child][t]
                
        # Solve
        logger.info(f"Solving Aggregated MILP ({len(unit_ids)} units)...")
        # Use same solver logic
        self._solve_prob(prob, time_limit)
        
        # 5. Map Results back to Blocks
        schedule_data = []
        
        status = pulp.LpStatus[prob.status]
        if status not in ['Optimal', 'Feasible']:
            logger.warning(f"Aggregated MILP failed: {status}. Falling back to greedy.")
            return self._create_simple_schedule()
            
        for u in unit_ids:
            mined_period = -1
            for t in periods:
                if y[u][t].varValue is not None and y[u][t].varValue > 0.5:
                    mined_period = t
                    break
            
            # Assign to all blocks in unit
            # Get block IDs for this unit
            # We stored them in the dataframe list
            # But accessing list in dataframe row by row is slow?
            # Better: create a mapping unit_id -> period
            
            # Let's optimize mapping
            pass
            
        # Efficient Mapping
        unit_period_map = {}
        for u in unit_ids:
            mined_period = -1
            for t in periods:
                if y[u][t].varValue is not None and y[u][t].varValue > 0.5:
                    mined_period = t
                    break
            unit_period_map[u] = mined_period
            
        # Explode units to blocks
        # Create a Series mapping Unit -> Period
        units['SCHEDULED_PERIOD'] = units['UNIT_ID'].map(unit_period_map)
        
        # We need to join this back to the block list
        # units dataframe has 'BLOCK_ID' as a list. explode it.
        # This creates a DataFrame with [BLOCK_ID, SCHEDULED_PERIOD]
        schedule_df = units[['BLOCK_ID', 'SCHEDULED_PERIOD']].explode('BLOCK_ID')
        
        schedule_df['PERIOD'] = schedule_df['SCHEDULED_PERIOD'].fillna(-1).astype(int)
        schedule_df['MINED'] = (schedule_df['PERIOD'] >= 0).astype(int)
        
        return schedule_df[['BLOCK_ID', 'PERIOD', 'MINED']]

    def _solve_prob(self, prob, time_limit):
        # Try to locate CBC explicitly on Windows/conda to avoid PATH issues
        cbc_path_env = os.environ.get('PULP_CBC_PATH')
        default_cbc_path = os.path.join(os.environ.get('CONDA_PREFIX', ''), 'Library', 'bin', 'cbc.exe')
        candidate_paths = [cbc_path_env, default_cbc_path, r"C:\\Program Files\\CBC\\cbc.exe"]
        cbc_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)
        
        solver = None
        if cbc_path:
            try:
                solver = pulp.COIN_CMD(path=cbc_path, timeLimit=time_limit, msg=0)
                logger.debug(f"Using COIN_CMD solver from: {cbc_path}")
            except Exception as e:
                # MP-012 FIX: Log the exception instead of silently ignoring
                logger.warning(f"Failed to initialize COIN_CMD solver at {cbc_path}: {e}")
        
        if solver is None:
            solver = pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0)
            logger.debug("Using default PULP_CBC_CMD solver")
            
        prob.solve(solver)

    def _solve_and_extract(self, prob, x, blocks, periods, time_limit):
        self._solve_prob(prob, time_limit)
        
        # Check solution status
        status = pulp.LpStatus[prob.status]
        logger.info(f"MILP solution status: {status}")
        
        if status not in ['Optimal', 'Feasible']:
            logger.warning(f"MILP did not find optimal solution: {status}")
            return self._create_simple_schedule()
        
        # Extract solution
        schedule_data = []
        for b in blocks:
            for t in periods:
                if x[b][t].varValue is not None and x[b][t].varValue > 0.5:
                    schedule_data.append({
                        'BLOCK_ID': b,
                        'PERIOD': t,
                        'MINED': 1
                    })
        
        schedule_df = pd.DataFrame(schedule_data)
        
        # Add unmined blocks
        mined_blocks = set(schedule_df['BLOCK_ID'].unique())
        unmined_blocks = set(blocks) - mined_blocks
        if unmined_blocks:
            unmined_df = pd.DataFrame([{'BLOCK_ID': b, 'PERIOD': -1, 'MINED': 0} for b in unmined_blocks])
            schedule_df = pd.concat([schedule_df, unmined_df], ignore_index=True)
        
        optimal_value = pulp.value(prob.objective)
        logger.info(f"Optimal NPV: ${optimal_value:,.2f}")
        
        return schedule_df
    
    def _create_simple_schedule(self) -> pd.DataFrame:
        """
        Create a simple greedy schedule (fallback if MILP fails).
        
        Returns:
            DataFrame with schedule
        """
        logger.info("Creating simple greedy schedule...")
        
        # Sort blocks by value/tonnage ratio (grade)
        self.block_model['value_per_tonne'] = self.block_model['VALUE'] / self.block_model['TONNAGE']
        sorted_blocks = self.block_model.sort_values('value_per_tonne', ascending=False)
        
        schedule_data = []
        period = 0
        period_tonnage = 0
        
        for _, block in sorted_blocks.iterrows():
            block_id = block['BLOCK_ID']
            tonnage = block['TONNAGE']
            
            # Check if we can fit this block in current period
            if period_tonnage + tonnage > self.production_capacity:
                period += 1
                period_tonnage = 0
            
            if period < self.num_periods:
                schedule_data.append({
                    'BLOCK_ID': block_id,
                    'PERIOD': period,
                    'MINED': 1
                })
                period_tonnage += tonnage
            else:
                # No more periods available
                schedule_data.append({
                    'BLOCK_ID': block_id,
                    'PERIOD': -1,
                    'MINED': 0
                })
        
        logger.info(f"Simple schedule created with {len([s for s in schedule_data if s['MINED'] == 1])} mined blocks")
        return pd.DataFrame(schedule_data)
    
    def get_schedule_periods(self, schedule_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Extract period information from schedule DataFrame (STEP 21).
        
        Returns list of period dictionaries with:
        - period_index: Period number
        - block_ids: List of block IDs mined in this period
        - tonnage: Total tonnage mined
        - metal: Total metal content (if GRADE available)
        
        Args:
            schedule_df: Schedule DataFrame with BLOCK_ID, PERIOD columns
        
        Returns:
            List of period info dicts
        """
        periods_info = []
        
        if schedule_df.empty or 'PERIOD' not in schedule_df.columns:
            return periods_info
        
        # Filter to mined blocks only
        mined_schedule = schedule_df[schedule_df.get('MINED', 1) == 1].copy()
        
        if mined_schedule.empty:
            return periods_info
        
        # Group by period
        for period_idx in sorted(mined_schedule['PERIOD'].unique()):
            if period_idx < 0:  # Skip unmined blocks
                continue
            
            period_blocks = mined_schedule[mined_schedule['PERIOD'] == period_idx]
            block_ids = period_blocks['BLOCK_ID'].tolist()
            
            # Get tonnage
            tonnage = 0.0
            if 'TONNAGE' in period_blocks.columns:
                tonnage = period_blocks['TONNAGE'].sum()
            
            # Get metal (from GRADE × TONNAGE if available)
            metal = 0.0
            if 'GRADE' in period_blocks.columns and 'TONNAGE' in period_blocks.columns:
                metal = (period_blocks['GRADE'] * period_blocks['TONNAGE']).sum()
            elif 'METAL' in period_blocks.columns:
                metal = period_blocks['METAL'].sum()
            
            # Get block coordinates if available
            locations = []
            if all(col in period_blocks.columns for col in ['X', 'Y', 'Z']):
                coords = period_blocks[['X', 'Y', 'Z']].values
                locations = coords.tolist()
            elif all(col in period_blocks.columns for col in ['XC', 'YC', 'ZC']):
                coords = period_blocks[['XC', 'YC', 'ZC']].values
                locations = coords.tolist()
            
            periods_info.append({
                'period_index': int(period_idx),
                'block_ids': block_ids,
                'locations': locations,
                'tonnage': float(tonnage),
                'metal': float(metal)
            })
        
        return periods_info
    
    @staticmethod
    def prepare_block_model(
        block_model: pd.DataFrame,
        economic_params: Dict,
        scenario_idx: Optional[int] = None,
        scenarios: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Prepare block model with economic values for optimization.
        
        NOTE: This calculates UNDISCOUNTED block values (revenue - cost).
        The greedy scheduler uses these static values to prioritize blocks.
        Time value of money is applied later during NPV calculation.
        
        For a more sophisticated approach, block values could be iteratively
        recalculated with discount factors based on their scheduled periods,
        but this adds significant complexity for marginal improvement in
        the greedy heuristic.
        
        Args:
            block_model: Raw block model DataFrame
            economic_params: Economic parameters
            scenario_idx: Index of scenario to use (if stochastic)
            scenarios: Scenario data (if stochastic)
            
        Returns:
            Block model with VALUE column added (undiscounted)
        """
        prepared = block_model.copy()
        
        # Calculate block values
        if scenarios is not None and scenario_idx is not None:
            # Use scenario-specific prices and grades
            avg_price = scenarios['prices'][scenario_idx, :].mean()
            recovery = scenarios['recoveries'][scenario_idx]
            avg_mining_cost = scenarios['costs']['mining_cost'][scenario_idx, :].mean()
            avg_processing_cost = scenarios['costs']['processing_cost'][scenario_idx, :].mean()
        else:
            # Use base case values
            avg_price = economic_params['metal_price']
            recovery = economic_params['recovery']
            avg_mining_cost = economic_params['mining_cost']
            avg_processing_cost = economic_params['processing_cost']
        
        # Calculate value for each block
        values = []
        for idx, row in prepared.iterrows():
            tonnage = row['TONNAGE']
            grade = row['GRADE']
            
            metal_content = tonnage * grade * recovery
            revenue = metal_content * avg_price
            cost = tonnage * (avg_mining_cost + avg_processing_cost)
            value = revenue - cost
            
            values.append(value)
        
        prepared['VALUE'] = values
        
        return prepared

