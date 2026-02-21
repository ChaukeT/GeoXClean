"""
Probabilistic Pit Shells - Spatial Mining Risk Analysis

Generates multiple pit shells under uncertainty to quantify block-level mining probability:
- Revenue factor sweeps with stochastic inputs
- Grade realization-driven pit shells  
- Block inclusion frequency analysis
- Risk maps: robust (>80%), marginal (20-80%), fringe (<20%)
- Integration with 3D visualization
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)


@dataclass
class ProbShellConfig:
    """Configuration for probabilistic pit shell analysis."""
    
    # Uncertainty source
    use_grade_realizations: bool = True
    n_realizations: int = 100
    grade_realization_pattern: str = "grade_real_{i}"
    
    # Parameter uncertainty (if not using realizations)
    price_distribution: Optional[Dict] = None      # {'type': 'normal', 'params': {...}}
    cost_distribution: Optional[Dict] = None
    recovery_distribution: Optional[Dict] = None
    slope_distribution: Optional[Dict] = None
    
    # Optimization settings
    revenue_factors: Optional[List[float]] = None   # If doing RF sweep
    pit_optimizer_config: Optional[Dict] = None
    
    # Processing
    parallel: bool = True
    n_workers: Optional[int] = None
    
    # Risk classification thresholds
    robust_threshold: float = 0.80      # >= 80% mining probability
    fringe_threshold: float = 0.20      # <= 20% mining probability
    
    random_seed: Optional[int] = None
    
    def validate(self):
        """Validate configuration."""
        if self.use_grade_realizations and self.n_realizations < 10:
            raise ValueError("n_realizations should be >= 10 for stable probabilities")
        
        if not 0 < self.fringe_threshold < self.robust_threshold < 1:
            raise ValueError("Invalid threshold values")


@dataclass
class ProbShellResult:
    """Results from probabilistic shell analysis."""
    
    # Per-block mining probability
    mining_probability: np.ndarray  # Shape: (n_blocks,)
    block_ids: np.ndarray
    
    # Risk classification
    robust_blocks: np.ndarray       # Block IDs with P >= robust_threshold
    marginal_blocks: np.ndarray     # Block IDs with fringe < P < robust
    fringe_blocks: np.ndarray       # Block IDs with P <= fringe_threshold
    
    # Statistics
    n_simulations: int
    n_blocks_total: int
    
    # Shell data (optional)
    shells: Optional[List[Dict]] = None  # List of {simulation_id, blocks, params}
    
    # Value at risk
    expected_ore_tonnes: float = 0.0
    p10_ore_tonnes: float = 0.0
    p90_ore_tonnes: float = 0.0
    
    expected_metal: float = 0.0
    p10_metal: float = 0.0
    p90_metal: float = 0.0
    
    def compute_risk_statistics(self, block_model: pd.DataFrame):
        """Compute ore and metal statistics by risk category."""
        if 'tonnage' not in block_model.columns:
            logger.warning("No tonnage column, skipping risk statistics")
            return
        
        # Tonnage by risk category
        robust_mask = np.isin(self.block_ids, self.robust_blocks)
        marginal_mask = np.isin(self.block_ids, self.marginal_blocks)
        fringe_mask = np.isin(self.block_ids, self.fringe_blocks)
        
        tonnage = block_model['tonnage'].values
        
        self.robust_tonnes = np.sum(tonnage[robust_mask])
        self.marginal_tonnes = np.sum(tonnage[marginal_mask])
        self.fringe_tonnes = np.sum(tonnage[fringe_mask])
        
        logger.info(f"Risk categories - Robust: {self.robust_tonnes/1e6:.1f}Mt, "
                   f"Marginal: {self.marginal_tonnes/1e6:.1f}Mt, "
                   f"Fringe: {self.fringe_tonnes/1e6:.1f}Mt")


class ProbabilisticShellAnalyzer:
    """
    Probabilistic pit shell analyzer for spatial risk quantification.
    """
    
    def __init__(self, config: ProbShellConfig):
        """Initialize analyzer."""
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        logger.info(f"Initialized Probabilistic Shell Analyzer: {config.n_realizations} realizations")
    
    def analyze(
        self,
        block_model: pd.DataFrame,
        pit_optimizer_func: Callable,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> ProbShellResult:
        """
        Run probabilistic shell analysis.
        
        Args:
            block_model: Base block model with grade realizations or base grades
            pit_optimizer_func: Function(block_model, params) -> (selected_blocks, kpis)
            progress_callback: Callback(current, total)
        
        Returns:
            ProbShellResult with mining probabilities and risk maps
        """
        logger.info("Starting probabilistic pit shell analysis...")
        start_time = time.time()
        
        n_blocks = len(block_model)
        block_ids = block_model.index.values
        
        # Generate scenarios
        scenarios = self._generate_scenarios(block_model)
        
        # Run optimizations
        if self.config.parallel:
            shells = self._run_parallel(
                block_model, pit_optimizer_func, scenarios, progress_callback
            )
        else:
            shells = self._run_sequential(
                block_model, pit_optimizer_func, scenarios, progress_callback
            )
        
        # Compute mining probability
        mining_count = np.zeros(n_blocks)
        
        for shell_data in shells:
            selected_mask = shell_data['selected_blocks']
            mining_count[selected_mask] += 1
        
        mining_probability = mining_count / len(shells)
        
        # Classify blocks by risk
        robust_blocks = block_ids[mining_probability >= self.config.robust_threshold]
        fringe_blocks = block_ids[mining_probability <= self.config.fringe_threshold]
        marginal_blocks = block_ids[
            (mining_probability > self.config.fringe_threshold) &
            (mining_probability < self.config.robust_threshold)
        ]
        
        # Create result
        result = ProbShellResult(
            mining_probability=mining_probability,
            block_ids=block_ids,
            robust_blocks=robust_blocks,
            marginal_blocks=marginal_blocks,
            fringe_blocks=fringe_blocks,
            n_simulations=len(shells),
            n_blocks_total=n_blocks,
            shells=shells if self.config.pit_optimizer_config.get('store_shells', False) else None
        )
        
        # Compute statistics
        result.compute_risk_statistics(block_model)
        self._compute_value_at_risk(result, shells, block_model)
        
        elapsed = time.time() - start_time
        logger.info(f"Probabilistic analysis complete: {len(shells)} shells in {elapsed:.1f}s")
        logger.info(f"Robust blocks: {len(robust_blocks)} ({len(robust_blocks)/n_blocks*100:.1f}%)")
        logger.info(f"Marginal blocks: {len(marginal_blocks)} ({len(marginal_blocks)/n_blocks*100:.1f}%)")
        logger.info(f"Fringe blocks: {len(fringe_blocks)} ({len(fringe_blocks)/n_blocks*100:.1f}%)")
        
        return result
    
    def _generate_scenarios(self, block_model: pd.DataFrame) -> List[Dict]:
        """Generate scenarios for probabilistic analysis."""
        scenarios = []
        
        if self.config.use_grade_realizations:
            # Use pre-generated grade realizations
            for i in range(self.config.n_realizations):
                real_col = self.config.grade_realization_pattern.format(i=i)
                if real_col not in block_model.columns:
                    logger.warning(f"Grade realization column {real_col} not found, skipping")
                    continue
                
                scenarios.append({
                    'type': 'grade_realization',
                    'realization_id': i,
                    'grade_column': real_col
                })
        
        else:
            # Generate parameter samples
            from .lhs_sampler import LHSSampler, LHSConfig
            
            # Build parameter specs
            param_specs = {}
            if self.config.price_distribution:
                param_specs['price'] = self.config.price_distribution
            if self.config.cost_distribution:
                param_specs['cost'] = self.config.cost_distribution
            if self.config.recovery_distribution:
                param_specs['recovery'] = self.config.recovery_distribution
            if self.config.slope_distribution:
                param_specs['slope_angle'] = self.config.slope_distribution
            
            # Sample using LHS
            lhs_config = LHSConfig(
                n_samples=self.config.n_realizations,
                n_dimensions=len(param_specs),
                random_seed=self.config.random_seed
            )
            
            sampler = LHSSampler(lhs_config)
            samples_df = sampler.sample_to_dataframe(param_specs)
            
            for i, row in samples_df.iterrows():
                scenarios.append({
                    'type': 'parameter_sample',
                    'sample_id': i,
                    'parameters': row.to_dict()
                })
        
        logger.info(f"Generated {len(scenarios)} scenarios")
        return scenarios
    
    def _run_sequential(
        self,
        block_model: pd.DataFrame,
        optimizer_func: Callable,
        scenarios: List[Dict],
        progress_callback: Optional[Callable]
    ) -> List[Dict]:
        """Run scenarios sequentially."""
        shells = []
        
        for i, scenario in enumerate(scenarios):
            try:
                shell_data = self._run_single_optimization(
                    i, block_model, optimizer_func, scenario
                )
                shells.append(shell_data)
                
                if progress_callback:
                    progress_callback(i + 1, len(scenarios))
            
            except Exception as e:
                logger.warning(f"Scenario {i} failed: {e}")
                continue
        
        return shells
    
    def _run_parallel(
        self,
        block_model: pd.DataFrame,
        optimizer_func: Callable,
        scenarios: List[Dict],
        progress_callback: Optional[Callable]
    ) -> List[Dict]:
        """Run scenarios in parallel."""
        shells = []
        n_workers = self.config.n_workers or min(8, len(scenarios))
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_optimization,
                    i, block_model, optimizer_func, scenario
                ): i
                for i, scenario in enumerate(scenarios)
            }
            
            completed = 0
            for future in as_completed(futures):
                try:
                    shell_data = future.result()
                    shells.append(shell_data)
                except Exception as e:
                    scenario_id = futures[future]
                    logger.warning(f"Scenario {scenario_id} failed: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(scenarios))
        
        # Sort by scenario ID
        shells.sort(key=lambda s: s['scenario_id'])
        return shells
    
    def _run_single_optimization(
        self,
        scenario_id: int,
        block_model: pd.DataFrame,
        optimizer_func: Callable,
        scenario: Dict
    ) -> Dict:
        """Run pit optimization for a single scenario."""
        
        # Prepare block model for this scenario
        bm = block_model.copy()
        opt_params = self.config.pit_optimizer_config.copy() if self.config.pit_optimizer_config else {}
        
        if scenario['type'] == 'grade_realization':
            # Use specific grade realization
            grade_col = scenario['grade_column']
            if grade_col in bm.columns:
                bm['grade'] = bm[grade_col]
        
        elif scenario['type'] == 'parameter_sample':
            # Update optimization parameters
            opt_params.update(scenario['parameters'])
        
        # Run optimization
        selected_blocks, kpis = optimizer_func(bm, opt_params)
        
        return {
            'scenario_id': scenario_id,
            'selected_blocks': selected_blocks,
            'scenario': scenario,
            'kpis': kpis
        }
    
    def _compute_value_at_risk(
        self,
        result: ProbShellResult,
        shells: List[Dict],
        block_model: pd.DataFrame
    ):
        """Compute value at risk metrics."""
        
        if 'tonnage' not in block_model.columns:
            return
        
        # Extract ore tonnes per shell
        ore_tonnes_list = []
        metal_list = []
        
        for shell_data in shells:
            selected = shell_data['selected_blocks']
            selected_df = block_model.iloc[selected]
            
            ore_tonnes = selected_df['tonnage'].sum()
            ore_tonnes_list.append(ore_tonnes)
            
            if 'grade' in selected_df.columns:
                metal = (selected_df['grade'] * selected_df['tonnage']).sum()
                metal_list.append(metal)
        
        # Compute percentiles
        if ore_tonnes_list:
            result.expected_ore_tonnes = np.mean(ore_tonnes_list)
            result.p10_ore_tonnes = np.percentile(ore_tonnes_list, 10)
            result.p90_ore_tonnes = np.percentile(ore_tonnes_list, 90)
        
        if metal_list:
            result.expected_metal = np.mean(metal_list)
            result.p10_metal = np.percentile(metal_list, 10)
            result.p90_metal = np.percentile(metal_list, 90)
        
        logger.info(f"Ore tonnes: {result.expected_ore_tonnes/1e6:.1f}Mt "
                   f"(P10={result.p10_ore_tonnes/1e6:.1f}Mt, P90={result.p90_ore_tonnes/1e6:.1f}Mt)")
        
        if metal_list:
            logger.info(f"Metal: {result.expected_metal:.0f}t "
                       f"(P10={result.p10_metal:.0f}t, P90={result.p90_metal:.0f}t)")
    
    def export_risk_map(
        self,
        result: ProbShellResult,
        block_model: pd.DataFrame,
        output_path: str
    ):
        """
        Export risk map to CSV with mining probabilities.
        
        Args:
            result: ProbShellResult
            block_model: Original block model with coordinates
            output_path: Output CSV path
        """
        # Merge probabilities with block model
        risk_df = block_model.copy()
        risk_df['mining_probability'] = result.mining_probability
        
        # Add risk category
        risk_df['risk_category'] = 'marginal'
        risk_df.loc[risk_df['mining_probability'] >= self.config.robust_threshold, 'risk_category'] = 'robust'
        risk_df.loc[risk_df['mining_probability'] <= self.config.fringe_threshold, 'risk_category'] = 'fringe'
        
        # Export
        risk_df.to_csv(output_path, index=False)
        logger.info(f"Risk map exported to {output_path}")
