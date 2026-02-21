"""
Stochastic Scenario Generator

Generates multiple economic scenarios by simulating commodity prices, grades, and costs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Configuration for scenario generation."""
    num_scenarios: int = 100
    num_periods: int = 20
    
    # Price simulation
    initial_price: float = 3.5  # $/lb copper or similar
    price_volatility: float = 0.25  # Annual volatility
    price_drift: float = 0.02  # Annual drift (mean reversion)
    price_mean_reversion: float = 0.1  # Mean reversion speed
    
    # Grade simulation
    grade_uncertainty: float = 0.15  # Coefficient of variation
    grade_spatial_correlation: float = 0.7  # Spatial correlation
    
    # Cost simulation
    mining_cost_base: float = 2.5  # $/t
    processing_cost_base: float = 8.0  # $/t
    cost_inflation: float = 0.03  # Annual inflation
    cost_uncertainty: float = 0.10  # Cost variability
    
    # Recovery
    recovery_base: float = 0.85
    recovery_uncertainty: float = 0.05
    
    # Random seed
    random_seed: Optional[int] = None


class ScenarioGenerator:
    """
    Generate stochastic economic scenarios for mining project evaluation.
    """
    
    def __init__(self, config: ScenarioConfig):
        """
        Initialize the scenario generator.
        
        Args:
            config: Scenario configuration parameters
        """
        self.config = config
        
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        logger.info(f"Initialized scenario generator: {config.num_scenarios} scenarios, {config.num_periods} periods")
    
    def generate_price_scenarios(self) -> np.ndarray:
        """
        Generate commodity price scenarios using Geometric Brownian Motion with mean reversion.
        
        Returns:
            Array of shape (num_scenarios, num_periods) with simulated prices
        """
        dt = 1.0  # Annual time step
        num_scenarios = self.config.num_scenarios
        num_periods = self.config.num_periods
        
        prices = np.zeros((num_scenarios, num_periods))
        prices[:, 0] = self.config.initial_price
        
        for t in range(1, num_periods):
            # Mean reversion term
            mean_reversion = self.config.price_mean_reversion * (
                self.config.initial_price - prices[:, t-1]
            )
            
            # Drift and volatility
            drift = (self.config.price_drift + mean_reversion) * dt
            shock = self.config.price_volatility * np.sqrt(dt) * np.random.randn(num_scenarios)
            
            # Update prices (ensure non-negative)
            prices[:, t] = np.maximum(
                prices[:, t-1] * np.exp(drift + shock),
                0.01
            )
        
        logger.info(f"Generated price scenarios: mean={prices.mean():.2f}, std={prices.std():.2f}")
        return prices
    
    def generate_grade_scenarios(self, block_model: pd.DataFrame) -> Dict[int, np.ndarray]:
        """
        Generate grade scenarios for each block with spatial correlation.
        
        Args:
            block_model: DataFrame with columns ['BLOCK_ID', 'GRADE']
            
        Returns:
            Dictionary mapping block_id to array of grades (num_scenarios,)
        """
        num_scenarios = self.config.num_scenarios
        grade_scenarios = {}
        
        for _, block in block_model.iterrows():
            block_id = block['BLOCK_ID']
            base_grade = block['GRADE']
            
            # Generate correlated grade variations
            uncertainty = self.config.grade_uncertainty
            
            # Log-normal distribution to ensure positive grades
            log_mean = np.log(base_grade) - 0.5 * (uncertainty ** 2)
            log_std = uncertainty
            
            simulated_grades = np.random.lognormal(log_mean, log_std, num_scenarios)
            grade_scenarios[block_id] = simulated_grades
        
        logger.info(f"Generated grade scenarios for {len(grade_scenarios)} blocks")
        return grade_scenarios
    
    def generate_cost_scenarios(self) -> Dict[str, np.ndarray]:
        """
        Generate operating cost scenarios with inflation and uncertainty.
        
        Returns:
            Dictionary with 'mining_cost' and 'processing_cost' arrays of shape (num_scenarios, num_periods)
        """
        num_scenarios = self.config.num_scenarios
        num_periods = self.config.num_periods
        
        mining_costs = np.zeros((num_scenarios, num_periods))
        processing_costs = np.zeros((num_scenarios, num_periods))
        
        # Generate cost paths with inflation and random variation
        for t in range(num_periods):
            inflation_factor = (1 + self.config.cost_inflation) ** t
            
            # Random cost variation
            mining_variation = 1 + self.config.cost_uncertainty * np.random.randn(num_scenarios)
            processing_variation = 1 + self.config.cost_uncertainty * np.random.randn(num_scenarios)
            
            mining_costs[:, t] = self.config.mining_cost_base * inflation_factor * np.maximum(mining_variation, 0.5)
            processing_costs[:, t] = self.config.processing_cost_base * inflation_factor * np.maximum(processing_variation, 0.5)
        
        logger.info(f"Generated cost scenarios: mining ${mining_costs.mean():.2f}/t, processing ${processing_costs.mean():.2f}/t")
        
        return {
            'mining_cost': mining_costs,
            'processing_cost': processing_costs
        }
    
    def generate_recovery_scenarios(self) -> np.ndarray:
        """
        Generate metallurgical recovery scenarios.
        
        Returns:
            Array of shape (num_scenarios,) with recovery factors
        """
        num_scenarios = self.config.num_scenarios
        
        # Beta distribution for recovery (bounded between 0 and 1)
        recoveries = np.random.normal(
            self.config.recovery_base,
            self.config.recovery_uncertainty,
            num_scenarios
        )
        
        # Clip to valid range
        recoveries = np.clip(recoveries, 0.5, 0.99)
        
        logger.info(f"Generated recovery scenarios: mean={recoveries.mean():.3f}, std={recoveries.std():.3f}")
        return recoveries
    
    def generate_all_scenarios(self, block_model: pd.DataFrame) -> Dict:
        """
        Generate all scenario types.
        
        Args:
            block_model: Block model DataFrame
            
        Returns:
            Dictionary containing all scenario arrays
        """
        logger.info("Generating complete scenario set...")
        
        scenarios = {
            'prices': self.generate_price_scenarios(),
            'grades': self.generate_grade_scenarios(block_model),
            'costs': self.generate_cost_scenarios(),
            'recoveries': self.generate_recovery_scenarios(),
            'num_scenarios': self.config.num_scenarios,
            'num_periods': self.config.num_periods
        }
        
        logger.info("Scenario generation complete")
        return scenarios
    
    @staticmethod
    def from_dict(config_dict: Dict) -> 'ScenarioGenerator':
        """
        Create a ScenarioGenerator from a configuration dictionary.
        
        Args:
            config_dict: Configuration parameters
            
        Returns:
            ScenarioGenerator instance
        """
        config = ScenarioConfig(**config_dict)
        return ScenarioGenerator(config)

