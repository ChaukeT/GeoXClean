"""
Configuration Loader for IRR Analysis

Handles loading and saving of scenario configuration files (YAML/JSON).
Provides typed configuration dataclasses for IRR and Pit optimization.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class IRRConfig:
    """Configuration for IRR analysis."""
    block_model: Any  # pd.DataFrame or BlockModel
    scenario_config: Dict[str, Any]
    economic_params: Dict[str, Any]
    irr_search: Dict[str, Any] = field(default_factory=lambda: {'alpha': 0.80, 'r_low': 0.0, 'r_high': 0.50})
    num_periods: int = 20
    production_capacity: float = 1000000
    tolerance: float = 0.001
    max_iterations: int = 30
    parallel: bool = False
    # New: Nested pit shells for dynamic pit selection per price scenario
    # If provided, the IRR analysis will select the optimal pit shell for each
    # price scenario, ensuring rational pit boundaries that respond to price changes.
    nested_shells: Optional[Any] = None  # Dict with shell data from LerchsGrossmann


@dataclass
class PitConfig:
    """Configuration for pit optimization."""
    block_model: Any  # pd.DataFrame or BlockModel
    slope_angles: Dict[str, float]
    economic_params: Dict[str, Any]
    nested_shells: bool = False
    shell_factors: Optional[list] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleConfig:
    """Configuration for production scheduling."""
    block_model: Any  # pd.DataFrame or BlockModel
    economic_params: Dict[str, Any]
    num_periods: int = 20
    production_capacity: float = 1000000
    discount_rate: float = 0.10
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif path.suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(f"Loaded configuration from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        if path.suffix in ['.yaml', '.yml']:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    logger.info(f"Saved configuration to {config_path}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for IRR analysis.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'scenario_generation': {
            'num_scenarios': 50,  # Reduced from 100 for faster execution
            'num_periods': 15,    # Reduced from 20 for faster execution
            'random_seed': 42,
            
            'price': {
                'initial': 60.0,  # $/g for gold (approximately $1,860/oz)
                'volatility': 0.25,
                'drift': 0.02,
                'mean_reversion': 0.1
            },
            
            'grade': {
                'uncertainty': 0.15,
                'spatial_correlation': 0.7
            },
            
            'costs': {
                'mining_cost_base': 2.5,
                'processing_cost_base': 8.0,
                'inflation': 0.03,
                'uncertainty': 0.10
            },
            
            'recovery': {
                'base': 0.85,
                'uncertainty': 0.05
            }
        },
        
        'economic_parameters': {
            'metal_price': 60.0,  # $/g for gold (approximately $1,860/oz)
            'mining_cost': 2.5,  # $/tonne
            'processing_cost': 8.0,  # $/tonne
            'recovery': 0.85,  # 85% metallurgical recovery
            'selling_cost': 0.1,  # $/g
            'capex': [],
            'by_products': []  # List of by-product configurations (e.g., Cu, Ag)
            # Example by-product entry:
            # {
            #     'grade_field': 'Cu',  # Field name in block model
            #     'price': 0.008,       # $/g (or appropriate units)
            #     'recovery': 0.80,     # 80% recovery
            #     'selling_cost': 0.0   # Optional selling cost
            # }
        },
        
        'optimization': {
            'num_periods': 15,    # Reduced from 20 for faster execution
            'production_capacity': 5000000,  # 5 Mt/period (increased from 1 Mt to allow mining full deposit)
            'time_limit_per_scenario': 30,  # Reduced from 60 for faster execution
            
            # Mining operations
            'annual_rom': 5000000,  # Annual Run-of-Mine tonnage capacity
            'min_bottom_width': 30,  # Minimum bottom width of pit (meters)
            
            # Pit slope angles by rock type (degrees)
            'slope_angles': {
                'ore': 55,
                'soil': 25,
                'weathered': 45,
                'fresh': 55
            },
            
            # Pit phase scheduling
            'enable_phases': True,   # Enable multi-phase mining optimization
            'num_phases': 3,         # Number of pit phases/pushbacks
            'phase_gap': 30,         # Minimum vertical gap between phase bottoms (meters)
            'use_lerchs_grossmann': True,  # Use LG algorithm for fast pit optimization
            'simple_phase_sequencing': True  # Use simple elevation-based phasing (faster)
        },
        
        'irr_search': {
            'alpha': 0.80,
            'r_low': 0.01,
            'r_high': 0.50,
            'tolerance': 0.005,  # Increased from 0.001 for faster convergence
            'max_iterations': 20,  # Reduced from 30 for faster execution
            'parallel': True     # Enabled for faster execution using multiple cores
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['scenario_generation', 'economic_parameters', 'optimization', 'irr_search']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate scenario generation
    sg = config['scenario_generation']
    if sg['num_scenarios'] < 10:
        raise ValueError("num_scenarios must be at least 10")
    
    if sg['num_periods'] < 1:
        raise ValueError("num_periods must be at least 1")
    
    # Validate IRR search
    irr = config['irr_search']
    if not (0 <= irr['alpha'] <= 1):
        raise ValueError("alpha must be between 0 and 1")
    
    if irr['r_low'] >= irr['r_high']:
        raise ValueError("r_low must be less than r_high")
    
    logger.info("Configuration validated successfully")
    return True

