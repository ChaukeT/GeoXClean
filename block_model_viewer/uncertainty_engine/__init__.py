"""
Uncertainty Analysis Engine for Mine Planning

Provides stochastic simulation and risk quantification tools including:
- Monte Carlo simulation with dynamic re-optimization
- Bootstrapping for confidence intervals
- Latin Hypercube Sampling for efficient sampling
- Probabilistic pit shells with spatial risk analysis
- Comprehensive uncertainty dashboards

Author: Mining Optimization AI
Date: 2025-11-06
"""

from .monte_carlo import (
    MonteCarloSimulator, 
    MonteCarloConfig,
    SimulationMode,
    DistributionType,
    ParameterDistribution,
    SimulationResult,
    MonteCarloResults
)
from .bootstrap import (
    BootstrapAnalyzer, 
    BootstrapConfig,
    BootstrapMethod,
    BootstrapResult
)
from .lhs_sampler import LHSSampler, LHSConfig
from .prob_shells import (
    ProbabilisticShellAnalyzer, 
    ProbShellConfig,
    ProbShellResult
)
from .dashboard_generator import UncertaintyDashboard, DashboardConfig

__all__ = [
    # Monte Carlo
    'MonteCarloSimulator',
    'MonteCarloConfig',
    'SimulationMode',
    'DistributionType',
    'ParameterDistribution',
    'SimulationResult',
    'MonteCarloResults',
    # Bootstrap
    'BootstrapAnalyzer',
    'BootstrapConfig',
    'BootstrapMethod',
    'BootstrapResult',
    # LHS
    'LHSSampler',
    'LHSConfig',
    # Probabilistic Shells
    'ProbabilisticShellAnalyzer',
    'ProbShellConfig',
    'ProbShellResult',
    # Dashboard
    'UncertaintyDashboard',
    'DashboardConfig'
]
