"""
Latin Hypercube Sampling (LHS) for Efficient Multi-Variate Sampling

Provides stratified sampling of correlated uncertain parameters:
- Efficient coverage of multi-dimensional parameter space
- Correlation control via rank-based methods
- Integration with Monte Carlo simulation
- Reduced variance vs simple random sampling
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm, qmc
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


@dataclass
class LHSConfig:
    """Configuration for Latin Hypercube Sampling."""
    
    n_samples: int = 100
    n_dimensions: int = 1
    
    # Correlation structure
    correlation_matrix: Optional[np.ndarray] = None
    
    # Optimization
    criterion: str = 'maximin'  # 'maximin', 'correlation', 'random'
    iterations: int = 1000       # For optimization
    
    random_seed: Optional[int] = None
    
    def validate(self):
        """Validate configuration."""
        if self.n_samples < 10:
            raise ValueError("n_samples must be >= 10")
        
        if self.n_dimensions < 1:
            raise ValueError("n_dimensions must be >= 1")
        
        if self.correlation_matrix is not None:
            if self.correlation_matrix.shape != (self.n_dimensions, self.n_dimensions):
                raise ValueError("correlation_matrix shape must match n_dimensions")
            
            # Check symmetry and valid correlation
            if not np.allclose(self.correlation_matrix, self.correlation_matrix.T):
                raise ValueError("correlation_matrix must be symmetric")
            
            if not np.allclose(np.diag(self.correlation_matrix), 1.0):
                raise ValueError("correlation_matrix diagonal must be 1.0")


class LHSSampler:
    """
    Latin Hypercube Sampler for efficient multi-variate parameter sampling.
    """
    
    def __init__(self, config: LHSConfig):
        """Initialize LHS sampler."""
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        logger.info(f"Initialized LHS sampler: {config.n_samples} samples, {config.n_dimensions} dimensions")
    
    def sample_uniform(self) -> np.ndarray:
        """
        Generate LHS samples in unit hypercube [0,1]^d.
        
        Returns:
            Array of shape (n_samples, n_dimensions)
        """
        # Use scipy's LHS implementation
        sampler = qmc.LatinHypercube(d=self.config.n_dimensions, seed=self.config.random_seed)
        samples = sampler.random(n=self.config.n_samples)
        
        # Apply optimization criterion if specified
        if self.config.criterion == 'maximin':
            samples = self._optimize_maximin(samples)
        elif self.config.criterion == 'correlation':
            samples = self._optimize_correlation(samples)
        
        return samples
    
    def sample_distributions(
        self,
        distributions: List[Dict[str, any]]
    ) -> np.ndarray:
        """
        Generate LHS samples from specified distributions.
        
        Args:
            distributions: List of distribution specs, each with:
                - 'type': 'normal', 'triangular', 'uniform', 'lognormal'
                - 'params': distribution parameters
        
        Returns:
            Array of shape (n_samples, n_dimensions)
        """
        if len(distributions) != self.config.n_dimensions:
            raise ValueError("Must provide distribution for each dimension")
        
        # Generate uniform LHS samples
        uniform_samples = self.sample_uniform()
        
        # Apply correlation if specified
        if self.config.correlation_matrix is not None:
            uniform_samples = self._induce_correlation(uniform_samples)
        
        # Transform to target distributions
        samples = np.zeros_like(uniform_samples)
        
        for i, dist_spec in enumerate(distributions):
            dist_type = dist_spec['type']
            params = dist_spec['params']
            
            if dist_type == 'normal':
                samples[:, i] = norm.ppf(uniform_samples[:, i], loc=params['mean'], scale=params['std'])
            
            elif dist_type == 'uniform':
                samples[:, i] = params['min'] + uniform_samples[:, i] * (params['max'] - params['min'])
            
            elif dist_type == 'triangular':
                samples[:, i] = self._triangular_ppf(
                    uniform_samples[:, i],
                    params['min'],
                    params['mode'],
                    params['max']
                )
            
            elif dist_type == 'lognormal':
                samples[:, i] = np.exp(
                    norm.ppf(uniform_samples[:, i], loc=params['mu'], scale=params['sigma'])
                )
            
            else:
                raise ValueError(f"Unknown distribution type: {dist_type}")
        
        return samples
    
    def sample_to_dataframe(
        self,
        param_specs: Dict[str, Dict[str, any]]
    ) -> pd.DataFrame:
        """
        Generate LHS samples as DataFrame with named parameters.
        
        Args:
            param_specs: Dict mapping parameter name -> distribution spec
                        e.g., {'price': {'type': 'normal', 'params': {'mean': 100, 'std': 10}}}
        
        Returns:
            DataFrame with columns for each parameter
        """
        param_names = list(param_specs.keys())
        distributions = [param_specs[name] for name in param_names]
        
        samples = self.sample_distributions(distributions)
        
        return pd.DataFrame(samples, columns=param_names)
    
    def _optimize_maximin(self, samples: np.ndarray) -> np.ndarray:
        """
        Optimize LHS design using maximin criterion (maximize minimum distance).
        """
        best_samples = samples.copy()
        best_min_dist = self._min_distance(samples)
        
        for _ in range(self.config.iterations):
            # Perturbation: swap random elements in random dimension
            trial = samples.copy()
            dim = self.rng.integers(0, self.config.n_dimensions)
            i, j = self.rng.choice(self.config.n_samples, size=2, replace=False)
            trial[i, dim], trial[j, dim] = trial[j, dim], trial[i, dim]
            
            # Evaluate
            min_dist = self._min_distance(trial)
            if min_dist > best_min_dist:
                best_samples = trial
                best_min_dist = min_dist
        
        logger.debug(f"Maximin optimization improved min distance to {best_min_dist:.6f}")
        return best_samples
    
    def _optimize_correlation(self, samples: np.ndarray) -> np.ndarray:
        """
        Optimize LHS design to minimize inter-variable correlation.
        """
        best_samples = samples.copy()
        best_corr = self._max_correlation(samples)
        
        for _ in range(self.config.iterations):
            # Perturbation
            trial = samples.copy()
            dim = self.rng.integers(0, self.config.n_dimensions)
            i, j = self.rng.choice(self.config.n_samples, size=2, replace=False)
            trial[i, dim], trial[j, dim] = trial[j, dim], trial[i, dim]
            
            # Evaluate
            max_corr = self._max_correlation(trial)
            if max_corr < best_corr:
                best_samples = trial
                best_corr = max_corr
        
        logger.debug(f"Correlation optimization reduced max correlation to {best_corr:.6f}")
        return best_samples
    
    def _induce_correlation(self, samples: np.ndarray) -> np.ndarray:
        """
        Induce correlation structure using Iman-Conover method.
        """
        if self.config.correlation_matrix is None:
            return samples
        
        # Rank transformation
        ranks = np.argsort(np.argsort(samples, axis=0), axis=0)
        
        # Cholesky decomposition of target correlation
        try:
            L = np.linalg.cholesky(self.config.correlation_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, using nearest PD matrix")
            L = self._nearest_pd_cholesky(self.config.correlation_matrix)
        
        # Generate correlated normal samples
        normal_samples = norm.ppf((ranks + 1) / (self.config.n_samples + 1))
        correlated_normal = normal_samples @ L.T
        
        # Back to uniform via CDF
        correlated_uniform = norm.cdf(correlated_normal)
        
        # Reorder original samples to match correlated ranks
        result = np.zeros_like(samples)
        for i in range(self.config.n_dimensions):
            result[:, i] = np.sort(samples[:, i])[np.argsort(correlated_uniform[:, i])]
        
        return result
    
    def _min_distance(self, samples: np.ndarray) -> float:
        """Compute minimum pairwise distance."""
        if len(samples) < 2:
            return 0.0
        distances = pdist(samples)
        return np.min(distances)
    
    def _max_correlation(self, samples: np.ndarray) -> float:
        """Compute maximum absolute correlation between dimensions."""
        if self.config.n_dimensions < 2:
            return 0.0
        
        corr_matrix = np.corrcoef(samples.T)
        # Exclude diagonal
        np.fill_diagonal(corr_matrix, 0)
        return np.max(np.abs(corr_matrix))
    
    def _triangular_ppf(self, p: np.ndarray, a: float, c: float, b: float) -> np.ndarray:
        """Percent point function (inverse CDF) for triangular distribution."""
        # a = min, c = mode, b = max
        Fc = (c - a) / (b - a)
        
        result = np.zeros_like(p)
        mask1 = p < Fc
        mask2 = ~mask1
        
        result[mask1] = a + np.sqrt(p[mask1] * (b - a) * (c - a))
        result[mask2] = b - np.sqrt((1 - p[mask2]) * (b - a) * (b - c))
        
        return result
    
    def _nearest_pd_cholesky(self, A: np.ndarray) -> np.ndarray:
        """Find nearest positive definite matrix and return Cholesky factor."""
        # Simple approximation: ensure positive eigenvalues
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals[eigvals < 0] = 1e-6
        A_pd = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return np.linalg.cholesky(A_pd)


def generate_lhs_samples(
    n_samples: int,
    param_specs: Dict[str, Dict[str, any]],
    correlation_matrix: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Quick LHS sample generation.
    
    Args:
        n_samples: Number of samples
        param_specs: Parameter specifications
        correlation_matrix: Optional correlation structure
        random_seed: Random seed
    
    Returns:
        DataFrame with sampled parameters
    """
    config = LHSConfig(
        n_samples=n_samples,
        n_dimensions=len(param_specs),
        correlation_matrix=correlation_matrix,
        random_seed=random_seed
    )
    
    sampler = LHSSampler(config)
    return sampler.sample_to_dataframe(param_specs)
