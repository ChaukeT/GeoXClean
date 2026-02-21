"""
Bootstrap Analysis for Non-Parametric Confidence Intervals (High-Performance).

OPTIMIZATION: Numba JIT for parallel resampling.

Provides distribution-free confidence interval estimation through resampling:
- Grade statistics CI (mean, variance, percentiles)
- Metal totals CI
- Variogram parameter uncertainty
- Annual KPI confidence bands
- Non-parametric hypothesis testing
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

logger = logging.getLogger(__name__)


class BootstrapMethod(Enum):
    """Bootstrap resampling methods."""
    SIMPLE = "simple"              # Simple random resampling with replacement
    STRATIFIED = "stratified"      # Stratify by domain/category
    BLOCK = "block"                # Block bootstrap for spatial data
    PARAMETRIC = "parametric"      # Parametric bootstrap (fit distribution, resample)


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap analysis."""
    
    n_iterations: int = 1000
    method: BootstrapMethod = BootstrapMethod.SIMPLE
    confidence_level: float = 0.95
    random_seed: Optional[int] = None
    
    # Stratified bootstrap
    stratify_column: Optional[str] = None
    
    # Block bootstrap
    block_length: Optional[int] = None  # For spatial blocks
    
    # Parametric bootstrap
    distribution: Optional[str] = None  # 'normal', 'lognormal', etc.
    
    def validate(self):
        """Validate configuration."""
        if self.n_iterations < 100:
            raise ValueError("n_iterations should be >= 100 for stable CI")
        
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        if self.method == BootstrapMethod.STRATIFIED and self.stratify_column is None:
            raise ValueError("stratify_column required for STRATIFIED method")
        
        if self.method == BootstrapMethod.BLOCK and self.block_length is None:
            raise ValueError("block_length required for BLOCK method")


@dataclass
class BootstrapResult:
    """Results from bootstrap analysis."""
    
    statistic_name: str
    original_statistic: float
    
    # Bootstrap distribution
    bootstrap_samples: np.ndarray
    
    # Confidence intervals
    ci_lower: float
    ci_upper: float
    confidence_level: float
    
    # Additional statistics
    bootstrap_mean: float
    bootstrap_std: float
    bootstrap_bias: float  # bootstrap_mean - original_statistic
    
    def __repr__(self) -> str:
        return (
            f"BootstrapResult({self.statistic_name})\n"
            f"  Original: {self.original_statistic:.4f}\n"
            f"  Bootstrap Mean: {self.bootstrap_mean:.4f}\n"
            f"  {self.confidence_level*100:.0f}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]\n"
            f"  Bias: {self.bootstrap_bias:.4f}"
        )


# --- NUMBA KERNELS ---

@jit(nopython=True, parallel=True, cache=True)
def _numba_bootstrap_mean(data: np.ndarray, n_iter: int) -> np.ndarray:
    """Ultra-fast bootstrap mean calculation."""
    n = len(data)
    results = np.empty(n_iter, dtype=np.float64)
    
    for i in prange(n_iter):
        sum_val = 0.0
        for j in range(n):
            idx = np.random.randint(0, n)
            sum_val += data[idx]
        results[i] = sum_val / n
        
    return results

@jit(nopython=True, parallel=True, cache=True)
def _numba_bootstrap_block_mean(data: np.ndarray, n_iter: int, block_len: int) -> np.ndarray:
    """Block bootstrap for time-series/spatial data."""
    n = len(data)
    results = np.empty(n_iter, dtype=np.float64)
    n_blocks = int(np.ceil(n / block_len))
    
    for i in prange(n_iter):
        sum_val = 0.0
        count = 0
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n - block_len + 1)
            for k in range(block_len):
                if count < n:
                    sum_val += data[start_idx + k]
                    count += 1
        results[i] = sum_val / count if count > 0 else 0.0
            
    return results


class BootstrapAnalyzer:
    """
    Bootstrap analyzer for non-parametric confidence intervals.
    """
    
    def __init__(self, config: BootstrapConfig):
        """Initialize bootstrap analyzer."""
        config.validate()
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        # Seed numba's random state if needed (mostly affects global numpy state)
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        logger.info(f"Initialized Bootstrap analyzer: {config.n_iterations} iterations, method={config.method.value}")
    
    def analyze_statistic(
        self,
        data: np.ndarray,
        statistic_func: Callable[[np.ndarray], float] = np.mean,
        statistic_name: str = "statistic"
    ) -> BootstrapResult:
        """
        Bootstrap a single statistic.
        
        Args:
            data: Input data array
            statistic_func: Function that computes statistic from data
            statistic_name: Name for reporting
        
        Returns:
            BootstrapResult with CI and distribution
        """
        logger.info(f"Bootstrapping {statistic_name} with {self.config.n_iterations} iterations...")
        
        data = np.asarray(data, dtype=np.float64)
        
        # Compute original statistic
        original_stat = statistic_func(data)
        
        # Optimization: Use Numba for Mean
        if statistic_func == np.mean:
            if self.config.method == BootstrapMethod.BLOCK and self.config.block_length:
                bootstrap_stats = _numba_bootstrap_block_mean(
                    data, self.config.n_iterations, self.config.block_length
                )
            elif self.config.method == BootstrapMethod.SIMPLE:
                bootstrap_stats = _numba_bootstrap_mean(data, self.config.n_iterations)
            else:
                bootstrap_stats = self._slow_bootstrap(data, statistic_func)
        else:
            bootstrap_stats = self._slow_bootstrap(data, statistic_func)
        
        # Compute confidence interval
        alpha = 1 - self.config.confidence_level
        ci_lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
        
        # Compute additional statistics
        bootstrap_mean = np.mean(bootstrap_stats)
        bootstrap_std = np.std(bootstrap_stats)
        bootstrap_bias = bootstrap_mean - original_stat
        
        result = BootstrapResult(
            statistic_name=statistic_name,
            original_statistic=original_stat,
            bootstrap_samples=bootstrap_stats,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=self.config.confidence_level,
            bootstrap_mean=bootstrap_mean,
            bootstrap_std=bootstrap_std,
            bootstrap_bias=bootstrap_bias
        )
        
        logger.info(f"Bootstrap complete: {statistic_name} = {original_stat:.4f}, "
                   f"CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return result
    
    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        statistics: Dict[str, Tuple[str, Callable[[pd.Series], float]]]
    ) -> Dict[str, BootstrapResult]:
        """
        Bootstrap multiple statistics from a DataFrame.
        
        Args:
            df: Input DataFrame
            statistics: Dict mapping name -> (column, function)
                       e.g., {'mean_grade': ('grade', np.mean)}
        
        Returns:
            Dict of BootstrapResult for each statistic
        """
        results = {}
        
        for stat_name, (column, func) in statistics.items():
            if column not in df.columns:
                logger.warning(f"Column {column} not found, skipping {stat_name}")
                continue
            
            data = df[column].dropna().values
            result = self.analyze_statistic(data, func, stat_name)
            results[stat_name] = result
        
        return results
    
    def analyze_grouped_statistic(
        self,
        df: pd.DataFrame,
        column: str,
        group_column: str,
        statistic_func: Callable[[np.ndarray], float],
        statistic_name: str = "statistic"
    ) -> pd.DataFrame:
        """
        Bootstrap a statistic for each group.
        
        Args:
            df: Input DataFrame
            column: Data column
            group_column: Grouping column
            statistic_func: Function to compute
            statistic_name: Name for reporting
        
        Returns:
            DataFrame with columns: group, original, ci_lower, ci_upper
        """
        results = []
        
        for group, group_df in df.groupby(group_column):
            data = group_df[column].dropna().values
            if len(data) < 10:
                logger.warning(f"Group {group} has only {len(data)} samples, skipping")
                continue
            
            result = self.analyze_statistic(data, statistic_func, f"{statistic_name}_{group}")
            
            results.append({
                'group': group,
                'original': result.original_statistic,
                'bootstrap_mean': result.bootstrap_mean,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'bias': result.bootstrap_bias
            })
        
        return pd.DataFrame(results)
    
    def analyze_ratio(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
        ratio_name: str = "ratio"
    ) -> BootstrapResult:
        """
        Bootstrap a ratio statistic (e.g., metal/ore ratio).
        
        Args:
            numerator: Numerator values
            denominator: Denominator values
            ratio_name: Name for reporting
        
        Returns:
            BootstrapResult for the ratio
        """
        if len(numerator) != len(denominator):
            raise ValueError("numerator and denominator must have same length")
        
        # Combine into structured array for joint resampling
        data = np.column_stack([numerator, denominator])
        
        def ratio_func(data_resample):
            num = data_resample[:, 0]
            den = data_resample[:, 1]
            return np.sum(num) / np.sum(den) if np.sum(den) > 0 else 0
        
        return self.analyze_statistic(data, ratio_func, ratio_name)
    
    def analyze_annual_series(
        self,
        df: pd.DataFrame,
        value_column: str,
        period_column: str = 'period',
        statistic_func: Callable[[np.ndarray], float] = np.mean
    ) -> pd.DataFrame:
        """
        Bootstrap annual time series (e.g., annual NPV, grade).
        
        Args:
            df: DataFrame with period and value columns
            value_column: Column to analyze
            period_column: Period/year column
            statistic_func: Statistic to compute per period
        
        Returns:
            DataFrame with columns: period, original, ci_lower, ci_upper
        """
        results = []
        
        for period in sorted(df[period_column].unique()):
            period_data = df[df[period_column] == period][value_column].values
            
            if len(period_data) < 10:
                continue
            
            result = self.analyze_statistic(
                period_data, 
                statistic_func, 
                f"{value_column}_period_{period}"
            )
            
            results.append({
                'period': period,
                'original': result.original_statistic,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper
            })
        
        return pd.DataFrame(results)
    
    def compare_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        statistic_func: Callable[[np.ndarray], float] = np.mean
    ) -> Tuple[float, float, bool]:
        """
        Test if two groups have significantly different statistics.
        
        Args:
            group1: First group data
            group2: Second group data
            statistic_func: Statistic to compare
        
        Returns:
            (difference, p_value, is_significant)
        """
        # Observed difference
        obs_diff = statistic_func(group1) - statistic_func(group2)
        
        # Permutation test via bootstrap
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        
        null_diffs = []
        for i in range(self.config.n_iterations):
            # Resample from combined
            perm = self.rng.permutation(combined)
            perm_group1 = perm[:n1]
            perm_group2 = perm[n1:]
            
            perm_diff = statistic_func(perm_group1) - statistic_func(perm_group2)
            null_diffs.append(perm_diff)
        
        null_diffs = np.array(null_diffs)
        
        # Two-tailed p-value
        p_value = np.mean(np.abs(null_diffs) >= np.abs(obs_diff))
        
        # Significance test
        is_significant = p_value < (1 - self.config.confidence_level)
        
        logger.info(f"Group comparison: difference={obs_diff:.4f}, p={p_value:.4f}, significant={is_significant}")
        
        return obs_diff, p_value, is_significant
    
    def _resample_advanced(self, data: np.ndarray, strata: Optional[np.ndarray] = None) -> np.ndarray:
        """Advanced resampling methods (stratified, block, parametric)."""
        
        if self.config.method == BootstrapMethod.STRATIFIED:
            # Stratified bootstrap: resample within each stratum proportionally
            if strata is None:
                # If no strata provided, fall back to simple
                logger.warning("Stratified bootstrap requires strata, using simple")
                return self.rng.choice(data, size=len(data), replace=True)
            
            unique_strata = np.unique(strata)
            resampled = []
            
            for s in unique_strata:
                stratum_mask = strata == s
                stratum_data = data[stratum_mask]
                n_stratum = len(stratum_data)
                
                # Resample within stratum with replacement
                stratum_resample = self.rng.choice(stratum_data, size=n_stratum, replace=True)
                resampled.extend(stratum_resample)
            
            return np.array(resampled)
        
        elif self.config.method == BootstrapMethod.BLOCK:
            # Block bootstrap for time series or spatial data
            n = len(data)
            block_len = self.config.block_length or int(np.sqrt(n))
            
            n_blocks = int(np.ceil(n / block_len))
            block_starts = self.rng.integers(0, n - block_len + 1, size=n_blocks)
            
            resample = []
            for start in block_starts:
                resample.extend(data[start:start + block_len])
            
            return np.array(resample[:n])
        
        elif self.config.method == BootstrapMethod.PARAMETRIC:
            # Fit distribution and resample
            if self.config.distribution == 'normal':
                mean, std = np.mean(data), np.std(data)
                return self.rng.normal(mean, std, size=len(data))
            elif self.config.distribution == 'lognormal':
                log_data = np.log(data[data > 0])
                mu, sigma = np.mean(log_data), np.std(log_data)
                return self.rng.lognormal(mu, sigma, size=len(data))
            else:
                logger.warning(f"Unknown distribution {self.config.distribution}, using simple")
                return self.rng.choice(data, size=len(data), replace=True)
        
        else:
            return self.rng.choice(data, size=len(data), replace=True)
    
    def _slow_bootstrap(self, data: np.ndarray, func: Callable) -> np.ndarray:
        """Fallback for complex statistics or methods not supported by Numba."""
        logger.debug("Using standard loop bootstrap (slow).")
        stats = np.empty(self.config.n_iterations)
        n = len(data)
        for i in range(self.config.n_iterations):
            resample = np.random.choice(data, n, replace=True)
            stats[i] = func(resample)
        return stats


def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float] = np.mean,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Quick bootstrap confidence interval calculation.
    
    Args:
        data: Input data
        statistic_func: Function to compute statistic
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        (ci_lower, ci_upper)
    """
    config = BootstrapConfig(
        n_iterations=n_iterations,
        confidence_level=confidence_level,
        random_seed=random_seed
    )
    
    analyzer = BootstrapAnalyzer(config)
    result = analyzer.analyze_statistic(data, statistic_func)
    
    return result.ci_lower, result.ci_upper


def bootstrap_bca_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float] = np.mean,
    n_iterations: int = 2000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Bias-Corrected and Accelerated (BCa) bootstrap confidence interval.
    
    Industry-standard method for small samples that corrects for bias and skewness.
    Reference: Efron, B. (1987) "Better Bootstrap Confidence Intervals"
    
    Args:
        data: Input data
        statistic_func: Function to compute statistic
        n_iterations: Number of bootstrap iterations (recommend >= 2000 for BCa)
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
    
    Returns:
        (ci_lower, ci_upper, original_statistic)
    """
    from scipy.stats import norm as scipy_norm
    
    rng = np.random.default_rng(random_seed)
    n = len(data)
    
    # Original statistic
    theta_hat = statistic_func(data)
    
    # Bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_iterations):
        resample = rng.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Bias correction factor (z0)
    prop_below = np.sum(bootstrap_stats < theta_hat) / n_iterations
    z0 = scipy_norm.ppf(prop_below) if 0 < prop_below < 1 else 0.0
    
    # Acceleration factor (a) using jackknife
    jackknife_stats = []
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats.append(statistic_func(jackknife_sample))
    
    jackknife_stats = np.array(jackknife_stats)
    jackknife_mean = np.mean(jackknife_stats)
    
    # Acceleration formula
    numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denominator = 6.0 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)
    
    if denominator != 0:
        a = numerator / denominator
    else:
        a = 0.0
    
    # BCa adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_lower = scipy_norm.ppf(alpha / 2)
    z_alpha_upper = scipy_norm.ppf(1 - alpha / 2)
    
    # BCa formula
    def bca_percentile(z_alpha):
        num = z0 + z_alpha
        denom = 1 - a * num
        if denom == 0:
            return scipy_norm.cdf(z_alpha)
        return scipy_norm.cdf(z0 + num / denom)
    
    p_lower = bca_percentile(z_alpha_lower)
    p_upper = bca_percentile(z_alpha_upper)
    
    # Clip to valid percentile range
    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)
    
    ci_lower = np.percentile(bootstrap_stats, p_lower * 100)
    ci_upper = np.percentile(bootstrap_stats, p_upper * 100)
    
    logger.info(f"BCa CI: [{ci_lower:.4f}, {ci_upper:.4f}] (z0={z0:.4f}, a={a:.4f})")
    
    return ci_lower, ci_upper, theta_hat
