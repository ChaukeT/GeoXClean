"""
Universal Kriging Validation & Residual Analysis Utilities

Handles proper UK workflow: drift fitting → residual variogram → sill correction.
Provides validation dashboard and post-processing filters.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ResidualVariogramCalculator:
    """
    Computes proper residual variograms for Universal Kriging.

    This fits the variogram model to residuals (Z - trend) rather than raw grades,
    which is essential for stable UK interpolation.
    """

    def __init__(self):
        self.drift_fitter = DriftFitter()

    def compute_residual_variogram(self, coords: np.ndarray, values: np.ndarray,
                                  drift_type: str = 'linear',
                                  variogram_model_type: str = 'spherical',
                                  n_lags: int = 15) -> Dict[str, Any]:
        """
        Compute residual variogram for UK.

        Args:
            coords: (N, 3) coordinate array
            values: (N,) value array
            drift_type: 'constant', 'linear', or 'quadratic'
            variogram_model_type: Variogram model type
            n_lags: Number of lag classes for experimental variogram

        Returns:
            Dict with residual variogram parameters and diagnostics
        """
        try:
            # 1. Fit drift to compute residuals
            drift_fit = self.drift_fitter.fit_drift(coords, values, drift_type)

            if 'error' in drift_fit:
                return {'error': f"Drift fitting failed: {drift_fit['error']}"}

            residuals = drift_fit['residuals']

            # 2. Compute experimental variogram on residuals
            exp_vario = self._compute_experimental_variogram(coords, residuals, n_lags)

            # 3. Fit theoretical model to experimental variogram
            fitted_model = self._fit_variogram_model(exp_vario, variogram_model_type)

            # 4. Validate the fit
            validation = self._validate_variogram_fit(exp_vario, fitted_model)

            return {
                'residuals': residuals,
                'drift_fit': drift_fit,
                'experimental_variogram': exp_vario,
                'fitted_model': fitted_model,
                'validation': validation,
                'variogram_params': {
                    'model_type': variogram_model_type,
                    'sill': fitted_model['sill'],
                    'range': fitted_model['range'],
                    'nugget': fitted_model['nugget'],
                    'anisotropy': fitted_model.get('anisotropy', None)
                }
            }

        except Exception as e:
            logger.error(f"Residual variogram computation failed: {e}")
            return {'error': str(e)}

    def _compute_experimental_variogram(self, coords: np.ndarray, values: np.ndarray,
                                       n_lags: int = 15) -> Dict[str, Any]:
        """
        Compute experimental variogram from data.

        Args:
            coords: (N, 3) coordinate array
            values: (N,) value array (residuals)
            n_lags: Number of lag classes

        Returns:
            Experimental variogram data
        """
        from scipy.spatial.distance import pdist, squareform

        # Compute all pairwise distances
        distances = pdist(coords)
        differences = pdist(values.reshape(-1, 1))

        # Create lag classes
        max_dist = np.max(distances)
        lag_size = max_dist / n_lags
        lag_centers = np.arange(lag_size/2, max_dist, lag_size)

        # Compute variogram for each lag
        gamma_values = []
        n_pairs = []

        for lag_center in lag_centers:
            lag_min = lag_center - lag_size/2
            lag_max = lag_center + lag_size/2

            # Find pairs in this lag
            in_lag = (distances >= lag_min) & (distances < lag_max)

            if np.sum(in_lag) > 0:
                gamma = 0.5 * np.mean(differences[in_lag]**2)
                gamma_values.append(gamma)
                n_pairs.append(np.sum(in_lag))
            else:
                gamma_values.append(np.nan)
                n_pairs.append(0)

        # Remove lags with insufficient pairs
        valid_mask = np.array(n_pairs) >= 10  # Minimum 10 pairs per lag
        lag_centers = lag_centers[valid_mask]
        gamma_values = np.array(gamma_values)[valid_mask]
        n_pairs = np.array(n_pairs)[valid_mask]

        return {
            'lag_centers': lag_centers,
            'gamma_values': gamma_values,
            'n_pairs': n_pairs,
            'lag_size': lag_size
        }

    def _fit_variogram_model(self, exp_vario: Dict[str, Any],
                           model_type: str = 'spherical') -> Dict[str, Any]:
        """
        Fit theoretical variogram model to experimental data.

        Args:
            exp_vario: Experimental variogram data
            model_type: Type of model to fit

        Returns:
            Fitted model parameters
        """
        from scipy.optimize import minimize

        lags = exp_vario['lag_centers']
        gamma_obs = exp_vario['gamma_values']

        # Remove NaN values
        valid = ~np.isnan(gamma_obs)
        lags = lags[valid]
        gamma_obs = gamma_obs[valid]

        if len(lags) < 5:
            # Fallback to simple estimation
            sill = np.max(gamma_obs)
            range_val = lags[np.argmax(gamma_obs)]
            nugget = gamma_obs[0] if len(gamma_obs) > 0 else 0
        else:
            # Fit model using optimization
            def objective(params):
                sill, range_val, nugget = params
                gamma_pred = self._variogram_model(lags, sill, range_val, nugget, model_type)
                return np.sum((gamma_obs - gamma_pred)**2)

            # Initial guess
            sill_init = np.max(gamma_obs)
            range_init = lags[np.argmax(gamma_obs)]
            nugget_init = max(0, gamma_obs[0])

            bounds = [(0.1, sill_init*2), (range_init*0.1, range_init*2), (0, sill_init*0.5)]
            result = minimize(objective, [sill_init, range_init, nugget_init],
                            bounds=bounds, method='L-BFGS-B')

            sill, range_val, nugget = result.x

        return {
            'sill': float(sill),
            'range': float(range_val),
            'nugget': float(nugget),
            'model_type': model_type
        }

    def _variogram_model(self, h: np.ndarray, sill: float, range_val: float,
                        nugget: float, model_type: str) -> np.ndarray:
        """Compute variogram values for given model."""
        if model_type == 'spherical':
            r = h / range_val
            return np.where(r >= 1, sill + nugget,
                          nugget + sill * (1.5 * r - 0.5 * r**3))
        elif model_type == 'exponential':
            return nugget + sill * (1 - np.exp(-3 * h / range_val))
        elif model_type == 'gaussian':
            return nugget + sill * (1 - np.exp(-3 * (h / range_val)**2))
        else:
            # Default to spherical
            r = h / range_val
            return np.where(r >= 1, sill + nugget,
                          nugget + sill * (1.5 * r - 0.5 * r**3))

    def _validate_variogram_fit(self, exp_vario: Dict[str, Any],
                              fitted_model: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the quality of variogram fit."""
        lags = exp_vario['lag_centers']
        gamma_obs = exp_vario['gamma_values']

        gamma_pred = self._variogram_model(
            lags,
            fitted_model['sill'],
            fitted_model['range'],
            fitted_model['nugget'],
            fitted_model['model_type']
        )

        # Compute fit statistics
        residuals = gamma_obs - gamma_pred
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((gamma_obs - np.mean(gamma_obs))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r_squared': float(r_squared),
            'fit_quality': 'good' if r_squared > 0.8 else 'fair' if r_squared > 0.6 else 'poor'
        }


class DriftFitter:
    """
    Fits polynomial drift models for Universal Kriging.

    Supports constant, linear, and quadratic drift functions.
    """

    def __init__(self):
        self.drift_types = {
            'constant': self._constant_basis,
            'linear': self._linear_basis,
            'quadratic': self._quadratic_basis
        }

    def fit_drift(self, coords: np.ndarray, values: np.ndarray,
                  drift_type: str = 'linear') -> Dict[str, Any]:
        """
        Fit drift model to data.

        Args:
            coords: (N, 3) coordinate array
            values: (N,) value array
            drift_type: 'constant', 'linear', or 'quadratic'

        Returns:
            Dict with coefficients, residuals, R², etc.
        """
        if drift_type not in self.drift_types:
            raise ValueError(f"Unknown drift type: {drift_type}")

        # Get design matrix
        X = self.drift_types[drift_type](coords)

        # Fit using weighted least squares (uniform weights for now)
        try:
            # Solve normal equations: (X^T X) β = X^T y
            XtX = X.T @ X
            Xty = X.T @ values

            # Add small regularization for numerical stability
            reg_factor = 1e-10 * np.trace(XtX) / X.shape[1]
            XtX += np.eye(X.shape[1]) * reg_factor

            beta = np.linalg.solve(XtX, Xty)

            # Compute predictions and residuals
            predictions = X @ beta
            residuals = values - predictions

            # Compute statistics
            ss_total = np.sum((values - np.mean(values))**2)
            ss_residual = np.sum(residuals**2)
            r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0

            rmse = np.sqrt(np.mean(residuals**2))
            mean_residual = np.mean(residuals)

            # Variance explained by drift vs residuals
            var_values = np.var(values)
            var_drift = np.var(predictions)
            var_residual = np.var(residuals)

            drift_contribution_pct = (var_drift / var_values * 100) if var_values > 0 else 0

            return {
                'coefficients': beta,
                'predictions': predictions,
                'residuals': residuals,
                'r_squared': r_squared,
                'rmse': rmse,
                'mean_residual': mean_residual,
                'variance_explained': {
                    'total': var_values,
                    'drift': var_drift,
                    'residual': var_residual,
                    'drift_percentage': drift_contribution_pct
                },
                'drift_type': drift_type,
                'design_matrix': X
            }

        except np.linalg.LinAlgError as e:
            logger.error(f"Drift fitting failed: {e}")
            return {
                'error': f"Linear algebra error: {e}",
                'fallback_residuals': values - np.mean(values),
                'fallback_variance': np.var(values)
            }

    def _constant_basis(self, coords: np.ndarray) -> np.ndarray:
        """Constant drift: β₀"""
        n = coords.shape[0]
        return np.ones((n, 1))

    def _linear_basis(self, coords: np.ndarray) -> np.ndarray:
        """Linear drift: β₀ + βₓx + βᵧy + β_z"""
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return np.column_stack([np.ones_like(x), x, y, z])

    def _quadratic_basis(self, coords: np.ndarray) -> np.ndarray:
        """Quadratic drift: full 2nd order polynomial"""
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        return np.column_stack([
            np.ones_like(x),  # β₀
            x, y, z,          # βₓ, βᵧ, β_z
            x*x, y*y, z*z,    # β_xx, β_yy, β_zz
            x*y, x*z, y*z     # β_xy, β_xz, β_yz
        ])

    def predict_drift(self, coords: np.ndarray, drift_fit: Dict[str, Any]) -> np.ndarray:
        """
        Predict drift values at new coordinates.

        Args:
            coords: (M, 3) coordinate array
            drift_fit: Result from fit_drift()

        Returns:
            Drift predictions at coords
        """
        if 'error' in drift_fit:
            return np.zeros(coords.shape[0])

        drift_type = drift_fit['drift_type']
        X = self.drift_types[drift_type](coords)
        beta = drift_fit['coefficients']

        return X @ beta


class ResidualVariogramAnalyzer:
    """
    Computes experimental variograms on residuals and fits models.
    """

    def __init__(self):
        pass

    def compute_residual_variogram(self, coords: np.ndarray,
                                 residuals: np.ndarray,
                                 variogram_params: Dict[str, Any],
                                 n_lags: int = 15,
                                 max_lag_distance: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute experimental variogram on residuals.

        Args:
            coords: (N, 3) coordinate array
            residuals: (N,) residual values
            variogram_params: Original variogram parameters for binning
            n_lags: Number of lag bins
            max_lag_distance: Maximum lag distance (auto if None)

        Returns:
            Experimental variogram data
        """
        from .geostats_utils import NeighborSearcher

        # Set up lag binning
        if max_lag_distance is None:
            coord_range = coords.max(axis=0) - coords.min(axis=0)
            max_lag_distance = np.max(coord_range) * 0.3  # 30% of max dimension

        lag_step = max_lag_distance / n_lags

        # Compute pairwise distances and semivariances
        n_points = len(coords)
        distances = []
        semivariances = []

        # Use batched computation for large datasets
        batch_size = min(10000, n_points)

        for i in range(0, n_points, batch_size):
            end_i = min(i + batch_size, n_points)
            batch_coords = coords[i:end_i]
            batch_residuals = residuals[i:end_i]

            # Compute distances within batch and to previous points
            for j in range(i, n_points):
                if j < end_i:
                    # Within batch
                    j_local = j - i
                    dist = np.sqrt(np.sum((batch_coords[j_local] - batch_coords)**2, axis=1))
                    gamma = 0.5 * (batch_residuals[j_local] - batch_residuals)**2
                else:
                    # To previous batches
                    dist = np.sqrt(np.sum((coords[j] - batch_coords)**2, axis=1))
                    gamma = 0.5 * (residuals[j] - batch_residuals)**2

                # Add valid pairs
                valid = (dist > 0) & (dist <= max_lag_distance) & np.isfinite(gamma)
                distances.extend(dist[valid])
                semivariances.extend(gamma[valid])

        distances = np.array(distances)
        semivariances = np.array(semivariances)

        if len(distances) == 0:
            return {'error': 'No valid distance pairs found'}

        # Bin into lag classes
        lag_centers = np.arange(lag_step/2, max_lag_distance, lag_step)
        lag_counts = np.zeros(len(lag_centers))
        lag_gammas = np.zeros(len(lag_centers))

        for i, center in enumerate(lag_centers):
            lag_min = center - lag_step/2
            lag_max = center + lag_step/2

            in_lag = (distances >= lag_min) & (distances <= lag_max)
            if np.any(in_lag):
                lag_counts[i] = np.sum(in_lag)
                lag_gammas[i] = np.mean(semivariances[in_lag])

        # Remove empty lags
        valid_lags = lag_counts > 0
        lag_centers = lag_centers[valid_lags]
        lag_gammas = lag_gammas[valid_lags]
        lag_counts = lag_counts[valid_lags]

        return {
            'lag_distances': lag_centers,
            'semivariances': lag_gammas,
            'pair_counts': lag_counts,
            'lag_step': lag_step,
            'max_lag_distance': max_lag_distance,
            'residual_variance': np.var(residuals)
        }

    def fit_residual_variogram_model(self, exp_variogram: Dict[str, Any],
                                   original_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fit variogram model to residual experimental variogram.

        Args:
            exp_variogram: Experimental variogram from compute_residual_variogram()
            original_params: Original variogram parameters

        Returns:
            Fitted residual variogram parameters
        """
        if 'error' in exp_variogram:
            return {'error': 'No experimental variogram available'}

        lag_distances = exp_variogram['lag_distances']
        semivariances = exp_variogram['semivariances']
        residual_var = exp_variogram['residual_variance']

        # Use same model type as original, but fit sill to residual variance
        model_type = original_params.get('model_type', 'spherical')
        nugget = original_params.get('nugget', 0.0)
        range_val = original_params.get('range', 100.0)

        # For residuals, sill should be residual_var - nugget
        # But cap at reasonable values
        fitted_sill = max(0.01, residual_var - nugget)

        return {
            'model_type': model_type,
            'range': range_val,  # Keep original range
            'sill': fitted_sill,  # Fitted to residual variance
            'nugget': nugget,     # Keep original nugget
            'anisotropy': original_params.get('anisotropy', {}),
            'residual_variance': residual_var,
            'fitted_sill': fitted_sill
        }


class UKValidationDashboard:
    """
    Comprehensive validation dashboard for Universal Kriging results.
    """

    def __init__(self):
        self.drift_fitter = DriftFitter()
        self.residual_analyzer = ResidualVariogramAnalyzer()

    def run_full_validation(self, uk_result: Dict[str, Any],
                          original_data: pd.DataFrame,
                          variable: str) -> Dict[str, Any]:
        """
        Run complete UK validation workflow.

        Args:
            uk_result: UK result dict
            original_data: Original data with coordinates
            variable: Variable name

        Returns:
            Complete validation report
        """
        logger.info("Starting UK validation workflow...")

        validation_report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'variable': variable,
            'validation_sections': {}
        }

        # 1. Drift Fit Analysis
        logger.info("Analyzing drift fit...")
        drift_analysis = self._analyze_drift_fit(uk_result, original_data, variable)
        validation_report['validation_sections']['drift_fit'] = drift_analysis

        # 2. Residual Analysis
        logger.info("Analyzing residuals...")
        residual_analysis = self._analyze_residuals(uk_result, original_data, variable)
        validation_report['validation_sections']['residual_analysis'] = residual_analysis

        # 3. Edge Instability Check
        logger.info("Checking edge instability...")
        edge_check = self._check_edge_instability(uk_result, original_data)
        validation_report['validation_sections']['edge_instability'] = edge_check

        # 4. UK vs OK Comparison (placeholder - would need OK results)
        validation_report['validation_sections']['uk_vs_ok_comparison'] = {
            'note': 'UK vs OK comparison requires separate OK run'
        }

        # Overall assessment
        validation_report['overall_assessment'] = self._compute_overall_assessment(validation_report)

        logger.info("UK validation complete")
        return validation_report

    def _analyze_drift_fit(self, uk_result: Dict[str, Any],
                          original_data: pd.DataFrame,
                          variable: str) -> Dict[str, Any]:
        """Analyze how well the drift fits the data."""
        if variable not in original_data.columns:
            return {'error': f'Variable {variable} not found'}

        # Get data coordinates and values
        data_coords = original_data[['X', 'Y', 'Z']].values
        data_values = original_data[variable].values

        # Get drift type from metadata
        drift_type = uk_result.get('metadata', {}).get('drift_type', 'linear')

        # Fit drift to original data
        drift_fit = self.drift_fitter.fit_drift(data_coords, data_values, drift_type)

        if 'error' in drift_fit:
            return {'error': drift_fit['error']}

        # Check for issues
        issues = []
        if drift_fit['r_squared'] < 0.05:
            issues.append("Drift explains <5% of variance - consider simpler model or OK")
        if abs(drift_fit['mean_residual']) > 0.1 * np.std(data_values):
            issues.append("Large mean residual - drift may be inappropriate")
        if drift_fit['variance_explained']['drift_percentage'] < 1.0:
            issues.append("Drift contributes negligible variance")

        return {
            'drift_type': drift_type,
            'r_squared': drift_fit['r_squared'],
            'rmse': drift_fit['rmse'],
            'mean_residual': drift_fit['mean_residual'],
            'variance_breakdown': drift_fit['variance_explained'],
            'issues': issues,
            'recommendations': self._get_drift_recommendations(drift_fit)
        }

    def _analyze_residuals(self, uk_result: Dict[str, Any],
                          original_data: pd.DataFrame,
                          variable: str) -> Dict[str, Any]:
        """Analyze residual distribution and variogram."""
        # Get original data
        data_coords = original_data[['X', 'Y', 'Z']].values
        data_values = original_data[variable].values

        # Get drift fit
        drift_type = uk_result.get('metadata', {}).get('drift_type', 'linear')
        drift_fit = self.drift_fitter.fit_drift(data_coords, data_values, drift_type)

        if 'error' in drift_fit:
            return {'error': 'Could not fit drift for residual analysis'}

        residuals = drift_fit['residuals']

        # Basic residual statistics
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'skewness': float(self._compute_skewness(residuals)),
            'kurtosis': float(self._compute_kurtosis(residuals))
        }

        # Check residual normality and independence
        issues = []
        if abs(residual_stats['mean']) > 0.05 * residual_stats['std']:
            issues.append("Residuals have significant mean bias")
        if abs(residual_stats['skewness']) > 1.0:
            issues.append("Residuals are significantly skewed")
        if residual_stats['kurtosis'] > 4.0 or residual_stats['kurtosis'] < 0.0:
            issues.append("Residual kurtosis suggests non-normal distribution")

        return {
            'statistics': residual_stats,
            'issues': issues,
            'distribution_check': self._check_residual_distribution(residuals)
        }

    def _check_edge_instability(self, uk_result: Dict[str, Any],
                              original_data: pd.DataFrame) -> Dict[str, Any]:
        """Check for instability at grid edges."""
        estimates = uk_result['estimates']
        data_std = original_data.select_dtypes(include=[np.number]).std().mean()

        # Find extreme values
        extreme_low = estimates < -3 * data_std
        extreme_high = estimates > 3 * data_std

        n_extreme_low = np.sum(extreme_low)
        n_extreme_high = np.sum(extreme_high)
        n_total = estimates.size

        pct_extreme = (n_extreme_low + n_extreme_high) / n_total * 100

        issues = []
        if pct_extreme > 5.0:
            issues.append(".1f")
        if n_extreme_low > n_extreme_high * 2:
            issues.append("Asymmetric extremes - primarily negative values")

        return {
            'n_extreme_low': int(n_extreme_low),
            'n_extreme_high': int(n_extreme_high),
            'percent_extreme': float(pct_extreme),
            'data_std': float(data_std),
            'issues': issues,
            'recommendations': ["Consider post-processing filters"] if issues else []
        }

    def _compute_overall_assessment(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall UK validation assessment."""
        sections = validation_report['validation_sections']

        # Scoring system
        score = 0
        max_score = 0

        # Drift fit quality (0-2 points)
        drift_section = sections.get('drift_fit', {})
        if drift_section.get('r_squared', 0) > 0.1:
            score += 2
        elif drift_section.get('r_squared', 0) > 0.05:
            score += 1
        max_score += 2

        # Residual quality (0-2 points)
        residual_section = sections.get('residual_analysis', {})
        residual_stats = residual_section.get('statistics', {})
        if abs(residual_stats.get('mean', 0)) < 0.01 * residual_stats.get('std', 1):
            score += 1  # Mean near zero
        if abs(residual_stats.get('skewness', 0)) < 0.5:
            score += 1  # Reasonable skewness
        max_score += 2

        # Edge stability (0-1 point)
        edge_section = sections.get('edge_instability', {})
        if edge_section.get('percent_extreme', 100) < 2.0:
            score += 1
        max_score += 1

        # Overall score
        overall_score = score / max_score if max_score > 0 else 0

        # Confidence level
        if overall_score > 0.8:
            confidence = 'high'
        elif overall_score > 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'

        # Collect all issues
        all_issues = []
        for section in sections.values():
            if isinstance(section, dict) and 'issues' in section:
                all_issues.extend(section['issues'])

        return {
            'overall_score': overall_score,
            'confidence_level': confidence,
            'total_issues': len(all_issues),
            'issues': all_issues,
            'recommendations': self._get_overall_recommendations(validation_report)
        }

    def _get_drift_recommendations(self, drift_fit: Dict[str, Any]) -> List[str]:
        """Get recommendations based on drift fit."""
        recommendations = []

        r2 = drift_fit.get('r_squared', 0)
        if r2 < 0.05:
            recommendations.append("Consider using Ordinary Kriging instead - drift explains minimal variance")
        elif r2 < 0.1:
            recommendations.append("Drift fit is weak - consider simpler drift model")

        return recommendations

    def _check_residual_distribution(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Check if residuals follow expected distribution."""
        # Simple normality checks
        from scipy import stats

        # Shapiro-Wilk test for normality (on subsample for large datasets)
        if len(residuals) > 5000:
            sample = np.random.choice(residuals, 5000, replace=False)
        else:
            sample = residuals

        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            normality_p = shapiro_p
        except (ValueError, RuntimeError) as e:
            logger.warning(f"Shapiro-Wilk test failed: {e}")
            normality_p = None

        return {
            'normality_test_p_value': normality_p,
            'is_normal': normality_p is not None and normality_p > 0.05
        }

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std)**3)

    def _compute_kurtosis(self, data: np.ndarray) -> float:
        """Compute kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std)**4) - 3

    def _get_overall_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Get overall recommendations."""
        recommendations = []
        assessment = validation_report.get('overall_assessment', {})

        if assessment.get('confidence_level') == 'low':
            recommendations.append("UK results may be unreliable - consider OK or review drift model")

        drift_section = validation_report['validation_sections'].get('drift_fit', {})
        if drift_section.get('r_squared', 0) < 0.05:
            recommendations.append("Switch to Ordinary Kriging - drift provides minimal benefit")

        edge_section = validation_report['validation_sections'].get('edge_instability', {})
        if edge_section.get('percent_extreme', 0) > 5:
            recommendations.append("Apply post-processing filters to handle edge instabilities")

        return recommendations


class UKPostProcessor:
    """
    Post-processing filters for UK results to eliminate unrealistic grades.
    """

    def __init__(self):
        pass

    def apply_percentile_clipping(self, estimates: np.ndarray,
                                lower_percentile: float = 1.0,
                                upper_percentile: float = 99.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply percentile-based clipping to remove extreme values.

        Args:
            estimates: UK estimate array
            lower_percentile: Lower percentile for clipping (e.g., 1.0)
            upper_percentile: Upper percentile for clipping (e.g., 99.0)

        Returns:
            Clipped estimates and statistics
        """
        # Compute percentiles
        p_low = np.percentile(estimates, lower_percentile)
        p_high = np.percentile(estimates, upper_percentile)

        # Apply clipping
        clipped = np.clip(estimates, p_low, p_high)

        # Statistics
        n_clipped_low = np.sum(estimates < p_low)
        n_clipped_high = np.sum(estimates > p_high)
        pct_clipped = (n_clipped_low + n_clipped_high) / len(estimates) * 100

        stats = {
            'lower_bound': float(p_low),
            'upper_bound': float(p_high),
            'n_clipped_low': int(n_clipped_low),
            'n_clipped_high': int(n_clipped_high),
            'percent_clipped': float(pct_clipped),
            'method': f'percentile_clipping_{lower_percentile}_{upper_percentile}'
        }

        return clipped, stats

    def apply_mean_reversion(self, estimates: np.ndarray,
                           drift_predictions: np.ndarray,
                           lambda_factor: float = 0.8) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply mean reversion to dampen extrapolation.

        Formula: Z* = m(x) + λ(Z* - m(x))

        Args:
            estimates: UK estimates
            drift_predictions: Drift predictions m(x)
            lambda_factor: Damping factor (0.6-0.9 recommended)

        Returns:
            Corrected estimates and statistics
        """
        correction = drift_predictions + lambda_factor * (estimates - drift_predictions)

        # Statistics
        original_range = np.ptp(estimates)  # peak-to-peak
        corrected_range = np.ptp(correction)

        stats = {
            'lambda_factor': lambda_factor,
            'original_range': float(original_range),
            'corrected_range': float(corrected_range),
            'damping_ratio': float(corrected_range / original_range) if original_range > 0 else 1.0,
            'method': f'mean_reversion_lambda_{lambda_factor}'
        }

        return correction, stats

    def apply_drift_contraction(self, estimates: np.ndarray,
                               drift_predictions: np.ndarray,
                               contraction_factor: float = 0.85) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply drift contraction to prevent UK extrapolation blow-up.

        Formula: Z* = m(x) + λ(Z* - m(x))

        This shrinks extreme values back toward the drift surface.

        Args:
            estimates: UK estimates
            drift_predictions: Drift predictions m(x) at estimation points
            contraction_factor: Contraction factor (0.7-0.9 recommended)

        Returns:
            Contracted estimates and statistics
        """
        if drift_predictions is None or len(drift_predictions) != len(estimates):
            # Fallback: use global mean as drift
            drift_predictions = np.full_like(estimates, np.mean(estimates))

        contracted = drift_predictions + contraction_factor * (estimates - drift_predictions)

        # Statistics
        original_range = np.ptp(estimates)
        contracted_range = np.ptp(contracted)
        n_contracted = np.sum(np.abs(contracted - estimates) > 0.1)  # Count significant changes

        stats = {
            'contraction_factor': contraction_factor,
            'original_range': float(original_range),
            'contracted_range': float(contracted_range),
            'range_reduction': float((original_range - contracted_range) / original_range) if original_range > 0 else 0,
            'n_contracted': int(n_contracted),
            'method': f'drift_contraction_lambda_{contraction_factor}'
        }

        return contracted, stats

    def apply_soft_truncation(self, estimates: np.ndarray,
                            data_values: np.ndarray,
                            lower_percentile: float = 1.0,
                            upper_percentile: float = 99.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply soft probabilistic truncation to extreme values.

        Instead of hard clipping, this applies a sigmoid-like transition
        that preserves the mean while preventing extreme outliers.

        Args:
            estimates: UK estimates
            data_values: Original data values for reference
            lower_percentile: Lower percentile for soft truncation
            upper_percentile: Upper percentile for soft truncation

        Returns:
            Soft-truncated estimates and statistics
        """
        # Compute reference bounds from data
        data_p01, data_p99 = np.percentile(data_values, [1, 99])
        data_range = data_p99 - data_p01

        # Define soft bounds (extend slightly beyond data range)
        soft_lower = data_p01 - 0.5 * data_range
        soft_upper = data_p99 + 0.5 * data_range

        # Apply sigmoid-like soft truncation
        def soft_clip(x, lower, upper):
            # For values within bounds, return unchanged
            within_bounds = (x >= lower) & (x <= upper)

            # For values below lower bound: smooth transition
            below_lower = x < lower
            if np.any(below_lower):
                # Exponential decay toward lower bound
                scale = (lower - soft_lower) / 3.0  # 3-sigma rule
                x_below = lower - (lower - x[below_lower]) * np.exp((x[below_lower] - lower) / scale)
                x = np.where(below_lower, x_below, x)

            # For values above upper bound: smooth transition
            above_upper = x > upper
            if np.any(above_upper):
                # Exponential decay toward upper bound
                scale = (soft_upper - upper) / 3.0
                x_above = upper + (x[above_upper] - upper) * np.exp((upper - x[above_upper]) / scale)
                x = np.where(above_upper, x_above, x)

            return x

        truncated = soft_clip(estimates, soft_lower, soft_upper)

        # Statistics
        n_modified = np.sum(np.abs(truncated - estimates) > 0.01)
        original_extremes = np.sum((estimates < data_p01) | (estimates > data_p99))
        final_extremes = np.sum((truncated < data_p01) | (truncated > data_p99))

        stats = {
            'soft_lower_bound': float(soft_lower),
            'soft_upper_bound': float(soft_upper),
            'data_p01': float(data_p01),
            'data_p99': float(data_p99),
            'n_modified': int(n_modified),
            'original_extremes': int(original_extremes),
            'final_extremes': int(final_extremes),
            'method': f'soft_truncation_p{lower_percentile}_p{upper_percentile}'
        }

        return truncated, stats

    def apply_combined_uk_filters(self, estimates: np.ndarray,
                                drift_predictions: np.ndarray = None,
                                data_values: np.ndarray = None,
                                config: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply combined UK stabilization filters in optimal order.

        Args:
            estimates: Raw UK estimates
            drift_predictions: Drift predictions at estimation points
            data_values: Original data values for reference
            config: Filter configuration

        Returns:
            Filtered estimates and combined statistics
        """
        if config is None:
            config = {}

        filtered = estimates.copy()
        applied_filters = []
        combined_stats = {}

        # 1. Apply drift contraction first (most important for stability)
        if config.get('drift_contraction', True) and drift_predictions is not None:
            filtered, stats = self.apply_drift_contraction(
                filtered, drift_predictions,
                contraction_factor=config.get('contraction_factor', 0.85)
            )
            applied_filters.append('drift_contraction')
            combined_stats['drift_contraction'] = stats

        # 2. Apply soft truncation for remaining extremes
        if config.get('soft_truncation', True) and data_values is not None:
            filtered, stats = self.apply_soft_truncation(
                filtered, data_values,
                lower_percentile=config.get('lower_percentile', 0.5),
                upper_percentile=config.get('upper_percentile', 99.5)
            )
            applied_filters.append('soft_truncation')
            combined_stats['soft_truncation'] = stats

        # 3. Apply positivity constraint if needed
        if config.get('positivity_constraint', False):
            filtered, stats = self.apply_positivity_constraint(
                filtered, replacement_value=config.get('min_value', 0.0)
            )
            applied_filters.append('positivity_constraint')
            combined_stats['positivity_constraint'] = stats

        combined_stats['applied_filters'] = applied_filters
        combined_stats['filter_count'] = len(applied_filters)

        return filtered, combined_stats

    def apply_positivity_constraint(self, estimates: np.ndarray,
                                  replacement_value: float = 0.0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply positivity constraint by clamping negative values.

        Args:
            estimates: UK estimates
            replacement_value: Value to use for negative estimates

        Returns:
            Constrained estimates and statistics
        """
        constrained = np.maximum(estimates, replacement_value)

        n_negative = np.sum(estimates < 0)
        pct_negative = n_negative / len(estimates) * 100

        stats = {
            'replacement_value': replacement_value,
            'n_negative_original': int(n_negative),
            'percent_negative': float(pct_negative),
            'method': 'positivity_constraint'
        }

        return constrained, stats

    def apply_combined_filters(self, estimates: np.ndarray,
                             drift_predictions: Optional[np.ndarray] = None,
                             config: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply multiple filters in sequence.

        Args:
            estimates: UK estimates
            drift_predictions: Drift predictions (for mean reversion)
            config: Filter configuration

        Returns:
            Filtered estimates and combined statistics
        """
        if config is None:
            config = {
                'percentile_clip': True,
                'lower_percentile': 1.0,
                'upper_percentile': 99.0,
                'mean_reversion': True,
                'lambda_factor': 0.8,
                'positivity_constraint': True,
                'replacement_value': 0.0
            }

        filtered = estimates.copy()
        applied_filters = []

        # 1. Percentile clipping
        if config.get('percentile_clip', False):
            filtered, clip_stats = self.apply_percentile_clipping(
                filtered,
                config.get('lower_percentile', 1.0),
                config.get('upper_percentile', 99.0)
            )
            applied_filters.append(('percentile_clipping', clip_stats))

        # 2. Mean reversion (if drift available)
        if config.get('mean_reversion', False) and drift_predictions is not None:
            filtered, mr_stats = self.apply_mean_reversion(
                filtered, drift_predictions, config.get('lambda_factor', 0.8)
            )
            applied_filters.append(('mean_reversion', mr_stats))

        # 3. Positivity constraint
        if config.get('positivity_constraint', False):
            filtered, pos_stats = self.apply_positivity_constraint(
                filtered, config.get('replacement_value', 0.0)
            )
            applied_filters.append(('positivity_constraint', pos_stats))

        return filtered, {
            'applied_filters': applied_filters,
            'final_range': (float(np.min(filtered)), float(np.max(filtered))),
            'final_mean': float(np.mean(filtered)),
            'final_std': float(np.std(filtered))
        }
