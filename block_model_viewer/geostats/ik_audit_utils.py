"""
Indicator Kriging Audit & Validation Utilities

Comprehensive validation toolkit for Indicator Kriging as per GSLIB/GeoStats best practices.

Includes:
- Validation workflow (5-step process)
- Benchmarking against OK/SK
- Diagnostic exports and QC checks
- CDF consistency verification
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class IndicatorKrigingAuditor:
    """
    Comprehensive audit toolkit for Indicator Kriging validation.

    Implements the 5-step validation workflow and benchmarking capabilities.
    """

    def __init__(self):
        self.validation_results = {}

    # =========================================================================
    # STEP 1: THRESHOLD & VARIOGRAM VALIDATION
    # =========================================================================

    def validate_thresholds_and_variogram(self, ik_result: Dict[str, Any],
                                        original_data: pd.DataFrame,
                                        variable: str) -> Dict[str, Any]:
        """
        Step 1: Validate thresholds and variogram setup.

        Args:
            ik_result: Result dict from run_indicator_kriging_job()
            original_data: Original drillhole data (composites/assays)
            variable: Variable name

        Returns:
            Validation dict with warnings and recommendations
        """
        thresholds = ik_result['thresholds']
        validation = {
            'threshold_warnings': [],
            'variogram_check': {},
            'recommendations': []
        }

        # Check threshold proportions
        if variable in original_data.columns:
            values = original_data[variable].dropna().values

            for i, thresh in enumerate(thresholds):
                proportion = np.mean(values <= thresh)
                if proportion < 0.02 or proportion > 0.98:
                    validation['threshold_warnings'].append({
                        'threshold': thresh,
                        'proportion': proportion,
                        'warning': f"Unstable threshold: {proportion:.1%} of samples ≤ {thresh}"
                    })

        # Variogram consistency check (placeholder for now)
        validation['variogram_check'] = {
            'model_type': ik_result['metadata'].get('variogram_model', 'unknown'),
            'range': ik_result['metadata'].get('range', 'unknown'),
            'nugget': ik_result['metadata'].get('nugget', 'unknown')
        }

        if validation['threshold_warnings']:
            validation['recommendations'].append(
                "Consider adjusting thresholds to avoid extreme proportions (<2% or >98%)"
            )

        return validation

    # =========================================================================
    # STEP 2: DATA-SPACE CROSS-CHECKS
    # =========================================================================

    def compute_data_space_cross_checks(self, ik_result: Dict[str, Any],
                                      original_data: pd.DataFrame,
                                      variable: str) -> Dict[str, Any]:
        """
        Step 2: Compare IK volume-weighted means vs global sample proportions.

        Args:
            ik_result: IK result dict
            original_data: Original data with X,Y,Z,Variable columns
            variable: Variable name

        Returns:
            Cross-check results
        """
        thresholds = ik_result['thresholds']
        probabilities = ik_result['probabilities']  # (nx, ny, nz, n_thresh)

        # Get grid definition
        grid_def = ik_result['grid_def']
        nx, ny, nz = grid_def['counts']
        dx, dy, dz = grid_def['spacing']

        # Compute global proportions from original data
        if variable in original_data.columns:
            values = original_data[variable].dropna().values
            global_props = [np.mean(values <= thresh) for thresh in thresholds]
        else:
            global_props = [np.nan] * len(thresholds)

        # Compute volume-weighted IK means
        block_volumes = dx * dy * dz
        total_volume = nx * ny * nz * block_volumes

        ik_means = []
        differences = []

        for i, thresh in enumerate(thresholds):
            # Volume-weighted average probability
            prob_grid = probabilities[:, :, :, i]
            weighted_mean = np.sum(prob_grid) * block_volumes / total_volume
            ik_means.append(weighted_mean)

            # Difference from global
            diff = weighted_mean - global_props[i] if not np.isnan(global_props[i]) else np.nan
            differences.append(diff)

        return {
            'thresholds': thresholds,
            'global_proportions': global_props,
            'ik_volume_weighted_means': ik_means,
            'differences': differences,
            'max_absolute_difference': np.nanmax(np.abs(differences)) if differences else np.nan,
            'acceptable_range': (-0.03, 0.03),  # ±3% acceptable
            'is_acceptable': all(abs(d) <= 0.03 for d in differences if not np.isnan(d))
        }

    # =========================================================================
    # STEP 3: NODE-BY-NODE BACK TO SAMPLES CHECK
    # =========================================================================

    def compute_node_sample_validation(self, ik_result: Dict[str, Any],
                                     original_data: pd.DataFrame,
                                     variable: str,
                                     max_distance: float = 50.0) -> Dict[str, Any]:
        """
        Step 3: For each composite, find nearest IK block and compare empirical vs kriged.

        Args:
            ik_result: IK result dict
            original_data: Data with X,Y,Z coordinates
            variable: Variable name
            max_distance: Max distance to consider (meters)

        Returns:
            Node validation results
        """
        if variable not in original_data.columns:
            return {'error': f'Variable {variable} not found in data'}

        # Filter data to valid samples
        valid_data = original_data.dropna(subset=['X', 'Y', 'Z', variable])
        if len(valid_data) == 0:
            return {'error': 'No valid data points found'}

        data_coords = valid_data[['X', 'Y', 'Z']].values
        data_values = valid_data[variable].values

        # Get IK grid
        grid_x, grid_y, grid_z = ik_result['grid_x'], ik_result['grid_y'], ik_result['grid_z']
        nx, ny, nz = grid_x.shape

        # Create flattened grid coordinates
        target_coords = []
        indices = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    target_coords.append([grid_x[i,j,k], grid_y[i,j,k], grid_z[i,j,k]])
                    indices.append((i,j,k))

        target_coords = np.array(target_coords)

        # Build KDTree for nearest neighbor lookup
        tree = cKDTree(target_coords)

        # Find nearest blocks for each data point
        distances, nearest_indices = tree.query(data_coords, k=1, distance_upper_bound=max_distance)

        # Filter valid matches (within distance)
        valid_mask = distances < max_distance
        valid_indices = nearest_indices[valid_mask]
        valid_values = data_values[valid_mask]

        if len(valid_indices) == 0:
            return {'error': f'No data points within {max_distance}m of grid'}

        # Extract IK probabilities for these blocks
        thresholds = ik_result['thresholds']
        probabilities = ik_result['probabilities']

        results = []
        for data_val, block_idx in zip(valid_values, valid_indices):
            i, j, k = indices[block_idx]
            block_probs = probabilities[i, j, k, :]  # All thresholds for this block

            # Empirical indicators
            empirical = (data_val <= thresholds).astype(float)

            results.append({
                'data_value': data_val,
                'empirical_indicators': empirical,
                'kriged_probabilities': block_probs,
                'block_index': (i, j, k)
            })

        # Summary statistics
        empirical_array = np.array([r['empirical_indicators'] for r in results])
        kriged_array = np.array([r['kriged_probabilities'] for r in results])

        # Compute correlations per threshold
        correlations = []
        for t in range(len(thresholds)):
            emp = empirical_array[:, t]
            krig = kriged_array[:, t]
            if np.std(emp) > 0 and np.std(krig) > 0:
                corr = np.corrcoef(emp, krig)[0, 1]
            else:
                corr = np.nan
            correlations.append(corr)

        return {
            'n_valid_points': len(results),
            'thresholds': thresholds,
            'correlations': correlations,
            'mean_correlation': np.nanmean(correlations),
            'individual_results': results[:1000],  # Limit for memory
            'assessment': 'good' if np.nanmean(correlations) > 0.7 else 'needs_review'
        }

    # =========================================================================
    # STEP 4: COMPARE IK MEDIAN/E-TYPE VS OK/SK
    # =========================================================================

    def compare_against_kriging(self, ik_result: Dict[str, Any],
                               ok_result: Optional[Dict[str, Any]] = None,
                               sk_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Step 4: Compare IK estimates vs OK/SK at same grid locations.

        Args:
            ik_result: IK result dict
            ok_result: Ordinary Kriging result dict (optional)
            sk_result: Simple Kriging result dict (optional)

        Returns:
            Comparison metrics
        """
        comparisons = {}

        if ok_result is not None:
            comparisons['OK'] = self._compare_single_method(ik_result, ok_result)

        if sk_result is not None:
            comparisons['SK'] = self._compare_single_method(ik_result, sk_result)

        return comparisons

    def _compare_single_method(self, ik_result: Dict[str, Any],
                              other_result: Dict[str, Any]) -> Dict[str, Any]:
        """Compare IK vs another kriging method."""
        # Extract estimates
        ik_median = ik_result.get('median')
        ik_mean = ik_result.get('mean')

        # Assume other method has 'estimates' or similar
        other_estimates = other_result.get('estimates') or other_result.get('mean')

        if ik_mean is None or other_estimates is None:
            return {'error': 'Missing estimates for comparison'}

        # Compute differences
        mean_diff = ik_mean - other_estimates
        median_diff = ik_median - other_estimates if ik_median is not None else None

        # Statistics
        stats = {
            'n_blocks': len(ik_mean.flatten()),
            'ik_mean_range': (float(np.nanmin(ik_mean)), float(np.nanmax(ik_mean))),
            'other_range': (float(np.nanmin(other_estimates)), float(np.nanmax(other_estimates))),
            'mean_difference': {
                'mean': float(np.nanmean(mean_diff)),
                'std': float(np.nanstd(mean_diff)),
                'rmse': float(np.sqrt(np.nanmean(mean_diff**2))),
                'mae': float(np.nanmean(np.abs(mean_diff)))
            }
        }

        if median_diff is not None:
            stats['median_difference'] = {
                'mean': float(np.nanmean(median_diff)),
                'std': float(np.nanstd(median_diff)),
                'rmse': float(np.sqrt(np.nanmean(median_diff**2))),
                'mae': float(np.nanmean(np.abs(median_diff)))
            }

        return stats

    # =========================================================================
    # STEP 5: SWATH PLOTS & TREND ANALYSIS
    # =========================================================================

    def compute_swath_analysis(self, ik_result: Dict[str, Any],
                             direction: str = 'X',
                             window_size: int = 10) -> Dict[str, Any]:
        """
        Step 5: Compute swath plots along specified direction.

        Args:
            ik_result: IK result dict
            direction: Direction for swath ('X', 'Y', 'Z')
            window_size: Moving window size

        Returns:
            Swath analysis results
        """
        grid_def = ik_result['grid_def']
        nx, ny, nz = grid_def['counts']

        # Get coordinate arrays
        grid_x, grid_y, grid_z = ik_result['grid_x'], ik_result['grid_y'], ik_result['grid_z']

        # Extract properties
        median = ik_result.get('median')
        mean = ik_result.get('mean')

        if direction == 'X':
            coord_array = grid_x[:, 0, 0]  # X coordinates along first row
            median_swath = np.nanmean(median, axis=(1, 2)) if median is not None else None
            mean_swath = np.nanmean(mean, axis=(1, 2)) if mean is not None else None
        elif direction == 'Y':
            coord_array = grid_y[0, :, 0]  # Y coordinates along first column
            median_swath = np.nanmean(median, axis=(0, 2)) if median is not None else None
            mean_swath = np.nanmean(mean, axis=(0, 2)) if mean is not None else None
        elif direction == 'Z':
            coord_array = grid_z[0, 0, :]  # Z coordinates along first level
            median_swath = np.nanmean(median, axis=(0, 1)) if median is not None else None
            mean_swath = np.nanmean(mean, axis=(0, 1)) if mean is not None else None

        # Moving averages
        def moving_average(data, window):
            if data is None or len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')

        return {
            'direction': direction,
            'coordinates': coord_array,
            'median_swath': median_swath,
            'mean_swath': mean_swath,
            'median_smoothed': moving_average(median_swath, window_size),
            'mean_smoothed': moving_average(mean_swath, window_size),
            'window_size': window_size
        }

    # =========================================================================
    # DIAGNOSTIC EXPORTS
    # =========================================================================

    def compute_probability_slope_map(self, ik_result: Dict[str, Any],
                                    band_center: float = None,
                                    band_width: float = 0.1) -> Dict[str, Any]:
        """
        Compute probability slope map for classification transition strength.

        High slope indicates sharp transitions between classes.

        Args:
            ik_result: IK result dict
            band_center: Center of grade band (e.g., ore cutoff)
            band_width: Width of band to analyze

        Returns:
            Slope map and statistics
        """
        thresholds = ik_result['thresholds']
        probabilities = ik_result['probabilities']

        if band_center is None:
            # Use middle threshold as default
            band_center = np.median(thresholds)

        # Find thresholds in the band
        band_mask = (thresholds >= band_center - band_width/2) & (thresholds <= band_center + band_width/2)
        band_thresholds = thresholds[band_mask]
        band_probs = probabilities[:, :, :, band_mask]

        if len(band_thresholds) < 2:
            return {'error': 'Not enough thresholds in specified band'}

        # Compute slope between consecutive thresholds
        slopes = []
        for i in range(len(band_thresholds) - 1):
            delta_prob = band_probs[:, :, :, i+1] - band_probs[:, :, :, i]
            delta_thresh = band_thresholds[i+1] - band_thresholds[i]
            slope = delta_prob / delta_thresh
            slopes.append(slope)

        # Average slope in band (transition strength)
        avg_slope = np.mean(slopes, axis=0)

        return {
            'slope_map': avg_slope,
            'band_center': band_center,
            'band_width': band_width,
            'n_thresholds_in_band': len(band_thresholds),
            'slope_range': (float(np.nanmin(avg_slope)), float(np.nanmax(avg_slope))),
            'high_transition_zones': np.sum(np.abs(avg_slope) > np.nanpercentile(np.abs(avg_slope), 75)),
            'grid_x': ik_result['grid_x'],
            'grid_y': ik_result['grid_y'],
            'grid_z': ik_result['grid_z'],
            'grid_def': ik_result['grid_def']
        }

    def compute_global_cdf_comparison(self, ik_result: Dict[str, Any],
                                    original_data: pd.DataFrame,
                                    variable: str) -> Dict[str, Any]:
        """
        Compare global empirical CDF vs IK-implied CDF.

        Args:
            ik_result: IK result dict
            original_data: Original data
            variable: Variable name

        Returns:
            CDF comparison
        """
        if variable not in original_data.columns:
            return {'error': f'Variable {variable} not found'}

        thresholds = ik_result['thresholds']
        probabilities = ik_result['probabilities']

        # Empirical CDF from data
        values = original_data[variable].dropna().values
        empirical_cdf = [np.mean(values <= t) for t in thresholds]

        # IK-implied CDF (volume-weighted average probabilities)
        grid_def = ik_result['grid_def']
        dx, dy, dz = grid_def['spacing']
        nx, ny, nz = grid_def['counts']
        block_volume = dx * dy * dz
        total_volume = nx * ny * nz * block_volume

        ik_cdf = []
        for i, t in enumerate(thresholds):
            prob_grid = probabilities[:, :, :, i]
            weighted_avg = np.sum(prob_grid) * block_volume / total_volume
            ik_cdf.append(weighted_avg)

        # Differences
        differences = np.array(ik_cdf) - np.array(empirical_cdf)
        max_diff = float(np.max(np.abs(differences)))

        return {
            'thresholds': thresholds,
            'empirical_cdf': empirical_cdf,
            'ik_cdf': ik_cdf,
            'differences': differences.tolist(),
            'max_absolute_difference': max_diff,
            'is_consistent': max_diff <= 0.05  # 5% tolerance
        }

    # =========================================================================
    # MAIN VALIDATION WORKFLOW
    # =========================================================================

    def run_full_validation_workflow(self, ik_result: Dict[str, Any],
                                   original_data: pd.DataFrame,
                                   variable: str,
                                   ok_result: Optional[Dict[str, Any]] = None,
                                   sk_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete 5-step validation workflow.

        Args:
            ik_result: Indicator Kriging result
            original_data: Original drillhole data
            variable: Variable name
            ok_result: Optional OK result for comparison
            sk_result: Optional SK result for comparison

        Returns:
            Complete validation report
        """
        logger.info("Starting IK validation workflow...")

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'variable': variable,
            'validation_steps': {}
        }

        # Step 1: Threshold and variogram validation
        logger.info("Step 1: Validating thresholds and variogram...")
        step1 = self.validate_thresholds_and_variogram(ik_result, original_data, variable)
        report['validation_steps']['1_threshold_variogram'] = step1

        # Step 2: Data-space cross-checks
        logger.info("Step 2: Computing data-space cross-checks...")
        step2 = self.compute_data_space_cross_checks(ik_result, original_data, variable)
        report['validation_steps']['2_data_space_checks'] = step2

        # Step 3: Node-by-node validation
        logger.info("Step 3: Node-by-node validation...")
        step3 = self.compute_node_sample_validation(ik_result, original_data, variable)
        report['validation_steps']['3_node_validation'] = step3

        # Step 4: Comparison with OK/SK
        if ok_result or sk_result:
            logger.info("Step 4: Comparing with OK/SK...")
            step4 = self.compare_against_kriging(ik_result, ok_result, sk_result)
            report['validation_steps']['4_kriging_comparison'] = step4

        # Step 5: Swath analysis
        logger.info("Step 5: Swath plot analysis...")
        step5_x = self.compute_swath_analysis(ik_result, 'X')
        step5_y = self.compute_swath_analysis(ik_result, 'Y')
        report['validation_steps']['5_swath_analysis'] = {
            'X_direction': step5_x,
            'Y_direction': step5_y
        }

        # Additional diagnostics
        logger.info("Computing additional diagnostics...")
        slope_map = self.compute_probability_slope_map(ik_result)
        cdf_comparison = self.compute_global_cdf_comparison(ik_result, original_data, variable)

        report['diagnostics'] = {
            'probability_slope_map': slope_map,
            'cdf_comparison': cdf_comparison
        }

        # Overall assessment
        report['overall_assessment'] = self._compute_overall_assessment(report)

        logger.info("IK validation workflow complete")
        return report

    def _compute_overall_assessment(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Compute overall validation assessment."""
        assessment = {
            'overall_score': 0,
            'issues': [],
            'recommendations': [],
            'confidence_level': 'unknown'
        }

        score = 0
        max_score = 0

        # Step 2: Data-space checks
        step2 = report['validation_steps'].get('2_data_space_checks', {})
        if step2.get('is_acceptable', False):
            score += 1
            max_score += 1
        else:
            assessment['issues'].append("IK CDF doesn't match global proportions")
            assessment['recommendations'].append("Check variogram model or grid extent")

        # Step 3: Node validation
        step3 = report['validation_steps'].get('3_node_validation', {})
        corr = step3.get('mean_correlation', 0)
        if corr > 0.7:
            score += 1
        elif corr > 0.5:
            score += 0.5
        max_score += 1

        if corr < 0.6:
            assessment['issues'].append("Poor correlation between empirical and kriged indicators")
            assessment['recommendations'].append("Review indicator variogram parameters")

        # CDF comparison
        cdf_comp = report['diagnostics'].get('cdf_comparison', {})
        if cdf_comp.get('is_consistent', False):
            score += 1
            max_score += 1
        else:
            assessment['issues'].append("IK-implied CDF differs from empirical CDF")
            assessment['recommendations'].append("Consider conditionalizing to global CDF")

        # Overall score
        assessment['overall_score'] = score / max_score if max_score > 0 else 0

        if assessment['overall_score'] > 0.8:
            assessment['confidence_level'] = 'high'
        elif assessment['overall_score'] > 0.6:
            assessment['confidence_level'] = 'medium'
        else:
            assessment['confidence_level'] = 'low'

        return assessment


class IndicatorKrigingBenchmark:
    """
    Benchmark Indicator Kriging against Ordinary/Simple Kriging.

    Provides accuracy metrics, classification performance, and variance analysis.
    """

    def __init__(self):
        self.benchmark_results = {}

    def run_accuracy_benchmark(self, ik_result: Dict[str, Any],
                              ok_result: Optional[Dict[str, Any]] = None,
                              sk_result: Optional[Dict[str, Any]] = None,
                              validation_data: Optional[pd.DataFrame] = None,
                              variable: str = 'grade',
                              holdout_fraction: float = 0.2) -> Dict[str, Any]:
        """
        Run accuracy benchmark using holdout validation.

        Args:
            ik_result: IK result dict
            ok_result: OK result dict
            sk_result: SK result dict
            validation_data: Holdout data for validation
            variable: Variable name
            holdout_fraction: Fraction of data to hold out

        Returns:
            Accuracy benchmark results
        """
        if validation_data is None or len(validation_data) == 0:
            return {'error': 'No validation data provided'}

        # Prepare holdout data
        valid_data = validation_data.dropna(subset=['X', 'Y', 'Z', variable])
        if len(valid_data) < 10:
            return {'error': 'Insufficient validation data'}

        # Sample holdout points
        n_holdout = max(10, int(len(valid_data) * holdout_fraction))
        holdout_indices = np.random.choice(len(valid_data), n_holdout, replace=False)
        holdout_data = valid_data.iloc[holdout_indices]

        results = {
            'n_holdout_points': n_holdout,
            'methods': {}
        }

        # Evaluate each method
        if ok_result:
            results['methods']['OK'] = self._evaluate_method_accuracy(
                ok_result, holdout_data, variable
            )

        if sk_result:
            results['methods']['SK'] = self._evaluate_method_accuracy(
                sk_result, holdout_data, variable
            )

        # IK evaluation (interpolate at holdout locations)
        results['methods']['IK_Etype'] = self._evaluate_ik_accuracy(
            ik_result, holdout_data, variable, use_median=False
        )

        results['methods']['IK_Median'] = self._evaluate_ik_accuracy(
            ik_result, holdout_data, variable, use_median=True
        )

        return results

    def _evaluate_method_accuracy(self, method_result: Dict[str, Any],
                                holdout_data: pd.DataFrame,
                                variable: str) -> Dict[str, Any]:
        """Evaluate accuracy of a kriging method."""
        # Interpolate at holdout locations (simplified - assume gridded results)
        # In practice, would need proper interpolation
        estimates = method_result.get('estimates') or method_result.get('mean')
        if estimates is None:
            return {'error': 'No estimates found'}

        # For now, return placeholder (would need spatial interpolation)
        return {
            'mae': None,  # Mean Absolute Error
            'rmse': None, # Root Mean Square Error
            'correlation': None,
            'note': 'Spatial interpolation needed for accurate evaluation'
        }

    def _evaluate_ik_accuracy(self, ik_result: Dict[str, Any],
                            holdout_data: pd.DataFrame,
                            variable: str,
                            use_median: bool = False) -> Dict[str, Any]:
        """Evaluate IK accuracy at holdout locations."""
        # Get IK grid
        grid_x, grid_y, grid_z = ik_result['grid_x'], ik_result['grid_y'], ik_result['grid_z']
        nx, ny, nz = grid_x.shape

        # Create coordinate arrays
        coords = []
        indices = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    coords.append([grid_x[i,j,k], grid_y[i,j,k], grid_z[i,j,k]])
                    indices.append((i,j,k))
        coords = np.array(coords)

        # Build KDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)

        # Find nearest blocks for holdout points
        holdout_coords = holdout_data[['X', 'Y', 'Z']].values
        true_values = holdout_data[variable].values

        distances, nearest_indices = tree.query(holdout_coords, k=1, distance_upper_bound=100.0)
        valid_mask = distances < 100.0

        if not np.any(valid_mask):
            return {'error': 'No holdout points near grid'}

        valid_indices = nearest_indices[valid_mask]
        valid_true = true_values[valid_mask]

        # Get IK estimates
        if use_median:
            ik_estimates = ik_result.get('median')
            if ik_estimates is None:
                return {'error': 'No median estimates available'}
            estimates = np.array([ik_estimates[idx] for idx in [indices[i] for i in valid_indices]])
        else:
            ik_estimates = ik_result.get('mean')
            if ik_estimates is None:
                return {'error': 'No mean estimates available'}
            estimates = np.array([ik_estimates[idx] for idx in [indices[i] for i in valid_indices]])

        # Remove NaNs
        valid_mask2 = ~np.isnan(estimates)
        final_true = valid_true[valid_mask2]
        final_estimates = estimates[valid_mask2]

        if len(final_true) < 5:
            return {'error': 'Insufficient valid estimates'}

        # Compute metrics
        errors = final_estimates - final_true
        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors**2)))

        correlation = float(np.corrcoef(final_true, final_estimates)[0, 1]) if len(final_true) > 1 else np.nan

        return {
            'mae': mae,
            'rmse': rmse,
            'correlation': correlation,
            'n_points': len(final_true),
            'estimate_type': 'median' if use_median else 'e-type'
        }

    def run_classification_benchmark(self, ik_result: Dict[str, Any],
                                   validation_data: pd.DataFrame,
                                   variable: str,
                                   cutoff: float,
                                   holdout_fraction: float = 0.2) -> Dict[str, Any]:
        """
        Benchmark classification performance against a cutoff.

        Args:
            ik_result: IK result dict
            validation_data: Validation data
            variable: Variable name
            cutoff: Classification cutoff
            holdout_fraction: Holdout fraction

        Returns:
            Classification benchmark results
        """
        if variable not in validation_data.columns:
            return {'error': f'Variable {variable} not found'}

        # Get holdout data
        valid_data = validation_data.dropna(subset=['X', 'Y', 'Z', variable])
        n_holdout = max(10, int(len(valid_data) * holdout_fraction))
        holdout_indices = np.random.choice(len(valid_data), n_holdout, replace=False)
        holdout_data = valid_data.iloc[holdout_indices]

        results = {
            'cutoff': cutoff,
            'n_holdout_points': n_holdout,
            'methods': {}
        }

        # True classifications
        true_classes = (holdout_data[variable].values >= cutoff).astype(int)

        # IK classification at different probability thresholds
        ik_probs = self._get_ik_probabilities_at_points(ik_result, holdout_data)

        if ik_probs is not None:
            # Find closest threshold to cutoff
            thresholds = ik_result['thresholds']
            cutoff_idx = np.argmin(np.abs(thresholds - cutoff))

            probs_above = 1.0 - ik_probs[:, cutoff_idx]  # P(Z > cutoff)

            # ROC-style analysis at different probability cutoffs
            prob_thresholds = np.linspace(0.1, 0.9, 9)

            ik_performance = []
            for prob_thresh in prob_thresholds:
                pred_classes = (probs_above >= prob_thresh).astype(int)

                # Confusion matrix
                tp = np.sum((pred_classes == 1) & (true_classes == 1))
                tn = np.sum((pred_classes == 0) & (true_classes == 0))
                fp = np.sum((pred_classes == 1) & (true_classes == 0))
                fn = np.sum((pred_classes == 0) & (true_classes == 1))

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                accuracy = (tp + tn) / len(true_classes)

                ik_performance.append({
                    'probability_threshold': prob_thresh,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
                })

            results['methods']['IK'] = {
                'performance_curve': ik_performance,
                'auc_approximation': np.mean([p['accuracy'] for p in ik_performance])
            }

        return results

    def _get_ik_probabilities_at_points(self, ik_result: Dict[str, Any],
                                      points: pd.DataFrame) -> Optional[np.ndarray]:
        """Get IK probabilities at arbitrary points (nearest neighbor)."""
        # Get IK grid
        grid_x, grid_y, grid_z = ik_result['grid_x'], ik_result['grid_y'], ik_result['grid_z']
        nx, ny, nz = grid_x.shape

        # Create coordinate arrays
        coords = []
        indices = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    coords.append([grid_x[i,j,k], grid_y[i,j,k], grid_z[i,j,k]])
                    indices.append((i,j,k))
        coords = np.array(coords)

        # Build KDTree
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)

        # Find nearest blocks
        point_coords = points[['X', 'Y', 'Z']].values
        distances, nearest_indices = tree.query(point_coords, k=1, distance_upper_bound=100.0)
        valid_mask = distances < 100.0

        if not np.any(valid_mask):
            return None

        # Extract probabilities
        probabilities = ik_result['probabilities']
        n_thresh = len(ik_result['thresholds'])

        result_probs = np.full((len(point_coords), n_thresh), np.nan)
        for i, (valid, idx) in enumerate(zip(valid_mask, nearest_indices)):
            if valid:
                block_i, block_j, block_k = indices[idx]
                result_probs[i, :] = probabilities[block_i, block_j, block_k, :]

        return result_probs

    def analyze_variance_behavior(self, ik_result: Dict[str, Any],
                                ok_result: Optional[Dict[str, Any]] = None,
                                sk_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze variance behavior and transition sharpness.

        Args:
            ik_result: IK result dict
            ok_result: OK result for comparison
            sk_result: SK result for comparison

        Returns:
            Variance analysis results
        """
        results = {
            'ik_slope_analysis': self._analyze_ik_transitions(ik_result)
        }

        if ok_result:
            results['ok_variance'] = self._extract_variance(ok_result)

        if sk_result:
            results['sk_variance'] = self._extract_variance(sk_result)

        return results

    def _analyze_ik_transitions(self, ik_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze IK transition sharpness."""
        probabilities = ik_result['probabilities']
        thresholds = ik_result['thresholds']

        # Compute probability slopes between consecutive thresholds
        slopes = []
        for i in range(len(thresholds) - 1):
            delta_prob = probabilities[:, :, :, i+1] - probabilities[:, :, :, i]
            delta_thresh = thresholds[i+1] - thresholds[i]
            slope = delta_prob / delta_thresh
            slopes.append(slope)

        avg_slope = np.mean(np.abs(slopes), axis=0)

        return {
            'mean_slope': float(np.nanmean(avg_slope)),
            'slope_std': float(np.nanstd(avg_slope)),
            'sharp_transition_blocks': int(np.sum(avg_slope > np.nanpercentile(avg_slope, 75))),
            'slope_range': (float(np.nanmin(avg_slope)), float(np.nanmax(avg_slope)))
        }

    def _extract_variance(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract variance information from kriging result."""
        variance = result.get('variance') or result.get('kriging_variance')
        if variance is None:
            return {'error': 'No variance information available'}

        return {
            'mean_variance': float(np.nanmean(variance)),
            'variance_range': (float(np.nanmin(variance)), float(np.nanmax(variance))),
            'variance_std': float(np.nanstd(variance))
        }
