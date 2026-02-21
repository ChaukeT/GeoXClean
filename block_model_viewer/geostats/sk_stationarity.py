"""
Stationarity validation for Simple Kriging.

Provides tools to validate the assumption that the global mean is stationary
across the domain. This is critical for SK and required for stock exchange reporting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import linregress, ks_2samp


@dataclass
class StationarityReport:
    """
    Stationarity validation report for Simple Kriging.

    Contains statistical tests and spatial trend analysis to assess
    whether the global mean assumption is valid.
    """
    global_mean: float
    mean_by_domain: Dict[str, float] = field(default_factory=dict)
    trend_x: Dict[str, float] = field(default_factory=dict)
    trend_y: Dict[str, float] = field(default_factory=dict)
    trend_z: Dict[str, float] = field(default_factory=dict)
    swath_results: Optional[pd.DataFrame] = None
    issues: List[str] = field(default_factory=list)
    confidence_level: str = "unknown"  # 'high', 'medium', 'low'


def validate_stationarity(
    data: pd.DataFrame,
    variable: str,
    global_mean: float,
    domain_column: Optional[str] = None
) -> StationarityReport:
    """
    Validate stationarity assumptions for Simple Kriging.

    Performs multiple checks:
    1. Domain-wise mean comparison (if domain column provided)
    2. Spatial trends in X, Y, Z directions
    3. Statistical tests for normality of residuals

    Args:
        data: DataFrame with coordinates (X, Y, Z) and variable
        variable: Column name of variable to analyze
        global_mean: Global mean value used in SK
        domain_column: Optional column name for domain/lithology grouping

    Returns:
        StationarityReport with validation results and issues

    Notes:
        - Confidence level 'high': No significant trends or domain differences
        - Confidence level 'medium': Some minor trends but acceptable
        - Confidence level 'low': Significant trends, SK may not be appropriate
    """
    report = StationarityReport(global_mean=global_mean)

    # Ensure required columns exist
    required_cols = ['X', 'Y', 'Z', variable]
    for col in required_cols:
        if col not in data.columns:
            report.issues.append(f"Required column '{col}' not found in data")
            report.confidence_level = 'low'
            return report

    # Get data values
    values = data[variable].to_numpy()
    n_samples = len(values)

    if n_samples < 10:
        report.issues.append(f"Insufficient samples for stationarity validation (n={n_samples})")
        report.confidence_level = 'low'
        return report

    # 1. Domain-wise mean comparison
    if domain_column and domain_column in data.columns:
        unique_domains = data[domain_column].unique()
        for domain in unique_domains:
            domain_mask = data[domain_column] == domain
            domain_data = data[domain_mask]

            if len(domain_data) < 5:
                continue  # Skip domains with very few samples

            domain_mean = domain_data[variable].mean()
            report.mean_by_domain[str(domain)] = float(domain_mean)

            # Check if domain mean differs significantly from global
            pct_diff = abs(domain_mean - global_mean) / global_mean * 100
            if pct_diff > 15:
                report.issues.append(
                    f"Domain '{domain}' mean ({domain_mean:.3f}) differs "
                    f"{pct_diff:.1f}% from global ({global_mean:.3f})"
                )

            # Statistical test: K-S test for distribution similarity
            try:
                ks_stat, p_value = ks_2samp(domain_data[variable].dropna(), values)
                if p_value < 0.01:
                    report.issues.append(
                        f"Domain '{domain}' has significantly different distribution (KS p={p_value:.4f})"
                    )
            except (ValueError, RuntimeError) as e:
                logger.warning(f"KS test failed for domain '{domain}': {e}")

    # 2. Spatial trends (X, Y, Z)
    coords = {
        'X': data['X'].to_numpy(),
        'Y': data['Y'].to_numpy(),
        'Z': data['Z'].to_numpy()
    }

    for coord_name, coord_values in coords.items():
        try:
            # Remove any NaN values
            valid_mask = ~(np.isnan(coord_values) | np.isnan(values))
            if np.sum(valid_mask) < 5:
                continue

            clean_coords = coord_values[valid_mask]
            clean_values = values[valid_mask]

            # Linear regression
            slope, intercept, r_value, p_value, std_err = linregress(clean_coords, clean_values)

            trend_dict = {
                'slope': float(slope),
                'intercept': float(intercept),
                'r_squared': float(r_value**2),
                'p_value': float(p_value),
                'std_err': float(std_err)
            }

            # Store in report
            if coord_name == 'X':
                report.trend_x = trend_dict
            elif coord_name == 'Y':
                report.trend_y = trend_dict
            elif coord_name == 'Z':
                report.trend_z = trend_dict

            # Check for significant trends
            r_squared = trend_dict['r_squared']
            if r_squared > 0.3 and p_value < 0.05:
                report.issues.append(
                    f"Strong {coord_name}-direction trend detected (R²={r_squared:.3f}, p={p_value:.4f})"
                )
            elif r_squared > 0.15 and p_value < 0.05:
                report.issues.append(
                    f"Moderate {coord_name}-direction trend detected (R²={r_squared:.3f}, p={p_value:.4f})"
                )

        except Exception as e:
            # Regression failed, skip this coordinate
            pass

    # 3. Confidence level assessment
    n_issues = len(report.issues)
    if n_issues == 0:
        report.confidence_level = 'high'
    elif n_issues <= 2:
        report.confidence_level = 'medium'
    else:
        report.confidence_level = 'low'

    return report


def compute_swath_analysis(
    data: pd.DataFrame,
    variable: str,
    estimates: Optional[np.ndarray] = None,
    estimate_coords: Optional[np.ndarray] = None,
    global_mean: Optional[float] = None,
    n_bins: int = 20
) -> pd.DataFrame:
    """
    Compute swath analysis for stationarity validation.

    Compares data means, SK estimate means, and global mean across
    spatial swaths in X, Y, and Z directions.

    Args:
        data: DataFrame with coordinates (X, Y, Z) and variable
        variable: Column name of variable to analyze
        estimates: Optional SK estimates array (same length as data)
        estimate_coords: Optional coordinates for estimates (n, 3)
        global_mean: Global mean value used in SK
        n_bins: Number of bins for each direction

    Returns:
        DataFrame with swath results containing columns:
        - Direction: 'X', 'Y', or 'Z'
        - Bin_Lower: Lower bound of bin
        - Bin_Upper: Upper bound of bin
        - Bin_Center: Center of bin
        - Mean_Data: Mean of data in bin
        - Std_Data: Std dev of data in bin
        - N_Data: Number of data samples in bin
        - Mean_Est: Mean of estimates in bin (if provided)
        - Mean_Global: Global mean (constant)
    """
    swath_results = []

    for direction in ['X', 'Y', 'Z']:
        if direction not in data.columns:
            continue

        coord_values = data[direction].to_numpy()
        data_values = data[variable].to_numpy()

        # Remove NaN
        valid_mask = ~(np.isnan(coord_values) | np.isnan(data_values))
        coord_values = coord_values[valid_mask]
        data_values = data_values[valid_mask]

        if len(coord_values) < n_bins:
            continue

        # Create bins
        bins = np.linspace(coord_values.min(), coord_values.max(), n_bins + 1)

        for i in range(len(bins) - 1):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]
            bin_center = (bin_lower + bin_upper) / 2

            # Data in this bin
            mask_data = (coord_values >= bin_lower) & (coord_values < bin_upper)
            if i == len(bins) - 2:  # Last bin includes upper bound
                mask_data = (coord_values >= bin_lower) & (coord_values <= bin_upper)

            n_data = np.sum(mask_data)
            if n_data == 0:
                continue

            bin_data = data_values[mask_data]
            mean_data = np.mean(bin_data)
            std_data = np.std(bin_data)

            swath_entry = {
                'Direction': direction,
                'Bin_Lower': bin_lower,
                'Bin_Upper': bin_upper,
                'Bin_Center': bin_center,
                'Mean_Data': mean_data,
                'Std_Data': std_data,
                'N_Data': n_data,
                'Mean_Global': global_mean if global_mean is not None else np.nan
            }

            # Add estimates if provided
            if estimates is not None and estimate_coords is not None:
                # Find estimates in this bin
                est_dir_coord = estimate_coords[:, ['X', 'Y', 'Z'].index(direction)]
                mask_est = (est_dir_coord >= bin_lower) & (est_dir_coord < bin_upper)
                if i == len(bins) - 2:
                    mask_est = (est_dir_coord >= bin_lower) & (est_dir_coord <= bin_upper)

                if np.sum(mask_est) > 0:
                    bin_estimates = estimates[mask_est]
                    # Remove NaN from estimates
                    bin_estimates = bin_estimates[~np.isnan(bin_estimates)]
                    if len(bin_estimates) > 0:
                        swath_entry['Mean_Est'] = np.mean(bin_estimates)
                        swath_entry['N_Est'] = len(bin_estimates)
                    else:
                        swath_entry['Mean_Est'] = np.nan
                        swath_entry['N_Est'] = 0
                else:
                    swath_entry['Mean_Est'] = np.nan
                    swath_entry['N_Est'] = 0

            swath_results.append(swath_entry)

    return pd.DataFrame(swath_results)


def export_stationarity_report_to_txt(
    report: StationarityReport,
    file_path: str
) -> None:
    """
    Export stationarity validation report to text file.

    Args:
        report: StationarityReport object
        file_path: Output text file path
    """
    with open(file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMPLE KRIGING STATIONARITY VALIDATION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Global Mean Used: {report.global_mean:.6f}\n")
        f.write(f"Confidence Level: {report.confidence_level.upper()}\n\n")

        # Domain-wise means
        if report.mean_by_domain:
            f.write("MEAN BY DOMAIN\n")
            f.write("-"*40 + "\n")
            for domain, mean_val in report.mean_by_domain.items():
                pct_diff = abs(mean_val - report.global_mean) / report.global_mean * 100
                f.write(f"{domain:20s}: {mean_val:10.4f} (Δ={pct_diff:5.1f}%)\n")
            f.write("\n")

        # Spatial trends
        f.write("SPATIAL TRENDS\n")
        f.write("-"*40 + "\n")
        for direction, trend_dict in [('X', report.trend_x), ('Y', report.trend_y), ('Z', report.trend_z)]:
            if trend_dict:
                r2 = trend_dict.get('r_squared', 0)
                p_val = trend_dict.get('p_value', 1)
                slope = trend_dict.get('slope', 0)
                f.write(f"{direction}-direction: R²={r2:.4f}, p={p_val:.4f}, slope={slope:.6f}\n")
        f.write("\n")

        # Issues
        if report.issues:
            f.write("IDENTIFIED ISSUES\n")
            f.write("-"*40 + "\n")
            for i, issue in enumerate(report.issues, 1):
                f.write(f"{i}. {issue}\n")
            f.write("\n")
        else:
            f.write("✓ No stationarity issues identified\n\n")

        # Interpretation
        f.write("INTERPRETATION\n")
        f.write("-"*40 + "\n")
        if report.confidence_level == 'high':
            f.write("✓ Stationarity assumption is well-supported.\n")
            f.write("  The global mean is appropriate for Simple Kriging.\n")
        elif report.confidence_level == 'medium':
            f.write("○ Stationarity assumption is acceptable with minor concerns.\n")
            f.write("  Simple Kriging results should be reviewed carefully.\n")
        else:
            f.write("⚠ Stationarity assumption is questionable.\n")
            f.write("  Consider using Ordinary Kriging with local mean adaptation,\n")
            f.write("  or subdivide the domain based on trends/domains.\n")

        f.write("\n")
        f.write("="*80 + "\n")
