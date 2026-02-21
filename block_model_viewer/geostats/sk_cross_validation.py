"""
Cross-validation framework for Simple Kriging.

Provides Leave-One-Out Cross-Validation (LOOCV) and validation metrics
to assess SK estimation accuracy and support stock exchange reporting.
"""

from dataclasses import dataclass
from typing import Optional, Callable
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import linregress


@dataclass
class CrossValidationResults:
    """
    Simple Kriging cross-validation results.

    Contains per-sample validation results and summary metrics
    suitable for JORC/NI 43-101 reporting.
    """
    # Per-sample results
    sample_ids: np.ndarray  # Original sample indices
    actual_values: np.ndarray  # True values
    estimated_values: np.ndarray  # CV estimates
    residuals: np.ndarray  # actual - estimated

    # Summary metrics
    me: float  # Mean error (bias indicator)
    mae: float  # Mean absolute error
    rmse: float  # Root mean squared error
    r_squared: float  # Coefficient of determination
    regression_slope: float  # Slope of actual vs estimated
    regression_intercept: float  # Intercept of regression line

    # Per-sample diagnostics
    distances_to_nearest: np.ndarray  # Distance to nearest sample in training set
    n_samples_used: np.ndarray  # Number of samples used per estimate


def run_loocv_simple_kriging(
    coords: np.ndarray,
    values: np.ndarray,
    sk_params,  # SKParameters from simple_kriging3d
    progress_callback: Optional[Callable] = None
) -> CrossValidationResults:
    """
    Leave-One-Out Cross-Validation for Simple Kriging.

    For each sample i:
    1. Remove sample i from dataset
    2. Estimate at location i using remaining samples
    3. Compare estimate to actual value

    Args:
        coords: Sample coordinates, shape (n, 3) [X, Y, Z]
        values: Sample values, shape (n,)
        sk_params: Simple kriging parameters (from simple_kriging3d module)
        progress_callback: Optional callback(progress_pct: int, message: str)

    Returns:
        CrossValidationResults with validation metrics

    Notes:
        - This is computationally intensive: O(n^2) for n samples
        - Progress callback recommended for large datasets
        - Results suitable for CP review and stock exchange reporting
    """
    from ..models.simple_kriging3d import simple_kriging_3d

    n = len(values)
    estimates = np.full(n, np.nan)
    distances_to_nearest = np.full(n, np.nan)
    n_samples_used = np.zeros(n, dtype=int)

    for i in range(n):
        if progress_callback and i % 100 == 0:
            progress_pct = int(100 * i / n)
            progress_callback(progress_pct, f"Cross-validation: {i}/{n} samples")

        # Create LOOCV dataset (all samples except i)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        coords_train = coords[mask]
        values_train = values[mask]

        # Estimate at target location
        target_coord = coords[i:i+1]  # Shape (1, 3)

        try:
            # Call simple_kriging_3d with SKParameters (correct API)
            est, var, nn, diagnostics = simple_kriging_3d(
                coords_train, values_train, target_coord, sk_params
            )

            estimates[i] = est[0]
            n_samples_used[i] = nn[0]

            # Distance to nearest sample in training set
            dists = np.linalg.norm(coords_train - coords[i], axis=1)
            distances_to_nearest[i] = np.min(dists) if len(dists) > 0 else np.nan

        except Exception as e:
            # If estimation fails for this point, leave as NaN
            # This can happen with extreme search parameters
            estimates[i] = np.nan
            n_samples_used[i] = 0
            distances_to_nearest[i] = np.nan

    # Final progress callback
    if progress_callback:
        progress_callback(100, "Cross-validation complete")

    # Remove any NaN estimates for metric calculation
    valid_mask = ~np.isnan(estimates)
    n_valid = np.sum(valid_mask)

    if n_valid < 2:
        # Not enough valid estimates to compute metrics
        return CrossValidationResults(
            sample_ids=np.arange(n),
            actual_values=values,
            estimated_values=estimates,
            residuals=estimates - values,
            me=np.nan,
            mae=np.nan,
            rmse=np.nan,
            r_squared=np.nan,
            regression_slope=np.nan,
            regression_intercept=np.nan,
            distances_to_nearest=distances_to_nearest,
            n_samples_used=n_samples_used
        )

    # Compute metrics on valid estimates only
    valid_actual = values[valid_mask]
    valid_estimates = estimates[valid_mask]

    residuals = estimates - values  # Keep full array for export
    valid_residuals = residuals[valid_mask]

    me = np.mean(valid_residuals)
    mae = mean_absolute_error(valid_actual, valid_estimates)
    rmse = np.sqrt(mean_squared_error(valid_actual, valid_estimates))

    try:
        r_squared = r2_score(valid_actual, valid_estimates)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"R² calculation failed: {e}, setting to NaN")
        r_squared = np.nan

    # Regression slope (actual vs estimated)
    try:
        slope, intercept, _, _, _ = linregress(valid_actual, valid_estimates)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Regression calculation failed: {e}, setting to NaN")
        slope = np.nan
        intercept = np.nan

    return CrossValidationResults(
        sample_ids=np.arange(n),
        actual_values=values,
        estimated_values=estimates,
        residuals=residuals,
        me=me,
        mae=mae,
        rmse=rmse,
        r_squared=r_squared,
        regression_slope=slope,
        regression_intercept=intercept,
        distances_to_nearest=distances_to_nearest,
        n_samples_used=n_samples_used
    )


def export_cv_results_to_csv(
    cv_results: CrossValidationResults,
    coords: np.ndarray,
    file_path: str
) -> None:
    """
    Export cross-validation results to CSV.

    Args:
        cv_results: CrossValidationResults object
        coords: Sample coordinates, shape (n, 3)
        file_path: Output CSV file path
    """
    import pandas as pd

    df = pd.DataFrame({
        'Sample_ID': cv_results.sample_ids,
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'Actual': cv_results.actual_values,
        'Estimated': cv_results.estimated_values,
        'Residual': cv_results.residuals,
        'Dist_To_Nearest': cv_results.distances_to_nearest,
        'N_Samples_Used': cv_results.n_samples_used
    })

    df.to_csv(file_path, index=False, float_format='%.6f')


def export_cv_summary_to_txt(
    cv_results: CrossValidationResults,
    file_path: str
) -> None:
    """
    Export cross-validation summary to text report.

    Args:
        cv_results: CrossValidationResults object
        file_path: Output text file path
    """
    with open(file_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SIMPLE KRIGING CROSS-VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write("VALIDATION METRICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Mean Error (ME):              {cv_results.me:>12.4f}\n")
        f.write(f"Mean Absolute Error (MAE):    {cv_results.mae:>12.4f}\n")
        f.write(f"Root Mean Squared Error:      {cv_results.rmse:>12.4f}\n")
        f.write(f"R-Squared:                    {cv_results.r_squared:>12.4f}\n")
        f.write(f"Regression Slope:             {cv_results.regression_slope:>12.4f}\n")
        f.write(f"Regression Intercept:         {cv_results.regression_intercept:>12.4f}\n")
        f.write("\n")

        f.write("INTERPRETATION\n")
        f.write("-"*40 + "\n")

        # ME interpretation
        if abs(cv_results.me) < 0.1 * np.std(cv_results.actual_values):
            f.write("✓ Mean Error: Minimal bias detected\n")
        else:
            f.write("⚠ Mean Error: Systematic bias present\n")

        # R² interpretation
        if cv_results.r_squared > 0.7:
            f.write("✓ R-Squared: Strong correlation (>0.7)\n")
        elif cv_results.r_squared > 0.5:
            f.write("○ R-Squared: Moderate correlation (0.5-0.7)\n")
        else:
            f.write("⚠ R-Squared: Weak correlation (<0.5)\n")

        # Slope interpretation
        if 0.9 <= cv_results.regression_slope <= 1.1:
            f.write("✓ Regression Slope: Near ideal (0.9-1.1)\n")
        else:
            f.write(f"○ Regression Slope: {cv_results.regression_slope:.3f} (ideal=1.0)\n")

        f.write("\n")
        f.write("PROFESSIONAL CONTEXT\n")
        f.write("-"*40 + "\n")
        f.write("Leave-One-Out Cross-Validation (LOOCV) provides an unbiased estimate\n")
        f.write("of Simple Kriging prediction error. Each sample is estimated using all\n")
        f.write("other samples, then compared to its true value.\n\n")
        f.write("These metrics are suitable for JORC/NI 43-101 reporting and CP review.\n")
        f.write("\n")

        # Sample statistics
        n_total = len(cv_results.sample_ids)
        n_valid = np.sum(~np.isnan(cv_results.estimated_values))
        f.write(f"Total samples validated:      {n_total}\n")
        f.write(f"Successfully estimated:       {n_valid}\n")
        if n_valid < n_total:
            f.write(f"Failed estimates:             {n_total - n_valid}\n")
        f.write("\n")

        f.write("="*80 + "\n")
