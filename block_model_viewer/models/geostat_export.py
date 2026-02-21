"""
Professional Geostatistical Results Export
==========================================

Export functions for comprehensive geostatistical results to CSV and VTK formats.
Handles all professional output attributes from estimation and simulation methods.

Author: Block Model Viewer Development Team
Date: 2025
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    import pyvista as pv
except ImportError:
    pv = None

from .geostat_results import (
    OrdinaryKrigingResults,
    SimpleKrigingResults,
    UniversalKrigingResults,
    IndicatorKrigingResults,
    CoKrigingResults,
    RegressionKrigingResults,
    SGSIMResults,
    SISResults,
    DBSResults,
    TurningBandsResults,
    MPSResults,
    CoSimulationResults,
)

logger = logging.getLogger(__name__)


def export_ok_results_to_csv(
    results: OrdinaryKrigingResults,
    coords: np.ndarray,
    output_path: str,
    include_optional: bool = False
) -> None:
    """
    Export Ordinary Kriging results to CSV with all professional attributes.
    
    Parameters
    ----------
    results : OrdinaryKrigingResults
        OK results object
    coords : np.ndarray
        (N, 3) array of coordinates (X, Y, Z)
    output_path : str
        Output CSV file path
    include_optional : bool
        Whether to include optional audit outputs (weights, sample coords)
    """
    logger.info(f"Exporting OK results to {output_path}")
    
    n_points = len(results.estimates)
    
    if coords.shape[0] != n_points:
        raise ValueError(f"Coordinate array size {coords.shape[0]} != results size {n_points}")
    
    # Build DataFrame with core attributes
    df_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'EST': results.estimates,
        'STATUS': results.status,
        'KM': results.kriging_mean,
        'KV': results.kriging_variance,
        'KE': results.kriging_efficiency,
        'SoR': results.slope_of_regression,
        'LM': results.lagrange_multiplier,
        'NS': results.num_samples,
        'SumW': results.sum_weights,
        'SumN': results.sum_negative_weights,
        'MinD': results.min_distance,
        'AvgD': results.avg_distance,
        'NearID': results.nearest_sample_id,
        'ND': results.num_duplicates_removed,
        'Pass': results.search_pass,
        'SearchVol': results.search_volume,
    }
    
    # Add anisotropy attributes if available
    if results.ellipsoid_rotation is not None:
        df_data['EllRot'] = results.ellipsoid_rotation
    if results.anisotropy_x is not None:
        df_data['AniX'] = results.anisotropy_x
    if results.anisotropy_y is not None:
        df_data['AniY'] = results.anisotropy_y
    if results.anisotropy_z is not None:
        df_data['AniZ'] = results.anisotropy_z
    
    # Add optional weight vectors if requested
    if include_optional and results.weight_vectors is not None:
        n_weights = results.weight_vectors.shape[1]
        for i in range(min(n_weights, 20)):  # Limit to first 20 weights
            df_data[f'W{i+1}'] = results.weight_vectors[:, i]
    
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(df)} points to {output_path}")


def export_ok_results_to_vtk(
    results: OrdinaryKrigingResults,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    output_path: str
) -> None:
    """
    Export Ordinary Kriging results to VTK format.
    
    Parameters
    ----------
    results : OrdinaryKrigingResults
        OK results object
    grid_x, grid_y, grid_z : np.ndarray
        Grid coordinate arrays
    output_path : str
        Output VTK file path
    """
    if pv is None:
        raise ImportError("PyVista is required for VTK export")
    
    logger.info(f"Exporting OK results to {output_path}")
    
    grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
    
    # Add all attributes as cell data
    grid["EST"] = results.estimates.ravel(order='F')
    grid["STATUS"] = results.status.ravel(order='F')
    grid["KM"] = results.kriging_mean.ravel(order='F')
    grid["KV"] = results.kriging_variance.ravel(order='F')
    grid["KE"] = results.kriging_efficiency.ravel(order='F')
    grid["SoR"] = results.slope_of_regression.ravel(order='F')
    grid["LM"] = results.lagrange_multiplier.ravel(order='F')
    grid["NS"] = results.num_samples.ravel(order='F')
    grid["SumW"] = results.sum_weights.ravel(order='F')
    grid["SumN"] = results.sum_negative_weights.ravel(order='F')
    grid["MinD"] = results.min_distance.ravel(order='F')
    grid["AvgD"] = results.avg_distance.ravel(order='F')
    grid["NearID"] = results.nearest_sample_id.ravel(order='F')
    grid["ND"] = results.num_duplicates_removed.ravel(order='F')
    grid["Pass"] = results.search_pass.ravel(order='F')
    grid["SearchVol"] = results.search_volume.ravel(order='F')
    
    if results.ellipsoid_rotation is not None:
        grid["EllRot"] = results.ellipsoid_rotation.ravel(order='F')
    if results.anisotropy_x is not None:
        grid["AniX"] = results.anisotropy_x.ravel(order='F')
    if results.anisotropy_y is not None:
        grid["AniY"] = results.anisotropy_y.ravel(order='F')
    if results.anisotropy_z is not None:
        grid["AniZ"] = results.anisotropy_z.ravel(order='F')
    
    grid.save(output_path)
    logger.info(f"Exported OK results to VTK: {output_path}")


def export_sgsim_results_to_csv(
    results: SGSIMResults,
    coords: np.ndarray,
    output_path: str,
    export_summary_only: bool = True
) -> None:
    """
    Export SGSIM results to CSV.
    
    Parameters
    ----------
    results : SGSIMResults
        SGSIM results object
    coords : np.ndarray
        (N, 3) array of coordinates
    output_path : str
        Output CSV file path
    export_summary_only : bool
        If True, export only summary statistics; if False, export all realizations
    """
    logger.info(f"Exporting SGSIM results to {output_path}")
    
    n_nodes = results.mean.size
    
    if coords.shape[0] != n_nodes:
        raise ValueError(f"Coordinate array size {coords.shape[0]} != results size {n_nodes}")
    
    df_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'MEAN': results.mean.ravel(),
        'VARIANCE': results.variance.ravel(),
        'STD_DEV': results.std_dev.ravel(),
        'CV': results.coefficient_of_variation.ravel(),
    }
    
    if results.p10 is not None:
        df_data['P10'] = results.p10.ravel()
    if results.p50 is not None:
        df_data['P50'] = results.p50.ravel()
    if results.p90 is not None:
        df_data['P90'] = results.p90.ravel()
    
    # Add probability maps if available
    if results.probability_above_cutoff is not None:
        n_cutoffs = results.probability_above_cutoff.shape[0]
        for i in range(n_cutoffs):
            cutoff_val = results.metadata.get('cutoffs', [])[i] if 'cutoffs' in results.metadata else i
            df_data[f'P_ABOVE_{cutoff_val:.2f}'] = results.probability_above_cutoff[i].ravel()
    
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported SGSIM summary to {output_path}")
    
    # Optionally export individual realizations
    if not export_summary_only and results.realizations is not None:
        base_path = output_path.replace('.csv', '')
        nreal = results.realizations.shape[0]
        for i in range(nreal):
            real_path = f"{base_path}_realization_{i+1:04d}.csv"
            real_df = pd.DataFrame({
                'X': coords[:, 0],
                'Y': coords[:, 1],
                'Z': coords[:, 2],
                'REALIZATION': results.realizations[i].ravel(),
            })
            real_df.to_csv(real_path, index=False)
        logger.info(f"Exported {nreal} individual realizations")


def export_sgsim_results_to_vtk(
    results: SGSIMResults,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    output_path: str,
    export_summary_only: bool = True
) -> None:
    """
    Export SGSIM results to VTK format.
    
    Parameters
    ----------
    results : SGSIMResults
        SGSIM results object
    grid_x, grid_y, grid_z : np.ndarray
        Grid coordinate arrays
    output_path : str
        Output VTK file path (base path for multiple files)
    export_summary_only : bool
        If True, export only summary; if False, export all realizations
    """
    if pv is None:
        raise ImportError("PyVista is required for VTK export")
    
    logger.info(f"Exporting SGSIM results to VTK")
    
    grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
    
    # SGSIM data is in (nz, ny, nx) format, but PyVista StructuredGrid expects (nx, ny, nz)
    # Transpose arrays to match PyVista's expected layout
    def transpose_sgsim_data(data):
        """Transpose SGSIM data from (nz, ny, nx) to (nx, ny, nz) for PyVista compatibility"""
        if data is not None and data.ndim == 3:
            # SGSIM: (nz, ny, nx) -> PyVista: (nx, ny, nz)
            return data.transpose(2, 1, 0)
        return data

    # Add summary statistics
    grid["MEAN"] = transpose_sgsim_data(results.mean).ravel(order='F')
    grid["VARIANCE"] = transpose_sgsim_data(results.variance).ravel(order='F')
    grid["STD_DEV"] = transpose_sgsim_data(results.std_dev).ravel(order='F')
    grid["CV"] = transpose_sgsim_data(results.coefficient_of_variation).ravel(order='F')

    if results.p10 is not None:
        grid["P10"] = transpose_sgsim_data(results.p10).ravel(order='F')
    if results.p50 is not None:
        grid["P50"] = transpose_sgsim_data(results.p50).ravel(order='F')
    if results.p90 is not None:
        grid["P90"] = transpose_sgsim_data(results.p90).ravel(order='F')

    # Add probability maps
    if results.probability_above_cutoff is not None:
        n_cutoffs = results.probability_above_cutoff.shape[0]
        cutoffs = results.metadata.get('cutoffs', [])
        for i in range(n_cutoffs):
            cutoff_val = cutoffs[i] if i < len(cutoffs) else i
            grid[f"P_ABOVE_{cutoff_val:.2f}"] = transpose_sgsim_data(results.probability_above_cutoff[i]).ravel(order='F')
    
    summary_path = output_path.replace('.vtk', '_summary.vtk')
    grid.save(summary_path)
    logger.info(f"Exported SGSIM summary to {summary_path}")
    
    # Export individual realizations if requested
    if not export_summary_only and results.realizations is not None:
        base_path = output_path.replace('.vtk', '')
        nreal = results.realizations.shape[0]
        for i in range(nreal):
            real_grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
            # Transpose each realization from (nz, ny, nx) to (nx, ny, nz)
            transposed_real = results.realizations[i].transpose(2, 1, 0)
            real_grid["REALIZATION"] = transposed_real.ravel(order='F')
            real_path = f"{base_path}_realization_{i+1:04d}.vtk"
            real_grid.save(real_path)
        logger.info(f"Exported {nreal} individual realizations")


def export_sk_results_to_csv(
    results: SimpleKrigingResults,
    coords: np.ndarray,
    output_path: str
) -> None:
    """Export Simple Kriging results to CSV."""
    logger.info(f"Exporting SK results to {output_path}")
    
    n_points = len(results.estimates)
    if coords.shape[0] != n_points:
        raise ValueError(f"Coordinate array size mismatch")
    
    df_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'EST': results.estimates,
        'STATUS': results.status,
        'GM': results.global_mean,
        'KV': results.kriging_variance,
        'KE': results.kriging_efficiency,
        'NS': results.num_samples,
        'SumW': results.sum_weights,
        'SumN': results.sum_negative_weights,
        'MinD': results.min_distance,
        'AvgD': results.avg_distance,
        'NearID': results.nearest_sample_id,
        'ND': results.num_duplicates_removed,
        'Pass': results.search_pass,
    }
    
    if results.search_volume is not None:
        df_data['SearchVol'] = results.search_volume
    
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported {len(df)} points to {output_path}")


def export_uk_results_to_csv(
    results: UniversalKrigingResults,
    coords: np.ndarray,
    output_path: str
) -> None:
    """Export Universal Kriging results to CSV."""
    logger.info(f"Exporting UK results to {output_path}")
    
    n_points = len(results.estimates)
    if coords.shape[0] != n_points:
        raise ValueError(f"Coordinate array size mismatch")
    
    df_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'EST': results.estimates,
        'STATUS': results.status,
        'KM': results.kriging_mean,
        'KV': results.kriging_variance,
        'KE': results.kriging_efficiency,
        'SoR': results.slope_of_regression,
        'LM': results.lagrange_multiplier,
        'NS': results.num_samples,
        'SumW': results.sum_weights,
        'SumN': results.sum_negative_weights,
        'MinD': results.min_distance,
        'AvgD': results.avg_distance,
        'NearID': results.nearest_sample_id,
        'ND': results.num_duplicates_removed,
        'Pass': results.search_pass,
        'DriftVal': results.drift_value,
        'EstResidual': results.residual_estimate,
    }
    
    if results.search_volume is not None:
        df_data['SearchVol'] = results.search_volume
    
    # Add trend coefficients if available
    if results.trend_coefficients is not None:
        n_beta = results.trend_coefficients.shape[1]
        for i in range(n_beta):
            df_data[f'Beta_{i}'] = results.trend_coefficients[:, i]
    
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported UK: {len(df)} points to {output_path}")


def export_ik_results_to_csv(
    results: IndicatorKrigingResults,
    coords: np.ndarray,
    output_path: str
) -> None:
    """Export Indicator Kriging results to CSV."""
    logger.info(f"Exporting IK results to {output_path}")
    
    n_points = len(results.indicator_probability)
    if coords.shape[0] != n_points:
        raise ValueError(f"Coordinate array size mismatch")
    
    df_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'Prob1': results.indicator_probability,
        'LCV': results.local_conditional_variance,
        'NS': results.num_samples,
        'IKV': results.indicator_kriging_variance,
        'Pass': results.search_pass,
        'MinD': results.min_distance,
        'AvgD': results.avg_distance,
        'ND': results.num_duplicates_removed,
    }
    
    # Add multiple probabilities if available
    if results.multiple_probabilities is not None:
        n_cutoffs = results.multiple_probabilities.shape[1]
        for i in range(n_cutoffs):
            threshold = results.metadata.get('thresholds', [])[i] if 'thresholds' in results.metadata else i
            df_data[f'Prob_{threshold:.2f}'] = results.multiple_probabilities[:, i]
    
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported IK: {len(df)} points to {output_path}")


def export_cok_results_to_csv(
    results: CoKrigingResults,
    coords: np.ndarray,
    output_path: str
) -> None:
    """Export Co-Kriging results to CSV."""
    logger.info(f"Exporting CoK results to {output_path}")
    
    n_points = len(results.primary_estimate)
    if coords.shape[0] != n_points:
        raise ValueError(f"Coordinate array size mismatch")
    
    df_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'Est1': results.primary_estimate,
        'Est2': results.secondary_estimate if results.secondary_estimate is not None else np.nan,
        'XCov': results.cross_covariance_contribution,
        'CoKV': results.cokriging_variance,
        'NS_Primary': results.num_samples_primary,
        'NS_Secondary': results.num_samples_secondary,
        'SumW_Primary': results.sum_weights_primary,
        'SumW_Secondary': results.sum_weights_secondary,
        'MinD': results.min_distance,
        'AvgD': results.avg_distance,
        'Pass': results.search_pass,
    }
    
    if results.primary_weight_fraction is not None:
        df_data['W_Frac_Primary'] = results.primary_weight_fraction
    if results.secondary_weight_fraction is not None:
        df_data['W_Frac_Secondary'] = results.secondary_weight_fraction
    
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported CoK: {len(df)} points to {output_path}")


def export_sis_results_to_csv(
    results: SISResults,
    coords: np.ndarray,
    output_path: str
) -> None:
    """Export Sequential Indicator Simulation results to CSV."""
    logger.info(f"Exporting SIS results to {output_path}")
    
    n_points = len(results.probability_indicator_one)
    if coords.shape[0] != n_points:
        raise ValueError(f"Coordinate array size mismatch")
    
    df_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'Prob1': results.probability_indicator_one.ravel() if results.probability_indicator_one.ndim > 1 else results.probability_indicator_one,
        'IndVar': results.indicator_variance_field.ravel() if results.indicator_variance_field.ndim > 1 else results.indicator_variance_field,
    }
    
    # Add realization data if available
    if results.indicator_realizations is not None:
        n_real = results.indicator_realizations.shape[0]
        for i in range(min(n_real, 10)):  # Limit to first 10 realizations
            real_data = results.indicator_realizations[i]
            df_data[f'Real_{i+1}'] = real_data.ravel() if real_data.ndim > 1 else real_data
    
    df = pd.DataFrame(df_data)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Exported SIS: {len(df)} points to {output_path}")


def export_uk_results_to_vtk(
    results: UniversalKrigingResults,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    output_path: str
) -> None:
    """Export Universal Kriging results to VTK format."""
    if pv is None:
        raise ImportError("PyVista is required for VTK export")
    
    logger.info(f"Exporting UK results to {output_path}")
    
    grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
    
    # Add all attributes as cell data
    grid["EST"] = results.estimates.ravel(order='F')
    grid["STATUS"] = results.status.ravel(order='F')
    grid["KM"] = results.kriging_mean.ravel(order='F')
    grid["KV"] = results.kriging_variance.ravel(order='F')
    grid["KE"] = results.kriging_efficiency.ravel(order='F')
    grid["SoR"] = results.slope_of_regression.ravel(order='F')
    grid["LM"] = results.lagrange_multiplier.ravel(order='F')
    grid["NS"] = results.num_samples.ravel(order='F')
    grid["SumW"] = results.sum_weights.ravel(order='F')
    grid["SumN"] = results.sum_negative_weights.ravel(order='F')
    grid["MinD"] = results.min_distance.ravel(order='F')
    grid["AvgD"] = results.avg_distance.ravel(order='F')
    grid["NearID"] = results.nearest_sample_id.ravel(order='F')
    grid["ND"] = results.num_duplicates_removed.ravel(order='F')
    grid["Pass"] = results.search_pass.ravel(order='F')
    grid["DriftVal"] = results.drift_value.ravel(order='F')
    grid["EstResidual"] = results.residual_estimate.ravel(order='F')
    
    if results.search_volume is not None:
        grid["SearchVol"] = results.search_volume.ravel(order='F')
    
    # Add trend coefficients if available
    if results.trend_coefficients is not None:
        n_beta = results.trend_coefficients.shape[1]
        for i in range(n_beta):
            grid[f"Beta_{i}"] = results.trend_coefficients[:, i].ravel(order='F')
    
    grid.save(output_path)
    logger.info(f"Exported UK results to VTK: {output_path}")

