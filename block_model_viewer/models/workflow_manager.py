"""
SGSIM Workflow Manager
======================

Professional workflow controller for Sequential Gaussian Simulation following
the Datamine/Surpac/Isatis architecture:

1. TRANSFORM: Raw Data → Gaussian Space (Normal Score Transform)
2. MODEL: Variogram on Gaussian Data
3. SIMULATE: SGSIM on Gaussian Data (returns Gaussian realizations)
4. BACK-TRANSFORM: Gaussian Realizations → Raw Grade Space
5. POST-PROCESS: Metal/Tonnage calculations on Raw Data

CRITICAL: Metal/Tonnage calculations MUST happen AFTER back-transformation.
The SGSIM engine is "blind" to physical units and only works in Gaussian space.

Author: Block Model Viewer Team
Date: 2025
"""

from typing import Dict, List, Optional, Callable, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Import SGSIM engine (Gaussian-only)
from .sgsim3d import run_sgsim_simulation, SGSIMParameters

# Import transformation
try:
    from .transform import NormalScoreTransformer
except ImportError:
    NormalScoreTransformer = None
    logger.warning("NormalScoreTransformer not available. Transformation may not work.")

# Import post-processing (for back-transformed data)
try:
    from .post_processing import (
        compute_summary_statistics_fast,
        compute_global_uncertainty
    )
except ImportError:
    compute_summary_statistics_fast = None
    compute_global_uncertainty = None
    logger.warning("Post-processing module not available. Some features may not work.")


def execute_simulation_workflow(
    raw_data_coords: np.ndarray,
    raw_data_values: np.ndarray,
    params: SGSIMParameters,
    cutoffs: Optional[List[float]] = None,
    transformer: Optional['NormalScoreTransformer'] = None,
    density: float = 2.7,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    """
    Execute complete SGSIM workflow following professional architecture.
    
    This is the CORRECT way to run SGSIM with metal/tonnage calculations.
    
    Workflow:
        1. TRANSFORM: Raw Data → Gaussian Space
        2. SIMULATE: SGSIM on Gaussian Data
        3. BACK-TRANSFORM: Gaussian → Raw Grade Space
        4. POST-PROCESS: Metal/Tonnage on Raw Data
    
    Parameters
    ----------
    raw_data_coords : np.ndarray
        Conditioning data coordinates (N, 3)
    raw_data_values : np.ndarray
        Raw grade values (N,) - e.g., Au in g/t, Cu in %
    params : SGSIMParameters
        Simulation parameters (variogram should be modeled on Gaussian data)
    cutoffs : List[float], optional
        Cutoff values for metal/tonnage calculations (in RAW grade units)
    transformer : NormalScoreTransformer, optional
        Pre-fitted transformer. If None, will be created automatically.
    density : float, optional
        Rock density (t/m³) for tonnage calculations
    progress_callback : Callable, optional
        Progress callback function(percent, message)
    
    Returns
    -------
    dict
        Complete results including:
        - 'gaussian_realizations': All realizations in Gaussian space (nreal, nz, ny, nx)
        - 'raw_realizations': Back-transformed realizations in physical space
        - 'summary': Summary statistics on raw data (for mining reports)
        - 'summary_gaussian': Summary statistics on Gaussian data (for quality check)
        - 'risk_analysis': Metal/tonnage uncertainty analysis (on raw data - VALID)
        - 'transformer': Fitted transformer (for future use)
        - 'params': Simulation parameters
    
    Progress breakdown:
        0-2%:   Setup and transformation
        2-85%:  SGSIM simulation
        85-88%: Back-transformation
        88-95%: Summary statistics
        95-100%: Metal/tonnage calculations
    """
    if NormalScoreTransformer is None:
        raise ImportError(
            "NormalScoreTransformer not available. "
            "Cannot transform raw data to Gaussian space."
        )
    
    # ========================================================================
    # STEP 1: TRANSFORM (Raw → Gaussian)
    # ========================================================================
    if progress_callback:
        progress_callback(0, "Fitting Normal Score Transformer...")
    
    if transformer is None:
        transformer = NormalScoreTransformer()
        transformer.fit(raw_data_values)
        logger.info("Normal Score Transformer fitted on raw data")
    
    if progress_callback:
        progress_callback(1, "Transforming data to Gaussian space...")
    
    # Transform data to Gaussian for SGSIM
    gaussian_data = transformer.transform(raw_data_values)
    logger.info(f"Data transformed to Gaussian space: {len(gaussian_data)} samples")
    
    if progress_callback:
        progress_callback(2, "Starting SGSIM simulation...")
    
    # ========================================================================
    # STEP 2: SIMULATE (Gaussian Space)
    # ========================================================================
    # Note: Ensure the variogram params passed here were modeled on 'gaussian_data'
    # The SGSIM engine only works in Gaussian space
    gaussian_realizations = run_sgsim_simulation(
        raw_data_coords,
        gaussian_data,
        params,
        progress_callback
    )
    
    logger.info(
        f"SGSIM complete: {gaussian_realizations.shape[0]} realizations in Gaussian space"
    )
    
    # ========================================================================
    # STEP 3: BACK-TRANSFORM (Gaussian → Raw)
    # ========================================================================
    # CRITICAL: This is the bridge that makes metal/tonnage calculations valid
    if progress_callback:
        progress_callback(85, "Back-transforming realizations to physical space...")
    
    logger.info("Back-transforming realizations from Gaussian to physical space...")
    
    n_real, nz, ny, nx = gaussian_realizations.shape
    raw_realizations = np.zeros_like(gaussian_realizations)
    
    for i in range(n_real):
        raw_realizations[i] = transformer.back_transform(gaussian_realizations[i])
    
    logger.info("Back-transformation complete. Realizations now in physical space.")
    
    if progress_callback:
        progress_callback(88, "Back-transformation complete")
    
    # ========================================================================
    # STEP 4: POST-PROCESS (Statistics & Metal/Tonnage on Raw Data)
    # ========================================================================
    # Now it is safe to calculate metal because we are back in g/t (or %)
    
    # Calculate Uncertainty (P10/P50/P90) on raw data
    if progress_callback:
        progress_callback(88, "Computing summary statistics (physical space)...")
    
    if compute_summary_statistics_fast is not None:
        summary = compute_summary_statistics_fast(raw_realizations)
        # Add missing fields for backward compatibility
        if 'var' not in summary:
            summary['var'] = summary['std'] ** 2
        if 'p25' not in summary:
            summary['p25'] = np.percentile(raw_realizations, 25, axis=0)
        if 'p75' not in summary:
            summary['p75'] = np.percentile(raw_realizations, 75, axis=0)
        if 'iqr' not in summary:
            summary['iqr'] = summary['p75'] - summary['p25']
    else:
        # Fallback
        summary = {
            'mean': np.mean(raw_realizations, axis=0),
            'var': np.var(raw_realizations, axis=0),
            'std': np.std(raw_realizations, axis=0),
            'p10': np.percentile(raw_realizations, 10, axis=0),
            'p25': np.percentile(raw_realizations, 25, axis=0),
            'p50': np.percentile(raw_realizations, 50, axis=0),
            'p75': np.percentile(raw_realizations, 75, axis=0),
            'p90': np.percentile(raw_realizations, 90, axis=0),
            'iqr': np.percentile(raw_realizations, 75, axis=0) - np.percentile(raw_realizations, 25, axis=0),
        }
        # Coefficient of variation
        mask = np.abs(summary['mean']) > 1e-6
        summary['cv'] = np.zeros_like(summary['mean'])
        summary['cv'][mask] = summary['std'][mask] / summary['mean'][mask]
    
    # Also compute on Gaussian for quality checking
    if compute_summary_statistics_fast is not None:
        summary_gaussian = compute_summary_statistics_fast(gaussian_realizations)
    else:
        summary_gaussian = {
            'mean': np.mean(gaussian_realizations, axis=0),
            'std': np.std(gaussian_realizations, axis=0),
            'p10': np.percentile(gaussian_realizations, 10, axis=0),
            'p50': np.percentile(gaussian_realizations, 50, axis=0),
            'p90': np.percentile(gaussian_realizations, 90, axis=0),
        }
    
    if progress_callback:
        progress_callback(90, "Summary statistics complete")
    
    # Calculate Metal/Tonnage curves (ON RAW DATA - VALID!)
    risk_analysis = {}
    if cutoffs:
        block_volume = params.xinc * params.yinc * params.zinc
        n_cutoffs = len(cutoffs)
        
        if compute_global_uncertainty is not None:
            # Use optimized vectorized implementation
            if progress_callback:
                progress_callback(90, "Computing metal/tonnage uncertainty...")
            
            risk_analysis = compute_global_uncertainty(
                raw_realizations,  # ✅ RAW DATA - metal calculations are valid
                cutoffs,
                block_volume,
                density,
                is_gaussian=False  # ✅ Explicitly mark as raw data
            )
            
            if progress_callback:
                progress_callback(100, f"Complete! {params.nreal} realizations, {len(cutoffs)} cutoffs")
        else:
            # Fallback: compute manually
            logger.warning(
                "compute_global_uncertainty not available. "
                "Using fallback metal/tonnage calculation."
            )
            
            for i, cutoff in enumerate(cutoffs):
                if progress_callback:
                    pct = 90 + int((i / n_cutoffs) * 10)
                    progress_callback(pct, f"Metal/tonnage {i+1}/{n_cutoffs} (cutoff={cutoff})...")
                
                # Simple exceedance calculation
                n_real = raw_realizations.shape[0]
                tonnages = []
                grades = []
                metals = []
                
                for j in range(n_real):
                    real = raw_realizations[j]
                    mask = real > cutoff
                    n_blocks = mask.sum()
                    
                    if n_blocks > 0:
                        volume = n_blocks * block_volume
                        tonnage = volume * density
                        grade = np.mean(real[mask])
                        metal = tonnage * grade / 100.0  # Assuming grade in %
                        
                        tonnages.append(tonnage)
                        grades.append(grade)
                        metals.append(metal)
                    else:
                        tonnages.append(0.0)
                        grades.append(0.0)
                        metals.append(0.0)
                
                risk_analysis[cutoff] = {
                    'tonnage': {
                        'mean': np.mean(tonnages),
                        'p10': np.percentile(tonnages, 10),
                        'p50': np.percentile(tonnages, 50),
                        'p90': np.percentile(tonnages, 90),
                    },
                    'grade': {
                        'mean': np.mean(grades),
                    },
                    'metal': {
                        'mean': np.mean(metals),
                        'p10': np.percentile(metals, 10),
                        'p50': np.percentile(metals, 50),
                        'p90': np.percentile(metals, 90),
                    }
                }
            
            if progress_callback:
                progress_callback(100, f"Complete! {params.nreal} realizations, {len(cutoffs)} cutoffs")
    
    logger.info(
        f"Workflow complete: {params.nreal} realizations, "
        f"{len(cutoffs) if cutoffs else 0} cutoffs, "
        f"metal/tonnage calculated on RAW data (VALID)"
    )
    
    # ========================================================================
    # RETURN RESULTS
    # ========================================================================
    return {
        'gaussian_realizations': gaussian_realizations,  # For validation
        'raw_realizations': raw_realizations,  # For mining reports
        'summary': summary,  # On raw data - CORRECT for mining
        'summary_gaussian': summary_gaussian,  # On Gaussian - for quality checks
        'risk_analysis': risk_analysis,  # On raw data - VALID metal/tonnage
        'transformer': transformer,  # For future use
        'params': params
    }

