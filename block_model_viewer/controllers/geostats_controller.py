"""
Geostatistics Controller - Handles all geostatistical operations.

This controller manages kriging, simulation, variogram analysis, and related
geostatistical computations. It is instantiated by AppController and provides
payload preparation methods for the job registry.

AUDIT COMPLIANCE (2025-12-16):
- Lineage gates integrated for variogram validation
- Pre-kriging validation enforced
- Data hash tracking for JORC/SAMREC compliance
"""

from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
import logging
import os

import numpy as np
import pandas as pd
import pyvista as pv

# Import lineage gates for variogram validation (AUDIT REQUIREMENT)
from ..geostats.variogram_gates import (
    validate_pre_kriging,
    VariogramGateError,
    STRICT_MODE as VARIOGRAM_STRICT_MODE,
    validate_data_source,  # AUDIT FIX (CRITICAL-002)
    DataSourceError,  # AUDIT FIX (CRITICAL-002)
)

if TYPE_CHECKING:
    from .app_controller import AppController

logger = logging.getLogger(__name__)


class GeostatsController:
    """
    Controller for geostatistical operations.
    
    Handles kriging (simple, ordinary, universal, co-kriging, indicator, bayesian),
    simulation (SGSIM, IK-SGSIM, CoSGSIM), variogram analysis, uncertainty propagation,
    grade statistics/transformations, swath analysis, and k-means clustering.
    """
    
    def __init__(self, app_controller: "AppController"):
        """
        Initialize geostatistics controller.
        
        Args:
            app_controller: Parent AppController instance for shared state access
        """
        self._app = app_controller
        self._block_model = None
    
    @property
    def block_model(self):
        """Return the currently loaded block model."""
        if self._block_model is not None:
            return self._block_model
        return self._app.block_model
    
    # =========================================================================
    # Simple Kriging
    # =========================================================================
    
    def _prepare_simple_kriging_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Run the Simple Kriging engine and package results for the UI.
        
        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data' into params before calling this.
        
        DEBUG MODE: Set params['debug_mode'] = True to enable comprehensive debugging
        with detailed logging to 'sk_debug_log.txt' file.
        """
        from ..models.simple_kriging3d import simple_kriging_3d, SKParameters
        
        # Check for debug mode
        debug_mode = params.get('debug_mode', False)
        if debug_mode:
            logger.warning("=" * 60)
            logger.warning("SIMPLE KRIGING DEBUG MODE ENABLED")
            logger.warning("Comprehensive logging will be written to sk_debug_log.txt")
            logger.warning("=" * 60)
            
            # Use debug wrapper for comprehensive error tracking
            from ..geostats.sk_debugger import SimpleKrigingDebugger
            debugger = SimpleKrigingDebugger()
            
            try:
                # Wrap the entire execution with debugging
                debugger.log_stage("Controller: _prepare_simple_kriging_payload called")
                debugger.log_stage("Debug mode activated by user")
                
                # Validate inputs immediately
                validation = debugger.validate_inputs(params)
                if not validation['valid']:
                    error_msg = "Input validation failed:\n" + "\n".join(f"  - {issue}" for issue in validation['issues'])
                    debugger.log_error(ValueError(error_msg), "Input Validation")
                    debugger.finalize()
                    raise ValueError(error_msg)
                
                # Create debug progress callback
                debug_progress = debugger.create_progress_callback(progress_callback)
                params['_debug_progress'] = debug_progress
                
                # Continue with normal execution but with debug tracking
                debugger.log_stage("Proceeding to normal execution path with debug tracking")
                
            except Exception as e:
                debugger.log_error(e, "Debug setup")
                debugger.finalize()
                raise
        else:
            debugger = None
        
        # Use debug progress callback if available
        if debug_mode and '_debug_progress' in params:
            progress_callback = params['_debug_progress']
        
        if debugger:
            debugger.log_stage("Starting data validation")
        
        df = params.get("data")
        if df is None or df.empty:
            error = ValueError(
                "No drillhole data provided for Simple Kriging. "
                "Controller must fetch data from DataRegistry and inject it before calling worker."
            )
            if debugger:
                debugger.log_error(error, "Data extraction")
                debugger.finalize()
            raise error
        
        # AUDIT FIX (CRITICAL-002): Validate data source at engine level
        # Defense-in-depth: even if controller passed data, ensure it's not raw assays
        if debugger:
            debugger.log_stage("Running data source validation")
        
        try:
            data_source_result = validate_data_source(df, "Simple Kriging", strict=VARIOGRAM_STRICT_MODE)
            if not data_source_result['valid']:
                raise DataSourceError(data_source_result['warnings'][0] if data_source_result['warnings'] else "Data source validation failed")
        except DataSourceError:
            if debugger:
                debugger.log_error(DataSourceError("Data source validation failed"), "Data source validation")
                debugger.finalize()
            raise
        except Exception as e:
            logger.warning(f"SK data source validation skipped (non-fatal): {e}")
            if debugger:
                debugger.log_stage(f"Data source validation skipped: {e}")

        if debugger:
            debugger.log_stage("Data source validation passed")
        
        variable = params.get("variable")
        if not variable or variable not in df.columns:
            error = ValueError("Selected variable not present in drillhole data.")
            if debugger:
                debugger.log_error(error, "Variable validation")
                debugger.finalize()
            raise error

        grid_spec = params.get("grid_spec")
        if not grid_spec:
            error = ValueError("Grid specification missing for Simple Kriging.")
            if debugger:
                debugger.log_error(error, "Grid spec validation")
                debugger.finalize()
            raise error
        
        if debugger:
            debugger.log_stage("Cleaning data", {
                'original_rows': len(df),
                'variable': variable,
                'columns': list(df.columns)
            })

        cleaned = df.dropna(subset=["X", "Y", "Z", variable])
        # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
        if hasattr(df, 'attrs') and df.attrs:
            cleaned.attrs = df.attrs.copy()
        
        if debugger:
            debugger.log_stage("Data cleaned", {
                'cleaned_rows': len(cleaned),
                'dropped_rows': len(df) - len(cleaned),
                'has_attrs': hasattr(cleaned, 'attrs')
            })
        
        if cleaned.empty:
            error = ValueError("No valid samples available after filtering missing coordinates or values.")
            if debugger:
                debugger.log_error(error, "Data cleaning")
                debugger.finalize()
            raise error

        if debugger:
            debugger.log_stage("Extracting coordinates and values")
        
        coords = cleaned[["X", "Y", "Z"]].to_numpy(float)
        values = cleaned[variable].to_numpy(float)
        
        if debugger:
            debugger.log_stage("Coordinates extracted", {
                'n_samples': len(coords),
                'coords_shape': coords.shape,
                'values_shape': values.shape,
                'values_min': float(np.nanmin(values)),
                'values_max': float(np.nanmax(values)),
                'values_mean': float(np.nanmean(values))
            })

        if debugger:
            debugger.log_stage("Building estimation grid")
        
        nx = int(grid_spec["nx"])
        ny = int(grid_spec["ny"])
        nz = int(grid_spec["nz"])
        xmin = float(grid_spec["xmin"])
        ymin = float(grid_spec["ymin"])
        zmin = float(grid_spec["zmin"])
        xinc = float(grid_spec["xinc"])
        yinc = float(grid_spec["yinc"])
        zinc = float(grid_spec["zinc"])

        if progress_callback:
            progress_callback(5, "Creating estimation grid...")

        gx = np.arange(nx) * xinc + xmin + xinc / 2.0
        gy = np.arange(ny) * yinc + ymin + yinc / 2.0
        gz = np.arange(nz) * zinc + zmin + zinc / 2.0
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        grid_coords = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])

        param_values = params.get("parameters", {})
        
        # AUDIT GATE: Validate variogram parameters for Simple Kriging (JORC/SAMREC requirement)
        sk_variogram_params = {
            'nugget': float(param_values.get("nugget", 0.0)),
            'sill': float(param_values.get("sill", 1.0)),
            'range': float(param_values.get("range_major", 100.0)),
            'model_type': str(param_values.get("variogram_type", "spherical")),
        }
        try:
            gate_result = validate_pre_kriging(
                variogram_params=sk_variogram_params,
                estimation_data=cleaned,
                estimation_coords=coords,
                variable_name=variable,
                strict=VARIOGRAM_STRICT_MODE,
            )
            if not gate_result['passed']:
                failures = [f for f in gate_result.get('failures', []) if f.get('severity') == 'FATAL']
                if failures:
                    raise VariogramGateError(
                        f"Simple Kriging pre-validation FAILED:\n" + 
                        "\n".join(f"- {f['check']}: {f['message']}" for f in failures)
                    )
                for w in gate_result.get('failures', []):
                    if w.get('severity') != 'FATAL':
                        logger.warning(f"SK variogram gate warning: {w['check']} - {w['message']}")
        except VariogramGateError:
            raise
        except Exception as e:
            logger.warning(f"SK variogram gate validation skipped: {e}")
        
        sk_params = SKParameters(
            global_mean=float(param_values.get("global_mean", float(np.nanmean(values)))),
            variogram_type=str(param_values.get("variogram_type", "spherical")),
            sill=float(param_values.get("sill", 1.0)),
            nugget=float(param_values.get("nugget", 0.0)),
            range_major=float(param_values.get("range_major", 100.0)),
            range_minor=float(param_values.get("range_minor", 50.0)),
            range_vert=float(param_values.get("range_vert", 25.0)),
            azimuth=float(param_values.get("azimuth", 0.0)),
            dip=float(param_values.get("dip", 0.0)),
            ndmax=int(param_values.get("ndmax", 12)),
            max_search_radius=float(param_values.get("max_search_radius", 200.0)),
            nmin=int(param_values.get("nmin", 1)),
            sectoring=str(param_values.get("sectoring", "No sectoring")),
        )

        # STATIONARITY VALIDATION (Component 3)
        # Validate that the global mean assumption is appropriate for SK
        from ..geostats.sk_stationarity import validate_stationarity

        domain_column = params.get("domain_column", None)
        if domain_column and domain_column not in cleaned.columns:
            domain_column = None

        stationarity_report = validate_stationarity(
            data=cleaned,
            variable=variable,
            global_mean=sk_params.global_mean,
            domain_column=domain_column
        )

        # Log stationarity confidence
        if stationarity_report.confidence_level == 'low':
            logger.warning(
                f"Stationarity confidence is LOW for variable '{variable}'. "
                f"Issues: {', '.join(stationarity_report.issues)}"
            )
        elif stationarity_report.confidence_level == 'medium':
            logger.info(
                f"Stationarity confidence is MEDIUM for variable '{variable}'. "
                f"Review recommended."
            )
        else:
            logger.info(f"Stationarity confidence is HIGH for variable '{variable}'.")

        if progress_callback:
            progress_callback(10, "Starting Simple Kriging...")
        
        if debugger:
            debugger.log_stage("CALLING simple_kriging_3d - CRITICAL EXECUTION POINT", {
                'n_data_samples': len(coords),
                'n_grid_points': len(grid_coords),
                'variogram_type': sk_params.variogram_type,
                'ndmax': sk_params.ndmax
            })

        try:
            estimates, variances, neighbour_counts, diagnostics = simple_kriging_3d(
                coords, values, grid_coords, sk_params, progress_callback=progress_callback
            )
            
            if debugger:
                debugger.log_stage("simple_kriging_3d COMPLETED SUCCESSFULLY", {
                    'estimates_shape': estimates.shape,
                    'n_nan': int(np.sum(np.isnan(estimates))),
                    'estimates_min': float(np.nanmin(estimates)) if not np.all(np.isnan(estimates)) else float('nan'),
                    'estimates_max': float(np.nanmax(estimates)) if not np.all(np.isnan(estimates)) else float('nan')
                })
                
        except Exception as e:
            if debugger:
                debugger.log_error(e, "simple_kriging_3d execution")
                debugger.finalize()
            raise

        # Calculate SK Mean Collapse Diagnostic
        estimates_flat = estimates.ravel()
        valid_estimates = estimates_flat[~np.isnan(estimates_flat)]

        std_estimates = np.std(valid_estimates) if len(valid_estimates) > 0 else 0.0
        std_data = np.std(values) if len(values) > 0 else 0.0
        collapse_threshold = 0.05 * std_data
        sk_mean_collapsed = std_estimates < collapse_threshold
        collapse_ratio = std_estimates / std_data if std_data > 0 else 0.0

        # Determine collapse severity
        if collapse_ratio < 0.01:
            collapse_severity = "SEVERE"
        elif collapse_ratio < 0.05:
            collapse_severity = "MODERATE"
        else:
            collapse_severity = "NONE"

        # Create SimpleKrigingResults object with diagnostics
        from ..models.geostat_results import SimpleKrigingResults

        # Compute stability flags
        m = len(estimates)
        stability_flags = np.zeros(m, dtype=int)
        pct_neg = diagnostics['pct_negative_weights']
        sum_w = diagnostics['sum_weights']
        slv_status = diagnostics['solver_status']

        # Bit 0: High negative weights (>20%)
        stability_flags[pct_neg > 20] |= 1
        # Bit 1: Bad sum of weights (|sum - 1.0| > 0.1)
        stability_flags[np.abs(sum_w - 1.0) > 0.1] |= 2
        # Bit 2: Solver issue (singular matrix)
        stability_flags[slv_status > 0] |= 4

        sk_results = SimpleKrigingResults(
            estimates=estimates,
            status=np.ones_like(estimates, dtype=int),  # 1 = estimated
            global_mean=float(sk_params.global_mean),
            kriging_variance=variances,
            kriging_efficiency=1.0 - (variances / np.var(values)) if np.var(values) > 0 else np.zeros_like(variances),
            num_samples=neighbour_counts,
            sum_weights=diagnostics['sum_weights'],
            sum_negative_weights=diagnostics['pct_negative_weights'] * neighbour_counts / 100.0,
            min_distance=diagnostics['min_distance'],
            avg_distance=diagnostics['avg_distance'],
            nearest_sample_id=np.full_like(estimates, -1, dtype=int),
            num_duplicates_removed=np.zeros_like(estimates, dtype=int),
            search_pass=np.ones_like(estimates, dtype=int),
            search_volume=None
        )

        # Calculate stability metrics for UI display
        n_unstable = np.sum(stability_flags > 0)
        pct_unstable = 100.0 * n_unstable / m if m > 0 else 0.0

        est_grid = estimates.reshape(nx, ny, nz)
        var_grid = variances.reshape(nx, ny, nz)
        grid = pv.StructuredGrid(GX, GY, GZ)

        property_name = params.get("property_name") or f"{variable}_SK"
        variance_property = params.get("variance_property") or f"{property_name}_var"

        grid[property_name] = est_grid.ravel(order="F")
        grid[variance_property] = var_grid.ravel(order="F")
        grid["SK_NN"] = neighbour_counts.reshape(nx, ny, nz).ravel(order="F")
        grid["SK_StabilityFlag"] = stability_flags.reshape(nx, ny, nz).ravel(order="F")

        try:
            est_min = float(np.nanmin(estimates))
            est_max = float(np.nanmax(estimates))
        except ValueError:
            est_min = est_max = float("nan")
        try:
            var_min = float(np.nanmin(variances))
            var_max = float(np.nanmax(variances))
        except ValueError:
            var_min = var_max = float("nan")

        try:
            nn_min = float(np.nanmin(neighbour_counts))
            nn_max = float(np.nanmax(neighbour_counts))
            nn_mean = float(np.nanmean(neighbour_counts))
        except ValueError:
            nn_min = nn_max = nn_mean = float("nan")

        # AUDIT FIX (HIGH-002): Add dataset version tracking for audit trail
        data_source_type = getattr(df, 'attrs', {}).get('source_type', 'unknown')
        data_validation_status = getattr(df, 'attrs', {}).get('validation_status', 'NOT_RUN')
        data_lineage_timestamp = getattr(df, 'attrs', {}).get('lineage_timestamp', None)
        
        metadata = {
            "variable": variable,
            "grid_dimensions": (nx, ny, nz),
            "grid_spacing": (xinc, yinc, zinc),
            "estimates_min": est_min,
            "estimates_max": est_max,
            "variance_min": var_min,
            "variance_max": var_max,
            "neighbour_count_min": nn_min,
            "neighbour_count_max": nn_max,
            "neighbour_count_mean": nn_mean,
            # AUDIT: Dataset provenance tracking (HIGH-002)
            "data_source_type": data_source_type,
            "data_validation_status": data_validation_status,
            "data_lineage_timestamp": data_lineage_timestamp,
            "data_row_count": len(df),
            "samples_used": int(len(coords)),
            "message": f"Simple Kriging generated {len(estimates)} estimates.",
            # SK Mean Collapse Diagnostic
            "sk_mean_collapse": {
                "flag": sk_mean_collapsed,
                "collapse_ratio": float(collapse_ratio),
                "severity": collapse_severity,
                "std_estimates": float(std_estimates),
                "std_data": float(std_data),
                "threshold": float(collapse_threshold),
                "interpretation": self._get_sk_collapse_interpretation(collapse_severity),
                "recommendation": "Use Ordinary Kriging or SGSIM for production estimation" if sk_mean_collapsed else "SK provides local spatial information"
            },
            # Stability metrics
            "stability_metrics": {
                "n_unstable_blocks": int(n_unstable),
                "pct_unstable": float(pct_unstable),
                "n_high_neg_weights": int(np.sum((stability_flags & 1) > 0)),
                "n_bad_sum_weights": int(np.sum((stability_flags & 2) > 0)),
                "n_solver_issues": int(np.sum((stability_flags & 4) > 0))
            },
            # Stationarity Validation (Component 3)
            "stationarity_validation": {
                "global_mean": float(stationarity_report.global_mean),
                "mean_by_domain": {k: float(v) for k, v in stationarity_report.mean_by_domain.items()},
                "trend_x": stationarity_report.trend_x,
                "trend_y": stationarity_report.trend_y,
                "trend_z": stationarity_report.trend_z,
                "issues": stationarity_report.issues,
                "confidence_level": stationarity_report.confidence_level,
                "interpretation": self._get_stationarity_interpretation(stationarity_report)
            },
            # Support Documentation (Component 5)
            "support_documentation": self._extract_support_documentation(cleaned, xinc, yinc, zinc),
            # Enhanced Provenance (Component 7)
            "full_provenance": self._extract_full_provenance(cleaned, param_values, variable, sk_params.global_mean),
            # Domain Controls (Component 6)
            "domain_controls": self._extract_domain_controls(cleaned, params)
        }

        payload = {
            "name": "simple_kriging",
            "property_name": property_name,
            "variance_property": variance_property,
            "metadata": metadata,
            "sk_results": sk_results,  # Enhanced results with diagnostics
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", "Simple Kriging"),
                "property": property_name,
            },
        }
        
        if debugger:
            debugger.log_stage("Payload packaged successfully")
            debugger.log_stage("Simple Kriging execution COMPLETED")
            debugger.finalize()
            logger.info(f"Debug log written to: {debugger.debug_log_path}")
        
        return payload

    # =========================================================================
    # Ordinary Kriging
    # =========================================================================
    
    def _prepare_kriging_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Ordinary Kriging payload.
        
        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..models import kriging3d as krig_engine
        from ..utils.variable_utils import validate_variable
        
        # Validate required parameters - data_df MUST be provided by Controller
        if "data_df" not in params or params["data_df"] is None:
            raise ValueError(
                "Missing required parameter 'data_df' in kriging task. "
                "Controller must fetch data from DataRegistry and inject it before calling worker."
            )
        
        data_df = params["data_df"]
        
        if data_df is None:
            raise ValueError("drillhole data (data_df) is None.")
        
        # AUDIT FIX (CRITICAL-002): Validate data source at engine level
        # Defense-in-depth: even if controller passed data, ensure it's not raw assays
        try:
            data_source_result = validate_data_source(data_df, "Ordinary Kriging", strict=VARIOGRAM_STRICT_MODE)
            if not data_source_result['valid']:
                raise DataSourceError(data_source_result['warnings'][0] if data_source_result['warnings'] else "Data source validation failed")
        except DataSourceError:
            raise
        except Exception as e:
            logger.warning(f"OK data source validation skipped (non-fatal): {e}")
        if not isinstance(data_df, pd.DataFrame):
            raise ValueError(f"Expected data_df to be a DataFrame, got {type(data_df)}")
        if data_df.empty:
            raise ValueError("drillhole data (data_df) is empty.")
        
        # Validate variable parameter - MUST be provided by Controller
        variable = params.get("variable")
        if not variable:
            raise ValueError(
                "Missing required parameter 'variable' in kriging task. "
                "Controller must provide variable name before calling worker."
            )
        
        validation_result = validate_variable(variable, data_df, context="Ordinary Kriging")
        if not validation_result.is_valid:
            raise ValueError(validation_result.error_message)
        
        variable = validation_result.variable
        
        variogram_params = params.get("variogram_params")
        if not variogram_params:
            raise ValueError("Missing required parameter 'variogram_params' in kriging task.")
        
        # AUDIT GATE: Validate variogram lineage before kriging (JORC/SAMREC requirement)
        try:
            data_coords = data_df[['X', 'Y', 'Z']].dropna().to_numpy(dtype=float)
            gate_result = validate_pre_kriging(
                variogram_params=variogram_params,
                estimation_data=data_df,
                estimation_coords=data_coords,
                variable_name=variable,
                strict=VARIOGRAM_STRICT_MODE,
            )
            
            if not gate_result['passed']:
                failures = [f for f in gate_result.get('failures', []) if f.get('severity') == 'FATAL']
                if failures:
                    raise VariogramGateError(
                        f"Pre-kriging validation FAILED:\n" + 
                        "\n".join(f"- {f['check']}: {f['message']}" for f in failures)
                    )
                # Log warnings for non-fatal issues
                warnings = [f for f in gate_result.get('failures', []) if f.get('severity') != 'FATAL']
                for w in warnings:
                    logger.warning(f"Variogram gate warning: {w['check']} - {w['message']}")
        except VariogramGateError:
            raise
        except Exception as e:
            logger.warning(f"Variogram gate validation skipped: {e}")
        
        grid_spacing = params.get("grid_spacing")
        if not grid_spacing:
            raise ValueError("Missing required parameter 'grid_spacing' in kriging task.")
        
        n_neighbors = params.get("n_neighbors")
        if not n_neighbors:
            raise ValueError("Missing required parameter 'n_neighbors' in kriging task.")
        
        max_distance = params.get("max_distance")
        model_type = params.get("model_type", "spherical")

        # Filter valid data
        data = data_df.dropna(subset=['X', 'Y', 'Z', variable]).copy()
        # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
        if hasattr(data_df, 'attrs') and data_df.attrs:
            data.attrs = data_df.attrs.copy()
        if data.empty:
            raise ValueError("No samples available after filtering missing coordinates or values.")
        
        data_coords = data[['X', 'Y', 'Z']].to_numpy(dtype=float)
        data_values = data[variable].to_numpy(dtype=float)

        # Calculate percentage-based buffer (10% padding)
        data_range = data_coords.max(axis=0) - data_coords.min(axis=0)
        padding_factor = 0.1
        buffer = tuple(data_range * padding_factor)
        
        # Create estimation grid
        grid_x, grid_y, grid_z, target_coords = krig_engine.create_estimation_grid(
            data_coords,
            grid_spacing=grid_spacing,
            buffer=buffer,
            max_points=50000,
            progress_callback=progress_callback
        )

        # Extract search configuration (multi-pass or legacy single-pass)
        search_passes = params.get("search_passes")  # None for legacy mode
        compute_qa_metrics = params.get("compute_qa_metrics", True)  # Enable by default for professional standard

        # Run Ordinary Kriging
        estimates, variances, qa_metrics = krig_engine.ordinary_kriging_3d(
            data_coords,
            data_values,
            target_coords,
            variogram_params,
            n_neighbors=n_neighbors,
            max_distance=max_distance,
            model_type=model_type,
            progress_callback=progress_callback,
            search_passes=search_passes,
            compute_qa_metrics=compute_qa_metrics
        )

        # Compute variogram signature for audit trail (JORC/NI 43-101 requirement)
        from ..geostats.variogram_model import compute_variogram_signature
        variogram_signature = compute_variogram_signature(variogram_params)

        # Reshape to grid dimensions for PyVista
        nx, ny, nz = grid_x.shape[0], grid_y.shape[1], grid_z.shape[2]
        estimates_grid = estimates.reshape(nx, ny, nz, order='C')
        variances_grid = variances.reshape(nx, ny, nz, order='C')

        grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
        property_name = f'{variable}_OK_est'
        variance_property = f'{variable}_OK_var'
        grid[property_name] = estimates_grid.ravel(order='F')
        grid[variance_property] = variances_grid.ravel(order='F')

        # Add QA metrics as grid properties (professional standard)
        if qa_metrics is not None:
            for qa_name, qa_values in qa_metrics.items():
                qa_grid = qa_values.reshape(nx, ny, nz, order='C')
                qa_prop_name = f'{variable}_OK_{qa_name}'
                grid[qa_prop_name] = qa_grid.ravel(order='F')
                logger.info(f"Added QA property: {qa_prop_name}")

        # Prepare metadata
        est_min, est_max = np.nanmin(estimates), np.nanmax(estimates)
        var_min, var_max = np.nanmin(variances), np.nanmax(variances)
        
        # AUDIT FIX (HIGH-002): Add dataset version tracking for audit trail
        data_source_type = getattr(data_df, 'attrs', {}).get('source_type', 'unknown')
        data_validation_status = getattr(data_df, 'attrs', {}).get('validation_status', 'NOT_RUN')
        data_lineage_timestamp = getattr(data_df, 'attrs', {}).get('lineage_timestamp', None)
        
        # Build search strategy definition for audit trail
        search_strategy = {
            'mode': 'multi_pass' if search_passes else 'single_pass',
            'compute_qa_metrics': compute_qa_metrics,
        }
        if search_passes:
            search_strategy['passes'] = [
                {
                    'pass_number': i + 1,
                    'min_neighbors': p.get('min_neighbors', p.get('min_samples', 0)),
                    'max_neighbors': p.get('max_neighbors', p.get('n_neighbors', n_neighbors)),
                    'ellipsoid_multiplier': p.get('ellipsoid_multiplier', 1.0)
                }
                for i, p in enumerate(search_passes)
            ]
        else:
            search_strategy['n_neighbors'] = n_neighbors
            search_strategy['max_distance'] = max_distance

        import datetime
        metadata = {
            "variable": variable,
            "variogram_model": model_type,
            "variogram_params": variogram_params,
            "variogram_signature": variogram_signature,  # AUDIT: Variogram traceability
            "n_neighbors": n_neighbors,
            "max_distance": max_distance,
            "search_strategy": search_strategy,  # AUDIT: Search strategy definition
            "grid_dims": (nx, ny, nz),
            "grid_spacing": grid_spacing,
            "estimates_min": est_min,
            "estimates_max": est_max,
            "variance_min": var_min,
            "variance_max": var_max,
            # AUDIT: Dataset provenance tracking (HIGH-002)
            "data_source_type": data_source_type,
            "data_validation_status": data_validation_status,
            "data_lineage_timestamp": data_lineage_timestamp,
            "data_row_count": len(data_df),
            "samples_used": int(len(data_coords)),
            "timestamp": datetime.datetime.now().isoformat(),  # AUDIT: Estimation timestamp
            "message": f"Ordinary Kriging generated {len(estimates)} estimates.",
        }

        # Add QA metrics summary to metadata (professional standard)
        if qa_metrics is not None:
            valid_estimates = ~np.isnan(estimates)
            n_valid = int(valid_estimates.sum())
            if n_valid > 0:
                ke_valid = qa_metrics['kriging_efficiency'][valid_estimates]
                sor_valid = qa_metrics['slope_of_regression'][valid_estimates]
                neg_wt_valid = qa_metrics['pct_negative_weights'][valid_estimates]
                pass_nums = qa_metrics['pass_number'][valid_estimates]

                metadata['qa_summary'] = {
                    'kriging_efficiency_mean': float(np.nanmean(ke_valid)),
                    'kriging_efficiency_min': float(np.nanmin(ke_valid)),
                    'slope_of_regression_mean': float(np.nanmean(sor_valid)),
                    'pct_negative_weights_max': float(np.nanmax(neg_wt_valid)),
                    'pass_1_count': int(np.sum(pass_nums == 1)),
                    'pass_2_count': int(np.sum(pass_nums == 2)),
                    'pass_3_count': int(np.sum(pass_nums == 3)),
                    'unestimated_count': int(np.sum(qa_metrics['pass_number'] == 0)),
                }

        payload = {
            "name": "kriging",
            "property_name": property_name,
            "variance_property": variance_property,
            "block_values": grid[property_name],
            "variance_values": grid[variance_property],
            "metadata": metadata,
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", "Ordinary Kriging"),
                "property": property_name,
            },
        }
        return payload

    # =========================================================================
    # Universal Kriging
    # =========================================================================
    
    def _prepare_universal_kriging_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Universal Kriging payload."""
        from ..geostats.universal_kriging import run_universal_kriging_job
        
        if progress_callback:
            progress_callback(5, "Preparing Universal Kriging...")
        
        params_with_progress = params.copy()
        params_with_progress['_progress_callback'] = progress_callback
        
        result = run_universal_kriging_job(params_with_progress)
        
        # Create PyVista grid for visualization
        grid_x = result.get('grid_x')
        grid_y = result.get('grid_y')
        grid_z = result.get('grid_z')
        estimates = result.get('estimates')
        variances = result.get('variances')
        property_name = result.get('property_name', 'UK_estimate')
        variance_property = result.get('variance_property', 'UK_variance')
        
        if grid_x is not None and estimates is not None:
            grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
            grid[property_name] = estimates.ravel(order='F')
            if variances is not None:
                grid[variance_property] = variances.ravel(order='F')
            
            result['visualization'] = {
                'mesh': grid,
                'layer_name': f"UK_{params.get('variable', 'estimate')}",
                'property': property_name
            }
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "universal_kriging",
            **result
        }

    # =========================================================================
    # Co-Kriging
    # =========================================================================
    
    def _prepare_cokriging_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Co-Kriging payload."""
        from ..geostats.cokriging3d import run_cokriging_job
        
        if progress_callback:
            progress_callback(5, "Preparing Co-Kriging...")
        
        params_with_progress = params.copy()
        params_with_progress['_progress_callback'] = progress_callback
        
        result = run_cokriging_job(params_with_progress)
        
        # Create PyVista grid for visualization
        grid_x = result.get('grid_x')
        grid_y = result.get('grid_y')
        grid_z = result.get('grid_z')
        estimates = result.get('estimates')
        variances = result.get('variances')
        property_name = result.get('property_name', 'CoK_estimate')
        variance_property = result.get('variance_property', 'CoK_variance')
        
        if grid_x is not None and estimates is not None:
            grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
            grid[property_name] = estimates.ravel(order='F')
            if variances is not None:
                grid[variance_property] = variances.ravel(order='F')
            
            result['visualization'] = {
                'mesh': grid,
                'layer_name': f"CoK_{params.get('primary_name', 'estimate')}",
                'property': property_name
            }
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "cokriging",
            **result
        }

    # =========================================================================
    # Indicator Kriging
    # =========================================================================
    
    def _prepare_indicator_kriging_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Indicator Kriging payload."""
        from ..geostats.indicator_kriging import run_indicator_kriging_job
        
        if progress_callback:
            progress_callback(5, "Preparing Indicator Kriging...")
        
        params_with_progress = params.copy()
        params_with_progress['_progress_callback'] = progress_callback
        
        result = run_indicator_kriging_job(params_with_progress)
        
        # Create PyVista grid for visualization
        grid_x = result.get('grid_x')
        grid_y = result.get('grid_y')
        grid_z = result.get('grid_z')
        probabilities = result.get('probabilities')
        median = result.get('median')
        mean = result.get('mean')
        property_name = result.get('property_name', 'IK_estimate')
        
        if grid_x is not None:
            grid = pv.StructuredGrid(grid_x, grid_y, grid_z)
            
            if median is not None:
                grid[f"{property_name}_median"] = median.ravel(order='F')
                display_property = f"{property_name}_median"
            elif mean is not None:
                grid[f"{property_name}_mean"] = mean.ravel(order='F')
                display_property = f"{property_name}_mean"
            elif probabilities is not None:
                if probabilities.ndim == 4:
                    first_prob = probabilities[:, :, :, 0]
                else:
                    first_prob = probabilities[:, 0]
                grid[f"{property_name}_p0"] = first_prob.ravel(order='F')
                display_property = f"{property_name}_p0"
            else:
                display_property = None
            
            if display_property:
                result['visualization'] = {
                    'mesh': grid,
                    'layer_name': f"IK_{params.get('variable', 'estimate')}",
                    'property': display_property
                }
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "indicator_kriging",
            **result
        }

    # =========================================================================
    # Bayesian/Soft Kriging
    # =========================================================================
    
    def _prepare_bayesian_kriging_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Bayesian/Soft Kriging payload."""
        from ..geostats.bayesian_kriging import run_bayesian_kriging_job

        if progress_callback:
            progress_callback(10, "Preparing Bayesian Kriging...")

        # Extract data from data_df (like other kriging methods)
        data_df = params.get("data_df")
        if data_df is None or data_df.empty:
            raise ValueError("No drillhole data provided for Bayesian Kriging.")

        variable = params.get("variable")
        if not variable or variable not in data_df.columns:
            raise ValueError(f"Variable '{variable}' not found in data.")

        # Clean and extract coordinates and values
        cleaned = data_df.dropna(subset=["X", "Y", "Z", variable])
        # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
        if hasattr(data_df, 'attrs') and data_df.attrs:
            cleaned.attrs = data_df.attrs.copy()
        if cleaned.empty:
            raise ValueError("No valid samples available after filtering missing coordinates or values.")

        coords = cleaned[["X", "Y", "Z"]].to_numpy(dtype=np.float64)
        values = cleaned[variable].to_numpy(dtype=np.float64)

        # Create target grid (simple approach - can be enhanced)
        grid_spec = params.get("grid", (10, 10, 10))  # (nx, ny, nz)
        nx, ny, nz = int(grid_spec[0]), int(grid_spec[1]), int(grid_spec[2])

        # Simple grid generation (can be enhanced with proper bounds)
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        z_min, z_max = coords[:, 2].min(), coords[:, 2].max()

        # Add some padding
        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        z_pad = (z_max - z_min) * 0.1

        x_min -= x_pad; x_max += x_pad
        y_min -= y_pad; y_max += y_pad
        z_min -= z_pad; z_max += z_pad

        dx = (x_max - x_min) / max(nx - 1, 1)
        dy = (y_max - y_min) / max(ny - 1, 1)
        dz = (z_max - z_min) / max(nz - 1, 1)

        gx = np.linspace(x_min, x_max, nx)
        gy = np.linspace(y_min, y_max, ny)
        gz = np.linspace(z_min, z_max, nz)

        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing='ij')
        target_coords = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])

        # Map soft kriging panel parameters to bayesian kriging format
        bayesian_params = params.get('bayesian', {})
        mode = bayesian_params.get('mode', 'Mean & Variance')
        weight = bayesian_params.get('weight', 0.5)

        # Map UI mode to bayesian config
        prior_type = 'mean_var' if mode == 'Mean & Variance' else 'mean_only'

        config = {
            'base_method': 'OK',  # Default to OK for soft kriging
            'prior_type': prior_type,
            'soft_weighting': weight
        }

        # Prepare parameters for bayesian kriging job
        job_params = {
            'coords': coords,
            'values': values,
            'locations': target_coords,
            'variogram_model': params.get('variogram', {}),
            'config': config,
            'grid_info': {
                'nx': nx, 'ny': ny, 'nz': nz,
                'dx': dx, 'dy': dy, 'dz': dz,
                'x_min': x_min, 'y_min': y_min, 'z_min': z_min
            }
        }

        # Handle soft data if provided
        soft_source = params.get('soft_source')
        if soft_source == "From Block Model":
            # TODO: Implement block model soft data extraction
            pass
        elif soft_source == "From External CSV":
            soft_path = params.get('soft_path')
            if soft_path:
                try:
                    from ..geostats.soft_data import load_soft_data_from_csv
                    soft_data = load_soft_data_from_csv(soft_path)
                    job_params['soft_data'] = soft_data.to_dict()
                except Exception as e:
                    logger.warning(f"Failed to load soft data from {soft_path}: {e}")

        if progress_callback:
            progress_callback(20, "Running Bayesian Kriging...")

        result = run_bayesian_kriging_job(job_params)

        if progress_callback:
            progress_callback(100, "Complete")

        # Add grid information for visualization
        if 'error' not in result:
            result.update({
                'grid_x': GX,
                'grid_y': GY,
                'grid_z': GZ,
                'grid_info': job_params['grid_info'],
                'variable': variable,
                'property_name': f'{variable}_soft_kriging',
                'variance_property': f'{variable}_soft_kriging_var'
            })

        return {
            "name": "bayesian_kriging",
            **result
        }

    # =========================================================================
    # SGSIM
    # =========================================================================
    
    def _prepare_sgsim_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare SGSIM payload.
        
        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..models.sgsim3d import SGSIMParameters, run_full_sgsim_workflow, create_pyvista_grid
        
        data_df = params.get("data_df")
        if data_df is None or data_df.empty:
            raise ValueError(
                "No drillhole data provided for SGSIM. "
                "Controller must fetch data from DataRegistry and inject it before calling worker."
            )
        
        variable = params.get("variable")
        if not variable:
            raise ValueError("Missing required parameter 'variable' in SGSIM task.")
        
        # Filter valid data
        data = data_df.dropna(subset=['X', 'Y', 'Z', variable]).copy()
        # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
        if hasattr(data_df, 'attrs') and data_df.attrs:
            data.attrs = data_df.attrs.copy()
        if data.empty:
            raise ValueError("No samples available after filtering missing coordinates or values.")
        
        # ✅ STANDARD API: Extract to numpy IMMEDIATELY (no pandas in loops)
        # This follows the industry-standard pattern for sample-based engines
        data_coords = data[['X', 'Y', 'Z']].to_numpy(dtype=np.float64)
        data_values = data[variable].to_numpy(dtype=np.float64)
        
        # Remove NaN/inf
        valid_mask = ~(np.isnan(data_coords).any(axis=1) | np.isnan(data_values) | np.isinf(data_values))
        data_coords = data_coords[valid_mask]
        data_values = data_values[valid_mask]
        
        if len(data_values) == 0:
            raise ValueError("All data points contain NaN or inf values.")
        
        # Create SGSIM parameters
        sgsim_params = SGSIMParameters(
            nreal=params["nreal"],
            nx=params["nx"],
            ny=params["ny"],
            nz=params["nz"],
            xmin=params["xmin"],
            ymin=params["ymin"],
            zmin=params["zmin"],
            xinc=params["xinc"],
            yinc=params["yinc"],
            zinc=params["zinc"],
            variogram_type=params["variogram_type"],
            range_major=params["range_major"],
            range_minor=params["range_minor"],
            range_vert=params["range_vert"],
            azimuth=params["azimuth"],
            dip=params["dip"],
            nugget=params["nugget"],
            sill=params["sill"],
            min_neighbors=params["min_neighbors"],
            max_neighbors=params["max_neighbors"],
            max_search_radius=params["max_search_radius"],
            seed=params.get("seed"),
        )
        
        cutoffs = params.get("cutoffs", [])

        # Use _progress_callback from params (injected by JobWorker) or the function argument
        effective_progress = params.get('_progress_callback') or progress_callback

        # Run SGSIM workflow
        # CRITICAL FIX: Use named parameter for progress_callback to avoid positional argument mismatch
        results = run_full_sgsim_workflow(
            data_coords,
            data_values,
            sgsim_params,
            cutoffs=cutoffs,
            progress_callback=effective_progress  # Must use named param - position 8, not 5!
        )
        
        results['params'] = sgsim_params
        
        # Create PyVista grid for visualization (mean realization)
        summary = results.get('summary', {})
        mean_data = summary.get('mean')
        if mean_data is None:
            mean_data = np.zeros((sgsim_params.nz, sgsim_params.ny, sgsim_params.nx))
        else:
            if not isinstance(mean_data, np.ndarray):
                mean_data = np.array(mean_data)
            if mean_data.ndim == 1:
                mean_data = mean_data.reshape((sgsim_params.nz, sgsim_params.ny, sgsim_params.nx))
            elif mean_data.ndim != 3:
                try:
                    mean_data = mean_data.reshape((sgsim_params.nz, sgsim_params.ny, sgsim_params.nx))
                except ValueError:
                    logger.warning(f"Could not reshape mean data, using zeros")
                    mean_data = np.zeros((sgsim_params.nz, sgsim_params.ny, sgsim_params.nx))
        
        property_name = f"{variable}_SGSIM_mean"
        mean_grid = create_pyvista_grid(
            mean_data,
            sgsim_params,
            property_name=property_name
        )
        
        # Verify property was added to grid
        if mean_grid is None:
            raise ValueError(f"SGSIM: Failed to create PyVista grid for visualization")
        
        available_props = list(mean_grid.cell_data.keys())
        if property_name not in available_props:
            logger.warning(
                f"SGSIM: Property '{property_name}' not found in grid after creation. "
                f"Available properties: {available_props}. Adding property manually."
            )
            # Add property manually as fallback
            # Ensure data is in correct shape (nz, ny, nx) and flatten with C-order for ImageData
            if mean_data.ndim == 3:
                flat_data = mean_data.ravel(order='C')
            else:
                # Reshape to (nz, ny, nx) first
                mean_data_3d = mean_data.reshape((sgsim_params.nz, sgsim_params.ny, sgsim_params.nx))
                flat_data = mean_data_3d.ravel(order='C')
            
            # Verify size matches grid cell count
            if len(flat_data) != mean_grid.n_cells:
                logger.error(
                    f"SGSIM: Data size mismatch: grid has {mean_grid.n_cells} cells, "
                    f"but data has {len(flat_data)} values. Grid dims: {mean_grid.dimensions}"
                )
                # Try to fix by reshaping
                if mean_data.ndim == 3:
                    # Data is (nz, ny, nx), need to match grid dimensions
                    expected_size = sgsim_params.nx * sgsim_params.ny * sgsim_params.nz
                    if len(flat_data) == expected_size:
                        mean_grid.cell_data[property_name] = flat_data
                    else:
                        raise ValueError(f"Cannot add property: data size {len(flat_data)} != expected {expected_size}")
                else:
                    raise ValueError(f"Cannot add property: data size mismatch")
            else:
                mean_grid.cell_data[property_name] = flat_data
                logger.info(f"SGSIM: Manually added property '{property_name}' to grid ({len(flat_data)} values)")
        
        # Final verification
        final_props = list(mean_grid.cell_data.keys())
        logger.info(f"SGSIM: Grid created with properties: {final_props}, property_name='{property_name}'")
        
        # AUDIT FIX (HIGH-002): Add dataset version tracking for audit trail
        data_source_type = getattr(data_df, 'attrs', {}).get('source_type', 'unknown')
        data_validation_status = getattr(data_df, 'attrs', {}).get('validation_status', 'NOT_RUN')
        data_lineage_timestamp = getattr(data_df, 'attrs', {}).get('lineage_timestamp', None)
        
        metadata = {
            "variable": variable,
            "nreal": sgsim_params.nreal,
            "grid_dims": (sgsim_params.nx, sgsim_params.ny, sgsim_params.nz),
            "grid_spacing": (sgsim_params.xinc, sgsim_params.yinc, sgsim_params.zinc),
            "grid_origin": (sgsim_params.xmin, sgsim_params.ymin, sgsim_params.zmin),
            "variogram_type": sgsim_params.variogram_type,
            "range_major": sgsim_params.range_major,
            "range_minor": sgsim_params.range_minor,
            "range_vert": sgsim_params.range_vert,
            "azimuth": sgsim_params.azimuth,
            "dip": sgsim_params.dip,
            "nugget": sgsim_params.nugget,
            "sill": sgsim_params.sill,
            "min_neighbors": sgsim_params.min_neighbors,
            "max_neighbors": sgsim_params.max_neighbors,
            "max_search_radius": sgsim_params.max_search_radius,
            "samples_used": int(len(data_coords)),
            "message": f"SGSIM generated {sgsim_params.nreal} realizations.",
            # AUDIT: Dataset provenance tracking (HIGH-002)
            "data_source_type": data_source_type,
            "data_validation_status": data_validation_status,
            "data_lineage_timestamp": data_lineage_timestamp,
            "data_row_count": len(data_df),
        }
        
        payload = {
            "name": "sgsim",
            "property_name": property_name,
            "results": results,
            "metadata": metadata,
            "visualization": {
                "mesh": mean_grid,
                "layer_name": params.get("layer_name", f"SGSIM_{variable}"),
                "property": property_name,
            },
        }
        return payload

    # =========================================================================
    # SIS (Sequential Indicator Simulation)
    # =========================================================================
    
    def _prepare_sis_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Sequential Indicator Simulation (SIS) payload using standardized workflow.

        SIS is used for category-based simulation for discrete domains, facies,
        or indicator-based grade distributions. Handles non-Gaussian, multi-modal,
        and highly skewed data.

        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..models.simulation_workflow_manager import (
            execute_standardized_simulation_workflow,
            SimulationParameters
        )
        from ..models.sgsim3d import create_pyvista_grid

        data_df = params.get("data_df")
        if data_df is None or data_df.empty:
            raise ValueError(
                "No drillhole data provided for SIS. "
                "Controller must fetch data from DataRegistry and inject it before calling worker."
            )

        variable = params.get("property")
        if not variable:
            raise ValueError("Missing required parameter 'property' in SIS task.")

        # Create standardized simulation parameters
        sim_params = SimulationParameters(
            method='sis',
            n_realizations=int(params.get("n_realizations", 10)),
            nx=int(params.get("nx", 20)),
            ny=int(params.get("ny", 20)),
            nz=int(params.get("nz", 10)),
            xmin=params.get("xmin", None),  # Will be auto-calculated if None
            ymin=params.get("ymin", None),
            zmin=params.get("zmin", None),
            xinc=params.get("xinc", 10.0),
            yinc=params.get("yinc", 10.0),
            zinc=params.get("zinc", 5.0),
            variogram_type=params.get("variogram_type", "spherical"),
            range_major=params.get("range_major", 100.0),
            range_minor=params.get("range_minor", 100.0),
            range_vert=params.get("range_vert", 50.0),
            azimuth=params.get("azimuth", 0.0),
            dip=params.get("dip", 0.0),
            nugget=params.get("nugget", 0.0),
            sill=params.get("sill", 1.0),
            min_neighbors=int(params.get("min_neighbors", 4)),
            max_neighbors=int(params.get("max_neighbors", 12)),
            max_search_radius=params.get("max_search_radius", 200.0),
            method_params={
                'thresholds': params.get("thresholds"),  # Will be auto-calculated if None
                'realization_prefix': params.get("realization_prefix", "sis"),
            },
            seed=params.get("seed")
        )

        # Get cutoffs for probability mapping
        cutoffs = params.get("cutoffs", [])

        # Execute standardized workflow
        results = execute_standardized_simulation_workflow(
            data_df=data_df,
            variable=variable,
            params=sim_params,
            cutoffs=cutoffs,
            progress_callback=progress_callback
        )

        if progress_callback:
            progress_callback(95, "Building visualization...")

        # Create PyVista grid for visualization using ImageData (proper cell data handling)
        from ..models.visualization import create_block_model
        
        property_name = f"{variable}_SIS_mean"
        
        # Get mean realization data
        mean_data = results['summary']['mean']
        if mean_data.ndim == 1:
            # Reshape to (nz, ny, nx) for create_block_model
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif mean_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            # Ensure correct shape (nz, ny, nx)
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        
        # Create ImageData grid with proper cell data
        grid = create_block_model(
            values=mean_data,
            origin=(sim_params.xmin, sim_params.ymin, sim_params.zmin),
            spacing=(sim_params.xinc, sim_params.yinc, sim_params.zinc),
            dims=(sim_params.nx, sim_params.ny, sim_params.nz),
            name=property_name
        )
        
        # Add variance field to cell_data
        var_data = results['summary']['var']
        if var_data.ndim == 1:
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif var_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        else:
            var_data_3d = var_data
        grid.cell_data[f"{property_name}_var"] = var_data_3d.ravel(order='C')

        # Update metadata with SIS-specific information
        results['metadata'].update({
            "thresholds": sim_params.method_params.get('thresholds'),
            "message": f"SIS generated {sim_params.n_realizations} realizations using standardized workflow.",
        })

        payload = {
            "name": "sis",
            "property_name": property_name,
            "results": results,
            "metadata": results['metadata'],
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", f"SIS_{variable}"),
                "property": property_name,
            },
        }

        if progress_callback:
            progress_callback(100, "Complete")

        return payload

    # =========================================================================
    # IK-SGSIM
    # =========================================================================
    
    def _prepare_ik_sgsim_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare IK-SGSIM payload using standardized workflow.

        Indicator Kriging Sequential Gaussian Simulation.
        """
        from ..models.simulation_workflow_manager import (
            execute_standardized_simulation_workflow,
            SimulationParameters
        )
        from ..models.sgsim3d import create_pyvista_grid

        data_df = params.get("data_df")
        variable = params.get("variable")

        # Create standardized simulation parameters
        sim_params = SimulationParameters(
            method='ik_sgsim',
            n_realizations=int(params.get("n_realizations", 10)),
            nx=int(params.get("nx", 20)),
            ny=int(params.get("ny", 20)),
            nz=int(params.get("nz", 10)),
            xmin=params.get("xmin", None),  # Will be auto-calculated if None
            ymin=params.get("ymin", None),
            zmin=params.get("zmin", None),
            xinc=params.get("xinc", 10.0),
            yinc=params.get("yinc", 10.0),
            zinc=params.get("zinc", 5.0),
            variogram_type=params.get("variogram_type", "spherical"),
            range_major=params.get("range_major", 100.0),
            range_minor=params.get("range_minor", 100.0),
            range_vert=params.get("range_vert", 50.0),
            azimuth=params.get("azimuth", 0.0),
            dip=params.get("dip", 0.0),
            nugget=params.get("nugget", 0.0),
            sill=params.get("sill", 1.0),
            min_neighbors=int(params.get("min_neighbors", 4)),
            max_neighbors=int(params.get("max_neighbors", 12)),
            max_search_radius=params.get("max_search_radius", 200.0),
            method_params={
                'thresholds': params.get("thresholds", []),  # Indicator thresholds
                'realization_prefix': params.get("realization_prefix", "ik_sgsim"),
            },
            seed=params.get("seed")
        )

        # Get cutoffs for probability mapping
        cutoffs = params.get("cutoffs", [])

        # Execute standardized workflow
        results = execute_standardized_simulation_workflow(
            data_df=data_df,
            variable=variable,
            params=sim_params,
            cutoffs=cutoffs,
            progress_callback=progress_callback
        )

        if progress_callback:
            progress_callback(95, "Building visualization...")

        # Create PyVista grid for visualization using ImageData (proper cell data handling)
        from ..models.visualization import create_block_model
        
        property_name = f"{variable or 'IK_SGSIM'}_mean" if variable else "IK_SGSIM_mean"
        
        # Get mean realization data
        mean_data = results['summary']['mean']
        if mean_data.ndim == 1:
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif mean_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        
        # Create ImageData grid with proper cell data
        grid = create_block_model(
            values=mean_data,
            origin=(sim_params.xmin, sim_params.ymin, sim_params.zmin),
            spacing=(sim_params.xinc, sim_params.yinc, sim_params.zinc),
            dims=(sim_params.nx, sim_params.ny, sim_params.nz),
            name=property_name
        )
        
        # Add variance field to cell_data
        var_data = results['summary']['var']
        if var_data.ndim == 1:
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif var_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        else:
            var_data_3d = var_data
        grid.cell_data[f"{property_name}_var"] = var_data_3d.ravel(order='C')

        # Update metadata with IK-SGSIM-specific information
        results['metadata'].update({
            "thresholds": sim_params.method_params.get('thresholds'),
            "message": f"IK-SGSIM generated {sim_params.n_realizations} realizations using standardized workflow.",
        })

        payload = {
            "name": "ik_sgsim",
            "property_name": property_name,
            "results": results,
            "metadata": results['metadata'],
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", f"IK_SGSIM_{variable or 'sim'}"),
                "property": property_name,
            },
        }

        if progress_callback:
            progress_callback(100, "Complete")

        return payload

    # =========================================================================
    # Turning Bands Simulation
    # =========================================================================
    
    def _prepare_turning_bands_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Turning Bands Simulation payload using standardized workflow.

        Fast Gaussian random field simulation using 1D line processes.
        Very fast for large domains (>100M cells).

        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..models.simulation_workflow_manager import (
            execute_standardized_simulation_workflow,
            SimulationParameters
        )
        from ..models.sgsim3d import create_pyvista_grid

        data_df = params.get("data_df")
        variable = params.get("variable")

        # Create standardized simulation parameters
        sim_params = SimulationParameters(
            method='turning_bands',
            n_realizations=int(params.get("n_realizations", 10)),
            nx=int(params.get("nx", 20)),
            ny=int(params.get("ny", 20)),
            nz=int(params.get("nz", 10)),
            xmin=params.get("xmin", None),  # Will be auto-calculated if None
            ymin=params.get("ymin", None),
            zmin=params.get("zmin", None),
            xinc=params.get("xinc", 10.0),
            yinc=params.get("yinc", 10.0),
            zinc=params.get("zinc", 5.0),
            variogram_type=params.get("variogram_type", "spherical"),
            range_major=params.get("range_major", 100.0),
            range_minor=params.get("range_minor", 100.0),
            range_vert=params.get("range_vert", 50.0),
            azimuth=params.get("azimuth", 0.0),
            dip=params.get("dip", 0.0),
            nugget=params.get("nugget", 0.0),
            sill=params.get("sill", 1.0),
            min_neighbors=int(params.get("min_neighbors", 4)),
            max_neighbors=int(params.get("max_neighbors", 12)),
            max_search_radius=params.get("max_search_radius", 200.0),
            method_params={
                'n_bands': int(params.get("n_bands", 1000)),
                'condition': params.get("condition", True),
                'realization_prefix': params.get("realization_prefix", "tb"),
            },
            seed=params.get("seed")
        )

        # Get cutoffs for probability mapping
        cutoffs = params.get("cutoffs", [])

        # Use _progress_callback from params (injected by JobWorker) or the function argument
        effective_progress = params.get('_progress_callback') or progress_callback

        # Execute standardized workflow
        results = execute_standardized_simulation_workflow(
            data_df=data_df,
            variable=variable,
            params=sim_params,
            cutoffs=cutoffs,
            progress_callback=effective_progress
        )

        if effective_progress:
            effective_progress(95, "Building visualization...")

        # Create PyVista grid for visualization using ImageData (proper cell data handling)
        from ..models.visualization import create_block_model

        property_name = f"{variable or 'TB'}_mean" if variable else "TB_mean"
        
        # Get mean realization data
        mean_data = results['summary']['mean']
        if mean_data.ndim == 1:
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif mean_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        
        # Create ImageData grid with proper cell data
        grid = create_block_model(
            values=mean_data,
            origin=(sim_params.xmin, sim_params.ymin, sim_params.zmin),
            spacing=(sim_params.xinc, sim_params.yinc, sim_params.zinc),
            dims=(sim_params.nx, sim_params.ny, sim_params.nz),
            name=property_name
        )
        
        # Add variance field to cell_data
        var_data = results['summary']['var']
        if var_data.ndim == 1:
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif var_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        else:
            var_data_3d = var_data
        grid.cell_data[f"{property_name}_var"] = var_data_3d.ravel(order='C')

        # Update metadata with Turning Bands-specific information
        results['metadata'].update({
            "n_bands": sim_params.method_params.get('n_bands'),
            "condition": sim_params.method_params.get('condition'),
            "message": f"Turning Bands generated {sim_params.n_realizations} realizations using standardized workflow.",
        })

        payload = {
            "name": "turning_bands",
            "property_name": property_name,
            "results": results,
            "metadata": results['metadata'],
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", f"TB_{variable or 'sim'}"),
                "property": property_name,
            },
        }

        if progress_callback:
            progress_callback(100, "Complete")

        return payload

    # =========================================================================
    # Direct Block Simulation (DBS)
    # =========================================================================
    
    def _prepare_dbs_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Direct Block Simulation (DBS) payload using standardized workflow.

        Simulation directly at block support without point simulation.
        Preserves correct block-scale variability.

        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..models.simulation_workflow_manager import (
            execute_standardized_simulation_workflow,
            SimulationParameters
        )
        from ..models.sgsim3d import create_pyvista_grid

        data_df = params.get("data_df")
        variable = params.get("variable")

        # Get block dimensions (DBS uses block dimensions, not grid cell dimensions)
        block_dx = float(params.get("block_dx", 10.0))
        block_dy = float(params.get("block_dy", 10.0))
        block_dz = float(params.get("block_dz", 5.0))

        # Create standardized simulation parameters
        # For DBS, we use block dimensions as grid spacing
        sim_params = SimulationParameters(
            method='dbs',
            n_realizations=int(params.get("n_realizations", 10)),
            nx=int(params.get("nx", 20)),
            ny=int(params.get("ny", 20)),
            nz=int(params.get("nz", 10)),
            xmin=params.get("xmin", None),  # Will be auto-calculated if None
            ymin=params.get("ymin", None),
            zmin=params.get("zmin", None),
            xinc=block_dx,  # Use block dimensions as spacing
            yinc=block_dy,
            zinc=block_dz,
            variogram_type=params.get("variogram_type", "spherical"),
            range_major=params.get("range_major", 100.0),
            range_minor=params.get("range_minor", 100.0),
            range_vert=params.get("range_vert", 50.0),
            azimuth=params.get("azimuth", 0.0),
            dip=params.get("dip", 0.0),
            nugget=params.get("nugget", 0.0),
            sill=params.get("sill", 1.0),
            min_neighbors=int(params.get("min_neighbors", 4)),
            max_neighbors=int(params.get("max_neighbors", 16)),
            max_search_radius=params.get("max_search_radius", 200.0),
            method_params={
                'block_dx': block_dx,
                'block_dy': block_dy,
                'block_dz': block_dz,
                'realization_prefix': params.get("realization_prefix", "dbs"),
            },
            seed=params.get("seed")
        )

        # Get cutoffs for probability mapping
        cutoffs = params.get("cutoffs", [])

        # Use _progress_callback from params (injected by JobWorker) or the function argument
        effective_progress = params.get('_progress_callback') or progress_callback

        # Execute standardized workflow
        results = execute_standardized_simulation_workflow(
            data_df=data_df,
            variable=variable,
            params=sim_params,
            cutoffs=cutoffs,
            progress_callback=effective_progress
        )

        if effective_progress:
            effective_progress(95, "Building visualization...")

        # Create PyVista grid for visualization using ImageData (proper cell data handling)
        from ..models.visualization import create_block_model

        property_name = f"{variable or 'DBS'}_mean" if variable else "DBS_mean"
        
        # Get mean realization data
        mean_data = results['summary']['mean']
        if mean_data.ndim == 1:
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif mean_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        
        # Create ImageData grid with proper cell data
        grid = create_block_model(
            values=mean_data,
            origin=(sim_params.xmin, sim_params.ymin, sim_params.zmin),
            spacing=(sim_params.xinc, sim_params.yinc, sim_params.zinc),
            dims=(sim_params.nx, sim_params.ny, sim_params.nz),
            name=property_name
        )
        
        # Add variance field to cell_data
        var_data = results['summary']['var']
        if var_data.ndim == 1:
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif var_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        else:
            var_data_3d = var_data
        grid.cell_data[f"{property_name}_var"] = var_data_3d.ravel(order='C')

        # Update metadata with DBS-specific information
        block_variance = np.mean(results['summary']['var']) if 'var' in results['summary'] else sim_params.sill
        results['metadata'].update({
            "block_dimensions": (block_dx, block_dy, block_dz),
            "block_variance": block_variance,
            "point_variance": sim_params.sill,
            "message": f"DBS generated {sim_params.n_realizations} realizations using standardized workflow.",
        })

        payload = {
            "name": "dbs",
            "property_name": property_name,
            "results": results,
            "metadata": results['metadata'],
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", f"DBS_{variable or 'sim'}"),
                "property": property_name,
            },
        }

        if progress_callback:
            progress_callback(100, "Complete")

        return payload

    # =========================================================================
    # Gaussian Random Field (GRF)
    # =========================================================================
    
    def _prepare_grf_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Gaussian Random Field (GRF) payload using standardized workflow.

        Unconditional Gaussian simulation from variogram or spectral methods.
        Very fast for large domains using FFT.

        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..models.simulation_workflow_manager import (
            execute_standardized_simulation_workflow,
            SimulationParameters
        )
        from ..models.sgsim3d import create_pyvista_grid

        data_df = params.get("data_df")
        variable = params.get("variable")

        # Create standardized simulation parameters
        sim_params = SimulationParameters(
            method='grf',
            n_realizations=int(params.get("n_realizations", 10)),
            nx=int(params.get("nx", 50)),
            ny=int(params.get("ny", 50)),
            nz=int(params.get("nz", 20)),
            xmin=params.get("xmin", 0.0),  # GRF often starts at origin
            ymin=params.get("ymin", 0.0),
            zmin=params.get("zmin", 0.0),
            xinc=params.get("dx", 10.0),  # Use dx, dy, dz parameter names
            yinc=params.get("dy", 10.0),
            zinc=params.get("dz", 5.0),
            variogram_type=params.get("covariance_type", "spherical"),  # Map covariance to variogram
            range_major=params.get("range_x", 100.0),  # Map range parameters
            range_minor=params.get("range_y", 100.0),
            range_vert=params.get("range_z", 50.0),
            azimuth=0.0,  # GRF typically isotropic
            dip=0.0,
            nugget=params.get("nugget", 0.0),
            sill=params.get("sill", 1.0),
            min_neighbors=int(params.get("min_neighbors", 4)),
            max_neighbors=int(params.get("max_neighbors", 12)),
            max_search_radius=params.get("max_search_radius", 200.0),
            method_params={
                'covariance_type': params.get("covariance_type", "spherical"),
                'method': params.get("method", "fft"),
                'condition': params.get("condition", True),
                'matern_nu': params.get("matern_nu", 1.5),
                'realization_prefix': params.get("realization_prefix", "grf"),
            },
            seed=params.get("seed")
        )

        # Get cutoffs for probability mapping
        cutoffs = params.get("cutoffs", [])

        # Use _progress_callback from params (injected by JobWorker) or the function argument
        effective_progress = params.get('_progress_callback') or progress_callback

        # Execute standardized workflow
        results = execute_standardized_simulation_workflow(
            data_df=data_df,
            variable=variable,
            params=sim_params,
            cutoffs=cutoffs,
            progress_callback=effective_progress
        )

        if effective_progress:
            effective_progress(95, "Building visualization...")

        # Create PyVista grid for visualization using ImageData (proper cell data handling)
        from ..models.visualization import create_block_model

        property_name = f"{variable or 'GRF'}_mean" if variable else "GRF_mean"
        
        # Get mean realization data
        mean_data = results['summary']['mean']
        if mean_data.ndim == 1:
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif mean_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        
        # Create ImageData grid with proper cell data
        grid = create_block_model(
            values=mean_data,
            origin=(sim_params.xmin, sim_params.ymin, sim_params.zmin),
            spacing=(sim_params.xinc, sim_params.yinc, sim_params.zinc),
            dims=(sim_params.nx, sim_params.ny, sim_params.nz),
            name=property_name
        )
        
        # Add variance field to cell_data
        var_data = results['summary']['var']
        if var_data.ndim == 1:
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif var_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        else:
            var_data_3d = var_data
        grid.cell_data[f"{property_name}_var"] = var_data_3d.ravel(order='C')

        # Update metadata with GRF-specific information
        results['metadata'].update({
            "covariance_type": sim_params.method_params.get('covariance_type'),
            "method": sim_params.method_params.get('method'),
            "matern_nu": sim_params.method_params.get('matern_nu'),
            "conditional": sim_params.method_params.get('condition'),
            "message": f"GRF generated {sim_params.n_realizations} realizations using standardized workflow.",
        })

        payload = {
            "name": "grf",
            "property_name": property_name,
            "results": results,
            "metadata": results['metadata'],
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", f"GRF_{variable or 'sim'}"),
                "property": property_name,
            },
        }

        if progress_callback:
            progress_callback(100, "Complete")

        return payload

    # =========================================================================
    # CoSGSIM
    # =========================================================================
    
    def _prepare_cosgsim_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Co-SGSIM payload using standardized workflow.

        Co-Simulation for multiple variables.
        """
        from ..models.simulation_workflow_manager import (
            execute_standardized_simulation_workflow,
            SimulationParameters
        )
        from ..models.sgsim3d import create_pyvista_grid

        data_df = params.get("data_df")
        if data_df is None:
            raise ValueError("Missing required parameter 'data_df' for Co-SGSIM. Data should be injected by Controller.")
        
        # Co-Simulation uses 'primary_name' instead of 'variable'
        variable = params.get("variable") or params.get("primary_name")
        if not variable:
            raise ValueError("Missing required parameter 'variable' or 'primary_name' for Co-SGSIM.")

        # Create standardized simulation parameters
        sim_params = SimulationParameters(
            method='cosgsim',
            n_realizations=int(params.get("n_realizations", 10)),
            nx=int(params.get("nx", 20)),
            ny=int(params.get("ny", 20)),
            nz=int(params.get("nz", 10)),
            xmin=params.get("xmin", None),  # Will be auto-calculated if None
            ymin=params.get("ymin", None),
            zmin=params.get("zmin", None),
            xinc=params.get("xinc", 10.0),
            yinc=params.get("yinc", 10.0),
            zinc=params.get("zinc", 5.0),
            variogram_type=params.get("variogram_type", "spherical"),
            range_major=params.get("range_major", 100.0),
            range_minor=params.get("range_minor", 100.0),
            range_vert=params.get("range_vert", 50.0),
            azimuth=params.get("azimuth", 0.0),
            dip=params.get("dip", 0.0),
            nugget=params.get("nugget", 0.0),
            sill=params.get("sill", 1.0),
            min_neighbors=int(params.get("min_neighbors", 4)),
            max_neighbors=int(params.get("max_neighbors", 12)),
            max_search_radius=params.get("max_search_radius", 200.0),
            method_params={
                'secondary_variable': params.get("secondary_variable"),
                'cross_variogram_type': params.get("cross_variogram_type", "spherical"),
                'cross_range_major': params.get("cross_range_major", 100.0),
                'cross_range_minor': params.get("cross_range_minor", 100.0),
                'cross_range_vert': params.get("cross_range_vert", 50.0),
                'cross_azimuth': params.get("cross_azimuth", 0.0),
                'cross_dip': params.get("cross_dip", 0.0),
                'cross_nugget': params.get("cross_nugget", 0.0),
                'cross_sill': params.get("cross_sill", 0.8),
                'realization_prefix': params.get("realization_prefix", "cosgsim"),
            },
            seed=params.get("seed")
        )

        # Get cutoffs for probability mapping
        cutoffs = params.get("cutoffs", [])

        # Use _progress_callback from params (injected by JobWorker) or the function argument
        effective_progress = params.get('_progress_callback') or progress_callback

        # Execute standardized workflow
        results = execute_standardized_simulation_workflow(
            data_df=data_df,
            variable=variable,
            params=sim_params,
            cutoffs=cutoffs,
            progress_callback=effective_progress
        )

        if effective_progress:
            effective_progress(95, "Building visualization...")

        # Create PyVista grid for visualization using ImageData (proper cell data handling)
        from ..models.visualization import create_block_model

        property_name = f"{variable or 'CoSGSIM'}_mean" if variable else "CoSGSIM_mean"
        
        # Get mean realization data
        mean_data = results['summary']['mean']
        if mean_data.ndim == 1:
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif mean_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            mean_data = mean_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        
        # Create ImageData grid with proper cell data
        grid = create_block_model(
            values=mean_data,
            origin=(sim_params.xmin, sim_params.ymin, sim_params.zmin),
            spacing=(sim_params.xinc, sim_params.yinc, sim_params.zinc),
            dims=(sim_params.nx, sim_params.ny, sim_params.nz),
            name=property_name
        )
        
        # Add variance field to cell_data
        var_data = results['summary']['var']
        if var_data.ndim == 1:
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        elif var_data.shape != (sim_params.nz, sim_params.ny, sim_params.nx):
            var_data_3d = var_data.reshape((sim_params.nz, sim_params.ny, sim_params.nx))
        else:
            var_data_3d = var_data
        grid.cell_data[f"{property_name}_var"] = var_data_3d.ravel(order='C')

        # Update metadata with Co-Simulation-specific information
        results['metadata'].update({
            "secondary_variable": sim_params.method_params.get('secondary_variable'),
            "cross_variogram_type": sim_params.method_params.get('cross_variogram_type'),
            "cross_range_major": sim_params.method_params.get('cross_range_major'),
            "cross_sill": sim_params.method_params.get('cross_sill'),
            "message": f"Co-SGSIM generated {sim_params.n_realizations} realizations using standardized workflow.",
        })

        payload = {
            "name": "cosgsim",
            "property_name": property_name,
            "results": results,
            "metadata": results['metadata'],
            "visualization": {
                "mesh": grid,
                "layer_name": params.get("layer_name", f"CoSGSIM_{variable or 'sim'}"),
                "property": property_name,
            },
        }

        if progress_callback:
            progress_callback(100, "Complete")

        return payload

    # =========================================================================
    # Variogram Analysis
    # =========================================================================
    
    def _prepare_variogram_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Variogram payload.
        
        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..models import variogram3d as vg3d
        
        data_df = params.get("data_df")
        if data_df is None or data_df.empty:
            raise ValueError(
                "No drillhole data provided for Variogram analysis. "
                "Controller must fetch data from DataRegistry and inject it before calling worker."
            )
        
        variable = params.get("variable")
        if not variable:
            raise ValueError("Missing required parameter 'variable' in Variogram task.")
        nlag = params["nlag"]
        lag_distance = params["lag_distance"]
        lag_tolerance = params["lag_tolerance"]
        model_type = params["model_type"]
        z_positive_up = params.get("z_positive_up", True)
        
        # Optional hole id column
        hole_col = None
        for cand in ["HOLEID", "HOLE_ID", "BHID"]:
            if cand in data_df.columns:
                hole_col = cand
                break
        
        # Depth columns for along-hole distances
        from_col = None
        to_col = None
        for cand in ["FROM", "FROM_", "START_DEPTH"]:
            if cand in data_df.columns:
                from_col = cand
                break
        for cand in ["TO", "TO_", "END_DEPTH"]:
            if cand in data_df.columns:
                to_col = cand
                break

        # Survey orientation defaults
        az_dip = {}
        if "AZIMUTH" in data_df.columns and "DIP" in data_df.columns:
            try:
                az_dip["azimuth"] = float(data_df["AZIMUTH"].dropna().mean())
                az_dip["dip"] = float(data_df["DIP"].dropna().mean())
            except Exception:
                pass
        
        # Filter valid data
        data = data_df.dropna(subset=['X', 'Y', 'Z', variable]).copy()
        # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
        if hasattr(data_df, 'attrs') and data_df.attrs:
            data.attrs = data_df.attrs.copy()
        if data.empty:
            raise ValueError("No samples available after filtering missing coordinates or values.")
        
        def progress_wrapper(step, message):
            if progress_callback:
                percentage = min(step * 20, 100) if step < 5 else 100
                progress_callback(percentage, message or "Computing variogram...")
        
        # Run variogram pipeline
        variogram_results = vg3d.run_variogram_pipeline(
            data,
            xcol="X", ycol="Y", zcol="Z", vcol=variable,
            hole_id_col=hole_col,
            from_col=from_col,
            to_col=to_col,
            default_azimuth=az_dip.get("azimuth"),
            default_dip=az_dip.get("dip"),
            z_positive_up=z_positive_up,
            nlag=nlag,
            lag_distance=lag_distance,
            lag_tolerance=lag_tolerance,
            model_types=[model_type],
            use_sill_norm=False,
            # DETERMINISM: Explicit seed for reproducibility
            random_state=42
        )
        
        metadata = {
            "variable": variable,
            "nlag": nlag,
            "lag_distance": lag_distance,
            "lag_tolerance": lag_tolerance,
            "model_type": model_type,
            "samples_used": int(len(data)),
            "message": "Variogram analysis completed successfully.",
        }
        
        payload = {
            "name": "variogram",
            "variogram_results": variogram_results,
            "metadata": metadata,
        }
        return payload

    # =========================================================================
    # Variogram Assistant
    # =========================================================================
    
    def _prepare_variogram_assistant_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Variogram Assistant payload."""
        from ..geostats.variogram_assistant import run_variogram_assistant_job
        
        if progress_callback:
            progress_callback(10, "Running Variogram Assistant...")
        
        result = run_variogram_assistant_job(params)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "variogram_assistant",
            **result
        }

    # =========================================================================
    # Uncertainty Analysis
    # =========================================================================
    
    def _prepare_uncertainty_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Uncertainty analysis payload."""
        analysis_type = params.get("analysis_type", "monte_carlo")
        block_model = params.get("block_model")
        
        if block_model is None or block_model.empty:
            raise ValueError("No block model provided for uncertainty analysis.")
        
        try:
            from ..uncertainty_engine import (
                MonteCarloSimulator, MonteCarloConfig, SimulationMode,
                ParameterDistribution, DistributionType,
                BootstrapAnalyzer, BootstrapConfig,
                LHSSampler, LHSConfig,
            )
        except ImportError:
            raise NotImplementedError("Uncertainty engine not available.")
        
        results = None
        
        if analysis_type == 'monte_carlo':
            mc_config = MonteCarloConfig(
                n_simulations=params['n_simulations'],
                mode=SimulationMode(params['mode']),
                random_seed=params.get('random_seed'),
                parallel=params.get('parallel', True)
            )
            
            for param_name, param_spec in params.get('parameters', {}).items():
                dist = ParameterDistribution(
                    name=param_name,
                    distribution=DistributionType(param_spec['distribution']),
                    base_value=param_spec['base_value'],
                    std_dev=param_spec.get('std_dev'),
                    min_value=param_spec.get('min_value'),
                    max_value=param_spec.get('max_value'),
                    mode_value=param_spec.get('mode_value')
                )
                mc_config.parameters[param_name] = dist
            
            simulator = MonteCarloSimulator(mc_config)
            
            def progress_wrapper(current, total):
                if progress_callback:
                    percentage = int((current / total) * 100) if total > 0 else 0
                    progress_callback(percentage, f"Running simulation {current}/{total}")
            
            results = simulator.run(block_model, progress_callback=progress_wrapper)
            
        elif analysis_type == 'bootstrap':
            config = BootstrapConfig(
                n_iterations=params['n_iterations'],
                confidence_level=params['confidence_level'],
                random_seed=params.get('random_seed')
            )
            
            analyzer = BootstrapAnalyzer(config)
            results = {}
            columns_to_analyze = params.get('columns', [])
            
            for col in columns_to_analyze:
                if col not in block_model.columns:
                    continue
                data = block_model[col].dropna().values
                result = analyzer.analyze_statistic(data, np.mean, f"{col}_mean")
                results[col] = result
                
        elif analysis_type == 'lhs':
            param_specs = params.get('parameters', {})
            config = LHSConfig(
                n_samples=params['n_samples'],
                n_dimensions=len(param_specs),
                random_seed=params.get('random_seed')
            )
            
            sampler = LHSSampler(config)
            results = sampler.sample_to_dataframe(param_specs)
            
        elif analysis_type == 'prob_shells':
            raise NotImplementedError("Probabilistic shells requires pit optimizer integration")
        else:
            raise ValueError(f"Unknown uncertainty analysis type: {analysis_type}")
        
        metadata = {
            "analysis_type": analysis_type,
            "blocks_analyzed": int(len(block_model)),
            "message": f"{analysis_type.replace('_', ' ').title()} analysis completed successfully.",
        }
        
        payload = {
            "name": "uncertainty",
            "analysis_type": analysis_type,
            "results": results,
            "metadata": metadata,
        }
        return payload

    # =========================================================================
    # Economic Uncertainty
    # =========================================================================
    
    def _prepare_economic_uncertainty_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Economic Uncertainty payload."""
        from ..uncertainty_engine.economic_propagation import run_economic_uncertainty_job
        
        if progress_callback:
            progress_callback(10, "Running Economic Uncertainty Propagation...")
        
        result = run_economic_uncertainty_job(params)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "economic_uncert",
            **result
        }

    # =========================================================================
    # Grade Statistics
    # =========================================================================
    
    def _prepare_grade_stats_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Grade Statistics payload."""
        assay_df = params["assay_df"]
        hole_id = params["hole_id"]
        property_name = params["property_name"]
        domain_filter = params["domain_filter"]
        
        hole_data = assay_df[assay_df['HOLEID'] == hole_id].copy()
        
        if property_name not in hole_data.columns:
            raise ValueError(f"Property '{property_name}' not found in data.")
        
        # Apply domain filter if not "All Domains"
        if domain_filter != "All Domains":
            for col in hole_data.columns:
                if hole_data[col].dtype == 'object' and col not in ['HOLEID']:
                    hole_data = hole_data[hole_data[col] == domain_filter]
                    break
        
        if hole_data.empty:
            raise ValueError("No data available for selected filters.")
        
        metadata = {
            "hole_id": hole_id,
            "property_name": property_name,
            "domain_filter": domain_filter,
            "samples_count": int(len(hole_data)),
            "message": f"Grade statistics computed for {hole_id}, property {property_name}.",
        }
        
        payload = {
            "name": "grade_stats",
            "hole_id": hole_id,
            "property_name": property_name,
            "filtered_data": hole_data,
            "metadata": metadata,
        }
        return payload

    # =========================================================================
    # Grade Transformation
    # =========================================================================
    
    def _prepare_grade_transform_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Grade Transformation payload."""
        original_data = params.get("original_data")
        transformed_data = params.get("transformed_data")
        col_name = params["column_name"]
        transform_type = params["transform_type"]
        add_const = params["add_const"]
        const = params["const"]
        power = params["power"]
        boxcox_lambda = params["boxcox_lambda"]
        existing_transformations = params.get("existing_transformations", {})
        
        if transformed_data is None:
            raise ValueError("No drillhole data provided.")
        
        # Handle "None" transformation (remove existing)
        if "None" in transform_type:
            transformations = existing_transformations.copy()
            if col_name in transformations:
                del transformations[col_name]
                if original_data is not None:
                    transformed_data = transformed_data.copy()
                    transformed_data[col_name] = original_data[col_name].copy()
            
            return {
                "name": "grade_transform",
                "transformed_data": transformed_data,
                "transformations": transformations,
                "column_name": col_name,
                "transform_type": transform_type,
                "message": f"Transformation removed from '{col_name}'.",
            }
        
        # Get original values
        if original_data is not None:
            values = original_data[col_name].values
        else:
            values = transformed_data[col_name].values
        
        # Auto-calculate Box-Cox lambda if needed
        if "Box-Cox" in transform_type and boxcox_lambda == 0.0:
            try:
                from scipy import stats
                valid = values[~np.isnan(values)]
                if add_const:
                    valid = valid + const
                valid = valid[valid > 0]
                
                if len(valid) > 10:
                    _, calculated_lambda = stats.boxcox(valid)
                    if np.isscalar(calculated_lambda):
                        boxcox_lambda = float(calculated_lambda)
                    else:
                        boxcox_lambda = float(np.mean(calculated_lambda)) if calculated_lambda.size > 0 else 0.0
            except Exception:
                boxcox_lambda = 0.0
        
        # Apply transformation
        transformed = self._apply_transformation_to_values(
            values, transform_type, add_const, const, power, boxcox_lambda
        )
        
        # Create transformed column name
        transform_suffix = self._get_transform_suffix(transform_type, boxcox_lambda, power)
        transformed_col_name = f"{col_name}_{transform_suffix}"
        
        # Store transformed column
        transformed_data = transformed_data.copy()
        transformed_data[transformed_col_name] = transformed
        
        # Store transformation info
        params_list = []
        if add_const and "Normal Score" not in transform_type:
            params_list.append(f"const={const:.4f}")
        if "Box-Cox" in transform_type:
            params_list.append(f"λ={boxcox_lambda:.4f}")
        if "Power" in transform_type:
            params_list.append(f"power={power:.4f}")
        
        param_str = ", ".join(params_list) if params_list else ""
        transform_info = {
            'transform_type': transform_type,
            'transformed_col_name': transformed_col_name,
            'original_col_name': col_name,
            'params': param_str,
            'add_const': add_const,
            'const': const if add_const else 0.0,
            'power': power if "Power" in transform_type else None,
            'boxcox_lambda': boxcox_lambda if "Box-Cox" in transform_type else None,
        }
        
        transformations = existing_transformations.copy()
        transformations[col_name] = transform_info
        transformations[transformed_col_name] = transform_info
        
        metadata = {
            "column_name": col_name,
            "transform_type": transform_type,
            "transformed_col_name": transformed_col_name,
            "samples_count": int(len(values[~np.isnan(values)])),
            "message": f"Applied '{transform_type}' to column '{col_name}'.",
        }
        
        payload = {
            "name": "grade_transform",
            "transformed_data": transformed_data,
            "transformations": transformations,
            "transform_info": transform_info,
            "column_name": col_name,
            "transform_type": transform_type,
            "original_values": values.copy(),
            "transformed_values": transformed.copy(),
            "metadata": metadata,
        }
        return payload
    
    def _apply_transformation_to_values(self, values: np.ndarray, transform_type: str,
                                       add_const: bool, const: float, power: float, boxcox_lambda: float) -> np.ndarray:
        """Apply transformation to values (helper method)."""
        try:
            from scipy import stats
            SCIPY_AVAILABLE = True
        except ImportError:
            SCIPY_AVAILABLE = False
        
        values = np.asarray(values, dtype=float)
        mask = ~np.isnan(values)
        
        if not mask.any():
            return values
        
        valid_values = values[mask].copy()
        
        # Normal Score transformation
        if "Normal Score" in transform_type or "Gaussian" in transform_type:
            if SCIPY_AVAILABLE:
                ranks = stats.rankdata(valid_values, method='average')
                percentiles = (ranks - 0.5) / len(valid_values)
                transformed = stats.norm.ppf(percentiles)
                transformed = np.where(np.isfinite(transformed), transformed, np.nan)
            else:
                return values
        else:
            if add_const:
                valid_values = valid_values + const
            
            if "Log10" in transform_type or "log10" in transform_type.lower():
                valid_for_log = np.maximum(valid_values, np.finfo(float).eps)
                transformed = np.log10(valid_for_log)
            elif "Log" in transform_type and "10" not in transform_type:
                valid_for_log = np.maximum(valid_values, np.finfo(float).eps)
                transformed = np.log(valid_for_log)
            elif "Square Root" in transform_type or "sqrt" in transform_type.lower():
                valid_for_sqrt = np.maximum(valid_values, 0.0)
                transformed = np.sqrt(valid_for_sqrt)
            elif "Box-Cox" in transform_type:
                if SCIPY_AVAILABLE:
                    valid_for_boxcox = np.maximum(valid_values, np.finfo(float).eps)
                    if boxcox_lambda == 0:
                        transformed = np.log(valid_for_boxcox)
                    else:
                        transformed = (valid_for_boxcox ** boxcox_lambda - 1) / boxcox_lambda
                else:
                    transformed = valid_values
            elif "Power" in transform_type:
                if power < 0 or (0 < power < 1):
                    valid_for_power = np.maximum(valid_values, 0.0)
                    transformed = valid_for_power ** power
                else:
                    transformed = valid_values ** power
            else:
                transformed = valid_values
        
        result = np.full_like(values, np.nan)
        result[mask] = transformed
        return result
    
    def _get_transform_suffix(self, transform_type: str, boxcox_lambda: float = 0.0, power: float = 0.5) -> str:
        """Get suffix for transformed column name."""
        if "Normal Score" in transform_type or "Gaussian" in transform_type:
            return "NormalScore"
        elif "Log10" in transform_type or "log10" in transform_type.lower():
            return "Log10"
        elif "Log" in transform_type:
            return "Log"
        elif "Square Root" in transform_type or "sqrt" in transform_type.lower():
            return "Sqrt"
        elif "Box-Cox" in transform_type:
            return f"BoxCox{boxcox_lambda:.3f}".replace('.', 'p').replace('-', 'm')
        elif "Power" in transform_type:
            return f"Power{power:.3f}".replace('.', 'p').replace('-', 'm')
        else:
            return "Transformed"

    def _get_sk_collapse_interpretation(self, severity: str) -> str:
        """Return professional interpretation of SK collapse severity."""
        interpretations = {
            "SEVERE": (
                "Simple Kriging estimates show strong reversion to global mean, "
                "indicating weak spatial continuity relative to nugget effect. "
                "This confirms that local data does not add significant information "
                "over the global mean."
            ),
            "MODERATE": (
                "Simple Kriging estimates show moderate reversion to global mean, "
                "suggesting spatial structure is present but relatively weak. "
                "Ordinary Kriging with local mean adaptation is recommended."
            ),
            "NONE": (
                "Simple Kriging estimates show sufficient local variation, "
                "indicating spatial structure effectively captures local trends."
            )
        }
        return interpretations.get(severity, "")

    def _get_stationarity_interpretation(self, report) -> str:
        """Return professional interpretation of stationarity validation."""
        from ..geostats.sk_stationarity import StationarityReport

        if report.confidence_level == 'high':
            return (
                "Stationarity assumption is well-supported. The global mean is "
                "appropriate for Simple Kriging across the domain."
            )
        elif report.confidence_level == 'medium':
            return (
                "Stationarity assumption is acceptable with minor concerns. "
                "Simple Kriging results should be reviewed carefully. Consider "
                "subdividing the domain if significant trends are present."
            )
        else:  # low
            return (
                "Stationarity assumption is questionable due to significant spatial "
                "trends or domain-specific mean differences. Ordinary Kriging with "
                "local mean adaptation or domain-based estimation is recommended."
            )

    def _extract_support_documentation(
        self, df, block_x: float, block_y: float, block_z: float
    ) -> Dict[str, Any]:
        """
        Extract support documentation from data provenance (Component 5).

        Extracts composite support information and documents change-of-support
        for JORC/NI 43-101 compliance.
        """
        # Extract composite support from provenance chain
        composite_support = None
        composite_method = "Unknown"

        if hasattr(df, 'attrs') and 'provenance' in df.attrs:
            from ..core.data_provenance import DataProvenance

            provenance = df.attrs['provenance']
            if isinstance(provenance, DataProvenance):
                # Search transformation chain for compositing step
                for step in provenance.transformation_chain:
                    if step.transformation_type.upper() == 'COMPOSITING':
                        composite_support = step.parameters.get('composite_length')
                        composite_method = step.parameters.get('method', 'Unknown')
                        if composite_support:
                            break

        # Calculate support ratios if composite support is known
        support_ratio_x = block_x / composite_support if composite_support and composite_support > 0 else None
        support_ratio_y = block_y / composite_support if composite_support and composite_support > 0 else None
        support_ratio_z = block_z / composite_support if composite_support and composite_support > 0 else None

        # Create change-of-support statement
        if composite_support:
            cos_statement = (
                f"Variogram fitted at {composite_support:.2f}m composite support ({composite_method} method). "
                f"Block model estimated at {block_x:.2f}×{block_y:.2f}×{block_z:.2f}m support. "
                f"Support ratios: X={support_ratio_x:.2f}, Y={support_ratio_y:.2f}, Z={support_ratio_z:.2f}. "
                "No block-to-point correction applied (standard SK practice for in-situ grade estimation)."
            )
            documentation_status = "COMPLETE"
        else:
            cos_statement = (
                "Composite support information not available in data provenance. "
                "Data source validation recommended. "
                f"Block model estimated at {block_x:.2f}×{block_y:.2f}×{block_z:.2f}m support."
            )
            documentation_status = "INCOMPLETE"

        return {
            "composite_support_m": float(composite_support) if composite_support else None,
            "composite_method": composite_method,
            "block_support_x_m": float(block_x),
            "block_support_y_m": float(block_y),
            "block_support_z_m": float(block_z),
            "block_volume_m3": float(block_x * block_y * block_z),
            "support_ratio": {
                "x": float(support_ratio_x) if support_ratio_x else None,
                "y": float(support_ratio_y) if support_ratio_y else None,
                "z": float(support_ratio_z) if support_ratio_z else None
            },
            "change_of_support_statement": cos_statement,
            "documentation_status": documentation_status,
            "professional_note": (
                "Block support is larger than point support, resulting in "
                "smoothing of block estimates relative to sample values. "
                "This is appropriate for resource estimation and mine planning."
                if support_ratio_x and max(support_ratio_x, support_ratio_y, support_ratio_z) > 1.0
                else "Block and sample supports are similar."
            )
        }

    def _extract_full_provenance(
        self, df, param_values: Dict, variable: str, global_mean: float
    ) -> Dict[str, Any]:
        """
        Extract comprehensive provenance chain for full reproducibility (Component 7).

        Captures all information needed to exactly reproduce the SK estimation
        for JORC/NI 43-101 compliance and stock exchange reporting.
        """
        import platform
        import hashlib
        from datetime import datetime

        # 1. Data Provenance
        data_provenance = {}
        if hasattr(df, 'attrs'):
            if 'provenance' in df.attrs:
                from ..core.data_provenance import DataProvenance

                prov = df.attrs['provenance']
                if isinstance(prov, DataProvenance):
                    data_provenance = {
                        'source_type': prov.source_type.value if hasattr(prov.source_type, 'value') else str(prov.source_type),
                        'source_file': prov.source_file,
                        'created_at': prov.created_at.isoformat() if hasattr(prov.created_at, 'isoformat') else str(prov.created_at),
                        'transformation_chain': [
                            {
                                'type': step.transformation_type,
                                'source_panel': step.source_panel,
                                'timestamp': step.timestamp.isoformat() if hasattr(step.timestamp, 'isoformat') else str(step.timestamp),
                                'parameters': step.parameters,
                                'description': step.description,
                                'row_count_before': step.row_count_before,
                                'row_count_after': step.row_count_after
                            }
                            for step in prov.transformation_chain
                        ]
                    }

            # Data hash for reproducibility
            if 'data_hash' in df.attrs:
                data_provenance['data_hash'] = df.attrs['data_hash']
            else:
                # Compute data hash if not present
                try:
                    data_str = f"{variable}_{len(df)}_{df[variable].sum():.6f}_{df[variable].mean():.6f}"
                    data_hash = hashlib.md5(data_str.encode()).hexdigest()
                    data_provenance['data_hash'] = data_hash
                except:
                    data_provenance['data_hash'] = None

        # 2. Variogram Signature
        variogram_signature = f"{param_values.get('variogram_type', 'spherical')}_" \
                             f"s{param_values.get('sill', 1.0):.3f}_" \
                             f"n{param_values.get('nugget', 0.0):.3f}_" \
                             f"r{param_values.get('range_major', 100.0):.1f}_" \
                             f"az{param_values.get('azimuth', 0.0):.0f}_" \
                             f"dip{param_values.get('dip', 0.0):.0f}"

        # 3. Software Information
        software_info = {
            'software': 'GeoX Block Model Viewer',
            'version': '1.0.0',  # TODO: Import from __version__
            'module': 'Simple Kriging',
            'platform': platform.system(),
            'platform_release': platform.release(),
            'python_version': platform.python_version(),
            'timestamp': datetime.now().isoformat(),
            'user': os.getenv('USERNAME', os.getenv('USER', 'unknown'))
        }

        # 4. SK-specific parameters
        sk_parameters = {
            'global_mean': float(global_mean),
            'mean_type': 'user_specified',  # or 'auto_calculated'
            'variogram_signature': variogram_signature,
            'search_strategy': {
                'max_neighbours': int(param_values.get('ndmax', 12)),
                'min_neighbours': int(param_values.get('nmin', 1)),
                'search_radius': float(param_values.get('max_search_radius', 200.0)),
                'sectoring': param_values.get('sectoring', 'No sectoring')
            }
        }

        # 5. Quality Gates Applied
        quality_gates = {
            'variogram_validation': 'PASS',  # Assuming gates passed if we got here
            'data_source_validation': 'PASS',
            'pre_kriging_checks': 'PASS'
        }

        # 6. Reproducibility Statement
        reproducibility_statement = (
            f"SK estimation fully reproducible using:\n"
            f"• Data: {data_provenance.get('source_file', 'N/A')} "
            f"(hash: {data_provenance.get('data_hash', 'N/A')[:8]}...)\n"
            f"• Variable: {variable}\n"
            f"• Global Mean: {global_mean:.6f}\n"
            f"• Variogram: {variogram_signature}\n"
            f"• Software: {software_info['software']} v{software_info['version']}\n"
            f"• Timestamp: {software_info['timestamp']}\n"
            f"• Platform: {software_info['platform']} {software_info['platform_release']}"
        )

        return {
            'data_provenance': data_provenance,
            'variogram_signature': variogram_signature,
            'sk_parameters': sk_parameters,
            'software_info': software_info,
            'quality_gates': quality_gates,
            'reproducibility_statement': reproducibility_statement,
            'compliance_note': (
                "This provenance record provides full reproducibility for "
                "JORC/NI 43-101 compliance and stock exchange reporting. "
                "All data transformations, parameters, and quality checks are documented."
            )
        }

    def _extract_domain_controls(self, df, params: Dict) -> Dict[str, Any]:
        """
        Extract domain control information for documentation (Component 6).

        Documents whether domain-based estimation was used and provides
        information for CP review regarding geological boundary enforcement.
        """
        # Check if domain enforcement was requested
        enforce_domains = params.get("enforce_domains", False)
        domain_column = params.get("domain_column", None)

        # Check if domain column exists in data
        domain_info = {
            "domain_enforcement_enabled": False,
            "domain_column": None,
            "domains_present": [],
            "n_domains": 0,
            "contact_handling": "no_boundaries",
            "estimation_strategy": "global"
        }

        if domain_column and domain_column in df.columns:
            unique_domains = df[domain_column].unique()
            domain_info["domain_column"] = domain_column
            domain_info["domains_present"] = [str(d) for d in unique_domains]
            domain_info["n_domains"] = len(unique_domains)

            if enforce_domains:
                domain_info["domain_enforcement_enabled"] = True
                domain_info["contact_handling"] = "hard_boundaries"
                domain_info["estimation_strategy"] = "domain_based"
                domain_statement = (
                    f"SK run separately within {len(unique_domains)} geological domains "
                    f"({', '.join([str(d) for d in unique_domains][:5])}) with hard boundaries. "
                    "No estimation across domain contacts to preserve geological interpretation."
                )
            else:
                domain_statement = (
                    f"Data contains {len(unique_domains)} domains, but SK run globally without "
                    "domain boundaries. Consider domain-based estimation if domains have "
                    "significantly different mean grades."
                )
        else:
            domain_statement = (
                "No domain column specified or available. SK run globally across all data. "
                "For production estimation, consider geological domain stratification if "
                "significant lithological or grade zonation exists."
            )

        domain_info["domain_statement"] = domain_statement
        domain_info["professional_note"] = (
            "Domain controls ensure geological boundaries are respected during estimation. "
            "Hard domain boundaries prevent smoothing across geological contacts, which is "
            "critical for JORC/NI 43-101 compliance when domains have distinct grade populations."
        )

        return domain_info

    # =========================================================================
    # Swath Analysis
    # =========================================================================
    
    def _prepare_swath_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Swath Analysis payload."""
        block_model = params.get("block_model")
        drillhole_df = params.get("drillhole_df")
        data_mode = params.get("data_mode", "block_model")
        direction = params["direction"]
        property_name = params["property_name"]
        n_bins = params["n_bins"]
        
        axis_map = {"X": 0, "Y": 1, "Z": 2}
        axis_col_name = f"{direction}C"
        axis_idx = axis_map[direction]
        
        logger.info(f"SwathPayload: data_mode={data_mode}, has_block_model={block_model is not None}, has_drillhole={drillhole_df is not None}")
        
        # Handle drillhole data mode
        if data_mode == "drillhole":
            if drillhole_df is None:
                raise ValueError("Drillhole data mode selected but no drillhole data provided.")
            
            # For drillhole data, use MID_X, MID_Y, MID_Z or X, Y, Z columns
            coord_col_map = {
                "X": ["MID_X", "X"],
                "Y": ["MID_Y", "Y"],
                "Z": ["MID_Z", "Z"]
            }
            
            # Find the appropriate coordinate column
            coord_col = None
            for col_name in coord_col_map[direction]:
                if col_name in drillhole_df.columns:
                    coord_col = col_name
                    break
            
            if coord_col is None:
                raise ValueError(f"Could not find coordinate column for {direction} axis in drillhole data.")
            
            if property_name not in drillhole_df.columns:
                raise ValueError(f"Property '{property_name}' not found in drillhole data.")
            
            # Create temporary dataframe with coordinate and property
            axis_coords = drillhole_df[coord_col].values
            prop_data = drillhole_df[property_name].values
            
            df_temp = pd.DataFrame({
                axis_col_name: axis_coords,
                property_name: prop_data
            })
        
        # Handle block model mode
        elif data_mode == "block_model":
            if block_model is None:
                raise ValueError("Block model mode selected but no block model provided.")
            
            # Handle DataFrame block models
            if isinstance(block_model, pd.DataFrame):
                # Extract coordinate column
                coord_col_map = {
                    "X": ["XC", "X"],
                    "Y": ["YC", "Y"],
                    "Z": ["ZC", "Z"]
                }
                
                coord_col = None
                for col_name in coord_col_map[direction]:
                    if col_name in block_model.columns:
                        coord_col = col_name
                        break
                
                if coord_col is None:
                    raise ValueError(f"Could not find coordinate column for {direction} axis in block model DataFrame.")
                
                if property_name not in block_model.columns:
                    raise ValueError(f"Property '{property_name}' not found in block model DataFrame.")
                
                axis_coords = block_model[coord_col].values
                prop_data = block_model[property_name].values
                
                df_temp = pd.DataFrame({
                    axis_col_name: axis_coords,
                    property_name: prop_data
                })
            else:
                # BlockModel instance with .get_property() and .positions
                prop_data = block_model.get_property(property_name)
                if prop_data is None:
                    raise ValueError(f"Could not retrieve property '{property_name}'.")
                
                positions = block_model.positions
                axis_coords = positions[:, axis_idx]
                
                df_temp = pd.DataFrame({
                    axis_col_name: axis_coords,
                    property_name: prop_data
                })
        else:
            raise ValueError(f"Invalid data_mode '{data_mode}'. Must be 'drillhole' or 'block_model'.")
        
        df_temp = df_temp.dropna()
        
        if len(df_temp) == 0:
            raise ValueError("No valid data for swath analysis.")
        
        bins = np.linspace(df_temp[axis_col_name].min(), df_temp[axis_col_name].max(), n_bins + 1)
        df_temp['bin'] = pd.cut(df_temp[axis_col_name], bins=bins, include_lowest=True)
        
        grouped = df_temp.groupby('bin')[property_name].agg(['mean', 'std', 'count']).reset_index()
        grouped['centre'] = grouped['bin'].apply(lambda x: x.mid)
        grouped['std'] = grouped['std'].fillna(0)
        
        metadata = {
            "direction": direction,
            "property_name": property_name,
            "n_bins": n_bins,
            "samples_count": int(len(df_temp)),
            "message": f"Swath plot generated for {property_name} along {direction}-axis.",
        }
        
        payload = {
            "name": "swath",
            "grouped_data": grouped,
            "bins": bins,
            "direction": direction,
            "property_name": property_name,
            "n_bins": n_bins,
            "metadata": metadata,
        }
        return payload

    # =========================================================================
    # K-Means Clustering
    # =========================================================================
    
    def _prepare_kmeans_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare K-Means Clustering payload."""
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        except ImportError:
            raise NotImplementedError("scikit-learn is required for K-means clustering.")
        
        data_df = params["data_df"]
        features = params["features"]
        n_clusters = params["n_clusters"]
        n_init = params["n_init"]
        max_iter = params["max_iter"]
        random_state = params.get("random_state")
        standardize = params["standardize"]
        
        def progress_wrapper(percentage, message):
            if progress_callback:
                progress_callback(percentage, message)
        
        progress_wrapper(10, "Preparing data...")
        
        X = data_df[features].values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid data after removing NaN values")
        
        progress_wrapper(20, "Standardizing features...")
        
        scaler = None
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean
        
        progress_wrapper(40, f"Running K-means (k={n_clusters})...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            algorithm='lloyd'
        )
        
        labels = kmeans.fit_predict(X_scaled)
        
        progress_wrapper(70, "Computing cluster statistics...")
        
        inertia = kmeans.inertia_
        silhouette = None
        calinski = None
        davies_bouldin = None
        
        if n_clusters > 1 and len(X_scaled) > n_clusters:
            try:
                silhouette = silhouette_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
                davies_bouldin = davies_bouldin_score(X_scaled, labels)
            except Exception:
                pass
        
        full_labels = np.full(len(X), -1, dtype=int)
        full_labels[valid_mask] = labels
        
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask = full_labels == i
            cluster_data = data_df[cluster_mask]
            
            stats = {
                'cluster_id': i,
                'count': cluster_mask.sum(),
                'percentage': (cluster_mask.sum() / len(data_df)) * 100,
                'center': kmeans.cluster_centers_[i].tolist()
            }
            
            for feat in features:
                stats[f'{feat}_mean'] = float(cluster_data[feat].mean())
                stats[f'{feat}_std'] = float(cluster_data[feat].std())
            
            cluster_stats.append(stats)
        
        progress_wrapper(90, "Finalizing results...")
        
        results = {
            'labels': full_labels,
            'centers': kmeans.cluster_centers_.tolist(),
            'inertia': float(inertia),
            'silhouette': float(silhouette) if silhouette is not None else None,
            'calinski_harabasz': float(calinski) if calinski is not None else None,
            'davies_bouldin': float(davies_bouldin) if davies_bouldin is not None else None,
            'n_clusters': n_clusters,
            'features': features,
            'cluster_stats': cluster_stats,
            'scaler': None,
            'valid_mask': valid_mask.tolist(),
            'n_iterations': int(kmeans.n_iter_)
        }
        
        metadata = {
            "n_clusters": n_clusters,
            "features": features,
            "samples_count": int(len(X_clean)),
            "message": f"K-means clustering completed: {n_clusters} clusters identified.",
        }
        
        payload = {
            "name": "kmeans",
            "results": results,
            "metadata": metadata,
        }
        return payload

    # =========================================================================
    # Public API Methods (delegated from AppController)
    # =========================================================================
    
    def run_simple_kriging(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Simple Kriging via task system."""
        self._app.run_task("simple_kriging", params, callback, progress_callback)
    
    def run_kriging(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Ordinary Kriging via task system."""
        self._app.run_task("kriging", params, callback, progress_callback)
    
    def run_universal_kriging(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Universal Kriging via task system."""
        self._app.run_task("universal_kriging", config, callback, progress_callback)
    
    def run_cokriging(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Co-Kriging via task system."""
        self._app.run_task("cokriging", config, callback, progress_callback)
    
    def run_indicator_kriging(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Indicator Kriging via task system."""
        self._app.run_task("indicator_kriging", config, callback, progress_callback)
    
    def run_soft_kriging(self, config: Dict[str, Any], callback=None) -> None:
        """Run Soft/Bayesian Kriging via task system."""
        self._app.run_task("soft_kriging", config, callback)
    
    def run_sgsim(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run SGSIM via task system."""
        self._app.run_task("sgsim", params, callback, progress_callback)
    
    def run_ik_sgsim(self, config: Dict[str, Any], callback=None) -> None:
        """Run IK-based SGSIM via task system."""
        self._app.run_task("ik_sgsim", config, callback)
    
    def run_sis(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Sequential Indicator Simulation (SIS) via task system."""
        self._app.run_task("sis", params, callback, progress_callback)
    
    def run_turning_bands(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Turning Bands Simulation via task system."""
        self._app.run_task("turning_bands", params, callback, progress_callback)
    
    def run_dbs(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Direct Block Simulation (DBS) via task system."""
        self._app.run_task("dbs", params, callback, progress_callback)
    
    def run_grf(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Gaussian Random Field (GRF) simulation via task system."""
        self._app.run_task("grf", params, callback, progress_callback)
    
    def run_cosgsim(self, config: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Co-Simulation via task system."""
        self._app.run_task("cosgsim", config, callback, progress_callback)
    
    def run_variogram(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Variogram analysis via task system."""
        self._app.run_task("variogram", params, callback, progress_callback)
    
    def run_variogram_assistant(self, config: Dict[str, Any], callback=None) -> None:
        """Run Variogram Assistant via task system."""
        self._app.run_task("variogram_assistant", config, callback)
    
    def run_uncertainty(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Uncertainty analysis via task system."""
        self._app.run_task("uncertainty", params, callback, progress_callback)
    
    def run_economic_uncertainty(self, config: Dict[str, Any], callback=None) -> None:
        """Run Economic Uncertainty Propagation via task system."""
        self._app.run_task("economic_uncert", config, callback)
    
    def run_grade_stats(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Grade Statistics via task system."""
        self._app.run_task("grade_stats", params, callback, progress_callback)
    
    def run_grade_transform(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Grade Transformation via task system."""
        self._app.run_task("grade_transform", params, callback, progress_callback)
    
    def run_swath_analysis(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Swath Analysis via task system."""
        self._app.run_task("swath", params, callback, progress_callback)
    
    def run_kmeans(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run K-Means Clustering via task system."""
        self._app.run_task("kmeans", params, callback, progress_callback)

    def _prepare_mps_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Multiple-Point Simulation (MPS) payload.

        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data_df' into params before calling this.
        """
        from ..geostats.mps import MPSConfig, run_mps
        from ..models.sgsim3d import create_pyvista_grid

        # Extract MPS-specific parameters
        training_image = params.get("training_image")
        if training_image is None:
            raise ValueError("Missing required parameter 'training_image' for MPS.")

        template_size = params.get("template_size", (5, 5, 3))
        max_patterns = params.get("max_patterns", 10000)
        grid_shape = params.get("grid_shape", (20, 100, 100))
        n_realizations = params.get("n_realizations", 10)
        use_servo = params.get("use_servo", True)

        # Optional conditioning data
        drillhole_data = params.get("drillhole_data")
        conditioning_coords = None
        conditioning_values = None

        if drillhole_data is not None and not drillhole_data.empty:
            # Use drillhole data as conditioning
            variable = params.get("variable")
            if variable and variable in drillhole_data.columns:
                # Filter valid data
                data = drillhole_data.dropna(subset=['X', 'Y', 'Z', variable]).copy()
                # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
                if hasattr(drillhole_data, 'attrs') and drillhole_data.attrs:
                    data.attrs = drillhole_data.attrs.copy()
                if not data.empty:
                    conditioning_coords = data[['X', 'Y', 'Z']].to_numpy(dtype=np.float64)
                    conditioning_values = data[variable].to_numpy(dtype=np.float64)

                    # Remove NaN/inf
                    valid_mask = ~(np.isnan(conditioning_coords).any(axis=1) |
                                 np.isnan(conditioning_values) |
                                 np.isinf(conditioning_values))
                    conditioning_coords = conditioning_coords[valid_mask]
                    conditioning_values = conditioning_values[valid_mask]

        # Create MPS configuration
        config = MPSConfig(
            n_realizations=n_realizations,
            template_size=template_size,
            max_patterns=max_patterns,
            use_servo=use_servo
        )

        # Run MPS simulation
        def progress_wrapper(percentage, message):
            if progress_callback:
                progress_callback(percentage, message)

        progress_wrapper(10, "Scanning training image...")
        result = run_mps(
            grid_shape=grid_shape,
            training_image=training_image,
            config=config,
            conditioning_coords=conditioning_coords,
            conditioning_values=conditioning_values,
            progress_callback=lambda real, msg: progress_wrapper(10 + (real * 80) // n_realizations, f"Realization {real}/{n_realizations}")
        )

        progress_wrapper(90, "Creating visualization grid...")

        # Create PyVista grid for visualization
        # For MPS, we create a structured grid manually
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista is required for MPS visualization")

        # Create coordinate arrays (assuming origin at 0,0,0 and spacing of 1)
        # This matches the default grid setup used in other geostats methods
        nz, ny, nx = grid_shape
        gx = np.arange(nx, dtype=np.float64)
        gy = np.arange(ny, dtype=np.float64)
        gz = np.arange(nz, dtype=np.float64)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")

        grid = pv.StructuredGrid(GX, GY, GZ)

        # Add first realization as the primary property
        grid["MPS_Realization_0001"] = result.realizations[0].ravel(order='F')

        # Add additional properties for analysis
        if result.realizations.shape[0] > 1:
            # Add mean of all realizations
            mean_values = np.mean(result.realizations, axis=0)
            grid["MPS_Mean"] = mean_values.ravel(order='F')

            # Add variance
            var_values = np.var(result.realizations, axis=0)
            grid["MPS_Variance"] = var_values.ravel(order='F')

        progress_wrapper(100, "MPS simulation complete")

        # Return payload for controller
        return {
            "grid": grid,
            "result": result,
            "realizations": result.realizations,
            "realization_names": result.realization_names,
            "category_proportions": result.category_proportions,
            "metadata": result.metadata,
            "method": "Multiple-Point Simulation (MPS)"
        }

    def run_mps(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run Multiple-Point Simulation (MPS) via task system."""
        self._app.run_task("mps", params, callback, progress_callback)

    # =========================================================================
    # RBF Interpolation
    # =========================================================================

    def _prepare_rbf_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Run RBF Interpolation engine and package results for the UI.

        CRITICAL: This is a pure worker function. It must NOT access DataRegistry.
        The Controller (main thread) must inject 'data' into params before calling this.
        """
        from ..geostats.rbf_interpolation import rbf_interpolate_3d, RBFAnisotropy, create_rbf_anisotropy_from_ranges

        df = params.get("data")
        if df is None or df.empty:
            raise ValueError(
                "No drillhole data provided for RBF interpolation. "
                "Controller must fetch data from DataRegistry and inject it before calling worker."
            )

        variable = params.get("variable")
        if not variable or variable not in df.columns:
            raise ValueError("Selected variable not present in drillhole data.")

        grid_spec = params.get("grid_spec")
        if not grid_spec:
            raise ValueError("Grid specification missing for RBF interpolation.")

        cleaned = df.dropna(subset=["X", "Y", "Z", variable])
        # CRITICAL: Preserve attrs for JORC/SAMREC data lineage tracking
        if hasattr(df, 'attrs') and df.attrs:
            cleaned.attrs = df.attrs.copy()
        if cleaned.empty:
            raise ValueError("No valid samples available after filtering missing coordinates or values.")

        coords = cleaned[["X", "Y", "Z"]].to_numpy(float)
        values = cleaned[variable].to_numpy(float)

        # RBF-specific parameters
        kernel = params.get("kernel", "thin_plate_spline")
        smoothing = params.get("smoothing", 0.0)
        neighbors = params.get("neighbors")
        method = params.get("method", "auto")
        use_gpu = params.get("use_gpu", False)
        classification = params.get("classification", False)
        trend_degree = params.get("trend_degree")
        random_state = params.get("random_state", 42)

        # Anisotropy parameters
        anisotropy = None
        anisotropy_enabled = params.get("anisotropy_enabled", False)
        if anisotropy_enabled:
            range_x = params.get("range_x", 50.0)
            range_y = params.get("range_y", 30.0)
            range_z = params.get("range_z", 10.0)
            azimuth = params.get("azimuth", 0.0)
            dip = params.get("dip", 0.0)
            plunge = params.get("plunge", 0.0)

            anisotropy = create_rbf_anisotropy_from_ranges(
                range_x=range_x,
                range_y=range_y,
                range_z=range_z,
                azimuth=azimuth,
                dip=dip,
                plunge=plunge
            )

        if progress_callback:
            progress_callback(10, "Setting up RBF interpolation...")

        # Run RBF interpolation
        x_coords, y_coords, z_coords, grid_values = rbf_interpolate_3d(
            coords=coords,
            values=values,
            grid_spec=grid_spec,
            anisotropy=anisotropy,
            kernel=kernel,
            smoothing=smoothing,
            neighbors=neighbors,
            method=method,
            use_gpu=use_gpu,
            classification=classification,
            trend_degree=trend_degree,
            random_state=random_state
        )

        if progress_callback:
            progress_callback(80, "Creating visualization grid...")

        # Create PyVista grid for visualization (following kriging pattern)
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError("PyVista is required for RBF visualization")

        # Create structured grid
        GX, GY, GZ = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
        grid = pv.StructuredGrid(GX, GY, GZ)
        grid[f"RBF_{variable}"] = grid_values.ravel(order='F')

        # Add metadata
        metadata = {
            "method": "RBF Interpolation",
            "kernel": kernel,
            "smoothing": smoothing,
            "method_type": method,
            "use_gpu": use_gpu,
            "classification": classification,
            "anisotropy_enabled": anisotropy_enabled,
            "n_samples": len(coords),
            "variable": variable
        }

        if anisotropy:
            metadata.update({
                "range_x": range_x,
                "range_y": range_y,
                "range_z": range_z,
                "azimuth": azimuth,
                "dip": dip,
                "plunge": plunge
            })

        if trend_degree is not None:
            metadata["trend_degree"] = trend_degree

        # Run diagnostics if requested
        diagnostics = None
        if params.get("run_diagnostics", False):
            if progress_callback:
                progress_callback(90, "Running diagnostics...")

            try:
                from ..geostats.rbf_interpolation import RBFModel3D
                model = RBFModel3D(
                    coords=coords,
                    values=values,
                    anisotropy=anisotropy,
                    kernel=kernel,
                    smoothing=smoothing,
                    neighbors=neighbors,
                    method=method,
                    use_gpu=use_gpu,
                    classification=classification,
                    trend_degree=trend_degree,
                    random_state=random_state
                )
                diagnostics = model.diagnostics_holdout(test_fraction=0.2)
                metadata["diagnostics"] = diagnostics
            except Exception as e:
                logger.warning(f"RBF diagnostics failed: {e}")
                diagnostics = None

        if progress_callback:
            progress_callback(100, "RBF interpolation complete")

        # Return payload for controller
        return {
            "grid": grid,
            "grid_values": grid_values,
            "x_coords": x_coords,
            "y_coords": y_coords,
            "z_coords": z_coords,
            "metadata": metadata,
            "diagnostics": diagnostics,
            "method": "RBF Interpolation"
        }

    def run_rbf_interpolation(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run RBF Interpolation via task system."""
        self._app.run_task("rbf", params, callback, progress_callback)
