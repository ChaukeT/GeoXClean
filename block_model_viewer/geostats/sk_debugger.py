"""
Simple Kriging Debugger - Comprehensive error tracking and diagnostics.

This module provides detailed debugging capabilities to track simple kriging
execution and identify failure points that cause software crashes.
"""

import logging
import traceback
import sys
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class SimpleKrigingDebugger:
    """
    Comprehensive debugger for Simple Kriging operations.
    
    Tracks all stages of execution with detailed logging, exception catching,
    and diagnostic output to identify crash causes.
    """
    
    def __init__(self, debug_log_path: Optional[str] = None):
        """
        Initialize the debugger.
        
        Args:
            debug_log_path: Path to write detailed debug logs. If None, uses default.
        """
        if debug_log_path is None:
            debug_log_path = Path("sk_debug_log.txt")
        else:
            debug_log_path = Path(debug_log_path)
        
        self.debug_log_path = debug_log_path
        self.execution_stages = []
        self.start_time = None
        self.error_caught = False
        self.last_progress = 0
        
        # Create debug log file
        self._init_debug_log()
    
    def _init_debug_log(self):
        """Initialize the debug log file."""
        try:
            with open(self.debug_log_path, 'w') as f:
                f.write(f"=== SIMPLE KRIGING DEBUG LOG ===\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
        except Exception as e:
            logger.error(f"Failed to create debug log: {e}")
    
    def log_stage(self, stage: str, details: Optional[Dict[str, Any]] = None):
        """
        Log an execution stage.
        
        Args:
            stage: Stage name/description
            details: Optional dictionary of stage details
        """
        timestamp = time.time()
        if self.start_time is None:
            self.start_time = timestamp
        
        elapsed = timestamp - self.start_time
        
        stage_info = {
            'stage': stage,
            'elapsed_seconds': elapsed,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.execution_stages.append(stage_info)
        
        # Write to log file
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(f"[{elapsed:.2f}s] {stage}\n")
                if details:
                    for key, value in details.items():
                        # Handle numpy arrays
                        if isinstance(value, np.ndarray):
                            value_str = f"array{value.shape} [{value.dtype}]"
                        else:
                            value_str = str(value)
                        f.write(f"  {key}: {value_str}\n")
                f.write("\n")
        except Exception as e:
            logger.error(f"Failed to write to debug log: {e}")
        
        logger.info(f"SK DEBUG [{elapsed:.2f}s]: {stage}")
    
    def log_error(self, error: Exception, stage: str = "Unknown"):
        """
        Log an error with full traceback.
        
        Args:
            error: The exception that occurred
            stage: Stage where the error occurred
        """
        self.error_caught = True
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stage': stage,
            'traceback': traceback.format_exc()
        }
        
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"ERROR CAUGHT AT STAGE: {stage}\n")
                f.write(f"Error Type: {error_info['error_type']}\n")
                f.write(f"Error Message: {error_info['error_message']}\n")
                f.write(f"\nFull Traceback:\n")
                f.write(error_info['traceback'])
                f.write(f"{'='*60}\n\n")
        except Exception as e:
            logger.error(f"Failed to write error to debug log: {e}")
        
        logger.error(f"SK ERROR at {stage}: {error_info['error_type']}: {error_info['error_message']}")
    
    def log_progress(self, progress: int, message: str):
        """
        Log progress updates.
        
        Args:
            progress: Progress percentage (0-100)
            message: Progress message
        """
        self.last_progress = progress
        
        try:
            with open(self.debug_log_path, 'a') as f:
                elapsed = time.time() - self.start_time if self.start_time else 0
                f.write(f"[{elapsed:.2f}s] Progress {progress}%: {message}\n")
        except Exception as e:
            logger.error(f"Failed to write progress to debug log: {e}")
    
    def validate_inputs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all input parameters before kriging starts.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Validation result dict with 'valid' bool and 'issues' list
        """
        issues = []
        
        self.log_stage("Input Validation", {'params_keys': list(params.keys())})
        
        # Check for required parameters
        required_keys = ['data', 'variable', 'grid_spec', 'parameters']
        for key in required_keys:
            if key not in params:
                issues.append(f"Missing required parameter: {key}")
        
        # Validate data
        if 'data' in params:
            data = params['data']
            if data is None:
                issues.append("data is None")
            elif hasattr(data, 'empty') and data.empty:
                issues.append("data DataFrame is empty")
            else:
                self.log_stage("Data Validation", {
                    'n_rows': len(data),
                    'columns': list(data.columns) if hasattr(data, 'columns') else 'N/A',
                    'data_type': type(data).__name__
                })
                
                # Check for coordinate columns
                for col in ['X', 'Y', 'Z']:
                    if not hasattr(data, 'columns') or col not in data.columns:
                        issues.append(f"Missing coordinate column: {col}")
        
        # Validate variable
        if 'variable' in params and 'data' in params:
            variable = params['variable']
            data = params['data']
            if hasattr(data, 'columns') and variable not in data.columns:
                issues.append(f"Variable '{variable}' not in data columns")
        
        # Validate grid_spec
        if 'grid_spec' in params:
            grid_spec = params['grid_spec']
            required_grid_keys = ['nx', 'ny', 'nz', 'xmin', 'ymin', 'zmin', 'xinc', 'yinc', 'zinc']
            for key in required_grid_keys:
                if key not in grid_spec:
                    issues.append(f"Missing grid_spec parameter: {key}")
                else:
                    try:
                        val = float(grid_spec[key])
                        if key.startswith('n') and (val <= 0 or val != int(val)):
                            issues.append(f"Invalid grid dimension {key}: must be positive integer")
                        elif key.endswith('inc') and val <= 0:
                            issues.append(f"Invalid grid increment {key}: must be positive")
                    except (ValueError, TypeError):
                        issues.append(f"Invalid grid_spec value for {key}: not numeric")
        
        # Validate parameters
        if 'parameters' in params:
            param_values = params['parameters']
            if not isinstance(param_values, dict):
                issues.append(f"parameters must be dict, got {type(param_values).__name__}")
            else:
                # Check numeric parameters
                numeric_params = ['global_mean', 'sill', 'nugget', 'range_major', 
                                 'range_minor', 'range_vert', 'azimuth', 'dip',
                                 'max_search_radius']
                for key in numeric_params:
                    if key in param_values:
                        try:
                            val = float(param_values[key])
                            if np.isnan(val) or np.isinf(val):
                                issues.append(f"Invalid {key}: NaN or Inf")
                        except (ValueError, TypeError):
                            issues.append(f"Invalid {key}: not numeric")
                
                # Check integer parameters
                int_params = ['ndmax', 'nmin']
                for key in int_params:
                    if key in param_values:
                        try:
                            val = int(param_values[key])
                            if val < 0:
                                issues.append(f"Invalid {key}: must be non-negative")
                        except (ValueError, TypeError):
                            issues.append(f"Invalid {key}: not integer")
        
        result = {
            'valid': len(issues) == 0,
            'issues': issues
        }
        
        if not result['valid']:
            self.log_stage("Validation FAILED", {'issues': issues})
        else:
            self.log_stage("Validation PASSED")
        
        return result
    
    def wrap_execution(self, func: Callable, *args, **kwargs) -> Any:
        """
        Wrap a function execution with comprehensive error catching.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result or raises exception
        """
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        
        self.log_stage(f"Executing: {func_name}", {
            'args_count': len(args),
            'kwargs_keys': list(kwargs.keys())
        })
        
        try:
            result = func(*args, **kwargs)
            self.log_stage(f"Completed: {func_name}", {
                'result_type': type(result).__name__
            })
            return result
        
        except Exception as e:
            self.log_error(e, func_name)
            raise
    
    def create_progress_callback(self, original_callback: Optional[Callable] = None) -> Callable:
        """
        Create a progress callback that logs to debug file.
        
        Args:
            original_callback: Original progress callback to chain
            
        Returns:
            Wrapped progress callback
        """
        def debug_progress_callback(progress: int, message: str):
            self.log_progress(progress, message)
            if original_callback:
                try:
                    original_callback(progress, message)
                except Exception as e:
                    self.log_error(e, "progress_callback")
        
        return debug_progress_callback
    
    def finalize(self):
        """Finalize the debug log with summary."""
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"DEBUG SESSION SUMMARY\n")
                f.write(f"{'='*60}\n")
                f.write(f"Total stages: {len(self.execution_stages)}\n")
                f.write(f"Error caught: {self.error_caught}\n")
                f.write(f"Last progress: {self.last_progress}%\n")
                
                if self.start_time:
                    total_time = time.time() - self.start_time
                    f.write(f"Total execution time: {total_time:.2f}s\n")
                
                f.write(f"\nStage Timeline:\n")
                for stage in self.execution_stages:
                    f.write(f"  [{stage['elapsed_seconds']:.2f}s] {stage['stage']}\n")
                
                f.write(f"\n{'='*60}\n")
                f.write(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            logger.error(f"Failed to finalize debug log: {e}")


def run_simple_kriging_with_debug(params: Dict[str, Any], 
                                   progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Run simple kriging with comprehensive debugging.
    
    This is a drop-in replacement for the normal simple kriging execution
    that adds extensive error tracking and diagnostics.
    
    Args:
        params: Simple kriging parameters
        progress_callback: Optional progress callback
        
    Returns:
        Result dictionary
        
    Raises:
        Exception: Any exception that occurs during execution (after logging)
    """
    debugger = SimpleKrigingDebugger()
    
    try:
        debugger.log_stage("Starting Simple Kriging Debug Session")
        
        # Validate inputs
        validation = debugger.validate_inputs(params)
        if not validation['valid']:
            error_msg = "Input validation failed:\n" + "\n".join(f"  - {issue}" for issue in validation['issues'])
            raise ValueError(error_msg)
        
        # Import the actual kriging function
        debugger.log_stage("Importing simple_kriging_3d module")
        from ..models.simple_kriging3d import simple_kriging_3d, SKParameters
        
        # Extract and validate data
        debugger.log_stage("Extracting data")
        df = params.get("data")
        variable = params.get("variable")
        
        debugger.log_stage("Cleaning data", {
            'original_rows': len(df),
            'variable': variable
        })
        
        cleaned = df.dropna(subset=["X", "Y", "Z", variable])
        if hasattr(df, 'attrs') and df.attrs:
            cleaned.attrs = df.attrs.copy()
        
        debugger.log_stage("Data cleaned", {
            'cleaned_rows': len(cleaned),
            'dropped_rows': len(df) - len(cleaned)
        })
        
        if cleaned.empty:
            raise ValueError("No valid samples after cleaning")
        
        # Extract coordinates and values
        debugger.log_stage("Extracting coordinates and values")
        coords = cleaned[["X", "Y", "Z"]].to_numpy(float)
        values = cleaned[variable].to_numpy(float)
        
        debugger.log_stage("Coordinates extracted", {
            'n_samples': len(coords),
            'coords_shape': coords.shape,
            'values_shape': values.shape,
            'values_min': float(np.nanmin(values)),
            'values_max': float(np.nanmax(values)),
            'values_mean': float(np.nanmean(values)),
            'values_std': float(np.nanstd(values))
        })
        
        # Build grid
        debugger.log_stage("Building estimation grid")
        grid_spec = params.get("grid_spec")
        
        nx = int(grid_spec["nx"])
        ny = int(grid_spec["ny"])
        nz = int(grid_spec["nz"])
        xmin = float(grid_spec["xmin"])
        ymin = float(grid_spec["ymin"])
        zmin = float(grid_spec["zmin"])
        xinc = float(grid_spec["xinc"])
        yinc = float(grid_spec["yinc"])
        zinc = float(grid_spec["zinc"])
        
        gx = np.arange(nx) * xinc + xmin + xinc / 2.0
        gy = np.arange(ny) * yinc + ymin + yinc / 2.0
        gz = np.arange(nz) * zinc + zmin + zinc / 2.0
        
        debugger.log_stage("Grid arrays created", {
            'nx': nx, 'ny': ny, 'nz': nz,
            'total_points': nx * ny * nz,
            'x_range': [float(gx.min()), float(gx.max())],
            'y_range': [float(gy.min()), float(gy.max())],
            'z_range': [float(gz.min()), float(gz.max())]
        })
        
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        grid_coords = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
        
        debugger.log_stage("Grid coordinates created", {
            'grid_coords_shape': grid_coords.shape,
            'memory_mb': grid_coords.nbytes / (1024 * 1024)
        })
        
        # Build SKParameters
        debugger.log_stage("Building SKParameters")
        param_values = params.get("parameters", {})
        
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
        
        debugger.log_stage("SKParameters created", {
            'global_mean': sk_params.global_mean,
            'variogram_type': sk_params.variogram_type,
            'range_major': sk_params.range_major,
            'ndmax': sk_params.ndmax
        })
        
        # Create debug progress callback
        debug_progress = debugger.create_progress_callback(progress_callback)
        
        # Execute kriging
        debugger.log_stage("CALLING simple_kriging_3d - CRITICAL POINT")
        
        estimates, variances, neighbour_counts, diagnostics = simple_kriging_3d(
            coords, values, grid_coords, sk_params, 
            progress_callback=debug_progress
        )
        
        debugger.log_stage("simple_kriging_3d COMPLETED SUCCESSFULLY", {
            'estimates_shape': estimates.shape,
            'estimates_min': float(np.nanmin(estimates)),
            'estimates_max': float(np.nanmax(estimates)),
            'estimates_mean': float(np.nanmean(estimates)),
            'n_nan': int(np.sum(np.isnan(estimates)))
        })
        
        # Package results
        debugger.log_stage("Packaging results")
        
        result = {
            'estimates': estimates,
            'variances': variances,
            'neighbour_counts': neighbour_counts,
            'diagnostics': diagnostics,
            'sk_params': sk_params,
            'grid_spec': grid_spec,
            'n_samples': len(coords)
        }
        
        debugger.log_stage("Results packaged successfully")
        debugger.finalize()
        
        logger.info(f"Simple Kriging DEBUG completed successfully. Log: {debugger.debug_log_path}")
        
        return result
        
    except Exception as e:
        debugger.log_error(e, "run_simple_kriging_with_debug")
        debugger.finalize()
        
        logger.error(f"Simple Kriging DEBUG FAILED. See log: {debugger.debug_log_path}")
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        
        raise
