"""
Comprehensive audit logging decorators and utilities for GeoX.

Provides automatic logging of:
- User actions (button clicks, menu selections)
- Data operations (size, duration, success/failure)
- Performance metrics
- Parameter changes
"""

import logging
import time
import functools
from typing import Callable, Any, Optional
import traceback

logger = logging.getLogger(__name__)


def log_user_action(action_name: Optional[str] = None):
    """
    Decorator to log user actions like button clicks, menu selections, etc.
    
    Usage:
        @log_user_action("Run SGSIM")
        def run_analysis(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get action name from decorator param or function name
            name = action_name or func.__name__.replace('_', ' ').title()
            
            # Get context (class name if method)
            context = ""
            if args and hasattr(args[0], '__class__'):
                context = f"{args[0].__class__.__name__}."
            
            logger.info(f"USER ACTION: {context}{name}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"USER ACTION COMPLETED: {context}{name}")
                return result
            except Exception as e:
                logger.error(
                    f"USER ACTION FAILED: {context}{name} - {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_data_operation(operation_name: Optional[str] = None, log_size: bool = True):
    """
    Decorator to log data operations with size and performance metrics.
    
    Usage:
        @log_data_operation("Load Drillhole Data")
        def load_data(self, data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = operation_name or func.__name__.replace('_', ' ').title()
            
            # Get context
            context = ""
            if args and hasattr(args[0], '__class__'):
                context = f"{args[0].__class__.__name__}."
            
            start_time = time.time()
            logger.info(f"DATA OPERATION START: {context}{name}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                
                # Log result size if applicable
                size_info = ""
                if log_size and result is not None:
                    try:
                        import pandas as pd
                        import numpy as np
                        
                        if isinstance(result, pd.DataFrame):
                            size_info = f" ({len(result)} rows, {len(result.columns)} cols)"
                        elif isinstance(result, (list, tuple)):
                            size_info = f" ({len(result)} items)"
                        elif isinstance(result, np.ndarray):
                            size_info = f" ({result.shape})"
                        elif isinstance(result, dict):
                            size_info = f" ({len(result)} keys)"
                    except Exception:
                        pass
                
                logger.info(
                    f"DATA OPERATION SUCCESS: {context}{name}{size_info} "
                    f"[{elapsed:.2f}s]"
                )
                return result
                
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    f"DATA OPERATION FAILED: {context}{name} after {elapsed:.2f}s - {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_parameter_change(param_name: str):
    """
    Decorator to log parameter/setting changes.
    
    Usage:
        @log_parameter_change("cutoff_grade")
        def set_cutoff(self, value):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get context
            context = ""
            if args and hasattr(args[0], '__class__'):
                context = f"{args[0].__class__.__name__}."
            
            # Extract new value (typically first argument after self)
            new_value = args[1] if len(args) > 1 else kwargs.get('value', '<unknown>')
            
            logger.info(f"PARAMETER CHANGE: {context}{param_name} = {new_value}")
            
            result = func(*args, **kwargs)
            return result
        
        return wrapper
    return decorator


def log_export_operation(export_type: str = "file"):
    """
    Decorator to log export operations for audit trail.
    
    Usage:
        @log_export_operation("CSV")
        def export_to_csv(self, path):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ""
            if args and hasattr(args[0], '__class__'):
                context = f"{args[0].__class__.__name__}."
            
            # Extract file path if available
            file_info = ""
            if args and len(args) > 1:
                if isinstance(args[1], str):
                    file_info = f" to {args[1]}"
            elif 'path' in kwargs or 'file_path' in kwargs:
                path = kwargs.get('path') or kwargs.get('file_path')
                file_info = f" to {path}"
            
            logger.info(f"EXPORT START: {context}{export_type}{file_info}")
            
            try:
                result = func(*args, **kwargs)
                logger.info(f"EXPORT SUCCESS: {context}{export_type}{file_info}")
                return result
            except Exception as e:
                logger.error(
                    f"EXPORT FAILED: {context}{export_type}{file_info} - {str(e)}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_performance(threshold_seconds: float = 1.0):
    """
    Decorator to log performance metrics for slow operations.
    Only logs if operation exceeds threshold.
    
    Usage:
        @log_performance(threshold_seconds=5.0)
        def expensive_calculation(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            if elapsed >= threshold_seconds:
                context = ""
                if args and hasattr(args[0], '__class__'):
                    context = f"{args[0].__class__.__name__}."
                
                logger.warning(
                    f"PERFORMANCE: {context}{func.__name__} took {elapsed:.2f}s "
                    f"(threshold: {threshold_seconds}s)"
                )
            
            return result
        
        return wrapper
    return decorator


class AuditLogger:
    """
    Helper class for structured audit logging.
    Use this for complex operations that need detailed logging.
    """
    
    def __init__(self, operation_name: str, context: str = ""):
        self.operation_name = operation_name
        self.context = context
        self.start_time = None
        self.metadata = {}
    
    def start(self, **metadata):
        """Start the operation and log it."""
        self.start_time = time.time()
        self.metadata = metadata
        
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        logger.info(
            f"AUDIT START: {self.context}{self.operation_name} "
            f"[{meta_str}]" if meta_str else ""
        )
    
    def success(self, **result_metadata):
        """Log successful completion."""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        meta_str = ", ".join(f"{k}={v}" for k, v in result_metadata.items())
        
        logger.info(
            f"AUDIT SUCCESS: {self.context}{self.operation_name} "
            f"[{elapsed:.2f}s] {meta_str if meta_str else ''}"
        )
    
    def failure(self, error: Exception):
        """Log failure."""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        logger.error(
            f"AUDIT FAILURE: {self.context}{self.operation_name} "
            f"[{elapsed:.2f}s] - {str(error)}",
            exc_info=True
        )
    
    def checkpoint(self, checkpoint_name: str, **metadata):
        """Log a checkpoint during long operation."""
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
        
        logger.debug(
            f"AUDIT CHECKPOINT: {self.context}{self.operation_name} - "
            f"{checkpoint_name} [{elapsed:.2f}s] {meta_str if meta_str else ''}"
        )
