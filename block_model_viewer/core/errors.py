"""
GeoX Error Handling Infrastructure
===================================

Provides a structured error handling system with:
- Typed exceptions with error codes
- User-friendly messages separate from technical details
- Error boundary decorator for consistent handling
- Integration with logging and UI dialogs

AUDIT COMPLIANCE:
- All errors are logged with full context
- Error codes enable tracking and support
- Recoverable vs non-recoverable distinction
"""

from __future__ import annotations

import functools
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)


# =============================================================================
# ERROR SEVERITY LEVELS
# =============================================================================

class ErrorSeverity(str, Enum):
    """Severity levels for GeoX errors."""
    INFO = "info"           # Informational, operation continued
    WARNING = "warning"     # Potential issue, operation continued
    ERROR = "error"         # Operation failed but app stable
    CRITICAL = "critical"   # App may be unstable


# =============================================================================
# ERROR CATEGORIES
# =============================================================================

class ErrorCategory(str, Enum):
    """Categories for error classification."""
    DATA = "DATA"           # Data loading, validation, quality issues
    GEOLOGY = "GEO"         # Geological modelling errors
    VISUALIZATION = "VIS"   # Rendering, display errors
    GEOSTATS = "STAT"       # Geostatistics errors
    IO = "IO"               # File I/O errors
    VALIDATION = "VAL"      # Validation gate failures
    SYSTEM = "SYS"          # System/infrastructure errors
    USER = "USR"            # User input errors


# =============================================================================
# BASE EXCEPTION CLASS
# =============================================================================

class GeoXError(Exception):
    """
    Base exception for all GeoX errors.
    
    Provides structured error information including:
    - Error code for tracking (e.g., "GEO-001")
    - User-friendly message
    - Technical details for debugging
    - Severity level
    - Recovery suggestions
    
    Usage:
        raise GeoXError(
            code="GEO-001",
            message="Failed to build geological surfaces",
            details="RBF matrix was singular",
            recoverable=True,
            suggestions=["Try increasing the nugget value", "Check for duplicate points"]
        )
    """
    
    def __init__(
        self,
        code: str,
        message: str,
        details: str = "",
        recoverable: bool = True,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.message = message
        self.details = details
        self.recoverable = recoverable
        self.severity = severity
        self.suggestions = suggestions or []
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        
        # Build full message for standard exception handling
        full_message = f"[{code}] {message}"
        if details:
            full_message += f"\nDetails: {details}"
        
        super().__init__(full_message)
        
        # Log the error
        self._log_error()
    
    def _log_error(self) -> None:
        """Log the error with appropriate level."""
        log_msg = f"{self.code}: {self.message}"
        if self.details:
            log_msg += f" | {self.details}"
        if self.context:
            log_msg += f" | Context: {self.context}"
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_msg, exc_info=True)
        elif self.severity == ErrorSeverity.ERROR:
            logger.error(log_msg)
        elif self.severity == ErrorSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
            "recoverable": self.recoverable,
            "severity": self.severity.value,
            "suggestions": self.suggestions,
            "context": self.context,
            "timestamp": self.timestamp,
        }
    
    def user_message(self) -> str:
        """Get user-friendly message for display."""
        msg = self.message
        if self.suggestions:
            msg += "\n\nSuggestions:\n"
            for suggestion in self.suggestions:
                msg += f"  • {suggestion}\n"
        return msg


# =============================================================================
# SPECIALIZED EXCEPTION CLASSES
# =============================================================================

class DataError(GeoXError):
    """Error related to data loading, validation, or quality."""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: str = "",
        recoverable: bool = True,
        **kwargs
    ):
        if not code.startswith("DATA-"):
            code = f"DATA-{code}"
        super().__init__(code, message, details, recoverable, **kwargs)


class GeologyError(GeoXError):
    """Error related to geological modelling."""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: str = "",
        recoverable: bool = True,
        **kwargs
    ):
        if not code.startswith("GEO-"):
            code = f"GEO-{code}"
        super().__init__(code, message, details, recoverable, **kwargs)


class VisualizationError(GeoXError):
    """Error related to rendering and display."""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: str = "",
        recoverable: bool = True,
        **kwargs
    ):
        if not code.startswith("VIS-"):
            code = f"VIS-{code}"
        super().__init__(code, message, details, recoverable, **kwargs)


class GeostatsError(GeoXError):
    """Error related to geostatistical analysis."""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: str = "",
        recoverable: bool = True,
        **kwargs
    ):
        if not code.startswith("STAT-"):
            code = f"STAT-{code}"
        super().__init__(code, message, details, recoverable, **kwargs)


class ValidationError(GeoXError):
    """Error related to data or parameter validation."""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: str = "",
        recoverable: bool = True,
        **kwargs
    ):
        if not code.startswith("VAL-"):
            code = f"VAL-{code}"
        super().__init__(code, message, details, recoverable, **kwargs)


class IOError(GeoXError):
    """Error related to file I/O operations."""
    
    def __init__(
        self,
        code: str,
        message: str,
        details: str = "",
        recoverable: bool = True,
        **kwargs
    ):
        if not code.startswith("IO-"):
            code = f"IO-{code}"
        super().__init__(code, message, details, recoverable, **kwargs)


# =============================================================================
# ERROR BOUNDARY DECORATOR
# =============================================================================

# Type variable for decorated function return type
T = TypeVar('T')

# Global error dialog function (set by UI layer)
_error_dialog_func: Optional[Callable[[GeoXError], None]] = None


def set_error_dialog_handler(handler: Callable[[GeoXError], None]) -> None:
    """
    Set the global error dialog handler.
    
    Called by the UI layer to register the error dialog function.
    This allows the error handling system to show dialogs without
    importing Qt directly.
    """
    global _error_dialog_func
    _error_dialog_func = handler


def show_error_dialog(error: GeoXError) -> None:
    """
    Show an error dialog to the user.
    
    Uses the registered handler, or logs if no handler is set.
    """
    if _error_dialog_func is not None:
        try:
            _error_dialog_func(error)
        except Exception as e:
            logger.error(f"Error dialog handler failed: {e}")
    else:
        # Fallback to console logging
        logger.error(f"ERROR DIALOG: {error.code} - {error.message}")
        if error.details:
            logger.error(f"  Details: {error.details}")


def error_boundary(
    operation_name: str,
    show_dialog: bool = True,
    default_return: Any = None,
    reraise: bool = False,
    error_class: Type[GeoXError] = GeoXError,
):
    """
    Decorator that provides consistent error handling for operations.
    
    Catches exceptions, logs them, optionally shows a dialog, and
    returns a default value or re-raises.
    
    Args:
        operation_name: Human-readable name of the operation (for error messages)
        show_dialog: Whether to show an error dialog to the user
        default_return: Value to return on error (if not re-raising)
        reraise: Whether to re-raise the exception after handling
        error_class: Exception class to use for wrapping unexpected errors
    
    Usage:
        @error_boundary("Build Surfaces", show_dialog=True)
        def build_surfaces(self):
            # ... code that might fail ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except GeoXError as e:
                # Already a GeoX error - handle it
                if show_dialog:
                    show_error_dialog(e)
                if reraise or not e.recoverable:
                    raise
                return default_return
            except Exception as e:
                # Unexpected error - wrap it
                tb = traceback.format_exc()
                wrapped = error_class(
                    code="UNEXPECTED",
                    message=f"Unexpected error in {operation_name}",
                    details=f"{type(e).__name__}: {str(e)}",
                    recoverable=True,
                    context={"traceback": tb},
                )
                logger.exception(f"Unexpected error in {operation_name}")
                if show_dialog:
                    show_error_dialog(wrapped)
                if reraise:
                    raise wrapped from e
                return default_return
        return wrapper
    return decorator


def safe_operation(
    operation_name: str,
    log_level: str = "warning",
):
    """
    Lightweight decorator for non-critical operations.
    
    Catches exceptions and logs them without showing dialogs.
    Useful for cleanup operations, optional features, etc.
    
    Args:
        operation_name: Name of the operation for logging
        log_level: Logging level for caught exceptions
    
    Usage:
        @safe_operation("Update legend")
        def update_legend(self):
            # ... code that might fail but shouldn't crash app ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Optional[T]]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Optional[T]:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_func = getattr(logger, log_level, logger.warning)
                log_func(f"{operation_name} failed: {e}", exc_info=True)
                return None
        return wrapper
    return decorator


# =============================================================================
# ERROR CODE REGISTRY
# =============================================================================

# Standard error codes for reference
ERROR_CODES = {
    # Data errors (DATA-xxx)
    "DATA-001": "Missing required column in dataset",
    "DATA-002": "Invalid data format",
    "DATA-003": "Data quality gate failed",
    "DATA-004": "Empty dataset",
    "DATA-005": "Duplicate records detected",
    
    # Geology errors (GEO-xxx)
    "GEO-001": "Composited data required but not found",
    "GEO-002": "Missing required columns for geological modelling",
    "GEO-003": "Insufficient contacts for surface fitting",
    "GEO-004": "Zero surfaces generated",
    "GEO-005": "RBF matrix singular",
    "GEO-006": "No valid domains found",
    "GEO-007": "Contact clustering failed",
    "GEO-008": "Empty classifier",
    
    # Visualization errors (VIS-xxx)
    "VIS-001": "Failed to render mesh",
    "VIS-002": "Invalid color mapping",
    "VIS-003": "Camera setup failed",
    "VIS-004": "Layer update failed",
    "VIS-005": "Legend synchronization failed",
    
    # Geostatistics errors (STAT-xxx)
    "STAT-001": "Variogram fitting failed",
    "STAT-002": "Kriging matrix singular",
    "STAT-003": "Insufficient data for analysis",
    "STAT-004": "Invalid variogram parameters",
    
    # Validation errors (VAL-xxx)
    "VAL-001": "Required parameter missing",
    "VAL-002": "Parameter out of range",
    "VAL-003": "Invalid parameter type",
    "VAL-004": "Constraint violation",
    
    # I/O errors (IO-xxx)
    "IO-001": "File not found",
    "IO-002": "Permission denied",
    "IO-003": "Invalid file format",
    "IO-004": "Write operation failed",
}


def get_error_description(code: str) -> str:
    """Get the description for an error code."""
    return ERROR_CODES.get(code, "Unknown error")

