"""
Background Worker Infrastructure
================================

Thread-safe background workers for long-running operations.

Phase 6.1 Implementation: Provides QRunnable-based workers that:
- Run operations in thread pool
- Emit progress signals (Qt-safe)
- Handle errors gracefully
- Support cancellation

Usage:
    from block_model_viewer.core.worker import GeologyWorker
    
    worker = GeologyWorker(
        build_multidomain_surfaces,
        drillhole_data={'composites': composites},
        domain_model=domain_model
    )
    worker.signals.progress.connect(on_progress)
    worker.signals.finished.connect(on_complete)
    worker.signals.error.connect(on_error)
    
    QThreadPool.globalInstance().start(worker)
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Dict, Optional, Tuple

from PyQt6.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot, QThreadPool

logger = logging.getLogger(__name__)


# =============================================================================
# WORKER SIGNALS
# =============================================================================

class WorkerSignals(QObject):
    """
    Signals for worker communication.
    
    Must be a separate QObject because QRunnable doesn't support signals directly.
    """
    # Progress update: (percentage 0-100, message)
    progress = pyqtSignal(int, str)
    
    # Operation completed: result object
    finished = pyqtSignal(object)
    
    # Error occurred: (error_code, error_message)
    error = pyqtSignal(str, str)
    
    # Cancellation acknowledged
    cancelled = pyqtSignal()


# =============================================================================
# BASE WORKER
# =============================================================================

class BaseWorker(QRunnable):
    """
    Base class for background workers.
    
    Provides:
    - Signal infrastructure
    - Error handling
    - Cancellation support
    - Progress callback injection
    """
    
    def __init__(self):
        super().__init__()
        self.signals = WorkerSignals()
        self._is_cancelled = False
    
    def cancel(self) -> None:
        """Request cancellation of the worker."""
        self._is_cancelled = True
        logger.debug(f"{self.__class__.__name__} cancellation requested")
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested."""
        return self._is_cancelled
    
    def _emit_progress(self, percentage: int, message: str) -> None:
        """Emit progress signal (thread-safe)."""
        if not self._is_cancelled:
            self.signals.progress.emit(percentage, message)
    
    def _emit_finished(self, result: Any) -> None:
        """Emit finished signal with result."""
        self.signals.finished.emit(result)
    
    def _emit_error(self, code: str, message: str) -> None:
        """Emit error signal."""
        self.signals.error.emit(code, message)


# =============================================================================
# GEOLOGY WORKER
# =============================================================================

class GeologyWorker(BaseWorker):
    """
    Background worker for long-running geology operations.
    
    Phase 6.1 Implementation: Runs geology operations (surface building,
    voxel modeling, etc.) in background threads with proper Qt integration.
    
    Features:
    - Automatic progress callback injection
    - Error categorization (GeoXError support)
    - Cancellation support
    - Result passing via signals
    
    Usage:
        def build_model(data, domain_model, progress_callback=None):
            # Your long-running operation
            pass
        
        worker = GeologyWorker(build_model, data=my_data, domain_model=model)
        worker.signals.finished.connect(on_complete)
        QThreadPool.globalInstance().start(worker)
    """
    
    def __init__(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs
    ):
        """
        Initialize geology worker.
        
        Args:
            operation: The function to run in background
            *args: Positional arguments for operation
            **kwargs: Keyword arguments for operation
        """
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        
        # Auto-scale so it doesn't block too many threads
        self.setAutoDelete(True)
    
    @pyqtSlot()
    def run(self) -> None:
        """Execute the operation in background thread."""
        try:
            # Inject progress callback that wraps our signal emission
            def progress_callback(percentage: int, message: str):
                if self._is_cancelled:
                    # Raise to stop the operation
                    raise InterruptedError("Operation cancelled by user")
                self._emit_progress(percentage, message)
            
            # Add progress callback to kwargs if operation accepts it
            self.kwargs['progress_callback'] = progress_callback
            
            # Run the operation
            result = self.operation(*self.args, **self.kwargs)
            
            if self._is_cancelled:
                self.signals.cancelled.emit()
                return
            
            self._emit_finished(result)
            
        except InterruptedError:
            # Clean cancellation
            self.signals.cancelled.emit()
            logger.info("Geology operation cancelled")
            
        except Exception as e:
            # Handle GeoXError specially
            try:
                from .errors import GeoXError
                if isinstance(e, GeoXError):
                    self._emit_error(e.code, e.message)
                    return
            except ImportError:
                pass
            
            # Generic error handling
            error_code = "UNEXPECTED"
            error_message = f"{type(e).__name__}: {str(e)}"
            
            logger.error(f"Geology worker error: {error_message}")
            logger.debug(traceback.format_exc())
            
            self._emit_error(error_code, error_message)


# =============================================================================
# DATA WORKER
# =============================================================================

class DataWorker(BaseWorker):
    """
    Background worker for data loading/processing operations.
    
    Similar to GeologyWorker but optimized for I/O-bound operations.
    """
    
    def __init__(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs
    ):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self.setAutoDelete(True)
    
    @pyqtSlot()
    def run(self) -> None:
        """Execute the data operation."""
        try:
            def progress_callback(percentage: int, message: str):
                if self._is_cancelled:
                    raise InterruptedError("Operation cancelled")
                self._emit_progress(percentage, message)
            
            self.kwargs['progress_callback'] = progress_callback
            result = self.operation(*self.args, **self.kwargs)
            
            if not self._is_cancelled:
                self._emit_finished(result)
            else:
                self.signals.cancelled.emit()
                
        except InterruptedError:
            self.signals.cancelled.emit()
        except Exception as e:
            self._emit_error("DATA-ERR", str(e))


# =============================================================================
# RENDER WORKER
# =============================================================================

class RenderWorker(BaseWorker):
    """
    Background worker for rendering preparation operations.
    
    For operations that prepare data for rendering but don't touch the
    VTK pipeline directly (which must be on main thread).
    """
    
    def __init__(
        self,
        operation: Callable[..., Any],
        *args,
        **kwargs
    ):
        super().__init__()
        self.operation = operation
        self.args = args
        self.kwargs = kwargs
        self.setAutoDelete(True)
    
    @pyqtSlot()
    def run(self) -> None:
        """Execute the render preparation."""
        try:
            result = self.operation(*self.args, **self.kwargs)
            if not self._is_cancelled:
                self._emit_finished(result)
        except Exception as e:
            self._emit_error("RENDER-ERR", str(e))


# =============================================================================
# WORKER POOL UTILITIES
# =============================================================================

def get_thread_pool() -> QThreadPool:
    """Get the global thread pool for workers."""
    return QThreadPool.globalInstance()


def submit_worker(worker: BaseWorker) -> None:
    """Submit a worker to the global thread pool."""
    get_thread_pool().start(worker)


def wait_for_all_workers(timeout_ms: int = 30000) -> bool:
    """
    Wait for all workers in the pool to complete.
    
    Args:
        timeout_ms: Maximum time to wait in milliseconds
    
    Returns:
        True if all workers completed, False if timed out
    """
    return get_thread_pool().waitForDone(timeout_ms)


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class WorkerContext:
    """
    Context manager for running a worker and waiting for result.
    
    Usage:
        with WorkerContext(my_operation, arg1, arg2) as ctx:
            ctx.wait()
            result = ctx.result
    """
    
    def __init__(
        self,
        operation: Callable[..., Any],
        *args,
        worker_class: type = GeologyWorker,
        **kwargs
    ):
        self.worker = worker_class(operation, *args, **kwargs)
        self.result = None
        self.error = None
        self._completed = False
        
        # Connect signals
        self.worker.signals.finished.connect(self._on_finished)
        self.worker.signals.error.connect(self._on_error)
    
    def __enter__(self):
        submit_worker(self.worker)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def _on_finished(self, result):
        self.result = result
        self._completed = True
    
    def _on_error(self, code, message):
        self.error = (code, message)
        self._completed = True
    
    def wait(self, timeout_ms: int = 60000) -> bool:
        """Wait for worker to complete."""
        import time
        start = time.time()
        while not self._completed:
            if (time.time() - start) * 1000 > timeout_ms:
                return False
            time.sleep(0.01)
        return True
    
    def cancel(self):
        """Cancel the worker."""
        self.worker.cancel()

