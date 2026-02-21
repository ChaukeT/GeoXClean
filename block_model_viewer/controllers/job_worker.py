"""
Unified Job Worker for Background Task Execution - Step 11

Provides a single, predictable QThread-based worker for all background tasks.
All analysis, resource, and planning operations run through this worker.
"""

import logging
import traceback
from typing import Callable, Dict, Any, Optional

try:
    from PyQt6.QtCore import QThread, pyqtSignal
except ImportError:
    # Fallback for headless environments
    QThread = object  # type: ignore
    def pyqtSignal(*args, **kwargs):  # type: ignore
        return None

logger = logging.getLogger(__name__)


class JobWorker(QThread):
    """
    Unified worker for executing background tasks.
    
    Runs exactly ONE function with exactly ONE params dict.
    Emits signals for progress, completion, and errors.
    """
    
    finished = pyqtSignal(object)  # results or error payload
    progress = pyqtSignal(int, str)    # percent, message
    error = pyqtSignal(str, str)    # message, traceback
    
    def __init__(self, func: Callable, params: Dict[str, Any], parent=None):
        """
        Initialize job worker.
        
        Args:
            func: Function to execute (must accept params dict)
            params: Parameters dictionary to pass to func
            parent: Parent QObject
        """
        super().__init__(parent)
        self.func = func
        
        # Safety check: Detect Qt objects in params (unsafe for worker threads)
        self._check_for_qt_objects(params)
        
        self.params = params
        self._cancelled = False
    
    def _check_for_qt_objects(self, params: Dict[str, Any], path: str = "params") -> None:
        """
        Recursively check for Qt objects in params dictionary.
        
        Qt objects (QObject instances) should not be passed to worker threads
        as they are not thread-safe and can cause crashes or silent failures.
        
        Args:
            params: Dictionary or value to check
            path: Current path in nested structure (for error messages)
        """
        try:
            from PyQt6.QtCore import QObject
        except ImportError:
            # Qt not available, skip check
            return
        
        if isinstance(params, dict):
            for key, val in params.items():
                current_path = f"{path}.{key}"
                if isinstance(val, QObject):
                    logger.warning(
                        f"Performance Warning: Passing QObject '{current_path}' ({type(val).__name__}) "
                        f"to worker thread is unsafe! Qt objects are not thread-safe and may cause crashes. "
                        f"Consider passing only data (dicts, lists, primitives) instead."
                    )
                elif isinstance(val, (dict, list)):
                    # Recursively check nested structures
                    self._check_for_qt_objects(val, current_path)
        elif isinstance(params, list):
            for idx, val in enumerate(params):
                current_path = f"{path}[{idx}]"
                if isinstance(val, QObject):
                    logger.warning(
                        f"Performance Warning: Passing QObject '{current_path}' ({type(val).__name__}) "
                        f"to worker thread is unsafe! Qt objects are not thread-safe and may cause crashes. "
                        f"Consider passing only data (dicts, lists, primitives) instead."
                    )
                elif isinstance(val, (dict, list)):
                    # Recursively check nested structures
                    self._check_for_qt_objects(val, current_path)
    
    def run(self):
        """
        Execute the task function.
        
        Emits finished signal with result on success, error signal on failure.
        
        FREEZE PROTECTION: Logs worker start/end to detect deadlocks.
        """
        try:
            logger.info("=" * 80)
            logger.info(f"JobWorker: Starting task {self.func.__name__}")
            logger.debug("WORKER START")  # Freeze protection: marks worker start
            logger.info("=" * 80)
            logger.debug(f"  Function: {self.func}")
            logger.debug(f"  Params keys: {list(self.params.keys()) if isinstance(self.params, dict) else 'Not a dict'}")
            
            # Patch: Wrap _progress_callback in params to emit our progress signal
            def progress_wrapper(percent, message):
                self.progress.emit(percent, message)

            # Always inject _progress_callback so any payload function can report progress
            if isinstance(self.params, dict):
                self.params['_progress_callback'] = progress_wrapper

            logger.debug("  Calling function...")
            try:
                result = self.func(self.params)
                logger.debug(f"  Function returned, result type: {type(result)}")
            except Exception as e:
                logger.error(f"EXCEPTION during function call: {e}", exc_info=True)
                raise
            
            if self._cancelled:
                logger.warning(f"JobWorker: Task {self.func.__name__} was cancelled")
                logger.debug("WORKER END (cancelled)")  # Freeze protection: marks worker end
                return
            
            logger.info(f"JobWorker: Task {self.func.__name__} completed successfully")
            logger.debug("WORKER END")  # Freeze protection: marks worker end
            logger.debug(f"  Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            logger.info("=" * 80)
            self.finished.emit(result)
            
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = str(e)
            logger.error("=" * 80)
            logger.error(f"JobWorker: Task {self.func.__name__} FAILED")
            logger.error("=" * 80)
            logger.error(f"Error: {error_msg}")
            logger.error(f"Traceback:\n{tb}")
            logger.error("=" * 80)
            self.error.emit(error_msg, tb)
    
    def cancel(self):
        """Request cancellation of the task."""
        self._cancelled = True
        logger.info(f"JobWorker: Cancellation requested for {self.func.__name__}")

