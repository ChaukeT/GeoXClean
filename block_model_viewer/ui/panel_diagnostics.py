"""
Panel Diagnostics & Crash Scanner (STEP 40)

Automated diagnostics for detecting panel construction issues.
"""

from __future__ import annotations

import inspect
import time
import logging
from typing import Type, Dict, Any, List, Optional, Callable
from pathlib import Path

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtCore import QTimer, QEventLoop

from .base_panel import BasePanel

logger = logging.getLogger(__name__)


class PanelDiagnosticsRunner:
    """
    Runs diagnostics on all UI panels to detect construction issues.
    """
    
    def __init__(
        self,
        main_window_factory: Optional[Callable[[], QMainWindow]] = None,
        timeout_s: float = 5.0,
        slow_threshold_s: float = 1.0
    ):
        """
        Initialize diagnostics runner.
        
        Args:
            main_window_factory: Optional factory function to create a main window
            timeout_s: Timeout in seconds for panel construction
            slow_threshold_s: Threshold in seconds for marking panels as slow
        """
        self.timeout_s = timeout_s
        self.slow_threshold_s = slow_threshold_s
        self.main_window_factory = main_window_factory
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def discover_panel_classes(self) -> List[Type[BasePanel]]:
        """
        Import all modules under block_model_viewer.ui and return a list of BasePanel subclasses.
        
        Excludes abstract/base classes.
        
        Returns:
            List of panel classes
        """
        import pkgutil
        import importlib
        import block_model_viewer.ui as ui_pkg
        
        panel_types: List[Type[BasePanel]] = []
        
        # Get all modules in the ui package
        for importer, modname, ispkg in pkgutil.iter_modules(ui_pkg.__path__, ui_pkg.__name__ + "."):
            if modname.startswith("_") or modname.endswith("._"):
                continue
            
            try:
                module = importlib.import_module(modname)
                
                # Find all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Check if it's a subclass of BasePanel
                    if not issubclass(obj, BasePanel):
                        continue
                    
                    # Skip BasePanel itself and abstract classes
                    if obj is BasePanel:
                        continue
                    
                    # Skip if it's an abstract base class (has ABCMeta)
                    if inspect.isabstract(obj):
                        continue
                    
                    # Skip if already added
                    if obj not in panel_types:
                        panel_types.append(obj)
            except Exception as e:
                logger.warning(f"Failed to import module {modname}: {e}")
                continue
        
        return panel_types
    
    def _make_dummy_main_window(self) -> QMainWindow:
        """Create a dummy main window for testing."""
        if self.main_window_factory is not None:
            try:
                return self.main_window_factory()
            except Exception as e:
                logger.warning(f"Failed to create main window via factory: {e}")
        
        # Fallback: create minimal QMainWindow
        return QMainWindow()
    
    def _make_dummy_controller(self):
        """Create a dummy controller for panels that require it."""
        class DummyController:
            def __init__(self):
                self.current_block_model = None
                self.scenario_store = None
                self.scenario_runner = None
            
            def run_task(self, *args, **kwargs):
                pass
            
            def refresh_block_model_view(self):
                pass
        
        return DummyController()
    
    def run(self) -> None:
        """
        Run diagnostics for all discovered panel classes.
        """
        logger.info("Starting panel diagnostics...")
        panel_classes = self.discover_panel_classes()
        logger.info(f"Discovered {len(panel_classes)} panel classes")
        
        for cls in panel_classes:
            self._test_panel_class(cls)
        
        logger.info("Panel diagnostics complete")
    
    def _test_panel_class(self, cls: Type[BasePanel]) -> None:
        """
        Test a single panel class.
        
        Args:
            cls: Panel class to test
        """
        name = getattr(cls, "PANEL_ID", cls.__name__)
        start = time.perf_counter()
        status = "OK"
        error: Optional[str] = None
        
        window: Optional[QMainWindow] = None
        panel_instance: Optional[BasePanel] = None
        
        try:
            # Create dummy main window
            window = self._make_dummy_main_window()
            
            # Create event loop for timeout handling
            loop = QEventLoop()
            timed_out = False
            
            def timeout():
                nonlocal timed_out
                timed_out = True
                loop.quit()
            
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(timeout)
            timer.start(int(self.timeout_s * 1000))
            
            # Try to instantiate panel
            try:
                # Create panel instance
                panel_instance = cls(parent=window)
                
                # Try to bind dummy controller if method exists
                if hasattr(panel_instance, 'bind_controller'):
                    dummy_controller = self._make_dummy_controller()
                    panel_instance.bind_controller(dummy_controller)
                
                # Process events briefly to catch deferred initialization
                QApplication.processEvents()
                
                # Wait for timeout or completion
                if not timed_out:
                    # Process events for a short time
                    QTimer.singleShot(100, loop.quit)
                    loop.exec()
                
            except Exception as exc:
                status = "EXCEPTION"
                error = repr(exc)
                timer.stop()
                elapsed = time.perf_counter() - start
                
                self.results[name] = {
                    "status": status,
                    "error": error,
                    "time_s": elapsed,
                }
                logger.error(f"Panel {name} raised exception: {error}")
                return
            
            elapsed = time.perf_counter() - start
            
            # Determine status
            if timed_out:
                status = "HANG"
                error = f"Timeout after {self.timeout_s}s"
            elif elapsed > self.slow_threshold_s:
                status = "SLOW"
            else:
                status = "OK"
            
            self.results[name] = {
                "status": status,
                "error": error,
                "time_s": elapsed,
            }
            
            # Cleanup
            timer.stop()
            if panel_instance is not None:
                try:
                    panel_instance.deleteLater()
                except Exception:
                    pass
            
            if window is not None:
                try:
                    window.close()
                except Exception:
                    pass
                    
        except Exception as e:
            # Outer exception handler for unexpected errors
            elapsed = time.perf_counter() - start
            status = "EXCEPTION"
            error = repr(e)
            self.results[name] = {
                "status": status,
                "error": error,
                "time_s": elapsed,
            }
            logger.error(f"Unexpected error testing panel {name}: {error}")
    
    def report_as_text(self) -> str:
        """
        Generate a human-readable text report.
        
        Returns:
            Formatted report string
        """
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("Panel Diagnostics Report")
        lines.append("=" * 60)
        lines.append("")
        
        # Group by status
        by_status: Dict[str, List[tuple[str, Dict[str, Any]]]] = {
            "OK": [],
            "SLOW": [],
            "HANG": [],
            "EXCEPTION": [],
        }
        
        for name, info in sorted(self.results.items()):
            status = info["status"]
            if status in by_status:
                by_status[status].append((name, info))
        
        # Print by status
        for status in ["OK", "SLOW", "HANG", "EXCEPTION"]:
            items = by_status[status]
            if items:
                lines.append(f"{status} ({len(items)} panels):")
                for name, info in items:
                    t = info["time_s"]
                    t_str = f"{t:.2f}s" if t is not None else "-"
                    if info["error"]:
                        lines.append(f"  {name}: {t_str} - {info['error']}")
                    else:
                        lines.append(f"  {name}: {t_str}")
                lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def report_to_file(self, log_path: Optional[Path] = None) -> None:
        """
        Write report to log file.
        
        Args:
            log_path: Optional path to log file
        """
        if log_path is None:
            from PyQt6.QtCore import QStandardPaths
            config_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
            log_dir = Path(config_dir) / "BlockModelViewer" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "panel_diagnostics.log"
        
        report_text = self.report_as_text()
        
        try:
            with open(log_path, 'w') as f:
                f.write(report_text)
                f.write("\n\n")
                # Also write detailed results
                f.write("Detailed Results:\n")
                f.write("-" * 60 + "\n")
                for name, info in sorted(self.results.items()):
                    f.write(f"{name}:\n")
                    for key, value in info.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
            
            logger.info(f"Diagnostics report written to {log_path}")
        except Exception as e:
            logger.error(f"Failed to write diagnostics report to file: {e}")

