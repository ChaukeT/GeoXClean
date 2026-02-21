"""
Main application entry point for GeoX.
"""

import sys
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import numbers

# Set matplotlib backend ONCE before any matplotlib imports
# This must be done before any module imports matplotlib to prevent backend conflicts
try:
    import matplotlib
    matplotlib.use('QtAgg')  # Use QtAgg for PyQt6 compatibility
except ImportError:
    pass  # Matplotlib not available, skip backend setup

# GUI imports are intentionally done inside main() to avoid importing GUI
# libraries (PyQt) at module-import time. This prevents child processes
# spawned via multiprocessing (spawn start method on Windows) from
# re-importing GUI modules and re-initializing the application.

# Set environment variables for better compatibility
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['QT_SCALE_FACTOR'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
# Suppress Qt warnings about QWidgetWindow (harmless warnings from embedded widgets)
os.environ['QT_LOGGING_RULES'] = 'qt.qpa.window.warning=false'

# Configure logging to work when packaged
def setup_logging():
    """Setup logging with fallback for packaged executable."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    # Setup session logging for geological modeling (to modeling_logs/)
    # SECURITY: Use secure directory creation
    modeling_log_dir = Path("modeling_logs")
    try:
        modeling_log_dir.mkdir(exist_ok=True, mode=0o755)  # Restrictive permissions
    except (PermissionError, OSError) as e:
        print(f"Warning: Could not create modeling log directory: {e}")
        modeling_log_dir = None
    
    if modeling_log_dir:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        modeling_log_path = modeling_log_dir / f"geox_session_{timestamp}.log"
        
        # Add modeling session log handler
        try:
            modeling_handler = logging.FileHandler(modeling_log_path, mode='w', encoding='utf-8')
            modeling_handler.setLevel(logging.DEBUG)  # Capture all details for modeling
            # Set restrictive permissions on log file
            os.chmod(modeling_log_path, 0o600)  # Owner read/write only
            handlers.append(modeling_handler)
            print(f"Modeling logs: {modeling_log_path.absolute()}")
        except Exception as e:
            print(f"Warning: Could not create modeling log file: {e}")
    
    # Setup main application log file in %LOCALAPPDATA%/GeoX
    # Prefer a deterministic log file in %LOCALAPPDATA%/GeoX so packaged apps
    # always write logs to a known, writable location regardless of CWD.
    try:
        local_appdata = os.getenv('LOCALAPPDATA') or os.getenv('APPDATA')
        if local_appdata:
            log_dir = Path(local_appdata) / 'GeoX'
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / 'block_model_viewer.log'
            rotating_handler = RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,  # 10 MB per file
                backupCount=5,  # Keep 5 backup files
                encoding='utf-8',
                errors='replace'
            )
            handlers.append(rotating_handler)
        else:
            # Fallback to current directory then home
            log_path = Path('block_model_viewer.log')
            rotating_handler = RotatingFileHandler(
                log_path, maxBytes=10*1024*1024, backupCount=5,
                encoding='utf-8', errors='replace'
            )
            handlers.append(rotating_handler)
    except (PermissionError, OSError):
        try:
            # Fallback to user's home directory
            log_path = Path.home() / 'block_model_viewer.log'
            rotating_handler = RotatingFileHandler(
                log_path, maxBytes=10*1024*1024, backupCount=5,
                encoding='utf-8', errors='replace'
            )
            handlers.append(rotating_handler)
        except Exception:
            # If all else fails, just use console logging
            pass
    
    # Use a safe formatter that prevents recursion and handles Unicode encoding errors
    class SafeFormatter(logging.Formatter):
        """Formatter that prevents recursive logging crashes and handles Unicode encoding errors."""

        def _safe_string(self, value, limit=400):
            try:
                text = str(value)
                if len(text) > limit:
                    text = text[:limit] + "..."
                # Handle Unicode encoding errors by replacing problematic characters
                try:
                    # Try to encode to check if it's safe
                    text.encode('cp1252', errors='strict')
                except UnicodeEncodeError:
                    # Replace problematic characters with safe alternatives
                    text = text.encode('cp1252', errors='replace').decode('cp1252')
                return text
            except RecursionError:
                return "<RecursionError>"
            except UnicodeEncodeError:
                # Fallback: replace all non-ASCII characters
                try:
                    return str(value).encode('ascii', errors='replace').decode('ascii')
                except Exception:
                    return "<UnicodeError>"
            except Exception as exc:
                return f"<Unprintable {type(value).__name__}: {exc}>"

        def format(self, record):
            if getattr(record, "_is_formatting", False):
                levelname = getattr(record, "levelname", "UNKNOWN")
                name = getattr(record, "name", "unknown")[:50]
                return f"<RecursionError: {levelname} - {name}>"
            try:
                record._is_formatting = True
                original_msg, original_args = record.msg, record.args
                original_exc_info = record.exc_info
                
                # For DEBUG logs with exc_info=True, suppress traceback formatting to prevent FormatError
                if record.levelno == logging.DEBUG and record.exc_info:
                    record.exc_info = None
                    record.exc_text = None
                
                try:
                    if original_args:
                        # Convert any Exception objects in args to strings before formatting.
                        # Preserve numeric values so '%0.2f' and similar specifiers still work.
                        safe_args_list = []
                        for arg in original_args:
                            if isinstance(arg, Exception):
                                try:
                                    safe_args_list.append(str(arg))
                                except Exception:
                                    safe_args_list.append(f"<{type(arg).__name__}>")
                            elif isinstance(arg, numbers.Number):
                                safe_args_list.append(arg)
                            else:
                                safe_args_list.append(self._safe_string(arg))
                        record.msg = self._safe_string(original_msg)
                        record.args = tuple(safe_args_list)
                    else:
                        record.msg = self._safe_string(original_msg)
                except RecursionError:
                    levelname = getattr(record, "levelname", "UNKNOWN")
                    name = getattr(record, "name", "unknown")[:50]
                    return f"<RecursionError: {levelname} - {name}>"
                except UnicodeEncodeError:
                    # Handle Unicode errors in message formatting
                    record.msg = "<UnicodeError in message>"
                    record.args = ()
                except Exception as exc:
                    # If formatting fails, try to get a safe message
                    try:
                        exc_type = type(exc).__name__
                    except Exception:
                        exc_type = "Unknown"
                    record.msg = f"<FormatError {exc_type}>"
                    record.args = ()
                    # Ensure exc_info is None to prevent further formatting attempts
                    record.exc_info = None
                    record.exc_text = None
                
                # Format the record, but catch any errors during super().format()
                try:
                    formatted = super().format(record)
                except Exception as format_exc:
                    # If super().format() fails, return a safe message
                    try:
                        levelname = getattr(record, "levelname", "UNKNOWN")
                        name = getattr(record, "name", "unknown")[:50]
                        format_exc_type = type(format_exc).__name__
                        return f"<FormatError in formatter: {levelname} - {name} - {format_exc_type}>"
                    except Exception:
                        return "<FormatError in formatter>"
                
                record.msg, record.args = original_msg, original_args
                record.exc_info = original_exc_info
                return formatted
            except RecursionError:
                levelname = getattr(record, "levelname", "UNKNOWN")
                name = getattr(record, "name", "unknown")[:50]
                return f"<RecursionError: {levelname} - {name}>"
            except UnicodeEncodeError:
                # Final fallback for Unicode errors
                levelname = getattr(record, "levelname", "UNKNOWN")
                return f"<UnicodeError: {levelname}>"
            except Exception as exc:
                levelname = getattr(record, "levelname", "UNKNOWN")
                # Try to get exception type name safely
                try:
                    exc_type_name = type(exc).__name__
                except Exception:
                    exc_type_name = "Unknown"
                return f"<FormatError: {levelname} - {exc_type_name}>"
            finally:
                if hasattr(record, "_is_formatting"):
                    delattr(record, "_is_formatting")
    
    # Apply safe formatter to all handlers
    formatter = SafeFormatter(
        '%(asctime)s [pid:%(process)d] [%(threadName)s] - %(name)s - %(levelname)s - %(message)s'
    )
    for handler in handlers:
        handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG for detailed troubleshooting
        handlers=handlers
    )

setup_logging()
logger = logging.getLogger(__name__)

# Log startup banner with file locations
logger.info("=" * 80)
logger.info("GeoX - Geological & Mining Block Model Viewer")
logger.info("=" * 80)
logger.info(f"Modeling logs directory: {Path('modeling_logs').absolute()}")
logger.info(f"Console and file logging active")
logger.info("=" * 80)

# Reduce extremely verbose matplotlib font_manager debug spam (keeping app logs readable)
try:
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
except Exception:
    # Non-fatal if logger hierarchy differs in some environments
    pass

# Suppress extremely verbose Numba JIT compilation debug logs
# These flood the terminal during first-run compilation and add overhead
try:
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numba.core').setLevel(logging.WARNING)
    logging.getLogger('numba.core.byteflow').setLevel(logging.WARNING)
    logging.getLogger('numba.core.interpreter').setLevel(logging.WARNING)
except Exception:
    pass


def main(headless=False):
    """
    Main application entry point.
    
    Args:
        headless: If True, run in headless mode (no UI, only test imports/initialization)
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Support headless mode and diagnostics via parameter or command-line argument
    if not headless:
        import argparse
        parser = argparse.ArgumentParser(description="GeoX")
        parser.add_argument("--headless-test", action="store_true", 
                           help="Run headless smoke test without UI")
        parser.add_argument("--diagnose-panels", action="store_true",
                           help="Run panel diagnostics and exit")
        args = parser.parse_args()
        headless = args.headless_test
        
        # STEP 40: Run panel diagnostics if requested
        if args.diagnose_panels:
            logger.info("Running panel diagnostics mode...")
            from PyQt6.QtWidgets import QApplication
            from .ui.panel_diagnostics import PanelDiagnosticsRunner
            
            # Create QApplication for diagnostics
            app = QApplication(sys.argv)
            app.setApplicationName("GeoX")
            
            # Create diagnostics runner with MainWindow factory
            try:
                from .ui.main_window import MainWindow
                runner = PanelDiagnosticsRunner(main_window_factory=lambda: MainWindow())
            except Exception as e:
                logger.warning(f"Could not create MainWindow factory: {e}")
                runner = PanelDiagnosticsRunner()
            
            # Run diagnostics
            runner.run()
            
            # Generate and print report
            report_text = runner.report_as_text()
            logger.info("Panel diagnostics:\n%s", report_text)
            print(report_text)
            
            # Write to log file
            runner.report_to_file()
            
            return 0
    
    if headless:
        # Run headless mode: only test imports, controllers, shortcuts
        logger.info("Running headless mode...")
        try:
            # Test shortcuts initialization
            from .ui.shortcuts import Shortcuts
            shortcuts = Shortcuts.get_all()
            logger.info(f"Shortcuts initialized: {len(shortcuts)} shortcuts loaded")
            
            # Test controller initialization (requires renderer, create minimal mock)
            from .controllers.app_controller import AppController
            from .visualization.renderer import Renderer
            # Create renderer in offscreen mode
            import os
            os.environ['PYVISTA_OFF_SCREEN'] = 'true'
            renderer = Renderer()
            controller = AppController(renderer)
            from .controllers.job_registry import JobRegistry
            JobRegistry.initialize(controller)
            tasks = JobRegistry.list_tasks()
            logger.info(f"Controller initialized: {len(tasks)} jobs registered")
            
            # Test basic model imports
            from .models.block_model import BlockModel
            import numpy as np
            import pandas as pd
            
            # Test basic model creation
            from .models.block_model import BlockModel, BlockMetadata
            metadata = BlockMetadata()
            block_model = BlockModel(metadata)
            assert block_model is not None
            logger.info("Headless test passed: All core components initialized successfully")
            return 0
        except Exception as e:
            logger.error(f"Headless test failed: {e}", exc_info=True)
            return 1
    
    try:
        logger.info("Initializing 3D Block Model Viewer application")
        # Import GUI modules lazily to prevent them from being imported by
        # child processes spawned by multiprocessing (Windows spawn method).
        # This avoids re-initialization of GUI subsystems in workers.
        from PyQt6.QtWidgets import QApplication, QMessageBox
        from PyQt6.QtGui import QSurfaceFormat
        from PyQt6.QtCore import Qt
        from .ui.main_window import MainWindow
        
        # Prefer a compatibility OpenGL context to support older GPUs/drivers
        # Use 32-bit depth buffer for better precision and reduced z-fighting on geological surfaces
        try:
            fmt = QSurfaceFormat()
            fmt.setVersion(2, 1)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
            fmt.setDepthBufferSize(32)  # 32-bit depth for better precision (was 24)
            fmt.setStencilBufferSize(8)  # Enable stencil buffer for advanced rendering
            QSurfaceFormat.setDefaultFormat(fmt)
        except Exception:
            pass

        # Create QApplication
        app = QApplication(sys.argv)
        # Ensure the app does not quit when a dialog window is closed
        try:
            app.setQuitOnLastWindowClosed(False)
        except Exception:
            pass
        app.setApplicationName("GeoX")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("GeoX")
        
        # Set application style
        app.setStyle('Fusion')
        
        # 1. Initialize Core Services (The Dependency Injection Container)
        # We create the specific instance here, AFTER QApplication exists (for Qt signals)
        from .core.data_registry import DataRegistry
        registry = DataRegistry()
        logger.info("DataRegistry initialized via dependency injection")
        
        # Show splash screen
        from .ui.splash_screen import SplashScreen
        from PyQt6.QtCore import Qt, QTimer
        splash = SplashScreen()
        splash.show()
        splash.show_message("Loading BlockModelViewer...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        
        logger.info("Creating main window...")
        
        # Use QTimer.singleShot to chain initialization steps safely
        # This prevents user interaction during initialization by deferring work to the event loop
        main_window = None
        initialization_error = None
        
        def step1_create_window():
            """Step 1: Create MainWindow instance."""
            nonlocal main_window, initialization_error
            try:
                logger.info("Creating MainWindow instance...")
                splash.show_message("Initializing UI components...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
                main_window = MainWindow(registry=registry)
                logger.info("MainWindow instance created")
                # Chain to next step
                QTimer.singleShot(0, step2_show_window)
            except Exception as e:
                logger.error(f"Failed to create main window: {e}", exc_info=True)
                initialization_error = e
                QTimer.singleShot(0, step_error)
        
        def step2_show_window():
            """Step 2: Show window and activate."""
            nonlocal initialization_error
            try:
                logger.info("Calling main_window.show()...")
                main_window.show()
                logger.info("main_window.show() completed")
                
                logger.info("Raising and activating window...")
                main_window.raise_()
                main_window.activateWindow()
                logger.info("Window raised and activated")
                
                # Chain to final step
                QTimer.singleShot(0, step3_finish_splash)
            except Exception as e:
                logger.error(f"Failed to show main window: {e}", exc_info=True)
                initialization_error = e
                QTimer.singleShot(0, step_error)
        
        def step3_finish_splash():
            """Step 3: Finish splash screen."""
            try:
                logger.info("Finishing splash screen...")
                splash.finish(main_window)
                logger.info("Splash screen finished successfully")
                logger.info("Main window created and shown successfully")
                
                # Pre-compile Numba kernels in background thread to avoid delay
                # when user first runs kriging
                import threading
                def _precompile_kernels():
                    try:
                        # Universal Kriging
                        from .geostats.universal_kriging import precompile_uk_kernels
                        precompile_uk_kernels()
                    except Exception as e:
                        logger.debug(f"UK kernel pre-compilation skipped: {e}")
                    
                    try:
                        # Indicator Kriging
                        from .geostats.indicator_kriging import precompile_ik_kernels
                        precompile_ik_kernels()
                    except Exception as e:
                        logger.debug(f"IK kernel pre-compilation skipped: {e}")
                    
                    try:
                        # Co-Kriging
                        from .geostats.cokriging3d import precompile_ck_kernels
                        precompile_ck_kernels()
                    except Exception as e:
                        logger.debug(f"CK kernel pre-compilation skipped: {e}")
                
                precompile_thread = threading.Thread(target=_precompile_kernels, daemon=True)
                precompile_thread.start()
                
            except Exception as e:
                logger.error(f"Error finishing splash screen: {e}", exc_info=True)
                # Continue anyway - splash screen error shouldn't block app
        
        def step_error():
            """Handle initialization error."""
            try:
                splash.close()
            except Exception:
                pass
            QMessageBox.critical(
                None,
                "Initialization Error",
                f"Failed to initialize the application window:\n\n{str(initialization_error)}\n\n"
                f"This may be due to:\n"
                f"• Missing or outdated graphics drivers\n"
                f"• Incompatible OpenGL version\n"
                f"• Missing system libraries\n\n"
                f"Please check the log file for details."
            )
            # Quit application instead of raising (safer during event loop)
            app.quit()
            sys.exit(1)
        
        # Start initialization chain
        QTimer.singleShot(0, step1_create_window)
        
        logger.info("Starting application event loop")
        
        # Install global exception handler for crash protection
        from .core.crash_handler import install_exception_handler
        install_exception_handler()
        logger.info("Global exception handler installed")
        
        # Run application
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Application failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # On Windows, frozen executables that use multiprocessing need freeze_support()
    # to avoid child processes re-importing and re-initializing GUI code.
    import multiprocessing

    try:
        multiprocessing.freeze_support()
    except Exception:
        # Non-fatal: freeze_support may not be necessary in some environments
        pass

    main()
