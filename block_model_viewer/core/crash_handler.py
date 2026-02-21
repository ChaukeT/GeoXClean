"""
Global Exception Handler.

Prevents Silent Crashes (CTD) and logs tracebacks to the Audit System.
"""

import sys
import traceback
import logging
from PyQt6.QtWidgets import QMessageBox, QApplication
from .audit_manager import AuditManager

logger = logging.getLogger(__name__)


def install_exception_handler():
    """Call this in main.py before app.exec()"""
    sys.excepthook = handle_exception


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Global catch for unhandled exceptions.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 1. Format the error
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    short_msg = f"{exc_type.__name__}: {exc_value}"

    logger.critical(f"Uncaught Exception: {short_msg}\n{error_msg}")

    # 2. Log to Audit System (Critical for debugging production issues)
    try:
        AuditManager().log_event(
            module="SYSTEM",
            action="CRASH",
            parameters={"error_type": exc_type.__name__},
            result_summary={"traceback": error_msg}
        )
    except Exception as audit_error:
        # Fallback: print to stderr if audit logging fails
        print(f"CRITICAL: Crash handler audit failed: {audit_error}", file=sys.stderr)
        print(f"Original exception: {exc_value}", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)
        # Don't re-raise - we're already handling a crash

    # 3. Show GUI Dialog (if UI exists)
    app = QApplication.instance()
    if app:
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("GeoX Application Error")
        msg_box.setText("An unexpected error occurred.")
        msg_box.setInformativeText(short_msg)
        msg_box.setDetailedText(error_msg)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Close)
        msg_box.exec()

