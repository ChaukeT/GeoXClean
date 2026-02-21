"""
GeoX Session Logger - Automatic Log File Creation
==================================================

Ensures all geological modeling logs are captured to files.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_session_logging(log_dir: str = "modeling_logs"):
    """
    Setup comprehensive logging to both console and file.
    
    Creates a timestamped log file for each GeoX session.
    
    Args:
        log_dir: Directory to store log files (default: "modeling_logs")
    """
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"geox_session_{timestamp}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Capture more detail in file
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)-40s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("GeoX Session Logging Started")
    logger.info(f"Log file: {log_file.absolute()}")
    logger.info("=" * 80)
    
    return str(log_file.absolute())


def flush_logs():
    """Flush all logging handlers to ensure logs are written to disk."""
    for handler in logging.getLogger().handlers:
        handler.flush()


# Auto-setup when imported
if __name__ != "__main__":
    # Only auto-setup if running as part of GeoX, not as script
    try:
        setup_session_logging()
    except Exception as e:
        print(f"Warning: Could not setup session logging: {e}")

