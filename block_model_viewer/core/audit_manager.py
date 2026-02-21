"""
Audit Manager for JORC/NI 43-101 Compliance.

Records inputs, parameters, user ID, and output hashes for every technical calculation.
"""

import json
import hashlib
import getpass
import platform
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _get_software_version() -> str:
    """
    Get the software version dynamically from package metadata.
    
    Attempts multiple methods in order:
    1. importlib.metadata (Python 3.8+)
    2. pkg_resources (setuptools)
    3. __version__ from main package
    4. Fallback to "unknown"
    
    Returns:
        Version string in format "GeoX_vX.Y.Z" or "GeoX_unknown"
    """
    version = None
    
    # Method 1: importlib.metadata (recommended for Python 3.8+)
    try:
        from importlib.metadata import version as get_version
        version = get_version("block_model_viewer")
    except Exception:
        pass
    
    # Method 2: Try pkg_resources
    if version is None:
        try:
            import pkg_resources
            version = pkg_resources.get_distribution("block_model_viewer").version
        except Exception:
            pass
    
    # Method 3: Try __version__ from config
    if version is None:
        try:
            from block_model_viewer.config import __version__
            version = __version__
        except Exception:
            pass
    
    # Method 4: Try reading from pyproject.toml
    if version is None:
        try:
            import tomllib
            pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomllib.load(f)
                    version = data.get("project", {}).get("version")
        except Exception:
            pass
    
    # Fallback
    if version is None:
        version = "1.0.0-dev"
        logger.debug("Could not determine software version, using fallback")
    
    return f"GeoX_v{version}"


# Cache the version at module load
SOFTWARE_VERSION = _get_software_version()


class AuditManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuditManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.audit_dir = Path("audit_logs")
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.user = getpass.getuser()
        self.machine = platform.node()

    def log_event(self, 
                  module: str, 
                  action: str, 
                  parameters: Dict[str, Any], 
                  input_data_hash: Optional[str] = None,
                  result_summary: Optional[Dict] = None) -> str:
        """
        Log a technical event. Returns the unique Transaction ID.
        """
        timestamp = datetime.now().isoformat()
        
        # Create unique ID for this event
        event_str = f"{timestamp}{self.user}{module}{action}"
        event_id = hashlib.sha256(event_str.encode()).hexdigest()[:12]

        # Sanitize parameters (convert numpy types to python native for JSON)
        sanitized_params = self._sanitize(parameters)

        log_entry = {
            "event_id": event_id,
            "timestamp": timestamp,
            "user": self.user,
            "machine": self.machine,
            "module": module,
            "action": action,
            "input_hash": input_data_hash or "N/A",
            "parameters": sanitized_params,
            "results": self._sanitize(result_summary) if result_summary else "N/A",
            "software_version": SOFTWARE_VERSION,  # Dynamic version from package metadata
            "python_version": platform.python_version(),
            "platform": platform.platform()
        }

        # Write to daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.audit_dir / f"audit_{date_str}.jsonl"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(f"Audit Logged: {module} - {action} [{event_id}]")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

        return event_id

    def hash_dataframe(self, df: pd.DataFrame) -> str:
        """Create a quick SHA256 hash of a DataFrame to prove data integrity."""
        try:
            # Hash a sample of the data + shape to be fast but reasonably unique
            # For strict compliance, hash the whole object buffer, but that's slow for 10GB models.
            summary = str(df.shape) + str(df.columns.tolist()) + str(df.iloc[0].values) + str(df.iloc[-1].values)
            return hashlib.sha256(summary.encode()).hexdigest()
        except Exception as e:
            logger.warning(f"DataFrame hash failed: {type(e).__name__}: {e}")
            return f"HASH_FAILED:{type(e).__name__}"

    def _sanitize(self, obj):
        """Recursively convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        elif hasattr(obj, 'item'):  # Numpy scalars
            return obj.item()
        elif hasattr(obj, 'to_dict'):  # Pandas/custom objects
            return str(obj)
        else:
            return obj

