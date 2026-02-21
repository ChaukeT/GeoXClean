"""
System Configuration Manager

Provides centralized system configuration management.
Supports configuration profiles, environment-specific settings, and dynamic configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class SystemConfig:
    """System configuration."""
    config_id: str
    environment: Environment
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Application settings
    app_name: str = "Block Model Viewer"
    app_version: str = "2.0"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Database settings
    database_path: Optional[str] = None
    backup_directory: Optional[str] = None
    max_backup_age_days: int = 30
    
    # Performance settings
    max_table_rows: int = 10000
    chunk_size: int = 500
    enable_virtual_scrolling: bool = True
    cache_size_mb: int = 512
    
    # QC settings
    default_qc_config: Optional[str] = None
    auto_run_qc_on_load: bool = False
    auto_fix_enabled: bool = True
    
    # UI settings
    theme: str = "dark"
    language: str = "en"
    show_tooltips: bool = True
    show_status_bar: bool = True
    
    # Security settings (simplified for local desktop app)
    enable_audit_logging: bool = True  # Audit trail for compliance
    
    # Export/Import settings
    default_export_format: str = "csv"
    default_import_template: Optional[str] = None
    
    # Advanced settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        data = asdict(self)
        data["environment"] = self.environment.value
        data["created_date"] = self.created_date.isoformat()
        data["last_modified"] = self.last_modified.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create config from dictionary."""
        data = data.copy()
        data["environment"] = Environment(data["environment"])
        data["created_date"] = datetime.fromisoformat(data["created_date"])
        data["last_modified"] = datetime.fromisoformat(data["last_modified"])
        return cls(**data)


class SystemConfigManager:
    """
    System configuration manager.
    
    Manages system-wide configuration with environment support.
    """
    
    def __init__(self, config_directory: Optional[Path] = None):
        if config_directory is None:
            config_directory = Path.home() / ".geox" / "config"
        
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(parents=True, exist_ok=True)
        
        # Detect environment
        env_str = os.getenv("GEOX_ENV", "development").lower()
        try:
            self.current_environment = Environment(env_str)
        except ValueError:
            self.current_environment = Environment.DEVELOPMENT
            logger.warning(f"Unknown environment '{env_str}', defaulting to development")
        
        self.current_config: Optional[SystemConfig] = None
        self._load_config()
        
        logger.info(f"SystemConfigManager initialized (environment: {self.current_environment.value})")
    
    def _load_config(self):
        """Load configuration for current environment."""
        config_file = self.config_directory / f"system_config_{self.current_environment.value}.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    data = json.load(f)
                self.current_config = SystemConfig.from_dict(data)
                logger.info(f"Loaded system config from {config_file}")
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
                self.current_config = self._create_default_config()
        else:
            self.current_config = self._create_default_config()
            self._save_config()
    
    def _create_default_config(self) -> SystemConfig:
        """Create default configuration."""
        return SystemConfig(
            config_id=f"config-{self.current_environment.value}",
            environment=self.current_environment,
            debug_mode=(self.current_environment == Environment.DEVELOPMENT),
            log_level="DEBUG" if self.current_environment == Environment.DEVELOPMENT else "INFO",
            database_path=str(Path.home() / ".geox" / "databases" / "default.db"),
            backup_directory=str(Path.home() / ".geox" / "backups"),
        )
    
    def get_config(self) -> SystemConfig:
        """Get current system configuration."""
        if self.current_config is None:
            self.current_config = self._create_default_config()
        return self.current_config
    
    def update_config(self, **kwargs) -> bool:
        """
        Update configuration values.
        
        Args:
            **kwargs: Configuration values to update
        
        Returns:
            True if successful, False otherwise
        """
        if self.current_config is None:
            self.current_config = self._create_default_config()
        
        try:
            for key, value in kwargs.items():
                if hasattr(self.current_config, key):
                    setattr(self.current_config, key, value)
                else:
                    # Store in custom_settings
                    self.current_config.custom_settings[key] = value
            
            self.current_config.last_modified = datetime.now()
            self._save_config()
            
            logger.info(f"Updated system config: {list(kwargs.keys())}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating config: {e}", exc_info=True)
            return False
    
    def _save_config(self):
        """Save current configuration to file."""
        if self.current_config is None:
            return
        
        config_file = self.config_directory / f"system_config_{self.current_environment.value}.json"
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.current_config.to_dict(), f, indent=2)
            logger.debug(f"Saved system config to {config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}", exc_info=True)
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults."""
        self.current_config = self._create_default_config()
        self._save_config()
        logger.info("Reset system config to defaults")
        return True
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration setting.
        
        Args:
            key: Setting key
            default: Default value if not found
        
        Returns:
            Setting value or default
        """
        if self.current_config is None:
            return default
        
        if hasattr(self.current_config, key):
            return getattr(self.current_config, key)
        else:
            return self.current_config.custom_settings.get(key, default)
    
    def set_setting(self, key: str, value: Any) -> bool:
        """
        Set a configuration setting.
        
        Args:
            key: Setting key
            value: Setting value
        
        Returns:
            True if successful, False otherwise
        """
        return self.update_config(**{key: value})


# Global config manager instance
_config_manager: Optional[SystemConfigManager] = None


def get_system_config_manager() -> SystemConfigManager:
    """Get the global system config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = SystemConfigManager()
    return _config_manager

