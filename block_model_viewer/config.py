"""
Configuration management for GeoX.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import os

try:
    import tomli
    import tomli_w
    HAS_TOML = True
except ImportError:
    HAS_TOML = False

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for the application.
    
    Handles saving and loading of user preferences and settings.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file (uses default if None)
        """
        if config_file is None:
            # Use default config file in user's home directory
            config_dir = Path.home() / '.geox'
            config_dir.mkdir(exist_ok=True)
            # Use TOML if available, otherwise JSON
            if HAS_TOML:
                config_file = config_dir / 'config.toml'
            else:
                config_file = config_dir / 'config.json'
        
        self.config_file = config_file
        self.config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            'ui': {
                'window_width': 1200,
                'window_height': 800,
                'splitter_position': [800, 200],
                'last_directory': str(Path.home()),
                'recent_files': [],
                'max_recent_files': 10,
                'show_axes': True,
                'show_bounds': True,
                'background_color': 'white',
                'edge_color': 'black',
                'edge_width': 0.5
            },
            'display': {
                'axis_font_family': 'Arial',
                'axis_font_size': 12,
                'axis_font_color': '#000000',  # Black
                'axis_label_font_size': 14,
                'show_grid': True,
                'grid_opacity': 0.3,
                'opacity': 1.0,
                'edge_color': 'black',
                'background': 'lightgrey',
                'lighting': True,
                'legend_font_size': 13
            },
            'scene': {
                'projection': 'perspective',  # 'perspective' or 'orthographic'
                'last_view': 'isometric',
                'show_axes': True,
                'show_legend': True,
                'show_grid': False,
                'trackball_mode': False
            },
            'visualization': {
                'default_colormap': 'viridis',
                'default_transparency': 1.0,
                'screenshot_resolution': [1920, 1080],
                'export_format': 'csv'
            },
            'data_analysis': {
                'active_property': '',
                'filter_property': '',
                'filter_min': 0.0,
                'filter_max': 100.0,
                'slice_axis': 'X',
                'slice_position': 0.0,
                'slice_keep': 'Above',
                'bins': 20
            },
            'parsing': {
                'csv_delimiter': 'auto',
                'csv_encoding': 'utf-8',
                'max_file_size_mb': 1000,
                'auto_detect_format': True
            },
            'resource_classification': {
                'Measured': {
                    'max_dist': 50.0,
                    'min_holes': 3
                },
                'Indicated': {
                    'max_dist': 100.0,
                    'min_holes': 2
                },
                'Inferred': {
                    'max_dist': 150.0,
                    'min_holes': 1
                }
            },
            'variogram': {
                # Determinism settings
                'random_state': 42,  # Default seed for reproducibility (JORC/SAMREC)
                'is_deterministic': True,  # Enforce deterministic calculations

                # Subsampling thresholds
                'max_directional_samples': 1500,  # Max samples for directional variograms
                'pair_cap': 200000,  # Maximum pairs for omnidirectional calculations
                'warn_subsample_fraction': 0.7,  # Warn if using less than this fraction

                # Weak direction thresholds (industry standards)
                'min_pairs_per_lag': 30,  # Minimum for reliable fitting
                'weak_threshold': 50,  # Below this is "concerning"
                'critical_lags_with_pairs': 3,  # Minimum lags needed

                # Sill constraint behavior
                'inherit_omni_sill_for_weak': True,  # Force weak dirs to use omni sill
                'sill_cap_multiplier': 1.3,  # Allow 30% above reference sill

                # Default lag parameters
                'default_n_lags': 12,
                'default_lag_tolerance_fraction': 0.3,  # 30% of lag distance
                'default_cone_tolerance': 15.0,  # Degrees
            },
            'mouse': {
                'interaction_style': 'trackball',  # 'trackball' | 'rubberband_zoom' | 'rubberband_pick'
                'invert_wheel': False,
                'wheel_zoom_factor': 1.2,  # base zoom factor per wheel step
                'double_click_fit_view': True
            },
            'recent_files': [],
            'version': '1.0.0'
        }
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_file.exists():
                if self.config_file.suffix == '.toml' and HAS_TOML:
                    with open(self.config_file, 'rb') as f:
                        loaded_config = tomli.load(f)
                else:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        loaded_config = json.load(f)
                
                # Merge with default config (preserve defaults for new keys)
                self._merge_config(self.config, loaded_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                logger.info("No existing configuration file, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def save_config(self) -> None:
        """Save configuration to file."""
        try:
            # Ensure directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config_file.suffix == '.toml' and HAS_TOML:
                with open(self.config_file, 'wb') as f:
                    tomli_w.dump(self.config, f)
            else:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _merge_config(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> None:
        """Recursively merge loaded config with default config."""
        for key, value in loaded.items():
            if key in default:
                if isinstance(value, dict) and isinstance(default[key], dict):
                    self._merge_config(default[key], value)
                else:
                    default[key] = value
            else:
                default[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the value (e.g., 'ui.window_width')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        config[keys[-1]] = value
    
    def add_recent_file(self, file_path: str) -> None:
        """
        Add a file to the recent files list.
        
        Args:
            file_path: Path to the file
        """
        recent_files = self.get('recent_files', [])
        
        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add to beginning
        recent_files.insert(0, file_path)
        
        # Limit to 10 files
        recent_files = recent_files[:10]
        
        self.set('recent_files', recent_files)
    
    def get_recent_files(self) -> list:
        """Get list of recent files."""
        return self.get('recent_files', [])
    
    def clear_recent_files(self) -> None:
        """Clear the recent files list."""
        self.set('recent_files', [])
    
    def get_last_directory(self) -> str:
        """Get the last used directory."""
        return self.get('ui.last_directory', str(Path.home()))
    
    def set_last_directory(self, directory: str) -> None:
        """Set the last used directory."""
        self.set('ui.last_directory', directory)
    
    def get_window_geometry(self) -> Dict[str, int]:
        """Get window geometry settings."""
        return {
            'width': self.get('ui.window_width', 1200),
            'height': self.get('ui.window_height', 800)
        }
    
    def set_window_geometry(self, width: int, height: int) -> None:
        """Set window geometry settings."""
        self.set('ui.window_width', width)
        self.set('ui.window_height', height)
    
    def get_splitter_position(self) -> list:
        """Get splitter position."""
        return self.get('ui.splitter_position', [800, 200])
    
    def set_splitter_position(self, position: list) -> None:
        """Set splitter position."""
        self.set('ui.splitter_position', position)
    
    def get_visualization_settings(self) -> Dict[str, Any]:
        """Get visualization settings."""
        return {
            'colormap': self.get('visualization.default_colormap', 'viridis'),
            'transparency': self.get('visualization.default_transparency', 1.0),
            'show_axes': self.get('ui.show_axes', True),
            'show_bounds': self.get('ui.show_bounds', True),
            'background_color': self.get('ui.background_color', 'white'),
            'edge_color': self.get('ui.edge_color', 'black'),
            'edge_width': self.get('ui.edge_width', 0.5)
        }
    
    def set_visualization_settings(self, settings: Dict[str, Any]) -> None:
        """Set visualization settings."""
        for key, value in settings.items():
            if key in ['colormap', 'transparency']:
                self.set(f'visualization.default_{key}', value)
            else:
                self.set(f'ui.{key}', value)
    
    def get_parsing_settings(self) -> Dict[str, Any]:
        """Get parsing settings."""
        return {
            'csv_delimiter': self.get('parsing.csv_delimiter', 'auto'),
            'csv_encoding': self.get('parsing.csv_encoding', 'utf-8'),
            'max_file_size_mb': self.get('parsing.max_file_size_mb', 1000),
            'auto_detect_format': self.get('parsing.auto_detect_format', True)
        }
    
    def set_parsing_settings(self, settings: Dict[str, Any]) -> None:
        """Set parsing settings."""
        for key, value in settings.items():
            self.set(f'parsing.{key}', value)
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self.config = self._load_default_config()
        logger.info("Reset configuration to defaults")
    
    def export_config(self, file_path: Path) -> None:
        """Export configuration to a file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported configuration to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting configuration: {e}")
    
    def import_config(self, file_path: Path) -> None:
        """Import configuration from a file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            self._merge_config(self.config, imported_config)
            logger.info(f"Imported configuration from {file_path}")
        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
    
    def get_resource_classification_defaults(self) -> Dict[str, Dict[str, float]]:
        """Get resource classification default parameters."""
        return {
            'Measured': {
                'max_dist': self.get('resource_classification.Measured.max_dist', 50.0),
                'min_holes': self.get('resource_classification.Measured.min_holes', 3)
            },
            'Indicated': {
                'max_dist': self.get('resource_classification.Indicated.max_dist', 100.0),
                'min_holes': self.get('resource_classification.Indicated.min_holes', 2)
            },
            'Inferred': {
                'max_dist': self.get('resource_classification.Inferred.max_dist', 150.0),
                'min_holes': self.get('resource_classification.Inferred.min_holes', 1)
            }
        }
    
    def set_resource_classification_defaults(self, category: str, max_dist: float, min_holes: int) -> None:
        """
        Set resource classification defaults for a specific category.
        
        Args:
            category: Category name ('Measured', 'Indicated', or 'Inferred')
            max_dist: Maximum distance threshold in meters
            min_holes: Minimum number of drillholes required
        """
        self.set(f'resource_classification.{category}.max_dist', max_dist)
        self.set(f'resource_classification.{category}.min_holes', min_holes)
        logger.info(f"Updated {category} classification defaults: max_dist={max_dist}m, min_holes={min_holes}")


# Global configuration instance
config = Config()
