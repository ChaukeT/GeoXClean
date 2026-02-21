"""
Color mapping utilities for block model visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Dict, Any, List, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ColorMapper:
    """
    Handles color mapping for block model properties.
    
    Supports both continuous and categorical data coloring.
    """
    
    def __init__(self):
        # Available colormaps
        self.continuous_colormaps = [
            'turbo', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
            'bone', 'copper', 'pink', 'gray', 'binary', 'gist_earth',
            'terrain', 'ocean', 'rainbow', 'seismic', 'coolwarm',
            'bwr', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral'
        ]
        
        # Categorical/discrete colormaps
        self.categorical_colormaps = [
            'tab10', 'tab20', 'tab20b', 'tab20c',
            'Set1', 'Set2', 'Set3',
            'Paired', 'Accent', 'Dark2',
            'Pastel1', 'Pastel2'
        ]
        
        # All available colormaps
        self.all_colormaps = self.continuous_colormaps + self.categorical_colormaps
        
        self.categorical_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        
        self.current_colormap = 'turbo'
        self.current_property = None
        self.color_range = None
        self.normalize_data = True
    
    def get_available_colormaps(self) -> List[str]:
        """Get list of available colormaps (both continuous and categorical)."""
        return self.all_colormaps.copy()
    
    def set_colormap(self, colormap: str) -> None:
        """
        Set the current colormap.
        
        Args:
            colormap: Name of the colormap
        """
        if colormap in self.all_colormaps:
            self.current_colormap = colormap
            logger.info(f"Set colormap: {colormap}")
        else:
            logger.warning(f"Unknown colormap: {colormap} - using viridis instead")
            self.current_colormap = 'viridis'
    
    def map_property_to_colors(self, values: np.ndarray, property_name: str,
                              colormap: Optional[Union[str, Any]] = None,
                              vmin: Optional[float] = None,
                              vmax: Optional[float] = None,
                              center_zero: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Map property values to colors.
        
        Args:
            values: Array of property values
            property_name: Name of the property
            colormap: Colormap to use (uses current if None)
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            center_zero: If True, normalize around zero (for divergent maps)
            
        Returns:
            Tuple of (colors_array, metadata_dict)
        """
        if colormap is None:
            colormap = self.current_colormap
        
        # Determine if data is categorical or continuous
        is_categorical = self._is_categorical(values)
        
        if is_categorical:
            return self._map_categorical_colors(values, property_name)
        else:
            return self._map_continuous_colors(values, property_name, colormap, vmin, vmax, center_zero)
    
    def _is_categorical(self, values: np.ndarray) -> bool:
        """Check if values represent categorical data."""
        # Check if values are strings or integers with limited unique values
        if not np.issubdtype(values.dtype, np.number):
            return True
        
        # Check if numeric values have limited unique values (less than 20)
        unique_values = np.unique(values[~np.isnan(values)])
        return len(unique_values) <= 20
    
    def _map_categorical_colors(self, values: np.ndarray, property_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Map categorical values to colors."""
        unique_values = np.unique(values[~np.isnan(values)])
        n_categories = len(unique_values)
        
        # Create color mapping
        colors = np.zeros((len(values), 4))  # RGBA
        color_map = {}
        
        for i, value in enumerate(unique_values):
            color_idx = i % len(self.categorical_colors)
            color_hex = self.categorical_colors[color_idx]
            
            # Convert hex to RGB
            rgb = self._hex_to_rgb(color_hex)
            rgba = [rgb[0]/255, rgb[1]/255, rgb[2]/255, 1.0]
            
            # Apply color to matching values
            mask = values == value
            colors[mask] = rgba
            
            color_map[value] = rgba
        
        metadata = {
            'type': 'categorical',
            'property': property_name,
            'unique_values': unique_values.tolist(),
            'color_map': color_map,
            'n_categories': n_categories
        }
        
        return colors, metadata
    
    def _map_continuous_colors(self, values: np.ndarray, property_name: str,
                              colormap: str, vmin: Optional[float], vmax: Optional[float],
                              center_zero: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Map continuous values to colors using matplotlib colormap.
        
        Args:
            values: Property values array
            property_name: Name of the property
            colormap: Colormap name or colormap object
            vmin: Minimum value for normalization
            vmax: Maximum value for normalization
            center_zero: If True, normalize around zero (for divergent maps)
        
        Returns:
            Tuple of (colors_array, metadata_dict)
        """
        # Handle NaN values
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        
        if len(valid_values) == 0:
            # All values are NaN
            colors = np.zeros((len(values), 4))
            colors[:, 3] = 0  # Transparent
            return colors, {'type': 'continuous', 'property': property_name, 'all_nan': True}
        
        # Set value range
        if vmin is None:
            vmin = np.min(valid_values)
        if vmax is None:
            vmax = np.max(valid_values)
        
        # Zero-centered normalization for divergent maps
        if center_zero:
            # Normalize around zero: [-max_abs, +max_abs] → [0, 1]
            max_abs = max(abs(vmin), abs(vmax))
            if max_abs > 0:
                # Map [-max_abs, +max_abs] to [0, 1]
                # Zero maps to 0.5
                normalized = (values + max_abs) / (2 * max_abs)
            else:
                normalized = np.full_like(values, 0.5)  # All zeros
            # Update vmin/vmax for metadata
            vmin = -max_abs
            vmax = max_abs
        else:
            # Standard normalization
            if vmax > vmin:
                normalized = (values - vmin) / (vmax - vmin)
            else:
                normalized = np.zeros_like(values)
        
        # Clamp values to [0, 1]
        normalized = np.clip(normalized, 0, 1)
        
        # Get colormap (handle both string names and colormap objects)
        if isinstance(colormap, str):
            cmap = cm.get_cmap(colormap)
        else:
            cmap = colormap  # Already a colormap object
        
        # Map to colors
        colors = cmap(normalized)
        
        # Handle NaN values (make transparent)
        colors[~valid_mask, 3] = 0
        
        metadata = {
            'type': 'continuous',
            'property': property_name,
            'colormap': colormap.name if hasattr(colormap, 'name') else str(colormap),
            'vmin': float(vmin),
            'vmax': float(vmax),
            'range': float(vmax - vmin),
            'n_valid': int(np.sum(valid_mask)),
            'n_nan': int(np.sum(~valid_mask)),
            'center_zero': center_zero
        }
        
        return colors, metadata
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def create_colorbar(self, metadata: Dict[str, Any], figsize: Tuple[int, int] = (2, 8)) -> plt.Figure:
        """
        Create a colorbar for the current color mapping.
        
        Args:
            metadata: Color mapping metadata
            figsize: Figure size (width, height)
            
        Returns:
            Matplotlib figure with colorbar
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if metadata['type'] == 'categorical':
            # Create categorical colorbar
            unique_values = metadata['unique_values']
            color_map = metadata['color_map']
            
            # Create color patches
            y_positions = np.arange(len(unique_values))
            colors = [color_map[val] for val in unique_values]
            
            for i, (val, color) in enumerate(zip(unique_values, colors)):
                ax.barh(i, 1, color=color, edgecolor='black', linewidth=0.5)
                ax.text(0.5, i, str(val), ha='center', va='center', fontsize=10)
            
            ax.set_ylim(-0.5, len(unique_values) - 0.5)
            ax.set_xlim(0, 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{metadata['property']}\n(Categorical)", fontsize=12)
            
        else:
            # Create continuous colorbar
            colormap = metadata['colormap']
            vmin = metadata['vmin']
            vmax = metadata['vmax']
            
            # Create gradient
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            ax.imshow(gradient, aspect='auto', cmap=colormap, extent=[0, 1, vmin, vmax])
            
            ax.set_xlim(0, 1)
            ax.set_ylim(vmin, vmax)
            ax.set_xticks([])
            ax.set_ylabel(f"{metadata['property']}\n({colormap})", fontsize=12)
            
            # Add value labels
            ax.set_yticks([vmin, vmax])
            ax.tick_params(axis='y', labelsize=10)
        
        plt.tight_layout()
        return fig
    
    def get_property_statistics(self, values: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics for a property array.
        
        Args:
            values: Property values
            
        Returns:
            Dictionary with statistics
        """
        valid_values = values[~np.isnan(values)]
        
        if len(valid_values) == 0:
            return {'all_nan': True, 'count': len(values)}
        
        stats = {
            'count': len(values),
            'valid_count': len(valid_values),
            'nan_count': len(values) - len(valid_values),
            'min': float(np.min(valid_values)),
            'max': float(np.max(valid_values)),
            'mean': float(np.mean(valid_values)),
            'std': float(np.std(valid_values)),
            'median': float(np.median(valid_values)),
            'is_categorical': self._is_categorical(values)
        }
        
        if stats['is_categorical']:
            unique_values = np.unique(valid_values)
            stats['unique_count'] = len(unique_values)
            stats['unique_values'] = unique_values.tolist()
        
        return stats
    
    def suggest_colormap(self, values: np.ndarray, property_name: str) -> str:
        """
        Suggest an appropriate colormap for a property.
        
        Args:
            values: Property values
            property_name: Name of the property
            
        Returns:
            Suggested colormap name
        """
        if self._is_categorical(values):
            return 'tab10'  # Good for categorical data
        
        # Suggest based on property name
        prop_lower = property_name.lower()
        
        if any(word in prop_lower for word in ['grade', 'concentration', 'content']):
            return 'viridis'  # Good for concentration data
        elif any(word in prop_lower for word in ['temperature', 'heat']):
            return 'hot'  # Good for temperature data
        elif any(word in prop_lower for word in ['depth', 'elevation', 'height']):
            return 'terrain'  # Good for elevation data
        elif any(word in prop_lower for word in ['density', 'weight']):
            return 'plasma'  # Good for density data
        else:
            return 'viridis'  # Default choice
    
    @staticmethod
    def get_nsr_map():
        """
        Get NSR-specific divergent colormap.
        
        Standard mining convention:
        - Negative NSR → Red (unprofitable)
        - Zero NSR → White (break-even)
        - Positive NSR → Blue (profitable)
        
        Returns:
            Matplotlib colormap object
        """
        from matplotlib.colors import LinearSegmentedColormap
        
        # Red to yellow (negative values)
        # White (zero)
        # Light blue to dark blue (positive values)
        colors = [
            '#d73027',  # Dark red
            '#f46d43',  # Red-orange
            '#fdae61',  # Orange-yellow
            '#fee08b',  # Yellow
            '#ffffff',  # White (zero)
            '#e0f3f8',  # Light blue
            '#abd9e9',  # Blue
            '#74add1',  # Medium blue
            '#4575b4',  # Dark blue
            '#313695'   # Very dark blue
        ]
        
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('nsr_diverging', colors, N=n_bins)
        return cmap
    
    @staticmethod
    def get_diverging_map(negative_color: str = '#d73027', 
                          zero_color: str = '#ffffff', 
                          positive_color: str = '#4575b4',
                          n_bins: int = 256):
        """
        Create custom divergent colormap.
        
        Args:
            negative_color: Color for negative values (default: red)
            zero_color: Color for zero (default: white)
            positive_color: Color for positive values (default: blue)
            n_bins: Number of color bins (default: 256)
        
        Returns:
            Matplotlib colormap
        """
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create smooth gradient from negative → zero → positive
        colors = [negative_color, zero_color, positive_color]
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=n_bins)
        return cmap
