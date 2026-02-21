"""
Drillhole Plotting - Downhole plots, strip logs, and fence diagrams.

Provides plotting capabilities for drillhole data visualization.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .datamodel import DrillholeDatabase

logger = logging.getLogger(__name__)


class DownholePlotter:
    """Downhole plot generator."""
    
    def __init__(self, db: DrillholeDatabase):
        """Initialize plotter with database."""
        self.db = db
    
    def create_downhole_plot(self, hole_id: str, element: str,
                            width: float = 8, height: float = 12,
                            fig: Optional[Figure] = None) -> Figure:
        """
        Create downhole plot for a single hole.
        
        Args:
            hole_id: Hole ID
            element: Element name to plot
            width: Figure width
            height: Figure height
            fig: Optional existing figure
        
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig, ax = plt.subplots(figsize=(width, height))
        else:
            ax = fig.gca()
        
        # Get hole data
        collar = next((c for c in self.db.collars if c.hole_id == hole_id), None)
        if not collar:
            logger.warning(f"Hole {hole_id} not found")
            return fig
        
        assays = self.db.get_assays_for(hole_id)
        if not assays:
            logger.warning(f"No assays found for hole {hole_id}")
            return fig
        
        # Extract depths and grades
        depths = []
        grades = []
        for assay in assays:
            if element in assay.values:
                depth = (assay.depth_from + assay.depth_to) / 2
                grade = assay.values[element]
                depths.append(depth)
                grades.append(grade)
        
        if not depths:
            logger.warning(f"No {element} data found for hole {hole_id}")
            return fig
        
        # Create plot
        ax.plot(grades, depths, 'b-', linewidth=2, label=element)
        ax.scatter(grades, depths, s=50, c='blue', alpha=0.6)
        
        # Format axes
        ax.set_xlabel(f'{element} Grade', fontsize=12)
        ax.set_ylabel('Depth (m)', fontsize=12)
        ax.set_title(f'Downhole Plot: {hole_id} - {element}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Invert y-axis (depth increases downward)
        ax.invert_yaxis()
        
        # Add collar information
        ax.text(0.02, 0.98, f'Collar: ({collar.x:.1f}, {collar.y:.1f}, {collar.z:.1f})',
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def create_multi_element_plot(self, hole_id: str, elements: List[str],
                                  width: float = 12, height: float = 12,
                                  fig: Optional[Figure] = None) -> Figure:
        """
        Create multi-element downhole plot.
        
        Args:
            hole_id: Hole ID
            elements: List of element names
            width: Figure width
            height: Figure height
            fig: Optional existing figure
        
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig, axes = plt.subplots(1, len(elements), figsize=(width, height), sharey=True)
        else:
            axes = fig.get_axes()
        
        if len(elements) == 1:
            axes = [axes]
        
        # Get hole data
        collar = next((c for c in self.db.collars if c.hole_id == hole_id), None)
        if not collar:
            logger.warning(f"Hole {hole_id} not found")
            return fig
        
        assays = self.db.get_assays_for(hole_id)
        if not assays:
            logger.warning(f"No assays found for hole {hole_id}")
            return fig
        
        # Extract depths
        depths = []
        for assay in assays:
            depth = (assay.depth_from + assay.depth_to) / 2
            depths.append(depth)
        
        # Plot each element
        for i, element in enumerate(elements):
            ax = axes[i]
            grades = []
            for assay in assays:
                if element in assay.values:
                    grades.append(assay.values[element])
                else:
                    grades.append(np.nan)
            
            ax.plot(grades, depths, 'b-', linewidth=2, label=element)
            ax.scatter(grades, depths, s=50, c='blue', alpha=0.6)
            
            ax.set_xlabel(f'{element} Grade', fontsize=12)
            ax.set_title(element, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.invert_yaxis()
        
        # Set y-label on first subplot
        axes[0].set_ylabel('Depth (m)', fontsize=12)
        
        # Set main title
        fig.suptitle(f'Multi-Element Downhole Plot: {hole_id}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig


class StripLogPlotter:
    """Strip log generator."""
    
    def __init__(self, db: DrillholeDatabase):
        """Initialize plotter with database."""
        self.db = db
    
    def create_strip_log(self, hole_id: str, elements: List[str],
                        width: float = 12, height: float = 12,
                        fig: Optional[Figure] = None) -> Figure:
        """
        Create strip log for a single hole.
        
        Args:
            hole_id: Hole ID
            elements: List of element names to plot
            width: Figure width
            height: Figure height
            fig: Optional existing figure
        
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig = plt.figure(figsize=(width, height))
        
        # Get hole data
        collar = next((c for c in self.db.collars if c.hole_id == hole_id), None)
        if not collar:
            logger.warning(f"Hole {hole_id} not found")
            return fig
        
        assays = self.db.get_assays_for(hole_id)
        lithology = self.db.get_lithology_for(hole_id)
        
        if not assays:
            logger.warning(f"No assays found for hole {hole_id}")
            return fig
        
        # Calculate plot layout
        num_columns = len(elements) + 1  # +1 for lithology column
        gs = fig.add_gridspec(1, num_columns, hspace=0.3, wspace=0.3)
        
        # Get depth range
        min_depth = min(a.depth_from for a in assays)
        max_depth = max(a.depth_to for a in assays)
        
        # Plot lithology column
        ax_lith = fig.add_subplot(gs[0, 0])
        self._plot_lithology_column(ax_lith, lithology, min_depth, max_depth)
        ax_lith.set_ylabel('Depth (m)', fontsize=12)
        ax_lith.set_title('Lithology', fontsize=12, fontweight='bold')
        ax_lith.invert_yaxis()
        
        # Plot element columns
        for i, element in enumerate(elements):
            ax = fig.add_subplot(gs[0, i + 1], sharey=ax_lith)
            self._plot_element_column(ax, assays, element, min_depth, max_depth)
            ax.set_title(element, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        # Set main title
        fig.suptitle(f'Strip Log: {hole_id}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def _plot_lithology_column(self, ax, lithology: List, min_depth: float, max_depth: float):
        """Plot lithology column."""
        # Define colors for lithology codes
        colors = {
            'W': 'white',
            'G': 'gray',
            'B': 'blue',
            'R': 'red',
            'Y': 'yellow',
            'G': 'green',
        }
        
        for lith in lithology:
            depth_from = lith.depth_from
            depth_to = lith.depth_to
            lith_code = lith.lith_code
            
            # Get color
            color = colors.get(lith_code[0] if lith_code else 'W', 'white')
            
            # Draw rectangle
            rect = patches.Rectangle(
                (0, depth_from), 1, depth_to - depth_from,
                linewidth=1, edgecolor='black', facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)
            
            # Add text label
            ax.text(0.5, (depth_from + depth_to) / 2, lith_code,
                   ha='center', va='center', fontsize=10)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(max_depth, min_depth)
        ax.set_xticks([])
    
    def _plot_element_column(self, ax, assays: List, element: str,
                            min_depth: float, max_depth: float):
        """Plot element column."""
        depths = []
        grades = []
        
        for assay in assays:
            if element in assay.values:
                depth = (assay.depth_from + assay.depth_to) / 2
                grade = assay.values[element]
                depths.append(depth)
                grades.append(grade)
        
        if depths:
            ax.plot(grades, depths, 'b-', linewidth=2)
            ax.scatter(grades, depths, s=50, c='blue', alpha=0.6)
            ax.set_xlabel(f'{element} Grade', fontsize=10)
        
        ax.set_ylim(max_depth, min_depth)


class FenceDiagramPlotter:
    """Fence diagram generator."""
    
    def __init__(self, db: DrillholeDatabase):
        """Initialize plotter with database."""
        self.db = db
    
    def create_fence_diagram(self, hole_ids: List[str], element: str,
                            section_axis: str = 'x',
                            width: float = 12, height: float = 10,
                            fig: Optional[Figure] = None) -> Figure:
        """
        Create fence diagram for multiple holes.
        
        Args:
            hole_ids: List of hole IDs
            element: Element name to plot
            section_axis: Section axis ('x', 'y', or 'z')
            width: Figure width
            height: Figure height
            fig: Optional existing figure
        
        Returns:
            Matplotlib figure
        """
        if fig is None:
            fig = plt.figure(figsize=(width, height))
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.gca()
        
        # Get hole data
        for hole_id in hole_ids:
            collar = next((c for c in self.db.collars if c.hole_id == hole_id), None)
            if not collar:
                continue
            
            assays = self.db.get_assays_for(hole_id)
            if not assays:
                continue
            
            # Extract coordinates and grades
            x_coords = []
            y_coords = []
            z_coords = []
            grades = []
            
            for assay in assays:
                if element in assay.values:
                    # Use midpoint depth
                    depth = (assay.depth_from + assay.depth_to) / 2
                    
                    # Calculate 3D position (simplified - assumes straight hole)
                    x_coords.append(collar.x)
                    y_coords.append(collar.y)
                    z_coords.append(collar.z - depth)  # Z decreases with depth
                    grades.append(assay.values[element])
            
            if x_coords:
                # Plot hole trace
                ax.plot(x_coords, y_coords, z_coords, 'k-', linewidth=2, alpha=0.5)
                
                # Plot grades as colored points
                scatter = ax.scatter(x_coords, y_coords, z_coords, c=grades,
                                   s=100, cmap='viridis', alpha=0.7)
        
        # Format axes
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f'Fence Diagram: {element}', fontsize=14, fontweight='bold')
        
        # Add colorbar
        if grades:
            fig.colorbar(scatter, ax=ax, label=f'{element} Grade')
        
        return fig

