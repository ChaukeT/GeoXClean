"""
Advanced Visualization System for Grade-Tonnage Curves and Cut-off Sensitivity Analysis.

This module provides comprehensive plotting capabilities for geostatistical mining analysis:
- Interactive grade-tonnage curves with confidence intervals
- Economic optimization plots (NPV, IRR, payback period)
- Sensitivity analysis tornado diagrams
- Statistical distribution plots
- Multi-panel dashboards
- Export functionality for reports

Features:
- Matplotlib and Plotly backends for different use cases
- Statistical overlays and confidence intervals
- Interactive hover information
- Multiple plot types in single view
- Publication-quality output
- Real-time updates during analysis

Author: GeoX Mining Software - Advanced Visualization Engine
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
import warnings

# Plotting libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as path_effects

# Try to import seaborn for enhanced plotting
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Try to import plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from .geostats_grade_tonnage import (
    GradeTonnageCurve, GradeTonnagePoint, CutoffSensitivityAnalysis,
    ConfidenceIntervalMethod
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class PlotBackend(str, Enum):
    """Plotting backend options."""
    MATPLOTLIB = "matplotlib"
    PLOTLY = "plotly"
    SEABORN = "seaborn"

class PlotType(str, Enum):
    """Types of plots available."""
    GRADE_TONNAGE_CURVE = "grade_tonnage_curve"
    NPV_OPTIMIZATION = "npv_optimization"
    SENSITIVITY_TORNADO = "sensitivity_tornado"
    ECONOMIC_SCATTER = "economic_scatter"
    DISTRIBUTION_HISTOGRAM = "distribution_histogram"
    CONFIDENCE_INTERVALS = "confidence_intervals"
    MULTI_CRITERIA_DASHBOARD = "multi_criteria_dashboard"
    VARIABILITY_ANALYSIS = "variability_analysis"

class ColorScheme(str, Enum):
    """Color schemes for plots."""
    DEFAULT = "default"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    GEOSTATS = "geostats"  # Custom mining/geostats colors
    HIGH_CONTRAST = "high_contrast"

# Color schemes
COLOR_SCHEMES = {
    ColorScheme.DEFAULT: {
        "primary": "#1f77b4",
        "secondary": "#ff7f0e",
        "tertiary": "#2ca02c",
        "accent": "#d62728",
        "neutral": "#7f7f7f",
        "background": "#ffffff",
        "grid": "#e0e0e0"
    },
    ColorScheme.GEOSTATS: {
        "ore": "#ff6b35",      # Red-orange for ore zones
        "waste": "#a8a8a8",    # Gray for waste
        "grade_high": "#ff4757",  # Bright red for high grades
        "grade_low": "#3742fa",   # Blue for low grades
        "confidence": "#ffa726",  # Orange for confidence intervals
        "optimal": "#4caf50",     # Green for optimal points
        "background": "#f5f5f5",
        "grid": "#e0e0e0"
    },
    ColorScheme.HIGH_CONTRAST: {
        "primary": "#000000",
        "secondary": "#ffffff",
        "tertiary": "#ff0000",
        "accent": "#00ff00",
        "neutral": "#808080",
        "background": "#ffffff",
        "grid": "#000000"
    }
}


# =============================================================================
# PLOT CONFIGURATION CLASSES
# =============================================================================

@dataclass
class PlotConfig:
    """
    Configuration for plot generation.

    Attributes:
        backend: Plotting backend to use
        figsize: Figure size (width, height) in inches
        dpi: Resolution for raster output
        color_scheme: Color scheme to use
        show_grid: Whether to show grid lines
        show_legend: Whether to show legend
        title_fontsize: Font size for titles
        label_fontsize: Font size for axis labels
        tick_fontsize: Font size for tick labels
        show_confidence_intervals: Whether to show confidence intervals
        interactive: Whether to enable interactive features
        export_format: Default export format
    """
    backend: PlotBackend = PlotBackend.MATPLOTLIB
    figsize: Tuple[float, float] = (10, 6)
    dpi: int = 150
    color_scheme: ColorScheme = ColorScheme.GEOSTATS
    show_grid: bool = True
    show_legend: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 12
    tick_fontsize: int = 10
    show_confidence_intervals: bool = True
    interactive: bool = False
    export_format: str = "png"

@dataclass
class PlotStyle:
    """
    Style configuration for individual plot elements.

    Attributes:
        linewidth: Line width for curves
        markersize: Size of markers
        alpha: Transparency level
        linestyle: Line style ('-', '--', '-.', ':')
        marker: Marker style ('o', 's', '^', etc.)
        colors: Custom color overrides
    """
    linewidth: float = 2.0
    markersize: float = 6.0
    alpha: float = 0.8
    linestyle: str = '-'
    marker: str = 'o'
    colors: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# MATPLOTLIB VISUALIZATION ENGINE
# =============================================================================

class MatplotlibVisualizationEngine:
    """
    Matplotlib-based visualization engine for grade-tonnage analysis.

    Provides high-quality static plots with advanced styling and statistical overlays.
    """

    def __init__(self, config: PlotConfig = None):
        """
        Initialize the matplotlib visualization engine.

        Args:
            config: Plot configuration
        """
        self.config = config or PlotConfig()
        self.colors = COLOR_SCHEMES[self.config.color_scheme]
        self._setup_style()

    def _setup_style(self):
        """Set up matplotlib style and defaults."""
        plt.style.use('default')

        # Set default parameters
        plt.rcParams.update({
            'figure.dpi': self.config.dpi,
            'font.size': self.config.tick_fontsize,
            'axes.labelsize': self.config.label_fontsize,
            'axes.titlesize': self.config.title_fontsize,
            'xtick.labelsize': self.config.tick_fontsize,
            'ytick.labelsize': self.config.tick_fontsize,
            'legend.fontsize': self.config.label_fontsize,
            'figure.facecolor': self.colors['background'],
            'axes.facecolor': self.colors['background'],
            'grid.color': self.colors['grid'],
            'grid.alpha': 0.3
        })

        if SEABORN_AVAILABLE:
            sns.set_palette("husl")

    def plot_grade_tonnage_curve(
        self,
        curve: GradeTonnageCurve,
        style: PlotStyle = None,
        title: str = "Grade-Tonnage Curve",
        show_confidence: bool = True
    ) -> Tuple[Figure, FigureCanvas]:
        """
        Plot grade-tonnage curve with confidence intervals.

        Args:
            curve: GradeTonnageCurve object
            style: Plot styling options
            title: Plot title
            show_confidence: Whether to show confidence intervals

        Returns:
            Tuple of (Figure, FigureCanvas)
        """
        style = style or PlotStyle()

        fig = Figure(figsize=self.config.figsize, dpi=self.config.dpi)
        canvas = FigureCanvas(fig)

        # Create subplot with twin y-axis
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()

        # Extract data
        cutoffs = [pt.cutoff_grade for pt in curve.points]
        tonnages = [pt.tonnage for pt in curve.points]
        grades = [pt.avg_grade for pt in curve.points]

        # Plot tonnage on primary y-axis
        line1, = ax1.plot(cutoffs, tonnages,
                         color=self.colors.get('primary', '#1f77b4'),
                         linewidth=style.linewidth,
                         marker=style.marker,
                         markersize=style.markersize,
                         alpha=style.alpha,
                         label='Tonnage')

        # Plot grade on secondary y-axis
        line2, = ax2.plot(cutoffs, grades,
                         color=self.colors.get('secondary', '#ff7f0e'),
                         linewidth=style.linewidth,
                         linestyle='--',
                         marker='s',
                         markersize=style.markersize,
                         alpha=style.alpha,
                         label='Average Grade')

        # Add confidence intervals if available and requested
        if show_confidence and self.config.show_confidence_intervals:
            self._add_cv_uncertainty_bands(ax1, curve.points, cutoffs)

        # Styling
        ax1.set_xlabel('Cutoff Grade', fontsize=self.config.label_fontsize, fontweight='bold')
        ax1.set_ylabel('Tonnage (tonnes)', color=self.colors.get('primary', '#1f77b4'),
                      fontsize=self.config.label_fontsize, fontweight='bold')
        ax2.set_ylabel('Average Grade', color=self.colors.get('secondary', '#ff7f0e'),
                      fontsize=self.config.label_fontsize, fontweight='bold')

        ax1.tick_params(axis='y', labelcolor=self.colors.get('primary', '#1f77b4'))
        ax2.tick_params(axis='y', labelcolor=self.colors.get('secondary', '#ff7f0e'))

        # Title
        fig.suptitle(title, fontsize=self.config.title_fontsize, fontweight='bold')

        # Legend
        if self.config.show_legend:
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='best')

        # Grid
        if self.config.show_grid:
            ax1.grid(True, alpha=0.3)

        # Add statistical annotations
        self._add_statistical_annotations(ax1, curve)

        fig.tight_layout()
        return fig, canvas

    def _add_cv_uncertainty_bands(self, ax, points: List[GradeTonnagePoint], cutoffs: List[float]):
        """Add CV-based uncertainty band shading to plot."""
        lower_bounds = [pt.cv_uncertainty_band[0] for pt in points]
        upper_bounds = [pt.cv_uncertainty_band[1] for pt in points]

        ax.fill_between(cutoffs, lower_bounds, upper_bounds,
                       color=self.colors.get('confidence', '#ffa726'),
                       alpha=0.2, label='CV Uncertainty Band (Heuristic)')

    def _add_statistical_annotations(self, ax, curve: GradeTonnageCurve):
        """Add statistical annotations to the plot."""
        if not curve.global_statistics:
            return

        stats = curve.global_statistics
        grade_stats = stats.get('grade_statistics', {})

        # Format tonnage with appropriate units
        total_tonnage = stats.get('total_tonnage', 0)
        if total_tonnage >= 1e9:
            tonnage_str = f"{total_tonnage/1e9:.2f}B t"
        elif total_tonnage >= 1e6:
            tonnage_str = f"{total_tonnage/1e6:.2f}M t"
        else:
            tonnage_str = f"{total_tonnage:,.0f} t"

        # Format metal quantity
        total_metal = stats.get('total_metal', 0)
        if total_metal >= 1e9:
            metal_str = f"{total_metal/1e9:.2f}B"
        elif total_metal >= 1e6:
            metal_str = f"{total_metal/1e6:.2f}M"
        else:
            metal_str = f"{total_metal:,.0f}"

        # Add text box with key statistics
        stats_text = (
            f"Tonnage: {tonnage_str}\n"
            f"Metal: {metal_str} units\n"
            f"CV: {grade_stats.get('cv', 0):.2f}\n"
            f"Samples: {stats.get('sample_count', 0):,}\n"
            f"Declustered: {stats.get('declustered', False)}"
        )

        # Position text box in upper right
        ax.text(0.98, 0.98, stats_text,
               transform=ax.transAxes,
               verticalalignment='top',
               horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white',
                        alpha=0.9,
                        edgecolor='#333'),
               fontsize=self.config.tick_fontsize,
               family='monospace')

    def plot_npv_optimization(
        self,
        sensitivity: CutoffSensitivityAnalysis,
        style: PlotStyle = None,
        title: str = "NPV Optimization Analysis"
    ) -> Tuple[Figure, FigureCanvas]:
        """
        Plot NPV optimization results.

        Args:
            sensitivity: CutoffSensitivityAnalysis object
            style: Plot styling options
            title: Plot title

        Returns:
            Tuple of (Figure, FigureCanvas)
        """
        style = style or PlotStyle()

        fig = Figure(figsize=self.config.figsize, dpi=self.config.dpi)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Plot NPV curve
        line, = ax.plot(sensitivity.cutoff_range, sensitivity.npv_by_cutoff,
                       color=self.colors.get('optimal', '#4caf50'),
                       linewidth=style.linewidth,
                       marker=style.marker,
                       markersize=style.markersize,
                       alpha=style.alpha,
                       label='NPV')

        # Highlight optimal point
        optimal_idx = np.argmax(sensitivity.npv_by_cutoff)
        optimal_cutoff = sensitivity.cutoff_range[optimal_idx]
        optimal_npv = sensitivity.npv_by_cutoff[optimal_idx]

        ax.scatter([optimal_cutoff], [optimal_npv],
                  color=self.colors.get('accent', '#d62728'),
                  s=100, marker='*', zorder=10,
                  label=f'Optimal Cutoff ({optimal_cutoff:.2f})')

        # Add vertical line at optimal cutoff
        ax.axvline(x=optimal_cutoff, color=self.colors.get('accent', '#d62728'),
                  linestyle='--', alpha=0.7, linewidth=1)

        # Styling
        ax.set_xlabel('Cutoff Grade', fontsize=self.config.label_fontsize, fontweight='bold')
        ax.set_ylabel('Net Present Value ($)', fontsize=self.config.label_fontsize, fontweight='bold')
        ax.set_title(title, fontsize=self.config.title_fontsize, fontweight='bold')

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        if self.config.show_grid:
            ax.grid(True, alpha=0.3)

        if self.config.show_legend:
            ax.legend()

        fig.tight_layout()
        return fig, canvas

    def plot_sensitivity_tornado(
        self,
        sensitivity: CutoffSensitivityAnalysis,
        style: PlotStyle = None,
        title: str = "Economic Sensitivity Analysis"
    ) -> Tuple[Figure, FigureCanvas]:
        """
        Create tornado diagram for sensitivity analysis.

        Args:
            sensitivity: CutoffSensitivityAnalysis object
            style: Plot styling options
            title: Plot title

        Returns:
            Tuple of (Figure, FigureCanvas)
        """
        style = style or PlotStyle()

        fig = Figure(figsize=(12, 8), dpi=self.config.dpi)
        canvas = FigureCanvas(fig)

        ax = fig.add_subplot(111)

        # Extract sensitivity data
        sensitivity_curves = sensitivity.sensitivity_curves
        if not sensitivity_curves:
            ax.text(0.5, 0.5, 'No sensitivity data available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            return fig, canvas

        # Calculate sensitivity ranges for each parameter
        param_sensitivities = []
        param_names = []

        base_npv = sensitivity.npv_by_cutoff.max()  # Use max NPV as base

        for param_name, curve_data in sensitivity_curves.items():
            npv_values = curve_data['npv_values']
            param_values = curve_data['param_values']

            if len(npv_values) >= 3:  # Need at least base, +10%, -10%
                # Calculate range from min to max NPV
                npv_range = max(npv_values) - min(npv_values)
                param_sensitivities.append(npv_range)
                param_names.append(param_name.replace('_', ' ').title())

        # Sort by sensitivity (most sensitive first)
        sorted_indices = np.argsort(param_sensitivities)[::-1]
        param_names = [param_names[i] for i in sorted_indices]
        param_sensitivities = [param_sensitivities[i] for i in sorted_indices]
        
        # Convert to millions for readability
        param_sensitivities_m = [v / 1e6 for v in param_sensitivities]

        # Create horizontal bar chart with better styling
        y_pos = np.arange(len(param_names))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Varied colors
        bar_colors = [colors[i % len(colors)] for i in range(len(param_names))]
        
        bars = ax.barh(y_pos, param_sensitivities_m,
                      color=bar_colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5, height=0.6)

        # Add value labels with better formatting
        for i, (bar, value_m) in enumerate(zip(bars, param_sensitivities_m)):
            width = bar.get_width()
            ax.text(width + max(param_sensitivities_m) * 0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'${value_m:.1f}M', ha='left', va='center', 
                   fontweight='bold', fontsize=12)

        # Styling with larger fonts
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names, fontsize=12)
        ax.set_xlabel('NPV Sensitivity ($M)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)

        # Format x-axis with clean labels (in millions)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.0f}M'))
        ax.tick_params(axis='x', labelsize=11)

        # Add grid
        if self.config.show_grid:
            ax.grid(True, alpha=0.3, axis='x', linestyle='--')
            ax.set_axisbelow(True)

        # Add base NPV reference line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

        fig.tight_layout()
        return fig, canvas

    def plot_multi_criteria_dashboard(
        self,
        curve: GradeTonnageCurve,
        sensitivity: CutoffSensitivityAnalysis,
        style: PlotStyle = None,
        title: str = "Multi-Criteria Mining Analysis Dashboard"
    ) -> Tuple[Figure, FigureCanvas]:
        """
        Create comprehensive multi-panel dashboard.

        Args:
            curve: GradeTonnageCurve object
            sensitivity: CutoffSensitivityAnalysis object
            style: Plot styling options
            title: Dashboard title

        Returns:
            Tuple of (Figure, FigureCanvas)
        """
        style = style or PlotStyle()

        # Create larger 2x2 subplot grid with better spacing
        fig = Figure(figsize=(18, 14), dpi=self.config.dpi)
        canvas = FigureCanvas(fig)

        # Increased spacing between subplots for better readability
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.4, 
                              left=0.08, right=0.95, top=0.92, bottom=0.08)

        # 1. Grade-Tonnage Curve (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_grade_tonnage_panel(ax1, curve)

        # 2. NPV Optimization (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_npv_panel(ax2, sensitivity)

        # 3. Economic Scatter Plot (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_economic_scatter(ax3, curve)

        # 4. Sensitivity Summary (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_sensitivity_panel(ax4, sensitivity)

        # Main title - larger and better positioned
        fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)

        return fig, canvas

    def _plot_grade_tonnage_panel(self, ax, curve: GradeTonnageCurve):
        """Plot grade-tonnage curve in dashboard panel."""
        cutoffs = [pt.cutoff_grade for pt in curve.points]
        tonnages = [pt.tonnage for pt in curve.points]
        grades = [pt.avg_grade for pt in curve.points]

        # Convert tonnage to millions of tonnes for better readability
        tonnages_mt = [t / 1e6 for t in tonnages]

        ax2 = ax.twinx()

        # Plot data with thicker lines and larger markers for visibility
        line1, = ax.plot(cutoffs, tonnages_mt, 'b-', linewidth=3, marker='o', 
                        markersize=6, alpha=0.9, label='Tonnage', zorder=2)
        line2, = ax2.plot(cutoffs, grades, 'r--', linewidth=3, marker='s', 
                         markersize=6, alpha=0.9, label='Average Grade', zorder=2)

        # Styling with larger fonts
        ax.set_xlabel('Cutoff Grade', fontweight='bold', fontsize=13)
        ax.set_ylabel('Tonnage (Mt)', color='b', fontweight='bold', fontsize=13)
        ax2.set_ylabel('Average Grade', color='r', fontweight='bold', fontsize=13)
        ax.set_title('Tonnage vs Cutoff Grade', fontweight='bold', fontsize=14, pad=10)

        ax.tick_params(axis='both', labelsize=11)
        ax.tick_params(axis='y', labelcolor='b', labelsize=11)
        ax2.tick_params(axis='y', labelcolor='r', labelsize=11)

        # Format y-axis to show clean numbers
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add legend
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=10, framealpha=0.9)

    def _plot_npv_panel(self, ax, sensitivity: CutoffSensitivityAnalysis):
        """Plot NPV optimization in dashboard panel."""
        # Convert NPV to millions for better readability
        npv_m = [n / 1e6 for n in sensitivity.npv_by_cutoff]
        
        ax.plot(sensitivity.cutoff_range, npv_m,
               color=self.colors.get('optimal', '#4caf50'),
               linewidth=3, marker='o', markersize=6, alpha=0.9, zorder=2, label='NPV')

        # Highlight optimal
        optimal_idx = np.argmax(sensitivity.npv_by_cutoff)
        optimal_cutoff = sensitivity.cutoff_range[optimal_idx]
        optimal_npv_m = npv_m[optimal_idx]

        ax.scatter([optimal_cutoff], [optimal_npv_m], color='red', s=200, 
                  marker='*', zorder=10, edgecolors='darkred', linewidths=1.5,
                  label=f'Optimal: {optimal_cutoff:.2f}')
        ax.axvline(x=optimal_cutoff, color='red', linestyle='--', alpha=0.7, linewidth=2)

        ax.set_xlabel('Cutoff Grade', fontweight='bold', fontsize=13)
        ax.set_ylabel('NPV ($M)', fontweight='bold', fontsize=13)
        ax.set_title('NPV vs Cutoff Grade', fontweight='bold', fontsize=14, pad=10)

        ax.tick_params(axis='both', labelsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.1f}M'))
        
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add legend
        ax.legend(loc='best', fontsize=10, framealpha=0.9)

    def _plot_economic_scatter(self, ax, curve: GradeTonnageCurve):
        """Plot economic scatter plot in dashboard panel."""
        tonnages = [pt.tonnage for pt in curve.points]
        net_values = [pt.net_value for pt in curve.points]
        cutoffs = [pt.cutoff_grade for pt in curve.points]

        # Convert to better units for readability
        tonnages_mt = [t / 1e6 for t in tonnages]
        net_values_b = [v / 1e9 for v in net_values]  # Convert to billions

        # Use larger markers and better colors
        scatter = ax.scatter(tonnages_mt, net_values_b,
                           c=cutoffs, cmap='plasma', alpha=0.8, 
                           s=100, edgecolors='black', linewidths=1.5, zorder=2)

        # Add colorbar with better formatting
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Cutoff Grade', fontweight='bold', fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        ax.set_xlabel('Tonnage (Mt)', fontweight='bold', fontsize=13)
        ax.set_ylabel('Net Value ($B)', fontweight='bold', fontsize=13)
        ax.set_title('Net Value vs Tonnage', fontweight='bold', fontsize=14, pad=10)

        ax.tick_params(axis='both', labelsize=11)
        
        # Format axes
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}B'))

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    def _plot_sensitivity_panel(self, ax, sensitivity: CutoffSensitivityAnalysis):
        """Plot sensitivity summary in dashboard panel."""
        if not sensitivity.sensitivity_curves:
            ax.text(0.5, 0.5, 'No sensitivity\ndata available',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Parameter Sensitivity', fontweight='bold', fontsize=14)
            return

        # Show top 3 most sensitive parameters
        sensitivities = []
        names = []

        for param_name, curve_data in sensitivity.sensitivity_curves.items():
            npv_values = curve_data['npv_values']
            if len(npv_values) >= 3:
                npv_range = max(npv_values) - min(npv_values)
                sensitivities.append(npv_range)
                names.append(param_name.replace('_', ' ').title())

        if sensitivities:
            # Sort and take top 3
            sorted_indices = np.argsort(sensitivities)[::-1][:3]
            top_names = [names[i] for i in sorted_indices]
            top_values = [sensitivities[i] for i in sorted_indices]
            
            # Convert to millions for better readability
            top_values_m = [v / 1e6 for v in top_values]

            # Use better colors and styling
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
            bars = ax.bar(range(len(top_names)), top_values_m,
                         color=colors[:len(top_names)], alpha=0.8, 
                         edgecolor='black', linewidth=1.5, zorder=2)

            ax.set_xticks(range(len(top_names)))
            ax.set_xticklabels(top_names, rotation=0, ha='center', fontsize=11)
            ax.set_ylabel('NPV Sensitivity ($M)', fontweight='bold', fontsize=13)
            ax.set_title('Economic Parameter Sensitivities', fontweight='bold', fontsize=14, pad=10)

            ax.tick_params(axis='both', labelsize=11)

            # Add values on bars with better formatting
            for bar, value_m in zip(bars, top_values_m):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(top_values_m)*0.03,
                       f'${value_m:.1f}M', ha='center', va='bottom', 
                       fontweight='bold', fontsize=11)

            ax.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
            ax.set_axisbelow(True)

    def export_plot(self, fig: Figure, filename: str, format: str = None) -> bool:
        """
        Export plot to file.

        Args:
            fig: Matplotlib figure
            filename: Output filename (without extension)
            format: Export format ('png', 'pdf', 'svg', etc.)

        Returns:
            True if successful
        """
        format = format or self.config.export_format

        try:
            full_filename = f"{filename}.{format}"
            fig.savefig(full_filename,
                       dpi=self.config.dpi,
                       bbox_inches='tight',
                       format=format)
            logger.info(f"Plot exported to {full_filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to export plot: {e}")
            return False


# =============================================================================
# PLOTLY VISUALIZATION ENGINE (INTERACTIVE)
# =============================================================================

class PlotlyVisualizationEngine:
    """
    Plotly-based interactive visualization engine.

    Provides interactive plots with hover information, zooming, and dynamic updates.
    """

    def __init__(self, config: PlotConfig = None):
        """
        Initialize the plotly visualization engine.

        Args:
            config: Plot configuration
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for interactive visualizations. Install with: pip install plotly")

        self.config = config or PlotConfig()
        self.colors = COLOR_SCHEMES[self.config.color_scheme]

    def plot_grade_tonnage_curve_interactive(
        self,
        curve: GradeTonnageCurve,
        title: str = "Interactive Grade-Tonnage Curve"
    ) -> go.Figure:
        """
        Create interactive grade-tonnage curve.

        Args:
            curve: GradeTonnageCurve object
            title: Plot title

        Returns:
            Plotly figure
        """
        # Extract data
        cutoffs = [pt.cutoff_grade for pt in curve.points]
        tonnages = [pt.tonnage for pt in curve.points]
        grades = [pt.avg_grade for pt in curve.points]
        metal_quantities = [pt.metal_quantity for pt in curve.points]
        net_values = [pt.net_value for pt in curve.points]

        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add tonnage trace
        fig.add_trace(
            go.Scatter(
                x=cutoffs,
                y=tonnages,
                mode='lines+markers',
                name='Tonnage',
                line=dict(color=self.colors.get('primary', '#1f77b4'), width=3),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>Cutoff:</b> %{x:.2f}<br>' +
                             '<b>Tonnage:</b> %{y:,.0f} tonnes<br>' +
                             '<extra></extra>'
            ),
            secondary_y=False
        )

        # Add grade trace
        fig.add_trace(
            go.Scatter(
                x=cutoffs,
                y=grades,
                mode='lines+markers',
                name='Average Grade',
                line=dict(color=self.colors.get('secondary', '#ff7f0e'), width=3, dash='dash'),
                marker=dict(size=8, symbol='square'),
                hovertemplate='<b>Cutoff:</b> %{x:.2f}<br>' +
                             '<b>Average Grade:</b> %{y:.3f}<br>' +
                             '<extra></extra>'
            ),
            secondary_y=True
        )

        # Add CV uncertainty bands if available
        if self.config.show_confidence_intervals and curve.points:
            lower_bounds = [pt.cv_uncertainty_band[0] for pt in curve.points]
            upper_bounds = [pt.cv_uncertainty_band[1] for pt in curve.points]

            fig.add_trace(
                go.Scatter(
                    x=cutoffs + cutoffs[::-1],
                    y=upper_bounds + lower_bounds[::-1],
                    fill='toself',
                    fillcolor=self.colors.get('confidence', '#ffa726'),
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    name='CV Uncertainty Band (Heuristic)',
                    showlegend=True
                ),
                secondary_y=False
            )

        # Update layout
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=self.config.title_fontsize + 4, weight='bold')
            ),
            hovermode='x unified',
            plot_bgcolor=self.colors.get('background', '#f5f5f5'),
            paper_bgcolor=self.colors.get('background', '#f5f5f5')
        )

        # Update axes
        fig.update_xaxes(
            title_text="Cutoff Grade",
            title_font=dict(size=self.config.label_fontsize, weight='bold'),
            tickfont=dict(size=self.config.tick_fontsize)
        )

        fig.update_yaxes(
            title_text="Tonnage (tonnes)",
            title_font=dict(size=self.config.label_fontsize, weight='bold'),
            tickfont=dict(size=self.config.tick_fontsize),
            secondary_y=False
        )

        fig.update_yaxes(
            title_text="Average Grade",
            title_font=dict(size=self.config.label_fontsize, weight='bold'),
            tickfont=dict(size=self.config.tick_fontsize),
            secondary_y=True
        )

        return fig

    def create_interactive_dashboard(
        self,
        curve: GradeTonnageCurve,
        sensitivity: CutoffSensitivityAnalysis,
        title: str = "Interactive Mining Analysis Dashboard"
    ) -> go.Figure:
        """
        Create comprehensive interactive dashboard.

        Args:
            curve: GradeTonnageCurve object
            sensitivity: CutoffSensitivityAnalysis object
            title: Dashboard title

        Returns:
            Plotly figure with subplots
        """
        # Create 2x2 subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Grade-Tonnage Curve', 'NPV Optimization',
                          'Economic Value Analysis', 'Parameter Sensitivity'),
            specs=[[{"secondary_y": True}, {}],
                   [{}, {"type": "bar"}]]
        )

        # 1. Grade-Tonnage Curve (top-left)
        cutoffs = [pt.cutoff_grade for pt in curve.points]
        tonnages = [pt.tonnage for pt in curve.points]
        grades = [pt.avg_grade for pt in curve.points]

        fig.add_trace(
            go.Scatter(x=cutoffs, y=tonnages, mode='lines+markers', name='Tonnage',
                      line=dict(color='#1f77b4', width=2)),
            row=1, col=1, secondary_y=False
        )

        fig.add_trace(
            go.Scatter(x=cutoffs, y=grades, mode='lines+markers', name='Grade',
                      line=dict(color='#ff7f0e', width=2, dash='dash')),
            row=1, col=1, secondary_y=True
        )

        # 2. NPV Optimization (top-right)
        fig.add_trace(
            go.Scatter(x=sensitivity.cutoff_range, y=sensitivity.npv_by_cutoff,
                      mode='lines+markers', name='NPV',
                      line=dict(color='#4caf50', width=2)),
            row=1, col=2
        )

        # Highlight optimal point
        optimal_idx = np.argmax(sensitivity.npv_by_cutoff)
        optimal_cutoff = sensitivity.cutoff_range[optimal_idx]
        optimal_npv = sensitivity.npv_by_cutoff[optimal_idx]

        fig.add_trace(
            go.Scatter(x=[optimal_cutoff], y=[optimal_npv],
                      mode='markers', name='Optimal Cutoff',
                      marker=dict(color='#d62728', size=12, symbol='star')),
            row=1, col=2
        )

        # 3. Economic Value Scatter (bottom-left)
        net_values = [pt.net_value for pt in curve.points]

        fig.add_trace(
            go.Scatter(x=tonnages, y=net_values, mode='markers',
                      name='Economic Value',
                      marker=dict(size=8, color=cutoffs, colorscale='Viridis',
                                showscale=True, colorbar=dict(title="Cutoff Grade")),
                      text=[f"Cutoff: {c:.2f}" for c in cutoffs],
                      hovertemplate='<b>Tonnage:</b> %{x:,.0f}<br>' +
                                   '<b>Net Value:</b> %{y:$,.0f}<br>' +
                                   '<b>Cutoff:</b> %{text}<br>' +
                                   '<extra></extra>'),
            row=2, col=1
        )

        # 4. Sensitivity Analysis (bottom-right)
        if sensitivity.sensitivity_curves:
            param_names = []
            sensitivities = []

            for param_name, curve_data in sensitivity.sensitivity_curves.items():
                npv_values = curve_data['npv_values']
                if len(npv_values) >= 3:
                    npv_range = max(npv_values) - min(npv_values)
                    param_names.append(param_name.replace('_', ' ').title())
                    sensitivities.append(npv_range)

            if sensitivities:
                # Sort by sensitivity
                sorted_indices = np.argsort(sensitivities)[::-1][:5]  # Top 5
                param_names = [param_names[i] for i in sorted_indices]
                sensitivities = [sensitivities[i] for i in sorted_indices]

                fig.add_trace(
                    go.Bar(x=param_names, y=sensitivities, name='Sensitivity',
                          marker_color='#1f77b4'),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, weight='bold')),
            showlegend=False,
            height=800,
            hovermode='closest'
        )

        # Update axes labels
        fig.update_xaxes(title_text="Cutoff Grade", row=1, col=1)
        fig.update_yaxes(title_text="Tonnage", secondary_y=False, row=1, col=1)
        fig.update_yaxes(title_text="Grade", secondary_y=True, row=1, col=1)

        fig.update_xaxes(title_text="Cutoff Grade", row=1, col=2)
        fig.update_yaxes(title_text="NPV ($)", row=1, col=2)

        fig.update_xaxes(title_text="Tonnage", row=2, col=1)
        fig.update_yaxes(title_text="Net Value ($)", row=2, col=1)

        fig.update_xaxes(title_text="Parameter", row=2, col=2)
        fig.update_yaxes(title_text="NPV Sensitivity ($)", row=2, col=2)

        return fig


# =============================================================================
# MAIN VISUALIZATION COORDINATOR
# =============================================================================

class MiningVisualizationCoordinator:
    """
    Main coordinator for mining visualization tasks.

    Provides unified interface to different visualization engines and plot types.
    """

    def __init__(self, config: PlotConfig = None):
        """
        Initialize the visualization coordinator.

        Args:
            config: Plot configuration
        """
        self.config = config or PlotConfig()

        # Initialize engines
        self.matplotlib_engine = MatplotlibVisualizationEngine(self.config)

        if PLOTLY_AVAILABLE:
            self.plotly_engine = PlotlyVisualizationEngine(self.config)
        else:
            self.plotly_engine = None
            logger.warning("Plotly not available - interactive plots disabled")

    def create_grade_tonnage_plot(
        self,
        curve: GradeTonnageCurve,
        plot_type: PlotType = PlotType.GRADE_TONNAGE_CURVE,
        interactive: bool = None,
        **kwargs
    ) -> Union[Tuple[Figure, FigureCanvas], go.Figure]:
        """
        Create grade-tonnage visualization.

        Args:
            curve: GradeTonnageCurve object
            plot_type: Type of plot to create
            interactive: Whether to use interactive backend (overrides config)
            **kwargs: Additional arguments for plotting functions

        Returns:
            Plot object (matplotlib Figure/Canvas tuple or plotly Figure)
        """
        interactive = interactive if interactive is not None else self.config.interactive

        if interactive and self.plotly_engine:
            return self.plotly_engine.plot_grade_tonnage_curve_interactive(
                curve, **kwargs
            )
        else:
            return self.matplotlib_engine.plot_grade_tonnage_curve(
                curve, **kwargs
            )

    def create_npv_optimization_plot(
        self,
        sensitivity: CutoffSensitivityAnalysis,
        interactive: bool = None,
        **kwargs
    ) -> Union[Tuple[Figure, FigureCanvas], go.Figure]:
        """
        Create NPV optimization visualization.

        Args:
            sensitivity: CutoffSensitivityAnalysis object
            interactive: Whether to use interactive backend
            **kwargs: Additional arguments

        Returns:
            Plot object
        """
        interactive = interactive if interactive is not None else self.config.interactive

        if interactive and self.plotly_engine:
            # For now, fall back to matplotlib for NPV plots
            # Could be extended to plotly in the future
            pass

        return self.matplotlib_engine.plot_npv_optimization(
            sensitivity, **kwargs
        )

    def create_sensitivity_plot(
        self,
        sensitivity: CutoffSensitivityAnalysis,
        plot_type: PlotType = PlotType.SENSITIVITY_TORNADO,
        **kwargs
    ) -> Union[Tuple[Figure, FigureCanvas], go.Figure]:
        """
        Create sensitivity analysis visualization.

        Args:
            sensitivity: CutoffSensitivityAnalysis object
            plot_type: Type of sensitivity plot
            **kwargs: Additional arguments

        Returns:
            Plot object
        """
        if plot_type == PlotType.SENSITIVITY_TORNADO:
            return self.matplotlib_engine.plot_sensitivity_tornado(
                sensitivity, **kwargs
            )
        else:
            raise ValueError(f"Unsupported sensitivity plot type: {plot_type}")

    def create_dashboard(
        self,
        curve: GradeTonnageCurve,
        sensitivity: CutoffSensitivityAnalysis,
        interactive: bool = None,
        **kwargs
    ) -> Union[Tuple[Figure, FigureCanvas], go.Figure]:
        """
        Create comprehensive analysis dashboard.

        Args:
            curve: GradeTonnageCurve object
            sensitivity: CutoffSensitivityAnalysis object
            interactive: Whether to use interactive backend
            **kwargs: Additional arguments

        Returns:
            Dashboard plot object
        """
        interactive = interactive if interactive is not None else self.config.interactive

        if interactive and self.plotly_engine:
            return self.plotly_engine.create_interactive_dashboard(
                curve, sensitivity, **kwargs
            )
        else:
            return self.matplotlib_engine.plot_multi_criteria_dashboard(
                curve, sensitivity, **kwargs
            )

    def export_plot(
        self,
        plot_obj: Union[Tuple[Figure, FigureCanvas], go.Figure],
        filename: str,
        format: str = None
    ) -> bool:
        """
        Export plot to file.

        Args:
            plot_obj: Plot object to export
            filename: Output filename
            format: Export format

        Returns:
            True if successful
        """
        if isinstance(plot_obj, tuple) and len(plot_obj) == 2:
            # Matplotlib Figure and Canvas
            fig, canvas = plot_obj
            return self.matplotlib_engine.export_plot(fig, filename, format)
        elif hasattr(plot_obj, 'write_image') and PLOTLY_AVAILABLE:
            # Plotly Figure
            format = format or 'png'
            try:
                plot_obj.write_image(f"{filename}.{format}")
                logger.info(f"Interactive plot exported to {filename}.{format}")
                return True
            except Exception as e:
                logger.error(f"Failed to export plotly plot: {e}")
                return False
        else:
            logger.error("Unsupported plot object type for export")
            return False


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_publication_quality_plot(
    curve: GradeTonnageCurve,
    sensitivity: CutoffSensitivityAnalysis = None,
    plot_type: PlotType = PlotType.MULTI_CRITERIA_DASHBOARD,
    output_filename: str = "mining_analysis_report",
    config: PlotConfig = None
) -> bool:
    """
    Create publication-quality plot for reports.

    Args:
        curve: GradeTonnageCurve object
        sensitivity: Optional CutoffSensitivityAnalysis object
        plot_type: Type of plot to create
        output_filename: Output filename (without extension)
        config: Plot configuration

    Returns:
        True if successful
    """
    config = config or PlotConfig()
    config.dpi = 300  # High resolution for publications
    config.export_format = 'pdf'  # Vector format for publications

    coordinator = MiningVisualizationCoordinator(config)

    try:
        if plot_type == PlotType.MULTI_CRITERIA_DASHBOARD and sensitivity:
            plot_obj = coordinator.create_dashboard(curve, sensitivity, interactive=False)
        elif plot_type == PlotType.GRADE_TONNAGE_CURVE:
            plot_obj = coordinator.create_grade_tonnage_plot(curve, interactive=False)
        elif plot_type == PlotType.NPV_OPTIMIZATION and sensitivity:
            plot_obj = coordinator.create_npv_optimization_plot(sensitivity, interactive=False)
        else:
            logger.error(f"Cannot create plot type {plot_type} with available data")
            return False

        return coordinator.export_plot(plot_obj, output_filename, 'pdf')

    except Exception as e:
        logger.error(f"Failed to create publication plot: {e}")
        return False


def quick_plot(
    curve: GradeTonnageCurve,
    sensitivity: CutoffSensitivityAnalysis = None,
    interactive: bool = True
) -> Union[Tuple[Figure, FigureCanvas], go.Figure]:
    """
    Quick plotting function for exploratory analysis.

    Args:
        curve: GradeTonnageCurve object
        sensitivity: Optional CutoffSensitivityAnalysis object
        interactive: Whether to use interactive plots

    Returns:
        Plot object for display
    """
    config = PlotConfig(interactive=interactive, figsize=(12, 8))
    coordinator = MiningVisualizationCoordinator(config)

    if sensitivity:
        return coordinator.create_dashboard(curve, sensitivity, interactive=interactive)
    else:
        return coordinator.create_grade_tonnage_plot(curve, interactive=interactive)


# Export main classes and functions
__all__ = [
    'PlotConfig',
    'PlotStyle',
    'PlotBackend',
    'PlotType',
    'ColorScheme',
    'MatplotlibVisualizationEngine',
    'PlotlyVisualizationEngine',
    'MiningVisualizationCoordinator',
    'create_publication_quality_plot',
    'quick_plot'
]
