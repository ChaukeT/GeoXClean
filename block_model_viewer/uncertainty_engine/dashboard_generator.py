"""
Uncertainty Dashboard Generator

Creates comprehensive risk visualization dashboards with:
- KPI summary tables (P10-P50-P90, CV)
- NPV histograms and cumulative distributions
- Fan charts for time series (grade, tonnes, cashflow)
- Risk maps and heatmaps
- Tornado charts for sensitivity
- Correlation matrices
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - visualization will be limited")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False


@dataclass
class DashboardConfig:
    """Configuration for uncertainty dashboard."""
    
    # Figure settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn-v0_8-darkgrid'
    
    # Percentiles to display
    percentiles: List[float] = None
    
    # Color scheme
    color_p10: str = '#d62728'      # Red
    color_p50: str = '#2ca02c'      # Green
    color_p90: str = '#1f77b4'      # Blue
    color_mean: str = '#ff7f0e'     # Orange
    
    # Export settings
    export_format: str = 'png'      # 'png', 'svg', 'pdf'
    export_dpi: int = 300
    
    def __post_init__(self):
        if self.percentiles is None:
            self.percentiles = [10, 50, 90]


class UncertaintyDashboard:
    """
    Generator for uncertainty analysis dashboards.
    """
    
    def __init__(self, config: DashboardConfig = None):
        """Initialize dashboard generator."""
        self.config = config or DashboardConfig()
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(self.config.style)
        
        logger.info("Initialized Uncertainty Dashboard generator")
    
    def create_summary_table(self, results: Any) -> pd.DataFrame:
        """
        Create KPI summary table with percentiles.
        
        Args:
            results: MonteCarloResults or similar with summary_stats
        
        Returns:
            DataFrame with formatted summary statistics
        """
        if hasattr(results, 'summary_stats') and results.summary_stats is not None:
            summary = results.summary_stats.copy()
            
            # Format for display
            summary = summary.round(2)
            
            # Add units column (if available)
            units = {
                'NPV': 'USD M',
                'IRR': '%',
                'Payback_Period': 'years',
                'Ore_Tonnes': 'Mt',
                'Waste_Tonnes': 'Mt',
                'Strip_Ratio': 't:t',
                'Head_Grade': '%',
                'Metal': 't'
            }
            
            if any(k in summary.index for k in units.keys()):
                summary['Units'] = summary.index.map(lambda x: units.get(x, '-'))
            
            return summary
        
        return pd.DataFrame()
    
    def plot_npv_distribution(
        self,
        npv_values: np.ndarray,
        title: str = "NPV Distribution"
    ) -> Figure:
        """
        Plot NPV histogram with percentiles.
        
        Args:
            npv_values: Array of NPV values
            title: Plot title
        
        Returns:
            Matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available")
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
        
        # Histogram
        ax1.hist(npv_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add percentile lines
        p10 = np.percentile(npv_values, 10)
        p50 = np.percentile(npv_values, 50)
        p90 = np.percentile(npv_values, 90)
        
        ax1.axvline(p10, color=self.config.color_p10, linestyle='--', linewidth=2, label=f'P10: ${p10/1e6:.1f}M')
        ax1.axvline(p50, color=self.config.color_p50, linestyle='--', linewidth=2, label=f'P50: ${p50/1e6:.1f}M')
        ax1.axvline(p90, color=self.config.color_p90, linestyle='--', linewidth=2, label=f'P90: ${p90/1e6:.1f}M')
        
        ax1.set_xlabel('NPV (USD M)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_npv = np.sort(npv_values)
        cumulative = np.arange(1, len(sorted_npv) + 1) / len(sorted_npv)
        
        ax2.plot(sorted_npv / 1e6, cumulative * 100, linewidth=2, color='navy')
        ax2.axhline(10, color=self.config.color_p10, linestyle='--', alpha=0.5)
        ax2.axhline(50, color=self.config.color_p50, linestyle='--', alpha=0.5)
        ax2.axhline(90, color=self.config.color_p90, linestyle='--', alpha=0.5)
        ax2.axvline(p10 / 1e6, color=self.config.color_p10, linestyle='--', alpha=0.5)
        ax2.axvline(p50 / 1e6, color=self.config.color_p50, linestyle='--', alpha=0.5)
        ax2.axvline(p90 / 1e6, color=self.config.color_p90, linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('NPV (USD M)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability (%)', fontsize=12)
        ax2.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add probability of loss
        prob_loss = np.sum(npv_values < 0) / len(npv_values) * 100
        ax2.text(0.05, 0.95, f'P(NPV < 0) = {prob_loss:.1f}%',
                transform=ax2.transAxes, fontsize=11,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_fan_chart(
        self,
        annual_data: pd.DataFrame,
        metric_name: str = "Grade",
        ylabel: str = "Head Grade (%)"
    ) -> Figure:
        """
        Plot fan chart showing P10-P50-P90 ribbons over time.
        
        Args:
            annual_data: DataFrame with columns P10, P50, P90, Mean per period
            metric_name: Name of metric
            ylabel: Y-axis label
        
        Returns:
            Matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        periods = annual_data.index
        
        # Plot ribbons
        ax.fill_between(periods, annual_data['P10'], annual_data['P90'],
                        alpha=0.2, color='blue', label='P10-P90 Range')
        
        # Plot lines
        ax.plot(periods, annual_data['P10'], color=self.config.color_p10,
               linestyle='--', linewidth=1.5, label='P10')
        ax.plot(periods, annual_data['P50'], color=self.config.color_p50,
               linestyle='-', linewidth=2, label='P50 (Median)')
        ax.plot(periods, annual_data['P90'], color=self.config.color_p90,
               linestyle='--', linewidth=1.5, label='P90')
        
        if 'Mean' in annual_data.columns:
            ax.plot(periods, annual_data['Mean'], color=self.config.color_mean,
                   linestyle=':', linewidth=2, label='Mean')
        
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{metric_name} Fan Chart - Uncertainty Over Time',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_tornado_chart(
        self,
        correlation_data: pd.DataFrame,
        target_metric: str = 'NPV',
        top_n: int = 10
    ) -> Figure:
        """
        Plot tornado chart showing parameter sensitivity.
        
        Args:
            correlation_data: DataFrame with correlations (output of compute_input_output_correlation)
            target_metric: Target output metric
            top_n: Number of top parameters to show
        
        Returns:
            Matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available")
            return None
        
        if target_metric not in correlation_data.columns:
            logger.error(f"Metric {target_metric} not in correlation data")
            return None
        
        # Get correlations and sort by absolute value
        corr_series = correlation_data[target_metric].abs().sort_values(ascending=False)
        top_params = corr_series.head(top_n)
        
        # Get actual (signed) correlations
        signed_corr = correlation_data.loc[top_params.index, target_metric]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Colors based on positive/negative
        colors = ['green' if c > 0 else 'red' for c in signed_corr]
        
        y_pos = np.arange(len(signed_corr))
        ax.barh(y_pos, signed_corr, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(signed_corr.index)
        ax.set_xlabel('Correlation with ' + target_metric, fontsize=12)
        ax.set_title(f'Sensitivity Analysis - {target_metric} Drivers',
                    fontsize=14, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Legend
        green_patch = mpatches.Patch(color='green', label='Positive Impact', alpha=0.7)
        red_patch = mpatches.Patch(color='red', label='Negative Impact', alpha=0.7)
        ax.legend(handles=[green_patch, red_patch])
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(
        self,
        correlation_data: pd.DataFrame,
        title: str = "Input-Output Correlations"
    ) -> Figure:
        """
        Plot correlation matrix heatmap.
        
        Args:
            correlation_data: Correlation matrix DataFrame
            title: Plot title
        
        Returns:
            Matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        if SEABORN_AVAILABLE:
            sns.heatmap(correlation_data, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, vmin=-1, vmax=1, square=True, ax=ax,
                       cbar_kws={'label': 'Correlation'})
        else:
            # Fallback without seaborn
            im = ax.imshow(correlation_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_xticks(np.arange(len(correlation_data.columns)))
            ax.set_yticks(np.arange(len(correlation_data.index)))
            ax.set_xticklabels(correlation_data.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_data.index)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Correlation')
            
            # Add text annotations
            for i in range(len(correlation_data.index)):
                for j in range(len(correlation_data.columns)):
                    text = ax.text(j, i, f'{correlation_data.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_risk_map_2d(
        self,
        block_model: pd.DataFrame,
        mining_probability: np.ndarray,
        slice_z: Optional[float] = None
    ) -> Figure:
        """
        Plot 2D risk map at specified elevation.
        
        Args:
            block_model: Block model with x, y, z coordinates
            mining_probability: Mining probability per block
            slice_z: Z elevation for slice (None = use median)
        
        Returns:
            Matplotlib Figure
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available")
            return None
        
        # Add probability to block model
        bm = block_model.copy()
        bm['mining_prob'] = mining_probability
        
        # Select slice
        if slice_z is None:
            slice_z = bm['z'].median()
        
        # Get blocks near this elevation (±tolerance)
        tolerance = bm['dz'].median() if 'dz' in bm.columns else 10
        slice_blocks = bm[np.abs(bm['z'] - slice_z) <= tolerance]
        
        if len(slice_blocks) == 0:
            logger.warning("No blocks found at specified elevation")
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Scatter plot colored by probability
        scatter = ax.scatter(slice_blocks['x'], slice_blocks['y'],
                           c=slice_blocks['mining_prob'], cmap='RdYlGn',
                           vmin=0, vmax=1, s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Mining Probability', fontsize=12)
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title(f'Risk Map at Elevation Z={slice_z:.0f}m', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend for risk categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Robust (≥80%)'),
            Patch(facecolor='yellow', label='Marginal (20-80%)'),
            Patch(facecolor='red', label='Fringe (≤20%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def export_dashboard(
        self,
        figures: Dict[str, Figure],
        output_dir: Path,
        prefix: str = "uncertainty"
    ):
        """
        Export all dashboard figures.
        
        Args:
            figures: Dict mapping name -> Figure
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            if fig is None:
                continue
            
            filename = f"{prefix}_{name}.{self.config.export_format}"
            filepath = output_dir / filename
            
            fig.savefig(filepath, dpi=self.config.export_dpi, bbox_inches='tight')
            logger.info(f"Exported {filename}")
        
        logger.info(f"Dashboard exported to {output_dir}")
    
    def create_full_dashboard(
        self,
        monte_carlo_results: Any,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Figure]:
        """
        Create complete dashboard from Monte Carlo results.
        
        Args:
            monte_carlo_results: MonteCarloResults object
            output_dir: Optional directory to export figures
        
        Returns:
            Dict of all generated figures
        """
        figures = {}
        
        # NPV distribution
        npvs = np.array([s.npv for s in monte_carlo_results.simulations])
        figures['npv_distribution'] = self.plot_npv_distribution(npvs)
        
        # Fan charts for annual metrics
        if monte_carlo_results.annual_percentiles:
            for metric, data in monte_carlo_results.annual_percentiles.items():
                ylabel = {
                    'npv': 'NPV (USD M)',
                    'tonnes': 'Tonnes (kt)',
                    'grade': 'Grade (%)',
                    'metal': 'Metal (t)'
                }.get(metric, metric)
                
                figures[f'fan_chart_{metric}'] = self.plot_fan_chart(
                    data, metric_name=metric.title(), ylabel=ylabel
                )
        
        # Tornado chart
        if monte_carlo_results.input_output_correlation is not None:
            figures['tornado_npv'] = self.plot_tornado_chart(
                monte_carlo_results.input_output_correlation, 'NPV'
            )
        
        # Correlation matrix
        if monte_carlo_results.input_output_correlation is not None:
            figures['correlation_matrix'] = self.plot_correlation_matrix(
                monte_carlo_results.input_output_correlation
            )
        
        # Export if directory specified
        if output_dir:
            self.export_dashboard(figures, output_dir)
        
        return figures
