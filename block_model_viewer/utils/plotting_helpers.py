"""
Shared plotting utility functions for Matplotlib charts.

Centralizes common plotting logic used across multiple panels.
"""

import logging
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

logger = logging.getLogger(__name__)


def create_figure(figsize: Tuple[float, float] = (8, 6), dpi: int = 100) -> Tuple[Figure, FigureCanvas]:
    """
    Create a matplotlib figure with Qt canvas.
    
    Args:
        figsize: Figure size in inches (width, height)
        dpi: Dots per inch
    
    Returns:
        Tuple of (Figure, FigureCanvas)
    """
    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvas(fig)
    return fig, canvas


def plot_histogram(
    ax,
    data: np.ndarray,
    bins: int = 30,
    title: str = "Histogram",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    color: str = 'skyblue',
    show_stats: bool = True
) -> None:
    """
    Create a histogram plot with optional statistics.
    
    Args:
        ax: Matplotlib axes
        data: Data array
        bins: Number of bins
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Bar color
        show_stats: Whether to show statistics text
    """
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return
    
    # Plot histogram
    n, bins_edges, patches = ax.hist(data, bins=bins, color=color, edgecolor='black', alpha=0.7)
    
    # Add statistics text box
    if show_stats:
        stats_text = (
            f"Count: {len(data)}\n"
            f"Mean: {np.mean(data):.2f}\n"
            f"Std: {np.std(data):.2f}\n"
            f"Min: {np.min(data):.2f}\n"
            f"Max: {np.max(data):.2f}"
        )
        ax.text(
            0.98, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9
        )
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_scatter(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Scatter Plot",
    xlabel: str = "X",
    ylabel: str = "Y",
    color: str = 'blue',
    alpha: float = 0.6,
    add_trendline: bool = True
) -> None:
    """
    Create a scatter plot with optional trendline.
    
    Args:
        ax: Matplotlib axes
        x: X-axis data
        y: Y-axis data
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        color: Point color
        alpha: Point transparency
        add_trendline: Whether to add linear trendline
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return
    
    # Scatter plot
    ax.scatter(x, y, c=color, alpha=alpha, s=30, edgecolors='black', linewidth=0.5)
    
    # Add trendline
    if add_trendline and len(x) > 1:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'y={z[0]:.2f}x+{z[1]:.2f}')
        
        # Calculate R²
        yhat = p(x)
        ybar = np.mean(y)
        ssreg = np.sum((yhat - ybar)**2)
        sstot = np.sum((y - ybar)**2)
        r_squared = ssreg / sstot if sstot != 0 else 0
        
        ax.text(
            0.05, 0.95, f'R² = {r_squared:.3f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        ax.legend()
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def plot_boxplot(
    ax,
    data: List[np.ndarray],
    labels: List[str],
    title: str = "Box Plot",
    ylabel: str = "Value",
    showmeans: bool = True
) -> None:
    """
    Create a box plot.
    
    Args:
        ax: Matplotlib axes
        data: List of data arrays
        labels: List of labels for each dataset
        title: Plot title
        ylabel: Y-axis label
        showmeans: Whether to show mean markers
    """
    # Remove NaN values from each dataset
    data = [d[~np.isnan(d)] for d in data]
    
    if all(len(d) == 0 for d in data):
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return
    
    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showmeans=showmeans,
        meanprops=dict(marker='D', markerfacecolor='red', markersize=6)
    )
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')


def plot_cumulative(
    ax,
    data: np.ndarray,
    title: str = "Cumulative Distribution",
    xlabel: str = "Value",
    ylabel: str = "Cumulative %",
    log_scale: bool = False
) -> None:
    """
    Create a cumulative distribution plot.
    
    Args:
        ax: Matplotlib axes
        data: Data array
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        log_scale: Use logarithmic scale
    """
    # Remove NaN values
    data = data[~np.isnan(data)]
    
    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return
    
    # Sort data
    sorted_data = np.sort(data)
    cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    
    if log_scale:
        ax.semilogy(sorted_data, cumulative, linewidth=2, color='blue')
    else:
        ax.plot(sorted_data, cumulative, linewidth=2, color='blue')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])


def plot_grade_tonnage(
    ax,
    grades: np.ndarray,
    tonnages: np.ndarray,
    title: str = "Grade-Tonnage Curve",
    xlabel: str = "Cutoff Grade",
    ylabel_left: str = "Tonnage",
    ylabel_right: str = "Average Grade",
    color_tonnage: str = 'blue',
    color_grade: str = 'red'
) -> None:
    """
    Create a grade-tonnage curve (dual y-axis).
    
    Args:
        ax: Matplotlib axes
        grades: Cutoff grades array
        tonnages: Tonnage above cutoff array
        title: Plot title
        xlabel: X-axis label
        ylabel_left: Left y-axis label (tonnage)
        ylabel_right: Right y-axis label (grade)
        color_tonnage: Tonnage line color
        color_grade: Grade line color
    """
    if len(grades) == 0 or len(tonnages) == 0:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
        return
    
    # Plot tonnage on primary y-axis
    ax.plot(grades, tonnages, linewidth=2, color=color_tonnage, marker='o', label='Tonnage')
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel_left, color=color_tonnage, fontweight='bold')
    ax.tick_params(axis='y', labelcolor=color_tonnage)
    
    # Create second y-axis for grade
    ax2 = ax.twinx()
    ax2.plot(grades, grades, linewidth=2, color=color_grade, marker='s', linestyle='--', label='Avg Grade')
    ax2.set_ylabel(ylabel_right, color=color_grade, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_grade)
    
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')


def apply_theme(fig: Figure, theme: str = 'default') -> None:
    """
    Apply visual theme to figure.
    
    Args:
        fig: Matplotlib figure
        theme: Theme name ('default', 'dark', 'minimal')
    """
    if theme == 'dark':
        fig.patch.set_facecolor('#2b2b2b')
        for ax in fig.get_axes():
            ax.set_facecolor('#1e1e1e')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
    
    elif theme == 'minimal':
        for ax in fig.get_axes():
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.5)
            ax.spines['left'].set_linewidth(0.5)


def save_figure(
    fig: Figure,
    filepath: str,
    dpi: int = 300,
    bbox_inches: str = 'tight'
) -> bool:
    """
    Save figure to file.
    
    Args:
        fig: Matplotlib figure
        filepath: Output file path
        dpi: Resolution
        bbox_inches: Bounding box setting
    
    Returns:
        True if successful
    """
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        logger.info(f"Saved figure to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving figure: {e}", exc_info=True)
        return False

















