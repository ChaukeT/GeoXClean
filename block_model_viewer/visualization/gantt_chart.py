"""
Interactive Gantt Chart for Production Schedule Visualization

This module provides interactive Gantt chart generation for underground mining
production schedules, showing stope extraction timelines, resource allocation,
and mining sequence visualization.

Features:
- Interactive matplotlib-based Gantt charts with hover tooltips
- Color-coding by stope value, grade, or custom attributes
- Period-based timeline visualization
- Export to PNG/PDF formats
- Customizable bar colors and labels

Author: Mining Software Team
Date: 2025
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.dates as mdates
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GanttChart:
    """
    Creates interactive Gantt charts for underground mining production schedules.
    
    This class visualizes the mining sequence showing which stopes are extracted
    in each time period, with interactive features and customizable appearance.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (14, 8), dpi: int = 100):
        """
        Initialize Gantt chart generator.
        
        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        self.fig = None
        self.ax = None
        
    def create_schedule_gantt(
        self,
        schedule: Dict[int, List[Any]],
        stope_attributes: Optional[Dict[int, Dict[str, float]]] = None,
        color_by: str = 'period',
        title: str = 'Underground Mining Production Schedule',
        period_duration_days: int = 90,
        start_date: Optional[datetime] = None,
        show_labels: bool = True,
        colormap: str = 'tab20'
    ) -> plt.Figure:
        """
        Create Gantt chart from production schedule.
        
        Args:
            schedule: Dict mapping period number to list of stope objects/IDs
            stope_attributes: Optional dict of stope_id -> {'nsr': float, 'grade': float, 'tonnes': float}
            color_by: Color bars by 'period', 'nsr', 'grade', or 'tonnes'
            title: Chart title
            period_duration_days: Duration of each mining period in days
            start_date: Start date for schedule (defaults to today)
            show_labels: Show stope labels on bars
            colormap: Matplotlib colormap name
            
        Returns:
            Matplotlib figure object
        """
        if not schedule:
            logger.warning("Empty schedule provided")
            return None
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Determine start date
        if start_date is None:
            start_date = datetime.now()
        
        # Prepare data
        periods = sorted(schedule.keys())
        num_periods = len(periods)
        
        # Calculate positions and colors
        bars_data = []
        y_pos = 0
        y_labels = []
        y_positions = []
        
        for period_idx, period in enumerate(periods):
            stopes = schedule[period]
            
            if not stopes:
                continue
            
            # Calculate period start and end dates
            period_start = start_date + timedelta(days=period * period_duration_days)
            period_end = period_start + timedelta(days=period_duration_days)
            
            for stope in stopes:
                # Extract stope ID
                stope_id = stope.stope_id if hasattr(stope, 'stope_id') else stope
                
                # Get attributes
                attrs = {}
                if stope_attributes and stope_id in stope_attributes:
                    attrs = stope_attributes[stope_id]
                elif hasattr(stope, 'nsr'):
                    attrs = {
                        'nsr': getattr(stope, 'nsr', 0),
                        'grade': getattr(stope, 'grade', 0),
                        'tonnes': getattr(stope, 'tonnes', 0)
                    }
                
                # Determine color value
                if color_by == 'period':
                    color_value = period_idx
                elif color_by == 'nsr':
                    color_value = attrs.get('nsr', 0)
                elif color_by == 'grade':
                    color_value = attrs.get('grade', 0)
                elif color_by == 'tonnes':
                    color_value = attrs.get('tonnes', 0)
                else:
                    color_value = period_idx
                
                # Store bar data
                bars_data.append({
                    'stope_id': stope_id,
                    'period': period,
                    'start': period_start,
                    'end': period_end,
                    'y_pos': y_pos,
                    'color_value': color_value,
                    'attributes': attrs
                })
                
                y_labels.append(f"Stope {stope_id}")
                y_positions.append(y_pos)
                y_pos += 1
        
        if not bars_data:
            logger.warning("No bars to plot")
            return None
        
        # Normalize color values
        color_values = np.array([b['color_value'] for b in bars_data])
        if color_by != 'period':
            # Normalize to 0-1 for continuous colormaps
            vmin, vmax = color_values.min(), color_values.max()
            if vmax > vmin:
                color_values_norm = (color_values - vmin) / (vmax - vmin)
            else:
                color_values_norm = np.zeros_like(color_values)
        else:
            # Use discrete colors for periods
            color_values_norm = color_values / max(color_values.max(), 1)
        
        # Get colormap
        cmap = plt.get_cmap(colormap)
        
        # Plot bars
        for i, bar in enumerate(bars_data):
            # Convert dates to matplotlib date format
            start_num = mdates.date2num(bar['start'])
            end_num = mdates.date2num(bar['end'])
            duration = end_num - start_num
            
            # Get color
            color = cmap(color_values_norm[i])
            
            # Create bar
            rect = Rectangle(
                (start_num, bar['y_pos'] - 0.4),
                duration,
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            self.ax.add_patch(rect)
            
            # Add label if requested
            if show_labels:
                label_x = start_num + duration / 2
                label_y = bar['y_pos']
                self.ax.text(
                    label_x,
                    label_y,
                    str(bar['stope_id']),
                    ha='center',
                    va='center',
                    fontsize=8,
                    fontweight='bold',
                    color='white' if color_values_norm[i] < 0.5 else 'black'
                )
        
        # Format x-axis as dates
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Set y-axis
        self.ax.set_yticks(y_positions)
        self.ax.set_yticklabels(y_labels, fontsize=8)
        self.ax.set_ylim(-1, len(bars_data))
        
        # Set labels and title
        self.ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Stopes', fontsize=12, fontweight='bold')
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        self.ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add colorbar if coloring by continuous variable
        if color_by != 'period':
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            cbar = self.fig.colorbar(sm, ax=self.ax, pad=0.02)
            cbar.set_label(color_by.upper(), fontsize=10, fontweight='bold')
        
        # Tight layout
        plt.tight_layout()
        
        # Store for interactive features
        self._bars_data = bars_data
        self._setup_interactivity()
        
        return self.fig
    
    def create_period_summary_gantt(
        self,
        schedule: Dict[int, List[Any]],
        stope_attributes: Optional[Dict[int, Dict[str, float]]] = None,
        title: str = 'Production Schedule by Period',
        period_duration_days: int = 90,
        start_date: Optional[datetime] = None,
        show_stats: bool = True
    ) -> plt.Figure:
        """
        Create simplified Gantt chart showing periods with aggregate statistics.
        
        Args:
            schedule: Dict mapping period number to list of stope objects/IDs
            stope_attributes: Optional dict of stope_id -> attributes
            title: Chart title
            period_duration_days: Duration of each period in days
            start_date: Start date (defaults to today)
            show_stats: Show statistics on bars (stope count, total tonnes, avg grade)
            
        Returns:
            Matplotlib figure object
        """
        if not schedule:
            logger.warning("Empty schedule provided")
            return None
        
        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        if start_date is None:
            start_date = datetime.now()
        
        periods = sorted(schedule.keys())
        y_positions = []
        y_labels = []
        
        for i, period in enumerate(periods):
            stopes = schedule[period]
            
            # Calculate period dates
            period_start = start_date + timedelta(days=period * period_duration_days)
            period_end = period_start + timedelta(days=period_duration_days)
            
            # Calculate statistics
            num_stopes = len(stopes)
            total_tonnes = 0
            total_grade = 0
            total_nsr = 0
            
            for stope in stopes:
                stope_id = stope.stope_id if hasattr(stope, 'stope_id') else stope
                
                if stope_attributes and stope_id in stope_attributes:
                    attrs = stope_attributes[stope_id]
                    total_tonnes += attrs.get('tonnes', 0)
                    total_grade += attrs.get('grade', 0) * attrs.get('tonnes', 0)
                    total_nsr += attrs.get('nsr', 0)
                elif hasattr(stope, 'tonnes'):
                    total_tonnes += getattr(stope, 'tonnes', 0)
                    total_grade += getattr(stope, 'grade', 0) * getattr(stope, 'tonnes', 0)
                    total_nsr += getattr(stope, 'nsr', 0)
            
            avg_grade = total_grade / total_tonnes if total_tonnes > 0 else 0
            
            # Convert dates
            start_num = mdates.date2num(period_start)
            end_num = mdates.date2num(period_end)
            duration = end_num - start_num
            
            # Color based on value (green = high, red = low)
            color_intensity = min(1.0, total_nsr / (num_stopes * 500))  # Normalize to 0-1
            color = plt.cm.RdYlGn(color_intensity)
            
            # Create bar
            rect = Rectangle(
                (start_num, i - 0.4),
                duration,
                0.8,
                facecolor=color,
                edgecolor='black',
                linewidth=1,
                alpha=0.8
            )
            self.ax.add_patch(rect)
            
            # Add statistics text
            if show_stats:
                label_x = start_num + duration / 2
                label_y = i
                
                stats_text = f"{num_stopes} stopes\n{total_tonnes:.0f}t"
                if avg_grade > 0:
                    stats_text += f"\n{avg_grade:.2f}%"
                
                self.ax.text(
                    label_x,
                    label_y,
                    stats_text,
                    ha='center',
                    va='center',
                    fontsize=9,
                    fontweight='bold',
                    color='white' if color_intensity < 0.5 else 'black'
                )
            
            y_positions.append(i)
            y_labels.append(f"Period {period}")
        
        # Format axes
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(self.ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        self.ax.set_yticks(y_positions)
        self.ax.set_yticklabels(y_labels, fontsize=10)
        self.ax.set_ylim(-1, len(periods))
        
        self.ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Mining Period', fontsize=12, fontweight='bold')
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        self.ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        return self.fig
    
    def _setup_interactivity(self):
        """Setup interactive hover tooltips for Gantt chart."""
        if not hasattr(self, '_bars_data') or not self._bars_data:
            return
        
        # Create annotation for hover text
        annot = self.ax.annotate(
            "",
            xy=(0, 0),
            xytext=(20, 20),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
            arrowprops=dict(arrowstyle="->"),
            fontsize=9,
            zorder=1000
        )
        annot.set_visible(False)
        
        def hover(event):
            """Handle hover events."""
            if event.inaxes != self.ax:
                if annot.get_visible():
                    annot.set_visible(False)
                    self.fig.canvas.draw_idle()
                return
            
            # Check if mouse is over any bar
            for bar in self._bars_data:
                start_num = mdates.date2num(bar['start'])
                end_num = mdates.date2num(bar['end'])
                
                if (start_num <= event.xdata <= end_num and
                    bar['y_pos'] - 0.4 <= event.ydata <= bar['y_pos'] + 0.4):
                    
                    # Build tooltip text
                    text_lines = [f"Stope: {bar['stope_id']}", f"Period: {bar['period']}"]
                    
                    attrs = bar['attributes']
                    if attrs:
                        if 'nsr' in attrs:
                            text_lines.append(f"NSR: ${attrs['nsr']:.2f}")
                        if 'grade' in attrs:
                            text_lines.append(f"Grade: {attrs['grade']:.2f}%")
                        if 'tonnes' in attrs:
                            text_lines.append(f"Tonnes: {attrs['tonnes']:.0f}")
                    
                    text_lines.append(f"Start: {bar['start'].strftime('%Y-%m-%d')}")
                    text_lines.append(f"End: {bar['end'].strftime('%Y-%m-%d')}")
                    
                    annot.xy = (event.xdata, event.ydata)
                    annot.set_text("\n".join(text_lines))
                    annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                    return
            
            # No bar found
            if annot.get_visible():
                annot.set_visible(False)
                self.fig.canvas.draw_idle()
        
        # Connect event
        self.fig.canvas.mpl_connect("motion_notify_event", hover)
    
    def save_chart(self, filepath: str, dpi: Optional[int] = None):
        """
        Save Gantt chart to file.
        
        Args:
            filepath: Output file path (.png, .pdf, .svg)
            dpi: Resolution for raster formats (defaults to figure dpi)
        """
        if self.fig is None:
            logger.error("No figure to save. Create a chart first.")
            return
        
        save_dpi = dpi if dpi else self.dpi
        self.fig.savefig(filepath, dpi=save_dpi, bbox_inches='tight')
        logger.info(f"Gantt chart saved to {filepath}")


# Convenience functions

def create_schedule_gantt(
    schedule: Dict[int, List[Any]],
    stope_attributes: Optional[Dict[int, Dict[str, float]]] = None,
    color_by: str = 'period',
    **kwargs
) -> plt.Figure:
    """
    Quick function to create a schedule Gantt chart.
    
    Args:
        schedule: Dict mapping period to list of stopes
        stope_attributes: Optional stope attributes dict
        color_by: Color bars by 'period', 'nsr', 'grade', or 'tonnes'
        **kwargs: Additional arguments for GanttChart.create_schedule_gantt
        
    Returns:
        Matplotlib figure
    """
    gantt = GanttChart()
    return gantt.create_schedule_gantt(schedule, stope_attributes, color_by, **kwargs)


def create_period_summary_gantt(
    schedule: Dict[int, List[Any]],
    stope_attributes: Optional[Dict[int, Dict[str, float]]] = None,
    **kwargs
) -> plt.Figure:
    """
    Quick function to create a period summary Gantt chart.
    
    Args:
        schedule: Dict mapping period to list of stopes
        stope_attributes: Optional stope attributes dict
        **kwargs: Additional arguments for GanttChart.create_period_summary_gantt
        
    Returns:
        Matplotlib figure
    """
    gantt = GanttChart()
    return gantt.create_period_summary_gantt(schedule, stope_attributes, **kwargs)


def export_gantt_to_file(
    schedule: Dict[int, List[Any]],
    filepath: str,
    chart_type: str = 'schedule',
    **kwargs
):
    """
    Create and export Gantt chart to file in one step.
    
    Args:
        schedule: Production schedule dict
        filepath: Output file path
        chart_type: 'schedule' or 'period_summary'
        **kwargs: Additional chart creation arguments
    """
    gantt = GanttChart()
    
    if chart_type == 'schedule':
        fig = gantt.create_schedule_gantt(schedule, **kwargs)
    else:
        fig = gantt.create_period_summary_gantt(schedule, **kwargs)
    
    if fig:
        gantt.save_chart(filepath)
        plt.close(fig)
