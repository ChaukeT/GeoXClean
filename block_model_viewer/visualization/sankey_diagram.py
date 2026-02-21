"""
Sankey Diagram for Water Balance Visualization

This module provides Sankey flow diagram generation for water balance analysis
in mining operations, showing water flows between different nodes in the system.

Features:
- Interactive matplotlib/plotly-based Sankey diagrams
- Water balance flow visualization (pit → process → tailings → recycling)
- Customizable node colors and labels
- Flow thickness proportional to water volume
- Export to PNG/PDF/HTML formats
- Integration with ESG water management metrics

Author: Mining Software Team
Date: 2025
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)

# Try to import plotly for interactive diagrams
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not available. Interactive Sankey diagrams will be limited.")


class WaterBalanceSankey:
    """
    Creates Sankey diagrams for water balance visualization in mining operations.
    
    This class visualizes water flows between different nodes in a mining system:
    - Pit dewatering
    - Process water
    - Tailings storage
    - Water recycling
    - Discharge/evaporation
    """
    
    def __init__(self, backend: str = 'plotly'):
        """
        Initialize Sankey diagram generator.
        
        Args:
            backend: Rendering backend ('plotly' or 'matplotlib')
        """
        self.backend = backend if (backend == 'matplotlib' or PLOTLY_AVAILABLE) else 'matplotlib'
        self.fig = None
        
        if backend == 'plotly' and not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, falling back to matplotlib")
            self.backend = 'matplotlib'
    
    def create_water_balance_diagram(
        self,
        flows: Dict[str, float],
        node_colors: Optional[Dict[str, str]] = None,
        title: str = 'Water Balance - Mining Operation',
        units: str = 'm³/day',
        show_values: bool = True,
        figsize: Tuple[int, int] = (14, 8)
    ) -> Any:
        """
        Create Sankey diagram from water balance flows.
        
        Args:
            flows: Dict mapping flow names to volumes, e.g.:
                {
                    'pit_to_process': 1000,
                    'process_to_tailings': 800,
                    'tailings_to_recycle': 600,
                    'recycle_to_process': 600,
                    'process_to_discharge': 200,
                    'tailings_to_evap': 200
                }
            node_colors: Optional dict mapping node names to colors
            title: Diagram title
            units: Flow units (e.g., 'm³/day', 'ML/day')
            show_values: Show flow values on arrows
            figsize: Figure size for matplotlib backend
            
        Returns:
            Plotly figure or matplotlib figure depending on backend
        """
        if self.backend == 'plotly':
            return self._create_plotly_sankey(flows, node_colors, title, units, show_values)
        else:
            return self._create_matplotlib_sankey(flows, node_colors, title, units, show_values, figsize)
    
    def _create_plotly_sankey(
        self,
        flows: Dict[str, float],
        node_colors: Optional[Dict[str, str]],
        title: str,
        units: str,
        show_values: bool
    ) -> 'go.Figure':
        """Create interactive Plotly Sankey diagram."""
        
        # Define standard nodes for water balance
        node_list = [
            'Pit Dewatering',
            'Fresh Water',
            'Process Plant',
            'Tailings Storage',
            'Recycled Water',
            'Discharge',
            'Evaporation',
            'Seepage'
        ]
        
        # Create node index mapping
        node_indices = {name: i for i, name in enumerate(node_list)}
        
        # Default colors if not provided
        if node_colors is None:
            node_colors = {
                'Pit Dewatering': '#1f77b4',  # Blue
                'Fresh Water': '#00bfff',      # Light blue
                'Process Plant': '#ff7f0e',    # Orange
                'Tailings Storage': '#d62728',  # Red
                'Recycled Water': '#2ca02c',   # Green
                'Discharge': '#9467bd',        # Purple
                'Evaporation': '#8c564b',      # Brown
                'Seepage': '#e377c2'           # Pink
            }
        
        # Build flows (source, target, value)
        sources = []
        targets = []
        values = []
        labels_list = []
        
        # Parse flow dictionary and map to nodes
        flow_mapping = {
            'pit_to_process': ('Pit Dewatering', 'Process Plant'),
            'fresh_to_process': ('Fresh Water', 'Process Plant'),
            'process_to_tailings': ('Process Plant', 'Tailings Storage'),
            'process_to_discharge': ('Process Plant', 'Discharge'),
            'tailings_to_recycle': ('Tailings Storage', 'Recycled Water'),
            'tailings_to_evap': ('Tailings Storage', 'Evaporation'),
            'tailings_to_seepage': ('Tailings Storage', 'Seepage'),
            'recycle_to_process': ('Recycled Water', 'Process Plant'),
            'pit_to_discharge': ('Pit Dewatering', 'Discharge')
        }
        
        for flow_name, volume in flows.items():
            if volume <= 0:
                continue
                
            if flow_name in flow_mapping:
                source_name, target_name = flow_mapping[flow_name]
                
                if source_name in node_indices and target_name in node_indices:
                    sources.append(node_indices[source_name])
                    targets.append(node_indices[target_name])
                    values.append(volume)
                    
                    if show_values:
                        labels_list.append(f"{volume:.1f} {units}")
                    else:
                        labels_list.append("")
        
        if not sources:
            logger.error("No valid flows to visualize")
            return None
        
        # Create Plotly Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_list,
                color=[node_colors.get(name, '#gray') for name in node_list]
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                label=labels_list if show_values else None,
                color=['rgba(0,0,255,0.3)'] * len(sources)  # Semi-transparent blue
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='black', family='Arial Black')
            ),
            font=dict(size=12),
            height=600,
            width=1000
        )
        
        self.fig = fig
        return fig
    
    def _create_matplotlib_sankey(
        self,
        flows: Dict[str, float],
        node_colors: Optional[Dict[str, str]],
        title: str,
        units: str,
        show_values: bool,
        figsize: Tuple[int, int]
    ) -> plt.Figure:
        """Create matplotlib-based Sankey diagram."""
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Matplotlib Sankey is more limited - we'll create a simplified version
        # showing main water flows
        
        # Extract key flows
        pit_water = flows.get('pit_to_process', 0)
        fresh_water = flows.get('fresh_to_process', 0)
        to_tailings = flows.get('process_to_tailings', 0)
        recycled = flows.get('tailings_to_recycle', 0)
        recycle_back = flows.get('recycle_to_process', 0)
        evaporation = flows.get('tailings_to_evap', 0)
        discharge = flows.get('process_to_discharge', 0) + flows.get('pit_to_discharge', 0)
        
        # Calculate total inflow and outflow
        total_in = pit_water + fresh_water
        total_out = evaporation + discharge
        
        # Create Sankey diagram
        sankey = Sankey(
            ax=ax,
            scale=0.01,
            offset=0.3,
            head_angle=120,
            format='%.0f',
            unit=f' {units}'
        )
        
        # Main process flows
        # Inflows: pit water, fresh water
        # Outflows: to tailings, discharge
        sankey.add(
            flows=[pit_water, fresh_water, recycle_back, -to_tailings, -discharge],
            labels=['Pit', 'Fresh', 'Recycled', 'Tailings', 'Discharge'],
            orientations=[0, 0, 0, 0, -1],
            pathlengths=[0.5, 0.5, 0.5, 0.5, 0.5],
            facecolor='#1f77b4',
            label='Process Plant'
        )
        
        # Tailings flows
        # Inflow: from process
        # Outflows: recycled, evaporation
        sankey.add(
            flows=[to_tailings, -recycled, -evaporation],
            labels=['From Process', 'Recycled', 'Evaporation'],
            orientations=[0, 0, -1],
            prior=0,
            connect=(3, 0),
            facecolor='#d62728',
            label='Tailings'
        )
        
        # Render
        diagrams = sankey.finish()
        
        # Add title
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor='#1f77b4', label='Process Water'),
            mpatches.Patch(facecolor='#d62728', label='Tailings Water'),
            mpatches.Patch(facecolor='#2ca02c', label='Recycled Water')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        # Add summary text
        summary_text = (
            f"Total Inflow: {total_in:.0f} {units}\n"
            f"Total Outflow: {total_out:.0f} {units}\n"
            f"Recycling Rate: {(recycled/to_tailings*100 if to_tailings > 0 else 0):.1f}%"
        )
        ax.text(
            0.98, 0.02,
            summary_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        
        self.fig = fig
        return fig
    
    def create_comprehensive_balance(
        self,
        water_balance: Dict[str, Any],
        title: str = 'Comprehensive Water Balance',
        backend: Optional[str] = None
    ) -> Any:
        """
        Create Sankey diagram from comprehensive water balance data.
        
        Args:
            water_balance: Dict containing water balance metrics:
                {
                    'pit_dewatering': 1000,
                    'fresh_water_supply': 500,
                    'process_consumption': 1500,
                    'tailings_input': 1200,
                    'recycled_water': 800,
                    'evaporation': 300,
                    'seepage': 100,
                    'discharge': 400
                }
            title: Diagram title
            backend: Optional backend override
            
        Returns:
            Figure object
        """
        # Convert water balance metrics to flows
        flows = {}
        
        pit = water_balance.get('pit_dewatering', 0)
        fresh = water_balance.get('fresh_water_supply', 0)
        process_total = water_balance.get('process_consumption', 0)
        tailings_in = water_balance.get('tailings_input', 0)
        recycled = water_balance.get('recycled_water', 0)
        evap = water_balance.get('evaporation', 0)
        seepage = water_balance.get('seepage', 0)
        discharge = water_balance.get('discharge', 0)
        
        # Build flows
        if pit > 0:
            flows['pit_to_process'] = pit * 0.8  # Most to process
            flows['pit_to_discharge'] = pit * 0.2  # Some direct discharge
        
        if fresh > 0:
            flows['fresh_to_process'] = fresh
        
        if tailings_in > 0:
            flows['process_to_tailings'] = tailings_in
        
        if recycled > 0:
            flows['tailings_to_recycle'] = recycled
            flows['recycle_to_process'] = recycled
        
        if evap > 0:
            flows['tailings_to_evap'] = evap
        
        if seepage > 0:
            flows['tailings_to_seepage'] = seepage
        
        if discharge > 0 and 'pit_to_discharge' in flows:
            flows['process_to_discharge'] = discharge - flows['pit_to_discharge']
        elif discharge > 0:
            flows['process_to_discharge'] = discharge
        
        # Use specified backend or default
        original_backend = self.backend
        if backend:
            self.backend = backend
        
        fig = self.create_water_balance_diagram(flows, title=title)
        
        # Restore original backend
        self.backend = original_backend
        
        return fig
    
    def save_diagram(self, filepath: str, **kwargs):
        """
        Save Sankey diagram to file.
        
        Args:
            filepath: Output file path (.png, .pdf, .html)
            **kwargs: Additional save arguments (dpi for matplotlib, etc.)
        """
        if self.fig is None:
            logger.error("No diagram to save. Create a diagram first.")
            return
        
        if self.backend == 'plotly':
            if filepath.endswith('.html'):
                self.fig.write_html(filepath)
            else:
                self.fig.write_image(filepath, **kwargs)
            logger.info(f"Plotly Sankey saved to {filepath}")
        else:
            dpi = kwargs.get('dpi', 150)
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            logger.info(f"Matplotlib Sankey saved to {filepath}")
    
    def calculate_efficiency_metrics(self, flows: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate water efficiency metrics from flows.
        
        Args:
            flows: Flow dictionary
            
        Returns:
            Dict with efficiency metrics:
                - recycling_rate: % of water recycled
                - fresh_water_intensity: Fresh water per unit process water
                - loss_rate: % of water lost (evap + seepage + discharge)
        """
        # Calculate totals
        fresh = flows.get('fresh_to_process', 0)
        recycled = flows.get('tailings_to_recycle', 0)
        to_tailings = flows.get('process_to_tailings', 0)
        evap = flows.get('tailings_to_evap', 0)
        seepage = flows.get('tailings_to_seepage', 0)
        discharge = flows.get('process_to_discharge', 0) + flows.get('pit_to_discharge', 0)
        
        total_in = fresh + flows.get('pit_to_process', 0)
        total_loss = evap + seepage + discharge
        
        # Calculate metrics
        recycling_rate = (recycled / to_tailings * 100) if to_tailings > 0 else 0
        fresh_intensity = (fresh / total_in * 100) if total_in > 0 else 0
        loss_rate = (total_loss / total_in * 100) if total_in > 0 else 0
        
        return {
            'recycling_rate': recycling_rate,
            'fresh_water_intensity': fresh_intensity,
            'loss_rate': loss_rate,
            'total_inflow': total_in,
            'total_outflow': total_loss,
            'water_balance_error': abs(total_in - recycled - total_loss)
        }


# Convenience functions

def create_water_sankey(
    flows: Dict[str, float],
    backend: str = 'plotly',
    **kwargs
) -> Any:
    """
    Quick function to create water balance Sankey diagram.
    
    Args:
        flows: Water flow dictionary
        backend: 'plotly' or 'matplotlib'
        **kwargs: Additional arguments for create_water_balance_diagram
        
    Returns:
        Figure object
    """
    sankey = WaterBalanceSankey(backend=backend)
    return sankey.create_water_balance_diagram(flows, **kwargs)


def create_esg_water_sankey(
    water_balance_data: Dict[str, Any],
    backend: str = 'plotly',
    **kwargs
) -> Any:
    """
    Create Sankey from ESG water balance data.
    
    Args:
        water_balance_data: Water balance metrics dict
        backend: 'plotly' or 'matplotlib'
        **kwargs: Additional arguments
        
    Returns:
        Figure object
    """
    sankey = WaterBalanceSankey(backend=backend)
    return sankey.create_comprehensive_balance(water_balance_data, **kwargs)


def export_sankey_to_file(
    flows: Dict[str, float],
    filepath: str,
    backend: str = 'plotly',
    **kwargs
):
    """
    Create and export Sankey diagram in one step.
    
    Args:
        flows: Water flow dictionary
        filepath: Output file path
        backend: 'plotly' or 'matplotlib'
        **kwargs: Additional diagram arguments
    """
    sankey = WaterBalanceSankey(backend=backend)
    fig = sankey.create_water_balance_diagram(flows, **kwargs)
    if fig:
        sankey.save_diagram(filepath)
