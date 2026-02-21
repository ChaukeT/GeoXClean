"""
GC Decision Engine Spider Chart (Radar Chart) Visualization

Creates radar charts to visualize the decision DNA of blocks,
showing all component scores in a spider/radar format for explainability.
"""

import logging
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)

# Try matplotlib first (better PyQt6 integration)
try:
    # Matplotlib backend is set in main.py
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available - spider charts will be disabled")

# Try plotly as alternative
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_block_dna_matplotlib(
    row: Dict[str, Any],
    block_id: Optional[str] = None,
    figsize: tuple = (8, 8)
) -> Optional[Figure]:
    """
    Create a matplotlib radar chart (spider chart) for block decision DNA.
    
    Args:
        row: Dictionary containing block data with score columns:
             - SCORE_GRADE
             - SCORE_PENALTY
             - SCORE_UNCERT
             - SCORE_GEO
             - SCORE_ECO
             - SCORE_CONTEXT
             - GC_SCORE (optional, for display)
             - GC_ROUTE (optional, for display)
        block_id: Optional block identifier for title
        figsize: Figure size tuple
        
    Returns:
        matplotlib Figure object or None if matplotlib unavailable
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available for spider chart")
        return None
    
    # Extract scores
    categories = [
        'Grade',
        'Cleanliness\n(Penalty)',
        'Low Risk\n(Uncert)',
        'Geo\nConfidence',
        'Economics',
        'Context'
    ]
    
    score_keys = [
        'SCORE_GRADE',
        'SCORE_PENALTY',
        'SCORE_UNCERT',
        'SCORE_GEO',
        'SCORE_ECO',
        'SCORE_CONTEXT'
    ]
    
    values = []
    for key in score_keys:
        val = row.get(key, 0.5)  # Default to 0.5 if missing
        if isinstance(val, (int, float)) and not np.isnan(val):
            values.append(float(val))
        else:
            values.append(0.5)
    
    # Close the loop for radar chart
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig = Figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='polar')
    
    # Plot the data
    ax.plot(angles, values_closed, 'o-', linewidth=2, label='Block Score', color='#1f77b4')
    ax.fill(angles, values_closed, alpha=0.25, color='#1f77b4')
    
    # Add grid circles
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    
    # Add title
    block_id_str = str(block_id) if block_id is not None else row.get('BLOCK_ID', 'Unknown')
    gc_score = row.get('GC_SCORE', None)
    gc_route = row.get('GC_ROUTE', 'UNKNOWN')
    
    title_parts = [f"Block {block_id_str} Decision DNA"]
    if gc_score is not None:
        title_parts.append(f"GC Score: {gc_score:.3f}")
    if gc_route:
        title_parts.append(f"Route: {gc_route}")
    
    ax.set_title('\n'.join(title_parts), size=12, fontweight='bold', pad=20)
    
    # Add value labels at each point
    for angle, value, category in zip(angles[:-1], values, categories):
        ax.text(angle, value + 0.1, f'{value:.2f}', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=9, fontweight='bold')
    
    fig.tight_layout()
    
    return fig


def plot_block_dna_plotly(
    row: Dict[str, Any],
    block_id: Optional[str] = None
) -> Optional['go.Figure']:
    """
    Create a plotly radar chart (spider chart) for block decision DNA.
    
    Args:
        row: Dictionary containing block data with score columns
        block_id: Optional block identifier for title
        
    Returns:
        plotly Figure object or None if plotly unavailable
    """
    if not PLOTLY_AVAILABLE:
        logger.error("Plotly not available for spider chart")
        return None
    
    # Extract scores
    categories = [
        'Grade',
        'Cleanliness (Penalty)',
        'Low Risk (Uncert)',
        'Geo Confidence',
        'Economics',
        'Context'
    ]
    
    score_keys = [
        'SCORE_GRADE',
        'SCORE_PENALTY',
        'SCORE_UNCERT',
        'SCORE_GEO',
        'SCORE_ECO',
        'SCORE_CONTEXT'
    ]
    
    values = []
    for key in score_keys:
        val = row.get(key, 0.5)
        if isinstance(val, (int, float)) and not np.isnan(val):
            values.append(float(val))
        else:
            values.append(0.5)
    
    # Close the loop
    categories_closed = [*categories, categories[0]]
    values_closed = [*values, values[0]]
    
    # Create figure
    block_id_str = str(block_id) if block_id is not None else row.get('BLOCK_ID', 'Unknown')
    gc_score = row.get('GC_SCORE', None)
    gc_route = row.get('GC_ROUTE', 'UNKNOWN')
    
    title_parts = [f"Block {block_id_str} Decision DNA"]
    if gc_score is not None:
        title_parts.append(f"GC Score: {gc_score:.3f} | Route: {gc_route}")
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill='toself',
        name='Block Score',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.25)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11)
            )
        ),
        showlegend=False,
        title={
            'text': '\n'.join(title_parts),
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': 'black'}
        },
        height=600,
        width=600
    )
    
    return fig


def plot_block_dna(
    row: Dict[str, Any],
    block_id: Optional[str] = None,
    backend: str = 'matplotlib',
    figsize: tuple = (8, 8)
):
    """
    Create a radar chart for block decision DNA using the specified backend.
    
    Args:
        row: Dictionary containing block data with score columns
        block_id: Optional block identifier for title
        backend: 'matplotlib' or 'plotly'
        figsize: Figure size for matplotlib (ignored for plotly)
        
    Returns:
        Figure object (matplotlib Figure or plotly Figure) or None
    """
    if backend == 'plotly' and PLOTLY_AVAILABLE:
        return plot_block_dna_plotly(row, block_id)
    elif backend == 'matplotlib' and MATPLOTLIB_AVAILABLE:
        return plot_block_dna_matplotlib(row, block_id, figsize)
    elif MATPLOTLIB_AVAILABLE:
        # Fallback to matplotlib if plotly requested but unavailable
        logger.warning("Plotly unavailable, falling back to matplotlib")
        return plot_block_dna_matplotlib(row, block_id, figsize)
    else:
        logger.error("No plotting backend available")
        return None

