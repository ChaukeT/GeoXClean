"""
Underground Mining Module

Comprehensive underground mine planning and optimization including:
- Stope optimization (maximum closure, morphological methods)
- Cut-and-fill scheduling with MILP
- Ground control and geotechnical analysis
- Ventilation network analysis
- Equipment planning and resource scheduling

Integrates with BlockModelViewer's existing geology and resource modules.

Author: BlockModelViewer Team
Date: 2025-11-06
"""

from .dataclasses import Stope, PeriodKPI, UGCapacities

__all__ = [
    'Stope',
    'PeriodKPI', 
    'UGCapacities'
]
