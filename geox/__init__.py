"""
GeoX namespace package.

Provides modular subsystems for:
- drillholes: Drillhole data handling
- rendering: Visualization utilities
"""

# Expose submodules for convenient import
from . import drillholes
from . import rendering

__all__ = [
    "drillholes",
    "rendering",
]
