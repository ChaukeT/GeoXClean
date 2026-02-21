"""
Environment, Social, and Governance (ESG) Module

Comprehensive ESG tracking and reporting including:
- Carbon footprint and energy consumption
- Water balance and tailings management
- Waste rock and land disturbance
- Governance and compliance reporting (GRI, ICMM, TCFD, SASB)

Integrates with mining schedules to provide activity-based ESG metrics.

Author: BlockModelViewer Team
Date: 2025-11-06
"""

from .dataclasses import EmissionFactor, WaterNode, ESGReport

__all__ = [
    'EmissionFactor',
    'WaterNode',
    'ESGReport'
]
