"""
Geostatistics Module

Provides Universal Kriging, Co-Kriging, and Indicator Kriging engines.
"""

from .universal_kriging import UniversalKriging3D, DriftModel
from .cokriging3d import CoKrigingConfig
from ..models.geostat_results import CoKrigingResults as CoKrigingResult
from .indicator_kriging import IKConfig, IKResult, run_indicator_kriging_job

__all__ = [
    'UniversalKriging3D',
    'DriftModel',
    'CoKrigingConfig',
    'CoKrigingResult',
    'IKConfig',  # Backward compatibility
    'IKResult',  # Backward compatibility
    'run_indicator_kriging_job',  # Main API
]

