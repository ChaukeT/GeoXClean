"""
Scan Analysis Subsystem
========================

Production-grade scan analysis pipeline for drone scans (point clouds and meshes).
Provides fragmentation analysis, PSD computation, and surface metrics for mining operations.

This subsystem is completely separate from drillholes, geostats, and block models.
All transformations are explicit, parameterized, versioned, persisted, and reproducible.
"""

from .scan_models import (
    ScanData,
    ValidationReport,
    ValidationViolation,
    CleaningReport,
    SegmentationParams,
    RegionGrowingParams,
    DBSCANParams,
    FragmentMetrics,
    PSDResults,
    ScanProcessingMode
)

__all__ = [
    'ScanData',
    'ValidationReport',
    'ValidationViolation',
    'CleaningReport',
    'SegmentationParams',
    'RegionGrowingParams',
    'DBSCANParams',
    'FragmentMetrics',
    'PSDResults',
    'ScanProcessingMode'
]
