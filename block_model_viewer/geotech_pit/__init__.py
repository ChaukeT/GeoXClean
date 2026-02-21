"""
Geotechnical Pit Analysis - Slope stability for open pits.
"""

from .slope_failure_surface import FailureSurface2D, FailureSurface3D, generate_circular_surfaces, generate_3d_surfaces
from .limit_equilibrium_2d import SlopeLEM2DConfig, LEM2DResult, compute_fos_2d, search_critical_surface_2d
from .limit_equilibrium_3d import SlopeLEM3DConfig, LEM3DResult, compute_fos_3d, search_critical_surface_3d
from .slope_probabilistic import ProbSlopeConfig, ProbSlopeResult, run_probabilistic_slope
from .slope_design import BenchDesignRule, BenchDesignSet, suggest_bench_design, evaluate_design_against_results

__all__ = [
    'FailureSurface2D',
    'FailureSurface3D',
    'generate_circular_surfaces',
    'generate_3d_surfaces',
    'SlopeLEM2DConfig',
    'LEM2DResult',
    'compute_fos_2d',
    'search_critical_surface_2d',
    'SlopeLEM3DConfig',
    'LEM3DResult',
    'compute_fos_3d',
    'search_critical_surface_3d',
    'ProbSlopeConfig',
    'ProbSlopeResult',
    'run_probabilistic_slope',
    'BenchDesignRule',
    'BenchDesignSet',
    'suggest_bench_design',
    'evaluate_design_against_results',
]

