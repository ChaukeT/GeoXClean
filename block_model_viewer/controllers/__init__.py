"""
Controllers package for orchestration layer between UI and Renderer.

Provides a centralized controller architecture with specialized sub-controllers:
- AppController: Main facade that coordinates all sub-controllers
- GeostatsController: Kriging, simulation, variogram, analysis operations
- MiningController: Resource calculation, IRR/NPV, scheduling, planning
- VisController: Visualization, layers, legend, overlays
- DataController: Drillholes, geology, structural, geotechnical
- SurveyDeformationController: Survey-based deformation & subsidence analysis
"""

from .app_controller import AppController, SessionState, PerformanceSettings
from .geostats_controller import GeostatsController
from .mining_controller import MiningController
from .vis_controller import VisController
from .data_controller import DataController
from .survey_deformation_controller import SurveyDeformationController
from .controller_signals import ControllerSignals
from .app_state import AppState, is_feature_enabled, get_empty_state_message
from .job_registry import JobRegistry
from .job_worker import JobWorker

__all__ = [
    'AppController',
    'AppState',
    'SessionState',
    'PerformanceSettings',
    'GeostatsController',
    'MiningController',
    'VisController',
    'DataController',
    'SurveyDeformationController',
    'ControllerSignals',
    'is_feature_enabled',
    'get_empty_state_message',
    'JobRegistry',
    'JobWorker',
]
