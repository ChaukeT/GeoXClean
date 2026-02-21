"""
Controller Signals - Step 11: Unified Signal System

Centralized Qt signals for controller events. All UI components subscribe to these
signals for consistent, predictable updates.
"""

import logging

try:
    from PyQt6.QtCore import QObject, pyqtSignal
except ImportError:
    # Fallback for headless environments
    QObject = object  # type: ignore
    def pyqtSignal(*args, **kwargs):  # type: ignore
        return None

from .app_state import AppState

logger = logging.getLogger(__name__)


class ControllerSignals(QObject):
    """
    Centralized signals for controller events.
    
    All UI components should subscribe to these signals for updates.
    """
    
    # Application state - single source of truth for UI gating
    app_state_changed = pyqtSignal(int)  # AppState enum value
    
    # Scene and model updates
    scene_updated = pyqtSignal()  # Emitted when renderer scene changes
    block_model_changed = pyqtSignal()  # Emitted when block model data changes
    
    # Task lifecycle
    task_started = pyqtSignal(str)  # task_name
    task_finished = pyqtSignal(str)  # task_name
    task_error = pyqtSignal(str, str)  # task_name, error_message
    task_progress = pyqtSignal(str, int, str)  # task_name, percent, message
    
    # Structural features signals
    structural_features_loaded = pyqtSignal(object)  # StructuralFeatureCollection
    fault_added = pyqtSignal(object)  # FaultFeature
    fold_added = pyqtSignal(object)  # FoldFeature
    unconformity_added = pyqtSignal(object)  # UnconformityFeature
    structural_feature_removed = pyqtSignal(str)  # feature_id
    structural_features_cleared = pyqtSignal()  # All features cleared
    
    def __init__(self, parent=None):
        """Initialize controller signals."""
        super().__init__(parent)
        logger.debug("ControllerSignals initialized")

