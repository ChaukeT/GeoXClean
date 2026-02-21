"""
Application State - Single source of truth for GeoX application state.

This module defines the authoritative AppState enum that gates UI visibility
and enablement. All panels must react to AppState changes, not inspect data directly.
"""

from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class AppState(IntEnum):
    """
    Application state enum - single source of truth.
    
    UI visibility and enablement must be gated by this state.
    Panels must NOT infer state from data presence.
    
    States:
        EMPTY: No data loaded (startup state)
        DATA_LOADED: Data loaded but not rendered
        RENDERED: At least one layer rendered
        BUSY: Engine running (e.g., kriging, simulation)
    """
    EMPTY = 0           # No data loaded - startup state
    DATA_LOADED = 1     # Data loaded but not rendered
    RENDERED = 2        # At least one layer rendered
    BUSY = 3            # Engine running


# UI Visibility rules by state
# Keys: widget/feature names, Values: True = visible/enabled, False = hidden/disabled
STATE_UI_RULES = {
    AppState.EMPTY: {
        # Visible
        "menu_bar": True,
        "scene_viewport": True,
        "status_bar": True,
        "footer_indicators": True,
        "file_import_menu": True,
        # Hidden or Disabled
        "legend": False,
        "property_controls": False,
        "active_layer_dropdown": False,
        "update_plot_buttons": False,
        "overlays": False,
        "drillhole_controls": False,
        "selection_tools": False,
        "quick_toggle": False,
    },
    AppState.DATA_LOADED: {
        # Visible
        "menu_bar": True,
        "scene_viewport": True,
        "status_bar": True,
        "footer_indicators": True,
        "file_import_menu": True,
        "file_info": True,
        "drillhole_controls": True,
        "update_plot_buttons": True,
        # Hidden
        "legend": False,
        "property_controls": False,
        "selection_tools": False,
        "overlays": False,
    },
    AppState.RENDERED: {
        # All visible
        "menu_bar": True,
        "scene_viewport": True,
        "status_bar": True,
        "footer_indicators": True,
        "file_import_menu": True,
        "file_info": True,
        "drillhole_controls": True,
        "update_plot_buttons": True,
        "legend": True,
        "property_controls": True,
        "active_layer_dropdown": True,
        "selection_tools": True,
        "overlays": True,
        "quick_toggle": True,
    },
    AppState.BUSY: {
        # Same as RENDERED but some controls disabled
        "menu_bar": True,
        "scene_viewport": True,
        "status_bar": True,
        "footer_indicators": True,
        "legend": True,
        "property_controls": False,  # Disabled during processing
        "active_layer_dropdown": False,
        "update_plot_buttons": False,  # Disabled during processing
        "drillhole_controls": False,  # Disabled during processing
        "selection_tools": False,
    },
}


def is_feature_enabled(state: AppState, feature: str) -> bool:
    """
    Check if a UI feature should be enabled/visible for the given state.
    
    Args:
        state: Current AppState
        feature: Feature name to check
        
    Returns:
        True if feature should be visible/enabled, False otherwise.
        Defaults to False if feature not found (safe default: hide unknown).
    """
    rules = STATE_UI_RULES.get(state, {})
    return rules.get(feature, False)


def get_empty_state_message(feature: str) -> str:
    """
    Get a user-friendly message for disabled features in EMPTY state.
    
    Args:
        feature: Feature name
        
    Returns:
        Appropriate message for the feature
    """
    messages = {
        "property_controls": "Available after data is loaded",
        "active_layer_dropdown": "Available after data is loaded",
        "legend": "Available after visualization",
        "drillhole_controls": "Load drillholes or a block model to enable controls",
        "selection_tools": "Available after visualization",
        "quick_toggle": "Available after data is loaded",
    }
    return messages.get(feature, "Available after data is loaded")

