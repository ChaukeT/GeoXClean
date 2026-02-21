"""
UI Signals Hub - Centralized Qt signals for panel-to-controller communication.

All panels should emit these signals instead of directly accessing the renderer.
"""

from PyQt6.QtCore import QObject, pyqtSignal


class UISignals(QObject):
    """
    Central signal hub for UI panel communication.
    
    Panels emit these signals; MainWindow connects them to AppController methods.
    This eliminates the need for MainWindow to manually connect hundreds of panel signals.
    """
    
    # =========================================================================
    # Property Panel Signals
    # =========================================================================
    propertyChanged = pyqtSignal(str)  # property_name
    colormapChanged = pyqtSignal(str)  # colormap_name
    colorModeChanged = pyqtSignal(str)  # color_mode
    filterChanged = pyqtSignal(str, float, float)  # property_name, min_value, max_value
    sliceChanged = pyqtSignal(str, float)  # axis, position
    transparencyChanged = pyqtSignal(float)  # alpha 0.0-1.0
    blockSizeChanged = pyqtSignal(float, float, float)  # dx, dy, dz
    legendSettingsChanged = pyqtSignal(str, int)  # property_name, decimals
    legendStyleChanged = pyqtSignal(dict)  # legend style dict
    axisFontChanged = pyqtSignal(str, int)  # font_name, font_size
    axisColorChanged = pyqtSignal(tuple)  # (r, g, b) tuple
    opacityChanged = pyqtSignal(float)  # opacity 0.0-1.0
    
    # =========================================================================
    # Scene Inspector Panel Signals
    # =========================================================================
    resetViewRequested = pyqtSignal()  # Reset camera view
    fitViewRequested = pyqtSignal()  # Fit model to view
    viewPresetRequested = pyqtSignal(str)  # preset_name
    projectionToggled = pyqtSignal(bool)  # True = orthographic, False = perspective
    scalarBarToggled = pyqtSignal(bool)  # show/hide scalar bar
    axesToggled = pyqtSignal(bool)  # show/hide axes
    gridToggled = pyqtSignal(bool)  # show/hide grid
    trackballModeToggled = pyqtSignal(bool)  # enable/disable trackball mode
    
    # =========================================================================
    # Block Info Panel Signals
    # =========================================================================
    clearSelectionRequested = pyqtSignal()  # Clear block selection
    
    # =========================================================================
    # Viewer Widget Signals
    # =========================================================================
    blockPicked = pyqtSignal(int, dict)  # block_index, properties
    mouseModeChanged = pyqtSignal(str)  # mouse_mode_name
    
    # =========================================================================
    # Legend Widget Signals
    # =========================================================================
    legendColormapChanged = pyqtSignal(str)  # colormap_name
    
    # =========================================================================
    # Drillhole Control Panel Signals
    # =========================================================================
    drillholePlotRequested = pyqtSignal(str)  # dataset_name
    drillholeClearRequested = pyqtSignal()
    drillholeRadiusChanged = pyqtSignal(float)  # radius
    drillholeColorModeChanged = pyqtSignal(str)  # color_mode
    drillholeAssayFieldChanged = pyqtSignal(str)  # assay_field
    drillholeShowIdsToggled = pyqtSignal(bool)  # show_ids
    drillholeVisibilityChanged = pyqtSignal(str, bool)  # hole_id, visible
    drillholeFocusRequested = pyqtSignal()  # Focus on selected holes
    drillholeLithFilterChanged = pyqtSignal(list)  # List of lithology codes to show (empty = show all)
    
    # =========================================================================
    # =========================================================================
    # Geological Model Signals
    # =========================================================================
    geologicalModelUpdated = pyqtSignal(object)  # surfaces/solids result dict
    geologicalSurfacesLoaded = pyqtSignal(object)  # implicit surfaces result
    geologicalSolidsLoaded = pyqtSignal(object)  # voxel solids result

    # =========================================================================
    # Geological Explorer Panel Signals
    # =========================================================================
    geologyRenderModeChanged = pyqtSignal(str)       # "surfaces", "solids", "both", "unified"
    geologyContactsVisibilityChanged = pyqtSignal(bool)  # show/hide contact points
    geologySurfacesVisibilityChanged = pyqtSignal(bool)  # show/hide surface meshes
    geologyMisfitVisibilityChanged = pyqtSignal(bool)    # show/hide misfit glyphs
    geologyFormationFilterChanged = pyqtSignal(list)     # list of formation names to show
    geologyOpacityChanged = pyqtSignal(float)            # opacity 0.0-1.0
    geologyColorPaletteChanged = pyqtSignal(str)         # colormap name
    geologyResetViewRequested = pyqtSignal()             # reset camera to fit geology
    geologyClearRequested = pyqtSignal()                 # clear all geology from view

    # Individual Layer Control Signals
    geologyLayerVisibilityChanged = pyqtSignal(str, bool)  # layer_name, visible
    geologyWireframeToggled = pyqtSignal(bool)             # show wireframe on solids
    geologySolidsOpacityChanged = pyqtSignal(float)        # solids-specific opacity
    geologyViewModeChanged = pyqtSignal(str)               # "surfaces_only", "solids_only", etc.

    # =========================================================================
    # Legacy/Compatibility Signals (for backward compatibility)
    # =========================================================================
    propertySelected = pyqtSignal(str)  # property_name (alias for propertyChanged)
    applyFilters = pyqtSignal(dict)  # {property_name: (min, max)} (legacy)
    
    # Export
    exportScreenshot = pyqtSignal(str)  # file_path
    
    # Data loading
    blockModelLoaded = pyqtSignal(object)  # block_model
    
    # Legend control
    legendToggled = pyqtSignal(bool)  # visible

    # =========================================================================
    # Cross-Section Panel Signals
    # =========================================================================
    crossSectionRequested = pyqtSignal(dict)  # cross_section_params
    crossSectionPropertyChanged = pyqtSignal(str)  # property_name
    crossSectionUpdated = pyqtSignal()  # section parameters changed
    crossSectionExported = pyqtSignal(str, str)  # file_path, format

    # =========================================================================
    # Cross-Section Manager Panel Signals
    # =========================================================================
    crossSectionManagerRenderRequested = pyqtSignal(str, str)  # section_name, property_name
    crossSectionManagerExportRequested = pyqtSignal(str, str, str)  # section_name, property_name, file_path
    crossSectionManagerSectionCreated = pyqtSignal(str, dict)  # section_name, section_spec
    crossSectionManagerSectionDeleted = pyqtSignal(str)  # section_name

