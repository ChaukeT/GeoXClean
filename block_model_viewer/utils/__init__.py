"""
Utility modules for Block Model Viewer.

Provides shared functionality:
- Coordinate management
- Variogram functions
- Export helpers
- Plotting helpers
- Data bridge (inter-panel communication)
"""

from .coordinate_manager import CoordinateManager, DatasetInfo, CoordinateBounds
from .coordinate_utils import ensure_xyz_columns, detect_coordinate_columns
from .variogram_functions import (
    spherical_variogram,
    exponential_variogram,
    gaussian_variogram,
    linear_variogram,
    power_variogram,
    get_variogram_function,
    fit_variogram,
    calculate_experimental_variogram,
    anisotropic_distance
)
from .export_helpers import (
    export_dataframe_to_csv,
    export_dataframe_to_excel,
    export_to_vtk,
    format_number,
    create_summary_dict,
    batch_export
)
from .plotting_helpers import (
    create_figure,
    plot_histogram,
    plot_scatter,
    plot_boxplot,
    plot_cumulative,
    plot_grade_tonnage,
    apply_theme,
    save_figure
)
from .data_bridge import (
    DataBridge,
    DataType,
    DataPackage,
    get_data_bridge,
    publish_schedule,
    get_schedule,
    get_stopes,
    clear_all_data
)
from .variable_utils import (
    get_numeric_columns,
    get_grade_columns,
    validate_variable,
    auto_select_variable,
    populate_variable_combo,
    get_variable_from_combo_or_fallback,
    ensure_required_columns,
    VariableValidationResult,
    COORDINATE_COLUMNS
)

__all__ = [
    # Coordinate management
    'CoordinateManager',
    'DatasetInfo',
    'CoordinateBounds',
    'ensure_xyz_columns',
    'detect_coordinate_columns',
    
    # Variogram functions
    'spherical_variogram',
    'exponential_variogram',
    'gaussian_variogram',
    'linear_variogram',
    'power_variogram',
    'get_variogram_function',
    'fit_variogram',
    'calculate_experimental_variogram',
    'anisotropic_distance',
    
    # Export helpers
    'export_dataframe_to_csv',
    'export_dataframe_to_excel',
    'export_to_vtk',
    'format_number',
    'create_summary_dict',
    'batch_export',
    
    # Plotting helpers
    'create_figure',
    'plot_histogram',
    'plot_scatter',
    'plot_boxplot',
    'plot_cumulative',
    'plot_grade_tonnage',
    'apply_theme',
    'save_figure',
    
    # Data bridge
    'DataBridge',
    'DataType',
    'DataPackage',
    'get_data_bridge',
    'publish_schedule',
    'get_schedule',
    'get_stopes',
    'clear_all_data',
    
    # Variable utilities
    'get_numeric_columns',
    'get_grade_columns',
    'validate_variable',
    'auto_select_variable',
    'populate_variable_combo',
    'get_variable_from_combo_or_fallback',
    'ensure_required_columns',
    'VariableValidationResult',
    'COORDINATE_COLUMNS'
]

