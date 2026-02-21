"""
Core module for centralized data management and communication.
"""

from .data_registry import DataRegistry
from .data_registry_simple import DataRegistrySimple, DataMetadata
from .audit_manager import AuditManager
from .crash_handler import install_exception_handler
from .data_provenance import (
    DataProvenance,
    DataSourceType,
    TransformationStep,
    create_raw_data_provenance,
    create_composited_provenance,
    create_declustered_provenance,
    create_estimation_provenance,
    format_lineage_for_display,
    get_available_data_sources,
)
from .process_history_tracker import ProcessHistoryTracker, get_process_history_tracker
from .errors import (
    GeoXError,
    DataError,
    GeologyError,
    VisualizationError,
    GeostatsError,
    ValidationError,
    ErrorSeverity,
    ErrorCategory,
    error_boundary,
    safe_operation,
    set_error_dialog_handler,
    show_error_dialog,
    get_error_description,
    ERROR_CODES,
)
from .state_manager import (
    UnifiedStateManager,
    LegendState,
    CameraState,
    LayerState,
    DrillholeState,
    GeologyState,
    get_state_manager,
    reset_state_manager,
)
from .worker import (
    WorkerSignals,
    BaseWorker,
    GeologyWorker,
    DataWorker,
    RenderWorker,
    get_thread_pool,
    submit_worker,
)
from .thread_safe_cache import (
    ThreadSafeCache,
    DrillholeCache,
    GeometryCache,
    get_drillhole_cache,
    get_geometry_cache,
    clear_all_caches,
)

__all__ = [
    "DataRegistry",
    "DataRegistrySimple",
    "DataMetadata",
    "AuditManager",
    "install_exception_handler",
    # Data Provenance
    "DataProvenance",
    "DataSourceType",
    "TransformationStep",
    "create_raw_data_provenance",
    "create_composited_provenance",
    "create_declustered_provenance",
    "create_estimation_provenance",
    "format_lineage_for_display",
    "get_available_data_sources",
    # Process History
    "ProcessHistoryTracker",
    "get_process_history_tracker",
    # Error Handling
    "GeoXError",
    "DataError",
    "GeologyError",
    "VisualizationError",
    "GeostatsError",
    "ValidationError",
    "ErrorSeverity",
    "ErrorCategory",
    "error_boundary",
    "safe_operation",
    "set_error_dialog_handler",
    "show_error_dialog",
    "get_error_description",
    "ERROR_CODES",
    # State Management
    "UnifiedStateManager",
    "LegendState",
    "CameraState",
    "LayerState",
    "DrillholeState",
    "GeologyState",
    "get_state_manager",
    "reset_state_manager",
    # Workers
    "WorkerSignals",
    "BaseWorker",
    "GeologyWorker",
    "DataWorker",
    "RenderWorker",
    "get_thread_pool",
    "submit_worker",
    # Thread-Safe Caches
    "ThreadSafeCache",
    "DrillholeCache",
    "GeometryCache",
    "get_drillhole_cache",
    "get_geometry_cache",
    "clear_all_caches",
]

