"""
Drillhole Package - Data model, I/O, compositing, database management,
reporting, and plotting for drillhole databases.
"""

from .datamodel import (
    Collar,
    SurveyInterval,
    Survey,  # Alias for backward compatibility
    AssayInterval,
    LithologyInterval,
    DrillholeDatabase,
)

from .data_io import load_from_csv
from .compositing_engine import (
    CompositingMethodEngine,
    CompositingMethod,
    BreakMode,
    WeightingMode,
    PartialStrategy,
    Interval,
    Composite,
    CompositeConfig,
)
from .compositing_ui_engines import (
    Severity,
    UIMessage,
    UIValidationResult,
    NumericalMode,
    NumericalUIState,
    NumericalUIEngine,
    LithologyUIState,
    LithologyUIEngine,
    EconomicUIState,
    EconomicUIEngine,
    WasteOreUIState,
    WasteOreUIEngine,
)
from .compositing_utils import (
    dataframes_to_intervals,
    dataframes_to_intervals_simple,
    get_intervals_from_registry,
    get_intervals_with_audit,
    IntervalConversionResult,
)
from .database import DrillholeDatabaseManager
from .reporting import DrillholeStatistics, ReportGenerator
from .plotting import DownholePlotter, StripLogPlotter, FenceDiagramPlotter

# Backward-compatibility re-exports — security.py and user_auth.py were moved
# to core/ (L-05 refactor). Import from core.security / core.user_auth directly.
from ..core.user_auth import (  # noqa: F401
    Permission,
    Role,
    ROLE_PERMISSIONS,
    User,
    UserManager,
    get_user_manager,
    get_current_user,
    require_permission,
)
from ..core.security import (  # noqa: F401
    AccessType,
    AccessLog,
    SecurityManager,
    get_security_manager,
)

__all__ = [
    # Data model
    "Collar",
    "SurveyInterval",
    "Survey",  # Alias for backward compatibility
    "AssayInterval",
    "LithologyInterval",
    "DrillholeDatabase",
    # I/O
    "load_from_csv",
    # Compositing
    "CompositingMethodEngine",
    "CompositingMethod",
    "BreakMode",
    "WeightingMode",
    "PartialStrategy",
    "Interval",
    "Composite",
    "CompositeConfig",
    # Compositing UI Engines
    "Severity",
    "UIMessage",
    "UIValidationResult",
    "NumericalMode",
    "NumericalUIState",
    "NumericalUIEngine",
    "LithologyUIState",
    "LithologyUIEngine",
    "EconomicUIState",
    "EconomicUIEngine",
    "WasteOreUIState",
    "WasteOreUIEngine",
    # Compositing Utilities
    "dataframes_to_intervals",
    "dataframes_to_intervals_simple",
    "get_intervals_from_registry",
    "get_intervals_with_audit",
    "IntervalConversionResult",
    # Database management
    "DrillholeDatabaseManager",
    # Reporting
    "DrillholeStatistics",
    "ReportGenerator",
    # Plotting
    "DownholePlotter",
    "StripLogPlotter",
    "FenceDiagramPlotter",
]

