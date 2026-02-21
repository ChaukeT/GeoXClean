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

