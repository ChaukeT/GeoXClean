from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .compositing_engine import (
    CompositeConfig,
    CompositingMethod,
    WeightingMode,
    PartialStrategy,
    EconomicDilutionRule,
)


# =============================================================================
# CORE TYPES - SHARED ACROSS ALL UI ENGINES
# =============================================================================


class Severity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class UIMessage:
    severity: Severity
    field: str
    message: str


@dataclass
class UIValidationResult:
    ok: bool
    messages: List[UIMessage] = field(default_factory=list)

    @property
    def errors(self) -> List[UIMessage]:
        return [m for m in self.messages if m.severity == Severity.ERROR]

    @property
    def warnings(self) -> List[UIMessage]:
        return [m for m in self.messages if m.severity == Severity.WARNING]

    def add_error(self, field: str, msg: str) -> None:
        self.messages.append(UIMessage(Severity.ERROR, field, msg))

    def add_warning(self, field: str, msg: str) -> None:
        self.messages.append(UIMessage(Severity.WARNING, field, msg))

    def add_info(self, field: str, msg: str) -> None:
        self.messages.append(UIMessage(Severity.INFO, field, msg))

    @classmethod
    def ok_result(cls) -> "UIValidationResult":
        return cls(ok=True)

    @classmethod
    def fail(cls, messages: List[UIMessage]) -> "UIValidationResult":
        return cls(ok=False, messages=messages)


# =============================================================================
# NUMERICAL COMPOSITING UI ENGINE
# =============================================================================


class NumericalMode(str, Enum):
    FIXED_LENGTH = "FIXED_LENGTH"
    EQUAL_MASS = "EQUAL_MASS"
    EQUAL_VOLUME = "EQUAL_VOLUME"
    TRUE_THICKNESS = "TRUE_THICKNESS"
    BENCH_ALIGNED = "BENCH_ALIGNED"
    ROLLING_WINDOW = "ROLLING_WINDOW"


@dataclass
class NumericalUIState:
    mode: NumericalMode = NumericalMode.FIXED_LENGTH

    # Generic
    weighting_mode: WeightingMode = WeightingMode.LENGTH
    composite_length: Optional[float] = 1.0              # m or true thickness
    target_mass: Optional[float] = None                  # t per composite
    rolling_window_length: Optional[float] = None        # m
    rolling_step: Optional[float] = None                 # m

    bench_height: Optional[float] = None                 # m
    bench_offset: float = 0.0                            # m

    partial_strategy: PartialStrategy = PartialStrategy.AUTO
    auto_partial_fraction: float = 0.5                   # e.g., keep if ≥ 50%

    # QA/QC filters
    exclude_qaqc: bool = False
    qaqc_flags: Tuple[str, ...] = ("STANDARD", "BLANK", "DUPLICATE")
    min_recovery: Optional[float] = None                 # %
    
    # Missing grade handling
    treat_null_as_zero: bool = True  # If True, treat None/NaN grades as 0.0; if False, exclude from weighting

    # For logging / display
    description: str = ""


class NumericalUIEngine:
    """
    Backend for the 'Numerical Compositing' panel.
    """

    @staticmethod
    def default_state() -> NumericalUIState:
        return NumericalUIState()

    def validate(self, state: NumericalUIState) -> UIValidationResult:
        res = UIValidationResult.ok_result()

        # Common checks
        if state.weighting_mode in (WeightingMode.MASS, WeightingMode.DENSITY):
            # Front-end should warn that density is required
            res.add_warning(
                "weighting_mode",
                "Mass / density weighting requires density to be present in intervals.",
            )

        # Mode-specific checks
        if state.mode == NumericalMode.FIXED_LENGTH:
            if not state.composite_length or state.composite_length <= 0:
                res.add_error("composite_length", "Composite length must be > 0 m.")

        elif state.mode == NumericalMode.EQUAL_VOLUME:
            if not state.composite_length or state.composite_length <= 0:
                res.add_error("composite_length", "Equal-volume requires a reference composite length.")
            if state.weighting_mode == WeightingMode.MASS:
                res.add_warning(
                    "weighting_mode",
                    "Equal-volume with mass weighting will behave like length-weighted grades; "
                    "borehole area is assumed constant.",
                )

        elif state.mode == NumericalMode.EQUAL_MASS:
            if not state.target_mass or state.target_mass <= 0:
                res.add_error("target_mass", "Target mass must be > 0 t for equal-mass compositing.")
            if state.weighting_mode not in (WeightingMode.MASS, WeightingMode.DENSITY):
                res.add_error(
                    "weighting_mode",
                    "Equal-mass must use MASS or DENSITY weighting.",
                )

        elif state.mode == NumericalMode.TRUE_THICKNESS:
            if not state.composite_length or state.composite_length <= 0:
                res.add_error("composite_length", "True-thickness target must be > 0 m.")
            res.add_warning(
                "true_thickness",
                "True-thickness compositing requires true_thickness on intervals; missing values will raise.",
            )

        elif state.mode == NumericalMode.BENCH_ALIGNED:
            if not state.bench_height or state.bench_height <= 0:
                res.add_error("bench_height", "Bench height must be > 0 m.")
            if state.bench_height and abs(state.bench_offset) > state.bench_height:
                res.add_warning(
                    "bench_offset",
                    "Bench offset is larger than bench height; please check.",
                )

        elif state.mode == NumericalMode.ROLLING_WINDOW:
            if not state.rolling_window_length or state.rolling_window_length <= 0:
                res.add_error("rolling_window_length", "Window length must be > 0 m.")
            if not state.rolling_step or state.rolling_step <= 0:
                res.add_error("rolling_step", "Window step must be > 0 m.")
            if (
                state.rolling_window_length
                and state.rolling_step
                and state.rolling_step > state.rolling_window_length
            ):
                res.add_warning(
                    "rolling_step",
                    "Step is larger than window length; rolling curve will be very sparse.",
                )

        # Partial strategy sanity
        if state.partial_strategy == PartialStrategy.AUTO:
            if not (0.0 < state.auto_partial_fraction <= 1.0):
                res.add_error(
                    "auto_partial_fraction",
                    "Auto partial fraction must be in (0, 1].",
                )

        res.ok = len(res.errors) == 0
        return res

    def to_config(self, state: NumericalUIState) -> CompositeConfig:
        """
        Convert UI state into CompositeConfig used by the backend engine.
        Assumes you've validated already.
        """

        # Base config
        cfg = CompositeConfig(
            method=self._map_mode_to_method(state.mode),
            weighting_mode=state.weighting_mode,
            composite_length=state.composite_length,
            target_mass=state.target_mass,
            rolling_window_length=state.rolling_window_length,
            rolling_step=state.rolling_step,
            bench_height=state.bench_height,
            bench_offset=state.bench_offset,
            partial_strategy=state.partial_strategy,
            auto_partial_fraction=state.auto_partial_fraction,
            exclude_qaqc=state.exclude_qaqc,
            qaqc_flags=state.qaqc_flags,
            min_recovery=state.min_recovery,
            treat_null_as_zero=state.treat_null_as_zero,
        )

        return cfg

    @staticmethod
    def _map_mode_to_method(mode: NumericalMode) -> CompositingMethod:
        if mode == NumericalMode.FIXED_LENGTH:
            return CompositingMethod.FIXED_LENGTH
        if mode == NumericalMode.EQUAL_MASS:
            return CompositingMethod.EQUAL_MASS
        if mode == NumericalMode.EQUAL_VOLUME:
            return CompositingMethod.EQUAL_VOLUME
        if mode == NumericalMode.TRUE_THICKNESS:
            return CompositingMethod.TRUE_THICKNESS
        if mode == NumericalMode.BENCH_ALIGNED:
            return CompositingMethod.BENCH_ALIGNED
        if mode == NumericalMode.ROLLING_WINDOW:
            return CompositingMethod.ROLLING_WINDOW
        raise ValueError(f"Unsupported numerical mode: {mode}")


# =============================================================================
# LITHOLOGY COMPOSITING UI ENGINE
# =============================================================================


@dataclass
class LithologyUIState:
    weighting_mode: WeightingMode = WeightingMode.LENGTH

    # minimum thickness filters per lith
    min_ore_thickness: Optional[float] = None
    min_mining_width: Optional[float] = None

    # optional: restrict to certain lith codes
    include_lithologies: List[str] = field(default_factory=list)
    exclude_lithologies: List[str] = field(default_factory=list)

    # QA/QC
    exclude_qaqc: bool = False
    qaqc_flags: Tuple[str, ...] = ("STANDARD", "BLANK", "DUPLICATE")
    min_recovery: Optional[float] = None
    
    # Missing grade handling
    treat_null_as_zero: bool = True  # If True, treat None/NaN grades as 0.0; if False, exclude from weighting


class LithologyUIEngine:
    @staticmethod
    def default_state() -> LithologyUIState:
        return LithologyUIState()

    def validate(self, state: LithologyUIState) -> UIValidationResult:
        res = UIValidationResult.ok_result()

        if state.min_ore_thickness is not None and state.min_ore_thickness < 0:
            res.add_error("min_ore_thickness", "Minimum thickness cannot be negative.")

        if state.min_mining_width is not None and state.min_mining_width < 0:
            res.add_error("min_mining_width", "Minimum mining width cannot be negative.")

        overlap = set(state.include_lithologies) & set(state.exclude_lithologies)
        if overlap:
            res.add_warning(
                "include_exclude_lithologies",
                f"Some lithologies are in both include and exclude lists: {sorted(overlap)}.",
            )

        res.ok = len(res.errors) == 0
        return res

    def to_config(self, state: LithologyUIState) -> CompositeConfig:
        """Convert LithologyUIState to CompositeConfig.
        
        Note: min_ore_thickness and min_mining_width are now defined on CompositeConfig.
        """
        cfg = CompositeConfig(
            method=CompositingMethod.LITHOLOGY,
            weighting_mode=state.weighting_mode,
            min_ore_thickness=state.min_ore_thickness,
            min_mining_width=state.min_mining_width,
            exclude_qaqc=state.exclude_qaqc,
            qaqc_flags=state.qaqc_flags,
            min_recovery=state.min_recovery,
            treat_null_as_zero=state.treat_null_as_zero,
        )

        # Note: include/exclude lithology lists from state are not yet supported
        # in CompositeConfig. They would need to be handled at a higher level.
        return cfg


# =============================================================================
# ECONOMIC COMPOSITING UI ENGINE
# =============================================================================


@dataclass
class EconomicUIState:
    # economic parameters
    cutoff_field: str = "Fe"
    cutoff_grade: float = 55.0
    cutoff_operator: str = ">="  # for future flexibility

    weighting_mode: WeightingMode = WeightingMode.LENGTH

    # Economic compositing parameters (NOT "mining constraints")
    min_ore_composite_length: Optional[float] = None  # Minimum length for ore composite
    max_included_waste: Optional[float] = None  # Maximum total waste length in ore composite
    max_consecutive_waste: Optional[float] = None  # Maximum consecutive waste segment length
    min_linear_grade: Optional[float] = None  # Min linear grade to keep short high-grade composites
    min_waste_composite_length: Optional[float] = None  # Min waste composite length (Advanced/Advanced+)
    keep_short_high_grade: bool = False  # Allow short composites if min_linear_grade met
    dilution_rule: str = "basic"  # "basic", "advanced", "advanced_plus"
    composite_twice: bool = False  # Two-pass compositing
    use_true_thickness: bool = False  # Use true thickness for economic compositing
    true_thickness_dip: Optional[float] = None  # Dip for true thickness calculation
    true_thickness_dip_azimuth: Optional[float] = None  # Dip azimuth for true thickness

    # classification labels
    ore_label: str = "ORE"
    waste_label: str = "WASTE"

    # QA/QC filters and recovery
    exclude_qaqc: bool = False
    qaqc_flags: Tuple[str, ...] = ("STANDARD", "BLANK", "DUPLICATE")
    min_recovery: Optional[float] = None
    
    # Missing grade handling
    treat_null_as_zero: bool = True  # If True, treat None/NaN grades as 0.0; if False, exclude from weighting


class EconomicUIEngine:
    @staticmethod
    def default_state() -> EconomicUIState:
        return EconomicUIState()

    def validate(self, state: EconomicUIState) -> UIValidationResult:
        res = UIValidationResult.ok_result()

        if not state.cutoff_field:
            res.add_error("cutoff_field", "Cut-off grade field is required.")

        if state.cutoff_operator not in (">=", ">", "<=", "<"):
            res.add_error("cutoff_operator", "Unsupported operator. Use >=, >, <=, or <.")

        if state.min_ore_composite_length is not None and state.min_ore_composite_length < 0:
            res.add_error("min_ore_composite_length", "Minimum ore composite length cannot be negative.")

        if state.max_included_waste is not None and state.max_included_waste < 0:
            res.add_error("max_included_waste", "Maximum included waste cannot be negative.")

        if state.max_consecutive_waste is not None and state.max_consecutive_waste < 0:
            res.add_error("max_consecutive_waste", "Maximum consecutive waste cannot be negative.")

        if state.min_linear_grade is not None and state.min_linear_grade < 0:
            res.add_error("min_linear_grade", "Minimum linear grade cannot be negative.")

        if state.min_waste_composite_length is not None and state.min_waste_composite_length < 0:
            res.add_error("min_waste_composite_length", "Minimum waste composite length cannot be negative.")

        if state.dilution_rule not in ("basic", "advanced", "advanced_plus"):
            res.add_error("dilution_rule", "Dilution rule must be 'basic', 'advanced', or 'advanced_plus'.")

        if state.use_true_thickness:
            if state.true_thickness_dip is None:
                res.add_error("true_thickness_dip", "True thickness dip is required when using true thickness.")
            if state.true_thickness_dip_azimuth is None:
                res.add_error("true_thickness_dip_azimuth", "True thickness dip azimuth is required when using true thickness.")

        res.ok = len(res.errors) == 0
        return res

    def to_config(self, state: EconomicUIState) -> CompositeConfig:
        # Map dilution rule string to enum
        dilution_rule_map = {
            "basic": EconomicDilutionRule.BASIC,
            "advanced": EconomicDilutionRule.ADVANCED,
            "advanced_plus": EconomicDilutionRule.ADVANCED_PLUS,
        }
        dilution_rule = dilution_rule_map.get(state.dilution_rule, EconomicDilutionRule.BASIC)
        
        cfg = CompositeConfig(
            method=CompositingMethod.ECONOMIC,
            weighting_mode=state.weighting_mode,
            cutoff_field=state.cutoff_field,
            cutoff_grade=state.cutoff_grade,
            # Economic compositing parameters (NOT "mining constraints")
            min_ore_composite_length=state.min_ore_composite_length,
            max_included_waste=state.max_included_waste,
            max_consecutive_waste=state.max_consecutive_waste,
            min_linear_grade=state.min_linear_grade,
            min_waste_composite_length=state.min_waste_composite_length,
            keep_short_high_grade=state.keep_short_high_grade,
            dilution_rule=dilution_rule,
            composite_twice=state.composite_twice,
            use_true_thickness=state.use_true_thickness,
            true_thickness_dip=state.true_thickness_dip,
            true_thickness_dip_azimuth=state.true_thickness_dip_azimuth,
            exclude_qaqc=state.exclude_qaqc,
            qaqc_flags=state.qaqc_flags,
            min_recovery=state.min_recovery,
        )
        # Note: cutoff_operator, ore_label, waste_label can be stored in metadata
        # for use by higher-level code if needed
        return cfg


# =============================================================================
# WASTE/ORE CODING (INDICATOR COMPOSITING) UI ENGINE
# =============================================================================


@dataclass
class WasteOreUIState:
    indicator_field: str = "Fe"
    cutoff_grade: float = 55.0
    cutoff_operator: str = ">="

    ore_code: int = 1
    waste_code: int = 0

    weighting_mode: WeightingMode = WeightingMode.LENGTH
    composite_length: Optional[float] = 1.0

    # partial control
    partial_strategy: PartialStrategy = PartialStrategy.AUTO
    auto_partial_fraction: float = 0.5

    # QA/QC
    exclude_qaqc: bool = True
    qaqc_flags: Tuple[str, ...] = ("STANDARD", "BLANK", "DUPLICATE")
    min_recovery: Optional[float] = None
    
    # Missing grade handling
    treat_null_as_zero: bool = True  # If True, treat None/NaN grades as 0.0; if False, exclude from weighting


class WasteOreUIEngine:
    @staticmethod
    def default_state() -> WasteOreUIState:
        return WasteOreUIState()

    def validate(self, state: WasteOreUIState) -> UIValidationResult:
        res = UIValidationResult.ok_result()

        if not state.indicator_field:
            res.add_error("indicator_field", "Indicator field is required.")

        if state.cutoff_operator not in (">=", ">", "<=", "<"):
            res.add_error("cutoff_operator", "Unsupported operator. Use >=, >, <=, or <.")

        if state.composite_length is None or state.composite_length <= 0:
            res.add_error("composite_length", "Composite length must be > 0 m.")

        if state.partial_strategy == PartialStrategy.AUTO:
            if not (0.0 < state.auto_partial_fraction <= 1.0):
                res.add_error(
                    "auto_partial_fraction",
                    "Auto partial fraction must be in (0, 1].",
                )

        res.ok = len(res.errors) == 0
        return res

    def to_config(self, state: WasteOreUIState) -> CompositeConfig:
        cfg = CompositeConfig(
            method=CompositingMethod.INDICATOR,
            weighting_mode=state.weighting_mode,
            composite_length=state.composite_length,
            partial_strategy=state.partial_strategy,
            auto_partial_fraction=state.auto_partial_fraction,
            exclude_qaqc=state.exclude_qaqc,
            qaqc_flags=state.qaqc_flags,
            min_recovery=state.min_recovery,
            treat_null_as_zero=state.treat_null_as_zero,
        )
        cfg.cutoff_field = state.indicator_field
        cfg.cutoff_grade = state.cutoff_grade
        # Note: cutoff_operator, ore_code, waste_code can be stored in metadata
        # for use by higher-level code if needed
        return cfg

