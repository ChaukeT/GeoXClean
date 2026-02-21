"""
Short-Term Scheduling (STEP 30)

Block-model-driven weekly/daily scheduling, blending, and shift planning.

Pipeline:
    1. Import block model (CSV / DataFrame)
    2. Filter by resource classification & material type
    3. Group into scheduling units (bench, pit, domain, etc.)
    4. Allocate units to periods with blend compliance
    5. Generate shift-by-shift operating plan
"""

from .block_model_scheduler import (
    # Enums
    ResourceClass,
    GroupByField,
    PeriodType,
    # Config
    GradeSpec,
    ColumnMapping,
    ShortTermScheduleConfig,
    # Results
    SchedulingUnit,
    ScheduleEntry,
    PeriodSummary,
    ShortTermScheduleResult,
    # Functions
    filter_blocks,
    group_blocks,
    build_short_term_schedule,
)

from .short_term_blend import (
    # Config
    BlendSource,
    BlendSpec,
    ShortTermBlendConfig,
    # Results
    BlendAllocation,
    BlendResult,
    # Functions
    optimise_short_term_blend,
    build_blend_sources_from_units,
)

from .shift_plan import (
    # Config
    ShiftConfig,
    EquipmentUnit,
    # Results
    ShiftAssignment,
    ShiftPlanEntry,
    ShiftPlanResult,
    # Functions
    generate_shift_plan,
)

__all__ = [
    # ── Block Model Scheduler ─────────────────────────────────────────────
    "ResourceClass",
    "GroupByField",
    "PeriodType",
    "GradeSpec",
    "ColumnMapping",
    "ShortTermScheduleConfig",
    "SchedulingUnit",
    "ScheduleEntry",
    "PeriodSummary",
    "ShortTermScheduleResult",
    "filter_blocks",
    "group_blocks",
    "build_short_term_schedule",
    # ── Blend Optimiser ───────────────────────────────────────────────────
    "BlendSource",
    "BlendSpec",
    "ShortTermBlendConfig",
    "BlendAllocation",
    "BlendResult",
    "optimise_short_term_blend",
    "build_blend_sources_from_units",
    # ── Shift Plan ────────────────────────────────────────────────────────
    "ShiftConfig",
    "EquipmentUnit",
    "ShiftAssignment",
    "ShiftPlanEntry",
    "ShiftPlanResult",
    "generate_shift_plan",
]
