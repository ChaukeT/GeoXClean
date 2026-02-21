"""
Short-Term Block Model Scheduler (STEP 30)

Block-model-driven weekly/daily scheduling with:
- Resource classification filtering (Measured/Indicated/Inferred)
- Configurable block grouping (bench, pit, domain, material)
- Tonnage-weighted grade blending with spec compliance
- Fleet capacity constraints (optional)
- Multi-destination routing (plant, stockpile, waste dump)

Author: BlockModelViewer Team
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from enum import Enum

logger = logging.getLogger(__name__)


# ─── Enums ────────────────────────────────────────────────────────────────────

class ResourceClass(Enum):
    """JORC / NI 43-101 resource classification."""
    MEASURED = "Measured"
    INDICATED = "Indicated"
    INFERRED = "Inferred"
    UNCLASSIFIED = "Unclassified"

    @classmethod
    def schedulable_default(cls) -> List["ResourceClass"]:
        """Default classes used in short-term scheduling."""
        return [cls.MEASURED, cls.INDICATED]


class GroupByField(Enum):
    """Available fields to group blocks into scheduling units."""
    BENCH = "bench"
    PIT = "pit"
    DOMAIN = "domain"
    MATERIAL = "material"
    RESOURCE_CLASS = "resource_class"
    CUSTOM = "custom"


class PeriodType(Enum):
    """Scheduling period granularity."""
    DAILY_7 = "daily_7"
    DAILY_5 = "daily_5"
    SHIFT_14 = "shift_14"
    WEEKLY_4 = "weekly_4"

    @property
    def count(self) -> int:
        return {
            self.DAILY_7: 7, self.DAILY_5: 5,
            self.SHIFT_14: 14, self.WEEKLY_4: 4,
        }[self]

    @property
    def labels(self) -> List[str]:
        mapping = {
            self.DAILY_7: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            self.DAILY_5: ["Mon", "Tue", "Wed", "Thu", "Fri"],
            self.SHIFT_14: [f"S{i+1}" for i in range(14)],
            self.WEEKLY_4: [f"Wk{i+1}" for i in range(4)],
        }
        return mapping[self]


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GradeSpec:
    """Grade specification for a single element."""
    element: str
    min_grade: float
    max_grade: float
    unit: str = "%"
    is_penalty: bool = False  # True for deleterious elements

    @property
    def mid(self) -> float:
        return (self.min_grade + self.max_grade) / 2.0

    def in_spec(self, value: float) -> bool:
        return self.min_grade <= value <= self.max_grade


@dataclass
class ColumnMapping:
    """
    Maps block model DataFrame columns to scheduler fields.

    All fields are column name strings referencing the source DataFrame.
    Only `tonnes` is strictly required; others are used when available.
    """
    block_id: str = "BLOCK_ID"
    x: str = "XC"
    y: str = "YC"
    z: str = "ZC"
    pit: str = "PIT"
    bench: str = "RL"
    domain: str = "DOMAIN"
    resource_class: str = "RESOURCE_CLASS"
    material: str = "MATERIAL"
    tonnes: str = "TONNES"
    density: str = "DENSITY"
    # Grade columns are specified separately via grade_specs


@dataclass
class ShortTermScheduleConfig:
    """
    Full configuration for the short-term block model scheduler.

    Attributes:
        column_mapping:       Maps DataFrame columns → scheduler fields.
        grade_specs:          Grade constraints per element.
        allowed_classes:      Resource classes to include.
        allowed_materials:    Material types to include (empty = all).
        min_block_tonnes:     Minimum block tonnage filter.
        group_by:             How to aggregate blocks into scheduling units.
        custom_group_field:   Column name when group_by == CUSTOM.
        plant_target_per_period: Ore target per scheduling period (tonnes).
        period_type:          Scheduling period granularity.
        destinations:         Material routing destinations.
        primary_destination:  Destination for ore (index into destinations).
        waste_destination:    Destination for waste (index into destinations).
    """
    column_mapping: ColumnMapping = field(default_factory=ColumnMapping)
    grade_specs: List[GradeSpec] = field(default_factory=list)
    allowed_classes: List[ResourceClass] = field(
        default_factory=ResourceClass.schedulable_default
    )
    allowed_materials: List[str] = field(
        default_factory=lambda: ["Ore"]
    )
    min_block_tonnes: float = 0.0
    group_by: GroupByField = GroupByField.BENCH
    custom_group_field: Optional[str] = None
    plant_target_per_period: float = 25_000.0
    period_type: PeriodType = PeriodType.DAILY_7
    destinations: List[str] = field(
        default_factory=lambda: ["Plant/Mill", "Waste Dump"]
    )
    primary_destination: int = 0
    waste_destination: int = 1


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class SchedulingUnit:
    """
    A group of blocks treated as a single scheduling entity.

    Created by aggregating filtered blocks based on the chosen grouping.
    """
    id: str
    name: str
    material: str
    block_count: int
    tonnes: float
    grades: Dict[str, float]           # element → tonnage-weighted avg grade
    block_ids: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduleEntry:
    """A single allocation: unit → period → destination."""
    period_index: int
    period_label: str
    unit_id: str
    unit_name: str
    material: str
    tonnes: float
    destination: str
    grades: Dict[str, float] = field(default_factory=dict)
    block_count: int = 0


@dataclass
class PeriodSummary:
    """Aggregated summary for one scheduling period."""
    index: int
    label: str
    total_ore_tonnes: float
    total_waste_tonnes: float
    blended_grades: Dict[str, float]
    entries: List[ScheduleEntry]
    grade_compliance: bool = True

    @property
    def total_tonnes(self) -> float:
        return self.total_ore_tonnes + self.total_waste_tonnes


@dataclass
class ShortTermScheduleResult:
    """Complete result from the short-term scheduler."""
    periods: List[PeriodSummary]
    units: List[SchedulingUnit]
    config: ShortTermScheduleConfig
    total_blocks_loaded: int = 0
    total_blocks_filtered: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience properties ────────────────────────────────────────────

    @property
    def total_ore(self) -> float:
        return sum(p.total_ore_tonnes for p in self.periods)

    @property
    def total_waste(self) -> float:
        return sum(p.total_waste_tonnes for p in self.periods)

    @property
    def avg_blended_grades(self) -> Dict[str, float]:
        if not self.periods:
            return {}
        grades = {}
        for el in self.periods[0].blended_grades:
            grades[el] = np.mean(
                [p.blended_grades.get(el, 0.0) for p in self.periods]
            )
        return grades

    @property
    def compliance_rate(self) -> float:
        if not self.periods:
            return 0.0
        return sum(1 for p in self.periods if p.grade_compliance) / len(self.periods)

    def get_entries_for_period(self, period_index: int) -> List[ScheduleEntry]:
        if 0 <= period_index < len(self.periods):
            return self.periods[period_index].entries
        return []

    def to_dataframe(self) -> pd.DataFrame:
        """Export schedule as a flat DataFrame."""
        rows = []
        for period in self.periods:
            for entry in period.entries:
                row = {
                    "period_index": entry.period_index,
                    "period_label": entry.period_label,
                    "unit_id": entry.unit_id,
                    "unit_name": entry.unit_name,
                    "material": entry.material,
                    "tonnes": entry.tonnes,
                    "destination": entry.destination,
                    "block_count": entry.block_count,
                }
                for el, grade in entry.grades.items():
                    row[f"grade_{el}"] = grade
                rows.append(row)
        return pd.DataFrame(rows)


# ─── Block Filtering ──────────────────────────────────────────────────────────

def filter_blocks(
    block_model: pd.DataFrame,
    config: ShortTermScheduleConfig,
) -> pd.DataFrame:
    """
    Filter block model by resource classification, material type, and
    minimum tonnage.

    Args:
        block_model: Raw block model DataFrame.
        config:      Scheduler configuration.

    Returns:
        Filtered DataFrame (copy).
    """
    cm = config.column_mapping
    df = block_model.copy()
    initial_count = len(df)

    # Resource classification filter
    if cm.resource_class in df.columns and config.allowed_classes:
        allowed = {c.value for c in config.allowed_classes}
        df = df[df[cm.resource_class].astype(str).isin(allowed)]
        logger.info(
            f"Resource classification filter: {initial_count} → {len(df)} "
            f"(allowed: {', '.join(allowed)})"
        )

    # Material type filter
    if cm.material in df.columns and config.allowed_materials:
        allowed_mat = {m.lower() for m in config.allowed_materials}
        df = df[df[cm.material].astype(str).str.lower().isin(allowed_mat)]
        logger.info(f"Material filter: → {len(df)} blocks")

    # Minimum tonnage filter
    if cm.tonnes in df.columns and config.min_block_tonnes > 0:
        df = df[df[cm.tonnes] >= config.min_block_tonnes]
        logger.info(f"Min tonnage filter ({config.min_block_tonnes}): → {len(df)} blocks")

    logger.info(
        f"Filtering complete: {initial_count} → {len(df)} blocks retained "
        f"({len(df) / max(initial_count, 1) * 100:.1f}%)"
    )
    return df


# ─── Block Grouping ──────────────────────────────────────────────────────────

def group_blocks(
    filtered_df: pd.DataFrame,
    config: ShortTermScheduleConfig,
) -> List[SchedulingUnit]:
    """
    Aggregate filtered blocks into scheduling units based on the chosen
    grouping field.  Grades are computed as tonnage-weighted averages.

    Args:
        filtered_df: Filtered block model DataFrame.
        config:      Scheduler configuration.

    Returns:
        List of SchedulingUnit.
    """
    cm = config.column_mapping

    # Determine grouping column
    group_col_map = {
        GroupByField.BENCH: cm.bench if cm.bench in filtered_df.columns else cm.z,
        GroupByField.PIT: cm.pit,
        GroupByField.DOMAIN: cm.domain,
        GroupByField.MATERIAL: cm.material,
        GroupByField.RESOURCE_CLASS: cm.resource_class,
        GroupByField.CUSTOM: config.custom_group_field,
    }
    group_col = group_col_map.get(config.group_by)

    if group_col is None or group_col not in filtered_df.columns:
        logger.warning(
            f"Grouping column '{group_col}' not found. "
            f"Falling back to single scheduling unit."
        )
        group_col = None

    # Grade element columns
    grade_cols = [gs.element for gs in config.grade_specs if gs.element in filtered_df.columns]

    units: List[SchedulingUnit] = []

    if group_col is None:
        # Single unit containing all blocks
        groups = [("All Blocks", filtered_df)]
    else:
        groups = list(filtered_df.groupby(group_col, sort=True))

    for i, (key, group_df) in enumerate(groups):
        tonnes_col = cm.tonnes if cm.tonnes in group_df.columns else None
        total_tonnes = group_df[tonnes_col].sum() if tonnes_col else float(len(group_df))

        # Tonnage-weighted average grades
        grades: Dict[str, float] = {}
        for el in grade_cols:
            if total_tonnes > 0:
                grades[el] = (
                    (group_df[el] * group_df[tonnes_col]).sum() / total_tonnes
                    if tonnes_col
                    else group_df[el].mean()
                )
            else:
                grades[el] = 0.0

        # Determine dominant material
        if cm.material in group_df.columns:
            mat_counts = group_df.groupby(cm.material)[tonnes_col or cm.material].sum() if tonnes_col else group_df[cm.material].value_counts()
            material = mat_counts.idxmax() if len(mat_counts) > 0 else "Ore"
        else:
            material = "Ore"

        # Determine unit name
        if config.group_by == GroupByField.BENCH:
            name = f"RL {key}"
        elif config.group_by == GroupByField.PIT:
            name = str(key)
        else:
            name = str(key)

        block_ids = (
            group_df[cm.block_id].tolist()
            if cm.block_id in group_df.columns
            else group_df.index.tolist()
        )

        units.append(SchedulingUnit(
            id=f"SU-{i+1:03d}",
            name=name,
            material=material,
            block_count=len(group_df),
            tonnes=float(total_tonnes),
            grades=grades,
            block_ids=block_ids,
            metadata={
                "group_key": str(key),
                "group_by": config.group_by.value,
            },
        ))

    logger.info(
        f"Grouped {len(filtered_df)} blocks into {len(units)} scheduling units "
        f"({config.group_by.value})"
    )
    return units


# ─── Blend Feasibility Check ─────────────────────────────────────────────────

def check_blend_feasibility(
    current_blend: Dict[str, float],
    current_tonnes: float,
    candidate_grades: Dict[str, float],
    candidate_tonnes: float,
    grade_specs: List[GradeSpec],
    leniency_fraction: float = 0.3,
    plant_target: float = 25_000.0,
) -> bool:
    """
    Check whether adding a candidate source keeps the blend within spec.

    Early in the period (below leniency_fraction of target), constraints
    are relaxed to avoid deadlocking the schedule.

    Args:
        current_blend:      Current tonnage-weighted blend grades.
        current_tonnes:     Tonnes already allocated this period.
        candidate_grades:   Grades of the candidate unit.
        candidate_tonnes:   Tonnes to add.
        grade_specs:        Grade specifications.
        leniency_fraction:  Fraction of target below which specs are relaxed.
        plant_target:       Plant target for the period.

    Returns:
        True if feasible.
    """
    new_total = current_tonnes + candidate_tonnes
    if new_total <= 0:
        return True

    for gs in grade_specs:
        el = gs.element
        current_grade = current_blend.get(el, 0.0)
        candidate_grade = candidate_grades.get(el, 0.0)

        blended = (
            (current_grade * current_tonnes + candidate_grade * candidate_tonnes)
            / new_total
        )

        # Strict check only after leniency threshold
        if current_tonnes >= plant_target * leniency_fraction:
            if blended < gs.min_grade or blended > gs.max_grade:
                return False

    return True


def compute_blended_grades(
    entries: List[ScheduleEntry],
    elements: List[str],
) -> Dict[str, float]:
    """Compute tonnage-weighted average grades from schedule entries."""
    total_tonnes = sum(e.tonnes for e in entries)
    if total_tonnes <= 0:
        return {el: 0.0 for el in elements}
    blend = {}
    for el in elements:
        blend[el] = sum(e.grades.get(el, 0.0) * e.tonnes for e in entries) / total_tonnes
    return blend


# ─── Main Scheduler ──────────────────────────────────────────────────────────

def build_short_term_schedule(
    block_model: pd.DataFrame,
    config: ShortTermScheduleConfig,
) -> ShortTermScheduleResult:
    """
    Build a short-term schedule from a block model.

    Pipeline:
        1. Filter blocks by resource class, material, min tonnes
        2. Group into scheduling units (bench, pit, domain, etc.)
        3. For each period, allocate units to meet plant target
           while respecting grade blend constraints
        4. Route waste to waste destination

    Args:
        block_model: Block model DataFrame.
        config:      Full scheduler configuration.

    Returns:
        ShortTermScheduleResult with periods, units, and metadata.
    """
    total_loaded = len(block_model)
    logger.info(f"Starting short-term scheduler: {total_loaded} blocks loaded")

    # ── Step 1: Filter ────────────────────────────────────────────────────
    filtered_df = filter_blocks(block_model, config)

    # ── Step 2: Group ─────────────────────────────────────────────────────
    all_units = group_blocks(filtered_df, config)

    ore_units = [u for u in all_units if u.material not in ("Waste", "Overburden")]
    waste_units = [u for u in all_units if u.material in ("Waste", "Overburden")]

    elements = [gs.element for gs in config.grade_specs]
    period_labels = config.period_type.labels
    num_periods = config.period_type.count
    plant_target = config.plant_target_per_period
    primary_dest = config.destinations[config.primary_destination] if config.destinations else "Plant"
    waste_dest = (
        config.destinations[config.waste_destination]
        if len(config.destinations) > config.waste_destination
        else "Waste Dump"
    )

    # ── Step 3: Allocate ──────────────────────────────────────────────────
    periods: List[PeriodSummary] = []

    for p_idx, p_label in enumerate(period_labels):
        remaining = plant_target
        total_ore = 0.0
        blend: Dict[str, float] = {el: 0.0 for el in elements}
        entries: List[ScheduleEntry] = []

        # Shuffle ore units slightly for variety across periods
        rng = np.random.default_rng(seed=42 + p_idx)
        shuffled_ore = list(ore_units)
        rng.shuffle(shuffled_ore)

        for unit in shuffled_ore:
            if remaining <= 0:
                break

            available = unit.tonnes / num_periods
            take = min(available, remaining * (0.3 + rng.random() * 0.5))
            take = round(take / 10) * 10  # round to nearest 10 t
            if take <= 0:
                continue

            # Check blend feasibility
            if not check_blend_feasibility(
                blend, total_ore, unit.grades, take,
                config.grade_specs, plant_target=plant_target,
            ):
                continue

            # Allocate
            entry = ScheduleEntry(
                period_index=p_idx,
                period_label=p_label,
                unit_id=unit.id,
                unit_name=unit.name,
                material=unit.material,
                tonnes=take,
                destination=primary_dest,
                grades=dict(unit.grades),
                block_count=unit.block_count,
            )
            entries.append(entry)

            # Update blend
            new_total = total_ore + take
            for el in elements:
                blend[el] = (
                    (blend[el] * total_ore + unit.grades.get(el, 0.0) * take)
                    / new_total
                )
            total_ore = new_total
            remaining -= take

        # Waste allocation
        total_waste = 0.0
        for unit in waste_units:
            waste_take = round(
                (unit.tonnes / num_periods * (0.6 + rng.random() * 0.4)) / 10
            ) * 10
            if waste_take > 0:
                entries.append(ScheduleEntry(
                    period_index=p_idx,
                    period_label=p_label,
                    unit_id=unit.id,
                    unit_name=unit.name,
                    material=unit.material,
                    tonnes=waste_take,
                    destination=waste_dest,
                    grades=dict(unit.grades),
                    block_count=unit.block_count,
                ))
                total_waste += waste_take

        # Check grade compliance
        ore_entries = [e for e in entries if e.destination == primary_dest]
        blended_grades = compute_blended_grades(ore_entries, elements)
        compliance = all(
            gs.in_spec(blended_grades.get(gs.element, 0.0))
            for gs in config.grade_specs
        )

        periods.append(PeriodSummary(
            index=p_idx,
            label=p_label,
            total_ore_tonnes=total_ore,
            total_waste_tonnes=total_waste,
            blended_grades=blended_grades,
            entries=entries,
            grade_compliance=compliance,
        ))

    result = ShortTermScheduleResult(
        periods=periods,
        units=all_units,
        config=config,
        total_blocks_loaded=total_loaded,
        total_blocks_filtered=len(filtered_df),
        metadata={
            "method": "heuristic_blend",
            "num_periods": num_periods,
            "plant_target": plant_target,
        },
    )

    logger.info(
        f"Schedule complete: {num_periods} periods, "
        f"{result.total_ore / 1000:.0f}k ore, "
        f"{result.total_waste / 1000:.0f}k waste, "
        f"{result.compliance_rate * 100:.0f}% grade compliance"
    )
    return result
