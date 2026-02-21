"""
Shift Plan Generator (STEP 30)

Convert period-level schedule into shift-by-shift operating plan
with equipment assignments and operator allocations.

Author: BlockModelViewer Team
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class ShiftConfig:
    """
    Configuration for a single shift.

    Attributes:
        name:             Shift display name (e.g. "Day", "Night").
        hours:            Shift duration in hours.
        availability:     Equipment availability factor (0.0–1.0).
        operator_count:   Number of operators available this shift.
    """
    name: str
    hours: float
    availability: float = 0.85
    operator_count: Optional[int] = None


@dataclass
class EquipmentUnit:
    """
    An individual piece of equipment available for assignment.

    Attributes:
        id:         Equipment identifier (e.g. "TR-01").
        name:       Display name.
        type:       Equipment type (Haul Truck, Shovel, Excavator, etc.).
        model:      Equipment model string.
        capacity_t: Payload or bucket capacity in tonnes.
        status:     Current status (Active, Standby, Maintenance, Down).
    """
    id: str
    name: str
    type: str
    model: str = ""
    capacity_t: float = 0.0
    status: str = "Active"

    @property
    def is_available(self) -> bool:
        return self.status in ("Active", "Standby")


# ─── Results ──────────────────────────────────────────────────────────────────

@dataclass
class ShiftAssignment:
    """
    Equipment/source assignment within a shift.

    Attributes:
        unit_id:        Scheduling unit ID.
        unit_name:      Scheduling unit name.
        material:       Material type.
        tonnes:         Tonnes allocated to this shift.
        destination:    Material destination.
        equipment_ids:  Assigned equipment IDs.
        truck_count:    Number of trucks allocated.
        loads:          Estimated number of truck loads.
        cycle_time_min: Estimated haul cycle time in minutes.
    """
    unit_id: str
    unit_name: str
    material: str
    tonnes: float
    destination: str
    equipment_ids: List[str] = field(default_factory=list)
    truck_count: int = 0
    loads: int = 0
    cycle_time_min: float = 0.0


@dataclass
class ShiftPlanEntry:
    """Plan for a single shift within a period."""
    period_index: int
    period_label: str
    shift_name: str
    shift_hours: float
    target_tonnes: float
    assignments: List[ShiftAssignment] = field(default_factory=list)

    @property
    def total_tonnes(self) -> float:
        return sum(a.tonnes for a in self.assignments)

    @property
    def ore_tonnes(self) -> float:
        return sum(
            a.tonnes for a in self.assignments
            if a.material not in ("Waste", "Overburden")
        )

    @property
    def waste_tonnes(self) -> float:
        return sum(
            a.tonnes for a in self.assignments
            if a.material in ("Waste", "Overburden")
        )


@dataclass
class ShiftPlanResult:
    """Complete shift plan across all periods."""
    entries: List[ShiftPlanEntry]
    shift_configs: List[ShiftConfig]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_entries_for_period(self, period_index: int) -> List[ShiftPlanEntry]:
        return [e for e in self.entries if e.period_index == period_index]

    def get_entries_for_shift(self, shift_name: str) -> List[ShiftPlanEntry]:
        return [e for e in self.entries if e.shift_name == shift_name]


# ─── Generator ────────────────────────────────────────────────────────────────

def generate_shift_plan(
    schedule_result: Any,  # ShortTermScheduleResult
    shift_configs: List[ShiftConfig],
    equipment: Optional[List[EquipmentUnit]] = None,
    default_cycle_time_min: float = 25.0,
    default_truck_capacity: float = 250.0,
) -> ShiftPlanResult:
    """
    Generate shift plan from a short-term schedule.

    Distributes each period's allocations across shifts proportional
    to shift duration and availability.  Optionally assigns equipment.

    Args:
        schedule_result:        ShortTermScheduleResult (from block_model_scheduler).
        shift_configs:          List of ShiftConfig defining the shifts.
        equipment:              Optional list of available equipment.
        default_cycle_time_min: Default haul cycle time for truck calculation.
        default_truck_capacity: Default truck payload if no equipment provided.

    Returns:
        ShiftPlanResult with per-shift assignments.
    """
    if not shift_configs:
        shift_configs = [
            ShiftConfig(name="Day", hours=12.0),
            ShiftConfig(name="Night", hours=12.0),
        ]

    total_shift_hours = sum(s.hours * s.availability for s in shift_configs)
    if total_shift_hours <= 0:
        total_shift_hours = sum(s.hours for s in shift_configs) or 24.0

    # Available trucks (for assignment)
    available_trucks = []
    available_loaders = []
    if equipment:
        available_trucks = [
            e for e in equipment
            if e.type in ("Haul Truck", "ADT") and e.is_available
        ]
        available_loaders = [
            e for e in equipment
            if e.type in ("Shovel", "Excavator", "Loader", "LHD") and e.is_available
        ]

    entries: List[ShiftPlanEntry] = []

    for period in schedule_result.periods:
        for shift in shift_configs:
            # Fraction of day this shift represents (weighted by availability)
            shift_fraction = (
                (shift.hours * shift.availability) / total_shift_hours
                if total_shift_hours > 0
                else 1.0 / len(shift_configs)
            )

            assignments: List[ShiftAssignment] = []
            truck_index = 0

            for entry in period.entries:
                shift_tonnes = round(entry.tonnes * shift_fraction)
                if shift_tonnes <= 0:
                    continue

                # Calculate truck requirements
                truck_cap = default_truck_capacity
                if available_trucks:
                    truck_cap = (
                        sum(t.capacity_t for t in available_trucks) / len(available_trucks)
                        if available_trucks else default_truck_capacity
                    )

                loads = max(1, int(np.ceil(shift_tonnes / truck_cap))) if truck_cap > 0 else 0
                productive_hours = shift.hours * shift.availability
                cycles_per_truck = (
                    (productive_hours * 60) / default_cycle_time_min
                    if default_cycle_time_min > 0 else 0
                )
                trucks_needed = (
                    max(1, int(np.ceil(loads / cycles_per_truck)))
                    if cycles_per_truck > 0 else 1
                )

                # Assign specific equipment IDs
                assigned_ids = []
                if available_trucks:
                    for _ in range(min(trucks_needed, len(available_trucks))):
                        assigned_ids.append(
                            available_trucks[truck_index % len(available_trucks)].id
                        )
                        truck_index += 1

                # Assign a loader
                if available_loaders and entry.material not in ("Waste", "Overburden"):
                    assigned_ids.append(available_loaders[0].id)

                assignments.append(ShiftAssignment(
                    unit_id=entry.unit_id,
                    unit_name=entry.unit_name,
                    material=entry.material,
                    tonnes=shift_tonnes,
                    destination=entry.destination,
                    equipment_ids=assigned_ids,
                    truck_count=trucks_needed,
                    loads=loads,
                    cycle_time_min=default_cycle_time_min,
                ))

            shift_target = period.total_ore_tonnes * shift_fraction

            entries.append(ShiftPlanEntry(
                period_index=period.index,
                period_label=period.label,
                shift_name=shift.name,
                shift_hours=shift.hours,
                target_tonnes=shift_target,
                assignments=assignments,
            ))

    result = ShiftPlanResult(
        entries=entries,
        shift_configs=shift_configs,
        metadata={
            "total_shifts": len(entries),
            "equipment_assigned": equipment is not None,
            "cycle_time_min": default_cycle_time_min,
        },
    )

    logger.info(
        f"Shift plan generated: {len(entries)} shifts "
        f"({len(shift_configs)} shifts/day × "
        f"{len(schedule_result.periods)} periods)"
    )
    return result


# ─── Numpy import (deferred for cycle time calc) ─────────────────────────────

import numpy as np
