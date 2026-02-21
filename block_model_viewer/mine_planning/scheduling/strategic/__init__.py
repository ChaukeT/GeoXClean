"""
Strategic Scheduling (STEP 30)

Annual LOM scheduling with MILP, nested shells, and cutoff optimization.
"""

from .strategic_milp import (
    StrategicScheduleConfig,
    build_strategic_milp_model,
    solve_strategic_schedule
)

from .nested_shell_scheduler import (
    NestedShellScheduleConfig,
    allocate_shells_to_periods
)

# ⚠️ REMOVED: cutoff_scheduler.py has been deleted.
# Use mine_planning.cutoff.cutoff_engine instead for robust pattern-based cutoff optimization.

__all__ = [
    # Strategic MILP
    "StrategicScheduleConfig",
    "build_strategic_milp_model",
    "solve_strategic_schedule",
    # Nested Shells
    "NestedShellScheduleConfig",
    "allocate_shells_to_periods",
    # Cutoff optimization moved to mine_planning.cutoff.cutoff_engine
]

