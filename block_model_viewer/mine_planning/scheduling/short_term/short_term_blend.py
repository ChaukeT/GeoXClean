"""
Short-Term Blend Optimiser (STEP 30)

Ensure daily/weekly plant feed respects grade bands and tonnage targets.

Supports two modes:
  1. Heuristic greedy blend (always available)
  2. LP/MILP blend via SciPy or PuLP (when installed)

Author: BlockModelViewer Team
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Optional LP solvers
try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class BlendSource:
    """
    A source of material for blending (scheduling unit, stockpile, etc.).

    Attributes:
        id:         Unique identifier.
        name:       Display name.
        tonnes:     Available tonnage for the period.
        grades:     Grade by element {element: value}.
        material:   Material type.
        min_tonnes: Minimum draw if selected (optional).
        max_tonnes: Maximum draw (defaults to available tonnes).
        priority:   Higher priority sources are preferred (optional).
    """
    id: str
    name: str
    tonnes: float
    grades: Dict[str, float]
    material: str = "Ore"
    min_tonnes: float = 0.0
    max_tonnes: Optional[float] = None
    priority: int = 0

    @property
    def effective_max(self) -> float:
        return self.max_tonnes if self.max_tonnes is not None else self.tonnes


@dataclass
class BlendSpec:
    """Grade specification for the blend target."""
    element: str
    min_grade: float
    max_grade: float
    weight: float = 1.0  # Penalty weight for out-of-spec (used in LP)

    @property
    def target(self) -> float:
        return (self.min_grade + self.max_grade) / 2.0


@dataclass
class ShortTermBlendConfig:
    """
    Configuration for short-term blending.

    Attributes:
        sources:            Available blend sources.
        specs:              Grade specifications.
        tonnage_target:     Target total tonnage for the period.
        tonnage_tolerance:  Acceptable % deviation from target (0.0–1.0).
        method:             'heuristic', 'scipy_lp', or 'pulp_milp'.
        allow_partial:      Allow under-target if blend is infeasible.
    """
    sources: List[BlendSource] = field(default_factory=list)
    specs: List[BlendSpec] = field(default_factory=list)
    tonnage_target: float = 25_000.0
    tonnage_tolerance: float = 0.05
    method: str = "heuristic"
    allow_partial: bool = True


@dataclass
class BlendAllocation:
    """Allocation from a single source."""
    source_id: str
    source_name: str
    tonnes: float
    grades: Dict[str, float]
    fraction: float = 0.0  # Fraction of total blend


@dataclass
class BlendResult:
    """Result from blend optimisation."""
    allocations: List[BlendAllocation]
    total_tonnes: float
    blended_grades: Dict[str, float]
    grade_compliance: Dict[str, bool]
    all_in_spec: bool
    tonnage_met: bool
    method_used: str
    objective_value: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Heuristic Blend ─────────────────────────────────────────────────────────

def _heuristic_blend(config: ShortTermBlendConfig) -> BlendResult:
    """
    Greedy heuristic: iterate sources by priority, add to blend if
    the resulting mix stays within spec.
    """
    sorted_sources = sorted(config.sources, key=lambda s: -s.priority)
    elements = [s.element for s in config.specs]
    spec_map = {s.element: s for s in config.specs}

    allocations: List[BlendAllocation] = []
    total_tonnes = 0.0
    blend: Dict[str, float] = {el: 0.0 for el in elements}
    remaining = config.tonnage_target

    for src in sorted_sources:
        if remaining <= 0:
            break
        if src.tonnes <= 0:
            continue

        take = min(src.effective_max, remaining)
        if take < src.min_tonnes:
            continue

        # Check if adding this source keeps blend in spec
        new_total = total_tonnes + take
        new_blend = {}
        feasible = True

        for el in elements:
            src_grade = src.grades.get(el, 0.0)
            new_blend[el] = (
                (blend[el] * total_tonnes + src_grade * take) / new_total
                if new_total > 0 else src_grade
            )
            spec = spec_map.get(el)
            if spec and total_tonnes > config.tonnage_target * 0.3:
                if new_blend[el] < spec.min_grade or new_blend[el] > spec.max_grade:
                    feasible = False
                    break

        if feasible:
            allocations.append(BlendAllocation(
                source_id=src.id,
                source_name=src.name,
                tonnes=take,
                grades=dict(src.grades),
            ))
            blend = new_blend
            total_tonnes = new_total
            remaining -= take

    # Compute fractions
    for a in allocations:
        a.fraction = a.tonnes / total_tonnes if total_tonnes > 0 else 0.0

    # Grade compliance
    compliance = {}
    for spec in config.specs:
        val = blend.get(spec.element, 0.0)
        compliance[spec.element] = spec.min_grade <= val <= spec.max_grade

    tol = config.tonnage_target * config.tonnage_tolerance
    tonnage_met = abs(total_tonnes - config.tonnage_target) <= tol

    return BlendResult(
        allocations=allocations,
        total_tonnes=total_tonnes,
        blended_grades=blend,
        grade_compliance=compliance,
        all_in_spec=all(compliance.values()),
        tonnage_met=tonnage_met,
        method_used="heuristic",
    )


# ─── SciPy LP Blend ──────────────────────────────────────────────────────────

def _scipy_lp_blend(config: ShortTermBlendConfig) -> BlendResult:
    """
    Linear programming blend using SciPy.

    Decision variables: tonnes drawn from each source.
    Objective:          Minimise deviation from target grades.
    Constraints:        Grade min/max, tonnage target, source availability.
    """
    if not SCIPY_AVAILABLE:
        logger.warning("SciPy not available, falling back to heuristic")
        return _heuristic_blend(config)

    n = len(config.sources)
    elements = [s.element for s in config.specs]

    # Decision variables: x_i = tonnes from source i
    # Objective: minimise slack (we just want feasibility, so use zeros)
    c = np.zeros(n)  # No preference (could weight by distance, cost, etc.)

    # Bounds: 0 <= x_i <= min(available, max_tonnes)
    bounds = [(0, src.effective_max) for src in config.sources]

    # Equality constraint: sum(x_i) = target
    A_eq = np.ones((1, n))
    b_eq = np.array([config.tonnage_target])

    # Inequality constraints for grade specs
    A_ub_rows = []
    b_ub_rows = []

    for spec in config.specs:
        grades = np.array([
            src.grades.get(spec.element, 0.0) for src in config.sources
        ])

        # grade_blend = sum(g_i * x_i) / sum(x_i) >= min_grade
        # → sum((g_i - min_grade) * x_i) >= 0
        # → sum((min_grade - g_i) * x_i) <= 0
        A_ub_rows.append(spec.min_grade - grades)
        b_ub_rows.append(0.0)

        # grade_blend <= max_grade
        # → sum((g_i - max_grade) * x_i) <= 0
        A_ub_rows.append(grades - spec.max_grade)
        b_ub_rows.append(0.0)

    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_rows) if b_ub_rows else None

    try:
        result = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
            bounds=bounds, method="highs",
        )

        if result.success:
            allocations = []
            total_tonnes = 0.0
            for i, src in enumerate(config.sources):
                t = result.x[i]
                if t > 0.5:  # Small threshold
                    allocations.append(BlendAllocation(
                        source_id=src.id,
                        source_name=src.name,
                        tonnes=t,
                        grades=dict(src.grades),
                    ))
                    total_tonnes += t

            # Compute blended grades
            blend = {}
            for el in elements:
                if total_tonnes > 0:
                    blend[el] = sum(
                        a.grades.get(el, 0.0) * a.tonnes for a in allocations
                    ) / total_tonnes
                else:
                    blend[el] = 0.0

            for a in allocations:
                a.fraction = a.tonnes / total_tonnes if total_tonnes > 0 else 0.0

            compliance = {}
            for spec in config.specs:
                val = blend.get(spec.element, 0.0)
                compliance[spec.element] = spec.min_grade <= val <= spec.max_grade

            return BlendResult(
                allocations=allocations,
                total_tonnes=total_tonnes,
                blended_grades=blend,
                grade_compliance=compliance,
                all_in_spec=all(compliance.values()),
                tonnage_met=True,
                method_used="scipy_lp",
                objective_value=result.fun,
                metadata={"solver_status": result.message},
            )
        else:
            logger.warning(f"LP infeasible ({result.message}), falling back to heuristic")
            return _heuristic_blend(config)

    except Exception as e:
        logger.warning(f"LP solver error: {e}, falling back to heuristic")
        return _heuristic_blend(config)


# ─── PuLP MILP Blend ─────────────────────────────────────────────────────────

def _pulp_milp_blend(config: ShortTermBlendConfig) -> BlendResult:
    """
    Mixed-Integer LP blend using PuLP.

    Adds binary selection variables and minimum-draw constraints.
    """
    if not PULP_AVAILABLE:
        logger.warning("PuLP not available, falling back to scipy LP")
        return _scipy_lp_blend(config)

    prob = pulp.LpProblem("ShortTermBlend", pulp.LpMinimize)
    n = len(config.sources)
    elements = [s.element for s in config.specs]

    # Decision variables
    x = [pulp.LpVariable(f"x_{i}", 0, src.effective_max) for i, src in enumerate(config.sources)]
    y = [pulp.LpVariable(f"y_{i}", cat="Binary") for i in range(n)]  # selection

    # Objective: minimise deviation from target grades (weighted)
    # Use auxiliary slack variables for grade deviation
    slack_vars = {}
    for spec in config.specs:
        s_lo = pulp.LpVariable(f"slack_lo_{spec.element}", 0)
        s_hi = pulp.LpVariable(f"slack_hi_{spec.element}", 0)
        slack_vars[spec.element] = (s_lo, s_hi)

    prob += pulp.lpSum(
        spec.weight * (slack_vars[spec.element][0] + slack_vars[spec.element][1])
        for spec in config.specs
    )

    # Tonnage target
    prob += pulp.lpSum(x) >= config.tonnage_target * (1 - config.tonnage_tolerance)
    prob += pulp.lpSum(x) <= config.tonnage_target * (1 + config.tonnage_tolerance)

    # Link x and y: x_i <= max_i * y_i
    for i, src in enumerate(config.sources):
        prob += x[i] <= src.effective_max * y[i]
        if src.min_tonnes > 0:
            prob += x[i] >= src.min_tonnes * y[i]

    # Grade constraints (linearised around tonnage target)
    T = config.tonnage_target
    for spec in config.specs:
        grades = [src.grades.get(spec.element, 0.0) for src in config.sources]
        grade_sum = pulp.lpSum(grades[i] * x[i] for i in range(n))
        s_lo, s_hi = slack_vars[spec.element]

        prob += grade_sum >= spec.min_grade * T - s_lo
        prob += grade_sum <= spec.max_grade * T + s_hi

    try:
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

        if prob.status == pulp.constants.LpStatusOptimal:
            allocations = []
            total_tonnes = 0.0
            for i, src in enumerate(config.sources):
                t = x[i].varValue or 0.0
                if t > 0.5:
                    allocations.append(BlendAllocation(
                        source_id=src.id,
                        source_name=src.name,
                        tonnes=t,
                        grades=dict(src.grades),
                    ))
                    total_tonnes += t

            blend = {}
            for el in elements:
                blend[el] = (
                    sum(a.grades.get(el, 0.0) * a.tonnes for a in allocations) / total_tonnes
                    if total_tonnes > 0 else 0.0
                )

            for a in allocations:
                a.fraction = a.tonnes / total_tonnes if total_tonnes > 0 else 0.0

            compliance = {}
            for spec in config.specs:
                val = blend.get(spec.element, 0.0)
                compliance[spec.element] = spec.min_grade <= val <= spec.max_grade

            return BlendResult(
                allocations=allocations,
                total_tonnes=total_tonnes,
                blended_grades=blend,
                grade_compliance=compliance,
                all_in_spec=all(compliance.values()),
                tonnage_met=True,
                method_used="pulp_milp",
                objective_value=pulp.value(prob.objective),
                metadata={"solver_status": pulp.LpStatus[prob.status]},
            )
        else:
            logger.warning("MILP infeasible, falling back to heuristic")
            return _heuristic_blend(config)

    except Exception as e:
        logger.warning(f"MILP solver error: {e}, falling back to heuristic")
        return _heuristic_blend(config)


# ─── Public Interface ─────────────────────────────────────────────────────────

def optimise_short_term_blend(config: ShortTermBlendConfig) -> BlendResult:
    """
    Run the blend optimiser with the configured method.

    Falls back gracefully: pulp_milp → scipy_lp → heuristic.

    Args:
        config: BlendConfig with sources, specs, and method.

    Returns:
        BlendResult with allocations and compliance.
    """
    method = config.method.lower()

    if method == "pulp_milp":
        return _pulp_milp_blend(config)
    elif method == "scipy_lp":
        return _scipy_lp_blend(config)
    else:
        return _heuristic_blend(config)


def build_blend_sources_from_units(
    units: List[Any],
    period_count: int,
) -> List[BlendSource]:
    """
    Convert scheduling units into blend sources for a single period.

    Divides each unit's total tonnes equally across periods.

    Args:
        units:        List of SchedulingUnit (from block_model_scheduler).
        period_count: Number of periods in the schedule.

    Returns:
        List of BlendSource.
    """
    sources = []
    for unit in units:
        if unit.material in ("Waste", "Overburden"):
            continue
        sources.append(BlendSource(
            id=unit.id,
            name=unit.name,
            tonnes=unit.tonnes / max(period_count, 1),
            grades=dict(unit.grades),
            material=unit.material,
        ))
    return sources
