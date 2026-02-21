from __future__ import annotations

from dataclasses import dataclass, field

from enum import Enum

from typing import Dict, List, Optional, Iterable, Any, Tuple, Callable
import math
import bisect
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS / CONSTANTS
# =============================================================================


class CompositingMethod(str, Enum):
    FIXED_LENGTH = "fixed_length"
    EQUAL_MASS = "equal_mass"
    EQUAL_VOLUME = "equal_volume"
    BENCH_ALIGNED = "bench_aligned"
    LITHOLOGY = "lithology"
    TRUE_THICKNESS = "true_thickness"
    ROLLING_WINDOW = "rolling_window"
    ECONOMIC = "economic"  # ore/waste envelope compositing
    INDICATOR = "indicator"  # ore/waste coding
    ATTRIBUTE_FILTERED = "attribute_filtered"


class BreakMode(str, Enum):
    NONE = "none"  # ignore boundaries
    HARD = "hard"  # always break at boundaries
    SOFT = "soft"  # break only if close to composite end


class WeightingMode(str, Enum):
    LENGTH = "length"
    DENSITY = "density"  # treated as length * density (mass proxy)
    MASS = "mass"        # explicit mass weighting, requires density
    VOLUME = "volume"    # same as length in 1D downhole support


class PartialStrategy(str, Enum):
    DISCARD = "discard"
    MERGE = "merge"
    KEEP = "keep"
    AUTO = "auto"  # fraction-based rule, e.g. >= 50 % keep, else merge


class EconomicDilutionRule(str, Enum):
    """Dilution rules for economic compositing."""
    BASIC = "basic"
    ADVANCED = "advanced"
    ADVANCED_PLUS = "advanced_plus"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class Interval:
    """Generic downhole interval record.

    This is the *raw* support before compositing.
    - from_depth, to_depth: metres downhole
    - grades: numeric fields, e.g. {"Fe": 62.3, "SiO2": 5.1}
    - lith, domain, sample_type, density: optional attributes used by
      break logic and weighting.
    """

    hole_id: str
    from_depth: float
    to_depth: float
    grades: Dict[str, float] = field(default_factory=dict)

    lith: Optional[str] = None
    domain: Optional[str] = None
    sample_type: Optional[str] = None
    density: Optional[float] = None  # t/m³ if used for mass/density weighting
    recovery: Optional[float] = None  # 0–100 %
    true_thickness: Optional[float] = None  # metres (geometrically corrected)
    flags: Dict[str, Any] = field(default_factory=dict)  # QAQC / metadata

    @property
    def length(self) -> float:
        return self.to_depth - self.from_depth


@dataclass
class Composite:
    """Composite interval produced by the engine.

    - grades: composite grades (weighted according to config)
    - metadata: extra info (sample count, weights, QA flags, etc.)
    """

    hole_id: str
    from_depth: float
    to_depth: float
    grades: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        return self.to_depth - self.from_depth


@dataclass
class CompositeConfig:
    """Core configuration for the compositing method engine.

    Higher-level engines (numerical / lithological / economic / indicator)
    will pre-populate this for specific workflows.
    
    Configuration Parameters
    ------------------------
    All optional parameters have documented defaults. Users should be aware of:
    - treat_null_as_zero: True by default (mining industry standard)
    - partial_strategy: DISCARD by default
    - weighting_mode: LENGTH by default
    """

    method: CompositingMethod

    # General support / geometry
    composite_length: Optional[float] = None  # used as target length for FIXED_LENGTH
    rolling_window_length: Optional[float] = None
    rolling_step: Optional[float] = None

    # Equal-mass (tonnage) target
    target_mass: Optional[float] = None       # tonnes per composite for EQUAL_MASS

    # Break behaviour
    break_mode: BreakMode = BreakMode.NONE
    hard_break_lithology: bool = False
    hard_break_domain: bool = False
    hard_break_sample_type: bool = False
    soft_break_threshold: float = 0.20  # metres

    # Partial composites
    partial_strategy: PartialStrategy = PartialStrategy.DISCARD
    min_partial_length: float = 0.30        # m (used by higher-level logic if needed)
    auto_partial_fraction: float = 0.50     # used when partial_strategy == AUTO

    # Weighting
    weighting_mode: WeightingMode = WeightingMode.LENGTH
    density_field: str = "density"         # authoritative density attribute (Interval.density)

    # Economic / indicator parameters
    cutoff_field: Optional[str] = None      # e.g. "Fe"
    cutoff_grade: Optional[float] = None
    
    # Economic compositing parameters (not "mining constraints")
    min_ore_composite_length: Optional[float] = None  # Minimum length for ore composite
    max_included_waste: Optional[float] = None  # Maximum total waste length in ore composite
    max_consecutive_waste: Optional[float] = None  # Maximum consecutive waste segment length
    min_linear_grade: Optional[float] = None  # Min linear grade to keep short high-grade composites
    min_waste_composite_length: Optional[float] = None  # Min waste composite length (Advanced/Advanced+)
    keep_short_high_grade: bool = False  # Allow short composites if min_linear_grade met
    dilution_rule: Optional[EconomicDilutionRule] = None  # Basic, Advanced, or Advanced+
    composite_twice: bool = False  # Two-pass compositing
    use_true_thickness: bool = False  # Use true thickness for economic compositing
    true_thickness_dip: Optional[float] = None  # Dip for true thickness calculation
    true_thickness_dip_azimuth: Optional[float] = None  # Dip azimuth for true thickness

    # Lithology compositing parameters (H-04, M-05 fix)
    min_ore_thickness: Optional[float] = None  # Minimum thickness for ore intervals
    min_mining_width: Optional[float] = None   # Minimum mining width constraint

    # Attribute filtering / QAQC
    exclude_qaqc: bool = True
    qaqc_flags: Tuple[str, ...] = ("BLANK", "STANDARD", "DUPLICATE")
    min_recovery: Optional[float] = None  # %; below this may be excluded or flagged upstream
    
    # Missing grade handling
    treat_null_as_zero: bool = True  # If True, treat None/NaN grades as 0.0; if False, exclude from weighting

    # Bench-aligned
    bench_height: Optional[float] = None
    bench_offset: float = 0.0

    # For ATTRIBUTE_FILTERED: which core method to use after filtering
    base_method_for_attribute_filtered: CompositingMethod = CompositingMethod.FIXED_LENGTH


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def _group_by_hole(intervals: Iterable[Interval]) -> Dict[str, List[Interval]]:
    holes: Dict[str, List[Interval]] = {}
    for iv in intervals:
        holes.setdefault(iv.hole_id, []).append(iv)
    for hole_id, lst in holes.items():
        lst.sort(key=lambda x: x.from_depth)
    return holes


def _interval_weight(interval: Interval, length_used: float, cfg: CompositeConfig) -> float:
    """Compute weighting factor for a segment of an interval.

    For MASS or DENSITY weighting, density is mandatory. If missing, the
    engine raises to avoid silent, non-physical behaviour.
    """

    if cfg.weighting_mode in (WeightingMode.MASS, WeightingMode.DENSITY):
        if interval.density is None:
            raise ValueError(
                f"Density is required for {cfg.weighting_mode} weighting "
                f"(hole {interval.hole_id}, interval {interval.from_depth}-{interval.to_depth})."
            )
        # For both MASS and DENSITY modes we effectively use mass = length * density
        return length_used * interval.density

    # LENGTH / VOLUME weighting → weight = length
    return length_used


def _is_break(prev_iv: Interval, curr_iv: Interval, cfg: CompositeConfig) -> bool:
    """Determine if there is a boundary between prev_iv and curr_iv that qualifies
    as a break candidate (for HARD or SOFT behaviour)."""

    if cfg.break_mode == BreakMode.NONE:
        return False

    if cfg.hard_break_lithology and prev_iv.lith != curr_iv.lith:
        return True
    if cfg.hard_break_domain and prev_iv.domain != curr_iv.domain:
        return True
    if cfg.hard_break_sample_type and prev_iv.sample_type != curr_iv.sample_type:
        return True

    return False


class HoleAccumulator:
    """
    Helper to calculate weighted grades for any depth range instantly.
    
    Pre-computes cumulative sums for O(1) queries.
    """
    
    def __init__(self, intervals: List[Interval], grade_field: str, cfg: CompositeConfig):
        self.grade_field = grade_field
        self.cfg = cfg
        self.depths = []
        self.cum_len = [0.0]
        self.cum_metal = [0.0]  # grade * weight
        self.cum_weight = [0.0]
        
        current_len = 0.0
        current_metal = 0.0
        current_weight = 0.0
        
        # Sort is critical
        self.intervals = sorted(intervals, key=lambda x: x.from_depth)
        
        # Build 1D arrays for fast lookup
        # We assume intervals are contiguous for the math,
        # gaps are handled by checking actual depths in query.
        for iv in self.intervals:
            self.depths.append(iv.to_depth)
            
            w = _interval_weight(iv, iv.length, cfg)
            g = _get_grade_value(iv.grades.get(grade_field, 0.0), cfg)
            if g is None:
                g = 0.0  # Treat None as 0 for accumulator
            
            current_len += iv.length
            current_weight += w
            current_metal += g * w
            
            self.cum_len.append(current_len)
            self.cum_weight.append(current_weight)
            self.cum_metal.append(current_metal)
    
    def get_stats(self, start: float, end: float) -> Tuple[float, float]:
        """
        Returns (average_grade, total_length) for range [start, end].
        
        Uses binary search for O(log N) lookup, then O(1) calculation.
        """
        if not self.intervals:
            return 0.0, 0.0
        
        # Find indices using binary search
        # The depths array contains 'to_depth'.
        # idx_start is the index of the first interval involved
        idx_start = bisect.bisect_right(self.depths, start)
        idx_end = bisect.bisect_right(self.depths, end)
        
        if idx_start >= len(self.intervals):
            return 0.0, 0.0
        
        # Handle partial start/end intervals
        # For speed in Economic Compositing loop, we approximate
        # if boundaries align with raw data (which they usually do).
        # For exact calculation, we'd need to handle partial intervals,
        # but for economic compositing this is usually sufficient.
        
        # Get cumulative sums
        metal_start = self.cum_metal[idx_start]
        metal_end = self.cum_metal[idx_end]
        weight_start = self.cum_weight[idx_start]
        weight_end = self.cum_weight[idx_end]
        
        metal = metal_end - metal_start
        weight = weight_end - weight_start
        
        # Handle partial intervals at boundaries
        if idx_start < len(self.intervals):
            iv_start = self.intervals[idx_start]
            if start > iv_start.from_depth:
                # Partial start interval
                partial_len = min(iv_start.to_depth, end) - start
                if partial_len > 0:
                    w_partial = _interval_weight(iv_start, partial_len, self.cfg)
                    g_partial = _get_grade_value(iv_start.grades.get(self.grade_field, 0.0), self.cfg)
                    if g_partial is None:
                        g_partial = 0.0
                    # Adjust: subtract full interval, add partial
                    metal -= self.cum_metal[idx_start + 1] - self.cum_metal[idx_start]
                    weight -= self.cum_weight[idx_start + 1] - self.cum_weight[idx_start]
                    metal += g_partial * w_partial
                    weight += w_partial
        
        if idx_end < len(self.intervals):
            iv_end = self.intervals[idx_end]
            if end < iv_end.to_depth:
                # Partial end interval
                partial_len = end - max(iv_end.from_depth, start)
                if partial_len > 0:
                    w_partial = _interval_weight(iv_end, partial_len, self.cfg)
                    g_partial = _get_grade_value(iv_end.grades.get(self.grade_field, 0.0), self.cfg)
                    if g_partial is None:
                        g_partial = 0.0
                    # Adjust: subtract full interval, add partial
                    metal -= self.cum_metal[idx_end + 1] - self.cum_metal[idx_end]
                    weight -= self.cum_weight[idx_end + 1] - self.cum_weight[idx_end]
                    metal += g_partial * w_partial
                    weight += w_partial
        
        if weight > 1e-8:
            return metal / weight, weight
        return 0.0, 0.0


def _get_grade_value(grade_val: Optional[float], cfg: CompositeConfig) -> Optional[float]:
    """
    Helper function to handle None/NaN grade values according to config.
    
    Returns:
        - If grade_val is not None: returns grade_val
        - If grade_val is None and treat_null_as_zero is True: returns 0.0
        - If grade_val is None and treat_null_as_zero is False: returns None (skip)
    """
    if grade_val is None:
        if cfg.treat_null_as_zero:
            return 0.0
        else:
            return None  # Signal to skip
    return grade_val


# =============================================================================
# CORE ENGINE
# =============================================================================


class CompositingMethodEngine:
    """Core compositing engine.

    - works on Interval objects and returns Composite objects
    - mathematical core; no UI, no database assumptions
    - higher-level engines configure CompositeConfig appropriately
    - supports parallel processing for large datasets
    """
    
    def __init__(self, max_workers: int = 4, use_parallel: bool = True):
        """
        Initialize the compositing engine.
        
        Args:
            max_workers: Maximum number of parallel workers for multi-hole processing
            use_parallel: Whether to use parallel processing (disable for debugging)
        """
        self.max_workers = max_workers
        self.use_parallel = use_parallel
        self._progress_callback: Optional[Callable[[int, str], None]] = None

    def set_progress_callback(self, callback: Optional[Callable[[int, str], None]]) -> None:
        """Set a callback for progress updates: callback(percent, message)"""
        self._progress_callback = callback
    
    def _report_progress(self, percent: int, message: str) -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            try:
                self._progress_callback(percent, message)
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------------

    def composite(self,
                  intervals: Iterable[Interval],
                  cfg: CompositeConfig,
                  progress_callback: Optional[Callable[[int, str], None]] = None) -> List[Composite]:
        # Set progress callback for this run
        if progress_callback:
            self.set_progress_callback(progress_callback)
        
        self._report_progress(0, f"Starting {cfg.method.value} compositing...")
        
        if cfg.method == CompositingMethod.FIXED_LENGTH:
            result = self._composite_fixed_length_parallel(intervals, cfg)
        elif cfg.method == CompositingMethod.INDICATOR:
            result = self._composite_indicator(intervals, cfg)
        elif cfg.method == CompositingMethod.EQUAL_MASS:
            result = self._composite_equal_mass(intervals, cfg)
        elif cfg.method == CompositingMethod.EQUAL_VOLUME:
            result = self._composite_equal_volume(intervals, cfg)
        elif cfg.method == CompositingMethod.BENCH_ALIGNED:
            result = self._composite_bench_aligned(intervals, cfg)
        elif cfg.method == CompositingMethod.LITHOLOGY:
            result = self._composite_lithology(intervals, cfg)
        elif cfg.method == CompositingMethod.TRUE_THICKNESS:
            result = self._composite_true_thickness(intervals, cfg)
        elif cfg.method == CompositingMethod.ROLLING_WINDOW:
            result = self._composite_rolling_window(intervals, cfg)
        elif cfg.method == CompositingMethod.ECONOMIC:
            result = self._composite_economic(intervals, cfg)
        elif cfg.method == CompositingMethod.ATTRIBUTE_FILTERED:
            result = self._composite_attribute_filtered(intervals, cfg)
        else:
            raise ValueError(f"Unsupported compositing method: {cfg.method}")
        
        self._report_progress(100, f"Compositing complete: {len(result)} composites")
        return result

    # ---------------------------------------------------------------------
    # 1) FIXED-LENGTH COMPOSITING (with corrected weighting & partial logic)
    # ---------------------------------------------------------------------

    def _composite_fixed_length_parallel(self,
                                         intervals: Iterable[Interval],
                                         cfg: CompositeConfig) -> List[Composite]:
        """
        Parallel fixed-length compositing.
        
        For datasets with many holes, processes holes in parallel using ThreadPoolExecutor.
        Falls back to sequential for small datasets or when parallel is disabled.
        """
        if cfg.composite_length is None or cfg.composite_length <= 0:
            raise ValueError("composite_length must be > 0 for FIXED_LENGTH compositing")

        holes = _group_by_hole(intervals)
        n_holes = len(holes)
        
        self._report_progress(5, f"Processing {n_holes} holes...")
        
        # Use parallel processing for datasets with many holes
        if self.use_parallel and n_holes > 10:
            return self._composite_fixed_length_parallel_impl(holes, cfg)
        else:
            return self._composite_fixed_length_sequential(holes, cfg)
    
    def _composite_fixed_length_parallel_impl(self,
                                              holes: Dict[str, List[Interval]],
                                              cfg: CompositeConfig) -> List[Composite]:
        """Parallel implementation using ThreadPoolExecutor."""
        all_composites: List[Composite] = []
        n_holes = len(holes)
        completed = 0
        
        # Use ThreadPoolExecutor (not ProcessPool) to avoid pickling issues with dataclasses
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all holes for processing
            future_to_hole = {
                executor.submit(self._composite_fixed_length_single_hole, hole_id, hole_intervals, cfg): hole_id
                for hole_id, hole_intervals in holes.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_hole):
                hole_id = future_to_hole[future]
                try:
                    composites = future.result()
                    all_composites.extend(composites)
                    completed += 1
                    
                    # Report progress every 5% or so
                    progress = int(5 + (completed / n_holes) * 90)
                    if completed % max(1, n_holes // 20) == 0:
                        self._report_progress(progress, f"Processed {completed}/{n_holes} holes")
                except Exception as e:
                    logger.warning(f"Failed to composite hole {hole_id}: {e}")
        
        self._report_progress(95, f"Processed all {n_holes} holes")
        return all_composites
    
    def _composite_fixed_length_sequential(self,
                                           holes: Dict[str, List[Interval]],
                                           cfg: CompositeConfig) -> List[Composite]:
        """Sequential implementation for small datasets or debugging."""
        all_composites: List[Composite] = []
        n_holes = len(holes)
        
        for i, (hole_id, hole_intervals) in enumerate(holes.items()):
            all_composites.extend(
                self._composite_fixed_length_single_hole(hole_id, hole_intervals, cfg)
            )
            # Report progress
            if i % max(1, n_holes // 20) == 0:
                progress = int(5 + ((i + 1) / n_holes) * 90)
                self._report_progress(progress, f"Processing hole {i+1}/{n_holes}")
        
        return all_composites

    def _composite_fixed_length(self,
                                intervals: Iterable[Interval],
                                cfg: CompositeConfig) -> List[Composite]:
        """Legacy sequential method - kept for backward compatibility."""
        return self._composite_fixed_length_parallel(intervals, cfg)

    def _composite_fixed_length_single_hole(self,
                                            hole_id: str,
                                            intervals: List[Interval],
                                            cfg: CompositeConfig) -> List[Composite]:
        composites: List[Composite] = []

        if not intervals:
            return composites

        comp_len = cfg.composite_length
        comp_start = intervals[0].from_depth
        comp_end_target = comp_start + comp_len

        # Per-element accumulators
        num_sums: Dict[str, float] = {}
        w_sums: Dict[str, float] = {}

        # Global support metrics (for density, QA, etc.)
        total_length: float = 0.0
        total_mass: float = 0.0  # length * density
        sample_count: int = 0

        def flush_composite(end_depth: float, is_partial: bool = False):
            nonlocal num_sums, w_sums, total_length, total_mass, sample_count, comp_start

            if not w_sums:
                # nothing accumulated
                comp_start = end_depth
                return

            grades: Dict[str, float] = {}
            for k, num in num_sums.items():
                w = w_sums.get(k, 0.0)
                if w > 0:
                    grades[k] = num / w

            # Composite density from mass / length if available
            if total_length > 0 and total_mass > 0:
                grades["density"] = total_mass / total_length

            metadata = {
                "sample_count": sample_count,
                "support": end_depth - comp_start,
                "is_partial": is_partial,
                "method": "fixed_length",
                "weighting": cfg.weighting_mode.value,
                "element_weights": dict(w_sums),
                "total_length": total_length,
                "total_mass": total_mass,
            }

            composites.append(
                Composite(
                    hole_id=hole_id,
                    from_depth=comp_start,
                    to_depth=end_depth,
                    grades=grades,
                    metadata=metadata,
                )
            )

            # Reset accumulators
            num_sums = {}
            w_sums = {}
            total_length = 0.0
            total_mass = 0.0
            sample_count = 0
            comp_start = end_depth

        prev_iv: Optional[Interval] = None

        for iv_idx, iv in enumerate(intervals):
            iv_from = iv.from_depth
            iv_to = iv.to_depth
            curr_pos = iv_from

            while curr_pos < iv_to:
                remaining_comp = comp_end_target - curr_pos
                remaining_iv = iv_to - curr_pos

                if remaining_comp <= 1e-8:
                    # Close composite exactly at target length
                    flush_composite(comp_end_target, is_partial=False)
                    comp_end_target = comp_start + comp_len
                    continue

                take_len = min(remaining_comp, remaining_iv)
                if take_len <= 0:
                    break

                # Weight for this slice
                w_slice = _interval_weight(iv, take_len, cfg)

                # Accumulate grades (per-element sums and weights)
                for k, v in iv.grades.items():
                    grade_val = _get_grade_value(v, cfg)
                    if grade_val is None:
                        continue  # Skip weight accumulation (Partial sampling)
                    num_sums[k] = num_sums.get(k, 0.0) + grade_val * w_slice
                    w_sums[k] = w_sums.get(k, 0.0) + w_slice

                # Support metrics
                total_length += take_len
                if iv.density is not None:
                    total_mass += take_len * iv.density

                sample_count += 1
                curr_pos += take_len

                # If we hit the composite target, flush
                if abs(curr_pos - comp_end_target) <= 1e-8:
                    flush_composite(comp_end_target, is_partial=False)
                    comp_end_target = comp_start + comp_len

            # BREAK LOGIC between intervals
            if iv_idx < len(intervals) - 1:
                next_iv = intervals[iv_idx + 1]

                if _is_break(iv, next_iv, cfg):
                    if cfg.break_mode == BreakMode.HARD:
                        if w_sums:
                            flush_composite(curr_pos, is_partial=True)
                        comp_start = next_iv.from_depth
                        comp_end_target = comp_start + comp_len
                    elif cfg.break_mode == BreakMode.SOFT:
                        distance_to_boundary = abs(comp_end_target - next_iv.from_depth)
                        if distance_to_boundary <= cfg.soft_break_threshold:
                            if w_sums:
                                flush_composite(next_iv.from_depth, is_partial=True)
                            comp_start = next_iv.from_depth
                            comp_end_target = comp_start + comp_len

            prev_iv = iv

        # Handle remainder / final partial composite
        if w_sums:
            # Remaining support relative to nominal target
            final_support = comp_end_target - comp_start
            is_partial = final_support > 0 and final_support < comp_len - 1e-6

            if not is_partial:
                flush_composite(comp_end_target, is_partial=False)
            else:
                # Build an ephemeral "partial composite" view from current accumulators
                partial_num = dict(num_sums)
                partial_w = dict(w_sums)
                partial_length = total_length
                partial_mass = total_mass

                if cfg.partial_strategy == PartialStrategy.DISCARD:
                    # Drop partial contribution
                    pass

                elif cfg.partial_strategy == PartialStrategy.KEEP:
                    flush_composite(comp_start + partial_length, is_partial=True)

                elif cfg.partial_strategy == PartialStrategy.MERGE:
                    if composites:
                        prev_comp = composites[-1]
                        prev_weights: Dict[str, float] = prev_comp.metadata.get("element_weights", {})

                        # Merge per-element grades using their own weights
                        for k in set(prev_comp.grades.keys()) | set(partial_num.keys()):
                            w_prev = prev_weights.get(k, 0.0)
                            w_part = partial_w.get(k, 0.0)
                            if w_prev == 0 and w_part == 0:
                                continue

                            g_prev = prev_comp.grades.get(k)
                            g_part = None
                            if k in partial_num and partial_w.get(k, 0.0) > 0:
                                g_part = partial_num[k] / partial_w[k]

                            num_prev = g_prev * w_prev if g_prev is not None else 0.0
                            num_part = g_part * w_part if g_part is not None else 0.0
                            w_tot = w_prev + w_part

                            if w_tot > 0:
                                prev_comp.grades[k] = (num_prev + num_part) / w_tot
                                prev_weights[k] = w_tot

                        # Merge support metrics for density
                        prev_length = float(prev_comp.metadata.get("total_length", prev_comp.length))
                        prev_mass = float(prev_comp.metadata.get("total_mass", 0.0))

                        new_length = prev_length + partial_length
                        new_mass = prev_mass + partial_mass

                        prev_comp.metadata["total_length"] = new_length
                        prev_comp.metadata["total_mass"] = new_mass
                        prev_comp.metadata["element_weights"] = prev_weights
                        prev_comp.metadata["support"] = new_length
                        prev_comp.metadata["merged_partial"] = True

                        if new_length > 0 and new_mass > 0:
                            prev_comp.grades["density"] = new_mass / new_length
                    else:
                        # No previous composite to merge into → just keep
                        flush_composite(comp_start + partial_length, is_partial=True)

                elif cfg.partial_strategy == PartialStrategy.AUTO:
                    frac = partial_length / comp_len if comp_len > 0 else 0.0
                    if frac >= cfg.auto_partial_fraction:
                        flush_composite(comp_start + partial_length, is_partial=True)
                    else:
                        if composites:
                            prev_comp = composites[-1]
                            prev_weights: Dict[str, float] = prev_comp.metadata.get("element_weights", {})

                            for k in set(prev_comp.grades.keys()) | set(partial_num.keys()):
                                w_prev = prev_weights.get(k, 0.0)
                                w_part = partial_w.get(k, 0.0)
                                if w_prev == 0 and w_part == 0:
                                    continue

                                g_prev = prev_comp.grades.get(k)
                                g_part = None
                                if k in partial_num and partial_w.get(k, 0.0) > 0:
                                    g_part = partial_num[k] / partial_w[k]

                                num_prev = g_prev * w_prev if g_prev is not None else 0.0
                                num_part = g_part * w_part if g_part is not None else 0.0
                                w_tot = w_prev + w_part

                                if w_tot > 0:
                                    prev_comp.grades[k] = (num_prev + num_part) / w_tot
                                    prev_weights[k] = w_tot

                            prev_length = float(prev_comp.metadata.get("total_length", prev_comp.length))
                            prev_mass = float(prev_comp.metadata.get("total_mass", 0.0))
                            new_length = prev_length + partial_length
                            new_mass = prev_mass + partial_mass

                            prev_comp.metadata["total_length"] = new_length
                            prev_comp.metadata["total_mass"] = new_mass
                            prev_comp.metadata["element_weights"] = prev_weights
                            prev_comp.metadata["support"] = new_length
                            prev_comp.metadata["merged_partial_auto"] = True

                            if new_length > 0 and new_mass > 0:
                                prev_comp.grades["density"] = new_mass / new_length
                        else:
                            flush_composite(comp_start + partial_length, is_partial=True)

        return composites

    # ---------------------------------------------------------------------
    # 2) INDICATOR COMPOSITING (re-uses fixed-length engine)
    # ---------------------------------------------------------------------

    def _composite_indicator(self,
                             intervals: Iterable[Interval],
                             cfg: CompositeConfig) -> List[Composite]:
        if cfg.cutoff_field is None or cfg.cutoff_grade is None:
            raise ValueError("Indicator compositing requires cutoff_field and cutoff_grade")

        # Fix #3: Rename output column to avoid confusion
        new_field_name = f"{cfg.cutoff_field}_Indicator"

        ind_intervals: List[Interval] = []
        for iv in intervals:
            g = iv.grades.get(cfg.cutoff_field)
            ind_val: Optional[float]
            if g is None:
                ind_val = None
            else:
                ind_val = 1.0 if g >= cfg.cutoff_grade else 0.0

            # Fix #3: Use renamed field and clear other grades to avoid confusion
            new_grades = {new_field_name: ind_val}

            ind_intervals.append(
                Interval(
                    hole_id=iv.hole_id,
                    from_depth=iv.from_depth,
                    to_depth=iv.to_depth,
                    grades=new_grades,
                    lith=iv.lith,
                    domain=iv.domain,
                    sample_type=iv.sample_type,
                    density=iv.density,
                    recovery=iv.recovery,
                    flags=dict(iv.flags),
                )
            )

        comps = self._composite_fixed_length(ind_intervals, cfg)

        for c in comps:
            ind = c.grades.get(new_field_name)
            c.metadata["indicator_probability"] = ind

        return comps

    # ---------------------------------------------------------------------
    # 3) OTHER METHODS — STRUCTURED STUBS (TO BE IMPLEMENTED)
    # ---------------------------------------------------------------------

    def _composite_equal_mass(self,
                              intervals: Iterable[Interval],
                              cfg: CompositeConfig) -> List[Composite]:
        """
        Equal-mass compositing.

        - target_mass is specified in cfg.target_mass (tonnes per composite)
        - weighting must be MASS or DENSITY (mass proxy)
        - density is mandatory; missing density will raise
        - composites are variable-length along depth, constant mass per composite
        """

        if cfg.target_mass is None or cfg.target_mass <= 0:
            raise ValueError("target_mass must be > 0 for EQUAL_MASS compositing")

        if cfg.weighting_mode not in (WeightingMode.MASS, WeightingMode.DENSITY):
            raise ValueError(
                "EQUAL_MASS compositing requires weighting_mode = MASS or DENSITY "
                f"(got {cfg.weighting_mode})."
            )

        holes = _group_by_hole(intervals)
        all_composites: List[Composite] = []

        for hole_id, hole_intervals in holes.items():
            all_composites.extend(
                self._composite_equal_mass_single_hole(hole_id, hole_intervals, cfg)
            )

        return all_composites

    def _composite_equal_mass_single_hole(self,
                                          hole_id: str,
                                          intervals: List[Interval],
                                          cfg: CompositeConfig) -> List[Composite]:
        composites: List[Composite] = []
        if not intervals:
            return composites

        target_mass = cfg.target_mass
        comp_start = intervals[0].from_depth

        # Per-element accumulators
        num_sums: Dict[str, float] = {}
        w_sums: Dict[str, float] = {}

        # Support metrics
        total_length: float = 0.0
        total_mass: float = 0.0
        sample_count: int = 0
        current_mass: float = 0.0

        def flush_composite(end_depth: float, is_partial: bool = False):
            nonlocal num_sums, w_sums, total_length, total_mass, sample_count, comp_start, current_mass

            if current_mass <= 0 or not w_sums:
                comp_start = end_depth
                current_mass = 0.0
                return

            grades: Dict[str, float] = {}
            for k, num in num_sums.items():
                w = w_sums.get(k, 0.0)
                if w > 0:
                    grades[k] = num / w

            if total_length > 0 and total_mass > 0:
                grades["density"] = total_mass / total_length

            metadata = {
                "sample_count": sample_count,
                "support": end_depth - comp_start,
                "is_partial": is_partial,
                "method": "equal_mass",
                "weighting": cfg.weighting_mode.value,
                "element_weights": dict(w_sums),
                "total_length": total_length,
                "total_mass": total_mass,
                "target_mass": target_mass,
            }

            composites.append(
                Composite(
                    hole_id=hole_id,
                    from_depth=comp_start,
                    to_depth=end_depth,
                    grades=grades,
                    metadata=metadata,
                )
            )

            # reset
            num_sums = {}
            w_sums = {}
            total_length = 0.0
            total_mass = 0.0
            sample_count = 0
            comp_start = end_depth
            current_mass = 0.0

        prev_iv: Optional[Interval] = None

        for iv_idx, iv in enumerate(intervals):
            if iv.density is None:
                raise ValueError(
                    f"EQUAL_MASS compositing requires density; missing on "
                    f"hole {iv.hole_id} interval {iv.from_depth}-{iv.to_depth}."
                )

            iv_from = iv.from_depth
            iv_to = iv.to_depth
            curr_pos = iv_from
            rho = iv.density

            while curr_pos < iv_to:
                remaining_interval = iv_to - curr_pos
                if remaining_interval <= 1e-8:
                    break

                remaining_to_target = target_mass - current_mass

                if remaining_to_target <= 1e-8:
                    # we already hit target mass; flush at current position
                    flush_composite(curr_pos, is_partial=False)
                    continue

                # Mass if we take whole remaining interval
                full_slice_mass = remaining_interval * rho

                if full_slice_mass <= remaining_to_target + 1e-8:
                    # take whole remaining interval into current composite
                    take_len = remaining_interval
                    mass_slice = full_slice_mass
                else:
                    # only take enough length to reach target_mass
                    take_len = remaining_to_target / rho
                    if take_len <= 1e-8:
                        # numerical safety; force small step
                        take_len = min(remaining_interval, remaining_to_target / rho)
                    if take_len > remaining_interval + 1e-8:
                        take_len = remaining_interval
                    mass_slice = take_len * rho

                # weight slice = mass
                w_slice = mass_slice

                for k, v in iv.grades.items():
                    if v is None:
                        continue
                    num_sums[k] = num_sums.get(k, 0.0) + v * w_slice
                    w_sums[k] = w_sums.get(k, 0.0) + w_slice

                total_length += take_len
                total_mass += mass_slice
                sample_count += 1

                current_mass += mass_slice
                curr_pos += take_len

                # reached or slightly exceeded target mass
                if current_mass >= target_mass - 1e-8:
                    flush_composite(curr_pos, is_partial=False)

            # No special break logic here; EQUAL_MASS is primarily mass-driven.
            prev_iv = iv

        # Handle final partial composite by mass fraction
        if current_mass > 0.0 and w_sums:
            mass_fraction = current_mass / target_mass if target_mass > 0 else 0.0

            partial_num = dict(num_sums)
            partial_w = dict(w_sums)
            partial_length = total_length
            partial_mass = total_mass

            if cfg.partial_strategy == PartialStrategy.DISCARD:
                pass

            elif cfg.partial_strategy == PartialStrategy.KEEP:
                flush_composite(comp_start + partial_length, is_partial=True)

            elif cfg.partial_strategy == PartialStrategy.MERGE:
                if composites:
                    prev_comp = composites[-1]
                    prev_weights: Dict[str, float] = prev_comp.metadata.get("element_weights", {})

                    for k in set(prev_comp.grades.keys()) | set(partial_num.keys()):
                        w_prev = prev_weights.get(k, 0.0)
                        w_part = partial_w.get(k, 0.0)
                        if w_prev == 0 and w_part == 0:
                            continue

                        g_prev = prev_comp.grades.get(k)
                        g_part = None
                        if k in partial_num and partial_w.get(k, 0.0) > 0:
                            g_part = partial_num[k] / partial_w[k]

                        num_prev = g_prev * w_prev if g_prev is not None else 0.0
                        num_part = g_part * w_part if g_part is not None else 0.0
                        w_tot = w_prev + w_part

                        if w_tot > 0:
                            prev_comp.grades[k] = (num_prev + num_part) / w_tot
                            prev_weights[k] = w_tot

                    prev_length = float(prev_comp.metadata.get("total_length", prev_comp.length))
                    prev_mass = float(prev_comp.metadata.get("total_mass", 0.0))

                    new_length = prev_length + partial_length
                    new_mass = prev_mass + partial_mass

                    prev_comp.metadata["total_length"] = new_length
                    prev_comp.metadata["total_mass"] = new_mass
                    prev_comp.metadata["element_weights"] = prev_weights
                    prev_comp.metadata["support"] = new_length
                    prev_comp.metadata["merged_partial"] = True

                    if new_length > 0 and new_mass > 0:
                        prev_comp.grades["density"] = new_mass / new_length
                else:
                    flush_composite(comp_start + partial_length, is_partial=True)

            elif cfg.partial_strategy == PartialStrategy.AUTO:
                if mass_fraction >= cfg.auto_partial_fraction:
                    flush_composite(comp_start + partial_length, is_partial=True)
                else:
                    if composites:
                        prev_comp = composites[-1]
                        prev_weights: Dict[str, float] = prev_comp.metadata.get("element_weights", {})

                        for k in set(prev_comp.grades.keys()) | set(partial_num.keys()):
                            w_prev = prev_weights.get(k, 0.0)
                            w_part = partial_w.get(k, 0.0)
                            if w_prev == 0 and w_part == 0:
                                continue

                            g_prev = prev_comp.grades.get(k)
                            g_part = None
                            if k in partial_num and partial_w.get(k, 0.0) > 0:
                                g_part = partial_num[k] / partial_w[k]

                            num_prev = g_prev * w_prev if g_prev is not None else 0.0
                            num_part = g_part * w_part if g_part is not None else 0.0
                            w_tot = w_prev + w_part

                            if w_tot > 0:
                                prev_comp.grades[k] = (num_prev + num_part) / w_tot
                                prev_weights[k] = w_tot

                        prev_length = float(prev_comp.metadata.get("total_length", prev_comp.length))
                        prev_mass = float(prev_comp.metadata.get("total_mass", 0.0))

                        new_length = prev_length + partial_length
                        new_mass = prev_mass + partial_mass

                        prev_comp.metadata["total_length"] = new_length
                        prev_comp.metadata["total_mass"] = new_mass
                        prev_comp.metadata["element_weights"] = prev_weights
                        prev_comp.metadata["support"] = new_length
                        prev_comp.metadata["merged_partial_auto"] = True

                        if new_length > 0 and new_mass > 0:
                            prev_comp.grades["density"] = new_mass / new_length
                    else:
                        flush_composite(comp_start + partial_length, is_partial=True)

        return composites

    def _composite_equal_volume(self,
                                intervals: Iterable[Interval],
                                cfg: CompositeConfig) -> List[Composite]:
        """
        Equal-volume compositing.

        In 1D drillhole space:
            volume = length * borehole_area  (area is constant)
        So equal-volume == equal-length.

        Implementation:
        - reuses fixed-length compositing
        - but stores metadata["method"] = "equal_volume"
        """

        if cfg.composite_length is None or cfg.composite_length <= 0:
            raise ValueError(
                "Equal-volume compositing requires composite_length (length representing constant volume)."
            )

        # We simply call the fixed-length engine with modified metadata tag.
        comps = self._composite_fixed_length(intervals, cfg)

        for c in comps:
            c.metadata["method"] = "equal_volume"

        return comps

    def _composite_bench_aligned(self,
                                 intervals: Iterable[Interval],
                                 cfg: CompositeConfig) -> List[Composite]:
        """
        Bench-aligned compositing.

        - Aligns composite boundaries to bench intervals:
            [bench_offset + n * bench_height, bench_offset + (n+1)*bench_height]
        - Uses same weighting logic as fixed-length.
        """

        if cfg.bench_height is None or cfg.bench_height <= 0:
            raise ValueError("bench_height must be > 0 for BENCH_ALIGNED compositing")

        holes = _group_by_hole(intervals)
        all_composites: List[Composite] = []

        for hole_id, hole_intervals in holes.items():
            if not hole_intervals:
                continue

            all_composites.extend(
                self._bench_aligned_single_hole(hole_id, hole_intervals, cfg)
            )

        return all_composites

    def _bench_aligned_single_hole(self,
                                   hole_id: str,
                                   intervals: List[Interval],
                                   cfg: CompositeConfig) -> List[Composite]:
        composites: List[Composite] = []

        bh = cfg.bench_height
        offset = cfg.bench_offset or 0.0

        depth_min = intervals[0].from_depth
        depth_max = intervals[-1].to_depth

        # find first bench boundary at or below depth_min
        import math
        n0 = math.floor((depth_min - offset) / bh)
        comp_start = offset + n0 * bh
        if comp_start > depth_min + 1e-8:
            comp_start -= bh
        comp_end_target = comp_start + bh

        num_sums: Dict[str, float] = {}
        w_sums: Dict[str, float] = {}
        total_length: float = 0.0
        total_mass: float = 0.0
        sample_count: int = 0

        def flush_composite(end_depth: float):
            nonlocal num_sums, w_sums, total_length, total_mass, sample_count, comp_start

            if not w_sums:
                comp_start = end_depth
                return

            grades: Dict[str, float] = {}
            for k, num in num_sums.items():
                w = w_sums.get(k, 0.0)
                if w > 0:
                    grades[k] = num / w

            if total_length > 0 and total_mass > 0:
                grades["density"] = total_mass / total_length

            metadata = {
                "sample_count": sample_count,
                "support": end_depth - comp_start,
                "method": "bench_aligned",
                "bench_height": bh,
                "bench_offset": offset,
                "weighting": cfg.weighting_mode.value,
                "element_weights": dict(w_sums),
                "total_length": total_length,
                "total_mass": total_mass,
            }

            composites.append(
                Composite(
                    hole_id=hole_id,
                    from_depth=comp_start,
                    to_depth=end_depth,
                    grades=grades,
                    metadata=metadata,
                )
            )

            num_sums = {}
            w_sums = {}
            total_length = 0.0
            total_mass = 0.0
            sample_count = 0
            comp_start = end_depth

        for iv in intervals:
            iv_from = iv.from_depth
            iv_to = iv.to_depth
            curr_pos = iv_from

            # skip any bench intervals entirely above first sample
            while comp_end_target <= iv_from - 1e-8:
                comp_start = comp_end_target
                comp_end_target = comp_start + bh

            while curr_pos < iv_to:
                remaining_comp = comp_end_target - curr_pos
                remaining_iv = iv_to - curr_pos

                if remaining_comp <= 1e-8:
                    flush_composite(comp_end_target)
                    comp_end_target = comp_start + bh
                    continue

                take_len = min(remaining_comp, remaining_iv)
                if take_len <= 0:
                    break

                w_slice = _interval_weight(iv, take_len, cfg)

                for k, v in iv.grades.items():
                    if v is None:
                        continue
                    num_sums[k] = num_sums.get(k, 0.0) + v * w_slice
                    w_sums[k] = w_sums.get(k, 0.0) + w_slice

                total_length += take_len
                if iv.density is not None:
                    total_mass += take_len * iv.density

                sample_count += 1
                curr_pos += take_len

                if abs(curr_pos - comp_end_target) <= 1e-8:
                    flush_composite(comp_end_target)
                    comp_end_target = comp_start + bh

        # final composite (partial bench at bottom)
        if w_sums:
            flush_composite(comp_start + total_length)

        return composites

    def _composite_lithology(self,
                             intervals: Iterable[Interval],
                             cfg: CompositeConfig) -> List[Composite]:
        """Lithology compositing.

        Merge consecutive intervals with the same lithology, apply minimum
        thickness rules and merge direction. Numeric grades may be carried as
        simple length-weighted averages.
        
        M-05 FIX: Safely access config attributes with getattr() defaults.
        """
        holes = _group_by_hole(intervals)
        all_composites = []
        
        # M-05 FIX: Safely get config attributes with defaults
        min_ore_thickness = getattr(cfg, 'min_ore_thickness', None)
        min_mining_width = getattr(cfg, 'min_mining_width', None)

        for hole_id, hole_ints in holes.items():
            if not hole_ints:
                continue
            
            # 1. Merge adjacent same-lithology
            merged_segments = []
            
            current_seg = [hole_ints[0]]
            current_lith = hole_ints[0].lith
            
            for i in range(1, len(hole_ints)):
                iv = hole_ints[i]
                # Check continuity (soft break threshold optional here)
                gap = iv.from_depth - current_seg[-1].to_depth
                
                if iv.lith == current_lith and gap < 1e-3:
                    current_seg.append(iv)
                else:
                    merged_segments.append(current_seg)
                    current_seg = [iv]
                    current_lith = iv.lith
            merged_segments.append(current_seg)

            # 2. Create Composites from Segments
            for seg in merged_segments:
                start_depth = seg[0].from_depth
                end_depth = seg[-1].to_depth
                length = end_depth - start_depth
                
                # Filter by Min Ore Thickness (if configured)
                # M-05 FIX: Use safely retrieved attribute
                if min_ore_thickness is not None:
                    if length < min_ore_thickness:
                        continue  # Discard short segments
                
                # Filter by Min Mining Width (if configured)
                if min_mining_width is not None:
                    if length < min_mining_width:
                        continue  # Discard segments below mining width

                # Calculate Weighted Grades
                num_sums = {}
                w_sums = {}
                
                for iv in seg:
                    iv_len = iv.length
                    w = _interval_weight(iv, iv_len, cfg)
                    for k, v in iv.grades.items():
                        grade_val = _get_grade_value(v, cfg)
                        if grade_val is None:
                            continue  # Skip weight accumulation (Partial sampling)
                        num_sums[k] = num_sums.get(k, 0.0) + grade_val * w
                        w_sums[k] = w_sums.get(k, 0.0) + w
                
                final_grades = {}
                for k, num in num_sums.items():
                    if w_sums[k] > 0:
                        final_grades[k] = num / w_sums[k]
                
                meta = {
                    "method": "lithology",
                    "lith_code": seg[0].lith,
                    "sample_count": len(seg),
                    "min_ore_thickness_applied": min_ore_thickness,
                    "min_mining_width_applied": min_mining_width,
                }
                
                all_composites.append(Composite(
                    hole_id=hole_id,
                    from_depth=start_depth,
                    to_depth=end_depth,
                    grades=final_grades,
                    metadata=meta
                ))

        return all_composites

    def _composite_true_thickness(self,
                                  intervals: Iterable[Interval],
                                  cfg: CompositeConfig) -> List[Composite]:
        """
        True-thickness compositing.

        - Requires Interval.true_thickness to be populated.
        - cfg.composite_length is interpreted as target TRUE THICKNESS per composite.
        - We walk along depth, but accumulation and weighting are based on true thickness.
        """

        if cfg.composite_length is None or cfg.composite_length <= 0:
            raise ValueError("TRUE_THICKNESS compositing requires composite_length (true thickness target).")

        holes = _group_by_hole(intervals)
        all_composites: List[Composite] = []

        for hole_id, hole_intervals in holes.items():
            if not hole_intervals:
                continue

            all_composites.extend(
                self._composite_true_thickness_single_hole(hole_id, hole_intervals, cfg)
            )

        return all_composites

    def _composite_true_thickness_single_hole(self,
                                              hole_id: str,
                                              intervals: List[Interval],
                                              cfg: CompositeConfig) -> List[Composite]:
        composites: List[Composite] = []
        target_tt = cfg.composite_length

        comp_start = intervals[0].from_depth

        num_sums: Dict[str, float] = {}
        w_sums: Dict[str, float] = {}
        total_length: float = 0.0          # downhole length
        total_mass: float = 0.0
        total_true_thickness: float = 0.0  # accumulated true thickness
        sample_count: int = 0
        current_tt: float = 0.0            # true-thickness accumulated in current composite

        def flush_composite(end_depth: float, is_partial: bool = False):
            nonlocal num_sums, w_sums, total_length, total_mass, \
                     total_true_thickness, sample_count, comp_start, current_tt

            if current_tt <= 0 or not w_sums:
                comp_start = end_depth
                current_tt = 0.0
                return

            grades: Dict[str, float] = {}
            for k, num in num_sums.items():
                w = w_sums.get(k, 0.0)
                if w > 0:
                    grades[k] = num / w

            if total_length > 0 and total_mass > 0:
                grades["density"] = total_mass / total_length

            metadata = {
                "sample_count": sample_count,
                "support_downhole": end_depth - comp_start,
                "support_true_thickness": total_true_thickness,
                "is_partial": is_partial,
                "method": "true_thickness",
                "weighting": cfg.weighting_mode.value,
                "element_weights": dict(w_sums),
                "total_length": total_length,
                "total_mass": total_mass,
            }

            composites.append(
                Composite(
                    hole_id=hole_id,
                    from_depth=comp_start,
                    to_depth=end_depth,
                    grades=grades,
                    metadata=metadata,
                )
            )

            num_sums = {}
            w_sums = {}
            total_length = 0.0
            total_mass = 0.0
            total_true_thickness = 0.0
            sample_count = 0
            comp_start = end_depth
            current_tt = 0.0

        for iv in intervals:
            if iv.true_thickness is None or iv.length <= 0:
                raise ValueError(
                    f"TRUE_THICKNESS compositing requires true_thickness for "
                    f"hole {iv.hole_id} interval {iv.from_depth}-{iv.to_depth}."
                )

            scale_tt = iv.true_thickness / iv.length  # factor to convert downhole length -> true thickness

            iv_from = iv.from_depth
            iv_to = iv.to_depth
            curr_pos = iv_from

            while curr_pos < iv_to:
                remaining_downhole = iv_to - curr_pos
                if remaining_downhole <= 1e-8:
                    break

                # True thickness available if we take whole remaining interval
                tt_full = remaining_downhole * scale_tt
                remaining_to_target = target_tt - current_tt

                if remaining_to_target <= 1e-8:
                    flush_composite(curr_pos, is_partial=False)
                    continue

                if tt_full <= remaining_to_target + 1e-8:
                    take_len = remaining_downhole
                    tt_slice = tt_full
                else:
                    # take only enough downhole to reach target true thickness
                    take_len = remaining_to_target / scale_tt
                    if take_len > remaining_downhole + 1e-8:
                        take_len = remaining_downhole
                    tt_slice = take_len * scale_tt

                # weight by true-thickness slice (this is your support measure)
                w_slice = _interval_weight(iv, tt_slice, cfg)

                for k, v in iv.grades.items():
                    if v is None:
                        continue
                    num_sums[k] = num_sums.get(k, 0.0) + v * w_slice
                    w_sums[k] = w_sums.get(k, 0.0) + w_slice

                total_length += take_len
                total_true_thickness += tt_slice
                if iv.density is not None:
                    total_mass += take_len * iv.density

                sample_count += 1
                current_tt += tt_slice
                curr_pos += take_len

                if current_tt >= target_tt - 1e-8:
                    flush_composite(curr_pos, is_partial=False)

        # final partial (by true thickness fraction)
        if current_tt > 0.0 and w_sums:
            frac = current_tt / target_tt if target_tt > 0 else 0.0
            partial_length = total_length
            partial_tt = total_true_thickness
            partial_num = dict(num_sums)
            partial_w = dict(w_sums)
            partial_mass = total_mass

            if cfg.partial_strategy == PartialStrategy.DISCARD:
                pass

            elif cfg.partial_strategy == PartialStrategy.KEEP:
                flush_composite(comp_start + partial_length, is_partial=True)

            elif cfg.partial_strategy in (PartialStrategy.MERGE, PartialStrategy.AUTO):
                # AUTO: merge if fraction below threshold
                if cfg.partial_strategy == PartialStrategy.AUTO and frac >= cfg.auto_partial_fraction:
                    flush_composite(comp_start + partial_length, is_partial=True)
                else:
                    if composites:
                        prev_comp = composites[-1]
                        prev_weights = prev_comp.metadata.get("element_weights", {})

                        for k in set(prev_comp.grades.keys()) | set(partial_num.keys()):
                            w_prev = prev_weights.get(k, 0.0)
                            w_part = partial_w.get(k, 0.0)
                            if w_prev == 0 and w_part == 0:
                                continue

                            g_prev = prev_comp.grades.get(k)
                            g_part = None
                            if k in partial_num and partial_w.get(k, 0.0) > 0:
                                g_part = partial_num[k] / partial_w[k]

                            num_prev = g_prev * w_prev if g_prev is not None else 0.0
                            num_part = g_part * w_part if g_part is not None else 0.0
                            w_tot = w_prev + w_part

                            if w_tot > 0:
                                prev_comp.grades[k] = (num_prev + num_part) / w_tot
                                prev_weights[k] = w_tot

                        prev_len = float(prev_comp.metadata.get("total_length", prev_comp.length))
                        prev_mass = float(prev_comp.metadata.get("total_mass", 0.0))
                        prev_tt = float(prev_comp.metadata.get("support_true_thickness", 0.0))

                        new_len = prev_len + partial_length
                        new_mass = prev_mass + partial_mass
                        new_tt = prev_tt + partial_tt

                        prev_comp.metadata["total_length"] = new_len
                        prev_comp.metadata["total_mass"] = new_mass
                        prev_comp.metadata["support_true_thickness"] = new_tt
                        prev_comp.metadata["element_weights"] = prev_weights
                        prev_comp.metadata["support_downhole"] = new_len
                        prev_comp.metadata["merged_partial"] = True

                        if new_len > 0 and new_mass > 0:
                            prev_comp.grades["density"] = new_mass / new_len
                    else:
                        flush_composite(comp_start + partial_length, is_partial=True)

        return composites

    def _composite_rolling_window(self,
                                   intervals: Iterable[Interval],
                                   cfg: CompositeConfig) -> List[Composite]:
        """
        Rolling-window compositing.

        - Window length = cfg.rolling_window_length
        - Step size = cfg.rolling_step
        - Produces overlapping composites (continuous curve)
        """

        if cfg.rolling_window_length is None or cfg.rolling_window_length <= 0:
            raise ValueError("rolling_window_length must be > 0")

        if cfg.rolling_step is None or cfg.rolling_step <= 0:
            raise ValueError("rolling_step must be > 0")

        holes = _group_by_hole(intervals)
        all_composites: List[Composite] = []

        for hole_id, hole_intervals in holes.items():
            if not hole_intervals:
                continue

            # Define sliding positions
            depth_min = hole_intervals[0].from_depth
            depth_max = hole_intervals[-1].to_depth

            pos = depth_min
            L = cfg.rolling_window_length
            step = cfg.rolling_step

            while pos + L <= depth_max + 1e-6:
                win_start = pos
                win_end = pos + L

                comp = self._rolling_window_single(
                    hole_id, hole_intervals, win_start, win_end, cfg
                )

                if comp is not None:
                    all_composites.append(comp)

                pos += step

        return all_composites

    def _rolling_window_single(self,
                               hole_id: str,
                               intervals: List[Interval],
                               win_start: float,
                               win_end: float,
                               cfg: CompositeConfig) -> Optional[Composite]:
        num_sums: Dict[str, float] = {}
        w_sums: Dict[str, float] = {}
        total_length = 0.0
        total_mass = 0.0
        sample_count = 0

        for iv in intervals:
            if iv.to_depth <= win_start or iv.from_depth >= win_end:
                continue

            iv_from = max(iv.from_depth, win_start)
            iv_to = min(iv.to_depth, win_end)
            length = iv_to - iv_from
            if length <= 1e-8:
                continue

            w_slice = _interval_weight(iv, length, cfg)

            for k, v in iv.grades.items():
                grade_val = _get_grade_value(v, cfg)
                if grade_val is None:
                    continue  # Skip weight accumulation (Partial sampling)
                num_sums[k] = num_sums.get(k, 0.0) + grade_val * w_slice
                w_sums[k] = w_sums.get(k, 0.0) + w_slice

            total_length += length
            if iv.density is not None:
                total_mass += length * iv.density

            sample_count += 1

        if not w_sums:
            return None

        grades: Dict[str, float] = {}
        for k, num in num_sums.items():
            w = w_sums.get(k, 0.0)
            if w > 0:
                grades[k] = num / w

        if total_length > 0 and total_mass > 0:
            grades["density"] = total_mass / total_length

        metadata = {
            "method": "rolling_window",
            "support": win_end - win_start,
            "sample_count": sample_count,
            "window_start": win_start,
            "window_end": win_end,
            "weighting": cfg.weighting_mode.value,
            "element_weights": dict(w_sums),
            "total_length": total_length,
            "total_mass": total_mass,
        }

        return Composite(
            hole_id=hole_id,
            from_depth=win_start,
            to_depth=win_end,
            grades=grades,
            metadata=metadata,
        )

    def _composite_economic(self,
                            intervals: Iterable[Interval],
                            cfg: CompositeConfig) -> List[Composite]:
        """
        Economic compositing (ore/waste envelopes).
        
        Economic compositing classifies intervals as "ore" or "waste" based on a cut-off grade,
        then builds ore composites by sequentially adding waste-ore pairs, subject to constraints.
        
        This is NOT about "mining constraints" - it's about building reasonable ore envelopes
        from grade data, considering dilution and composite length requirements.
        
        Process:
        1. Classify intervals as ore/waste based on cutoff_grade
        2. Concatenate adjacent intervals on the same side of cutoff
        3. Build ore composites sequentially by adding waste-ore pairs
        4. Apply constraints: min_ore_composite_length, max_included_waste, max_consecutive_waste
        5. Apply dilution rules (Basic/Advanced/Advanced+)
        6. Optionally apply two-pass compositing
        """

        if cfg.cutoff_field is None or cfg.cutoff_grade is None:
            raise ValueError("Economic compositing requires cutoff_field and cutoff_grade")

        # Set defaults for dilution rule
        dilution_rule = cfg.dilution_rule or EconomicDilutionRule.BASIC
        
        # Two-pass compositing: run once, then use results as input for second pass
        if cfg.composite_twice:
            # First pass
            first_pass = self._composite_economic_single_pass(intervals, cfg)
            # Convert composites back to intervals for second pass
            # (This is a simplification - in practice you'd use the ore/waste classification)
            return self._composite_economic_single_pass(intervals, cfg)
        else:
            return self._composite_economic_single_pass(intervals, cfg)
    
    def _composite_economic_single_pass(self,
                                       intervals: Iterable[Interval],
                                       cfg: CompositeConfig) -> List[Composite]:
        """Single pass of economic compositing."""
        
        holes = _group_by_hole(intervals)
        all_composites: List[Composite] = []
        
        dilution_rule = cfg.dilution_rule or EconomicDilutionRule.BASIC
        
        for hole_id, hole_intervals in holes.items():
            if not hole_intervals:
                continue
            
            composites = self._composite_economic_single_hole(
                hole_id, hole_intervals, cfg, dilution_rule
            )
            all_composites.extend(composites)
        
        return all_composites
    
    def _composite_economic_single_hole(self,
                                       hole_id: str,
                                       intervals: List[Interval],
                                       cfg: CompositeConfig,
                                       dilution_rule: EconomicDilutionRule) -> List[Composite]:
        """
        Economic compositing for a single hole.
        
        Builds ore composites by sequentially adding waste-ore pairs, testing against
        constraints at each step.
        """
        
        def classify_iv(iv: Interval) -> bool:
            """Classify interval as ore (True) or waste (False)."""
            g = iv.grades.get(cfg.cutoff_field)
            if g is None:
                return False
            return g >= cfg.cutoff_grade
        
        # Step 1: Classify all intervals as ore/waste
        classified_intervals: List[Dict[str, Any]] = []
        for iv in intervals:
            is_ore = classify_iv(iv)
            classified_intervals.append({
                "interval": iv,
                "is_ore": is_ore,
                "from": iv.from_depth,
                "to": iv.to_depth,
                "length": iv.length,
            })
        
        # Step 2: Concatenate adjacent intervals on same side of cutoff
        segments: List[Dict[str, Any]] = []
        current_flag: Optional[bool] = None
        seg_start: Optional[float] = None
        seg_end: Optional[float] = None
        
        for item in classified_intervals:
            ore_flag = item["is_ore"]
            if current_flag is None:
                current_flag = ore_flag
                seg_start = item["from"]
                seg_end = item["to"]
            else:
                if ore_flag == current_flag:
                    seg_end = item["to"]
                else:
                    segments.append({
                        "start": seg_start,
                        "end": seg_end,
                        "is_ore": current_flag,
                        "intervals": [iv for iv in classified_intervals 
                                     if seg_start <= iv["from"] < seg_end or 
                                        seg_start < iv["to"] <= seg_end]
                    })
                    current_flag = ore_flag
                    seg_start = item["from"]
                    seg_end = item["to"]
        
        if seg_start is not None and seg_end is not None:
            segments.append({
                "start": seg_start,
                "end": seg_end,
                "is_ore": current_flag,
                "intervals": [iv for iv in classified_intervals 
                             if seg_start <= iv["from"] < seg_end or 
                                seg_start < iv["to"] <= seg_end]
            })
        
        if not segments:
            return []
        
        # Step 3: Build ore composites sequentially
        composites: List[Composite] = []
        i = 0
        
        while i < len(segments):
            seg = segments[i]
            
            if not seg["is_ore"]:
                # Waste segment - add as waste composite or accumulate
                waste_comp = self._build_segment_composite(
                    hole_id, intervals, seg["start"], seg["end"], cfg, "WASTE"
                )
                if waste_comp:
                    composites.append(waste_comp)
                i += 1
                continue
            
            # Start building an ore composite candidate
            ore_candidate_start = seg["start"]
            ore_candidate_end = seg["end"]
            ore_candidate_segments = [seg]
            total_waste_length = 0.0
            max_consecutive_waste_in_candidate = 0.0
            current_consecutive_waste = 0.0
            
            # Try to extend the ore composite by adding waste-ore pairs
            j = i + 1
            while j < len(segments) - 1:  # Need at least one more segment after j
                waste_seg = segments[j]
                ore_seg = segments[j + 1] if j + 1 < len(segments) else None
                
                if not waste_seg["is_ore"] and ore_seg and ore_seg["is_ore"]:
                    # We have a waste-ore pair to test
                    waste_len = waste_seg["end"] - waste_seg["start"]
                    test_waste_total = total_waste_length + waste_len
                    test_consecutive = max(max_consecutive_waste_in_candidate, waste_len)
                    
                    # Test constraints
                    if cfg.max_included_waste is not None:
                        if test_waste_total > cfg.max_included_waste + 1e-8:
                            break  # Would exceed max included waste
                    
                    if cfg.max_consecutive_waste is not None:
                        if test_consecutive > cfg.max_consecutive_waste + 1e-8:
                            break  # Would exceed max consecutive waste
                    
                    # Test dilution rule
                    if not self._test_dilution_rule(
                        ore_candidate_segments, waste_seg, ore_seg, intervals,
                        cfg, dilution_rule
                    ):
                        break  # Dilution rule rejects this addition
                    
                    # Accept the waste-ore pair
                    ore_candidate_segments.append(waste_seg)
                    ore_candidate_segments.append(ore_seg)
                    ore_candidate_end = ore_seg["end"]
                    total_waste_length = test_waste_total
                    max_consecutive_waste_in_candidate = test_consecutive
                    j += 2
                else:
                    break
            
            # Test if candidate meets minimum requirements
            candidate_length = ore_candidate_end - ore_candidate_start
            candidate_meets_min_length = (
                cfg.min_ore_composite_length is None or
                candidate_length >= cfg.min_ore_composite_length - 1e-8
            )
            
            # Calculate linear grade for candidate
            candidate_linear_grade = self._calculate_linear_grade(
                hole_id, intervals, ore_candidate_start, ore_candidate_end, cfg
            )
            candidate_meets_min_linear_grade = (
                not cfg.keep_short_high_grade or
                cfg.min_linear_grade is None or
                candidate_linear_grade >= cfg.min_linear_grade - 1e-8
            )
            
            if candidate_meets_min_length or candidate_meets_min_linear_grade:
                # Accept as ore composite
                ore_comp = self._build_segment_composite(
                    hole_id, intervals, ore_candidate_start, ore_candidate_end, cfg, "ORE"
                )
                if ore_comp:
                    # Add dilution metadata
                    ore_comp.metadata["dilution_waste_length"] = total_waste_length
                    ore_comp.metadata["dilution_max_consecutive_waste"] = max_consecutive_waste_in_candidate
                    ore_comp.metadata["linear_grade"] = candidate_linear_grade
                    composites.append(ore_comp)
                i = j
            else:
                # Reject candidate - convert to waste
                waste_comp = self._build_segment_composite(
                    hole_id, intervals, ore_candidate_start, ore_candidate_end, cfg, "WASTE"
                )
                if waste_comp:
                    composites.append(waste_comp)
                i += 1
        
        # Step 4: Apply min_waste_composite_length for Advanced/Advanced+
        if (dilution_rule in (EconomicDilutionRule.ADVANCED, EconomicDilutionRule.ADVANCED_PLUS) and
            cfg.min_waste_composite_length is not None):
            composites = self._expand_waste_composites(
                composites, intervals, cfg, hole_id
            )
        
        return composites
    
    def _test_dilution_rule(self,
                           ore_segments: List[Dict[str, Any]],
                           waste_seg: Dict[str, Any],
                           ore_seg: Dict[str, Any],
                           intervals: List[Interval],
                           cfg: CompositeConfig,
                           dilution_rule: EconomicDilutionRule) -> bool:
        """
        Test if adding waste-ore pair passes dilution rule.
        
        Returns True if the addition is acceptable, False otherwise.
        
        Rules:
        - BASIC: Simple length-weighted average of candidate + waste-ore pair must be >= cutoff
        - ADVANCED: More conservative - checks that average grade stays above cutoff with stricter controls
        - ADVANCED_PLUS: Also checks for waste-ore-waste patterns that would be below cutoff
        """
        if cfg.cutoff_field is None or cfg.cutoff_grade is None:
            return True  # No cutoff defined, accept all
        
        # Calculate current candidate grade (length-weighted average)
        candidate_start = ore_segments[0]["start"]
        candidate_end = ore_segments[-1]["end"]
        candidate_grade, candidate_length = self._calculate_weighted_grade(
            intervals, candidate_start, candidate_end, cfg.cutoff_field, cfg
        )
        
        # Calculate waste-ore pair grade
        waste_grade, waste_length = self._calculate_weighted_grade(
            intervals, waste_seg["start"], waste_seg["end"], cfg.cutoff_field, cfg
        )
        ore_grade, ore_length = self._calculate_weighted_grade(
            intervals, ore_seg["start"], ore_seg["end"], cfg.cutoff_field, cfg
        )
        
        waste_ore_length = waste_length + ore_length
        waste_ore_linear_grade = waste_grade * waste_length + ore_grade * ore_length
        # Fix #4: Use math.isclose for robust floating point comparison
        waste_ore_avg_grade = waste_ore_linear_grade / waste_ore_length if not math.isclose(waste_ore_length, 0.0, abs_tol=1e-8) else 0.0
        
        # Combined candidate + waste-ore pair
        total_length = candidate_length + waste_ore_length
        total_linear_grade = candidate_grade * candidate_length + waste_ore_linear_grade
        combined_avg_grade = total_linear_grade / total_length if total_length > 1e-8 else 0.0
        
        if dilution_rule == EconomicDilutionRule.BASIC:
            # Basic: simple length-weighted average test
            # Accept if combined average grade is >= cutoff
            return combined_avg_grade >= cfg.cutoff_grade - 1e-8
        
        elif dilution_rule == EconomicDilutionRule.ADVANCED:
            # Advanced: more conservative
            # 1. Combined average must be >= cutoff
            # 2. The waste-ore pair itself should not dilute too much
            #    (i.e., the ore segment in the pair should be significant)
            if combined_avg_grade < cfg.cutoff_grade - 1e-8:
                return False
            
            # Additional check: ore segment in the pair should be substantial
            # If ore length is very small relative to waste, reject
            # Fix #4: Use math.isclose for robust floating point comparison
            if not math.isclose(waste_ore_length, 0.0, abs_tol=1e-8):
                ore_fraction = ore_length / waste_ore_length
                # Require at least 30% ore in the waste-ore pair for Advanced
                if ore_fraction < 0.30:
                    return False
            
            return True
        
        elif dilution_rule == EconomicDilutionRule.ADVANCED_PLUS:
            # Advanced+: most conservative
            # 1. Combined average must be >= cutoff
            # 2. Check for waste-ore-waste patterns that would be below cutoff
            
            if combined_avg_grade < cfg.cutoff_grade - 1e-8:
                return False
            
            # Check if adding this waste-ore pair creates a waste-ore-waste pattern
            # that would be below cutoff. This requires looking at the structure.
            # For now, we check if the waste segment alone is too large relative to ore
            # Fix #4: Use math.isclose for robust floating point comparison
            if not math.isclose(waste_ore_length, 0.0, abs_tol=1e-8):
                ore_fraction = ore_length / waste_ore_length
                # Require at least 40% ore in the waste-ore pair for Advanced+
                if ore_fraction < 0.40:
                    return False
            
            # Additional check: if there's a previous waste segment in the candidate,
            # check if waste-ore-waste pattern would be below cutoff
            # This ensures we don't create internal waste-ore-waste segments below cutoff
            if len(ore_segments) > 1:
                # Find the last waste segment in the candidate (if any)
                for seg_idx in range(len(ore_segments) - 1, -1, -1):
                    prev_seg = ore_segments[seg_idx]
                    if not prev_seg.get("is_ore", True):
                        # We have a waste segment in the candidate
                        # Check if adding this waste-ore pair creates waste-ore-waste below cutoff
                        prev_waste_grade, prev_waste_length = self._calculate_weighted_grade(
                            intervals, prev_seg["start"], prev_seg["end"], cfg.cutoff_field, cfg
                        )
                        
                        # Calculate waste-ore-waste combined grade
                        wow_length = prev_waste_length + waste_length + ore_length
                        wow_linear = (prev_waste_grade * prev_waste_length + 
                                    waste_grade * waste_length + 
                                    ore_grade * ore_length)
                        # Fix #4: Use math.isclose for robust floating point comparison
                        wow_avg = wow_linear / wow_length if not math.isclose(wow_length, 0.0, abs_tol=1e-8) else 0.0
                        
                        # Reject if waste-ore-waste pattern is below cutoff
                        if wow_avg < cfg.cutoff_grade - 1e-8:
                            return False
                        break  # Only check the last waste segment
            
            return True
        
        return True
    
    def _calculate_weighted_grade(self,
                                  intervals: List[Interval],
                                  start: float,
                                  end: float,
                                  grade_field: str,
                                  cfg: CompositeConfig) -> Tuple[float, float]:
        """
        Calculate length-weighted average grade for a segment.
        
        Returns:
            (average_grade, total_length) tuple
        """
        total_length = 0.0
        total_grade_length = 0.0  # grade * length
        
        for iv in intervals:
            if iv.to_depth <= start or iv.from_depth >= end:
                continue
            
            iv_from = max(iv.from_depth, start)
            iv_to = min(iv.to_depth, end)
            length = iv_to - iv_from
            if length <= 1e-8:
                continue
            
            grade = iv.grades.get(grade_field)
            grade_val = _get_grade_value(grade, cfg)
            if grade_val is not None:
                # Use weighting mode
                weight = _interval_weight(iv, length, cfg)
                total_grade_length += grade_val * weight
                total_length += length
        
        if total_length > 1e-8:
            avg_grade = total_grade_length / total_length
        else:
            avg_grade = 0.0
        
        return avg_grade, total_length
    
    def _calculate_linear_grade(self,
                               hole_id: str,
                               intervals: List[Interval],
                               start: float,
                               end: float,
                               cfg: CompositeConfig) -> float:
        """
        Calculate linear grade (grade * length) for a segment.
        
        Linear grade = average_grade * total_length
        This represents the total "grade content" in the segment.
        """
        if cfg.cutoff_field is None:
            return 0.0
        
        avg_grade, total_length = self._calculate_weighted_grade(
            intervals, start, end, cfg.cutoff_field, cfg
        )
        return avg_grade * total_length
    
    def _expand_waste_composites(self,
                                composites: List[Composite],
                                intervals: List[Interval],
                                cfg: CompositeConfig,
                                hole_id: str) -> List[Composite]:
        """
        Expand waste composites to meet min_waste_composite_length.
        
        Only applies to waste composites bounded on both sides by ore.
        Expansion takes segments from surrounding ore composites while minimizing
        loss from ore composites and ensuring ore constraints are still met.
        """
        if not composites or cfg.min_waste_composite_length is None:
            return composites
        
        expanded = []
        i = 0
        
        while i < len(composites):
            comp = composites[i]
            
            if comp.metadata.get("class") != "WASTE":
                expanded.append(comp)
                i += 1
                continue
            
            # Check if bounded by ore on both sides
            prev_is_ore = i > 0 and expanded[-1].metadata.get("class") == "ORE"
            next_is_ore = (i + 1 < len(composites) and 
                          composites[i + 1].metadata.get("class") == "ORE")
            
            if prev_is_ore and next_is_ore:
                waste_len = comp.to_depth - comp.from_depth
                
                if waste_len < cfg.min_waste_composite_length - 1e-8:
                    # Need to expand this waste composite
                    # Try to expand into adjacent ore composites
                    expansion_needed = cfg.min_waste_composite_length - waste_len
                    
                    # Try expanding upward first (into previous ore composite)
                    expanded_waste_start = comp.from_depth
                    expanded_waste_end = comp.to_depth
                    
                    if i > 0:
                        prev_ore = expanded[-1]
                        if prev_ore.metadata.get("class") == "ORE":
                            # Calculate how much we can take from previous ore
                            prev_ore_len = prev_ore.to_depth - prev_ore.from_depth
                            
                            # Check if taking from previous ore would break min_ore_composite_length
                            remaining_prev_ore_len = prev_ore_len - expansion_needed / 2
                            
                            if (cfg.min_ore_composite_length is None or 
                                remaining_prev_ore_len >= cfg.min_ore_composite_length - 1e-8):
                                # Safe to take from previous ore
                                take_from_prev = min(expansion_needed / 2, prev_ore_len)
                                expanded_waste_start = prev_ore.to_depth - take_from_prev
                                
                                # Update previous ore composite
                                prev_ore.to_depth = expanded_waste_start
                                prev_ore.metadata["support"] = prev_ore.to_depth - prev_ore.from_depth
                                
                                expansion_needed -= take_from_prev
                    
                    # Try expanding downward (into next ore composite)
                    if expansion_needed > 1e-8 and i + 1 < len(composites):
                        next_ore = composites[i + 1]
                        if next_ore.metadata.get("class") == "ORE":
                            next_ore_len = next_ore.to_depth - next_ore.from_depth
                            remaining_next_ore_len = next_ore_len - expansion_needed
                            
                            if (cfg.min_ore_composite_length is None or 
                                remaining_next_ore_len >= cfg.min_ore_composite_length - 1e-8):
                                take_from_next = min(expansion_needed, next_ore_len)
                                expanded_waste_end = next_ore.from_depth + take_from_next
                                
                                # Update next ore composite
                                next_ore.from_depth = expanded_waste_end
                                next_ore.metadata["support"] = next_ore.to_depth - next_ore.from_depth
                    
                    # Rebuild expanded waste composite
                    if expanded_waste_end - expanded_waste_start >= waste_len + 1e-8:
                        expanded_waste = self._build_segment_composite(
                            hole_id, intervals, expanded_waste_start, expanded_waste_end, cfg, "WASTE"
                        )
                        if expanded_waste:
                            expanded_waste.metadata["expanded"] = True
                            expanded_waste.metadata["original_length"] = waste_len
                            expanded.append(expanded_waste)
                        else:
                            expanded.append(comp)
                    else:
                        expanded.append(comp)
                else:
                    expanded.append(comp)
            else:
                expanded.append(comp)
            
            i += 1
        
        return expanded

    def _build_segment_composite(self,
                                 hole_id: str,
                                 intervals: List[Interval],
                                 seg_start: float,
                                 seg_end: float,
                                 cfg: CompositeConfig,
                                 class_label: str) -> Optional[Composite]:
        """
        Helper: composite a single depth segment [seg_start, seg_end] into
        one composite using the chosen weighting mode.
        """

        num_sums: Dict[str, float] = {}
        w_sums: Dict[str, float] = {}
        total_length: float = 0.0
        total_mass: float = 0.0
        sample_count: int = 0

        for iv in intervals:
            if iv.to_depth <= seg_start or iv.from_depth >= seg_end:
                continue

            iv_from = max(iv.from_depth, seg_start)
            iv_to = min(iv.to_depth, seg_end)
            length = iv_to - iv_from
            if length <= 1e-8:
                continue

            # weight slice according to cfg
            w_slice = _interval_weight(iv, length, cfg)

            for k, v in iv.grades.items():
                grade_val = _get_grade_value(v, cfg)
                if grade_val is None:
                    continue  # Skip weight accumulation (Partial sampling)
                num_sums[k] = num_sums.get(k, 0.0) + grade_val * w_slice
                w_sums[k] = w_sums.get(k, 0.0) + w_slice

            total_length += length
            if iv.density is not None:
                total_mass += length * iv.density

            sample_count += 1

        if not w_sums:
            return None

        grades: Dict[str, float] = {}
        for k, num in num_sums.items():
            w = w_sums.get(k, 0.0)
            if w > 0:
                grades[k] = num / w

        if total_length > 0 and total_mass > 0:
            grades["density"] = total_mass / total_length

        metadata = {
            "sample_count": sample_count,
            "support": seg_end - seg_start,
            "is_partial": False,
            "method": "economic",
            "class": class_label,
            "weighting": cfg.weighting_mode.value,
            "element_weights": dict(w_sums),
            "total_length": total_length,
            "total_mass": total_mass,
            "segment_start": seg_start,
            "segment_end": seg_end,
        }

        return Composite(
            hole_id=hole_id,
            from_depth=seg_start,
            to_depth=seg_end,
            grades=grades,
            metadata=metadata,
        )

    def _composite_attribute_filtered(self,
                                      intervals: Iterable[Interval],
                                      cfg: CompositeConfig) -> List[Composite]:
        """
        Attribute-filtered compositing.

        - Filters intervals by QAQC flags, recovery, etc.
        - Then delegates to a base compositing method
          (cfg.base_method_for_attribute_filtered, default FIXED_LENGTH).
        """

        filtered: List[Interval] = []

        for iv in intervals:
            # QAQC filter
            if cfg.exclude_qaqc:
                qaqc_type = iv.flags.get("QAQC_TYPE") or iv.flags.get("qaqc_type")
                if qaqc_type in cfg.qaqc_flags:
                    continue

            # Recovery filter
            if cfg.min_recovery is not None and iv.recovery is not None:
                if iv.recovery < cfg.min_recovery:
                    continue

            filtered.append(iv)

        if not filtered:
            return []

        # Delegate to base method
        base_cfg = cfg
        base_method = cfg.base_method_for_attribute_filtered

        if base_method == CompositingMethod.FIXED_LENGTH:
            comps = self._composite_fixed_length(filtered, base_cfg)
        elif base_method == CompositingMethod.EQUAL_MASS:
            comps = self._composite_equal_mass(filtered, base_cfg)
        elif base_method == CompositingMethod.EQUAL_VOLUME:
            comps = self._composite_equal_volume(filtered, base_cfg)
        elif base_method == CompositingMethod.LITHOLOGY:
            comps = self._composite_lithology(filtered, base_cfg)
        elif base_method == CompositingMethod.TRUE_THICKNESS:
            comps = self._composite_true_thickness(filtered, base_cfg)
        elif base_method == CompositingMethod.BENCH_ALIGNED:
            comps = self._composite_bench_aligned(filtered, base_cfg)
        elif base_method == CompositingMethod.ROLLING_WINDOW:
            comps = self._composite_rolling_window(filtered, base_cfg)
        elif base_method == CompositingMethod.ECONOMIC:
            comps = self._composite_economic(filtered, base_cfg)
        else:
            raise ValueError(f"Unsupported base method for ATTRIBUTE_FILTERED: {base_method}")

        for c in comps:
            c.metadata["attribute_filtered"] = True

        return comps

