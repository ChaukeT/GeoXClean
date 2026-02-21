"""
Utility functions for converting drillhole data to compositing engine format.

AUDIT COMPLIANCE:
- All interval exclusions are logged with reason
- Validation status is checked before conversion
- Gap/overlap detection warns but does not block (validation handles blocking)
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import bisect
import pandas as pd
import logging

from .compositing_engine import Interval

logger = logging.getLogger(__name__)


@dataclass
class IntervalConversionResult:
    """
    Result of converting DataFrames to Interval objects.
    
    Provides audit trail of which rows were included/excluded and why.
    """
    intervals: List[Interval] = field(default_factory=list)
    total_rows_processed: int = 0
    rows_included: int = 0
    rows_excluded: int = 0
    exclusion_reasons: Dict[str, int] = field(default_factory=dict)  # reason -> count
    warnings: List[str] = field(default_factory=list)
    validation_status: Optional[str] = None  # "passed", "warnings", "errors", "not_run"
    
    def add_exclusion(self, reason: str) -> None:
        """Track an exclusion with reason."""
        self.rows_excluded += 1
        self.exclusion_reasons[reason] = self.exclusion_reasons.get(reason, 0) + 1
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Interval Conversion Summary:",
            f"  Total rows processed: {self.total_rows_processed}",
            f"  Rows included: {self.rows_included}",
            f"  Rows excluded: {self.rows_excluded}",
        ]
        if self.exclusion_reasons:
            lines.append("  Exclusion reasons:")
            for reason, count in sorted(self.exclusion_reasons.items()):
                lines.append(f"    - {reason}: {count}")
        if self.warnings:
            lines.append(f"  Warnings: {len(self.warnings)}")
        if self.validation_status:
            lines.append(f"  Validation status: {self.validation_status}")
        return "\n".join(lines)


def dataframes_to_intervals(
    collars_df: Optional[pd.DataFrame] = None,
    assays_df: Optional[pd.DataFrame] = None,
    surveys_df: Optional[pd.DataFrame] = None,
    lithology_df: Optional[pd.DataFrame] = None,
    validation_result: Optional[Any] = None,
    strict_validation: bool = False,
    excluded_rows: Optional[Dict[str, List[int]]] = None,
) -> IntervalConversionResult:
    """
    Convert drillhole DataFrames to Interval objects for compositing.

    AUDIT COMPLIANCE (C-01, C-02, H-01):
    - All exclusions are logged with reasons
    - Validation status is checked if provided
    - Gap/overlap detection is performed
    - Rows with validation errors are excluded from downstream processing

    Args:
        collars_df: Collar data with hole_id, x, y, z, etc.
        assays_df: Assay data with hole_id, depth_from, depth_to, grade columns
        surveys_df: Survey data (optional, for trajectory calculation)
        lithology_df: Lithology data (optional, for lithology compositing)
        validation_result: Optional ValidationResult from drillhole_validation.run_drillhole_validation()
        strict_validation: If True, raises error when validation has errors; if False, logs warning
        excluded_rows: Dict mapping table name -> list of row indices to exclude.
                      Rows with ERROR violations should be excluded from compositing
                      even if "ignored" in the UI. Ignoring hides from display but
                      does NOT make invalid data valid for analysis.

    Returns:
        IntervalConversionResult with intervals and audit trail

    Raises:
        ValueError: If required columns are missing or if strict_validation=True and validation failed
    """
    result = IntervalConversionResult()

    if assays_df is None or assays_df.empty:
        logger.warning("No assay data provided for compositing")
        result.add_warning("No assay data provided")
        return result

    result.total_rows_processed = len(assays_df)

    # Get excluded assay row indices (rows with ERROR violations)
    excluded_assay_indices: set = set()
    if excluded_rows:
        # Try both 'assays' and 'assay' keys (common variations)
        for key in ['assays', 'assay', 'Assays', 'ASSAYS']:
            if key in excluded_rows:
                excluded_assay_indices.update(excluded_rows[key])
        if excluded_assay_indices:
            logger.info(
                f"Will exclude {len(excluded_assay_indices)} assay rows with validation errors"
            )
    
    # C-02 FIX: Check validation status if provided
    if validation_result is not None:
        error_count = sum(1 for v in validation_result.violations if v.severity == "ERROR")
        warning_count = sum(1 for v in validation_result.violations if v.severity == "WARNING")
        
        if error_count > 0:
            result.validation_status = "errors"
            msg = f"Validation has {error_count} ERROR(s) and {warning_count} WARNING(s). Compositing may produce invalid results."
            logger.warning(msg)
            result.add_warning(msg)
            
            if strict_validation:
                raise ValueError(
                    f"Cannot composite data with validation errors. "
                    f"Found {error_count} error(s). Run validation and fix issues first."
                )
        elif warning_count > 0:
            result.validation_status = "warnings"
            logger.info(f"Validation passed with {warning_count} warning(s)")
        else:
            result.validation_status = "passed"
            logger.info("Validation passed with no issues")
    else:
        result.validation_status = "not_run"
        logger.info("Compositing without validation check - consider running validation first")
    
    # Find column names (flexible matching)
    def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find a column by common aliases (case-insensitive)."""
        lowered = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in lowered:
                return lowered[cand.lower()]
        return None
    
    # Required columns for assays
    hole_col = find_column(assays_df, ["hole_id", "holeid", "hole", "HOLEID", "HOLE"])
    from_col = find_column(assays_df, ["depth_from", "from", "mfrom", "FROM", "DEPTH_FROM"])
    to_col = find_column(assays_df, ["depth_to", "to", "mto", "TO", "DEPTH_TO"])
    
    if not hole_col or not from_col or not to_col:
        raise ValueError(
            f"Assay data missing required columns. Found: {list(assays_df.columns)}. "
            f"Need: hole_id, depth_from, depth_to"
        )
    
    # Optional columns
    density_col = find_column(assays_df, ["density", "DENSITY", "dens", "DENS"])
    lith_col = find_column(assays_df, ["lith", "lithology", "LITH", "LITHOLOGY", "lith_code", "LITH_CODE"])
    domain_col = find_column(assays_df, ["domain", "DOMAIN", "geological_domain", "GEOLOGICAL_DOMAIN"])
    sample_type_col = find_column(assays_df, ["sample_type", "SAMPLE_TYPE", "sampletype", "SAMPLETYPE"])
    recovery_col = find_column(assays_df, ["recovery", "RECOVERY", "rec", "REC"])
    
    # Create a mapping from hole_id to lithology if lithology_df is provided
    lithology_map: Dict[str, List[Tuple[float, str]]] = {}
    if lithology_df is not None and not lithology_df.empty:
        lith_hole_col = find_column(lithology_df, ["hole_id", "holeid", "hole", "HOLEID", "HOLE"])
        lith_from_col = find_column(lithology_df, ["depth_from", "from", "mfrom", "FROM", "DEPTH_FROM"])
        lith_to_col = find_column(lithology_df, ["depth_to", "to", "mto", "TO", "DEPTH_TO"])
        lith_code_col = find_column(lithology_df, ["lith_code", "lithology", "code", "LITH_CODE", "LITHOLOGY", "LITH"])
        
        if lith_hole_col and lith_from_col and lith_to_col and lith_code_col:
            hole_ids = lithology_df[lith_hole_col].astype(str).values
            from_depths = lithology_df[lith_from_col].values.astype(float)
            to_depths = lithology_df[lith_to_col].values.astype(float)
            lith_codes = lithology_df[lith_code_col].astype(str).values
            mid_depths = (from_depths + to_depths) / 2.0
            
            temp_map: Dict[str, List[Tuple[float, str]]] = {}
            for i in range(len(hole_ids)):
                hole_id = hole_ids[i]
                if hole_id not in temp_map:
                    temp_map[hole_id] = []
                temp_map[hole_id].append((mid_depths[i], lith_codes[i]))
            
            for hole_id, lith_list in temp_map.items():
                lithology_map[hole_id] = sorted(lith_list, key=lambda x: x[0])
    
    # OPTIMIZATION: Use to_dict('records') instead of iterrows()
    assay_records = assays_df.to_dict('records')
    
    # Pre-calculate column names to avoid lookup inside loop
    col_map = {
        'hole': hole_col, 'from': from_col, 'to': to_col,
        'dens': density_col, 'lith': lith_col, 'dom': domain_col,
        'type': sample_type_col, 'rec': recovery_col
    }
    
    # Identify grade columns once (M-02: documented exclusion list)
    # These columns are excluded from grade detection:
    # - Coordinate columns (X, Y, Z, etc.)
    # - System ID columns (GLOBAL_INTERVAL_ID)
    # - Compositing metadata columns (from previous compositing runs)
    exclude_cols = {
        hole_col, from_col, to_col, density_col, lith_col, 
        domain_col, sample_type_col, recovery_col,
        "X", "Y", "Z", "x", "y", "z", "EAST", "NORTH", "RL",
        "GLOBAL_INTERVAL_ID", "global_interval_id",
        "SAMPLE_COUNT", "sample_count", "TOTAL_MASS", "total_mass",
        "TOTAL_LENGTH", "total_length", "SUPPORT", "support",
        "IS_PARTIAL", "is_partial", "METHOD", "method",
        "WEIGHTING", "weighting", "ELEMENT_WEIGHTS", "element_weights",
        "MERGED_PARTIAL", "merged_partial", "MERGED_PARTIAL_AUTO", "merged_partial_auto"
    }
    exclude_cols = {c for c in exclude_cols if c}
    grade_cols = [c for c in assays_df.columns if c not in exclude_cols]
    
    # H-01 FIX: Track intervals per hole for gap/overlap detection
    hole_intervals: Dict[str, List[Tuple[float, float, int]]] = {}  # hole_id -> [(from, to, row_idx), ...]
    
    # Convert each assay row to an Interval with full audit trail
    for row_idx, row in enumerate(assay_records):
        hole_id = str(row[col_map['hole']])

        # Check if this row has validation errors and should be excluded
        # This ensures data with ERROR violations doesn't flow to compositing
        # even if the user chose to "ignore" them in the QC UI
        if row_idx in excluded_assay_indices:
            result.add_exclusion("validation_error")
            logger.debug(f"Row {row_idx}: Excluded - validation error (hole={hole_id})")
            continue

        # C-01 FIX: Log all exclusions with reasons
        from_val = row.get(col_map['from'])
        to_val = row.get(col_map['to'])
        
        # Check for missing depths
        if pd.isna(from_val) or from_val is None:
            result.add_exclusion("missing_from_depth")
            logger.debug(f"Row {row_idx}: Excluded - missing from_depth (hole={hole_id})")
            continue
        
        if pd.isna(to_val) or to_val is None:
            result.add_exclusion("missing_to_depth")
            logger.debug(f"Row {row_idx}: Excluded - missing to_depth (hole={hole_id})")
            continue
        
        # Try to convert depths to float
        try:
            from_depth = float(from_val)
            to_depth = float(to_val)
        except (ValueError, TypeError) as e:
            result.add_exclusion("invalid_depth_format")
            logger.warning(f"Row {row_idx}: Excluded - invalid depth format (hole={hole_id}, from={from_val}, to={to_val}): {e}")
            continue
        
        # Check for negative or zero-length intervals
        interval_length = to_depth - from_depth
        if interval_length <= 0:
            result.add_exclusion("zero_or_negative_length")
            logger.warning(f"Row {row_idx}: Excluded - zero/negative length (hole={hole_id}, from={from_depth}, to={to_depth})")
            continue
        
        # Check for negative depths
        if from_depth < 0 or to_depth < 0:
            result.add_exclusion("negative_depth")
            logger.warning(f"Row {row_idx}: Excluded - negative depth (hole={hole_id}, from={from_depth}, to={to_depth})")
            continue
        
        # Track for gap/overlap detection
        if hole_id not in hole_intervals:
            hole_intervals[hole_id] = []
        hole_intervals[hole_id].append((from_depth, to_depth, row_idx))
        
        # Extract grades
        grades: Dict[str, float] = {}
        for gc in grade_cols:
            val = row.get(gc)
            if val is not None and isinstance(val, (int, float)) and not pd.isna(val):
                grades[gc] = float(val)
        
        # Optional attributes
        density = float(row[col_map['dens']]) if col_map['dens'] and pd.notna(row.get(col_map['dens'])) else None
        lith = str(row[col_map['lith']]) if col_map['lith'] and pd.notna(row.get(col_map['lith'])) else None
        
        # If lith not in assay data, try to get from lithology map
        if lith is None and hole_id in lithology_map:
            mid_depth = (from_depth + to_depth) / 2.0
            hole_lith = lithology_map[hole_id]
            if hole_lith:
                lith_depths = [x[0] for x in hole_lith]
                idx = bisect.bisect_left(lith_depths, mid_depth)
                if idx == 0:
                    lith = hole_lith[0][1]
                elif idx >= len(hole_lith):
                    lith = hole_lith[-1][1]
                else:
                    if abs(lith_depths[idx] - mid_depth) < abs(lith_depths[idx-1] - mid_depth):
                        lith = hole_lith[idx][1]
                    else:
                        lith = hole_lith[idx-1][1]
        
        domain = str(row[col_map['dom']]) if col_map['dom'] and pd.notna(row.get(col_map['dom'])) else None
        sample_type = str(row[col_map['type']]) if col_map['type'] and pd.notna(row.get(col_map['type'])) else None
        recovery = float(row[col_map['rec']]) if col_map['rec'] and pd.notna(row.get(col_map['rec'])) else None
        
        # Extract flags (QAQC metadata)
        flags: Dict[str, Any] = {}
        if sample_type:
            flags["sample_type"] = sample_type
        
        # Create interval
        interval = Interval(
            hole_id=hole_id,
            from_depth=from_depth,
            to_depth=to_depth,
            grades=grades,
            lith=lith,
            domain=domain,
            sample_type=sample_type,
            density=density,
            recovery=recovery,
            flags=flags,
        )
        
        result.intervals.append(interval)
        result.rows_included += 1
    
    # H-01 FIX: Detect gaps and overlaps per hole
    gap_count = 0
    overlap_count = 0
    for hole_id, intervals_list in hole_intervals.items():
        sorted_intervals = sorted(intervals_list, key=lambda x: x[0])
        prev_to = None
        for from_depth, to_depth, row_idx in sorted_intervals:
            if prev_to is not None:
                if from_depth > prev_to + 0.001:  # Gap tolerance 1mm
                    gap_count += 1
                    logger.debug(f"Gap detected in hole {hole_id}: {prev_to} to {from_depth} (gap={from_depth - prev_to:.3f}m)")
                elif from_depth < prev_to - 0.001:  # Overlap tolerance 1mm
                    overlap_count += 1
                    logger.warning(f"Overlap detected in hole {hole_id}: interval starts at {from_depth} but previous ends at {prev_to}")
            prev_to = to_depth
    
    if gap_count > 0:
        result.add_warning(f"Detected {gap_count} gaps in interval data")
    if overlap_count > 0:
        result.add_warning(f"Detected {overlap_count} overlaps in interval data - composites may be incorrect")
    
    # Log summary
    logger.info(result.summary())
    
    return result


def dataframes_to_intervals_simple(
    collars_df: Optional[pd.DataFrame] = None,
    assays_df: Optional[pd.DataFrame] = None,
    surveys_df: Optional[pd.DataFrame] = None,
    lithology_df: Optional[pd.DataFrame] = None,
) -> List[Interval]:
    """
    Simplified wrapper that returns just the intervals list.
    
    For backward compatibility with existing code that expects List[Interval].
    New code should use dataframes_to_intervals() for full audit trail.
    """
    result = dataframes_to_intervals(
        collars_df=collars_df,
        assays_df=assays_df,
        surveys_df=surveys_df,
        lithology_df=lithology_df,
    )
    return result.intervals


def get_intervals_from_registry(
    drillhole_data: Optional[Dict[str, Any]] = None,
    validation_result: Optional[Any] = None,
    strict_validation: bool = False,
    excluded_rows: Optional[Dict[str, List[int]]] = None,
    registry: Optional[Any] = None,
) -> Optional[List[Interval]]:
    """
    Get drillhole intervals from drillhole_data dict and convert to Interval objects.

    This function should be called from the UI thread. Data should be fetched from
    DataRegistry in the UI thread and passed explicitly to avoid race conditions.

    IMPORTANT: Rows with validation errors are automatically excluded from conversion
    to prevent invalid data from flowing downstream to compositing/kriging. This
    happens even if the user "ignored" errors in the QC UI - ignoring hides from
    display but does NOT make invalid data valid for analysis.

    Args:
        drillhole_data: Optional drillhole data dict with keys 'collars', 'assays',
                       'surveys', 'lithology'. If None, will attempt to fetch from
                       DataRegistry (UI thread only - deprecated pattern).
        validation_result: Optional ValidationResult to check before compositing
        strict_validation: If True, returns None when validation has errors
        excluded_rows: Optional dict of table -> row indices to exclude. If None,
                      will attempt to fetch from registry validation state.
        registry: Optional DataRegistry instance to fetch validation state from.

    Returns:
        List of Interval objects, or None if no data available
    """
    try:
        _registry = registry

        # If no data provided, fetch from registry (UI thread only - backward compatibility)
        if drillhole_data is None:
            import warnings
            warnings.warn(
                "get_intervals_from_registry() called without data parameter. "
                "This accesses DataRegistry directly and should only be called from UI thread. "
                "Prefer passing drillhole_data explicitly.",
                DeprecationWarning,
                stacklevel=2
            )
            from ..core.data_registry import DataRegistry

            _registry = DataRegistry.instance()
            if _registry is None:
                return None

            drillhole_data = _registry.get_drillhole_data()
            if drillhole_data is None:
                return None

        # Extract DataFrames
        collars_df = drillhole_data.get('collars') if isinstance(drillhole_data, dict) else None
        assays_df = drillhole_data.get('assays') if isinstance(drillhole_data, dict) else None
        surveys_df = drillhole_data.get('surveys') if isinstance(drillhole_data, dict) else None
        lithology_df = drillhole_data.get('lithology') if isinstance(drillhole_data, dict) else None

        if assays_df is None or assays_df.empty:
            return None

        # Fetch excluded_rows from validation state if not provided
        if excluded_rows is None and _registry is not None:
            try:
                validation_state = _registry.get_drillholes_validation_state()
                if validation_state is not None:
                    excluded_rows = validation_state.get('excluded_rows', {})
                    if excluded_rows:
                        total_excluded = sum(len(rows) for rows in excluded_rows.values())
                        logger.info(
                            f"Fetched {total_excluded} excluded rows from validation state"
                        )
            except Exception as e:
                logger.debug(f"Could not fetch excluded_rows from registry: {e}")

        # Use the full conversion function with validation support
        conversion_result = dataframes_to_intervals(
            collars_df=collars_df,
            assays_df=assays_df,
            surveys_df=surveys_df,
            lithology_df=lithology_df,
            validation_result=validation_result,
            strict_validation=strict_validation,
            excluded_rows=excluded_rows,
        )
        
        # Log any warnings or exclusions for audit
        if conversion_result.rows_excluded > 0:
            logger.warning(
                f"Interval conversion excluded {conversion_result.rows_excluded} rows. "
                f"Reasons: {conversion_result.exclusion_reasons}"
            )
        
        return conversion_result.intervals if conversion_result.intervals else None
        
    except ValueError as e:
        # Re-raise ValueError for strict validation failures
        logger.error(f"Validation error in interval conversion: {e}")
        raise
    except Exception as e:
        logger.error(f"Error getting intervals from registry: {e}", exc_info=True)
        return None


def get_intervals_with_audit(
    drillhole_data: Dict[str, Any],
    validation_result: Optional[Any] = None,
) -> IntervalConversionResult:
    """
    Get drillhole intervals with full audit trail.
    
    Use this function when you need to see exclusion reasons and warnings.
    
    Args:
        drillhole_data: Drillhole data dict with keys 'collars', 'assays', 'surveys', 'lithology'
        validation_result: Optional ValidationResult to check before compositing
    
    Returns:
        IntervalConversionResult with intervals and full audit trail
    """
    collars_df = drillhole_data.get('collars') if isinstance(drillhole_data, dict) else None
    assays_df = drillhole_data.get('assays') if isinstance(drillhole_data, dict) else None
    surveys_df = drillhole_data.get('surveys') if isinstance(drillhole_data, dict) else None
    lithology_df = drillhole_data.get('lithology') if isinstance(drillhole_data, dict) else None
    
    return dataframes_to_intervals(
        collars_df=collars_df,
        assays_df=assays_df,
        surveys_df=surveys_df,
        lithology_df=lithology_df,
        validation_result=validation_result,
        strict_validation=False,
    )

