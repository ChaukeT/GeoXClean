"""
Base Data Registry: Pure Python, Thread-Safe.

Handles thread-safe storage and validation. No Qt dependencies.
This allows headless tests to run without a display server.
"""

import threading
import copy
import logging
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataMetadata:
    """
    Metadata container for registered datasets.
    
    Includes provenance tracking to eliminate the "black box" problem:
    - source_file: Original file path
    - source_panel: Which panel registered this data
    - parent_data_key: Link to parent data (for transformation chains)
    - transformation_type: What transformation was applied
    - transformation_params: Parameters used in transformation
    """
    data_type: str
    source_panel: str
    timestamp: datetime
    row_count: Optional[int] = None
    columns: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    
    # Provenance tracking fields (NEW)
    source_file: Optional[str] = None
    parent_data_key: Optional[str] = None  # e.g., "raw_assays" -> "composites"
    transformation_type: Optional[str] = None  # e.g., "compositing", "declustering"
    transformation_params: Optional[Dict[str, Any]] = None  # e.g., {"length": 2.0}
    provenance_chain: Optional[List[Dict[str, Any]]] = None  # Full transformation history


class DataRegistrySimple:
    """
    Base Data Registry: Pure Python, Thread-Safe.

    Handles storage, locking, and validation. Knows nothing about Qt.
    This allows you to run headless tests on Cloud Run or GitHub Actions without a display.

    Thread Safety:
    - Uses RLock (Reentrant Lock) so the same thread can acquire it multiple times
    - All read/write operations are protected by the lock
    - Returns copies of data to prevent mutation
    """

    # Provenance chain depth limit to prevent unbounded memory growth
    MAX_PROVENANCE_DEPTH = 10
    
    def __init__(self):
        """Initialize the registry with thread-safe storage."""
        # Use RLock (Reentrant Lock) so the same thread can acquire it multiple times
        # This works in both GUI and Headless modes
        self._lock = threading.RLock()
        
        # Data storage - simple dictionaries and attributes
        self._data_store: Dict[str, Dict[str, Any]] = {}
        
        # Status flags for quick checks
        self._status_flags: Dict[str, bool] = {
            "drillhole_data": False,
            "block_model": False,
            "domain_model": False,
        }
        
        # Callback lists for panels that want notifications (alternative to signals)
        self._drillhole_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._block_model_callbacks: List[Callable[[Any], None]] = []
        self._domain_model_callbacks: List[Callable[[Any], None]] = []

        # Multi-model support for block models
        self._current_block_model_id: Optional[str] = None  # Active model pointer
        self._block_model_index: Dict[str, str] = {}  # {model_id: storage_key}
        self._last_registered_model_id: Optional[str] = None  # Last registered model ID (for signal emission)
        
        # ========================================================================
        # DRILLHOLE INTERVAL ID MANAGEMENT (Critical for GPU picking stability)
        # ========================================================================
        # Global counter for assigning persistent, immutable interval IDs
        # These IDs are assigned ONCE at ingestion and NEVER change
        # This ensures GPU picking remains stable across data reloads
        self._next_interval_id = 1  # Start at 1 (0 reserved for background)
        
        # Fast O(1) lookup: {GLOBAL_INTERVAL_ID: (df_key, row_idx)}
        # Enables instant interval retrieval during GPU picking
        self._interval_lookup: Dict[int, Tuple[str, int]] = {}
        
        logger.info("DataRegistrySimple initialized (thread-safe, pure Python)")
    
    # ========================================================================
    # Generic Storage Methods
    # ========================================================================
    
    def register_model(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None, source_panel: str = "unknown") -> bool:
        """
        Generic storage method.

        Args:
            key: Storage key (e.g., "drillhole_data", "block_model")
            data: Data to store
            metadata: Optional metadata dict
            source_panel: Source panel identifier

        Returns:
            True if successful, False if validation failed
        """
        # BUG FIX #20: Validate key is not None or empty
        if not key or not isinstance(key, str):
            logger.error(f"Invalid key: {key!r} - must be a non-empty string")
            return False

        # BUG FIX #20: Warn about unusual key patterns
        if key.startswith('.') or key.startswith('_') or ' ' in key:
            logger.warning(f"Unusual key pattern: {key!r} - consider using standard naming")

        with self._lock:
            if not self._validate(key, data):
                logger.error(f"Validation failed for {key}")
                return False
            
            # Build metadata
            meta = self._build_metadata(key, data, source_panel, metadata)
            
            # Store the data
            self._data_store[key] = {
                'data': data,
                'metadata': meta
            }
            
            # Update status flag (create if doesn't exist)
            self._status_flags[key] = True
            
            logger.info(f"Registered {key} from {source_panel}")
            return True
    
    def get_data(self, key: str, copy_data: bool = True) -> Optional[Any]:
        """
        Thread-safe retrieval.
        
        Args:
            key: Storage key
            copy_data: If True, return a copy to prevent mutation
            
        Returns:
            Stored data or None if not found
        """
        with self._lock:
            item = self._data_store.get(key)
            if item is None:
                return None
            
            # Return raw data or copy based on flag
            data = item['data']
            if copy_data:
                try:
                    # CRITICAL FIX: Use deep copy for DataFrames to prevent shared numpy arrays
                    # data.copy() is shallow - underlying arrays are shared!
                    if isinstance(data, pd.DataFrame):
                        return data.copy(deep=True)  # Deep copy prevents mutation of underlying arrays
                    elif isinstance(data, dict):
                        # For dicts containing DataFrames, need deep copy
                        return copy.deepcopy(data)
                    else:
                        return copy.deepcopy(data)
                except Exception as e1:
                    try:
                        # Fallback to shallow copy if deep copy fails
                        if isinstance(data, pd.DataFrame):
                            return data.copy()  # Shallow copy as fallback
                        return copy.copy(data)
                    except Exception as e2:
                        # BUG FIX: Raise exception instead of returning None to make failure explicit
                        # Returning None can cause cryptic downstream crashes
                        error_msg = (f"Failed to copy data for key '{key}' (deep: {e1}, shallow: {e2}). "
                                   f"Data may be corrupted or contain uncopyable objects.")
                        logger.error(error_msg, exc_info=True)
                        raise ValueError(error_msg) from e2
            return data
    
    def get_metadata(self, key: str) -> Optional[DataMetadata]:
        """Get metadata for a stored key."""
        with self._lock:
            item = self._data_store.get(key)
            if item is None:
                return None
            return item.get('metadata')
    
    def has_data(self, key: str) -> bool:
        """Check if data exists for a key."""
        with self._lock:
            return self._status_flags.get(key, False)
    
    def clear_data(self, key: str) -> None:
        """Clear data for a specific key."""
        with self._lock:
            if key in self._data_store:
                del self._data_store[key]
            if key in self._status_flags:
                self._status_flags[key] = False
            logger.info(f"Cleared {key}")
    
    def clear_all(self) -> None:
        """Clear all stored data."""
        with self._lock:
            self._data_store.clear()
            for key in self._status_flags:
                self._status_flags[key] = False
            logger.info("Cleared all data")
    
    # ========================================================================
    # Validation Methods
    # ========================================================================
    
    def _validate(self, key: str, data: Any) -> bool:
        """
        Base validation logic.
        
        Args:
            key: Storage key
            data: Data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if data is None:
            return False
        
        # Key-specific validation
        if key == "drillhole_data":
            valid, _ = self._validate_drillholes(data)
            return valid
        elif key == "block_model":
            valid, _ = self._validate_block_model(data)
            return valid
        elif key == "domain_model":
            return True  # Domain models can be various types
        
        # Default: accept non-None data
        return True
    
    def _validate_drillholes(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate drillhole data structure.

        Allows collar-only imports for visualization, but requires assays/composites
        for estimation operations.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(data, dict):
            logger.error(f"❌ Validation: data is not dict, type={type(data)}")
            return False, "Expected dictionary"

        logger.info(f"🔍 Validating drillhole data with keys: {list(data.keys())}")

        # Check collars (mandatory)
        collars = data.get("collars")
        if not isinstance(collars, pd.DataFrame):
            logger.error(f"❌ Validation failed: collars is not DataFrame, type={type(collars)}")
            return False, "Collars not a DataFrame"

        if len(collars) == 0:
            logger.error(f"❌ Validation failed: collars DataFrame is empty")
            return False, "Collars are empty"

        logger.info(f"✅ Collars validated: {len(collars)} drillholes")

        # Check assays/composites (optional - needed for estimation only)
        composites = data.get("composites")
        logger.info(f"🔍 composites type: {type(composites)}, is None: {composites is None}")

        if composites is None:
            composites = data.get("assays")
            logger.info(f"🔍 Falling back to assays, type: {type(composites)}, is None: {composites is None}")

        if composites is None or not isinstance(composites, pd.DataFrame) or len(composites) == 0:
            # Collar-only mode - valid for visualization but not for estimation
            logger.warning(f"⚠️ COLLAR-ONLY MODE: No assays/composites data")
            logger.warning(f"   Drillholes can be visualized but estimation operations will be disabled")
            logger.info(f"✅ Validation passed: collar-only mode ({len(collars)} collars)")
            return True, None  # Valid, but collar-only

        # Full validation - assays/composites exist
        logger.info(f"✅ Validation passed: {len(collars)} collars, {len(composites)} assay/composite records")
        return True, None
    
    def _validate_block_model(self, block_model: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate block model structure.
        
        ✅ Supports BlockModel (standard API) or DataFrame (legacy).
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if block_model is None:
            return False, "Block model is None"
        
        # ✅ NEW: Validate BlockModel (standard API)
        if hasattr(block_model, 'block_count') and hasattr(block_model, 'positions'):
            # It's a BlockModel instance
            if block_model.block_count == 0:
                return False, "BlockModel is empty (block_count = 0)"
            if block_model.positions is None:
                return False, "BlockModel missing positions"
            return True, None
        
        # Legacy: Validate DataFrame
        if isinstance(block_model, pd.DataFrame):
            required = {"X", "Y", "Z"}
            missing = required - set(block_model.columns)
            if missing:
                return False, f"Missing columns: {', '.join(sorted(missing))}"
            return True, None
        
        # Unknown type
        return False, f"Unsupported block model type: {type(block_model)}"
    
    def _build_metadata(
        self,
        data_type: str,
        data: Any,
        source_panel: str,
        custom_metadata: Optional[Dict[str, Any]] = None,
        source_file: Optional[str] = None,
        parent_data_key: Optional[str] = None,
        transformation_type: Optional[str] = None,
        transformation_params: Optional[Dict[str, Any]] = None,
    ) -> DataMetadata:
        """
        Build metadata object from data with provenance tracking.
        
        Args:
            data_type: Type of data being stored
            data: The actual data
            source_panel: Panel that registered this data
            custom_metadata: Additional custom metadata
            source_file: Original file path (for provenance)
            parent_data_key: Key of parent data (for transformation chains)
            transformation_type: Type of transformation applied
            transformation_params: Parameters used in transformation
            
        Returns:
            DataMetadata object with full provenance information
        """
        row_count: Optional[int] = None
        columns: Optional[List[str]] = None

        if isinstance(data, pd.DataFrame):
            row_count = len(data)
            columns = list(data.columns)
        elif isinstance(data, dict):
            df = next((v for v in data.values() if isinstance(v, pd.DataFrame)), None)
            if isinstance(df, pd.DataFrame):
                row_count = len(df)
                columns = list(df.columns)
            elif "estimated_values" in data and hasattr(data["estimated_values"], "__len__"):
                row_count = len(data["estimated_values"])
        elif isinstance(data, list):
            row_count = len(data)

        # Build provenance chain
        provenance_chain = None
        if parent_data_key:
            # Try to get parent's provenance chain
            parent_item = self._data_store.get(parent_data_key)
            if parent_item and 'metadata' in parent_item:
                parent_meta = parent_item['metadata']
                if parent_meta.provenance_chain:
                    # BUG FIX #9: Deep copy provenance chain to prevent shared dict references
                    provenance_chain = copy.deepcopy(parent_meta.provenance_chain)

                    # Trim oldest entries if chain too long (prevent unbounded memory growth)
                    if len(provenance_chain) >= self.MAX_PROVENANCE_DEPTH:
                        origin = provenance_chain[:2]  # Keep first 2 entries (original data)
                        recent = provenance_chain[-(self.MAX_PROVENANCE_DEPTH-3):]  # Keep recent entries
                        provenance_chain = origin + [
                            {"transformation_type": "... (trimmed)",
                             "note": f"{len(provenance_chain)-self.MAX_PROVENANCE_DEPTH+1} steps omitted"}
                        ] + recent
                        logger.debug(f"Provenance chain trimmed to {self.MAX_PROVENANCE_DEPTH} entries")
                else:
                    provenance_chain = []

                # Add parent to chain
                provenance_chain.append({
                    'data_type': parent_meta.data_type,
                    'source_panel': parent_meta.source_panel,
                    'timestamp': parent_meta.timestamp.isoformat(),
                    'source_file': parent_meta.source_file,
                    'transformation_type': parent_meta.transformation_type,
                    'transformation_params': parent_meta.transformation_params,
                })
            else:
                provenance_chain = []
        
        # Add current transformation to chain if present
        if transformation_type and provenance_chain is not None:
            provenance_chain.append({
                'data_type': data_type,
                'source_panel': source_panel,
                'timestamp': datetime.now().isoformat(),
                'transformation_type': transformation_type,
                'transformation_params': transformation_params,
            })

        return DataMetadata(
            data_type=data_type,
            source_panel=source_panel,
            timestamp=datetime.now(),
            row_count=row_count,
            columns=columns,
            custom_metadata=custom_metadata,
            source_file=source_file,
            parent_data_key=parent_data_key,
            transformation_type=transformation_type,
            transformation_params=transformation_params,
            provenance_chain=provenance_chain,
        )
    
    # ========================================================================
    # Drillhole Data Methods
    # ========================================================================
    
    def register_drillhole_data(
        self,
        data: Dict[str, Any],
        source_panel: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        is_raw_import: bool = False,
    ) -> bool:
        """
        Register drillhole data - thread-safe with persistent interval IDs.

        THREE-LAYER DATA SYSTEM (JORC/SAMREC Compliance):
        1. RAW DATA: Immutable original import, never touched
        2. VALIDATED DATA: Current working data with validation state
        3. ESTIMATION-READY: Generated on-demand, excludes error rows

        CRITICAL: Assigns GLOBAL_INTERVAL_ID to each interval at ingestion time.
        These IDs are immutable and survive data reloads, ensuring GPU picking stability.

        Args:
            data: Drillhole data dictionary
            source_panel: Source panel identifier
            metadata: Optional custom metadata
            source_id: Optional source ID to prevent signal loops
            is_raw_import: If True, this is a fresh import and raw data will be stored

        Returns:
            True if successful, False if validation failed
        """
        logger.debug("DataRegistrySimple.register_drillhole_data: STARTED")

        with self._lock:
            # Validate
            logger.debug("DataRegistrySimple.register_drillhole_data: Validating...")
            valid, error = self._validate_drillholes(data)
            if not valid:
                logger.error(f"Validation failed for drillhole data: {error}")
                return False
            logger.debug("DataRegistrySimple.register_drillhole_data: Validation passed")

            # CRITICAL: Ensure all interval data has 3D coordinates (X, Y, Z)
            # This is required for geological modeling (LoopStructural) to work correctly.
            # Without this, surfaces are built in a different coordinate space than drillholes!
            logger.info("DataRegistrySimple.register_drillhole_data: Ensuring 3D coordinates (Minimum Curvature)...")
            data = self._ensure_interval_coordinates(data)
            logger.info("DataRegistrySimple.register_drillhole_data: 3D coordinates ensured")

            # CRITICAL: Assign persistent GLOBAL_INTERVAL_ID to each DataFrame
            # This ensures GPU picking IDs remain stable across data reloads
            logger.debug("DataRegistrySimple.register_drillhole_data: Assigning interval IDs...")
            data_with_ids = self._assign_interval_ids(data)
            logger.debug("DataRegistrySimple.register_drillhole_data: Interval IDs assigned")

            # THREE-LAYER SYSTEM: Store raw data on first import
            # Raw data is IMMUTABLE - never overwritten once stored
            if is_raw_import or "drillhole_data_raw" not in self._data_store:
                raw_copy = copy.deepcopy(data_with_ids)
                raw_meta = self._build_metadata("drillhole_data_raw", raw_copy, source_panel, {
                    **(metadata or {}),
                    "layer": "raw",
                    "immutable": True,
                    "import_timestamp": datetime.now().isoformat(),
                })
                self._data_store["drillhole_data_raw"] = {
                    'data': raw_copy,
                    'metadata': raw_meta
                }
                logger.info(f"Stored RAW drillhole data (immutable layer) from {source_panel}")

            # Store validated/working data
            logger.debug("DataRegistrySimple.register_drillhole_data: Building metadata...")
            meta = self._build_metadata("drillhole_data", data_with_ids, source_panel, {
                **(metadata or {}),
                "layer": "validated",
            })
            logger.debug("DataRegistrySimple.register_drillhole_data: Storing data...")
            self._data_store["drillhole_data"] = {
                'data': data_with_ids,
                'metadata': meta
            }
            self._status_flags["drillhole_data"] = True

            logger.info(f"Registered drillhole data from {source_panel} with {len(self._interval_lookup)} total intervals")
            logger.debug("DataRegistrySimple.register_drillhole_data: FINISHED")
            return True
    
    def get_drillhole_data(
        self,
        copy_data: bool = True,
        mode: str = "validated",
        override_reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get drillhole data - thread-safe read with data layer selection.

        THREE-LAYER DATA SYSTEM (JORC/SAMREC Compliance):

        Mode "raw":
            Returns original imported data. NEVER modified.
            Use for: Audit trail, comparison, re-import.

        Mode "validated" (DEFAULT):
            Returns current working data (after QC edits).
            Includes all rows. Caller must check validation state.
            Use for: QC Window, data review.

        Mode "estimation_ready":
            Returns filtered data ready for geostatistics.
            EXCLUDES all rows with ERROR violations.
            REQUIRES validation to have passed (PASS or WARN).
            BLOCKS if validation not run or FAIL.
            Use for: Kriging, Simulation, Variogram, Resource Estimation.

        Mode "including_errors":
            Returns validated data INCLUDING error rows.
            REQUIRES override_reason (logged to audit trail).
            Use for: Expert analysis, debugging.

        Args:
            copy_data: If True, return a copy to prevent mutation
            mode: One of "raw", "validated", "estimation_ready", "including_errors"
            override_reason: Required for "including_errors" mode (logged)

        Returns:
            Drillhole data dictionary or None

        Raises:
            ValueError: If mode is invalid
            ValueError: If "estimation_ready" requested but validation not passed
            ValueError: If "including_errors" requested without override_reason
        """
        with self._lock:
            # MODE: RAW - Return immutable original import
            if mode == "raw":
                item = self._data_store.get("drillhole_data_raw")
                if item is None:
                    # Fallback to current data if raw not stored
                    logger.warning("Raw data not found, returning current data")
                    return self.get_data("drillhole_data", copy_data=copy_data)
                data = item['data']
                if copy_data:
                    return copy.deepcopy(data)
                return data

            # MODE: VALIDATED - Return current working data (default)
            elif mode == "validated":
                return self.get_data("drillhole_data", copy_data=copy_data)

            # MODE: ESTIMATION_READY - Return filtered data, BLOCK if not validated
            elif mode == "estimation_ready":
                # HARD GATE: Check validation status
                allowed, message = self.require_validation_for_estimation()
                if not allowed:
                    logger.error(f"BLOCKED: Cannot return estimation_ready data: {message}")
                    raise ValueError(
                        f"Data not ready for estimation: {message}\n\n"
                        "Run QC validation and fix or exclude error rows before proceeding."
                    )

                # Get data and filter out error rows
                data = self.get_data("drillhole_data", copy_data=True)
                if data is None:
                    return None

                # Apply excluded_rows filter
                data = self._filter_excluded_rows(data)

                logger.info(
                    f"Returning estimation_ready data (filtered by excluded_rows)"
                )
                return data

            # MODE: INCLUDING_ERRORS - Requires explicit override with reason
            elif mode == "including_errors":
                if not override_reason:
                    raise ValueError(
                        "Mode 'including_errors' requires override_reason parameter.\n"
                        "This is required for JORC/SAMREC audit compliance."
                    )

                # Log override to audit trail
                self._log_data_override(
                    mode="including_errors",
                    reason=override_reason,
                )

                logger.warning(
                    f"DATA OVERRIDE: Returning data including error rows. "
                    f"Reason: {override_reason}"
                )

                return self.get_data("drillhole_data", copy_data=copy_data)

            else:
                raise ValueError(
                    f"Invalid mode: {mode}. "
                    f"Must be one of: 'raw', 'validated', 'estimation_ready', 'including_errors'"
                )

    def _filter_excluded_rows(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter out rows with ERROR violations from drillhole data.

        Used by get_drillhole_data(mode="estimation_ready") to ensure
        invalid data never reaches geostatistical engines.

        Args:
            data: Drillhole data dictionary (already copied)

        Returns:
            Filtered data dictionary
        """
        validation_state = self.get_drillholes_validation_state()
        if validation_state is None:
            return data

        excluded_rows = validation_state.get("excluded_rows", {})
        if not excluded_rows:
            return data

        total_excluded = 0

        for table_key, row_indices in excluded_rows.items():
            if not row_indices:
                continue

            # Find the DataFrame in data
            df = data.get(table_key)
            if df is None:
                # Try common variations
                for key in [table_key, table_key.lower(), table_key.upper()]:
                    df = data.get(key)
                    if df is not None:
                        table_key = key
                        break

            if df is None or not isinstance(df, pd.DataFrame):
                continue

            # Filter out excluded rows
            original_count = len(df)
            valid_indices = [i for i in df.index if i not in row_indices]
            if len(valid_indices) < original_count:
                data[table_key] = df.loc[valid_indices].reset_index(drop=True)
                excluded_count = original_count - len(valid_indices)
                total_excluded += excluded_count
                logger.info(f"Filtered {excluded_count} error rows from {table_key}")

        if total_excluded > 0:
            logger.info(f"Total rows filtered for estimation_ready: {total_excluded}")

        return data

    def require_validation_for_estimation(self) -> Tuple[bool, str]:
        """
        Check if data is ready for estimation (kriging, simulation, etc.)

        HARD GATE: Estimation engines MUST call this before processing.
        Returns (False, message) if estimation should be BLOCKED.

        This is NOT a warning. This is a STOP.

        Returns:
            (allowed: bool, message: str)
            If allowed=False, estimation MUST NOT proceed.
        """
        validation_state = self.get_drillholes_validation_state()

        # BLOCK: Never validated
        if validation_state is None:
            return (False, "Data has NOT been validated. Run QC validation first.")

        status = validation_state.get("status", "NOT_RUN")

        # BLOCK: Validation not run
        if status == "NOT_RUN":
            return (False, "Data has NOT been validated. Run QC validation first.")

        # BLOCK: Validation failed
        if status == "FAIL":
            fatal_count = validation_state.get("fatal_count", 0)
            return (
                False,
                f"Data FAILED validation with {fatal_count} ERROR(s). "
                f"Fix errors or explicitly exclude them before estimation."
            )

        # ALLOW: Validation passed (PASS or WARN)
        if status in ("PASS", "WARN"):
            excluded = validation_state.get("excluded_rows", {})
            total_excluded = sum(len(rows) for rows in excluded.values())
            if total_excluded > 0:
                return (
                    True,
                    f"Validation {status}. {total_excluded} error rows will be excluded."
                )
            return (True, f"Validation {status}. All data ready for estimation.")

        # Unknown status - block to be safe
        return (False, f"Unknown validation status: {status}")

    def _log_data_override(self, mode: str, reason: str) -> None:
        """
        Log data access override to audit trail for JORC/SAMREC compliance.

        Every time a user/system accesses data with errors included,
        it MUST be logged with a reason.
        """
        try:
            from .audit_manager import AuditManager
            audit = AuditManager()
            audit.log_event(
                module="data_registry",
                action="data_override",
                parameters={
                    "mode": mode,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat(),
                },
                result_summary={
                    "validation_status": self.get_drillholes_validation_status(),
                    "warning": "Data accessed with errors included - requires justification",
                }
            )
            logger.warning(f"DATA OVERRIDE logged to audit trail: {reason}")
        except Exception as e:
            # Log to file even if audit manager fails
            logger.warning(
                f"DATA OVERRIDE (audit failed): mode={mode}, reason={reason}, error={e}"
            )
    
    def get_drillhole_metadata(self) -> Optional[DataMetadata]:
        """Get drillhole metadata."""
        return self.get_metadata("drillhole_data")
    
    # ========================================================================
    # Drillhole Coordinate Calculation (Critical for Geological Modeling)
    # ========================================================================
    
    def _ensure_interval_coordinates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all interval DataFrames have proper 3D coordinates (X, Y, Z).
        
        CRITICAL: Without proper coordinates, geological surfaces (LoopStructural)
        are built in a different coordinate space than the drillhole visualization.
        This causes the classic "surface passes through drillhole intervals" bug.
        
        This method uses Minimum Curvature desurveying to calculate the 3D position
        of each interval midpoint, ensuring:
        - Surfaces are constrained by actual drillhole positions
        - Rendered drillholes and model constraints are in the same coordinate space
        - Deviated holes are handled correctly (not assumed vertical)
        
        Args:
            data: Drillhole data dictionary with collars, surveys, assays, etc.
            
        Returns:
            Data dictionary with X, Y, Z columns added to all interval DataFrames
        """
        from ..utils.desurvey import add_coordinates_to_intervals
        
        data_copy = copy.deepcopy(data)
        
        # Get collar and survey data (required for desurveying)
        collar_df = data_copy.get('collars')
        if collar_df is None:
            collar_df = data_copy.get('collar')  # Alternative key
        survey_df = data_copy.get('surveys')
        if survey_df is None:
            survey_df = data_copy.get('survey')  # Alternative key
            
        if collar_df is None or (isinstance(collar_df, pd.DataFrame) and collar_df.empty):
            logger.warning("Cannot calculate 3D coordinates: no collar data available")
            return data_copy
        
        # Detect hole ID column in collar
        collar_id_col = self._detect_holeid_column(collar_df)
        if not collar_id_col:
            logger.warning("Cannot calculate 3D coordinates: no hole ID column in collars")
            return data_copy
        
        # Process each interval DataFrame
        interval_keys = ['assays', 'lithology', 'composites', 'intervals']
        
        for key in interval_keys:
            df = data_copy.get(key)
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                continue
            
            logger.info(f"Processing {key}: {len(df)} rows, columns: {list(df.columns)[:10]}...")  # First 10 columns
            
            # Check if already has coordinates
            has_x = 'X' in df.columns and df['X'].notna().any() and (df['X'] != 0).any()
            has_y = 'Y' in df.columns and df['Y'].notna().any() and (df['Y'] != 0).any()
            has_z = 'Z' in df.columns and df['Z'].notna().any() and (df['Z'] != 0).any()
            
            if has_x and has_y and has_z:
                # Log coordinate ranges to verify they're reasonable
                x_range = (df['X'].min(), df['X'].max())
                y_range = (df['Y'].min(), df['Y'].max())
                z_range = (df['Z'].min(), df['Z'].max())
                logger.info(f"{key} already has X, Y, Z coordinates - skipping calculation")
                logger.info(f"  Existing {key} coordinate ranges: X={x_range}, Y={y_range}, Z={z_range}")
                continue
            
            # Detect column names
            hole_id_col = self._detect_holeid_column(df)
            from_col = self._detect_column(df, {'from', 'depth_from', 'start'})
            to_col = self._detect_column(df, {'to', 'depth_to', 'end'})
            
            if not hole_id_col or not from_col or not to_col:
                logger.warning(f"Cannot calculate coordinates for {key}: missing required columns")
                continue
            
            # Calculate 3D coordinates using Minimum Curvature desurveying
            logger.info(f"Calculating 3D coordinates for {len(df)} {key} intervals using Minimum Curvature...")
            try:
                df_with_coords = add_coordinates_to_intervals(
                    intervals_df=df,
                    collar_df=collar_df,
                    survey_df=survey_df,
                    hole_id_col=hole_id_col,
                    from_col=from_col,
                    to_col=to_col,
                )
                data_copy[key] = df_with_coords
                
                # Log coordinate ranges for verification
                if 'X' in df_with_coords.columns:
                    x_range = (df_with_coords['X'].min(), df_with_coords['X'].max())
                    y_range = (df_with_coords['Y'].min(), df_with_coords['Y'].max())
                    z_range = (df_with_coords['Z'].min(), df_with_coords['Z'].max())
                    logger.info(f"  {key} coordinate ranges: X={x_range}, Y={y_range}, Z={z_range}")
                    
            except Exception as e:
                logger.error(f"Failed to calculate coordinates for {key}: {e}", exc_info=True)
                # Continue with other DataFrames even if one fails
        
        return data_copy
    
    def _detect_holeid_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect hole ID column (case-insensitive)."""
        return self._detect_column(df, {'holeid', 'hole_id', 'hole', 'bhid', 'drillhole_id'})
    
    def _detect_column(self, df: pd.DataFrame, candidates: set) -> Optional[str]:
        """Detect column by name (case-insensitive)."""
        lower_candidates = {name.strip().lower() for name in candidates}
        for col in df.columns:
            if col and str(col).strip().lower() in lower_candidates:
                return col
        return None
    
    # ========================================================================
    # Drillhole Interval ID Management (Critical for GPU Picking)
    # ========================================================================
    
    def _assign_interval_ids(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assign persistent GLOBAL_INTERVAL_ID to all interval DataFrames.
        
        CRITICAL: This is called ONCE at ingestion time. IDs are NEVER reassigned.
        This ensures GPU picking remains stable across:
        - Data reloads
        - DataFrame reindexing
        - Filtering operations
        - Compositing operations
        - Worker thread updates
        
        Args:
            data: Drillhole data dictionary
            
        Returns:
            Data dictionary with GLOBAL_INTERVAL_ID columns added
        """
        data_copy = copy.deepcopy(data)
        
        # Process assays DataFrame
        if 'assays' in data_copy:
            df = data_copy['assays']
            if isinstance(df, pd.DataFrame) and not df.empty:
                if 'GLOBAL_INTERVAL_ID' not in df.columns:
                    ids = np.arange(self._next_interval_id, 
                                   self._next_interval_id + len(df), 
                                   dtype=np.int32)
                    df['GLOBAL_INTERVAL_ID'] = ids
                    
                    # Build O(1) lookup table
                    for i, global_id in enumerate(ids):
                        self._interval_lookup[int(global_id)] = ('assays', i)
                    
                    self._next_interval_id += len(df)
                    logger.info(f"Assigned {len(df)} persistent IDs to assay intervals (range: {ids[0]}-{ids[-1]})")
        
        # Process lithology DataFrame
        if 'lithology' in data_copy:
            df = data_copy['lithology']
            if isinstance(df, pd.DataFrame) and not df.empty:
                if 'GLOBAL_INTERVAL_ID' not in df.columns:
                    ids = np.arange(self._next_interval_id,
                                   self._next_interval_id + len(df),
                                   dtype=np.int32)
                    df['GLOBAL_INTERVAL_ID'] = ids
                    
                    # Build O(1) lookup table
                    for i, global_id in enumerate(ids):
                        self._interval_lookup[int(global_id)] = ('lithology', i)
                    
                    self._next_interval_id += len(df)
                    logger.info(f"Assigned {len(df)} persistent IDs to lithology intervals (range: {ids[0]}-{ids[-1]})")
        
        # Process composites DataFrame (if present)
        if 'composites' in data_copy:
            df = data_copy['composites']
            if isinstance(df, pd.DataFrame) and not df.empty:
                if 'GLOBAL_INTERVAL_ID' not in df.columns:
                    ids = np.arange(self._next_interval_id,
                                   self._next_interval_id + len(df),
                                   dtype=np.int32)
                    df['GLOBAL_INTERVAL_ID'] = ids
                    
                    # Build O(1) lookup table
                    for i, global_id in enumerate(ids):
                        self._interval_lookup[int(global_id)] = ('composites', i)
                    
                    self._next_interval_id += len(df)
                    logger.info(f"Assigned {len(df)} persistent IDs to composite intervals (range: {ids[0]}-{ids[-1]})")
        
        # Process survey intervals (if present as DataFrame)
        if 'surveys' in data_copy:
            df = data_copy['surveys']
            if isinstance(df, pd.DataFrame) and not df.empty:
                if 'GLOBAL_INTERVAL_ID' not in df.columns:
                    ids = np.arange(self._next_interval_id,
                                   self._next_interval_id + len(df),
                                   dtype=np.int32)
                    df['GLOBAL_INTERVAL_ID'] = ids
                    
                    # Build O(1) lookup table
                    for i, global_id in enumerate(ids):
                        self._interval_lookup[int(global_id)] = ('surveys', i)
                    
                    self._next_interval_id += len(df)
                    logger.info(f"Assigned {len(df)} persistent IDs to survey intervals (range: {ids[0]}-{ids[-1]})")
        
        return data_copy
    
    def lookup_interval_by_id(self, global_id: int) -> Optional[Tuple[str, pd.Series]]:
        """
        Fast O(1) lookup of interval by GLOBAL_INTERVAL_ID.
        
        This is the critical method for GPU picking - it translates a picked
        color ID back to the actual interval data.
        
        Args:
            global_id: The GLOBAL_INTERVAL_ID to look up
            
        Returns:
            Tuple of (df_name, interval_row) or None if not found
            where df_name is 'assays', 'lithology', 'composites', or 'surveys'
            and interval_row is a pandas Series with all interval data
        """
        with self._lock:
            if global_id not in self._interval_lookup:
                logger.debug(f"Interval ID {global_id} not found in lookup table")
                return None
            
            df_key, row_idx = self._interval_lookup[global_id]
            
            # Get the DataFrame
            drillhole_data = self._data_store.get("drillhole_data")
            if not drillhole_data:
                logger.warning("No drillhole data in registry")
                return None
            
            data = drillhole_data['data']
            if df_key not in data:
                logger.warning(f"DataFrame key '{df_key}' not found in drillhole data")
                return None
            
            df = data[df_key]
            if not isinstance(df, pd.DataFrame) or row_idx >= len(df):
                logger.warning(f"Invalid DataFrame or row index for interval {global_id}")
                return None
            
            return (df_key, df.iloc[row_idx])
    
    def rebuild_interval_lookup(self) -> None:
        """
        Rebuild the interval lookup table after data changes.
        
        Call this after operations that might change DataFrame order:
        - Filtering operations
        - Sorting operations  
        - Merging DataFrames
        - Adding new intervals
        
        This ensures the lookup table stays synchronized with DataFrame indices.
        """
        with self._lock:
            self._interval_lookup.clear()
            
            drillhole_data = self._data_store.get("drillhole_data")
            if not drillhole_data:
                logger.debug("No drillhole data to rebuild lookup table")
                return
            
            data = drillhole_data['data']
            
            # Rebuild from each DataFrame
            for df_key in ['assays', 'lithology', 'composites', 'surveys']:
                if df_key not in data:
                    continue
                    
                df = data[df_key]
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                
                if 'GLOBAL_INTERVAL_ID' not in df.columns:
                    logger.warning(f"{df_key} missing GLOBAL_INTERVAL_ID column - data may be from old version")
                    continue
                
                for i, global_id in enumerate(df['GLOBAL_INTERVAL_ID']):
                    self._interval_lookup[int(global_id)] = (df_key, i)
            
            logger.info(f"Rebuilt interval lookup table with {len(self._interval_lookup)} entries")
    
    def get_interval_id_stats(self) -> Dict[str, Any]:
        """
        Get statistics about interval ID assignment.
        
        Useful for debugging and monitoring data integrity.
        
        Returns:
            Dictionary with ID statistics
        """
        with self._lock:
            return {
                'next_id': self._next_interval_id,
                'lookup_size': len(self._interval_lookup),
                'ids_assigned': self._next_interval_id - 1,
                'dataframes': {}
            }

    
    def has_drillhole_data(self) -> bool:
        """Check if drillhole data is available."""
        return self.has_data("drillhole_data")
    
    def clear_drillhole_data(self) -> None:
        """Clear drillhole data and interval lookup table (DR-002 fix).
        
        CRITICAL: Also clears _interval_lookup to prevent memory leak
        on repeated data imports. The next_interval_id is NOT reset
        to preserve ID uniqueness across sessions.
        
        Also clears validation state since it no longer applies to new data.
        """
        with self._lock:
            self.clear_data("drillhole_data")
            # Clear interval lookup to free memory (DR-002 fix)
            cleared_count = len(self._interval_lookup)
            self._interval_lookup.clear()
            logger.info(f"Cleared {cleared_count} interval ID mappings from lookup table")
            
            # Clear validation state - it no longer applies to new data
            self.clear_drillholes_validation_state()
    
    def connect_drillhole_loaded(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register callback for drillhole data loaded events.

        BUG FIX #3-4: Improved callback management to prevent duplicates.
        """
        with self._lock:
            # Check by identity, not just equality
            for existing in self._drillhole_callbacks:
                if existing is callback:
                    logger.debug(f"Callback {callback} already registered, skipping duplicate")
                    return
            self._drillhole_callbacks.append(callback)
            logger.debug(f"Registered drillhole callback, total: {len(self._drillhole_callbacks)}")

    def disconnect_drillhole_loaded(self, callback: Callable[[Dict[str, Any]], None]) -> bool:
        """Unregister callback for drillhole data loaded events.

        BUG FIX #3-4: Added disconnect method for proper callback lifecycle.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        with self._lock:
            # Remove by identity
            for i, existing in enumerate(self._drillhole_callbacks):
                if existing is callback:
                    del self._drillhole_callbacks[i]
                    logger.debug(f"Unregistered drillhole callback, remaining: {len(self._drillhole_callbacks)}")
                    return True
            return False
    
    # ========================================================================
    # Block Model Methods
    # ========================================================================
    
    def register_block_model(
        self,
        block_model: Any,
        source_panel: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
        model_id: Optional[str] = None,
        set_as_current: bool = True,
    ) -> bool:
        """
        Register block model with multi-model support.

        Args:
            block_model: Block model object (BlockModel or DataFrame)
            source_panel: Source panel identifier
            metadata: Optional custom metadata
            source_id: Optional source ID to prevent signal loops
            model_id: Optional unique identifier (auto-generated if None)
            set_as_current: Whether to set this as the active model (default True)

        Returns:
            True if successful, False if validation failed
        """
        with self._lock:
            # Validate
            valid, error = self._validate_block_model(block_model)
            if not valid:
                logger.error(f"Validation failed for block model: {error}")
                return False

            # Generate model_id if not provided
            if model_id is None:
                model_id = self._generate_block_model_id(source_panel, metadata)

            # Sanitize and ensure uniqueness
            model_id = self._sanitize_model_id(model_id)
            model_id = self._ensure_unique_model_id(model_id)

            # Build storage key
            storage_key = f"block_model_{model_id}"

            # Store
            meta = self._build_metadata("block_model", block_model, source_panel, metadata)
            self._data_store[storage_key] = {
                'data': block_model,
                'metadata': meta
            }
            self._status_flags["block_model"] = True  # Keep for backward compat

            # Update index
            self._block_model_index[model_id] = storage_key

            # Store last registered ID for signal emission
            self._last_registered_model_id = model_id

            # Set as current if requested
            if set_as_current or self._current_block_model_id is None:
                self._current_block_model_id = model_id

            # Warn if many models registered
            if len(self._block_model_index) > 20:
                logger.warning(
                    f"Registry contains {len(self._block_model_index)} block models. "
                    "Consider clearing old models to free memory."
                )

            logger.info(f"Registered block model '{model_id}' from {source_panel} "
                       f"(current={set_as_current})")
            return True
    
    def get_block_model(self, model_id: Optional[str] = None, copy_data: bool = True) -> Optional[Any]:
        """
        Get block model - retrieves specific model or current model.

        Args:
            model_id: Model identifier. If None, returns current active model.
            copy_data: If True, return a copy to prevent mutation

        Returns:
            Block model or None
        """
        with self._lock:
            # If no model_id specified, return current model
            if model_id is None or model_id == "current":
                model_id = self._current_block_model_id

            if model_id is None:
                # Check for legacy "block_model" key (backward compat)
                if "block_model" in self._data_store:
                    return self.get_data("block_model", copy_data=copy_data)
                return None

            # Lookup storage key
            if model_id not in self._block_model_index:
                return None

            storage_key = self._block_model_index[model_id]
            return self.get_data(storage_key, copy_data=copy_data)

    def get_all_block_models(self) -> Dict[str, Any]:
        """
        Get all registered block models.

        Returns:
            Dict mapping {model_id: model_data}
        """
        with self._lock:
            models = {}
            for model_id, storage_key in self._block_model_index.items():
                data = self.get_data(storage_key, copy_data=False)
                if data is not None:
                    models[model_id] = data
            return models

    def get_block_model_list(self) -> List[Dict[str, Any]]:
        """
        Get list of available block models with metadata.

        Returns:
            List of dicts with keys: model_id, source_panel, row_count,
                                     timestamp, is_current, columns
        """
        with self._lock:
            models = []
            for model_id, storage_key in self._block_model_index.items():
                meta = self.get_metadata(storage_key)
                if meta:
                    models.append({
                        'model_id': model_id,
                        'source_panel': meta.source_panel,
                        'row_count': meta.row_count,
                        'timestamp': meta.timestamp.isoformat() if meta.timestamp else None,
                        'is_current': model_id == self._current_block_model_id,
                        'columns': meta.columns,
                    })
            return models

    def set_current_block_model(self, model_id: str) -> bool:
        """
        Set the active/current block model.

        Args:
            model_id: Model identifier to set as current

        Returns:
            True if successful, False if model_id not found
        """
        with self._lock:
            if model_id not in self._block_model_index:
                logger.error(f"Cannot set current block model: '{model_id}' not found")
                return False

            self._current_block_model_id = model_id
            logger.info(f"Set current block model to '{model_id}'")
            return True

    def get_current_block_model_id(self) -> Optional[str]:
        """Get the ID of the currently active block model."""
        with self._lock:
            return self._current_block_model_id

    def get_last_registered_model_id(self) -> Optional[str]:
        """Get the ID of the last registered block model (for signal emission)."""
        with self._lock:
            return self._last_registered_model_id

    def get_block_model_metadata(self, model_id: Optional[str] = None) -> Optional[DataMetadata]:
        """
        Get block model metadata.

        Args:
            model_id: Model identifier. If None, returns current model metadata.
        """
        with self._lock:
            if model_id is None:
                model_id = self._current_block_model_id

            if model_id is None or model_id not in self._block_model_index:
                # Fallback to legacy "block_model" key
                return self.get_metadata("block_model")

            storage_key = self._block_model_index[model_id]
            return self.get_metadata(storage_key)
    
    def has_block_model(self) -> bool:
        """Check if any block model is available."""
        return self._status_flags.get("block_model", False)

    def clear_block_model(self, model_id: Optional[str] = None) -> None:
        """
        Clear block model(s).

        Args:
            model_id: Specific model to clear. If None, clears ALL models.
        """
        with self._lock:
            if model_id is None:
                # Clear all block models
                for storage_key in list(self._block_model_index.values()):
                    self.clear_data(storage_key)
                self._block_model_index.clear()
                self._current_block_model_id = None
                self._last_registered_model_id = None
                self._status_flags["block_model"] = False
                logger.info("Cleared all block models")
            else:
                # Clear specific model
                if model_id in self._block_model_index:
                    storage_key = self._block_model_index[model_id]
                    self.clear_data(storage_key)
                    del self._block_model_index[model_id]

                    # If we cleared current model, reset pointer
                    if self._current_block_model_id == model_id:
                        if self._block_model_index:
                            self._current_block_model_id = next(iter(self._block_model_index))
                        else:
                            self._current_block_model_id = None
                            self._status_flags["block_model"] = False

                    # If we cleared the last registered model, clear that tracking too
                    if self._last_registered_model_id == model_id:
                        self._last_registered_model_id = None

                    logger.info(f"Cleared block model '{model_id}'")
    
    def connect_block_model_loaded(self, callback: Callable[[Any], None]) -> None:
        """Register callback for block model loaded events."""
        with self._lock:
            if callback not in self._block_model_callbacks:
                self._block_model_callbacks.append(callback)

    def _generate_block_model_id(self, source_panel: str, metadata: Optional[Dict]) -> str:
        """
        Auto-generate block model ID from source and metadata.

        Strategy:
        - SGSIM: "sgsim_{variable}_{statistic}" e.g. "sgsim_FE_PCT_mean"
        - Kriging: "kriging_{variable}" e.g. "kriging_CU_PCT"
        - Import: "imported_{filename}" e.g. "imported_Au_model"
        - Builder: "{panel}_{timestamp}" e.g. "builder_20260214_1430"
        """
        from datetime import datetime

        if metadata:
            # SGSIM pattern
            if 'variable' in metadata and 'statistic' in metadata:
                var = metadata['variable'].replace(' ', '_')
                stat = metadata['statistic'].replace(' ', '_')
                return f"sgsim_{var}_{stat}"

            # Kriging pattern
            if 'variable' in metadata and 'kriging' in source_panel.lower():
                var = metadata['variable'].replace(' ', '_')
                return f"kriging_{var}"

            # Import pattern
            if 'source_file' in metadata:
                from pathlib import Path
                filename = Path(metadata['source_file']).stem
                return f"imported_{filename}"

        # Default: panel + timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        panel = source_panel.replace(' ', '_').lower()
        return f"{panel}_{ts}"

    def _sanitize_model_id(self, model_id: str) -> str:
        """Sanitize model ID to be filesystem-safe."""
        import re
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', model_id)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized or "model"

    def _ensure_unique_model_id(self, model_id: str) -> str:
        """Ensure model_id is unique by adding counter if needed."""
        if model_id not in self._block_model_index:
            return model_id

        counter = 2
        while f"{model_id}_{counter}" in self._block_model_index:
            counter += 1

        return f"{model_id}_{counter}"

    # ========================================================================
    # Domain Model Methods
    # ========================================================================
    
    def register_domain_model(
        self,
        domain_model: Any,
        source_panel: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> bool:
        """
        Register domain model - thread-safe.
        
        Args:
            domain_model: Domain model object
            source_panel: Source panel identifier
            metadata: Optional custom metadata
            source_id: Optional source ID to prevent signal loops
            
        Returns:
            True if successful, False if validation failed
        """
        with self._lock:
            # Store (domain models don't have strict validation)
            meta = self._build_metadata("domain_model", domain_model, source_panel, metadata)
            self._data_store["domain_model"] = {
                'data': domain_model,
                'metadata': meta
            }
            self._status_flags["domain_model"] = True
            
            logger.info(f"Registered domain model from {source_panel}")
            return True
    
    def get_domain_model(self, copy_data: bool = True) -> Optional[Any]:
        """
        Get domain model - thread-safe read.
        
        Args:
            copy_data: If True, return a copy to prevent mutation
            
        Returns:
            Domain model or None
        """
        return self.get_data("domain_model", copy_data=copy_data)
    
    def get_domain_model_metadata(self) -> Optional[DataMetadata]:
        """Get domain model metadata."""
        return self.get_metadata("domain_model")
    
    def has_domain_model(self) -> bool:
        """Check if domain model is available."""
        return self.has_data("domain_model")
    
    def clear_domain_model(self) -> None:
        """Clear domain model."""
        self.clear_data("domain_model")
    
    def connect_domain_model_loaded(self, callback: Callable[[Any], None]) -> None:
        """Register callback for domain model loaded events."""
        with self._lock:
            if callback not in self._domain_model_callbacks:
                self._domain_model_callbacks.append(callback)
    
    # ========================================================================
    # Validation Helpers (Public API)
    # ========================================================================
    
    def validate_drillhole_data(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Public validation method for drillhole data."""
        return self._validate_drillholes(data)
    
    def validate_block_model(self, block_model: Any) -> Tuple[bool, Optional[str]]:
        """Public validation method for block model."""
        return self._validate_block_model(block_model)
    
    def check_data_integrity(self) -> Dict[str, Any]:
        """Check integrity of stored data."""
        results = {
            "checks_passed": 0,
            "checks_failed": 0,
            "warnings": [],
            "errors": [],
        }
        
        with self._lock:
            if self.has_drillhole_data():
                data = self.get_drillhole_data(copy_data=False)
                valid, error = self.validate_drillhole_data(data or {})
                if valid:
                    results["checks_passed"] += 1
                else:
                    results["checks_failed"] += 1
                    results["errors"].append(f"Drillhole data invalid: {error}")
            else:
                results["warnings"].append("Drillhole data not loaded")

            if self.has_block_model():
                data = self.get_block_model(copy_data=False)
                valid, error = self.validate_block_model(data)
                if valid:
                    results["checks_passed"] += 1
                else:
                    results["checks_failed"] += 1
                    results["errors"].append(f"Block model invalid: {error}")
            else:
                results["warnings"].append("Block model not loaded")
        
        return results
    
    def get_status_summary(self) -> Dict[str, bool]:
        """Get status summary of all stored data."""
        with self._lock:
            return dict(self._status_flags)
    
    # ========================================================================
    # Data Provenance & Lineage Methods
    # ========================================================================
    
    def get_provenance_chain(self, key: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get the full provenance chain for a dataset.
        
        This shows the complete transformation history, allowing users
        to see exactly how this data was derived.
        
        Args:
            key: Data storage key
            
        Returns:
            List of transformation steps, or None if no provenance
            
        Example return:
            [
                {"data_type": "raw_assays", "source_file": "drillholes.csv", ...},
                {"data_type": "composites", "transformation_type": "compositing", 
                 "transformation_params": {"length": 2.0}, ...},
                {"data_type": "declustered", "transformation_type": "declustering", 
                 "transformation_params": {"cell_size": [50, 50, 10]}, ...}
            ]
        """
        with self._lock:
            metadata = self.get_metadata(key)
            if not metadata:
                return None
            return metadata.provenance_chain
    
    def get_provenance_summary(self, key: str) -> str:
        """
        Get a human-readable summary of data provenance.
        
        Returns something like:
            "Raw Data (drillholes.csv) → Composited (2m) → Declustered (50x50x10)"
        
        Args:
            key: Data storage key
            
        Returns:
            Human-readable provenance string
        """
        with self._lock:
            metadata = self.get_metadata(key)
            if not metadata:
                return "No data"
            
            parts = []
            
            # Add source info
            if metadata.source_file:
                parts.append(f"Source: {metadata.source_file}")
            else:
                parts.append(f"Source: {metadata.source_panel}")
            
            # Add transformation chain
            if metadata.provenance_chain:
                for step in metadata.provenance_chain:
                    trans_type = step.get('transformation_type', '')
                    if trans_type:
                        params = step.get('transformation_params', {})
                        param_str = ""
                        if params:
                            # Show key parameters
                            if 'length' in params:
                                param_str = f" ({params['length']}m)"
                            elif 'cell_size' in params:
                                cs = params['cell_size']
                                if isinstance(cs, list):
                                    param_str = f" ({cs[0]}x{cs[1]}x{cs[2]})"
                                else:
                                    param_str = f" ({cs})"
                        parts.append(f"{trans_type.title()}{param_str}")
            
            # Add current type
            parts.append(f"→ {metadata.data_type}")
            
            return " → ".join(parts)
    
    def register_with_provenance(
        self,
        key: str,
        data: Any,
        source_panel: str,
        source_file: Optional[str] = None,
        parent_key: Optional[str] = None,
        transformation_type: Optional[str] = None,
        transformation_params: Optional[Dict[str, Any]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register data with explicit provenance tracking.
        
        This method should be used when registering transformed data
        to maintain the full transformation chain.
        
        Args:
            key: Storage key for this data
            data: The data to store
            source_panel: Panel performing the registration
            source_file: Original source file (if this is raw data)
            parent_key: Key of the parent data (if this is transformed)
            transformation_type: Type of transformation applied
            transformation_params: Parameters used in transformation
            custom_metadata: Additional metadata
            
        Returns:
            True if successful
            
        Example:
            # Register composited data
            registry.register_with_provenance(
                key="composites",
                data=composited_df,
                source_panel="CompositingWindow",
                parent_key="raw_assays",
                transformation_type="compositing",
                transformation_params={"length": 2.0, "method": "length_weighted"}
            )
        """
        with self._lock:
            if not self._validate(key, data):
                logger.error(f"Validation failed for {key}")
                return False
            
            # Build metadata with provenance
            meta = self._build_metadata(
                data_type=key,
                data=data,
                source_panel=source_panel,
                custom_metadata=custom_metadata,
                source_file=source_file,
                parent_data_key=parent_key,
                transformation_type=transformation_type,
                transformation_params=transformation_params,
            )
            
            # Store
            self._data_store[key] = {
                'data': data,
                'metadata': meta
            }
            self._status_flags[key] = True
            
            # Log provenance info
            if transformation_type:
                logger.info(f"Registered {key} from {source_panel} "
                           f"(transformed from {parent_key} via {transformation_type})")
            else:
                logger.info(f"Registered {key} from {source_panel} "
                           f"(source: {source_file or 'unknown'})")
            
            return True
    
    def get_all_data_sources(self) -> List[Dict[str, Any]]:
        """
        Get information about all available data sources.
        
        This is useful for populating data source selection dropdowns.
        
        Returns:
            List of data source info dicts with keys:
            - key: Storage key
            - data_type: Type of data
            - source_panel: Source panel
            - source_file: Original file (if known)
            - row_count: Number of rows
            - transformation_type: Transformation applied (if any)
            - has_provenance: Whether full provenance is available
        """
        sources = []
        
        with self._lock:
            for key, item in self._data_store.items():
                if not item:
                    continue
                    
                metadata = item.get('metadata')
                if not metadata:
                    continue
                
                sources.append({
                    'key': key,
                    'data_type': metadata.data_type,
                    'source_panel': metadata.source_panel,
                    'source_file': metadata.source_file,
                    'row_count': metadata.row_count,
                    'timestamp': metadata.timestamp.isoformat() if metadata.timestamp else None,
                    'transformation_type': metadata.transformation_type,
                    'has_provenance': metadata.provenance_chain is not None,
                    'columns': metadata.columns,
                })
        
        return sources
    
    def get_data_lineage_tree(self) -> Dict[str, Any]:
        """
        Get the full data lineage as a tree structure.
        
        Returns a nested dictionary showing how data flows
        through the system.
        
        Returns:
            Dictionary with data lineage tree
        """
        tree = {
            'sources': [],  # Root data sources (no parent)
            'derived': {},  # Data derived from other data
        }
        
        with self._lock:
            for key, item in self._data_store.items():
                if not item:
                    continue
                    
                metadata = item.get('metadata')
                if not metadata:
                    continue
                
                node = {
                    'key': key,
                    'data_type': metadata.data_type,
                    'source_panel': metadata.source_panel,
                    'transformation_type': metadata.transformation_type,
                    'row_count': metadata.row_count,
                }
                
                if metadata.parent_data_key:
                    # This is derived data
                    parent = metadata.parent_data_key
                    if parent not in tree['derived']:
                        tree['derived'][parent] = []
                    tree['derived'][parent].append(node)
                else:
                    # This is a root source
                    tree['sources'].append(node)
        
        return tree
    
    # ========================================================================
    # Category Label Maps (for legend label aliasing)
    # ========================================================================
    
    def get_category_label_maps(self) -> Dict[str, Dict[str, str]]:
        """
        Get all category label mappings.
        
        Returns:
            Dictionary mapping namespace -> (code -> label)
            Example: {"drillholes.lithology": {"BIF": "Banded Iron Formation"}}
        """
        with self._lock:
            data = self.get_data("category_label_maps", copy_data=False)
            if data is None:
                return {}
            return dict(data)
    
    def get_category_label(self, namespace: str, code: str) -> str:
        """
        Get the display label for a category code.
        
        Args:
            namespace: Namespace identifier (e.g., "drillholes.lithology")
            code: Category code (e.g., "BIF")
            
        Returns:
            Display label if mapped, otherwise the code as-is
        """
        with self._lock:
            maps = self.get_category_label_maps()
            namespace_map = maps.get(namespace, {})
            return namespace_map.get(code, str(code))
    
    def set_category_label(self, namespace: str, code: str, label: str, source: str = "user") -> None:
        """
        Set a display label for a category code.
        
        This creates a per-project alias mapping without mutating source data.
        
        Args:
            namespace: Namespace identifier (e.g., "drillholes.lithology")
            code: Category code (e.g., "BIF")
            label: Display label (e.g., "Banded Iron Formation")
            source: Source of the change (for audit trail)
        """
        with self._lock:
            # Get existing maps or create new
            maps = self.get_category_label_maps()
            
            # Ensure namespace exists
            if namespace not in maps:
                maps[namespace] = {}
            
            # Set the label
            maps[namespace][code] = label
            
            # Store back (creates new metadata entry if first time)
            success = self.register_model(
                "category_label_maps",
                maps,
                metadata={"source": source, "namespace": namespace, "code": code, "label": label},
                source_panel="LegendManager"
            )
            
            if success:
                logger.info(f"Set category label: {namespace}.{code} = '{label}'")
    
    def clear_category_label(self, namespace: str, code: str) -> None:
        """
        Clear a category label mapping (reset to showing code).
        
        Args:
            namespace: Namespace identifier (e.g., "drillholes.lithology")
            code: Category code to reset
        """
        with self._lock:
            maps = self.get_category_label_maps()
            
            if namespace in maps and code in maps[namespace]:
                del maps[namespace][code]
                
                # Clean up empty namespace
                if not maps[namespace]:
                    del maps[namespace]
                
                # Store back
                self.register_model("category_label_maps", maps, source_panel="LegendManager")
                logger.info(f"Cleared category label: {namespace}.{code}")
    
    # ========================================================================
    # Drillhole Validation State Management
    # ========================================================================
    
    def set_drillholes_validation_state(
        self,
        status: str,
        timestamp: str,
        config_hash: str,
        fatal_count: int,
        warn_count: int,
        info_count: int = 0,
        violations_summary: Optional[Dict[str, Any]] = None,
        tables_validated: Optional[List[str]] = None,
        schema_errors: Optional[List[str]] = None,
        excluded_rows: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """
        Store drillhole validation state for downstream engines to query.

        This enables compositing and other engines to check if validation
        has been run and whether data is safe to process.

        Args:
            status: One of "NOT_RUN", "PASS", "WARN", "FAIL"
            timestamp: ISO8601 timestamp of validation run
            config_hash: SHA256 hash of ValidationConfig for reproducibility
            fatal_count: Number of ERROR-level violations
            warn_count: Number of WARNING-level violations
            info_count: Number of INFO-level violations
            violations_summary: Optional breakdown by table/type
            tables_validated: List of tables that were validated
            schema_errors: List of schema-level errors encountered
            excluded_rows: Dict mapping table name -> list of row indices that have
                          unresolved ERROR violations and should be excluded from
                          downstream processing (compositing, kriging, etc.)
        """
        with self._lock:
            state = {
                "status": status,
                "timestamp": timestamp,
                "config_hash": config_hash,
                "fatal_count": fatal_count,
                "warn_count": warn_count,
                "info_count": info_count,
                "violations_summary": violations_summary or {},
                "tables_validated": tables_validated or [],
                "schema_errors": schema_errors or [],
                "excluded_rows": excluded_rows or {},
            }
            
            self._data_store["drillholes_validation_state"] = {
                "data": state,
                "metadata": None  # Lightweight storage, no full DataMetadata needed
            }
            
            logger.info(
                f"Drillhole validation state updated: status={status}, "
                f"fatal={fatal_count}, warn={warn_count}"
            )
    
    def get_drillholes_validation_state(self) -> Optional[Dict[str, Any]]:
        """
        Get the current drillhole validation state.
        
        Returns:
            Dictionary with validation state or None if never validated.
            Keys: status, timestamp, config_hash, fatal_count, warn_count,
                  info_count, violations_summary, tables_validated, schema_errors
        """
        with self._lock:
            item = self._data_store.get("drillholes_validation_state")
            if item is None:
                return None
            return item.get("data")
    
    def get_drillholes_validation_status(self) -> str:
        """
        Quick check for validation status.
        
        Returns:
            "NOT_RUN" if never validated, otherwise "PASS", "WARN", or "FAIL"
        """
        state = self.get_drillholes_validation_state()
        if state is None:
            return "NOT_RUN"
        return state.get("status", "NOT_RUN")
    
    def is_drillholes_validated(self) -> bool:
        """
        Check if drillholes have been validated (regardless of result).
        
        Returns:
            True if validation has been run at least once.
        """
        return self.get_drillholes_validation_status() != "NOT_RUN"
    
    def is_drillholes_valid(self) -> bool:
        """
        Check if drillholes are valid (PASS or WARN, not FAIL).
        
        Returns:
            True if validation passed or passed with warnings.
            False if validation failed or was never run.
        """
        status = self.get_drillholes_validation_status()
        return status in ("PASS", "WARN")
    
    def clear_drillholes_validation_state(self) -> None:
        """Clear validation state (e.g., when new data is loaded)."""
        with self._lock:
            if "drillholes_validation_state" in self._data_store:
                del self._data_store["drillholes_validation_state"]
                logger.info("Drillhole validation state cleared")

    def get_estimation_ready_data(
        self,
        data_type: str = "assays",
        raise_on_block: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Get estimation-ready data with HARD VALIDATION GATE.

        USE THIS METHOD IN ALL ESTIMATION PANELS:
        - Kriging (ordinary, simple, universal, indicator)
        - Simulation (SGSIM, SISIM, etc.)
        - Variogram modeling
        - Resource estimation/classification

        This method:
        1. BLOCKS if validation not run or FAILED
        2. Filters out all rows with ERROR violations
        3. Returns clean data ready for geostatistics
        4. Logs access for audit trail

        Args:
            data_type: Which data to return ("assays", "composites", "lithology")
            raise_on_block: If True, raises ValueError when blocked.
                           If False, returns None with logged warning.

        Returns:
            Filtered DataFrame ready for estimation, or None if blocked

        Raises:
            ValueError: If raise_on_block=True and data not ready

        Example:
            # In kriging panel:
            try:
                df = registry.get_estimation_ready_data("composites")
            except ValueError as e:
                QMessageBox.critical(self, "Cannot Run Kriging", str(e))
                return
        """
        # Check validation gate
        allowed, message = self.require_validation_for_estimation()

        if not allowed:
            logger.error(f"ESTIMATION BLOCKED: {message}")
            if raise_on_block:
                raise ValueError(message)
            return None

        # Get filtered data
        try:
            data = self.get_drillhole_data(mode="estimation_ready", copy_data=True)
        except ValueError as e:
            logger.error(f"ESTIMATION BLOCKED: {e}")
            if raise_on_block:
                raise
            return None

        if data is None:
            if raise_on_block:
                raise ValueError("No drillhole data available")
            return None

        # Extract requested data type
        df = data.get(data_type)
        if df is None:
            # Try common variations
            for key in [data_type, data_type.lower(), data_type.upper()]:
                df = data.get(key)
                if df is not None:
                    break

        if df is None or (hasattr(df, 'empty') and df.empty):
            if raise_on_block:
                raise ValueError(f"No {data_type} data available for estimation")
            return None

        # Log successful access
        logger.info(
            f"Estimation-ready {data_type} data accessed: {len(df)} rows"
        )

        return df

    # ========================================================================
    # DATA LINEAGE ENFORCEMENT (Geostatistical Pipeline Gates)
    # ========================================================================
    
    def has_composites(self) -> bool:
        """
        Check if composited data exists in the registry.
        
        Returns:
            True if composites exist and are non-empty.
        """
        with self._lock:
            dh_data = self.get_drillhole_data(copy_data=False)
            if not dh_data:
                return False
            composites = dh_data.get('composites')
            if composites is None:
                return False
            if hasattr(composites, 'empty'):
                return not composites.empty
            return True
    
    def get_validated_composites(
        self, 
        require_validation: bool = True,
        allow_validation_override: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Single point of truth for retrieving geostatistically-ready data.
        
        ENFORCEMENT: 
        - Returns ONLY composited data (never raw assays)
        - Requires validation unless explicitly bypassed
        - Attaches lineage metadata to DataFrame attrs
        
        This method implements the data lineage gate for the geostatistical
        pipeline. All estimation engines (kriging, SGSIM, etc.) should use
        this method to ensure they never accidentally process raw data.
        
        Args:
            require_validation: If True, raises error if validation not passed
            allow_validation_override: If True, allows WARN status to proceed
            
        Returns:
            Composited DataFrame with provenance attrs, or None if no data
            
        Raises:
            ValueError: If validation required but not passed
            ValueError: If no composited data found
            
        Example:
            # Standard usage - enforces all gates
            df = registry.get_validated_composites()
            
            # Skip validation (e.g., for testing)
            df = registry.get_validated_composites(require_validation=False)
        """
        with self._lock:
            # GATE 1: Validation check
            if require_validation:
                validation_status = self.get_drillholes_validation_status()
                
                if validation_status == "NOT_RUN":
                    raise ValueError(
                        "LINEAGE GATE FAILED: Drillhole validation has not been run. "
                        "Run validation from the QC panel before proceeding to estimation. "
                        "This gate ensures data quality for JORC/SAMREC compliance."
                    )
                
                if validation_status == "FAIL":
                    raise ValueError(
                        "LINEAGE GATE FAILED: Drillhole validation FAILED with errors. "
                        "Fix validation errors before proceeding to estimation. "
                        "Check the QC panel for details on validation failures."
                    )
                
                if validation_status == "WARN" and not allow_validation_override:
                    logger.warning(
                        "LINEAGE WARNING: Drillhole validation passed with warnings. "
                        "Proceeding with estimation, but review warnings for JORC/SAMREC compliance."
                    )
            
            # GATE 2: Compositing check
            dh_data = self.get_drillhole_data(copy_data=True)
            if not dh_data:
                raise ValueError(
                    "LINEAGE GATE FAILED: No drillhole data loaded in registry. "
                    "Import drillhole data before proceeding."
                )
            
            composites = dh_data.get('composites')
            if composites is None:
                raise ValueError(
                    "LINEAGE GATE FAILED: No composited data found. "
                    "Run compositing before proceeding to estimation. "
                    "Raw assays cannot be used directly for kriging/simulation - "
                    "this would violate change-of-support principles."
                )
            
            if hasattr(composites, 'empty') and composites.empty:
                raise ValueError(
                    "LINEAGE GATE FAILED: Composited data is empty. "
                    "Re-run compositing or check source data."
                )
            
            # ATTACH PROVENANCE METADATA to DataFrame
            # This enables downstream checks without re-querying registry
            validation_state = self.get_drillholes_validation_state() or {}
            composites.attrs['source_type'] = 'composites'
            composites.attrs['validation_status'] = validation_state.get('status', 'NOT_RUN')
            composites.attrs['validation_config_hash'] = validation_state.get('config_hash', '')
            composites.attrs['lineage_gate_passed'] = True
            composites.attrs['lineage_timestamp'] = datetime.now().isoformat()
            
            logger.info(
                f"LINEAGE: Returning validated composites. "
                f"Validation status: {validation_state.get('status', 'NOT_RUN')}, "
                f"Rows: {len(composites)}"
            )
            
            return composites
    
    def get_estimation_ready_data(
        self,
        prefer_declustered: bool = True,
        require_validation: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get the best available data for estimation (kriging/simulation).
        
        Priority order:
        1. Declustered composites (if available and prefer_declustered=True)
        2. Composites
        
        NEVER returns raw assays.
        
        Args:
            prefer_declustered: If True, returns declustered data if available
            require_validation: If True, enforces validation gate
            
        Returns:
            DataFrame ready for estimation with provenance attrs
            
        Raises:
            ValueError: If lineage gates fail
        """
        with self._lock:
            # First, ensure composites exist and are validated
            # This will raise ValueError if gates fail
            composites = self.get_validated_composites(
                require_validation=require_validation
            )
            
            # Check for declustered data
            if prefer_declustered:
                declustering_results = self.get_data("declustering_results", copy_data=True)
                if declustering_results:
                    declustered_df = declustering_results.get('weighted_dataframe')
                    if declustered_df is not None and not declustered_df.empty:
                        # Attach provenance
                        declustered_df.attrs['source_type'] = 'declustered'
                        declustered_df.attrs['parent_source'] = 'composites'
                        declustered_df.attrs['lineage_gate_passed'] = True
                        declustered_df.attrs['lineage_timestamp'] = datetime.now().isoformat()
                        
                        logger.info(
                            f"LINEAGE: Returning declustered data. Rows: {len(declustered_df)}"
                        )
                        return declustered_df
            
            return composites