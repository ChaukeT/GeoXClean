
"""
Central Data Registry
=====================

This module provides a lightweight singleton that stores drillhole data,
block models, estimation results, schedules, and other artefacts so that
panels under *Drillholes* and *Estimates* can talk to each other through a
shared memory buffer.  Heavy computation continues to live inside the
controller/workers; the registry simply keeps references, metadata, and
optional Qt signals so dependant panels can refresh automatically.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from .data_registry_simple import DataRegistrySimple, DataMetadata

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QObject, pyqtSignal

    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - Qt not available during some tests
    _QT_AVAILABLE = False
    pyqtSignal = object  # type: ignore
    QObject = object  # type: ignore


class _SignalEmitter(QObject if _QT_AVAILABLE else object):
    """Isolated QObject holding all registry signals."""

    if _QT_AVAILABLE:
        drillholeDataLoaded = pyqtSignal(object)
        drillholeDataCleared = pyqtSignal()

        # Legacy block model signals (maintained for backward compatibility)
        blockModelLoaded = pyqtSignal(object)
        blockModelGenerated = pyqtSignal(object)
        blockModelClassified = pyqtSignal(object)
        blockModelCleared = pyqtSignal()

        # New multi-model block model signals
        blockModelLoadedEx = pyqtSignal(str, object)  # (model_id, model_data)
        blockModelGeneratedEx = pyqtSignal(str, object)
        blockModelClassifiedEx = pyqtSignal(str, object)
        blockModelClearedEx = pyqtSignal(str)  # (model_id or "" for all)
        currentBlockModelChanged = pyqtSignal(str)  # (model_id)

        domainModelLoaded = pyqtSignal(object)
        domainModelCleared = pyqtSignal()
        contactSetLoaded = pyqtSignal(object)

        variogramResultsLoaded = pyqtSignal(object)
        declusteringResultsLoaded = pyqtSignal(object)
        transformationMetadataLoaded = pyqtSignal(object)
        krigingResultsLoaded = pyqtSignal(object)
        simpleKrigingResultsLoaded = pyqtSignal(object)
        cokrigingResultsLoaded = pyqtSignal(object)
        indicatorKrigingResultsLoaded = pyqtSignal(object)
        universalKrigingResultsLoaded = pyqtSignal(object)
        softKrigingResultsLoaded = pyqtSignal(object)
        rbfResultsLoaded = pyqtSignal(object)
        sgsimResultsLoaded = pyqtSignal(object)

        geometResultsLoaded = pyqtSignal(object)
        geometOreTypesLoaded = pyqtSignal(object)
        resourceCalculated = pyqtSignal(object)
        
        # Geological Model signals (Phase 6 - UI integration)
        geologicalModelUpdated = pyqtSignal(object)  # Surfaces/solids generated or updated
        geologicalSurfacesLoaded = pyqtSignal(object)  # Implicit surfaces result
        geologicalSolidsLoaded = pyqtSignal(object)  # Voxel solids result
        
        # LoopStructural Model signals (Industry-grade JORC/SAMREC compliant)
        loopstructuralModelLoaded = pyqtSignal(object)  # LoopStructural model result
        loopstructuralComplianceChecked = pyqtSignal(object)  # Compliance audit report

        pitOptimizationResultsLoaded = pyqtSignal(object)
        scheduleGenerated = pyqtSignal(object)
        irrResultsLoaded = pyqtSignal(object)
        reconciliationResultsLoaded = pyqtSignal(object)
        haulageEvaluationLoaded = pyqtSignal(object)
        experimentResultsLoaded = pyqtSignal(object)
        categoryLabelMapsChanged = pyqtSignal(str)  # namespace

        def __init__(self) -> None:  # pragma: no cover - trivial
            super().__init__()
            logger.debug("DataRegistry signal emitter initialised")


class DataRegistry(QObject if _QT_AVAILABLE else object, DataRegistrySimple):
    """
    GUI Data Registry: Adds Signals to the base logic.
    
    Implements Singleton pattern for legacy compatibility.
    Inherits storage and validation from DataRegistrySimple.
    """

    _instance: Optional["DataRegistry"] = None

    _DATA_TYPES = [
        "drillhole_data",
        "block_model",
        "domain_model",
        "contact_set",
        "variogram_results",
        "transformation_metadata",
        "kriging_results",
        "sgsim_results",
        "simple_kriging_results",
        "cokriging_results",
        "indicator_kriging_results",
        "universal_kriging_results",
        "soft_kriging_results",
        "classified_block_model",
        "resource_summary",
        "geomet_results",
        "geomet_ore_types",
        "pit_optimization_results",
        "schedule",
        "irr_results",
        "reconciliation_results",
        "haulage_evaluation",
        "experiment_results",
    ]

    @classmethod
    def instance(cls) -> "DataRegistry":
        """Global Access Point (Legacy Support)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def get_existing(cls) -> Optional["DataRegistry"]:
        return cls._instance

    def __new__(cls):
        """Thread-safe Singleton Pattern."""
        if cls._instance is None:
            cls._instance = super(DataRegistry, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize both QObject and DataRegistrySimple."""
        # Prevent double-initialization in Singletons
        if hasattr(self, "_initialised"):
            return

        # Initialize BOTH parents
        if _QT_AVAILABLE:
            QObject.__init__(self)
        DataRegistrySimple.__init__(self)

        self._initialised = True
        
        # Initialize Qt signals
        self._signals: Optional[_SignalEmitter] = None
        if _QT_AVAILABLE:
            try:
                from PyQt6.QtWidgets import QApplication

                if QApplication.instance():
                    self._signals = _SignalEmitter()
                    logger.info("DataRegistry: Qt signal support enabled")
            except Exception as exc:
                logger.debug("DataRegistry: Qt signals disabled (%s)", exc)

        # Transformers storage (special case - stores transformer objects, not migrated to base class yet)
        self._transformers: Optional[Dict[str, Any]] = None
        
        # Note: All other data types now use base class storage via register_model/get_data
        # Legacy status flags are managed by base class

        logger.info("DataRegistry initialised (inherits from DataRegistrySimple)")

    # ---------------------------------------------------------------- helpers
    def _emit(self, signal_name: str, *args: Any) -> None:
        """Emit Qt signal (outside lock to prevent deadlocks)."""
        if not self._signals:
            return
        signal = getattr(self._signals, signal_name, None)
        if signal is not None:
            try:
                signal.emit(*args)  # type: ignore[attr-defined]
            except Exception as e:  # pragma: no cover - signal exceptions ignored
                # BUG FIX #7: Log at WARNING level for visibility, not DEBUG
                logger.warning(f"Failed to emit signal '{signal_name}': {e}")
        else:
            # BUG FIX #16: Warn when signal doesn't exist
            logger.warning(f"Signal '{signal_name}' not found in _SignalEmitter - check signal definition")

    def _set_flag(self, key: str, value: bool) -> None:
        """Set status flag (for legacy methods)."""
        if key in self._status_flags:
            self._status_flags[key] = value

    # ---------------------------------------------------------------- signals
    @property
    def drillholeDataLoaded(self):
        return self._signals.drillholeDataLoaded if self._signals else None

    @property
    def drillholeDataCleared(self):
        return self._signals.drillholeDataCleared if self._signals else None

    @property
    def blockModelLoaded(self):
        return self._signals.blockModelLoaded if self._signals else None

    @property
    def blockModelGenerated(self):
        return self._signals.blockModelGenerated if self._signals else None

    @property
    def blockModelClassified(self):
        return self._signals.blockModelClassified if self._signals else None

    @property
    def blockModelCleared(self):
        return self._signals.blockModelCleared if self._signals else None

    @property
    def domainModelLoaded(self):
        return self._signals.domainModelLoaded if self._signals else None

    @property
    def domainModelCleared(self):
        return self._signals.domainModelCleared if self._signals else None

    @property
    def contactSetLoaded(self):
        return self._signals.contactSetLoaded if self._signals else None

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._signals and hasattr(self._signals, name):
            return getattr(self._signals, name)
        raise AttributeError(name)

    # ---------------------------------------------------------------- drillhole
    def register_drillhole_data(
        self,
        data: Dict[str, Any],
        source_panel: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> bool:
        """
        Register drillhole data - calls base class then emits signal.
        
        TRF-011 COMPLIANCE: When new drillhole data is registered from a different
        source than GradeTransformer, existing transformers are invalidated to
        prevent stale transform usage.
        
        Args:
            data: Drillhole data dictionary
            source_panel: Source panel identifier
            metadata: Optional custom metadata
            source_id: Optional source ID to prevent signal loops
            
        Returns:
            True if successful, False if validation failed
        """
        logger.debug("DataRegistry.register_drillhole_data: STARTED")
        
        # TRF-011 COMPLIANCE: Check if transformers should be invalidated
        # Only invalidate if source is NOT the GradeTransformer (which manages its own transforms)
        # and if the data is fundamentally new (not just adding transform columns)
        if source_panel != "Grade Transformer" and self._transformers:
            # Check if this is a data reload vs transform update
            is_transform_update = (
                metadata is not None and 
                ('transformations' in metadata or 'transform_history' in metadata)
            )
            if not is_transform_update:
                n_transformers = len(self._transformers)
                self._transformers = None
                logger.info(
                    f"TRF-011 LINEAGE: Invalidated {n_transformers} transformer(s) due to "
                    f"new drillhole data from {source_panel}. Re-run transformations if needed."
                )
        
        # 1. Perform Logic (Inherited from Simple)
        logger.debug("DataRegistry.register_drillhole_data: Calling super()...")
        success = super().register_drillhole_data(data, source_panel, metadata, source_id)
        logger.debug(f"DataRegistry.register_drillhole_data: super() returned success={success}")
        
        # 2. Emit Signal (GUI specific) - OUTSIDE lock to prevent deadlocks
        if success:
            # Notify callbacks
            logger.debug(f"DataRegistry.register_drillhole_data: Notifying {len(self._drillhole_callbacks)} callbacks...")
            for i, callback in enumerate(self._drillhole_callbacks):
                try:
                    logger.debug(f"DataRegistry.register_drillhole_data: Calling callback {i}...")
                    callback(data)
                    logger.debug(f"DataRegistry.register_drillhole_data: Callback {i} completed")
                except Exception:
                    logger.warning("Drillhole callback failed", exc_info=True)
            
            # Emit Qt signal
            logger.debug("DataRegistry.register_drillhole_data: Emitting drillholeDataLoaded signal...")
            self._emit("drillholeDataLoaded", data)
            logger.debug("DataRegistry.register_drillhole_data: Signal emitted")
        
        logger.debug("DataRegistry.register_drillhole_data: FINISHED")
        return success

    def clear_drillhole_data(self) -> None:
        """
        Clear drillhole data and emit signal.
        
        LINEAGE ENFORCEMENT: Also clears downstream results that depend on
        drillhole data to prevent stale/inconsistent data from being used.
        
        Clears:
        - drillhole_data
        - declustering_results (depends on drillholes)
        - variogram_results (depends on drillholes)
        - transformers (TRF-011: depends on drillholes)
        - transformation_metadata (TRF-011: depends on drillholes)
        """
        # Clear drillhole data
        super().clear_drillhole_data()
        
        # LINEAGE: Clear dependent downstream results
        # Declustering is derived from drillhole data - must be invalidated
        if self.has_data("declustering_results"):
            self.clear_data("declustering_results")
            logger.info("LINEAGE: Cleared declustering_results (drillhole data cleared)")
        
        # Variogram results depend on drillhole data
        if self.has_data("variogram_results"):
            self.clear_data("variogram_results")
            logger.info("LINEAGE: Cleared variogram_results (drillhole data cleared)")
        
        # TRF-011 COMPLIANCE: Transformers depend on drillhole data - must be invalidated
        # Stale transforms can produce incorrect back-transformations
        if self._transformers:
            n_transformers = len(self._transformers)
            self._transformers = None
            logger.info(f"TRF-011 LINEAGE: Cleared {n_transformers} transformer(s) (drillhole data cleared)")
        
        # Clear transformation metadata as well
        if self.has_data("transformation_metadata"):
            self.clear_data("transformation_metadata")
            logger.info("TRF-011 LINEAGE: Cleared transformation_metadata (drillhole data cleared)")
        
        self._emit("drillholeDataCleared")

    # ---------------------------------------------------------------- block models
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
        # 1. Perform Logic (Inherited from Simple)
        success = super().register_block_model(
            block_model, source_panel, metadata, source_id, model_id, set_as_current
        )

        # 2. Emit Signal (GUI specific) - OUTSIDE lock to prevent deadlocks
        if success:
            # Get the actual model_id (might be auto-generated, sanitized, made unique)
            actual_model_id = super().get_last_registered_model_id()

            # Notify callbacks
            for callback in self._block_model_callbacks:
                try:
                    callback(block_model)
                except Exception:
                    logger.warning("Block model callback failed", exc_info=True)

            # Emit both old and new signals
            self._emit("blockModelLoaded", block_model)
            self._emit("blockModelLoadedEx", actual_model_id, block_model)

            if set_as_current:
                self._emit("currentBlockModelChanged", actual_model_id)

        return success

    def register_block_model_generated(
        self,
        block_model: Any,
        source_panel: str = "BlockModelBuilder",
        metadata: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
        set_as_current: bool = True,
    ) -> None:
        self.register_block_model(block_model, source_panel, metadata, model_id=model_id, set_as_current=set_as_current)
        # Get the actual model_id for new signal (the one just registered)
        actual_model_id = super().get_last_registered_model_id()
        self._emit("blockModelGenerated", block_model)
        self._emit("blockModelGeneratedEx", actual_model_id, block_model)

    # Alias for backwards compatibility
    def register_generated_block_model(
        self,
        block_model: Any,
        source_panel: str = "BlockModelBuilder",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Alias for register_block_model_generated (backwards compatibility)."""
        self.register_block_model_generated(block_model, source_panel, metadata)

    def register_classified_block_model(
        self,
        block_model: Any,
        source_panel: str = "ResourceClassificationPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register classified block model - delegates to base class then emits signal."""
        success = super().register_model("classified_block_model", block_model, metadata, source_panel)
        if success:
            self._emit("blockModelClassified", block_model)
        return success

    def register_resource_summary(
        self,
        summary: Any,
        source_panel: str = "ResourceReportingPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register resource summary - delegates to base class then emits signal."""
        success = super().register_model("resource_summary", summary, metadata, source_panel)
        if success:
            self._emit("resourceSummaryLoaded", summary)
        return success

    def get_resource_summary(self, copy_data: bool = True) -> Optional[Any]:
        """Get the most recent resource summary."""
        return self.get_model("resource_summary", copy_data)

    def get_block_model(self, model_id: Optional[str] = None, copy_data: bool = True) -> Optional[Any]:
        """
        Get block model - retrieves specific model or current model.

        Args:
            model_id: Model identifier. If None, returns current active model.
            copy_data: If True, return a copy to prevent mutation

        Returns:
            Block model or None
        """
        return super().get_block_model(model_id=model_id, copy_data=copy_data)

    def get_classified_block_model(self, copy_data: bool = True) -> Optional[Any]:
        """Get classified block model - delegates to base class."""
        return super().get_data("classified_block_model", copy_data=copy_data)

    def clear_block_model(self, model_id: Optional[str] = None) -> None:
        """
        Clear block model(s) and emit signal.

        Args:
            model_id: Specific model to clear. If None, clears ALL models.
        """
        super().clear_block_model(model_id=model_id)
        self._emit("blockModelCleared")
        self._emit("blockModelClearedEx", model_id or "")

    def get_all_block_models(self) -> Dict[str, Any]:
        """Get all registered block models."""
        return super().get_all_block_models()

    def get_block_model_list(self) -> List[Dict[str, Any]]:
        """Get list of available block models with metadata."""
        return super().get_block_model_list()

    def set_current_block_model(self, model_id: str) -> bool:
        """
        Set the active/current block model.

        Args:
            model_id: Model identifier to set as current

        Returns:
            True if successful, False if model_id not found
        """
        success = super().set_current_block_model(model_id)
        if success:
            self._emit("currentBlockModelChanged", model_id)
        return success

    def get_current_block_model_id(self) -> Optional[str]:
        """Get the ID of the currently active block model."""
        return super().get_current_block_model_id()

    def get_last_registered_model_id(self) -> Optional[str]:
        """Get the ID of the last registered block model (for signal emission)."""
        return super().get_last_registered_model_id()

    # ---------------------------------------------------------------- geology/domain
    def register_domain_model(
        self,
        domain_model: Any,
        source_panel: str = "GeologyPanel",
        metadata: Optional[Dict[str, Any]] = None,
        source_id: Optional[str] = None,
    ) -> bool:
        """
        Register domain model - calls base class then emits signal.
        
        Args:
            domain_model: Domain model object
            source_panel: Source panel identifier
            metadata: Optional custom metadata
            source_id: Optional source ID to prevent signal loops
            
        Returns:
            True if successful, False if validation failed
        """
        # 1. Perform Logic (Inherited from Simple)
        success = super().register_domain_model(domain_model, source_panel, metadata, source_id)
        
        # 2. Emit Signal (GUI specific) - OUTSIDE lock to prevent deadlocks
        if success:
            # Notify callbacks
            for callback in self._domain_model_callbacks:
                try:
                    callback(domain_model)
                except Exception:
                    logger.warning("Domain model callback failed", exc_info=True)
            
            # Emit Qt signal
            self._emit("domainModelLoaded", domain_model)
        
        return success

    def clear_domain_model(self) -> None:
        """Clear domain model and emit signal."""
        super().clear_domain_model()
        self._emit("domainModelCleared")

    def register_contact_set(
        self,
        contact_set: Any,
        source_panel: str = "GeologyPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register contact set - delegates to base class then emits signal."""
        success = super().register_model("contact_set", contact_set, metadata, source_panel)
        if success:
            self._emit("contactSetLoaded", contact_set)
        return success

    def get_contact_set(self, copy_data: bool = True) -> Optional[Any]:
        """Get contact set - delegates to base class."""
        return super().get_data("contact_set", copy_data=copy_data)

    # ---------------------------------------------------------------- variogram & estimation
    def register_variogram_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "VariogramPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register variogram results - stores per variable to support co-kriging."""
        # Extract variable name from results
        variable_name = results.get('variable', 'default')
        
        # Store variogram per variable
        key = f"variogram_results_{variable_name}"
        success = super().register_model(key, results, metadata, source_panel)
        
        # Also store as "variogram_results" for backward compatibility (latest one)
        super().register_model("variogram_results", results, metadata, source_panel)
        
        if success:
            self._emit("variogramResultsLoaded", results)
        return success

    def get_variogram_results(self, variable_name: Optional[str] = None, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get variogram results - can retrieve by variable name for co-kriging."""
        if variable_name:
            # Get variogram for specific variable
            key = f"variogram_results_{variable_name}"
            result = super().get_data(key, copy_data=copy_data)
            if result:
                return result
        
        # Fallback to latest variogram (backward compatibility)
        return super().get_data("variogram_results", copy_data=copy_data)

    # ---------------------------------------------------------------- declustering results
    def register_declustering_results(
        self,
        results: Tuple[pd.DataFrame, Dict[str, Any]],
        source_panel: str = "DeclusteringPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register declustering results with full provenance tracking.
        
        LINEAGE ENFORCEMENT: Captures complete transformation metadata including:
        - Parent data source (composites vs assays)
        - Validation status at time of processing
        - Cell size and configuration
        - Processing timestamp
        
        Args:
            results: Tuple of (weighted_dataframe, summary)
            source_panel: Source panel identifier
            metadata: Additional provenance metadata with required keys:
                - parent_data_key: 'composites' or 'assays'
                - transformation_type: 'declustering'
                - transformation_params: cell size, method, etc.
                - validation_status: drillhole validation status
                - source_was_raw_assays: True if declustered from raw assays
                
        Returns:
            True if successful, False if validation failed
        """
        df_weighted, summary = results
        
        # Build complete provenance metadata
        provenance = metadata or {}
        
        # MANDATORY PROVENANCE FIELDS
        parent_key = provenance.get('parent_data_key', 'unknown')
        transformation_type = provenance.get('transformation_type', 'declustering')
        validation_status = provenance.get('validation_status', 'NOT_RUN')
        source_was_raw = provenance.get('source_was_raw_assays', False)
        
        # Extract transformation params from metadata or summary
        transformation_params = provenance.get('transformation_params', {})
        if not transformation_params and isinstance(summary, dict):
            # Try to extract from summary
            transformation_params = {
                'cell_size': summary.get('cell_size_summary', 'unknown'),
                'method': 'cell_declustering',
            }
        
        # Build complete declust_data with provenance
        declust_data = {
            'weighted_dataframe': df_weighted,
            'summary': summary,
            'timestamp': datetime.now().isoformat(),
            # PROVENANCE METADATA
            'source_type': 'declustered',
            'parent_data_key': parent_key,
            'transformation_type': transformation_type,
            'transformation_params': transformation_params,
            'validation_status': validation_status,
            'source_was_raw_assays': source_was_raw,
            # AUDIT TRAIL
            'sample_count': provenance.get('sample_count', len(df_weighted) if hasattr(df_weighted, '__len__') else 0),
            'occupied_cells': provenance.get('occupied_cells', 0),
            'lineage_warning': 'CAUTION: Source was raw assays (not composites)' if source_was_raw else None,
        }

        # Store with provenance-aware registration
        success = super().register_with_provenance(
            key="declustering_results",
            data=declust_data,
            source_panel=source_panel,
            parent_key=parent_key if parent_key != 'unknown' else None,
            transformation_type=transformation_type,
            transformation_params=transformation_params,
            custom_metadata=provenance,
        )
        
        if success:
            self._emit("declusteringResultsLoaded", declust_data)
            logger.info(
                f"LINEAGE: Registered declustering results from {source_panel}. "
                f"Parent: {parent_key}, Validation: {validation_status}, "
                f"Raw assays: {source_was_raw}"
            )
        
        return success

    def get_declustering_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get declustering results - returns weighted DataFrame and summary."""
        return super().get_data("declustering_results", copy_data=copy_data)

    def get_declustered_dataframe(self, copy_data: bool = True) -> Optional[pd.DataFrame]:
        """Get just the declustered DataFrame with weights."""
        results = self.get_declustering_results(copy_data=copy_data)
        if results:
            return results.get('weighted_dataframe')
        return None

    def get_declustered_dataframe_with_lineage_check(
        self,
        require_composites_source: bool = False,
        copy_data: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get declustered DataFrame with lineage validation.
        
        DOWNSTREAM SAFETY: This method ensures that estimation engines
        can verify the provenance of declustered data before using it.
        
        Args:
            require_composites_source: If True, raises error if source was raw assays
            copy_data: If True, return a copy to prevent mutation
            
        Returns:
            Declustered DataFrame with provenance attrs
            
        Raises:
            ValueError: If require_composites_source=True and source was raw assays
        """
        results = self.get_declustering_results(copy_data=copy_data)
        if not results:
            return None
        
        df = results.get('weighted_dataframe')
        if df is None:
            return None
        
        # LINEAGE CHECK: Verify source was composites if required
        source_was_raw = results.get('source_was_raw_assays', False)
        if require_composites_source and source_was_raw:
            raise ValueError(
                "LINEAGE GATE FAILED: Declustering was performed on raw assays, "
                "not composited data. This violates change-of-support principles "
                "and may produce biased estimation results. "
                "Re-run declustering on composited data."
            )
        
        # Attach provenance to DataFrame attrs for downstream verification
        df.attrs['source_type'] = 'declustered'
        df.attrs['parent_data_key'] = results.get('parent_data_key', 'unknown')
        df.attrs['source_was_raw_assays'] = source_was_raw
        df.attrs['validation_status'] = results.get('validation_status', 'NOT_RUN')
        df.attrs['lineage_gate_passed'] = True
        
        if source_was_raw:
            logger.warning(
                "LINEAGE WARNING: Returning declustered data that was derived from raw assays. "
                "Consider re-running on composited data for JORC/SAMREC compliance."
            )
        
        return df

    # ---------------------------------------------------------------- transformation metadata
    def register_transformation_metadata(
        self,
        metadata: Dict[str, Any],
        source_panel: str = "GradeTransformationPanel",
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register grade transformation metadata - delegates to base class then emits signal."""
        try:
            success = super().register_model("transformation_metadata", metadata, custom_metadata, source_panel)
            if success:
                self._emit("transformationMetadataLoaded", metadata)
            return success
        except Exception:
            logger.error("Failed to register transformation metadata", exc_info=True)
            return False

    def get_transformation_metadata(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get grade transformation metadata - delegates to base class."""
        return super().get_data("transformation_metadata", copy_data=copy_data)

    def has_transformation_metadata(self) -> bool:
        """Check if transformation metadata is available - delegates to base class."""
        return super().has_data("transformation_metadata")

    def register_transformers(self, transformers: Dict[str, Any]) -> None:
        """
        Register NormalScoreTransformer objects for back-transformation in SGSIM.
        
        Parameters
        ----------
        transformers : Dict[str, Any]
            Dictionary mapping column names to NormalScoreTransformer objects
        """
        # BUG FIX #2: Use lock for thread safety
        with self._lock:
            try:
                if self._transformers is None:
                    self._transformers = {}
                self._transformers.update(transformers)
                logger.info(f"Registered {len(transformers)} transformer(s) for back-transformation")
            except Exception:
                logger.error("Failed to register transformers", exc_info=True)

    def get_transformers(self) -> Optional[Dict[str, Any]]:
        """Get registered transformer objects."""
        # BUG FIX #1: Return a copy to prevent external mutation
        with self._lock:
            if self._transformers is None:
                return None
            return dict(self._transformers)  # Shallow copy of dict

    def register_kriging_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "KrigingPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register kriging results - delegates to base class then emits signal."""
        success = super().register_model("kriging_results", results, metadata, source_panel)
        if success:
            self._emit("krigingResultsLoaded", results)
        return success

    def get_kriging_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get kriging results - delegates to base class."""
        return super().get_data("kriging_results", copy_data=copy_data)

    def register_simple_kriging_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "SimpleKrigingPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register simple kriging results - delegates to base class then emits signal."""
        success = super().register_model("simple_kriging_results", results, metadata, source_panel)
        if success:
            self._emit("simpleKrigingResultsLoaded", results)
        return success

    def get_simple_kriging_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get simple kriging results - delegates to base class."""
        return super().get_data("simple_kriging_results", copy_data=copy_data)

    def register_cokriging_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "CoKrigingPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register cokriging results - delegates to base class then emits signal."""
        success = super().register_model("cokriging_results", results, metadata, source_panel)
        if success:
            self._emit("cokrigingResultsLoaded", results)
        return success

    def get_cokriging_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get cokriging results - delegates to base class."""
        return super().get_data("cokriging_results", copy_data=copy_data)

    def register_indicator_kriging_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "IndicatorKrigingPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register indicator kriging results - delegates to base class then emits signal."""
        success = super().register_model("indicator_kriging_results", results, metadata, source_panel)
        if success:
            self._emit("indicatorKrigingResultsLoaded", results)
        return success

    def get_indicator_kriging_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get indicator kriging results - delegates to base class."""
        return super().get_data("indicator_kriging_results", copy_data=copy_data)

    def register_universal_kriging_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "UniversalKrigingPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register universal kriging results - delegates to base class then emits signal."""
        success = super().register_model("universal_kriging_results", results, metadata, source_panel)
        if success:
            self._emit("universalKrigingResultsLoaded", results)
        return success

    def get_universal_kriging_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get universal kriging results - delegates to base class."""
        return super().get_data("universal_kriging_results", copy_data=copy_data)

    def register_soft_kriging_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "SoftKrigingPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register soft kriging results - delegates to base class then emits signal."""
        success = super().register_model("soft_kriging_results", results, metadata, source_panel)
        if success:
            self._emit("softKrigingResultsLoaded", results)
        return success

    def get_soft_kriging_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get soft kriging results - delegates to base class."""
        return super().get_data("soft_kriging_results", copy_data=copy_data)

    def register_rbf_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "RBFPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register RBF interpolation results - delegates to base class then emits signal."""
        success = super().register_model("rbf_results", results, metadata, source_panel)
        if success:
            self._emit("rbfResultsLoaded", results)
        return success

    def get_rbf_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get RBF interpolation results - delegates to base class."""
        return super().get_data("rbf_results", copy_data=copy_data)

    def register_sgsim_results(
        self,
        results: Dict[str, Any],
        source_panel: str = "SGSIMPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register SGSIM results - delegates to base class then emits signal."""
        success = super().register_model("sgsim_results", results, metadata, source_panel)
        if success:
            self._emit("sgsimResultsLoaded", results)
        return success

    def get_sgsim_results(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get SGSIM results - delegates to base class."""
        return super().get_data("sgsim_results", copy_data=copy_data)

    # ---------------------------------------------------------------- resources & geomet
    def register_resource_summary(
        self,
        summary: Dict[str, Any],
        source_panel: str = "BlockResourcePanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register resource summary - delegates to base class then emits signal."""
        success = super().register_model("resource_summary", summary, metadata, source_panel)
        if success:
            self._emit("resourceCalculated", summary)
        return success

    def get_resource_summary(self, copy_data: bool = True) -> Optional[Dict[str, Any]]:
        """Get resource summary - delegates to base class."""
        return super().get_data("resource_summary", copy_data=copy_data)

    def register_geomet_results(
        self,
        results: Any,
        source_panel: str = "GeometPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register geomet results - delegates to base class then emits signal."""
        success = super().register_model("geomet_results", results, metadata, source_panel)
        if success:
            self._emit("geometResultsLoaded", results)
        return success

    def get_geomet_results(self, copy_data: bool = True) -> Optional[Any]:
        """Get geomet results - delegates to base class."""
        return super().get_data("geomet_results", copy_data=copy_data)

    def register_geomet_ore_types(
        self,
        ore_types: Any,
        source_panel: str = "GeometDomainPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register geomet ore types - delegates to base class then emits signal."""
        success = super().register_model("geomet_ore_types", ore_types, metadata, source_panel)
        if success:
            self._emit("geometOreTypesLoaded", ore_types)
        return success

    def get_geomet_ore_types(self, copy_data: bool = True) -> Optional[Any]:
        """Get geomet ore types - delegates to base class."""
        return super().get_data("geomet_ore_types", copy_data=copy_data)

    # ---------------------------------------------------------------- geological model
    def register_geological_surfaces(
        self,
        surfaces_result: Any,
        source_panel: str = "GeologicalModellingWizard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register implicit geological surfaces result.
        
        This emits geologicalSurfacesLoaded and geologicalModelUpdated signals
        so that panels like the Geological Model Explorer and Property Panel
        can update their UI.
        
        Args:
            surfaces_result: Result dict from build_multidomain_surfaces
            source_panel: Name of panel that generated the surfaces
            metadata: Optional metadata dict
        """
        success = super().register_model("implicit_surfaces", surfaces_result, metadata, source_panel)
        if success:
            self._emit("geologicalSurfacesLoaded", surfaces_result)
            self._emit("geologicalModelUpdated", surfaces_result)
            logger.info(f"Registered geological surfaces from {source_panel}")
        return success

    def get_geological_surfaces(self, copy_data: bool = True) -> Optional[Any]:
        """Get implicit geological surfaces result."""
        return super().get_data("implicit_surfaces", copy_data=copy_data)

    def register_geological_solids(
        self,
        solids_result: Any,
        source_panel: str = "GeologicalModellingWizard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register voxel geological solids result.
        
        This emits geologicalSolidsLoaded and geologicalModelUpdated signals
        so that panels like the Geological Model Explorer and Property Panel
        can update their UI.
        
        Args:
            solids_result: Result dict from build_voxel_solids
            source_panel: Name of panel that generated the solids
            metadata: Optional metadata dict
        """
        success = super().register_model("voxel_solids", solids_result, metadata, source_panel)
        if success:
            self._emit("geologicalSolidsLoaded", solids_result)
            self._emit("geologicalModelUpdated", solids_result)
            logger.info(f"Registered geological solids from {source_panel}")
        return success

    def get_geological_solids(self, copy_data: bool = True) -> Optional[Any]:
        """Get voxel geological solids result."""
        return super().get_data("voxel_solids", copy_data=copy_data)

    # ---------------------------------------------------------------- loopstructural model
    def register_loopstructural_model(
        self,
        model_result: Any,
        source_panel: str = "LoopStructuralModelPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register LoopStructural geological model result.
        
        This emits loopstructuralModelLoaded and geologicalModelUpdated signals
        so that panels can update their UI.
        
        Args:
            model_result: Result dict from LoopStructural modeling
            source_panel: Name of panel that generated the model
            metadata: Optional metadata dict
        """
        success = super().register_model("loopstructural_model", model_result, metadata, source_panel)
        if success:
            self._emit("loopstructuralModelLoaded", model_result)
            self._emit("geologicalModelUpdated", model_result)
            logger.info(f"Registered LoopStructural model from {source_panel}")
        return success

    def get_loopstructural_model(self, copy_data: bool = True) -> Optional[Any]:
        """Get LoopStructural model result."""
        return super().get_data("loopstructural_model", copy_data=copy_data)

    def register_loopstructural_compliance(
        self,
        report: Any,
        source_panel: str = "LoopStructuralModelPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register LoopStructural compliance audit report.
        
        Args:
            report: AuditReport from compliance validation
            source_panel: Name of panel that generated the report
            metadata: Optional metadata dict
        """
        success = super().register_model("loopstructural_compliance", report, metadata, source_panel)
        if success:
            self._emit("loopstructuralComplianceChecked", report)
            logger.info(f"Registered LoopStructural compliance report from {source_panel}")
        return success

    def get_loopstructural_compliance(self, copy_data: bool = True) -> Optional[Any]:
        """Get LoopStructural compliance report."""
        return super().get_data("loopstructural_compliance", copy_data=copy_data)

    # ---------------------------------------------------------------- planning / schedules
    def register_pit_optimization_results(
        self,
        results: Any,
        source_panel: str = "PitOptimisationPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register pit optimization results - delegates to base class then emits signal."""
        success = super().register_model("pit_optimization_results", results, metadata, source_panel)
        if success:
            self._emit("pitOptimizationResultsLoaded", results)
        return success

    def get_pit_optimization_results(self, copy_data: bool = True) -> Optional[Any]:
        """Get pit optimization results - delegates to base class."""
        return super().get_data("pit_optimization_results", copy_data=copy_data)

    def register_schedule(
        self,
        schedule: Any,
        source_panel: str = "NPVPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register schedule - delegates to base class then emits signal."""
        success = super().register_model("schedule", schedule, metadata, source_panel)
        if success:
            self._emit("scheduleGenerated", schedule)
        return success

    def get_schedule(self, copy_data: bool = True) -> Optional[Any]:
        """Get schedule - delegates to base class."""
        return super().get_data("schedule", copy_data=copy_data)

    def register_irr_results(
        self,
        irr_results: Any,
        source_panel: str = "IRRPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register IRR results - delegates to base class then emits signal."""
        success = super().register_model("irr_results", irr_results, metadata, source_panel)
        if success:
            self._emit("irrResultsLoaded", irr_results)
        return success

    def get_irr_results(self, copy_data: bool = True) -> Optional[Any]:
        """Get IRR results - delegates to base class."""
        return super().get_data("irr_results", copy_data=copy_data)

    def register_reconciliation_results(
        self,
        results: Any,
        source_panel: str = "ReconciliationPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register reconciliation results - delegates to base class then emits signal."""
        success = super().register_model("reconciliation_results", results, metadata, source_panel)
        if success:
            self._emit("reconciliationResultsLoaded", results)
        return success

    def get_reconciliation_results(self, copy_data: bool = True) -> Optional[Any]:
        """Get reconciliation results - delegates to base class."""
        return super().get_data("reconciliation_results", copy_data=copy_data)

    def register_haulage_evaluation(
        self,
        haulage: Any,
        source_panel: str = "FleetPanel",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register haulage evaluation - delegates to base class then emits signal."""
        success = super().register_model("haulage_evaluation", haulage, metadata, source_panel)
        if success:
            self._emit("haulageEvaluationLoaded", haulage)
        return success

    def get_haulage_evaluation(self, copy_data: bool = True) -> Optional[Any]:
        """Get haulage evaluation - delegates to base class."""
        return super().get_data("haulage_evaluation", copy_data=copy_data)

    # ---------------------------------------------------------------- research
    def register_experiment_results(
        self,
        results: Any,
        source_panel: str = "ResearchDashboard",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register experiment results - delegates to base class then emits signal."""
        success = super().register_model("experiment_results", results, metadata, source_panel)
        if success:
            self._emit("experimentResultsLoaded", results)
        return success

    def get_experiment_results(self, copy_data: bool = True) -> Optional[Any]:
        """Get experiment results - delegates to base class."""
        return super().get_data("experiment_results", copy_data=copy_data)

    # ---------------------------------------------------------------- validation / integrity
    # Note: validate_block_model and validate_drillhole_data are inherited from DataRegistrySimple
    
    def check_data_integrity(self) -> Dict[str, Any]:
        """Check integrity of stored data (uses parent's method, adds legacy checks)."""
        results = super().check_data_integrity()

        # Add checks for non-core methods using base class methods
        if super().has_data("variogram_results"):
            results["checks_passed"] += 1
        else:
            results["warnings"].append("Variogram results not available")

        if super().has_data("kriging_results") or super().has_data("sgsim_results"):
            results["checks_passed"] += 1
        else:
            results["warnings"].append("No estimation results registered yet")

        return results

    # ---------------------------------------------------------------- summaries / exports
    # Note: get_status_summary delegates to base class (status flags are managed by base class)
    def get_status_summary(self) -> Dict[str, bool]:
        """Get status summary - delegates to base class."""
        return super().get_status_summary()

    def get_data_flow_graph(self) -> Dict[str, List[str]]:
        return {
            "drillhole_data": [
                "DrillholeImportPanel",
                "VariogramAnalysisPanel",
                "Kriging Panels",
                "SGSIM Panels",
            ],
            "variogram_results": [
                "KrigingPanel",
                "SimpleKrigingPanel",
                "SGSIMPanel",
                "UniversalKrigingPanel",
            ],
            "kriging_results": [
                "BlockModelBuilderPanel",
                "UncertaintyPanel",
                "ResourcePanels",
            ],
            "sgsim_results": [
                "UncertaintyPanel",
                "UncertaintyPropagationPanel",
                "Planning Dashboards",
            ],
            "block_model": [
                "Visualization",
                "ResourcePanels",
                "NPVS/IRR/Pit Optimisation",
            ],
            "classified_block_model": ["Resource Dashboards", "ReconciliationPanel"],
            "schedule": [
                "PlanningDashboard",
                "ProductionDashboard",
                "FleetPanel",
            ],
            "pit_optimization_results": [
                "NPVSPanel",
                "StrategicSchedulePanel",
                "UncertaintyPanel",
            ],
            "resource_summary": ["ReconciliationPanel", "PlanningDashboard"],
            "geomet_results": ["GeometPanels", "PlanningDashboard"],
            "geomet_ore_types": ["MainWindow Export", "GeometDomainPanel"],
            "irr_results": ["PlanningDashboard"],
            "reconciliation_results": ["ProductionDashboard", "ESG Dashboard"],
            "haulage_evaluation": ["ProductionDashboard"],
            "experiment_results": ["ResearchDashboard"],
            "contact_set": ["StructuralPanel", "GradeControlPanel"],
        }

    def export_status_summary(self, file_path: str) -> bool:
        """Export status summary - uses base class methods for metadata."""
        try:
            export_payload = {
                "generated_at": datetime.now().isoformat(),
                "status_summary": self.get_status_summary(),
                "metadata": {},
                "integrity_check": self.check_data_integrity(),
                "data_flow_graph": self.get_data_flow_graph(),
            }

            # Get metadata from base class storage
            for key in self._DATA_TYPES:
                metadata = super().get_metadata(key)
                if metadata:
                    export_payload["metadata"][key] = {
                        "source_panel": metadata.source_panel,
                        "timestamp": metadata.timestamp.isoformat(),
                        "row_count": metadata.row_count,
                        "columns": metadata.columns,
                        "custom_metadata": metadata.custom_metadata,
                    }

            with open(file_path, "w", encoding="utf-8") as handle:
                json.dump(export_payload, handle, indent=2)

            logger.info("Exported status summary to %s", file_path)
            return True
        except Exception:
            logger.error("Failed to export status summary", exc_info=True)
            return False

    def export_data_flow_graph(self, file_path: str, fmt: str = "json") -> bool:
        graph = self.get_data_flow_graph()
        try:
            fmt = fmt.lower()
            if fmt == "json":
                with open(file_path, "w", encoding="utf-8") as handle:
                    json.dump(graph, handle, indent=2)
            elif fmt == "csv":
                import csv

                with open(file_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["data_type", "consuming_panel"])
                    for data_type, consumers in graph.items():
                        for consumer in consumers:
                            writer.writerow([data_type, consumer])
            elif fmt == "dot":
                with open(file_path, "w", encoding="utf-8") as handle:
                    handle.write("digraph DataFlow {\\n")
                    for data_type, consumers in graph.items():
                        for consumer in consumers:
                            handle.write(f'  "{data_type}" -> "{consumer}";\\n')
                    handle.write("}\\n")
            else:
                raise ValueError(f"Unsupported format {fmt}")

            logger.info("Exported data flow graph (%s) to %s", fmt, file_path)
            return True
        except Exception:
            logger.error("Failed to export data flow graph", exc_info=True)
            return False

    # ---------------------------------------------------------------- category label maps
    def set_category_label(self, namespace: str, code: str, label: str, source: str = "user") -> None:
        """
        Set a display label for a category code with signal emission and audit logging.
        
        Args:
            namespace: Namespace identifier (e.g., "drillholes.lithology")
            code: Category code (e.g., "BIF")
            label: Display label (e.g., "Banded Iron Formation")
            source: Source of the change (for audit trail)
        """
        # Call base class method
        super().set_category_label(namespace, code, label, source)
        
        # Emit signal
        self._emit("categoryLabelMapsChanged", namespace)
        
        # Audit log
        try:
            from .audit_manager import AuditManager
            audit = AuditManager()
            audit.log_event(
                module="DataRegistry",
                action="set_category_label",
                parameters={
                    "namespace": namespace,
                    "code": code,
                    "label": label,
                    "source": source
                },
                result_summary={"success": True}
            )
        except Exception as e:
            logger.debug(f"Audit logging failed: {e}")
    
    def clear_category_label(self, namespace: str, code: str) -> None:
        """
        Clear a category label mapping with signal emission.
        
        Args:
            namespace: Namespace identifier
            code: Category code to reset
        """
        super().clear_category_label(namespace, code)
        self._emit("categoryLabelMapsChanged", namespace)
        
        # Audit log
        try:
            from .audit_manager import AuditManager
            audit = AuditManager()
            audit.log_event(
                module="DataRegistry",
                action="clear_category_label",
                parameters={"namespace": namespace, "code": code},
                result_summary={"success": True}
            )
        except Exception as e:
            logger.debug(f"Audit logging failed: {e}")
    
    # ---------------------------------------------------------------- housekeeping
    def clear_all(self) -> None:
        """Clear all data - delegates to base class then emits signals."""
        logger.info("Clearing entire DataRegistry")
        # Clear all data via base class (handles all storage)
        super().clear_all()
        
        # Clear transformers (special case - not in base class storage)
        self._transformers = None
        
        # Emit signals for complete clear
        self._emit("drillholeDataCleared")
        self._emit("blockModelCleared")
        self._emit("domainModelCleared")
