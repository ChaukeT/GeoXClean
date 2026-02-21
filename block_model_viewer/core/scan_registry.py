"""
Scan Registry
=============

Central registry for scan data, separate from DataRegistry to maintain isolation.
Stores scan datasets, provenance chains, and derived products with auditability.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtCore import QObject, pyqtSignal

    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - Qt not available during some tests
    _QT_AVAILABLE = False
    pyqtSignal = object  # type: ignore
    QObject = object  # type: ignore


class _ScanSignalEmitter(QObject if _QT_AVAILABLE else object):
    """Isolated QObject holding all scan registry signals."""

    if _QT_AVAILABLE:
        scanLoaded = pyqtSignal(object)  # ScanMetadata
        scanValidated = pyqtSignal(object)  # ScanMetadata
        scanProcessed = pyqtSignal(object)  # ScanMetadata
        scanCleared = pyqtSignal()  # All scans cleared
        fragmentsComputed = pyqtSignal(object)  # FragmentResults

        def __init__(self) -> None:  # pragma: no cover - trivial
            super().__init__()
            logger.debug("Scan registry signal emitter initialized")


@dataclass
class ProcessingStep:
    """Single processing step in provenance chain."""
    step_name: str  # "validation", "cleaning", "segmentation", "metrics"
    timestamp: datetime
    parameters: Dict[str, Any]  # Complete parameter snapshot
    input_version: UUID  # Input scan_id
    output_version: UUID  # Output scan_id
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "step_name": self.step_name,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "input_version": str(self.input_version),
            "output_version": str(self.output_version),
            "warnings": self.warnings,
            "errors": self.errors
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ProcessingStep:
        """Create from dict (for deserialization)."""
        return cls(
            step_name=data["step_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parameters=data["parameters"],
            input_version=UUID(data["input_version"]),
            output_version=UUID(data["output_version"]),
            warnings=data.get("warnings", []),
            errors=data.get("errors", [])
        )


@dataclass
class DerivedProduct:
    """Link to derived datasets (fragments, metrics)."""
    product_type: str  # "fragments", "psd_curve", "roughness_map"
    product_id: UUID
    scan_id: UUID  # Parent scan
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "product_type": self.product_type,
            "product_id": str(self.product_id),
            "scan_id": str(self.scan_id),
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DerivedProduct:
        """Create from dict (for deserialization)."""
        return cls(
            product_type=data["product_type"],
            product_id=UUID(data["product_id"]),
            scan_id=UUID(data["scan_id"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )


@dataclass
class ScanMetadata:
    """Metadata for a registered scan dataset."""
    scan_id: UUID  # Immutable identifier
    source_file: Path  # Original file path
    source_hash: str  # SHA-256 checksum
    crs: Optional[str]  # CRS code (e.g., "EPSG:32633")
    units: str  # "meters" or "feet" (explicit, not inferred)
    point_count: int  # For point clouds
    mesh_face_count: Optional[int]  # For meshes
    file_format: str  # "LAS", "E57", "OBJ", etc.

    # Provenance chain
    timestamp: datetime
    user: str
    processing_history: List[ProcessingStep] = field(default_factory=list)  # Immutable list
    derived_products: List[DerivedProduct] = field(default_factory=list)  # Links to fragments, metrics

    # Validation state
    validation_result: Optional[Dict[str, Any]] = None

    # Parent chain (if derived from another scan)
    parent_scan_id: Optional[UUID] = None
    transformation_type: Optional[str] = None  # "cleaned", "segmented", etc.
    transformation_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "scan_id": str(self.scan_id),
            "source_file": str(self.source_file),
            "source_hash": self.source_hash,
            "crs": self.crs,
            "units": self.units,
            "point_count": self.point_count,
            "mesh_face_count": self.mesh_face_count,
            "file_format": self.file_format,
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "processing_history": [step.to_dict() for step in self.processing_history],
            "derived_products": [product.to_dict() for product in self.derived_products],
            "validation_result": self.validation_result,
            "parent_scan_id": str(self.parent_scan_id) if self.parent_scan_id else None,
            "transformation_type": self.transformation_type,
            "transformation_params": self.transformation_params
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScanMetadata:
        """Create from dict (for deserialization)."""
        return cls(
            scan_id=UUID(data["scan_id"]),
            source_file=Path(data["source_file"]),
            source_hash=data["source_hash"],
            crs=data["crs"],
            units=data["units"],
            point_count=data["point_count"],
            mesh_face_count=data["mesh_face_count"],
            file_format=data["file_format"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            user=data["user"],
            processing_history=[ProcessingStep.from_dict(step) for step in data.get("processing_history", [])],
            derived_products=[DerivedProduct.from_dict(product) for product in data.get("derived_products", [])],
            validation_result=data.get("validation_result"),
            parent_scan_id=UUID(data["parent_scan_id"]) if data.get("parent_scan_id") else None,
            transformation_type=data.get("transformation_type"),
            transformation_params=data.get("transformation_params", {})
        )

    def add_processing_step(self, step: ProcessingStep) -> None:
        """Add a processing step to the immutable history."""
        # Create new list to maintain immutability
        self.processing_history = self.processing_history + [step]

    def add_derived_product(self, product: DerivedProduct) -> None:
        """Add a derived product link."""
        self.derived_products = self.derived_products + [product]


class ScanRegistry(QObject if _QT_AVAILABLE else object):
    """
    Scan Registry: Separate from DataRegistry for scan isolation.

    Stores scan datasets with full provenance tracking. Maintains separation
    from drillhole and block model data to prevent cross-contamination.
    """

    _instance: Optional["ScanRegistry"] = None
    _signal_emitter: Optional[_ScanSignalEmitter] = None
    _init_lock = None  # Class-level lock for thread-safe initialization

    @classmethod
    def _get_init_lock(cls):
        """Get or create the initialization lock (lazy, thread-safe)."""
        import threading
        if cls._init_lock is None:
            # Use RLock (Reentrant Lock) to avoid deadlock when __init__ is called
            # from within instance() which already holds the lock
            cls._init_lock = threading.RLock()
        return cls._init_lock

    @classmethod
    def instance(cls) -> "ScanRegistry":
        """Global Access Point (Legacy Support)."""
        # BUG FIX #11: Use lock for thread-safe singleton
        with cls._get_init_lock():
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def get_existing(cls) -> Optional["ScanRegistry"]:
        return cls._instance

    def __init__(self):
        """Initialize scan registry with thread-safe storage."""
        super().__init__()
        self._scans: Dict[UUID, ScanMetadata] = {}
        self._scan_data: Dict[UUID, Any] = {}  # Actual scan data storage

        # BUG FIX #11: Use lock for thread-safe signal emitter initialization
        with self._get_init_lock():
            if _QT_AVAILABLE and self.__class__._signal_emitter is None:
                self.__class__._signal_emitter = _ScanSignalEmitter()

        logger.info("ScanRegistry initialized (separate from DataRegistry)")

    @property
    def signals(self) -> _ScanSignalEmitter:
        """Access to Qt signals (if available)."""
        if not _QT_AVAILABLE:
            raise RuntimeError("Qt signals not available (Qt not loaded)")
        return self._signal_emitter

    def register_scan(self, scan_id: Optional[UUID], data: Any, metadata: ScanMetadata) -> bool:
        """
        Register a scan dataset with metadata.

        Args:
            scan_id: Optional UUID (generated if None)
            data: Scan data (points, mesh, etc.)
            metadata: Complete metadata

        Returns:
            True if successful
        """
        if scan_id is None:
            scan_id = uuid4()

        # Deep copy metadata to prevent mutations
        metadata_copy = copy.deepcopy(metadata)
        metadata_copy.scan_id = scan_id

        # Compute checksum if not provided
        if not metadata_copy.source_hash:
            metadata_copy.source_hash = self._compute_data_hash(data)

        self._scans[scan_id] = metadata_copy
        self._scan_data[scan_id] = data

        logger.info(f"Registered scan {scan_id} ({metadata_copy.file_format}, {metadata_copy.point_count} points)")

        # Emit signal
        if _QT_AVAILABLE:
            self.signals.scanLoaded.emit(metadata_copy)

        return True

    def get_scan(self, scan_id: UUID) -> Optional[Tuple[ScanMetadata, Any]]:
        """
        Retrieve scan metadata and data.

        Returns:
            Tuple of (metadata, data) or None if not found
        """
        metadata = self._scans.get(scan_id)
        data = self._scan_data.get(scan_id)

        if metadata is None or data is None:
            return None

        # Return deep copies to prevent mutation
        return copy.deepcopy(metadata), copy.deepcopy(data)

    def get_scan_metadata(self, scan_id: UUID) -> Optional[ScanMetadata]:
        """Get scan metadata only."""
        metadata = self._scans.get(scan_id)
        return copy.deepcopy(metadata) if metadata else None

    def get_scan_data(self, scan_id: UUID) -> Optional[Any]:
        """Get scan data only."""
        data = self._scan_data.get(scan_id)
        return copy.deepcopy(data) if data else None

    def has_scan(self, scan_id: UUID) -> bool:
        """Check if scan exists."""
        return scan_id in self._scans and scan_id in self._scan_data

    def list_scans(self) -> List[ScanMetadata]:
        """List all registered scans."""
        return [copy.deepcopy(metadata) for metadata in self._scans.values()]

    def get_provenance_chain(self, scan_id: UUID) -> List[ProcessingStep]:
        """Get complete provenance chain for a scan."""
        metadata = self._scans.get(scan_id)
        if metadata is None:
            return []
        return copy.deepcopy(metadata.processing_history)

    def update_scan_metadata(self, scan_id: UUID, updates: Dict[str, Any]) -> bool:
        """
        Update scan metadata (creates new version).

        Args:
            scan_id: Scan to update
            updates: Metadata fields to update

        Returns:
            True if successful
        """
        metadata = self._scans.get(scan_id)
        if metadata is None:
            return False

        # Create new metadata copy
        new_metadata = copy.deepcopy(metadata)

        # Apply updates
        for key, value in updates.items():
            if hasattr(new_metadata, key):
                setattr(new_metadata, key, value)

        self._scans[scan_id] = new_metadata

        # Emit signal
        if _QT_AVAILABLE:
            self.signals.scanProcessed.emit(new_metadata)

        return True

    def clear_scan(self, scan_id: UUID) -> bool:
        """Remove a scan from registry."""
        if scan_id in self._scans:
            del self._scans[scan_id]
        if scan_id in self._scan_data:
            del self._scan_data[scan_id]

        logger.info(f"Cleared scan {scan_id}")
        return True

    def clear_all(self) -> None:
        """Clear all scans from registry."""
        self._scans.clear()
        self._scan_data.clear()

        logger.info("Cleared all scans")

        # Emit signal
        if _QT_AVAILABLE:
            self.signals.scanCleared.emit()

    def save_state(self, filepath: Path) -> bool:
        """Save registry state to file."""
        try:
            state = {
                "version": "1.0",
                "scans": [metadata.to_dict() for metadata in self._scans.values()]
            }

            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"Saved scan registry state to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to save scan registry state: {e}")
            return False

    def load_state(self, filepath: Path) -> bool:
        """Load registry state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            if state.get("version") != "1.0":
                logger.warning("Registry state version mismatch, attempting load anyway")

            # Clear existing state
            self.clear_all()

            # Load scans
            for scan_dict in state.get("scans", []):
                metadata = ScanMetadata.from_dict(scan_dict)
                # Note: We can't restore the actual data, only metadata
                self._scans[metadata.scan_id] = metadata

            logger.info(f"Loaded scan registry state from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to load scan registry state: {e}")
            return False

    def _compute_data_hash(self, data: Any) -> str:
        """Compute SHA-256 hash of scan data for integrity checking."""
        try:
            # Convert data to string representation for hashing
            if hasattr(data, 'tobytes'):
                # NumPy arrays
                data_bytes = data.tobytes()
            elif hasattr(data, '__array__'):
                # Array-like objects
                import numpy as np
                data_bytes = np.asarray(data).tobytes()
            else:
                # Fallback to string representation
                data_bytes = str(data).encode('utf-8')

            return hashlib.sha256(data_bytes).hexdigest()

        except Exception as e:
            logger.warning(f"Could not compute data hash: {e}")
            return "HASH_FAILED"

