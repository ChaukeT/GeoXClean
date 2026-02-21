"""
Scan Controller - Orchestrates scan analysis pipeline.

Manages the complete scan processing pipeline from ingestion to fragmentation analysis.
Follows GeoX controller patterns with job registry integration.
"""

from typing import Optional, Dict, Any, Callable, List, Tuple, TYPE_CHECKING
import logging
from uuid import UUID, uuid4
from pathlib import Path

import numpy as np

if TYPE_CHECKING:
    from .app_controller import AppController
from ..core.scan_registry import ScanRegistry, ScanMetadata, ProcessingStep
from ..scans.scan_ingest import ScanIngestor
from ..scans.scan_validation import ScanValidator, ValidationReport
from ..scans.scan_cleaning import ScanCleaner, CleaningReport
from ..scans.segmentation_region_growing import RegionGrowingSegmenter, RegionGrowingParams
from ..scans.segmentation_dbscan import DBSCANSegmenter, DBSCANParams
from ..scans.fragment_metrics import FragmentMetricsComputer, FragmentMetrics, PSDResults
from ..scans.scan_lod_controller import ScanLODController
from ..scans.scan_models import ScanData, ScanProcessingMode

logger = logging.getLogger(__name__)


class ScanController:
    """
    Controller for scan analysis operations.

    Orchestrates the scan processing pipeline: ingest -> validate -> clean -> segment -> metrics.
    Integrates with job registry for background processing.
    """

    def __init__(self, app_controller: "AppController"):
        """
        Initialize scan controller.

        Args:
            app_controller: Parent AppController instance
        """
        self._app = app_controller  # type: ignore
        self._registry = ScanRegistry.instance()
        self._lod_controller = ScanLODController()

        # Initialize engines
        self._ingestor = ScanIngestor()
        self._validator = ScanValidator()
        self._cleaner = ScanCleaner()
        self._region_growing_segmenter = RegionGrowingSegmenter()
        self._dbscan_segmenter = DBSCANSegmenter()
        self._metrics_computer = FragmentMetricsComputer()

        logger.info("ScanController initialized")

    @property
    def registry(self) -> ScanRegistry:
        """Access to scan registry."""
        return self._registry

    @property
    def lod_controller(self) -> ScanLODController:
        """Access to LOD controller."""
        return self._lod_controller

    # =========================================================================
    # Pipeline Orchestration Methods
    # =========================================================================

    def load_scan(self, filepath: Path, format_hint: Optional[str] = None,
                 progress_callback: Optional[Callable[[float, str], None]] = None) -> Optional[UUID]:
        """
        Load and register a scan file.

        Args:
            filepath: Path to scan file
            format_hint: Optional format hint
            progress_callback: Optional progress callback

        Returns:
            Scan UUID if successful, None otherwise
        """
        try:
            # Load scan data
            result = self._ingestor.load_scan(filepath, format_hint)

            if not result.success:
                logger.error(f"Failed to load scan {filepath}: {result.errors}")
                return None

            scan_data = result.result_data

            # Register in scan registry
            scan_id = uuid4()  # Generate new UUID
            metadata = ScanMetadata(
                scan_id=scan_id,
                source_file=filepath,
                source_hash="",  # Will be computed by registry
                crs=scan_data.crs,
                units=scan_data.units,
                point_count=scan_data.point_count(),
                mesh_face_count=scan_data.face_count(),
                file_format=scan_data.file_format,
                timestamp=result.timestamp or None,
                user="current_user"  # TODO: Get from user context
            )

            # Use the pre-generated scan_id so UI/registry stay in sync
            success = self._registry.register_scan(scan_id, scan_data, metadata)

            if success:
                logger.info(f"Successfully loaded scan {scan_id} from {filepath}")
                return scan_id
            else:
                logger.error(f"Failed to register scan in registry")
                return None

        except Exception as e:
            logger.error(f"Error loading scan {filepath}: {e}", exc_info=True)
            return None

    def validate_scan(self, scan_id: UUID, config: Optional[Dict[str, Any]] = None,
                     progress_callback: Optional[Callable[[float, str], None]] = None) -> Optional[ValidationReport]:
        """
        Validate a registered scan.

        Args:
            scan_id: Scan UUID
            config: Optional validation configuration
            progress_callback: Optional progress callback

        Returns:
            ValidationReport if successful, None otherwise
        """
        try:
            # Get scan data
            metadata, scan_data = self._registry.get_scan(scan_id)
            if metadata is None or scan_data is None:
                logger.error(f"Scan {scan_id} not found in registry")
                return None

            # Validate
            report = self._validator.validate_scan(scan_data, scan_id)

            # Update metadata with validation result
            self._registry.update_scan_metadata(scan_id, {
                "validation_result": {
                    "is_valid": report.is_valid,
                    "error_count": report.error_count(),
                    "warning_count": report.warning_count(),
                    "total_points": report.total_points
                }
            })

            # Record processing step
            step = ProcessingStep(
                step_name="validation",
                timestamp=report.timestamp,
                parameters=config or {},
                input_version=scan_id,
                output_version=scan_id,  # Same version, just validated
                warnings=[v.message for v in report.violations if v.is_warning()],
                errors=[v.message for v in report.violations if v.is_error()]
            )

            # Update processing history
            metadata.add_processing_step(step)
            self._registry.update_scan_metadata(scan_id, {
                "processing_history": metadata.processing_history
            })

            logger.info(f"Validation completed for scan {scan_id}: {report.error_count()} errors, {report.warning_count()} warnings")
            return report

        except Exception as e:
            logger.error(f"Error validating scan {scan_id}: {e}", exc_info=True)
            return None

    def clean_scan(self, scan_id: UUID, outlier_method: str = "statistical",
                  normal_method: str = "auto", outlier_params: Optional[Dict[str, Any]] = None,
                  normal_params: Optional[Dict[str, Any]] = None,
                  progress_callback: Optional[Callable[[float, str], None]] = None) -> Optional[UUID]:
        """
        Clean scan data (outlier removal and normal estimation).

        Args:
            scan_id: Input scan UUID
            outlier_method: Outlier removal method
            normal_method: Normal estimation method
            outlier_params: Outlier removal parameters
            normal_params: Normal estimation parameters
            progress_callback: Optional progress callback

        Returns:
            New scan UUID if successful, None otherwise
        """
        try:
            # Get input scan
            metadata, scan_data = self._registry.get_scan(scan_id)
            if metadata is None or scan_data is None:
                logger.error(f"Scan {scan_id} not found in registry")
                return None

            # Clean scan
            cleaned_data, report = self._cleaner.clean_scan(
                scan_data, scan_id, outlier_method, normal_method,
                outlier_params, normal_params
            )

            # Register cleaned scan as new version
            new_scan_id = uuid4()
            new_metadata = ScanMetadata(
                scan_id=new_scan_id,
                source_file=metadata.source_file,
                source_hash="",  # Will be computed
                crs=cleaned_data.crs,
                units=cleaned_data.units,
                point_count=cleaned_data.point_count(),
                mesh_face_count=cleaned_data.face_count(),
                file_format=cleaned_data.file_format,
                timestamp=report.timestamp,
                user=metadata.user,
                parent_scan_id=scan_id,
                transformation_type="cleaned",
                transformation_params={
                    "outlier_method": outlier_method,
                    "normal_method": normal_method,
                    "outlier_params": outlier_params or {},
                    "normal_params": normal_params or {}
                }
            )

            # Preserve the pre-generated UUID to keep lineage consistent
            success = self._registry.register_scan(new_scan_id, cleaned_data, new_metadata)

            if success:
                # Record processing step
                step = ProcessingStep(
                    step_name="cleaning",
                    timestamp=report.timestamp,
                    parameters={
                        "outlier_method": outlier_method,
                        "normal_method": normal_method,
                        "outlier_params": outlier_params or {},
                        "normal_params": normal_params or {}
                    },
                    input_version=scan_id,
                    output_version=new_scan_id,
                    warnings=[],
                    errors=[]
                )

                new_metadata.add_processing_step(step)
                self._registry.update_scan_metadata(new_scan_id, {
                    "processing_history": new_metadata.processing_history
                })

                logger.info(f"Cleaning completed: {scan_id} -> {new_scan_id} ({report.input_point_count} -> {report.output_point_count} points)")
                return new_scan_id
            else:
                logger.error("Failed to register cleaned scan")
                return None

        except Exception as e:
            logger.error(f"Error cleaning scan {scan_id}: {e}", exc_info=True)
            return None

    def segment_scan(self, scan_id: UUID, strategy: str = "region_growing",
                    strategy_params: Optional[Dict[str, Any]] = None,
                    progress_callback: Optional[Callable[[float, str], None]] = None) -> Optional[Tuple[UUID, List[int]]]:
        """
        Segment scan into fragments.

        Args:
            scan_id: Input scan UUID
            strategy: Segmentation strategy ("region_growing" or "dbscan")
            strategy_params: Strategy-specific parameters
            progress_callback: Optional progress callback

        Returns:
            Tuple of (new_scan_id, fragment_labels) if successful, None otherwise
        """
        try:
            # Get input scan
            metadata, scan_data = self._registry.get_scan(scan_id)
            if metadata is None or scan_data is None:
                logger.error(f"Scan {scan_id} not found in registry")
                return None

            # Perform segmentation based on strategy
            if strategy == "region_growing":
                params = RegionGrowingParams(**strategy_params) if strategy_params else RegionGrowingParams()
                result = self._region_growing_segmenter.segment(scan_data, scan_id, params, progress_callback)
            elif strategy == "dbscan":
                params = DBSCANParams(**strategy_params) if strategy_params else DBSCANParams()
                result = self._dbscan_segmenter.segment(scan_data, scan_id, params, progress_callback)
            else:
                raise ValueError(f"Unknown segmentation strategy: {strategy}")

            if not result.success:
                logger.error(f"Segmentation failed: {result.warnings}")
                return None

            # Create new scan metadata with segmentation results
            new_scan_id = uuid4()
            new_metadata = ScanMetadata(
                scan_id=new_scan_id,
                source_file=metadata.source_file,
                source_hash="",  # Will be computed
                crs=scan_data.crs,
                units=scan_data.units,
                point_count=scan_data.point_count(),
                mesh_face_count=scan_data.face_count(),
                file_format=scan_data.file_format,
                timestamp=result.timestamp,
                user=metadata.user,
                parent_scan_id=scan_id,
                transformation_type="segmented",
                transformation_params={"strategy": strategy, "params": strategy_params or {}}
            )

            # Register the segmented scan (same data, but with segmentation metadata)
            # Use the generated UUID so downstream references stay valid
            success = self._registry.register_scan(new_scan_id, scan_data, new_metadata)

            if success:
                # Record processing step
                step = ProcessingStep(
                    step_name="segmentation",
                    timestamp=result.timestamp,
                    parameters={"strategy": strategy, "params": strategy_params or {}},
                    input_version=scan_id,
                    output_version=new_scan_id,
                    warnings=result.warnings,
                    errors=[]
                )

                new_metadata.add_processing_step(step)
                self._registry.update_scan_metadata(new_scan_id, {
                    "processing_history": new_metadata.processing_history
                })

                # Store derived product (fragment labels)
                from ..core.scan_registry import DerivedProduct
                fragment_product = DerivedProduct(
                    product_type="fragment_labels",
                    product_id=uuid4(),  # Would store actual fragment data
                    scan_id=new_scan_id,
                    timestamp=result.timestamp
                )
                new_metadata.add_derived_product(fragment_product)

                logger.info(f"Segmentation completed: {scan_id} -> {new_scan_id} ({result.fragment_count} fragments)")
                return new_scan_id, result.fragment_labels.tolist()
            else:
                logger.error("Failed to register segmented scan")
                return None

        except Exception as e:
            logger.error(f"Error segmenting scan {scan_id}: {e}", exc_info=True)
            return None

    def compute_fragment_metrics(self, scan_id: UUID, fragment_labels: List[int],
                                progress_callback: Optional[Callable[[float, str], None]] = None) -> Optional[Tuple[UUID, List[FragmentMetrics]]]:
        """
        Compute fragment metrics and PSD.

        Args:
            scan_id: Segmented scan UUID
            fragment_labels: Fragment labels from segmentation
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results_id, fragment_metrics) if successful, None otherwise
        """
        try:
            # Get scan data
            metadata, scan_data = self._registry.get_scan(scan_id)
            if metadata is None or scan_data is None:
                logger.error(f"Scan {scan_id} not found in registry")
                return None

            # Convert labels to numpy array
            labels_array = np.array(fragment_labels, dtype=np.int32)

            # Compute fragment metrics
            fragment_metrics = self._metrics_computer.compute_fragment_metrics(
                scan_data, labels_array, progress_callback
            )

            if not fragment_metrics:
                logger.warning(f"No fragment metrics computed for scan {scan_id}")
                return None

            # Compute PSD
            psd_results = self._metrics_computer.compute_psd(fragment_metrics, scan_id)

            # Store results in registry as derived products
            results_id = uuid4()

            from ..core.scan_registry import DerivedProduct
            metrics_product = DerivedProduct(
                product_type="fragment_metrics",
                product_id=results_id,
                scan_id=scan_id,
                timestamp=psd_results.timestamp
            )

            metadata.add_derived_product(metrics_product)
            self._registry.update_scan_metadata(scan_id, {
                "derived_products": metadata.derived_products
            })

            # Record processing step
            step = ProcessingStep(
                step_name="metrics",
                timestamp=psd_results.timestamp,
                parameters={},
                input_version=scan_id,
                output_version=scan_id,  # Same scan, just added metrics
                warnings=[],
                errors=[]
            )

            metadata.add_processing_step(step)
            self._registry.update_scan_metadata(scan_id, {
                "processing_history": metadata.processing_history
            })

            logger.info(
                f"Fragment metrics computed for scan {scan_id}: {len(fragment_metrics)} fragments, "
                f"P10={psd_results.p10_m:.3f}m, P50={psd_results.p50_m:.3f}m, P80={psd_results.p80_m:.3f}m"
            )

            return results_id, fragment_metrics

        except Exception as e:
            logger.error(f"Error computing fragment metrics for scan {scan_id}: {e}", exc_info=True)
            return None

    # =========================================================================
    # Job Registry Payload Preparation Methods
    # =========================================================================

    def _prepare_scan_validate_payload(self, params: Dict[str, Any],
                                      progress_callback: Optional[Callable[[float, str], None]] = None) -> ValidationReport:
        """
        Prepare payload for scan validation job.

        Args:
            params: Job parameters containing scan_id and config
            progress_callback: Optional progress callback

        Returns:
            ValidationReport
        """
        scan_id = UUID(params["scan_id"])
        config = params.get("config")

        report = self.validate_scan(scan_id, config, progress_callback)
        if report is None:
            raise RuntimeError(f"Scan validation failed for {scan_id}")

        return report

    def _prepare_scan_clean_payload(self, params: Dict[str, Any],
                                   progress_callback: Optional[Callable[[float, str], None]] = None) -> UUID:
        """
        Prepare payload for scan cleaning job.

        Args:
            params: Job parameters
            progress_callback: Optional progress callback

        Returns:
            New scan UUID
        """
        scan_id = UUID(params["scan_id"])
        outlier_method = params.get("outlier_method", "statistical")
        normal_method = params.get("normal_method", "auto")
        outlier_params = params.get("outlier_params")
        normal_params = params.get("normal_params")

        new_scan_id = self.clean_scan(scan_id, outlier_method, normal_method,
                                    outlier_params, normal_params, progress_callback)

        if new_scan_id is None:
            raise RuntimeError(f"Scan cleaning failed for {scan_id}")

        return new_scan_id

    def _prepare_scan_segment_payload(self, params: Dict[str, Any],
                                     progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[UUID, List[int]]:
        """
        Prepare payload for scan segmentation job.

        Args:
            params: Job parameters
            progress_callback: Optional progress callback

        Returns:
            Tuple of (new_scan_id, fragment_labels)
        """
        scan_id = UUID(params["scan_id"])
        strategy = params.get("strategy", "region_growing")
        strategy_params = params.get("strategy_params")

        result = self.segment_scan(scan_id, strategy, strategy_params, progress_callback)

        if result is None:
            raise RuntimeError(f"Scan segmentation failed for {scan_id}")

        return result

    def _prepare_scan_compute_metrics_payload(self, params: Dict[str, Any],
                                             progress_callback: Optional[Callable[[float, str], None]] = None) -> Tuple[UUID, List[FragmentMetrics]]:
        """
        Prepare payload for fragment metrics computation job.

        Args:
            params: Job parameters
            progress_callback: Optional progress callback

        Returns:
            Tuple of (results_id, fragment_metrics)
        """
        scan_id = UUID(params["scan_id"])
        fragment_labels = params["fragment_labels"]

        result = self.compute_fragment_metrics(scan_id, fragment_labels, progress_callback)

        if result is None:
            raise RuntimeError(f"Fragment metrics computation failed for {scan_id}")

        return result

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_scan_list(self) -> List[ScanMetadata]:
        """Get list of all registered scans."""
        return self._registry.list_scans()

    def get_scan_provenance(self, scan_id: UUID) -> List[ProcessingStep]:
        """Get complete provenance chain for a scan."""
        return self._registry.get_provenance_chain(scan_id)

    def delete_scan(self, scan_id: UUID) -> bool:
        """Delete a scan from registry."""
        return self._registry.clear_scan(scan_id) is None  # clear_scan returns None on success

    def build_lod_octree(self, scan_id: UUID) -> bool:
        """
        Build LOD octree for a scan.

        Args:
            scan_id: Scan UUID

        Returns:
            True if successful
        """
        metadata, scan_data = self._registry.get_scan(scan_id)
        if metadata is None or scan_data is None or scan_data.points is None:
            return False

        return self._lod_controller.build_octree(scan_data.points)
