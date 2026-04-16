"""
Fragmentation Controller
========================

Sub-controller of AppController that coordinates the fragmentation analysis
workflow: import -> preprocess -> segment -> size extraction -> spatial mapping.

Follows the same patterns as GeostatsController and ScanController.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4
from datetime import datetime

import numpy as np

if TYPE_CHECKING:
    from .app_controller import AppController

logger = logging.getLogger(__name__)


class FragmentationController:
    """
    Controller for fragmentation analysis operations.

    Orchestrates: import -> preprocess -> segment -> size extract -> spatial map.
    Integrates with job registry for background processing.
    """

    def __init__(self, app_controller: "AppController"):
        self._app = app_controller
        logger.info("FragmentationController initialized")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dataset(self):
        """Get current FragmentDataset from registry."""
        return self._app.registry.get_fragment_dataset(copy_data=False)

    def _ensure_dataset(self):
        """Get dataset or raise."""
        ds = self._get_dataset()
        if ds is None:
            raise ValueError("No FragmentDataset loaded. Import data first.")
        return ds

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def _prepare_import_lidar_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Import a point cloud file (LAS/LAZ/PLY/OBJ/XYZ/DXF) and create a
        new FragmentDataset.

        Params keys: las_path, crs, name, acquisition_date
        """
        from ..scans.fragment_dataset import FragmentDataset, ProvenanceLog

        file_path = Path(params["las_path"])
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = file_path.suffix.lower()

        if progress_callback:
            progress_callback(10, f"Loading {ext} file...")

        if ext == ".dxf":
            cloud, crs_hint = self._import_dxf(file_path, progress_callback)
        else:
            cloud, crs_hint = self._import_scan(file_path, progress_callback)

        if cloud is None or len(cloud) < 100:
            raise ValueError(f"Point cloud too small ({len(cloud) if cloud is not None else 0} points). Minimum 100.")

        if progress_callback:
            progress_callback(80, "Creating FragmentDataset...")

        dataset = FragmentDataset(
            dataset_id=uuid4(),
            name=params.get("name", file_path.stem),
            source_lidar_path=file_path,
            crs=params.get("crs") or crs_hint,
            acquisition_date=params.get("acquisition_date"),
            fused_cloud=cloud,
        )

        dataset.provenance.record(
            operation="import",
            parameters={"file": str(file_path), "format": ext, "crs": dataset.crs},
            input_array=cloud,
            output_summary={
                "point_count": len(cloud),
                "columns": cloud.shape[1],
                "bounds_min": cloud[:, :3].min(axis=0).tolist(),
                "bounds_max": cloud[:, :3].max(axis=0).tolist(),
            },
        )

        self._app.registry.register_fragment_dataset(
            dataset, source_panel="FragmentImportPanel"
        )

        if progress_callback:
            progress_callback(100, "Import complete")

        return {
            "dataset_id": str(dataset.dataset_id),
            "name": dataset.name,
            "point_count": dataset.point_count,
            "crs": dataset.crs,
        }

    @staticmethod
    def _import_scan(file_path: Path, progress_callback) -> tuple:
        """Import LAS/LAZ/PLY/OBJ/XYZ via ScanIngestor."""
        from ..scans.scan_ingest import ScanIngestor

        ingestor = ScanIngestor()
        scan_data = ingestor.ingest_file(file_path)

        if scan_data.points is None:
            raise ValueError("No point data found in file.")

        cloud = scan_data.points.copy()
        if scan_data.colors is not None:
            cloud = np.column_stack([cloud, scan_data.colors])
        if scan_data.intensities is not None:
            cloud = np.column_stack([cloud, scan_data.intensities.reshape(-1, 1)])

        return cloud, scan_data.crs

    @staticmethod
    def _import_dxf(file_path: Path, progress_callback) -> tuple:
        """
        Import DXF — extract all 3D vertices from surfaces, lines, and points
        into a unified point cloud for fragmentation analysis.
        """
        from ..parsers.dxf_parser import DXFParser

        if progress_callback:
            progress_callback(20, "Parsing DXF geometry...")

        result = DXFParser().parse(file_path)

        # Collect all XYZ vertices from every geometry type
        all_points = []

        for surf in result.get("surfaces", []):
            if surf.n_points > 0:
                all_points.append(np.asarray(surf.points))

        for line in result.get("lines", []):
            if line.n_points > 0:
                all_points.append(np.asarray(line.points))

        pts_pv = result.get("points")
        if pts_pv is not None and pts_pv.n_points > 0:
            all_points.append(np.asarray(pts_pv.points))

        if not all_points:
            raise ValueError("DXF file contains no 3D geometry.")

        cloud = np.vstack(all_points)

        # Remove exact duplicates (DXF surfaces share vertices)
        cloud = np.unique(cloud, axis=0)

        if progress_callback:
            progress_callback(60, f"Extracted {len(cloud):,} unique vertices from DXF")

        return cloud, None  # DXF files don't carry CRS

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _prepare_preprocessing_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run the preprocessing pipeline on the current FragmentDataset.

        Params keys: config (PreprocessingConfig), image (optional np.ndarray)
        """
        from ..scans.preprocessing import (
            PreprocessingConfig, run_preprocessing_pipeline,
        )
        from ..scans.fragment_dataset import FragmentDataset

        dataset = self._ensure_dataset()

        config = params.get("config")
        if config is None:
            config = PreprocessingConfig()

        image = params.get("image")

        result = run_preprocessing_pipeline(
            cloud=dataset.fused_cloud,
            config=config,
            image=image,
            progress_callback=progress_callback,
        )

        # Build fused cloud array: XYZ, Nx,Ny,Nz, [R,G,B if present], Curvature
        cloud = result["cloud"]
        normals = result["normals"]
        curvature = result["curvature"]

        # Assemble: X,Y,Z,Nx,Ny,Nz,Curvature + original extra cols
        xyz = cloud[:, :3]
        extra = cloud[:, 3:] if cloud.shape[1] > 3 else np.empty((len(cloud), 0))
        fused = np.column_stack([xyz, normals, extra, curvature.reshape(-1, 1)])

        dataset.fused_cloud = fused

        # Record provenance
        dataset.provenance.record(
            operation="preprocessing",
            parameters={
                "enable_sor": config.enable_sor,
                "enable_ror": config.enable_ror,
                "enable_ground_filter": config.enable_ground_filter,
                "enable_voxel_downsample": config.enable_voxel_downsample,
                "enable_fusion": config.enable_fusion,
            },
            input_array=cloud,
            output_summary=result["stats"],
        )

        # Update registry
        self._app.registry.register_fragment_dataset(
            dataset, source_panel="FragmentPreprocessingPanel"
        )

        return {
            "point_count": len(fused),
            "stats": result["stats"],
        }

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _prepare_segmentation_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run segmentation on the current FragmentDataset.

        Params keys: method ('geometric'|'watershed'|'dl'|'hybrid'), method-specific params
        """
        dataset = self._ensure_dataset()
        method = params.get("method", "geometric")

        if progress_callback:
            progress_callback(5, f"Starting {method} segmentation...")

        if method == "geometric":
            labels = self._run_geometric_segmentation(dataset, params, progress_callback)
        elif method == "watershed":
            labels = self._run_watershed_segmentation(dataset, params, progress_callback)
        else:
            raise ValueError(f"Unsupported segmentation method: {method}")

        dataset.fragment_labels = labels

        # Record provenance
        dataset.provenance.record(
            operation=f"segmentation_{method}",
            parameters=params,
            output_summary={
                "fragment_count": int(np.unique(labels[labels >= 0]).size),
                "noise_points": int(np.sum(labels == -1)),
            },
        )

        self._app.registry.register_fragment_dataset(
            dataset, source_panel="FragmentSegmentationPanel"
        )

        frag_count = int(np.unique(labels[labels >= 0]).size)
        if progress_callback:
            progress_callback(100, f"Segmentation complete: {frag_count} fragments")

        return {
            "fragment_count": frag_count,
            "noise_points": int(np.sum(labels == -1)),
            "method": method,
        }

    def _run_geometric_segmentation(
        self, dataset, params: Dict, progress_callback: Optional[Callable]
    ) -> np.ndarray:
        """Geometric segmentation: region growing + RANSAC + DBSCAN fallback."""
        from ..scans.geometric_segmentation import run_geometric_segmentation

        cloud = dataset.fused_cloud
        labels = run_geometric_segmentation(
            cloud=cloud,
            normal_threshold_deg=params.get("normal_threshold_deg", 30.0),
            curvature_threshold=params.get("curvature_threshold", 0.01),
            min_region_size=params.get("min_region_size", 100),
            max_region_size=params.get("max_region_size", 1_000_000),
            k_neighbors=params.get("k_neighbors", 15),
            ransac_dist_thresh=params.get("ransac_dist_thresh", 0.05),
            enable_ransac=params.get("enable_ransac", False),
            enable_dbscan_fallback=params.get("enable_dbscan_fallback", True),
            dbscan_eps=params.get("dbscan_eps", 0.05),
            dbscan_min_samples=params.get("dbscan_min_samples", 20),
            merge_threshold=params.get("merge_threshold", 0.8),
            progress_callback=progress_callback,
        )
        return labels

    def _run_watershed_segmentation(
        self, dataset, params: Dict, progress_callback: Optional[Callable]
    ) -> np.ndarray:
        """Watershed segmentation on orthorectified RGB image."""
        from ..scans.watershed_segmentation import run_watershed_segmentation

        cloud = dataset.fused_cloud
        labels = run_watershed_segmentation(
            cloud=cloud,
            canny_low=params.get("canny_low", 50),
            canny_high=params.get("canny_high", 150),
            dist_thresh_ratio=params.get("dist_thresh_ratio", 0.5),
            min_fragment_size=params.get("min_fragment_size", 50),
            progress_callback=progress_callback,
        )
        return labels

    # ------------------------------------------------------------------
    # Size Extraction
    # ------------------------------------------------------------------

    def _prepare_size_extraction_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Extract size metrics for all fragments in the current dataset."""
        from ..scans.size_extraction import (
            compute_all_fragment_metrics, build_fsd,
            fit_rosin_rammler, fit_swebrec,
        )
        from ..scans.fragment_dataset import (
            FragmentRecord, FSD, OBB, SegmentationMethod,
        )

        dataset = self._ensure_dataset()
        if dataset.fragment_labels is None:
            raise ValueError("No segmentation results. Run segmentation first.")

        cloud = dataset.fused_cloud
        labels = dataset.fragment_labels

        # Compute metrics using the new size extraction engine
        metrics_list = compute_all_fragment_metrics(cloud, labels, progress_callback)

        # Convert to FragmentRecord objects
        fragments = []
        for m in metrics_list:
            obb = None
            if m.get("obb"):
                obb = OBB(
                    center=m["obb"]["center"],
                    axes=m["obb"]["axes"],
                    half_extents=m["obb"]["half_extents"],
                )
            fr = FragmentRecord(
                fragment_id=m["fragment_id"],
                centroid_xyz=m["centroid"],
                point_indices=m["point_indices"],
                bbox_3d=obb,
                equiv_diameter=m["equiv_diameter"],
                feret_max=m["feret_max"],
                feret_min=m["feret_min"],
                projected_area=m["projected_area"],
                volume_estimate=m["volume"],
                surface_area=m["surface_area"],
                aspect_ratio=m["aspect_ratio"],
                sphericity=m["sphericity"],
                elongation=m["elongation"],
                confidence=0.5,  # TODO: compute from point density
                source_method=SegmentationMethod.GEOMETRIC,
            )
            fragments.append(fr)

        dataset.fragments = fragments

        # Build FSD with distribution fitting
        diameters = np.array([f.equiv_diameter for f in fragments if f.equiv_diameter > 0])
        if len(diameters) > 0:
            fsd_data = build_fsd(diameters)
            rr_n, rr_xc = fit_rosin_rammler(
                fsd_data["diameters_sorted"], fsd_data["cumulative_passing"]
            )
            sw_xmax, sw_x50, sw_b = fit_swebrec(
                fsd_data["diameters_sorted"], fsd_data["cumulative_passing"]
            )
            fsd = FSD(
                diameters_sorted=fsd_data["diameters_sorted"],
                cumulative_passing=fsd_data["cumulative_passing"],
                d10=fsd_data["d10"],
                d50=fsd_data["d50"],
                d80=fsd_data["d80"],
                d90=fsd_data["d90"],
                rr_n=rr_n, rr_xc=rr_xc,
                swebrec_xmax=sw_xmax, swebrec_x50=sw_x50, swebrec_b=sw_b,
            )
            dataset.fsd_global = fsd

        dataset.provenance.record(
            operation="size_extraction",
            parameters={},
            output_summary={
                "fragment_count": len(fragments),
                "d50": dataset.fsd_global.d50 if dataset.fsd_global else 0,
                "d80": dataset.fsd_global.d80 if dataset.fsd_global else 0,
                "rr_n": rr_n if 'rr_n' in dir() else None,
                "rr_xc": rr_xc if 'rr_xc' in dir() else None,
            },
        )

        self._app.registry.register_fragment_dataset(
            dataset, source_panel="FragmentResultsPanel"
        )

        return {
            "fragment_count": len(fragments),
            "d50": dataset.fsd_global.d50 if dataset.fsd_global else 0,
            "d80": dataset.fsd_global.d80 if dataset.fsd_global else 0,
            "d90": dataset.fsd_global.d90 if dataset.fsd_global else 0,
            "rr_n": dataset.fsd_global.rr_n if dataset.fsd_global else None,
            "rr_xc": dataset.fsd_global.rr_xc if dataset.fsd_global else None,
        }

    # ------------------------------------------------------------------
    # Spatial Mapping
    # ------------------------------------------------------------------

    def _prepare_spatial_mapping_payload(
        self, params: Dict[str, Any], progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Placeholder for spatial mapping (Phase 3)."""
        raise NotImplementedError("Spatial mapping will be implemented in Phase 3")
