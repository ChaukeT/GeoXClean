"""
Scan Ingestion Engine
=====================

Parses various scan file formats and extracts point clouds/meshes with metadata.
Handles CRS extraction, unit detection, and format validation.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

from .scan_models import ScanData, ScanOperationResult
from ..core.scan_registry import ScanRegistry

logger = logging.getLogger(__name__)


class ScanIngestError(Exception):
    """Base exception for scan ingestion errors."""
    pass


class UnsupportedFormatError(ScanIngestError):
    """Raised when file format is not supported."""
    pass


class CorruptedFileError(ScanIngestError):
    """Raised when file is corrupted or unreadable."""
    pass


class ScanIngestor:
    """
    Main scan ingestion engine.

    Supports multiple file formats with automatic format detection.
    """

    SUPPORTED_FORMATS = {
        '.las': 'las',
        '.laz': 'las',
        '.ply': 'ply',
        '.obj': 'obj',
        '.stl': 'stl',
        '.xyz': 'xyz',
        '.e57': 'e57'  # Optional support
    }

    def __init__(self):
        """Initialize ingestor with available libraries."""
        # LAZY INITIALIZATION: Don't check dependencies at startup to avoid slow imports
        # Open3D in particular can hang for 30+ seconds or indefinitely on some GPU configs
        self._laspy_available: Optional[bool] = None
        self._open3d_available: Optional[bool] = None
        self._trimesh_available: Optional[bool] = None
        logger.info("ScanIngestor initialized (lazy dependency checks)")

    def _ensure_dependencies_checked(self):
        """Check dependencies lazily on first use."""
        if self._laspy_available is None:
            self._laspy_available = self._check_laspy()
            self._open3d_available = self._check_open3d()
            self._trimesh_available = self._check_trimesh()
            logger.info("Scan ingestor dependencies checked:")
            logger.info(f"  LAS/LAZ support: {self._laspy_available}")
            logger.info(f"  Open3D support: {self._open3d_available}")
            logger.info(f"  Trimesh support: {self._trimesh_available}")

    def _check_laspy(self) -> bool:
        """Check if laspy is available."""
        try:
            import laspy
            return True
        except ImportError:
            return False

    def _check_open3d(self) -> bool:
        """Check if open3d is available."""
        try:
            import open3d as o3d
            return True
        except ImportError:
            return False

    def _check_trimesh(self) -> bool:
        """Check if trimesh is available."""
        try:
            import trimesh
            return True
        except ImportError:
            return False

    def load_scan(self, filepath: Union[str, Path],
                  format_hint: Optional[str] = None) -> ScanOperationResult:
        """
        Load scan from file with automatic format detection.

        Args:
            filepath: Path to scan file
            format_hint: Optional format hint ('las', 'ply', 'obj', etc.)

        Returns:
            ScanOperationResult with ScanData or error information
        """
        import time
        start_time = time.time()

        filepath = Path(filepath)
        if not filepath.exists():
            return ScanOperationResult(
                operation="ingest",
                scan_id=None,  # No scan_id yet
                success=False,
                timestamp=None,
                result_data=None,
                errors=[f"File does not exist: {filepath}"],
                processing_time_seconds=time.time() - start_time
            )

        # Detect format
        format_type = format_hint or self._detect_format(filepath)
        if format_type is None:
            return ScanOperationResult(
                operation="ingest",
                scan_id=None,
                success=False,
                timestamp=None,
                result_data=None,
                errors=[f"Unsupported file format: {filepath.suffix}"],
                processing_time_seconds=time.time() - start_time
            )

        # Load based on format - ensure dependencies are checked first (lazy init)
        self._ensure_dependencies_checked()
        try:
            if format_type == 'las':
                scan_data = self._load_las(filepath)
            elif format_type == 'ply':
                scan_data = self._load_ply(filepath)
            elif format_type == 'obj':
                scan_data = self._load_obj(filepath)
            elif format_type == 'stl':
                scan_data = self._load_stl(filepath)
            elif format_type == 'xyz':
                scan_data = self._load_xyz(filepath)
            elif format_type == 'e57':
                scan_data = self._load_e57(filepath)
            else:
                raise UnsupportedFormatError(f"Unsupported format: {format_type}")

            # Set file format metadata
            scan_data.file_format = format_type.upper()

            return ScanOperationResult(
                operation="ingest",
                scan_id=None,  # Will be set by registry
                success=True,
                timestamp=None,
                result_data=scan_data,
                processing_time_seconds=time.time() - start_time
            )

        except Exception as e:
            error_msg = f"Failed to load {format_type.upper()} file: {str(e)}"
            logger.error(error_msg, exc_info=True)

            return ScanOperationResult(
                operation="ingest",
                scan_id=None,
                success=False,
                timestamp=None,
                result_data=None,
                errors=[error_msg],
                processing_time_seconds=time.time() - start_time
            )

    def _detect_format(self, filepath: Path) -> Optional[str]:
        """Detect file format from extension."""
        suffix = filepath.suffix.lower()
        return self.SUPPORTED_FORMATS.get(suffix)

    def _load_las(self, filepath: Path) -> ScanData:
        """Load LAS/LAZ file using laspy."""
        if not self._laspy_available:
            raise ImportError("laspy not available. Install with: pip install laspy")

        import laspy

        # Read LAS file
        with laspy.open(filepath) as las_file:
            # Read header first
            header = las_file.header

            # Extract CRS if available
            crs = None
            if hasattr(header, 'global_encoding') and hasattr(header.global_encoding, 'crs'):
                crs = str(header.global_encoding.crs)

            # Read point data
            points = np.vstack([las_file.x, las_file.y, las_file.z]).T

            # Extract additional attributes if available
            colors = None
            if hasattr(las_file, 'red') and hasattr(las_file, 'green') and hasattr(las_file, 'blue'):
                try:
                    red = np.array(las_file.red, dtype=np.float32) / 65535.0
                    green = np.array(las_file.green, dtype=np.float32) / 65535.0
                    blue = np.array(las_file.blue, dtype=np.float32) / 65535.0
                    colors = np.column_stack([red, green, blue])
                except:
                    pass  # Skip colors if conversion fails

            intensities = None
            if hasattr(las_file, 'intensity'):
                intensities = np.array(las_file.intensity, dtype=np.float32)

            classifications = None
            if hasattr(las_file, 'classification'):
                classifications = np.array(las_file.classification, dtype=np.uint8)

            # Detect units (LAS is typically in meters, but check header)
            units = "meters"  # Default assumption
            if hasattr(header, 'units') and header.units:
                if "foot" in str(header.units).lower():
                    units = "feet"

        return ScanData(
            points=points,
            colors=colors,
            intensities=intensities,
            classifications=classifications,
            crs=crs,
            units=units,
            file_format="LAS"
        )

    def _load_ply(self, filepath: Path) -> ScanData:
        """Load PLY file using Open3D or trimesh."""
        if self._open3d_available:
            return self._load_ply_open3d(filepath)
        elif self._trimesh_available:
            return self._load_ply_trimesh(filepath)
        else:
            raise ImportError("Neither Open3D nor trimesh available for PLY loading")

    def _load_ply_open3d(self, filepath: Path) -> ScanData:
        """Load PLY using Open3D."""
        import open3d as o3d

        # Load mesh/point cloud
        mesh = o3d.io.read_triangle_mesh(str(filepath))
        if len(mesh.triangles) > 0:
            # It's a mesh
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.triangles, dtype=np.int32)
            vertex_colors = None
            if mesh.has_vertex_colors():
                vertex_colors = np.asarray(mesh.vertex_colors, dtype=np.float32)
            return ScanData(
                points=vertices,
                faces=faces,
                colors=vertex_colors,
                file_format="PLY"
            )
        else:
            # Try as point cloud
            pcd = o3d.io.read_point_cloud(str(filepath))
            points = np.asarray(pcd.points, dtype=np.float64)
            colors = None
            if pcd.has_colors():
                colors = np.asarray(pcd.colors, dtype=np.float32)
            normals = None
            if pcd.has_normals():
                normals = np.asarray(pcd.normals, dtype=np.float32)

            return ScanData(
                points=points,
                colors=colors,
                normals=normals,
                file_format="PLY"
            )

    def _load_ply_trimesh(self, filepath: Path) -> ScanData:
        """Load PLY using trimesh."""
        import trimesh

        mesh = trimesh.load(str(filepath))

        # Convert to numpy arrays
        points = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32) if hasattr(mesh, 'faces') and mesh.faces is not None else None
        colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0 if hasattr(mesh.visual, 'vertex_colors') else None

        return ScanData(
            points=points,
            faces=faces,
            colors=colors,
            file_format="PLY"
        )

    def _load_obj(self, filepath: Path) -> ScanData:
        """Load OBJ file using trimesh."""
        if not self._trimesh_available:
            raise ImportError("trimesh not available. Install with: pip install trimesh")

        import trimesh

        mesh = trimesh.load(str(filepath))

        points = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)
        colors = np.array(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0 if hasattr(mesh.visual, 'vertex_colors') else None

        return ScanData(
            points=points,
            faces=faces,
            colors=colors,
            file_format="OBJ"
        )

    def _load_stl(self, filepath: Path) -> ScanData:
        """Load STL file using trimesh."""
        if not self._trimesh_available:
            raise ImportError("trimesh not available. Install with: pip install trimesh")

        import trimesh

        mesh = trimesh.load(str(filepath))

        points = np.array(mesh.vertices, dtype=np.float64)
        faces = np.array(mesh.faces, dtype=np.int32)

        return ScanData(
            points=points,
            faces=faces,
            file_format="STL"
        )

    def _load_xyz(self, filepath: Path) -> ScanData:
        """Load XYZ file (custom parser for simple text format)."""
        try:
            # Read the file
            data = np.loadtxt(filepath, dtype=np.float64)

            # Handle different column layouts
            if data.shape[1] >= 3:
                points = data[:, :3]

                # Check for additional columns
                colors = None
                intensities = None

                if data.shape[1] >= 6:  # X, Y, Z, R, G, B
                    colors = data[:, 3:6].astype(np.float32)
                    if colors.max() > 1.0:  # Assume 0-255 range
                        colors /= 255.0

                elif data.shape[1] >= 4:  # X, Y, Z, Intensity
                    intensities = data[:, 3].astype(np.float32)

                return ScanData(
                    points=points,
                    colors=colors,
                    intensities=intensities,
                    file_format="XYZ"
                )
            else:
                raise ValueError(f"XYZ file must have at least 3 columns (X, Y, Z), got {data.shape[1]}")

        except Exception as e:
            raise CorruptedFileError(f"Failed to parse XYZ file: {str(e)}")

    def _load_e57(self, filepath: Path) -> ScanData:
        """Load E57 file (placeholder - complex format, may not be implemented)."""
        raise UnsupportedFormatError("E57 format not yet supported")

    def register_scan(self, registry: ScanRegistry, filepath: Union[str, Path],
                     format_hint: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """
        Load scan and register it in the registry.

        Args:
            registry: ScanRegistry instance
            filepath: Path to scan file
            format_hint: Optional format hint

        Returns:
            (success, scan_id_or_error_message)
        """
        result = self.load_scan(filepath, format_hint)

        if not result.success:
            return False, "; ".join(result.errors)

        # Create metadata
        from uuid import uuid4
        from datetime import datetime
        from ..core.scan_registry import ScanMetadata

        scan_id = uuid4()
        scan_data = result.result_data

        metadata = ScanMetadata(
            scan_id=scan_id,
            source_file=Path(filepath),
            source_hash="",  # Will be computed by registry
            crs=scan_data.crs,
            units=scan_data.units,
            point_count=scan_data.point_count(),
            mesh_face_count=scan_data.face_count() if scan_data.is_mesh() else None,
            file_format=scan_data.file_format,
            timestamp=datetime.now(),
            user="unknown",  # TODO: Get from user context
        )

        success = registry.register_scan(None, scan_data, metadata)

        if success:
            return True, str(scan_id)
        else:
            return False, "Failed to register scan in registry"
