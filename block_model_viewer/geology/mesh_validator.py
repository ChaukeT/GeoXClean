"""
Mesh Validation Utilities.

Validates mesh integrity for mining software compatibility.

GeoX Invariant Compliance:
- Watertight validation for volume calculations
- Manifold checks for export compatibility
- Unit continuity validation (isolated volume detection)
- Detailed error reporting for debugging
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

import numpy as np

logger = logging.getLogger(__name__)

# Check trimesh availability
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None
    logger.warning("trimesh library not available - mesh validation disabled")


@dataclass
class MeshValidationResult:
    """Result of mesh validation checks."""
    is_watertight: bool
    is_manifold: bool
    volume: Optional[float]  # Cubic meters (only if watertight)
    surface_area: Optional[float]
    n_vertices: int
    n_faces: int
    n_edges: int
    n_holes: int
    n_degenerate_faces: int
    status: str  # 'SUCCESS', 'WARNING', 'ERROR'
    message: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "is_watertight": self.is_watertight,
            "is_manifold": self.is_manifold,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_edges": self.n_edges,
            "n_holes": self.n_holes,
            "n_degenerate_faces": self.n_degenerate_faces,
            "status": self.status,
            "message": self.message,
        }


def verify_mesh_integrity(
    verts: np.ndarray,
    faces: np.ndarray
) -> str:
    """
    Quick check if a mesh is Manifold (Watertight).
    
    Mining software (Surpac/Deswik/Datamine) will reject meshes with 'holes'.
    This is a critical validation for JORC-compliant resource estimation.
    
    Args:
        verts: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices
        
    Returns:
        Status string: "SUCCESS: Volume = X m3" or "ERROR: description"
    """
    if not TRIMESH_AVAILABLE:
        return "WARNING: trimesh library not available. Cannot validate mesh."
    
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        if not mesh.is_watertight:
            # Try to identify the issue
            edges = mesh.edges_unique
            n_holes = len(mesh.facets) if hasattr(mesh, 'facets') else 0
            
            return (
                f"ERROR: Mesh has holes. Volume calculation invalid for JORC. "
                f"Vertices: {len(verts)}, Faces: {len(faces)}, Unique edges: {len(edges)}"
            )
        
        return f"SUCCESS: Volume = {mesh.volume:.0f} m³, Watertight mesh validated"
        
    except Exception as e:
        return f"ERROR: Mesh validation failed: {e}"


def validate_mesh_detailed(
    verts: np.ndarray,
    faces: np.ndarray
) -> MeshValidationResult:
    """
    Detailed mesh validation for export compatibility.
    
    Performs comprehensive checks required by mining software:
    - Watertight (closed) mesh
    - Manifold topology
    - No degenerate faces
    - Valid volume calculation
    
    Args:
        verts: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices
        
    Returns:
        MeshValidationResult with detailed metrics
    """
    if not TRIMESH_AVAILABLE:
        return MeshValidationResult(
            is_watertight=False,
            is_manifold=False,
            volume=None,
            surface_area=None,
            n_vertices=len(verts),
            n_faces=len(faces),
            n_edges=0,
            n_holes=-1,
            n_degenerate_faces=-1,
            status="WARNING",
            message="trimesh library not available. Cannot validate mesh.",
        )
    
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Basic metrics
        n_verts = len(mesh.vertices)
        n_faces = len(mesh.faces)
        n_edges = len(mesh.edges_unique)
        
        # Topology checks
        is_watertight = mesh.is_watertight
        
        # Check for manifold edges (each edge should be shared by exactly 2 faces)
        # In trimesh, is_watertight implies manifold for closed meshes
        is_manifold = is_watertight
        
        # Count degenerate faces (zero-area triangles)
        face_areas = mesh.area_faces
        n_degenerate = int(np.sum(face_areas < 1e-10))
        
        # Volume and surface area (only valid if watertight)
        volume = float(mesh.volume) if is_watertight else None
        surface_area = float(mesh.area)
        
        # Count holes (boundary edges)
        # In a watertight mesh, there should be no boundary edges
        if hasattr(mesh, 'facets'):
            n_holes = len(mesh.facets) if not is_watertight else 0
        else:
            n_holes = 0 if is_watertight else -1  # Unknown
        
        # Determine status
        if is_watertight and n_degenerate == 0:
            status = "SUCCESS"
            message = f"Mesh is watertight with volume {volume:.1f} m³"
        elif is_watertight:
            status = "WARNING"
            message = f"Mesh is watertight but has {n_degenerate} degenerate faces"
        else:
            status = "ERROR"
            message = "Mesh is not watertight. Volume calculation will be invalid."
        
        return MeshValidationResult(
            is_watertight=is_watertight,
            is_manifold=is_manifold,
            volume=volume,
            surface_area=surface_area,
            n_vertices=n_verts,
            n_faces=n_faces,
            n_edges=n_edges,
            n_holes=n_holes,
            n_degenerate_faces=n_degenerate,
            status=status,
            message=message,
        )
        
    except Exception as e:
        logger.error(f"Mesh validation error: {e}")
        return MeshValidationResult(
            is_watertight=False,
            is_manifold=False,
            volume=None,
            surface_area=None,
            n_vertices=len(verts),
            n_faces=len(faces),
            n_edges=0,
            n_holes=-1,
            n_degenerate_faces=-1,
            status="ERROR",
            message=f"Validation failed: {e}",
        )


def repair_mesh(
    verts: np.ndarray,
    faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Attempt to repair common mesh issues.
    
    Repairs:
    - Remove duplicate vertices
    - Remove degenerate faces
    - Fill small holes
    
    Args:
        verts: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices
        
    Returns:
        Tuple of (repaired_verts, repaired_faces, repair_log)
    """
    repair_log = {
        "original_vertices": len(verts),
        "original_faces": len(faces),
        "repairs": [],
    }
    
    if not TRIMESH_AVAILABLE:
        repair_log["error"] = "trimesh library not available"
        return verts, faces, repair_log
    
    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # Remove duplicate vertices
        mesh.merge_vertices()
        if len(mesh.vertices) < repair_log["original_vertices"]:
            repair_log["repairs"].append({
                "type": "merge_vertices",
                "removed": repair_log["original_vertices"] - len(mesh.vertices),
            })
        
        # Remove degenerate faces
        original_faces = len(mesh.faces)
        mesh.remove_degenerate_faces()
        if len(mesh.faces) < original_faces:
            repair_log["repairs"].append({
                "type": "remove_degenerate",
                "removed": original_faces - len(mesh.faces),
            })
        
        # Fill holes (if trimesh supports it)
        if hasattr(mesh, 'fill_holes') and not mesh.is_watertight:
            try:
                mesh.fill_holes()
                repair_log["repairs"].append({"type": "fill_holes"})
            except Exception:
                pass
        
        repair_log["final_vertices"] = len(mesh.vertices)
        repair_log["final_faces"] = len(mesh.faces)
        repair_log["is_watertight"] = mesh.is_watertight
        
        return mesh.vertices, mesh.faces, repair_log
        
    except Exception as e:
        logger.error(f"Mesh repair failed: {e}")
        repair_log["error"] = str(e)
        return verts, faces, repair_log


# =============================================================================
# Unit Continuity Validation
# =============================================================================

@dataclass
class IsolatedVolume:
    """A disconnected component within a unit mesh."""
    component_id: int
    volume_m3: float
    centroid: Tuple[float, float, float]
    bounding_box: Dict[str, float]  # xmin, xmax, ymin, ymax, zmin, zmax
    n_vertices: int
    n_faces: int
    is_artifact: bool  # True if likely a modelling artifact
    artifact_reason: Optional[str]  # Reason for artifact classification

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "volume_m3": self.volume_m3,
            "centroid": self.centroid,
            "bounding_box": self.bounding_box,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "is_artifact": self.is_artifact,
            "artifact_reason": self.artifact_reason,
        }


@dataclass
class UnitContinuityResult:
    """Result of unit continuity analysis."""
    unit_name: str
    n_components: int  # Total connected components
    main_volume_m3: float  # Volume of largest component
    total_volume_m3: float  # Total volume of all components
    isolated_volumes: List[IsolatedVolume] = field(default_factory=list)
    total_isolated_volume_m3: float = 0.0
    isolation_ratio: float = 0.0  # isolated_volume / total_volume
    status: str = "UNKNOWN"  # 'CONTINUOUS', 'MINOR_ISOLATION', 'SIGNIFICANT_ISOLATION', 'FRAGMENTED'
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_name": self.unit_name,
            "n_components": self.n_components,
            "main_volume_m3": self.main_volume_m3,
            "total_volume_m3": self.total_volume_m3,
            "isolated_volumes": [v.to_dict() for v in self.isolated_volumes],
            "total_isolated_volume_m3": self.total_isolated_volume_m3,
            "isolation_ratio": self.isolation_ratio,
            "status": self.status,
            "message": self.message,
        }


@dataclass
class ContinuityValidationReport:
    """Complete continuity validation for all units."""
    unit_results: Dict[str, UnitContinuityResult] = field(default_factory=dict)
    total_isolated_volume_m3: float = 0.0
    total_artifact_count: int = 0
    worst_unit: Optional[str] = None  # Unit with highest isolation ratio
    worst_isolation_ratio: float = 0.0
    status: str = "NOT_EVALUATED"  # Overall status
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit_results": {k: v.to_dict() for k, v in self.unit_results.items()},
            "total_isolated_volume_m3": self.total_isolated_volume_m3,
            "total_artifact_count": self.total_artifact_count,
            "worst_unit": self.worst_unit,
            "worst_isolation_ratio": self.worst_isolation_ratio,
            "status": self.status,
            "message": self.message,
        }


def validate_unit_continuity(
    verts: np.ndarray,
    faces: np.ndarray,
    unit_name: str,
    min_artifact_volume_m3: float = 100.0,
    max_isolated_components: int = 3,
    model_extent: Optional[Dict[str, float]] = None
) -> UnitContinuityResult:
    """
    Analyze mesh connectivity to detect isolated volumes.

    Uses trimesh's split() method to identify connected components.
    Isolated components are classified as artifacts if they are small
    and/or located at model boundaries.

    Args:
        verts: (N, 3) array of vertex coordinates
        faces: (M, 3) array of face indices
        unit_name: Name of the geological unit
        min_artifact_volume_m3: Volumes smaller than this are likely artifacts
        max_isolated_components: More than this triggers warning
        model_extent: Optional dict with xmin, xmax, ymin, ymax, zmin, zmax
                     for boundary detection

    Returns:
        UnitContinuityResult with component analysis
    """
    if not TRIMESH_AVAILABLE:
        return UnitContinuityResult(
            unit_name=unit_name,
            n_components=1,
            main_volume_m3=0.0,
            total_volume_m3=0.0,
            status="UNKNOWN",
            message="trimesh library not available - cannot validate continuity",
        )

    try:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # Split into connected components
        components = mesh.split(only_watertight=False)

        if len(components) == 0:
            return UnitContinuityResult(
                unit_name=unit_name,
                n_components=0,
                main_volume_m3=0.0,
                total_volume_m3=0.0,
                status="ERROR",
                message="No valid mesh components found",
            )

        # Sort by volume (largest first)
        component_data = []
        for comp in components:
            try:
                vol = abs(comp.volume) if comp.is_watertight else 0.0
            except Exception:
                vol = 0.0
            component_data.append((comp, vol))

        component_data.sort(key=lambda x: x[1], reverse=True)

        # Main component is the largest
        main_mesh, main_volume = component_data[0]
        total_volume = sum(vol for _, vol in component_data)

        # Analyze isolated components
        isolated_volumes = []
        total_isolated = 0.0

        for i, (comp, vol) in enumerate(component_data[1:], 1):
            # Get component properties
            try:
                centroid = tuple(comp.centroid)
            except Exception:
                centroid = (0.0, 0.0, 0.0)

            bounds = comp.bounds if comp.bounds is not None else np.array([[0, 0, 0], [0, 0, 0]])
            bbox = {
                "xmin": float(bounds[0, 0]),
                "xmax": float(bounds[1, 0]),
                "ymin": float(bounds[0, 1]),
                "ymax": float(bounds[1, 1]),
                "zmin": float(bounds[0, 2]),
                "zmax": float(bounds[1, 2]),
            }

            # Classify as artifact
            is_artifact, reason = _classify_as_artifact(
                vol, centroid, bbox, main_mesh, min_artifact_volume_m3, model_extent
            )

            isolated_volumes.append(IsolatedVolume(
                component_id=i,
                volume_m3=vol,
                centroid=centroid,
                bounding_box=bbox,
                n_vertices=len(comp.vertices),
                n_faces=len(comp.faces),
                is_artifact=is_artifact,
                artifact_reason=reason,
            ))

            total_isolated += vol

        # Calculate isolation ratio
        isolation_ratio = total_isolated / total_volume if total_volume > 0 else 0.0

        # Determine status
        if len(component_data) == 1:
            status = "CONTINUOUS"
            message = "Unit is fully continuous (single component)"
        elif isolation_ratio < 0.01:  # Less than 1% isolated
            status = "MINOR_ISOLATION"
            message = f"{len(component_data)-1} isolated component(s), {isolation_ratio*100:.2f}% of volume"
        elif isolation_ratio < 0.10:  # Less than 10% isolated
            status = "SIGNIFICANT_ISOLATION"
            message = f"{len(component_data)-1} isolated component(s), {isolation_ratio*100:.1f}% of volume - review recommended"
        else:
            status = "FRAGMENTED"
            message = f"Unit is fragmented: {len(component_data)} components, {isolation_ratio*100:.1f}% isolated"

        return UnitContinuityResult(
            unit_name=unit_name,
            n_components=len(component_data),
            main_volume_m3=main_volume,
            total_volume_m3=total_volume,
            isolated_volumes=isolated_volumes,
            total_isolated_volume_m3=total_isolated,
            isolation_ratio=isolation_ratio,
            status=status,
            message=message,
        )

    except Exception as e:
        logger.error(f"Continuity validation error for {unit_name}: {e}")
        return UnitContinuityResult(
            unit_name=unit_name,
            n_components=0,
            main_volume_m3=0.0,
            total_volume_m3=0.0,
            status="ERROR",
            message=f"Validation failed: {e}",
        )


def _classify_as_artifact(
    volume_m3: float,
    centroid: Tuple[float, float, float],
    bbox: Dict[str, float],
    main_mesh: "trimesh.Trimesh",
    min_volume_m3: float,
    model_extent: Optional[Dict[str, float]]
) -> Tuple[bool, Optional[str]]:
    """
    Determine if an isolated volume is likely a modelling artifact.

    Criteria for artifact classification:
    1. Volume < min_volume_m3 (small fragments)
    2. Located at model boundary
    3. Thin sliver geometry (high aspect ratio)
    """
    reasons = []

    # Check 1: Small volume
    if volume_m3 < min_volume_m3:
        reasons.append(f"small volume ({volume_m3:.1f} m³ < {min_volume_m3:.1f} m³)")

    # Check 2: At model boundary
    if model_extent is not None:
        boundary_tolerance = 0.01  # 1% of extent
        for dim, (min_key, max_key, bbox_min, bbox_max) in enumerate([
            ('xmin', 'xmax', bbox['xmin'], bbox['xmax']),
            ('ymin', 'ymax', bbox['ymin'], bbox['ymax']),
            ('zmin', 'zmax', bbox['zmin'], bbox['zmax']),
        ]):
            extent_range = model_extent[max_key] - model_extent[min_key]
            tol = extent_range * boundary_tolerance

            if abs(bbox_min - model_extent[min_key]) < tol or \
               abs(bbox_max - model_extent[max_key]) < tol:
                reasons.append("at model boundary")
                break

    # Check 3: Thin sliver (high aspect ratio)
    x_range = bbox['xmax'] - bbox['xmin']
    y_range = bbox['ymax'] - bbox['ymin']
    z_range = bbox['zmax'] - bbox['zmin']
    ranges = sorted([x_range, y_range, z_range])

    if ranges[0] > 0 and ranges[2] / ranges[0] > 20:
        reasons.append(f"thin sliver geometry (aspect ratio {ranges[2]/ranges[0]:.1f})")

    is_artifact = len(reasons) > 0
    reason = "; ".join(reasons) if reasons else None

    return is_artifact, reason


def validate_all_units_continuity(
    solids: List[Dict[str, Any]],
    min_artifact_volume_m3: float = 100.0,
    max_isolation_ratio: float = 0.10,
    model_extent: Optional[Dict[str, float]] = None
) -> ContinuityValidationReport:
    """
    Validate continuity for all extracted unit solids.

    Args:
        solids: List of solid dicts from ChronosEngine.extract_solids()
                Each should have 'vertices', 'faces', 'unit_name'
        min_artifact_volume_m3: Volumes smaller than this are likely artifacts
        max_isolation_ratio: Isolation ratio above this triggers warning
        model_extent: Optional dict with model bounds for boundary detection

    Returns:
        Complete continuity report with per-unit analysis
    """
    report = ContinuityValidationReport()

    if not solids:
        report.status = "NO_DATA"
        report.message = "No solids provided for continuity validation"
        return report

    worst_ratio = 0.0
    worst_unit = None

    for solid in solids:
        verts = solid.get('vertices')
        faces = solid.get('faces')
        unit_name = solid.get('unit_name', solid.get('name', 'Unknown'))

        if verts is None or faces is None:
            logger.warning(f"Skipping {unit_name}: missing vertices or faces")
            continue

        result = validate_unit_continuity(
            verts=np.asarray(verts),
            faces=np.asarray(faces),
            unit_name=unit_name,
            min_artifact_volume_m3=min_artifact_volume_m3,
            model_extent=model_extent,
        )

        report.unit_results[unit_name] = result
        report.total_isolated_volume_m3 += result.total_isolated_volume_m3
        report.total_artifact_count += sum(
            1 for v in result.isolated_volumes if v.is_artifact
        )

        if result.isolation_ratio > worst_ratio:
            worst_ratio = result.isolation_ratio
            worst_unit = unit_name

    report.worst_unit = worst_unit
    report.worst_isolation_ratio = worst_ratio

    # Determine overall status
    if not report.unit_results:
        report.status = "NO_DATA"
        report.message = "No valid units for continuity analysis"
    elif worst_ratio < 0.01:
        report.status = "ACCEPTABLE"
        report.message = f"All units are continuous or have minimal isolation (<1%)"
    elif worst_ratio < max_isolation_ratio:
        report.status = "NEEDS_REVIEW"
        report.message = f"Some isolation detected (worst: {worst_unit} at {worst_ratio*100:.1f}%)"
    else:
        report.status = "CRITICAL"
        report.message = f"Significant fragmentation in {worst_unit} ({worst_ratio*100:.1f}% isolated)"

    logger.info(
        f"Continuity validation complete: {len(report.unit_results)} units, "
        f"{report.total_artifact_count} artifacts, status={report.status}"
    )

    return report

