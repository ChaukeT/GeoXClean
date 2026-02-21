"""
Render Payloads - Typed interfaces for Renderer operations.

These dataclasses form the strict interface between UI/Controller and Renderer.
No PyVista objects should be passed directly - only these payloads.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np


@dataclass
class MeshPayload:
    """Payload for mesh-based visualization."""
    name: str
    vertices: np.ndarray  # (N, 3) array of vertex coordinates
    faces: Optional[np.ndarray] = None  # (M, 3) or (M, 4) face indices, or None for point cloud
    scalars: Optional[np.ndarray] = None  # (N,) scalar values per vertex
    colors: Optional[np.ndarray] = None  # (N, 3) RGB colors per vertex
    opacity: float = 1.0
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GridPayload:
    """Payload for structured grid visualization."""
    name: str
    grid: np.ndarray  # Structured grid array or tuple of (x, y, z) coordinate arrays
    scalars: np.ndarray  # Scalar values for grid points
    origin: Optional[np.ndarray] = None  # (3,) origin point
    spacing: Optional[np.ndarray] = None  # (3,) spacing per axis
    dimensions: Optional[np.ndarray] = None  # (3,) grid dimensions
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossSectionPayload:
    """Payload for cross-section visualization."""
    name: str
    points: np.ndarray  # (N, 3) points defining the section
    lines: Optional[np.ndarray] = None  # Line connectivity
    thickness: float = 0.0  # Thickness for 3D extrusion
    scalars: Optional[np.ndarray] = None  # Scalar values along section
    color: Optional[np.ndarray] = None  # RGB color
    opacity: float = 1.0
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PitShellPayload:
    """Payload for pit shell visualization."""
    name: str
    benches: List[MeshPayload] = field(default_factory=list)  # Individual bench meshes
    phases: List[MeshPayload] = field(default_factory=list)  # Phase meshes
    wireframe: bool = False
    opacity: float = 0.7
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PointCloudPayload:
    """Payload for point cloud visualization."""
    name: str
    points: np.ndarray  # (N, 3) point coordinates
    scalars: Optional[np.ndarray] = None  # (N,) scalar values
    colors: Optional[np.ndarray] = None  # (N, 3) RGB colors
    point_size: float = 5.0
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LinePayload:
    """Payload for line/polyline visualization."""
    name: str
    points: np.ndarray  # (N, 3) point coordinates
    lines: Optional[np.ndarray] = None  # Line connectivity
    color: Optional[np.ndarray] = None  # RGB color
    width: float = 2.0
    visible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecondaryViewPayload:
    """Payload for secondary viewer windows (e.g., PyQtGraph)."""
    view_id: str
    view_type: str  # 'grid', 'plot', 'table', etc.
    data: Any  # View-specific data (DataFrame, array, etc.)
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

