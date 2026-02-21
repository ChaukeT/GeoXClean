"""
Scan LOD (Level of Detail) Controller
=====================================

Manages point cloud rendering performance through spatial indexing and viewport culling.
Supports preview vs analysis modes with configurable point budgets.
"""

import logging
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OctreeNode:
    """Node in the octree spatial index."""
    bounds_min: np.ndarray  # (3,) min coordinates
    bounds_max: np.ndarray  # (3,) max coordinates
    center: np.ndarray      # (3,) center coordinates
    point_indices: List[int]  # Indices of points in this node
    children: Optional[List['OctreeNode']] = None  # 8 children for internal nodes
    depth: int = 0

    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.children is None

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if point is within node bounds."""
        return np.all(point >= self.bounds_min) and np.all(point <= self.bounds_max)

    def intersects_frustum(self, frustum_planes: List[Tuple[np.ndarray, float]]) -> bool:
        """
        Check if node intersects view frustum.

        Args:
            frustum_planes: List of (normal, distance) tuples for frustum planes

        Returns:
            True if node intersects frustum
        """
        # Simple AABB-frustum intersection test
        for normal, distance in frustum_planes:
            # Check if all points are outside one plane
            if np.dot(self.bounds_min, normal) + distance < 0 and \
               np.dot(self.bounds_max, normal) + distance < 0:
                # Check corners
                corners = self._get_corners()
                all_outside = all(np.dot(corner, normal) + distance < 0 for corner in corners)
                if all_outside:
                    return False
        return True

    def _get_corners(self) -> List[np.ndarray]:
        """Get 8 corner points of the AABB."""
        corners = []
        for i in range(8):
            corner = np.array([
                self.bounds_min[0] if (i & 1) == 0 else self.bounds_max[0],
                self.bounds_min[1] if (i & 2) == 0 else self.bounds_max[1],
                self.bounds_min[2] if (i & 4) == 0 else self.bounds_max[2]
            ])
            corners.append(corner)
        return corners


@dataclass
class LODConfig:
    """Configuration for LOD management."""
    max_points_viewport: int = 1_000_000  # Maximum points to show in viewport
    preview_downsample_ratio: float = 0.1  # Ratio of points to show in preview mode
    octree_max_depth: int = 8  # Maximum octree depth
    octree_min_points_per_node: int = 100  # Minimum points per octree node


class ScanLODController:
    """
    Manages Level of Detail for scan visualization.

    Uses octree spatial indexing to efficiently cull points outside the viewport
    and manage rendering performance.
    """

    def __init__(self, config: Optional[LODConfig] = None):
        """
        Initialize LOD controller.

        Args:
            config: LOD configuration
        """
        self.config = config or LODConfig()
        self.octree: Optional[OctreeNode] = None
        self.original_points: Optional[np.ndarray] = None
        self.original_indices: Optional[np.ndarray] = None

        logger.info("ScanLODController initialized")

    def build_octree(self, points: np.ndarray, max_depth: Optional[int] = None,
                    min_points_per_node: Optional[int] = None) -> bool:
        """
        Build octree spatial index for the point cloud.

        Args:
            points: Point coordinates (N, 3)
            max_depth: Maximum octree depth
            min_points_per_node: Minimum points per node

        Returns:
            True if successful
        """
        if len(points) == 0:
            logger.warning("Cannot build octree for empty point cloud")
            return False

        self.original_points = points.copy()
        self.original_indices = np.arange(len(points))

        max_depth = max_depth or self.config.octree_max_depth
        min_points_per_node = min_points_per_node or self.config.octree_min_points_per_node

        # Compute bounds
        bounds_min = points.min(axis=0)
        bounds_max = points.max(axis=0)

        # Build root node
        self.octree = self._build_octree_recursive(
            points=self.original_points,
            indices=self.original_indices,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            depth=0,
            max_depth=max_depth,
            min_points_per_node=min_points_per_node
        )

        logger.info(f"Built octree with max depth {max_depth}, {self._count_nodes()} nodes")
        return True

    def _build_octree_recursive(self, points: np.ndarray, indices: np.ndarray,
                               bounds_min: np.ndarray, bounds_max: np.ndarray,
                               depth: int, max_depth: int, min_points_per_node: int) -> OctreeNode:
        """
        Recursively build octree nodes.

        Args:
            points: All points
            indices: Indices of points in this subtree
            bounds_min/max: Bounds of this node
            depth: Current depth
            max_depth: Maximum depth
            min_points_per_node: Minimum points per node

        Returns:
            OctreeNode
        """
        center = (bounds_min + bounds_max) / 2.0

        # Check if this should be a leaf node
        if depth >= max_depth or len(indices) <= min_points_per_node:
            return OctreeNode(
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                center=center,
                point_indices=indices.tolist(),
                depth=depth
            )

        # Create children
        children = []
        for i in range(8):
            # Compute child bounds
            child_bounds_min = np.array([
                bounds_min[0] if (i & 1) == 0 else center[0],
                bounds_min[1] if (i & 2) == 0 else center[1],
                bounds_min[2] if (i & 4) == 0 else center[2]
            ])
            child_bounds_max = np.array([
                center[0] if (i & 1) == 0 else bounds_max[0],
                center[1] if (i & 2) == 0 else bounds_max[1],
                center[2] if (i & 4) == 0 else bounds_max[2]
            ])

            # Find points in this child
            child_mask = np.all((points[indices] >= child_bounds_min) &
                               (points[indices] <= child_bounds_max), axis=1)
            child_indices = indices[child_mask]

            if len(child_indices) > 0:
                child_node = self._build_octree_recursive(
                    points=points,
                    indices=child_indices,
                    bounds_min=child_bounds_min,
                    bounds_max=child_bounds_max,
                    depth=depth + 1,
                    max_depth=max_depth,
                    min_points_per_node=min_points_per_node
                )
                children.append(child_node)

        return OctreeNode(
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            center=center,
            point_indices=[],  # Internal nodes don't store points directly
            children=children,
            depth=depth
        )

    def _count_nodes(self) -> int:
        """Count total nodes in octree."""
        if self.octree is None:
            return 0

        def count_recursive(node: OctreeNode) -> int:
            count = 1
            if node.children:
                for child in node.children:
                    count += count_recursive(child)
            return count

        return count_recursive(self.octree)

    def get_viewport_points(self, camera_pos: np.ndarray, camera_forward: np.ndarray,
                           fov_degrees: float = 60.0, aspect_ratio: float = 1.333,
                           max_distance: float = 1000.0) -> Optional[np.ndarray]:
        """
        Get points visible in viewport using frustum culling.

        Args:
            camera_pos: Camera position (3,)
            camera_forward: Camera forward vector (3,)
            fov_degrees: Field of view in degrees
            aspect_ratio: Viewport aspect ratio
            max_distance: Maximum render distance

        Returns:
            Indices of visible points, or None if no octree
        """
        if self.octree is None or self.original_points is None:
            return None

        # Create simple frustum planes (approximation)
        frustum_planes = self._create_frustum_planes(
            camera_pos, camera_forward, fov_degrees, aspect_ratio, max_distance
        )

        # Traverse octree and collect visible point indices
        visible_indices = set()

        def traverse_frustum(node: OctreeNode):
            if node.intersects_frustum(frustum_planes):
                if node.is_leaf():
                    visible_indices.update(node.point_indices)
                elif node.children:
                    for child in node.children:
                        traverse_frustum(child)

        traverse_frustum(self.octree)

        visible_indices = np.array(list(visible_indices))

        # Apply point budget limit
        if len(visible_indices) > self.config.max_points_viewport:
            # Random downsampling to meet budget
            np.random.shuffle(visible_indices)
            visible_indices = visible_indices[:self.config.max_points_viewport]

        return visible_indices

    def _create_frustum_planes(self, camera_pos: np.ndarray, camera_forward: np.ndarray,
                              fov_degrees: float, aspect_ratio: float,
                              max_distance: float) -> List[Tuple[np.ndarray, float]]:
        """
        Create simplified frustum planes for culling.

        This is a basic approximation - real frustum culling would use proper
        view/projection matrices.

        Args:
            camera_pos: Camera position
            camera_forward: Camera forward direction
            fov_degrees: Field of view
            aspect_ratio: Aspect ratio
            max_distance: Far plane distance

        Returns:
            List of (normal, distance) tuples for frustum planes
        """
        # Normalize forward vector
        camera_forward = camera_forward / np.linalg.norm(camera_forward)

        # Create up and right vectors
        up = np.array([0, 0, 1])  # Assume Z-up
        right = np.cross(camera_forward, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, camera_forward)

        # Field of view in radians
        fov_rad = np.deg2rad(fov_degrees)

        # Compute frustum dimensions at distance 1
        height = 2 * np.tan(fov_rad / 2)
        width = height * aspect_ratio

        # Frustum corners at near/far planes (simplified)
        near_dist = 0.1
        far_dist = max_distance

        # Create planes (normal, distance)
        planes = []

        # Near plane
        planes.append((-camera_forward, -np.dot(camera_pos + near_dist * camera_forward, -camera_forward)))

        # Far plane
        planes.append((camera_forward, -np.dot(camera_pos + far_dist * camera_forward, camera_forward)))

        # Left plane
        left_normal = np.cross(up, camera_forward + right * (width / 2))
        left_normal = left_normal / np.linalg.norm(left_normal)
        planes.append((left_normal, -np.dot(camera_pos, left_normal)))

        # Right plane
        right_normal = np.cross(camera_forward - right * (width / 2), up)
        right_normal = right_normal / np.linalg.norm(right_normal)
        planes.append((right_normal, -np.dot(camera_pos, right_normal)))

        # Top plane
        top_normal = np.cross(camera_forward + up * (height / 2), right)
        top_normal = top_normal / np.linalg.norm(top_normal)
        planes.append((top_normal, -np.dot(camera_pos, top_normal)))

        # Bottom plane
        bottom_normal = np.cross(right, camera_forward - up * (height / 2))
        bottom_normal = bottom_normal / np.linalg.norm(bottom_normal)
        planes.append((bottom_normal, -np.dot(camera_pos, bottom_normal)))

        return planes

    def get_preview_points(self, mode: str = "preview") -> Optional[np.ndarray]:
        """
        Get downsampled points for preview mode.

        Args:
            mode: "preview" or "analysis"

        Returns:
            Point indices for preview, or None
        """
        if self.original_indices is None:
            return None

        if mode == "preview":
            # Downsample for preview
            n_preview = int(len(self.original_indices) * self.config.preview_downsample_ratio)
            n_preview = max(1000, min(n_preview, self.config.max_points_viewport))  # Reasonable bounds

            indices = np.random.choice(self.original_indices, size=n_preview, replace=False)
            return indices
        else:
            # Analysis mode - return all points (subject to budget)
            if len(self.original_indices) <= self.config.max_points_viewport:
                return self.original_indices
            else:
                # Random downsampling to meet budget
                indices = np.random.choice(self.original_indices,
                                         size=self.config.max_points_viewport, replace=False)
                return indices

    def get_points_in_region(self, region_bounds: Tuple[np.ndarray, np.ndarray],
                           max_points: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get points within a specific 3D region.

        Args:
            region_bounds: (min_coords, max_coords) tuple
            max_points: Maximum points to return

        Returns:
            Point indices in region
        """
        if self.octree is None or self.original_points is None:
            return None

        region_min, region_max = region_bounds

        # Traverse octree and find intersecting nodes
        region_indices = set()

        def traverse_region(node: OctreeNode):
            # Check if node intersects region
            if (node.bounds_max[0] >= region_min[0] and node.bounds_min[0] <= region_max[0] and
                node.bounds_max[1] >= region_min[1] and node.bounds_min[1] <= region_max[1] and
                node.bounds_max[2] >= region_min[2] and node.bounds_min[2] <= region_max[2]):

                if node.is_leaf():
                    # Check individual points
                    points_in_node = self.original_points[node.point_indices]
                    mask = np.all((points_in_node >= region_min) &
                                (points_in_node <= region_max), axis=1)
                    region_indices.update(np.array(node.point_indices)[mask])
                elif node.children:
                    for child in node.children:
                        traverse_region(child)

        traverse_region(self.octree)

        region_indices = np.array(list(region_indices))

        # Apply point limit
        if max_points and len(region_indices) > max_points:
            np.random.shuffle(region_indices)
            region_indices = region_indices[:max_points]

        return region_indices

    def estimate_point_density(self) -> Optional[float]:
        """
        Estimate average point density (points per cubic unit).

        Returns:
            Points per cubic unit, or None if no octree
        """
        if self.octree is None or self.original_points is None:
            return None

        # Use root node bounds to estimate density
        bounds = self.octree.bounds_max - self.octree.bounds_min
        volume = np.prod(bounds)

        if volume > 0:
            return len(self.original_points) / volume
        else:
            return None

    def clear(self):
        """Clear octree and cached data."""
        self.octree = None
        self.original_points = None
        self.original_indices = None

        logger.info("ScanLODController cleared")
