"""
Stereonet Projections - Schmidt (Equal-Area) and Wulff (Equal-Angle).

Implements vectorized stereographic projections for efficiency.
All computations use unit vectors; angles are derived as needed.

Conventions:
- Center of net is the vertical axis (Z)
- Lower hemisphere projection by default
- North is at the top of the plot (Y direction)
- East is to the right (X direction)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional, Union

from .models import NetType, Hemisphere


# =============================================================================
# PROJECTION FUNCTIONS
# =============================================================================

def project_schmidt(
    normals: np.ndarray,
    hemisphere: Hemisphere = Hemisphere.LOWER
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project unit vectors onto Schmidt (equal-area) stereonet.
    
    Uses Lambert azimuthal equal-area projection.
    
    Args:
        normals: Nx3 array of unit vectors
        hemisphere: Which hemisphere to project
    
    Returns:
        Tuple of (x, y) projected coordinates (range -1 to 1)
    
    Formula:
        r = sqrt(2) * sin(theta / 2)
        where theta is the angle from the projection axis (Z)
    """
    normals = np.atleast_2d(normals)
    
    # Ensure proper hemisphere
    if hemisphere == Hemisphere.LOWER:
        # Flip vectors pointing up
        flip_mask = normals[:, 2] > 0
        normals = normals.copy()
        normals[flip_mask] = -normals[flip_mask]
    else:
        # Flip vectors pointing down
        flip_mask = normals[:, 2] < 0
        normals = normals.copy()
        normals[flip_mask] = -normals[flip_mask]
    
    # For lower hemisphere, we project onto plane below
    # theta is angle from -Z axis (for lower hemisphere)
    if hemisphere == Hemisphere.LOWER:
        cos_theta = -normals[:, 2]  # Angle from -Z
    else:
        cos_theta = normals[:, 2]   # Angle from +Z
    
    # Clamp to valid range
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # theta / 2
    theta = np.arccos(cos_theta)
    
    # Equal-area radius
    r = np.sqrt(2) * np.sin(theta / 2)
    
    # Azimuth angle in the horizontal plane
    # atan2(x, y) gives angle from north (Y), clockwise positive
    azimuth = np.arctan2(normals[:, 0], normals[:, 1])
    
    # Project coordinates
    x = r * np.sin(azimuth)
    y = r * np.cos(azimuth)
    
    return x, y


def project_wulff(
    normals: np.ndarray,
    hemisphere: Hemisphere = Hemisphere.LOWER
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project unit vectors onto Wulff (equal-angle) stereonet.
    
    Uses standard stereographic projection.
    
    Args:
        normals: Nx3 array of unit vectors
        hemisphere: Which hemisphere to project
    
    Returns:
        Tuple of (x, y) projected coordinates (range -1 to 1)
    
    Formula:
        r = tan(theta / 2)
        where theta is the angle from the projection axis
        
        Normalized to unit circle: r = tan(theta / 2) / tan(90 / 2) = tan(theta / 2)
        For theta = 90, r = 1
    """
    normals = np.atleast_2d(normals)
    
    # Ensure proper hemisphere
    if hemisphere == Hemisphere.LOWER:
        flip_mask = normals[:, 2] > 0
        normals = normals.copy()
        normals[flip_mask] = -normals[flip_mask]
    else:
        flip_mask = normals[:, 2] < 0
        normals = normals.copy()
        normals[flip_mask] = -normals[flip_mask]
    
    # theta from projection axis
    if hemisphere == Hemisphere.LOWER:
        cos_theta = -normals[:, 2]
    else:
        cos_theta = normals[:, 2]
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    # Equal-angle radius (stereographic)
    # tan(theta/2), normalized so theta=90 gives r=1
    r = np.tan(theta / 2)
    
    # Clamp very large values (near theta = 180, which shouldn't happen for proper hemisphere)
    r = np.clip(r, 0, 10)
    
    # Azimuth
    azimuth = np.arctan2(normals[:, 0], normals[:, 1])
    
    x = r * np.sin(azimuth)
    y = r * np.cos(azimuth)
    
    return x, y


def inverse_project_schmidt(
    x: np.ndarray,
    y: np.ndarray,
    hemisphere: Hemisphere = Hemisphere.LOWER
) -> np.ndarray:
    """
    Inverse Schmidt projection: (x, y) to unit vectors.
    
    Args:
        x: X coordinates on stereonet
        y: Y coordinates on stereonet
        hemisphere: Which hemisphere
    
    Returns:
        Nx3 array of unit vectors
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    
    r = np.sqrt(x**2 + y**2)
    r = np.clip(r, 0, np.sqrt(2))  # Max radius for equal-area
    
    # Inverse formula: theta = 2 * arcsin(r / sqrt(2))
    theta = 2 * np.arcsin(np.clip(r / np.sqrt(2), 0, 1))
    
    # Azimuth
    azimuth = np.arctan2(x, y)
    
    # Unit vector components
    sin_theta = np.sin(theta)
    vx = sin_theta * np.sin(azimuth)
    vy = sin_theta * np.cos(azimuth)
    vz_mag = np.cos(theta)
    
    if hemisphere == Hemisphere.LOWER:
        vz = -vz_mag
    else:
        vz = vz_mag
    
    return np.column_stack([vx, vy, vz])


def inverse_project_wulff(
    x: np.ndarray,
    y: np.ndarray,
    hemisphere: Hemisphere = Hemisphere.LOWER
) -> np.ndarray:
    """
    Inverse Wulff projection: (x, y) to unit vectors.
    
    Args:
        x: X coordinates on stereonet
        y: Y coordinates on stereonet
        hemisphere: Which hemisphere
    
    Returns:
        Nx3 array of unit vectors
    """
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    
    r = np.sqrt(x**2 + y**2)
    
    # Inverse formula: theta = 2 * arctan(r)
    theta = 2 * np.arctan(r)
    
    azimuth = np.arctan2(x, y)
    
    sin_theta = np.sin(theta)
    vx = sin_theta * np.sin(azimuth)
    vy = sin_theta * np.cos(azimuth)
    vz_mag = np.cos(theta)
    
    if hemisphere == Hemisphere.LOWER:
        vz = -vz_mag
    else:
        vz = vz_mag
    
    return np.column_stack([vx, vy, vz])


# =============================================================================
# GREAT CIRCLES AND SMALL CIRCLES
# =============================================================================

def compute_great_circle(
    normal: np.ndarray,
    n_points: int = 180,
    net_type: NetType = NetType.SCHMIDT,
    hemisphere: Hemisphere = Hemisphere.LOWER
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the great circle (plane trace) for a given pole.
    
    Args:
        normal: 3-element unit vector (pole to plane)
        n_points: Number of points to sample along the great circle
        net_type: Projection type
        hemisphere: Which hemisphere to project
    
    Returns:
        Tuple of (x, y) arrays for the great circle path
    """
    normal = np.asarray(normal, dtype=np.float64).flatten()
    normal = normal / np.linalg.norm(normal)
    
    # Generate points along the great circle
    # Great circle is all vectors perpendicular to the pole
    
    # Find two orthogonal vectors in the plane
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, [0, 0, 1])
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Parameterize great circle
    t = np.linspace(0, 2 * np.pi, n_points)
    circle_points = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
    
    # Project
    if net_type == NetType.SCHMIDT:
        x, y = project_schmidt(circle_points, hemisphere)
    else:
        x, y = project_wulff(circle_points, hemisphere)
    
    return x, y


def compute_small_circle(
    axis: np.ndarray,
    cone_angle: float,
    n_points: int = 180,
    net_type: NetType = NetType.SCHMIDT,
    hemisphere: Hemisphere = Hemisphere.LOWER
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a small circle (cone) around a given axis.
    
    Useful for plotting confidence cones.
    
    Args:
        axis: 3-element unit vector (axis of cone)
        cone_angle: Half-angle of cone in degrees
        n_points: Number of points to sample
        net_type: Projection type
        hemisphere: Which hemisphere to project
    
    Returns:
        Tuple of (x, y) arrays for the small circle path
    """
    axis = np.asarray(axis, dtype=np.float64).flatten()
    axis = axis / np.linalg.norm(axis)
    
    # Find two orthogonal vectors perpendicular to axis
    if abs(axis[2]) < 0.9:
        u = np.cross(axis, [0, 0, 1])
    else:
        u = np.cross(axis, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)
    
    # Cone angle
    theta = np.radians(cone_angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Parameterize small circle
    t = np.linspace(0, 2 * np.pi, n_points)
    circle_points = (cos_theta * axis + 
                     sin_theta * np.outer(np.cos(t), u) + 
                     sin_theta * np.outer(np.sin(t), v))
    
    # Normalize
    circle_points = circle_points / np.linalg.norm(circle_points, axis=1, keepdims=True)
    
    # Project
    if net_type == NetType.SCHMIDT:
        x, y = project_schmidt(circle_points, hemisphere)
    else:
        x, y = project_wulff(circle_points, hemisphere)
    
    return x, y


# =============================================================================
# GRID GENERATION
# =============================================================================

def generate_stereonet_grid(
    n_lat: int = 19,
    n_lon: int = 37,
    net_type: NetType = NetType.SCHMIDT,
    hemisphere: Hemisphere = Hemisphere.LOWER
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a stereonet grid for density calculations.
    
    Creates an equal-area or equal-angle grid on the hemisphere.
    
    Args:
        n_lat: Number of latitude divisions (from center to edge)
        n_lon: Number of longitude divisions (around the circle)
        net_type: Projection type
        hemisphere: Which hemisphere
    
    Returns:
        Tuple of:
            - x: Projected x coordinates (n_lat x n_lon)
            - y: Projected y coordinates (n_lat x n_lon)
            - vectors: Unit vectors at grid points (n_lat x n_lon x 3)
    """
    # Latitude from 0 (center/pole) to 90 (edge/equator)
    lat_deg = np.linspace(0, 90, n_lat)
    
    # Longitude from 0 to 360
    lon_deg = np.linspace(0, 360, n_lon, endpoint=False)
    
    # Create meshgrid
    lat_grid, lon_grid = np.meshgrid(lat_deg, lon_deg, indexing='ij')
    
    # Convert to radians
    lat_rad = np.radians(lat_grid)
    lon_rad = np.radians(lon_grid)
    
    # Create unit vectors
    # lat = 0 is the projection axis (vertical)
    # lat = 90 is the equator (horizontal)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    vx = sin_lat * sin_lon
    vy = sin_lat * cos_lon
    
    if hemisphere == Hemisphere.LOWER:
        vz = -cos_lat
    else:
        vz = cos_lat
    
    vectors = np.stack([vx, vy, vz], axis=-1)
    
    # Project to get x, y coordinates
    vectors_flat = vectors.reshape(-1, 3)
    
    if net_type == NetType.SCHMIDT:
        x_flat, y_flat = project_schmidt(vectors_flat, hemisphere)
    else:
        x_flat, y_flat = project_wulff(vectors_flat, hemisphere)
    
    x = x_flat.reshape(n_lat, n_lon)
    y = y_flat.reshape(n_lat, n_lon)
    
    return x, y, vectors


def generate_primitive_circle(n_points: int = 360) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the primitive (outer) circle of the stereonet.
    
    Args:
        n_points: Number of points
    
    Returns:
        Tuple of (x, y) for unit circle
    """
    t = np.linspace(0, 2 * np.pi, n_points)
    x = np.cos(t)
    y = np.sin(t)
    return x, y


def generate_graticule(
    dip_spacing: float = 10.0,
    strike_spacing: float = 10.0,
    net_type: NetType = NetType.SCHMIDT,
    hemisphere: Hemisphere = Hemisphere.LOWER,
    n_points: int = 90
) -> Tuple[list, list]:
    """
    Generate graticule lines for stereonet plotting.
    
    Args:
        dip_spacing: Spacing between dip circles in degrees
        strike_spacing: Spacing between strike lines in degrees
        net_type: Projection type
        hemisphere: Which hemisphere
        n_points: Points per line
    
    Returns:
        Tuple of:
            - dip_circles: List of (x, y) tuples for dip circles
            - strike_lines: List of (x, y) tuples for strike lines
    """
    dip_circles = []
    strike_lines = []
    
    # Dip circles (small circles around vertical axis)
    for dip in np.arange(dip_spacing, 90, dip_spacing):
        x, y = compute_small_circle(
            axis=np.array([0, 0, -1 if hemisphere == Hemisphere.LOWER else 1]),
            cone_angle=dip,
            n_points=n_points,
            net_type=net_type,
            hemisphere=hemisphere
        )
        dip_circles.append((x, y))
    
    # Strike lines (great circles through vertical axis)
    for strike in np.arange(0, 180, strike_spacing):
        # Create a pole perpendicular to the strike direction
        strike_rad = np.radians(strike)
        pole = np.array([np.cos(strike_rad), np.sin(strike_rad), 0])
        
        x, y = compute_great_circle(
            normal=pole,
            n_points=n_points,
            net_type=net_type,
            hemisphere=hemisphere
        )
        strike_lines.append((x, y))
    
    return dip_circles, strike_lines

