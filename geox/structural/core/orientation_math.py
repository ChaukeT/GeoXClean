"""
Orientation Mathematics - Vector-Based Operations.

All orientations are stored as unit vectors. Angles are derived, never stored.
Implements vectorized operations for efficiency.

Conventions:
- Dip: 0-90 degrees, angle from horizontal
- Dip Direction: 0-360 degrees, azimuth of dip direction (clockwise from north)
- Strike: dip_direction - 90 (right-hand rule)
- Plunge: 0-90 degrees, angle below horizontal
- Trend: 0-360 degrees, azimuth of plunge direction

Alpha/Beta (oriented core/televiewer style):
- Alpha: acute angle between plane and borehole axis (0-90)
- Beta: clockwise angle around core from reference line (Top-of-hole/Highside) to plane trace (0-360)
- Reference line is typically the highside mark or north-oriented scribe

Coordinate System:
- X: East
- Y: North  
- Z: Up (right-handed)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Union


# =============================================================================
# DIP / DIP-DIRECTION CONVERSIONS
# =============================================================================

def dip_dipdir_to_normal(
    dip: Union[float, np.ndarray],
    dip_direction: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Convert dip/dip-direction to unit normal vectors (poles to planes).
    
    The normal vector points in the direction of the steepest descent
    (down-dip direction).
    
    Args:
        dip: Dip angle(s) in degrees (0-90)
        dip_direction: Dip direction(s) in degrees (0-360, azimuth)
    
    Returns:
        Nx3 array of unit normal vectors
    
    Convention:
        - Normal points DOWN-dip (into the lower hemisphere for plotting poles)
        - For a horizontal plane (dip=0), normal is vertical downward [0, 0, -1]
        - For a vertical plane (dip=90), normal is horizontal
    """
    dip = np.atleast_1d(np.asarray(dip, dtype=np.float64))
    dip_direction = np.atleast_1d(np.asarray(dip_direction, dtype=np.float64))
    
    # Convert to radians
    dip_rad = np.radians(dip)
    dd_rad = np.radians(dip_direction)
    
    # Normal vector components
    # The normal points in the dip direction (down-dip)
    # X = East, Y = North, Z = Up
    nx = np.sin(dip_rad) * np.sin(dd_rad)  # East component
    ny = np.sin(dip_rad) * np.cos(dd_rad)  # North component
    nz = -np.cos(dip_rad)                   # Down component (negative Z for down-dip)
    
    normals = np.column_stack([nx, ny, nz])
    
    return normals


def normal_to_dip_dipdir(normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert unit normal vectors to dip/dip-direction.
    
    Args:
        normals: Nx3 array of unit normal vectors
    
    Returns:
        Tuple of (dip, dip_direction) arrays in degrees
    
    Note:
        Normals pointing upward (positive Z) are treated as overturned
        and the dip direction is flipped 180 degrees.
    """
    normals = np.atleast_2d(normals)
    
    # Handle normals pointing up by flipping them down
    # (convention: poles in lower hemisphere)
    flip_mask = normals[:, 2] > 0
    normals_lower = normals.copy()
    normals_lower[flip_mask] = -normals_lower[flip_mask]
    
    # Dip from vertical component
    # dip = arccos(|nz|) but since nz is negative, dip = arccos(-nz)
    dip = np.degrees(np.arccos(-normals_lower[:, 2]))
    
    # Dip direction from horizontal components
    # atan2(nx, ny) gives azimuth
    dip_direction = np.degrees(np.arctan2(normals_lower[:, 0], normals_lower[:, 1]))
    dip_direction = dip_direction % 360  # Wrap to 0-360
    
    return dip, dip_direction


# =============================================================================
# PLUNGE / TREND CONVERSIONS (for lineations)
# =============================================================================

def plunge_trend_to_vector(
    plunge: Union[float, np.ndarray],
    trend: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Convert plunge/trend to unit direction vectors.
    
    Args:
        plunge: Plunge angle(s) in degrees (0-90, positive downward)
        trend: Trend azimuth(s) in degrees (0-360, clockwise from north)
    
    Returns:
        Nx3 array of unit direction vectors
    
    Convention:
        - Vector points in the direction of plunge (downward)
        - Horizontal lineation (plunge=0) has Z=0
        - Vertical lineation (plunge=90) points straight down [0, 0, -1]
    """
    plunge = np.atleast_1d(np.asarray(plunge, dtype=np.float64))
    trend = np.atleast_1d(np.asarray(trend, dtype=np.float64))
    
    # Convert to radians
    plunge_rad = np.radians(plunge)
    trend_rad = np.radians(trend)
    
    # Direction vector components
    vx = np.cos(plunge_rad) * np.sin(trend_rad)  # East
    vy = np.cos(plunge_rad) * np.cos(trend_rad)  # North
    vz = -np.sin(plunge_rad)                      # Down (negative Z)
    
    vectors = np.column_stack([vx, vy, vz])
    
    return vectors


def vector_to_plunge_trend(vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert unit direction vectors to plunge/trend.
    
    Args:
        vectors: Nx3 array of unit direction vectors
    
    Returns:
        Tuple of (plunge, trend) arrays in degrees
    """
    vectors = np.atleast_2d(vectors)
    
    # Ensure vectors point downward (or horizontal)
    # Flip those pointing up
    flip_mask = vectors[:, 2] > 0
    vectors_down = vectors.copy()
    vectors_down[flip_mask] = -vectors_down[flip_mask]
    
    # Plunge from vertical component
    # plunge = arcsin(|vz|)
    plunge = np.degrees(np.arcsin(-vectors_down[:, 2]))
    
    # Trend from horizontal components
    trend = np.degrees(np.arctan2(vectors_down[:, 0], vectors_down[:, 1]))
    trend = trend % 360
    
    return plunge, trend


# =============================================================================
# ALPHA / BETA CONVERSIONS (oriented core measurements)
# =============================================================================

def alpha_beta_to_normal(
    alpha: Union[float, np.ndarray],
    beta: Union[float, np.ndarray],
    hole_azimuth: Union[float, np.ndarray],
    hole_dip: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Convert alpha/beta measurements to world-space normal vectors.
    
    Uses oriented-core/televiewer convention:
    - Alpha: acute angle between plane and borehole axis (0-90 degrees)
    - Beta: clockwise angle around core from reference line (highside/top-of-hole)
           to the trace of the plane on the core surface (0-360 degrees)
    
    The reference line is the highside mark (top of hole when looking down-hole).
    
    Args:
        alpha: Alpha angle(s) in degrees (0-90)
        beta: Beta angle(s) in degrees (0-360)
        hole_azimuth: Borehole azimuth(s) in degrees (0-360, direction of drilling)
        hole_dip: Borehole dip(s) in degrees (negative for downward, -90 for vertical)
    
    Returns:
        Nx3 array of unit normal vectors in world coordinates
    
    Algorithm:
        1. Compute pole to plane in core coordinates (core axis = Z)
        2. Rotate from core coordinates to world coordinates using hole orientation
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=np.float64))
    beta = np.atleast_1d(np.asarray(beta, dtype=np.float64))
    hole_azimuth = np.atleast_1d(np.asarray(hole_azimuth, dtype=np.float64))
    hole_dip = np.atleast_1d(np.asarray(hole_dip, dtype=np.float64))
    
    # Broadcast to same length
    n = max(len(alpha), len(beta), len(hole_azimuth), len(hole_dip))
    if len(alpha) == 1:
        alpha = np.full(n, alpha[0])
    if len(beta) == 1:
        beta = np.full(n, beta[0])
    if len(hole_azimuth) == 1:
        hole_azimuth = np.full(n, hole_azimuth[0])
    if len(hole_dip) == 1:
        hole_dip = np.full(n, hole_dip[0])
    
    # Convert to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    az_rad = np.radians(hole_azimuth)
    dip_rad = np.radians(-hole_dip)  # Convert to positive angle from horizontal
    
    # Step 1: Pole in core reference frame
    # In core frame: Z-axis along core (down-hole), X-axis = highside (reference)
    # The pole makes angle alpha with the core axis
    # Beta is the rotation around the core axis from the highside
    
    # Pole in core coordinates
    pole_core_x = np.sin(alpha_rad) * np.cos(beta_rad)  # Highside direction
    pole_core_y = np.sin(alpha_rad) * np.sin(beta_rad)  # 90° from highside
    pole_core_z = np.cos(alpha_rad)                      # Along core axis
    
    # Step 2: Build rotation matrix from core to world coordinates
    # This requires two rotations:
    # 1. Rotate around X (east) axis by (90 - hole_inclination) to orient Z to hole direction
    # 2. Rotate around Z (up) axis by hole_azimuth
    
    normals = np.zeros((n, 3))
    
    for i in range(n):
        # Core to world transformation
        # First: rotate to align core Z with hole direction
        # Hole direction vector
        hole_x = np.sin(dip_rad[i]) * np.sin(az_rad[i])
        hole_y = np.sin(dip_rad[i]) * np.cos(az_rad[i])
        hole_z = -np.cos(dip_rad[i])  # Negative because holes go down
        
        hole_dir = np.array([hole_x, hole_y, hole_z])
        
        # Highside direction (perpendicular to hole, pointing up in vertical plane)
        # For a vertical hole, highside could be any horizontal direction (use north)
        if np.abs(np.cos(dip_rad[i])) > 0.999:  # Nearly vertical hole
            # Use north as reference for nearly vertical holes
            highside = np.array([0, 1, 0])
        else:
            # Highside is in the vertical plane containing the hole, perpendicular to hole, pointing up
            # It's the direction you'd see at the "top" of the core when looking down-hole
            up = np.array([0, 0, 1])
            # Project up onto plane perpendicular to hole direction
            highside = up - np.dot(up, hole_dir) * hole_dir
            highside_norm = np.linalg.norm(highside)
            if highside_norm > 1e-10:
                highside = highside / highside_norm
            else:
                highside = np.array([0, 1, 0])  # Fallback
        
        # Third axis (right-hand rule)
        third = np.cross(hole_dir, highside)
        third = third / np.linalg.norm(third)
        
        # Rotation matrix: columns are [highside, third, hole_dir]
        # This transforms from core frame to world frame
        R = np.column_stack([highside, third, hole_dir])
        
        # Transform pole from core to world
        pole_core = np.array([pole_core_x[i], pole_core_y[i], pole_core_z[i]])
        pole_world = R @ pole_core
        
        # Normalize
        pole_world = pole_world / np.linalg.norm(pole_world)
        
        normals[i] = pole_world
    
    return normals


def normal_to_alpha_beta(
    normals: np.ndarray,
    hole_azimuth: Union[float, np.ndarray],
    hole_dip: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert world-space normals to alpha/beta measurements.
    
    Inverse of alpha_beta_to_normal.
    
    Args:
        normals: Nx3 array of unit normal vectors
        hole_azimuth: Borehole azimuth(s) in degrees
        hole_dip: Borehole dip(s) in degrees (negative for downward)
    
    Returns:
        Tuple of (alpha, beta) arrays in degrees
    """
    normals = np.atleast_2d(normals)
    hole_azimuth = np.atleast_1d(np.asarray(hole_azimuth, dtype=np.float64))
    hole_dip = np.atleast_1d(np.asarray(hole_dip, dtype=np.float64))
    
    n = len(normals)
    if len(hole_azimuth) == 1:
        hole_azimuth = np.full(n, hole_azimuth[0])
    if len(hole_dip) == 1:
        hole_dip = np.full(n, hole_dip[0])
    
    az_rad = np.radians(hole_azimuth)
    dip_rad = np.radians(-hole_dip)
    
    alphas = np.zeros(n)
    betas = np.zeros(n)
    
    for i in range(n):
        # Build same rotation matrix as forward transform
        hole_x = np.sin(dip_rad[i]) * np.sin(az_rad[i])
        hole_y = np.sin(dip_rad[i]) * np.cos(az_rad[i])
        hole_z = -np.cos(dip_rad[i])
        hole_dir = np.array([hole_x, hole_y, hole_z])
        
        if np.abs(np.cos(dip_rad[i])) > 0.999:
            highside = np.array([0, 1, 0])
        else:
            up = np.array([0, 0, 1])
            highside = up - np.dot(up, hole_dir) * hole_dir
            highside_norm = np.linalg.norm(highside)
            if highside_norm > 1e-10:
                highside = highside / highside_norm
            else:
                highside = np.array([0, 1, 0])
        
        third = np.cross(hole_dir, highside)
        third = third / np.linalg.norm(third)
        
        R = np.column_stack([highside, third, hole_dir])
        
        # Inverse transform: world to core
        R_inv = R.T
        pole_core = R_inv @ normals[i]
        
        # Extract alpha and beta
        alpha = np.degrees(np.arccos(np.clip(pole_core[2], -1, 1)))
        beta = np.degrees(np.arctan2(pole_core[1], pole_core[0]))
        beta = beta % 360
        
        alphas[i] = alpha
        betas[i] = beta
    
    return alphas, betas


# =============================================================================
# HEMISPHERE OPERATIONS
# =============================================================================

def canonicalize_to_lower_hemisphere(normals: np.ndarray) -> np.ndarray:
    """
    Flip normals to point into the lower hemisphere (negative Z).
    
    This is essential for clustering plane poles since antipodal vectors
    represent the same plane.
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Nx3 array with all vectors having Z <= 0
    """
    normals = np.atleast_2d(normals).copy()
    
    # Flip vectors pointing up
    flip_mask = normals[:, 2] > 0
    normals[flip_mask] = -normals[flip_mask]
    
    return normals


def canonicalize_to_upper_hemisphere(normals: np.ndarray) -> np.ndarray:
    """
    Flip normals to point into the upper hemisphere (positive Z).
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        Nx3 array with all vectors having Z >= 0
    """
    normals = np.atleast_2d(normals).copy()
    
    flip_mask = normals[:, 2] < 0
    normals[flip_mask] = -normals[flip_mask]
    
    return normals


# =============================================================================
# VECTOR UTILITIES
# =============================================================================

def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length.
    
    Args:
        vectors: Nx3 array of vectors
    
    Returns:
        Nx3 array of unit vectors
    """
    vectors = np.atleast_2d(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    return vectors / norms


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute angle between two sets of vectors.
    
    Args:
        v1: Nx3 array of unit vectors (or single vector)
        v2: Nx3 array of unit vectors (or single vector)
    
    Returns:
        Array of angles in degrees
    """
    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)
    
    # Dot product
    dots = np.sum(v1 * v2, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    
    return np.degrees(np.arccos(dots))


def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix from axis-angle representation.
    
    Args:
        axis: Unit vector defining rotation axis
        angle: Rotation angle in degrees
    
    Returns:
        3x3 rotation matrix
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    
    angle_rad = np.radians(angle)
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t = 1 - c
    
    x, y, z = axis
    
    R = np.array([
        [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
        [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
        [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
    ])
    
    return R


def rotate_vectors(vectors: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Rotate vectors by a rotation matrix.
    
    Args:
        vectors: Nx3 array of vectors
        R: 3x3 rotation matrix
    
    Returns:
        Nx3 array of rotated vectors
    """
    vectors = np.atleast_2d(vectors)
    return vectors @ R.T


def cross_product_vectorized(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute cross product for arrays of vectors.
    
    Args:
        v1: Nx3 array of vectors
        v2: Nx3 array of vectors (or single vector to broadcast)
    
    Returns:
        Nx3 array of cross products
    """
    v1 = np.atleast_2d(v1)
    v2 = np.atleast_2d(v2)
    
    return np.cross(v1, v2)


def strike_from_dip_direction(dip_direction: Union[float, np.ndarray]) -> np.ndarray:
    """
    Compute strike from dip direction using right-hand rule.
    
    Strike = dip_direction - 90
    
    Args:
        dip_direction: Dip direction(s) in degrees
    
    Returns:
        Strike angle(s) in degrees (0-360)
    """
    dip_direction = np.atleast_1d(np.asarray(dip_direction, dtype=np.float64))
    strike = (dip_direction - 90) % 360
    return strike


def dip_direction_from_strike(strike: Union[float, np.ndarray]) -> np.ndarray:
    """
    Compute dip direction from strike using right-hand rule.
    
    Dip direction = strike + 90
    
    Args:
        strike: Strike angle(s) in degrees
    
    Returns:
        Dip direction(s) in degrees (0-360)
    """
    strike = np.atleast_1d(np.asarray(strike, dtype=np.float64))
    dip_direction = (strike + 90) % 360
    return dip_direction

