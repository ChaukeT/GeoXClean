"""
Fault geometry handling for geological modeling.

This module provides the FaultPlane class to convert fault parameters
(dip, azimuth, point, displacement) into fault trace points and orientations
that can be consumed by LoopStructural.

GeoX Invariant Compliance:
- All calculations are deterministic
- Fault geometry generation is auditable
- Coordinate systems follow right-hand rule conventions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FaultPlane:
    """
    Represents a fault plane with geometric parameters.
    
    Attributes:
        name: Fault identifier
        point: A point on the fault plane [x, y, z]
        dip: Dip angle in degrees (0-90, measured from horizontal)
        azimuth: Dip direction in degrees (0-360, measured clockwise from north)
        throw_magnitude: Displacement along fault in meters
        throw_direction: Direction of throw (up, down, left, right)
        influence: Influence distance in meters
        active: Whether the fault is active in modeling
        metadata: Additional metadata
    """
    name: str
    point: np.ndarray
    dip: float
    azimuth: float
    throw_magnitude: float = 0.0
    throw_direction: str = "down"
    influence: float = 1000.0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dip_azimuth(
        cls,
        name: str,
        point: np.ndarray,
        dip: float,
        azimuth: float,
        throw_magnitude: float = 0.0,
        throw_direction: str = "down",
        influence: float = 1000.0
    ) -> FaultPlane:
        """
        Create a FaultPlane from dip and azimuth parameters.
        
        Args:
            name: Fault identifier
            point: Point on fault plane [x, y, z]
            dip: Dip angle in degrees (0-90)
            azimuth: Dip direction in degrees (0-360, clockwise from north)
            throw_magnitude: Displacement in meters
            throw_direction: Direction of throw
            influence: Influence distance in meters
            
        Returns:
            FaultPlane instance
        """
        return cls(
            name=name,
            point=np.asarray(point),
            dip=dip,
            azimuth=azimuth,
            throw_magnitude=throw_magnitude,
            throw_direction=throw_direction,
            influence=influence
        )
    
    def get_normal_vector(self) -> np.ndarray:
        """
        Calculate the normal vector to the fault plane.
        
        The normal vector points in the dip direction (downward along the dip).
        Uses right-hand rule: azimuth is dip direction, measured clockwise from north.
        
        Returns:
            Unit normal vector [nx, ny, nz]
        """
        # Convert angles to radians
        dip_rad = np.radians(self.dip)
        az_rad = np.radians(self.azimuth)
        
        # Calculate normal vector components
        # Azimuth is measured clockwise from north (Y-axis)
        # X-axis points East, Y-axis points North, Z-axis points Up
        nx = np.sin(az_rad) * np.sin(dip_rad)  # East component
        ny = np.cos(az_rad) * np.sin(dip_rad)  # North component
        nz = -np.cos(dip_rad)  # Down component (negative because dip is downward)
        
        normal = np.array([nx, ny, nz])
        return normal / np.linalg.norm(normal)  # Normalize
    
    def generate_fault_trace_points(
        self,
        extent: Dict[str, float],
        num_points: int = 20,
        grid_spacing: float = 100.0
    ) -> pd.DataFrame:
        """
        Generate fault trace points across the model extent.
        
        This creates a grid of points on the fault plane that LoopStructural
        can use to interpolate the fault surface.
        
        Args:
            extent: Model bounds {'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'}
            num_points: Number of points along each dimension
            grid_spacing: Approximate spacing between points in meters
            
        Returns:
            DataFrame with columns [X, Y, Z, feature_name]
        """
        # Get fault plane parameters
        center = self.point
        normal = self.get_normal_vector()
        
        # Create two orthogonal vectors in the fault plane
        # Pick an arbitrary vector not parallel to normal
        if abs(normal[2]) < 0.9:
            v1 = np.array([0, 0, 1])
        else:
            v1 = np.array([1, 0, 0])
        
        # Use Gram-Schmidt to get two perpendicular vectors in the plane
        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Determine the extent of the fault plane based on model bounds
        x_range = extent['xmax'] - extent['xmin']
        y_range = extent['ymax'] - extent['ymin']
        z_range = extent['zmax'] - extent['zmin']
        max_extent = np.sqrt(x_range**2 + y_range**2 + z_range**2)
        
        # Generate grid of points on the fault plane
        points = []
        n_grid = int(np.sqrt(num_points))
        
        for i in np.linspace(-max_extent/2, max_extent/2, n_grid):
            for j in np.linspace(-max_extent/2, max_extent/2, n_grid):
                pt = center + i * v1 + j * v2
                
                # Check if point is within model extent
                if (extent['xmin'] <= pt[0] <= extent['xmax'] and
                    extent['ymin'] <= pt[1] <= extent['ymax'] and
                    extent['zmin'] <= pt[2] <= extent['zmax']):
                    points.append(pt)
        
        if not points:
            logger.warning(f"No fault trace points generated for '{self.name}' within extent - using center point")
            # Generate at least a few points near the center to ensure valid data
            # Create points along the fault plane near the center
            spacing = min(x_range, y_range, z_range) * 0.1  # 10% of smallest dimension
            for i in [-spacing, 0, spacing]:
                for j in [-spacing, 0, spacing]:
                    pt = center + i * v1 + j * v2
                    points.append(pt)
        
        # Create DataFrame
        if len(points) == 0:
            # Last resort: single center point
            points = [center]
        
        points_array = np.array(points)
        
        # Ensure we have valid 2D array shape
        if points_array.ndim == 1:
            points_array = points_array.reshape(1, -1)
        
        df = pd.DataFrame({
            'X': points_array[:, 0],
            'Y': points_array[:, 1],
            'Z': points_array[:, 2],
            'feature_name': self.name
        })
        
        logger.info(f"Generated {len(df)} fault trace points for '{self.name}'")
        return df
    
    def generate_fault_orientations(
        self,
        extent: Dict[str, float],
        num_orientations: int = 10
    ) -> pd.DataFrame:
        """
        Generate orientation data for the fault plane.
        
        These orientations constrain the fault plane's dip and direction.
        
        Args:
            extent: Model bounds
            num_orientations: Number of orientation measurements
            
        Returns:
            DataFrame with columns [X, Y, Z, gx, gy, gz, feature_name]
        """
        # Get normal vector (gradient direction)
        normal = self.get_normal_vector()
        
        # Generate orientation points distributed across the fault plane
        # Use the same approach as trace points but with fewer samples
        center = self.point
        
        # Create orthogonal vectors in fault plane
        if abs(normal[2]) < 0.9:
            v1 = np.array([0, 0, 1])
        else:
            v1 = np.array([1, 0, 0])
        
        v1 = v1 - np.dot(v1, normal) * normal
        v1 = v1 / np.linalg.norm(v1)
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Extent
        x_range = extent['xmax'] - extent['xmin']
        y_range = extent['ymax'] - extent['ymin']
        z_range = extent['zmax'] - extent['zmin']
        max_extent = np.sqrt(x_range**2 + y_range**2 + z_range**2)
        
        # Generate orientation points
        orientations = []
        n_grid = int(np.sqrt(num_orientations))
        
        for i in np.linspace(-max_extent/3, max_extent/3, n_grid):
            for j in np.linspace(-max_extent/3, max_extent/3, n_grid):
                pt = center + i * v1 + j * v2
                
                # Check bounds
                if (extent['xmin'] <= pt[0] <= extent['xmax'] and
                    extent['ymin'] <= pt[1] <= extent['ymax'] and
                    extent['zmin'] <= pt[2] <= extent['zmax']):
                    orientations.append(pt)
        
        if not orientations:
            # At least a few orientations near center
            spacing = min(x_range, y_range, z_range) * 0.1
            for i in [-spacing, 0, spacing]:
                for j in [-spacing, 0, spacing]:
                    pt = center + i * v1 + j * v2
                    orientations.append(pt)
        
        if len(orientations) == 0:
            # Last resort: single center point
            orientations = [center]
        
        # Create DataFrame with gradient directions
        orientations_array = np.array(orientations)
        
        # Ensure valid shape
        if orientations_array.ndim == 1:
            orientations_array = orientations_array.reshape(1, -1)
        
        df = pd.DataFrame({
            'X': orientations_array[:, 0],
            'Y': orientations_array[:, 1],
            'Z': orientations_array[:, 2],
            'gx': normal[0],
            'gy': normal[1],
            'gz': normal[2],
            'feature_name': self.name
        })
        
        logger.info(f"Generated {len(df)} fault orientations for '{self.name}'")
        return df
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Returns:
            Dictionary representation
        """
        return {
            'name': self.name,
            'point': self.point.tolist(),
            'dip': self.dip,
            'azimuth': self.azimuth,
            'throw_magnitude': self.throw_magnitude,
            'throw_direction': self.throw_direction,
            'influence': self.influence,
            'displacement': self.throw_magnitude,  # Alias for compatibility
            'type': 'normal',  # Default type
            'active': self.active,
            'metadata': self.metadata,
            # Include normal vector for reference
            'normal': self.get_normal_vector().tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FaultPlane:
        """
        Create FaultPlane from dictionary.
        
        Args:
            data: Dictionary with fault parameters
            
        Returns:
            FaultPlane instance
        """
        return cls(
            name=data.get('name', 'Unnamed_Fault'),
            point=np.array(data.get('point', [0, 0, 0])),
            dip=data.get('dip', 45.0),
            azimuth=data.get('azimuth', 0.0),
            throw_magnitude=data.get('throw_magnitude', data.get('displacement', 0.0)),
            throw_direction=data.get('throw_direction', 'down'),
            influence=data.get('influence', 1000.0),
            active=data.get('active', True),
            metadata=data.get('metadata', {})
        )

