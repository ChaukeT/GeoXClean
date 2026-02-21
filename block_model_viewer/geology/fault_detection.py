"""
FaultDetectionEngine - Automatic Fault Plane Suggestion.

Analyzes systematic model errors to suggest missing fault planes.
Required for high-level JORC/SAMREC structural reconciliation.

GeoX Invariant Compliance:
- Suggestions include confidence metrics
- PCA-based plane fitting is deterministic
- All suggestions recorded for audit trail
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

logger = logging.getLogger("GeoX_Compliance")


@dataclass
class SuggestedFault:
    """
    A fault plane suggested by automated error analysis.
    
    Contains geometric parameters and confidence metrics for
    geologist review.
    """
    name: str
    center: np.ndarray  # (3,) center point in world coordinates
    dip: float  # Dip angle in degrees
    dip_dir: float  # Dip direction in degrees (azimuth)
    confidence: float  # 0-1 confidence score based on planarity
    avg_misfit: float  # Average misfit magnitude of cluster points
    n_points: int = 0  # Number of error points in cluster
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON export."""
        return {
            "name": self.name,
            "center": self.center.tolist(),
            "dip": self.dip,
            "dip_dir": self.dip_dir,
            "confidence": self.confidence,
            "avg_misfit": self.avg_misfit,
            "n_points": self.n_points,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @property
    def strike(self) -> float:
        """Calculate strike from dip direction (right-hand rule)."""
        return (self.dip_dir - 90) % 360
    
    @property
    def summary(self) -> str:
        """Human-readable summary for display."""
        return (
            f"{self.name}: Dip {self.dip:.1f}° → {self.dip_dir:.1f}° "
            f"(Strike {self.strike:.1f}°), Confidence: {self.confidence:.0%}"
        )


class FaultDetectionEngine:
    """
    Analyzes systematic model errors to suggest missing fault planes.
    
    This is an advanced QC tool for high-level JORC/SAMREC structural
    reconciliation. When the geological model shows systematic errors
    clustered in space, it often indicates a missing structural control
    (fault, unconformity, intrusion contact).
    
    Algorithm:
    1. Filter high-error points (> threshold)
    2. Cluster errors using DBSCAN
    3. Fit planes to each cluster using PCA
    4. Calculate dip/dip-direction from normal vectors
    5. Assign confidence based on planarity (eigenvalue ratio)
    
    Usage:
        engine = FaultDetectionEngine(error_threshold_m=3.0)
        suggestions = engine.detect_potential_faults(misfit_data)
        for fault in suggestions:
            print(fault.summary)
    """
    
    def __init__(
        self,
        error_threshold_m: float = 3.0,
        cluster_eps: float = 50.0,
        cluster_min_samples: int = 4,
        min_confidence: float = 0.6
    ):
        """
        Initialize the fault detection engine.
        
        Args:
            error_threshold_m: Minimum misfit (meters) to consider as 'high error'
            cluster_eps: DBSCAN eps parameter - max distance between cluster points
            cluster_min_samples: DBSCAN min_samples - minimum cluster size
            min_confidence: Minimum planarity confidence to suggest a fault
        """
        self.error_threshold = error_threshold_m
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_confidence = min_confidence
        
        # Audit trail
        self.detection_log: List[Dict[str, Any]] = []
    
    def detect_potential_faults(
        self,
        misfit_data: pd.DataFrame
    ) -> List[SuggestedFault]:
        """
        Analyze misfit data and suggest potential fault planes.
        
        Args:
            misfit_data: DataFrame with X, Y, Z, residual_m columns
                        (typically from ComplianceManager.generate_misfit_report)
                        
        Returns:
            List of SuggestedFault objects for geologist review
        """
        if 'residual_m' not in misfit_data.columns:
            logger.warning("misfit_data missing 'residual_m' column")
            return []
        
        # 1. Filter points where the model 'missed' significantly
        high_error_pts = misfit_data[misfit_data['residual_m'] > self.error_threshold]
        
        if len(high_error_pts) < self.cluster_min_samples:
            logger.info(
                f"Only {len(high_error_pts)} high-error points found "
                f"(threshold={self.error_threshold}m). No fault suggestions."
            )
            return []
        
        coords = high_error_pts[['X', 'Y', 'Z']].values
        
        # 2. Cluster the errors (find where errors are grouped together)
        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples
        ).fit(coords)
        labels = clustering.labels_
        
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        logger.info(f"Found {len(unique_labels)} error clusters from {len(high_error_pts)} high-error points")
        
        # 3. Analyze each cluster for potential fault plane
        suggestions = []
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_coords = coords[cluster_mask]
            cluster_misfits = high_error_pts.iloc[cluster_mask]['residual_m'].values
            
            avg_err = float(np.mean(cluster_misfits))
            n_points = len(cluster_coords)
            
            # 4. Fit a plane to the error cluster using PCA
            # The smallest principal component is the Normal Vector to the fault
            pca = PCA(n_components=3)
            pca.fit(cluster_coords)
            
            normal_vec = pca.components_[2]  # Direction of least variance
            center = pca.mean_
            
            # 5. Convert Normal Vector to Dip and Dip Direction
            dip, dip_dir = self._normal_to_dip_dipdir(normal_vec)
            
            # 6. Calculate confidence based on planarity
            # High confidence if errors form a clear plane (low 3rd eigenvalue)
            explained_var = pca.explained_variance_ratio_
            if explained_var[1] > 1e-10:
                confidence = 1.0 - (explained_var[2] / explained_var[1])
            else:
                confidence = 0.0
            
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
            
            # Only suggest if it actually looks like a plane
            if confidence >= self.min_confidence:
                fault = SuggestedFault(
                    name=f"Suggested_Fault_{len(suggestions)+1}",
                    center=center,
                    dip=dip,
                    dip_dir=dip_dir,
                    confidence=confidence,
                    avg_misfit=avg_err,
                    n_points=n_points,
                )
                suggestions.append(fault)
                
                logger.info(f"Suggested fault: {fault.summary}")
            else:
                logger.debug(
                    f"Cluster {cluster_id} rejected: confidence {confidence:.2f} "
                    f"< threshold {self.min_confidence}"
                )
        
        # Record detection for audit
        self.detection_log.append({
            "timestamp": datetime.now().isoformat(),
            "n_high_error_points": len(high_error_pts),
            "error_threshold": self.error_threshold,
            "n_clusters": len(unique_labels),
            "n_suggestions": len(suggestions),
            "suggestions": [f.to_dict() for f in suggestions],
        })
        
        return suggestions
    
    @staticmethod
    def _normal_to_dip_dipdir(n: np.ndarray) -> tuple:
        """
        Convert a plane normal vector to standard geological notation.
        
        Dip: Angle of steepest descent from horizontal (0-90°)
        Dip Direction: Azimuth of dip direction (0-360°)
        
        Args:
            n: Normal vector (3,)
            
        Returns:
            Tuple of (dip, dip_direction) in degrees
        """
        n = n.copy()
        
        # Ensure upward-pointing normal
        if n[2] < 0:
            n = -n
        
        # Normalize
        n = n / np.linalg.norm(n)
        
        # Dip is angle from vertical (cos = z component)
        dip = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        
        # Dip direction is azimuth of horizontal projection
        dip_dir = np.degrees(np.arctan2(n[0], n[1])) % 360
        
        return round(dip, 1), round(dip_dir, 1)
    
    @staticmethod
    def dip_dipdir_to_normal(dip: float, dip_dir: float) -> np.ndarray:
        """
        Convert dip/dip-direction to unit normal vector.
        
        Args:
            dip: Dip angle in degrees
            dip_dir: Dip direction (azimuth) in degrees
            
        Returns:
            Unit normal vector (3,)
        """
        dip_rad = np.radians(dip)
        azim_rad = np.radians(dip_dir)
        
        nx = np.sin(dip_rad) * np.sin(azim_rad)
        ny = np.sin(dip_rad) * np.cos(azim_rad)
        nz = np.cos(dip_rad)
        
        return np.array([nx, ny, nz])
    
    def get_detection_log(self) -> List[Dict[str, Any]]:
        """Get the complete detection log for audit purposes."""
        return self.detection_log.copy()
    
    def generate_fault_wireframe(
        self,
        fault: SuggestedFault,
        extent_m: float = 100.0
    ) -> Dict[str, np.ndarray]:
        """
        Generate a simple wireframe mesh for fault visualization.
        
        Args:
            fault: SuggestedFault to visualize
            extent_m: Half-width of the fault plane in meters
            
        Returns:
            Dict with 'vertices' and 'faces' arrays
        """
        # Get normal and center
        normal = self.dip_dipdir_to_normal(fault.dip, fault.dip_dir)
        center = fault.center
        
        # Create two vectors in the fault plane
        # v1 is the strike direction
        v1 = np.array([-normal[1], normal[0], 0])
        if np.linalg.norm(v1) < 1e-10:
            v1 = np.array([1, 0, 0])
        v1 = v1 / np.linalg.norm(v1)
        
        # v2 is perpendicular to both normal and v1
        v2 = np.cross(normal, v1)
        v2 = v2 / np.linalg.norm(v2)
        
        # Create rectangle corners
        vertices = np.array([
            center - extent_m * v1 - extent_m * v2,
            center + extent_m * v1 - extent_m * v2,
            center + extent_m * v1 + extent_m * v2,
            center - extent_m * v1 + extent_m * v2,
        ])
        
        # Two triangles forming a quad
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ])
        
        return {"vertices": vertices, "faces": faces}

