"""
Stereonet Analysis - Compute stereonet coordinates and clustering.

Includes support for structural features (faults, folds, unconformities)
loaded from CSV import.
"""

from typing import List, Tuple, Union, TYPE_CHECKING
import numpy as np
from sklearn.cluster import KMeans

from .datasets import PlaneMeasurement, LineationMeasurement

if TYPE_CHECKING:
    from .feature_types import (
        FaultFeature, FoldFeature, UnconformityFeature, 
        StructuralFeatureCollection
    )


def plane_poles_to_stereonet(planes: List[PlaneMeasurement]) -> np.ndarray:
    """
    Convert plane measurements to stereonet pole vectors.
    
    Args:
        planes: List of PlaneMeasurement objects
    
    Returns:
        Array of shape (N, 3) with pole vectors (x, y, z) on unit sphere
    """
    poles = []
    
    for plane in planes:
        # Convert dip/dip_direction to pole vector
        dip_rad = np.radians(plane.dip)
        dir_rad = np.radians(plane.dip_direction)
        
        # Pole is perpendicular to plane
        # Pole direction is opposite to dip direction, at 90 - dip angle
        pole_plunge = 90.0 - plane.dip
        pole_trend = (plane.dip_direction + 180.0) % 360.0
        
        plunge_rad = np.radians(pole_plunge)
        trend_rad = np.radians(pole_trend)
        
        # Convert to Cartesian coordinates
        x = np.sin(plunge_rad) * np.cos(trend_rad)
        y = np.sin(plunge_rad) * np.sin(trend_rad)
        z = np.cos(plunge_rad)
        
        poles.append([x, y, z])
    
    return np.array(poles)


def density_grid(
    poles: np.ndarray,
    grid_size: int = 100,
    smoothing: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute density grid for stereonet contour plot.
    
    Args:
        poles: Array of pole vectors (N, 3)
        grid_size: Size of density grid
        smoothing: Smoothing parameter (degrees)
    
    Returns:
        Tuple of (x_grid, y_grid, density_grid)
    """
    # Create grid in stereonet coordinates
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Compute density at each grid point
    density = np.zeros_like(x_grid)
    
    for pole in poles:
        # Project pole to stereonet (equal area projection)
        # For equal area: r = sqrt(2) * sin(plunge/2)
        z = pole[2]
        if z < 0:
            continue  # Only plot lower hemisphere
        
        plunge = np.arccos(z)
        r = np.sqrt(2) * np.sin(plunge / 2)
        
        if r > 1:
            continue
        
        # Convert to grid coordinates
        azimuth = np.arctan2(pole[1], pole[0])
        px = r * np.cos(azimuth)
        py = r * np.sin(azimuth)
        
        # Add Gaussian kernel
        dist_sq = (x_grid - px)**2 + (y_grid - py)**2
        sigma_sq = (smoothing / 90.0)**2  # Convert degrees to normalized units
        density += np.exp(-dist_sq / (2 * sigma_sq))
    
    return x_grid, y_grid, density


def cluster_planes(
    planes: List[PlaneMeasurement],
    n_sets: int
) -> List[str]:
    """
    Cluster planes into structural sets using k-means on pole vectors.
    
    Args:
        planes: List of PlaneMeasurement objects
        n_sets: Number of structural sets to identify
    
    Returns:
        List of set_id strings (one per plane)
    """
    if len(planes) < n_sets:
        # Not enough planes to cluster
        return [f"set_{i % n_sets}" for i in range(len(planes))]
    
    # Convert to pole vectors
    poles = plane_poles_to_stereonet(planes)
    
    # Cluster using k-means
    kmeans = KMeans(n_clusters=n_sets, random_state=42, n_init=10)
    labels = kmeans.fit_predict(poles)
    
    # Return set IDs
    return [f"set_{label}" for label in labels]


# =============================================================================
# STRUCTURAL FEATURE SUPPORT
# =============================================================================

def fault_orientations_to_planes(fault_feature: "FaultFeature") -> List[PlaneMeasurement]:
    """
    Extract plane measurements from a FaultFeature.
    
    Args:
        fault_feature: FaultFeature from structural CSV import
        
    Returns:
        List of PlaneMeasurement objects for stereonet analysis
    """
    planes = []
    
    if hasattr(fault_feature, 'orientations'):
        for orient in fault_feature.orientations:
            planes.append(PlaneMeasurement(
                dip=orient.dip,
                dip_direction=orient.azimuth,
                set_id=fault_feature.name,
                metadata={
                    'feature_type': 'fault',
                    'feature_name': fault_feature.name,
                    'x': orient.x,
                    'y': orient.y,
                    'z': orient.z,
                }
            ))
    
    return planes


def unconformity_orientations_to_planes(unconformity_feature: "UnconformityFeature") -> List[PlaneMeasurement]:
    """
    Extract plane measurements from an UnconformityFeature.
    
    Args:
        unconformity_feature: UnconformityFeature from structural CSV import
        
    Returns:
        List of PlaneMeasurement objects for stereonet analysis
    """
    planes = []
    
    if hasattr(unconformity_feature, 'orientations'):
        for orient in unconformity_feature.orientations:
            planes.append(PlaneMeasurement(
                dip=orient.dip,
                dip_direction=orient.azimuth,
                set_id=unconformity_feature.name,
                metadata={
                    'feature_type': 'unconformity',
                    'feature_name': unconformity_feature.name,
                    'x': orient.x,
                    'y': orient.y,
                    'z': orient.z,
                }
            ))
    
    return planes


def fold_axes_to_lineations(fold_feature: "FoldFeature") -> List[LineationMeasurement]:
    """
    Extract lineation measurements from a FoldFeature.
    
    Args:
        fold_feature: FoldFeature from structural CSV import
        
    Returns:
        List of LineationMeasurement objects for stereonet analysis
    """
    lineations = []
    
    if hasattr(fold_feature, 'fold_axes'):
        for axis in fold_feature.fold_axes:
            lineations.append(LineationMeasurement(
                plunge=axis.plunge,
                trend=axis.trend,
                set_id=fold_feature.name,
                metadata={
                    'feature_type': 'fold_axis',
                    'feature_name': fold_feature.name,
                    'fold_style': fold_feature.fold_style.value if hasattr(fold_feature.fold_style, 'value') else str(fold_feature.fold_style),
                }
            ))
    
    return lineations


def fold_limbs_to_planes(fold_feature: "FoldFeature") -> List[PlaneMeasurement]:
    """
    Extract limb plane measurements from a FoldFeature.
    
    Args:
        fold_feature: FoldFeature from structural CSV import
        
    Returns:
        List of PlaneMeasurement objects for stereonet analysis
    """
    planes = []
    
    if hasattr(fold_feature, 'limb_orientations'):
        for limb in fold_feature.limb_orientations:
            # limb_orientations are already PlaneMeasurement objects
            if isinstance(limb, PlaneMeasurement):
                limb.set_id = fold_feature.name
                limb.metadata['feature_type'] = 'fold_limb'
                limb.metadata['feature_name'] = fold_feature.name
                planes.append(limb)
    
    return planes


def collection_to_planes(collection: "StructuralFeatureCollection") -> List[PlaneMeasurement]:
    """
    Extract all plane measurements from a StructuralFeatureCollection.
    
    Combines fault orientations, unconformity orientations, and fold limbs
    into a single list for stereonet analysis.
    
    Args:
        collection: StructuralFeatureCollection from CSV import
        
    Returns:
        List of all PlaneMeasurement objects
    """
    planes = []
    
    # Add fault planes
    for fault in collection.faults:
        planes.extend(fault_orientations_to_planes(fault))
    
    # Add unconformity planes
    for unconformity in collection.unconformities:
        planes.extend(unconformity_orientations_to_planes(unconformity))
    
    # Add fold limb planes
    for fold in collection.folds:
        planes.extend(fold_limbs_to_planes(fold))
    
    return planes


def collection_to_lineations(collection: "StructuralFeatureCollection") -> List[LineationMeasurement]:
    """
    Extract all lineation measurements from a StructuralFeatureCollection.
    
    Extracts fold axes from all fold features.
    
    Args:
        collection: StructuralFeatureCollection from CSV import
        
    Returns:
        List of all LineationMeasurement objects
    """
    lineations = []
    
    for fold in collection.folds:
        lineations.extend(fold_axes_to_lineations(fold))
    
    return lineations


def lineations_to_stereonet(lineations: List[LineationMeasurement]) -> np.ndarray:
    """
    Convert lineation measurements to stereonet point vectors.
    
    Args:
        lineations: List of LineationMeasurement objects (plunge/trend)
        
    Returns:
        Array of shape (N, 3) with lineation vectors on unit sphere
    """
    points = []
    
    for lin in lineations:
        plunge_rad = np.radians(lin.plunge)
        trend_rad = np.radians(lin.trend)
        
        # Convert to Cartesian coordinates
        x = np.cos(plunge_rad) * np.sin(trend_rad)
        y = np.cos(plunge_rad) * np.cos(trend_rad)
        z = -np.sin(plunge_rad)  # Negative because plunge is downward
        
        points.append([x, y, z])
    
    return np.array(points) if points else np.empty((0, 3))


def analyze_structural_collection(
    collection: "StructuralFeatureCollection",
    cluster_planes: bool = True,
    n_clusters: int = 3,
) -> dict:
    """
    Perform comprehensive stereonet analysis on a structural feature collection.
    
    Args:
        collection: StructuralFeatureCollection from CSV import
        cluster_planes: Whether to perform K-means clustering
        n_clusters: Number of clusters for K-means
        
    Returns:
        Dict with analysis results:
        - 'planes': List of PlaneMeasurement objects
        - 'lineations': List of LineationMeasurement objects
        - 'plane_poles': np.ndarray of pole vectors
        - 'lineation_vectors': np.ndarray of lineation vectors
        - 'density_grid': (x, y, density) tuple for contour plotting
        - 'clusters': List of cluster IDs (if clustering enabled)
        - 'statistics': Summary statistics
    """
    results = {
        'planes': [],
        'lineations': [],
        'plane_poles': np.empty((0, 3)),
        'lineation_vectors': np.empty((0, 3)),
        'density_grid': None,
        'clusters': [],
        'statistics': {},
    }
    
    # Extract all measurements
    planes = collection_to_planes(collection)
    lineations = collection_to_lineations(collection)
    
    results['planes'] = planes
    results['lineations'] = lineations
    
    # Convert to stereonet vectors
    if planes:
        results['plane_poles'] = plane_poles_to_stereonet(planes)
        
        # Compute density grid
        try:
            x, y, density = density_grid(results['plane_poles'])
            results['density_grid'] = (x, y, density)
        except Exception as e:
            pass  # Skip if density computation fails
        
        # Cluster if requested
        if cluster_planes and len(planes) >= n_clusters:
            try:
                from .stereonet import cluster_planes as do_cluster
                results['clusters'] = do_cluster(planes, n_clusters)
            except Exception:
                results['clusters'] = [f"set_0"] * len(planes)
    
    if lineations:
        results['lineation_vectors'] = lineations_to_stereonet(lineations)
    
    # Compute statistics
    results['statistics'] = {
        'n_planes': len(planes),
        'n_lineations': len(lineations),
        'n_faults': len(collection.faults),
        'n_folds': len(collection.folds),
        'n_unconformities': len(collection.unconformities),
    }
    
    # Mean orientation statistics
    if planes:
        dips = [p.dip for p in planes]
        results['statistics']['mean_dip'] = np.mean(dips)
        results['statistics']['std_dip'] = np.std(dips)
    
    if lineations:
        plunges = [l.plunge for l in lineations]
        results['statistics']['mean_plunge'] = np.mean(plunges)
        results['statistics']['std_plunge'] = np.std(plunges)
    
    return results

