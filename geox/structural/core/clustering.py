"""
Structural Set Clustering - HDBSCAN/DBSCAN in Orientation Space.

Identifies structural sets (joint sets, bedding families, etc.) from
orientation data using density-based clustering.

Key Features:
- Hemisphere canonicalization for axial data (plane poles)
- HDBSCAN preferred (better at finding clusters of varying density)
- DBSCAN fallback if HDBSCAN not available
- Deterministic with explicit random seed
- Per-set Fisher statistics
"""

from __future__ import annotations

import logging
import numpy as np
from typing import List, Optional, Tuple, Dict, Any

from .models import OrientationData, StructuralSet
from .orientation_math import canonicalize_to_lower_hemisphere
from .spherical_stats import fisher_mean, fisher_kappa, fisher_statistics, confidence_cone

logger = logging.getLogger(__name__)


# Check for HDBSCAN availability
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.debug("hdbscan not available, will use sklearn DBSCAN")


def cluster_orientations(
    data: OrientationData,
    method: str = "auto",
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None,
    eps: Optional[float] = None,
    metric: str = "angular",
    random_seed: int = 42,
) -> Tuple[np.ndarray, List[StructuralSet]]:
    """
    Cluster orientation data into structural sets.
    
    Args:
        data: OrientationData with unit normal vectors
        method: "hdbscan", "dbscan", "kmeans", or "auto" (prefer hdbscan)
        n_clusters: Number of clusters (required for kmeans, ignored for dbscan/hdbscan)
        min_cluster_size: Minimum cluster size for hdbscan
        min_samples: Core sample threshold for dbscan/hdbscan
        eps: Neighborhood radius for dbscan (in angular degrees)
        metric: "angular" (recommended) or "euclidean"
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of:
            - labels: Array of cluster labels (-1 for noise)
            - sets: List of StructuralSet objects with statistics
    """
    normals = data.normals.copy()
    
    # Canonicalize to lower hemisphere for axial data
    normals = canonicalize_to_lower_hemisphere(normals)
    
    n = len(normals)
    if n < min_cluster_size:
        # Too few points to cluster
        labels = np.zeros(n, dtype=int)
        sets = [_create_structural_set("set_0", normals)]
        return labels, sets
    
    # Select method
    if method == "auto":
        if HDBSCAN_AVAILABLE:
            method = "hdbscan"
        else:
            method = "dbscan"
    
    # Prepare distance metric
    if metric == "angular":
        # Compute pairwise angular distances
        distance_matrix = _compute_angular_distance_matrix(normals)
        metric_param = "precomputed"
    else:
        distance_matrix = normals
        metric_param = "euclidean"
    
    # Run clustering
    if method == "hdbscan":
        labels = _cluster_hdbscan(
            distance_matrix, 
            metric_param, 
            min_cluster_size, 
            min_samples
        )
    elif method == "dbscan":
        if eps is None:
            # Default eps based on data dispersion
            eps = _estimate_eps(normals)
        labels = _cluster_dbscan(
            distance_matrix, 
            metric_param, 
            eps, 
            min_samples or min_cluster_size
        )
    elif method == "kmeans":
        if n_clusters is None:
            n_clusters = _estimate_n_clusters(normals)
        labels = _cluster_kmeans(normals, n_clusters, random_seed)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Create StructuralSet objects
    sets = _create_sets_from_labels(normals, labels)
    
    return labels, sets


def identify_sets(
    data: OrientationData,
    n_sets: Optional[int] = None,
    method: str = "auto",
    random_seed: int = 42,
) -> List[StructuralSet]:
    """
    Convenience function to identify structural sets.
    
    Args:
        data: OrientationData
        n_sets: Optional number of sets (if known)
        method: Clustering method
        random_seed: Random seed
    
    Returns:
        List of StructuralSet objects
    """
    labels, sets = cluster_orientations(
        data,
        method=method,
        n_clusters=n_sets,
        random_seed=random_seed
    )
    
    # Update data with set assignments
    data.set_ids = np.array([s.set_id if i >= 0 else "noise" 
                             for i, s in zip(labels, [None] + sets)])
    
    return sets


def merge_sets(
    sets: List[StructuralSet],
    angle_threshold: float = 15.0
) -> List[StructuralSet]:
    """
    Merge similar structural sets based on mean direction.
    
    Args:
        sets: List of StructuralSet objects
        angle_threshold: Maximum angle between mean directions to merge (degrees)
    
    Returns:
        Merged list of StructuralSet objects
    """
    if len(sets) <= 1:
        return sets
    
    # Compute pairwise angles between set means
    n_sets = len(sets)
    merged = [False] * n_sets
    merged_sets = []
    
    for i in range(n_sets):
        if merged[i]:
            continue
        
        # Find sets to merge with i
        to_merge = [i]
        for j in range(i + 1, n_sets):
            if merged[j]:
                continue
            
            # Angular distance between means
            dot = np.dot(sets[i].mean_normal, sets[j].mean_normal)
            angle = np.degrees(np.arccos(np.clip(np.abs(dot), 0, 1)))
            
            if angle < angle_threshold:
                to_merge.append(j)
                merged[j] = True
        
        # Combine normals from all sets to merge
        combined_normals = np.vstack([sets[k].normals for k in to_merge])
        
        # Create new merged set
        new_set = _create_structural_set(f"set_{len(merged_sets)}", combined_normals)
        merged_sets.append(new_set)
        merged[i] = True
    
    return merged_sets


def compute_set_statistics(structural_set: StructuralSet) -> Dict[str, Any]:
    """
    Compute detailed statistics for a structural set.
    
    Args:
        structural_set: StructuralSet object
    
    Returns:
        Dictionary with comprehensive statistics
    """
    stats = fisher_statistics(structural_set.normals)
    
    # Add set-specific info
    stats["set_id"] = structural_set.set_id
    stats["n_members"] = structural_set.n_members
    
    return stats


# =============================================================================
# INTERNAL FUNCTIONS
# =============================================================================

def _compute_angular_distance_matrix(normals: np.ndarray) -> np.ndarray:
    """
    Compute pairwise angular distance matrix.
    
    For axial data (plane poles), use minimum of angle and 180-angle.
    
    Args:
        normals: Nx3 array of unit vectors
    
    Returns:
        NxN distance matrix in degrees
    """
    n = len(normals)
    
    # Compute all pairwise dot products
    dots = normals @ normals.T
    dots = np.clip(dots, -1.0, 1.0)
    
    # Angles in degrees
    angles = np.degrees(np.arccos(np.abs(dots)))  # Use abs for axial data
    
    # Ensure diagonal is zero
    np.fill_diagonal(angles, 0.0)
    
    return angles


def _estimate_eps(normals: np.ndarray) -> float:
    """
    Estimate a good eps value for DBSCAN based on data characteristics.
    
    Uses a heuristic based on nearest neighbor distances.
    """
    n = len(normals)
    if n < 3:
        return 20.0
    
    # Compute pairwise distances
    dist_matrix = _compute_angular_distance_matrix(normals)
    
    # Find k-th nearest neighbor distance for each point (k = min_samples)
    k = min(5, n - 1)
    knn_distances = np.sort(dist_matrix, axis=1)[:, k]
    
    # Use the "elbow" point or a percentile
    eps = np.percentile(knn_distances, 90)
    
    return max(5.0, min(30.0, eps))


def _estimate_n_clusters(normals: np.ndarray) -> int:
    """
    Estimate number of clusters using silhouette analysis.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    n = len(normals)
    if n < 4:
        return 1
    
    max_k = min(8, n // 3)
    best_k = 1
    best_score = -1
    
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(normals)
        
        if len(np.unique(labels)) < 2:
            continue
        
        score = silhouette_score(normals, labels)
        if score > best_score:
            best_score = score
            best_k = k
    
    return best_k


def _cluster_hdbscan(
    distance_matrix: np.ndarray,
    metric: str,
    min_cluster_size: int,
    min_samples: Optional[int]
) -> np.ndarray:
    """Run HDBSCAN clustering."""
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan package not installed")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method='eom',  # Excess of mass
    )
    
    if metric == "precomputed":
        labels = clusterer.fit_predict(distance_matrix)
    else:
        labels = clusterer.fit_predict(distance_matrix)
    
    return labels


def _cluster_dbscan(
    distance_matrix: np.ndarray,
    metric: str,
    eps: float,
    min_samples: int
) -> np.ndarray:
    """Run DBSCAN clustering."""
    from sklearn.cluster import DBSCAN
    
    clusterer = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
    )
    
    if metric == "precomputed":
        labels = clusterer.fit_predict(distance_matrix)
    else:
        labels = clusterer.fit_predict(distance_matrix)
    
    return labels


def _cluster_kmeans(
    normals: np.ndarray,
    n_clusters: int,
    random_seed: int
) -> np.ndarray:
    """Run KMeans clustering."""
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_seed,
        n_init=10,
    )
    
    labels = kmeans.fit_predict(normals)
    
    return labels


def _create_sets_from_labels(
    normals: np.ndarray,
    labels: np.ndarray
) -> List[StructuralSet]:
    """Create StructuralSet objects from cluster labels."""
    sets = []
    unique_labels = np.unique(labels)
    
    # Skip noise label (-1)
    cluster_labels = [l for l in unique_labels if l >= 0]
    
    for i, label in enumerate(cluster_labels):
        mask = labels == label
        cluster_normals = normals[mask]
        
        structural_set = _create_structural_set(f"set_{i}", cluster_normals)
        sets.append(structural_set)
    
    return sets


def _create_structural_set(set_id: str, normals: np.ndarray) -> StructuralSet:
    """Create a single StructuralSet with computed statistics."""
    normals = np.atleast_2d(normals)
    
    # Compute Fisher statistics
    mean_dir = fisher_mean(normals)
    kappa = fisher_kappa(normals)
    conf_95 = confidence_cone(normals, confidence=0.95)
    
    # Dispersion in degrees
    from .spherical_stats import spherical_dispersion
    dispersion = spherical_dispersion(normals)
    
    return StructuralSet(
        set_id=set_id,
        normals=normals,
        mean_normal=mean_dir,
        kappa=kappa,
        confidence_cone_95=conf_95,
        dispersion=dispersion,
        n_members=len(normals),
    )


def compute_clustering_quality(
    normals: np.ndarray,
    labels: np.ndarray
) -> Dict[str, float]:
    """
    Compute clustering quality metrics.
    
    Args:
        normals: Nx3 array of unit vectors
        labels: Cluster labels
    
    Returns:
        Dictionary with quality metrics
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    unique_labels = np.unique(labels)
    n_clusters = len([l for l in unique_labels if l >= 0])
    n_noise = np.sum(labels == -1)
    
    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / len(labels),
    }
    
    # Only compute scores if we have multiple clusters and not all noise
    if n_clusters >= 2 and n_noise < len(labels) - 1:
        # Remove noise points for scoring
        non_noise = labels >= 0
        if np.sum(non_noise) > n_clusters:
            try:
                metrics["silhouette_score"] = silhouette_score(
                    normals[non_noise], 
                    labels[non_noise]
                )
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                    normals[non_noise],
                    labels[non_noise]
                )
            except Exception:
                pass
    
    return metrics

