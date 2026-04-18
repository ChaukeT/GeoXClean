"""
Ellipsoidal search neighbourhood with octant control.

Uses scipy.spatial.cKDTree for fast spatial lookup, then applies
octant filtering to ensure spatial coverage around each query point.

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .config import RBFConfig
from ..utils.distance import _rotation_matrix

logger = logging.getLogger(__name__)


@dataclass
class SearchStats:
    """Per-block search statistics for JORC audit."""

    num_samples: int
    num_octants: int
    avg_distance: float
    search_pass: int  # 1=half-range, 2=full-range, 3=double-range
    selected_indices: Optional[NDArray[np.intp]] = None  # indices into original data


class SearchNeighbourhood:
    """
    Ellipsoidal search with octant control.

    Parameters match Leapfrog Geo / Datamine conventions.  The KDTree
    is built once on the isotropic-equivalent coordinates (scaled by
    the inverse ellipsoid ratios after rotation) so that a single
    ball query in the transformed space corresponds to an ellipsoidal
    query in the original space.
    """

    def __init__(
        self,
        config: RBFConfig,
        data_points: NDArray[np.float64],
    ) -> None:
        self.config = config
        self.data_points = np.asarray(data_points, dtype=np.float64)

        # Transform points into anisotropic space for KDTree
        self._R = _rotation_matrix(config.azimuth, config.dip, config.pitch)
        self._scale = np.array(
            [1.0 / config.ratio_major, 1.0 / config.ratio_semi, 1.0 / config.ratio_minor]
        )

        self._transformed = (self.data_points @ self._R.T) * self._scale
        self._tree = cKDTree(self._transformed)

        logger.debug(
            "SearchNeighbourhood built: %d points, max=%d, min=%d, octants=%d",
            len(data_points),
            config.search_max_samples,
            config.search_min_samples,
            config.search_min_octants,
        )

    def _transform_query(self, query: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform a query point into anisotropic KDTree space."""
        return (query @ self._R.T) * self._scale

    def select(
        self,
        query_point: NDArray[np.float64],
        all_values: NDArray[np.float64],
        search_radius: Optional[float] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], SearchStats]:
        """
        Select samples for one query location using octant-controlled search.

        Parameters
        ----------
        query_point : (3,) ndarray
        all_values : (N,) ndarray
        search_radius : float, optional
            Override the base_range for multi-pass search.

        Returns
        -------
        selected_points : (K, 3) ndarray
        selected_values : (K,) ndarray
        stats : SearchStats
        """
        query_point = np.asarray(query_point, dtype=np.float64).ravel()
        radius = search_radius if search_radius is not None else self.config.base_range

        # Query KDTree in transformed space
        q_transformed = self._transform_query(query_point)

        # Ball query in transformed (isotropic-equivalent) space.
        # Transform: x_t = (x @ R.T) * [1/ratio_major, 1/ratio_semi, 1/ratio_minor]
        # Euclidean distance in transformed space corresponds to an
        # ellipsoid with semi-axes (radius×ratio_major, radius×ratio_semi,
        # radius×ratio_minor) in original rotated coordinates.
        candidates_idx = self._tree.query_ball_point(q_transformed, r=radius)

        if len(candidates_idx) == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                SearchStats(0, 0, 0.0, 0),
            )

        candidates_idx = np.array(candidates_idx, dtype=np.intp)
        cand_points = self.data_points[candidates_idx]
        cand_values = all_values[candidates_idx]

        # Compute actual anisotropic distances for sorting
        cand_transformed = self._transformed[candidates_idx]
        dists = np.linalg.norm(cand_transformed - q_transformed, axis=1)

        # Sort by distance
        order = np.argsort(dists)
        candidates_idx = candidates_idx[order]
        cand_points = cand_points[order]
        cand_values = cand_values[order]
        dists = dists[order]

        # Octant assignment
        diff = cand_points - query_point[np.newaxis, :]
        octants = (
            (diff[:, 0] >= 0).astype(np.int32) * 4
            + (diff[:, 1] >= 0).astype(np.int32) * 2
            + (diff[:, 2] >= 0).astype(np.int32)
        )

        # Octant-controlled selection
        selected_mask = np.zeros(len(cand_points), dtype=bool)
        octant_counts = np.zeros(8, dtype=np.int32)
        max_per_oct = self.config.search_max_per_octant
        max_total = self.config.search_max_samples

        count = 0
        for i in range(len(cand_points)):
            if count >= max_total:
                break
            oct = octants[i]
            if octant_counts[oct] < max_per_oct:
                selected_mask[i] = True
                octant_counts[oct] += 1
                count += 1

        sel_points = cand_points[selected_mask]
        sel_values = cand_values[selected_mask]
        sel_dists = dists[selected_mask]
        sel_indices = candidates_idx[selected_mask]
        num_octants = int(np.sum(octant_counts > 0))

        stats = SearchStats(
            num_samples=len(sel_points),
            num_octants=num_octants,
            avg_distance=float(sel_dists.mean()) if len(sel_dists) > 0 else 0.0,
            search_pass=0,  # set by caller
            selected_indices=sel_indices,
        )

        return sel_points, sel_values, stats

    def multi_pass_select(
        self,
        query_point: NDArray[np.float64],
        all_values: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], SearchStats]:
        """
        Multi-pass search (Leapfrog/Datamine convention).

        Pass 1: half variogram range  → Measured candidate
        Pass 2: full variogram range  → Indicated candidate
        Pass 3: double variogram range → Inferred candidate
        Insufficient in all passes    → Unclassified

        Returns the result from the first pass that meets minimum
        sample/octant requirements.  If no pass meets requirements,
        returns the pass with the most samples (best available data).
        """
        base = self.config.base_range
        min_samples = self.config.search_min_samples
        min_octants = self.config.search_min_octants

        best_pts, best_vals, best_stats = None, None, None
        best_n = -1

        for pass_num, factor in enumerate([0.5, 1.0, 2.0], start=1):
            radius = base * factor
            pts, vals, stats = self.select(query_point, all_values, search_radius=radius)

            if stats.num_samples >= min_samples and stats.num_octants >= min_octants:
                stats.search_pass = pass_num
                return pts, vals, stats

            # Track best result in case no pass meets requirements
            if stats.num_samples > best_n:
                best_n = stats.num_samples
                best_pts, best_vals, best_stats = pts, vals, stats

        # All passes failed — return the one with most samples
        best_stats.search_pass = 0  # Unclassified
        return best_pts, best_vals, best_stats
