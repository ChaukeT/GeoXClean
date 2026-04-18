"""
Block model estimation loop with local search neighbourhood.

Iterates over block centroids, applies octant-controlled search,
solves a local RBF system per block, evaluates at discretisation
points, and records full diagnostics.

Multi-pass search for JORC classification:
  Pass 1 (half range)   → Measured candidates
  Pass 2 (full range)   → Indicated candidates
  Pass 3 (double range) → Inferred candidates
  Insufficient          → Unclassified

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .config import RBFConfig
from .fastrbf_engine import FastRBFEngine, FittedRBF
from .search_neighbourhood import SearchNeighbourhood, SearchStats

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10_000  # blocks per memory-release batch
_CACHE_MAX_SIZE = 2048  # max cached FittedRBF objects (LRU eviction)


@dataclass
class EstimationResult:
    """
    Complete result of a block model estimation run.

    Every array has length ``n_blocks``.
    """

    estimated_values: NDArray[np.float64]
    num_samples: NDArray[np.int32]
    num_octants: NDArray[np.int32]
    avg_distance: NDArray[np.float64]
    search_pass: NDArray[np.int32]
    classification: NDArray  # dtype=object, str
    n_blocks_estimated: int
    n_blocks_total: int
    elapsed_seconds: float


class BlockModelEstimator:
    """
    Production block model estimation using local RBF interpolation.

    For each block centroid:
      1. Multi-pass search neighbourhood selection (octant-controlled)
      2. Build local kernel matrix from selected samples
      3. Solve local RBF system
      4. Evaluate at block discretisation points
      5. Average for block grade
      6. Record diagnostics (samples, octants, distance, pass)
    """

    def __init__(self, config: RBFConfig) -> None:
        self.config = config

    def estimate(
        self,
        block_centroids: NDArray[np.float64],
        block_sizes: NDArray[np.float64],
        sample_points: NDArray[np.float64],
        sample_values: NDArray[np.float64],
    ) -> EstimationResult:
        """
        Main estimation loop.

        Parameters
        ----------
        block_centroids : (B, 3) ndarray
        block_sizes : (3,) or (B, 3) ndarray
        sample_points : (N, 3) ndarray
        sample_values : (N,) ndarray

        Returns
        -------
        EstimationResult
        """
        block_centroids = np.asarray(block_centroids, dtype=np.float64)
        block_sizes = np.asarray(block_sizes, dtype=np.float64)
        sample_points = np.asarray(sample_points, dtype=np.float64)
        sample_values = np.asarray(sample_values, dtype=np.float64)

        n_blocks = block_centroids.shape[0]
        n_samples = sample_points.shape[0]

        if block_sizes.ndim == 1:
            block_sizes_full = np.tile(block_sizes, (n_blocks, 1))
        else:
            block_sizes_full = block_sizes

        logger.info(
            "Starting block estimation: %d blocks, %d samples", n_blocks, n_samples
        )
        t0 = time.perf_counter()

        # Build search neighbourhood (KDTree built once)
        search = SearchNeighbourhood(self.config, sample_points)
        engine = FastRBFEngine(self.config)

        # Build discretisation offsets — adaptive: use fewer sub-points
        # when block count is large to keep total query count manageable.
        # 4³=64 sub-points × 100k blocks = 6.4M queries; 2³=8 saves 87%.
        n_disc = self.config.discretisation_points
        if n_blocks > 50_000 and n_disc > 2:
            n_disc = 2
            logger.info(
                "Adaptive discretisation: reduced to %d³=%d sub-points "
                "(block count %d > 50k)", n_disc, n_disc ** 3, n_blocks,
            )
        elif n_blocks > 10_000 and n_disc > 3:
            n_disc = 3
            logger.info(
                "Adaptive discretisation: reduced to %d³=%d sub-points "
                "(block count %d > 10k)", n_disc, n_disc ** 3, n_blocks,
            )
        offsets_1d = np.linspace(-0.5, 0.5, n_disc, endpoint=True)
        gx, gy, gz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
        sub_offsets = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        n_sub = sub_offsets.shape[0]

        # Result arrays
        estimated = np.full(n_blocks, np.nan, dtype=np.float64)
        num_samples_arr = np.zeros(n_blocks, dtype=np.int32)
        num_octants_arr = np.zeros(n_blocks, dtype=np.int32)
        avg_dist_arr = np.zeros(n_blocks, dtype=np.float64)
        search_pass_arr = np.zeros(n_blocks, dtype=np.int32)
        classification = np.empty(n_blocks, dtype=object)
        classification[:] = "Unclassified"

        n_estimated = 0
        log_interval = max(1, n_blocks // 20)

        # LRU cache for fitted RBF models — adjacent blocks often share
        # the same neighbourhood, avoiding redundant O(k³) solves.
        from collections import OrderedDict
        fit_cache: OrderedDict[bytes, FittedRBF] = OrderedDict()
        cache_hits = 0

        for i in range(n_blocks):
            if i > 0 and i % log_interval == 0:
                pct = 100.0 * i / n_blocks
                logger.info(
                    "Estimation progress: %d/%d (%.0f%%) [cache hits: %d]",
                    i, n_blocks, pct, cache_hits,
                )

            centroid = block_centroids[i]
            sizes = block_sizes_full[i]

            # Multi-pass search
            sel_pts, sel_vals, stats = search.multi_pass_select(
                centroid, sample_values
            )

            num_samples_arr[i] = stats.num_samples
            num_octants_arr[i] = stats.num_octants
            avg_dist_arr[i] = stats.avg_distance
            search_pass_arr[i] = stats.search_pass

            if stats.num_samples < self.config.search_min_samples:
                classification[i] = "Unclassified"
                continue

            # Cache key: sorted tuple of sample indices (O(k log k) vs O(k*8) for tobytes)
            # Adjacent blocks sharing the same neighbourhood get exact cache hits
            cache_key = tuple(sorted(stats.selected_indices.tolist()))

            fitted = fit_cache.get(cache_key)
            if fitted is not None:
                # Move to end (most recently used)
                fit_cache.move_to_end(cache_key)
                cache_hits += 1
            else:
                # Solve local RBF system
                try:
                    fitted = engine.fit(sel_pts, sel_vals)
                except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
                    logger.debug("Block %d: local solve failed: %s", i, exc)
                    classification[i] = "Unclassified"
                    continue

                # LRU eviction
                fit_cache[cache_key] = fitted
                if len(fit_cache) > _CACHE_MAX_SIZE:
                    fit_cache.popitem(last=False)

            # Evaluate at discretisation points
            disc_points = centroid[np.newaxis, :] + sub_offsets * sizes[np.newaxis, :]
            disc_values = engine.predict(fitted, disc_points)
            block_grade = disc_values.mean()

            # Clip
            if self.config.clip_min is not None:
                block_grade = max(block_grade, self.config.clip_min)
            if self.config.clip_max is not None:
                block_grade = min(block_grade, self.config.clip_max)

            estimated[i] = block_grade
            n_estimated += 1

            # Classification by search pass
            if stats.search_pass == 1:
                classification[i] = "Measured"
            elif stats.search_pass == 2:
                classification[i] = "Indicated"
            elif stats.search_pass == 3:
                classification[i] = "Inferred"
            else:
                classification[i] = "Unclassified"

        elapsed = time.perf_counter() - t0
        cache_rate = 100.0 * cache_hits / max(n_estimated, 1)
        logger.info(
            "Estimation complete: %d/%d blocks estimated in %.1fs "
            "(cache hits: %d, %.0f%%)",
            n_estimated,
            n_blocks,
            elapsed,
            cache_hits,
            cache_rate,
        )

        return EstimationResult(
            estimated_values=estimated,
            num_samples=num_samples_arr,
            num_octants=num_octants_arr,
            avg_distance=avg_dist_arr,
            search_pass=search_pass_arr,
            classification=classification,
            n_blocks_estimated=n_estimated,
            n_blocks_total=n_blocks,
            elapsed_seconds=elapsed,
        )
