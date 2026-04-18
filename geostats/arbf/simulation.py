"""
ARBF Sequential Conditional Simulation.

Provides a separate simulation workflow so that estimation, estimation
variance, and conditional simulation are no longer conflated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .engine import ARBFEstimator
from .gpr import compute_l_inv, predict_mean_and_variance
from .transforms import normal_score_backtransform, normal_score_transform
from .utils import rotation_matrix, scale_matrix
from .variogram import LocalVariogramResult

logger = logging.getLogger(__name__)


@dataclass
class ARBFSimulationResult:
    """Summary outputs from sequential conditional simulation."""

    realizations: np.ndarray
    mean: np.ndarray
    variance: np.ndarray
    p10: np.ndarray
    p50: np.ndarray
    p90: np.ndarray
    gaussian_realizations: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class ARBFSequentialSimulation(ARBFEstimator):
    """Sequential conditional simulation using ARBF local-neighbourhood GPR."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.n_realizations: int = int(cfg.get("n_realizations", 20))
        self.simulation_seed: int = int(cfg.get("simulation_seed", self.seed))
        self.simulation_use_normal_score: bool = bool(
            cfg.get("simulation_use_normal_score", cfg.get("use_normal_score", True)),
        )
        self.return_gaussian_realizations: bool = bool(
            cfg.get("return_gaussian_realizations", self.simulation_use_normal_score),
        )
        self.simulation_min_variance_fraction: float = float(
            cfg.get("simulation_min_variance_fraction", 1e-6),
        )
        if self.n_realizations <= 0:
            raise ValueError("n_realizations must be positive")

    def simulate(self) -> ARBFSimulationResult:
        """Generate sequential conditional simulations on the block centroids."""
        self._validate_inputs()
        self._domain_diagnostics = self._summarize_domain_state()
        self._local_drift_fallbacks = 0
        self._stabilization_events = []

        self._progress(0, "Starting ARBF sequential simulation")

        hard_coords = np.asarray(self._composite_coords, dtype=np.float64)
        hard_values_raw = np.asarray(self._composite_values, dtype=np.float64).ravel()
        block_centroids = np.asarray(self._block_centroids, dtype=np.float64)
        B = len(block_centroids)

        ns_table = None
        working_values = hard_values_raw.copy()
        if self.simulation_use_normal_score:
            self._progress(5, "Normal-score transform for simulation")
            working_values, ns_table = normal_score_transform(
                hard_values_raw,
                seed=self.seed,
                coords=hard_coords,
            )

        self._working_values = working_values.copy()
        self._effective_alpha = self.alpha
        if self.simulation_use_normal_score:
            ns_var = float(np.var(working_values))
            if self.sill > 0:
                total = self.sill + self.nugget
                nugget_ratio = self.nugget / max(total, 1e-12)
                self._effective_sill = ns_var * (1.0 - nugget_ratio)
                self._effective_nugget = ns_var * nugget_ratio
            else:
                self._effective_sill = ns_var
                self._effective_nugget = self.nugget
        else:
            self._effective_sill = self.sill if self.sill > 0 else float(np.var(working_values))
            self._effective_nugget = self.nugget

        self._effective_drift_type = self._resolve_effective_drift_type(
            hard_coords,
            working_values,
        )

        if self.use_lva and self._orientation_field is None:
            self._progress(10, "Building orientation field")
            self._orientation_field = self._build_orientation_field(
                hard_coords,
                working_values,
            )

        domain_labels, hard_domain_labels = self._resolve_simulation_domains()
        transformed_hard = self._transform_for_search(hard_coords)
        transformed_blocks = self._transform_for_search(block_centroids)
        hard_exact_mask, hard_exact_values = self._resolve_hard_matches(
            hard_coords,
            working_values,
            block_centroids,
        )

        global_vp = self._get_global_variogram_params()
        vp_local = LocalVariogramResult(
            sill=global_vp["sill"],
            nugget=global_vp["nugget"],
            range_=global_vp["range_"],
            alpha=global_vp["alpha"],
            kernel_type=global_vp["kernel_type"],
            fit_residual=0.0,
            n_pairs=0,
            n_lags=0,
        )
        R_global = rotation_matrix(self.azimuth, self.dip, self.pitch)
        ratio_mid = self.range_mid / max(self.range_max, 1e-12)
        ratio_min = self.range_min / max(self.range_max, 1e-12)
        S_global = scale_matrix(
            vp_local.range_,
            vp_local.range_ * ratio_mid,
            vp_local.range_ * ratio_min,
        )
        unconditional_variance = max(float(vp_local.sill + vp_local.nugget), 1e-12)
        min_variance = max(
            unconditional_variance * self.simulation_min_variance_fraction,
            1e-12,
        )

        hard_data_by_domain = {}
        for dom in np.unique(domain_labels):
            mask = hard_domain_labels == dom
            hard_data_by_domain[dom] = (
                hard_coords[mask],
                transformed_hard[mask],
                working_values[mask],
            )

        gaussian_realizations = np.empty((self.n_realizations, B), dtype=np.float64)
        conditioning_mean = np.zeros(self.n_realizations, dtype=np.float64)
        conditioning_max = np.zeros(self.n_realizations, dtype=np.int32)
        unconditional_counts = np.zeros(self.n_realizations, dtype=np.int32)

        self._progress(15, f"Running {self.n_realizations} realizations")
        for ireal in range(self.n_realizations):
            rng = np.random.default_rng(self.simulation_seed + ireal)
            path = rng.permutation(B)
            sim_values = np.full(B, np.nan, dtype=np.float64)
            simulated = np.zeros(B, dtype=bool)
            conditioning_total = 0
            conditioning_peak = 0
            unconditional_used = 0

            for b in path:
                if hard_exact_mask[b]:
                    sim_values[b] = hard_exact_values[b]
                    simulated[b] = True
                    continue

                dom = domain_labels[b]
                hard_dom_coords, hard_dom_trans, hard_dom_values = hard_data_by_domain[dom]
                soft_mask = simulated & (domain_labels == dom) & (~hard_exact_mask)
                if np.any(soft_mask):
                    soft_coords = block_centroids[soft_mask]
                    soft_trans = transformed_blocks[soft_mask]
                    soft_values = sim_values[soft_mask]
                    conditioning_coords = np.vstack([hard_dom_coords, soft_coords])
                    conditioning_trans = np.vstack([hard_dom_trans, soft_trans])
                    conditioning_values = np.concatenate([hard_dom_values, soft_values])
                else:
                    conditioning_coords = hard_dom_coords
                    conditioning_trans = hard_dom_trans
                    conditioning_values = hard_dom_values

                selected, _ = self._select_simulation_conditioning(
                    conditioning_trans,
                    transformed_blocks[b],
                )
                if len(selected) == 0:
                    mean = self._unconditional_mean(
                        block_centroids[b],
                        hard_dom_coords,
                        hard_dom_values,
                    )
                    variance = unconditional_variance
                    unconditional_used += 1
                else:
                    local_coords = conditioning_coords[selected]
                    local_values = conditioning_values[selected]
                    local_drift = self._local_neighbourhood_drift_type(local_coords)
                    try:
                        fact, weights, poly_coeffs, _, _ = self._factorise_kernel_system(
                            local_coords,
                            local_values,
                            vp_local,
                            R_global,
                            S_global,
                            context=f"ARBF SGS node n={len(local_values)}",
                            drift_type=local_drift,
                        )
                        l_inv = compute_l_inv(fact)
                        mean_arr, var_arr = predict_mean_and_variance(
                            block_centroids[b:b + 1],
                            local_coords,
                            weights,
                            poly_coeffs,
                            fact,
                            kernel_type=vp_local.kernel_type,
                            alpha=vp_local.alpha,
                            sill=vp_local.sill,
                            range_=vp_local.range_,
                            nugget=vp_local.nugget,
                            R=R_global,
                            S=S_global,
                            drift_type=local_drift,
                            orientation_field=self._orientation_field if self.use_lva else None,
                            l_inv=l_inv,
                        )
                        mean = float(mean_arr[0])
                        variance = max(float(var_arr[0]), min_variance)
                    except np.linalg.LinAlgError:
                        logger.warning(
                            "Simulation node failed to factorize with %d conditioning points; "
                            "falling back to unconditional draw.",
                            len(local_values),
                        )
                        mean = self._unconditional_mean(
                            block_centroids[b],
                            hard_dom_coords,
                            hard_dom_values,
                        )
                        variance = unconditional_variance
                        unconditional_used += 1

                sim_values[b] = float(rng.normal(mean, np.sqrt(max(variance, min_variance))))
                simulated[b] = True
                conditioning_total += int(len(selected))
                conditioning_peak = max(conditioning_peak, int(len(selected)))

            gaussian_realizations[ireal] = sim_values
            conditioning_mean[ireal] = conditioning_total / max(B, 1)
            conditioning_max[ireal] = conditioning_peak
            unconditional_counts[ireal] = unconditional_used

            pct = 15 + int(75 * (ireal + 1) / self.n_realizations)
            self._progress(pct, f"Completed realization {ireal + 1}/{self.n_realizations}")

        if ns_table is not None:
            self._progress(92, "Back-transforming realizations")
            realizations = np.empty_like(gaussian_realizations)
            for ireal in range(self.n_realizations):
                realizations[ireal] = normal_score_backtransform(
                    gaussian_realizations[ireal],
                    ns_table,
                )
        else:
            realizations = gaussian_realizations.copy()

        self._progress(98, "Computing simulation summaries")
        mean = np.mean(realizations, axis=0)
        variance = np.var(realizations, axis=0)
        p10 = np.percentile(realizations, 10, axis=0)
        p50 = np.percentile(realizations, 50, axis=0)
        p90 = np.percentile(realizations, 90, axis=0)

        diagnostics = {
            "n_realizations": int(self.n_realizations),
            "seed": int(self.simulation_seed),
            "simulation_space": (
                "normal_score" if ns_table is not None else "raw"
            ),
            "effective_drift_type": self._active_drift_type(),
            "domains_enforced": bool(self._domain_diagnostics.get("domains_enforced", False)),
            "n_geological_domains": int(self._domain_diagnostics.get("n_geological_domains", 0)),
            "hard_data_honoured_nodes": int(np.sum(hard_exact_mask)),
            "mean_conditioning_samples": float(np.mean(conditioning_mean)),
            "max_conditioning_samples": int(np.max(conditioning_max)),
            "mean_unconditional_draws": float(np.mean(unconditional_counts)),
            "local_drift_fallbacks": int(self._local_drift_fallbacks),
            "n_stabilized_systems": int(len(self._stabilization_events)),
        }

        self._progress(100, "ARBF sequential simulation complete")
        return ARBFSimulationResult(
            realizations=realizations,
            mean=mean,
            variance=variance,
            p10=p10,
            p50=p50,
            p90=p90,
            gaussian_realizations=(
                gaussian_realizations if self.return_gaussian_realizations else None
            ),
            diagnostics=diagnostics,
        )

    def _resolve_simulation_domains(self) -> Tuple[np.ndarray, np.ndarray]:
        """Resolve composite and block domain labels for simulation."""
        if self._composite_domains is None:
            domain_labels = np.zeros(len(self._block_centroids), dtype=np.int32)
            hard_domain_labels = np.zeros(len(self._composite_coords), dtype=np.int32)
            return domain_labels, hard_domain_labels

        hard_domain_labels = np.asarray(self._composite_domains).ravel()
        if self._block_domains is not None:
            domain_labels = np.asarray(self._block_domains).ravel()
            return domain_labels, hard_domain_labels

        tree = cKDTree(np.asarray(self._composite_coords, dtype=np.float64))
        _, nn_idx = tree.query(np.asarray(self._block_centroids, dtype=np.float64), k=1)
        domain_labels = hard_domain_labels[nn_idx]
        return domain_labels, hard_domain_labels

    def _resolve_hard_matches(
        self,
        hard_coords: np.ndarray,
        hard_values: np.ndarray,
        block_centroids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify block nodes that coincide with hard data locations."""
        bs = np.asarray(self._block_sizes, dtype=np.float64)
        if bs.ndim == 2 and len(bs) > 0:
            bs = bs[0]
        tol = max(float(np.min(bs)), 1.0) * 1e-6 if bs.size > 0 else 1e-6

        tree = cKDTree(hard_coords)
        dists, nn_idx = tree.query(block_centroids, k=1)
        matched = dists <= tol
        values = np.full(len(block_centroids), np.nan, dtype=np.float64)
        values[matched] = hard_values[nn_idx[matched]]
        return matched, values

    def _select_simulation_conditioning(
        self,
        conditioning_coords: np.ndarray,
        query_point: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """Select conditioning samples for one simulation node."""
        if len(conditioning_coords) == 0:
            return np.empty(0, dtype=np.intp), len(self.local_search_radii)

        selected = np.empty(0, dtype=np.intp)
        chosen_pass = len(self.local_search_radii)
        dists = np.linalg.norm(conditioning_coords - query_point, axis=1)

        for pass_idx, radius in enumerate(self.local_search_radii, start=1):
            idx = np.where(dists <= radius)[0]
            if len(idx) == 0:
                continue
            order = np.argsort(dists[idx])
            if self.max_samples > 0:
                order = order[:self.max_samples]
            selected = idx[order]
            chosen_pass = pass_idx
            return selected.astype(np.intp), chosen_pass

        return selected.astype(np.intp), chosen_pass

    def _unconditional_mean(
        self,
        query_point: np.ndarray,
        conditioning_coords: np.ndarray,
        conditioning_values: np.ndarray,
    ) -> float:
        """Approximate unconditional mean under the configured drift model."""
        if (
            self._active_drift_type() == "linear"
            and len(conditioning_values) >= 4
        ):
            X = np.column_stack([
                np.ones(len(conditioning_coords), dtype=np.float64),
                np.asarray(conditioning_coords, dtype=np.float64),
            ])
            try:
                beta, *_ = np.linalg.lstsq(X, conditioning_values, rcond=None)
                return float(np.dot(np.array([1.0, *query_point], dtype=np.float64), beta))
            except np.linalg.LinAlgError:
                pass
        return float(np.mean(conditioning_values)) if len(conditioning_values) > 0 else 0.0
