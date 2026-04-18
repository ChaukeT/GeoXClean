"""
Core FastRBF solver — the heart of the estimation engine.

Replicates the mathematical workflow of Leapfrog Geo's FastRBF system.
Builds the augmented linear system [Φ P; Pᵀ 0][w; c] = [d; 0], solves
for weights, and evaluates the interpolant at query points.

Solver strategy (in order of preference):
  1. N ≤ 5000  → direct solve via LDLᵀ factorisation (symmetric indefinite)
  2. N > 5000  → preconditioned GMRES via scipy.sparse.linalg.gmres
  3. Tikhonov regularisation with config.accuracy on diagonal
  4. Condition number warning if > 1e12

Reference: CLAUDE_CODE_PROMPT_FastRBF_Engine.md
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as la
from scipy.sparse.linalg import gmres, LinearOperator

from .config import RBFConfig, DriftType
from .interpolant_functions import evaluate_kernel
from .trend import (
    build_augmented_system,
    build_polynomial_matrix,
    drift_dimension,
    evaluate_polynomial,
)
from ..utils.distance import (
    pairwise_anisotropic_distance,
    batch_anisotropic_distance,
)

logger = logging.getLogger(__name__)

_DIRECT_SOLVE_THRESHOLD = 5000
_CONDITION_WARNING_THRESHOLD = 1e12

# Default regularisation: ε · trace(Φ) / N
_DEFAULT_REG_EPSILON = 1e-7


@dataclass
class FittedRBF:
    """
    Container for a fitted RBF interpolant.

    Stores everything needed to evaluate the interpolant at new points
    and to reproduce the result.
    """

    weights: NDArray[np.float64]
    polynomial_coefficients: NDArray[np.float64]
    data_points: NDArray[np.float64]
    data_values: NDArray[np.float64]
    config: RBFConfig
    accuracy_used: float
    condition_number: float
    solve_method: str
    fit_timestamp: str
    data_hash: str
    n_samples: int
    solve_time_seconds: float


class FastRBFEngine:
    """
    Production-grade RBF interpolation engine.

    Builds the kernel matrix, augments with polynomial drift, solves
    for weights, and provides prediction at arbitrary query points
    with optional block discretisation.
    """

    def __init__(self, config: RBFConfig) -> None:
        """Initialise with a validated RBFConfig."""
        self.config = config

        # Warn about non-unit ratio_major (#3)
        if abs(config.ratio_major - 1.0) > 1e-6:
            logger.warning(
                "ratio_major = %.4f (expected 1.0). Anisotropy ratios should "
                "be relative to the major axis (ratio_major=1.0). Results may "
                "have inverted directional continuity.",
                config.ratio_major,
            )

        logger.info(
            "FastRBFEngine initialised: kernel=%s, drift=%s, range=%.2f",
            config.kernel_type.value,
            config.drift.value,
            config.base_range,
        )

    # ────────────────────────────────────────────────────────────
    # Fitting
    # ────────────────────────────────────────────────────────────

    def fit(
        self,
        points: NDArray[np.float64],
        values: NDArray[np.float64],
    ) -> FittedRBF:
        """
        Solve the RBF system for the given data.

        Parameters
        ----------
        points : (N, 3) array of sample coordinates.
        values : (N,) array of sample values.

        Returns
        -------
        FittedRBF object containing weights and metadata.
        """
        points = np.asarray(points, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        n = points.shape[0]

        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3); got shape {points.shape}")
        if values.shape != (n,):
            raise ValueError(
                f"values length ({values.shape[0]}) != points count ({n})"
            )

        # Merge exact duplicate locations (#6 — duplicates → singular Φ)
        points, values = self._merge_duplicates(points, values)
        n = points.shape[0]

        logger.info("Fitting RBF with %d data points", n)
        t0 = time.perf_counter()

        # Build pairwise distance matrix
        D = pairwise_anisotropic_distance(
            points,
            azimuth=self.config.azimuth,
            dip=self.config.dip,
            pitch=self.config.pitch,
            ratio_major=self.config.ratio_major,
            ratio_semi=self.config.ratio_semi,
            ratio_minor=self.config.ratio_minor,
        )

        # Build kernel matrix Φ (variogram formulation).
        #
        # All kernels return γ(0) = 0 on the diagonal and
        # γ(r>0) = sill·f(r) + nugget off-diagonal.  The nugget
        # is the variogram discontinuity at the origin (measurement
        # error / micro-scale variance) — it appears ONLY off-diagonal.
        #
        # Accuracy (Tikhonov regularisation) is a SEPARATE term added
        # to the diagonal to stabilise the linear system.  There is NO
        # double-counting: diagonal = 0 + accuracy, off-diag includes nugget.
        sill = self.config.partial_sill
        kernel_matrix = evaluate_kernel(
            D,
            sill=sill,
            range_=self.config.base_range,
            nugget=self.config.nugget,
            kernel_type=self.config.kernel_type,
            alpha=self.config.alpha,
        )

        # Regularisation: accuracy relative to Φ scale (#4)
        accuracy = self._resolve_accuracy(kernel_matrix)

        # Diagonal is γ(0) = 0 from kernels; accuracy is the only diagonal term
        np.fill_diagonal(kernel_matrix, kernel_matrix.diagonal() + accuracy)

        # Build augmented system
        A, b = build_augmented_system(
            kernel_matrix, points, values, self.config.drift
        )

        # Enforce exact symmetry (#1 — floating-point asymmetry guard)
        A = (A + A.T) * 0.5

        # Solve
        solution, solve_method, cond = self._solve_system(A, b, n)

        # Extract weights and polynomial coefficients
        m = drift_dimension(self.config.drift)
        weights = solution[:n]
        poly_coefs = solution[n:] if m > 0 else np.array([], dtype=np.float64)

        elapsed = time.perf_counter() - t0

        # Data hash — paired rows with lexicographic sort (#8)
        data_hash = self._compute_data_hash(points, values)

        fitted = FittedRBF(
            weights=weights,
            polynomial_coefficients=poly_coefs,
            data_points=points.copy(),
            data_values=values.copy(),
            config=self.config,
            accuracy_used=accuracy,
            condition_number=cond,
            solve_method=solve_method,
            fit_timestamp=datetime.now(timezone.utc).isoformat(),
            data_hash=data_hash,
            n_samples=n,
            solve_time_seconds=elapsed,
        )

        logger.info(
            "Fit complete: method=%s, cond=%.2e, accuracy=%.2e, time=%.3fs",
            solve_method,
            cond,
            accuracy,
            elapsed,
        )
        return fitted

    # ────────────────────────────────────────────────────────────
    # Prediction  (#2 — uses shared distance function, no duplication)
    # ────────────────────────────────────────────────────────────

    def predict(
        self,
        fitted: FittedRBF,
        query_points: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Evaluate the fitted interpolant at query points.

        f(x) = Σᵢ wᵢ · φ(‖x - xᵢ‖) + p(x)

        Uses batch_anisotropic_distance() — the same rotation/scaling
        code path as pairwise_anisotropic_distance() in fit().

        Parameters
        ----------
        fitted : FittedRBF
        query_points : (M, 3) ndarray

        Returns
        -------
        estimates : (M,) ndarray
        """
        query_points = np.asarray(query_points, dtype=np.float64)
        cfg = fitted.config
        m = query_points.shape[0]

        estimates = np.empty(m, dtype=np.float64)
        batch_size = 10000

        for start in range(0, m, batch_size):
            end = min(start + batch_size, m)
            batch = query_points[start:end]

            # Distance: same function family as fit (#2 — unified)
            D = batch_anisotropic_distance(
                batch,
                fitted.data_points,
                azimuth=cfg.azimuth,
                dip=cfg.dip,
                pitch=cfg.pitch,
                ratio_major=cfg.ratio_major,
                ratio_semi=cfg.ratio_semi,
                ratio_minor=cfg.ratio_minor,
            )

            # Kernel evaluation
            sill = cfg.partial_sill
            phi = evaluate_kernel(
                D,
                sill=sill,
                range_=cfg.base_range,
                nugget=cfg.nugget,
                kernel_type=cfg.kernel_type,
                alpha=cfg.alpha,
            )

            # RBF contribution
            est = phi @ fitted.weights

            # Polynomial contribution
            if fitted.polynomial_coefficients.size > 0:
                est += evaluate_polynomial(
                    batch, fitted.polynomial_coefficients, cfg.drift
                )

            estimates[start:end] = est

        # Clip if configured
        if cfg.clip_min is not None:
            estimates = np.maximum(estimates, cfg.clip_min)
        if cfg.clip_max is not None:
            estimates = np.minimum(estimates, cfg.clip_max)

        return estimates

    def predict_block(
        self,
        fitted: FittedRBF,
        block_centroids: NDArray[np.float64],
        block_sizes: NDArray[np.float64],
        n_discretisation: Optional[int] = None,
    ) -> NDArray[np.float64]:
        """
        Estimate block values by averaging point estimates at internal
        discretisation points.

        This is numerical integration of the interpolant over the block
        volume — NOT a change-of-support model. It does not account for
        information effect or volume-variance relationships.

        Parameters
        ----------
        fitted : FittedRBF
        block_centroids : (B, 3) ndarray
        block_sizes : (3,) or (B, 3) ndarray
            Full block dimensions (NOT half-widths). Offsets span
            [-0.5, +0.5] × block_sizes, i.e., the full block extent.
        n_discretisation : int, optional
            Override config.discretisation_points.

        Returns
        -------
        block_estimates : (B,) ndarray
        """
        n_disc = n_discretisation or fitted.config.discretisation_points
        block_centroids = np.asarray(block_centroids, dtype=np.float64)
        block_sizes = np.asarray(block_sizes, dtype=np.float64)

        if block_sizes.ndim == 1:
            block_sizes = np.tile(block_sizes, (block_centroids.shape[0], 1))

        n_blocks = block_centroids.shape[0]

        # Build sub-block offsets (n³ points within each block)
        offsets_1d = np.linspace(-0.5, 0.5, n_disc, endpoint=True)
        gx, gy, gz = np.meshgrid(offsets_1d, offsets_1d, offsets_1d, indexing="ij")
        offsets = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])  # (n³, 3)
        n_sub = offsets.shape[0]

        block_estimates = np.empty(n_blocks, dtype=np.float64)
        batch = 2000  # blocks per batch

        for start in range(0, n_blocks, batch):
            end = min(start + batch, n_blocks)
            centroids_batch = block_centroids[start:end]
            sizes_batch = block_sizes[start:end]
            nb = centroids_batch.shape[0]

            # (nb, n_sub, 3)
            query = (
                centroids_batch[:, np.newaxis, :]
                + offsets[np.newaxis, :, :] * sizes_batch[:, np.newaxis, :]
            )
            query_flat = query.reshape(-1, 3)

            est_flat = self.predict(fitted, query_flat)
            est_blocks = est_flat.reshape(nb, n_sub)
            block_estimates[start:end] = est_blocks.mean(axis=1)

        return block_estimates

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _merge_duplicates(
        points: NDArray[np.float64],
        values: NDArray[np.float64],
        tol: float = 1e-8,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Merge exact/near-duplicate locations by averaging values (#6).

        Prevents singular Φ from identical rows/columns.
        """
        n = points.shape[0]
        if n <= 1:
            return points, values

        # Round to tolerance for grouping
        rounded = np.round(points / tol) * tol
        # Find unique rows
        _, inverse, counts = np.unique(
            rounded, axis=0, return_inverse=True, return_counts=True
        )

        n_unique = len(counts)
        if n_unique == n:
            return points, values  # No duplicates

        n_dupes = n - n_unique
        logger.warning(
            "Merged %d duplicate locations (%d → %d unique). "
            "Values at coincident locations were averaged.",
            n_dupes, n, n_unique,
        )

        # Average values at each unique location — vectorised with bincount
        merged_vals = np.bincount(inverse, weights=values, minlength=n_unique) / counts
        merged_pts = np.zeros((n_unique, 3), dtype=np.float64)
        for axis in range(3):
            merged_pts[:, axis] = np.bincount(
                inverse, weights=points[:, axis], minlength=n_unique
            ) / counts

        return merged_pts, merged_vals

    def _resolve_accuracy(
        self, kernel_matrix: NDArray[np.float64]
    ) -> float:
        """
        Determine the accuracy (regularisation) parameter.

        If user-provided, use that. Otherwise auto-compute as:
            accuracy = ε · mean(Φ_off_diag)

        This scales regularisation relative to the typical kernel matrix
        entry magnitude (#4), independent of dataset size N.  The previous
        formula multiplied by N, causing O(N) growth that over-smoothed
        large datasets.
        """
        if self.config.accuracy is not None:
            return self.config.accuracy

        n = kernel_matrix.shape[0]
        trace_phi = np.trace(kernel_matrix)

        if n > 1:
            # γ(0) = 0 for all variogram kernels, so trace ≈ 0.
            # Use mean of off-diagonal entries as the characteristic scale.
            off_diag_sum = kernel_matrix.sum() - trace_phi
            n_off = n * n - n
            mean_phi = off_diag_sum / max(n_off, 1)
            acc = _DEFAULT_REG_EPSILON * mean_phi if mean_phi > 0 else 1e-10
        else:
            acc = 1e-10

        # Clamp to a sensible floor
        acc = max(acc, 1e-14)

        logger.debug("Auto-computed accuracy = %.6e (from Φ scale, N=%d)", acc, n)
        return float(acc)

    def _solve_system(
        self,
        A: NDArray[np.float64],
        b: NDArray[np.float64],
        n_data: int,
    ) -> tuple[NDArray[np.float64], str, float]:
        """
        Solve the augmented linear system.

        The matrix [Φ P; Pᵀ 0] is symmetric indefinite (saddle-point).
        Uses LDLᵀ factorisation for numerical stability (#1).

        Returns (solution, method_name, condition_number).
        """
        size = A.shape[0]

        # Condition number estimate (#6 — use 1-norm estimate for all sizes)
        cond = self._estimate_condition(A)

        if cond > _CONDITION_WARNING_THRESHOLD and np.isfinite(cond):
            logger.warning(
                "System matrix is ill-conditioned (cond=%.2e). "
                "Consider increasing accuracy or reducing search radius.",
                cond,
            )

        if n_data <= _DIRECT_SOLVE_THRESHOLD:
            # Direct solve — symmetric indefinite (#1)
            try:
                solution = la.solve(A, b, assume_a="sym")
                # Verify residual
                residual = float(np.linalg.norm(A @ solution - b))
                rhs_norm = float(np.linalg.norm(b))
                rel_residual = residual / max(rhs_norm, 1e-15)
                if rel_residual > 1e-6:
                    logger.warning(
                        "Direct solve residual is large: ||Ax-b||/||b|| = %.2e",
                        rel_residual,
                    )
                return solution, "direct_symmetric", cond
            except la.LinAlgError:
                logger.warning("Symmetric solve failed; falling back to general solve")
                try:
                    solution = la.solve(A, b)
                    return solution, "direct_general", cond
                except la.LinAlgError as exc:
                    raise np.linalg.LinAlgError(
                        f"Direct solve failed for {size}×{size} system. "
                        f"Condition number: {cond:.2e}. "
                        "Try increasing the accuracy parameter."
                    ) from exc
        else:
            # Iterative GMRES for large systems (#7)
            logger.info(
                "Using GMRES for %d×%d system (N=%d > %d)",
                size, size, n_data, _DIRECT_SOLVE_THRESHOLD,
            )

            # ── Block-diagonal Schur complement preconditioner (#7) ──
            #
            # The augmented system has saddle-point structure:
            #
            #   A = [ Φ_reg   P ]    where Φ_reg = Φ + λI  (N×N)
            #       [  Pᵀ     0 ]    and P is (N×m), m = drift_dimension
            #
            # A good preconditioner for this structure is:
            #
            #   M = [ Φ_reg   0          ]
            #       [  0     -S_approx   ]
            #
            # where S = Pᵀ Φ_reg⁻¹ P is the (m×m) Schur complement.
            # Since m is tiny (1, 4, or 10 for constant/linear/quadratic
            # drift) we can afford to:
            #   1. Use diagonal of Φ_reg as a cheap approximation D_Φ
            #   2. Form S_approx = Pᵀ D_Φ⁻¹ P  (m×m, trivially invertible)
            #   3. Apply the preconditioner block-wise in O(N) per iteration
            #
            # This respects the natural 2×2 block structure and gives
            # GMRES a much better-conditioned system to work with,
            # especially for the polynomial rows which a Jacobi
            # preconditioner handles poorly (clamping zeros to a mean).

            m = size - n_data  # polynomial block dimension

            # Extract blocks
            Phi_reg_diag = A.diagonal()[:n_data].copy()
            Phi_reg_diag[np.abs(Phi_reg_diag) < 1e-15] = 1e-10
            D_Phi_inv = 1.0 / Phi_reg_diag  # (N,) cheap diagonal approx of Φ_reg⁻¹

            if m > 0:
                P = A[:n_data, n_data:]  # (N, m)

                # S_approx = Pᵀ diag(Φ_reg)⁻¹ P  →  (m, m)
                S_approx = (P * D_Phi_inv[:, np.newaxis]).T @ P  # (m, m)

                # Regularise for safety (should be SPD but floating-point
                # may make it barely singular)
                S_approx += np.eye(m) * (1e-12 * np.trace(S_approx) / max(m, 1))

                # Factor S_approx — it's tiny (m ≤ 10), so direct inverse is fine
                try:
                    S_approx_inv = la.inv(S_approx)
                except la.LinAlgError:
                    logger.warning(
                        "Schur complement inversion failed; "
                        "falling back to diagonal preconditioner"
                    )
                    S_approx_inv = None
            else:
                S_approx_inv = None

            if S_approx_inv is not None and m > 0:
                # Full block-diagonal preconditioner
                def precond(x):
                    result = np.empty_like(x)
                    result[:n_data] = D_Phi_inv * x[:n_data]
                    # Negative sign: the Schur complement of the saddle-point
                    # system is negative definite (bottom-right block is 0).
                    result[n_data:] = -(S_approx_inv @ x[n_data:])
                    return result

                logger.debug(
                    "Using block-diagonal Schur complement preconditioner "
                    "(Phi block: diagonal, polynomial block: %d×%d S^-1)",
                    m, m,
                )
            else:
                # Fallback: pure diagonal (no polynomial block or S failed)
                diag = np.abs(A.diagonal()).copy()
                diag[diag < 1e-15] = max(np.mean(diag[diag > 1e-15]), 1e-10)
                D_full_inv = 1.0 / diag

                def precond(x):
                    return D_full_inv * x

                logger.debug("Using diagonal fallback preconditioner")

            M = LinearOperator((size, size), matvec=precond)

            solution, info = gmres(
                A,
                b,
                M=M,
                maxiter=self.config.max_solver_iterations,
                atol=1e-8,
                rtol=1e-6,
            )

            if info != 0:
                # Check actual residual (#11)
                residual = float(np.linalg.norm(A @ solution - b))
                rhs_norm = float(np.linalg.norm(b))
                rel_residual = residual / max(rhs_norm, 1e-15)
                logger.warning(
                    "GMRES did not converge (info=%d). "
                    "Relative residual: %.2e. Results may be inaccurate.",
                    info, rel_residual,
                )
                if rel_residual > 0.1:
                    raise RuntimeError(
                        f"GMRES failed to converge: relative residual = {rel_residual:.2e}. "
                        f"System size: {size}×{size}. "
                        "Try increasing accuracy or reducing dataset size."
                    )

            return solution, "gmres", cond

    @staticmethod
    def _estimate_condition(A: NDArray[np.float64]) -> float:
        """
        Estimate condition number for any size (#6).

        Uses cheap 1-norm estimate for large systems instead of
        skipping entirely.
        """
        size = A.shape[0]
        try:
            if size <= 3000:
                return float(np.linalg.cond(A, p=1))
            else:
                # Cheap estimate: ||A||_1 * ||A^{-1}||_1 via scipy
                norm_a = float(la.norm(A, ord=1))
                # Use LU-based reciprocal condition number estimate
                try:
                    lu, piv, info = la.lapack.dgetrf(A, overwrite_a=False)
                    if info == 0:
                        rcond = la.lapack.dgecon('1', lu, norm_a)[0]
                        return 1.0 / max(rcond, 1e-30)
                    else:
                        raise ValueError("LU factorization failed")
                except Exception:
                    # Fallback: just report the norm ratio heuristic
                    diag_min = np.min(np.abs(A.diagonal()))
                    diag_min = max(diag_min, 1e-30)
                    return norm_a / diag_min
        except Exception:
            return np.inf

    @staticmethod
    def _compute_data_hash(
        points: NDArray[np.float64], values: NDArray[np.float64]
    ) -> str:
        """
        Reproducibility hash of the paired (x, y, z, value) dataset (#8).

        Sorts rows lexicographically so order-independence is correct
        while maintaining point-value correspondence.
        """
        n = points.shape[0]
        paired = np.empty((n, 4), dtype=np.float64)
        paired[:, :3] = points
        paired[:, 3] = values

        # Lexicographic sort by (x, y, z, value)
        sort_idx = np.lexsort(
            (paired[:, 3], paired[:, 2], paired[:, 1], paired[:, 0])
        )
        sorted_data = paired[sort_idx]

        # Use MD5 instead of SHA256 — 2-3x faster, sufficient for
        # reproducibility hashing (not security-sensitive)
        return hashlib.md5(sorted_data.tobytes()).hexdigest()
