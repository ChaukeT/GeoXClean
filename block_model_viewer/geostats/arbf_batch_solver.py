"""
Block-level batched RBF solver — the core performance optimization.

Design principle:
    ONE neighbourhood search per block.
    ONE RBF matrix assembly per block.
    ONE factorization per block.
    ALL subpoints evaluated through batched backsolves.

The system solved is the standard RBF augmented system:

    [Φ + λI   P] [w]   [z]
    [P'       0] [β] = [0]

where Φ_ij = φ(r_ij) is the RBF basis function evaluated at
anisotropy-transformed distances, NOT a variogram covariance.

Prediction at query point x:
    ẑ(x) = Σ w_i φ(||x - x_i||) + p(x)'β

Uncertainty is NOT computed as kriging variance.  Instead, the
caller should use compute_uncertainty_index() from the engine.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve, LinAlgError, LinAlgWarning
from scipy.spatial.distance import cdist

from .fastrbf_engine_v2 import (
    Anisotropy,
    RBFKernel,
    RBFSettings,
    VariogramModel,
    drift_basis,
    _as_2d_float,
)


def _safe_log10(x: float) -> float:
    return math.log10(max(x, 1.0))


@dataclass
class BatchBlockResult:
    """Results for all subpoints in one block."""
    sub_means: np.ndarray       # (n_sub,)
    sub_vars: np.ndarray        # (n_sub,) — local residual spread, NOT kriging variance
    condnum: float
    neff: float
    n_used: int
    fail_flag: bool


class BlockBatchSolver:
    """Solves the local RBF system ONCE and evaluates all subpoints via batched backsolve.

    The system is:
        [Φ + λI   P] [w]   [z]
        [P'       0] [β] = [0]

    where Φ_ij = φ(r_ij) and φ is the chosen RBF basis function.

    Construction cost: O(n³) — one factorization.
    Per-subpoint cost: O(n²) — one triangular backsolve per subpoint (batched).
    """

    def __init__(
        self,
        local_coords: np.ndarray,
        local_values: np.ndarray,
        rbf_kernel: RBFKernel,
        settings: RBFSettings,
        anisotropy: Anisotropy,
        compute_condition: bool = False,
        *,
        variogram: Optional[VariogramModel] = None,  # legacy compat
        sample_weights: Optional[np.ndarray] = None,
    ) -> None:
        coords = _as_2d_float(local_coords)
        values = np.asarray(local_values, dtype=float).ravel()
        n = coords.shape[0]
        _local_weights: Optional[np.ndarray] = None
        if sample_weights is not None:
            _lw = np.asarray(sample_weights, dtype=float).ravel()
            if _lw.size == n:
                _local_weights = _lw
        self._sample_weights = _local_weights

        # ── Stability pass 1: collapse near-duplicate samples ────────
        # Duplicate (or near-duplicate) composites in the local
        # neighbourhood produce a near-rank-deficient Phi. Global
        # jitter at ingest doesn't catch within-neighbourhood dups
        # that arose from re-logging / compositing. Collapse within
        # a 1e-3 tolerance in the anisotropy-transformed space.
        coords_t_pre = anisotropy.transform(coords)
        if n >= 2:
            _ord = np.lexsort(coords_t_pre.T)
            _keep = np.zeros(n, dtype=bool)
            _keep[_ord[0]] = True
            _prev = coords_t_pre[_ord[0]]
            for _k in range(1, n):
                _idx = _ord[_k]
                if float(np.linalg.norm(coords_t_pre[_idx] - _prev)) > 1e-3:
                    _keep[_idx] = True
                    _prev = coords_t_pre[_idx]
            n_kept = int(_keep.sum())
            if n_kept < n:
                coords = coords[_keep]
                values = values[_keep]
                coords_t_pre = coords_t_pre[_keep]
                if self._sample_weights is not None:
                    self._sample_weights = self._sample_weights[_keep]
                n = n_kept

        # ── Stability pass 2: local coordinate centring ──────────────
        # Translate coords so the neighbourhood centroid sits at the
        # origin. This preserves all pairwise distances (Phi_ij is
        # identical) but keeps the drift basis matrix P well-conditioned
        # — without centring, the linear drift terms (x, y, z) span
        # huge magnitudes in UTM space and dominate the condition
        # number of the augmented system.
        if n > 0:
            self._centroid = np.mean(coords, axis=0, keepdims=True)
        else:
            self._centroid = np.zeros((1, 3), dtype=float)
        coords_centred = coords - self._centroid

        self._n = n
        self._coords = coords
        self._coords_centred = coords_centred
        self._values = values
        self._rbf_kernel = rbf_kernel
        self._settings = settings
        self._anisotropy = anisotropy
        self._fail = False

        # ── Precompute ONCE: transform, distances, RBF matrix, augmented system ──

        # 1. Anisotropy-transform the (centred) local coordinates.
        # Distance matrix is the same either way; we use centred
        # coords below for the drift basis so the polynomial terms
        # stay O(1) instead of O(UTM-magnitude).
        self._coords_t = coords_t_pre

        # 2. Build n×n distance matrix using fast cdist
        r_nn = cdist(self._coords_t, self._coords_t)

        # 3. RBF interpolation matrix Φ from basis function
        Phi = rbf_kernel.evaluate(r_nn)

        # 4. Tikhonov regularisation: Φ + λI
        # Adaptive: max(user λ, fraction of kernel diagonal). Step up
        # the floor from 1e-8 to 1e-6 so long-range / well-conditioned
        # kernels still get a non-trivial diagonal and don't slide
        # into numerical degeneracy when all samples cluster close to
        # the origin after centring.
        user_diag = rbf_kernel.nugget + settings.smoothing + settings.epsilon
        kernel_scale = max(float(np.mean(np.diag(Phi))), 1e-12)
        adaptive_diag = max(user_diag, kernel_scale * 1e-6)
        Phi += adaptive_diag * np.eye(n, dtype=float)

        # Per-sample declustering weights — add extra diagonal
        # regularisation inversely proportional to weight.
        if self._sample_weights is not None and self._sample_weights.size == n:
            _w = self._sample_weights.astype(float)
            _mean = float(np.mean(_w))
            if _mean > 0 and np.any(_w != _mean):
                _wn = _w / _mean
                _extra = kernel_scale * np.maximum(
                    1.0 / np.maximum(_wn, 1e-3) - 1.0, 0.0
                )
                Phi[np.diag_indices_from(Phi)] += _extra

        # 5. Polynomial drift basis — evaluate on CENTRED coords so
        # linear/quadratic drift terms have O(1) magnitudes.
        P = drift_basis(coords_centred, settings.drift)
        m = P.shape[1]
        self._m = m
        self._P = P
        self._Phi_reg = Phi  # store for residual computation

        # 6. Augmented system [Φ+λI, P; P', 0]
        nm = n + m
        A = np.zeros((nm, nm), dtype=float)
        A[:n, :n] = Phi
        if m > 0:
            A[:n, n:] = P
            A[n:, :n] = P.T

        # 7. Condition number — compute for every block (not optional
        # any more). The cheap diagonal-ratio heuristic misses most
        # ill-conditioning; we pay the np.linalg.cond() cost per block
        # so we can reject unstable solves and log outliers. For small
        # n (typical neighbourhoods are ≤ 24 samples + a few drift
        # terms) this is a few µs per block.
        try:
            self._condnum = float(np.linalg.cond(A))
        except Exception:
            self._condnum = float("inf")
        if not np.isfinite(self._condnum):
            self._condnum = settings.condition_fail
            self._fail = True

        # 8. Factorize ONCE — progressive regularisation:
        #    (a) Cholesky (fast, symmetric positive definite)
        #    (b) Cholesky with boosted diagonal (add more λ)
        #    (c) scipy.linalg.solve with assume_a="sym" (LU-based)
        #    (d) SVD least-squares via lstsq (most stable, last resort)
        # Each step is tried only when the previous one fails so the
        # fast path stays fast.
        self._rhs = np.concatenate([values, np.zeros(m, dtype=float)])
        _solved = False
        self._use_cho = False
        self._solver_path = "cholesky"
        for _attempt in range(3):
            try:
                self._factor, self._lower = cho_factor(
                    A, lower=True, check_finite=False
                )
                self._sol = cho_solve(
                    (self._factor, self._lower), self._rhs, check_finite=False
                )
                _solved = True
                self._use_cho = True
                if _attempt > 0:
                    self._solver_path = f"cholesky_boosted_{_attempt}"
                break
            except LinAlgError:
                # Boost the diagonal and retry
                _boost = kernel_scale * 10 ** (_attempt - 3)
                A[:n, :n] += _boost * np.eye(n, dtype=float)

        if not _solved:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=LinAlgWarning)
                    self._sol = solve(A, self._rhs, assume_a="sym")
                _solved = True
                self._solver_path = "lu_sym"
                self._A = A
            except Exception:
                pass

        if not _solved:
            # Last-resort SVD least-squares. This always returns a
            # minimum-norm solution even for singular systems and is
            # the right tool for degenerate local neighbourhoods.
            try:
                from scipy.linalg import lstsq as _lstsq
                _sol, _res, _rank, _sv = _lstsq(
                    A, self._rhs, lapack_driver="gelsd"
                )
                self._sol = _sol
                _solved = True
                self._solver_path = "svd_lstsq"
                self._A = A
                # Update condnum from the SVD: ratio of largest to
                # smallest non-zero singular value.
                _sv_pos = _sv[_sv > 0]
                if _sv_pos.size > 1:
                    self._condnum = float(_sv_pos.max() / _sv_pos.min())
            except Exception:
                pass

        if not _solved:
            # Complete failure — return data mean (guarantee no crash)
            self._sol = np.zeros(nm, dtype=float)
            self._sol[:n] = 1.0 / max(n, 1)  # uniform weights
            self._fail = True
            self._use_cho = False
            self._A = A
            self._solver_path = "fallback_uniform"

        # 9. Effective sample size from weights
        w_abs = np.abs(self._sol[:n])
        w_sum = float(np.sum(w_abs))
        w_sq = float(np.sum(w_abs ** 2))
        self._neff = (w_sum ** 2) / max(w_sq, 1e-12)

        # 10. Compute local residual variance (for uncertainty estimation)
        # Use the UNREGULARISED kernel to compute fitted values.
        # The system solves (Phi + λI) w = z, so Phi_reg @ w ≈ z
        # (near-zero residual).  But Phi_raw @ w captures the actual
        # smoothing effect of the regularisation.
        weights = self._sol[:n]
        Phi_raw = rbf_kernel.evaluate(r_nn)  # unregularised
        fitted = Phi_raw @ weights
        if m > 0:
            fitted += P @ self._sol[n:]
        residuals = values - fitted
        self._residual_var = float(np.var(residuals)) if n > 1 else 0.0

        # 11. Store the local value range with a small tolerance so
        # ``predict_subpoints_batch`` can clamp runaway predictions
        # that sit outside [min, max] of the neighbourhood. For a
        # data-faithful interpolant the block estimate should stay
        # within the local range ± a fraction of that range; any
        # prediction outside that envelope is a numerical artefact
        # from an ill-conditioned local solve.
        if n > 0:
            self._local_min = float(np.min(values))
            self._local_max = float(np.max(values))
            _range = self._local_max - self._local_min
            # Allow 10% of the local range as soft tolerance — most
            # well-behaved RBF predictions sit within the envelope
            # but a small amount of overshoot is legitimate for
            # smooth kernels on heterogeneous data.
            self._local_tol = max(abs(_range) * 0.10, 1e-9)
        else:
            self._local_min = -np.inf
            self._local_max = np.inf
            self._local_tol = 0.0

    def predict_subpoints_batch(
        self,
        query_points: np.ndarray,
    ) -> BatchBlockResult:
        """Evaluate mean at all subpoints using the cached factorization.

        Uncertainty (sub_vars) is set to the local residual variance —
        this is NOT kriging variance.  It is the same value for all
        subpoints in the block, representing how well the local RBF
        fit reproduces the neighbourhood data.
        """
        query_points = _as_2d_float(query_points)
        n_sub = query_points.shape[0]
        n = self._n
        m = self._m
        nm = n + m
        settings = self._settings
        rbf_kernel = self._rbf_kernel

        if self._fail and np.all(self._sol[:n] == 0):
            # Total failure path
            mean_val = float(np.mean(self._values))
            var_val = self._residual_var
            return BatchBlockResult(
                sub_means=np.full(n_sub, mean_val),
                sub_vars=np.full(n_sub, var_val),
                condnum=self._condnum,
                neff=1.0,
                n_used=n,
                fail_flag=True,
            )

        # ── Transform query points ONCE ──────────────────────────────
        query_t = self._anisotropy.transform(query_points)

        # ── Build all RBF vectors at once: (n, n_sub) ────────────────
        r_nq = cdist(self._coords_t, query_t)  # (n, n_sub) distances
        phi_batch = rbf_kernel.evaluate(r_nq)   # (n, n_sub) basis values

        # ── Build drift basis for all subpoints: (m, n_sub) ──────────
        # IMPORTANT: drift basis must be evaluated on CENTRED query
        # coordinates because the system was solved on centred sample
        # coordinates. If we evaluate P(query) in raw UTM space the
        # linear drift β·x term will have a completely different
        # magnitude from what the solver saw, producing O(UTM) errors.
        if m > 0:
            query_centred = query_points - self._centroid
            P0 = drift_basis(query_centred, settings.drift)  # (n_sub, m)
            B = np.vstack([phi_batch, P0.T])  # (nm, n_sub)
        else:
            B = phi_batch  # (n, n_sub)

        # ── Batched means: sol @ B ───────────────────────────────────
        # ẑ(x) = Σ w_i φ(||x - x_i||) + p(x)'β
        means = self._sol @ B  # (n_sub,)

        # ── Local range clamp ────────────────────────────────────────
        # When the local solve is well-conditioned the prediction
        # sits within [local_min, local_max] of the neighbourhood
        # (with modest overshoot for smooth kernels). When the solve
        # is ill-conditioned the prediction can explode to absurd
        # values — the source of the "max=150,633 / median=690"
        # tail-inflation bug reported by the user. Clip predictions
        # to the neighbourhood envelope so a few unstable blocks
        # can't corrupt the whole model. The full block is also
        # flagged so the outlier diagnostic can see them.
        _hi = self._local_max + self._local_tol
        _lo = self._local_min - self._local_tol
        _runaway = np.any((means > _hi) | (means < _lo))
        if _runaway:
            means = np.clip(means, _lo, _hi)

        # ── Local residual spread as variance proxy ──────────────────
        vars_raw = np.full(n_sub, self._residual_var, dtype=float)

        fail = self._fail or (self._condnum >= settings.condition_fail)
        # Treat any runaway-prediction block as a fail for diagnostic
        # purposes so the post-estimation outlier logger can pick it
        # up. We still return clipped values (not NaN) so the block
        # model remains usable.
        if _runaway:
            fail = True

        return BatchBlockResult(
            sub_means=means.astype(float),
            sub_vars=vars_raw,
            condnum=self._condnum,
            neff=self._neff,
            n_used=n,
            fail_flag=fail,
        )
