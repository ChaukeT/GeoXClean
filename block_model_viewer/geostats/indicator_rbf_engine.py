"""
Indicator RBF Domain Engine
================================
Implicit domain estimation from drillhole indicator data using RBF
interpolation and robust isosurface extraction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

# ── Result container ─────────────────────────────────────────────

@dataclass
class IndicatorRBFResult:
    """All outputs from one run."""
    # probability volume
    prob: np.ndarray               # (nz, ny, nx) clipped [0,1]
    x: np.ndarray                  # (nx,)
    y: np.ndarray                  # (ny,)
    z: np.ndarray                  # (nz,)
    # isosurface
    verts: Optional[np.ndarray] = None   # (V,3) world coords
    faces: Optional[np.ndarray] = None   # (F,3) int64
    # per-sample
    labels: Optional[np.ndarray] = None  # (N,) "Inside"/"Outside"/"Unclassified"
    probs: Optional[np.ndarray] = None   # (N,) float, NaN for unclassifiable
    inside_mask: Optional[np.ndarray] = None  # (nz*ny*nx,) bool
    sign_flipped: bool = False
    iso_value: float = 0.5
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def probability_field(self) -> np.ndarray:
        """Alias for ``prob`` — the (nz, ny, nx) probability volume."""
        return self.prob

    @property
    def inside_mask_grid(self) -> Optional[np.ndarray]:
        """Alias for ``inside_mask`` — the flattened boolean mask."""
        return self.inside_mask

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["probability_field"] = self.prob
        d["inside_mask_grid"] = self.inside_mask
        return d


# ── Helpers ──────────────────────────────────────────────────────

def _build_cell_centre_coords(n: int, origin: float, spacing: float) -> np.ndarray:
    """Return 1-D array of cell-centre coordinates.

    Parameters
    ----------
    n : int
        Number of cells along this axis.
    origin : float
        Coordinate of the first cell's lower edge.
    spacing : float
        Cell size along this axis.

    Returns
    -------
    np.ndarray
        1-D array of length *n* with cell centres.
    """
    return np.arange(n) * spacing + origin + spacing / 2.0


def _volume_filter(
    mask: np.ndarray,
    shape: tuple,
    cell_volume: float,
    min_volume: float,
) -> tuple:
    """Remove connected components smaller than *min_volume*.

    Parameters
    ----------
    mask : 1-D bool array
        Flattened inside/outside mask (length = product of *shape*).
    shape : (nz, ny, nx)
        3-D shape for reshaping *mask*.
    cell_volume : float
        Volume of a single cell.
    min_volume : float
        Minimum volume to keep a connected component.

    Returns
    -------
    filtered : 1-D bool array  (same length as *mask*)
    n_components_before : int
    n_components_after : int
    """
    from scipy.ndimage import label as ndlabel

    if min_volume <= 0:
        n_before = ndlabel(mask.reshape(shape))[1] if mask.any() else 0
        return mask.copy(), n_before, n_before

    vol3 = mask.reshape(shape)
    labelled, n_comp = ndlabel(vol3)
    min_cells = max(1, int(np.ceil(min_volume / cell_volume)))

    keep = np.zeros_like(vol3, dtype=bool)
    n_after = 0
    for lbl in range(1, n_comp + 1):
        component = labelled == lbl
        if component.sum() >= min_cells:
            keep |= component
            n_after += 1

    return keep.ravel(), n_comp, n_after


def resample_mask_to_grid(
    prob_field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    target_coords: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Resample a probability field onto arbitrary target coordinates.

    Parameters
    ----------
    prob_field : (nz, ny, nx) array
        Probability field on a regular grid.
    x, y, z : 1-D arrays
        Cell-centre coordinates of the regular grid axes (lengths nx, ny, nz).
    target_coords : (M, 3) array
        XYZ query coordinates.
    threshold : float
        Probability threshold for the returned boolean mask.

    Returns
    -------
    np.ndarray
        Boolean mask of length M — True where interpolated probability >= threshold.
    """
    from scipy.interpolate import RegularGridInterpolator

    # prob_field shape is (nz, ny, nx) → axes order is (z, y, x)
    interp = RegularGridInterpolator(
        (z, y, x), prob_field, method="nearest",
        bounds_error=False, fill_value=0.0,
    )
    # target_coords columns are (X, Y, Z) → reorder to (Z, Y, X) for the interpolator
    probs = interp(target_coords[:, [2, 1, 0]])
    return probs >= threshold


def _auto_resolution(coords: np.ndarray, max_cells: int = 150_000) -> float:
    """Choose a grid cell size that keeps total cells near *max_cells*."""
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    nn_med = float(np.median(d[:, 1]))
    res = max(nn_med * 2.0, 1.0)

    ranges = coords.max(axis=0) - coords.min(axis=0)
    pad = np.maximum(ranges * 0.10, res)
    est = np.prod((ranges + 2 * pad) / res)
    if est > max_cells:
        scale = (est / max_cells) ** (1.0 / 3.0)
        res = float(np.ceil(res * scale))
    return res


def _make_grid(coords: np.ndarray, res: float):
    """Return (x, y, z) 1-D cell-centre arrays with 10% padding."""
    lo = coords.min(axis=0) - np.maximum((coords.max(axis=0) - coords.min(axis=0)) * 0.10, res)
    hi = coords.max(axis=0) + np.maximum((coords.max(axis=0) - coords.min(axis=0)) * 0.10, res)
    x = np.arange(lo[0] + res / 2, hi[0], res)
    y = np.arange(lo[1] + res / 2, hi[1], res)
    z = np.arange(lo[2] + res / 2, hi[2], res)
    if len(x) < 3: x = np.linspace(lo[0], hi[0], 3)
    if len(y) < 3: y = np.linspace(lo[1], hi[1], 3)
    if len(z) < 3: z = np.linspace(lo[2], hi[2], 3)
    return x, y, z


def _clip_far(prob, x, y, z, data, factor=2.0):
    """NaN-out grid cells far from any data point."""
    tree = cKDTree(data)
    d, _ = tree.query(data, k=2)
    clip_r = max(float(np.median(d[:, 1])) * factor,
                 5.0 * np.sqrt(sum((a[1]-a[0])**2 for a in (x, y, z) if len(a) > 1)))
    ZZ, YY, XX = np.meshgrid(z, y, x, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    dist, _ = tree.query(pts)
    out = prob.copy()
    out.ravel()[dist > clip_r] = np.nan
    return out


def _fix_sign(prob, coords, indicators, x, y, z, iso):
    """If most indicator=1 points are outside the iso, flip the field."""
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((z, y, x), prob, method="nearest",
                                bounds_error=False, fill_value=np.nan)
    p = f(np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0]]))
    mask1 = indicators >= 0.5
    if mask1.sum() == 0:
        return prob, False
    frac = np.nansum(p[mask1] >= iso) / mask1.sum()
    if frac >= 0.5:
        return prob, False
    logger.warning("Sign flip: only %.0f%% of indicator=1 inside iso=%.2f. Inverting.", frac*100, iso)
    return 1.0 - prob, True


def _extract_iso(prob, x, y, z, iso):
    """Marching cubes → (verts, faces) in world coords, or (None, None).

    Falls back to PyVista contouring if scikit-image is unavailable.
    """
    pmin, pmax = float(np.nanmin(prob)), float(np.nanmax(prob))
    if iso <= pmin or iso >= pmax:
        logger.warning("iso %.3f outside range [%.3f, %.3f]", iso, pmin, pmax)
        return None, None

    dz = float(z[1]-z[0]) if len(z) > 1 else 1.0
    dy = float(y[1]-y[0]) if len(y) > 1 else 1.0
    dx = float(x[1]-x[0]) if len(x) > 1 else 1.0

    # Try scikit-image first
    try:
        from skimage.measure import marching_cubes
        vi, fi, _, _ = marching_cubes(prob, level=iso, spacing=(dz, dy, dx))
        if len(vi) == 0:
            return None, None
        v = np.empty_like(vi)
        v[:, 0] = vi[:, 2] + x[0]
        v[:, 1] = vi[:, 1] + y[0]
        v[:, 2] = vi[:, 0] + z[0]
        return v, fi.astype(np.int64)
    except ImportError:
        logger.info("scikit-image not found, falling back to pyvista contouring")
    except Exception as e:
        logger.warning("skimage marching_cubes failed: %s, falling back to pyvista", e)

    # PyVista fallback
    try:
        import pyvista as pv
        grid = pv.ImageData(
            dimensions=(len(x), len(y), len(z)),
            spacing=(dx, dy, dz),
            origin=(x[0], y[0], z[0]),
        )
        grid.point_data["prob"] = prob.flatten(order="C")
        mesh = grid.contour([iso], scalars="prob")
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        if mesh.n_points == 0:
            return None, None
        faces = mesh.faces.reshape(-1, 4)[:, 1:]
        return np.asarray(mesh.points), faces.astype(np.int64)
    except Exception as e:
        logger.error("Isosurface extraction failed: %s", e)
        return None, None


def _close_mesh(verts, faces):
    """Attempt to close open boundaries on the isosurface mesh.

    Uses PyVista's ``fill_holes`` to cap open edges.  Returns the
    (possibly modified) vertices and faces.  If closing fails, returns
    the originals unchanged with a logged warning.
    """
    if verts is None or faces is None:
        return verts, faces
    try:
        import pyvista as pv
        fp = np.hstack([
            np.full((len(faces), 1), 3, dtype=np.int64),
            faces.astype(np.int64),
        ]).ravel()
        mesh = pv.PolyData(verts.astype(np.float64), fp)

        n_open = mesh.n_open_edges
        if n_open == 0:
            return verts, faces  # already watertight

        closed = mesh.fill_holes(hole_size=mesh.length)
        if closed.n_open_edges < n_open:
            if not closed.is_all_triangles:
                closed = closed.triangulate()
            cv = np.asarray(closed.points)
            cf_raw = closed.faces.reshape(-1, 4)[:, 1:]
            logger.info(
                "Mesh closing: %d open edges → %d (filled %d holes)",
                n_open, closed.n_open_edges, n_open - closed.n_open_edges,
            )
            return cv, cf_raw.astype(np.int64)
        else:
            logger.warning(
                "Isosurface has %d open edges that could not be closed. "
                "Block classification near boundaries may be unreliable.",
                n_open,
            )
    except Exception as exc:
        logger.warning("Mesh closing failed: %s", exc)
    return verts, faces


def _smooth(verts, faces, n_iter=15, relax=0.1):
    if n_iter <= 0 or verts is None:
        return verts, faces
    try:
        import pyvista as pv
        fp = np.hstack([np.full((len(faces), 1), 3, dtype=np.int64), faces]).ravel()
        m = pv.PolyData(verts.copy(), fp)
        s = m.smooth(n_iter=n_iter, relaxation_factor=relax, boundary_smoothing=False)
        return np.asarray(s.points), faces
    except Exception:
        return verts, faces


# ── Main entry point ─────────────────────────────────────────────

def run_indicator_rbf(
    coords: np.ndarray,
    values: np.ndarray,
    cutoff: float,
    *,
    iso_value: float = 0.5,
    kernel: str = "thin_plate_spline",
    smoothing: float = 0.5,
    trend_degree: int = 1,
    resolution: Optional[float] = None,
    clip_far_field: bool = True,
    clip_factor: float = 2.0,
    smooth_iterations: int = 15,
    anisotropy_ranges: Optional[tuple] = None,
    anisotropy_rotation: Optional[np.ndarray] = None,
    progress: Optional[Callable[[int, str], None]] = None,
    interp_nx: Optional[int] = None,
    interp_ny: Optional[int] = None,
    interp_nz: Optional[int] = None,
    interp_dx: Optional[float] = None,
    interp_dy: Optional[float] = None,
    interp_dz: Optional[float] = None,
) -> IndicatorRBFResult:
    """Run the full Indicator RBF pipeline.

    Parameters
    ----------
    anisotropy_ranges : (range_major, range_semi, range_minor) or None
        If provided, coordinates are scaled by 1/range per axis before
        RBF fitting.  Produces elongated domain boundaries.
    anisotropy_rotation : (3, 3) array or None
        Rotation matrix applied before range scaling.  Build from
        azimuth/dip/plunge angles.
    """

    from .rbf_interpolation import RBFModel3D, RBFAnisotropy

    def P(pct, msg):
        if progress:
            progress(pct, msg)

    # 1 — clean
    P(0, "Cleaning input...")
    coords = np.asarray(coords, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).ravel()
    if coords.shape[0] != len(values):
        raise ValueError(
            f"coords and values must have the same number of rows "
            f"(got {coords.shape[0]} vs {len(values)})"
        )

    # Clip iso_value to (0.1, 0.9) to avoid degenerate isosurfaces
    iso_value = float(np.clip(iso_value, 0.1, 0.9))

    ok = np.isfinite(values) & np.all(np.isfinite(coords), axis=1)
    cC, vC = coords[ok], values[ok]
    if len(cC) < 4:
        raise ValueError(f"Only {len(cC)} valid samples (need >= 4)")

    # 2 — binarise (strict >  so that cutoff=0 puts zeros outside)
    P(5, "Binarising...")
    ind = (vC > cutoff).astype(np.float64)
    n_in, n_out = int(ind.sum()), int((1 - ind).sum())
    if n_in == 0 or n_out == 0:
        raise ValueError(
            f"Cannot split: {n_in} inside, {n_out} outside at cutoff {cutoff}. "
            f"Adjust the cut-off so both classes have samples."
        )
    logger.info("Indicator: %d in, %d out (cutoff %.4g)", n_in, n_out, cutoff)

    # 2b — handle collocated points
    from scipy.spatial import cKDTree as _cKDTree
    _tree = _cKDTree(cC)
    _nn_dists, _ = _tree.query(cC, k=2)
    _median_nn = float(np.median(_nn_dists[:, 1]))
    _tol = max(_median_nn * 1e-4, 1e-6)
    _pairs = _tree.query_pairs(r=_tol)
    if _pairs:
        # Merge collocated duplicates: keep majority indicator vote
        _to_remove = set()
        for i, j in _pairs:
            if i in _to_remove or j in _to_remove:
                continue
            if ind[i] == ind[j]:
                _to_remove.add(j)  # same indicator — drop duplicate
            else:
                # conflicting indicators — keep indicator=1 (conservative)
                if ind[i] == 1.0:
                    _to_remove.add(j)
                else:
                    _to_remove.add(i)
        if _to_remove:
            _keep = np.array(sorted(set(range(len(cC))) - _to_remove))
            logger.info(
                "IRBF: Removed %d collocated duplicates (tolerance=%.2e)",
                len(_to_remove), _tol,
            )
            cC = cC[_keep]
            ind = ind[_keep]
            n_in, n_out = int(ind.sum()), int((1 - ind).sum())

    # Jitter remaining near-collocated points to prevent singularity
    _tree2 = _cKDTree(cC)
    _near_pairs = _tree2.query_pairs(r=_tol)
    if _near_pairs:
        _jittered = set()
        _rng = np.random.RandomState(42)
        for i, j in _near_pairs:
            if j not in _jittered:
                cC[j] += _rng.normal(0, _tol * 0.1, size=3)
                _jittered.add(j)
        if _jittered:
            logger.info("IRBF: Jittered %d near-collocated points", len(_jittered))

    # 3 — grid
    P(10, "Building grid...")
    if interp_nx is not None and interp_dx is not None:
        # Explicit grid dimensions provided
        origin = cC.min(axis=0)
        x = _build_cell_centre_coords(interp_nx, origin[0], interp_dx)
        y = _build_cell_centre_coords(interp_ny or interp_nx, origin[1], interp_dy or interp_dx)
        z = _build_cell_centre_coords(interp_nz or interp_nx, origin[2], interp_dz or interp_dx)
        res = min(interp_dx, interp_dy or interp_dx, interp_dz or interp_dx)
    else:
        res = resolution or _auto_resolution(cC)
        x, y, z = _make_grid(cC, res)
    nx, ny, nz = len(x), len(y), len(z)
    total = nx * ny * nz
    if total > 10_000_000:
        raise ValueError(f"Grid {nx}x{ny}x{nz}={total:,} too large (max 10M). Increase resolution.")
    logger.info("Grid %dx%dx%d = %s cells, res=%.1fm", nx, ny, nz, f"{total:,}", res)

    # 4 — fit
    P(20, "Fitting RBF...")
    _aniso = None
    if anisotropy_ranges is not None:
        rot = anisotropy_rotation if anisotropy_rotation is not None else np.eye(3)
        ranges = np.array(anisotropy_ranges, dtype=float)
        scales = np.diag(1.0 / np.maximum(ranges, 1e-6))
        _aniso = RBFAnisotropy(metric_matrix=scales @ rot)
        logger.info("IRBF anisotropy: ranges=%s", anisotropy_ranges)
    model = RBFModel3D(cC, ind, kernel=kernel,
                       smoothing=max(smoothing, 0.1),
                       classification=False, trend_degree=trend_degree,
                       anisotropy=_aniso)

    # 5 — evaluate (chunked by Z for progress)
    P(30, "Evaluating on grid...")
    prob = np.empty((nz, ny, nx), dtype=np.float64)
    chunk = max(1, min(5, nz))
    for i0 in range(0, nz, chunk):
        i1 = min(i0 + chunk, nz)
        _, _, _, sl = model.interpolate_grid(x, y, z[i0:i1], chunk_z=1)
        prob[i0:i1] = sl
        P(30 + int(25 * i1 / nz), f"Evaluating... {i1}/{nz}")
    prob = np.clip(prob, 0.0, 1.0)

    # 6 — clip far field
    P(58, "Clipping far field...")
    if clip_far_field:
        prob = _clip_far(prob, x, y, z, cC, clip_factor)

    # 7 — sign check
    P(62, "Checking sign convention...")
    pC = np.nan_to_num(prob, nan=0.0)
    pC, flipped = _fix_sign(pC, cC, ind, x, y, z, iso_value)

    # 8 — inside mask
    imask = (pC >= iso_value).ravel()

    # 9 — isosurface
    P(68, "Extracting isosurface...")
    verts, faces = _extract_iso(pC, x, y, z, iso_value)

    # 9b — close open boundaries
    P(74, "Closing mesh boundaries...")
    verts, faces = _close_mesh(verts, faces)

    # 10 — smooth
    P(80, "Smoothing mesh...")
    verts, faces = _smooth(verts, faces, smooth_iterations)

    # 11 — label all input samples
    P(90, "Labelling samples...")
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((z, y, x), pC, method="nearest",
                                     bounds_error=False, fill_value=np.nan)
    pts = np.column_stack([coords[:, 2], coords[:, 1], coords[:, 0]])
    sp = interp(pts)

    labels = np.full(len(coords), "Unclassified", dtype=object)
    classifiable = ok & np.isfinite(sp)
    labels[classifiable & (sp >= iso_value)] = "Inside"
    labels[classifiable & (sp < iso_value)] = "Outside"
    sp = np.where(np.isfinite(sp), np.clip(sp, 0, 1), np.nan)

    # 12 — stats
    acc = None
    if len(cC) >= 20:
        rng = np.random.RandomState(42)
        ti = rng.permutation(len(cC))[:max(5, len(cC) // 5)]
        tp = interp(np.column_stack([cC[ti, 2], cC[ti, 1], cC[ti, 0]]))
        acc = float(np.mean((tp >= iso_value) == (ind[ti] >= 0.5)))

    P(100, "Done")
    return IndicatorRBFResult(
        prob=pC, x=x, y=y, z=z,
        verts=verts, faces=faces,
        labels=labels, probs=sp,
        inside_mask=imask, sign_flipped=flipped, iso_value=iso_value,
        stats={
            "n_samples": len(cC), "n_inside": n_in, "n_outside": n_out,
            "cutoff": cutoff, "iso_value": iso_value,
            "kernel": kernel, "smoothing": max(smoothing, 0.1),
            "nx": nx, "ny": ny, "nz": nz, "total_cells": total,
            "resolution": res, "n_faces": len(faces) if faces is not None else 0,
            "sign_flipped": flipped, "holdout_accuracy": acc,
            "inside_cells": int(imask.sum()),
        },
    )
