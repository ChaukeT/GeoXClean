"""
Gradient Estimator - Geologically-Sound Orientation Computation.

This module derives bedding orientations from drillhole contact data using
three complementary strategies that mirror professional geological practice:

  1. PER-BOUNDARY PLANE FITTING  (Primary)
     For each stratigraphic boundary, fit a plane through all drillhole
     intercepts of that boundary. Distribute the resulting gradient to
     EVERY contact point on that boundary — not just the centroid.

  2. DRILLHOLE-SEQUENCE APPARENT DIP  (Augmentation)
     Within each drillhole, consecutive contacts define an apparent dip.
     This provides orientation data exactly where the observations are,
     anchoring the interpolation to hard data.

  3. CROSS-HOLE DIP ESTIMATION  (Regional)
     Compare the same boundary across nearby drillholes to resolve the
     full 3D dip tensor. This captures lateral dip variation that
     single-hole apparent dips cannot.

  4. LOCAL k-NN PCA  (Complex Geology)
     For formations with spatially variable dip (folds, drag zones),
     compute local gradients at each contact point using its nearest
     neighbours. This handles geology that a single global plane cannot.

The OLD code computed ONE gradient at the centroid per formation using global
PCA. This starved LoopStructural of orientation data, causing:
  - Surfaces that ignored observed dip ("hallucinated" flat-lying beds)
  - No constraint between drillholes → wild surface extrapolation
  - Failure to honour contacts → chronic audit failures

GeoX Invariant Compliance:
- All calculations are deterministic
- Provenance metadata for every computed gradient
- JORC/SAMREC audit trail support
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ContactGradient:
    """Gradient computed from contact point cloud via PCA plane fitting."""

    formation: str
    location: np.ndarray       # Centroid of the contact points (X, Y, Z)
    gradient: np.ndarray       # Unit normal vector (gx, gy, gz)
    dip: float                 # Dip angle in degrees (0-90)
    dip_direction: float       # Dip direction in degrees (0-360)
    confidence: float          # Based on planarity (eigenvalue ratio), 0-1
    n_points: int              # Number of contact points used
    source: str = "pca"        # "pca", "drillhole", "cross_hole", "local_knn"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialisation."""
        return {
            "formation": self.formation,
            "location": self.location.tolist(),
            "gradient": self.gradient.tolist(),
            "dip": self.dip,
            "dip_direction": self.dip_direction,
            "confidence": self.confidence,
            "n_points": self.n_points,
            "source": self.source,
        }

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"{self.formation}: dip={self.dip:.1f}° → "
            f"{self.dip_direction:.1f}° "
            f"conf={self.confidence:.2f} n={self.n_points} "
            f"[{self.source}]"
        )


@dataclass
class GradientEstimationReport:
    """
    Audit report for gradient estimation process.

    Tracks which formations had gradients computed vs. fell back to synthetic,
    along with confidence metrics for QC review.
    """

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    computed_gradients: List[ContactGradient] = field(default_factory=list)
    synthetic_formations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    gradient_source: str = "unknown"

    def add_computed(self, gradient: ContactGradient):
        self.computed_gradients.append(gradient)

    def add_synthetic(self, formation: str, reason: str):
        self.synthetic_formations.append(formation)
        self.warnings.append(f"{formation}: {reason}")

    def finalize(self):
        """Determine overall gradient source."""
        n_computed = len(self.computed_gradients)
        n_synthetic = len(self.synthetic_formations)

        if n_computed > 0 and n_synthetic == 0:
            self.gradient_source = "computed"
        elif n_computed > 0:
            self.gradient_source = "mixed"
        elif n_synthetic > 0:
            self.gradient_source = "synthetic"
        else:
            self.gradient_source = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gradient_source": self.gradient_source,
            "n_computed": len(self.computed_gradients),
            "n_synthetic": len(self.synthetic_formations),
            "computed_gradients": [g.to_dict() for g in self.computed_gradients],
            "synthetic_formations": self.synthetic_formations,
            "warnings": self.warnings,
        }

    @property
    def summary(self) -> str:
        return (
            f"Gradient Estimation: {self.gradient_source} "
            f"(computed={len(self.computed_gradients)}, "
            f"synthetic={len(self.synthetic_formations)})"
        )


# ---------------------------------------------------------------------------
# Core geometric helpers
# ---------------------------------------------------------------------------

def _normal_to_dip_dipdir(n: np.ndarray) -> Tuple[float, float]:
    """
    Convert a plane normal vector to standard geological notation.

    Convention:
      - Normal is forced upward (nz > 0) for stratigraphic surfaces.
      - Dip = angle of steepest descent from horizontal (0–90°).
      - Dip direction = azimuth of steepest descent (0–360°, clockwise from N).
    """
    n = n.copy()

    # Ensure upward-pointing normal (stratigraphic convention)
    if n[2] < 0:
        n = -n

    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return 0.0, 0.0
    n /= norm

    # Dip from horizontal = 90° - arccos(nz)  →  but standard dip = arccos(nz)
    # when nz=1 → dip=0 (horizontal), nz=0 → dip=90° (vertical)
    dip = np.degrees(np.arccos(np.clip(n[2], -1, 1)))

    # Dip direction: azimuth of the horizontal projection of normal
    dip_dir = np.degrees(np.arctan2(n[0], n[1])) % 360

    return round(dip, 1), round(dip_dir, 1)


def _dip_dipdir_to_gradient(dip: float, dip_dir: float) -> np.ndarray:
    """
    Convert dip/dip-direction to gradient vector (gx, gy, gz).

    The gradient points in the direction of *increasing* stratigraphic value.
    For normal (right-way-up) stratigraphy this is broadly upward.
    """
    dip_rad = np.radians(dip)
    azim_rad = np.radians(dip_dir)

    gx = np.sin(dip_rad) * np.sin(azim_rad)
    gy = np.sin(dip_rad) * np.cos(azim_rad)
    gz = np.cos(dip_rad)

    return np.array([gx, gy, gz])


def _fit_plane_pca(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a plane to a point cloud using PCA.

    Returns (normal_vector, centroid, confidence).
    Confidence is a planarity score [0-1]; higher = more planar.
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")

    pca = PCA(n_components=3)
    pca.fit(points)

    normal_vec = pca.components_[2]     # direction of least variance
    centroid = pca.mean_

    ev = pca.explained_variance_ratio_
    # Planarity: ratio of smallest to second-smallest eigenvalue
    if ev[1] > 1e-10:
        confidence = 1.0 - (ev[2] / ev[1])
    else:
        confidence = 0.0

    return normal_vec, centroid, float(np.clip(confidence, 0.0, 1.0))


# ---------------------------------------------------------------------------
# STRATEGY 1: Per-boundary plane fitting (PRIMARY)
# ---------------------------------------------------------------------------

def _compute_boundary_gradients(
    contacts_df: pd.DataFrame,
    stratigraphy: List[str],
    min_points: int = 3,
    min_confidence: float = 0.25,
) -> Tuple[List[Dict[str, float]], List[ContactGradient], List[str]]:
    """
    Fit a plane to each stratigraphic BOUNDARY (not each formation).

    A boundary is the contact surface between formation[i] and formation[i+1].
    We collect all contact points for formation[i+1] (the formation immediately
    above the boundary) and fit a plane through them.

    Crucially, the resulting gradient is then distributed to EVERY contact
    point — not just the centroid — giving LoopStructural dense orientation
    constraints anchored to hard data.
    """
    orientations: List[Dict[str, float]] = []
    gradients: List[ContactGradient] = []
    warnings: List[str] = []

    for formation in stratigraphy:
        mask = contacts_df["formation"] == formation
        pts = contacts_df.loc[mask, ["X", "Y", "Z"]].values

        if len(pts) < min_points:
            warnings.append(
                f"Boundary '{formation}': only {len(pts)} points "
                f"(need {min_points}), skipping plane fit."
            )
            continue

        try:
            normal_vec, centroid, confidence = _fit_plane_pca(pts)
        except Exception as e:
            warnings.append(f"Boundary '{formation}': PCA failed: {e}")
            continue

        if confidence < min_confidence:
            warnings.append(
                f"Boundary '{formation}': low planarity "
                f"({confidence:.2f} < {min_confidence}), skipping."
            )
            continue

        dip, dip_dir = _normal_to_dip_dipdir(normal_vec)
        gradient = _dip_dipdir_to_gradient(dip, dip_dir)

        cg = ContactGradient(
            formation=formation,
            location=centroid,
            gradient=gradient,
            dip=dip,
            dip_direction=dip_dir,
            confidence=confidence,
            n_points=len(pts),
            source="pca_boundary",
        )
        gradients.append(cg)

        # ── KEY FIX: distribute to EVERY contact point ──
        for pt in pts:
            orientations.append({
                "X": float(pt[0]),
                "Y": float(pt[1]),
                "Z": float(pt[2]),
                "gx": float(gradient[0]),
                "gy": float(gradient[1]),
                "gz": float(gradient[2]),
                "formation": formation,
                "confidence": confidence,
            })

        logger.info(f"  Boundary '{formation}': {cg.summary}")

    return orientations, gradients, warnings


# ---------------------------------------------------------------------------
# STRATEGY 2: Drillhole-sequence apparent dip (AUGMENTATION)
# ---------------------------------------------------------------------------

def _compute_drillhole_gradients(
    contacts_df: pd.DataFrame,
    stratigraphy: List[str],
    formation_values: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, float]], List[ContactGradient], List[str]]:
    """
    For each drillhole, compute apparent dip from consecutive contact depths.

    Within a single (vertical) drillhole the dip cannot be fully resolved
    (no horizontal component), but the vertical gradient magnitude and
    polarity are directly observed. These measurements anchor the interpolator
    at every drillhole location — preventing the surface from "floating"
    away from hard data.

    For vertical drillholes the gradient is approximately (0, 0, ±1) with
    magnitude proportional to val_change / depth_change. Even this seemingly
    trivial constraint dramatically improves contact snapping because it tells
    the solver "here, the surface changes at this rate with depth".
    """
    orientations: List[Dict[str, float]] = []
    gradients: List[ContactGradient] = []
    warnings: List[str] = []

    if "hole_id" not in contacts_df.columns:
        warnings.append(
            "No 'hole_id' column — skipping drillhole-sequence gradients."
        )
        return orientations, gradients, warnings

    # Build formation → scalar mapping
    if formation_values is None:
        formation_values = {fm: float(i) for i, fm in enumerate(stratigraphy)}

    strat_set = set(stratigraphy)

    for hole_id, group in contacts_df.groupby("hole_id"):
        # Sort deepest first (highest Z first for elevation, or lowest Z first for depth)
        grp = group.sort_values("Z", ascending=True).reset_index(drop=True)

        if len(grp) < 2:
            continue

        for i in range(len(grp) - 1):
            fm_lower = grp.loc[i, "formation"]
            fm_upper = grp.loc[i + 1, "formation"]

            if fm_lower not in strat_set or fm_upper not in strat_set:
                continue

            val_lower = formation_values.get(fm_lower, float(stratigraphy.index(fm_lower)) if fm_lower in stratigraphy else None)
            val_upper = formation_values.get(fm_upper, float(stratigraphy.index(fm_upper)) if fm_upper in stratigraphy else None)

            if val_lower is None or val_upper is None:
                continue

            dval = val_upper - val_lower
            x0, y0, z0 = grp.loc[i, "X"], grp.loc[i, "Y"], grp.loc[i, "Z"]
            x1, y1, z1 = grp.loc[i + 1, "X"], grp.loc[i + 1, "Y"], grp.loc[i + 1, "Z"]

            dz = z1 - z0
            dx = x1 - x0
            dy = y1 - y0
            dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

            if dist < 1e-6 or abs(dval) < 1e-10:
                continue

            # Gradient vector: direction of increasing scalar value
            # For most vertical holes this is approximately (0, 0, sign)
            gx = dx / dist * np.sign(dval)
            gy = dy / dist * np.sign(dval)
            gz = dz / dist * np.sign(dval)

            g_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
            if g_mag < 1e-10:
                continue
            gx /= g_mag
            gy /= g_mag
            gz /= g_mag

            # Midpoint between the two contacts
            mx = (x0 + x1) / 2.0
            my = (y0 + y1) / 2.0
            mz = (z0 + z1) / 2.0

            # Confidence scales with how close the hole is to vertical
            # (vertical holes give reliable gz but no gx/gy information)
            verticality = abs(dz) / dist if dist > 0 else 0
            confidence = 0.5 + 0.5 * verticality  # 0.5–1.0

            dip, dip_dir = _normal_to_dip_dipdir(np.array([gx, gy, gz]))

            cg = ContactGradient(
                formation=fm_upper,
                location=np.array([mx, my, mz]),
                gradient=np.array([gx, gy, gz]),
                dip=dip,
                dip_direction=dip_dir,
                confidence=confidence,
                n_points=2,
                source="drillhole_sequence",
            )
            gradients.append(cg)

            # Place orientation at BOTH contact points + midpoint
            for px, py, pz in [(x0, y0, z0), (mx, my, mz), (x1, y1, z1)]:
                orientations.append({
                    "X": float(px),
                    "Y": float(py),
                    "Z": float(pz),
                    "gx": float(gx),
                    "gy": float(gy),
                    "gz": float(gz),
                    "formation": fm_upper,
                    "confidence": confidence,
                })

    logger.info(
        f"  Drillhole-sequence: {len(gradients)} apparent-dip vectors "
        f"from {contacts_df['hole_id'].nunique()} holes"
    )
    return orientations, gradients, warnings


# ---------------------------------------------------------------------------
# STRATEGY 3: Cross-hole dip estimation (REGIONAL)
# ---------------------------------------------------------------------------

def _compute_crosshole_gradients(
    contacts_df: pd.DataFrame,
    stratigraphy: List[str],
    formation_values: Optional[Dict[str, float]] = None,
    max_distance_m: float = 2000.0,
    min_pairs: int = 3,
) -> Tuple[List[Dict[str, float]], List[ContactGradient], List[str]]:
    """
    Estimate dip by comparing the same boundary across nearby drillholes.

    If boundary B is at depth Z1 in hole H1 and depth Z2 in hole H2, and the
    horizontal distance is D, then the apparent dip between the two holes is
    arctan((Z2-Z1)/D). By combining multiple pairs we recover the full 3D dip.
    """
    from scipy.spatial import cKDTree

    orientations: List[Dict[str, float]] = []
    gradients: List[ContactGradient] = []
    warnings: List[str] = []

    if "hole_id" not in contacts_df.columns:
        warnings.append("No 'hole_id' — skipping cross-hole gradients.")
        return orientations, gradients, warnings

    if formation_values is None:
        formation_values = {fm: float(i) for i, fm in enumerate(stratigraphy)}

    # For each formation boundary, collect the (X, Y, Z) of every drillhole intercept
    for formation in stratigraphy:
        mask = contacts_df["formation"] == formation
        fm_df = contacts_df.loc[mask].copy()

        if len(fm_df) < min_pairs:
            continue

        # Use hole centroids (one point per hole) to avoid within-hole duplication
        if "hole_id" in fm_df.columns:
            hole_pts = fm_df.groupby("hole_id")[["X", "Y", "Z"]].mean().values
        else:
            hole_pts = fm_df[["X", "Y", "Z"]].values

        if len(hole_pts) < min_pairs:
            continue

        # Fit plane to these drillhole intercepts
        try:
            normal_vec, centroid, confidence = _fit_plane_pca(hole_pts)
        except Exception:
            continue

        if confidence < 0.2:
            continue

        dip, dip_dir = _normal_to_dip_dipdir(normal_vec)
        gradient = _dip_dipdir_to_gradient(dip, dip_dir)

        cg = ContactGradient(
            formation=formation,
            location=centroid,
            gradient=gradient,
            dip=dip,
            dip_direction=dip_dir,
            confidence=confidence * 0.8,  # slightly discount vs per-boundary
            n_points=len(hole_pts),
            source="cross_hole",
        )
        gradients.append(cg)

        # Distribute to hole intercept midpoints
        for pt in hole_pts:
            orientations.append({
                "X": float(pt[0]),
                "Y": float(pt[1]),
                "Z": float(pt[2]),
                "gx": float(gradient[0]),
                "gy": float(gradient[1]),
                "gz": float(gradient[2]),
                "formation": formation,
                "confidence": confidence * 0.8,
            })

    logger.info(f"  Cross-hole: {len(gradients)} regional dip estimates")
    return orientations, gradients, warnings


# ---------------------------------------------------------------------------
# STRATEGY 4: Local k-NN PCA (for complex / folded geology)
# ---------------------------------------------------------------------------

def compute_local_gradients(
    contacts_df: pd.DataFrame,
    stratigraphy: List[str],
    k_neighbors: int = 10,
    min_points_per_formation: int = 3,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Compute local gradient vectors using k-nearest neighbour PCA.

    For formations with spatially variable dip (folds, drag zones), a single
    global plane is wrong. This function fits a local plane at each contact
    point using its k nearest neighbours.
    """
    from scipy.spatial import cKDTree

    logger.info("Computing local gradients using k-NN analysis...")

    required_cols = ["X", "Y", "Z", "formation"]
    missing = [c for c in required_cols if c not in contacts_df.columns]
    if missing:
        raise ValueError(f"contacts_df missing required columns: {missing}")

    warnings: List[str] = []
    orientations_list: List[Dict[str, float]] = []

    for formation in stratigraphy:
        mask = contacts_df["formation"] == formation
        pts = contacts_df.loc[mask, ["X", "Y", "Z"]].values
        n = len(pts)

        if n < min_points_per_formation:
            warnings.append(f"Formation '{formation}': only {n} points, skipping.")
            continue

        if n < k_neighbors + 1:
            # Fall back to global PCA
            try:
                normal_vec, centroid, confidence = _fit_plane_pca(pts)
                if confidence >= 0.25:
                    dip, dip_dir = _normal_to_dip_dipdir(normal_vec)
                    gradient = _dip_dipdir_to_gradient(dip, dip_dir)
                    orientations_list.append({
                        "X": float(centroid[0]),
                        "Y": float(centroid[1]),
                        "Z": float(centroid[2]),
                        "gx": float(gradient[0]),
                        "gy": float(gradient[1]),
                        "gz": float(gradient[2]),
                        "formation": formation,
                    })
            except Exception as e:
                warnings.append(f"Formation '{formation}': {e}")
            continue

        tree = cKDTree(pts)

        for i, point in enumerate(pts):
            _, idx = tree.query(point, k=min(k_neighbors + 1, n))
            nbrs = pts[idx[1:]]  # skip self

            if len(nbrs) < 3:
                continue

            try:
                normal_vec, _, confidence = _fit_plane_pca(nbrs)
                if confidence >= 0.25:
                    dip, dip_dir = _normal_to_dip_dipdir(normal_vec)
                    gradient = _dip_dipdir_to_gradient(dip, dip_dir)
                    orientations_list.append({
                        "X": float(point[0]),
                        "Y": float(point[1]),
                        "Z": float(point[2]),
                        "gx": float(gradient[0]),
                        "gy": float(gradient[1]),
                        "gz": float(gradient[2]),
                        "formation": formation,
                    })
            except Exception:
                continue

    if orientations_list:
        df = pd.DataFrame(orientations_list)
        logger.info(f"Computed {len(df)} local gradients")
    else:
        df = pd.DataFrame(columns=["X", "Y", "Z", "gx", "gy", "gz"])
        warnings.append("No local gradients could be computed.")

    return df, warnings


# ---------------------------------------------------------------------------
# PUBLIC API: Unified gradient computation
# ---------------------------------------------------------------------------

def compute_contact_gradients(
    contacts_df: pd.DataFrame,
    stratigraphy: List[str],
    min_points_per_formation: int = 3,
    min_confidence: float = 0.25,
    formation_values: Optional[Dict[str, float]] = None,
    use_drillhole_sequence: bool = True,
    use_crosshole: bool = True,
    use_local_knn: bool = False,
    k_neighbors: int = 8,
) -> Tuple[pd.DataFrame, List[ContactGradient], List[str]]:
    """
    Compute gradient vectors from contact data using multiple strategies.

    This is the MAIN entry point for gradient computation. It combines:
      1. Per-boundary PCA plane fitting (always)
      2. Drillhole-sequence apparent dip (if hole_id present)
      3. Cross-hole dip estimation (if hole_id present + enough holes)
      4. Optional local k-NN for complex geology

    The result is a DENSE set of orientation constraints — at least one
    orientation per contact point — that guides LoopStructural's FDI solver
    to produce surfaces that honour drillhole dip and pass through contacts.

    Args:
        contacts_df: DataFrame with X, Y, Z, formation (and optionally hole_id)
        stratigraphy: Formation names from oldest to youngest
        min_points_per_formation: Minimum contacts for PCA (default 3)
        min_confidence: Minimum planarity for acceptance (default 0.25)
        formation_values: Optional {formation: scalar_value} mapping
        use_drillhole_sequence: Compute apparent dip from hole sequences
        use_crosshole: Compute cross-hole dip estimates
        use_local_knn: Use local k-NN PCA (for folded geology)
        k_neighbors: k for local k-NN if enabled

    Returns:
        (orientations_df, computed_gradients, warnings)
    """
    logger.info("=" * 60)
    logger.info("GRADIENT ESTIMATOR - Multi-Strategy Orientation Computation")
    logger.info("=" * 60)

    required_cols = ["X", "Y", "Z", "formation"]
    missing = [c for c in required_cols if c not in contacts_df.columns]
    if missing:
        raise ValueError(f"contacts_df missing required columns: {missing}")

    all_orientations: List[Dict[str, float]] = []
    all_gradients: List[ContactGradient] = []
    all_warnings: List[str] = []

    # ── Strategy 1: Per-boundary plane fitting (PRIMARY) ──
    logger.info("Strategy 1: Per-boundary PCA plane fitting...")
    o1, g1, w1 = _compute_boundary_gradients(
        contacts_df, stratigraphy, min_points_per_formation, min_confidence
    )
    all_orientations.extend(o1)
    all_gradients.extend(g1)
    all_warnings.extend(w1)
    logger.info(f"  → {len(o1)} orientation points from {len(g1)} boundaries")

    # ── Strategy 2: Drillhole-sequence apparent dip ──
    if use_drillhole_sequence and "hole_id" in contacts_df.columns:
        logger.info("Strategy 2: Drillhole-sequence apparent dip...")
        o2, g2, w2 = _compute_drillhole_gradients(
            contacts_df, stratigraphy, formation_values
        )
        all_orientations.extend(o2)
        all_gradients.extend(g2)
        all_warnings.extend(w2)
        logger.info(f"  → {len(o2)} orientation points from drillhole sequences")

    # ── Strategy 3: Cross-hole dip estimation ──
    if use_crosshole and "hole_id" in contacts_df.columns:
        n_holes = contacts_df["hole_id"].nunique()
        if n_holes >= 3:
            logger.info("Strategy 3: Cross-hole dip estimation...")
            o3, g3, w3 = _compute_crosshole_gradients(
                contacts_df, stratigraphy, formation_values
            )
            all_orientations.extend(o3)
            all_gradients.extend(g3)
            all_warnings.extend(w3)
            logger.info(f"  → {len(o3)} orientation points from cross-hole analysis")
        else:
            all_warnings.append(
                f"Only {n_holes} drillholes — need 3+ for cross-hole dip."
            )

    # ── Strategy 4: Local k-NN (optional, for complex geology) ──
    if use_local_knn:
        logger.info("Strategy 4: Local k-NN gradient computation...")
        local_df, local_warns = compute_local_gradients(
            contacts_df, stratigraphy, k_neighbors, min_points_per_formation
        )
        all_warnings.extend(local_warns)
        if len(local_df) > 0:
            for _, row in local_df.iterrows():
                all_orientations.append({
                    "X": float(row["X"]),
                    "Y": float(row["Y"]),
                    "Z": float(row["Z"]),
                    "gx": float(row["gx"]),
                    "gy": float(row["gy"]),
                    "gz": float(row["gz"]),
                    "formation": row.get("formation", "unknown"),
                    "confidence": 0.6,
                })
            logger.info(f"  → {len(local_df)} orientation points from local k-NN")

    # ── Deduplicate and compile ──
    if all_orientations:
        orientations_df = pd.DataFrame(all_orientations)

        # Remove exact-duplicate locations (keep the highest-confidence)
        orientations_df = (
            orientations_df
            .sort_values("confidence", ascending=False)
            .drop_duplicates(subset=["X", "Y", "Z"], keep="first")
            .reset_index(drop=True)
        )

        logger.info(
            f"GRADIENT TOTAL: {len(orientations_df)} unique orientation points "
            f"from {len(all_gradients)} gradient computations"
        )
    else:
        orientations_df = pd.DataFrame(columns=["X", "Y", "Z", "gx", "gy", "gz"])
        all_warnings.append(
            "No valid gradients could be computed from contact geometry. "
            "Ensure sufficient contact points per formation."
        )
        logger.warning("No valid gradients computed!")

    return orientations_df, all_gradients, all_warnings
