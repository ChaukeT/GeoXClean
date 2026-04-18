"""
Signed Distance Function Construction from Drillhole Contacts.
===============================================================

Two methods (Eq. 2.1 from math spec):

Method A — Offset Points (Cowan et al. 2003):
    Contact: f(x_c) = 0
    Above:   f(x_c + eps*n) = +eps
    Below:   f(x_c - eps*n) = -eps

Method B — Gradient Constraints (Hillier et al. 2014):
    Contact: f(x_c) = 0
    Gradient: grad f(x_c) . n = 1
    No offset points needed.  Cleaner and avoids epsilon parameter.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .contact_data import (
    ContactSet,
    StratigraphicColumn,
    dip_azimuth_to_normal,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# SDF Constraint Construction
# ═══════════════════════════════════════════════════════════════════

def construct_sdf_constraints(
    contact_points: np.ndarray,
    contact_normals: np.ndarray,
    method: str = "gradient",
    offset_distance: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct SDF constraints from contact points and normals.

    Parameters
    ----------
    contact_points : np.ndarray, shape (N_c, 3)
        Contact locations.
    contact_normals : np.ndarray, shape (N_c, 3)
        Unit surface normals at contacts.
    method : str
        ``"gradient"`` (Hillier 2014) or ``"offset"`` (Cowan 2003).
    offset_distance : float
        Offset epsilon for the offset method (metres).

    Returns
    -------
    value_coords : np.ndarray, shape (N_v, 3)
    value_data : np.ndarray, shape (N_v,)
    gradient_coords : np.ndarray, shape (N_g, 3)
    gradient_normals : np.ndarray, shape (N_g, 3)
    """
    N_c = contact_points.shape[0]
    if N_c == 0:
        empty3 = np.empty((0, 3), dtype=np.float64)
        empty1 = np.empty((0,), dtype=np.float64)
        return empty3, empty1, empty3.copy(), empty3.copy()

    # Ensure unit normals
    norms = np.linalg.norm(contact_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals = contact_normals / norms

    if method == "gradient":
        # Value constraints: f(x_c) = 0
        value_coords = contact_points.copy()
        value_data = np.zeros(N_c, dtype=np.float64)
        # Gradient constraints: grad f . n = 1 at each contact
        gradient_coords = contact_points.copy()
        gradient_normals_out = normals.copy()

    elif method == "offset":
        # Three value constraints per contact: 0, +eps, -eps
        eps = offset_distance
        above = contact_points + eps * normals
        below = contact_points - eps * normals

        value_coords = np.vstack([contact_points, above, below])
        value_data = np.concatenate([
            np.zeros(N_c),
            np.full(N_c, +eps),
            np.full(N_c, -eps),
        ])
        # No gradient constraints for offset method
        gradient_coords = np.empty((0, 3), dtype=np.float64)
        gradient_normals_out = np.empty((0, 3), dtype=np.float64)

    else:
        raise ValueError(f"Unknown SDF method: '{method}'. Use 'gradient' or 'offset'.")

    return value_coords, value_data, gradient_coords, gradient_normals_out


# ═══════════════════════════════════════════════════════════════════
# Survey-Based Direction Interpolation
# ═══════════════════════════════════════════════════════════════════

def get_hole_direction_at_depth(
    surveys_df: pd.DataFrame,
    hole_id: str,
    depth: float,
) -> np.ndarray:
    """Return the unit direction vector of a drillhole at a given depth.

    Linearly interpolates between survey stations.

    Parameters
    ----------
    surveys_df : pd.DataFrame
        Survey table. Accepts both uppercase (HOLEID, FROM, DIP, AZIMUTH)
        and lowercase (hole_id, depth, dip, azimuth) column names.
    hole_id : str
    depth : float (metres along hole)

    Returns
    -------
    np.ndarray, shape (3,) — unit vector pointing along the hole (downward).
    """
    if surveys_df is None or surveys_df.empty:
        return np.array([0.0, 0.0, -1.0])  # vertical down

    col_lc = {c.lower(): c for c in surveys_df.columns}

    # Resolve column names flexibly
    id_col  = col_lc.get("holeid") or col_lc.get("hole_id") or col_lc.get("bhid")
    dep_col = (col_lc.get("from") or col_lc.get("depth_from")
               or col_lc.get("depth") or col_lc.get("md"))
    dip_col = col_lc.get("dip") or col_lc.get("inclination") or col_lc.get("inc")
    az_col  = col_lc.get("azimuth") or col_lc.get("azi") or col_lc.get("bearing")

    if not all([id_col, dep_col, dip_col, az_col]):
        return np.array([0.0, 0.0, -1.0])

    hole_surv = surveys_df[surveys_df[id_col].astype(str) == str(hole_id)].copy()
    if hole_surv.empty:
        return np.array([0.0, 0.0, -1.0])

    hole_surv = hole_surv.sort_values(dep_col)
    depths = hole_surv[dep_col].values.astype(np.float64)
    dips   = hole_surv[dip_col].values.astype(np.float64)
    azs    = hole_surv[az_col].values.astype(np.float64)

    if len(depths) == 1 or depth <= depths[0]:
        dip, az = dips[0], azs[0]
    elif depth >= depths[-1]:
        dip, az = dips[-1], azs[-1]
    else:
        k = int(np.searchsorted(depths, depth)) - 1
        k = max(0, min(k, len(depths) - 2))
        frac = (depth - depths[k]) / max(depths[k + 1] - depths[k], 1e-6)
        dip = dips[k] + frac * (dips[k + 1] - dips[k])
        az  = azs[k]  + frac * (azs[k + 1]  - azs[k])

    # Convert to direction vector.
    # GeoX convention: dip is measured from horizontal, negative = downward.
    #   dip = -90° → straight down   dip = 0° → horizontal
    # Formula (inclination from horizontal):
    #   horizontal component = cos(|dip|)   (→ 0 for vertical hole)
    #   vertical component   = sin(|dip|)   (→ 1 for vertical hole)
    inc = np.radians(abs(dip))   # inclination magnitude from horizontal
    azr = np.radians(az)
    dx =  np.cos(inc) * np.sin(azr)   # East
    dy =  np.cos(inc) * np.cos(azr)   # North
    dz = -np.sin(inc)                  # Down (negative)

    direction = np.array([dx, dy, dz], dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm > 1e-12:
        direction /= norm
    return direction


def estimate_surface_normal_from_contacts_and_surveys(
    contact_coords: np.ndarray,
    hole_directions: np.ndarray,
) -> np.ndarray:
    """Estimate surface normal using contact positions + drillhole directions.

    Primary: PCA of contact positions.
    Refinement: drillhole direction vectors are added as soft constraints
    (as virtual points offset along each hole direction).  This biases the
    fitted plane towards orientations that the holes can reasonably intersect,
    resolving the ambiguity that pure positional PCA has in flat/thin bodies.

    Parameters
    ----------
    contact_coords : (K, 3) — 3D positions of contacts on this surface
    hole_directions : (K, 3) — unit direction vectors of the hole at each contact

    Returns
    -------
    np.ndarray, shape (3,) — upward-pointing unit normal
    """
    K = contact_coords.shape[0]

    if K < 2:
        t = hole_directions[0]
        # Normal perpendicular to hole direction in the vertical plane
        # Rotate t 90° towards the vertical axis
        horizontal = np.array([t[0], t[1], 0.0])
        h_len = np.linalg.norm(horizontal)
        if h_len > 1e-8:
            n = np.cross(horizontal / h_len, t)
            n_len = np.linalg.norm(n)
            if n_len > 1e-8:
                n /= n_len
                if n[2] < 0:
                    n = -n
                return n
        return np.array([0.0, 0.0, 1.0])

    if K == 2:
        d12 = contact_coords[1] - contact_coords[0]
        d12_len = np.linalg.norm(d12)
        if d12_len < 1e-8:
            return np.array([0.0, 0.0, 1.0])
        d12 /= d12_len
        t_avg = np.mean(hole_directions, axis=0)
        t_avg_len = np.linalg.norm(t_avg)
        if t_avg_len > 1e-8:
            t_avg /= t_avg_len
            n = np.cross(d12, t_avg)
            n_len = np.linalg.norm(n)
            if n_len > 1e-8:
                n /= n_len
                if n[2] < 0:
                    n = -n
                return n

    # K >= 3 — cross-product normal estimation.
    #
    # For each adjacent pair of contacts, compute cross(d_ij, t_ij) where:
    #   d_ij = unit vector from contact i to contact j (strikes across the surface)
    #   t_ij = mean drillhole direction at the two contacts
    # The cross product is perpendicular to both → the surface normal.
    # This avoids the PCA virtual-point bias that drags the plane to contain the drillhole.
    cross_normals = []
    for k in range(K - 1):
        d = contact_coords[k + 1] - contact_coords[k]
        d_len = np.linalg.norm(d)
        if d_len < 1e-8:
            continue
        d /= d_len
        t_pair = (hole_directions[k] + hole_directions[k + 1]) * 0.5
        t_len = np.linalg.norm(t_pair)
        if t_len < 1e-8:
            continue
        t_pair /= t_len
        cross = np.cross(d, t_pair)
        c_len = np.linalg.norm(cross)
        if c_len > 1e-8:
            cross_normals.append(cross / c_len)

    if cross_normals:
        n = np.mean(cross_normals, axis=0)
        n_len = np.linalg.norm(n)
        if n_len > 1e-8:
            n /= n_len
            if n[2] < 0.0:
                n = -n
            logger.debug(
                "Cross-product normal (K=%d): [%.3f, %.3f, %.3f]  dip=%.1f° from horizontal",
                K, n[0], n[1], n[2],
                float(np.degrees(np.arccos(np.clip(abs(n[2]), 0.0, 1.0)))),
            )
            return n

    # Fallback: pure positional PCA (no drillhole-direction bias)
    centroid = contact_coords.mean(axis=0)
    centered = contact_coords - centroid
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    n = eigenvectors[:, 0].copy()
    if n[2] < 0.0:
        n = -n
    return n


# ═══════════════════════════════════════════════════════════════════
# Contact Extraction from Lithology Logs
# ═══════════════════════════════════════════════════════════════════

def extract_contacts_from_lithology(
    drillhole_data: pd.DataFrame,
    lithology_column: str = "lith_code",
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
    from_col: str = "depth_from",
    to_col: str = "depth_to",
    hole_id_col: str = "hole_id",
    grouping: Optional[Dict[str, List[str]]] = None,
    surveys_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract geological contacts from drillhole lithology logs.

    Scans each drillhole for transitions between different lithology
    units.  At each transition, records the contact position.

    Parameters
    ----------
    drillhole_data : pd.DataFrame
        Lithology log with hole_id, from/to depths, coordinates, lithology.
    lithology_column : str
        Column containing lithology codes.
    grouping : dict, optional
        Maps {unit_name: [list of raw codes]}.  If provided, raw codes
        are mapped to modelling units before contact extraction.

    Returns
    -------
    pd.DataFrame
        Columns: hole_id, depth, X, Y, Z, unit_above, unit_below, surface_name,
        and optionally hole_dir_x/y/z when surveys_df is provided.
    """
    df = drillhole_data.copy()

    # Apply grouping if provided
    if grouping:
        code_to_unit = {}
        for unit_name, codes in grouping.items():
            for code in codes:
                code_to_unit[code] = unit_name
        df["_grouped_lith"] = df[lithology_column].map(
            lambda c: code_to_unit.get(str(c).strip(), str(c).strip())
        )
        lith_col = "_grouped_lith"
    else:
        lith_col = lithology_column

    contacts = []

    for hole_id, hole_df in df.groupby(hole_id_col):
        # Sort by depth
        hole_df = hole_df.sort_values(from_col).reset_index(drop=True)
        if len(hole_df) < 2:
            continue

        for i in range(len(hole_df) - 1):
            unit_above = str(hole_df.iloc[i][lith_col]).strip()
            unit_below = str(hole_df.iloc[i + 1][lith_col]).strip()

            if unit_above == unit_below:
                continue

            # Contact depth = bottom of upper interval = top of lower interval
            contact_depth = float(hole_df.iloc[i][to_col])

            # Interpolate 3D position at contact depth
            # Use the midpoint between the two interval endpoints
            row_above = hole_df.iloc[i]
            row_below = hole_df.iloc[i + 1]

            # Simple interpolation: use coordinates from above row
            # adjusted to contact depth (linear interpolation along drillhole)
            if to_col in row_above.index and from_col in row_below.index:
                frac_above = 1.0  # contact is at bottom of above interval
                x_above = float(row_above.get(x_col, 0.0))
                y_above = float(row_above.get(y_col, 0.0))
                z_above = float(row_above.get(z_col, 0.0))

                x_below = float(row_below.get(x_col, x_above))
                y_below = float(row_below.get(y_col, y_above))
                z_below = float(row_below.get(z_col, z_above))

                # Average the two endpoints for contact position
                cx = (x_above + x_below) / 2.0
                cy = (y_above + y_below) / 2.0
                cz = (z_above + z_below) / 2.0
            else:
                cx = float(row_above.get(x_col, 0.0))
                cy = float(row_above.get(y_col, 0.0))
                cz = float(row_above.get(z_col, 0.0))

            surface_name = f"{unit_above}_{unit_below}"

            contact_dict: Dict[str, Any] = {
                "hole_id": str(hole_id),
                "depth": contact_depth,
                "X": cx,
                "Y": cy,
                "Z": cz,
                "unit_above": unit_above,
                "unit_below": unit_below,
                "surface_name": surface_name,
            }

            # Attach drillhole direction at contact depth if surveys available
            if surveys_df is not None:
                direction = get_hole_direction_at_depth(
                    surveys_df, str(hole_id), contact_depth
                )
                contact_dict["hole_dir_x"] = direction[0]
                contact_dict["hole_dir_y"] = direction[1]
                contact_dict["hole_dir_z"] = direction[2]

            contacts.append(contact_dict)

    if not contacts:
        return pd.DataFrame(columns=[
            "hole_id", "depth", "X", "Y", "Z",
            "unit_above", "unit_below", "surface_name",
        ])

    result = pd.DataFrame(contacts)
    logger.info(
        "Extracted %d contacts across %d surfaces from %d drillholes",
        len(result),
        result["surface_name"].nunique(),
        result["hole_id"].nunique(),
    )
    return result


# ═══════════════════════════════════════════════════════════════════
# Contact Normal Estimation
# ═══════════════════════════════════════════════════════════════════

def estimate_contact_normals(
    contacts: pd.DataFrame,
    method: str = "drillhole",
    structural_data: Optional[pd.DataFrame] = None,
    x_col: str = "X",
    y_col: str = "Y",
    z_col: str = "Z",
) -> np.ndarray:
    """Estimate surface normal directions at contact points.

    Parameters
    ----------
    contacts : pd.DataFrame
        Must have X, Y, Z columns.
    method : str
        ``"drillhole"`` — assume normal is vertical [0,0,1] (minimum assumption).
        ``"local_plane"`` — fit local plane to nearby contacts using PCA.
        ``"structural"`` — use nearest structural measurement.
    structural_data : pd.DataFrame, optional
        Required for ``"structural"`` method.  Must have X, Y, Z, dip, azimuth.

    Returns
    -------
    np.ndarray, shape (N_c, 3)
        Unit normal vectors.
    """
    N = len(contacts)
    if N == 0:
        return np.empty((0, 3), dtype=np.float64)

    coords = contacts[[x_col, y_col, z_col]].values.astype(np.float64)

    if method == "drillhole":
        # Default: vertical normal (surface perpendicular to drillhole)
        normals = np.zeros((N, 3), dtype=np.float64)
        normals[:, 2] = 1.0
        return normals

    elif method == "local_plane":
        # Fit local plane to contacts on the same surface
        normals = np.zeros((N, 3), dtype=np.float64)
        normals[:, 2] = 1.0  # default

        surface_groups = contacts.groupby("surface_name")
        for surface_name, group in surface_groups:
            group_coords = group[[x_col, y_col, z_col]].values.astype(np.float64)
            idx = group.index

            if len(group_coords) < 3:
                # Not enough points for PCA, keep vertical
                continue

            # PCA: normal is the eigenvector with smallest eigenvalue
            centroid = np.mean(group_coords, axis=0)
            centered = group_coords - centroid
            cov = centered.T @ centered / (len(centered) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Smallest eigenvalue → normal direction
            plane_normal = eigenvectors[:, 0]

            # Ensure consistent orientation (point upward)
            if plane_normal[2] < 0:
                plane_normal = -plane_normal

            # Assign to all contacts on this surface
            for i, original_idx in enumerate(idx):
                pos = contacts.index.get_loc(original_idx)
                normals[pos] = plane_normal

        return normals

    elif method == "structural":
        if structural_data is None:
            raise ValueError("structural_data required for method='structural'")

        struct_coords = structural_data[["X", "Y", "Z"]].values.astype(np.float64)
        struct_normals = np.array([
            dip_azimuth_to_normal(row["dip"], row["azimuth"])
            for _, row in structural_data.iterrows()
        ], dtype=np.float64)

        from scipy.spatial import cKDTree
        tree = cKDTree(struct_coords)

        normals = np.zeros((N, 3), dtype=np.float64)
        for i in range(N):
            _, idx = tree.query(coords[i], k=1)
            normals[i] = struct_normals[idx]

        return normals

    elif method == "survey":
        # Use drillhole direction vectors stored in the contacts DataFrame
        # (added by extract_contacts_from_lithology when surveys_df is provided)
        has_dirs = all(
            c in contacts.columns for c in ["hole_dir_x", "hole_dir_y", "hole_dir_z"]
        )
        if not has_dirs:
            logger.warning(
                "estimate_contact_normals(method='survey'): no hole_dir_* columns; "
                "falling back to local_plane"
            )
            return estimate_contact_normals(
                contacts, method="local_plane",
                x_col=x_col, y_col=y_col, z_col=z_col,
            )

        normals = np.zeros((N, 3), dtype=np.float64)
        normals[:, 2] = 1.0  # upward default

        if "surface_name" not in contacts.columns:
            # No grouping — treat all contacts as one surface
            group_coords = coords
            group_dirs = contacts[["hole_dir_x", "hole_dir_y", "hole_dir_z"]].values.astype(np.float64)
            n = estimate_surface_normal_from_contacts_and_surveys(group_coords, group_dirs)
            normals[:] = n
            return normals

        for _sname, group in contacts.groupby("surface_name"):
            group_coords = group[[x_col, y_col, z_col]].values.astype(np.float64)
            group_dirs = group[["hole_dir_x", "hole_dir_y", "hole_dir_z"]].values.astype(np.float64)
            idx = group.index

            n = estimate_surface_normal_from_contacts_and_surveys(group_coords, group_dirs)

            for original_idx in idx:
                pos = contacts.index.get_loc(original_idx)
                normals[pos] = n

        logger.info(
            "Survey-based normals estimated for %d contacts across %d surfaces",
            N, contacts["surface_name"].nunique(),
        )
        return normals

    else:
        raise ValueError(f"Unknown normal estimation method: '{method}'")
