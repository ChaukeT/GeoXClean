"""
GeologicalModelEngine - Industry-Grade Implicit Geological Modelling.

Complete rewrite implementing the Leapfrog/Vulcan/GemPy workflow:
1. Extract lithological contacts from drillholes (where formations change)
2. Compute orientations from contact geometry (strike/dip from spatial patterns)
3. Assign stratigraphic scalar values (cumulative thickness-proportional)
4. Build LoopStructural model with proper constraints
5. Extract surfaces, solids, and unified meshes

Key differences from previous ChronosEngine:
- Contacts are EXTRACTED from drillhole lithology logs, not passed raw
- Orientations are COMPUTED from contact geometry, not synthetic (0,0,1)
- Scalar values are PROPORTIONAL to true stratigraphic thickness
- Structure detection feeds back into the model
- Full JORC/SAMREC audit trail

References:
- Grose et al. (2021) "LoopStructural 1.0: Time aware geological modelling"
- Lajaunie et al. (1997) "Foliation fields and 3D cartography in geology"
- Cowan et al. (2003) "Practical Implicit Geological Modelling"

GeoX Invariant Compliance:
- All operations logged for audit trail
- Provenance metadata for every output
- Deterministic results for same input
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ─── LoopStructural availability ────────────────────────────────────────────
try:
    from LoopStructural import GeologicalModel
    LS_AVAILABLE = True
except ImportError:
    LS_AVAILABLE = False
    GeologicalModel = None
    logger.warning("LoopStructural not available – geological modelling disabled")

# ─── PyVista availability ───────────────────────────────────────────────────
try:
    import pyvista as pv
    PV_AVAILABLE = True
except ImportError:
    PV_AVAILABLE = False
    pv = None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrillholeContact:
    """A lithological contact extracted from a drillhole."""
    hole_id: str
    x: float
    y: float
    z: float               # Z at the contact point (midpoint of transition)
    formation_above: str   # Younger formation (shallower)
    formation_below: str   # Older formation (deeper)
    contact_name: str      # e.g. "FormA|FormB"
    scalar_value: float    # Assigned stratigraphic scalar
    depth_from: float      # Depth of the contact interval top
    depth_to: float        # Depth of the contact interval bottom
    confidence: float = 1.0  # 1.0 = logged contact, 0.5 = inferred


@dataclass
class ComputedOrientation:
    """An orientation (strike/dip) computed from contact geometry."""
    x: float
    y: float
    z: float
    gx: float          # Gradient X component (normal to contact surface)
    gy: float          # Gradient Y component
    gz: float          # Gradient Z component
    dip: float         # Dip angle in degrees
    dip_direction: float  # Dip direction (azimuth) in degrees
    contact_name: str
    method: str        # 'plane_fit', 'tangent', 'user_supplied'
    confidence: float = 1.0
    n_points_used: int = 0


@dataclass
class DetectedStructure:
    """A structural feature auto-detected from drillhole patterns."""
    structure_type: str   # 'fault', 'fold', 'unconformity', 'intrusion'
    name: str
    center: np.ndarray    # (3,) center point
    confidence: float     # 0-1
    evidence: str         # Human-readable description of evidence
    parameters: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = False  # User must accept before use in model


@dataclass
class StratigraphicColumn:
    """Ordered stratigraphic column with scalar value assignments."""
    formations: List[str]           # Oldest first
    scalar_values: Dict[str, float]  # Formation → scalar value
    thicknesses: Dict[str, float]    # Formation → avg thickness (m)
    contact_names: List[str]         # Ordered contact boundary names
    formation_colors: Dict[str, str] = field(default_factory=dict)


@dataclass
class ModelBuildLog:
    """Complete provenance record for JORC/SAMREC auditing."""
    timestamp: datetime = field(default_factory=datetime.now)
    engine: str = "LoopStructural"
    engine_version: str = "1.6+"
    parameters: Dict[str, Any] = field(default_factory=dict)
    contact_extraction: Dict[str, Any] = field(default_factory=dict)
    orientation_computation: Dict[str, Any] = field(default_factory=dict)
    structure_detection: Dict[str, Any] = field(default_factory=dict)
    coordinate_transform: Dict[str, Any] = field(default_factory=dict)
    event_stack: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "engine": self.engine,
            "engine_version": self.engine_version,
            "parameters": self.parameters,
            "contact_extraction": self.contact_extraction,
            "orientation_computation": self.orientation_computation,
            "structure_detection": self.structure_detection,
            "coordinate_transform": self.coordinate_transform,
            "event_stack": self.event_stack,
            "warnings": self.warnings,
            "data_hash": self.data_hash,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONTACT EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class DrillholeContactExtractor:
    """
    Extract lithological contacts from drillhole data.

    This is the FIRST and MOST CRITICAL step in geological modelling.
    A "contact" is the exact 3D point where one formation transitions
    to another along a drillhole. These contacts are the primary training
    data for the implicit model.

    Industry standard (Leapfrog/Vulcan):
    - Walk each drillhole from top to bottom
    - Where formation changes, record the contact point
    - Contact Z = midpoint of the transition interval
    - Each contact gets a scalar value from the stratigraphic column
    """

    @staticmethod
    def extract_contacts(
        lithology_df: pd.DataFrame,
        stratigraphy: List[str],
        scalar_values: Optional[Dict[str, float]] = None,
        hole_id_col: str = 'hole_id',
        formation_col: str = 'formation',
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: str = 'Z',
        from_col: Optional[str] = None,
        to_col: Optional[str] = None,
    ) -> Tuple[List[DrillholeContact], Dict[str, Any]]:
        """
        Extract contact points from drillhole lithology logs.

        For each drillhole, walks the intervals and identifies where
        the formation changes. The contact point is placed at the
        midpoint of the transition.

        Args:
            lithology_df: DataFrame with drillhole lithology data.
                Must contain hole_id, formation, X, Y, Z columns.
                Optionally 'from' and 'to' depth columns.
            stratigraphy: List of formation names, OLDEST first.
            scalar_values: Optional pre-computed scalar value mapping.
                If None, computed from cumulative thickness.
            hole_id_col: Column name for drillhole ID.
            formation_col: Column name for formation/lithology.
            x_col, y_col, z_col: Coordinate column names.
            from_col, to_col: Optional interval depth columns.

        Returns:
            Tuple of (contacts_list, extraction_stats)
        """
        if lithology_df is None or len(lithology_df) == 0:
            return [], {"error": "Empty DataFrame"}

        # Auto-detect column names
        hole_id_col = DrillholeContactExtractor._find_column(
            lithology_df, [hole_id_col, 'HOLEID', 'HoleID', 'Hole_ID', 'BHID', 'hole', 'drillhole_id']
        )
        formation_col = DrillholeContactExtractor._find_column(
            lithology_df, [formation_col, 'Formation', 'FORMATION', 'lith', 'lithology', 'LITHOLOGY', 'unit', 'rock_type']
        )

        if hole_id_col is None:
            return [], {"error": "Cannot find hole_id column"}
        if formation_col is None:
            return [], {"error": "Cannot find formation column"}

        # Detect from/to columns if not specified
        if from_col is None:
            from_col = DrillholeContactExtractor._find_column(
                lithology_df, ['from', 'From', 'FROM', 'depth_from', 'DEPTH_FROM', 'from_m']
            )
        if to_col is None:
            to_col = DrillholeContactExtractor._find_column(
                lithology_df, ['to', 'To', 'TO', 'depth_to', 'DEPTH_TO', 'to_m']
            )

        # Build formation → index mapping
        formation_idx = {f: i for i, f in enumerate(stratigraphy)}

        # Compute scalar values if not provided
        if scalar_values is None:
            scalar_values = DrillholeContactExtractor._compute_scalar_values(
                lithology_df, stratigraphy, hole_id_col, formation_col, z_col, from_col, to_col
            )

        contacts: List[DrillholeContact] = []
        stats = {
            "holes_processed": 0,
            "contacts_extracted": 0,
            "formations_found": set(),
            "contacts_per_hole": [],
            "skipped_unknown": 0,
        }

        # Process each drillhole
        for hole_id, hole_data in lithology_df.groupby(hole_id_col):
            stats["holes_processed"] += 1

            # Sort by depth: shallowest first (highest Z or lowest from-depth)
            if from_col and from_col in hole_data.columns:
                hole_sorted = hole_data.sort_values(from_col, ascending=True)
            else:
                hole_sorted = hole_data.sort_values(z_col, ascending=False)

            hole_contacts = 0
            prev_formation = None
            prev_row = None

            for idx, row in hole_sorted.iterrows():
                current_formation = str(row[formation_col]).strip() if pd.notna(row[formation_col]) else None

                if current_formation is None:
                    continue

                stats["formations_found"].add(current_formation)

                if prev_formation is not None and current_formation != prev_formation:
                    # ═══════════════════════════════════════════════════
                    # CONTACT FOUND: formation changed along drillhole
                    # ═══════════════════════════════════════════════════

                    # Determine which is above (younger) and below (older)
                    # In downhole order: prev is shallower (younger), current is deeper (older)
                    formation_above = prev_formation
                    formation_below = current_formation

                    # Skip contacts with unknown formations (not in stratigraphy)
                    if formation_above not in formation_idx or formation_below not in formation_idx:
                        stats["skipped_unknown"] += 1
                        prev_formation = current_formation
                        prev_row = row
                        continue

                    # Compute contact position
                    if from_col and to_col and from_col in hole_data.columns:
                        # Use interval boundaries for precise Z
                        prev_to = float(prev_row[to_col]) if pd.notna(prev_row[to_col]) else None
                        curr_from = float(row[from_col]) if pd.notna(row[from_col]) else None

                        if prev_to is not None and curr_from is not None:
                            contact_depth = (prev_to + curr_from) / 2.0
                        else:
                            contact_depth = None
                    else:
                        contact_depth = None

                    # Get X, Y, Z of the contact
                    if contact_depth is not None and z_col in row.index:
                        # Interpolate X, Y based on relative position
                        z_above = float(prev_row[z_col])
                        z_below = float(row[z_col])
                        if abs(z_above - z_below) > 1e-6:
                            # Linear interpolation between the two sample points
                            frac = 0.5  # midpoint between contacts
                            cx = float(prev_row[x_col]) * (1 - frac) + float(row[x_col]) * frac
                            cy = float(prev_row[y_col]) * (1 - frac) + float(row[y_col]) * frac
                            cz = z_above * (1 - frac) + z_below * frac
                        else:
                            cx = float(row[x_col])
                            cy = float(row[y_col])
                            cz = float(row[z_col])
                    else:
                        # Use midpoint between adjacent samples
                        cx = (float(prev_row[x_col]) + float(row[x_col])) / 2.0
                        cy = (float(prev_row[y_col]) + float(row[y_col])) / 2.0
                        cz = (float(prev_row[z_col]) + float(row[z_col])) / 2.0

                    # Contact name and scalar value
                    # The contact scalar = boundary between the two formations
                    # Use the midpoint of their scalar values
                    val_above = scalar_values.get(formation_above, 0.0)
                    val_below = scalar_values.get(formation_below, 0.0)
                    contact_scalar = (val_above + val_below) / 2.0

                    contact_name = f"{formation_below}|{formation_above}"

                    contact = DrillholeContact(
                        hole_id=str(hole_id),
                        x=cx, y=cy, z=cz,
                        formation_above=formation_above,
                        formation_below=formation_below,
                        contact_name=contact_name,
                        scalar_value=contact_scalar,
                        depth_from=float(prev_row.get(to_col, cz)) if to_col and to_col in prev_row.index else cz,
                        depth_to=float(row.get(from_col, cz)) if from_col and from_col in row.index else cz,
                        confidence=1.0,
                    )
                    contacts.append(contact)
                    hole_contacts += 1

                prev_formation = current_formation
                prev_row = row

            stats["contacts_per_hole"].append(hole_contacts)

        stats["contacts_extracted"] = len(contacts)
        stats["formations_found"] = list(stats["formations_found"])
        stats["avg_contacts_per_hole"] = (
            np.mean(stats["contacts_per_hole"]) if stats["contacts_per_hole"] else 0
        )

        logger.info(
            f"Contact extraction: {stats['contacts_extracted']} contacts from "
            f"{stats['holes_processed']} drillholes "
            f"(avg {stats['avg_contacts_per_hole']:.1f} per hole)"
        )

        return contacts, stats

    @staticmethod
    def _compute_scalar_values(
        df: pd.DataFrame,
        stratigraphy: List[str],
        hole_id_col: str,
        formation_col: str,
        z_col: str,
        from_col: Optional[str],
        to_col: Optional[str],
    ) -> Dict[str, float]:
        """
        Compute scalar values proportional to cumulative thickness.

        Industry standard: scalar value represents cumulative stratigraphic
        position. Thin units get small scalar ranges, thick units get large.
        This ensures the implicit function correctly represents real geology.
        """
        thicknesses = {}

        for hole_id, hole_data in df.groupby(hole_id_col):
            if from_col and to_col and from_col in hole_data.columns and to_col in hole_data.columns:
                for _, row in hole_data.iterrows():
                    fm = str(row[formation_col]).strip() if pd.notna(row[formation_col]) else None
                    if fm and fm in stratigraphy:
                        try:
                            thick = abs(float(row[to_col]) - float(row[from_col]))
                            if thick > 0:
                                thicknesses.setdefault(fm, []).append(thick)
                        except (ValueError, TypeError):
                            pass
            else:
                # Estimate from Z-spacing between consecutive same-formation points
                hole_sorted = hole_data.sort_values(z_col, ascending=False)
                prev_fm = None
                prev_z = None
                for _, row in hole_sorted.iterrows():
                    fm = str(row[formation_col]).strip() if pd.notna(row[formation_col]) else None
                    z = float(row[z_col]) if pd.notna(row[z_col]) else None
                    if fm and z is not None and prev_fm == fm and prev_z is not None:
                        thick = abs(prev_z - z)
                        if thick > 0:
                            thicknesses.setdefault(fm, []).append(thick)
                    prev_fm = fm
                    prev_z = z

        # Compute average thicknesses
        avg_thicknesses = {}
        for fm in stratigraphy:
            if fm in thicknesses and len(thicknesses[fm]) > 0:
                avg_thicknesses[fm] = float(np.median(thicknesses[fm]))
            else:
                avg_thicknesses[fm] = 1.0  # Default for unknown units

        # Compute cumulative scalar values
        # Oldest formation starts at 0, each subsequent adds its thickness
        scalar_values = {}
        cumulative = 0.0
        for fm in stratigraphy:
            scalar_values[fm] = cumulative
            cumulative += avg_thicknesses[fm]

        # Normalize to [0, total_thickness] range
        # (LoopStructural works well with real-world thickness values)
        total = cumulative if cumulative > 0 else 1.0

        logger.info(f"Scalar values (cumulative thickness): {scalar_values}")
        logger.info(f"Total stratigraphic thickness: {total:.1f}m")

        return scalar_values

    @staticmethod
    def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find the first matching column name from a list of candidates."""
        for col in candidates:
            if col in df.columns:
                return col
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# ORIENTATION CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class OrientationCalculator:
    """
    Compute orientations (strike/dip → gradient normals) from contact geometry.

    This is the KEY step that makes geological modelling actually geological.
    Without real orientations, the model just interpolates between points
    with no understanding of geological structure.

    Methods:
    1. Plane fitting: Fit a plane to nearby contacts on the same boundary.
       The plane normal gives the gradient direction (strike/dip).
    2. Drillhole tangent: Use the drillhole trajectory at the contact
       as a proxy for the dip direction (perpendicular to drillhole).
    3. Three-point: Use three nearby contacts on the same surface to
       determine the plane orientation.
    """

    @staticmethod
    def compute_orientations(
        contacts: List[DrillholeContact],
        method: str = 'plane_fit',
        search_radius: Optional[float] = None,
        min_contacts_for_fit: int = 3,
    ) -> Tuple[List[ComputedOrientation], Dict[str, Any]]:
        """
        Compute orientations from contact spatial patterns.

        For each unique contact surface, finds clusters of nearby contacts
        and fits planes to determine the local orientation.

        Args:
            contacts: List of extracted drillhole contacts.
            method: Orientation computation method.
                'plane_fit' - PCA plane fitting to nearby contacts (default)
                'three_point' - Three-point problems for nearby triplets
                'default_horizontal' - Fallback: assume horizontal (gz=1)
            search_radius: Search radius for finding nearby contacts.
                If None, auto-computed from contact spacing.
            min_contacts_for_fit: Minimum contacts needed for plane fitting.

        Returns:
            Tuple of (orientations_list, computation_stats)
        """
        if not contacts:
            return [], {"error": "No contacts provided"}

        # Group contacts by contact boundary name
        contact_groups: Dict[str, List[DrillholeContact]] = {}
        for c in contacts:
            contact_groups.setdefault(c.contact_name, []).append(c)

        # Auto-compute search radius if not specified
        if search_radius is None:
            all_coords = np.array([[c.x, c.y, c.z] for c in contacts])
            if len(all_coords) > 1:
                # Use ~2x the average nearest-neighbor distance
                tree = cKDTree(all_coords)
                dists, _ = tree.query(all_coords, k=min(4, len(all_coords)))
                if dists.shape[1] > 1:
                    avg_nn = np.mean(dists[:, 1])
                    search_radius = avg_nn * 3.0
                else:
                    search_radius = 500.0  # Default fallback
            else:
                search_radius = 500.0

        orientations: List[ComputedOrientation] = []
        stats = {
            "total_contacts": len(contacts),
            "unique_boundaries": len(contact_groups),
            "orientations_computed": 0,
            "method": method,
            "search_radius": search_radius,
            "boundaries_processed": {},
        }

        for boundary_name, boundary_contacts in contact_groups.items():
            coords = np.array([[c.x, c.y, c.z] for c in boundary_contacts])
            n_pts = len(coords)

            if n_pts < min_contacts_for_fit:
                # Not enough contacts for plane fitting → use default
                for c in boundary_contacts:
                    orientations.append(ComputedOrientation(
                        x=c.x, y=c.y, z=c.z,
                        gx=0.0, gy=0.0, gz=1.0,  # Default horizontal
                        dip=0.0, dip_direction=0.0,
                        contact_name=c.contact_name,
                        method='default_horizontal',
                        confidence=0.3,
                        n_points_used=1,
                    ))
                stats["boundaries_processed"][boundary_name] = {
                    "n_contacts": n_pts,
                    "method_used": "default_horizontal",
                    "reason": f"Too few contacts ({n_pts} < {min_contacts_for_fit})",
                }
                continue

            if method == 'plane_fit':
                boundary_orientations = OrientationCalculator._plane_fit_orientations(
                    boundary_contacts, coords, search_radius, min_contacts_for_fit
                )
            elif method == 'three_point':
                boundary_orientations = OrientationCalculator._three_point_orientations(
                    boundary_contacts, coords
                )
            else:
                # Default horizontal for all
                boundary_orientations = [
                    ComputedOrientation(
                        x=c.x, y=c.y, z=c.z,
                        gx=0.0, gy=0.0, gz=1.0,
                        dip=0.0, dip_direction=0.0,
                        contact_name=c.contact_name,
                        method='default_horizontal',
                        confidence=0.3, n_points_used=1,
                    ) for c in boundary_contacts
                ]

            orientations.extend(boundary_orientations)
            stats["boundaries_processed"][boundary_name] = {
                "n_contacts": n_pts,
                "n_orientations": len(boundary_orientations),
                "method_used": method,
            }

        stats["orientations_computed"] = len(orientations)

        logger.info(
            f"Orientation computation: {len(orientations)} orientations from "
            f"{len(contact_groups)} boundaries using '{method}'"
        )

        return orientations, stats

    @staticmethod
    def _plane_fit_orientations(
        contacts: List[DrillholeContact],
        coords: np.ndarray,
        search_radius: float,
        min_pts: int,
    ) -> List[ComputedOrientation]:
        """Fit planes to local neighborhoods of contacts using PCA."""
        from sklearn.decomposition import PCA

        tree = cKDTree(coords)
        orientations = []

        for i, contact in enumerate(contacts):
            # Find neighbors within search radius
            pt = coords[i]
            neighbor_idxs = tree.query_ball_point(pt, search_radius)

            if len(neighbor_idxs) < min_pts:
                # Fall back to global fit for this boundary
                neighbor_idxs = list(range(len(coords)))

            if len(neighbor_idxs) < min_pts:
                # Still too few → horizontal default
                orientations.append(ComputedOrientation(
                    x=contact.x, y=contact.y, z=contact.z,
                    gx=0.0, gy=0.0, gz=1.0,
                    dip=0.0, dip_direction=0.0,
                    contact_name=contact.contact_name,
                    method='default_horizontal',
                    confidence=0.3, n_points_used=1,
                ))
                continue

            # Fit plane using PCA
            local_coords = coords[neighbor_idxs]
            pca = PCA(n_components=3)
            pca.fit(local_coords)

            # Normal vector = direction of least variance (3rd component)
            normal = pca.components_[2].copy()

            # Ensure normal points upward (towards younger stratigraphy)
            if normal[2] < 0:
                normal = -normal

            # Normalize
            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-10:
                normal = normal / norm_mag
            else:
                normal = np.array([0.0, 0.0, 1.0])

            # Confidence from planarity (how well the points define a plane)
            ev = pca.explained_variance_ratio_
            if ev[1] > 1e-10:
                planarity = 1.0 - (ev[2] / ev[1])
            else:
                planarity = 0.0
            confidence = np.clip(planarity, 0.0, 1.0)

            # Convert to dip / dip-direction
            dip, dip_dir = OrientationCalculator._normal_to_dip_dipdir(normal)

            orientations.append(ComputedOrientation(
                x=contact.x, y=contact.y, z=contact.z,
                gx=normal[0], gy=normal[1], gz=normal[2],
                dip=dip, dip_direction=dip_dir,
                contact_name=contact.contact_name,
                method='plane_fit',
                confidence=confidence,
                n_points_used=len(neighbor_idxs),
            ))

        return orientations

    @staticmethod
    def _three_point_orientations(
        contacts: List[DrillholeContact],
        coords: np.ndarray,
    ) -> List[ComputedOrientation]:
        """Compute orientations using three-point problems."""
        tree = cKDTree(coords)
        orientations = []

        for i, contact in enumerate(contacts):
            pt = coords[i]
            # Find 2 nearest neighbors (gives 3 points total)
            k = min(3, len(coords))
            dists, idxs = tree.query(pt, k=k)

            if k < 3 or len(idxs) < 3:
                orientations.append(ComputedOrientation(
                    x=contact.x, y=contact.y, z=contact.z,
                    gx=0.0, gy=0.0, gz=1.0,
                    dip=0.0, dip_direction=0.0,
                    contact_name=contact.contact_name,
                    method='default_horizontal',
                    confidence=0.3, n_points_used=1,
                ))
                continue

            # Three-point plane fit
            p1 = coords[idxs[0]]
            p2 = coords[idxs[1]]
            p3 = coords[idxs[2]]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)

            norm_mag = np.linalg.norm(normal)
            if norm_mag > 1e-10:
                normal = normal / norm_mag
                if normal[2] < 0:
                    normal = -normal
            else:
                normal = np.array([0.0, 0.0, 1.0])

            dip, dip_dir = OrientationCalculator._normal_to_dip_dipdir(normal)

            orientations.append(ComputedOrientation(
                x=contact.x, y=contact.y, z=contact.z,
                gx=normal[0], gy=normal[1], gz=normal[2],
                dip=dip, dip_direction=dip_dir,
                contact_name=contact.contact_name,
                method='three_point',
                confidence=0.7,
                n_points_used=3,
            ))

        return orientations

    @staticmethod
    def _normal_to_dip_dipdir(n: np.ndarray) -> Tuple[float, float]:
        """Convert normal vector to dip/dip-direction (degrees)."""
        n = n.copy()
        if n[2] < 0:
            n = -n
        norm_mag = np.linalg.norm(n)
        if norm_mag > 1e-10:
            n = n / norm_mag

        dip = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
        dip_dir = np.degrees(np.arctan2(n[0], n[1])) % 360

        return round(dip, 1), round(dip_dir, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class StructureDetector:
    """
    Automatic detection of geological structures from drillhole patterns.

    Detects:
    1. FAULTS: Offset contacts (same boundary at different Z in nearby holes)
    2. FOLDS: Systematic dip changes across contacts
    3. REPEATED SECTIONS: Same unit appearing twice in a drillhole (faulting)
    4. MISSING UNITS: Expected formation absent (erosion, faulting, or pinch-out)
    5. THICKNESS ANOMALIES: Rapid thickness changes indicating proximity to faults

    All detections are SUGGESTIONS that require geologist acceptance.
    """

    @staticmethod
    def detect_structures(
        contacts: List[DrillholeContact],
        orientations: List[ComputedOrientation],
        stratigraphy: List[str],
        lithology_df: pd.DataFrame,
        hole_id_col: str = 'hole_id',
        formation_col: str = 'formation',
        z_col: str = 'Z',
        offset_threshold_m: float = 10.0,
        thickness_anomaly_factor: float = 2.0,
    ) -> Tuple[List[DetectedStructure], Dict[str, Any]]:
        """
        Run all structure detection algorithms.

        Args:
            contacts: Extracted drillhole contacts.
            orientations: Computed orientations.
            stratigraphy: Ordered formation names (oldest first).
            lithology_df: Raw drillhole lithology data.
            offset_threshold_m: Min Z-offset to flag as potential fault.
            thickness_anomaly_factor: Factor of std dev for anomaly detection.

        Returns:
            Tuple of (detected_structures, detection_stats)
        """
        structures = []
        stats = {
            "faults_detected": 0,
            "folds_detected": 0,
            "repeated_sections": 0,
            "missing_units": 0,
            "thickness_anomalies": 0,
        }

        # 1. Detect FAULTS from contact offsets
        fault_structures = StructureDetector._detect_faults_from_offsets(
            contacts, offset_threshold_m
        )
        structures.extend(fault_structures)
        stats["faults_detected"] = len(fault_structures)

        # 2. Detect REPEATED SECTIONS (strong fault indicator)
        repeated = StructureDetector._detect_repeated_sections(
            lithology_df, stratigraphy, hole_id_col, formation_col, z_col
        )
        structures.extend(repeated)
        stats["repeated_sections"] = len(repeated)

        # 3. Detect MISSING UNITS (possible erosion or faulting)
        missing = StructureDetector._detect_missing_units(
            lithology_df, stratigraphy, hole_id_col, formation_col
        )
        structures.extend(missing)
        stats["missing_units"] = len(missing)

        # 4. Detect FOLDS from orientation patterns
        if orientations:
            fold_structures = StructureDetector._detect_folds_from_orientations(
                orientations
            )
            structures.extend(fold_structures)
            stats["folds_detected"] = len(fold_structures)

        # 5. Detect THICKNESS ANOMALIES
        thickness_anomalies = StructureDetector._detect_thickness_anomalies(
            contacts, stratigraphy, thickness_anomaly_factor
        )
        structures.extend(thickness_anomalies)
        stats["thickness_anomalies"] = len(thickness_anomalies)

        logger.info(
            f"Structure detection: {len(structures)} total "
            f"({stats['faults_detected']} faults, {stats['folds_detected']} folds, "
            f"{stats['repeated_sections']} repeats, {stats['missing_units']} missing, "
            f"{stats['thickness_anomalies']} thickness anomalies)"
        )

        return structures, stats

    @staticmethod
    def _detect_faults_from_offsets(
        contacts: List[DrillholeContact],
        threshold_m: float,
    ) -> List[DetectedStructure]:
        """Detect faults by looking for offset contacts between nearby drillholes."""
        from sklearn.cluster import DBSCAN

        # Group contacts by boundary name
        by_boundary: Dict[str, List[DrillholeContact]] = {}
        for c in contacts:
            by_boundary.setdefault(c.contact_name, []).append(c)

        structures = []

        for boundary_name, boundary_contacts in by_boundary.items():
            if len(boundary_contacts) < 3:
                continue

            coords_xy = np.array([[c.x, c.y] for c in boundary_contacts])
            z_values = np.array([c.z for c in boundary_contacts])

            # For each contact, find nearby contacts (same boundary)
            # and check for Z-offset
            tree = cKDTree(coords_xy)

            # Use average XY spacing to define "nearby"
            if len(coords_xy) > 1:
                dists, _ = tree.query(coords_xy, k=min(3, len(coords_xy)))
                avg_spacing = np.mean(dists[:, 1]) if dists.shape[1] > 1 else 500.0
            else:
                avg_spacing = 500.0

            search_r = avg_spacing * 2.0

            offsets = []
            for i in range(len(boundary_contacts)):
                neighbors = tree.query_ball_point(coords_xy[i], search_r)
                neighbors = [j for j in neighbors if j != i]

                for j in neighbors:
                    offset = abs(z_values[i] - z_values[j])
                    if offset > threshold_m:
                        midx = (coords_xy[i][0] + coords_xy[j][0]) / 2.0
                        midy = (coords_xy[i][1] + coords_xy[j][1]) / 2.0
                        midz = (z_values[i] + z_values[j]) / 2.0
                        offsets.append({
                            "center": np.array([midx, midy, midz]),
                            "offset_m": offset,
                            "boundary": boundary_name,
                            "hole_i": boundary_contacts[i].hole_id,
                            "hole_j": boundary_contacts[j].hole_id,
                        })

            # Cluster offset detections to avoid duplicates
            if offsets:
                offset_coords = np.array([o["center"] for o in offsets])
                if len(offset_coords) >= 2:
                    clustering = DBSCAN(eps=avg_spacing, min_samples=1).fit(offset_coords)
                    unique_labels = set(clustering.labels_)
                    unique_labels.discard(-1)

                    for label in unique_labels:
                        mask = clustering.labels_ == label
                        cluster_offsets = [o for o, m in zip(offsets, mask) if m]

                        avg_offset = np.mean([o["offset_m"] for o in cluster_offsets])
                        center = np.mean([o["center"] for o in cluster_offsets], axis=0)
                        confidence = min(1.0, avg_offset / (threshold_m * 3.0))

                        holes_involved = set()
                        for o in cluster_offsets:
                            holes_involved.add(o["hole_i"])
                            holes_involved.add(o["hole_j"])

                        structures.append(DetectedStructure(
                            structure_type='fault',
                            name=f"AutoFault_{boundary_name}_{len(structures)+1}",
                            center=center,
                            confidence=confidence,
                            evidence=(
                                f"Contact '{boundary_name}' offset by {avg_offset:.1f}m "
                                f"between {len(holes_involved)} drillholes"
                            ),
                            parameters={
                                "offset_m": float(avg_offset),
                                "boundary": boundary_name,
                                "n_detections": len(cluster_offsets),
                                "holes_involved": list(holes_involved),
                            },
                        ))
                else:
                    o = offsets[0]
                    structures.append(DetectedStructure(
                        structure_type='fault',
                        name=f"AutoFault_{boundary_name}_1",
                        center=o["center"],
                        confidence=min(1.0, o["offset_m"] / (threshold_m * 3.0)),
                        evidence=(
                            f"Contact '{boundary_name}' offset by {o['offset_m']:.1f}m "
                            f"between holes {o['hole_i']} and {o['hole_j']}"
                        ),
                        parameters={
                            "offset_m": o["offset_m"],
                            "boundary": boundary_name,
                        },
                    ))

        return structures

    @staticmethod
    def _detect_repeated_sections(
        df: pd.DataFrame,
        stratigraphy: List[str],
        hole_id_col: str,
        formation_col: str,
        z_col: str,
    ) -> List[DetectedStructure]:
        """Detect repeated stratigraphic sections (strong fault indicator)."""
        structures = []

        if hole_id_col not in df.columns or formation_col not in df.columns:
            return structures

        for hole_id, hole_data in df.groupby(hole_id_col):
            hole_sorted = hole_data.sort_values(z_col, ascending=False)
            formations_sequence = hole_sorted[formation_col].dropna().tolist()

            # Look for a formation that appears, disappears, then reappears
            seen_last = {}
            for i, fm in enumerate(formations_sequence):
                fm_str = str(fm).strip()
                if fm_str in seen_last:
                    gap = i - seen_last[fm_str]
                    if gap > 1:
                        # Formation reappears after other formations
                        z_vals = hole_sorted[z_col].values
                        midz = float(z_vals[i]) if i < len(z_vals) else 0.0
                        midx = float(hole_sorted.iloc[i].get('X', 0))
                        midy = float(hole_sorted.iloc[i].get('Y', 0))

                        structures.append(DetectedStructure(
                            structure_type='fault',
                            name=f"RepeatedSection_{hole_id}_{fm_str}",
                            center=np.array([midx, midy, midz]),
                            confidence=0.8,
                            evidence=(
                                f"Formation '{fm_str}' repeated in hole {hole_id} "
                                f"(gap of {gap} formations between occurrences). "
                                f"This strongly suggests faulting."
                            ),
                            parameters={
                                "hole_id": str(hole_id),
                                "formation": fm_str,
                                "gap_formations": gap,
                            },
                        ))
                seen_last[fm_str] = i

        return structures

    @staticmethod
    def _detect_missing_units(
        df: pd.DataFrame,
        stratigraphy: List[str],
        hole_id_col: str,
        formation_col: str,
    ) -> List[DetectedStructure]:
        """Detect missing units in drillholes (possible erosion or faulting)."""
        structures = []

        if hole_id_col not in df.columns or formation_col not in df.columns:
            return structures

        strat_set = set(stratigraphy)

        for hole_id, hole_data in df.groupby(hole_id_col):
            hole_formations = set(
                str(f).strip() for f in hole_data[formation_col].dropna().unique()
            ) & strat_set

            if len(hole_formations) < 2:
                continue

            # Check which units in the stratigraphic range are missing
            hole_indices = sorted(
                stratigraphy.index(f) for f in hole_formations
            )
            min_idx = hole_indices[0]
            max_idx = hole_indices[-1]

            expected = set(stratigraphy[min_idx:max_idx + 1])
            missing = expected - hole_formations

            for fm in missing:
                x = float(hole_data.iloc[0].get('X', 0))
                y = float(hole_data.iloc[0].get('Y', 0))
                z = float(hole_data.iloc[0].get('Z', 0))

                structures.append(DetectedStructure(
                    structure_type='unconformity',
                    name=f"MissingUnit_{hole_id}_{fm}",
                    center=np.array([x, y, z]),
                    confidence=0.5,
                    evidence=(
                        f"Unit '{fm}' missing in hole {hole_id}. "
                        f"Expected between {stratigraphy[min_idx]} and "
                        f"{stratigraphy[max_idx]}. May indicate erosion, "
                        f"faulting, or lateral pinch-out."
                    ),
                    parameters={
                        "hole_id": str(hole_id),
                        "missing_formation": fm,
                        "expected_range": [stratigraphy[min_idx], stratigraphy[max_idx]],
                    },
                ))

        return structures

    @staticmethod
    def _detect_folds_from_orientations(
        orientations: List[ComputedOrientation],
    ) -> List[DetectedStructure]:
        """Detect folds from systematic dip changes across space."""
        structures = []

        if len(orientations) < 5:
            return structures

        # Group by contact boundary
        by_boundary: Dict[str, List[ComputedOrientation]] = {}
        for o in orientations:
            by_boundary.setdefault(o.contact_name, []).append(o)

        for boundary_name, boundary_orients in by_boundary.items():
            if len(boundary_orients) < 4:
                continue

            coords = np.array([[o.x, o.y] for o in boundary_orients])
            dips = np.array([o.dip for o in boundary_orients])
            dip_dirs = np.array([o.dip_direction for o in boundary_orients])

            # Check for significant dip variation (indicator of folding)
            dip_std = np.std(dips)
            mean_dip = np.mean(dips)

            # Check for dip direction reversal (strong fold indicator)
            # Convert to vectors for proper angular comparison
            dip_vectors_x = np.sin(np.radians(dip_dirs))
            dip_vectors_y = np.cos(np.radians(dip_dirs))

            # Check directional variance
            mean_vx = np.mean(dip_vectors_x)
            mean_vy = np.mean(dip_vectors_y)
            resultant_length = np.sqrt(mean_vx**2 + mean_vy**2)

            # Low resultant length = high directional dispersion = possible fold
            if dip_std > 15.0 and resultant_length < 0.7 and mean_dip > 10.0:
                center = np.array([
                    np.mean(coords[:, 0]),
                    np.mean(coords[:, 1]),
                    np.mean([o.z for o in boundary_orients]),
                ])

                confidence = min(1.0, dip_std / 45.0) * (1.0 - resultant_length)

                structures.append(DetectedStructure(
                    structure_type='fold',
                    name=f"AutoFold_{boundary_name}",
                    center=center,
                    confidence=confidence,
                    evidence=(
                        f"Contact '{boundary_name}' shows dip variation of "
                        f"{dip_std:.1f}° (mean {mean_dip:.1f}°) with directional "
                        f"dispersion (R={resultant_length:.2f}). Suggests folding."
                    ),
                    parameters={
                        "dip_std": float(dip_std),
                        "mean_dip": float(mean_dip),
                        "resultant_length": float(resultant_length),
                        "n_orientations": len(boundary_orients),
                    },
                ))

        return structures

    @staticmethod
    def _detect_thickness_anomalies(
        contacts: List[DrillholeContact],
        stratigraphy: List[str],
        anomaly_factor: float,
    ) -> List[DetectedStructure]:
        """Detect anomalous thickness changes indicating proximity to faults."""
        structures = []

        # Group contacts by hole
        by_hole: Dict[str, List[DrillholeContact]] = {}
        for c in contacts:
            by_hole.setdefault(c.hole_id, []).append(c)

        # For each consecutive pair of boundaries, compute thickness per hole
        for i in range(len(stratigraphy) - 1):
            fm_below = stratigraphy[i]
            fm_above = stratigraphy[i + 1]
            boundary_name = f"{fm_below}|{fm_above}"

            thicknesses = {}
            for hole_id, hole_contacts in by_hole.items():
                sorted_contacts = sorted(hole_contacts, key=lambda c: -c.z)
                for j in range(len(sorted_contacts) - 1):
                    c_upper = sorted_contacts[j]
                    c_lower = sorted_contacts[j + 1]
                    if (c_upper.formation_above == fm_above and
                            c_lower.formation_below == fm_below):
                        thickness = abs(c_upper.z - c_lower.z)
                        thicknesses[hole_id] = thickness
                        break

            if len(thicknesses) < 3:
                continue

            vals = np.array(list(thicknesses.values()))
            mean_t = np.mean(vals)
            std_t = np.std(vals)

            if std_t < 1e-6:
                continue

            # Flag holes with anomalous thickness
            for hole_id, thickness in thicknesses.items():
                z_score = abs(thickness - mean_t) / std_t
                if z_score > anomaly_factor:
                    hole_contacts_list = by_hole[hole_id]
                    if hole_contacts_list:
                        cx = hole_contacts_list[0].x
                        cy = hole_contacts_list[0].y
                        cz = hole_contacts_list[0].z
                    else:
                        cx = cy = cz = 0.0

                    structures.append(DetectedStructure(
                        structure_type='fault',
                        name=f"ThicknessAnomaly_{hole_id}_{fm_above}",
                        center=np.array([cx, cy, cz]),
                        confidence=min(1.0, z_score / 5.0),
                        evidence=(
                            f"Unit '{fm_above}' has anomalous thickness "
                            f"{thickness:.1f}m in hole {hole_id} "
                            f"(mean={mean_t:.1f}m, Z-score={z_score:.1f}). "
                            f"May indicate fault proximity."
                        ),
                        parameters={
                            "hole_id": str(hole_id),
                            "formation": fm_above,
                            "thickness_m": float(thickness),
                            "mean_thickness_m": float(mean_t),
                            "z_score": float(z_score),
                        },
                    ))

        return structures


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class GeologicalModelEngine:
    """
    Industry-grade geological modelling engine using LoopStructural.

    This is the COMPLETE replacement for ChronosEngine. It implements the
    standard geological modelling workflow:

    1. EXTRACT: Get lithological contacts from drillhole data
    2. ORIENT: Compute strike/dip from contact spatial patterns
    3. DETECT: Automatically identify structural features
    4. BUILD: Construct LoopStructural model with proper constraints
    5. VALIDATE: Check model against input data

    Usage:
        engine = GeologicalModelEngine(extent, resolution=80)
        result = engine.build_geological_model(
            lithology_df=drillhole_data,
            stratigraphy=['Basement', 'Unit_A', 'Unit_B', 'Cover'],
            auto_detect_structures=True,
        )
        surfaces = engine.extract_surfaces()
        solids = engine.extract_solids()
    """

    FEATURE_NAME = "Stratigraphy"

    def __init__(
        self,
        extent: Dict[str, float],
        resolution: int = 80,
        boundary_padding: float = 0.05,
    ):
        if not LS_AVAILABLE:
            raise RuntimeError(
                "LoopStructural not found. Install: pip install LoopStructural>=1.6.0"
            )

        self.raw_extent = extent
        self.resolution = [resolution, resolution, resolution]
        self.boundary_padding = boundary_padding

        # Coordinate scaler: world → [0, 1]
        self.scaler = MinMaxScaler()
        bbox = np.array([
            [extent['xmin'], extent['ymin'], extent['zmin']],
            [extent['xmax'], extent['ymax'], extent['zmax']],
        ])
        self.scaler.fit(bbox)

        # Model state
        self.model: Optional[GeologicalModel] = None
        self.contacts: List[DrillholeContact] = []
        self.orientations: List[ComputedOrientation] = []
        self.detected_structures: List[DetectedStructure] = []
        self.strat_column: Optional[StratigraphicColumn] = None
        self.build_log = ModelBuildLog()

        # Record coordinate transform
        self.build_log.coordinate_transform = {
            "method": "MinMaxScaler",
            "original_extent": extent,
            "scaled_extent": {"min": [0, 0, 0], "max": [1, 1, 1]},
            "scale": self.scaler.scale_.tolist(),
            "boundary_padding": boundary_padding,
        }

        logger.info(
            f"GeologicalModelEngine initialized: res={resolution}³, "
            f"padding={boundary_padding}"
        )

    def build_geological_model(
        self,
        lithology_df: pd.DataFrame,
        stratigraphy: List[str],
        faults: Optional[List[Dict[str, Any]]] = None,
        orientation_method: str = 'plane_fit',
        auto_detect_structures: bool = True,
        cgw: float = 0.1,
        interpolator_type: str = 'FDI',
        user_orientations: Optional[pd.DataFrame] = None,
        accepted_structures: Optional[List[str]] = None,
        scalar_values: Optional[Dict[str, float]] = None,
        hole_id_col: str = 'hole_id',
        formation_col: str = 'formation',
    ) -> Dict[str, Any]:
        """
        Execute the complete geological modelling pipeline.

        This is the SINGLE entry point. It performs:
        1. Contact extraction from drillholes
        2. Orientation computation from contact geometry
        3. Structure detection (optional)
        4. LoopStructural model construction
        5. Validation

        Args:
            lithology_df: Drillhole lithology data with columns:
                hole_id, formation, X, Y, Z (and optionally from, to)
            stratigraphy: Formation names ordered OLDEST to YOUNGEST.
            faults: Optional manually-defined fault parameters.
            orientation_method: 'plane_fit', 'three_point', or 'default_horizontal'.
            auto_detect_structures: Whether to run structure detection.
            cgw: Regularization weight (0.01=tight, 0.1=smooth).
            interpolator_type: 'FDI' or 'PLI'.
            user_orientations: Optional user-supplied orientation data.
            accepted_structures: Names of auto-detected structures to include.
            scalar_values: Optional pre-computed scalar values.
            hole_id_col: Column name for drillhole ID.
            formation_col: Column name for formation.

        Returns:
            Dict with model result, build log, and stats.
        """
        import time
        start_time = time.time()
        warnings = []

        logger.info("=" * 60)
        logger.info("GEOLOGICAL MODEL ENGINE - BUILDING MODEL")
        logger.info("=" * 60)

        # Compute data hash for reproducibility
        self.build_log.data_hash = hashlib.sha256(
            pd.util.hash_pandas_object(lithology_df).values.tobytes()
        ).hexdigest()[:16]

        # ════════════════════════════════════════════════════════════
        # STEP 1: EXTRACT CONTACTS FROM DRILLHOLES
        # ════════════════════════════════════════════════════════════
        logger.info("STEP 1: Extracting lithological contacts from drillholes...")

        self.contacts, contact_stats = DrillholeContactExtractor.extract_contacts(
            lithology_df=lithology_df,
            stratigraphy=stratigraphy,
            scalar_values=scalar_values,
            hole_id_col=hole_id_col,
            formation_col=formation_col,
        )
        self.build_log.contact_extraction = contact_stats

        if not self.contacts:
            raise ValueError(
                "No contacts could be extracted from drillhole data. "
                "Check that formation names match the stratigraphy list."
            )

        logger.info(
            f"  Extracted {len(self.contacts)} contacts from "
            f"{contact_stats['holes_processed']} drillholes"
        )

        # Build stratigraphic column
        scalar_vals = {}
        thicknesses = {}
        for c in self.contacts:
            if c.formation_above not in scalar_vals:
                scalar_vals[c.formation_above] = c.scalar_value
            if c.formation_below not in scalar_vals:
                scalar_vals[c.formation_below] = c.scalar_value

        self.strat_column = StratigraphicColumn(
            formations=stratigraphy,
            scalar_values=scalar_vals,
            thicknesses=thicknesses,
            contact_names=list(set(c.contact_name for c in self.contacts)),
        )

        # ════════════════════════════════════════════════════════════
        # STEP 2: COMPUTE ORIENTATIONS FROM CONTACT GEOMETRY
        # ════════════════════════════════════════════════════════════
        logger.info(f"STEP 2: Computing orientations (method='{orientation_method}')...")

        self.orientations, orient_stats = OrientationCalculator.compute_orientations(
            contacts=self.contacts,
            method=orientation_method,
        )
        self.build_log.orientation_computation = orient_stats

        # Merge user-supplied orientations if provided
        if user_orientations is not None and len(user_orientations) > 0:
            n_user = len(user_orientations)
            for _, row in user_orientations.iterrows():
                self.orientations.append(ComputedOrientation(
                    x=float(row['X']), y=float(row['Y']), z=float(row['Z']),
                    gx=float(row.get('gx', 0)), gy=float(row.get('gy', 0)),
                    gz=float(row.get('gz', 1)),
                    dip=float(row.get('dip', 0)), dip_direction=float(row.get('dip_dir', 0)),
                    contact_name='user_supplied',
                    method='user_supplied',
                    confidence=1.0,
                ))
            logger.info(f"  Added {n_user} user-supplied orientations")

        logger.info(
            f"  Total {len(self.orientations)} orientations "
            f"({orient_stats.get('orientations_computed', 0)} computed)"
        )

        # Log orientation statistics
        if self.orientations:
            dips = [o.dip for o in self.orientations]
            logger.info(
                f"  Dip statistics: mean={np.mean(dips):.1f}°, "
                f"max={np.max(dips):.1f}°, std={np.std(dips):.1f}°"
            )

        # ════════════════════════════════════════════════════════════
        # STEP 3: DETECT STRUCTURES (optional)
        # ════════════════════════════════════════════════════════════
        if auto_detect_structures:
            logger.info("STEP 3: Detecting geological structures...")

            self.detected_structures, detect_stats = StructureDetector.detect_structures(
                contacts=self.contacts,
                orientations=self.orientations,
                stratigraphy=stratigraphy,
                lithology_df=lithology_df,
                hole_id_col=hole_id_col,
                formation_col=formation_col,
            )
            self.build_log.structure_detection = detect_stats

            # Mark accepted structures
            if accepted_structures:
                for s in self.detected_structures:
                    if s.name in accepted_structures:
                        s.accepted = True

            logger.info(
                f"  Detected {len(self.detected_structures)} structures "
                f"({sum(1 for s in self.detected_structures if s.accepted)} accepted)"
            )
        else:
            logger.info("STEP 3: Structure detection SKIPPED (auto_detect=False)")

        # ════════════════════════════════════════════════════════════
        # STEP 4: BUILD LOOPSTRUCTURAL MODEL
        # ════════════════════════════════════════════════════════════
        logger.info("STEP 4: Building LoopStructural model...")

        self._build_loopstructural_model(
            stratigraphy=stratigraphy,
            faults=faults,
            cgw=cgw,
            interpolator_type=interpolator_type,
        )

        # ════════════════════════════════════════════════════════════
        # COMPILE RESULTS
        # ════════════════════════════════════════════════════════════
        elapsed = time.time() - start_time

        self.build_log.parameters = {
            "stratigraphy": stratigraphy,
            "n_contacts": len(self.contacts),
            "n_orientations": len(self.orientations),
            "n_faults": len(faults) if faults else 0,
            "cgw": cgw,
            "interpolator_type": interpolator_type,
            "orientation_method": orientation_method,
            "resolution": self.resolution,
            "elapsed_seconds": elapsed,
        }

        logger.info("=" * 60)
        logger.info(f"MODEL BUILD COMPLETE in {elapsed:.1f}s")
        logger.info("=" * 60)

        return {
            "model": self.model,
            "contacts": self.contacts,
            "orientations": self.orientations,
            "detected_structures": self.detected_structures,
            "strat_column": self.strat_column,
            "build_log": self.build_log.to_dict(),
            "warnings": warnings,
            "elapsed_seconds": elapsed,
        }

    def _build_loopstructural_model(
        self,
        stratigraphy: List[str],
        faults: Optional[List[Dict[str, Any]]],
        cgw: float,
        interpolator_type: str,
    ) -> None:
        """Build the LoopStructural model from extracted data."""

        pad = self.boundary_padding
        origin = [-pad, -pad, -pad]
        maximum = [1 + pad, 1 + pad, 1 + pad]
        self.model = GeologicalModel(origin, maximum)

        # ─── Prepare contact data for LoopStructural ────────────────────
        contact_rows = []
        for c in self.contacts:
            # Scale to [0,1] space
            scaled = self.scaler.transform([[c.x, c.y, c.z]])[0]
            contact_rows.append({
                'X': scaled[0],
                'Y': scaled[1],
                'Z': scaled[2],
                'val': c.scalar_value,
                'feature_name': self.FEATURE_NAME,
            })

        contacts_df = pd.DataFrame(contact_rows)

        # ─── Prepare orientation data for LoopStructural ────────────────
        orient_rows = []
        for o in self.orientations:
            scaled = self.scaler.transform([[o.x, o.y, o.z]])[0]

            # Validate gradient vector
            gvec = np.array([o.gx, o.gy, o.gz])
            gnorm = np.linalg.norm(gvec)
            if gnorm < 1e-10:
                gvec = np.array([0.0, 0.0, 1.0])
            else:
                gvec = gvec / gnorm

            orient_rows.append({
                'X': scaled[0],
                'Y': scaled[1],
                'Z': scaled[2],
                'gx': gvec[0],
                'gy': gvec[1],
                'gz': gvec[2],
                'feature_name': self.FEATURE_NAME,
            })

        orientations_df = pd.DataFrame(orient_rows)

        # ─── Set model data ─────────────────────────────────────────────
        self.model.data = pd.concat(
            [contacts_df, orientations_df], ignore_index=True
        )

        logger.info(
            f"  Model data: {len(contacts_df)} contacts + "
            f"{len(orientations_df)} orientations"
        )
        logger.info(f"  Val range: [{contacts_df['val'].min():.2f}, {contacts_df['val'].max():.2f}]")

        # ─── Add faults FIRST (they deform the space) ──────────────────
        if faults:
            for f in faults:
                try:
                    fname = f.get('name', f'Fault_{id(f)}')

                    # Add fault data if geometric parameters provided
                    if all(k in f for k in ['dip', 'azimuth', 'point']):
                        self._add_fault_geometry(f, fname)

                    # Scale displacement
                    avg_scale = np.mean(self.scaler.scale_)
                    scaled_disp = f['displacement'] / avg_scale

                    self.model.create_and_add_fault(
                        fname,
                        displacement=scaled_disp,
                        fault_type=f.get('type', 'normal'),
                        force_mesh_geometry=True,
                    )

                    self.build_log.event_stack.append(f"Fault: {fname}")
                    logger.info(f"  Added fault '{fname}' (disp={f['displacement']}m)")

                except Exception as e:
                    logger.error(f"  Failed to add fault '{fname}': {e}")
                    self.build_log.warnings.append(f"Fault '{fname}' failed: {e}")

        # Also add accepted auto-detected faults
        for structure in self.detected_structures:
            if structure.accepted and structure.structure_type == 'fault':
                try:
                    offset = structure.parameters.get('offset_m', 50.0)
                    avg_scale = np.mean(self.scaler.scale_)
                    scaled_disp = offset / avg_scale

                    self.model.create_and_add_fault(
                        structure.name,
                        displacement=scaled_disp,
                        force_mesh_geometry=True,
                    )
                    self.build_log.event_stack.append(f"AutoFault: {structure.name}")
                    logger.info(f"  Added auto-detected fault '{structure.name}'")
                except Exception as e:
                    logger.warning(f"  Auto-fault '{structure.name}' failed: {e}")

        # ─── Add stratigraphic foliation ────────────────────────────────
        try:
            self.model.create_and_add_foliation(
                self.FEATURE_NAME,
                interpolatortype=interpolator_type,
                cgw=cgw,
            )
            self.build_log.event_stack.append(
                f"{self.FEATURE_NAME}: {interpolator_type} foliation (cgw={cgw})"
            )
        except Exception as e:
            logger.error(f"  Failed to create foliation: {e}")
            raise

        # ─── Solve ──────────────────────────────────────────────────────
        logger.info("  Solving implicit function...")
        self.model.update()
        logger.info("  Model solve complete")

    def _add_fault_geometry(self, fault_dict: Dict[str, Any], fault_name: str) -> None:
        """Generate and add fault geometry data to the model."""
        try:
            from .faults import FaultPlane
            fault_plane = FaultPlane.from_dict(fault_dict)
            traces = fault_plane.generate_fault_trace_points(
                extent=self.raw_extent, num_points=20
            )
            orients = fault_plane.generate_fault_orientations(
                extent=self.raw_extent, num_orientations=10
            )

            if traces.empty or orients.empty:
                return

            # Scale to model space
            scaled_traces = pd.DataFrame(
                self.scaler.transform(traces[['X', 'Y', 'Z']].values),
                columns=['X', 'Y', 'Z'],
            )
            scaled_traces['feature_name'] = fault_name
            scaled_traces['val'] = 0.0

            scaled_orients = pd.DataFrame(
                self.scaler.transform(orients[['X', 'Y', 'Z']].values),
                columns=['X', 'Y', 'Z'],
            )
            scaled_orients['gx'] = orients['gx'].values
            scaled_orients['gy'] = orients['gy'].values
            scaled_orients['gz'] = orients['gz'].values
            scaled_orients['feature_name'] = fault_name

            self.model.data = pd.concat(
                [self.model.data, scaled_traces, scaled_orients],
                ignore_index=True,
            )

        except Exception as e:
            logger.warning(f"  Fault geometry generation failed: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # MESH EXTRACTION
    # ═══════════════════════════════════════════════════════════════════

    def extract_surfaces(self) -> List[Dict[str, Any]]:
        """
        Extract isosurfaces at contact boundaries.

        Returns surfaces in WORLD coordinates with proper mesh format.
        """
        if not self.model:
            logger.warning("No model built")
            return []

        surfaces = []

        try:
            feature = self.model[self.FEATURE_NAME]
        except KeyError:
            logger.error(f"Feature '{self.FEATURE_NAME}' not found")
            return []

        # Get unique contact values
        unique_vals = np.sort(self.model.data['val'].dropna().unique())
        logger.info(f"Extracting surfaces at vals: {unique_vals}")

        for val in unique_vals:
            try:
                result_list = feature.surfaces(val)
                if result_list is None:
                    continue

                if not isinstance(result_list, (list, tuple)):
                    result_list = [result_list]

                for mesh_idx, result in enumerate(result_list):
                    verts, faces = self._extract_mesh_data(result)
                    if verts is None or faces is None:
                        continue

                    # Inverse transform to world coordinates
                    verts_world = self.scaler.inverse_transform(verts)

                    name = f"Surface_{val:.2f}"
                    if len(result_list) > 1:
                        name += f"_{mesh_idx}"

                    surfaces.append({
                        "vertices": verts_world,
                        "faces": faces,
                        "val": float(val),
                        "name": name,
                    })

                    logger.info(
                        f"  Surface at val={val:.2f}: "
                        f"{len(verts)} verts, {len(faces)} faces"
                    )

            except Exception as e:
                logger.warning(f"  Surface extraction failed at val={val}: {e}")

        logger.info(f"Extracted {len(surfaces)} surfaces")
        return surfaces

    def extract_solids(
        self, stratigraphy: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract solid volumes for each geological unit.

        Uses threshold extraction with proper boundary calculation
        in WORLD coordinates.
        """
        if not self.model or not PV_AVAILABLE:
            return []

        try:
            feature = self.model[self.FEATURE_NAME]
        except KeyError:
            return []

        unique_vals = np.sort(self.model.data['val'].dropna().unique())
        pad = self.boundary_padding
        nx, ny, nz = self.resolution

        # Create evaluation grid
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=((1 + 2*pad)/(nx-1), (1 + 2*pad)/(ny-1), (1 + 2*pad)/(nz-1)),
            origin=(-pad, -pad, -pad),
        )

        # Evaluate scalar field
        field_values = feature.evaluate_value(grid.points)
        grid['scalar'] = field_values

        field_min = float(np.nanmin(field_values))
        field_max = float(np.nanmax(field_values))

        # Compute unit boundaries (midpoints)
        boundaries = [(unique_vals[i] + unique_vals[i+1]) / 2.0
                      for i in range(len(unique_vals) - 1)]

        solids = []
        for i, val in enumerate(unique_vals):
            unit_name = stratigraphy[i] if stratigraphy and i < len(stratigraphy) else f"Unit_{i}"

            if i == 0:
                v_min = field_min - 1.0
                v_max = boundaries[0] if boundaries else field_max + 1.0
            elif i == len(unique_vals) - 1:
                v_min = boundaries[-1]
                v_max = field_max + 1.0
            else:
                v_min = boundaries[i-1]
                v_max = boundaries[i]

            try:
                clipped = grid.threshold([v_min, v_max], scalars='scalar')
                if clipped is None or clipped.n_cells == 0:
                    continue

                surface = clipped.extract_surface().triangulate()
                if surface.n_points == 0:
                    continue

                # Fill holes
                if hasattr(surface, 'fill_holes'):
                    try:
                        surface = surface.fill_holes(hole_size=1000)
                    except Exception:
                        pass

                # Transform to world coordinates
                verts_world = self.scaler.inverse_transform(
                    np.asarray(surface.points, dtype=np.float64)
                )

                faces = self._extract_pyvista_faces(surface)
                if faces is None:
                    continue

                # Volume calculation
                try:
                    faces_pv = np.hstack([
                        np.full((len(faces), 1), 3, dtype=np.int64), faces
                    ]).flatten()
                    world_mesh = pv.PolyData(verts_world, faces_pv)
                    volume_m3 = abs(float(world_mesh.volume))
                except Exception:
                    volume_m3 = 0.0

                solids.append({
                    "vertices": verts_world,
                    "faces": faces.astype(np.int64),
                    "unit_name": unit_name,
                    "val": float(val),
                    "val_range": [float(v_min), float(v_max)],
                    "volume_m3": volume_m3,
                    "n_cells": surface.n_cells,
                    "name": f"Solid_{unit_name}",
                })

                logger.info(
                    f"  Solid '{unit_name}': {len(verts_world)} verts, "
                    f"volume={volume_m3:,.0f} m³"
                )

            except Exception as e:
                logger.error(f"  Solid extraction failed for '{unit_name}': {e}")

        return solids

    def extract_unified_mesh(
        self, stratigraphy: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract unified partition mesh (Leapfrog-style).
        One mesh, each cell assigned to exactly one formation.
        """
        if not self.model or not PV_AVAILABLE:
            return None

        try:
            feature = self.model[self.FEATURE_NAME]
        except KeyError:
            return None

        unique_vals = np.sort(self.model.data['val'].dropna().unique())
        n_units = len(unique_vals)
        pad = self.boundary_padding
        res = min(180, max(self.resolution[0], 80))

        grid = pv.ImageData(
            dimensions=(res, res, res),
            spacing=(
                (1 + 2*pad)/(res-1),
                (1 + 2*pad)/(res-1),
                (1 + 2*pad)/(res-1),
            ),
            origin=(-pad, -pad, -pad),
        )

        cell_centers = grid.cell_centers().points
        field_values = feature.evaluate_value(cell_centers)

        # Partition into formations
        boundaries = [(unique_vals[i] + unique_vals[i+1]) / 2.0
                      for i in range(len(unique_vals) - 1)]

        formation_ids = np.zeros(len(field_values), dtype=np.int32)
        for i in range(n_units):
            if i == 0:
                mask = field_values < boundaries[0] if boundaries else np.ones(len(field_values), dtype=bool)
            elif i == n_units - 1:
                mask = field_values >= boundaries[-1]
            else:
                mask = (field_values >= boundaries[i-1]) & (field_values < boundaries[i])
            formation_ids[mask] = i

        grid.cell_data['Formation_ID'] = formation_ids
        verts_world = self.scaler.inverse_transform(cell_centers)

        formation_names = {}
        for i in range(n_units):
            if stratigraphy and i < len(stratigraphy):
                formation_names[i] = stratigraphy[i]
            else:
                formation_names[i] = f"Unit_{i}"

        return {
            'vertices': verts_world,
            'formation_ids': formation_ids,
            'formation_names': formation_names,
            'unique_vals': unique_vals.tolist(),
            'boundaries': boundaries,
            'n_units': n_units,
            'grid_dimensions': (res, res, res),
            '_pyvista_grid': grid,
            '_scaler': self.scaler,
        }

    # ═══════════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════════

    def validate_contacts(self) -> Dict[str, Any]:
        """
        Validate that the model honours the input contacts.

        For each extracted contact, evaluates the scalar field at that
        point and computes the deviation from the expected value.
        """
        if not self.model or not self.contacts:
            return {"error": "No model or contacts"}

        try:
            feature = self.model[self.FEATURE_NAME]
        except KeyError:
            return {"error": "Feature not found"}

        residuals = []
        for c in self.contacts:
            scaled = self.scaler.transform([[c.x, c.y, c.z]])[0]
            predicted = feature.evaluate_value(scaled.reshape(1, -1))[0]

            if np.isnan(predicted):
                continue

            residual = abs(predicted - c.scalar_value)
            residuals.append({
                "hole_id": c.hole_id,
                "contact_name": c.contact_name,
                "x": c.x, "y": c.y, "z": c.z,
                "expected": c.scalar_value,
                "predicted": float(predicted),
                "residual": float(residual),
            })

        if not residuals:
            return {"error": "No residuals computed"}

        residual_values = [r["residual"] for r in residuals]

        return {
            "n_contacts": len(residuals),
            "mean_residual": float(np.mean(residual_values)),
            "median_residual": float(np.median(residual_values)),
            "p90_residual": float(np.percentile(residual_values, 90)),
            "max_residual": float(np.max(residual_values)),
            "residuals": residuals,
        }

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _extract_mesh_data(self, result) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract vertices and faces from various mesh formats."""
        verts, faces = None, None

        if hasattr(result, 'vertices') and hasattr(result, 'triangles'):
            verts = np.asarray(result.vertices)
            faces = np.asarray(result.triangles)
        elif hasattr(result, 'points') and hasattr(result, 'faces'):
            verts = np.asarray(result.points)
            faces = self._extract_pyvista_faces(result)
        elif hasattr(result, 'vertices') and hasattr(result, 'faces'):
            verts = np.asarray(result.vertices)
            faces = np.asarray(result.faces)
        elif isinstance(result, (list, tuple)) and len(result) == 4:
            verts = np.asarray(result[0]) if result[0] is not None else None
            faces = np.asarray(result[1]) if result[1] is not None else None

        if verts is not None and len(verts) == 0:
            verts = None
        if faces is not None and len(faces) == 0:
            faces = None

        return verts, faces

    @staticmethod
    def _extract_pyvista_faces(mesh) -> Optional[np.ndarray]:
        """Extract faces from PyVista mesh format."""
        if not hasattr(mesh, 'faces') or mesh.faces is None or len(mesh.faces) == 0:
            return None

        faces_raw = np.asarray(mesh.faces)
        try:
            n_faces = len(faces_raw) // 4
            return faces_raw.reshape(n_faces, 4)[:, 1:4]
        except ValueError:
            # Manual extraction for mixed face types
            faces = []
            idx = 0
            while idx < len(faces_raw):
                n = faces_raw[idx]
                if n == 3 and idx + 3 < len(faces_raw):
                    faces.append(faces_raw[idx+1:idx+4])
                idx += n + 1
            return np.array(faces, dtype=np.int64) if faces else None

    @staticmethod
    def is_available() -> bool:
        return LS_AVAILABLE

    def get_build_log(self) -> Dict[str, Any]:
        return self.build_log.to_dict()

    def get_contacts_dataframe(self) -> pd.DataFrame:
        """Get contacts as a DataFrame for display/export."""
        if not self.contacts:
            return pd.DataFrame()

        return pd.DataFrame([{
            'hole_id': c.hole_id,
            'X': c.x, 'Y': c.y, 'Z': c.z,
            'formation_above': c.formation_above,
            'formation_below': c.formation_below,
            'contact_name': c.contact_name,
            'scalar_value': c.scalar_value,
            'confidence': c.confidence,
        } for c in self.contacts])

    def get_orientations_dataframe(self) -> pd.DataFrame:
        """Get orientations as a DataFrame for display/export."""
        if not self.orientations:
            return pd.DataFrame()

        return pd.DataFrame([{
            'X': o.x, 'Y': o.y, 'Z': o.z,
            'gx': o.gx, 'gy': o.gy, 'gz': o.gz,
            'dip': o.dip, 'dip_direction': o.dip_direction,
            'contact_name': o.contact_name,
            'method': o.method,
            'confidence': o.confidence,
            'n_points': o.n_points_used,
        } for o in self.orientations])

    def get_structures_dataframe(self) -> pd.DataFrame:
        """Get detected structures as a DataFrame for display/export."""
        if not self.detected_structures:
            return pd.DataFrame()

        return pd.DataFrame([{
            'name': s.name,
            'type': s.structure_type,
            'confidence': s.confidence,
            'evidence': s.evidence,
            'accepted': s.accepted,
            'center_x': s.center[0],
            'center_y': s.center[1],
            'center_z': s.center[2],
        } for s in self.detected_structures])
