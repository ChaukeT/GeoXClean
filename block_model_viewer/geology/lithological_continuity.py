"""
LithologicalContinuityEnforcer - Prevents Disconnected Lithology Islands.

The #1 artefact in implicit geological modelling is disconnected "islands":
small blobs of one lithology floating inside another, disconnected from the
main body. These arise because the scalar field can create closed isosurfaces
where data is sparse.

REAL EXAMPLE OF THE PROBLEM:
    A sandstone unit that should be a continuous sheet from surface to 200m
    instead appears as:
      - Main body: 95% of the expected volume (correct)
      - Island at (500300, 7000150, -80): 12 cells of sandstone floating
        inside the underlying shale, 50m from any drillhole
      - Island at (500500, 7000500, -10): 3 cells near the surface edge

    These islands are GEOLOGICALLY IMPOSSIBLE for sedimentary deposits.
    They indicate poor interpolation in data-sparse regions.

SOLUTION (industry standard - Leapfrog/Vulcan/GOCAD):
1. Build 3D voxel model from scalar field (already done by extract_unified_mesh)
2. For each lithological unit, run 3D connected component labeling
3. Identify the primary body (largest component) for each unit
4. Flag small disconnected components as islands
5. Reassign island voxels to the geologically-correct neighbour
6. Report continuity metrics for JORC/SAMREC audit

Algorithm: 3D flood fill on regular grid with 6-connectivity (face-adjacent).

References:
- Cowan et al. (2003) "Practical Implicit Geological Modelling" §4.3
- He et al. (2017) "Topology checking for geological 3D models" Math Geosci
- Wellmann et al. (2019) "Structural geological modelling" CAGEO

GeoX Invariant Compliance:
- Deterministic (same input -> same output)
- Full audit trail of every reassignment
- Original formation_ids preserved in audit log
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConnectedComponent:
    """A single connected body of one lithological unit."""
    formation_id: int
    formation_name: str
    component_label: int
    cell_count: int
    volume_m3: float
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    is_primary: bool
    cell_indices: np.ndarray


@dataclass
class IslandReassignment:
    """Record of a single island being reassigned."""
    original_formation_id: int
    original_formation_name: str
    new_formation_id: int
    new_formation_name: str
    cell_count: int
    centroid: np.ndarray
    reason: str
    method: str  # 'neighbour_majority', 'nearest_contact', 'enclosing'


@dataclass
class ContinuityReport:
    """Full report on lithological continuity analysis."""
    formation_stats: Dict[int, Dict[str, Any]]
    all_components: List[ConnectedComponent]
    islands: List[ConnectedComponent]
    reassignments: List[IslandReassignment]
    total_cells: int
    total_islands: int
    total_island_cells: int
    island_volume_fraction: float
    all_continuous: bool
    grid_dimensions: Tuple[int, int, int]
    cell_volume_m3: float
    timestamp: str = ""
    method_used: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# CORE 3D CONNECTED COMPONENT LABELING
# ═══════════════════════════════════════════════════════════════════════════════

class ConnectedComponentLabeler:
    """
    3D connected component labeling on a regular voxel grid.
    Uses 6-connectivity (face-adjacent cells only, no diagonals).
    This is geological standard: diagonals imply unrealistic single-cell pinch-outs.
    Algorithm: BFS flood fill, O(N) for N cells.
    """

    OFFSETS_6 = np.array([
        [-1, 0, 0], [1, 0, 0],
        [0, -1, 0], [0, 1, 0],
        [0, 0, -1], [0, 0, 1],
    ], dtype=np.int32)

    @staticmethod
    def _determine_cell_dims(formation_ids, grid_dims):
        """Determine the effective cell grid dimensions from array length and node dims."""
        nx, ny, nz = grid_dims
        n_nodes = nx * ny * nz
        n_cells = (nx - 1) * (ny - 1) * (nz - 1)
        n = len(formation_ids)

        if n == n_cells and n_cells > 0:
            return nx - 1, ny - 1, nz - 1
        elif n == n_nodes:
            return nx, ny, nz
        else:
            cube = round(n ** (1.0/3.0))
            if cube ** 3 == n:
                return cube, cube, cube
            return nx - 1, ny - 1, nz - 1

    @staticmethod
    def label_components_3d(
        formation_ids: np.ndarray,
        grid_dims: Tuple[int, int, int],
        target_formation_id: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Label connected components for a single formation in a 3D grid.

        Returns:
            labels: 0 = not this formation, 1..N = component labels.
            n_components: Number of distinct components found.
        """
        cx, cy, cz = ConnectedComponentLabeler._determine_cell_dims(formation_ids, grid_dims)

        is_target = (formation_ids == target_formation_id)
        labels = np.zeros(len(formation_ids), dtype=np.int32)

        # Reshape to 3D for efficient indexing
        mask_3d = is_target[:cx*cy*cz].reshape(cx, cy, cz)
        labels_3d = labels[:cx*cy*cz].reshape(cx, cy, cz)

        current_label = 0

        for ix in range(cx):
            for iy in range(cy):
                for iz in range(cz):
                    if mask_3d[ix, iy, iz] and labels_3d[ix, iy, iz] == 0:
                        current_label += 1
                        ConnectedComponentLabeler._flood_fill_bfs(
                            mask_3d, labels_3d, ix, iy, iz,
                            current_label, cx, cy, cz
                        )

        return labels_3d.ravel(), current_label

    @staticmethod
    def _flood_fill_bfs(mask, labels, sx, sy, sz, label, nx, ny, nz):
        """BFS flood fill from a starting cell."""
        queue = deque()
        queue.append((sx, sy, sz))
        labels[sx, sy, sz] = label

        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in ConnectedComponentLabeler.OFFSETS_6:
                nx2, ny2, nz2 = x + dx, y + dy, z + dz
                if (0 <= nx2 < nx and 0 <= ny2 < ny and 0 <= nz2 < nz):
                    if mask[nx2, ny2, nz2] and labels[nx2, ny2, nz2] == 0:
                        labels[nx2, ny2, nz2] = label
                        queue.append((nx2, ny2, nz2))


# ═══════════════════════════════════════════════════════════════════════════════
# LITHOLOGICAL CONTINUITY ENFORCER
# ═══════════════════════════════════════════════════════════════════════════════

class LithologicalContinuityEnforcer:
    """
    Detects and eliminates disconnected lithological islands.

    After implicit modelling, this class:
    1. Labels connected components for each lithological unit
    2. Identifies the primary body (largest component)
    3. Classifies smaller components as islands
    4. Reassigns island cells to the correct neighbour
    5. Reports full continuity metrics

    Reassignment strategies (priority order):
    1. NEIGHBOUR_MAJORITY: Most common adjacent non-island formation
    2. ENCLOSING: Formation that entirely surrounds the island
    3. NEAREST_CONTACT: Based on nearest drillhole contact (tiebreaker)
    """

    def __init__(
        self,
        min_component_fraction: float = 0.05,
        min_component_cells: int = 10,
        max_islands_per_formation: int = 20,
        enable_reassignment: bool = True,
    ):
        self.min_component_fraction = min_component_fraction
        self.min_component_cells = min_component_cells
        self.max_islands_per_formation = max_islands_per_formation
        self.enable_reassignment = enable_reassignment

    def enforce_continuity(
        self,
        formation_ids: np.ndarray,
        grid_dimensions: Tuple[int, int, int],
        formation_names: Dict[int, str],
        cell_positions: Optional[np.ndarray] = None,
        cell_volume_m3: float = 1.0,
        contact_points: Optional[np.ndarray] = None,
        contact_formation_ids: Optional[np.ndarray] = None,
    ) -> ContinuityReport:
        """
        Run full continuity analysis and optional island elimination.

        Args:
            formation_ids: Flat array of formation IDs for each voxel (MODIFIED IN PLACE if reassignment enabled).
            grid_dimensions: (nx, ny, nz) of the voxel grid.
            formation_names: {formation_id: name} mapping.
            cell_positions: (N, 3) world coordinates of cell centres.
            cell_volume_m3: Volume of each cell in cubic metres.
            contact_points: (M, 3) coordinates of drillhole contacts.
            contact_formation_ids: (M,) formation IDs of contacts.

        Returns:
            ContinuityReport with all analysis, islands, and reassignments.
        """
        from datetime import datetime

        logger.info(
            f"Continuity analysis: {len(formation_ids)} cells, "
            f"grid {grid_dimensions}, {len(formation_names)} formations"
        )

        working_ids = formation_ids  # We modify in place
        all_components: List[ConnectedComponent] = []
        islands: List[ConnectedComponent] = []
        reassignments: List[IslandReassignment] = []
        formation_stats: Dict[int, Dict[str, Any]] = {}

        unique_formations = np.unique(formation_ids)

        # ── STEP 1: Label components for each formation ──
        for fid in unique_formations:
            fname = formation_names.get(int(fid), f"Unit_{fid}")
            total_cells = int(np.sum(formation_ids == fid))
            if total_cells == 0:
                continue

            labels, n_components = ConnectedComponentLabeler.label_components_3d(
                formation_ids, grid_dimensions, int(fid)
            )

            # ── STEP 2: Analyze each component ──
            comps: List[ConnectedComponent] = []
            for comp_label in range(1, n_components + 1):
                cell_indices = np.where(labels == comp_label)[0]
                cell_count = len(cell_indices)

                if cell_positions is not None and len(cell_positions) >= len(formation_ids):
                    pos = cell_positions[cell_indices]
                    centroid = np.mean(pos, axis=0)
                    bbox_min = np.min(pos, axis=0)
                    bbox_max = np.max(pos, axis=0)
                else:
                    centroid = np.zeros(3)
                    bbox_min = np.zeros(3)
                    bbox_max = np.zeros(3)

                comps.append(ConnectedComponent(
                    formation_id=int(fid), formation_name=fname,
                    component_label=comp_label, cell_count=cell_count,
                    volume_m3=cell_count * cell_volume_m3,
                    centroid=centroid, bbox_min=bbox_min, bbox_max=bbox_max,
                    is_primary=False, cell_indices=cell_indices,
                ))

            # ── STEP 3: Primary body vs islands ──
            if comps:
                comps.sort(key=lambda c: c.cell_count, reverse=True)
                comps[0].is_primary = True

                formation_islands = []
                for comp in comps[1:]:
                    frac = comp.cell_count / total_cells
                    if frac < self.min_component_fraction or comp.cell_count < self.min_component_cells:
                        formation_islands.append(comp)
                    else:
                        logger.info(f"  {fname}: secondary body {comp.cell_count} cells ({frac:.1%}) — kept")

                islands.extend(formation_islands)
                primary = comps[0]
                formation_stats[int(fid)] = {
                    'name': fname,
                    'total_cells': total_cells,
                    'n_components': n_components,
                    'primary_cells': primary.cell_count,
                    'primary_fraction': primary.cell_count / total_cells,
                    'n_islands': len(formation_islands),
                    'island_cells': sum(c.cell_count for c in formation_islands),
                    'island_fraction': sum(c.cell_count for c in formation_islands) / total_cells,
                    'is_continuous': len(formation_islands) == 0,
                }
                all_components.extend(comps)

                if formation_islands:
                    logger.warning(
                        f"  {fname}: {len(formation_islands)} islands "
                        f"({sum(c.cell_count for c in formation_islands)} cells)"
                    )
                    if len(formation_islands) > self.max_islands_per_formation:
                        logger.error(f"  {fname}: EXCESSIVE ISLANDS — severe model instability")

        # ── STEP 4: Reassign island cells ──
        if self.enable_reassignment and islands:
            logger.info(f"Reassigning {len(islands)} islands ({sum(i.cell_count for i in islands)} cells)")

            cx, cy, cz = ConnectedComponentLabeler._determine_cell_dims(
                formation_ids, grid_dimensions
            )

            for island in islands:
                ra = self._reassign_island(
                    island, working_ids, (cx, cy, cz),
                    formation_names, cell_positions,
                    contact_points, contact_formation_ids,
                )
                if ra is not None:
                    working_ids[island.cell_indices] = ra.new_formation_id
                    reassignments.append(ra)
                    logger.info(
                        f"  {island.formation_name} → {ra.new_formation_name} "
                        f"({island.cell_count} cells, {ra.method})"
                    )

        # ── STEP 5: Build report ──
        total_island_cells = sum(i.cell_count for i in islands)
        total_cells = len(formation_ids)

        report = ContinuityReport(
            formation_stats=formation_stats,
            all_components=all_components,
            islands=islands,
            reassignments=reassignments,
            total_cells=total_cells,
            total_islands=len(islands),
            total_island_cells=total_island_cells,
            island_volume_fraction=total_island_cells / max(total_cells, 1),
            all_continuous=len(islands) == 0,
            grid_dimensions=grid_dimensions,
            cell_volume_m3=cell_volume_m3,
            timestamp=datetime.now().isoformat(),
            method_used="6-connectivity BFS flood fill + neighbour majority reassignment",
        )

        logger.info(
            f"Continuity complete: {len(islands)} islands, "
            f"{total_island_cells} cells ({report.island_volume_fraction:.2%})"
        )
        return report

    def _reassign_island(self, island, formation_ids, cell_dims,
                         formation_names, cell_positions,
                         contact_points, contact_formation_ids):
        """Determine correct formation for an island via neighbour analysis."""
        cx, cy, cz = cell_dims
        island_set = set(island.cell_indices.tolist())

        # Find face-adjacent neighbours NOT in this island
        neighbour_counts: Dict[int, int] = {}
        for flat_idx in island.cell_indices:
            ix = flat_idx // (cy * cz)
            rem = flat_idx % (cy * cz)
            iy = rem // cz
            iz = rem % cz

            for dx, dy, dz in ConnectedComponentLabeler.OFFSETS_6:
                nx2, ny2, nz2 = ix + dx, iy + dy, iz + dz
                if 0 <= nx2 < cx and 0 <= ny2 < cy and 0 <= nz2 < cz:
                    nflat = nx2 * cy * cz + ny2 * cz + nz2
                    if nflat not in island_set:
                        nfid = int(formation_ids[nflat])
                        if nfid != island.formation_id:
                            neighbour_counts[nfid] = neighbour_counts.get(nfid, 0) + 1

        if not neighbour_counts:
            # Internal island — try nearest contact
            if (contact_points is not None and contact_formation_ids is not None
                    and cell_positions is not None):
                return self._reassign_by_nearest_contact(
                    island, cell_positions, contact_points,
                    contact_formation_ids, formation_names
                )
            return None

        best_fid = max(neighbour_counts, key=neighbour_counts.get)
        best_count = neighbour_counts[best_fid]
        total_nb = sum(neighbour_counts.values())

        # Clear majority (>60%) or single enclosing formation
        if len(neighbour_counts) == 1:
            method = 'enclosing'
            reason = f"Entirely enclosed by '{formation_names.get(best_fid, f'Unit_{best_fid}')}'"
        elif best_count / max(total_nb, 1) > 0.6:
            method = 'neighbour_majority'
            reason = f"{best_count}/{total_nb} neighbours ({best_count/total_nb:.0%})"
        else:
            # No clear majority — try nearest contact
            if (contact_points is not None and contact_formation_ids is not None
                    and cell_positions is not None):
                nc = self._reassign_by_nearest_contact(
                    island, cell_positions, contact_points,
                    contact_formation_ids, formation_names
                )
                if nc is not None:
                    return nc

            method = 'neighbour_majority'
            reason = f"Plurality: {best_count}/{total_nb} ({best_count/total_nb:.0%})"

        return IslandReassignment(
            original_formation_id=island.formation_id,
            original_formation_name=island.formation_name,
            new_formation_id=best_fid,
            new_formation_name=formation_names.get(best_fid, f"Unit_{best_fid}"),
            cell_count=island.cell_count,
            centroid=island.centroid,
            reason=reason,
            method=method,
        )

    def _reassign_by_nearest_contact(self, island, cell_positions,
                                     contact_points, contact_formation_ids,
                                     formation_names):
        """Reassign based on nearest drillhole contact."""
        from scipy.spatial import cKDTree

        if len(contact_points) == 0:
            return None

        tree = cKDTree(contact_points)
        dist, idx = tree.query(island.centroid.reshape(1, -1), k=1)
        nearest_fid = int(contact_formation_ids[idx[0]])

        if nearest_fid == island.formation_id:
            if len(contact_points) > 1:
                _, indices = tree.query(island.centroid.reshape(1, -1), k=min(5, len(contact_points)))
                for i in indices.ravel():
                    cand = int(contact_formation_ids[i])
                    if cand != island.formation_id:
                        nearest_fid = cand
                        dist = np.linalg.norm(contact_points[i] - island.centroid)
                        break
                else:
                    return None
            else:
                return None

        d = float(dist[0] if hasattr(dist, '__len__') else dist)
        return IslandReassignment(
            original_formation_id=island.formation_id,
            original_formation_name=island.formation_name,
            new_formation_id=nearest_fid,
            new_formation_name=formation_names.get(nearest_fid, f"Unit_{nearest_fid}"),
            cell_count=island.cell_count,
            centroid=island.centroid,
            reason=f"Nearest contact at {d:.1f}m",
            method='nearest_contact',
        )

    # ═══════════════════════════════════════════════════════════════════
    # POST-REASSIGNMENT VALIDATION
    # ═══════════════════════════════════════════════════════════════════

    def validate_after_reassignment(self, formation_ids, grid_dimensions, formation_names):
        """Quick validation that no significant islands remain."""
        result = {'clean': True, 'formation_components': {}}
        for fid in np.unique(formation_ids):
            _, n_comp = ConnectedComponentLabeler.label_components_3d(
                formation_ids, grid_dimensions, int(fid)
            )
            fname = formation_names.get(int(fid), f"Unit_{fid}")
            result['formation_components'][fname] = n_comp
            if n_comp > 1:
                total = int(np.sum(formation_ids == fid))
                labels, _ = ConnectedComponentLabeler.label_components_3d(
                    formation_ids, grid_dimensions, int(fid)
                )
                for cl in range(2, n_comp + 1):
                    if int(np.sum(labels == cl)) >= self.min_component_cells:
                        result['clean'] = False
                        break
        return result

    # ═══════════════════════════════════════════════════════════════════
    # STRATIGRAPHIC CONTINUITY CHECKS
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def check_lateral_continuity(formation_ids, grid_dimensions, target_formation_id):
        """
        Check if a formation forms a laterally continuous sheet.
        For sedimentary deposits, each formation should appear in most vertical columns.

        Returns dict with coverage_fraction, gap_columns, is_continuous.
        """
        cx, cy, cz = ConnectedComponentLabeler._determine_cell_dims(formation_ids, grid_dimensions)

        try:
            ids_3d = formation_ids[:cx*cy*cz].reshape(cx, cy, cz)
        except ValueError:
            return {'error': 'Cannot reshape', 'is_continuous': False}

        columns_with = 0
        gap_locations = []
        for ix in range(cx):
            for iy in range(cy):
                if np.any(ids_3d[ix, iy, :] == target_formation_id):
                    columns_with += 1
                else:
                    if len(gap_locations) < 100:
                        gap_locations.append((ix, iy))

        total_columns = cx * cy
        coverage = columns_with / max(total_columns, 1)
        return {
            'coverage_fraction': coverage,
            'gap_columns': total_columns - columns_with,
            'total_columns': total_columns,
            'is_continuous': coverage > 0.80,
            'gap_locations': gap_locations,
        }

    @staticmethod
    def check_vertical_stacking(formation_ids, grid_dimensions, stratigraphy_order):
        """
        Check that formations respect stratigraphic ordering vertically.
        Violations indicate stratigraphic inversion (strong island indicator).

        Args:
            stratigraphy_order: List of formation IDs from youngest (top) to oldest (bottom).
        Returns dict with violations, violation_fraction, is_ordered.
        """
        cx, cy, cz = ConnectedComponentLabeler._determine_cell_dims(formation_ids, grid_dimensions)

        try:
            ids_3d = formation_ids[:cx*cy*cz].reshape(cx, cy, cz)
        except ValueError:
            return {'error': 'Cannot reshape', 'is_ordered': False}

        rank_map = {fid: rank for rank, fid in enumerate(stratigraphy_order)}
        violations = 0
        violation_locs = []

        for ix in range(cx):
            for iy in range(cy):
                prev_rank = -1
                for iz in range(cz - 1, -1, -1):  # top to bottom
                    fid = int(ids_3d[ix, iy, iz])
                    rank = rank_map.get(fid, -1)
                    if rank == -1:
                        continue
                    if rank < prev_rank:
                        violations += 1
                        if len(violation_locs) < 50:
                            violation_locs.append((ix, iy, iz))
                    prev_rank = rank

        total_checks = cx * cy * cz
        return {
            'violations': violations,
            'violation_fraction': violations / max(total_checks, 1),
            'is_ordered': violations / max(total_checks, 1) < 0.01,
            'violation_locations': violation_locs,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def enforce_lithological_continuity(
    unified_mesh: Dict[str, Any],
    min_island_fraction: float = 0.05,
    min_island_cells: int = 10,
    enable_reassignment: bool = True,
    contact_points: Optional[np.ndarray] = None,
    contact_formation_ids: Optional[np.ndarray] = None,
) -> ContinuityReport:
    """
    Convenience: run continuity enforcement on extract_unified_mesh() output.
    """
    enforcer = LithologicalContinuityEnforcer(
        min_component_fraction=min_island_fraction,
        min_component_cells=min_island_cells,
        enable_reassignment=enable_reassignment,
    )

    formation_ids = unified_mesh['formation_ids']
    grid_dims = unified_mesh['grid_dimensions']
    formation_names = unified_mesh['formation_names']
    cell_positions = unified_mesh.get('vertices')

    cell_volume = 1.0
    if '_pyvista_grid' in unified_mesh:
        try:
            s = unified_mesh['_pyvista_grid'].spacing
            cell_volume = s[0] * s[1] * s[2]
        except Exception:
            pass

    return enforcer.enforce_continuity(
        formation_ids=formation_ids,
        grid_dimensions=grid_dims,
        formation_names=formation_names,
        cell_positions=cell_positions,
        cell_volume_m3=cell_volume,
        contact_points=contact_points,
        contact_formation_ids=contact_formation_ids,
    )


def generate_continuity_audit_markdown(report: ContinuityReport) -> str:
    """Generate a markdown audit report from a ContinuityReport."""
    lines = [
        "# Lithological Continuity Audit Report",
        "",
        f"**Timestamp:** {report.timestamp}",
        f"**Method:** {report.method_used}",
        f"**Grid:** {report.grid_dimensions[0]} x {report.grid_dimensions[1]} x {report.grid_dimensions[2]}",
        f"**Total cells:** {report.total_cells:,}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total islands detected | {report.total_islands} |",
        f"| Total island cells | {report.total_island_cells:,} |",
        f"| Island volume fraction | {report.island_volume_fraction:.2%} |",
        f"| All formations continuous | {'YES' if report.all_continuous else 'NO'} |",
        f"| Reassignments made | {len(report.reassignments)} |",
        "",
        "## Per-Formation Analysis",
        "",
        "| Formation | Cells | Components | Primary % | Islands | Island Cells | OK |",
        "|-----------|-------|------------|-----------|---------|-------------|-----|",
    ]

    for fid, stats in sorted(report.formation_stats.items()):
        ok = 'Y' if stats['is_continuous'] else 'N'
        lines.append(
            f"| {stats['name']} | {stats['total_cells']:,} | "
            f"{stats['n_components']} | {stats['primary_fraction']:.1%} | "
            f"{stats['n_islands']} | {stats['island_cells']:,} | {ok} |"
        )

    if report.reassignments:
        lines.extend([
            "",
            "## Reassignment Log",
            "",
            "| From | To | Cells | Method | Reason |",
            "|------|----|-------|--------|--------|",
        ])
        for r in report.reassignments:
            lines.append(
                f"| {r.original_formation_name} | {r.new_formation_name} | "
                f"{r.cell_count} | {r.method} | {r.reason} |"
            )

    return "\n".join(lines)
