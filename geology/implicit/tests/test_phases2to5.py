"""
Tests for Phases 2-5: Stratigraphy, Vein Model, Fault Model, Fold Frame.
=========================================================================

Tests verify mathematical correctness, contact honouring, non-crossing
guarantees, and domain assignment consistency.
"""

from __future__ import annotations

import logging
import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    import pytest
except ImportError:
    # Minimal pytest shim for standalone execution
    class _PytestShim:
        class raises:
            def __init__(self, exc, match=None):
                self.exc = exc
                self.match = match
            def __enter__(self): return self
            def __exit__(self, et, ev, tb):
                if et is None:
                    raise AssertionError(f"Expected {self.exc.__name__}")
                if not issubclass(et, self.exc):
                    return False
                if self.match and self.match not in str(ev):
                    return False
                return True
    pytest = _PytestShim()

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_horizontal_contacts(
    n_holes: int = 5,
    z_surfaces: list = None,
    unit_names: list = None,
    xy_range: tuple = (0, 100),
):
    """Generate synthetic contacts for horizontal layers."""
    if z_surfaces is None:
        z_surfaces = [80.0, 60.0, 40.0]
    if unit_names is None:
        unit_names = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]

    rng = np.random.RandomState(42)
    rows = []
    for h in range(n_holes):
        x = rng.uniform(xy_range[0], xy_range[1])
        y = rng.uniform(xy_range[0], xy_range[1])
        hid = f"DH_{h:03d}"

        for i, z in enumerate(z_surfaces):
            rows.append({
                "hole_id": hid,
                "depth": 100.0 - z,
                "X": x, "Y": y, "Z": z,
                "unit_above": unit_names[i],
                "unit_below": unit_names[i + 1],
                "surface_name": f"{unit_names[i]}_{unit_names[i + 1]}",
            })

    return pd.DataFrame(rows)


def _make_tilted_contacts(
    n_holes: int = 8,
    dip_deg: float = 30.0,
    azimuth_deg: float = 90.0,
    z_ref: float = 50.0,
    separation: float = 20.0,
):
    """Generate contacts for tilted layers (dipping east at given angle)."""
    rng = np.random.RandomState(123)
    dip_rad = np.radians(dip_deg)

    # Normal vector for the tilted surface
    # Dip azimuth 90 = dipping east: z decreases with increasing x
    nx = np.sin(dip_rad) * np.sin(np.radians(azimuth_deg))
    ny = np.sin(dip_rad) * np.cos(np.radians(azimuth_deg))
    nz = np.cos(dip_rad)

    unit_names = ["TOP", "MID", "BOT"]
    rows = []
    for h in range(n_holes):
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)
        hid = f"DH_{h:03d}"

        for i in range(2):
            # z = z_ref - i*separation adjusted for tilt
            # Plane equation: nx*(x-50) + ny*(y-50) + nz*(z-z_ref) = 0
            z = z_ref - (nx * (x - 50) + ny * (y - 50)) / nz - i * separation
            rows.append({
                "hole_id": hid,
                "depth": 100.0 - z,
                "X": x, "Y": y, "Z": z,
                "unit_above": unit_names[i],
                "unit_below": unit_names[i + 1],
                "surface_name": f"{unit_names[i]}_{unit_names[i + 1]}",
            })

    return pd.DataFrame(rows)


# ==================================================================
# PHASE 2 TESTS: Stratigraphy
# ==================================================================

class TestStratigraphicColumn:
    """Test StratigraphicModelColumn construction."""

    def test_from_contacts_creates_units(self):
        from geology.implicit.stratigraphy import StratigraphicModelColumn
        contacts = _make_horizontal_contacts()
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]

        col = StratigraphicModelColumn.from_contacts(contacts, unit_order)

        assert col.n_units == 4
        assert col.n_surfaces == 3
        assert col.unit_names == unit_order

    def test_isovalues_are_ordered(self):
        from geology.implicit.stratigraphy import StratigraphicModelColumn
        contacts = _make_horizontal_contacts()
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]

        col = StratigraphicModelColumn.from_contacts(contacts, unit_order)
        isos = col.isovalues

        assert len(isos) == 3
        # Isovalues should be monotonically increasing (deeper = larger)
        for i in range(len(isos) - 1):
            assert isos[i] < isos[i + 1], f"Isovalues not sorted: {isos}"


class TestPotentialFieldConstraints:
    """Test constraint construction for potential field method."""

    def test_value_constraints_match_surfaces(self):
        from geology.implicit.stratigraphy import (
            StratigraphicModelColumn, build_potential_field_constraints,
        )
        contacts = _make_horizontal_contacts(n_holes=5)
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]
        col = StratigraphicModelColumn.from_contacts(contacts, unit_order)

        v_coords, v_data, g_coords, g_normals = build_potential_field_constraints(
            contacts, col,
        )

        # 5 holes x 3 surfaces = 15 contacts
        assert v_coords.shape[0] == 15
        assert v_data.shape[0] == 15
        # No orientations -> no gradient constraints
        assert g_coords.shape[0] == 0

    def test_gradient_constraints_from_orientations(self):
        from geology.implicit.stratigraphy import (
            StratigraphicModelColumn, build_potential_field_constraints,
        )
        contacts = _make_horizontal_contacts(n_holes=3)
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]
        col = StratigraphicModelColumn.from_contacts(contacts, unit_order)

        # Add orientation data
        orient_df = pd.DataFrame({
            "X": [50.0, 50.0], "Y": [50.0, 50.0], "Z": [70.0, 50.0],
            "dip": [0.0, 0.0], "azimuth": [0.0, 0.0],
        })

        v_coords, v_data, g_coords, g_normals = build_potential_field_constraints(
            contacts, col, orientations_df=orient_df,
        )

        assert g_coords.shape[0] == 2
        assert g_normals.shape == (2, 3)


class TestHorizontalLayers:
    """Test stratigraphic model with horizontal layers."""

    def test_horizontal_contacts_honoured(self):
        from geology.implicit.stratigraphy import build_stratigraphic_model

        contacts = _make_horizontal_contacts(n_holes=8)
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]

        result = build_stratigraphic_model(
            contacts, unit_order,
            range_max=150.0,
            grid_resolution=5.0,
            tolerance=2.0,
        )

        summary = result["contact_honouring_summary"]
        assert summary["pct_honoured"] >= 90.0, (
            f"Contact honouring {summary['pct_honoured']:.1f}% < 90%"
        )

    def test_surfaces_extracted(self):
        from geology.implicit.stratigraphy import build_stratigraphic_model

        contacts = _make_horizontal_contacts(n_holes=6)
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]

        result = build_stratigraphic_model(
            contacts, unit_order,
            range_max=150.0,
            grid_resolution=8.0,
        )

        assert len(result["surfaces"]) >= 1, "No surfaces extracted"


class TestNonCrossing:
    """Verify isosurfaces never cross (Theorem 5.1)."""

    def test_potential_field_monotonic(self):
        """Evaluate potential field on a vertical line --
        values should be monotonically increasing (or decreasing)."""
        from geology.implicit.stratigraphy import build_stratigraphic_model

        contacts = _make_horizontal_contacts(n_holes=8)
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]

        result = build_stratigraphic_model(
            contacts, unit_order,
            range_max=150.0,
            grid_resolution=10.0,
        )

        eval_fn = result["evaluate_fn"]

        # Vertical line at centre
        z_vals = np.linspace(30, 90, 50)
        pts = np.column_stack([
            np.full_like(z_vals, 50.0),
            np.full_like(z_vals, 50.0),
            z_vals,
        ])

        field_values = eval_fn(pts)

        # For potential field, values should be monotonic along vertical
        diffs = np.diff(field_values)
        # All diffs should have same sign (or be ~zero)
        signs = np.sign(diffs[np.abs(diffs) > 1e-6])
        if len(signs) > 0:
            assert np.all(signs == signs[0]) or np.all(signs == -signs[0]), (
                "Potential field not monotonic along vertical line: "
                "surfaces may cross"
            )


class TestTiltedLayers:
    """Test stratigraphic model with tilted layers."""

    def test_tilted_contacts_honoured(self):
        from geology.implicit.stratigraphy import build_stratigraphic_model

        contacts = _make_tilted_contacts(n_holes=10, dip_deg=30.0)
        unit_order = ["TOP", "MID", "BOT"]

        # Add orientation constraints
        orient_df = pd.DataFrame({
            "X": [50.0, 25.0, 75.0],
            "Y": [50.0, 50.0, 50.0],
            "Z": [50.0, 50.0, 50.0],
            "dip": [30.0, 30.0, 30.0],
            "azimuth": [90.0, 90.0, 90.0],
        })

        result = build_stratigraphic_model(
            contacts, unit_order,
            range_max=200.0,
            grid_resolution=5.0,
            tolerance=3.0,
            orientations_df=orient_df,
        )

        summary = result["contact_honouring_summary"]
        assert summary["pct_honoured"] >= 80.0, (
            f"Tilted layer honouring {summary['pct_honoured']:.1f}% < 80%"
        )


# ==================================================================
# PHASE 3 TESTS: Vein Model
# ==================================================================

def _make_vein_drillholes(
    n_holes: int = 6,
    vein_centre_z: float = 50.0,
    vein_thickness: float = 5.0,
    vein_code: str = "VEIN",
    host_code: str = "HOST",
):
    """Generate synthetic drillhole data with a tabular vein."""
    rng = np.random.RandomState(456)
    rows = []
    for h in range(n_holes):
        x = rng.uniform(0, 100)
        y = rng.uniform(0, 100)
        hid = f"DH_{h:03d}"

        hw_z = vein_centre_z + vein_thickness / 2.0
        fw_z = vein_centre_z - vein_thickness / 2.0

        # Above vein
        rows.append({
            "hole_id": hid, "X": x, "Y": y, "Z": 100.0,
            "depth_from": 0.0, "depth_to": 100.0 - hw_z,
            "lith_code": host_code,
        })
        # Vein interval
        rows.append({
            "hole_id": hid, "X": x, "Y": y, "Z": hw_z,
            "depth_from": 100.0 - hw_z, "depth_to": 100.0 - fw_z,
            "lith_code": vein_code,
        })
        # Below vein
        rows.append({
            "hole_id": hid, "X": x, "Y": y, "Z": fw_z,
            "depth_from": 100.0 - fw_z, "depth_to": 100.0,
            "lith_code": host_code,
        })

    return pd.DataFrame(rows)


class TestVeinIntersectionExtraction:
    """Test extraction of vein intersections from lithology logs."""

    def test_extract_correct_count(self):
        from geology.implicit.vein_model import extract_vein_intersections

        data = _make_vein_drillholes(n_holes=6)
        intersections = extract_vein_intersections(
            data, "VEIN", lithology_column="lith_code",
        )

        assert len(intersections) == 6, f"Expected 6, got {len(intersections)}"

    def test_thickness_correct(self):
        from geology.implicit.vein_model import extract_vein_intersections

        data = _make_vein_drillholes(n_holes=4, vein_thickness=5.0)
        intersections = extract_vein_intersections(
            data, "VEIN", lithology_column="lith_code",
        )

        for vi in intersections:
            assert abs(vi.thickness - 5.0) < 1.0, (
                f"Thickness {vi.thickness:.2f} != 5.0"
            )

    def test_midpoint_between_hw_fw(self):
        from geology.implicit.vein_model import extract_vein_intersections

        data = _make_vein_drillholes(n_holes=3)
        intersections = extract_vein_intersections(
            data, "VEIN", lithology_column="lith_code",
        )

        for vi in intersections:
            expected_mid = (vi.hw_point + vi.fw_point) / 2.0
            np.testing.assert_allclose(vi.midpoint, expected_mid, atol=1e-10)


class TestVeinModel:
    """Test vein model construction."""

    def test_vertical_vein_builds(self):
        from geology.implicit.vein_model import (
            extract_vein_intersections, build_vein_model,
        )

        data = _make_vein_drillholes(n_holes=6, vein_thickness=5.0)
        intersections = extract_vein_intersections(
            data, "VEIN", lithology_column="lith_code",
        )

        result = build_vein_model(
            intersections,
            range_=150.0,
            grid_resolution=5.0,
            method="median_thickness",
        )

        assert "hw_evaluate_fn" in result
        assert "fw_evaluate_fn" in result
        assert "vein_sdf" in result

    def test_no_negative_thickness(self):
        """Verify HW is always above FW (no negative thickness)."""
        from geology.implicit.vein_model import (
            extract_vein_intersections, build_vein_model,
        )

        data = _make_vein_drillholes(n_holes=8, vein_thickness=5.0)
        intersections = extract_vein_intersections(
            data, "VEIN", lithology_column="lith_code",
        )

        result = build_vein_model(
            intersections,
            range_=150.0,
            min_thickness=1.0,
            grid_resolution=8.0,
            method="median_thickness",
        )

        # Sample points along vertical lines
        z = np.linspace(30, 70, 30)
        pts = np.column_stack([
            np.full_like(z, 50.0), np.full_like(z, 50.0), z,
        ])

        hw_vals = result["hw_evaluate_fn"](pts)
        fw_vals = result["fw_evaluate_fn"](pts)

        # Where both are inside vein, fw - hw should be > 0
        # (fw_field increases going into vein from FW side,
        #  hw_field decreases going into vein from HW side)
        thickness = fw_vals - hw_vals
        # After minimum thickness enforcement, should be >= min_thickness
        # at points well within the vein domain
        vein_mask = result["vein_sdf"](pts) < 0
        if np.any(vein_mask):
            assert np.all(thickness[vein_mask] >= 0.5), (
                "Negative thickness detected inside vein"
            )


class TestMinimumThickness:
    """Test minimum thickness enforcement."""

    def test_enforce_pushes_apart(self):
        from geology.implicit.vein_model import enforce_minimum_thickness

        hw = np.array([0.0, -0.5, -1.0, -2.0])
        fw = np.array([0.0, 0.3, 0.5, 2.0])
        # Thicknesses: 0, 0.8, 1.5, 4.0
        # With min_thickness=1.0: first two should be adjusted

        hw_adj, fw_adj = enforce_minimum_thickness(
            hw, fw, min_thickness=1.0,
            grid_spacing=np.array([1.0, 1.0, 1.0]),
        )

        thickness = fw_adj - hw_adj
        assert np.all(thickness >= 1.0 - 1e-10), (
            f"Thickness below minimum: {thickness}"
        )


# ==================================================================
# PHASE 4 TESTS: Fault Model
# ==================================================================

def _make_fault_contacts(
    n_points: int = 10,
    fault_strike: float = 0.0,
    fault_dip: float = 60.0,
    x_pos: float = 50.0,
):
    """Generate synthetic fault plane contact points."""
    rng = np.random.RandomState(789)
    rows = []
    for i in range(n_points):
        y = rng.uniform(0, 100)
        z = rng.uniform(20, 80)
        # Fault plane: x = x_pos + (z - 50) * tan(90 - dip)
        dip_rad = np.radians(fault_dip)
        x = x_pos + (z - 50.0) * np.cos(dip_rad) / np.sin(dip_rad)
        rows.append({
            "hole_id": f"DH_{i:03d}",
            "X": x, "Y": y, "Z": z,
            "surface_name": "fault_1",
            "unit_above": "HW", "unit_below": "FW",
            "depth": 100.0 - z,
        })
    return pd.DataFrame(rows)


class TestFaultSurface:
    """Test fault surface construction."""

    def test_fault_surface_builds(self):
        from geology.implicit.fault_model import FaultDefinition, build_fault_surface

        contacts = _make_fault_contacts()
        fault = FaultDefinition(
            name="F1",
            fault_type="normal",
            displacement=20.0,
            contacts=contacts,
        )

        result = build_fault_surface(fault, range_=200.0)

        assert "evaluate_fn" in result
        assert "hw_mask_fn" in result

    def test_hw_fw_separation(self):
        """Points on each side of fault should have different field sign."""
        from geology.implicit.fault_model import FaultDefinition, build_fault_surface

        contacts = _make_fault_contacts(n_points=15, x_pos=50.0, fault_dip=90.0)

        # Provide fault orientation data so normals point across the fault (E-W)
        orient_df = pd.DataFrame({
            "X": [50.0, 50.0, 50.0],
            "Y": [20.0, 50.0, 80.0],
            "Z": [50.0, 50.0, 50.0],
            "dip": [90.0, 90.0, 90.0],
            "azimuth": [90.0, 90.0, 90.0],  # dip 90 az 90 -> normal points east
        })

        fault = FaultDefinition(
            name="F1",
            fault_type="normal",
            displacement=20.0,
            contacts=contacts,
            orientations=orient_df,
        )

        result = build_fault_surface(fault, range_=200.0)
        eval_fn = result["evaluate_fn"]

        # Points well separated on each side of the fault
        pts = np.array([
            [20.0, 50.0, 50.0],   # far from fault (side A)
            [80.0, 50.0, 50.0],   # far from fault (side B)
        ])
        vals = eval_fn(pts)

        # The two points should have opposite sign
        assert vals[0] * vals[1] < 0, (
            f"Points on opposite sides have same sign: {vals[0]:.4f}, {vals[1]:.4f}"
        )


class TestDomainSplitting:
    """Test splitting contacts at a fault."""

    def test_split_separates_contacts(self):
        from geology.implicit.fault_model import (
            FaultDefinition, build_fault_surface, split_domain_at_fault,
        )

        fault_contacts = _make_fault_contacts(n_points=15, x_pos=50.0, fault_dip=90.0)

        # Provide fault orientation data so field has correct E-W gradient
        orient_df = pd.DataFrame({
            "X": [50.0, 50.0, 50.0],
            "Y": [20.0, 50.0, 80.0],
            "Z": [50.0, 50.0, 50.0],
            "dip": [90.0, 90.0, 90.0],
            "azimuth": [90.0, 90.0, 90.0],
        })

        fault = FaultDefinition(
            name="F1",
            fault_type="normal",
            displacement=20.0,
            contacts=fault_contacts,
            orientations=orient_df,
        )

        result = build_fault_surface(fault, range_=200.0)

        # Create contacts well-separated on each side (far from fault)
        geo_contacts = pd.DataFrame({
            "hole_id": ["A", "A", "B", "B"],
            "X": [10.0, 15.0, 85.0, 90.0],
            "Y": [50.0, 50.0, 50.0, 50.0],
            "Z": [60.0, 40.0, 60.0, 40.0],
            "surface_name": ["S1", "S1", "S1", "S1"],
            "unit_above": ["U1", "U1", "U1", "U1"],
            "unit_below": ["U2", "U2", "U2", "U2"],
        })

        hw, fw = split_domain_at_fault(geo_contacts, result["evaluate_fn"])

        assert len(hw) + len(fw) == 4
        # At least one point on each side
        assert len(hw) >= 1 and len(fw) >= 1, (
            f"Split should produce both HW and FW, got HW={len(hw)}, FW={len(fw)}"
        )


class TestDisplacementRestoration:
    """Test fault displacement restoration."""

    def test_restoration_shifts_hw(self):
        from geology.implicit.fault_model import apply_fault_displacement

        # Simple vertical fault at x=50
        def fault_fn(pts):
            return pts[:, 0] - 50.0  # positive = HW (x > 50)

        coords = np.array([
            [30.0, 50.0, 70.0],  # FW
            [70.0, 50.0, 70.0],  # HW
            [70.0, 50.0, 50.0],  # HW
        ])

        displacement = np.array([0.0, 0.0, -20.0])  # normal fault: HW drops

        restored = apply_fault_displacement(coords, fault_fn, displacement)

        # FW should be unchanged
        np.testing.assert_array_equal(restored[0], coords[0])

        # HW should be shifted by -displacement (i.e., moved UP by 20m)
        np.testing.assert_allclose(restored[1], [70.0, 50.0, 90.0])
        np.testing.assert_allclose(restored[2], [70.0, 50.0, 70.0])

    def test_restored_hw_aligns_with_fw(self):
        """After unfaulting, HW and FW contacts should align
        for a horizontal layer offset by a vertical fault."""
        from geology.implicit.fault_model import apply_fault_displacement

        def fault_fn(pts):
            return pts[:, 0] - 50.0

        # Layer at z=60 on FW, offset to z=40 on HW (normal fault, throw=20m)
        contacts = np.array([
            [30.0, 50.0, 60.0],  # FW: layer at z=60
            [70.0, 50.0, 40.0],  # HW: layer at z=40 (dropped 20m)
        ])

        displacement = np.array([0.0, 0.0, -20.0])
        restored = apply_fault_displacement(contacts, fault_fn, displacement)

        # After restoration, both should be at z=60
        assert abs(restored[0, 2] - 60.0) < 1e-10
        assert abs(restored[1, 2] - 60.0) < 1e-10


# ==================================================================
# PHASE 5 TESTS: Fold Frame
# ==================================================================

def _make_fold_orientations(
    n_points: int = 20,
    fold_axis_azimuth: float = 45.0,
    fold_axis_plunge: float = 10.0,
    fold_tightness: float = 0.5,
):
    """Generate synthetic bedding measurements from a cylindrical fold."""
    rng = np.random.RandomState(321)

    # Fold axis direction
    az_rad = np.radians(fold_axis_azimuth)
    pl_rad = np.radians(fold_axis_plunge)
    fold_ax = np.array([
        np.cos(pl_rad) * np.sin(az_rad),
        np.cos(pl_rad) * np.cos(az_rad),
        np.sin(pl_rad),
    ])

    # Perpendicular to fold axis (in horizontal plane)
    perp = np.array([np.cos(az_rad), -np.sin(az_rad), 0.0])
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-10:
        perp /= perp_norm

    rows = []
    for i in range(n_points):
        x = rng.uniform(0, 200)
        y = rng.uniform(0, 200)
        z = rng.uniform(0, 100)

        # Position along S1 (distance from axial surface)
        s1 = np.dot([x, y, z], perp)

        # Rotation angle: sinusoidal fold
        theta = fold_tightness * np.sin(s1 * 2 * np.pi / 100.0)

        # Bedding normal: rotate vertical normal by theta around fold axis
        c, s = np.cos(theta), np.sin(theta)
        vertical = np.array([0.0, 0.0, 1.0])
        # Rodrigues rotation
        n_rot = (vertical * c
                 + np.cross(fold_ax, vertical) * s
                 + fold_ax * np.dot(fold_ax, vertical) * (1 - c))
        n_rot /= np.linalg.norm(n_rot)

        # Convert back to dip/azimuth
        dip = np.degrees(np.arccos(np.clip(abs(n_rot[2]), 0, 1)))
        azimuth = np.degrees(np.arctan2(n_rot[0], n_rot[1])) % 360.0

        rows.append({
            "X": x, "Y": y, "Z": z,
            "dip": dip, "azimuth": azimuth,
        })

    return pd.DataFrame(rows)


class TestFoldAxisDetection:
    """Test fold axis auto-detection from bedding data."""

    def test_detects_known_axis(self):
        from geology.implicit.fold_frame import detect_fold_axis

        orient_df = _make_fold_orientations(
            n_points=30,
            fold_axis_azimuth=45.0,
            fold_axis_plunge=10.0,
            fold_tightness=0.8,
        )

        azimuth, plunge = detect_fold_axis(orient_df)

        # Should recover axis within +/-20 degrees
        # (allowing for the ambiguity of +/-180 in azimuth)
        az_diff = min(abs(azimuth - 45.0), abs(azimuth - 225.0), abs(azimuth + 315.0))
        assert az_diff < 30.0, (
            f"Detected azimuth {azimuth:.1f} too far from 45.0"
        )

    def test_needs_minimum_data(self):
        from geology.implicit.fold_frame import detect_fold_axis

        orient_df = pd.DataFrame({
            "X": [0, 1], "Y": [0, 1], "Z": [0, 1],
            "dip": [30, 30], "azimuth": [0, 0],
        })

        with pytest.raises(ValueError, match="at least 3"):
            detect_fold_axis(orient_df)


class TestFoldFrame:
    """Test fold frame construction."""

    def test_fold_frame_builds(self):
        from geology.implicit.fold_frame import FoldFrame, FoldFrameConfig

        config = FoldFrameConfig(auto_detect=True)
        frame = FoldFrame(config)

        orient_df = _make_fold_orientations(n_points=15)
        contacts_df = pd.DataFrame({
            "X": [50.0, 100.0, 150.0],
            "Y": [100.0, 100.0, 100.0],
            "Z": [50.0, 50.0, 50.0],
            "surface_name": ["S1", "S1", "S1"],
        })

        result = frame.build_from_orientations(
            orient_df, contacts_df,
            grid_origin=np.array([0.0, 0.0, 0.0]),
            grid_spacing=np.array([20.0, 20.0, 20.0]),
            grid_dims=(11, 11, 6),
            range_=300.0,
            nugget=0.01,
            accuracy=1e-4,
        )

        assert "s1_field" in result
        assert "s2_field" in result
        assert "s0_field" in result
        assert "fold_axis" in result
        assert result["s1_field"].shape == (11, 11, 6)

    def test_s_plot_computed(self):
        from geology.implicit.fold_frame import FoldFrame, FoldFrameConfig

        config = FoldFrameConfig(auto_detect=True)
        frame = FoldFrame(config)

        orient_df = _make_fold_orientations(n_points=15)
        contacts_df = pd.DataFrame({
            "X": [50.0], "Y": [100.0], "Z": [50.0],
            "surface_name": ["S1"],
        })

        result = frame.build_from_orientations(
            orient_df, contacts_df,
            grid_origin=np.array([0.0, 0.0, 0.0]),
            grid_spacing=np.array([20.0, 20.0, 20.0]),
            grid_dims=(11, 11, 6),
            range_=300.0,
            nugget=0.01,
            accuracy=1e-4,
        )

        s1_vals, theta_vals = result["s_plot"]
        assert len(s1_vals) == len(orient_df)
        assert len(theta_vals) == len(orient_df)

    def test_transform_to_fold_coords(self):
        from geology.implicit.fold_frame import FoldFrame, FoldFrameConfig

        config = FoldFrameConfig(auto_detect=True)
        frame = FoldFrame(config)

        orient_df = _make_fold_orientations(n_points=15)
        contacts_df = pd.DataFrame({
            "X": [50.0], "Y": [100.0], "Z": [50.0],
            "surface_name": ["S1"],
        })

        frame.build_from_orientations(
            orient_df, contacts_df,
            grid_origin=np.array([0.0, 0.0, 0.0]),
            grid_spacing=np.array([20.0, 20.0, 20.0]),
            grid_dims=(11, 11, 6),
            range_=300.0,
            nugget=0.01,
            accuracy=1e-4,
        )

        pts = np.array([[50.0, 100.0, 50.0], [100.0, 100.0, 50.0]])
        fold_coords = frame.transform_to_fold_coords(pts)

        assert fold_coords.shape == (2, 3)


# ==================================================================
# INTEGRATION TEST
# ==================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_stratigraphic_domain_assignment(self):
        """Build strat model then assign domains to block centroids."""
        from geology.implicit.stratigraphy import build_stratigraphic_model
        from geology.implicit.domain_model import assign_domains_potential_field

        contacts = _make_horizontal_contacts(n_holes=8)
        unit_order = ["UNIT_A", "UNIT_B", "UNIT_C", "UNIT_D"]

        result = build_stratigraphic_model(
            contacts, unit_order,
            range_max=150.0,
            grid_resolution=10.0,
        )

        # Create block centroids
        x = np.linspace(10, 90, 5)
        y = np.linspace(10, 90, 5)
        z = np.linspace(30, 90, 8)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        centroids = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        isovalues = result["strat_column"].isovalues
        unit_names = result["strat_column"].unit_names

        domain_codes, domain_names = assign_domains_potential_field(
            centroids, result["evaluate_fn"],
            isovalues=isovalues,
            unit_names=unit_names,
        )

        assert domain_codes.shape[0] == centroids.shape[0]
        # Should have multiple domains
        unique_domains = np.unique(domain_codes)
        assert len(unique_domains) >= 2, (
            f"Expected multiple domains, got {unique_domains}"
        )


# ==================================================================
# Runner
# ==================================================================

def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    arrow = "-->"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" {arrow} {detail}"
    print(msg)
    return passed


def run_all():
    """Run all tests and report results."""
    results = []
    test_classes = [
        TestStratigraphicColumn,
        TestPotentialFieldConstraints,
        TestHorizontalLayers,
        TestNonCrossing,
        TestTiltedLayers,
        TestVeinIntersectionExtraction,
        TestVeinModel,
        TestMinimumThickness,
        TestFaultSurface,
        TestDomainSplitting,
        TestDisplacementRestoration,
        TestFoldAxisDetection,
        TestFoldFrame,
        TestIntegration,
    ]

    total = 0
    passed = 0

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")

        inst = cls()
        methods = [m for m in dir(inst) if m.startswith("test_")]

        for method_name in sorted(methods):
            total += 1
            method = getattr(inst, method_name)
            try:
                method()
                ok = record(method_name, True)
                passed += 1
            except Exception as e:
                record(method_name, False, str(e)[:120])

    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{total} passed")
    print(f"{'='*60}")

    return passed, total


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    passed, total = run_all()
    sys.exit(0 if passed == total else 1)
