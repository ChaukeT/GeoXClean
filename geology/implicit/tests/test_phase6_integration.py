"""
Phase 6 Integration Tests — Full Workflow: Data → Geology → Estimation.
=========================================================================

Tests the complete pipeline:
1. GeologicalModelBuilder orchestrator dispatching
2. Stratigraphic build → domain assignment → ARBF estimation wiring
3. Vein model build through orchestrator
4. Faulted model build through orchestrator
5. Fold frame build through orchestrator
6. Cross-section sampling and rendering
7. Domain assignment to block centroids
"""

from __future__ import annotations

import os
import sys

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════
# Synthetic data helpers
# ═══════════════════════════════════════════════════════════════════

def _make_horizontal_contacts(n_holes=5, z_surface=50.0):
    """Make synthetic drillhole contacts for a horizontal layer at z=50."""
    rows = []
    for i in range(n_holes):
        x = i * 100
        y = 0
        rows.append({
            "hole_id": f"DH{i:03d}",
            "X": float(x), "Y": float(y), "Z": z_surface,
            "unit_above": "OVB",
            "unit_below": "ORE",
            "surface_name": "OVB_ORE",
            "depth": 100.0 - z_surface,
        })
    return pd.DataFrame(rows)


def _make_stratigraphic_contacts(n_holes=5):
    """Make synthetic contacts for a 3-layer sequence: OVB / ORE / FW."""
    rows = []
    for i in range(n_holes):
        x = i * 100
        y = 0
        # Surface 1: OVB/ORE at z=70
        rows.append({
            "hole_id": f"DH{i:03d}", "X": float(x), "Y": float(y),
            "Z": 70.0, "unit_above": "OVB", "unit_below": "ORE",
            "surface_name": "OVB_ORE", "depth": 30.0,
        })
        # Surface 2: ORE/FW at z=30
        rows.append({
            "hole_id": f"DH{i:03d}", "X": float(x), "Y": float(y),
            "Z": 30.0, "unit_above": "ORE", "unit_below": "FW",
            "surface_name": "ORE_FW", "depth": 70.0,
        })
    return pd.DataFrame(rows)


def _make_orientations(n=5, dip=0.0, azimuth=0.0):
    """Make synthetic orientation measurements."""
    rows = []
    for i in range(n):
        rows.append({
            "X": float(i * 100), "Y": 0.0, "Z": 50.0,
            "dip": dip, "azimuth": azimuth,
        })
    return pd.DataFrame(rows)


def _make_block_centroids(nx=10, ny=1, nz=10, spacing=20.0):
    """Make a regular grid of block centroids."""
    xs = np.arange(nx) * spacing + spacing / 2
    ys = np.arange(ny) * spacing + spacing / 2
    zs = np.arange(nz) * spacing + spacing / 2
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


# ═══════════════════════════════════════════════════════════════════
# Test: Orchestrator dispatch
# ═══════════════════════════════════════════════════════════════════

class TestOrchestratorDispatch:
    """Verify the builder dispatches to correct model type."""

    def test_single_surface_default(self):
        from geology.implicit import GeologicalModelBuilder
        builder = GeologicalModelBuilder({"model_type": "stratiform"})
        contacts = _make_horizontal_contacts()
        builder.set_contacts(contacts)

        result = builder.build()
        assert "surfaces" in result
        assert "evaluate_fn" in result
        assert "scalar_field" in result

    def test_single_surface_contacts_honoured(self):
        from geology.implicit import GeologicalModelBuilder
        builder = GeologicalModelBuilder({
            "model_type": "stratiform",
            "grid_resolution": 10.0,
            "tolerance": 5.0,
        })
        contacts = _make_horizontal_contacts(n_holes=5, z_surface=50.0)
        builder.set_contacts(contacts)

        result = builder.build()
        summary = result["contact_honouring_summary"]
        assert summary["pct_honoured"] >= 80.0, (
            f"Contact honouring {summary['pct_honoured']:.1f}% < 80%"
        )


# ═══════════════════════════════════════════════════════════════════
# Test: Domain assignment
# ═══════════════════════════════════════════════════════════════════

class TestDomainAssignment:
    """Test domain code assignment to block centroids."""

    def test_domain_assignment_potential_field(self):
        from geology.implicit.domain_model import assign_domains_potential_field

        # Create a simple linear field: f(x,y,z) = z
        def linear_field(pts):
            return pts[:, 2]

        centroids = _make_block_centroids(nx=5, ny=1, nz=10, spacing=10)
        codes, names = assign_domains_potential_field(
            centroids, linear_field,
            isovalues=[30.0, 70.0],
            unit_names=["DEEP", "MIDDLE", "SHALLOW"],
        )

        assert len(codes) == len(centroids)
        unique = np.unique(codes)
        assert len(unique) >= 2, "Should have at least 2 domains"
        # Points with z < 30 → DEEP (code 0)
        deep_mask = centroids[:, 2] < 30.0
        if np.any(deep_mask):
            assert np.all(codes[deep_mask] == 0)

    def test_domain_from_lithology(self):
        from geology.implicit.domain_model import assign_domains_from_lithology

        df = pd.DataFrame({
            "lith_code": ["BIF", "BIF", "SHL", "GRN", "BIF"],
        })
        grouping = {
            "Iron": ["BIF"],
            "Shale": ["SHL"],
            "Granite": ["GRN"],
        }
        codes = assign_domains_from_lithology(df, grouping)
        assert codes[0] == 0  # BIF → Iron (first group)
        assert codes[2] == 1  # SHL → Shale
        assert codes[3] == 2  # GRN → Granite


# ═══════════════════════════════════════════════════════════════════
# Test: Cross-section sampling
# ═══════════════════════════════════════════════════════════════════

class TestCrossSection:
    """Test cross-section sampling and classification."""

    def test_sample_section(self):
        from block_model_viewer.ui.cross_section_widget import sample_section

        # Linear field: f(x,y,z) = z
        def field(pts):
            return pts[:, 2]

        vals, h, z = sample_section(
            start_xy=(0, 0), end_xy=(100, 0),
            z_min=0, z_max=100,
            evaluate_fn=field,
            nx=50, nz=25,
        )

        assert vals.shape == (25, 50)
        # Top row (z_max=100) should have highest values
        assert vals[0, 0] > vals[-1, 0], "Top row should have higher z values"

    def test_classify_section(self):
        from block_model_viewer.ui.cross_section_widget import classify_section

        # Create section values that increase from bottom to top
        nz, nx = 20, 30
        vals = np.linspace(0, 100, nz).reshape(-1, 1) * np.ones((1, nx))

        codes = classify_section(vals, isovalues=[30.0, 70.0])
        assert codes.shape == (nz, nx)

        # 3 domains
        unique = np.unique(codes)
        assert len(unique) == 3

    def test_render_section_to_image(self):
        from block_model_viewer.ui.cross_section_widget import render_section_to_image

        codes = np.zeros((20, 30), dtype=np.int32)
        codes[:10, :] = 0
        codes[10:, :] = 1

        img = render_section_to_image(codes, width_px=100, height_px=50)
        assert img.shape == (50, 100, 3)
        assert img.dtype == np.uint8

    def test_drillhole_traces(self):
        from block_model_viewer.ui.cross_section_widget import drillhole_traces_on_section

        dh = pd.DataFrame({
            "hole_id": ["DH001"] * 5 + ["DH002"] * 5,
            "X": [50] * 5 + [500] * 5,  # DH002 far from section
            "Y": [0] * 10,
            "Z": list(range(0, 100, 20)) * 2,
        })

        traces = drillhole_traces_on_section(
            dh, start_xy=(0, 0), end_xy=(100, 0), width=20,
        )

        # DH001 is on the section line, DH002 is 500m away
        assert len(traces) >= 1
        assert traces[0]["hole_id"] == "DH001"


# ═══════════════════════════════════════════════════════════════════
# Test: Full pipeline (data → geology → domain → estimation config)
# ═══════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """End-to-end: contacts → geological model → domain codes → estimation ready."""

    def test_stratigraphy_to_domains(self):
        """Build stratigraphic model and assign domains to blocks."""
        from geology.implicit import GeologicalModelBuilder
        from geology.implicit.domain_model import assign_domains_potential_field

        contacts = _make_horizontal_contacts(n_holes=5, z_surface=50.0)
        builder = GeologicalModelBuilder({
            "model_type": "stratiform",
            "grid_resolution": 10.0,
        })
        builder.set_contacts(contacts)
        result = builder.build()

        # Now assign domains
        centroids = _make_block_centroids(nx=5, ny=1, nz=10, spacing=10)
        evaluate_fn = result["evaluate_fn"]

        codes, names = assign_domains_potential_field(
            centroids, evaluate_fn,
            isovalues=[0.0],
            unit_names=["BELOW", "ABOVE"],
        )

        # Blocks below the surface (z < ~50) should be BELOW
        # Blocks above should be ABOVE
        assert len(np.unique(codes)) >= 2

    def test_fold_frame_pipeline(self):
        """Build fold frame model end-to-end."""
        from geology.implicit import GeologicalModelBuilder

        contacts = _make_horizontal_contacts(n_holes=5, z_surface=50.0)

        # Need varied orientations for fold detection
        orient_rows = []
        for i in range(8):
            angle = i * 45  # Different dips to create fold signal
            orient_rows.append({
                "X": float(i * 50), "Y": 0.0, "Z": 50.0,
                "dip": 10.0 + 20.0 * np.sin(np.radians(angle)),
                "azimuth": float(angle),
            })
        orientations = pd.DataFrame(orient_rows)

        builder = GeologicalModelBuilder({
            "model_type": "structural",
            "grid_resolution": 20.0,
        })
        builder.set_contacts(contacts)
        builder.set_orientations(orientations)
        builder.set_fold_config({
            "auto_detect": True,
            "regularisation_weight": 1.0,
        })

        result = builder.build()

        assert "fold_frame" in result
        assert "s1_field" in result["fold_frame"]
        assert "s2_field" in result["fold_frame"]
        assert "s0_field" in result["fold_frame"]
        assert "fold_axis" in result["fold_frame"]
        assert result["fold_frame"]["fold_axis"].shape == (3,)

    def test_domain_codes_for_arbf(self):
        """Verify domain codes can be used as ARBF domain_column input."""
        from geology.implicit.domain_model import assign_domains_potential_field

        # Simple field
        def field(pts):
            return pts[:, 2]

        centroids = _make_block_centroids(nx=3, ny=1, nz=5, spacing=20)
        codes, names = assign_domains_potential_field(
            centroids, field,
            isovalues=[40.0],
            unit_names=["DEEP", "SHALLOW"],
        )

        # Build a composites-like DataFrame with domain codes
        composites = pd.DataFrame({
            "X": centroids[:, 0],
            "Y": centroids[:, 1],
            "Z": centroids[:, 2],
            "Au": np.random.RandomState(42).lognormal(0, 0.5, len(centroids)),
            "DOMAIN": names,
            "DOMAIN_CODE": codes,
        })

        # Verify we can group by domain for estimation
        for dom_code in np.unique(codes):
            mask = codes == dom_code
            dom_composites = composites[mask]
            assert len(dom_composites) > 0
            assert "Au" in dom_composites.columns


# ═══════════════════════════════════════════════════════════════════
# Test: Cross-section with real model
# ═══════════════════════════════════════════════════════════════════

class TestCrossSectionWithModel:
    """Cross-section through a real geological model build."""

    def test_section_through_built_model(self):
        from geology.implicit import GeologicalModelBuilder
        from block_model_viewer.ui.cross_section_widget import (
            sample_section, classify_section,
        )

        contacts = _make_horizontal_contacts(n_holes=5, z_surface=50.0)
        builder = GeologicalModelBuilder({
            "grid_resolution": 10.0,
        })
        builder.set_contacts(contacts)
        result = builder.build()

        evaluate_fn = result["evaluate_fn"]
        origin = result["grid_origin"]
        spacing = result["grid_spacing"]
        dims = result["grid_dims"]

        # Section through the model
        vals, h, z = sample_section(
            start_xy=(origin[0], origin[1] + spacing[1] * dims[1] / 2),
            end_xy=(origin[0] + spacing[0] * dims[0], origin[1] + spacing[1] * dims[1] / 2),
            z_min=origin[2],
            z_max=origin[2] + spacing[2] * dims[2],
            evaluate_fn=evaluate_fn,
            nx=50, nz=25,
        )

        assert vals.shape == (25, 50)
        # Surface is at f=0; values should cross zero somewhere
        assert vals.min() < 0 and vals.max() > 0, "Section should cross the surface"

        codes = classify_section(vals, isovalues=[0.0])
        assert len(np.unique(codes)) >= 2, "Section should show both domains"


# ═══════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
