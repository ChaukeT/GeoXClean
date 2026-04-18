"""
Phase 1 Test Suite — Implicit Geological Modelling Engine.
===========================================================

Tests:
    1. Gradient kernel derivatives (analytical vs numerical finite difference)
    2. Augmented matrix symmetry and positive-definiteness
    3. Scalar field interpolation (contact honouring)
    4. Signed distance construction (gradient and offset methods)
    5. Surface extraction (marching cubes on known sphere)
    6. Domain assignment
    7. Full integration pipeline
"""

from __future__ import annotations

import logging
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from geology.implicit.gradient_kernels import (
    phi_prime,
    phi_double_prime,
    phi_prime_over_r_limit,
    kernel_derivative_1,
    kernel_derivative_2,
    polynomial_drift,
    polynomial_drift_gradient,
)
from geology.implicit.scalar_field import (
    assemble_augmented_matrix,
    solve_augmented_system,
    evaluate_scalar_field,
    make_evaluate_fn,
)
from geology.implicit.signed_distance import (
    construct_sdf_constraints,
    extract_contacts_from_lithology,
    estimate_contact_normals,
)
from geology.implicit.contact_data import (
    ContactSet,
    StratigraphicColumn,
    dip_azimuth_to_normal,
)
from geology.implicit.surface_extraction import (
    evaluate_field_on_grid,
    extract_isosurface,
    cleanup_mesh,
)
from geology.implicit.domain_model import (
    assign_domains_potential_field,
    assign_domains_from_lithology,
)
from geology.implicit.validation import (
    check_contact_honouring,
    contact_honouring_summary,
)

from geostats.arbf.kernels import evaluate_kernel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PASS = 0
FAIL = 0
RESULTS = []


def record(name: str, passed: bool, detail: str = ""):
    global PASS, FAIL
    if passed:
        PASS += 1
        tag = "PASS"
    else:
        FAIL += 1
        tag = "FAIL"
    RESULTS.append((name, tag, detail))
    print(f"  [{tag}] {name}" + (f" -- {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════════
# 1. Gradient Kernel Tests
# ═══════════════════════════════════════════════════════════════════

def test_phi_prime_at_zero():
    """phi'(0) should be 0 for all kernels."""
    for kernel in ["spheroidal", "gaussian", "matern_32", "matern_52"]:
        r = np.array([0.0])
        val = phi_prime(r, kernel, alpha=1.0)
        record(f"phi'(0)=0 [{kernel}]", abs(val[0]) < 1e-12, f"got {val[0]:.2e}")


def test_phi_double_prime_at_zero():
    """phi''(0) should match analytical value."""
    expected = {
        "spheroidal": -2.0,   # -2*alpha with alpha=1
        "gaussian": -2.0,
        "matern_32": -3.0,
        "matern_52": -5.0 / 3.0,
    }
    for kernel, exp in expected.items():
        r = np.array([0.0])
        val = phi_double_prime(r, kernel, alpha=1.0)
        record(
            f"phi''(0)={exp} [{kernel}]",
            abs(val[0] - exp) < 1e-10,
            f"got {val[0]:.6f}",
        )


def test_phi_prime_finite_difference():
    """phi'(r) matches numerical finite difference for all kernels."""
    eps = 1e-6
    r_test = np.array([0.5, 1.0, 2.0])

    for kernel in ["spheroidal", "gaussian", "matern_32", "matern_52"]:
        phi_plus = evaluate_kernel(r_test + eps, kernel, alpha=1.0)
        phi_minus = evaluate_kernel(r_test - eps, kernel, alpha=1.0)
        numerical = (phi_plus - phi_minus) / (2 * eps)
        analytical = phi_prime(r_test, kernel, alpha=1.0)

        max_err = np.max(np.abs(numerical - analytical))
        record(
            f"phi' finite diff [{kernel}]",
            max_err < 1e-4,
            f"max_err={max_err:.2e}",
        )


def test_phi_double_prime_finite_difference():
    """phi''(r) matches numerical second derivative."""
    eps = 1e-5
    r_test = np.array([0.5, 1.0, 2.0])

    for kernel in ["spheroidal", "gaussian", "matern_32", "matern_52"]:
        pp_plus = phi_prime(r_test + eps, kernel, alpha=1.0)
        pp_minus = phi_prime(r_test - eps, kernel, alpha=1.0)
        numerical = (pp_plus - pp_minus) / (2 * eps)
        analytical = phi_double_prime(r_test, kernel, alpha=1.0)

        max_err = np.max(np.abs(numerical - analytical))
        record(
            f"phi'' finite diff [{kernel}]",
            max_err < 1e-3,
            f"max_err={max_err:.2e}",
        )


def test_phi_prime_over_r_limit():
    """lim phi'(r)/r as r→0 = phi''(0)."""
    for kernel in ["spheroidal", "gaussian", "matern_32", "matern_52"]:
        limit = phi_prime_over_r_limit(kernel, alpha=1.0)
        pp0 = phi_double_prime(np.array([0.0]), kernel, alpha=1.0)[0]
        record(
            f"phi'/r limit [{kernel}]",
            abs(limit - pp0) < 1e-10,
            f"limit={limit:.6f}, phi''(0)={pp0:.6f}",
        )


# ═══════════════════════════════════════════════════════════════════
# 2. Augmented Matrix Tests
# ═══════════════════════════════════════════════════════════════════

def test_augmented_matrix_symmetry():
    """Augmented matrix should be symmetric."""
    np.random.seed(42)
    N_v = 10
    N_g = 5
    value_coords = np.random.randn(N_v, 3) * 50
    grad_coords = np.random.randn(N_g, 3) * 50
    grad_normals = np.random.randn(N_g, 3)
    grad_normals /= np.linalg.norm(grad_normals, axis=1, keepdims=True)

    K, nv, ng = assemble_augmented_matrix(
        value_coords, grad_coords, grad_normals,
        kernel_type="spheroidal", alpha=1.0, range_=100.0,
        nugget=0.01, accuracy=1e-6, drift_type="constant",
    )

    asym = np.max(np.abs(K - K.T))
    record("Augmented matrix symmetric", asym < 1e-10, f"max asymmetry={asym:.2e}")


def test_augmented_matrix_solvable():
    """Augmented system should be solvable (kernel sub-block is SPD).

    The full augmented matrix with the polynomial zero block is a
    saddle-point system, so it is indefinite by design. The kernel
    sub-block K = [[K_vv, K_vg], [K_gv, K_gg]] should be SPD.
    """
    np.random.seed(42)
    N_v = 8
    N_g = 4
    value_coords = np.random.randn(N_v, 3) * 50
    grad_coords = np.random.randn(N_g, 3) * 50
    grad_normals = np.random.randn(N_g, 3)
    grad_normals /= np.linalg.norm(grad_normals, axis=1, keepdims=True)

    K_aug, nv, ng = assemble_augmented_matrix(
        value_coords, grad_coords, grad_normals,
        kernel_type="spheroidal", alpha=1.0, range_=100.0,
        nugget=0.01, accuracy=1e-4, drift_type="constant",
    )

    # The K_vv sub-block (value-only kernel) should be SPD
    K_vv = K_aug[:nv, :nv]
    eigenvalues_vv = np.linalg.eigvalsh(K_vv)
    min_eig_vv = eigenvalues_vv.min()
    record(
        "K_vv sub-block SPD",
        min_eig_vv > 0,
        f"min_eig={min_eig_vv:.2e}",
    )

    # Also verify the full system is solvable
    rhs = np.random.randn(K_aug.shape[0])
    rhs[-1] = 0.0  # polynomial constraint
    try:
        x = np.linalg.solve(K_aug, rhs)
        residual = np.max(np.abs(K_aug @ x - rhs))
        record("Full system solvable", residual < 1e-6, f"residual={residual:.2e}")
    except np.linalg.LinAlgError:
        record("Full system solvable", False, "LinAlgError")


# ═══════════════════════════════════════════════════════════════════
# 3. Scalar Field Interpolation Tests
# ═══════════════════════════════════════════════════════════════════

def test_interpolation_through_contacts():
    """Scalar field should be ~0 at contact points (value constraints)."""
    np.random.seed(42)
    # Contacts on a tilted plane: z = 0.5*x + 0.3*y
    N = 20
    xy = np.random.randn(N, 2) * 50
    z = 0.5 * xy[:, 0] + 0.3 * xy[:, 1]
    contact_coords = np.column_stack([xy, z])
    contact_normals = np.tile([0.0, 0.0, 1.0], (N, 1))

    value_coords, value_data, grad_coords, grad_normals = \
        construct_sdf_constraints(contact_coords, contact_normals, method="gradient")

    K, N_v, N_g = assemble_augmented_matrix(
        value_coords, grad_coords, grad_normals,
        kernel_type="spheroidal", alpha=1.0, range_=100.0,
        nugget=0.0, accuracy=1e-6,
    )

    grad_values = np.ones(N_g, dtype=np.float64)
    w, u, c = solve_augmented_system(K, value_data, grad_values, N_v, N_g)

    evaluate_fn = make_evaluate_fn(
        value_coords, grad_coords, grad_normals, w, u, c,
        "spheroidal", 1.0, 100.0,
    )

    # Evaluate at contact points
    f_at_contacts = evaluate_fn(contact_coords)
    max_misfit = np.max(np.abs(f_at_contacts))

    record(
        "Interpolation through contacts",
        max_misfit < 0.1,
        f"max |f(contact)| = {max_misfit:.4e}",
    )


def test_known_tilted_plane():
    """Scalar field for a tilted plane with known dip/azimuth."""
    # Plane: z = 10 (horizontal, dip=0)
    # Contacts at z=10
    N = 15
    np.random.seed(123)
    xy = np.random.randn(N, 2) * 30
    z = np.full(N, 10.0)
    contact_coords = np.column_stack([xy, z])
    contact_normals = np.tile([0.0, 0.0, 1.0], (N, 1))

    value_coords, value_data, grad_coords, grad_normals = \
        construct_sdf_constraints(contact_coords, contact_normals, method="gradient")

    K, N_v, N_g = assemble_augmented_matrix(
        value_coords, grad_coords, grad_normals,
        kernel_type="spheroidal", alpha=1.0, range_=100.0,
        nugget=0.0, accuracy=1e-6,
    )
    w, u, c = solve_augmented_system(K, value_data, np.ones(N_g), N_v, N_g)

    evaluate_fn = make_evaluate_fn(
        value_coords, grad_coords, grad_normals, w, u, c,
        "spheroidal", 1.0, 100.0,
    )

    # Test points above and below the plane
    above = np.array([[0.0, 0.0, 15.0]])
    below = np.array([[0.0, 0.0, 5.0]])
    on_plane = np.array([[0.0, 0.0, 10.0]])

    f_above = evaluate_fn(above)[0]
    f_below = evaluate_fn(below)[0]
    f_on = evaluate_fn(on_plane)[0]

    # f should be positive above, negative below, ~0 on plane
    record(
        "Tilted plane: f>0 above",
        f_above > 0,
        f"f(above)={f_above:.4f}",
    )
    record(
        "Tilted plane: f<0 below",
        f_below < 0,
        f"f(below)={f_below:.4f}",
    )
    record(
        "Tilted plane: f~0 on plane",
        abs(f_on) < 0.5,
        f"f(on_plane)={f_on:.4e}",
    )


# ═══════════════════════════════════════════════════════════════════
# 4. Signed Distance Tests
# ═══════════════════════════════════════════════════════════════════

def test_sdf_gradient_method():
    """Gradient method: N_c value + N_c gradient constraints."""
    N = 10
    coords = np.random.randn(N, 3)
    normals = np.tile([0, 0, 1], (N, 1)).astype(float)

    vc, vd, gc, gn = construct_sdf_constraints(coords, normals, method="gradient")

    record("Gradient SDF: N_v == N_c", vc.shape[0] == N, f"N_v={vc.shape[0]}")
    record("Gradient SDF: N_g == N_c", gc.shape[0] == N, f"N_g={gc.shape[0]}")
    record("Gradient SDF: values all 0", np.all(vd == 0.0))


def test_sdf_offset_method():
    """Offset method: 3*N_c value constraints, no gradients."""
    N = 10
    coords = np.random.randn(N, 3)
    normals = np.tile([0, 0, 1], (N, 1)).astype(float)

    vc, vd, gc, gn = construct_sdf_constraints(coords, normals, method="offset", offset_distance=2.0)

    record("Offset SDF: N_v == 3*N_c", vc.shape[0] == 3 * N, f"N_v={vc.shape[0]}")
    record("Offset SDF: N_g == 0", gc.shape[0] == 0)
    record("Offset SDF: has +eps/-eps", np.any(vd > 0) and np.any(vd < 0))


# ═══════════════════════════════════════════════════════════════════
# 5. Contact Extraction Tests
# ═══════════════════════════════════════════════════════════════════

def test_extract_contacts():
    """Extract contacts from synthetic lithology log."""
    data = pd.DataFrame({
        "hole_id": ["DH01"] * 4 + ["DH02"] * 3,
        "depth_from": [0, 10, 20, 30, 0, 15, 25],
        "depth_to": [10, 20, 30, 40, 15, 25, 35],
        "lith_code": ["OX", "OX", "TR", "FR", "OX", "TR", "FR"],
        "X": [100, 100, 100, 100, 200, 200, 200],
        "Y": [100, 100, 100, 100, 200, 200, 200],
        "Z": [90, 80, 70, 60, 85, 75, 65],
    })

    contacts = extract_contacts_from_lithology(data)

    record(
        "Contact extraction count",
        len(contacts) == 4,
        f"got {len(contacts)} contacts (expected 4)",
    )

    # Check surfaces found
    surfaces = set(contacts["surface_name"].unique())
    record(
        "Contact surfaces found",
        "OX_TR" in surfaces and "TR_FR" in surfaces,
        f"surfaces: {surfaces}",
    )


def test_extract_contacts_with_grouping():
    """Contact extraction with lithology grouping."""
    data = pd.DataFrame({
        "hole_id": ["DH01"] * 3,
        "depth_from": [0, 10, 20],
        "depth_to": [10, 20, 30],
        "lith_code": ["OX_BIF", "OX-BIF", "FRESH"],
        "X": [100, 100, 100],
        "Y": [100, 100, 100],
        "Z": [90, 80, 70],
    })

    grouping = {
        "OXIDE": ["OX_BIF", "OX-BIF"],
        "FRESH": ["FRESH"],
    }

    contacts = extract_contacts_from_lithology(data, grouping=grouping)

    record(
        "Grouped contacts: 1 transition",
        len(contacts) == 1,
        f"got {len(contacts)}",
    )


# ═══════════════════════════════════════════════════════════════════
# 6. Surface Extraction Tests
# ═══════════════════════════════════════════════════════════════════

def test_extract_sphere():
    """Extract a sphere surface from a known SDF."""
    # SDF for a sphere of radius 5 centered at origin
    nx, ny, nz = 30, 30, 30
    res = 0.5
    origin = np.array([-8.0, -8.0, -8.0])
    spacing = np.array([res, res, res])

    x = origin[0] + np.arange(nx) * res
    y = origin[1] + np.arange(ny) * res
    z = origin[2] + np.arange(nz) * res
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    sdf = np.sqrt(xx**2 + yy**2 + zz**2) - 5.0  # positive outside, negative inside

    verts, faces = extract_isosurface(sdf, origin, spacing, isovalue=0.0)

    record("Sphere: got vertices", verts.shape[0] > 100, f"V={verts.shape[0]}")
    record("Sphere: got triangles", faces.shape[0] > 50, f"F={faces.shape[0]}")

    # Check radius of extracted vertices
    radii = np.linalg.norm(verts, axis=1)
    mean_r = np.mean(radii)
    record(
        "Sphere: mean radius ~5",
        abs(mean_r - 5.0) < 0.5,
        f"mean_r={mean_r:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════
# 7. Domain Assignment Tests
# ═══════════════════════════════════════════════════════════════════

def test_domain_potential_field():
    """Domain assignment from potential field."""
    # Simple: f(x) = x[2] (z coordinate)
    def f(pts): return pts[:, 2]

    centroids = np.array([
        [0, 0, -5],   # below both surfaces → unit C
        [0, 0, 5],    # between surfaces → unit B
        [0, 0, 15],   # above both surfaces → unit A
    ], dtype=float)

    codes, names = assign_domains_potential_field(
        centroids, f,
        isovalues=[0.0, 10.0],
        unit_names=["Below", "Middle", "Above"],
    )

    record("Domain: z=-5 -> Below", names[0] == "Below", f"got {names[0]}")
    record("Domain: z=5 -> Middle", names[1] == "Middle", f"got {names[1]}")
    record("Domain: z=15 -> Above", names[2] == "Above", f"got {names[2]}")


def test_domain_from_lithology():
    """Domain codes from lithology grouping."""
    df = pd.DataFrame({
        "lith_code": ["OX", "OX", "TR", "FR", "FR"],
    })
    grouping = {"OXIDE": ["OX"], "TRANS": ["TR"], "FRESH": ["FR"]}

    codes = assign_domains_from_lithology(df, grouping)

    record("Lithology domain: OX -> 0", codes[0] == 0, f"got {codes[0]}")
    record("Lithology domain: TR -> 1", codes[2] == 1, f"got {codes[2]}")
    record("Lithology domain: FR -> 2", codes[3] == 2, f"got {codes[3]}")


# ═══════════════════════════════════════════════════════════════════
# 8. Validation Tests
# ═══════════════════════════════════════════════════════════════════

def test_contact_honouring():
    """Contact honouring QC with perfect model."""
    contacts = pd.DataFrame({
        "X": [0.0, 10.0, 20.0],
        "Y": [0.0, 0.0, 0.0],
        "Z": [0.0, 0.0, 0.0],
        "surface_name": ["S1", "S1", "S1"],
        "hole_id": ["DH01", "DH02", "DH03"],
    })

    # Perfect model: f(x) = 0 everywhere
    def f(pts): return np.zeros(pts.shape[0])

    qc = check_contact_honouring(contacts, f, tolerance=1.0)
    summary = contact_honouring_summary(qc)

    record(
        "Contact honouring 100%",
        summary["pct_honoured"] == 100.0,
        f"got {summary['pct_honoured']:.1f}%",
    )


# ═══════════════════════════════════════════════════════════════════
# 9. Integration Test
# ═══════════════════════════════════════════════════════════════════

def test_full_pipeline():
    """Full pipeline: contacts → surface → domain assignment."""
    from geology.implicit.geological_model import GeologicalModelBuilder

    # Create synthetic contacts for a horizontal surface at z=50
    np.random.seed(42)
    N = 25
    xy = np.random.randn(N, 2) * 30
    z = np.full(N, 50.0) + np.random.randn(N) * 0.5  # slight noise
    contacts_df = pd.DataFrame({
        "hole_id": [f"DH{i:02d}" for i in range(N)],
        "depth": z,
        "X": xy[:, 0],
        "Y": xy[:, 1],
        "Z": z,
        "unit_above": "OXIDE",
        "unit_below": "FRESH",
        "surface_name": "OXIDE_FRESH",
    })

    config = {
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "range_max": 80.0,
        "range_mid": 80.0,
        "range_min": 40.0,
        "nugget": 0.0,
        "accuracy": 1e-4,
        "grid_resolution": 5.0,
        "constraint_method": "gradient",
        "tolerance": 2.0,
    }

    builder = GeologicalModelBuilder(config)
    builder.set_contacts(contacts_df)

    result = builder.build()

    record("Pipeline: got surfaces", len(result["surfaces"]) > 0)
    record("Pipeline: got scalar field", result["scalar_field"] is not None)
    record(
        "Pipeline: contact honouring",
        result["contact_honouring_summary"]["pct_honoured"] >= 80.0,
        f"{result['contact_honouring_summary']['pct_honoured']:.1f}%",
    )
    record("Pipeline: audit record", "run_id" in result["audit_record"])


# ═══════════════════════════════════════════════════════════════════
# 10. Dip/Azimuth Conversion
# ═══════════════════════════════════════════════════════════════════

def test_dip_azimuth_to_normal():
    """Verify dip/azimuth → normal vector conversion."""
    # Horizontal surface (dip=0): normal should be [0, 0, 1]
    n = dip_azimuth_to_normal(0.0, 0.0)
    record("Dip 0: vertical normal", abs(n[2] - 1.0) < 1e-10, f"n={n}")

    # Vertical surface dipping north (dip=90, azimuth=0): normal ~ [0, 1, 0]
    n = dip_azimuth_to_normal(90.0, 0.0)
    record("Dip 90, Az 0: north normal", abs(n[1] - 1.0) < 1e-10, f"n={n}")

    # Vertical surface dipping east (dip=90, azimuth=90): normal ~ [1, 0, 0]
    n = dip_azimuth_to_normal(90.0, 90.0)
    record("Dip 90, Az 90: east normal", abs(n[0] - 1.0) < 1e-10, f"n={n}")


# ═══════════════════════════════════════════════════════════════════
# Run All
# ═══════════════════════════════════════════════════════════════════

def run_all():
    print("\n" + "=" * 70)
    print("  GeoX Implicit Geological Modelling — Phase 1 Test Suite")
    print("=" * 70)

    print("\n--- 1. Gradient Kernel Tests ---")
    test_phi_prime_at_zero()
    test_phi_double_prime_at_zero()
    test_phi_prime_finite_difference()
    test_phi_double_prime_finite_difference()
    test_phi_prime_over_r_limit()

    print("\n--- 2. Augmented Matrix Tests ---")
    test_augmented_matrix_symmetry()
    test_augmented_matrix_solvable()

    print("\n--- 3. Scalar Field Interpolation Tests ---")
    test_interpolation_through_contacts()
    test_known_tilted_plane()

    print("\n--- 4. Signed Distance Tests ---")
    test_sdf_gradient_method()
    test_sdf_offset_method()

    print("\n--- 5. Contact Extraction Tests ---")
    test_extract_contacts()
    test_extract_contacts_with_grouping()

    print("\n--- 6. Surface Extraction Tests ---")
    test_extract_sphere()

    print("\n--- 7. Domain Assignment Tests ---")
    test_domain_potential_field()
    test_domain_from_lithology()

    print("\n--- 8. Validation Tests ---")
    test_contact_honouring()

    print("\n--- 9. Dip/Azimuth Tests ---")
    test_dip_azimuth_to_normal()

    print("\n--- 10. Integration Test ---")
    test_full_pipeline()

    print("\n" + "=" * 70)
    print(f"  RESULTS: {PASS} passed, {FAIL} failed, {PASS + FAIL} total")
    print("=" * 70 + "\n")

    return FAIL == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
