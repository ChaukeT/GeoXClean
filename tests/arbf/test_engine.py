"""Integration test: ARBF engine end-to-end."""

import numpy as np
import pytest

from geostats.arbf.engine import ARBFEstimator


def _make_synthetic_block_model(seed=42):
    """Create a synthetic block model and composites for testing."""
    rng = np.random.RandomState(seed)

    # Composites: scattered in a 100x100x50 domain
    N = 100
    composite_coords = np.column_stack([
        rng.uniform(0, 100, N),
        rng.uniform(0, 100, N),
        rng.uniform(0, 50, N),
    ])
    # Smooth spatial field + noise
    composite_values = (
        0.5 * np.sin(composite_coords[:, 0] / 20.0)
        + 0.3 * np.cos(composite_coords[:, 1] / 30.0)
        + 2.0
        + rng.normal(0, 0.1, N)
    )

    # Block model: 10x10x5 grid
    nx, ny, nz = 10, 10, 5
    dx, dy, dz = 10.0, 10.0, 10.0
    centroids = np.array([
        [ix * dx + dx / 2, iy * dy + dy / 2, iz * dz + dz / 2]
        for ix in range(nx)
        for iy in range(ny)
        for iz in range(nz)
    ])
    block_sizes = np.array([dx, dy, dz])

    return composite_coords, composite_values, centroids, block_sizes


def _make_vertical_hole_lognormal_case(seed=20260321):
    """Build a drillhole-style skewed case that stresses NS masking."""
    rng = np.random.default_rng(seed)
    center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    azimuth_rad = np.deg2rad(236.3)
    rotation = np.array(
        [
            [np.sin(azimuth_rad), np.cos(azimuth_rad)],
            [np.cos(azimuth_rad), -np.sin(azimuth_rad)],
        ],
        dtype=np.float64,
    )
    range_max, range_mid, range_min = 123.9, 92.2, 28.0
    nugget_fraction = 0.06297780428508051
    structured_sill = 0.9363666305607734
    clip_max = 32468.3333
    reference_range = 80.0

    hole_u = np.linspace(-180.0, 180.0, 4)
    hole_v = np.linspace(-120.0, 120.0, 3)
    collars_local = np.array(
        np.meshgrid(hole_u, hole_v, indexing="ij"),
        dtype=np.float64,
    ).reshape(2, -1).T
    collars_xy = collars_local + rng.normal(0.0, 8.0, size=collars_local.shape)

    coords = []
    for xy in collars_xy:
        for interval_index in range(30):
            mid_depth = (interval_index + 0.5) * 10.0
            coords.append([xy[0], xy[1], 300.0 - mid_depth])
    coords = np.asarray(coords, dtype=np.float64)

    directions = rng.normal(0.0, 1.0, size=(180, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(directions))

    def _raw_field(xyz: np.ndarray) -> np.ndarray:
        local_xy = (xyz[:, :2] - center[:2]) @ rotation
        local = np.column_stack(
            [
                local_xy[:, 0] * (reference_range / range_max),
                local_xy[:, 1] * (reference_range / range_mid),
                (xyz[:, 2] - center[2]) * (reference_range / range_min),
            ],
        )
        values = np.zeros(len(xyz), dtype=np.float64)
        for direction, phase in zip(directions, phases):
            values += np.cos(
                2.0 * np.pi * (local @ direction) / reference_range + phase,
            )
        return values

    raw = _raw_field(coords)
    structured = (
        (raw - float(np.mean(raw)))
        * np.sqrt(structured_sill)
        / max(float(np.std(raw)), 1e-12)
    )
    gaussian = structured + rng.normal(
        0.0, np.sqrt(nugget_fraction), size=len(structured),
    )
    grades = np.clip(np.exp(8.0 + 0.9 * gaussian), 0.0, clip_max)

    return coords, grades


class TestARBFEstimator:
    """End-to-end integration tests."""

    def test_basic_estimation(self):
        """ARBF estimation runs without errors and produces valid results."""
        coords, values, centroids, block_sizes = _make_synthetic_block_model()

        config = {
            "kernel_type": "spheroidal",
            "alpha": 1.0,
            "drift_type": "constant",
            "nugget": 0.05,
            "accuracy": 1e-6,
            "n_subdomains": 5,
            "max_samples": 50,
            "min_samples": 4,
            "range_max": 50.0,
            "range_mid": 50.0,
            "range_min": 50.0,
            "discretisation": "fixed",
            "discretisation_density": 8,
            "run_cv": False,
            "change_of_support": False,
            "verbose": False,
        }

        estimator = ARBFEstimator(config)
        estimator.set_composites(coords, values)
        estimator.set_block_model(centroids, block_sizes)

        results = estimator.estimate()

        assert "grades" in results
        assert "variances" in results
        assert "classifications" in results
        assert "audit_record" in results

        B = len(centroids)
        assert len(results["grades"]) == B
        assert len(results["variances"]) == B
        assert len(results["classifications"]) == B

        # Variances must be non-negative
        assert np.all(results["variances"] >= 0)

        # Classification codes must be in {0, 1, 2, 3}
        assert np.all(np.isin(results["classifications"], [0, 1, 2, 3]))

    def test_with_cross_validation(self):
        """ARBF with CV enabled produces validation results."""
        coords, values, centroids, block_sizes = _make_synthetic_block_model()

        config = {
            "kernel_type": "spheroidal",
            "alpha": 1.0,
            "drift_type": "constant",
            "nugget": 0.05,
            "n_subdomains": 4,
            "max_samples": 40,
            "range_max": 50.0,
            "range_mid": 50.0,
            "range_min": 50.0,
            "discretisation": "fixed",
            "discretisation_density": 8,
            "run_cv": True,
            "cv_max_samples": 50,
            "change_of_support": False,
            "verbose": False,
        }

        estimator = ARBFEstimator(config)
        estimator.set_composites(coords, values)
        estimator.set_block_model(centroids, block_sizes)

        results = estimator.estimate()

        cv = results.get("cv_result")
        assert cv is not None
        assert cv.n_samples > 0
        assert cv.rmse >= 0

    def test_with_change_of_support(self):
        """ARBF with change-of-support produces corrected estimates."""
        coords, values, centroids, block_sizes = _make_synthetic_block_model()

        config = {
            "kernel_type": "spheroidal",
            "alpha": 1.0,
            "nugget": 0.05,
            "n_subdomains": 4,
            "max_samples": 40,
            "range_max": 50.0,
            "range_mid": 50.0,
            "range_min": 50.0,
            "discretisation": "fixed",
            "discretisation_density": 8,
            "run_cv": False,
            "change_of_support": True,
            "verbose": False,
        }

        estimator = ARBFEstimator(config)
        estimator.set_composites(coords, values)
        estimator.set_block_model(centroids, block_sizes)

        results = estimator.estimate()

        cos = results.get("cos_result")
        assert cos is not None
        assert cos.support_ratio <= 1.0  # r < 1 always

    def test_audit_record(self):
        """Audit record is populated and exportable."""
        coords, values, centroids, block_sizes = _make_synthetic_block_model()

        config = {
            "kernel_type": "spheroidal",
            "n_subdomains": 3,
            "max_samples": 40,
            "range_max": 50.0,
            "range_mid": 50.0,
            "range_min": 50.0,
            "discretisation_density": 8,
            "run_cv": False,
            "change_of_support": False,
            "verbose": False,
            "operator": "Test Geologist",
        }

        estimator = ARBFEstimator(config)
        estimator.set_composites(coords, values)
        estimator.set_block_model(centroids, block_sizes)

        results = estimator.estimate()
        audit = results["audit_record"]

        # Can convert to JSON
        json_str = audit.to_json()
        assert len(json_str) > 100
        assert "ARBF" in audit.software_version

        # Can generate JORC report
        report = audit.to_jorc_table1_section3()
        assert "JORC TABLE 1" in report
        assert "Test Geologist" in report

        # Data hash is populated
        assert len(audit.data_hash) == 64  # SHA-256 hex

    def test_missing_composites_raises(self):
        """Calling estimate without composites raises ValueError."""
        estimator = ARBFEstimator({})
        with pytest.raises(ValueError, match="Composite data not set"):
            estimator.estimate()

    def test_normal_score_transform(self):
        """ARBF with normal-score transform runs correctly."""
        coords, values, centroids, block_sizes = _make_synthetic_block_model()
        # Make values positively skewed
        values = np.abs(values) + 0.1

        config = {
            "kernel_type": "spheroidal",
            "n_subdomains": 4,
            "max_samples": 40,
            "range_max": 50.0,
            "range_mid": 50.0,
            "range_min": 50.0,
            "discretisation_density": 8,
            "use_normal_score": True,
            "run_cv": False,
            "change_of_support": False,
            "verbose": False,
        }

        estimator = ARBFEstimator(config)
        estimator.set_composites(coords, values)
        estimator.set_block_model(centroids, block_sizes)

        results = estimator.estimate()

        # Grades should be back-transformed to positive values
        # (may not be strictly positive due to interpolation, but should be close)
        assert np.median(results["grades"]) > 0

    def test_normal_score_mask_is_opt_in_for_estimation(self):
        """NS estimation should not blank the reporting model by default."""
        coords, values = _make_vertical_hole_lognormal_case()

        config = {
            "kernel_type": "spherical",
            "alpha": 1.0,
            "drift_type": "constant",
            "sill": 0.9363666305607734,
            "nugget": 0.06297780428508051,
            "accuracy": 1e-6,
            "n_subdomains": 0,
            "max_samples": 300,
            "min_samples": 4,
            "range_max": 123.9,
            "range_mid": 92.2,
            "range_min": 28.0,
            "azimuth": 236.3,
            "dip": 0.0,
            "pitch": 0.0,
            "discretisation": "fixed",
            "discretisation_density": 27,
            "run_cv": False,
            "change_of_support": True,
            "use_normal_score": True,
            "use_lva": True,
            "lva_source": "data",
            "estimation_mode": "local_neighbourhood_gpr",
            "local_search_radii": (0.75, 1.5, 3.0),
            "balanced_neighbourhood_selection": True,
            "search_min_octants": 3,
            "search_min_octants_linear": 4,
            "max_samples_per_octant": 4,
            "parallel": False,
            "verbose": False,
        }

        point_support_targets = coords[:120].copy()
        zero_block_sizes = np.zeros(3, dtype=np.float64)

        estimator = ARBFEstimator(config)
        estimator.set_composites(coords, values)
        estimator.set_block_model(point_support_targets, zero_block_sizes)
        result = estimator.estimate()

        masked_estimator = ARBFEstimator(
            {**config, "mask_uninformed_ns_blocks": True},
        )
        masked_estimator.set_composites(coords, values)
        masked_estimator.set_block_model(point_support_targets, zero_block_sizes)
        masked_result = masked_estimator.estimate()

        assert np.isfinite(result["grades"]).sum() == len(point_support_targets)
        assert np.isfinite(masked_result["grades"]).sum() < len(point_support_targets)

    def test_footprint_clip_limits_reporting_model(self):
        """Footprint clipping must change reporting counts, not just UI state."""
        rng = np.random.default_rng(1234)
        coords = rng.uniform([0.0, 0.0, 0.0], [40.0, 40.0, 20.0], size=(80, 3))
        values = 1.0 + 0.2 * coords[:, 0] / 40.0 + 0.1 * coords[:, 1] / 40.0

        xs = np.arange(18) * 10.0 + 5.0
        ys = np.arange(18) * 10.0 + 5.0
        zs = np.arange(6) * 10.0 + 5.0
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        centroids = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        block_sizes = np.array([10.0, 10.0, 10.0], dtype=np.float64)

        config = {
            "kernel_type": "spheroidal",
            "alpha": 1.0,
            "drift_type": "constant",
            "nugget": 0.05,
            "n_subdomains": 0,
            "max_samples": 40,
            "min_samples": 4,
            "range_max": 20.0,
            "range_mid": 20.0,
            "range_min": 15.0,
            "discretisation_density": 8,
            "run_cv": False,
            "change_of_support": False,
            "clip_to_drill_footprint": True,
            "footprint_buffer_ranges": 1.0,
            "verbose": False,
        }

        estimator = ARBFEstimator(config)
        estimator.set_composites(coords, values)
        estimator.set_block_model(centroids, block_sizes)
        results = estimator.estimate()

        assert len(results["grades"]) == len(centroids)
        assert results["audit_record"].clip_to_drill_footprint is True
        assert results["audit_record"].n_blocks_total < len(centroids)
        assert results["diagnostics"]["n_blocks_total"] == results["audit_record"].n_blocks_total
        assert results["diagnostics"]["n_blocks_estimated"] <= results["audit_record"].n_blocks_total

    def test_utm_scale_coordinates(self):
        """Regression: UTM-scale coords must NOT collapse to constant predictions.

        Prior to the fix, large sub-domain radii (~200 km from k-means on
        UTM data) caused max_lag = 0.8*radius ≈ 160 km.  With 15 lag bins
        each >10 km wide, the first bin already exceeded the true
        correlation range (~100 m).  The experimental variogram appeared
        flat → optimizer fitted sill≈0 (nugget-only) → every prediction
        collapsed to the mean with zero variance.
        """
        rng = np.random.RandomState(99)

        # Simulate drillholes in UTM-like coordinates
        N = 200
        composite_coords = np.column_stack([
            rng.uniform(499000, 501000, N),  # 2 km spread in X
            rng.uniform(6000000, 6002000, N),  # 2 km spread in Y
            rng.uniform(100, 200, N),  # 100 m spread in Z
        ])
        # Spatially structured grade field
        composite_values = (
            3.0
            + 1.5 * np.sin(composite_coords[:, 0] / 200.0)
            + 0.8 * np.cos(composite_coords[:, 1] / 300.0)
            + rng.normal(0, 0.15, N)
        )

        # Block model covering the composite area
        nx, ny, nz = 10, 10, 5
        dx, dy, dz = 200.0, 200.0, 20.0
        x0 = composite_coords[:, 0].min()
        y0 = composite_coords[:, 1].min()
        z0 = composite_coords[:, 2].min()
        centroids = np.array([
            [x0 + ix * dx + dx / 2, y0 + iy * dy + dy / 2, z0 + iz * dz + dz / 2]
            for ix in range(nx)
            for iy in range(ny)
            for iz in range(nz)
        ])
        block_sizes = np.array([dx, dy, dz])

        config = {
            "kernel_type": "spheroidal",
            "alpha": 1.0,
            "drift_type": "constant",
            "nugget": 0.05,
            "n_subdomains": 4,
            "max_samples": 100,
            "min_samples": 4,
            "range_max": 300.0,
            "range_mid": 300.0,
            "range_min": 300.0,
            "discretisation": "fixed",
            "discretisation_density": 8,
            "run_cv": False,
            "change_of_support": False,
            "verbose": False,
        }

        estimator = ARBFEstimator(config)
        estimator.set_composites(composite_coords, composite_values)
        estimator.set_block_model(centroids, block_sizes)

        results = estimator.estimate()
        grades = results["grades"]
        variances = results["variances"]

        # KEY ASSERTIONS: grades must NOT be constant
        assert np.std(grades) > 0.01, (
            f"Grades collapsed to constant: std={np.std(grades):.6f}, "
            f"mean={np.mean(grades):.4f}"
        )

        # Variances must NOT all be zero
        assert np.max(variances) > 1e-8, (
            f"All variances zero: max={np.max(variances):.2e}"
        )
