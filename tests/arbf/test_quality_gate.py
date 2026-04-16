import numpy as np

from geostats.arbf.quality_gate import evaluate_geostatistical_gate


def _base_audit():
    return {
        "cv_slope_of_regression": 0.98,
        "cv_r_squared": 0.62,
        "cv_mean_error": 0.005,
        "conditional_bias_binned_slope": 0.97,
        "conditional_bias_max_abs_bin_bias": 0.01,
        "support_swath_panels_total": 24,
        "support_swath_panels_with_data": 24,
        "support_swath_mean_rmse": 0.05,
        "support_swath_mean_bias": 0.005,
        "n_blocks_total": 64,
        "n_blocks_estimated": 64,
        "unclassified_blocks": 2,
        "domains_enforced": True,
        "azimuth": 0.0,
        "dip": 0.0,
        "pitch": 0.0,
        "range_max": 20.0,
        "range_mid": 20.0,
        "range_min": 20.0,
        "local_drift_fallbacks": 0,
    }


def test_quality_gate_passes_for_well_behaved_run():
    audit = _base_audit()
    sample_coords = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [10.0, 10.0, 0.0],
        [5.0, 5.0, 10.0],
    ])
    block_centroids = np.array([
        [2.0, 2.0, 1.0],
        [8.0, 2.0, 1.0],
        [2.0, 8.0, 1.0],
        [8.0, 8.0, 1.0],
    ])
    sample_values = np.array([1.0, 1.2, 0.8, 1.1, 1.0])

    gate = evaluate_geostatistical_gate(
        audit,
        {},
        sample_values=sample_values,
        sample_coords=sample_coords,
        block_centroids=block_centroids,
        domain_column="domain_code",
        available_domain_columns=["domain_code"],
        change_of_support=True,
        estimation_mode="local_neighbourhood_gpr",
    )

    assert gate.overall_status == "pass"
    assert gate.actions["allow_register"] is True
    assert gate.actions["allow_jorc_export"] is True


def test_quality_gate_fails_without_domain_selection_when_domains_exist():
    audit = _base_audit()
    gate = evaluate_geostatistical_gate(
        audit,
        {},
        sample_values=np.array([1.0, 1.1, 0.9]),
        sample_coords=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
        block_centroids=np.array([[2.0, 2.0, 0.0], [8.0, 8.0, 0.0]]),
        domain_column=None,
        available_domain_columns=["domain_code", "lith_domain"],
        change_of_support=True,
    )

    assert gate.overall_status == "fail"
    assert any(check.code == "domains" and check.status == "fail" for check in gate.checks)
    assert gate.actions["allow_register"] is False


def test_quality_gate_warns_for_marginal_cv_but_keeps_registration_enabled():
    audit = _base_audit()
    audit["cv_slope_of_regression"] = 0.84
    audit["conditional_bias_binned_slope"] = 0.86

    gate = evaluate_geostatistical_gate(
        audit,
        {},
        sample_values=np.array([1.0, 1.3, 0.7, 1.1]),
        sample_coords=np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [8.0, 8.0, 0.0]]),
        block_centroids=np.array([[2.0, 2.0, 0.0], [6.0, 6.0, 0.0]]),
        domain_column="domain_code",
        available_domain_columns=["domain_code"],
        change_of_support=True,
    )

    assert gate.overall_status == "warn"
    assert gate.actions["allow_register"] is True
    assert gate.actions["allow_jorc_export"] is False


def test_quality_gate_fails_on_support_and_extrapolation_risk():
    audit = _base_audit()
    audit["support_swath_mean_rmse"] = 1.6
    audit["support_swath_mean_bias"] = 0.4
    audit["local_drift_fallbacks"] = 10
    audit["n_blocks_total"] = 100
    audit["n_blocks_estimated"] = 100

    gate = evaluate_geostatistical_gate(
        audit,
        {},
        sample_values=np.array([1.0, 1.1, 0.9, 1.2]),
        sample_coords=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [5.0, 5.0, 0.0]]),
        block_centroids=np.array([[80.0, 80.0, 0.0], [90.0, 90.0, 0.0]]),
        domain_column="domain_code",
        available_domain_columns=["domain_code"],
        change_of_support=False,
    )

    assert gate.overall_status == "fail"
    assert any(check.code == "support_mode" and check.status == "fail" for check in gate.checks)
    assert any(check.code == "extrapolation" and check.status == "fail" for check in gate.checks)


def test_quality_gate_scopes_extrapolation_to_clipped_reporting_model():
    audit = _base_audit()
    audit["clip_to_drill_footprint"] = True
    audit["footprint_buffer_ranges"] = 0.75
    audit["n_blocks_total"] = 4
    audit["n_blocks_estimated"] = 4
    audit["unclassified_blocks"] = 0

    sample_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [10.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    block_centroids = np.array(
        [
            [2.0, 2.0, 0.0],
            [8.0, 2.0, 0.0],
            [2.0, 8.0, 0.0],
            [8.0, 8.0, 0.0],
            [200.0, 200.0, 0.0],
            [220.0, 220.0, 0.0],
        ],
        dtype=np.float64,
    )

    gate = evaluate_geostatistical_gate(
        audit,
        {},
        sample_values=np.array([1.0, 1.1, 0.9, 1.2]),
        sample_coords=sample_coords,
        block_centroids=block_centroids,
        domain_column="domain_code",
        available_domain_columns=["domain_code"],
        change_of_support=True,
    )

    extrapolation = next(check for check in gate.checks if check.code == "extrapolation")
    assert extrapolation.status == "pass"
