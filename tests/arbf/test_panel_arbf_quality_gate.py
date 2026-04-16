import numpy as np

from block_model_viewer.geostats.arbf_quality_gate import evaluate_geostatistical_gate


def test_panel_quality_gate_fails_on_miscalibrated_cv_uncertainty():
    audit = {
        "cv_slope_of_regression": 1.0,
        "cv_r_squared": 0.55,
        "cv_mean_error": 0.0,
        "cv_rmse": 0.5,
        "cv_std_resid_var": 2.6,
        "cv_cover_95": 0.78,
        "fail_ratio": 0.01,
        "mean_uncertainty_index": 0.2,
        "sigma_block": 0.4,
        "support_ratio": 0.2,
        "variance_ratio": 0.04,
        "grade_max": 10.0,
        "grade_median": 5.0,
        "n_negative_grades": 0,
        "n_blocks_active": 100,
        "measured_blocks": 0,
        "indicated_blocks": 10,
        "inferred_blocks": 80,
        "unclassified_blocks": 10,
        "num_composites": 100,
    }

    gate = evaluate_geostatistical_gate(
        audit,
        {},
        sample_values=np.array([1.0, 2.0, 3.0, 4.0], dtype=float),
        sample_coords=np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [10.0, 10.0, 0.0]],
            dtype=float,
        ),
        block_centroids=np.array([[2.0, 2.0, 0.0], [8.0, 8.0, 0.0]], dtype=float),
        domain_column="IRBF_Domain",
        available_domain_columns=["IRBF_Domain"],
        change_of_support=True,
        estimation_mode="adaptive_local_rbf",
    )

    calibration = next(check for check in gate.checks if check.code == "CV_CALIBRATION")
    assert calibration.status == "FAIL"
    assert gate.overall_status == "FAIL"
