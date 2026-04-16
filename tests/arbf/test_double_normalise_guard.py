"""
Test: Double normal-score transform guard.

Verifies that the ARBF engine detects already-normalised data and
skips the forward transform to prevent double-normalisation.
"""

import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from geostats.arbf.transforms import (
    detect_already_normal_scored,
    normal_score_transform,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_detect_raw_grades():
    """Raw grade values (ppm) should NOT be flagged."""
    rng = np.random.default_rng(42)
    # Lognormal grades: mean ~5000, all positive
    raw = np.exp(rng.normal(8.5, 1.5, size=500))
    assert not detect_already_normal_scored(raw), \
        "Raw lognormal grades should NOT be detected as normal-scored"
    print("PASS: raw lognormal grades not flagged")


def test_detect_low_grade():
    """Low-grade percentages (0-5%) should NOT be flagged."""
    rng = np.random.default_rng(42)
    raw = rng.uniform(0.1, 5.0, size=500)
    assert not detect_already_normal_scored(raw), \
        "Low-grade percentages should NOT be detected as normal-scored"
    print("PASS: low-grade percentages not flagged")


def test_detect_ns_values():
    """Normal-scored values should be flagged."""
    rng = np.random.default_rng(42)
    raw = np.exp(rng.normal(5.0, 1.0, size=500))
    ns_values, _ = normal_score_transform(raw, seed=42)
    assert detect_already_normal_scored(ns_values), \
        "Normal-scored values SHOULD be detected"
    print("PASS: normal-scored values detected")


def test_detect_standard_normal():
    """Standard normal samples should be flagged."""
    rng = np.random.default_rng(42)
    z = rng.standard_normal(500)
    assert detect_already_normal_scored(z), \
        "Standard normal samples SHOULD be detected"
    print("PASS: standard normal samples detected")


def test_no_false_positive_on_centred_residuals():
    """Residuals centred at 0 but with large range should NOT be flagged."""
    rng = np.random.default_rng(42)
    # Residuals with std ~1 but range well beyond [-5, 5]
    residuals = rng.normal(0, 3.0, size=500)
    assert not detect_already_normal_scored(residuals), \
        "Wide residuals should NOT be flagged (range too large)"
    print("PASS: wide-range residuals not flagged")


def test_no_false_positive_on_positive_only():
    """Positive-only data near 1.0 should NOT be flagged."""
    rng = np.random.default_rng(42)
    # e.g. porosity values 0.0 - 1.0
    porosity = rng.beta(2, 5, size=500)
    assert not detect_already_normal_scored(porosity), \
        "Positive-only porosity values should NOT be flagged"
    print("PASS: positive-only values not flagged")


def test_engine_guard_prevents_double_transform():
    """Full integration: engine should skip transform on NS input."""
    from geostats.arbf.engine import ARBFEstimator

    rng = np.random.default_rng(42)

    # Generate raw data and pre-transform
    N = 200
    coords = rng.uniform(0, 100, (N, 3))
    raw_grades = np.exp(rng.normal(3.0, 0.8, N))
    ns_values, _ = normal_score_transform(raw_grades, seed=42, coords=coords)

    # Block model (small for speed)
    bx = np.arange(5) * 20 + 10
    BX, BY, BZ = np.meshgrid(bx, bx, bx[:3], indexing="ij")
    centroids = np.column_stack([BX.ravel(), BY.ravel(), BZ.ravel()])
    sizes = np.array([20.0, 20.0, 20.0])

    config = {
        "kernel_type": "spheroidal",
        "alpha": 1.5,
        "sill": 1.0,
        "nugget": 0.1,
        "range_max": 50.0,
        "range_mid": 50.0,
        "range_min": 50.0,
        "drift_type": "constant",
        "accuracy": 1e-6,
        "use_normal_score": True,  # <-- user accidentally set this
        "change_of_support": False,
        "run_cv": False,
        "verbose": False,
        "seed": 42,
    }

    est = ARBFEstimator(config)
    est.set_composites(coords, ns_values)  # <-- passing NS values
    est.set_block_model(centroids, sizes)

    # Capture log messages
    import io
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.WARNING)
    eng_logger = logging.getLogger("geostats.arbf.engine")
    eng_logger.addHandler(handler)

    results = est.estimate()
    eng_logger.removeHandler(handler)

    log_output = log_capture.getvalue()
    assert "DOUBLE-TRANSFORM GUARD" in log_output, \
        f"Expected double-transform warning in log, got: {log_output[:200]}"

    # Grades should be in NS-space range (roughly -3 to +3), NOT
    # in the crazy back-transform-of-double-NS range
    grades = results.grades
    valid = np.isfinite(grades) & (grades != 0)
    if np.any(valid):
        assert np.max(np.abs(grades[valid])) < 10.0, \
            f"Grades should be in NS range, got max={np.max(np.abs(grades[valid])):.1f}"

    print("PASS: engine detected double-normalise and skipped transform")


def test_engine_normal_path_works():
    """Raw grades + use_normal_score=True should still work normally."""
    from geostats.arbf.engine import ARBFEstimator

    rng = np.random.default_rng(42)

    N = 200
    coords = rng.uniform(0, 100, (N, 3))
    raw_grades = np.exp(rng.normal(3.0, 0.8, N))  # raw, not NS

    bx = np.arange(5) * 20 + 10
    BX, BY, BZ = np.meshgrid(bx, bx, bx[:3], indexing="ij")
    centroids = np.column_stack([BX.ravel(), BY.ravel(), BZ.ravel()])
    sizes = np.array([20.0, 20.0, 20.0])

    config = {
        "kernel_type": "spheroidal",
        "alpha": 1.5,
        "sill": 1.0,
        "nugget": 0.1,
        "range_max": 50.0,
        "range_mid": 50.0,
        "range_min": 50.0,
        "drift_type": "constant",
        "accuracy": 1e-6,
        "use_normal_score": True,
        "change_of_support": False,
        "run_cv": False,
        "verbose": False,
        "seed": 42,
    }

    est = ARBFEstimator(config)
    est.set_composites(coords, raw_grades)
    est.set_block_model(centroids, sizes)

    results = est.estimate()
    grades = results.grades
    valid = np.isfinite(grades) & (grades != 0)

    # Grades should be back in original scale (positive, similar range to input)
    assert np.all(grades[valid] > 0), "Back-transformed grades should be positive"
    assert np.mean(grades[valid]) > 1.0, "Mean grade should be > 1 (original scale)"

    print("PASS: normal path (raw + use_normal_score=True) works correctly")


if __name__ == "__main__":
    test_detect_raw_grades()
    test_detect_low_grade()
    test_detect_ns_values()
    test_detect_standard_normal()
    test_no_false_positive_on_centred_residuals()
    test_no_false_positive_on_positive_only()
    test_engine_guard_prevents_double_transform()
    test_engine_normal_path_works()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)
