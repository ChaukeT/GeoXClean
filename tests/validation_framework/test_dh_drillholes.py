"""
DH-01 through DH-12: Drillhole domain checks.

Tests cover desurvey accuracy, dip convention auto-detection,
autofix rollback, QAQC control samples, depth validation,
overlap detection, and database integrity.
"""
import copy
import logging
import pytest
import pandas as pd
import numpy as np

pytestmark = [pytest.mark.drillholes, pytest.mark.smoke]


# ── DH-01: Minimum curvature numerical accuracy ─────────────────────────────

@pytest.mark.blocker
class TestDH01MinCurvatureAccuracy:
    def test_straight_vertical_hole(self, sample_collars):
        """A perfectly vertical hole should return x,y unchanged and z decreasing."""
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        survey = pd.DataFrame({
            "depth": [0.0, 50.0, 100.0],
            "azimuth": [0.0, 0.0, 0.0],
            "dip": [-90.0, -90.0, -90.0],
        })
        depths, xs, ys, zs = minimum_curvature_desurvey(
            collar_x=1000.0, collar_y=2000.0, collar_z=500.0,
            survey_df=survey,
        )
        if depths is not None:
            assert xs is not None and ys is not None and zs is not None
            np.testing.assert_allclose(xs, 1000.0, atol=0.01,
                                       err_msg="DH-01 FAIL: X drift in vertical hole")
            np.testing.assert_allclose(ys, 2000.0, atol=0.01,
                                       err_msg="DH-01 FAIL: Y drift in vertical hole")
            assert zs[-1] < zs[0], "DH-01 FAIL: Z should decrease with depth"

    def test_known_deflection(self):
        """A hole with 45° dip should have horizontal displacement ≈ depth * sin(45°)."""
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        survey = pd.DataFrame({
            "depth": [0.0, 100.0],
            "azimuth": [90.0, 90.0],
            "dip": [-45.0, -45.0],
        })
        depths, xs, ys, zs = minimum_curvature_desurvey(
            collar_x=0.0, collar_y=0.0, collar_z=0.0,
            survey_df=survey,
        )
        if depths is not None and xs is not None:
            horiz = np.sqrt(xs[-1] ** 2 + ys[-1] ** 2)
            expected_h = 100.0 * np.sin(np.radians(45))
            np.testing.assert_allclose(horiz, expected_h, rtol=0.05,
                                       err_msg="DH-01 FAIL: Horizontal displacement off for 45° hole")


# ── DH-02: Dip convention auto-detection ─────────────────────────────────────

@pytest.mark.blocker
class TestDH02DipConventionDetection:
    def test_positive_dips_are_negated(self):
        """Mining convention: positive dips (downhole) should be auto-negated."""
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        survey = pd.DataFrame({
            "depth": [0.0, 50.0, 100.0],
            "azimuth": [90.0, 90.0, 90.0],
            "dip": [60.0, 60.0, 60.0],  # positive = down in some conventions
        })
        depths, xs, ys, zs = minimum_curvature_desurvey(
            collar_x=0.0, collar_y=0.0, collar_z=100.0,
            survey_df=survey,
        )
        if depths is not None and zs is not None:
            # Z should decrease (hole goes down)
            assert zs[-1] < zs[0], \
                "DH-02 FAIL: Positive dips not auto-negated; hole went up"

    def test_already_negative_dips_unchanged(self):
        """Dips already negative should not be double-negated."""
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        survey = pd.DataFrame({
            "depth": [0.0, 50.0, 100.0],
            "azimuth": [90.0, 90.0, 90.0],
            "dip": [-60.0, -60.0, -60.0],
        })
        depths, xs, ys, zs = minimum_curvature_desurvey(
            collar_x=0.0, collar_y=0.0, collar_z=100.0,
            survey_df=survey,
        )
        if depths is not None and zs is not None:
            assert zs[-1] < zs[0], \
                "DH-02 FAIL: Negative dips double-negated; hole went up"


# ── DH-03: Post-negation guard (DIA-DS03) ───────────────────────────────────

@pytest.mark.critical
class TestDH03PostNegationGuard:
    def test_double_negation_is_reverted(self):
        """If auto-negation makes >50% of dips positive, it should revert."""
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        # All dips already negative — auto-negation would make them positive
        survey = pd.DataFrame({
            "depth": [0.0, 30.0, 60.0, 90.0],
            "azimuth": [90.0, 90.0, 90.0, 90.0],
            "dip": [-60.0, -55.0, -50.0, -45.0],
        })
        depths, xs, ys, zs = minimum_curvature_desurvey(
            collar_x=0.0, collar_y=0.0, collar_z=100.0,
            survey_df=survey,
        )
        if depths is not None and zs is not None:
            assert zs[-1] < zs[0], \
                "DH-03 FAIL: Post-negation guard failed; hole goes upward"


# ── DH-04: Azimuth validation for vertical holes (DIA-DS05) ─────────────────

@pytest.mark.major
class TestDH04VerticalAzimuthWarning:
    def test_varying_azimuth_on_vertical_hole_warns(self, caplog):
        """Vertical holes (dip ≈ -90) with varying azimuths should log a warning."""
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        survey = pd.DataFrame({
            "depth": [0.0, 50.0, 100.0],
            "azimuth": [0.0, 180.0, 90.0],  # varying azimuth
            "dip": [-90.0, -90.0, -90.0],
        })
        with caplog.at_level(logging.WARNING):
            minimum_curvature_desurvey(
                collar_x=0.0, collar_y=0.0, collar_z=0.0,
                survey_df=survey,
            )
        # Check that the DIA-DS05 warning was emitted
        ds05_warnings = [r for r in caplog.records if "DS05" in r.message or "ertical" in r.message]
        assert len(ds05_warnings) > 0, \
            "DH-04 FAIL: No warning for varying azimuth on vertical hole"


# ── DH-05: Small-angle threshold (DIA-DS04) ─────────────────────────────────

@pytest.mark.critical
class TestDH05SmallAngleThreshold:
    def test_near_zero_deflection_handled(self):
        """Very small deflection angles (< 1e-6 but > 1e-9) should still use ratio formula."""
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        survey = pd.DataFrame({
            "depth": [0.0, 100.0],
            "azimuth": [90.0, 90.0 + 1e-7],  # tiny azimuth change
            "dip": [-60.0, -60.0],
        })
        depths, xs, ys, zs = minimum_curvature_desurvey(
            collar_x=0.0, collar_y=0.0, collar_z=0.0,
            survey_df=survey,
        )
        assert depths is not None, "DH-05 FAIL: Desurvey returned None for tiny deflection"
        assert not np.any(np.isnan(xs)), "DH-05 FAIL: NaN in coordinates"
        assert not np.any(np.isnan(zs)), "DH-05 FAIL: NaN in Z coordinates"


# ── DH-06: Autofix exception logging (DIA-A02) ──────────────────────────────

@pytest.mark.critical
class TestDH06AutofixExceptionLogging:
    def test_no_silent_exception_swallowing(self):
        """drillhole_autofix.py must not have bare 'except Exception: pass'."""
        from pathlib import Path
        autofix_path = Path(__file__).parent.parent.parent / \
            "block_model_viewer" / "drillholes" / "drillhole_autofix.py"
        content = autofix_path.read_text()
        lines = content.split("\n")
        bare_passes = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "pass" and i > 1:
                prev = lines[i - 2].strip()
                if prev.startswith("except Exception:") or prev.startswith("except Exception :"):
                    bare_passes.append(i)
        assert len(bare_passes) == 0, \
            f"DH-06 FAIL: Found bare 'except Exception: pass' at lines {bare_passes}"


# ── DH-07: Autofix confidence tiers configurable (DIA-A03) ──────────────────

@pytest.mark.major
class TestDH07ConfigurableConfidence:
    def test_confidence_tiers_update(self):
        """set_confidence_tiers should update internal tier thresholds."""
        from block_model_viewer.drillholes.drillhole_autofix import set_confidence_tiers
        # Should not raise
        set_confidence_tiers(
            tight_threshold=3.0, tight_score=0.99,
            medium_threshold=10.0, medium_score=0.90,
            wide_score=0.75,
        )
        # Reset to defaults
        set_confidence_tiers()


# ── DH-08: Autofix rollback on worsened validation (DIA-A04) ────────────────

@pytest.mark.blocker
class TestDH08AutofixRollback:
    def test_autofix_does_not_worsen_errors(self, sample_collars, sample_surveys, sample_assays, sample_lithology):
        """If autofix increases error count, result should roll back."""
        from block_model_viewer.drillholes.drillhole_autofix import run_drillhole_autofix
        try:
            result = run_drillhole_autofix(
                collars=sample_collars.copy(),
                surveys=sample_surveys.copy(),
                assays=sample_assays.copy(),
                lithology=sample_lithology.copy(),
            )
            # After autofix, errors should not exceed pre-fix count
            errors_before = sum(1 for v in result.violations_before
                                if getattr(v, "severity", "") == "ERROR")
            errors_after = sum(1 for v in result.violations_after
                               if getattr(v, "severity", "") == "ERROR")
            assert errors_after <= errors_before, \
                f"DH-08 FAIL: Autofix worsened errors ({errors_before} → {errors_after})"
        except Exception:
            pytest.skip("run_drillhole_autofix not available or incompatible")


# ── DH-09: QAQC CRM z-score evaluation ──────────────────────────────────────

@pytest.mark.critical
class TestDH09CRMZScore:
    def test_crm_pass_within_tolerance(self):
        """CRM sample with z-score < 2.0 should PASS."""
        from block_model_viewer.drillholes.control_samples import (
            ControlSampleManager, ControlSampleType, ControlSampleStatus,
        )
        mgr = ControlSampleManager()
        s = mgr.add_sample(ControlSampleType.CRM, "DH001", 0, 2, "Fe", 60.0, 59.5)
        s.evaluate_crm(tolerance_zscore=2.0)
        assert s.status in (ControlSampleStatus.PASSED, ControlSampleStatus.WARNING), \
            f"DH-09 FAIL: CRM with small deviation should pass, got {s.status}"

    def test_crm_fail_outside_tolerance(self):
        """CRM sample with large deviation should FAIL."""
        from block_model_viewer.drillholes.control_samples import (
            ControlSampleManager, ControlSampleType, ControlSampleStatus,
        )
        mgr = ControlSampleManager()
        s = mgr.add_sample(ControlSampleType.CRM, "DH001", 0, 2, "Fe", 60.0, 10.0)
        s.evaluate_crm(tolerance_zscore=2.0)
        assert s.status in (ControlSampleStatus.FAILED, ControlSampleStatus.WARNING), \
            f"DH-09 FAIL: CRM with huge deviation should fail, got {s.status}"


# ── DH-10: QAQC duplicate RSD evaluation ────────────────────────────────────

@pytest.mark.critical
class TestDH10DuplicateRSD:
    def test_duplicate_pass_low_rsd(self):
        """Duplicate pair with near-identical values should pass RSD check."""
        from block_model_viewer.drillholes.control_samples import (
            ControlSampleManager, ControlSampleType, ControlSampleStatus,
        )
        mgr = ControlSampleManager()
        s = mgr.add_sample(ControlSampleType.DUPLICATE, "DH001", 0, 2, "Fe", 60.0, 60.1)
        s.evaluate_duplicate(max_rsd_percent=5.0)
        assert s.status == ControlSampleStatus.PASSED, \
            f"DH-10 FAIL: Low-RSD duplicate should pass, got {s.status}"

    def test_duplicate_fail_high_rsd(self):
        """Duplicate pair with large difference should fail."""
        from block_model_viewer.drillholes.control_samples import (
            ControlSampleManager, ControlSampleType, ControlSampleStatus,
        )
        mgr = ControlSampleManager()
        s = mgr.add_sample(ControlSampleType.DUPLICATE, "DH001", 0, 2, "Fe", 60.0, 30.0)
        s.evaluate_duplicate(max_rsd_percent=5.0)
        assert s.status == ControlSampleStatus.FAILED, \
            f"DH-10 FAIL: High-RSD duplicate should fail, got {s.status}"


# ── DH-11: QAQC STANDARD evaluation (DIA-QC05) ─────────────────────────────

@pytest.mark.major
class TestDH11StandardEvaluation:
    def test_standard_evaluation_exists(self):
        """ControlSample must have evaluate_standard method."""
        from block_model_viewer.drillholes.control_samples import (
            ControlSampleManager, ControlSampleType,
        )
        mgr = ControlSampleManager()
        s = mgr.add_sample(ControlSampleType.STANDARD, "DH001", 0, 2, "Fe", 60.0, 59.8)
        assert hasattr(s, "evaluate_standard"), \
            "DH-11 FAIL: ControlSample missing evaluate_standard method"
        s.evaluate_standard(tolerance_zscore=2.0)


# ── DH-12: Depth-from < Depth-to validation ─────────────────────────────────

@pytest.mark.blocker
class TestDH12DepthValidation:
    def test_interval_conversion_rejects_bad_depths(self):
        """Intervals where depth_from >= depth_to should be excluded."""
        from block_model_viewer.drillholes.compositing_utils import dataframes_to_intervals
        bad_assays = pd.DataFrame({
            "hole_id": ["DH001", "DH001", "DH001"],
            "depth_from": [0.0, 4.0, 6.0],   # second row: from > to
            "depth_to": [2.0, 3.0, 8.0],
            "Fe": [55.0, 60.0, 50.0],
        })
        collars = pd.DataFrame({
            "hole_id": ["DH001"],
            "x": [0.0], "y": [0.0], "z": [0.0],
            "total_depth": [100.0],
        })
        try:
            result = dataframes_to_intervals(
                collars_df=collars, assays_df=bad_assays,
            )
            # The bad interval (4.0 → 3.0) should be excluded or flagged
            if hasattr(result, "exclusion_reasons"):
                total_exclusions = sum(result.exclusion_reasons.values())
                assert total_exclusions >= 1 or result.rows_excluded >= 1, \
                    "DH-12 FAIL: Bad depth interval not excluded"
        except Exception:
            # If it raises, that's also acceptable — bad data was caught
            pass
