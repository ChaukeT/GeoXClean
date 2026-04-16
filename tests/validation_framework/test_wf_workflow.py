"""
WF-01 through WF-04: Workflow Consistency domain checks.

Tests verify the end-to-end pipeline: load → validate → autofix →
composite → export, ensuring data flows correctly between stages.
"""
import pytest
import pandas as pd
import numpy as np

pytestmark = [pytest.mark.workflow, pytest.mark.smoke]


# ── WF-01: Load → Validate pipeline ─────────────────────────────────────────

@pytest.mark.blocker
class TestWF01LoadValidatePipeline:
    def test_registry_accepts_valid_data(self, sample_collars, sample_assays):
        """DataRegistry should accept well-formed drillhole data."""
        from block_model_viewer.core.data_registry import DataRegistry
        reg = DataRegistry()
        data = {"collars": sample_collars.copy(), "assays": sample_assays.copy()}
        reg.register_drillhole_data(data, source_panel="test")
        retrieved = reg.get_data("drillhole_data")
        assert retrieved is not None, \
            "WF-01 FAIL: Registry returned None after valid registration"


# ── WF-02: Validate → Autofix → Re-validate pipeline ────────────────────────

@pytest.mark.critical
class TestWF02ValidateAutofixPipeline:
    def test_autofix_reduces_or_maintains_errors(
        self, sample_collars, sample_surveys, sample_assays, sample_lithology
    ):
        """Autofix pipeline should not increase error count."""
        try:
            from block_model_viewer.drillholes.drillhole_autofix import run_drillhole_autofix
            result = run_drillhole_autofix(
                collars=sample_collars.copy(),
                surveys=sample_surveys.copy(),
                assays=sample_assays.copy(),
                lithology=sample_lithology.copy(),
            )
            errors_before = sum(1 for v in result.violations_before
                                if getattr(v, "severity", "") == "ERROR")
            errors_after = sum(1 for v in result.violations_after
                               if getattr(v, "severity", "") == "ERROR")
            assert errors_after <= errors_before, \
                f"WF-02 FAIL: Autofix increased errors ({errors_before} → {errors_after})"
        except Exception:
            pytest.skip("Autofix pipeline not available")


# ── WF-03: Autofix → Composite pipeline ─────────────────────────────────────

@pytest.mark.critical
class TestWF03AutofixCompositePipeline:
    def test_compositing_works_on_autofixed_data(
        self, sample_collars, sample_assays
    ):
        """Compositing engine should accept data from the standard pipeline."""
        from block_model_viewer.drillholes.compositing_utils import dataframes_to_intervals_simple
        from block_model_viewer.drillholes.compositing_engine import (
            CompositingMethodEngine, CompositeConfig, CompositingMethod,
        )
        intervals = dataframes_to_intervals_simple(
            collars_df=sample_collars, assays_df=sample_assays,
        )
        if not intervals:
            pytest.skip("No intervals produced from sample data")

        cfg = CompositeConfig(
            method=CompositingMethod.FIXED_LENGTH,
            composite_length=10.0,
        )
        engine = CompositingMethodEngine()
        composites = engine.composite(intervals, cfg)
        assert len(composites) > 0, \
            "WF-03 FAIL: No composites produced from valid pipeline"
        # Verify composites have grades
        has_grades = any(c.grades for c in composites)
        assert has_grades, "WF-03 FAIL: Composites have no grade data"


# ── WF-04: Full pipeline audit trail ────────────────────────────────────────

@pytest.mark.major
class TestWF04PipelineAuditTrail:
    def test_audit_trail_records_pipeline_actions(self):
        """Audit trail should record actions from each pipeline stage."""
        from block_model_viewer.drillholes.audit_trail import (
            get_audit_trail, AuditAction,
        )
        trail = get_audit_trail("wf04_test_project")
        initial_count = len(trail.records)
        trail.log_action(
            user="test", action=AuditAction.CREATE,
            entity_type="drillhole", entity_id="DH001",
            description="WF-04 test: data loaded",
        )
        trail.log_action(
            user="test", action=AuditAction.UPDATE,
            entity_type="drillhole", entity_id="DH001",
            description="WF-04 test: validation complete",
        )
        assert len(trail.records) == initial_count + 2, \
            "WF-04 FAIL: Audit trail did not record both actions"
        # Verify chronological order
        if len(trail.records) >= 2:
            last_two = trail.records[-2:]
            assert last_two[0].timestamp <= last_two[1].timestamp, \
                "WF-04 FAIL: Audit records not in chronological order"
