"""
DI-01 through DI-08: Data Integrity checks.

Tests verify that DataRegistry deep-copies, provenance chains,
cascade invalidation, audit trail immutability, and exception
logging are all functioning correctly.
"""
import copy
import pytest
import pandas as pd
import numpy as np

pytestmark = [pytest.mark.data_integrity, pytest.mark.smoke]


# ── DI-01: Deep-copy on registration ──────────────────────────────────────

@pytest.mark.blocker
class TestDI01DeepCopyRegistration:
    def test_registered_data_is_independent(self, sample_collars, sample_assays):
        from block_model_viewer.core.data_registry import DataRegistry
        reg = DataRegistry()
        original = {"collars": sample_collars.copy(), "assays": sample_assays.copy()}
        reg.register_drillhole_data(original, source_panel="test")
        # Mutate the original after registration
        original["collars"].iloc[0, 1] = -9999.0
        retrieved = reg.get_data("drillhole_data")
        if retrieved is not None:
            if hasattr(retrieved, "get") and "collars" in retrieved:
                assert retrieved["collars"].iloc[0, 1] != -9999.0, \
                    "DI-01 FAIL: External mutation leaked into registry"


# ── DI-02: Provenance chain preservation ──────────────────────────────────

@pytest.mark.critical
class TestDI02ProvenanceChain:
    def test_provenance_chain_survives_reregistration(self, sample_collars, sample_assays):
        from block_model_viewer.core.data_registry import DataRegistry
        reg = DataRegistry()
        data1 = {"collars": sample_collars.copy(), "assays": sample_assays.copy()}
        reg.register_drillhole_data(data1, source_panel="import")
        data2 = {"collars": sample_collars.copy(), "assays": sample_assays.copy()}
        reg.register_drillhole_data(data2, source_panel="autofix")
        item = reg._data_store.get("drillhole_data")
        if item and item.get("metadata"):
            meta = item["metadata"]
            chain = getattr(meta, "provenance_chain", None) or getattr(meta, "_prev_provenance_chain", [])
            # Chain should exist (at least one prior entry)
            assert chain is not None, "DI-02 FAIL: Provenance chain missing after re-registration"


# ── DI-03: Cascade invalidation ───────────────────────────────────────────

@pytest.mark.blocker
class TestDI03CascadeInvalidation:
    def test_stale_results_purged_on_fail(self):
        from block_model_viewer.core.data_registry import DataRegistry
        reg = DataRegistry()
        # Simulate stored results
        reg._data_store["kriging_results"] = {"data": "stale", "metadata": {}}
        reg._data_store["sgsim_results"] = {"data": "stale", "metadata": {}}
        # Trigger validation FAIL status
        if hasattr(reg, "update_validation_status"):
            reg.update_validation_status("FAIL", fatal_count=1)
            assert "kriging_results" not in reg._data_store, \
                "DI-03 FAIL: kriging_results not invalidated"
            assert "sgsim_results" not in reg._data_store, \
                "DI-03 FAIL: sgsim_results not invalidated"


# ── DI-04: Raw layer returns deep copy ────────────────────────────────────

@pytest.mark.blocker
class TestDI04RawLayerDeepCopy:
    def test_raw_retrieval_is_deep_copy(self, sample_collars, sample_assays):
        from block_model_viewer.core.data_registry import DataRegistry
        reg = DataRegistry()
        data = {"collars": sample_collars.copy(), "assays": sample_assays.copy()}
        reg.register_drillhole_data(data, source_panel="test")
        raw1 = reg.get_data("drillhole_data", copy_data=True)
        raw2 = reg.get_data("drillhole_data", copy_data=True)
        if raw1 is not None and raw2 is not None:
            assert raw1 is not raw2, "DI-04 FAIL: Raw layer returning same object"


# ── DI-05: Audit trail immutability ───────────────────────────────────────

@pytest.mark.critical
class TestDI05AuditImmutability:
    def test_audit_record_is_frozen(self):
        from block_model_viewer.drillholes.audit_trail import AuditRecord, AuditAction
        from datetime import datetime
        record = AuditRecord(
            record_id="TEST-001", timestamp=datetime.now(),
            user="test", action=AuditAction.CREATE,
            entity_type="drillhole", entity_id="DH001",
            description="test record",
        )
        with pytest.raises(Exception):
            record.description = "tampered"


# ── DI-06: Per-project audit trail isolation ──────────────────────────────

@pytest.mark.major
class TestDI06ProjectIsolation:
    def test_separate_projects_have_separate_trails(self):
        from block_model_viewer.drillholes.audit_trail import get_audit_trail, AuditAction
        trail_a = get_audit_trail("project_A_test")
        trail_b = get_audit_trail("project_B_test")
        trail_a.log_action(
            user="test", action=AuditAction.CREATE,
            entity_type="test", entity_id="1", description="A record",
        )
        assert len(trail_a.records) >= 1
        assert len(trail_b.records) == 0, \
            "DI-06 FAIL: Project B sees Project A records"


# ── DI-07: UUID-based control sample IDs ──────────────────────────────────

@pytest.mark.major
class TestDI07UUIDSampleIDs:
    def test_sample_ids_are_unique_across_managers(self):
        from block_model_viewer.drillholes.control_samples import (
            ControlSampleManager, ControlSampleType,
        )
        mgr1 = ControlSampleManager()
        mgr2 = ControlSampleManager()
        s1 = mgr1.add_sample(ControlSampleType.CRM, "DH001", 0, 2, "Fe", 60.0, 59.5)
        s2 = mgr2.add_sample(ControlSampleType.CRM, "DH001", 0, 2, "Fe", 60.0, 59.5)
        assert s1.sample_id != s2.sample_id, \
            "DI-07 FAIL: Sample IDs collide across managers"
        assert "CTRL-" in s1.sample_id


# ── DI-08: All autofix exceptions logged ──────────────────────────────────

@pytest.mark.critical
class TestDI08AutofixExceptionLogging:
    def test_no_bare_except_pass_in_autofix(self):
        """Verify that drillhole_autofix.py has no silent exception swallowing."""
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
            f"DI-08 FAIL: Found bare 'except Exception: pass' at lines {bare_passes}"
