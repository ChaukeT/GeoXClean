from datetime import datetime

import numpy as np
import pytest

from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.ui.cosgsim_panel import CoSGSIMPanel
from block_model_viewer.ui.kriging_panel import KrigingPanel
from block_model_viewer.ui.simple_kriging_panel import SimpleKrigingPanel
from tests.panel_debugger.tests.test_variogram_panel_workflow import (
    _attach_two_lithologies,
    _build_structured_drillhole_package,
)


def _fresh_registry():
    existing = DataRegistry.get_existing()
    if existing is not None:
        try:
            existing.clear_drillhole_data()
        except Exception:
            pass
    DataRegistry._instance = None
    return DataRegistry.instance()


def _register_package(registry, package, source_panel):
    assert registry.register_drillhole_data(
        package,
        source_panel=source_panel,
        is_raw_import=True,
    )
    registry.set_drillholes_validation_state(
        status="PASS",
        timestamp=datetime.now().isoformat(),
        config_hash=source_panel,
        fatal_count=0,
        warn_count=0,
        info_count=0,
        excluded_rows={},
    )


def _with_secondary_variable(package):
    enriched = dict(package)
    assays = package["assays"].copy()
    composites = package["composites"].copy()

    rng = np.random.RandomState(20260319)
    assays["MN_PCT"] = assays["FE_PCT"] * 0.7 + rng.normal(0.0, 0.15, len(assays))
    composites["MN_PCT"] = composites["FE_PCT"] * 0.7 + rng.normal(0.0, 0.1, len(composites))

    enriched["assays"] = assays
    enriched["composites"] = composites
    return enriched


@pytest.mark.integration
def test_ordinary_kriging_panel_uses_selected_lithology_value(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    package = _attach_two_lithologies(package)
    _register_package(registry, package, "ordinary-kriging-domain-filter-test")

    panel = KrigingPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        assert panel.drillhole_data is not None
        assert "lith_code" in panel.drillhole_data.columns

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "lith_code: SANDSTONE" in combo_items
        assert "lith_code: LIMESTONE" in combo_items

        panel.data_source_composited.setChecked(True)
        panel._on_drillhole_data_loaded(registry.get_drillhole_data())
        panel.variable_combo.setCurrentText("FE_PCT")
        panel.domain_combo.setCurrentText("lith_code: SANDSTONE")

        filtered = panel._get_filtered_data()
        assert filtered is not None
        assert not filtered.empty
        assert set(filtered["lith_code"].dropna().astype(str).unique()) == {"SANDSTONE"}

        params = panel.gather_parameters()

        expected_rows = len(
            panel.drillhole_data[
                panel.drillhole_data["lith_code"].astype(str) == "SANDSTONE"
            ].dropna(subset=["X", "Y", "Z", "FE_PCT"])
        )
        assert len(params["data_df"]) == expected_rows
        assert params["domain_column"] == "lith_code"
        assert params["domain_value"] == "SANDSTONE"
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_ordinary_kriging_panel_lists_irbf_domain_after_registration(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "ordinary-kriging-irbf-domain-test")

    panel = KrigingPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" not in combo_items

        composites = package["composites"]
        labels = np.where(
            composites["FE_PCT"].to_numpy(dtype=float) >= float(composites["FE_PCT"].median()),
            "Inside",
            "Outside",
        )
        registry.register_indicator_rbf_domain(
            {
                "domain_name": "IRBF_Inside",
                "sample_domain_labels": labels,
                "sample_domain_index": list(composites.index),
                "sample_domain_column": "IRBF_Domain",
            },
            source_panel="ordinary-kriging-irbf-domain-test",
            metadata={"domain_name": "IRBF_Inside"},
        )
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items
        assert "IRBF_Domain: Outside" in combo_items
        assert panel.drillhole_data is not None
        assert "IRBF_Domain" in panel.drillhole_data.columns

        panel.domain_combo.setCurrentText("IRBF_Domain: Inside")
        filtered = panel._get_filtered_data()
        assert filtered is not None
        assert set(filtered["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside"}
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_cosgsim_panel_uses_selected_lithology_value(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    package = _attach_two_lithologies(_with_secondary_variable(package))
    _register_package(registry, package, "cosgsim-domain-filter-test")

    panel = CoSGSIMPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        assert panel.drillhole_data is not None
        assert "lith_code" in panel.drillhole_data.columns

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "lith_code: SANDSTONE" in combo_items
        assert "lith_code: LIMESTONE" in combo_items

        panel.primary_combo.setCurrentText("FE_PCT")
        panel.domain_combo.setCurrentText("lith_code: LIMESTONE")

        matched_index = None
        for i in range(panel.secondary_list.count()):
            if panel.secondary_list.item(i).text() == "MN_PCT":
                matched_index = i
                break
        assert matched_index is not None
        panel.secondary_list.item(matched_index).setSelected(True)

        filtered = panel._get_filtered_data()
        assert filtered is not None
        assert not filtered.empty
        assert set(filtered["lith_code"].dropna().astype(str).unique()) == {"LIMESTONE"}

        params = panel.gather_parameters()

        expected_rows = len(
            panel.drillhole_data[
                panel.drillhole_data["lith_code"].astype(str) == "LIMESTONE"
            ].dropna(subset=["X", "Y", "Z", "FE_PCT", "MN_PCT"])
        )
        assert len(params["data_df"]) == expected_rows
        assert params["domain_column"] == "lith_code"
        assert params["domain_value"] == "LIMESTONE"
        assert params["primary_name"] == "FE_PCT"
        assert params["secondary_names"] == ["MN_PCT"]
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_simple_kriging_panel_lists_irbf_domain_after_registration(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "simple-kriging-irbf-domain-test")

    panel = SimpleKrigingPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" not in combo_items

        composites = package["composites"]
        labels = np.where(composites["FE_PCT"].to_numpy(dtype=float) >= float(composites["FE_PCT"].median()), "Inside", "Outside")
        registry.register_indicator_rbf_domain(
            {
                "domain_name": "IRBF_Inside",
                "sample_domain_labels": labels,
                "sample_domain_index": list(composites.index),
                "sample_domain_column": "IRBF_Domain",
            },
            source_panel="simple-kriging-irbf-domain-test",
            metadata={"domain_name": "IRBF_Inside"},
        )
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items
        assert "IRBF_Domain: Outside" in combo_items
        assert panel.data_df is not None
        assert "IRBF_Domain" in panel.data_df.columns
        panel.domain_combo.setCurrentText("IRBF_Domain: Inside")
        filtered = panel._get_filtered_data()
        assert filtered is not None
        assert set(filtered["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside"}
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None
