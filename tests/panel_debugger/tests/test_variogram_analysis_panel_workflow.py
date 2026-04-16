from datetime import datetime

import pytest

from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.ui.variogram_analysis_panel import VariogramPanel
from tests.panel_debugger.tests.test_variogram_panel_workflow import (
    _attach_two_lithologies,
    _build_irbf_labels,
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


@pytest.mark.integration
def test_active_variogram_panel_runs_end_to_end_with_registry_data(mock_qapp):
    registry = _fresh_registry()
    package, true_azimuth = _build_structured_drillhole_package()
    _register_package(registry, package, "active-variogram-panel-workflow")

    panel = VariogramPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())

        assert panel.drillhole_data is not None
        assert {"X", "Y", "Z", "FE_PCT"}.issubset(panel.drillhole_data.columns)

        panel.var_combo.setCurrentText("FE_PCT")
        panel.cb_auto_lags.setChecked(True)
        panel.model_combo.setCurrentText("Spherical")
        panel.seed_spin.setValue(42)

        panel._on_compute()
        mock_qapp.processEvents()

        results = panel.variogram_results
        assert results is not None
        assert abs((results["major_azimuth"] % 180.0) - true_azimuth) <= 2.0
        assert results["metadata"]["source_data_n_samples"] == len(
            panel.drillhole_data.dropna(subset=["X", "Y", "Z", "FE_PCT"])
        )

        combined = results["combined_3d_model"]
        assert combined["major_range"] >= combined["minor_range"] > 0.0
        assert combined["vertical_range"] > 0.0
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_variogram_panel_filters_by_imported_lithology_values(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    package = _attach_two_lithologies(package)
    _register_package(registry, package, "active-variogram-panel-domain-filter")

    panel = VariogramPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())

        assert panel.drillhole_data is not None
        assert "lith_code" in panel.drillhole_data.columns

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "lith_code: SANDSTONE" in combo_items
        assert "lith_code: LIMESTONE" in combo_items

        panel.var_combo.setCurrentText("FE_PCT")
        panel.domain_combo.setCurrentText("lith_code: SANDSTONE")
        panel.cb_auto_lags.setChecked(True)
        panel.model_combo.setCurrentText("Spherical")
        panel.seed_spin.setValue(42)

        expected_rows = len(
            panel.drillhole_data[
                panel.drillhole_data["lith_code"].astype(str) == "SANDSTONE"
            ].dropna(subset=["X", "Y", "Z", "FE_PCT"])
        )

        panel._on_compute()
        mock_qapp.processEvents()

        results = panel.variogram_results
        assert results is not None
        assert results["metadata"]["domain_filter_selection"] == "lith_code: SANDSTONE"
        assert results["metadata"]["domain_filter_column"] == "lith_code"
        assert results["metadata"]["domain_filter_value"] == "SANDSTONE"
        assert results["metadata"]["domain_filter_n_samples"] == expected_rows
        assert results["metadata"]["source_data_n_samples"] == expected_rows
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_variogram_panel_recommends_applies_and_fits(mock_qapp):
    registry = _fresh_registry()
    package, true_azimuth = _build_structured_drillhole_package()
    _register_package(registry, package, "active-variogram-panel-recommend-fit")

    panel = VariogramPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())

        panel.var_combo.setCurrentText("FE_PCT")
        panel.seed_spin.setValue(42)
        panel._on_recommend_and_fit()
        mock_qapp.processEvents()

        recommendation = panel._latest_recommendation
        assert recommendation is not None
        assert recommendation["settings"]["model_type"] == "spherical"
        assert recommendation["settings"]["auto_lags"] is True
        assert abs((recommendation["settings"]["default_azimuth"] % 180.0) - true_azimuth) <= 5.0

        assert panel.model_combo.currentText() == "Spherical"
        assert panel.cb_auto_lags.isChecked()
        assert "DEEP VARIOGRAM ANALYSIS" in panel._recommend_text.toPlainText()

        results = panel.variogram_results
        assert results is not None
        assert results["metadata"]["recommendation_applied"] is True
        assert results["metadata"]["recommended_model_type"] == "spherical"
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_variogram_panel_lists_irbf_domain_for_declustered_data(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "active-variogram-panel-irbf-declustered")

    composites = registry.get_drillhole_data()["composites"]
    declustered = composites.copy()
    declustered["declust_weight"] = 1.0
    declustered["declust_cell"] = range(len(declustered))
    registry.register_declustering_results(
        (declustered, {"total_samples": len(declustered)}),
        source_panel="active-variogram-panel-irbf-declustered",
        metadata={"parent_data_key": "composites"},
    )

    registry.register_indicator_rbf_domain(
        {
            "domain_name": "Inside",
            "sample_domain_labels": _build_irbf_labels(composites),
            "sample_domain_index": list(composites.index),
            "sample_domain_column": "IRBF_Domain",
        },
        source_panel="active-variogram-panel-irbf-declustered",
    )

    panel = VariogramPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded()

        assert panel.drillhole_data is not None
        assert "IRBF_Domain" in panel.drillhole_data.columns

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items
        assert "IRBF_Domain: Outside" in combo_items
        assert "Declustered" in panel._data_status.text()
    finally:
        panel.close()
        registry.clear_drillhole_data()
        registry.clear_data("declustering_results")
        DataRegistry._instance = None
