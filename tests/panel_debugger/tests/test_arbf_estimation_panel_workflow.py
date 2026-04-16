from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.models.transform import NormalScoreTransformer
from block_model_viewer.ui.arbf_estimation_panel import ARBFEstimationPanel
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


def _build_irbf_labels(df):
    cutoff = float(df["FE_PCT"].median())
    return np.where(df["FE_PCT"].to_numpy(dtype=float) >= cutoff, "Inside", "Outside")


@pytest.mark.integration
def test_active_arbf_panel_respects_source_selection_and_domain_filter(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    package = _attach_two_lithologies(package)
    _register_package(registry, package, "active-arbf-panel-source-domain-test")

    panel = ARBFEstimationPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel.source_combo.setCurrentText("Raw Assays")
        panel._on_load_data()
        mock_qapp.processEvents()

        assert panel.drillhole_data is not None
        assert len(panel.drillhole_data) == len(package["assays"])
        assert "FE_PCT" in [panel.variable_combo.itemText(i) for i in range(panel.variable_combo.count())]

        panel.source_combo.setCurrentText("Composited Drillholes")
        mock_qapp.processEvents()

        assert panel.drillhole_data is not None
        assert len(panel.drillhole_data) == len(package["composites"])
        assert "lith_code" in panel.drillhole_data.columns

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "lith_code: SANDSTONE" in combo_items
        assert "lith_code: LIMESTONE" in combo_items

        panel.domain_combo.setCurrentText("lith_code: LIMESTONE")
        filtered = panel._get_filtered_data()
        assert filtered is not None
        assert not filtered.empty
        assert set(filtered["lith_code"].dropna().astype(str).unique()) == {"LIMESTONE"}
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_arbf_panel_uses_registered_indicator_rbf_domain_on_load(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "active-arbf-panel-irbf-load-test")

    composites = package["composites"]
    labels = _build_irbf_labels(composites)
    registry.register_indicator_rbf_domain(
        {
            "domain_name": "IRBF_Inside",
            "sample_domain_labels": labels,
            "sample_domain_index": list(composites.index),
            "sample_domain_column": "IRBF_Domain",
        },
        source_panel="indicator-rbf-test",
        metadata={"domain_name": "IRBF_Inside"},
    )

    panel = ARBFEstimationPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_load_data()
        panel.source_combo.setCurrentText("Composited Drillholes")
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items
        assert "IRBF_Domain: Outside" in combo_items

        panel.domain_combo.setCurrentText("IRBF_Domain: Inside")
        filtered = panel._get_filtered_data()
        assert filtered is not None
        assert not filtered.empty
        assert set(filtered["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside"}

        params = panel.gather_parameters()
        assert params["domain_column"] == "IRBF_Domain"
        assert params["domain_value"] == "Inside"
        assert len(params["data"]) == int(np.sum(labels == "Inside"))
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_arbf_panel_refreshes_indicator_rbf_domain_from_composites_signal(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "active-arbf-panel-irbf-signal-test")

    panel = ARBFEstimationPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_load_data()
        panel.source_combo.setCurrentText("Composited Drillholes")
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" not in combo_items

        refreshed = package["composites"].copy()
        refreshed["IRBF_Domain"] = pd.Categorical(_build_irbf_labels(refreshed))
        registry.signals.compositesLoaded.emit(refreshed)
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items

        panel.domain_combo.setCurrentText("IRBF_Domain: Inside")
        filtered = panel._get_filtered_data()
        assert filtered is not None
        assert not filtered.empty
        assert set(filtered["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside"}
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_arbf_panel_refreshes_indicator_rbf_domain_without_manual_load(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "active-arbf-panel-irbf-open-test")

    panel = ARBFEstimationPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        assert panel.drillhole_data is None
        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" not in combo_items

        composites = package["composites"]
        labels = _build_irbf_labels(composites)
        registry.register_indicator_rbf_domain(
            {
                "domain_name": "IRBF_Inside",
                "sample_domain_labels": labels,
                "sample_domain_index": list(composites.index),
                "sample_domain_column": "IRBF_Domain",
            },
            source_panel="indicator-rbf-open-test",
            metadata={"domain_name": "IRBF_Inside"},
        )
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items
        assert "IRBF_Domain: Outside" in combo_items
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_arbf_panel_maps_irbf_domain_for_reindexed_transformed_composites(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()

    original_composites = package["composites"].copy()
    transformed_composites = original_composites.copy()
    transformed_composites["CU_NS"] = (
        (transformed_composites["FE_PCT"] - transformed_composites["FE_PCT"].mean())
        / transformed_composites["FE_PCT"].std()
    )
    transformed_composites["IRBF_Domain"] = pd.Categorical(["Unclassified"] * len(transformed_composites))
    transformed_composites.index = pd.Index(np.arange(10_000, 10_000 + len(transformed_composites)), name="row_id")

    transformed_package = dict(package)
    transformed_package["composites"] = transformed_composites
    _register_package(registry, transformed_package, "active-arbf-panel-irbf-reindexed-test")

    labels = _build_irbf_labels(original_composites)
    registry.register_indicator_rbf_domain(
        {
            "domain_name": "IRBF_Inside",
            "sample_domain_labels": labels,
            "sample_domain_index": list(original_composites.index),
            "sample_domain_column": "IRBF_Domain",
            "sample_interval_ids": original_composites["INTERVAL_ID"].astype(str).tolist(),
            "sample_hole_ids": original_composites["HOLEID"].astype(str).tolist(),
            "sample_from_values": original_composites["FROM"].astype(float).tolist(),
            "sample_to_values": original_composites["TO"].astype(float).tolist(),
        },
        source_panel="indicator-rbf-reindexed-test",
        metadata={"domain_name": "IRBF_Inside"},
    )

    panel = ARBFEstimationPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_load_data()
        panel.source_combo.setCurrentText("Composited Drillholes")
        mock_qapp.processEvents()

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items
        assert "IRBF_Domain: Outside" in combo_items
        assert panel.drillhole_data is not None
        assert set(panel.drillhole_data["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside", "Outside"}
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_arbf_panel_recommends_and_applies_settings(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "active-arbf-panel-recommend-apply-test")

    panel = ARBFEstimationPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_load_data()
        panel.source_combo.setCurrentText("Composited Drillholes")
        panel.variable_combo.setCurrentText("FE_PCT")

        panel.kernel_combo.setCurrentText("Spheroidal")
        panel.cb_run_cv.setChecked(False)
        panel.max_samples_spin.setValue(40)
        panel.min_samples_spin.setValue(2)

        panel._on_auto_recommend()
        mock_qapp.processEvents()

        recommendation = panel._last_recommendation
        assert recommendation is not None
        assert recommendation.settings
        report_text = panel.auto_recommend_info.toPlainText()
        assert "UNIVARIATE DISTRIBUTION ANALYSIS" in report_text
        assert "DIRECTIONAL VARIOGRAPHY" in report_text
        assert "RECOMMENDATIONS" in report_text

        assert panel.kernel_combo.currentText() == "Spheroidal"
        assert panel.cb_run_cv.isChecked() is False
        assert panel.max_samples_spin.value() == 40
        assert panel.min_samples_spin.value() == 2

        panel._on_apply_recommendations()
        mock_qapp.processEvents()

        kernel_map = {
            "spheroidal": "Spheroidal",
            "spherical": "Spherical",
            "gaussian": "Gaussian",
            "matern_32": "Matern-3/2",
            "matern_52": "Matern-5/2",
            "cubic": "Cubic",
        }
        expected_kernel = kernel_map.get(recommendation.settings.get("kernel_type"))
        if expected_kernel is not None and panel.kernel_combo.findText(expected_kernel) >= 0:
            assert panel.kernel_combo.currentText() == expected_kernel

        if "max_samples" in recommendation.settings:
            assert panel.max_samples_spin.value() == recommendation.settings["max_samples"]
        if "min_samples" in recommendation.settings:
            assert panel.min_samples_spin.value() == recommendation.settings["min_samples"]
        if "run_cv" in recommendation.settings:
            assert panel.cb_run_cv.isChecked() == recommendation.settings["run_cv"]
        if "clip_to_drill_footprint" in recommendation.settings and hasattr(panel, "chk_clip_to_footprint"):
            assert (
                panel.chk_clip_to_footprint.isChecked()
                == recommendation.settings["clip_to_drill_footprint"]
            )

        assert "Settings applied to panel controls" in panel.auto_recommend_info.toPlainText()
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_active_arbf_panel_recommendation_keeps_ns_variable_with_registered_transformer(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    package = dict(package)
    composites = package["composites"].copy()
    assays = package["assays"].copy()

    for frame in (composites, assays):
        raw = np.exp(frame["FE_PCT"].to_numpy(dtype=float) - frame["FE_PCT"].min() + 1.0)
        ns_like = (np.log(raw) - np.mean(np.log(raw))) / np.std(np.log(raw))
        frame["CU"] = raw
        frame["CU_NS"] = ns_like

    package["composites"] = composites
    package["assays"] = assays
    _register_package(registry, package, "active-arbf-panel-ns-recommend-test")
    transformer = NormalScoreTransformer()
    transformer.fit(composites["CU"].to_numpy(dtype=float))
    registry.register_transformation_metadata(
        {"transformations": {"CU": {"new_col": "CU_NS", "method": "Normal Score"}}}
    )
    registry.register_transformers({"CU": transformer})

    panel = ARBFEstimationPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_load_data()
        panel.source_combo.setCurrentText("Composited Drillholes")
        panel.cb_normal_score.setChecked(True)
        panel.variable_combo.setCurrentText("CU_NS")
        assert panel.cb_normal_score.isChecked() is False

        assert panel._resolve_normal_score() is False
        assert panel._external_ns_original_var == "CU"
        assert panel._external_ns_transformer is transformer

        panel._on_auto_recommend()
        mock_qapp.processEvents()

        recommendation = panel._last_recommendation
        assert recommendation is not None
        assert "recommended_variable" not in recommendation.settings
        assert recommendation.settings.get("use_normal_score") is False

        panel._on_apply_recommendations()
        mock_qapp.processEvents()

        assert panel.variable_combo.currentText() == "CU_NS"
        assert panel.cb_normal_score.isChecked() is False
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None
