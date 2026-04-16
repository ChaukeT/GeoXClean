import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtTest import QTest

from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.ui.variogram_panel import VariogramAnalysisPanel


def _fresh_registry():
    existing = DataRegistry.get_existing()
    if existing is not None:
        try:
            existing.clear_drillhole_data()
        except Exception:
            pass
    DataRegistry._instance = None
    return DataRegistry.instance()


def _build_structured_drillhole_package(seed: int = 20260319):
    rng = np.random.RandomState(seed)
    azimuth_deg = 35.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    rotation = np.array([
        [np.sin(azimuth_rad), np.cos(azimuth_rad)],
        [np.cos(azimuth_rad), -np.sin(azimuth_rad)],
    ])

    u = np.linspace(-180.0, 180.0, 7)
    v = np.linspace(-90.0, 90.0, 5)
    collars_uv = np.array(np.meshgrid(u, v, indexing="ij")).reshape(2, -1).T
    collars_xy = collars_uv @ rotation.T + np.array([500000.0, 6000000.0])

    collars = []
    surveys = []
    assays = []
    composites = []
    assay_xyz = []
    composite_xyz = []

    for hole_index, (uv, xy) in enumerate(zip(collars_uv, collars_xy), start=1):
        hole_id = f"DH{hole_index:03d}"
        collars.append({
            "HOLEID": hole_id,
            "EAST": float(xy[0]),
            "NORTH": float(xy[1]),
            "ELEV": 500.0,
            "DEPTH": 60.0,
            "AZIMUTH": 0.0,
            "DIP": -90.0,
        })
        surveys.extend([
            {"HOLEID": hole_id, "DEPTH": 0.0, "AZIMUTH": 0.0, "DIP": -90.0},
            {"HOLEID": hole_id, "DEPTH": 60.0, "AZIMUTH": 0.0, "DIP": -90.0},
        ])

        for interval_index in range(24):
            from_depth = interval_index * 2.5
            to_depth = (interval_index + 1) * 2.5
            mid_depth = 0.5 * (from_depth + to_depth)
            assays.append({
                "HOLEID": hole_id,
                "FROM": from_depth,
                "TO": to_depth,
                "INTERVAL_ID": f"{hole_id}_A_{interval_index}",
            })
            assay_xyz.append([uv[0], uv[1], -mid_depth])

        for interval_index in range(12):
            from_depth = interval_index * 5.0
            to_depth = (interval_index + 1) * 5.0
            mid_depth = 0.5 * (from_depth + to_depth)
            composites.append({
                "HOLEID": hole_id,
                "FROM": from_depth,
                "TO": to_depth,
                "INTERVAL_ID": f"{hole_id}_C_{interval_index}",
            })
            composite_xyz.append([uv[0], uv[1], -mid_depth])

    assays_df = pd.DataFrame(assays)
    composites_df = pd.DataFrame(composites)
    assay_xyz = np.asarray(assay_xyz, dtype=float)
    composite_xyz = np.asarray(composite_xyz, dtype=float)

    true_major = 120.0
    true_minor = 45.0
    true_vertical = 25.0
    true_nugget = 0.1
    true_sill = 1.0
    reference_range = 80.0

    all_xyz = np.vstack([assay_xyz, composite_xyz])
    scaled_xyz = np.column_stack([
        all_xyz[:, 0] * (reference_range / true_major),
        all_xyz[:, 1] * (reference_range / true_minor),
        all_xyz[:, 2] * (reference_range / true_vertical),
    ])

    directions = rng.randn(300, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    values = np.zeros(len(all_xyz))
    for direction in directions:
        projection = scaled_xyz @ direction
        phase = rng.uniform(0.0, 2.0 * np.pi)
        values += np.cos(2.0 * np.pi * (1.0 / reference_range) * projection + phase)

    values -= values.mean()
    values *= np.sqrt(true_sill - true_nugget) / (np.std(values) + 1e-12)
    values += rng.normal(0.0, np.sqrt(true_nugget), len(values))

    assays_df["FE_PCT"] = values[: len(assays_df)]
    composites_df["FE_PCT"] = values[len(assays_df) :]

    package = {
        "collars": pd.DataFrame(collars),
        "surveys": pd.DataFrame(surveys),
        "assays": assays_df,
        "composites": composites_df,
    }
    return package, azimuth_deg


def _attach_two_lithologies(drillhole_package):
    lithology = []
    collars = drillhole_package["collars"]
    for hole_id in collars["HOLEID"].astype(str):
        lithology.extend(
            [
                {"HOLEID": hole_id, "FROM": 0.0, "TO": 30.0, "LITH": "SANDSTONE"},
                {"HOLEID": hole_id, "FROM": 30.0, "TO": 60.0, "LITH": "LIMESTONE"},
            ]
        )
    enriched = dict(drillhole_package)
    enriched["lithology"] = pd.DataFrame(lithology)
    return enriched


def _build_irbf_labels(df: pd.DataFrame) -> list[str]:
    x_values = pd.to_numeric(df["X"], errors="coerce")
    threshold = float(x_values.median())
    return ["Inside" if value >= threshold else "Outside" for value in x_values]


@pytest.mark.integration
@pytest.mark.critical
def test_variogram_panel_runs_end_to_end_with_real_registry_workflow(mock_qapp):
    registry = _fresh_registry()
    drillhole_package, true_azimuth = _build_structured_drillhole_package()

    assert registry.register_drillhole_data(
        drillhole_package,
        source_panel="variogram_panel_workflow_test",
        is_raw_import=True,
    )
    registry.set_drillholes_validation_state(
        status="PASS",
        timestamp=datetime.now().isoformat(),
        config_hash="variogram-panel-workflow-test",
        fatal_count=0,
        warn_count=0,
        info_count=0,
        excluded_rows={},
    )

    panel = VariogramAnalysisPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        assert panel.drillhole_data is not None
        assert {"X", "Y", "Z", "FE_PCT"}.issubset(panel.drillhole_data.columns)

        panel.data_source_composited.setChecked(True)
        panel.var_combo.setCurrentText("FE_PCT")
        panel.auto_lags_checkbox.setChecked(True)
        panel.model_combo.setCurrentText("Spherical")
        panel.random_seed_spin.setValue(42)

        panel.run_analysis()
        QTest.qWait(700)
        mock_qapp.processEvents()

        results = panel.get_variogram_results()
        stored = registry.get_variogram_results("FE_PCT")

        assert results is not None
        assert stored is not None
        assert results["metadata"]["orientation_source"] == "horizontal_pca_support"
        assert results["metadata"]["source_dataset_type"] == "composites"
        assert abs((results["major_azimuth"] % 180.0) - true_azimuth) <= 2.0
        assert abs((results["minor_azimuth"] % 180.0) - ((true_azimuth + 90.0) % 180.0)) <= 2.0
        assert stored["data_source_type"] == "composites"

        combined = results["combined_3d_model"]
        assert combined["major_range"] >= combined["minor_range"] > 0.0
        assert combined["vertical_range"] > 0.0
        assert "horizontal support geometry" in panel.azimuth_info_label.text().lower()
    finally:
        panel.close()
        QTest.qWait(100)
        mock_qapp.processEvents()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_variogram_panel_filters_by_imported_lithology_values(mock_qapp):
    registry = _fresh_registry()
    drillhole_package, _ = _build_structured_drillhole_package()
    drillhole_package = _attach_two_lithologies(drillhole_package)

    assert registry.register_drillhole_data(
        drillhole_package,
        source_panel="variogram_panel_domain_filter_test",
        is_raw_import=True,
    )
    registry.set_drillholes_validation_state(
        status="PASS",
        timestamp=datetime.now().isoformat(),
        config_hash="variogram-panel-domain-filter-test",
        fatal_count=0,
        warn_count=0,
        info_count=0,
        excluded_rows={},
    )

    panel = VariogramAnalysisPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        assert panel.drillhole_data is not None
        assert "lith_code" in panel.drillhole_data.columns

        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "lith_code: SANDSTONE" in combo_items
        assert "lith_code: LIMESTONE" in combo_items

        panel.data_source_composited.setChecked(True)
        panel._apply_data_update(registry.get_drillhole_data())
        panel.var_combo.setCurrentText("FE_PCT")
        panel.domain_combo.setCurrentText("lith_code: SANDSTONE")
        panel.auto_lags_checkbox.setChecked(True)
        panel.model_combo.setCurrentText("Spherical")
        panel.random_seed_spin.setValue(42)

        expected_rows = len(
            panel.drillhole_data[
                panel.drillhole_data["lith_code"].astype(str) == "SANDSTONE"
            ].dropna(subset=["X", "Y", "Z", "FE_PCT"])
        )

        panel.run_analysis()
        QTest.qWait(700)
        mock_qapp.processEvents()

        results = panel.get_variogram_results()

        assert results is not None
        assert results["metadata"]["domain_filter_selection"] == "lith_code: SANDSTONE"
        assert results["metadata"]["domain_filter_column"] == "lith_code"
        assert results["metadata"]["domain_filter_value"] == "SANDSTONE"
        assert results["metadata"]["domain_filter_n_samples"] == expected_rows
        assert results["metadata"]["source_data_n_samples"] == expected_rows
    finally:
        panel.close()
        QTest.qWait(100)
        mock_qapp.processEvents()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_variogram_panel_lists_registered_indicator_rbf_domain(mock_qapp):
    registry = _fresh_registry()
    drillhole_package, _ = _build_structured_drillhole_package()

    assert registry.register_drillhole_data(
        drillhole_package,
        source_panel="variogram-panel-irbf-domain-test",
        is_raw_import=True,
    )
    registry.set_drillholes_validation_state(
        status="PASS",
        timestamp=datetime.now().isoformat(),
        config_hash="variogram-panel-irbf-domain-test",
        fatal_count=0,
        warn_count=0,
        info_count=0,
        excluded_rows={},
    )

    composites = registry.get_drillhole_data()["composites"]
    labels = _build_irbf_labels(composites)
    registry.register_indicator_rbf_domain(
        {
            "domain_name": "Inside",
            "sample_domain_labels": labels,
            "sample_domain_index": list(composites.index),
            "sample_domain_column": "IRBF_Domain",
        },
        source_panel="variogram-panel-irbf-domain-test",
    )

    panel = VariogramAnalysisPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        combo_items = [panel.domain_combo.itemText(i) for i in range(panel.domain_combo.count())]
        assert "IRBF_Domain: Inside" in combo_items
        assert "IRBF_Domain: Outside" in combo_items

        panel.domain_combo.setCurrentText("IRBF_Domain: Inside")
        filtered = panel._apply_selected_domain_filter(registry.get_drillhole_data()["composites"])
        assert set(filtered["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside"}

        hydrated = registry.get_drillhole_data()["composites"]
        assert "IRBF_Domain" in hydrated.columns
        assert set(hydrated["IRBF_Domain"].dropna().astype(str).unique()) == {"Inside", "Outside"}
    finally:
        panel.close()
        QTest.qWait(100)
        mock_qapp.processEvents()
        registry.clear_drillhole_data()
        DataRegistry._instance = None
