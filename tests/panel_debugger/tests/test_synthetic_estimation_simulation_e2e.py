import time
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QMessageBox

from block_model_viewer.controllers.app_controller import AppController
from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.models.kriging3d import get_variogram_function
from block_model_viewer.models.sgsim3d import (
    _covariance_matrix_from_distances,
    _covariance_vector_from_distances,
    _normalize_simulation_method,
)
from block_model_viewer.models.transform import NormalScoreTransformer
from block_model_viewer.ui.kriging_panel import KrigingPanel
from block_model_viewer.ui.sgsim_panel import SGSIMPanel
from tests.panel_debugger.core.fixtures import MockRenderer
from tests.panel_debugger.tests.test_variogram_panel_workflow import (
    _build_structured_drillhole_package,
)


COLLAR_ELEV = 500.0


def test_sgsim_covariance_uses_total_variance_for_self_covariances():
    sill = 1.0
    nugget = 0.2
    vario_func = get_variogram_function("spherical")

    pair_dists = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=float)
    cov_matrix = _covariance_matrix_from_distances(
        pair_dists,
        vario_func,
        sill,
        nugget,
    )
    cov_vector = _covariance_vector_from_distances(
        np.array([0.0, 0.5], dtype=float),
        vario_func,
        sill,
        nugget,
    )

    expected_offdiag = sill - float(vario_func(np.array([0.5]), 1.0, sill, nugget)[0])

    np.testing.assert_allclose(np.diag(cov_matrix), np.array([sill, sill]), atol=1e-12)
    assert cov_matrix[0, 1] == pytest.approx(expected_offdiag)
    assert cov_matrix[1, 0] == pytest.approx(expected_offdiag)
    assert cov_vector[0] == pytest.approx(sill)
    assert cov_vector[1] == pytest.approx(expected_offdiag)


def test_sgsim_method_aliases_normalize_to_two_supported_methods():
    assert _normalize_simulation_method("sgs") == "sgs"
    assert _normalize_simulation_method("sgsim") == "sgs"
    assert _normalize_simulation_method("sequential") == "sgs"
    assert _normalize_simulation_method("fft_ma") == "fft_ma"
    assert _normalize_simulation_method("fftma") == "fft_ma"


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


def _wait_until(mock_qapp, predicate, timeout_ms=30000):
    start = time.time()
    while time.time() - start < timeout_ms / 1000.0:
        mock_qapp.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _patch_message_boxes(monkeypatch):
    monkeypatch.setattr(
        QMessageBox,
        "warning",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok),
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes),
    )


def _build_truth_function(seed: int = 20260319):
    rng = np.random.RandomState(seed)
    azimuth_deg = 35.0
    azimuth_rad = np.deg2rad(azimuth_deg)
    rotation = np.array(
        [
            [np.sin(azimuth_rad), np.cos(azimuth_rad)],
            [np.cos(azimuth_rad), -np.sin(azimuth_rad)],
        ],
        dtype=float,
    )

    true_major = 120.0
    true_minor = 45.0
    true_vertical = 25.0
    true_nugget = 0.1
    true_sill = 1.0
    reference_range = 80.0
    xy_offset = np.array([500000.0, 6000000.0], dtype=float)

    u = np.linspace(-180.0, 180.0, 7)
    v = np.linspace(-90.0, 90.0, 5)
    collars_uv = np.array(np.meshgrid(u, v, indexing="ij")).reshape(2, -1).T

    support_xyz = []
    for uv in collars_uv:
        for interval_index in range(24):
            mid_depth = 0.5 * (interval_index * 2.5 + (interval_index + 1) * 2.5)
            support_xyz.append([uv[0], uv[1], -mid_depth])
        for interval_index in range(12):
            mid_depth = 0.5 * (interval_index * 5.0 + (interval_index + 1) * 5.0)
            support_xyz.append([uv[0], uv[1], -mid_depth])

    support_xyz = np.asarray(support_xyz, dtype=float)
    scaled_support = np.column_stack(
        [
            support_xyz[:, 0] * (reference_range / true_major),
            support_xyz[:, 1] * (reference_range / true_minor),
            support_xyz[:, 2] * (reference_range / true_vertical),
        ]
    )

    directions = rng.randn(300, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    phases = rng.uniform(0.0, 2.0 * np.pi, len(directions))

    raw_values = np.zeros(len(support_xyz), dtype=float)
    for direction, phase in zip(directions, phases):
        raw_values += np.cos(
            2.0 * np.pi * (1.0 / reference_range) * (scaled_support @ direction) + phase
        )

    raw_mean = raw_values.mean()
    raw_std = raw_values.std()

    def truth_fn(global_xyz):
        pts = np.asarray(global_xyz, dtype=float).reshape(-1, 3)
        local_xy = (pts[:, :2] - xy_offset) @ rotation
        local_z = pts[:, 2] - COLLAR_ELEV
        uvz = np.column_stack([local_xy[:, 0], local_xy[:, 1], local_z])
        scaled = np.column_stack(
            [
                uvz[:, 0] * (reference_range / true_major),
                uvz[:, 1] * (reference_range / true_minor),
                uvz[:, 2] * (reference_range / true_vertical),
            ]
        )
        values = np.zeros(len(pts), dtype=float)
        for direction, phase in zip(directions, phases):
            values += np.cos(
                2.0 * np.pi * (1.0 / reference_range) * (scaled @ direction) + phase
            )
        values -= raw_mean
        values *= np.sqrt(true_sill - true_nugget) / (raw_std + 1e-12)
        return values

    return truth_fn


def _make_transformed_package(package):
    transformer = NormalScoreTransformer()
    transformer.fit(package["composites"]["FE_PCT"].to_numpy())

    transformed = dict(package)
    transformed["assays"] = package["assays"].copy()
    transformed["composites"] = package["composites"].copy()
    transformed["assays"]["FE_PCT_NS"] = transformer.transform(
        transformed["assays"]["FE_PCT"].to_numpy()
    )
    transformed["composites"]["FE_PCT_NS"] = transformer.transform(
        transformed["composites"]["FE_PCT"].to_numpy()
    )
    return transformed, transformer


def _shift_grade_package(package, offset):
    shifted = dict(package)
    shifted["assays"] = package["assays"].copy()
    shifted["composites"] = package["composites"].copy()
    shifted["assays"]["FE_PCT"] = shifted["assays"]["FE_PCT"].astype(float) + float(offset)
    shifted["composites"]["FE_PCT"] = shifted["composites"]["FE_PCT"].astype(float) + float(offset)
    return shifted


def _build_collocated_grid_package():
    coords = []
    collars = []
    surveys = []
    assays = []
    composites = []
    values = np.array([0.20, 0.85, -0.35, 0.60, 1.10, -0.55, 0.15, 0.95], dtype=float)

    sample_index = 0
    for ix, x in enumerate((5.0, 15.0), start=1):
        for iy, y in enumerate((5.0, 15.0), start=1):
            for iz, z in enumerate((5.0, 15.0), start=1):
                hole_id = f"CG{sample_index + 1:02d}"
                collars.append(
                    {
                        "HOLEID": hole_id,
                        "EAST": x,
                        "NORTH": y,
                        "ELEV": z + 5.0,
                        "DEPTH": 1.0,
                        "AZIMUTH": 0.0,
                        "DIP": -90.0,
                    }
                )
                surveys.extend(
                    [
                        {"HOLEID": hole_id, "DEPTH": 0.0, "AZIMUTH": 0.0, "DIP": -90.0},
                        {"HOLEID": hole_id, "DEPTH": 1.0, "AZIMUTH": 0.0, "DIP": -90.0},
                    ]
                )
                row = {
                    "HOLEID": hole_id,
                    "FROM": 0.0,
                    "TO": 1.0,
                    "INTERVAL_ID": f"{hole_id}_C_0",
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "FE_PCT": float(values[sample_index]),
                }
                composites.append(row)
                assays.append(dict(row, INTERVAL_ID=f"{hole_id}_A_0"))
                sample_index += 1

    return {
        "collars": pd.DataFrame(collars),
        "surveys": pd.DataFrame(surveys),
        "assays": pd.DataFrame(assays),
        "composites": pd.DataFrame(composites),
    }


@pytest.mark.integration
def test_synthetic_truth_reconstruction_matches_panel_sample_support(mock_qapp):
    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    truth_fn = _build_truth_function()
    _register_package(registry, package, "synthetic-truth-support-test")

    panel = KrigingPanel()
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_drillhole_data_loaded(registry.get_drillhole_data())
        df = panel.drillhole_data.copy()

        truth = truth_fn(df[["X", "Y", "Z"]].to_numpy())
        observed = df["FE_PCT"].to_numpy(dtype=float)
        corr = float(np.corrcoef(truth, observed)[0, 1])
        rmse = float(np.sqrt(np.mean((truth - observed) ** 2)))

        assert len(df) == 420
        assert corr >= 0.94
        assert rmse <= 0.35
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_ordinary_kriging_panel_runs_end_to_end_on_synthetic_data(mock_qapp, monkeypatch):
    _patch_message_boxes(monkeypatch)

    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    _register_package(registry, package, "synthetic-ordinary-kriging-e2e")

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = KrigingPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        panel.data_source_composited.setChecked(True)
        panel._on_drillhole_data_loaded(registry.get_drillhole_data())
        panel.variable_combo.setCurrentText("FE_PCT")
        panel.grid_x_spin.setValue(40.0)
        panel.grid_y_spin.setValue(40.0)
        panel.grid_z_spin.setValue(20.0)
        panel.model_combo.setCurrentText("Spherical")
        panel.range_spin.setValue(120.0)
        panel.sill_spin.setValue(0.9)
        panel.nugget_spin.setValue(0.1)
        panel.azimuth_spin.setValue(35.0)
        panel.dip_spin.setValue(0.0)
        panel.variogram_results = {
            "combined_3d_model": {
                "major_range": 120.0,
                "minor_range": 45.0,
                "vertical_range": 25.0,
                "nugget": 0.1,
                "sill": 1.0,
                "model_type": "spherical",
            }
        }

        assert panel.validate_inputs()
        panel.run_analysis()

        assert _wait_until(
            mock_qapp,
            lambda: getattr(panel, "kriging_results", None) is not None,
        )

        results = panel.kriging_results
        estimates = np.asarray(results["estimates"], dtype=float).ravel()
        coords = np.column_stack(
            [
                np.asarray(results["grid_x"]).ravel(),
                np.asarray(results["grid_y"]).ravel(),
                np.asarray(results["grid_z"]).ravel(),
            ]
        )

        assert coords.shape == (528, 3)
        assert estimates.shape == (528,)
        assert np.isfinite(estimates).all()
        assert 0.3 <= float(np.std(estimates)) <= 0.8
        assert abs(float(np.mean(estimates))) <= 0.15
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_sgsim_panel_runs_end_to_end_on_transformed_synthetic_data(mock_qapp, monkeypatch):
    _patch_message_boxes(monkeypatch)

    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    transformed_package, transformer = _make_transformed_package(package)
    _register_package(registry, transformed_package, "synthetic-sgsim-e2e")
    registry.register_transformation_metadata(
        {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS", "method": "normal_score"}}}
    )
    registry.register_transformers({"FE_PCT": transformer})

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = SGSIMPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())
        panel.data_source_composited.setChecked(True)
        panel.variable_combo.setCurrentText("FE_PCT_NS")
        panel.nx.setValue(11)
        panel.ny.setValue(12)
        panel.nz.setValue(4)
        panel.xmin_spin.setValue(499800.0)
        panel.ymin_spin.setValue(5999780.0)
        panel.zmin_spin.setValue(435.0)
        panel.dx.setValue(40.0)
        panel.dy.setValue(40.0)
        panel.dz.setValue(20.0)
        panel.nreal_spin.setValue(5)
        panel.seed_spin.setValue(42)
        panel.vario_type.setCurrentText("Spherical")
        panel.rmaj.setValue(120.0)
        panel.rmin.setValue(45.0)
        panel.rver.setValue(25.0)
        panel.azim.setValue(35.0)
        panel.dip.setValue(0.0)
        panel.nug.setValue(0.1)
        panel.sill.setValue(1.0)

        assert panel.validate_inputs()
        panel.run_analysis()

        assert _wait_until(
            mock_qapp,
            lambda: getattr(panel, "sgsim_results", None) is not None,
        )

        summary = panel.sgsim_results["summary"]
        mean_grid = np.asarray(summary["mean"], dtype=float).ravel()
        std_grid = np.asarray(summary["std"], dtype=float).ravel()
        mesh = ((panel.sgsim_payload or {}).get("visualization") or {}).get("mesh")
        metadata = (panel.sgsim_payload or {}).get("metadata", {})
        diagnostics = panel.sgsim_results.get("diagnostics", {})

        assert mesh is not None
        assert mesh.n_cells == 528
        assert mean_grid.shape == (528,)
        assert std_grid.shape == (528,)
        assert np.isfinite(mean_grid).all()
        assert np.isfinite(std_grid).all()
        assert abs(float(np.mean(mean_grid))) <= 0.2
        assert float(np.nanmean(std_grid)) >= 0.3
        assert metadata["simulation_method"] == "sgs"
        assert metadata["simulation_method_label"] == "SGS (Datamine-style)"
        assert metadata["conditioning_mode"] == "exact_sgs"
        assert metadata["conditioning_is_approximate"] is False
        assert metadata["execution_engine"] == "python_exact"
        assert diagnostics["conditioning_support"]["median_nearest_node_distance"] > 0.0
        assert diagnostics["conditioning_support"]["n_unique_collocated_nodes"] == 0
        assert diagnostics["variogram_reproduction"]["n_lags_compared"] >= 3
        assert np.isfinite(diagnostics["variogram_reproduction"]["normalized_rmse"])
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_sgsim_panel_marks_fft_ma_runs_as_approximate(mock_qapp, monkeypatch):
    _patch_message_boxes(monkeypatch)

    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    transformed_package, transformer = _make_transformed_package(package)
    _register_package(registry, transformed_package, "synthetic-sgsim-fftma-e2e")
    registry.register_transformation_metadata(
        {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS", "method": "normal_score"}}}
    )
    registry.register_transformers({"FE_PCT": transformer})

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = SGSIMPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())
        panel.data_source_composited.setChecked(True)
        panel.variable_combo.setCurrentText("FE_PCT_NS")
        panel.nx.setValue(11)
        panel.ny.setValue(12)
        panel.nz.setValue(4)
        panel.xmin_spin.setValue(499800.0)
        panel.ymin_spin.setValue(5999780.0)
        panel.zmin_spin.setValue(435.0)
        panel.dx.setValue(40.0)
        panel.dy.setValue(40.0)
        panel.dz.setValue(20.0)
        panel.nreal_spin.setValue(2)
        panel.seed_spin.setValue(42)
        panel.vario_type.setCurrentText("Spherical")
        panel.rmaj.setValue(120.0)
        panel.rmin.setValue(45.0)
        panel.rver.setValue(25.0)
        panel.azim.setValue(35.0)
        panel.dip.setValue(0.0)
        panel.nug.setValue(0.1)
        panel.sill.setValue(1.0)
        panel.sim_method_combo.setCurrentIndex(panel.sim_method_combo.findData("fft_ma"))

        assert panel.validate_inputs()
        panel.run_analysis()

        assert _wait_until(
            mock_qapp,
            lambda: getattr(panel, "sgsim_results", None) is not None,
        )

        metadata = (panel.sgsim_payload or {}).get("metadata", {})
        assert metadata["simulation_method"] == "fft_ma"
        assert metadata["conditioning_mode"] == "approximate_fft_ma"
        assert metadata["conditioning_is_approximate"] is True
        assert metadata["execution_engine"] == "fft_ma"
        assert "Approximate FFT-MA" in metadata["message"]
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_sgsim_panel_is_reproducible_with_fixed_seed(mock_qapp, monkeypatch):
    _patch_message_boxes(monkeypatch)

    def run_once(source_name):
        registry = _fresh_registry()
        package, _ = _build_structured_drillhole_package()
        transformed_package, transformer = _make_transformed_package(package)
        _register_package(registry, transformed_package, source_name)
        registry.register_transformation_metadata(
            {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS", "method": "normal_score"}}}
        )
        registry.register_transformers({"FE_PCT": transformer})

        controller = AppController(renderer=MockRenderer(), registry=registry)
        panel = SGSIMPanel()
        panel.bind_controller(controller)
        panel.show()
        mock_qapp.processEvents()

        try:
            panel._on_data_loaded(registry.get_drillhole_data())
            panel.data_source_composited.setChecked(True)
            panel.variable_combo.setCurrentText("FE_PCT_NS")
            panel.nx.setValue(11)
            panel.ny.setValue(12)
            panel.nz.setValue(4)
            panel.xmin_spin.setValue(499800.0)
            panel.ymin_spin.setValue(5999780.0)
            panel.zmin_spin.setValue(435.0)
            panel.dx.setValue(40.0)
            panel.dy.setValue(40.0)
            panel.dz.setValue(20.0)
            panel.nreal_spin.setValue(3)
            panel.seed_spin.setValue(42)
            panel.vario_type.setCurrentText("Spherical")
            panel.rmaj.setValue(120.0)
            panel.rmin.setValue(45.0)
            panel.rver.setValue(25.0)
            panel.azim.setValue(35.0)
            panel.dip.setValue(0.0)
            panel.nug.setValue(0.1)
            panel.sill.setValue(1.0)

            assert panel.validate_inputs()
            panel.run_analysis()

            assert _wait_until(
                mock_qapp,
                lambda: getattr(panel, "sgsim_results", None) is not None,
            )

            summary = panel.sgsim_results["summary"]
            metadata = (panel.sgsim_payload or {}).get("metadata", {})
            return (
                np.asarray(summary["mean"], dtype=float).copy(),
                np.asarray(summary["std"], dtype=float).copy(),
                metadata.copy(),
            )
        finally:
            panel.close()
            registry.clear_drillhole_data()
            DataRegistry._instance = None

    mean_a, std_a, meta_a = run_once("synthetic-sgsim-repro-a")
    mean_b, std_b, meta_b = run_once("synthetic-sgsim-repro-b")

    assert np.allclose(mean_a, mean_b, equal_nan=True)
    assert np.allclose(std_a, std_b, equal_nan=True)
    assert meta_a["simulation_method"] == meta_b["simulation_method"] == "sgs"
    assert meta_a["simulation_method_label"] == meta_b["simulation_method_label"] == "SGS (Datamine-style)"
    assert meta_a["execution_engine"] == meta_b["execution_engine"] == "python_exact"


@pytest.mark.integration
def test_sgsim_validation_table_separates_gaussian_and_physical_metrics(mock_qapp, monkeypatch):
    _patch_message_boxes(monkeypatch)

    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    shifted_package = _shift_grade_package(package, 2.5)
    transformed_package, transformer = _make_transformed_package(shifted_package)
    _register_package(registry, transformed_package, "synthetic-sgsim-validation-metrics")
    registry.register_transformation_metadata(
        {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS", "method": "normal_score"}}}
    )
    registry.register_transformers({"FE_PCT": transformer})

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = SGSIMPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())
        panel.data_source_composited.setChecked(True)
        panel.variable_combo.setCurrentText("FE_PCT_NS")
        panel.nx.setValue(11)
        panel.ny.setValue(12)
        panel.nz.setValue(4)
        panel.xmin_spin.setValue(499800.0)
        panel.ymin_spin.setValue(5999780.0)
        panel.zmin_spin.setValue(435.0)
        panel.dx.setValue(40.0)
        panel.dy.setValue(40.0)
        panel.dz.setValue(20.0)
        panel.nreal_spin.setValue(5)
        panel.seed_spin.setValue(42)
        panel.vario_type.setCurrentText("Spherical")
        panel.rmaj.setValue(120.0)
        panel.rmin.setValue(45.0)
        panel.rver.setValue(25.0)
        panel.azim.setValue(35.0)
        panel.dip.setValue(0.0)
        panel.nug.setValue(0.1)
        panel.sill.setValue(1.0)

        assert panel.validate_inputs()
        panel.run_analysis()

        assert _wait_until(
            mock_qapp,
            lambda: getattr(panel, "sgsim_results", None) is not None,
        )

        gaussian_mean = float(np.nanmean(panel.sgsim_results["summary_gaussian"]["mean"]))
        physical_mean = float(np.nanmean(panel.sgsim_results["summary"]["mean"]))
        log_text = panel.results_text.toPlainText()

        assert abs(gaussian_mean) <= 0.2
        assert physical_mean >= 1.5
        assert "Gaussian data var." in log_text
        assert "Gaussian mean mean" in log_text
        assert "Physical mean mean" in log_text
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_sgsim_collocated_hard_data_diagnostics_detect_exact_honoring(mock_qapp, monkeypatch):
    _patch_message_boxes(monkeypatch)

    registry = _fresh_registry()
    package = _build_collocated_grid_package()
    transformed_package, transformer = _make_transformed_package(package)
    _register_package(registry, transformed_package, "synthetic-sgsim-collocated-hard-data")
    registry.register_transformation_metadata(
        {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS", "method": "normal_score"}}}
    )
    registry.register_transformers({"FE_PCT": transformer})

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = SGSIMPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())
        panel.data_source_composited.setChecked(True)
        panel.variable_combo.setCurrentText("FE_PCT_NS")
        panel.nx.setValue(2)
        panel.ny.setValue(2)
        panel.nz.setValue(2)
        panel.xmin_spin.setValue(0.0)
        panel.ymin_spin.setValue(0.0)
        panel.zmin_spin.setValue(0.0)
        panel.dx.setValue(10.0)
        panel.dy.setValue(10.0)
        panel.dz.setValue(10.0)
        panel.nreal_spin.setValue(4)
        panel.seed_spin.setValue(42)
        panel.vario_type.setCurrentText("Spherical")
        panel.rmaj.setValue(30.0)
        panel.rmin.setValue(30.0)
        panel.rver.setValue(30.0)
        panel.azim.setValue(0.0)
        panel.dip.setValue(0.0)
        panel.nug.setValue(0.05)
        panel.sill.setValue(1.0)

        assert panel.validate_inputs()
        panel.run_analysis()

        assert _wait_until(
            mock_qapp,
            lambda: getattr(panel, "sgsim_results", None) is not None,
        )

        diagnostics = panel.sgsim_results["diagnostics"]["conditioning_support"]

        assert diagnostics["n_unique_collocated_nodes"] == 8
        assert diagnostics["n_duplicate_collocated_nodes"] == 0
        assert diagnostics["n_strict_hard_data_nodes"] == 8
        assert diagnostics["hard_data_honoured_nodes"] == 8
        assert diagnostics["hard_data_mean_abs_error"] <= 1e-3
        assert diagnostics["hard_data_max_realization_std"] <= 1e-3
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None
