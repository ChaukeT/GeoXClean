"""SGSIM level 5 — real-UI panel + controller-worker coverage.

Drives the real Qt ``SGSIMPanel`` through the full configuration
path a user clicks through: ``QApplication → DataRegistry →
AppController → SGSIMPanel → _on_data_loaded → widget configuration →
panel.gather_parameters → controller._geostats._prepare_sgsim_payload
→ run_full_sgsim_workflow → payload back``.

**Note on the QThread layer.** The shipped `JobWorker` uses
`multiprocessing.Pool` inside SGSIM (via ``SGSIMParameters.parallel``),
which deadlocks reliably on Windows when invoked from a pytest-driven
QThread — both this test and the existing
``test_sgsim_panel_runs_end_to_end_on_transformed_synthetic_data`` in
the panel-debugger suite hang for that reason in this environment.
To get real Qt-panel coverage without the deadlock, these tests
bypass `JobWorker.start()` and call
``controller._geostats._prepare_sgsim_payload(params)`` on the
**same thread**, with the default ``SGSIMParameters(parallel=True)``
replaced with a serial wrapper for the duration of the test. The
panel, its widgets, the registry wiring, the controller worker
function, the transformer handling, and the mesh-creation path are
all real. Only the QThread hand-off is elided.

Mirror in spirit of the real-UI ARBF test. The two existing
panel-debugger tests
(``test_sgsim_panel_runs_end_to_end_on_transformed_synthetic_data``,
``test_sgsim_panel_marks_fft_ma_runs_as_approximate``) share the
same fixture and remain in ``tests/panel_debugger/`` — these live
alongside the rest of the SGSIM suite in ``tests/geostats/`` so the
full pyramid is in one place.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pytest
from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QMessageBox

from block_model_viewer.controllers.app_controller import AppController
from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.models.transform import NormalScoreTransformer
from block_model_viewer.ui.sgsim_panel import SGSIMPanel
from tests.panel_debugger.core.fixtures import MockRenderer
from tests.panel_debugger.tests.test_variogram_panel_workflow import (
    _build_structured_drillhole_package,
)


def _fresh_registry() -> DataRegistry:
    existing = DataRegistry.get_existing()
    if existing is not None:
        try:
            existing.clear_drillhole_data()
        except Exception:
            pass
    DataRegistry._instance = None
    return DataRegistry.instance()


def _register_package(registry, package, source):
    assert registry.register_drillhole_data(
        package, source_panel=source, is_raw_import=True,
    )
    registry.set_drillholes_validation_state(
        status="PASS",
        timestamp=datetime.now().isoformat(),
        config_hash=source,
        fatal_count=0, warn_count=0, info_count=0, excluded_rows={},
    )


def _wait_until(app, predicate, timeout_ms: int = 60000) -> bool:
    start = time.time()
    while time.time() - start < timeout_ms / 1000.0:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _patch_message_boxes(monkeypatch):
    for name in ("warning", "critical", "information"):
        monkeypatch.setattr(
            QMessageBox, name,
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
        )
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
    )


def _force_serial_sgsim(monkeypatch):
    """Force SGSIM to run sequentially inside the worker thread.

    Rationale: the default ``SGSIMParameters`` has ``parallel=True``
    and ``n_jobs=-1``, which spawns a multiprocessing ``Pool`` inside
    the pytest-driven Qt worker on Windows. Spawning new processes
    from inside a worker thread while pytest is holding the interpreter
    deadlocks. For the real-UI test the grid is tiny (10×10×3 = 300
    cells, 3 realisations), so serial execution is actually faster
    than the pool spin-up cost.

    We wrap ``run_full_sgsim_workflow`` at the call sites the worker
    uses (both ``sgsim3d`` and ``geostats_controller``), force
    ``params.parallel = False`` and ``params.use_numba = False``, then
    delegate to the real implementation. This is strictly additive —
    no production code is touched, and the patch is session-scoped
    by pytest ``monkeypatch``.
    """
    from block_model_viewer.models import sgsim3d
    from block_model_viewer.controllers import geostats_controller

    _original_workflow = sgsim3d.run_full_sgsim_workflow

    def _serial_workflow(data_coords, data_values, params, *args, **kwargs):
        try:
            params.parallel = False
            params.n_jobs = 1
            params.use_numba = False
        except Exception:
            pass
        return _original_workflow(
            data_coords, data_values, params, *args, **kwargs
        )

    monkeypatch.setattr(sgsim3d, "run_full_sgsim_workflow", _serial_workflow)
    if hasattr(geostats_controller, "run_full_sgsim_workflow"):
        monkeypatch.setattr(
            geostats_controller,
            "run_full_sgsim_workflow",
            _serial_workflow,
        )


def _make_transformed_package(package):
    """Add an FE_PCT_NS column to the existing fixture package,
    fitting a NormalScoreTransformer on the raw composite data."""
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


def _configure_panel_standard(panel, registry, seed: int = 42,
                                method_text: str = "SGS"):
    """Apply the shared widget configuration. Signals on each widget
    are blocked during setValue/setCurrentText so the panel's
    auto-detection and variogram-autoload callbacks do not fire
    mid-config and overwrite the values we just set.
    """
    panel._on_data_loaded(registry.get_drillhole_data())
    panel.data_source_composited.blockSignals(True)
    panel.data_source_composited.setChecked(True)
    panel.data_source_composited.blockSignals(False)
    panel.variable_combo.blockSignals(True)
    panel.variable_combo.setCurrentText("FE_PCT_NS")
    panel.variable_combo.blockSignals(False)
    # Grid matches the panel-debugger reference test: 11×12×4 = 528
    # cells at origin (499800, 5999780, 435), spacing (40, 40, 20).
    # This covers the structured fixture's drillhole extent.
    for w, v in [
        (panel.nx, 11), (panel.ny, 12), (panel.nz, 4),
        (panel.xmin_spin, 499800.0), (panel.ymin_spin, 5999780.0),
        (panel.zmin_spin, 435.0),
        (panel.dx, 40.0), (panel.dy, 40.0), (panel.dz, 20.0),
        (panel.nreal_spin, 3), (panel.seed_spin, seed),
        (panel.rmaj, 120.0), (panel.rmin, 45.0), (panel.rver, 25.0),
        (panel.azim, 35.0), (panel.dip, 0.0),
        (panel.nug, 0.1), (panel.sill, 1.0),
    ]:
        w.blockSignals(True)
        w.setValue(v)
        w.blockSignals(False)
    panel.vario_type.blockSignals(True)
    panel.vario_type.setCurrentText("Spherical")
    panel.vario_type.blockSignals(False)
    # method_combo: match by stored data() rather than display label
    target_data = "fft_ma" if "FFT" in method_text.upper() else "sgs"
    for i in range(panel.method_combo.count()):
        if panel.method_combo.itemData(i) == target_data:
            panel.method_combo.setCurrentIndex(i)
            break


def _synchronous_sgsim(panel, controller, registry):
    """Configure panel, gather parameters, inject data_df, call the
    real ``_prepare_sgsim_payload`` on the same thread, return the
    payload. Avoids the JobWorker QThread deadlock on Windows.
    """
    assert panel.validate_inputs()
    params = panel.gather_parameters()
    # Controller normally injects data_df before dispatching to the
    # worker function. The dispatch happens inside
    # AppController.run_task which is what we're eliding — do the
    # same injection here.
    if "data_df" not in params or params["data_df"] is None:
        dh = registry.get_drillhole_data()
        if isinstance(dh, dict):
            df = dh.get("composites")
            if df is None:
                df = dh.get("assays")
            params["data_df"] = df
        else:
            params["data_df"] = dh
    geostats = controller._geostats
    return geostats._prepare_sgsim_payload(params)


@pytest.mark.integration
def test_sgsim_panel_end_to_end_on_fixture_deposit(mock_qapp, monkeypatch):
    """Real Qt panel + real controller worker function, same-thread
    dispatch. Uses the default ``sgs`` method.
    """
    _patch_message_boxes(monkeypatch)
    _force_serial_sgsim(monkeypatch)

    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    transformed_package, transformer = _make_transformed_package(package)
    _register_package(registry, transformed_package, "sgsim-real-ui-sgs")
    registry.register_transformation_metadata(
        {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS",
                                          "method": "normal_score"}}}
    )
    registry.register_transformers({"FE_PCT": transformer})

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = SGSIMPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        _configure_panel_standard(panel, registry, seed=42, method_text="SGS")
        mock_qapp.processEvents()

        payload = _synchronous_sgsim(panel, controller, registry)
        assert payload is not None
        assert payload.get("name") == "sgsim"
        results = payload.get("results") or {}
        assert "summary" in results
        assert "realizations_raw" in results
        assert "realizations_gaussian" in results

        # Core result shape — grid is (nx=10, ny=10, nz=3) = 300 cells
        summary = results["summary"]
        mean_grid = np.asarray(summary["mean"], float).ravel()
        std_grid = np.asarray(summary["std"], float).ravel()
        assert mean_grid.size == 11 * 12 * 4
        assert np.isfinite(mean_grid).all()
        assert np.isfinite(std_grid).all()

        # Metadata + diagnostics from the payload
        metadata = payload.get("metadata", {}) or {}
        assert "variable" in metadata
        assert metadata["variable"] == "FE_PCT_NS"
        results_meta = results.get("metadata", {}) or {}
        assert results_meta.get("simulation_method") == "sgs"
        assert results_meta.get("conditioning_is_approximate") is False

        diagnostics = results.get("diagnostics", {}) or {}
        cs = diagnostics.get("conditioning_support", {}) or {}
        vr = diagnostics.get("variogram_reproduction", {}) or {}
        assert float(cs.get("median_nearest_node_distance", -1)) >= 0.0
        assert int(vr.get("n_lags_compared", 0)) >= 3
        assert np.isfinite(float(vr.get("normalized_rmse", float("nan"))))

        # Visualization mesh should be present with summary cell data
        viz = payload.get("visualization") or {}
        assert viz.get("mesh") is not None
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_sgsim_panel_block_size_change_with_loaded_data(mock_qapp, monkeypatch):
    """Regression: changing the dx spinbox while drillhole data is
    loaded used to crash in ``_on_block_size_changed`` with
    ``ValueError: The truth value of a DataFrame is ambiguous`` because
    the handler tested ``self.drillhole_data`` in a boolean context.

    The handler now does an explicit ``is None`` + ``.empty`` check.
    This test fires the handler via the real dx valueChanged signal
    with a live DataFrame loaded, so the pandas truth-value path is
    actually exercised.
    """
    _patch_message_boxes(monkeypatch)

    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    transformed_package, transformer = _make_transformed_package(package)
    _register_package(registry, transformed_package, "sgsim-block-size")
    registry.register_transformation_metadata(
        {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS",
                                          "method": "normal_score"}}}
    )
    registry.register_transformers({"FE_PCT": transformer})

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = SGSIMPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        panel._on_data_loaded(registry.get_drillhole_data())
        panel.variable_combo.setCurrentText("FE_PCT_NS")
        # drillhole_data is now a real DataFrame on the panel
        assert panel.drillhole_data is not None
        # Fire the signal path WITHOUT blockSignals — this is what
        # reproduces the pandas-truth-value crash before the fix.
        panel.dx.setValue(35.0)
        mock_qapp.processEvents()
        # No exception above = fix holds. Verify dx actually changed.
        assert panel.dx.value() == 35.0
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None


@pytest.mark.integration
def test_sgsim_panel_method_combo_selects_fft_ma(mock_qapp, monkeypatch):
    """Real Qt panel widget-path verification for the ``fft_ma``
    method selection. Asserts that selecting the FFT-MA entry in
    ``method_combo`` is correctly reported by ``gather_parameters``
    as ``simulation_method="fft_ma"``.

    **Why this test is scoped to the panel widget path, not the
    full controller-worker path:** in the current codebase,
    ``GeostatsController._prepare_sgsim_payload`` at lines 1366–1389
    constructs ``SGSIMParameters`` without passing the user's method
    choice through — it always defaults to the dataclass default
    ``method='sgs'``. That's a pre-existing wiring gap between the
    panel and the worker, documented here so future fixes can
    promote this test to full end-to-end coverage. Fixing the
    wiring is out of scope for this test-only addition.

    The corresponding end-to-end FFT-MA test in the panel-debugger
    suite references ``panel.sim_method_combo``, which does not
    exist in the current sgsim_panel.py — so that test is itself
    stale against the live code. Both gaps are for a follow-up.
    """
    _patch_message_boxes(monkeypatch)

    registry = _fresh_registry()
    package, _ = _build_structured_drillhole_package()
    transformed_package, transformer = _make_transformed_package(package)
    _register_package(registry, transformed_package, "sgsim-real-ui-fftma")
    registry.register_transformation_metadata(
        {"transformations": {"FE_PCT": {"new_col": "FE_PCT_NS",
                                          "method": "normal_score"}}}
    )
    registry.register_transformers({"FE_PCT": transformer})

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = SGSIMPanel()
    panel.bind_controller(controller)
    panel.show()
    mock_qapp.processEvents()

    try:
        _configure_panel_standard(panel, registry, seed=42, method_text="FFT")
        mock_qapp.processEvents()

        # Panel widget-path: the method combo must report fft_ma
        assert panel.method_combo.currentData() == "fft_ma"

        # gather_parameters must surface the selection
        params = panel.gather_parameters()
        assert params.get("simulation_method") == "fft_ma"

        # Other params must still be valid (panel config applied cleanly)
        assert params.get("nreal") == 3
        assert params.get("nx") == 11
        assert params.get("ny") == 12
        assert params.get("nz") == 4
        assert params.get("variable") == "FE_PCT_NS"
        assert params.get("transformer") is not None
    finally:
        panel.close()
        registry.clear_drillhole_data()
        DataRegistry._instance = None
