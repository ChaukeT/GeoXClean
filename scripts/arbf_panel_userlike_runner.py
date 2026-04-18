import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QEventLoop
from PyQt6.QtWidgets import QApplication, QMessageBox

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from block_model_viewer.controllers.app_controller import AppController
from block_model_viewer.core.data_registry import DataRegistry
from block_model_viewer.ui.arbf_estimation_panel import ARBFEstimationPanel
from tests.panel_debugger.core.fixtures import MockRenderer


OUTPUT_PATH = ROOT / "audit_logs" / "arbf_panel_userlike_20260320.json"


def _patch_message_boxes() -> None:
    QMessageBox.warning = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.critical = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.information = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Ok)
    QMessageBox.question = staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.Yes)


def _to_builtin(value):
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _mapping_from_object(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


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


def _rotation_matrix(azimuth_deg: float) -> np.ndarray:
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    return np.array(
        [
            [np.sin(azimuth_rad), np.cos(azimuth_rad)],
            [np.cos(azimuth_rad), -np.sin(azimuth_rad)],
        ],
        dtype=float,
    )


def _build_userlike_cu_package(seed: int = 20260320):
    rng = np.random.default_rng(seed)

    grid_spec = {
        "nx": 155,
        "ny": 123,
        "nz": 49,
        "dx": 10.0,
        "dy": 10.0,
        "dz": 10.0,
        "x0": 257740.0,
        "y0": 7921980.0,
        "z0": 90.0,
    }
    center = np.array(
        [
            grid_spec["x0"] + 0.5 * grid_spec["nx"] * grid_spec["dx"],
            grid_spec["y0"] + 0.5 * grid_spec["ny"] * grid_spec["dy"],
            grid_spec["z0"] + 0.5 * grid_spec["nz"] * grid_spec["dz"],
        ],
        dtype=float,
    )

    major_azimuth = 236.3
    rotation = _rotation_matrix(major_azimuth)
    range_max = 123.9
    range_mid = 92.2
    range_min = 28.0
    nugget_fraction = 0.06297780428508051
    structured_sill = 0.9363666305607734
    clip_max = 32468.3333
    reference_range = 80.0

    hole_u = np.linspace(-420.0, 420.0, 7)
    hole_v = np.linspace(-300.0, 300.0, 6)
    collars_local = np.array(np.meshgrid(hole_u, hole_v, indexing="ij")).reshape(2, -1).T
    collars_xy = collars_local + rng.normal(0.0, 12.0, size=collars_local.shape)
    collars_xy += center[:2]

    holes_with_48 = set(range(20))
    collar_elev = 580.0
    assay_length = 10.0

    raw_points = []
    assays = []
    composites = []
    collars = []
    surveys = []

    for hole_index, xy in enumerate(collars_xy, start=1):
        hole_id = f"CU{hole_index:03d}"
        n_intervals = 48 if (hole_index - 1) in holes_with_48 else 47
        depth = float(n_intervals * assay_length)
        collars.append(
            {
                "HOLEID": hole_id,
                "EAST": float(xy[0]),
                "NORTH": float(xy[1]),
                "ELEV": collar_elev,
                "DEPTH": depth,
                "AZIMUTH": 0.0,
                "DIP": -90.0,
            }
        )
        surveys.extend(
            [
                {"HOLEID": hole_id, "DEPTH": 0.0, "AZIMUTH": 0.0, "DIP": -90.0},
                {"HOLEID": hole_id, "DEPTH": depth, "AZIMUTH": 0.0, "DIP": -90.0},
            ]
        )

        for interval_index in range(n_intervals):
            from_depth = float(interval_index * assay_length)
            to_depth = float((interval_index + 1) * assay_length)
            mid_depth = 0.5 * (from_depth + to_depth)
            mid_xyz = np.array([xy[0], xy[1], collar_elev - mid_depth], dtype=float)
            raw_points.append(mid_xyz)

            row = {
                "HOLEID": hole_id,
                "FROM": from_depth,
                "TO": to_depth,
                "INTERVAL_ID": f"{hole_id}_I_{interval_index:03d}",
                "X": float(mid_xyz[0]),
                "Y": float(mid_xyz[1]),
                "Z": float(mid_xyz[2]),
            }
            assays.append(dict(row))
            composites.append(dict(row))

    composite_coords = np.asarray(raw_points, dtype=float)

    directions = rng.normal(0.0, 1.0, size=(280, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=len(directions))

    def _raw_field(xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xyz, dtype=float).reshape(-1, 3)
        local_xy = (pts[:, :2] - center[:2]) @ rotation
        local = np.column_stack(
            [
                local_xy[:, 0] * (reference_range / range_max),
                local_xy[:, 1] * (reference_range / range_mid),
                (pts[:, 2] - center[2]) * (reference_range / range_min),
            ]
        )
        values = np.zeros(len(pts), dtype=float)
        for direction, phase in zip(directions, phases):
            values += np.cos(
                2.0 * np.pi * (local @ direction) / reference_range + phase,
            )
        return values

    raw_support = _raw_field(composite_coords)
    raw_mean = float(np.mean(raw_support))
    raw_std = float(np.std(raw_support))

    def _structured_field(xyz: np.ndarray) -> np.ndarray:
        raw = _raw_field(xyz)
        return (raw - raw_mean) * np.sqrt(structured_sill) / max(raw_std, 1e-12)

    def _grade_from_gaussian(gaussian_values: np.ndarray) -> np.ndarray:
        grades = np.exp(8.0 + 0.90 * np.asarray(gaussian_values, dtype=float))
        return np.clip(grades, 0.0, clip_max)

    structured_support = _structured_field(composite_coords)
    gaussian_samples = structured_support + rng.normal(
        0.0,
        np.sqrt(nugget_fraction),
        size=len(structured_support),
    )
    sample_grades = _grade_from_gaussian(gaussian_samples)

    assays_df = pd.DataFrame(assays)
    composites_df = pd.DataFrame(composites)
    assays_df["Cu"] = sample_grades
    composites_df["Cu"] = sample_grades

    package = {
        "collars": pd.DataFrame(collars),
        "surveys": pd.DataFrame(surveys),
        "assays": assays_df,
        "composites": composites_df,
    }

    meta = {
        "grid_spec": grid_spec,
        "center": center,
        "major_azimuth": major_azimuth,
        "range_max": range_max,
        "range_mid": range_mid,
        "range_min": range_min,
        "nugget_fraction": nugget_fraction,
        "structured_sill": structured_sill,
        "clip_max": clip_max,
        "n_composites": int(len(composites_df)),
        "n_holes": int(len(collars)),
        "sample_mean": float(np.mean(sample_grades)),
        "sample_std": float(np.std(sample_grades)),
        "sample_min": float(np.min(sample_grades)),
        "sample_max": float(np.max(sample_grades)),
    }

    def truth_grade(xyz: np.ndarray) -> np.ndarray:
        return _grade_from_gaussian(_structured_field(xyz))

    return package, truth_grade, meta


def _extract_result_coords(payload: dict) -> np.ndarray:
    grades = np.asarray(payload["grades"], dtype=float).ravel()
    x_coords = np.asarray(payload["x_coords"], dtype=float).ravel()
    y_coords = np.asarray(payload["y_coords"], dtype=float).ravel()
    z_coords = np.asarray(payload["z_coords"], dtype=float).ravel()

    if x_coords.size * y_coords.size * z_coords.size == grades.size and x_coords.size != grades.size:
        return np.column_stack(
            [
                axis.ravel()
                for axis in np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")
            ]
        )

    if x_coords.size == y_coords.size == z_coords.size == grades.size:
        return np.column_stack([x_coords, y_coords, z_coords])

    raise ValueError(
        f"Could not reconstruct block coordinates: x={x_coords.size}, y={y_coords.size}, "
        f"z={z_coords.size}, grades={grades.size}",
    )


def _wait_until(app: QApplication, predicate, timeout_s: float = 900.0) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _configure_panel(panel: ARBFEstimationPanel, *, clip_to_footprint: bool) -> None:
    panel.source_combo.setCurrentText("Composited Drillholes")
    panel._on_load_data()
    panel.variable_combo.setCurrentText("Cu")

    panel.kernel_combo.setCurrentText("Spherical")
    panel.alpha_spin.setValue(1.0)
    panel.sill_spin.setValue(0.9363666305607734)
    panel.nugget_spin.setValue(0.06297780428508051)
    panel.accuracy_spin.setValue(1e-6)
    panel.drift_combo.setCurrentText("Constant")
    panel.variogram_mode_combo.setCurrentText("Global")

    panel.range_max_spin.setValue(123.9)
    panel.range_mid_spin.setValue(92.2)
    panel.range_min_spin.setValue(28.0)
    panel.azimuth_spin.setValue(236.3)
    panel.dip_spin.setValue(0.0)
    panel.pitch_spin.setValue(0.0)
    panel.rotation_convention_combo.setCurrentText("GeoX (native)")

    panel.nx_spin.setValue(155)
    panel.ny_spin.setValue(123)
    panel.nz_spin.setValue(49)
    panel.dx_spin.setValue(10.0)
    panel.dy_spin.setValue(10.0)
    panel.dz_spin.setValue(10.0)
    panel.x0_spin.setValue(257740.0)
    panel.y0_spin.setValue(7921980.0)
    panel.z0_spin.setValue(90.0)
    panel.chk_clip_to_footprint.setChecked(bool(clip_to_footprint))
    panel.footprint_buffer_spin.setValue(1.5)

    panel.cb_normal_score.setChecked(True)
    panel.cb_ilr.setChecked(False)
    panel.cb_grade_clip.setChecked(True)
    panel.clip_min_spin.setValue(0.0)
    panel.clip_max_spin.setValue(32468.3333)
    panel.estimation_mode_combo.setCurrentText("Local Neighbourhood GPR")

    panel.max_samples_spin.setValue(300)
    panel.min_samples_spin.setValue(4)
    panel.search_radius_1_spin.setValue(0.75)
    panel.search_radius_2_spin.setValue(1.50)
    panel.search_radius_3_spin.setValue(3.00)
    panel.cb_balanced_search.setChecked(True)
    panel.search_min_octants_spin.setValue(3)
    panel.search_min_octants_linear_spin.setValue(4)
    panel.max_samples_per_octant_spin.setValue(4)
    panel.auto_drift_slope_spin.setValue(0.20)
    panel.cb_single_domain.setChecked(False)
    panel.pum_threshold_spin.setValue(3000)
    panel.overlap_spin.setValue(2.0)
    panel.disc_mode_combo.setCurrentText("Fixed")
    panel.disc_density_combo.setCurrentText("27 (3x3x3)")
    panel.cb_lva.setChecked(True)
    panel.lva_source_combo.setCurrentIndex(0)
    panel.cb_geodesic.setChecked(False)

    panel.cb_run_cv.setChecked(True)
    panel.cv_mode_combo.setCurrentText("Spatial K-Fold")
    panel.cv_folds_spin.setValue(5)
    panel.cb_cos.setChecked(True)
    panel.cos_mode_combo.setCurrentText("Discretized Block Support")
    panel.cb_prefilter_blocks.setChecked(False)
    panel.decluster_spin.setValue(0.0)
    panel.seed_spin.setValue(42)
    panel.simulation_seed_spin.setValue(1042)


def _summarize_payload(case_name: str, payload: dict, truth_fn) -> dict:
    diagnostics = payload.get("diagnostics", {}) or {}
    audit = _mapping_from_object(payload.get("audit_record", {}) or {})
    gate = payload.get("geostatistical_gate", {}) or {}

    grades = np.asarray(payload["grades"], dtype=float).ravel()
    coords = _extract_result_coords(payload)
    truth = truth_fn(coords)
    valid = np.isfinite(grades) & np.isfinite(truth)

    if np.any(valid):
        corr = float(np.corrcoef(grades[valid], truth[valid])[0, 1]) if np.sum(valid) >= 2 else float("nan")
        rmse = float(np.sqrt(np.mean((grades[valid] - truth[valid]) ** 2)))
        mean_ratio = float(np.mean(grades[valid]) / max(np.mean(truth[valid]), 1e-12))
    else:
        corr = float("nan")
        rmse = float("nan")
        mean_ratio = float("nan")

    failed_codes = [
        check.get("code")
        for check in gate.get("checks", [])
        if check.get("status") == "fail"
    ]

    metrics = (gate.get("metrics") or {})
    return {
        "case": case_name,
        "gate_status": gate.get("overall_status"),
        "gate_headline": gate.get("headline"),
        "gate_fail_count": gate.get("fail_count"),
        "failed_codes": failed_codes,
        "n_blocks_total": int(diagnostics.get("n_blocks_total", len(grades))),
        "n_blocks_estimated": int(diagnostics.get("n_blocks_estimated", int(np.isfinite(grades).sum()))),
        "estimated_fraction": float(np.isfinite(grades).sum() / max(len(grades), 1)),
        "cv_execution_mode": diagnostics.get("cv_execution_mode"),
        "cv_r_squared": _to_builtin(audit.get("cv_r_squared")),
        "cv_slope_of_regression": _to_builtin(audit.get("cv_slope_of_regression")),
        "cv_mean_error": _to_builtin(audit.get("cv_mean_error")),
        "conditional_bias_binned_slope": _to_builtin(audit.get("conditional_bias_binned_slope")),
        "support_swath_mean_rmse": _to_builtin(audit.get("support_swath_mean_rmse")),
        "support_swath_panels_with_data": _to_builtin(audit.get("support_swath_panels_with_data")),
        "support_swath_panels_total": _to_builtin(audit.get("support_swath_panels_total")),
        "fraction_beyond_one_range": _to_builtin(metrics.get("fraction_beyond_one_range")),
        "nearest_distance_p95_in_range_units": _to_builtin(metrics.get("nearest_distance_p95_in_range_units")),
        "truth_correlation": corr,
        "truth_rmse": rmse,
        "truth_mean_ratio": mean_ratio,
        "grade_mean_estimated": float(np.nanmean(grades)),
        "grade_std_estimated": float(np.nanstd(grades)),
    }


def _run_case(app: QApplication, package: dict, truth_fn, *, clip_to_footprint: bool) -> dict:
    registry = _fresh_registry()
    source_name = (
        "arbf-userlike-footprint-clip"
        if clip_to_footprint else
        "arbf-userlike-no-clip"
    )
    _register_package(registry, package, source_name)

    controller = AppController(renderer=MockRenderer(), registry=registry)
    panel = ARBFEstimationPanel()
    panel.bind_controller(controller)
    panel.show()
    app.processEvents()

    try:
        _configure_panel(panel, clip_to_footprint=clip_to_footprint)
        if not panel.validate_inputs():
            raise RuntimeError("ARBF panel validation failed before run.")

        t0 = time.time()
        panel.run_analysis()
        ok = _wait_until(app, lambda: getattr(panel, "arbf_results", None) is not None, timeout_s=1800.0)
        elapsed = time.time() - t0
        if not ok:
            raise TimeoutError("Timed out waiting for ARBF panel result.")

        payload = panel.arbf_results
        summary = _summarize_payload(
            "user_like_clip" if clip_to_footprint else "user_like_no_clip",
            payload,
            truth_fn,
        )
        summary["elapsed_seconds"] = float(elapsed)
        summary["summary_text"] = panel._summary_text.toPlainText()
        summary["defensibility_text"] = panel._gate_text.toPlainText()
        summary["audit_record"] = _to_builtin(_mapping_from_object(payload.get("audit_record") or {}))
        summary["diagnostics"] = _to_builtin(payload.get("diagnostics") or {})
        summary["geostatistical_gate"] = _to_builtin(payload.get("geostatistical_gate") or {})
        return summary
    finally:
        try:
            panel.close()
        except Exception:
            pass
        try:
            registry.clear_drillhole_data()
        except Exception:
            pass
        DataRegistry._instance = None


def main():
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("geostats.arbf.engine").setLevel(logging.ERROR)
    logging.getLogger("geostats.arbf.cross_validation").setLevel(logging.ERROR)
    logging.getLogger("block_model_viewer.controllers.app_controller").setLevel(logging.ERROR)
    _patch_message_boxes()
    app = QApplication.instance() or QApplication([])

    package, truth_fn, meta = _build_userlike_cu_package()

    print("=" * 78)
    print("ARBF PANEL USER-LIKE SOFTWARE-PATH RUN")
    print("=" * 78)
    print(f"Composites: {meta['n_composites']}")
    print(f"Holes:      {meta['n_holes']}")
    print(f"Cu mean:    {meta['sample_mean']:.2f}")
    print(f"Cu std:     {meta['sample_std']:.2f}")
    print(f"Cu min/max: {meta['sample_min']:.2f} / {meta['sample_max']:.2f}")
    print("Grid:       155 x 123 x 49 @ 10 m")
    print("Settings:   spherical, NS on, CoS on, LVA on, spatial k-fold, max_samples=300")
    print("Angles:     azimuth=236.3 dip=0.0 pitch=0.0")
    print("Ranges:     123.9 / 92.2 / 28.0")
    print("")

    results = {
        "package_meta": meta,
        "cases": [],
    }

    for clip_to_footprint in (False, True):
        label = "NO FOOTPRINT CLIP" if not clip_to_footprint else "FOOTPRINT CLIP ON"
        print("-" * 78)
        print(f"Running case: {label}")
        print("-" * 78)
        case_summary = _run_case(app, package, truth_fn, clip_to_footprint=clip_to_footprint)
        results["cases"].append(case_summary)
        print(f"Status:      {str(case_summary['gate_status']).upper()}")
        print(f"Fails:       {case_summary['gate_fail_count']} -> {case_summary['failed_codes']}")
        print(f"Blocks:      {case_summary['n_blocks_estimated']:,} / {case_summary['n_blocks_total']:,}")
        print(f"CV mode:     {case_summary['cv_execution_mode']}")
        print(f"CV R2:       {case_summary['cv_r_squared']}")
        print(f"CV slope:    {case_summary['cv_slope_of_regression']}")
        print(f"Cond bias:   {case_summary['conditional_bias_binned_slope']}")
        print(f"Truth corr:  {case_summary['truth_correlation']:.4f}")
        print(f"Truth RMSE:  {case_summary['truth_rmse']:.2f}")
        print(f"Mean ratio:  {case_summary['truth_mean_ratio']:.4f}")
        print(f"Elapsed:     {case_summary['elapsed_seconds']:.1f}s")
        print("")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(_to_builtin(results), indent=2), encoding="utf-8")
    print(f"Saved full report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
