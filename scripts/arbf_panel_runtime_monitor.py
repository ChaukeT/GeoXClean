import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from block_model_viewer.controllers.geostats_controller import GeostatsController
from scripts.arbf_method_audit import build_case_data, make_observations
from tests.arbf.test_arbf_production import make_combined_worst_case


OUTPUT_PATH = ROOT / "audit_logs" / "arbf_panel_runtime_monitor_20260319.json"


class _DummyApp:
    def __init__(self):
        self.block_model = None


class _DummyBlockModel:
    def __init__(self, positions: np.ndarray, dimensions: np.ndarray):
        self.positions = np.asarray(positions, dtype=np.float64)
        self.dimensions = np.asarray(dimensions, dtype=np.float64)


def _to_builtin(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    return value


def _parse_arbf_log(log_text: str) -> dict:
    neigh_all = [
        int(match)
        for match in re.findall(r"LGPR neighbourhood n=(\d+)", log_text)
    ]
    neigh_stabilized = [
        int(match)
        for match in re.findall(r"LGPR neighbourhood n=(\d+): stabilized with accuracy", log_text)
    ]
    kernel_cond_count = len(re.findall(r"Kernel cond .* > 1e10", log_text))
    retry_count = len(re.findall(r"Retrying with stronger diagonal regularization", log_text))
    stabilized_count = len(neigh_stabilized)
    fallback_count = len(re.findall(r"falling back to constant drift", log_text, flags=re.IGNORECASE))
    factorisation_fail_count = len(re.findall(r"factorisation failed", log_text, flags=re.IGNORECASE))

    summary = {
        "kernel_cond_warnings": kernel_cond_count,
        "regularization_retries": retry_count,
        "stabilized_neighbourhoods": stabilized_count,
        "local_drift_fallback_logs": fallback_count,
        "factorisation_fail_logs": factorisation_fail_count,
        "log_lines": len([line for line in log_text.splitlines() if line.strip()]),
    }
    if neigh_all:
        neigh_arr = np.asarray(neigh_all, dtype=np.int32)
        summary.update(
            {
                "neighbourhood_entries": int(len(neigh_all)),
                "neighbourhood_n_min": int(np.min(neigh_arr)),
                "neighbourhood_n_p50": float(np.median(neigh_arr)),
                "neighbourhood_n_p95": float(np.percentile(neigh_arr, 95)),
                "neighbourhood_n_max": int(np.max(neigh_arr)),
            }
        )
    if neigh_stabilized:
        stable_arr = np.asarray(neigh_stabilized, dtype=np.int32)
        summary.update(
            {
                "stabilized_n_p50": float(np.median(stable_arr)),
                "stabilized_n_p95": float(np.percentile(stable_arr, 95)),
                "stabilized_n_max": int(np.max(stable_arr)),
            }
        )
    return summary


def _panel_default_params(
    df: pd.DataFrame,
    variable: str,
    *,
    variogram: dict,
    panel_overrides: dict | None = None,
) -> dict:
    # These defaults mirror the current values in
    # block_model_viewer/ui/arbf_estimation_panel.py.
    params = {
        "data": df,
        "variable": variable,
        "domain_column": None,
        "domain_policy": "warn",
        "grid_spec": {
            "nx": 100,
            "ny": 100,
            "nz": 50,
            "dx": 10.0,
            "dy": 10.0,
            "dz": 10.0,
            "x0": 0.0,
            "y0": 0.0,
            "z0": 0.0,
        },
        "use_block_model_grid": True,
        "kernel_type": "spheroidal",
        "alpha": 1.0,
        "sill": float(variogram["sill"]),
        "nugget": float(variogram["nugget"]),
        "accuracy": 1e-6,
        "drift_type": "auto",
        "estimation_mode": "local_neighbourhood_gpr",
        "range_max": float(variogram["range_max"]),
        "range_mid": float(variogram["range_mid"]),
        "range_min": float(variogram["range_min"]),
        "azimuth": float(variogram["azimuth"]),
        "dip": float(variogram["dip"]),
        "pitch": float(variogram["pitch"]),
        "local_search_radii": (0.75, 1.50, 3.00),
        "balanced_neighbourhood_selection": True,
        "search_min_octants": 3,
        "search_min_octants_linear": 4,
        "max_samples_per_octant": 4,
        "auto_drift_max_slope_deviation": 0.20,
        "n_subdomains": 0,
        "pum_threshold": 3000,
        "subdomain_method": "kmeans",
        "overlap_factor": 2.0,
        "max_samples": 300,
        "min_samples": 4,
        "use_lva": True,
        "lva_source": "data",
        "use_normal_score": True,
        "use_ilr": False,
        "discretisation": "fixed",
        "discretisation_density": 27,
        "run_cv": True,
        "cv_mode": "spatial_kfold",
        "cv_folds": 5,
        "change_of_support": True,
        "change_of_support_mode": "discretized",
        "prefilter_blocks": False,
        "operator": "",
        "seed": 42,
        "clip_min": None,
        "clip_max": None,
        "variogram_mode": "hybrid",
        "rotation_convention": "geox",
        "use_geodesic": False,
        "decluster_cell_size": 0.0,
        "n_realizations": 20,
        "simulation_seed": 1042,
        "simulation_use_normal_score": True,
    }
    if panel_overrides:
        params.update(panel_overrides)
    return params


def _run_panel_worker(
    run_name: str,
    df: pd.DataFrame,
    centroids: np.ndarray,
    block_sizes: np.ndarray,
    *,
    variogram: dict,
    panel_overrides: dict | None = None,
) -> dict:
    params = _panel_default_params(
        df,
        "grade",
        variogram=variogram,
        panel_overrides=panel_overrides,
    )

    dummy_app = _DummyApp()
    controller = GeostatsController(dummy_app)
    controller._block_model = _DummyBlockModel(centroids, block_sizes)

    progress_events: list[dict] = []

    def progress_callback(percent: int, message: str):
        progress_events.append(
            {
                "t": round(time.perf_counter() - t0, 3),
                "percent": int(percent),
                "message": str(message),
            }
        )

    t0 = time.perf_counter()
    payload = controller._prepare_arbf_payload(params, progress_callback=progress_callback)
    elapsed = time.perf_counter() - t0

    log_path = Path(os.getenv("LOCALAPPDATA", ".")) / "GeoX" / "arbf_run.log"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""

    gate = payload.get("geostatistical_gate", {})
    failed_codes = [
        check["code"]
        for check in gate.get("checks", [])
        if check.get("status") == "fail"
    ]

    diagnostics = payload.get("diagnostics", {})
    result = {
        "run_name": run_name,
        "elapsed_seconds": float(elapsed),
        "progress_events": progress_events,
        "progress_event_count": int(len(progress_events)),
        "final_progress": progress_events[-1] if progress_events else None,
        "geostatistical_status": payload.get("metadata", {}).get("geostatistical_status"),
        "failed_gate_codes": failed_codes,
        "diagnostics": {
            "n_blocks_estimated": diagnostics.get("n_blocks_estimated"),
            "n_blocks_total": diagnostics.get("n_blocks_total"),
            "effective_drift_type": diagnostics.get("effective_drift_type"),
            "local_drift_fallbacks": diagnostics.get("local_drift_fallbacks"),
            "cv_r2": diagnostics.get("cv_r2"),
            "cv_rmse": diagnostics.get("cv_rmse"),
            "cv_slope": diagnostics.get("cv_slope"),
            "coverage": diagnostics.get("coverage"),
            "estimation_mode": diagnostics.get("estimation_mode"),
        },
        "log_summary": _parse_arbf_log(log_text),
    }
    return _to_builtin(result)


def build_anisotropic_control_case():
    case = {
        c["name"]: c
        for c in build_case_data()
    }["anisotropic_field"]
    obs_coords, obs_values, _ = make_observations(case, obs_seed=42)
    df = pd.DataFrame(obs_coords, columns=["X", "Y", "Z"])
    df["grade"] = obs_values
    variogram = {
        "sill": case["sill"],
        "nugget": case["nugget"],
        "range_max": case["ranges"][0],
        "range_mid": case["ranges"][1],
        "range_min": case["ranges"][2],
        "azimuth": case["angles"][0],
        "dip": case["angles"][1],
        "pitch": case["angles"][2],
    }
    return df, case["grid_coords"], case["spacing"], variogram


def build_combined_worst_case():
    coords, values, centroids, block_sizes, _ = make_combined_worst_case(n=250)
    df = pd.DataFrame(coords, columns=["X", "Y", "Z"])
    df["grade"] = values
    total_var = float(np.var(values))
    nugget = 0.10 * total_var
    variogram = {
        "sill": max(total_var - nugget, 1e-9),
        "nugget": nugget,
        "range_max": 80.0,
        "range_mid": 80.0,
        "range_min": 40.0,
        "azimuth": 0.0,
        "dip": 0.0,
        "pitch": 0.0,
    }
    return df, centroids, block_sizes, variogram


def main():
    runs = []

    df_a, cent_a, block_a, vario_a = build_anisotropic_control_case()
    runs.append(
        _run_panel_worker(
            "anisotropic_control_panel_defaults",
            df_a,
            cent_a,
            block_a,
            variogram=vario_a,
        )
    )

    df_w, cent_w, block_w, vario_w = build_combined_worst_case()
    runs.append(
        _run_panel_worker(
            "combined_worst_panel_defaults",
            df_w,
            cent_w,
            block_w,
            variogram=vario_w,
        )
    )
    runs.append(
        _run_panel_worker(
            "combined_worst_lva_off",
            df_w,
            cent_w,
            block_w,
            variogram=vario_w,
            panel_overrides={"use_lva": False},
        )
    )
    runs.append(
        _run_panel_worker(
            "combined_worst_fast_exec",
            df_w,
            cent_w,
            block_w,
            variogram=vario_w,
            panel_overrides={
                "use_lva": False,
                "run_cv": False,
                "change_of_support": False,
                "max_samples": 80,
            },
        )
    )

    OUTPUT_PATH.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")

    print(f"Wrote runtime monitor output to {OUTPUT_PATH}")
    print()
    print("ARBF panel runtime monitor")
    print("-" * 120)
    print(
        f"{'run':36} {'elapsed_s':>10} {'status':>8} {'cv_slope':>10} "
        f"{'retries':>10} {'stabilized':>11} {'n_p95':>8} {'fails'}"
    )
    for run in runs:
        diag = run["diagnostics"]
        log_summary = run["log_summary"]
        print(
            f"{run['run_name'][:36]:36} "
            f"{run['elapsed_seconds']:10.2f} "
            f"{str(run['geostatistical_status']):>8} "
            f"{str(diag.get('cv_slope'))[:10]:>10} "
            f"{int(log_summary.get('regularization_retries', 0)):10d} "
            f"{int(log_summary.get('stabilized_neighbourhoods', 0)):11d} "
            f"{str(log_summary.get('neighbourhood_n_p95', '-'))[:8]:>8} "
            f"{','.join(run['failed_gate_codes']) if run['failed_gate_codes'] else '-'}"
        )


if __name__ == "__main__":
    main()
