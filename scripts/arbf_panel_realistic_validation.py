import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "audit_logs" / "arbf_panel_realistic_validation_20260319.json"

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geostats.arbf.quality_gate import evaluate_geostatistical_gate
from scripts.arbf_method_audit import (
    build_case_data,
    compute_metrics,
    compute_panel_validation,
    make_grid,
    make_observations,
    run_estimator,
    sample_gp_field,
    select_uniform,
)


def _coerce_float_dict(mapping):
    out = {}
    for key, value in mapping.items():
        if isinstance(value, (np.floating, float)):
            out[key] = float(value)
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _serialize_gate(gate):
    payload = gate.to_dict()
    payload["failed_codes"] = [
        check["code"] for check in payload["checks"] if check["status"] == "fail"
    ]
    payload["warning_codes"] = [
        check["code"] for check in payload["checks"] if check["status"] == "warn"
    ]
    return payload


def run_panel_case(
    case,
    *,
    case_label=None,
    change_of_support=True,
    discretisation_density=27,
    available_domain_columns=None,
    domain_column=None,
    config_overrides=None,
    composite_domains=None,
    block_domains=None,
    obs_seed=42,
    estimator_seed=42,
):
    obs_coords, obs_values, obs_idx = make_observations(case, obs_seed)
    result, elapsed = run_estimator(
        case,
        obs_coords,
        obs_values,
        n_subdomains=0,
        seed=estimator_seed,
        change_of_support=change_of_support,
        discretisation_density=discretisation_density,
        max_samples=40,
        run_cv=True,
        config_overrides=config_overrides,
        composite_domains=composite_domains,
        block_domains=block_domains,
    )
    gate = evaluate_geostatistical_gate(
        result.audit_record,
        result.diagnostics,
        sample_values=obs_values,
        sample_coords=obs_coords,
        block_centroids=case["grid_coords"],
        domain_column=domain_column,
        available_domain_columns=available_domain_columns or [],
        change_of_support=change_of_support,
        estimation_mode=result.diagnostics.get("estimation_mode"),
        simulation_run=False,
    )
    return {
        "case": case_label or case["name"],
        "n_obs": int(len(obs_values)),
        "elapsed_seconds": float(elapsed),
        "metrics": compute_metrics(case["truth"], result.grades, obs_coords, case["grid_coords"]),
        "panel_metrics": compute_panel_validation(case, result.grades, obs_coords),
        "cv": {
            "r2": float(result.cv_result.r_squared) if result.cv_result is not None else None,
            "rmse": float(result.cv_result.rmse) if result.cv_result is not None else None,
            "slope": (
                float(result.cv_result.slope_of_regression)
                if result.cv_result is not None else None
            ),
        },
        "diagnostics": _coerce_float_dict(result.diagnostics),
        "gate": _serialize_gate(gate),
        "observation_indices": obs_idx.tolist(),
    }


def build_domained_realistic_case():
    domain = (60.0, 40.0, 24.0)
    grid_shape = (13, 9, 5)
    grid_coords, spacing = make_grid(shape=grid_shape, domain=domain)
    domain_labels = np.where(grid_coords[:, 0] + 0.20 * grid_coords[:, 1] < 30.0, 1, 2)
    trend = (
        0.15 * (grid_coords[:, 0] / domain[0])
        - 0.10 * (grid_coords[:, 1] / domain[1])
        + 0.12 * (grid_coords[:, 2] / domain[2])
    )
    residual = sample_gp_field(
        grid_coords,
        mean=0.0,
        sill=0.10,
        alpha=1.5,
        ranges=(18.0, 10.0, 6.0),
        angles=(28.0, 8.0, 0.0),
        seed=1111,
    )
    truth = np.where(domain_labels == 1, 0.9, 2.1) + trend + 0.25 * residual
    case = {
        "name": "domained_realistic_deposit",
        "truth": truth,
        "obs_selector": lambda coords, rng: select_uniform(len(coords), 120, rng),
        "noise_sd": 0.0,
        "ranges": (18.0, 10.0, 6.0),
        "angles": (28.0, 8.0, 0.0),
        "alpha": 1.5,
        "sill": 0.10,
        "nugget": 0.02,
        "drift_type": "auto",
        "domain": domain,
        "grid_coords": grid_coords,
        "grid_shape": grid_shape,
        "panel_factors": (1, 3, 1),
        "spacing": spacing,
    }
    return case, domain_labels


def build_realistic_suite():
    cases = {case["name"]: case for case in build_case_data()}
    out = []

    out.append(
        run_panel_case(
            cases["trend_plus_residual"],
            case_label="trend_residual_block_support",
            change_of_support=True,
            discretisation_density=27,
        )
    )
    out.append(
        run_panel_case(
            cases["anisotropic_field"],
            case_label="anisotropic_good_coverage",
            change_of_support=True,
            discretisation_density=27,
        )
    )
    out.append(
        run_panel_case(
            cases["clustered_sampling"],
            case_label="clustered_drilling_realistic",
            change_of_support=True,
            discretisation_density=27,
        )
    )
    out.append(
        run_panel_case(
            cases["sparse_sampling"],
            case_label="sparse_drilling_realistic",
            change_of_support=True,
            discretisation_density=27,
        )
    )
    out.append(
        run_panel_case(
            cases["boundary_extrapolation"],
            case_label="pit_shell_extrapolation",
            change_of_support=True,
            discretisation_density=27,
        )
    )

    domained_case, domain_labels = build_domained_realistic_case()
    obs_coords, obs_values, obs_idx = make_observations(domained_case, 42)
    obs_domains = domain_labels[obs_idx]

    no_domain_result, no_domain_elapsed = run_estimator(
        domained_case,
        obs_coords,
        obs_values,
        n_subdomains=0,
        seed=42,
        change_of_support=True,
        discretisation_density=27,
        max_samples=40,
        run_cv=True,
        config_overrides={"domain_policy": "warn"},
    )
    no_domain_gate = evaluate_geostatistical_gate(
        no_domain_result.audit_record,
        no_domain_result.diagnostics,
        sample_values=obs_values,
        sample_coords=obs_coords,
        block_centroids=domained_case["grid_coords"],
        domain_column=None,
        available_domain_columns=["domain_code"],
        change_of_support=True,
        estimation_mode=no_domain_result.diagnostics.get("estimation_mode"),
    )
    out.append(
        {
            "case": "domained_deposit_without_domain_selection",
            "n_obs": int(len(obs_values)),
            "elapsed_seconds": float(no_domain_elapsed),
            "metrics": compute_metrics(
                domained_case["truth"],
                no_domain_result.grades,
                obs_coords,
                domained_case["grid_coords"],
            ),
            "panel_metrics": compute_panel_validation(
                domained_case, no_domain_result.grades, obs_coords,
            ),
            "cv": {
                "r2": (
                    float(no_domain_result.cv_result.r_squared)
                    if no_domain_result.cv_result is not None else None
                ),
                "rmse": (
                    float(no_domain_result.cv_result.rmse)
                    if no_domain_result.cv_result is not None else None
                ),
                "slope": (
                    float(no_domain_result.cv_result.slope_of_regression)
                    if no_domain_result.cv_result is not None else None
                ),
            },
            "diagnostics": _coerce_float_dict(no_domain_result.diagnostics),
            "gate": _serialize_gate(no_domain_gate),
            "observation_indices": obs_idx.tolist(),
        }
    )

    hard_domain_result, hard_domain_elapsed = run_estimator(
        domained_case,
        obs_coords,
        obs_values,
        n_subdomains=0,
        seed=42,
        change_of_support=True,
        discretisation_density=27,
        max_samples=40,
        run_cv=True,
        config_overrides={"domain_policy": "require"},
        composite_domains=obs_domains,
        block_domains=domain_labels,
    )
    hard_domain_gate = evaluate_geostatistical_gate(
        hard_domain_result.audit_record,
        hard_domain_result.diagnostics,
        sample_values=obs_values,
        sample_coords=obs_coords,
        block_centroids=domained_case["grid_coords"],
        domain_column="domain_code",
        available_domain_columns=["domain_code"],
        change_of_support=True,
        estimation_mode=hard_domain_result.diagnostics.get("estimation_mode"),
    )
    out.append(
        {
            "case": "domained_deposit_with_hard_domains",
            "n_obs": int(len(obs_values)),
            "elapsed_seconds": float(hard_domain_elapsed),
            "metrics": compute_metrics(
                domained_case["truth"],
                hard_domain_result.grades,
                obs_coords,
                domained_case["grid_coords"],
            ),
            "panel_metrics": compute_panel_validation(
                domained_case, hard_domain_result.grades, obs_coords,
            ),
            "cv": {
                "r2": (
                    float(hard_domain_result.cv_result.r_squared)
                    if hard_domain_result.cv_result is not None else None
                ),
                "rmse": (
                    float(hard_domain_result.cv_result.rmse)
                    if hard_domain_result.cv_result is not None else None
                ),
                "slope": (
                    float(hard_domain_result.cv_result.slope_of_regression)
                    if hard_domain_result.cv_result is not None else None
                ),
            },
            "diagnostics": _coerce_float_dict(hard_domain_result.diagnostics),
            "gate": _serialize_gate(hard_domain_gate),
            "observation_indices": obs_idx.tolist(),
        }
    )
    return out


def summarize_cases(rows):
    print("ARBF panel realistic synthetic validation")
    print("-" * 132)
    print(
        f"{'case':38s} {'gate':7s} {'rmse':>8s} {'r2':>8s} {'panel_r2':>9s} "
        f"{'cv_slope':>9s} {'sw_rmse':>9s} {'fails':>32s}"
    )
    for row in rows:
        metrics = row["metrics"]
        panel_metrics = row["panel_metrics"] or {}
        diagnostics = row["diagnostics"]
        gate = row["gate"]
        fails = ",".join(gate["failed_codes"]) or "-"
        print(
            f"{row['case'][:38]:38s} "
            f"{gate['overall_status'][:7]:7s} "
            f"{metrics['rmse']:8.4f} "
            f"{metrics['r2']:8.4f} "
            f"{float(panel_metrics.get('r2', np.nan)):9.4f} "
            f"{float((row.get('cv') or {}).get('slope', np.nan)):9.4f} "
            f"{float(diagnostics.get('support_swath_mean_rmse', np.nan)):9.4f} "
            f"{fails:>32s}"
        )


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = build_realistic_suite()
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": float(time.time() - started),
        "cases": rows,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote realistic synthetic panel validation to {OUTPUT_PATH}")
    print()
    summarize_cases(rows)


if __name__ == "__main__":
    main()
