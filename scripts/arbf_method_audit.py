import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geostats.arbf.engine import ARBFEstimator
from geostats.arbf.kernels import evaluate_kernel
from geostats.arbf.utils import pairwise_anisotropic_distance, rotation_matrix, scale_matrix


OUTPUT_PATH = ROOT / "audit_logs" / "arbf_method_audit_20260319.json"


def make_grid(shape=(9, 9, 5), domain=(40.0, 40.0, 20.0)):
    axes = [np.linspace(0.0, float(domain[i]), int(shape[i])) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    spacing = np.array(
        [
            float(domain[0]) / max(shape[0] - 1, 1),
            float(domain[1]) / max(shape[1] - 1, 1),
            float(domain[2]) / max(shape[2] - 1, 1),
        ],
        dtype=np.float64,
    )
    return coords.astype(np.float64), spacing


def sample_gp_field(coords, *, mean, sill, alpha, ranges, angles, seed):
    rng = np.random.default_rng(seed)
    R = rotation_matrix(*angles)
    S = scale_matrix(*ranges)
    D = pairwise_anisotropic_distance(coords, R, S)
    K = sill * evaluate_kernel(D, kernel_type="spheroidal", alpha=alpha)
    K = 0.5 * (K + K.T)
    K += 1e-8 * np.eye(K.shape[0], dtype=np.float64)
    L = np.linalg.cholesky(K)
    return mean + L @ rng.standard_normal(len(coords))


def select_uniform(n_total, n_obs, rng):
    return rng.choice(n_total, size=n_obs, replace=False)


def select_clustered(coords, n_obs, rng):
    centers = np.array([[10.0, 10.0, 5.0], [14.0, 12.0, 8.0]], dtype=np.float64)
    d1 = np.linalg.norm(coords - centers[0], axis=1)
    d2 = np.linalg.norm(coords - centers[1], axis=1)
    weights = np.exp(-(d1 / 6.0) ** 2) + 0.7 * np.exp(-(d2 / 7.0) ** 2) + 0.02
    weights /= weights.sum()
    return rng.choice(len(coords), size=n_obs, replace=False, p=weights)


def select_boundary(coords, n_obs, rng):
    mask = (
        (coords[:, 0] >= 8.0)
        & (coords[:, 0] <= 26.0)
        & (coords[:, 1] >= 8.0)
        & (coords[:, 1] <= 26.0)
    )
    candidate = np.where(mask)[0]
    return rng.choice(candidate, size=min(n_obs, len(candidate)), replace=False)


def trend_component(coords, domain):
    x = coords[:, 0] / float(domain[0])
    y = coords[:, 1] / float(domain[1])
    z = coords[:, 2] / float(domain[2])
    return 0.9 * x - 0.6 * y + 0.45 * z + 0.35 * x * y


def build_case_data():
    domain = (40.0, 40.0, 20.0)
    grid_shape = (9, 9, 5)
    grid_coords, spacing = make_grid(shape=grid_shape, domain=domain)

    cases = []

    cases.append(
        {
            "name": "smooth_stationary",
            "truth": sample_gp_field(
                grid_coords,
                mean=1.25,
                sill=1.0,
                alpha=1.5,
                ranges=(18.0, 18.0, 18.0),
                angles=(0.0, 0.0, 0.0),
                seed=101,
            ),
            "obs_selector": lambda coords, rng: select_uniform(len(coords), 90, rng),
            "noise_sd": 0.0,
            "ranges": (18.0, 18.0, 18.0),
            "angles": (0.0, 0.0, 0.0),
            "alpha": 1.5,
            "sill": 1.0,
            "nugget": 0.0,
            "drift_type": "auto",
            "domain": domain,
            "grid_coords": grid_coords,
            "grid_shape": grid_shape,
            "panel_factors": (3, 3, 1),
            "spacing": spacing,
        }
    )

    cases.append(
        {
            "name": "anisotropic_field",
            "truth": sample_gp_field(
                grid_coords,
                mean=1.0,
                sill=1.0,
                alpha=1.5,
                ranges=(24.0, 10.0, 5.0),
                angles=(35.0, 10.0, 0.0),
                seed=202,
            ),
            "obs_selector": lambda coords, rng: select_uniform(len(coords), 90, rng),
            "noise_sd": 0.0,
            "ranges": (24.0, 10.0, 5.0),
            "angles": (35.0, 10.0, 0.0),
            "alpha": 1.5,
            "sill": 1.0,
            "nugget": 0.0,
            "drift_type": "auto",
            "domain": domain,
            "grid_coords": grid_coords,
            "grid_shape": grid_shape,
            "panel_factors": (3, 3, 1),
            "spacing": spacing,
        }
    )

    residual = sample_gp_field(
        grid_coords,
        mean=0.0,
        sill=0.35,
        alpha=1.5,
        ranges=(14.0, 14.0, 10.0),
        angles=(0.0, 0.0, 0.0),
        seed=303,
    )
    cases.append(
        {
            "name": "trend_plus_residual",
            "truth": 1.1 + trend_component(grid_coords, domain) + residual,
            "obs_selector": lambda coords, rng: select_uniform(len(coords), 90, rng),
            "noise_sd": 0.0,
            "ranges": (14.0, 14.0, 10.0),
            "angles": (0.0, 0.0, 0.0),
            "alpha": 1.5,
            "sill": 0.35,
            "nugget": 0.0,
            "drift_type": "auto",
            "domain": domain,
            "grid_coords": grid_coords,
            "grid_shape": grid_shape,
            "panel_factors": (3, 3, 1),
            "spacing": spacing,
        }
    )

    cases.append(
        {
            "name": "clustered_sampling",
            "truth": sample_gp_field(
                grid_coords,
                mean=1.2,
                sill=1.0,
                alpha=1.5,
                ranges=(18.0, 18.0, 18.0),
                angles=(0.0, 0.0, 0.0),
                seed=404,
            ),
            "obs_selector": lambda coords, rng: select_clustered(coords, 90, rng),
            "noise_sd": 0.0,
            "ranges": (18.0, 18.0, 18.0),
            "angles": (0.0, 0.0, 0.0),
            "alpha": 1.5,
            "sill": 1.0,
            "nugget": 0.0,
            "drift_type": "auto",
            "domain": domain,
            "grid_coords": grid_coords,
            "grid_shape": grid_shape,
            "panel_factors": (3, 3, 1),
            "spacing": spacing,
        }
    )

    cases.append(
        {
            "name": "sparse_sampling",
            "truth": sample_gp_field(
                grid_coords,
                mean=1.0,
                sill=1.0,
                alpha=1.5,
                ranges=(18.0, 18.0, 18.0),
                angles=(0.0, 0.0, 0.0),
                seed=505,
            ),
            "obs_selector": lambda coords, rng: select_uniform(len(coords), 24, rng),
            "noise_sd": 0.0,
            "ranges": (18.0, 18.0, 18.0),
            "angles": (0.0, 0.0, 0.0),
            "alpha": 1.5,
            "sill": 1.0,
            "nugget": 0.0,
            "drift_type": "auto",
            "domain": domain,
            "grid_coords": grid_coords,
            "grid_shape": grid_shape,
            "panel_factors": (3, 3, 1),
            "spacing": spacing,
        }
    )

    cases.append(
        {
            "name": "noisy_observations",
            "truth": sample_gp_field(
                grid_coords,
                mean=1.1,
                sill=1.0,
                alpha=1.5,
                ranges=(18.0, 18.0, 18.0),
                angles=(0.0, 0.0, 0.0),
                seed=606,
            ),
            "obs_selector": lambda coords, rng: select_uniform(len(coords), 90, rng),
            "noise_sd": 0.45,
            "ranges": (18.0, 18.0, 18.0),
            "angles": (0.0, 0.0, 0.0),
            "alpha": 1.5,
            "sill": 1.0,
            "nugget": 0.45 ** 2,
            "drift_type": "auto",
            "domain": domain,
            "grid_coords": grid_coords,
            "grid_shape": grid_shape,
            "panel_factors": (3, 3, 1),
            "spacing": spacing,
        }
    )

    cases.append(
        {
            "name": "boundary_extrapolation",
            "truth": sample_gp_field(
                grid_coords,
                mean=1.0,
                sill=1.0,
                alpha=1.5,
                ranges=(18.0, 18.0, 18.0),
                angles=(0.0, 0.0, 0.0),
                seed=707,
            ),
            "obs_selector": lambda coords, rng: select_boundary(coords, 75, rng),
            "noise_sd": 0.0,
            "ranges": (18.0, 18.0, 18.0),
            "angles": (0.0, 0.0, 0.0),
            "alpha": 1.5,
            "sill": 1.0,
            "nugget": 0.0,
            "drift_type": "auto",
            "domain": domain,
            "grid_coords": grid_coords,
            "grid_shape": grid_shape,
            "panel_factors": (3, 3, 1),
            "spacing": spacing,
        }
    )

    return cases


def make_observations(case, obs_seed):
    rng = np.random.default_rng(obs_seed)
    idx = case["obs_selector"](case["grid_coords"], rng)
    obs_coords = case["grid_coords"][idx]
    obs_truth = case["truth"][idx]
    if case["noise_sd"] > 0.0:
        obs_values = obs_truth + rng.normal(0.0, case["noise_sd"], size=len(idx))
    else:
        obs_values = obs_truth.copy()
    return obs_coords, obs_values, idx


def run_estimator(
    case,
    obs_coords,
    obs_values,
    *,
    n_subdomains,
    seed,
    change_of_support=False,
    discretisation_density=1,
    max_samples=40,
    run_cv=True,
    pum_threshold=999999,
    change_of_support_mode="discretized",
    estimation_mode="local_neighbourhood_gpr",
    config_overrides=None,
    composite_domains=None,
    block_domains=None,
):
    config = {
        "kernel_type": "spheroidal",
        "alpha": case["alpha"],
        "sill": case["sill"],
        "nugget": case["nugget"],
        "range_max": case["ranges"][0],
        "range_mid": case["ranges"][1],
        "range_min": case["ranges"][2],
        "azimuth": case["angles"][0],
        "dip": case["angles"][1],
        "pitch": case["angles"][2],
        "drift_type": case.get("drift_type", "auto"),
        "n_subdomains": n_subdomains,
        "pum_threshold": pum_threshold,
        "max_samples": max_samples,
        "min_samples": 4,
        "overlap_factor": 1.5,
        "estimation_mode": estimation_mode,
        "variogram_mode": "global",
        "change_of_support": change_of_support,
        "change_of_support_mode": change_of_support_mode,
        "discretisation": "fixed",
        "discretisation_density": discretisation_density,
        "run_cv": run_cv,
        "cv_max_samples": min(120, len(obs_values)),
        "parallel": False,
        "verbose": False,
        "seed": seed,
    }
    if config_overrides:
        config.update(config_overrides)

    estimator = ARBFEstimator(config)
    estimator.set_composites(obs_coords, obs_values)
    estimator.set_block_model(case["grid_coords"], case["spacing"])
    if composite_domains is not None:
        estimator.set_domains(composite_domains, block_domains)

    start = time.time()
    result = estimator.estimate()
    elapsed = time.time() - start
    return result, elapsed


def compute_metrics(truth, pred, obs_coords, eval_coords):
    mask = np.isfinite(pred)
    finite_fraction = float(np.mean(mask))
    if not np.any(mask):
        return {
            "finite_fraction": finite_fraction,
            "mae": math.nan,
            "rmse": math.nan,
            "bias": math.nan,
            "r2": math.nan,
            "slope": math.nan,
            "std_ratio": math.nan,
            "mean_diff": math.nan,
            "distance_error_corr": math.nan,
            "rmse_near": math.nan,
            "rmse_far": math.nan,
        }

    resid = pred[mask] - truth[mask]
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    bias = float(np.mean(resid))
    sst = float(np.sum((truth[mask] - np.mean(truth[mask])) ** 2))
    r2 = float(1.0 - np.sum(resid ** 2) / max(sst, 1e-12))
    est_var = float(np.var(pred[mask]))
    if np.sum(mask) > 1 and est_var > 1e-12:
        cov = float(np.cov(pred[mask], truth[mask], ddof=0)[0, 1])
        slope = cov / est_var
    else:
        slope = math.nan
    truth_std = float(np.std(truth[mask]))
    pred_std = float(np.std(pred[mask]))
    std_ratio = pred_std / truth_std if truth_std > 1e-12 else math.nan
    mean_diff = float(np.mean(pred[mask]) - np.mean(truth[mask]))

    nn_dist = cKDTree(obs_coords).query(eval_coords)[0]
    abs_err = np.abs(pred - truth)
    if np.sum(mask) > 2 and np.std(abs_err[mask]) > 1e-12 and np.std(nn_dist[mask]) > 1e-12:
        distance_error_corr = float(np.corrcoef(abs_err[mask], nn_dist[mask])[0, 1])
    else:
        distance_error_corr = math.nan

    q = np.nanmedian(nn_dist[mask])
    near = mask & (nn_dist <= q)
    far = mask & (nn_dist > q)
    rmse_near = float(np.sqrt(np.mean((pred[near] - truth[near]) ** 2))) if np.any(near) else math.nan
    rmse_far = float(np.sqrt(np.mean((pred[far] - truth[far]) ** 2))) if np.any(far) else math.nan

    return {
        "finite_fraction": finite_fraction,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "r2": r2,
        "slope": slope,
        "std_ratio": std_ratio,
        "mean_diff": mean_diff,
        "distance_error_corr": distance_error_corr,
        "rmse_near": rmse_near,
        "rmse_far": rmse_far,
    }


def aggregate_support(values, coords, grid_shape, panel_factors):
    nx, ny, nz = [int(x) for x in grid_shape]
    fx, fy, fz = [int(x) for x in panel_factors]
    if nx % fx != 0 or ny % fy != 0 or nz % fz != 0:
        raise ValueError("panel_factors must divide the grid shape exactly")

    value_grid = np.asarray(values, dtype=np.float64).reshape(nx, ny, nz)
    coord_grid = np.asarray(coords, dtype=np.float64).reshape(nx, ny, nz, 3)

    agg_shape = (nx // fx, ny // fy, nz // fz)
    value_panels = value_grid.reshape(
        agg_shape[0], fx, agg_shape[1], fy, agg_shape[2], fz,
    ).mean(axis=(1, 3, 5))
    coord_panels = coord_grid.reshape(
        agg_shape[0], fx, agg_shape[1], fy, agg_shape[2], fz, 3,
    ).mean(axis=(1, 3, 5))
    return value_panels.reshape(-1), coord_panels.reshape(-1, 3), agg_shape


def compute_panel_validation(case, pred, obs_coords):
    if "grid_shape" not in case or "panel_factors" not in case:
        return None

    truth_panel, panel_coords, panel_shape = aggregate_support(
        case["truth"],
        case["grid_coords"],
        case["grid_shape"],
        case["panel_factors"],
    )
    pred_panel, _, _ = aggregate_support(
        pred,
        case["grid_coords"],
        case["grid_shape"],
        case["panel_factors"],
    )
    metrics = compute_metrics(truth_panel, pred_panel, obs_coords, panel_coords)
    metrics["panel_shape"] = list(panel_shape)
    metrics["panel_factors"] = list(case["panel_factors"])
    return metrics


def serialize_cv(cv_result):
    if cv_result is None:
        return None
    return {
        "n_samples": int(cv_result.n_samples),
        "rmse": float(cv_result.rmse),
        "mae": float(cv_result.mae),
        "r2": float(cv_result.r_squared),
        "bias": float(cv_result.mean_error),
        "slope": float(cv_result.slope_of_regression),
        "correlation": float(cv_result.correlation),
    }


def run_case(
    case,
    *,
    n_subdomains,
    obs_seed=42,
    estimator_seed=42,
    change_of_support=False,
    discretisation_density=1,
    max_samples=40,
    run_cv=True,
    pum_threshold=999999,
    estimation_mode="local_neighbourhood_gpr",
):
    obs_coords, obs_values, obs_idx = make_observations(case, obs_seed)
    result, elapsed = run_estimator(
        case,
        obs_coords,
        obs_values,
        n_subdomains=n_subdomains,
        seed=estimator_seed,
        change_of_support=change_of_support,
        discretisation_density=discretisation_density,
        max_samples=max_samples,
        run_cv=run_cv,
        pum_threshold=pum_threshold,
        estimation_mode=estimation_mode,
    )
    metrics = compute_metrics(case["truth"], result.grades, obs_coords, case["grid_coords"])
    panel_metrics = compute_panel_validation(case, result.grades, obs_coords)
    return {
        "case": case["name"],
        "n_obs": int(len(obs_values)),
        "obs_seed": int(obs_seed),
        "estimator_seed": int(estimator_seed),
        "n_subdomains": int(n_subdomains),
        "estimation_mode": estimation_mode,
        "metrics": metrics,
        "panel_metrics": panel_metrics,
        "cv": serialize_cv(result.cv_result),
        "elapsed_seconds": float(elapsed),
        "grade_mean": float(np.nanmean(result.grades)),
        "grade_std": float(np.nanstd(result.grades)),
        "variance_mean": float(np.nanmean(result.variances)),
        "variance_max": float(np.nanmax(result.variances)),
        "n_nan": int(np.sum(~np.isfinite(result.grades))),
        "diagnostics": {
            k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
            for k, v in result.diagnostics.items()
        },
    }


def run_support_experiment():
    domain = (40.0, 40.0, 20.0)
    fine_coords, spacing = make_grid(shape=(10, 10, 6), domain=domain)
    truth = sample_gp_field(
        fine_coords,
        mean=1.0,
        sill=1.0,
        alpha=1.5,
        ranges=(18.0, 18.0, 18.0),
        angles=(0.0, 0.0, 0.0),
        seed=808,
    )
    rng = np.random.default_rng(42)
    obs_idx = rng.choice(len(fine_coords), size=100, replace=False)
    obs_coords = fine_coords[obs_idx]
    obs_values = truth[obs_idx]

    coarse_shape = (5, 5, 3)
    coarse_coords, coarse_spacing = make_grid(shape=coarse_shape, domain=domain)
    nx, ny, nz = 10, 10, 6
    field3d = truth.reshape(nx, ny, nz)
    coarse_truth = np.zeros(len(coarse_coords), dtype=np.float64)
    block_id = 0
    for ix in range(0, nx, 2):
        for iy in range(0, ny, 2):
            for iz in range(0, nz, 2):
                coarse_truth[block_id] = float(np.mean(field3d[ix:ix + 2, iy:iy + 2, iz:iz + 2]))
                block_id += 1

    base_case = {
        "name": "support_effects",
        "truth": coarse_truth,
        "grid_coords": coarse_coords,
        "spacing": coarse_spacing * 2.0,
        "ranges": (18.0, 18.0, 18.0),
        "angles": (0.0, 0.0, 0.0),
        "alpha": 1.5,
        "sill": 1.0,
        "nugget": 0.0,
        "drift_type": "auto",
    }

    out = {}
    support_runs = [
        ("point_support_centroid", False, 1, "discretized"),
        ("discretized_block_support_auto", True, 1, "discretized"),
        ("discretized_block_support_dense", True, 27, "discretized"),
        ("affine_legacy_centroid", True, 1, "affine_legacy"),
    ]
    for label, cos_flag, disc_density, cos_mode in support_runs:
        config_case = dict(base_case)
        result, elapsed = run_estimator(
            config_case,
            obs_coords,
            obs_values,
            n_subdomains=1,
            seed=42,
            change_of_support=cos_flag,
            discretisation_density=disc_density,
            max_samples=999,
            change_of_support_mode=cos_mode,
        )
        out[label] = {
            "metrics": compute_metrics(coarse_truth, result.grades, obs_coords, coarse_coords),
            "elapsed_seconds": float(elapsed),
            "grade_std": float(np.nanstd(result.grades)),
            "truth_std": float(np.std(coarse_truth)),
            "variance_mean": float(np.nanmean(result.variances)),
            "stitching_variance_mean": float(
                np.nanmean(result.stitching_variance)
            ) if result.stitching_variance is not None else 0.0,
            "cos_method": (
                result.cos_result.method
                if result.cos_result is not None and hasattr(result.cos_result, "method")
                else "none"
            ),
            "support_ratio": (
                float(result.cos_result.support_ratio)
                if result.cos_result is not None
                else 0.0
            ),
        }
    return out


def run_reproducibility_check(case):
    obs_coords, obs_values, _ = make_observations(case, 42)
    runs = []
    predictions = []
    for global_seed in [0, 1, 2]:
        np.random.seed(global_seed)
        result, elapsed = run_estimator(
            case,
            obs_coords,
            obs_values,
            n_subdomains=0,
            seed=42,
            change_of_support=False,
            discretisation_density=1,
            max_samples=40,
        )
        runs.append(
            {
                "global_seed": global_seed,
                "elapsed_seconds": float(elapsed),
                "r2": float(compute_metrics(case["truth"], result.grades, obs_coords, case["grid_coords"])["r2"]),
                "grade_mean": float(np.nanmean(result.grades)),
            }
        )
        predictions.append(result.grades.copy())

    pairwise = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            diff = np.abs(predictions[i] - predictions[j])
            pairwise.append(
                {
                    "pair": [i, j],
                    "max_abs_diff": float(np.nanmax(diff)),
                    "mean_abs_diff": float(np.nanmean(diff)),
                }
            )
    return {"runs": runs, "pairwise_prediction_differences": pairwise}


def run_domain_bug_check():
    comp_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [11.0, 10.0, 0.0],
        ],
        dtype=np.float64,
    )
    comp_values = np.array([1.0, 1.1, 2.0, 2.1], dtype=np.float64)
    block_centroids = np.array([[0.5, 0.0, 0.0], [50.0, 50.0, 0.0]], dtype=np.float64)
    block_sizes = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    estimator = ARBFEstimator(
        {
            "kernel_type": "spheroidal",
            "alpha": 1.5,
            "sill": 1.0,
            "nugget": 0.0,
            "range_max": 8.0,
            "range_mid": 8.0,
            "range_min": 8.0,
            "n_subdomains": 1,
            "run_cv": False,
            "change_of_support": False,
            "verbose": False,
        }
    )
    estimator.set_composites(comp_coords, comp_values)
    estimator.set_block_model(block_centroids, block_sizes)
    estimator.set_domains(
        composite_domains=np.array([1, 1, 2, 2]),
        block_domains=np.array([1, 999]),
    )
    result = estimator.estimate()
    return {
        "classification_codes": result.classifications.tolist(),
        "classification_names": result.classification_names.tolist(),
        "uncovered_block_grade": float(result.grades[1]),
        "uncovered_block_variance": float(result.variances[1]),
    }


def run_domaining_experiment():
    domain = (40.0, 40.0, 20.0)
    grid_shape = (9, 9, 5)
    grid_coords, spacing = make_grid(shape=grid_shape, domain=domain)
    domain_labels = np.where(grid_coords[:, 0] < 20.0, 1, 2)
    residual = sample_gp_field(
        grid_coords,
        mean=0.0,
        sill=0.12,
        alpha=1.5,
        ranges=(10.0, 10.0, 8.0),
        angles=(0.0, 0.0, 0.0),
        seed=909,
    )
    truth = np.where(domain_labels == 1, 0.8, 2.6) + 0.15 * residual
    case = {
        "name": "domaining_contact",
        "truth": truth,
        "obs_selector": lambda coords, rng: select_uniform(len(coords), 100, rng),
        "noise_sd": 0.0,
        "ranges": (12.0, 12.0, 8.0),
        "angles": (0.0, 0.0, 0.0),
        "alpha": 1.5,
        "sill": 0.12,
        "nugget": 0.0,
        "drift_type": "auto",
        "domain": domain,
        "grid_coords": grid_coords,
        "grid_shape": grid_shape,
        "panel_factors": (3, 3, 1),
        "spacing": spacing,
    }
    obs_coords, obs_values, obs_idx = make_observations(case, 42)
    obs_domains = domain_labels[obs_idx]
    contact_mask = np.abs(grid_coords[:, 0] - 20.0) <= spacing[0] * 1.5

    no_domain_result, _ = run_estimator(
        case,
        obs_coords,
        obs_values,
        n_subdomains=0,
        seed=42,
        change_of_support=False,
        discretisation_density=1,
        max_samples=40,
        run_cv=True,
        config_overrides={"domain_policy": "warn"},
    )
    hard_domain_result, _ = run_estimator(
        case,
        obs_coords,
        obs_values,
        n_subdomains=0,
        seed=42,
        change_of_support=False,
        discretisation_density=1,
        max_samples=40,
        run_cv=True,
        config_overrides={"domain_policy": "require"},
        composite_domains=obs_domains,
        block_domains=domain_labels,
    )

    return {
        "no_domains": {
            "metrics": compute_metrics(truth, no_domain_result.grades, obs_coords, grid_coords),
            "panel_metrics": compute_panel_validation(case, no_domain_result.grades, obs_coords),
            "contact_metrics": compute_metrics(
                truth[contact_mask],
                no_domain_result.grades[contact_mask],
                obs_coords,
                grid_coords[contact_mask],
            ),
            "diagnostics": no_domain_result.diagnostics,
        },
        "hard_domains": {
            "metrics": compute_metrics(truth, hard_domain_result.grades, obs_coords, grid_coords),
            "panel_metrics": compute_panel_validation(case, hard_domain_result.grades, obs_coords),
            "contact_metrics": compute_metrics(
                truth[contact_mask],
                hard_domain_result.grades[contact_mask],
                obs_coords,
                grid_coords[contact_mask],
            ),
            "diagnostics": hard_domain_result.diagnostics,
        },
    }


def run_pum_threshold_check(case):
    obs_coords, obs_values, _ = make_observations(case, 42)
    preds = []
    for pum_threshold in [1, 999999]:
        np.random.seed(123)
        result, _ = run_estimator(
            case,
            obs_coords,
            obs_values,
            n_subdomains=0,
            seed=42,
            change_of_support=False,
            discretisation_density=1,
            max_samples=40,
            run_cv=False,
            pum_threshold=pum_threshold,
            estimation_mode="pum_legacy",
        )
        preds.append(result.grades.copy())
    diff = np.abs(preds[0] - preds[1])
    return {
        "max_abs_diff": float(np.nanmax(diff)),
        "mean_abs_diff": float(np.nanmean(diff)),
    }


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cases = build_case_data()

    case_results = []
    for case in cases:
        case_results.append(run_case(case, n_subdomains=0))

    sensitivity = {
        "smooth_stationary": {
            str(n_sd): run_case(cases[0], n_subdomains=n_sd)
            for n_sd in [1, 0]
        }
    }
    sensitivity["smooth_stationary"]["legacy_pum_4"] = run_case(
        cases[0],
        n_subdomains=4,
        estimation_mode="pum_legacy",
    )
    sensitivity["smooth_stationary"]["legacy_pum_8"] = run_case(
        cases[0],
        n_subdomains=8,
        estimation_mode="pum_legacy",
    )
    sensitivity["smooth_stationary"]["1_forced_global"] = run_case(
        cases[0],
        n_subdomains=1,
        max_samples=999,
    )

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": case_results,
        "sensitivity": sensitivity,
        "reproducibility": run_reproducibility_check(cases[0]),
        "pum_threshold_check": run_pum_threshold_check(cases[0]),
        "support_effects": run_support_experiment(),
        "domaining_experiment": run_domaining_experiment(),
        "domain_bug_check": run_domain_bug_check(),
    }

    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("ARBF method audit results written to", OUTPUT_PATH)
    print()
    print("Case summary")
    print("-" * 128)
    print(
        f"{'case':24s} {'drift':12s} {'cov':>6s} {'rmse':>8s} {'r2':>8s} "
        f"{'panel_r2':>10s} {'std_rt':>8s} {'cv_r2':>8s} {'cv_slope':>10s}"
    )
    for row in case_results:
        cv = row["cv"] or {}
        m = row["metrics"]
        pm = row["panel_metrics"] or {}
        drift = str(row["diagnostics"].get("effective_drift_type", "n/a"))
        print(
            f"{row['case'][:24]:24s} "
            f"{drift[:12]:12s} "
            f"{m['finite_fraction']:6.2f} "
            f"{m['rmse']:8.4f} "
            f"{m['r2']:8.4f} "
            f"{float(pm.get('r2', math.nan)):10.4f} "
            f"{float(pm.get('std_ratio', math.nan)):8.4f} "
            f"{float(cv.get('r2', math.nan)):8.4f} "
            f"{float(cv.get('slope', math.nan)):10.4f}"
        )

    print()
    print("Reproducibility")
    print("-" * 96)
    for row in results["reproducibility"]["runs"]:
        print(
            f"global_seed={row['global_seed']} "
            f"r2={row['r2']:.4f} "
            f"grade_mean={row['grade_mean']:.4f}"
        )
    for row in results["reproducibility"]["pairwise_prediction_differences"]:
        print(
            f"pair={row['pair']} "
            f"max_abs_diff={row['max_abs_diff']:.6f} "
            f"mean_abs_diff={row['mean_abs_diff']:.6f}"
        )

    print()
    print("Support effects")
    print("-" * 96)
    for label, row in results["support_effects"].items():
        print(
            f"{label:10s} "
            f"rmse={row['metrics']['rmse']:.4f} "
            f"bias={row['metrics']['bias']:.4f} "
            f"pred_std={row['grade_std']:.4f} "
            f"truth_std={row['truth_std']:.4f}"
        )

    print()
    print("Domaining experiment")
    print("-" * 96)
    for label, row in results["domaining_experiment"].items():
        print(
            f"{label:12s} "
            f"rmse={row['metrics']['rmse']:.4f} "
            f"panel_r2={row['panel_metrics']['r2']:.4f} "
            f"contact_rmse={row['contact_metrics']['rmse']:.4f} "
            f"domains={row['diagnostics'].get('geological_domains', 0)}"
        )

    print()
    print("Domain bug check")
    print("-" * 96)
    print(json.dumps(results["domain_bug_check"], indent=2))


if __name__ == "__main__":
    main()
