import numpy as np

from geostats.arbf.recommend import (
    ARBFRecommendation,
    _build_recommendations,
    recommend_arbf_settings,
)


def _random_coords(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 100.0, size=(n, 3))


def _simple_blocks() -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(5) * 20.0 + 10.0
    ys = np.arange(5) * 20.0 + 10.0
    zs = np.arange(3) * 20.0 + 10.0
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    centroids = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    sizes = np.array([20.0, 20.0, 20.0], dtype=float)
    return centroids, sizes


def test_recommendation_skips_ns_and_clipping_for_standardized_gaussian_support():
    rng = np.random.default_rng(123)
    coords = _random_coords(400, seed=123)
    values = rng.standard_normal(400)
    block_centroids, block_sizes = _simple_blocks()

    rec = recommend_arbf_settings(coords, values, block_centroids, block_sizes)

    assert rec.univariate["appears_normal_scored"] is True
    assert rec.settings["use_normal_score"] is False
    assert "clip_max" not in rec.settings
    assert "clip_min" not in rec.settings
    assert rec.univariate["outlier_analysis"]["applicable"] is False
    assert "normal-scored" in " ".join(rec.reasons).lower()


def test_recommendation_keeps_ns_and_top_cut_for_positive_skewed_grade_data():
    rng = np.random.default_rng(456)
    coords = _random_coords(500, seed=456)
    values = np.exp(rng.normal(2.2, 1.0, size=500))
    values[:8] *= 20.0
    block_centroids, block_sizes = _simple_blocks()

    rec = recommend_arbf_settings(coords, values, block_centroids, block_sizes)

    assert rec.univariate["positive_grade_like"] is True
    assert rec.univariate["cv_is_meaningful"] is True
    assert rec.settings["use_normal_score"] is True
    assert "clip_max" in rec.settings
    assert "clip_min" not in rec.settings
    assert rec.univariate["outlier_analysis"]["applicable"] is True
    assert rec.univariate["outlier_analysis"]["metal_at_risk_pct"] > 20.0


def test_recommendation_only_bottom_clips_zero_bounded_data_with_small_negative_tail():
    rng = np.random.default_rng(789)
    coords = _random_coords(450, seed=789)
    values = np.exp(rng.normal(1.8, 0.8, size=450))
    values[:6] = -0.05
    block_centroids, block_sizes = _simple_blocks()

    rec = recommend_arbf_settings(coords, values, block_centroids, block_sizes)

    assert rec.univariate["positive_grade_like"] is True
    assert rec.univariate["negative_fraction"] < 0.05
    assert rec.settings["clip_min"] == 0.0
    assert rec.settings["use_normal_score"] is True


def test_recommendation_tightens_neighbourhood_for_nonstationary_unsupported_case():
    rec = ARBFRecommendation(
        univariate={
            "n": 1314,
            "cv": 1.4,
            "skewness": 2.0,
            "cv_is_meaningful": True,
            "appears_normal_scored": False,
            "positive_grade_like": True,
            "centered_signed": False,
            "negative_fraction": 0.0,
            "distribution_type": "lognormal",
            "outlier_analysis": {"applicable": False, "reason": "not needed"},
            "min": 0.1,
        },
        spatial={
            "median_spacing": 35.0,
            "clustering_coeff": 0.35,
            "data_volume": 8.0e6,
            "preferential_sampling": True,
        },
        variography={
            "anisotropy": {
                "range_max": 231.1,
                "range_mid": 142.6,
                "range_min": 12.1,
                "ratio_max_min": 19.1,
                "nugget": 0.1977,
                "sill": 0.8047,
                "nugget_fraction": 0.1977,
                "major_azimuth": 146.3,
                "major_dip": 0.0,
            },
        },
        stationarity={
            "trend_detected": True,
            "trend_direction": "Z",
            "trend_r2": 0.189,
            "proportional_effect": True,
        },
        contact={"multimodal": False},
        compositing={"is_regular": False, "length_cv": 2.61, "median_length": 3.0},
        block_coverage={"pct_within_1_range": 0.0, "clip_recommended": True, "volume_ratio": 5.0},
    )

    _build_recommendations(rec, block_sizes=np.array([10.0, 10.0, 10.0], dtype=float))

    assert rec.settings["drift_type"] == "linear"
    assert rec.settings["clip_to_drill_footprint"] is True
    assert rec.settings["max_samples"] <= 32
    assert rec.settings["min_samples"] >= 6
    assert rec.settings["search_radius_1"] <= 0.75
    assert rec.settings["search_radius_2"] <= 1.50
    assert rec.settings["search_radius_3"] <= 3.00
    assert rec.settings["balanced_octant"] is True
    assert rec.settings["max_per_octant"] <= 4
