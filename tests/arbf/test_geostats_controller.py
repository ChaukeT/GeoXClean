import numpy as np

from block_model_viewer.controllers.geostats_controller import (
    GeostatsController,
    _arbf_transform_cv_payload_to_original,
)


class _DummyApp:
    block_model = None


def test_build_arbf_config_preserves_footprint_clip_settings():
    controller = GeostatsController(_DummyApp())

    config = controller._build_arbf_config(
        {
            "clip_to_drill_footprint": True,
            "footprint_buffer_ranges": 0.75,
            "range_max": 80.0,
            "range_mid": 60.0,
            "range_min": 20.0,
        },
    )

    assert config["clip_to_drill_footprint"] is True
    assert config["footprint_buffer_ranges"] == 0.75


class _AffineTransformer:
    def back_transform(self, values):
        arr = np.asarray(values, dtype=float)
        return arr * 10.0 + 5.0


def test_transform_cv_payload_to_original_recomputes_aliases_and_scale():
    payload = {
        "actual": [0.0, 1.0],
        "estimated": [0.1, 0.9],
        "pred_std": [0.2, 0.3],
    }

    transformed = _arbf_transform_cv_payload_to_original(payload, _AffineTransformer())

    np.testing.assert_allclose(transformed["actual"], [5.0, 15.0])
    np.testing.assert_allclose(transformed["estimated"], [6.0, 14.0])
    np.testing.assert_allclose(transformed["pred_std"], [2.0, 3.0])
    assert transformed["rmse"] == transformed["RMSE"]
    assert transformed["r_squared"] == transformed["R2"]
    assert transformed["mean_error"] == transformed["ME"]
    assert transformed["mae"] == transformed["MAE"]
    assert transformed["n_samples"] == 2
