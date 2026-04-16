from types import SimpleNamespace

import numpy as np

from block_model_viewer.ui.property_panel import PropertyPanel


class _RegistryStub:
    def __init__(self, results):
        self._results = results

    def get_sgsim_results(self):
        return self._results


def test_build_grid_for_model_derives_domain_mask_from_nan_summary_cells():
    params = SimpleNamespace(
        nx=2,
        ny=2,
        nz=1,
        xinc=10.0,
        yinc=10.0,
        zinc=5.0,
        xmin=0.0,
        ymin=0.0,
        zmin=0.0,
    )
    mean = np.array([[[1.0, np.nan], [2.0, np.nan]]], dtype=float)
    fake_self = SimpleNamespace(
        registry=_RegistryStub(
            {
                "params": params,
                "summary": {"mean": mean},
                "variable": "Au",
            }
        )
    )

    grid = PropertyPanel._build_grid_for_model(fake_self, "AU_SGSIM_MEAN")

    assert grid is not None
    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["domain_mask"]),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["DOMAIN"]),
        np.array([1, 0, 1, 0], dtype=np.int32),
    )
    values = np.asarray(grid.cell_data["Au_SGSIM_MEAN"])
    assert np.isnan(values[1])
    assert np.isnan(values[3])
