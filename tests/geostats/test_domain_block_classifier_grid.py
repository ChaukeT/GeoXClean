import numpy as np
import pyvista as pv

from block_model_viewer.geostats import domain_block_classifier as dbc


def _make_grid():
    grid = pv.ImageData(dimensions=(3, 3, 2), spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
    grid.cell_data["grade"] = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    return grid


def test_apply_domain_mask_to_grid_adds_explicit_domain_arrays(monkeypatch):
    mask = np.array([True, False, True, False], dtype=bool)
    monkeypatch.setattr(dbc, "get_domain_mask", lambda *args, **kwargs: mask)

    grid = _make_grid()
    dbc.apply_domain_mask_to_grid(
        grid,
        {"domain_column": "LITH", "domain_value": "Inside", "_registry": object()},
        method_name="SGSIM",
    )

    grade = np.asarray(grid.cell_data["grade"])
    assert np.isnan(grade[1])
    assert np.isnan(grade[3])
    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["domain_mask"]),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["DOMAIN"]),
        np.array([1, 0, 1, 0], dtype=np.int32),
    )
    assert "DOMAIN_LABEL_MAP" in grid.field_data


def test_apply_domain_mask_to_grid_is_stable_on_repeat(monkeypatch):
    mask = np.array([True, False, True, False], dtype=bool)
    monkeypatch.setattr(dbc, "get_domain_mask", lambda *args, **kwargs: mask)

    grid = _make_grid()
    dbc.apply_domain_mask_to_grid(
        grid,
        {"domain_column": "LITH", "domain_value": "Inside", "_registry": object()},
        method_name="SGSIM",
    )
    dbc.apply_domain_mask_to_grid(
        grid,
        {"domain_column": "LITH", "domain_value": "Inside", "_registry": object()},
        method_name="SGSIM",
    )

    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["domain_mask"]),
        np.array([1, 0, 1, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["DOMAIN"]),
        np.array([1, 0, 1, 0], dtype=np.int32),
    )


def test_apply_domain_mask_to_grid_adds_arrays_even_when_all_cells_are_inside(monkeypatch):
    mask = np.ones(4, dtype=bool)
    monkeypatch.setattr(dbc, "get_domain_mask", lambda *args, **kwargs: mask)

    grid = _make_grid()
    dbc.apply_domain_mask_to_grid(
        grid,
        {"domain_column": "LITH", "domain_value": "Inside", "_registry": object()},
        method_name="SGSIM",
    )

    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["domain_mask"]),
        np.ones(4, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        np.asarray(grid.cell_data["DOMAIN"]),
        np.ones(4, dtype=np.int32),
    )
