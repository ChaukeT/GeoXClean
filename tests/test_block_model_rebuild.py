"""Regression tests for the block model rebuild.

Tests the core patterns from block_model_patterns.md:
- Tolerance-aware grid detection
- Ghost cell domain toggle
- ImageData vs UnstructuredGrid selection
- SGSIM round-trip geometry preservation
- In-place LUT construction
"""

import numpy as np
import pyvista as pv
import pytest
import vtk


class TestGridUtils:
    """Test tolerance-aware coordinate utilities."""

    def test_unique_with_tolerance_recovers_from_jitter(self):
        """Exercise 4.1: np.unique gives 3991 values, ours gives 10."""
        from block_model_viewer.utils.grid_utils import unique_with_tolerance

        rng = np.random.default_rng(42)
        true_z = np.repeat(np.arange(10) * 2.5 + 1.25, 400)
        noisy_z = true_z + rng.uniform(-1e-10, 1e-10, len(true_z))

        # np.unique fails
        raw = np.unique(noisy_z)
        assert len(raw) > 100, f"np.unique should give many false uniques, got {len(raw)}"

        # Ours succeeds
        tol = unique_with_tolerance(noisy_z)
        assert len(tol) == 10, f"Expected 10 unique Z values, got {len(tol)}"

    def test_infer_spacing_from_jittered_coords(self):
        from block_model_viewer.utils.grid_utils import infer_spacing_from_coordinates

        rng = np.random.default_rng(42)
        coords = np.repeat(np.arange(20) * 5.0 + 2.5, 200)
        coords += rng.uniform(-1e-10, 1e-10, len(coords))

        spacing = infer_spacing_from_coordinates(coords)
        assert abs(spacing - 5.0) < 0.01, f"Expected 5.0, got {spacing}"

    def test_detect_uniform_grid(self):
        from block_model_viewer.utils.grid_utils import detect_uniform_grid

        rng = np.random.default_rng(42)
        nx, ny, nz = 10, 10, 5
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )
        positions = np.column_stack([
            ii.ravel() * 5.0 + 2.5,
            jj.ravel() * 5.0 + 2.5,
            kk.ravel() * 2.5 + 1.25,
        ])
        positions += rng.uniform(-1e-10, 1e-10, positions.shape)

        is_uniform, info = detect_uniform_grid(positions)
        assert is_uniform
        assert info["nx"] == 10
        assert info["ny"] == 10
        assert info["nz"] == 5
        assert abs(info["dx"] - 5.0) < 0.01
        assert abs(info["dz"] - 2.5) < 0.01


class TestMeshBuilder:
    """Test block model mesh builder."""

    def test_uniform_grid_produces_imagedata(self):
        from block_model_viewer.models.block_model import BlockModel
        from block_model_viewer.visualization.block_model_mesh_builder import build_mesh

        bm = BlockModel()
        nx, ny, nz = 10, 10, 5
        ii, jj, kk = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij'
        )
        positions = np.column_stack([
            ii.ravel() * 5 + 2.5, jj.ravel() * 5 + 2.5, kk.ravel() * 2.5 + 1.25
        ])
        dims = np.full((500, 3), [5, 5, 2.5])
        bm.set_geometry(positions, dims)
        bm.add_property("Fe", np.random.default_rng(42).uniform(0, 65, 500))

        mesh, info = build_mesh(bm)
        assert isinstance(mesh, pv.ImageData), f"Expected ImageData, got {type(mesh)}"
        assert info["grid_type"] == "ImageData"
        assert mesh.n_cells == 500
        assert "Original_ID" in mesh.cell_data

    def test_nonuniform_dims_produces_unstructured(self):
        from block_model_viewer.models.block_model import BlockModel
        from block_model_viewer.visualization.block_model_mesh_builder import build_mesh

        bm = BlockModel()
        pos = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float64)
        dim = np.array([[10, 10, 5], [5, 5, 5]], dtype=np.float64)
        bm.set_geometry(pos, dim)
        bm.add_property("Fe", np.array([10, 20], dtype=np.float64))

        mesh, info = build_mesh(bm)
        assert info["grid_type"] == "UnstructuredGrid"

    def test_build_imagedata_direct(self):
        from block_model_viewer.visualization.block_model_mesh_builder import build_imagedata_direct

        grid = build_imagedata_direct(
            origin=(100, 200, 50),
            spacing=(5, 5, 2.5),
            dimensions=(20, 20, 10),
            cell_data={"Fe": np.random.default_rng(42).uniform(0, 65, 4000)},
        )
        assert isinstance(grid, pv.ImageData)
        assert grid.n_cells == 4000
        assert "Fe" in grid.cell_data
        assert "Original_ID" in grid.cell_data


class TestGhostCells:
    """Test ghost cell domain toggle."""

    def test_ghost_toggle_preserves_topology(self):
        from block_model_viewer.visualization.block_model_mesh_builder import (
            build_imagedata_direct, update_ghost_cells, clear_ghost_cells, GHOST_NAME
        )

        grid = build_imagedata_direct(
            origin=(0, 0, 0), spacing=(5, 5, 2.5), dimensions=(10, 10, 5),
            cell_data={"Domain": np.repeat(np.arange(1, 6), 100).astype(np.int32)},
        )
        assert grid.n_cells == 500

        # Hide domain 5
        visible = grid.cell_data["Domain"] != 5
        update_ghost_cells(grid, visible)

        assert GHOST_NAME in grid.cell_data
        assert grid.n_cells == 500  # Topology preserved!
        assert isinstance(grid, pv.ImageData)  # Type preserved!

        # Clear ghost — all visible
        clear_ghost_cells(grid)
        assert GHOST_NAME not in grid.cell_data
        assert grid.n_cells == 500


class TestRendererHelpers:
    """Test renderer auto-cmap and LUT building."""

    def test_auto_cmap_continuous_for_float(self):
        from block_model_viewer.visualization.renderer.renderers.block_model_renderer import _auto_cmap

        grid = pv.ImageData(dimensions=(11, 11, 6), spacing=(5, 5, 2.5))
        grid.cell_data["Fe"] = np.random.default_rng(42).uniform(0, 65, grid.n_cells)
        cmap, n_colors = _auto_cmap(grid, "Fe")
        assert cmap == "turbo"
        assert n_colors is None

    def test_auto_cmap_categorical_for_integer(self):
        from block_model_viewer.visualization.renderer.renderers.block_model_renderer import _auto_cmap

        grid = pv.ImageData(dimensions=(11, 11, 6), spacing=(5, 5, 2.5))
        grid.cell_data["Domain"] = np.repeat(np.arange(1, 5), 125).astype(np.int32)
        cmap, n_colors = _auto_cmap(grid, "Domain")
        assert cmap == "tab10"
        assert n_colors == 4

    def test_build_lut_continuous(self):
        from block_model_viewer.visualization.renderer.renderers.block_model_renderer import _build_lut

        grid = pv.ImageData(dimensions=(11, 11, 6), spacing=(5, 5, 2.5))
        grid.cell_data["Fe"] = np.random.default_rng(42).uniform(0, 65, grid.n_cells)

        lut = _build_lut(grid, "Fe", "turbo", (0, 65))
        assert lut.GetNumberOfTableValues() == 256
        assert lut.GetRange() == (0.0, 65.0)

    def test_build_lut_categorical(self):
        from block_model_viewer.visualization.renderer.renderers.block_model_renderer import _build_lut

        grid = pv.ImageData(dimensions=(11, 11, 6), spacing=(5, 5, 2.5))
        grid.cell_data["Domain"] = np.repeat(np.arange(1, 5), 125).astype(np.int32)

        lut = _build_lut(grid, "Domain", "tab10", (1, 4))
        assert lut.GetNumberOfTableValues() == 4
        assert lut.GetRange() == (1.0, 4.0)


class TestSGSIMRoundTrip:
    """Test that SGSIM results preserve geometry through the pipeline."""

    def test_direct_imagedata_preserves_spacing(self):
        """Exercise 4.2 fix: direct path avoids np.unique vulnerability."""
        from block_model_viewer.visualization.block_model_mesh_builder import build_imagedata_direct

        grid = build_imagedata_direct(
            origin=(100, 200, 50),
            spacing=(5, 5, 2.5),
            dimensions=(20, 20, 10),
        )
        assert grid.spacing == (5.0, 5.0, 2.5)
        assert grid.origin == (100.0, 200.0, 50.0)
        assert grid.dimensions == (21, 21, 11)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
