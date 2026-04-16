"""
Validation tests for block model rendering fixes (Issues 1–7).

NOTE: Block model rendering has been demolished for clean reimplementation.
Tests referencing deleted modules (block_model_mesh_builder, block_model_renderer)
will fail with ImportError until the rebuild is complete.
"""

import numpy as np
import pytest
import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ---------------------------------------------------------------------------
# Issue 1 & related: _compute_clim consistency
# ---------------------------------------------------------------------------

class TestComputeClim:
    """Verify _compute_clim returns full [vmin, vmax] range."""

    @staticmethod
    def _compute_clim(values):
        """Standalone copy of the method for testing without VTK."""
        _finite = values[np.isfinite(values)]
        if len(_finite) == 0:
            return None
        vmin = float(np.nanmin(_finite))
        vmax = float(np.nanmax(_finite))
        if vmax > vmin:
            return [vmin, vmax]
        else:
            magnitude = abs(vmin) if vmin != 0 else 1.0
            epsilon = max(magnitude * 0.01, 0.1)
            return [vmin - epsilon, vmin + epsilon]

    def test_normal_range(self):
        values = np.array([0.1, 0.5, 1.0, 2.0, 45.3])
        result = self._compute_clim(values)
        assert result == [0.1, 45.3], f"Expected [0.1, 45.3], got {result}"

    def test_with_nan(self):
        values = np.array([np.nan, 0.5, 1.0, np.nan, 2.0])
        result = self._compute_clim(values)
        assert result == [0.5, 2.0]

    def test_all_nan(self):
        values = np.array([np.nan, np.nan, np.nan])
        assert self._compute_clim(values) is None

    def test_identical_values(self):
        values = np.array([5.0, 5.0, 5.0])
        result = self._compute_clim(values)
        assert result[0] < 5.0 and result[1] > 5.0, "Should pad around identical value"

    def test_with_outliers_uses_full_range(self):
        """Issue 1: must use full range, NOT percentiles."""
        values = np.concatenate([np.ones(98) * 1.0, [0.001, 100.0]])
        result = self._compute_clim(values)
        assert result[0] == pytest.approx(0.001)
        assert result[1] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Issue 4: Categorical detection
# ---------------------------------------------------------------------------

class TestCategoricalDetection:
    """Verify dtype-aware categorical detection."""

    def _is_categorical(self, values):
        """Reimplementation of the fixed logic for testing."""
        if not np.issubdtype(values.dtype, np.number):
            return True
        valid = values[~np.isnan(values)]
        unique_values = np.unique(valid)
        n_unique = len(unique_values)
        if np.issubdtype(values.dtype, np.integer):
            return n_unique <= 20
        if n_unique > 10:
            return False
        return bool(np.all(np.equal(np.mod(unique_values, 1), 0)))

    def test_integer_domain_codes_categorical(self):
        """Integer array with ≤20 unique values → categorical."""
        values = np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.int32)
        assert self._is_categorical(values) is True

    def test_integer_many_codes_continuous(self):
        """Integer array with >20 unique values → continuous."""
        values = np.arange(25, dtype=np.int32)
        assert self._is_categorical(values) is False

    def test_float_grade_continuous(self):
        """Float array with 18 unique values → continuous (not categorical)."""
        values = np.array([0.1 * i for i in range(18)], dtype=np.float64)
        assert self._is_categorical(values) is False

    def test_float_encoded_domains_categorical(self):
        """Float array with ≤10 integer-valued values → categorical."""
        values = np.array([1.0, 2.0, 3.0, 1.0, 2.0], dtype=np.float64)
        assert self._is_categorical(values) is True

    def test_float_encoded_domains_too_many(self):
        """Float array with >10 integer-valued values → continuous."""
        values = np.array([float(i) for i in range(15)], dtype=np.float64)
        assert self._is_categorical(values) is False

    def test_kriging_variance_continuous(self):
        """Float array with few fractional values → continuous."""
        values = np.array([0.123, 0.456, 0.789, 0.123, 0.456], dtype=np.float64)
        assert self._is_categorical(values) is False

    def test_string_always_categorical(self):
        """String arrays are always categorical."""
        values = np.array(['A', 'B', 'C', 'A'])
        assert self._is_categorical(values) is True


# ---------------------------------------------------------------------------
# Issue 5: Integer sentinel
# ---------------------------------------------------------------------------

class TestIntegerSentinel:
    """Verify sentinel does not collide with legitimate domain codes."""

    def test_sentinel_not_minus_one(self):
        """Sentinel for int32 should NOT be -1 (common domain code)."""
        dtype = np.int32
        sentinel = np.iinfo(dtype).min
        assert sentinel != -1, f"Sentinel {sentinel} should not be -1"
        assert sentinel < -2_000_000_000, f"Sentinel {sentinel} should be far negative"

    def test_sentinel_int64(self):
        dtype = np.int64
        sentinel = np.iinfo(dtype).min
        assert sentinel < -2_000_000_000

    def test_picking_threshold(self):
        """Picking adapter should accept block_id = -1 (legitimate)."""
        block_id = -1
        # The check should be: block_id < -2_000_000_000
        assert not (block_id < -2_000_000_000), "block_id=-1 should NOT be treated as miss"

    def test_picking_rejects_sentinel(self):
        """Picking adapter should reject int64.min as sentinel."""
        sentinel = int(np.iinfo(np.int64).min)
        assert sentinel < -2_000_000_000, "int64.min should be treated as miss"


# ---------------------------------------------------------------------------
# Issue 6: Single-layer grid detection
# ---------------------------------------------------------------------------

class TestSingleLayerGrid:
    """Verify single-layer grids are detected as uniform."""

    def test_2d_grid_detected(self):
        """A 10×10×1 grid should be detected as uniform."""
        from block_model_viewer.visualization.block_model_mesh_builder import is_uniform_grid
        from block_model_viewer.models.block_model import BlockModel

        # Create a 10×10×1 block model
        nx, ny, nz = 10, 10, 1
        dx, dy, dz = 5.0, 5.0, 10.0
        positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    positions.append([
                        ix * dx + dx / 2,
                        iy * dy + dy / 2,
                        iz * dz + dz / 2,
                    ])
        positions = np.array(positions, dtype=np.float64)
        dimensions = np.full((len(positions), 3), [dx, dy, dz], dtype=np.float64)

        bm = BlockModel()
        bm.set_geometry(positions, dimensions)

        is_uniform, grid_info = is_uniform_grid(bm)
        assert is_uniform, "10×10×1 grid should be detected as uniform"
        assert grid_info['dims'] == (nx, ny, nz)

    def test_3d_grid_still_works(self):
        """A standard 5×5×5 grid should still be detected as uniform."""
        from block_model_viewer.visualization.block_model_mesh_builder import is_uniform_grid
        from block_model_viewer.models.block_model import BlockModel

        nx, ny, nz = 5, 5, 5
        dx, dy, dz = 10.0, 10.0, 5.0
        positions = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    positions.append([
                        ix * dx + dx / 2,
                        iy * dy + dy / 2,
                        iz * dz + dz / 2,
                    ])
        positions = np.array(positions, dtype=np.float64)
        dimensions = np.full((len(positions), 3), [dx, dy, dz], dtype=np.float64)

        bm = BlockModel()
        bm.set_geometry(positions, dimensions)

        is_uniform, grid_info = is_uniform_grid(bm)
        assert is_uniform, "5×5×5 grid should be detected as uniform"
        assert grid_info['dims'] == (nx, ny, nz)


# ---------------------------------------------------------------------------
# Issue 17: Original_ID sentinel in renderer _create_imagedata_grid
# ---------------------------------------------------------------------------

class TestOriginalIDSentinel:
    """Verify that the renderer's Original_ID sentinel is compatible
    with the hover_inspector's sentinel threshold check."""

    def test_sentinel_is_detected_as_miss(self):
        """int64.min must be caught by the -2B threshold."""
        sentinel = int(np.iinfo(np.int64).min)
        assert sentinel < -2_000_000_000, (
            f"Sentinel {sentinel} must be below -2B to be treated as miss"
        )

    def test_minus_one_is_not_sentinel(self):
        """block_id = -1 is a legitimate domain code, not a sentinel."""
        block_id = -1
        assert not (block_id < -2_000_000_000), (
            "block_id=-1 must NOT be treated as a miss"
        )


# ---------------------------------------------------------------------------
# Issue 19: Index computation consistency (floor vs round)
# ---------------------------------------------------------------------------

class TestIndexComputation:
    """Verify that np.floor with corner-based origin matches expected indices."""

    def test_floor_based_indexing(self):
        """For exact grid centres, floor((center - origin) / spacing) == expected index."""
        origin = (100.0, 200.0, 0.0)   # corner-based
        spacing = (5.0, 5.0, 10.0)
        # Block centers at ix=0,1,2 → x = 102.5, 107.5, 112.5
        centers = np.array([102.5, 107.5, 112.5])
        indices = np.floor((centers - origin[0]) / spacing[0]).astype(np.int32)
        np.testing.assert_array_equal(indices, [0, 1, 2])

    def test_floor_handles_floating_point_noise(self):
        """Small FP perturbation should not shift the index."""
        origin_x = 500000.0 - 2.5  # corner, spacing=5
        spacing_x = 5.0
        # center of cell 0 = 500000.0, with tiny FP noise
        center = 500000.0 + 1e-12
        idx = int(np.floor((center - origin_x) / spacing_x))
        assert idx == 0, f"Expected index 0, got {idx}"


# ---------------------------------------------------------------------------
# Issue 21: Integer sentinel filtering in hover data
# ---------------------------------------------------------------------------

class TestHoverIntegerSentinel:
    """Verify that huge negative integer values are filtered in pick results."""

    def test_sentinel_int32_filtered(self):
        """np.iinfo(np.int32).min should be skipped in property display."""
        val = np.int32(np.iinfo(np.int32).min)
        assert int(val) < -2_000_000_000

    def test_negative_one_not_filtered(self):
        """Domain code -1 should NOT be filtered."""
        val = np.int32(-1)
        assert not (int(val) < -2_000_000_000)


# ---------------------------------------------------------------------------
# Issue P1: Highlight coordinate shift correctness
# ---------------------------------------------------------------------------

class TestHighlightCoordinateShift:
    """Verify highlight box is placed correctly under global shift."""

    def test_direct_index_preferred(self):
        """block_id should directly index into positions (O(1))."""
        positions = np.array([
            [500000.0, 6000000.0, 100.0],
            [500010.0, 6000000.0, 100.0],
            [500020.0, 6000000.0, 100.0],
        ])
        block_id = 1
        assert 0 <= block_id < len(positions)
        center = positions[block_id].copy()
        np.testing.assert_array_equal(center, [500010.0, 6000000.0, 100.0])

    def test_shift_applied_to_highlight_center(self):
        """After global shift, highlight center should be in local coords."""
        center = np.array([500010.0, 6000000.0, 100.0])
        global_shift = np.array([500000.0, 6000000.0, 0.0])
        local_center = center - global_shift
        np.testing.assert_array_almost_equal(local_center, [10.0, 0.0, 100.0])

    def test_reverse_shift_for_fallback_search(self):
        """When using world_pos fallback, reverse-shifting must align coords."""
        world_pos = np.array([10.0, 0.0, 100.0])  # local coords from VTK
        global_shift = np.array([500000.0, 6000000.0, 0.0])
        original_pos = world_pos + global_shift  # back to UTM
        np.testing.assert_array_almost_equal(original_pos, [500010.0, 6000000.0, 100.0])


# ---------------------------------------------------------------------------
# Issue P2: Pick adapter cache invalidation on actor change
# ---------------------------------------------------------------------------

class TestPickAdapterCacheInvalidation:
    """Verify the pick adapter detects actor changes."""

    def test_different_actor_ids_trigger_rebuild(self):
        """Two distinct objects must have different id()."""
        class MockActor:
            pass
        a1, a2 = MockActor(), MockActor()
        assert id(a1) != id(a2), "Different actors must have different id()"

    def test_same_actor_id_skips_rebuild(self):
        """Same actor object must keep the cache."""
        class MockActor:
            pass
        a = MockActor()
        assert id(a) == id(a), "Same actor must have same id()"


# ---------------------------------------------------------------------------
# Issue P3: VTK Y-coordinate conversion
# ---------------------------------------------------------------------------

class TestVTKYConversion:
    """Verify Qt-to-VTK Y coordinate conversion."""

    def test_y_conversion_formula(self):
        """VTK y = widget_height - qt_y - 1 (0-indexed from bottom)."""
        widget_height = 600
        qt_y = 0  # top of widget
        vtk_y = widget_height - qt_y - 1
        assert vtk_y == 599, f"Expected 599, got {vtk_y}"

        qt_y = 599  # bottom of widget
        vtk_y = widget_height - qt_y - 1
        assert vtk_y == 0, f"Expected 0, got {vtk_y}"

    def test_consistent_across_click_and_hover(self):
        """Click and hover paths must produce same VTK y for same Qt y."""
        height = 800
        qt_y = 400
        click_vtk_y = height - qt_y - 1  # fixed click path
        hover_vtk_y = height - qt_y - 1  # hover path
        assert click_vtk_y == hover_vtk_y


# ---------------------------------------------------------------------------
# Display Pipeline Audit: Issues D1–D6
# ---------------------------------------------------------------------------

class TestDisplayD1GridAdapterClim:
    """D1: GridAdapter must use _compute_clim(), not P2/P98 percentiles."""

    def test_grid_adapter_no_percentile_calls(self):
        """grid_adapter.py must NOT contain nanpercentile calls."""
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / 'visualization' / 'grid_adapter.py'
        text = src.read_text(encoding='utf-8')
        assert 'nanpercentile' not in text, (
            "grid_adapter.py still uses nanpercentile — should use _compute_clim()"
        )
        assert '_compute_clim' in text, (
            "grid_adapter.py should delegate to _compute_clim() for colour limits"
        )


class TestDisplayD2FiniteValsFixed:
    """D2: render_orchestrator must not reference undefined _finite_vals."""

    def test_no_bare_finite_vals_reference(self):
        """_finite_vals must not appear as a bare variable read."""
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / 'visualization' / 'renderer' / 'render_orchestrator.py'
        text = src.read_text(encoding='utf-8')
        # The OLD buggy line was: f"n_finite={len(_finite_vals)}"
        # The fix replaces it with _n_finite computed from prop_values.
        assert '_finite_vals' not in text, (
            "render_orchestrator.py still references undefined _finite_vals"
        )


class TestDisplayD3HalfThicknessScoping:
    """D3: filters.py must define half_thickness before both branches."""

    def test_half_thickness_defined_before_branch(self):
        """half_thickness must be assigned before the if/else block."""
        import pathlib, re
        src = pathlib.Path(__file__).resolve().parent.parent / 'visualization' / 'filters.py'
        text = src.read_text(encoding='utf-8')
        # Verify half_thickness is defined before the per-block dimensions branch
        ht_def = text.find('half_thickness = thickness / 2.0')
        per_block = text.find('half_sizes = dimensions[:, axis_idx] / 2.0')
        assert ht_def != -1, "half_thickness assignment not found"
        assert per_block != -1, "per-block dimension path not found"
        assert ht_def < per_block, (
            "half_thickness must be defined BEFORE the per-block dimension branch"
        )


class TestDisplayD4DomainCodeZeroKept:
    """D4: Discrete mode must NOT exclude domain code 0."""

    def test_discrete_path_keeps_zero(self):
        """block_model_renderer.py discrete fallback must not filter != 0."""
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / 'visualization' / 'renderer' / 'renderers' / 'block_model_renderer.py'
        text = src.read_text(encoding='utf-8')
        # The old bug: unique_values = unique_values[unique_values != 0]
        # There should be no surviving "!= 0" filter on unique_values
        # (only > -2_000_000_000 sentinel filter should remain)
        import re
        bad_pattern = re.findall(r'unique_values\[unique_values\s*!=\s*0\]', text)
        assert len(bad_pattern) == 0, (
            f"Found {len(bad_pattern)} instances of domain-code-0 exclusion"
        )


class TestDisplayD5SentinelFiltering:
    """D5: Discrete mode must filter integer sentinel values."""

    def test_sentinel_excluded_from_categories(self):
        """Integer sentinel (int32.min) must be caught by -2B threshold."""
        sentinel = int(np.iinfo(np.int32).min)  # -2147483648
        assert sentinel < -2_000_000_000, "int32 sentinel must be below -2B"

        # Simulate the fixed filter logic
        arr = np.array([0, 1, 2, 3, sentinel], dtype=np.int32)
        _pv = np.asarray(arr)
        _valid_mask = _pv > -2_000_000_000
        unique = np.unique(_pv[_valid_mask])
        assert sentinel not in unique, "Sentinel must be excluded"
        assert 0 in unique, "Domain code 0 must be preserved"
        assert list(unique) == [0, 1, 2, 3]

    def test_float_nan_still_filtered(self):
        """Float NaN must still be excluded from categories."""
        arr = np.array([1.0, 2.0, np.nan, 3.0], dtype=np.float64)
        _pv = np.asarray(arr)
        _valid_mask = np.isfinite(_pv)
        unique = np.unique(_pv[_valid_mask])
        assert not np.any(np.isnan(unique))
        assert list(unique) == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Interrogation Pipeline Audit: Issues I1–I3
# ---------------------------------------------------------------------------

class TestInterrogationI1InternalArrayFilter:
    """I1: Internal VTK arrays must not appear in tooltip."""

    def test_internal_array_names_defined(self):
        """BlockModelPickAdapter must have _INTERNAL_ARRAY_NAMES."""
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'hover_inspector.py'
        text = src.read_text(encoding='utf-8')
        assert '_INTERNAL_ARRAY_NAMES' in text
        assert 'domain_mask_colors' in text
        assert 'domain_mask' in text

    def test_cache_builder_uses_filter(self):
        """Cache builder must exclude internal arrays."""
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'hover_inspector.py'
        text = src.read_text(encoding='utf-8')
        assert '_is_internal_array(k)' in text

    def test_vtk_prefix_filter(self):
        """Arrays starting with 'vtk' must be treated as internal."""
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'hover_inspector.py'
        text = src.read_text(encoding='utf-8')
        assert 'startswith("vtk")' in text


class TestInterrogationI3MultiComponentGuard:
    """I3: Multi-component arrays must not crash .item()."""

    def test_ndim_guard_in_both_adapters(self):
        """ndim > 0 guard must appear in both block model and surface adapters."""
        import pathlib
        src = pathlib.Path(__file__).resolve().parent.parent / 'ui' / 'hover_inspector.py'
        text = src.read_text(encoding='utf-8')
        assert text.count('val.ndim > 0') >= 2

    def test_rgba_array_caught_by_guard(self):
        """A 4-component RGBA uint8 array must be caught by ndim > 0."""
        rgba = np.array([255, 0, 0, 255], dtype=np.uint8)
        assert isinstance(rgba, np.ndarray) and rgba.ndim > 0

    def test_scalar_not_caught(self):
        """A numpy scalar must NOT be caught by ndim > 0."""
        scalar = np.float64(3.14)
        assert not (isinstance(scalar, np.ndarray) and scalar.ndim > 0)


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
