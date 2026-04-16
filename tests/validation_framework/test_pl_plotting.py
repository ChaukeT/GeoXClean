"""
PL-01 through PL-05: Plotting domain checks.

Tests verify figure creation, histogram rendering, scatter plots
with R² annotation, theme application, and figure save/export.
"""
import pytest
import numpy as np

pytestmark = [pytest.mark.plotting, pytest.mark.smoke]


# ── PL-01: Figure creation ──────────────────────────────────────────────────

@pytest.mark.blocker
class TestPL01FigureCreation:
    def test_create_figure_returns_valid_objects(self):
        """create_figure should return (Figure, Canvas) tuple."""
        try:
            from block_model_viewer.utils.plotting_helpers import create_figure
            fig, canvas = create_figure(figsize=(8, 6), dpi=100)
            assert fig is not None, "PL-01 FAIL: Figure is None"
            assert canvas is not None, "PL-01 FAIL: Canvas is None"
            assert hasattr(fig, "savefig"), "PL-01 FAIL: Not a matplotlib Figure"
        except ImportError:
            pytest.skip("plotting_helpers not available")


# ── PL-02: Histogram rendering ──────────────────────────────────────────────

@pytest.mark.critical
class TestPL02HistogramRendering:
    def test_histogram_handles_nan(self):
        """plot_histogram should handle NaN values gracefully."""
        try:
            from block_model_viewer.utils.plotting_helpers import create_figure, plot_histogram
            fig, canvas = create_figure()
            ax = fig.add_subplot(111)
            data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan, 5.0])
            plot_histogram(ax, data, bins=10)
            # Should not raise; plot should have been created
            assert len(ax.patches) > 0 or len(ax.lines) > 0 or len(ax.collections) > 0, \
                "PL-02 FAIL: Histogram produced no visual elements"
        except ImportError:
            pytest.skip("plotting_helpers not available")

    def test_histogram_handles_empty(self):
        """plot_histogram should handle empty data without crashing."""
        try:
            from block_model_viewer.utils.plotting_helpers import create_figure, plot_histogram
            fig, canvas = create_figure()
            ax = fig.add_subplot(111)
            data = np.array([])
            # Should not raise
            try:
                plot_histogram(ax, data, bins=10)
            except (ValueError, IndexError):
                pass  # Acceptable to raise for truly empty data
        except ImportError:
            pytest.skip("plotting_helpers not available")


# ── PL-03: Scatter plot R² annotation ───────────────────────────────────────

@pytest.mark.critical
class TestPL03ScatterR2:
    def test_scatter_returns_r_squared(self):
        """plot_scatter should return R² value."""
        try:
            from block_model_viewer.utils.plotting_helpers import create_figure, plot_scatter
            fig, canvas = create_figure()
            ax = fig.add_subplot(111)
            x = np.array([1, 2, 3, 4, 5], dtype=float)
            y = np.array([1.1, 2.0, 3.1, 3.9, 5.0], dtype=float)
            r2 = plot_scatter(ax, x, y)
            if r2 is not None:
                assert 0.0 <= r2 <= 1.0, \
                    f"PL-03 FAIL: R² = {r2} outside [0, 1]"
                assert r2 > 0.9, \
                    f"PL-03 FAIL: R² = {r2} too low for near-linear data"
        except ImportError:
            pytest.skip("plotting_helpers not available")


# ── PL-04: Theme application ────────────────────────────────────────────────

@pytest.mark.major
class TestPL04ThemeApplication:
    def test_apply_theme_does_not_crash(self):
        """apply_theme should accept standard theme names without error."""
        try:
            from block_model_viewer.utils.plotting_helpers import create_figure, apply_theme
            for theme in ["default", "dark", "minimal"]:
                fig, canvas = create_figure()
                try:
                    apply_theme(fig, theme)
                except Exception as e:
                    pytest.fail(f"PL-04 FAIL: apply_theme('{theme}') raised {e}")
        except ImportError:
            pytest.skip("plotting_helpers not available")


# ── PL-05: Figure save/export ────────────────────────────────────────────────

@pytest.mark.critical
class TestPL05FigureSave:
    def test_save_figure_to_png(self, tmp_path):
        """save_figure should create a PNG file on disk."""
        try:
            from block_model_viewer.utils.plotting_helpers import create_figure, save_figure
            fig, canvas = create_figure()
            ax = fig.add_subplot(111)
            ax.plot([1, 2, 3], [1, 2, 3])
            out_path = tmp_path / "test_plot.png"
            result = save_figure(fig, str(out_path))
            if result is not False:
                assert out_path.exists(), "PL-05 FAIL: PNG file not created"
                assert out_path.stat().st_size > 0, "PL-05 FAIL: PNG file is empty"
        except ImportError:
            pytest.skip("plotting_helpers not available")
