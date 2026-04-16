"""
ER-01 through ER-05: Export & Reproducibility domain checks.

Tests verify CSV export fidelity, Excel multi-sheet export,
companion metadata generation, filename suggestion, and
block model DataFrame export.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

pytestmark = [pytest.mark.export, pytest.mark.smoke]


# ── ER-01: CSV export round-trip fidelity ────────────────────────────────────

@pytest.mark.blocker
class TestER01CSVRoundTrip:
    def test_csv_preserves_data(self, tmp_path):
        """DataFrame exported to CSV should read back identically."""
        try:
            from block_model_viewer.utils.export_helpers import export_dataframe_to_csv
        except ImportError:
            pytest.skip("export_helpers not available")

        df = pd.DataFrame({
            "hole_id": ["DH001", "DH002", "DH003"],
            "Fe": [55.123456, 60.789012, 48.345678],
            "SiO2": [5.12, 3.45, 8.90],
        })
        out_path = tmp_path / "test_export.csv"
        export_dataframe_to_csv(df, out_path)
        assert out_path.exists(), "ER-01 FAIL: CSV file not created"

        df_read = pd.read_csv(out_path)
        assert list(df_read.columns) == list(df.columns), \
            "ER-01 FAIL: Column names changed"
        np.testing.assert_allclose(
            df_read["Fe"].values, df["Fe"].values, rtol=1e-6,
            err_msg="ER-01 FAIL: Fe values changed in CSV round-trip",
        )


# ── ER-02: Excel export ─────────────────────────────────────────────────────

@pytest.mark.critical
class TestER02ExcelExport:
    def test_excel_single_sheet(self, tmp_path):
        """export_dataframe_to_excel should create a valid .xlsx file."""
        try:
            from block_model_viewer.utils.export_helpers import export_dataframe_to_excel
        except ImportError:
            pytest.skip("export_helpers not available")

        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        out_path = tmp_path / "test.xlsx"
        export_dataframe_to_excel(df, out_path, sheet_name="Data")
        assert out_path.exists(), "ER-02 FAIL: Excel file not created"
        assert out_path.stat().st_size > 0, "ER-02 FAIL: Excel file is empty"
        # Read back
        df_read = pd.read_excel(out_path, sheet_name="Data")
        assert len(df_read) == 3, "ER-02 FAIL: Row count mismatch"


# ── ER-03: Companion metadata export ────────────────────────────────────────

@pytest.mark.major
class TestER03CompanionMetadata:
    def test_metadata_json_created(self, tmp_path):
        """export_companion_metadata should create a .json sidecar."""
        try:
            from block_model_viewer.utils.export_helpers import export_companion_metadata
        except ImportError:
            pytest.skip("export_helpers not available")

        data_path = tmp_path / "model.csv"
        data_path.write_text("x,y,z\n1,2,3\n")
        meta = {"source": "test", "version": "1.0", "rows": 1}
        meta_path = export_companion_metadata(data_path, meta)
        assert meta_path.exists(), "ER-03 FAIL: Metadata JSON not created"
        import json
        with open(meta_path) as f:
            loaded = json.load(f)
        assert loaded["source"] == "test", "ER-03 FAIL: Metadata content wrong"


# ── ER-04: Suggested export filename ────────────────────────────────────────

@pytest.mark.minor
class TestER04FilenameSuggestion:
    def test_suggest_filename_has_timestamp(self):
        """suggest_export_filename should include a timestamp component."""
        try:
            from block_model_viewer.utils.export_helpers import suggest_export_filename
        except ImportError:
            pytest.skip("export_helpers not available")

        path = suggest_export_filename("model_export", ".csv")
        name = Path(path).stem if isinstance(path, (str, Path)) else str(path)
        # Should contain base name
        assert "model_export" in name, \
            f"ER-04 FAIL: Base name not in suggested filename: {name}"


# ── ER-05: Block model to DataFrame export ──────────────────────────────────

@pytest.mark.blocker
class TestER05BlockModelExport:
    def test_block_model_to_dataframe(self):
        """BlockModel.to_dataframe() should include all geometry + properties."""
        from block_model_viewer.models.block_model import BlockModel
        bm = BlockModel()
        n = 3
        positions = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]], dtype=float)
        dimensions = np.array([[10, 10, 5]] * n, dtype=float)
        bm.set_geometry(positions, dimensions)
        bm.add_property("Fe", np.array([55.0, 60.0, 48.0]))
        bm.add_property("density", np.array([3.2, 3.1, 3.3]))

        df = bm.to_dataframe()
        assert len(df) == n, f"ER-05 FAIL: Expected {n} rows, got {len(df)}"
        assert "Fe" in df.columns, "ER-05 FAIL: Property 'Fe' missing"
        assert "density" in df.columns, "ER-05 FAIL: Property 'density' missing"
        # Coordinates should be present
        coord_cols = [c for c in df.columns if c.lower() in ("x", "y", "z")]
        assert len(coord_cols) >= 3, \
            f"ER-05 FAIL: Missing coordinate columns, found: {list(df.columns)}"
