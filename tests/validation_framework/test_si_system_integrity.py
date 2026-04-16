"""
SI-01 through SI-06: System Integrity domain checks.

Tests verify module import health, py_compile on all source files,
configuration loading, logging infrastructure, and dependency
version constraints.
"""
import pytest
import sys
from pathlib import Path

pytestmark = [pytest.mark.system_integrity, pytest.mark.smoke]

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ── SI-01: All Python source files compile ───────────────────────────────────

@pytest.mark.blocker
class TestSI01SourceCompilation:
    def test_all_py_files_compile(self):
        """Every .py file under block_model_viewer/ must pass py_compile."""
        import py_compile
        src_dir = PROJECT_ROOT / "block_model_viewer"
        failures = []
        py_files = list(src_dir.rglob("*.py"))
        assert len(py_files) > 0, "SI-01 FAIL: No .py files found"
        for f in py_files:
            try:
                py_compile.compile(str(f), doraise=True)
            except py_compile.PyCompileError as e:
                failures.append(f"{f.name}: {e}")
        assert len(failures) == 0, \
            f"SI-01 FAIL: {len(failures)} files failed compilation:\n" + \
            "\n".join(failures[:10])


# ── SI-02: Core module imports ───────────────────────────────────────────────

@pytest.mark.blocker
class TestSI02CoreImports:
    def test_import_data_registry(self):
        from block_model_viewer.core.data_registry import DataRegistry
        assert DataRegistry is not None

    def test_import_block_model(self):
        from block_model_viewer.models.block_model import BlockModel
        assert BlockModel is not None

    def test_import_desurvey(self):
        from block_model_viewer.utils.desurvey import minimum_curvature_desurvey
        assert minimum_curvature_desurvey is not None

    def test_import_compositing_engine(self):
        from block_model_viewer.drillholes.compositing_engine import CompositingMethodEngine
        assert CompositingMethodEngine is not None

    def test_import_control_samples(self):
        from block_model_viewer.drillholes.control_samples import ControlSampleManager
        assert ControlSampleManager is not None

    def test_import_audit_trail(self):
        from block_model_viewer.drillholes.audit_trail import AuditTrail, AuditRecord
        assert AuditTrail is not None and AuditRecord is not None

    def test_import_variogram_model(self):
        from block_model_viewer.geostats.variogram_model import VariogramModel
        assert VariogramModel is not None


# ── SI-03: No circular imports ──────────────────────────────────────────────

@pytest.mark.critical
class TestSI03NoCircularImports:
    def test_fresh_import_of_core_modules(self):
        """Importing core modules should not trigger circular import errors."""
        import importlib
        modules = [
            "block_model_viewer.core.data_registry",
            "block_model_viewer.models.block_model",
            "block_model_viewer.drillholes.compositing_engine",
            "block_model_viewer.drillholes.compositing_utils",
            "block_model_viewer.drillholes.control_samples",
            "block_model_viewer.drillholes.audit_trail",
            "block_model_viewer.utils.desurvey",
        ]
        for mod_name in modules:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                if "circular" in str(e).lower():
                    pytest.fail(f"SI-03 FAIL: Circular import in {mod_name}: {e}")


# ── SI-04: Logging infrastructure ───────────────────────────────────────────

@pytest.mark.major
class TestSI04LoggingInfrastructure:
    def test_autofix_has_logger(self):
        """drillhole_autofix.py must define a logger."""
        from block_model_viewer.drillholes import drillhole_autofix
        assert hasattr(drillhole_autofix, "logger"), \
            "SI-04 FAIL: drillhole_autofix missing logger"

    def test_audit_trail_has_logger(self):
        """audit_trail.py must define a logger."""
        from block_model_viewer.drillholes import audit_trail
        assert hasattr(audit_trail, "logger"), \
            "SI-04 FAIL: audit_trail missing logger"


# ── SI-05: Dependency version bounds ────────────────────────────────────────

@pytest.mark.major
class TestSI05DependencyVersions:
    def test_numpy_version(self):
        import numpy
        major, minor = map(int, numpy.__version__.split(".")[:2])
        assert (major, minor) >= (1, 24), \
            f"SI-05 FAIL: numpy {numpy.__version__} < 1.24"

    def test_pandas_version(self):
        import pandas
        major, minor = map(int, pandas.__version__.split(".")[:2])
        assert (major, minor) >= (2, 0), \
            f"SI-05 FAIL: pandas {pandas.__version__} < 2.0"

    def test_scipy_version(self):
        try:
            import scipy
            major, minor = map(int, scipy.__version__.split(".")[:2])
            assert (major, minor) >= (1, 10), \
                f"SI-05 FAIL: scipy {scipy.__version__} < 1.10"
        except ImportError:
            pytest.skip("scipy not installed")


# ── SI-06: Python version compatibility ─────────────────────────────────────

@pytest.mark.blocker
class TestSI06PythonVersion:
    def test_python_310_or_higher(self):
        """Project requires Python >= 3.10."""
        assert sys.version_info >= (3, 10), \
            f"SI-06 FAIL: Python {sys.version} < 3.10"
