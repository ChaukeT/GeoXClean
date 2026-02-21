# GeoX Panel Debugger - Quick Start Guide

## Installation

1. Install test dependencies:
```bash
cd c:/Users/chauk/Documents/GeoX_Clean
pip install -r tests/requirements-test.txt
```

## Basic Usage

### Run All Tests
```bash
python -m panel_debugger
```

### Run Critical Tests Only (Recommended First)
```bash
python -m panel_debugger -m critical
```

This will run:
- Silent exception detection (`test_signals.py::TestSilentExceptionDetection`)
- Coordinate alignment tests (`test_coordinates.py::TestCoordinateAlignment`)
- End-to-end drillhole loading (`test_integration.py::TestDrillholeLoadingWorkflow`)

### Run Specific Categories

**Detect Silent Exceptions** (Will find drillhole_control_panel.py:255 and dock_setup.py:129):
```bash
python -m panel_debugger --category signals --verbose
```

**Test Coordinate System Alignment** (Will detect coordinate mismatches):
```bash
python -m panel_debugger --category coordinates --verbose
```

**Test Panel Instantiation** (Will find panels that can't be created):
```bash
python -m panel_debugger --category panel_init
```

**Test Data Flow** (Will find broken registry connections):
```bash
python -m panel_debugger --category data_flow
```

**Test End-to-End Integration** (Will test complete drillhole loading workflow):
```bash
python -m panel_debugger --category integration
```

### Generate HTML Report

```bash
python -m panel_debugger --html tests/panel_debugger/reports/debug_report.html
```

Then open `tests/panel_debugger/reports/debug_report.html` in your browser.

## What Will Be Detected

### CRITICAL Issues (Will Cause Test Failures)

1. **Silent Exceptions** - `test_signals.py`
   - Location: `drillhole_control_panel.py:255`
   - Location: `dock_setup.py:129`
   - Pattern: `except: pass` or `except Exception: pass`

2. **Coordinate System Mismatches** - `test_coordinates.py`
   - Drillholes and block models in different coordinate systems
   - Causes layers to appear 500km apart

3. **Broken Signal Connections** - `test_signals.py`
   - Panels that don't connect to required signals
   - Signal handlers that crash

4. **End-to-End Workflow Failures** - `test_integration.py`
   - Drillhole loading not triggering visualization
   - Data not propagating through registry

### ERROR Issues (Will Cause Test Failures)

1. **Import Errors** - `test_panel_init.py`
   - Missing dependencies
   - Broken import paths

2. **Instantiation Errors** - `test_panel_init.py`
   - Panels that can't be created
   - Missing constructor parameters

### WARNING Issues (Tests Pass But Issues Noted)

1. **Missing Attributes** - `test_panel_init.py`
   - Panels missing `PANEL_ID` or `PANEL_NAME`

2. **No Signal Handlers** - `test_signals.py`
   - Panels that consume signals but have no handler methods

## Expected Output

```
╔══════════════════════════════════════════════════════════════════╗
║  GeoX Panel Debugger v1.0                                        ║
╚══════════════════════════════════════════════════════════════════╝

======================== test session starts =========================
platform win32 -- Python 3.x.x, pytest-7.x.x, pluggy-1.x.x
collected 45 items

tests/panel_debugger/tests/test_panel_init.py ................  [ 35%]
tests/panel_debugger/tests/test_signals.py .......F............ [ 65%]
tests/panel_debugger/tests/test_data_flow.py ..........        [ 85%]
tests/panel_debugger/tests/test_coordinates.py .....           [ 95%]
tests/panel_debugger/tests/test_integration.py .....           [100%]

============================== FAILURES ==============================
__________ TestSilentExceptionDetection.test_no_silent_exceptions[DrillholeControlPanel] __________

CRITICAL: Silent exceptions found in DrillholeControlPanel
File: block_model_viewer.ui.drillhole_control_panel

Line 255: Silent exception detected
Pattern: except\s+Exception\s*:\s*\n\s*pass

Context:
252:             try:
253:                 registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
254:                 if registry.get_drillhole_data():
255:                     self._on_drillhole_data_loaded(registry.get_drillhole_data())
256:             except Exception:
257:                 pass

FIX: Replace 'except: pass' with:
  - Explicit exception logging: logger.error('Error', exc_info=True)
  - Specific exception types: except ImportError as e:
  - Or remove try-except if error should propagate

==================== 1 failed, 44 passed in 12.34s ====================
```

## Next Steps After Running Tests

1. **Fix Critical Issues First**:
   - Remove silent exceptions at `drillhole_control_panel.py:255` and `dock_setup.py:129`
   - Add proper error logging instead of `except: pass`

2. **Fix Coordinate Mismatches**:
   - Ensure all renderer methods call `_to_local_precision()`
   - Verify block models use same coordinate transformation as drillholes

3. **Fix Broken Signal Connections**:
   - Add missing signal connections in panel `__init__` methods
   - Test signal handlers don't crash with valid data

4. **Re-run Tests**:
   - After fixes, re-run to verify all tests pass
   - Use `python -m panel_debugger -m critical` to verify critical issues resolved

## Troubleshooting

**Error: "PyQt6 not available"**
```bash
pip install PyQt6>=6.4.0
```

**Error: "PyVista not available"**
```bash
pip install pyvista>=0.40.0
```

**Error: "No module named 'block_model_viewer'"**
- Make sure you're in the project root directory
- The conftest.py adds the project to Python path automatically

**Tests hang or timeout**
- Check if Qt application is running in background
- Kill any orphaned Python processes
- Re-run with `--verbose` flag

## Advanced Usage

### Run with Coverage Report
```bash
python -m pytest tests/panel_debugger/tests/ --cov=block_model_viewer --cov-report=html
```

### Run Specific Test
```bash
python -m pytest tests/panel_debugger/tests/test_signals.py::TestDrillholeSignalFlow::test_drillhole_control_panel_connects_to_signal -v
```

### Stop on First Failure
```bash
python -m panel_debugger --failfast
```

### Run Tests Matching Pattern
```bash
python -m pytest tests/panel_debugger/tests/ -k "drillhole" -v
```

## Support

For issues or questions:
- Check `tests/panel_debugger/README.md` for detailed documentation
- Review test files in `tests/panel_debugger/tests/` for examples
- See plan at `.claude/plans/happy-purring-hearth.md` for design details
