# Simple Kriging Debug System - User Guide

## Overview

A comprehensive debugging system has been implemented to track and diagnose Simple Kriging failures that cause software crashes. This system provides detailed execution logging, error tracking, and diagnostics to identify exactly where and why failures occur.

## How to Use

### Method 1: UI Debug Mode (Recommended)

1. **Open Simple Kriging Panel** in GeoX
2. **Check the "Debug Mode" checkbox** (orange text, located with the action buttons)
3. **Configure your kriging parameters** as normal
4. **Click "RUN ESTIMATION"**

When debug mode is enabled:
- Comprehensive logging is written to `sk_debug_log.txt` in your working directory
- Every stage of execution is tracked with timestamps
- All errors are caught and logged with full tracebacks
- Progress updates are recorded
- Input validation is performed upfront

### Method 2: Programmatic Debug Mode

```python
from block_model_viewer.controllers.geostats_controller import GeostatsController

params = {
    "data": your_dataframe,
    "variable": "Au",
    "debug_mode": True,  # Enable debug mode
    "grid_spec": {...},
    "parameters": {...}
}

controller.run_simple_kriging(params, callback=..., progress_callback=...)
```

## Debug Log Location

The debug log is written to:
- **File**: `sk_debug_log.txt`
- **Location**: Current working directory (where you launched GeoX)
- **Format**: Plain text with timestamps and structured information

## What Gets Logged

### 1. **Execution Stages**
Every major stage of execution with elapsed time:
```
[0.00s] Starting Simple Kriging Debug Session
[0.02s] Controller: _prepare_simple_kriging_payload called
[0.05s] Data Validation
[0.10s] Extracting coordinates and values
[0.25s] Building estimation grid
[1.50s] CALLING simple_kriging_3d - CRITICAL EXECUTION POINT
[45.30s] simple_kriging_3d COMPLETED SUCCESSFULLY
```

### 2. **Input Validation**
Comprehensive validation before execution starts:
- Data presence and size
- Variable existence in dataset
- Grid specification completeness
- Parameter value validation (numerics, ranges, types)
- Missing required fields

### 3. **Data Details**
```
Data cleaned
  cleaned_rows: 1523
  dropped_rows: 47
  has_attrs: True

Coordinates extracted
  n_samples: 1523
  coords_shape: (1523, 3)
  values_min: 0.05
  values_max: 25.3
  values_mean: 2.47
```

### 4. **Grid Information**
```
Grid arrays created
  nx: 100, ny: 150, nz: 50
  total_points: 750000
  x_range: [100.0, 1100.0]
  y_range: [200.0, 1700.0]
  z_range: [-150.0, 350.0]
  memory_mb: 17.1
```

### 5. **Kriging Parameters**
```
SKParameters created
  global_mean: 2.5
  variogram_type: spherical
  range_major: 150.0
  ndmax: 12
```

### 6. **Progress Tracking**
All progress callbacks are logged:
```
[5.23s] Progress 10%: Starting Simple Kriging...
[8.45s] Progress 25%: Estimating block 1 of 4...
[12.67s] Progress 50%: Estimating block 2 of 4...
```

### 7. **Error Details** (if failure occurs)
Full exception information:
```
============================================================
ERROR CAUGHT AT STAGE: simple_kriging_3d execution
Error Type: ValueError
Error Message: Covariance matrix is singular

Full Traceback:
Traceback (most recent call last):
  File "...", line XXX, in _prepare_simple_kriging_payload
    estimates, variances, neighbour_counts, diagnostics = simple_kriging_3d(...)
  File "...", line YYY, in simple_kriging_3d
    ...
ValueError: Covariance matrix is singular
============================================================
```

### 8. **Results Summary** (if successful)
```
simple_kriging_3d COMPLETED SUCCESSFULLY
  estimates_shape: (750000,)
  n_nan: 45230
  estimates_min: 0.12
  estimates_max: 18.45
  estimates_mean: 2.51
```

## Common Failure Scenarios

### Scenario 1: Crash During Grid Creation
**Symptoms**: Software closes after "Building estimation grid"
**Debug Log Shows**:
```
[0.25s] Building estimation grid
[ERROR] memory allocation failed
```
**Solution**: Reduce grid dimensions (nx, ny, nz)

### Scenario 2: Crash During Kriging
**Symptoms**: Software closes after "CALLING simple_kriging_3d"
**Debug Log Shows**:
```
[1.50s] CALLING simple_kriging_3d - CRITICAL EXECUTION POINT
[ERROR] Numba compilation failed
```
**Solution**: Check variogram parameters, ensure Numba is installed

### Scenario 3: Invalid Input Data
**Symptoms**: Immediate crash or error
**Debug Log Shows**:
```
Validation FAILED
  issues:
    - Missing coordinate column: Z
    - Variable 'Au' not in data columns
```
**Solution**: Fix data issues before running

### Scenario 4: Singular Matrix Error
**Symptoms**: Kriging starts but fails partway through
**Debug Log Shows**:
```
[ERROR] Covariance matrix is singular
solver_status: FAILED for 1523 blocks
```
**Solution**: Add nugget effect, increase search radius, or check for duplicate samples

## Interpreting the Debug Log

### Timeline Analysis
1. **Look at elapsed times** - identify where delays occur
2. **Check last successful stage** - shows exactly where failure happened
3. **Review error section** - detailed traceback of what went wrong

### Memory Issues
If the log shows:
```
Grid coordinates created
  memory_mb: 2500.5
```
And then crashes, you likely have a memory problem. Reduce grid size.

### Data Issues
If validation fails or data stats look wrong:
```
values_min: -999.0
values_max: 999999.0
```
This indicates data quality issues (flags, outliers, etc.)

## Debug Output Summary

At the end of every debug session, a summary is written:
```
============================================================
DEBUG SESSION SUMMARY
============================================================
Total stages: 15
Error caught: True/False
Last progress: 75%
Total execution time: 45.3s

Stage Timeline:
  [0.00s] Starting Simple Kriging Debug Session
  [0.02s] Controller: _prepare_simple_kriging_payload called
  ...
  [45.30s] simple_kriging_3d COMPLETED SUCCESSFULLY
============================================================
```

## Performance Considerations

**Debug mode adds minimal overhead**:
- File I/O is buffered and non-blocking
- Logging happens asynchronously
- Typical overhead: < 2% execution time
- Safe to use in production for troubleshooting

**When to use debug mode**:
- ✅ When kriging fails or crashes unexpectedly
- ✅ When testing new variogram parameters
- ✅ When working with new/unfamiliar datasets
- ✅ For performance profiling (see stage timings)
- ❌ Not needed for routine production runs (unless issues arise)

## Troubleshooting Tips

1. **Always check the debug log first** before asking for help
2. **Look for the ERROR section** - this contains the root cause
3. **Check validation failures** - input issues are common
4. **Review parameter values** - ensure they're reasonable for your data
5. **Compare stage timings** - identify bottlenecks

## Advanced: Custom Debug Wrapper

For custom integrations, you can use the debugger directly:

```python
from block_model_viewer.geostats.sk_debugger import SimpleKrigingDebugger

debugger = SimpleKrigingDebugger("my_custom_log.txt")

try:
    debugger.log_stage("My custom stage")
    # Your code here
    result = some_function()
    debugger.log_stage("Function completed", {'result_type': type(result).__name__})
    
except Exception as e:
    debugger.log_error(e, "My custom stage")
    raise
finally:
    debugger.finalize()
```

## Files Created

When debug mode is enabled, the following file is created:
- `sk_debug_log.txt` - Main debug log (overwritten each run)

Location: Current working directory (typically the GeoX installation folder or where you launched the app)

## Getting Help

If Simple Kriging fails:
1. Enable debug mode and re-run
2. Open `sk_debug_log.txt`
3. Find the ERROR section
4. Copy the error details and stage timeline
5. Contact support with this information

The debug log provides all necessary information to diagnose and fix issues.

## Example Debug Log

Here's an abbreviated example of a successful run:

```
=== SIMPLE KRIGING DEBUG LOG ===
Started: 2026-02-08 21:30:00
============================================================

[0.00s] Starting Simple Kriging Debug Session
[0.01s] Debug mode activated by user
[0.02s] Controller: _prepare_simple_kriging_payload called

[0.03s] Input Validation
[0.04s] Validation PASSED

[0.05s] Starting data validation
[0.06s] Data source validation passed

[0.08s] Cleaning data
  original_rows: 1570
  variable: Au
  columns: ['HoleID', 'X', 'Y', 'Z', 'Au', 'Ag', 'Cu']

[0.10s] Data cleaned
  cleaned_rows: 1523
  dropped_rows: 47

[0.12s] Extracting coordinates and values
[0.15s] Coordinates extracted
  n_samples: 1523
  coords_shape: (1523, 3)
  values_min: 0.05
  values_max: 25.3
  values_mean: 2.47

[0.18s] Building estimation grid
[0.25s] Grid arrays created
  nx: 100, ny: 150, nz: 50
  total_points: 750000

[0.40s] Grid coordinates created
  grid_coords_shape: (750000, 3)
  memory_mb: 17.14

[0.45s] SKParameters created
  global_mean: 2.5
  variogram_type: spherical
  range_major: 150.0

[0.50s] CALLING simple_kriging_3d - CRITICAL EXECUTION POINT
  n_data_samples: 1523
  n_grid_points: 750000

[0.55s] Progress 10%: Starting Simple Kriging...
[5.23s] Progress 25%: Estimating block 1 of 4...
[12.45s] Progress 50%: Estimating block 2 of 4...
[20.12s] Progress 75%: Estimating block 3 of 4...
[27.89s] Progress 100%: Complete

[28.15s] simple_kriging_3d COMPLETED SUCCESSFULLY
  estimates_shape: (750000,)
  n_nan: 45230
  estimates_min: 0.12
  estimates_max: 18.45

[28.30s] Payload packaged successfully
[28.31s] Simple Kriging execution COMPLETED

============================================================
DEBUG SESSION SUMMARY
============================================================
Total stages: 18
Error caught: False
Last progress: 100%
Total execution time: 28.31s

Stage Timeline:
  [0.00s] Starting Simple Kriging Debug Session
  [0.50s] CALLING simple_kriging_3d - CRITICAL EXECUTION POINT
  [28.15s] simple_kriging_3d COMPLETED SUCCESSFULLY
  [28.31s] Simple Kriging execution COMPLETED
============================================================
End: 2026-02-08 21:30:28
```

## Summary

The Simple Kriging Debug System provides:
- ✅ **Comprehensive execution tracking** - know exactly what's happening
- ✅ **Detailed error logging** - understand why failures occur
- ✅ **Input validation** - catch problems before execution
- ✅ **Performance profiling** - identify bottlenecks
- ✅ **Easy to use** - just check a box in the UI

Enable it whenever you encounter issues, and the debug log will guide you to the solution.
