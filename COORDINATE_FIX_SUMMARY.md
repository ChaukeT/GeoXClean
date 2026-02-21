# Geological Model Coordinate System Fix

## The Problem

The BIF surfaces built by LoopStructural were not honouring drillhole constraints. Drillholes would visually "slice through" geological unit boundaries instead of terminating on them. This made the surfaces "mathematical artefacts" rather than true geological surfaces.

### Root Cause Analysis

The investigation revealed a critical gap in the data pipeline:

1. **Drillhole Visualization**: The renderer correctly calculated 3D coordinates using Minimum Curvature desurveying via `minimum_curvature_path_from_surveys()` in `drillhole_layer.py`

2. **Geological Modeling**: LoopStructural expected constraint data with X, Y, Z columns, but:
   - `registry_utils.py` → `_populate_assays()` only extracted `hole_id`, `depth_from`, `depth_to`, and grades - **no X, Y, Z**
   - `loopstructural_panel.py` required data to already have X, Y, Z but never calculated them
   - `ensure_xyz_columns()` only **renamed** existing columns, didn't calculate coordinates

3. **The Unused Function**: `add_coordinates_to_intervals()` in `desurvey.py` was the correct function to calculate 3D coordinates using Minimum Curvature, but **it was never called anywhere in the codebase**.

### Result
The geological model received data without proper 3D coordinates. The surface was built in a completely different coordinate space than the rendered drillholes, causing them to appear to pass through unit boundaries.

## The Fix

### Files Modified

1. **`block_model_viewer/core/data_registry_simple.py`**
   - Added `_ensure_interval_coordinates()` method
   - Modified `register_drillhole_data()` to call coordinate calculation after validation
   - Added helper methods `_detect_holeid_column()` and `_detect_column()`

2. **`block_model_viewer/utils/desurvey.py`**
   - Fixed `_apply_vertical_coords()` to use proper column names instead of hardcoded 'FROM'/'TO'

### How It Works

When drillhole data is registered:

```
register_drillhole_data()
    ↓
_validate_drillholes()          # Validate data structure
    ↓
_ensure_interval_coordinates()  # NEW: Calculate X, Y, Z using Minimum Curvature
    ↓
_assign_interval_ids()          # Assign GPU picking IDs
    ↓
Store in registry
```

The `_ensure_interval_coordinates()` method:
1. Checks each interval DataFrame (assays, lithology, composites, intervals)
2. If X, Y, Z are missing or all zeros, calls `add_coordinates_to_intervals()`
3. Uses Minimum Curvature desurveying for deviated holes
4. Falls back to vertical assumption if no survey data
5. Logs coordinate ranges for verification

### Coordinate Calculation

For each interval, the 3D midpoint is calculated:
- **With Survey Data**: Minimum Curvature algorithm interpolates position along the desurveyed path
- **Without Survey**: Vertical assumption: X = collar_X, Y = collar_Y, Z = collar_Z - midpoint_depth

## Verification

After loading drillhole data, check the log output:

```
INFO - Calculating 3D coordinates for 15000 assays intervals using Minimum Curvature...
INFO -   assays coordinate ranges: X=(450123.5, 451890.2), Y=(6789012.3, 6790456.7), Z=(450.2, 1250.8)
```

To verify the fix works:
1. Load drillhole data
2. Build a geological surface with LoopStructural
3. Confirm that drillhole unit boundaries align with the modeled surface
4. BIF intercepts should terminate ON the BIF surface, not pass through it

## Technical Notes

### Why Coordinates Were Missing

The original workflow assumed:
1. Assays would have X, Y, Z from some upstream process
2. `ensure_xyz_columns()` would normalize column names
3. LoopStructural would receive properly positioned data

But in reality:
1. Raw assays only have HOLEID, FROM, TO, and grade values
2. No code path actually calculated 3D coordinates for assays
3. The desurveying was only applied during visualization, not data registration

### Coordinate Space Consistency

Now both:
- **Drillhole visualization** (via `drillhole_layer.py`)
- **Geological modeling constraints** (via `data_registry_simple.py`)

Use the **same** Minimum Curvature algorithm from `desurvey.py`, ensuring:
- Surfaces are constrained by actual drillhole positions
- Rendered drillholes and model constraints are in the same coordinate space
- Deviated holes are handled correctly

## Files Touched

| File | Change |
|------|--------|
| `core/data_registry_simple.py` | Added coordinate calculation during registration |
| `utils/desurvey.py` | Fixed column name handling in `_apply_vertical_coords()` |

## Date

2024-01-30
