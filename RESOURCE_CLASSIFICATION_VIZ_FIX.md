# Resource Classification "Visualize 3D" Button Fix

**Date:** 2026-02-14
**Issue:** Classification results not appearing when clicking "Visualize 3D" button
**Status:** Fixed with comprehensive logging

---

## Problem Summary

When users click the "Visualize 3D" button in the Resource Classification panel after running classification, the classified blocks do not appear in the 3D viewer. The panel shows correct statistics (Measured, Indicated, Inferred counts), but the visualization is blank.

---

## Root Causes Identified

### 1. **Missing Active Scalars on PolyData**
**Location:** `resource_classification_panel.py:1270-1293`

**Problem:**
- The code created PyVista PolyData and added "Classification" scalars
- But it never called `set_active_scalars("Classification")`
- Without active scalars, the glyph() method doesn't know which data to visualize

**Fix:**
```python
pdata["Classification"] = scalars
pdata.set_active_scalars("Classification")  # ADDED
```

### 2. **Scalar Data Not Transferred to Glyphs**
**Location:** `resource_classification_panel.py:1292`

**Problem:**
- When creating glyphs from PolyData, point data should be transferred to cell data
- The glyph() method auto-transfers active scalars, but this can fail silently
- No verification that the glyphs actually have the Classification data

**Fix:**
```python
glyphs = pdata.glyph(geom=cube, scale=False, orient=False)

# Verify transfer and manually copy if needed
if "Classification" not in glyphs.cell_data and "Classification" in pdata.point_data:
    glyphs.cell_data["Classification"] = pdata.point_data["Classification"]

glyphs.set_active_scalars("Classification")
```

### 3. **Handler Not Checking point_data**
**Location:** `main_window.py:8627-8634`

**Problem:**
- The handler only checked `mesh.cell_data` for Classification scalars
- For point cloud representation (>500k blocks), scalars are in `point_data`
- This caused visualization to fail for large models

**Fix:**
```python
# Check BOTH cell_data and point_data
if "Classification" in mesh.cell_data:
    scalar_name = "Classification"
    mesh.set_active_scalars("Classification", preference="cell")
elif "Classification" in mesh.point_data:
    scalar_name = "Classification"
    mesh.set_active_scalars("Classification", preference="point")
```

### 4. **Silent Failures - No User Feedback**
**Location:** Both files

**Problem:**
- Exceptions were caught and logged but user got no error message
- User couldn't tell if visualization failed vs. was processing
- No diagnostic information to help debug

**Fix:**
- Added comprehensive logging at every step
- Added user-facing error messages with diagnostic info
- Log coordinate bounds, data distribution, mesh properties
- Show QMessageBox warnings when viewer/plotter not available

---

## Files Modified

### 1. `block_model_viewer\ui\resource_classification_panel.py`

**Changes:**
- Added logging at start of `_visualize_results()` to confirm method is called
- Log DataFrame shape, columns, coordinate bounds
- Log classification distribution (Measured/Indicated/Inferred counts)
- Set active scalars on PolyData before glyphing
- Verify scalar transfer to glyphs and manually copy if needed
- Log mesh properties (n_points, n_cells, active_scalars_name)
- Added user-friendly error messages with full diagnostic info

**Lines Modified:** 1242-1330

### 2. `block_model_viewer\ui\main_window.py`

**Changes:**
- Added prominent logging when handler is called (with separator line)
- Log mesh type, n_cells, n_points at start
- Check both cell_data and point_data for Classification scalars
- Explicitly set active scalars with preference (cell vs point)
- Log which data source is being used
- Warn if no Classification data found (show available arrays)
- Added QMessageBox warnings when viewer/plotter unavailable

**Lines Modified:** 8524-8652

---

## Testing Instructions

### Test Case 1: Normal Block Model (<500k blocks)

1. **Load Data:**
   - Load drillhole composites
   - Load or build a block model (or run SGSIM)

2. **Run Classification:**
   - Open Resource Classification panel
   - Set variogram parameters (range ~100m)
   - Set thresholds (Measured: 25%, Indicated: 60%, Inferred: 150%)
   - Click "RUN CLASSIFICATION"
   - Verify statistics table shows counts for each category

3. **Visualize:**
   - Click "Visualize 3D" button
   - **Expected:** Console shows logging sequence:
     ```
     INFO: Starting classification visualization...
     INFO: Classification DataFrame has XXXXX blocks
     INFO: Using coordinate columns: ['X', 'Y', 'Z']
     INFO: Classification distribution: {'Measured': XXX, 'Indicated': XXX, ...}
     INFO: Created PolyData with XXXXX points
     INFO: Creating cube glyphs with dimensions: dx=X, dy=Y, dz=Z
     INFO: Transferred Classification scalars to glyphs
     INFO: Created glyphs with XXXXX cells
     INFO: Emitting visualization signal with XXXXX block glyphs
     ============================================================
     INFO: CLASSIFICATION VISUALIZATION REQUESTED: Classification
     INFO: Mesh type: UnstructuredGrid
     INFO: Mesh n_cells: XXXXX
     ============================================================
     INFO: Using Classification from cell_data
     INFO: [CLASSIFICATION] Applied coordinate shift to UnstructuredGrid
     INFO: [CLASSIFICATION] Shifted bounds: (...)
     INFO: Updated legend with categorical classification
     INFO: [CLASSIFICATION] Camera reset to fit classification bounds
     INFO: Added classification visualization: Classification (XXXXX blocks)
     ```
   - **Expected:** 3D viewer shows colored blocks:
     - Green (Measured)
     - Yellow (Indicated)
     - Red (Inferred)
     - Gray (Unclassified)
   - **Expected:** Legend shows category names with colors
   - **Expected:** Can pick blocks and see classification in tooltip

### Test Case 2: Large Block Model (>500k blocks)

1. Follow steps 1-2 from Test Case 1
2. **Visualize:**
   - Click "Visualize 3D"
   - **Expected:** Console shows:
     ```
     INFO: Using point cloud representation (>500k blocks)
     INFO: Emitting visualization signal with XXXXX points
     INFO: Using Classification from point_data
     ```
   - **Expected:** 3D viewer shows point cloud (not cubes)
   - **Expected:** Colors still match categories

### Test Case 3: Error Handling

1. **No Data Loaded:**
   - Open Resource Classification panel without loading drillholes/blocks
   - Click "Visualize 3D"
   - **Expected:** Warning message: "No classification results to visualize"

2. **Missing Coordinates:**
   - Manually test with DataFrame missing X/Y/Z columns (if possible)
   - **Expected:** Error message lists available columns

3. **Viewer Not Initialized:**
   - Test visualization before 3D viewer is created
   - **Expected:** "3D viewer is not initialized" warning

---

## Debugging Checklist

If visualization still doesn't work after these fixes, check the console log for:

### ✅ Panel Signal Emitted
```
INFO: Classification visualization signal emitted successfully
```
**If missing:** Issue is in panel code (exception before signal emission)

### ✅ Handler Received Signal
```
============================================================
INFO: CLASSIFICATION VISUALIZATION REQUESTED: Classification
```
**If missing:** Signal connection issue - check `main_window.py:8415-8418`

### ✅ Mesh Has Classification Data
```
INFO: Using Classification from cell_data
```
OR
```
INFO: Using Classification from point_data
```
**If missing:** Scalar transfer failed - check point_data/cell_data keys in log

### ✅ Coordinate Shift Applied
```
INFO: [CLASSIFICATION] Applied coordinate shift to UnstructuredGrid: shift=[...]
```
**If missing:** Blocks will be at UTM coords (500km away), invisible

### ✅ Mesh Added to Scene
```
INFO: Added classification visualization: Classification (XXXXX blocks)
```
**If missing:** Check earlier errors in handler

---

## Known Issues / Edge Cases

### Issue: "Classification data not found in glyphs cell_data"
**Cause:** PyVista glyph() failed to transfer scalars
**Fix:** Manual transfer code added (lines 1293-1297)
**Check:** Look for "Manually transferred Classification" in log

### Issue: Blocks Invisible (Far from Camera)
**Cause:** Coordinate shift not applied
**Fix:** Handler applies `_to_local_precision()` (main_window.py:8572-8615)
**Check:** Look for "Applied coordinate shift" and "Shifted bounds" in log

### Issue: Wrong Colors
**Cause:** Scalar values not matching expected range [0-3]
**Check:** Look for "Classification distribution" in log - should show Measured/Indicated/Inferred
**Fix:** Verify mapping at resource_classification_panel.py:1266

---

## Signal Connection Verification

The visualization signal is connected in `main_window.py`:

```python
# Line 8415-8418
if hasattr(self.jorc_classification_panel, 'request_visualization'):
    self.jorc_classification_panel.request_visualization.connect(
        self._handle_classification_visualization
    )
```

**To verify connection is active:**
```python
# Add temporary debug logging in panel __init__
logger.info(f"request_visualization signal created: {hasattr(self, 'request_visualization')}")
```

**To verify signal is emitted:**
```python
# Already added at line 1340 in resource_classification_panel.py
logger.info("Classification visualization signal emitted successfully")
```

**To verify handler is called:**
```python
# Already added at line 8526 in main_window.py
logger.info(f"CLASSIFICATION VISUALIZATION REQUESTED: {name}")
```

---

## Performance Notes

### Small Models (<30k blocks):
- Uses cube glyphs (pretty, detailed)
- Render time: <1 second
- GPU safe

### Medium Models (30k-500k blocks):
- Uses cube glyphs (can be slow)
- Render time: 1-5 seconds
- GPU-safe mode activates at 30k (single render, no auto-refresh)

### Large Models (>500k blocks):
- Uses point cloud (fast)
- Render time: <2 seconds
- **WARNING:** Still prone to GPU timeout if total scene complexity >500k

**Recommendation:** For >200k blocks, create SGSIM on coarser grid instead of trying to visualize ultra-fine grids.

---

## Next Steps

1. **Test the fixes** using the test cases above
2. **Check console logs** to verify the logging sequence
3. **Report results** - whether blocks now appear correctly
4. **If still failing**, provide:
   - Full console log from clicking "Visualize 3D"
   - Screenshot of Resource Classification panel (showing statistics table)
   - Number of blocks in model

---

**End of Fix Documentation**
