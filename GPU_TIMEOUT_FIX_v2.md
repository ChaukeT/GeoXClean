# GPU Timeout (TDR) Fix - Version 2
## Date: 2026-02-13

## Problem
GPU driver timeout occurring when resizing window with SGSIM block models rendered alongside drillholes.

**Key Observation:** Issue appears when SGSIM grid is created on fine drillhole spacing, potentially creating **hundreds of thousands of cells** (vs coarse block model with ~30k cells).

---

## Root Causes

### 1. Rendering Re-enabled Too Early
- `_on_resize_complete()` was calling `EnableRenderOn()` BEFORE checking model size
- This allowed VTK to auto-render from other events even though we skipped manual `render()` call
- For very large models, ANY render during/after resize causes GPU timeout

### 2. No Protection Against Extremely Large Grids
- SGSIM grids on drillhole spacing can have 200k-500k+ cells
- No warning to user before visualizing these
- 30k threshold too low for detecting "extreme" cases

### 3. Total Scene Complexity Not Considered
- Checking individual layer size wasn't enough
- drillholes (50k samples) + SGSIM grid (200k cells) = 250k total
- GPU overwhelmed by combined scene complexity

---

## Fixes Applied

### Fix 1: Delayed Render Re-enable for Large Models
**File:** `viewer_widget.py:_on_resize_complete()` (line ~250)

**Change:**
- Check `_has_large_model` flag FIRST
- For small models: Enable rendering + render immediately (old behavior)
- For large models:
  - DON'T call `EnableRenderOn()` immediately
  - Start 2-second delay timer before re-enabling
  - Prevents automatic render triggers during critical post-resize period

**Code:**
```python
# Check model size BEFORE re-enabling rendering
has_large_model = getattr(self.renderer, '_has_large_model', False)

if not has_large_model:
    # Normal: re-enable and render
    interactor.EnableRenderOn()
    self.plotter.render()
else:
    # Large: delay re-enable by 2 seconds
    logger.debug("GPU-SAFE: Keeping render disabled after resize")
    self._delayed_render_timer.start(2000)
```

### Fix 2: Warn About Extremely Large Grids
**File:** `main_window.py:visualize_sgsim_results()` (line ~7354)

**Thresholds:**
- **200k cells:** Warning (slow, may timeout)
- **500k cells:** Critical warning with confirmation (very likely to timeout)

**Message:**
- Suggests creating SGSIM on coarser grid
- Recommends <100k cells for stable visualization
- Allows proceeding but warns of risks

### Fix 3: Total Scene Complexity Logging
**File:** `viewer_widget.py:resizeEvent()` (line ~242)

**Change:**
- Calculate total cells across all active layers during resize
- Log total complexity for debugging
- Helps identify when combined scene is too large

**Log Output:**
```
GPU-SAFE: Resize triggered with 2 layers, 250,000 total cells
```

---

## Testing

### How to Test
1. Load drillholes (with composites)
2. Create SGSIM on drillhole grid (fine spacing)
3. Visualize mean/variance
4. Try resizing window (maximize/restore)

### Expected Behavior
- **Small grids (<30k):** Resize renders immediately (smooth)
- **Large grids (30k-200k):** Resize skips render, 2-second delay before re-enable
- **Very large (200k-500k):** Warning message before visualization
- **Extreme (>500k):** Critical warning requiring confirmation

### Check Logs For
```
DEBUG - GPU-SAFE: Resize triggered with X layers, XXX,XXX total cells
DEBUG - GPU-SAFE: Keeping render disabled for large model after resize
DEBUG - GPU-SAFE: Re-enabled rendering after resize delay
```

---

## Additional Recommendations

### 1. Reduce SGSIM Grid Size
Instead of using drillhole sample spacing (1-2m), use:
- Composite spacing as minimum (e.g., 5-10m)
- Target grid: 50x50x20 cells = 50,000 total (safe)
- Avoid: 200x200x50 cells = 2,000,000 total (extreme!)

### 2. Visualize Subsets
For very large SGSIM results:
- Extract and visualize slices (single Z level)
- Use threshold filters to show only high-grade blocks
- Export to VTK and visualize in ParaView for full resolution

### 3. Monitor GPU Usage
- Task Manager → Performance → GPU
- Watch for "3D" and "GPU Memory" during resize
- If either spikes to 100%, grid is too large

---

## Files Modified
- `viewer_widget.py`: Delayed render re-enable for large models
- `main_window.py`: Grid size warnings before visualization
- `main_window.py`: Camera reset threshold aligned (50k → 30k)

## Memory Updated
- `MEMORY.md`: GPU Timeout section updated with new fix details

---

## If Issue Persists

If you still get driver timeout even with these fixes:

1. **Check grid size in logs:**
   ```
   RECEIVED SGSIM visualization request: property='...', grid has XXX,XXX cells
   ```

2. **If >200k cells:** Grid is TOO LARGE for real-time visualization
   - Create SGSIM on coarser grid (increase cell size in SGSIM panel)
   - Or visualize only a subset/slice

3. **Check total scene complexity:**
   ```
   GPU-SAFE: Resize triggered with X layers, XXX,XXX total cells
   ```
   - If >300k total: Disable some layers before resizing
   - Or simplify drillhole display (fewer properties/samples)

4. **AMD Driver Settings:**
   - AMD Radeon Software → Gaming → Graphics
   - Disable "Radeon Anti-Lag"
   - Disable "Radeon Boost"
   - Set "Texture Filtering Quality" to "High Quality"
   - Increase TDR timeout (advanced - requires registry edit)

---

## Key Takeaway

**The fix prevents timeout for "large" models (30k-200k cells), but cannot overcome physics for "extreme" models (>500k cells).**

If you have 500k+ cells, the solution is NOT better GPU timeout protection - it's **reducing the grid size** to something the GPU can handle for real-time interaction.
