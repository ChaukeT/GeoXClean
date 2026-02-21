# Layout Legend Sync Fix - 2026-02-16

## Problem Summary

**Issue:** Layout legends show different properties/values than the main viewer legend. The two legends are not synchronized.

**Visual Symptoms:**
- Layout legend shows values 0 to 1 with generic colormap
- Main viewer legend shows "AL2O3_PCT" with values 0 to 25.65 and "turbo" colormap
- When exporting layouts, the legend doesn't match the actual data being visualized

## Root Cause

The `LegendManager.get_state()` method only captured **UI positioning information**, not the actual **legend content**.

### What Was Captured Before (Incomplete)
- anchor, position, size, margin
- visibility, orientation, background color

### What Was Missing (Critical)
- property name (e.g., "AL2O3_PCT")
- colormap name (e.g., "turbo", "viridis")
- vmin/vmax (value range like 0-25.65)
- categories (for discrete legends)
- category colors

## Technical Details

The `LegendManager` class ([legend_manager.py](block_model_viewer/ui/legend_manager.py)) maintains two separate data structures:

1. **`_state` (LegendState)**: UI positioning only
2. **`_current_payload` (LegendPayload)**: Actual content (property, colormap, values)

The `_state_as_dict()` method only serialized #1, ignoring #2.

## The Fix

**File Modified:** [block_model_viewer/ui/legend_manager.py](block_model_viewer/ui/legend_manager.py#L1677-L1714)

Extended `_state_as_dict()` to include current payload data:

```python
def _state_as_dict(self) -> Dict[str, Any]:
    """Get complete legend state including both UI positioning and content."""
    state = {
        "anchor": self._state.anchor,
        "position": list(self._state.position),
        "size": list(self._state.size),
        "margin": int(self._state.margin),
        "visible": bool(self._visible),
        "orientation": self._state.orientation,
        "background_rgba": list(self._state.background_rgba),
    }

    # Include current payload data (property, colormap, values, categories)
    if self._current_payload is not None:
        payload = self._current_payload
        state["property"] = payload.property
        state["title"] = payload.title
        state["mode"] = payload.mode
        state["colormap"] = payload.colormap

        if payload.vmin is not None:
            state["vmin"] = float(payload.vmin)
        if payload.vmax is not None:
            state["vmax"] = float(payload.vmax)

        state["log_scale"] = payload.log_scale
        state["reverse"] = payload.reverse

        if payload.categories:
            state["categories"] = payload.categories
        if payload.category_colors:
            state["category_colors"] = {
                str(k): list(v) for k, v in payload.category_colors.items()
            }

    return state
```

## How Layouts Capture Legend State

When creating a layout or clicking "Capture View":

1. **MainWindow Quick Export** ([main_window.py:2245](block_model_viewer/ui/main_window.py#L2245)):
   ```python
   viewport.legend_state = renderer.legend_manager.get_state()
   legend = LegendItem(legend_state=viewport.legend_state)
   ```

2. **Layout Window Capture View** ([layout_window.py:586,618](block_model_viewer/ui/layout/layout_window.py#L586)):
   ```python
   legend_state = legend_mgr.get_state()
   # Update all legend items
   item.legend_state = legend_state
   ```

## How Layout Renderer Uses The State

The layout renderer ([layout_renderer.py:255-375](block_model_viewer/layout/layout_renderer.py#L255-L375)) already reads these fields:

```python
def _render_legend(self, painter, item, width, height, dpi):
    legend_state = item.legend_state or {}
    title = legend_state.get("property", "Legend")  # ✅ Now captured
    categories = legend_state.get("categories") or []  # ✅ Now captured

    if categories:
        self._render_discrete_legend(...)  # Uses category_colors ✅
    else:
        self._render_continuous_legend(...)  # Uses vmin, vmax, colormap ✅

def _render_continuous_legend(...):
    vmin = legend_state.get("vmin", 0.0)  # ✅ Now provided
    vmax = legend_state.get("vmax", 1.0)  # ✅ Now provided
    cmap_name = legend_state.get("colormap", "viridis")  # ✅ Now provided
```

**The renderer was already looking for these fields - they just weren't being provided before!**

## Testing Instructions

### Test 1: Quick Export (Main Window)
1. Load a block model and visualize a property (e.g., "AL2O3_PCT")
2. Set colormap to "turbo"
3. File → Export → Quick Layout Export
4. Check the exported PDF/PNG - the legend should show:
   - Title: "AL2O3_PCT"
   - Colormap: turbo gradient (red-yellow-cyan-blue)
   - Values: actual data range (e.g., 0-25.65)

### Test 2: Layout Composer
1. Load a block model and visualize a property
2. Open Layout Composer (View → Layout Composer)
3. Add a Legend item to the layout
4. Click "Capture View" button
5. The layout legend should now match the main viewer legend exactly:
   - Same property name
   - Same colormap
   - Same value range

### Test 3: Discrete/Categorical Legends
1. Visualize lithology or a categorical property
2. Export to layout
3. The layout legend should show:
   - All categories with correct labels
   - Correct colors for each category

### Test 4: Layout Save/Load
1. Create a layout with captured legend
2. Save the layout (.geox_layout file)
3. Close and reopen the layout
4. Legend should still show the correct property/colormap/values

## Expected Results

✅ **Before Fix:**
- Layout legend: generic (0-1, viridis)
- Main legend: "AL2O3_PCT" (0-25.65, turbo)
- ❌ Not synchronized

✅ **After Fix:**
- Layout legend: "AL2O3_PCT" (0-25.65, turbo)
- Main legend: "AL2O3_PCT" (0-25.65, turbo)
- ✅ Perfectly synchronized

## Impact

- All layout exports (PDF, PNG, TIFF) now have correct legends
- Saved layouts preserve the exact legend state
- No more manual legend editing after export
- Professional, audit-ready outputs for JORC/SAMREC compliance

## Related Files

- [legend_manager.py](block_model_viewer/ui/legend_manager.py) - State capture (FIXED)
- [layout_renderer.py](block_model_viewer/layout/layout_renderer.py) - State rendering (already correct)
- [layout_window.py](block_model_viewer/ui/layout/layout_window.py) - Capture View functionality
- [main_window.py](block_model_viewer/ui/main_window.py) - Quick Export functionality
- [layout_document.py](block_model_viewer/layout/layout_document.py) - LegendItem definition

## Backward Compatibility

✅ **Fully backward compatible**
- Old layouts without legend content will still render (using defaults)
- New layouts capture full state
- No migration needed
