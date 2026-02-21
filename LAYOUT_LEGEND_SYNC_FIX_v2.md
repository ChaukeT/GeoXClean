# Layout Legend Sync Fix v2 - ACTUAL ROOT CAUSE - 2026-02-16

## The Real Problem (Found via Debug Logging)

The debug logs revealed the **actual root cause**:

```
[STATE CAPTURE] No current payload available - legend content will be missing!
[QUICK EXPORT] Captured legend state: {'anchor': 'top_right', 'position': [1212, 47], ...}
```

**`_current_payload` was None** even though the legend had been updated with AL2O3_PCT!

## Root Cause Analysis

The legend update methods were **updating the widget but not storing the payload**:

### What Was Happening

1. ✅ `update_continuous("AL2O3_PCT", data, "turbo")` was called
2. ✅ The legend widget was updated correctly → user sees correct legend
3. ❌ `_current_payload` was **never set** → export captures empty state
4. ❌ When exporting, `get_state()` returns only UI positioning, no content

### The Bug

Looking at `update_continuous()` (and other update methods):

```python
def update_continuous(self, property_name, data, cmap_name, ...):
    # ... calculate vmin, vmax ...

    # Update the widget (user sees this)
    self.widget.set_continuous(
        title=property_name,
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        ...
    )

    # Store for tracking
    self._current_data = array_data
    self._current_property = property_name

    # ❌ BUG: _current_payload is NEVER SET!
    # (It's only set when widget is None, which never happens)
```

## The Complete Fix

**File Modified:** [legend_manager.py](block_model_viewer/ui/legend_manager.py)

Added `_current_payload` creation to **all 6 legend update methods**:

### 1. `update_continuous()` (line ~498)
```python
# Store current payload for state capture (layout exports, etc.)
final_cmap_name = cmap_name
if hasattr(cmap_name, 'name'):
    final_cmap_name = cmap_name.name
    if isinstance(final_cmap_name, str) and final_cmap_name.endswith('_custom'):
        final_cmap_name = final_cmap_name[:-7]
elif not isinstance(cmap_name, str):
    final_cmap_name = str(cmap_name)

self._current_payload = LegendPayload(
    layer=None,
    property=property_name,
    title=property_name,
    mode="continuous",
    colormap=final_cmap_name if isinstance(final_cmap_name, str) else 'viridis',
    data=array_data,
    log_scale=log_scale,
    subtitle=subtitle,
    vmin=vmin,
    vmax=vmax,
)
```

### 2. `update_discrete()` (line ~638)
```python
# Store current payload for state capture
final_cmap_name = cmap_name
if hasattr(cmap_name, 'name'):
    final_cmap_name = cmap_name.name
elif cmap_name is None:
    final_cmap_name = "tab20"
elif not isinstance(cmap_name, str):
    final_cmap_name = str(cmap_name)

self._current_payload = LegendPayload(
    layer=None,
    property=property_name,
    title=property_name,
    mode="discrete",
    colormap=final_cmap_name if isinstance(final_cmap_name, str) else 'tab20',
    categories=list(categories),
    category_colors=category_colors or {},
    subtitle=subtitle,
)
```

### 3. `set_continuous()` (line ~1015)
```python
# Store as LegendPayload for state capture
self._current_payload = LegendPayload(
    layer=None,
    property=field,
    title=field,
    mode="continuous",
    colormap=self._colormap or "viridis",
    vmin=self._vmin,
    vmax=self._vmax,
)
```

### 4. `set_discrete_bins()` (line ~1049)
```python
# Store as LegendPayload for state capture
category_colors = {}
for cat_info in self._categories:
    label = cat_info.get("label", "")
    colour = cat_info.get("colour", (0.5, 0.5, 0.5))
    if len(colour) == 3:
        colour = (*colour, 1.0)
    category_colors[label] = colour

self._current_payload = LegendPayload(
    layer=None,
    property=field,
    title=field,
    mode="discrete",
    colormap="custom",
    categories=[cat["label"] for cat in self._categories],
    category_colors=category_colors,
)
```

### 5. `set_lithology_lut()` (line ~1068)
```python
# Store as LegendPayload for state capture
category_colors = {}
for cat_info in self._categories:
    label = cat_info.get("label", "")
    colour = cat_info.get("colour", (0.5, 0.5, 0.5))
    if len(colour) == 3:
        colour = (*colour, 1.0)
    category_colors[label] = colour

self._current_payload = LegendPayload(
    layer=None,
    property=field,
    title=field,
    mode="discrete",
    colormap="lithology",
    categories=[cat["label"] for cat in self._categories],
    category_colors=category_colors,
)
```

### 6. `set_custom_lut()` (line ~1095)
```python
# Store as LegendPayload for state capture
category_colors = {}
for cat_info in self._categories:
    label = cat_info.get("label", "")
    colour = cat_info.get("colour", (0.5, 0.5, 0.5))
    if len(colour) == 3:
        colour = (*colour, 1.0)
    category_colors[label] = colour

self._current_payload = LegendPayload(
    layer=None,
    property=field,
    title=field,
    mode="discrete",
    colormap="custom",
    categories=[cat["label"] for cat in self._categories],
    category_colors=category_colors,
)
```

## Impact

Now when ANY legend update method is called:
1. ✅ The widget is updated (user sees correct legend)
2. ✅ `_current_payload` is stored with complete data
3. ✅ `get_state()` captures the payload (via `_state_as_dict()`)
4. ✅ Layout exports have complete legend information

## Testing

Try the export again:

1. Visualize AL2O3_PCT with turbo colormap
2. Export to layout (File → Export → Quick Layout Export)
3. Check the logs for:
   ```
   [STATE CAPTURE] Included payload: property=AL2O3_PCT, colormap=turbo, vmin=0.0, vmax=25.65
   [QUICK EXPORT] Captured legend state: {...'property': 'AL2O3_PCT', 'colormap': 'turbo', 'vmin': 0.0, 'vmax': 25.65...}
   ```
4. Open the PDF - the legend should show AL2O3_PCT with turbo colormap and correct value range

## Debug Logging Added

For troubleshooting, debug logging was added:

**In `_state_as_dict()` (line ~1714):**
```python
if self._current_payload is not None:
    # ... include payload data ...
    logger.info(f"[STATE CAPTURE] Included payload: property={payload.property}, ...")
else:
    logger.warning("[STATE CAPTURE] No current payload available - legend content will be missing!")
```

**In `quick_layout_export()` (main_window.py:2245):**
```python
viewport.legend_state = renderer.legend_manager.get_state()
logger.info(f"[QUICK EXPORT] Captured legend state: {viewport.legend_state}")
```

## Summary

- **v1 Fix**: Extended `_state_as_dict()` to include payload data ✅ (Correct but insufficient)
- **v2 Fix**: Made sure `_current_payload` is actually SET by all update methods ✅ (This was the real bug!)

Both fixes were necessary:
1. `_state_as_dict()` needed to know HOW to serialize the payload
2. The update methods needed to actually CREATE and STORE the payload

Now the layout legends will perfectly match the main viewer legend!
