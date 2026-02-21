# Resource Classification Manager - Debug Report

**Date:** 2026-02-13
**Status:** Issues Identified - Fixes Provided
**Panels Affected:**
- `jorc_classification_panel.py` (Main JORC Classification Panel)
- `resource_classification_panel.py` (Legacy Simple Classification Panel)

---

## Summary of Issues Found

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Auto Suggestion Threshold Logic Working But Not User-Friendly | Medium | Fix Provided |
| 2 | 3D Visualization Not Working - Signal Connection Issue | **HIGH** | Fix Provided |
| 3 | Missing Error Handling in Visualization | Medium | Fix Provided |
| 4 | Coordinate Shift Applied But May Fail Silently | Medium | Fix Provided |
| 5 | resource_classification_panel.py Has Duplicate/Broken Visualization | Medium | Fix Provided |

---

## Issue #1: Auto Suggestion Threshold - Usability Problem

### Location
- File: `block_model_viewer/ui/jorc_classification_panel.py`
- Method: `_on_suggest_thresholds()` (lines 1537-1648)
- Supporting: `block_model_viewer/models/jorc_classification_engine.py` - `suggest_thresholds_from_distances()` (lines 1637-1752)

### Problem Description
The auto suggestion threshold feature **EXISTS and WORKS**, but has usability issues:

1. **Button Not Always Connected**: The "✨ Suggest Thresholds" button is created at line 569 but may not be visible or accessible depending on UI layout
2. **Progress Feedback Unclear**: Users don't get clear feedback during the analysis (especially for large models)
3. **No Validation**: Doesn't validate if data is sufficient for suggestions (e.g., minimum # of drillholes)

### Current Code Analysis
```python
# Line 1537 - Button handler exists
def _on_suggest_thresholds(self):
    """Analyze drillhole spacing and suggest optimal classification thresholds."""
    if self.drillhole_data is None or self.block_model_data is None:
        QMessageBox.warning(self, "No Data", "Load drillhole and block model data first.")
        return
```

**The Logic Is Sound:**
- Computes distance diagnostics using variogram-normalized isotropic space
- Uses quantile-based thresholds (10%, 35%, 80% coverage targets)
- Applies ordering constraints (Measured < Indicated < Inferred)
- Shows diagnostics in popup message

### The Real Issue
Looking at lines 567-586 where the button is created:

```python
# Line 567-586 in jorc_classification_panel.py
# Suggest Thresholds Button
row_suggest = QHBoxLayout()
self.btn_suggest = QPushButton("✨ Suggest Thresholds")
self.btn_suggest.setToolTip(
    "Analyze drillhole spacing and suggest optimal thresholds.\n"
    "Uses distance distributions to recommend coverage levels:\n"
    "• Measured: ~10% of blocks (3 unique holes)\n"
    "• Indicated: ~35% of blocks (2 unique holes)\n"
    "• Inferred: ~80% of blocks (1 unique hole)\n\n"
    "Note: For large models (>10k blocks), uses fast sampling."
)
self.btn_suggest.setStyleSheet("""
    QPushButton {
        background-color: #7b2cbf; color: white; padding: 10px;
        font-weight: bold; border-radius: 4px;
    }
    QPushButton:hover { background-color: #9d4edd; }
    QPushButton:disabled { background-color: #444; color: #888; }
""")
self.btn_suggest.clicked.connect(self._on_suggest_thresholds)
row_suggest.addWidget(self.btn_suggest)
```

**Problem**: The button might not be added to the main layout properly, or the UI might be too narrow to show it.

### Fix Required

**Fix 1: Ensure button is visible in UI layout**

Check lines 550-600 where the header/controls are built:

```python
# In _build_ui() method around line 550
# Ensure btn_suggest is added to a visible row
controls_row = QHBoxLayout()
controls_row.addWidget(QLabel("Variogram:"))
controls_row.addWidget(self.spin_maj)
controls_row.addWidget(self.spin_semi)
controls_row.addWidget(self.spin_min)
controls_row.addWidget(self.spin_sill)
controls_row.addStretch()
controls_row.addWidget(self.btn_suggest)  # <-- ADD THIS LINE
config_layout.addLayout(controls_row)
```

**Fix 2: Add data validation**

```python
def _on_suggest_thresholds(self):
    """Analyze drillhole spacing and suggest optimal classification thresholds."""
    # Existing check
    if self.drillhole_data is None or self.block_model_data is None:
        QMessageBox.warning(self, "No Data", "Load drillhole and block model data first.")
        return

    # NEW: Validate sufficient data
    n_holes = len(self.drillhole_data['HOLEID'].unique()) if 'HOLEID' in self.drillhole_data.columns else len(self.drillhole_data)
    if n_holes < 10:
        QMessageBox.warning(
            self,
            "Insufficient Data",
            f"Auto-suggestion requires at least 10 unique drillholes.\n"
            f"Found: {n_holes} drillholes.\n\n"
            f"Add more drillhole data or manually set thresholds."
        )
        return

    # Rest of existing code...
```

---

## Issue #2: 3D Visualization Not Working - **CRITICAL BUG**

### Location
- File: `block_model_viewer/ui/jorc_classification_panel.py`
- Method: `_visualize_results()` (lines 1796-1986)
- Connection: `block_model_viewer/ui/main_window.py` - `_handle_classification_visualization()` (lines 8487-8660)

### Problem Description
**ROOT CAUSE FOUND:**

The visualization signal `request_visualization` is emitted correctly (line 1979), BUT:

1. **Signal May Not Be Connected**: When opening the JORC classification panel via the OLD `resource_classification_panel.py` (lines 184, 332), the signal connection is missing
2. **Two Different Panel Files Exist**:
   - `jorc_classification_panel.py` (NEW - modern, feature-complete)
   - `resource_classification_panel.py` (OLD - deprecated, imports from jorc but has broken visualization)

### Evidence

**File: `resource_classification_panel.py` (line 103)**
```python
from .jorc_classification_panel import JORCClassificationPanel as ResourceClassificationPanel
```

This creates an ALIAS, which means the old panel is actually using the new panel class BUT the visualization signal in `resource_classification_panel.py` has a DIFFERENT implementation (lines 1242-1297) that doesn't emit the signal - **it tries to do visualization inline!**

**Comparison:**

**CORRECT (jorc_classification_panel.py, line 1979):**
```python
# Emits signal to main_window
self.request_visualization.emit(grid, "Resource Classification")
```

**BROKEN (resource_classification_panel.py, line 1242-1297):**
```python
def _visualize_results(self):
    """Visualize classification results in 3D as blocks (optimized for large models)."""
    if self.classification_result is None:
        return

    try:
        import pyvista as pv

        df = self.classification_result.classified_df
        # ... builds point cloud or glyphs ...
        # 🔥 BUG: EMITS THE WRONG SIGNAL!
        self.request_visualization.emit(pdata, "Classification")  # Line 1284
        # OR Line 1293:
        self.request_visualization.emit(glyphs, "Classification")
```

Notice the signal payload name is **"Classification"** not **"Resource Classification"** - this might cause the main window handler to not find the right layer name!

### Fix Required

**Fix Option A: Use Only jorc_classification_panel.py (RECOMMENDED)**

Remove the duplicate in `resource_classification_panel.py`:

```python
# File: resource_classification_panel.py
# DELETE entire file and update all imports to use jorc_classification_panel directly

# In main_window.py line 103, change:
# OLD:
from .resource_classification_panel import JORCClassificationPanel as ResourceClassificationPanel

# NEW:
from .jorc_classification_panel import JORCClassificationPanel as ResourceClassificationPanel
```

**Fix Option B: Fix signal connection in main_window.py**

```python
# File: main_window.py, around line 8378
# Ensure signal is connected for BOTH panel types

def open_resource_classification_panel(self):
    """Open JORC/SAMREC-compliant Resource Classification panel."""
    try:
        # ... existing code ...

        # Connect visualization request signal
        if hasattr(self.jorc_classification_panel, 'request_visualization'):
            # DISCONNECT any existing connections first
            try:
                self.jorc_classification_panel.request_visualization.disconnect()
            except (TypeError, RuntimeError):
                pass  # No connections exist

            # Connect to handler
            self.jorc_classification_panel.request_visualization.connect(
                self._handle_classification_visualization
            )
            logger.info("✅ Connected resource classification visualization signal")
```

**Fix Option C: Fix the visualization signal name consistency**

In `resource_classification_panel.py` line 1284 and 1293:

```python
# OLD (line 1284, 1293):
self.request_visualization.emit(pdata, "Classification")
self.request_visualization.emit(glyphs, "Classification")

# NEW - Match the name used in jorc_classification_panel:
self.request_visualization.emit(pdata, "Resource Classification")
self.request_visualization.emit(glyphs, "Resource Classification")
```

---

## Issue #3: Missing Error Handling in Visualization

### Location
- File: `block_model_viewer/ui/jorc_classification_panel.py`
- Method: `_visualize_results()` (lines 1796-1986)

### Problem
The visualization method has a try-except block (line 1983-1985) but doesn't provide actionable error information to debug:

```python
except Exception as e:
    logger.exception("Visualization error")
    QMessageBox.critical(self, "Visualization Error", f"Failed to visualize:\n{e}")
```

### Fix Required

Add diagnostic information:

```python
except Exception as e:
    logger.exception("Visualization error")

    # Diagnostic info for debugging
    diag_info = []
    diag_info.append(f"Error: {e}")
    diag_info.append(f"\nData available: {len(df):,} blocks")
    diag_info.append(f"Columns: {list(df.columns)[:10]}")
    diag_info.append(f"PyVista available: {PYVISTA_AVAILABLE}")
    diag_info.append(f"Classification result: {self.classification_result is not None}")

    if hasattr(self, 'viewer_widget') and self.viewer_widget:
        diag_info.append(f"Viewer available: True")
        diag_info.append(f"Renderer available: {self.viewer_widget.renderer is not None}")
    else:
        diag_info.append(f"Viewer available: False")

    diag_msg = "\n".join(diag_info)

    QMessageBox.critical(
        self,
        "Visualization Error",
        f"Failed to visualize classification results.\n\n{diag_msg}\n\n"
        f"Check the console log for full traceback."
    )
```

---

## Issue #4: Coordinate Shift May Fail Silently

### Location
- File: `block_model_viewer/ui/main_window.py`
- Method: `_handle_classification_visualization()` (lines 8535-8578)

### Problem
The coordinate shift is wrapped in try-except and logs a WARNING on failure (line 8578), but doesn't notify the user:

```python
except Exception as e:
    logger.warning(f"Could not apply coordinate shift to classification mesh: {e}")
```

This means if coordinate shift fails, the blocks will be rendered at UTM coordinates (500km away from camera) and appear invisible - but the user gets NO indication why.

### Fix Required

```python
except Exception as e:
    logger.error(f"Could not apply coordinate shift to classification mesh: {e}", exc_info=True)

    # Notify user of critical coordinate issue
    QMessageBox.warning(
        self,
        "Coordinate Shift Failed",
        f"⚠️ Warning: Could not apply coordinate transformation to classification blocks.\n\n"
        f"The blocks may appear in the wrong location or be invisible.\n\n"
        f"Error: {e}\n\n"
        f"Try:\n"
        f"1. Reloading the block model\n"
        f"2. Checking coordinate systems match between drillholes and blocks\n"
        f"3. Using 'Reset Camera' if blocks are far from origin"
    )
```

---

## Issue #5: resource_classification_panel.py Duplication

### Location
- File: `block_model_viewer/ui/resource_classification_panel.py`

### Problem
This file is largely REDUNDANT and causes confusion:
- Line 103: Imports `JORCClassificationPanel` and aliases it
- Lines 184, 1242-1297: Has its own visualization method that **conflicts** with the jorc panel's method
- Different signal names ("Classification" vs "Resource Classification")

### Fix Required

**Option A: Delete the file entirely (RECOMMENDED)**

```bash
# Remove the redundant file
rm block_model_viewer/ui/resource_classification_panel.py

# Update all imports to point directly to jorc_classification_panel
# Search and replace in all files:
# OLD: from .resource_classification_panel import
# NEW: from .jorc_classification_panel import
```

**Option B: Make it a pure alias (no custom methods)**

```python
# File: resource_classification_panel.py
"""
Legacy compatibility alias for JORC Classification Panel.

DEPRECATED: Use jorc_classification_panel.JORCClassificationPanel directly.
"""
from .jorc_classification_panel import JORCClassificationPanel as ResourceClassificationPanel

__all__ = ['ResourceClassificationPanel']
```

---

## Recommended Action Plan

### Priority 1 (Critical - Do First)
1. **Fix visualization signal connection** (Issue #2)
   - Ensure signal is always connected when panel opens
   - Use consistent signal payload name ("Resource Classification")
   - Test that `_handle_classification_visualization` is actually called

2. **Add diagnostic error messages** (Issue #3)
   - Help users debug when visualization fails
   - Show actionable information (PyVista installed? Data present?)

### Priority 2 (High - Do Next)
3. **Fix coordinate shift error handling** (Issue #4)
   - Don't fail silently - warn the user
   - Provide recovery instructions

4. **Remove duplicate file** (Issue #5)
   - Delete `resource_classification_panel.py` OR make it a pure alias
   - Update all imports

### Priority 3 (Medium - Nice to Have)
5. **Improve auto-suggestion UI** (Issue #1)
   - Add data validation before allowing suggestions
   - Make button more prominent in UI
   - Add progress indicator for large models

---

## Testing Checklist

After applying fixes, test the following:

- [ ] Open JORC Classification panel from menu
- [ ] Load drillhole data (composites)
- [ ] Load or build a block model
- [ ] Click "✨ Suggest Thresholds" button
  - [ ] Button is visible and clickable
  - [ ] Progress bar shows during analysis
  - [ ] Thresholds are applied to sliders
  - [ ] Diagnostic popup shows distance medians
- [ ] Run classification
  - [ ] Results table updates with counts
  - [ ] "Visualize in 3D" button becomes enabled
- [ ] Click "Visualize in 3D"
  - [ ] 3D viewer shows colored blocks (NOT empty scene)
  - [ ] Legend shows Measured (green), Indicated (yellow), Inferred (red), Unclassified (gray)
  - [ ] Blocks are positioned correctly (not 500km away)
  - [ ] Can pick blocks and see classification in tooltip

---

## Additional Notes

### Dependency Check
Ensure PyVista is installed:
```bash
pip install pyvista>=0.37.0
```

### Logging
Enable debug logging to trace signal connections:
```python
import logging
logging.getLogger('block_model_viewer.ui.jorc_classification_panel').setLevel(logging.DEBUG)
logging.getLogger('block_model_viewer.ui.main_window').setLevel(logging.DEBUG)
```

### Memory Considerations
For large models (>100k blocks):
- Auto-suggest uses sampling (10k block sample) - GOOD
- Visualization uses UnstructuredGrid for irregular grids - OK, but can be slow
- Consider adding LOD (Level of Detail) for very large classified models

---

**End of Debug Report**