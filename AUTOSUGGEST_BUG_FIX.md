# Auto-Suggest Returns 0% Bug - Complete Fix

## Problem
Auto-suggest sometimes returns 0% for all categories, resulting in zero blocks being classified.

## Root Causes Identified

### 1. Coordinate System Mismatch
- Blocks and drillholes in different coordinate systems
- Results in invalid distance calculations

### 2. Empty/Invalid Distance Arrays
- No valid drillhole coordinates
- Distance computation returns all NaN/inf

### 3. UI Update Exceptions
- Malformed suggestions dict
- KeyError/TypeError during slider updates
- Sliders remain at previous values (possibly 0)

### 4. Silent Failures
- No validation of computed distances
- No feedback when suggestion quality is poor

---

## Complete Fix

### Fix 1: Add Distance Validation

**File:** `block_model_viewer/models/jorc_classification_engine.py`
**Location:** After line 1676 in `suggest_thresholds_from_distances()`

```python
def suggest_thresholds_from_distances(
    dist_to_1st: np.ndarray,
    dist_to_2nd: np.ndarray,
    dist_to_3rd: np.ndarray,
    target_measured: float = 0.10,
    target_indicated: float = 0.35,
    target_inferred: float = 0.80,
) -> Dict[str, Dict[str, Any]]:
    """Suggest classification thresholds based on distance distributions."""

    # === NEW: Validate input arrays ===
    if dist_to_1st is None or len(dist_to_1st) == 0:
        logger.error("Auto-suggest FAILED: dist_to_1st is None or empty")
        return {
            "measured": {"dist_pct": 25, "min_holes": 3},
            "indicated": {"dist_pct": 60, "min_holes": 2},
            "inferred": {"dist_pct": 100, "min_holes": 1},
            "diagnostics": {"error": "Empty distance arrays - check coordinate systems"}
        }

    # Get finite values only
    finite1 = dist_to_1st[np.isfinite(dist_to_1st)]
    finite2 = dist_to_2nd[np.isfinite(dist_to_2nd)]
    finite3 = dist_to_3rd[np.isfinite(dist_to_3rd)]

    # === NEW: Validate finite arrays have reasonable values ===
    if len(finite1) > 0:
        median1 = np.median(finite1)
        if median1 > 100.0:  # More than 100× variogram range!
            logger.error(
                f"Auto-suggest FAILED: Distances are HUGE (median={median1:.1f} × range). "
                f"This indicates coordinate system mismatch between blocks and drillholes. "
                f"Check that both use the same coordinate system (UTM or Local)."
            )
            return {
                "measured": {"dist_pct": 25, "min_holes": 3},
                "indicated": {"dist_pct": 60, "min_holes": 2},
                "inferred": {"dist_pct": 100, "min_holes": 1},
                "diagnostics": {
                    "error": "COORDINATE MISMATCH DETECTED",
                    "median_distance": float(median1),
                    "hint": "Blocks and drillholes may be in different coordinate systems"
                }
            }

    # Handle edge cases
    if len(finite1) == 0:
        logger.warning("No finite dist_to_1st values, using defaults")
        return {
            "measured": {"dist_pct": 25, "min_holes": 3},
            "indicated": {"dist_pct": 60, "min_holes": 2},
            "inferred": {"dist_pct": 100, "min_holes": 1},
            "diagnostics": {"error": "No finite distance values - no drillholes found"}
        }

    # === NEW: Warn if too few samples for reliable statistics ===
    if len(finite1) < 100:
        logger.warning(
            f"Auto-suggest: Only {len(finite1)} valid distance samples. "
            f"Results may be unreliable. Recommend >1000 samples for stable percentiles."
        )

    # Rest of existing code...
    # (Continue with quantile calculations as before)
```

---

### Fix 2: Robust UI Update with Validation

**File:** `block_model_viewer/ui/jorc_classification_panel.py`
**Location:** Replace lines 1602-1614 with:

```python
# Get suggested thresholds
suggestions = suggest_thresholds_from_distances(dist_1st, dist_2nd, dist_3rd)

# === NEW: Validate suggestions before applying ===
if suggestions is None:
    raise ValueError("suggest_thresholds_from_distances returned None")

# Check for error diagnostics
if "diagnostics" in suggestions and "error" in suggestions["diagnostics"]:
    error_msg = suggestions["diagnostics"]["error"]
    hint = suggestions["diagnostics"].get("hint", "")
    median_dist = suggestions["diagnostics"].get("median_distance")

    if median_dist and median_dist > 100.0:
        # Coordinate mismatch detected
        QMessageBox.critical(
            self,
            "Coordinate System Mismatch",
            f"⚠️ CRITICAL: Blocks and drillholes appear to be in different coordinate systems!\n\n"
            f"Median distance: {median_dist:.1f} × variogram range\n"
            f"(This is impossibly large - indicates ~{median_dist * self.spin_maj.value():.0f}m separation)\n\n"
            f"Possible causes:\n"
            f"• Blocks in UTM coordinates (500,000m) but drillholes in Local (0m)\n"
            f"• Blocks in Local but drillholes in UTM\n"
            f"• Incorrect variogram range (too small)\n\n"
            f"Fix:\n"
            f"1. Check coordinate columns in both datasets\n"
            f"2. Ensure both use same coordinate system\n"
            f"3. Use coordinate transformation if needed\n\n"
            f"Using default thresholds for now (25%, 60%, 100%)."
        )
    else:
        # Other error (e.g., empty arrays)
        QMessageBox.warning(
            self,
            "Auto-Suggest Failed",
            f"Could not compute reliable thresholds:\n\n{error_msg}\n\n{hint}\n\n"
            f"Using default thresholds (25%, 60%, 100%)."
        )

    # Fall through to apply defaults (suggestions already has defaults)

# === NEW: Validate suggestion values ===
for cat_name in ["measured", "indicated", "inferred"]:
    if cat_name not in suggestions:
        raise ValueError(f"Missing category '{cat_name}' in suggestions")
    if "dist_pct" not in suggestions[cat_name]:
        raise ValueError(f"Missing 'dist_pct' for category '{cat_name}'")
    if "min_holes" not in suggestions[cat_name]:
        raise ValueError(f"Missing 'min_holes' for category '{cat_name}'")

    # Validate range
    dist_pct = suggestions[cat_name]["dist_pct"]
    if not isinstance(dist_pct, (int, float)) or dist_pct < 0 or dist_pct > 500:
        logger.error(f"Invalid dist_pct for {cat_name}: {dist_pct}")
        suggestions[cat_name]["dist_pct"] = 25 if cat_name == "measured" else (60 if cat_name == "indicated" else 100)

# Apply to UI - set all sliders with suggested values
logger.info(f"Applying auto-suggestions: "
           f"Measured={suggestions['measured']['dist_pct']}%, "
           f"Indicated={suggestions['indicated']['dist_pct']}%, "
           f"Inferred={suggestions['inferred']['dist_pct']}%")

self.cards["Measured"].dist_slider.slider.setValue(suggestions["measured"]["dist_pct"])
self.cards["Measured"].holes_spin.setValue(suggestions["measured"]["min_holes"])

self.cards["Indicated"].dist_slider.slider.setValue(suggestions["indicated"]["dist_pct"])
self.cards["Indicated"].holes_spin.setValue(suggestions["indicated"]["min_holes"])

self.cards["Inferred"].dist_slider.slider.setValue(suggestions["inferred"]["dist_pct"])
self.cards["Inferred"].holes_spin.setValue(suggestions["inferred"]["min_holes"])

# Force UI repaint BEFORE showing message box
QApplication.processEvents()

# Show diagnostics (only if no errors)
diag = suggestions.get("diagnostics", {})
if not diag.get("error"):
    # ... existing success message code ...
```

---

### Fix 3: Add Pre-Flight Validation

**File:** `block_model_viewer/ui/jorc_classification_panel.py`
**Location:** After line 1548 in `_on_suggest_thresholds()`

```python
def _on_suggest_thresholds(self):
    """Analyze drillhole spacing and suggest optimal classification thresholds."""
    if self.drillhole_data is None or self.block_model_data is None:
        QMessageBox.warning(self, "No Data", "Load drillhole and block model data first.")
        return

    # === NEW: Validate data quality BEFORE running expensive computation ===

    # Check drillhole count
    n_holes = len(self.drillhole_data)
    if 'HOLEID' in self.drillhole_data.columns:
        n_unique_holes = len(self.drillhole_data['HOLEID'].unique())
    else:
        n_unique_holes = n_holes

    if n_unique_holes < 5:
        QMessageBox.warning(
            self,
            "Insufficient Drillholes",
            f"Auto-suggest requires at least 5 unique drillholes.\n\n"
            f"Found: {n_unique_holes} drillholes\n\n"
            f"Add more drillhole data or manually set thresholds."
        )
        return

    # Check coordinate columns exist
    required_dh_cols = ['X', 'Y', 'Z']
    missing_dh = [c for c in required_dh_cols if c not in self.drillhole_data.columns]
    if missing_dh:
        QMessageBox.critical(
            self,
            "Missing Coordinates",
            f"Drillhole data missing coordinate columns: {missing_dh}\n\n"
            f"Available columns: {list(self.drillhole_data.columns)[:20]}\n\n"
            f"Ensure drillholes have X, Y, Z coordinates."
        )
        return

    # Check blocks have coordinates
    required_bm_cols = ['XC', 'YC', 'ZC']
    if not all(c in self.block_model_data.columns for c in required_bm_cols):
        # Try alternate column names
        alt_cols = [['X', 'Y', 'Z'], ['x', 'y', 'z'], ['XCENTER', 'YCENTER', 'ZCENTER']]
        found = False
        for alt in alt_cols:
            if all(c in self.block_model_data.columns for c in alt):
                found = True
                break

        if not found:
            QMessageBox.critical(
                self,
                "Missing Coordinates",
                f"Block model missing coordinate columns.\n\n"
                f"Expected: XC/YC/ZC or X/Y/Z\n"
                f"Available: {list(self.block_model_data.columns)[:20]}\n\n"
                f"Ensure blocks have centroid coordinates."
            )
            return

    # Check for NaN coordinates
    dh_coords_valid = self.drillhole_data[['X', 'Y', 'Z']].notna().all(axis=1).sum()
    if dh_coords_valid < n_unique_holes * 0.5:
        QMessageBox.warning(
            self,
            "Invalid Coordinates",
            f"More than 50% of drillhole samples have NaN coordinates!\n\n"
            f"Valid samples: {dh_coords_valid} / {len(self.drillhole_data)}\n\n"
            f"Clean your drillhole data before running auto-suggest."
        )
        return

    # Check variogram range is reasonable
    var_range = self.spin_maj.value()
    if var_range < 1.0:
        QMessageBox.warning(
            self,
            "Invalid Variogram",
            f"Variogram major range is too small: {var_range}m\n\n"
            f"Set a realistic variogram range before auto-suggesting thresholds."
        )
        return

    # Warn if range seems unrealistically large
    if var_range > 10000:
        reply = QMessageBox.question(
            self,
            "Large Variogram Range",
            f"Variogram major range is very large: {var_range:,.0f}m\n\n"
            f"This seems unusually large. Typical ranges are 50-500m.\n\n"
            f"Continue anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

    # If all checks pass, proceed with existing code...
    self.btn_suggest.setEnabled(False)
    # ... rest of existing code ...
```

---

## Testing

After applying fixes, test these scenarios:

### Test 1: Coordinate Mismatch
1. Load blocks in UTM (500,000m)
2. Load drillholes in Local (0m)
3. Click Auto-Suggest
4. **Expected**: Error dialog about coordinate mismatch, defaults applied

### Test 2: Too Few Drillholes
1. Load only 2-3 drillholes
2. Click Auto-Suggest
3. **Expected**: Warning about insufficient drillholes, operation cancelled

### Test 3: Missing Coordinates
1. Load drillholes without X/Y/Z columns
2. Click Auto-Suggest
3. **Expected**: Error about missing coordinates, operation cancelled

### Test 4: Normal Case
1. Load properly formatted drillholes (10+ holes)
2. Load properly formatted blocks (same coordinate system)
3. Set reasonable variogram range (50-200m)
4. Click Auto-Suggest
5. **Expected**: Thresholds applied (non-zero values), diagnostic popup shows coverage

---

## Summary

The bug occurs when:
1. **Coordinate mismatch** → huge distances → invalid percentiles
2. **Empty arrays** → no valid data → defaults (not zeros, but could fail in UI)
3. **UI exceptions** → sliders never updated → remain at previous values

The fix adds:
- ✅ Input validation before computation
- ✅ Distance sanity checks (detect coordinate mismatches)
- ✅ Robust error handling with user feedback
- ✅ Graceful degradation (use defaults on failure)
- ✅ Detailed diagnostic messages

**Result**: User always gets meaningful feedback and valid thresholds, never mysterious zeros!
