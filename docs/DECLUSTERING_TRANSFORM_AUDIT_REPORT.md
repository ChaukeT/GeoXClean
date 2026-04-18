# DECLUSTERING & GRADE TRANSFORMATION — CORRECTNESS AUDIT

## Files Reviewed
- `declustering.py` (1,199 lines) — Cell-based declustering engine
- `declustering_panel.py` (1,638 lines) — Qt UI for declustering
- `transform.py` (514 lines) — Normal Score Transformer (forward/backward)
- `grade_transformation_panel.py` (862 lines) — Qt UI for grade transformations

---

## CRITICAL BUGS

### BUG-D01: Sensitivity analysis uses wrong stability metric
**File:** declustering.py lines 943–956
**Severity:** CRITICAL — recommends wrong cell size
**Affects:** Multi-cell-size sensitivity analysis, `get_recommended_cell_size()`

**The problem:**
```python
weight_change = abs(summary.mean_weight - prev_mean_weight)
stability_achieved = weight_change < stability_threshold  # 0.001
```

The standard cell-size sensitivity methodology (Deutsch, 1989; Supervisor/Datamine/Leapfrog) tracks the **declustered mean grade** across cell sizes. The optimal cell size is where the declustered mean stops changing. This code tracks `mean_weight` instead, which equals `occupied_cells / total_samples` — a purely geometric quantity that says nothing about grade bias correction.

`mean_weight` can stabilize at a small cell size (because the grid quickly saturates), while the declustered grade mean is still drifting significantly. The recommendation will be a cell size that is too small, under-correcting for clustering bias.

The correct data is already computed and available in `summary.variable_summaries[var]['mean_declust']` but is never used for stability determination.

**Impact:** `get_recommended_cell_size()` returns the wrong cell size. Users relying on the automated recommendation get inadequate bias correction, leading to biased resource estimates.

**Fix:** Track the maximum absolute change in declustered mean across all grade variables. Stability = max grade change < threshold.

---

### BUG-D02: Forward transform collapses tied values to single Gaussian value
**File:** transform.py lines 258–268
**Severity:** CRITICAL — violates SGSIM normality assumption
**Affects:** All Normal Score transforms on data with tied grade values

**The problem:**
```python
f = interp1d(
    self.raw_data_sorted,      # ← contains duplicate x-values when grades are tied
    self.gaussian_data_sorted,
    kind='linear',
    ...
)
```

`fit()` carefully assigns unique Gaussian values to tied raw values via deterministic tie-breaking (TRF-003). But `transform()` builds an `interp1d` from `raw_data_sorted` → `gaussian_data_sorted`. When `raw_data_sorted` has duplicates (e.g., 50 samples all at grade=1.0), `interp1d` picks an arbitrary mapping (typically the last duplicate's y-value). ALL tied values then map to the **same** Gaussian value.

This creates a spike at one Gaussian value, violating the normality assumption that SGSIM requires. The deterministic tie-breaking from `fit()` is completely lost in the forward transform.

**Impact:** In datasets with significant ties (common with rounded grades, indicator data, or detection limits), the Gaussian distribution has spikes instead of being smooth. SGSIM quality degrades, and back-transformed realizations have artifacts.

**Fix:** Use `np.searchsorted` on `raw_data_sorted` to find each value's position in the sorted array, then index into `gaussian_data_sorted` directly. Interpolate only between unique values for values not exactly in the training set.

---

## HIGH-SEVERITY BUGS

### BUG-D03: Box-Cox/Log shift value not stored — back-transform impossible
**File:** grade_transformation_panel.py lines 472–549
**Severity:** HIGH — data loss for inverse transform
**Affects:** Any Box-Cox or Log transform with "Add Constant" checked

When a user applies a shift before transformation:
```python
valid = valid + shift_value   # line 477
...
result[pos_mask] = stats.boxcox(raw[pos_mask] + shift_value, lmbda=l_val)  # line 545
```

The metadata stores `boxcox_lambda` but **not** `shift_value` or `shift_applied`. To correctly back-transform: `raw = inv_boxcox(transformed) - shift`. Without the shift, any back-transform will be systematically offset by `shift_value`.

For Log transforms, the same issue: `log(raw + shift)` → to invert you need `exp(transformed) - shift`, but shift is lost.

**Fix:** Store `shift_value` and `shift_applied` in `meta` dict.

---

### BUG-D04: Declustering panel has unreachable duplicate `except` block
**File:** declustering_panel.py lines 1432–1438
**Severity:** MEDIUM (code smell, not runtime error)

```python
    except Exception as e:
        logger.error(f"Cell weights export failed: {e}")
        QMessageBox.critical(self, "Export Failed", f"Failed to export cell weights: {e}")

    except Exception as e:   # ← UNREACHABLE — first except catches everything
        logger.error(f"Export failed: {e}")
        QMessageBox.critical(self, "Export Failed", f"Failed to export results: {e}")
```

Python silently ignores the second `except` block since the first catches all exceptions. Likely a copy-paste artifact.

---

### BUG-D05: `_execute_multi_cell_analysis` forces 3D cells regardless of config
**File:** declustering_panel.py lines 810–818
**Severity:** HIGH — wrong dimensionality in sensitivity analysis

```python
def _execute_multi_cell_analysis(self, df, cell_sizes):
    cell_specs = [(size, size, size) for size in cell_sizes]  # ← always 3D tuple
    results = self.engine.analyze_cell_sizes(df, cell_specs, ...)
```

This always creates 3D prismatic cell specs `(sx, sy, sz)` regardless of the user's 2D/3D checkbox setting. If the user configured 2D declustering (no Z), the sensitivity analysis still runs in 3D with tiny Z cells, producing completely different weights than the single-run result.

**Fix:** Check `self.is_3d_checkbox.isChecked()` and create 2D specs `(size, size)` when 2D is selected.

---

### BUG-D06: `analyze_cell_sizes` doesn't pass value_cols through
**File:** declustering_panel.py line 816, declustering.py line 933
**Severity:** MEDIUM — sensitivity analysis computes weights but no grade statistics

In the panel:
```python
results = self.engine.analyze_cell_sizes(df, cell_specs, value_cols=None)
```

And in the engine:
```python
df_result, summary = temp_engine.compute_weights(df, x_col, y_col, z_col, value_cols)
```

Since `value_cols=None`, `compute_weights` generates weights but the summary's `variable_summaries` dict is populated with ALL numeric columns (including coordinates, IDs, etc.). This means the grade delta shown in the multi-cell table may use a coordinate column instead of a grade column. More importantly, even with BUG-D01 fixed, the stability check would use non-grade columns.

**Fix:** Pass the user's selected grade variable as `value_cols` to `analyze_cell_sizes`.

---

## MEDIUM BUGS

### BUG-D07: `_on_data_loaded` stores direct reference to registry data
**File:** grade_transformation_panel.py line 286
**Severity:** MEDIUM — mutation risk

```python
self.registry_snapshot = data  # Keep full ref — NOT a copy
```

If the registry mutates this dict after the panel stores it, the panel's reference becomes stale or corrupted. The `_send_to_registry` method later does `self.registry_snapshot.copy()` (shallow), but the snapshot itself is a live reference.

**Fix:** Deep-copy the relevant DataFrames on receipt, or at minimum do `self.registry_snapshot = dict(data)`.

---

### BUG-D08: Log transform check order catches "Log10" in "Log" branch
**File:** grade_transformation_panel.py lines 484–505
**Severity:** LOW (mitigated by check order)

```python
if "Log10" in method:     # Checks first — correct
    ...
elif "Log" in method:     # "Log10" would also match this, but elif prevents it
    ...
```

The current check order happens to be correct ("Log10" checked before "Log"). But this is fragile — if someone reorders the conditions, Log10 silently falls through to natural log. Using exact equality `method == "Log"` would be safer.

---

## FIX PRIORITY ORDER

| Priority | Bug  | Impact                                        | Effort |
|----------|------|-----------------------------------------------|--------|
| 1        | D01  | Wrong cell size recommendation                | Medium |
| 2        | D02  | Forward transform collapses ties              | Medium |
| 3        | D05  | Sensitivity analysis uses wrong dimensionality | Low    |
| 4        | D03  | Shift value lost for back-transform           | Low    |
| 5        | D06  | value_cols not passed in sensitivity           | Low    |
| 6        | D04  | Unreachable except block                       | Low    |
| 7        | D07  | Registry snapshot mutation risk                | Low    |

---

## NOTES — WHAT'S DONE WELL

- Cell declustering math is correct: `weight = 1/n_k` per cell, vectorized with integer encoding
- Weighted CDF in NST uses proper Hazen-like midpoint formula
- Deterministic tie-breaking via `lexsort` in `fit()` is textbook correct
- PCHIP for back-transform is a good choice (monotonic, smooth)
- Lineage enforcement at both UI and engine layers is thorough
- Validation gates with override+audit logging match JORC/SAMREC requirements
- Coordinate column detection with flexible case-insensitive matching is robust
