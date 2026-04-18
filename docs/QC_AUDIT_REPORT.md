# QC VALIDATION PIPELINE — CORRECTNESS AUDIT
## Files reviewed: drillhole_validation.py, drillhole_autofix.py, qc_window.py, drillhole_manual_edit.py, drillhole_ignore.py, drillhole_audit_trail.py, audit_export.py

---

## CRITICAL — Will produce wrong results downstream

### BUG-01: Autofix silently destroys data on large overlaps
**File:** `drillhole_autofix.py` line 416–422
**Severity:** CRITICAL — corrupts assay data

The overlap fix snaps `from_depth` to `prev_to_depth` regardless of overlap size.
If interval A = [0, 100] and interval B = [50, 150], autofix changes B to [100, 150].
The 50m of assay data from 50–100 is silently discarded.

The config has `max_small_overlap = 0.02` but autofix ignores it:
```python
# Current code — fixes ALL overlaps:
if overlap > 0:
    f_new = prev_to_float  # Truncates regardless of size!

# Should be:
if overlap > 0 and overlap <= cfg.max_small_overlap:
    f_new = prev_to_float  # Only fix tiny overlaps
# else: leave for manual review
```

**Impact:** A geologist runs autofix thinking it's "safe", and 50m of grade data vanishes. Compositing and kriging downstream use truncated intervals. Resource estimate is wrong.

---

### BUG-02: Autofix zero/negative length fix is destructive
**File:** `drillhole_autofix.py` line 396–401
**Severity:** CRITICAL — creates phantom intervals

If `to_depth <= from_depth` (e.g., from=100, to=90 — a data entry swap), autofix creates:
```python
t_new = f + cfg.standard_sample_length  # from=100, to=101
```
Instead of investigating whether from/to are swapped (which would give [90, 100] with real data), it creates a phantom 1m interval at [100, 101] with the original assay value. The real 10m interval data is lost.

**Fix:** Zero/negative length should NEVER be auto-fixed. Flag as ERROR for manual review.

---

### BUG-03: No negative assay value detection
**File:** `drillhole_validation.py` — missing entirely
**Severity:** HIGH — invalid grades enter pipeline

The test data shows `au_ppm: -5.0` but validation has NO rule for negative grade values. Negative grades are physically impossible and indicate data entry errors, lab errors, or detection limit issues.

**Impact:** Negative grades flow through compositing, pull down block estimates in kriging, and contaminate resource calculations. A single -5.0 ppm Au in a 0.5 g/t zone visibly distorts the local estimate.

**Fix:** Add `ASSAY_NEGATIVE_VALUE` check for all numeric assay columns.

---

### BUG-04: Overlap severity ignores max_small_overlap config
**File:** `drillhole_validation.py` line 693–698
**Severity:** HIGH — false errors flood the tree

ALL overlaps are flagged as ERROR regardless of size. The config field `max_small_overlap` exists but is never used in validation:
```python
# Current code:
if f < prev_to:
    v.append(... severity="ERROR" ...)  # 0.001m overlap = ERROR

# Should be:
overlap = prev_to - f
if overlap > cfg.max_small_overlap:
    severity = "ERROR"    # Real overlap
else:
    severity = "WARNING"  # Rounding artifact
```

**Impact:** Datasets with minor floating-point rounding (0.001m overlaps) show hundreds of ERRORs. Users either ignore ALL errors (missing real issues) or waste hours fixing non-issues.

---

### BUG-05: Apply to Registry has no validation gate
**File:** `qc_window.py` line 1864–1993
**Severity:** HIGH — bad data enters downstream pipeline

`_apply_to_registry()` pushes data to the DataRegistry regardless of validation status. A user can:
1. Load data with 500 ERRORs
2. Click "Apply to Registry" immediately
3. All downstream tools (compositing, kriging) use uncleaned data

**Fix:** Check `self.violations_all` for ERROR count before allowing apply. Show warning if fatal_count > 0.

---

### BUG-06: ManualEditEngine stores strings in numeric columns
**File:** `qc_window.py` line 252–293, `drillhole_manual_edit.py` line 108–127
**Severity:** HIGH — crashes downstream numeric operations

`PandasTableModel.setData` passes raw string values from the UI to `ManualEditEngine.edit_cell`. If the user types "abc" in a from_depth column, it stores the string "abc". Next time validation runs `float(f)`, it throws a ValueError caught at line 669 — but the bad value persists in the DataFrame.

**Fix:** Type-check in `setData` before calling `edit_cell`:
```python
if pd.api.types.is_numeric_dtype(self._df[col_name]):
    try:
        value = float(value)
    except ValueError:
        return False  # Reject non-numeric input
```

---

### BUG-07: Autofix destroys undo history
**File:** `qc_window.py` line 838–844
**Severity:** MEDIUM — user can't recover from autofix mistakes

After autofix, a brand new `ManualEditEngine` is created:
```python
self.editor_engine = ManualEditEngine(
    collars=af.collars, surveys=af.surveys, ...
)
```
This discards the entire undo stack. If autofix made destructive changes (BUG-01), the user cannot Ctrl+Z to recover.

**Fix:** Either preserve undo stack across autofix, or snapshot pre-autofix state as a single undo entry.

---

### BUG-08: _find_column is duplicated and diverging
**File:** `drillhole_validation.py` line 167, `drillhole_autofix.py` line 143
**Severity:** MEDIUM — schema accepted by validation but rejected by autofix

Two separate copies of `_find_column()`. If you add "BHID" support to validation, autofix won't know about it. The autofix will silently skip fixing holes it can't find columns for.

**Fix:** Import from validation module instead of duplicating.

---

## IMPORTANT — Correctness adjacent, misleading to user

### BUG-09: Autofix confidence is uniform 0.9 for all interval fixes
**File:** `drillhole_autofix.py` line 451
A 0.01m gap fix and a 50m overlap fix both get confidence=0.9. Confidence should scale inversely with the magnitude of the change.

### BUG-10: No high-grade outlier detection
No statistical check for assay values that are 3+ standard deviations from the mean within a lithological domain. This is standard QC practice (JORC Table 1 Section 1).

### BUG-11: No check for assay values exceeding detection limits
If a lab returns ">10000 ppm" as text, it fails numeric conversion silently. No specific rule to catch detection limit markers.

### BUG-12: Find & Replace searches a copy, edits the original
**File:** `qc_window.py` line 1484
`df = model._df.copy()` used for searching, but edits go through `editor_engine.edit_cell()` using `df.index[row_idx]`. If autofix has been run (resetting indices), the copy's index may not match the engine's current DataFrame.

### BUG-13: NaN vs None comparison in setData
**File:** `qc_window.py` line 261
`str(current_val) == str(value)` — `str(np.nan)` = "nan", `str(None)` = "None". Editing a NaN cell by typing "None" would be rejected as "no change", and vice versa.

---

## MISSING VALIDATION RULES (industry standard)

1. **ASSAY_NEGATIVE_VALUE** — Negative grades in any numeric assay column
2. **ASSAY_HIGH_OUTLIER** — Grade > mean + N*stdev within domain
3. **ASSAY_DETECTION_LIMIT** — Non-numeric markers ("<0.01", ">10000")
4. **COLLAR_COORDINATE_OUTLIER** — Collar location > N*stdev from centroid
5. **SURVEY_DIP_SIGN_CONVENTION** — Mixed positive/negative dip conventions
6. **LITH_CODE_UNKNOWN** — Lithology code not in approved list
7. **INTERVAL_COMPLETE_COVERAGE** — Assay intervals should cover collar-to-TD
8. **DUPLICATE_SAMPLE_ID** — Sample IDs must be unique within a hole

---

## FIX PRIORITY ORDER

1. BUG-01 (autofix overlap — data destruction)
2. BUG-02 (autofix negative length — phantom intervals)
3. BUG-03 (negative assay detection — add rule)
4. BUG-06 (string in numeric column — type checking)
5. BUG-05 (registry gate — validation check before apply)
6. BUG-04 (overlap severity — use max_small_overlap)
7. BUG-07 (undo preservation across autofix)
8. BUG-08 (_find_column dedup)
