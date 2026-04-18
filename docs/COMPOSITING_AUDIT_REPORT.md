# COMPOSITING PIPELINE — CORRECTNESS AUDIT

## Files Reviewed
- `compositing_engine.py` (2,357 lines) — Core math engine
- `compositing_utils.py` (471 lines) — DataFrame → Interval conversion
- `compositing_ui_engines.py` (499 lines) — UI state → CompositeConfig translation
- `compositing_window.py` (2,105 lines) — Qt UI, data flow, registry integration

---

## CRITICAL BUGS

### BUG-C01: `_calculate_weighted_grade` wrong under density/mass weighting
**File:** compositing_engine.py lines 2059–2097
**Severity:** CRITICAL — produces nonsensical grades
**Affects:** ALL economic compositing grade calculations (dilution rule tests, linear grade, segment grades)

**The problem:**
```python
weight = _interval_weight(iv, length, cfg)   # returns length * density for MASS mode
total_grade_length += grade_val * weight      # grade * length * density
total_length += length                         # ← accumulates LENGTH, not WEIGHT
...
avg_grade = total_grade_length / total_length  # = grade * density  ← WRONG
```

When `weighting_mode` is DENSITY or MASS, `_interval_weight()` returns `length × density`. The numerator accumulates `grade × weight`, but the denominator accumulates raw `length` instead of `weight`. Result: `avg_grade = Σ(grade × length × density) / Σ(length) = grade × density`, which is physically meaningless.

For LENGTH weighting this works by accident because weight == length.

**Impact:** Every dilution rule test, every linear grade calculation, and every segment composite in economic mode returns wrong grades when density weighting is used. Ore/waste boundaries shift, resource tonnages change.

**Fix:** Replace `total_length += length` with `total_weight += weight`, divide by `total_weight`.

---

### BUG-C02: `treat_null_as_zero` silently inflates grades
**File:** compositing_engine.py lines 629–634, compositing_utils.py line 265
**Severity:** CRITICAL — systematic grade bias upward
**Affects:** ALL compositing methods on multi-element datasets with incomplete assays

**The problem:** Two-part failure:

1. In `dataframes_to_intervals` (utils line 265), NaN grade values are **excluded** from the `grades` dict entirely:
   ```python
   if val is not None and isinstance(val, (int, float)) and not pd.isna(val):
       grades[gc] = float(val)
   # NaN → key simply absent from dict
   ```

2. In the engine, the compositing loop iterates only over keys present in each interval:
   ```python
   for k, v in iv.grades.items():  # ← only visits keys that exist
       ...
       w_sums[k] += w_slice
   ```

So if Interval A has `{"Au": 5.0, "Cu": 1.0}` and Interval B has `{"Au": 3.0}` (Cu was NaN), with `treat_null_as_zero=True`:
- **Expected:** Cu average = `(1.0 × w_A + 0.0 × w_B) / (w_A + w_B)` (diluted)
- **Actual:** Cu average = `1.0 × w_A / w_A` = `1.0` (undiluted, inflated)

The engine's `_get_grade_value` has the logic to convert None→0.0, but it's never invoked because the key doesn't exist in `iv.grades`.

**Impact:** Grade composites are systematically biased upward for any element with incomplete assay coverage. In a gold deposit where half the holes weren't assayed for Cu, Cu composites report double the true average. This flows directly into kriging estimates and resource tonnages.

**Fix:** In `dataframes_to_intervals`, include ALL grade columns for every interval, using `None` for missing values instead of omitting the key. The engine's `_get_grade_value` then correctly handles the None→0.0 conversion.

---

### BUG-C03: Partial merge doesn't update `to_depth`
**File:** compositing_engine.py lines 691–730 (and AUTO merge at 735–777)
**Severity:** HIGH — composite geometry is wrong
**Affects:** FIXED_LENGTH, EQUAL_MASS, TRUE_THICKNESS when partial_strategy = MERGE or AUTO

**The problem:** When a partial composite is merged into the previous one:
- `prev_comp.metadata["support"]` is updated ✓
- `prev_comp.metadata["total_length"]` is updated ✓
- `prev_comp.grades` are recalculated ✓
- `prev_comp.to_depth` is **never updated** ✗

The composite reports `to_depth = old_boundary` but contains grade data past that point. The `Composite.length` property returns `to_depth - from_depth` which is now shorter than the actual support, and spatial positioning in kriging places the composite at the wrong centroid.

**Fix:** Add `prev_comp.to_depth = comp_start + partial_length` after merge.

---

### BUG-C04: Two-pass economic compositing is a no-op
**File:** compositing_engine.py lines 1729–1736
**Severity:** HIGH — feature doesn't work

```python
if cfg.composite_twice:
    first_pass = self._composite_economic_single_pass(intervals, cfg)
    # The comment says "Convert composites back to intervals for second pass"
    # but this line re-runs with original intervals:
    return self._composite_economic_single_pass(intervals, cfg)
```

First pass result is computed then discarded. Second pass runs on the same raw intervals and produces identical output. The `composite_twice` UI option does nothing.

**Fix:** Convert first-pass composites to Interval objects, run second pass on those.

---

### BUG-C05: HoleAccumulator double-adjusts single-interval queries
**File:** compositing_engine.py lines 284–352
**Severity:** HIGH — wrong grades from accumulator

When `idx_start == idx_end` (query range falls within a single interval), both the "partial start" and "partial end" adjustments fire on the same interval. Each one subtracts the full interval and adds its partial contribution, but since they share the cumulative sums, the full interval gets subtracted twice and two different partials get added.

**Example:** Query [25, 75] within interval [0, 100]:
- Partial start: subtract full [0,100], add partial [25,100] 
- Partial end: subtract full [0,100] again, add partial [25,75]
- Net: original_metal - 2×full_metal + partial_25_100 + partial_25_75

**Impact:** Economic compositing uses HoleAccumulator for fast grade lookups. Wrong grades → wrong ore/waste classification → wrong resource estimate.

**Fix:** Handle the single-interval case (`idx_start == idx_end`) as a special case.

---

## HIGH-SEVERITY BUGS

### BUG-C06: `_calculate_weighted_grade` returns `total_length` not `total_weight`
**File:** compositing_engine.py line 2097
**Severity:** HIGH (consequence of BUG-C01)

The function returns `(avg_grade, total_length)`. Callers use the second value as the weighting factor for combining segments in dilution tests:
```python
total_linear_grade = candidate_grade * candidate_length + waste_ore_linear_grade
```

Under density weighting, `candidate_length` is raw length, not mass. The dilution test becomes length-weighted instead of mass-weighted, defeating the purpose of density weighting.

**Fix:** Return `(avg_grade, total_weight)` where `total_weight` is the sum of weights.

---

### BUG-C07: `bench_aligned` doesn't use `_get_grade_value` consistently
**File:** compositing_engine.py lines 1252–1256
**Severity:** HIGH — bench compositing ignores treat_null_as_zero

```python
for k, v in iv.grades.items():
    if v is None:
        continue   # ← hard-coded skip, ignores treat_null_as_zero
    num_sums[k] = num_sums.get(k, 0.0) + v * w_slice
    w_sums[k] = w_sums.get(k, 0.0) + w_slice
```

While fixed-length uses `_get_grade_value(v, cfg)`, bench-aligned does a raw `if v is None: continue`. This means `treat_null_as_zero=True` is honored in fixed-length but ignored in bench-aligned. User gets different composite grades for the same data depending on which method they pick.

**Fix:** Replace `if v is None: continue` with `grade_val = _get_grade_value(v, cfg)` pattern.

---

### BUG-C08: `equal_mass` doesn't use `_get_grade_value` either
**File:** compositing_engine.py lines 979–983
**Severity:** HIGH — same inconsistency as BUG-C07

```python
for k, v in iv.grades.items():
    if v is None:
        continue   # ← ignores treat_null_as_zero
```

Equal-mass compositing has the same raw None check. Only fixed-length and rolling-window properly call `_get_grade_value`.

---

### BUG-C09: Audit trail config hash is incomplete
**File:** compositing_window.py lines 1488–1498  
**Severity:** MEDIUM — undermines reproducibility

Only 6 fields hashed: method, composite_length, weighting_mode, partial_strategy, treat_null_as_zero, exclude_qaqc. Missing: cutoff_field, cutoff_grade, bench_height, bench_offset, rolling_window_length, rolling_step, all economic parameters (dilution_rule, min_ore_composite_length, etc.).

Two runs with different cutoff grades produce the same config_hash. Audit trail cannot distinguish them.

**Fix:** Hash all cfg fields, or use `dataclasses.asdict(cfg)`.

---

## MEDIUM BUGS

### BUG-C10: Economic segment `intervals` field uses broken list comprehension
**File:** compositing_engine.py lines 1810–1826

The list comp `[iv for iv in classified_intervals if seg_start <= iv["from"] < seg_end or seg_start < iv["to"] <= seg_end]` has operator precedence issues and can include intervals outside the segment. The field is never used downstream (dead code), but it wastes memory on large datasets and misleads anyone reading the code.

### BUG-C11: `_expand_waste_composites` mutates Composite.from_depth/to_depth directly
**File:** compositing_engine.py lines 2178, 2196

`prev_ore.to_depth = expanded_waste_start` and `next_ore.from_depth = expanded_waste_end` mutate composites but don't recalculate their grades. The ore composite now reports a shorter depth range but still has grades computed over the original range. Grades are wrong.

### BUG-C12: `LithologyUIEngine` import in compositing_window.py but not defined in compositing_ui_engines.py
**File:** compositing_window.py line 67 imports `LithologyUIEngine, LithologyUIState` but compositing_ui_engines.py has no `LithologyUIEngine` class. The try/except silently catches this. If the lithology tab is actually used, it will fail at runtime.

---

## FIX PRIORITY ORDER

| Priority | Bug    | Impact                            | Effort |
|----------|--------|-----------------------------------|--------|
| 1        | C02    | Grade inflation on incomplete data | Medium |
| 2        | C01+C06| Wrong grades under density weight  | Low    |
| 3        | C03    | Merged composite geometry wrong    | Low    |
| 4        | C07+C08| Inconsistent null handling          | Low    |
| 5        | C05    | Accumulator double-adjustment      | Medium |
| 6        | C04    | Two-pass is no-op                  | Medium |
| 7        | C11    | Waste expansion mutates grades     | Medium |
| 8        | C09    | Incomplete audit hash              | Low    |
