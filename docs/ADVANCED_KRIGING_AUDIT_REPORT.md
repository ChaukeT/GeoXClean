# Advanced Kriging Subsystem Audit Report
## Indicator Kriging · Co-Kriging · Bayesian Kriging · Soft Data

**Date:** 2026-02-23  
**Auditor:** Claude (Anthropic)  
**Files audited:** indicator_kriging.py, indicator_kriging_panel.py, cokriging3d.py, cokriging_panel.py, bayesian_kriging.py, soft_kriging_panel.py, soft_data.py  
**Continues from:** Variogram subsystem audit + OK/SK/UK kriging audit

---

## EXECUTIVE SUMMARY

The advanced kriging subsystem contains **4 CRITICAL**, **3 HIGH**, and **3 MEDIUM** bugs. The most severe finding is that **the entire Bayesian kriging module is non-functional**: a legacy compatibility shim (`SoftDataSet.points` always returning `[]`) causes every Bayesian update path to silently fall through to standard kriging. Users believe they are running Bayesian kriging with soft data, but all soft data is discarded. Additionally, the grid reshape `order='F'` bug (K-02) reappears in both Indicator Kriging and Co-Kriging engines, mapping every estimate to the wrong spatial block. The sill convention mismatch pattern continues here with triple-subtraction in the Co-Kriging pipeline.

| Severity | Count | Impact |
|----------|-------|--------|
| CRITICAL | 4 | Dead Bayesian module, wrong block mapping, wrong covariance matrices |
| HIGH | 3 | Sill errors in secondary SK, coordinate mismatch cascade, job runner crashes |
| MEDIUM | 3 | Label ambiguity, non-deterministic warmup, assisted loading sill confusion |

---

## BUG CATALOG

### BUG A-01: Bayesian kriging `soft_data.points` always returns `[]` (CRITICAL)

**Location:** bayesian_kriging.py lines 60, 125, 273, 502, 659  
**Impact:** ALL Bayesian updates are silently skipped — entire module is dead code

**Root Cause:** `SoftDataSet` uses a Structure-of-Arrays (SoA) design. It stores `coords`, `means`, `variances` as numpy arrays. The `.points` property is a legacy compatibility shim:

```python
# soft_data.py line 97
@property
def points(self) -> list:
    """Compatibility property for legacy code that expects a 'points' attribute."""
    return []  # ← ALWAYS empty!
```

Every Bayesian function checks `len(soft_data.points) == 0` before applying updates:

```python
# bayesian_kriging.py line 125
if soft_data is None or len(soft_data.points) == 0:
    logger.info("No soft data provided, using standard OK")  # ← Always hits this!
    return ok_estimates, ok_variances
```

**Affected functions:** `_find_nearest_soft_data`, `run_bayesian_ok`, `run_bayesian_uk`, `run_bayesian_ik`, `run_bayesian_cok`

**Fix:** Replace `len(soft_data.points) == 0` with `soft_data.n_points == 0` everywhere:

```python
if soft_data is None or soft_data.n_points == 0:
```

---

### BUG A-02: Bayesian job runners import non-existent `SoftPoint` class (CRITICAL)

**Location:** bayesian_kriging.py lines 770-772, 820-822, 879-881, 940-942  
**Impact:** All Bayesian jobs crash with ImportError

The job runners attempt to reconstruct `SoftDataSet` from a serialized dict:

```python
from .soft_data import SoftPoint  # ← Does not exist! ImportError
points.append(SoftPoint(**p))
soft_data = SoftDataSet(points=points, metadata=...)  # ← Wrong constructor signature
```

`SoftPoint` was never defined in `soft_data.py`. And `SoftDataSet.__init__` takes `(coords, means, variances)`, not `(points=...)`.

**Fix:** Replace dict reconstruction with proper SoftDataSet construction:

```python
if isinstance(soft_dict, dict):
    coords = np.array(soft_dict['coords'])
    means = np.array(soft_dict['means'])
    variances = np.array(soft_dict['variances'])
    soft_data = SoftDataSet(
        coords=coords, means=means, variances=variances,
        metadata=soft_dict.get('metadata', {})
    )
```

---

### BUG A-03: IK grid reshape uses wrong memory order (CRITICAL)

**Location:** indicator_kriging.py line 556  
**Impact:** Every IK probability mapped to the wrong spatial block

Same root cause as Bug K-02 (UK engine). Grid created with `np.meshgrid(indexing='ij')` and flattened with `flatten()` (C-order), but reshaped with `order='F'` (Fortran order):

```python
# Line 465: meshgrid with 'ij' indexing
grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
# Line 467: flatten uses C-order (default)
target_coords = np.column_stack([grid_x.flatten(), ...])

# Line 556: WRONG - reshape with Fortran order
probs_reshaped = probs.reshape((nx, ny, nz, n_thresh), order='F')
```

For a 10×10×5 grid, block at index 15 in C-order (x=0,y=1,z=5) gets mapped to position (x=5,y=1,z=0) in F-order. Every non-corner block gets the wrong probability.

The same wrong `order='F'` propagates to median (line 566), mean (line 571), and all threshold properties (line 579).

**Fix:** Change all `order='F'` to `order='C'` in reshape AND ravel calls (lines 556, 566-567, 571-572, 579, 586, 606, 610).

---

### BUG A-04: CoK grid reshape uses wrong memory order (CRITICAL)

**Location:** cokriging3d.py lines 1812-1838  
**Impact:** Every CoK estimate mapped to the wrong spatial block

Identical pattern to A-03. Grid created with `meshgrid(indexing='ij')` + `flatten()` (C-order), reshaped with `order='F'`:

```python
# Line 1719: meshgrid with 'ij' indexing
grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
# Line 1721: flatten (C-order)
target_coords = np.column_stack([grid_x.flatten(), ...])

# Line 1812: WRONG
estimates = results.primary_estimate.reshape((nx, ny, nz), order='F')
variances = results.cokriging_variance.reshape((nx, ny, nz), order='F')
```

Same `order='F'` bug in secondary estimate (line 1826), secondary influence (line 1832), and neighbor count (line 1837).

**Fix:** Change all `order='F'` to `order='C'`.

---

### BUG A-05: IK sill convention mismatch (HIGH)

**Location:** indicator_kriging.py lines 500-505, _get_cov lines 62-63  
**Impact:** Wrong covariance matrix diagonal for IK with non-zero nugget

The wrapper subtracts nugget before passing to the kernel:

```python
# Line 500-502
kern_params = np.array([
    range_, sill_total - nug, nug, m_code  # params[1] = partial sill
])
```

The kernel passes this directly to `_get_cov`:

```python
# run_ik_kernel line 249
sill = params[1]  # partial sill
# _solve_single_ik_point line 127
cov = _get_cov(d, rng, sill, nugget, model_code)
```

But `_get_cov` expects TOTAL sill and subtracts nugget internally:

```python
partial_sill = max(sill - nugget, 0.0)  # Double-subtracts!
```

**Example:** nugget=0.2, sill_total=1.0 (standard indicator)
- Wrapper sends: sill = 1.0 - 0.2 = 0.8
- Kernel computes: partial_sill = 0.8 - 0.2 = 0.6 (should be 0.8)
- C(0) = 0.8 instead of 1.0

For standard IK (nugget ≈ 0), impact is minimal. For IK with significant nugget (common in gold/diamond deposits), probabilities are systematically biased.

**Fix:** Pass total sill directly: `kern_params[1] = sill_total`

---

### BUG A-06: CoK sill triple-subtraction pipeline (HIGH)

**Location:** cokriging_panel.py → cokriging3d.py → _get_cov kernel  
**Impact:** CoK covariance matrix has drastically wrong partial sill

The sill is subtracted three times in the CoK pipeline:

| Step | Operation | Value (nugget=0.3, total_sill=1.0) |
|------|-----------|-------------------------------------|
| 1. Panel loads combined model | `partial = total - nugget` | 0.7 |
| 2. Panel sends as 'sill' | `gather_parameters: 'sill': 0.7` | 0.7 |
| 3. Engine subtracts nugget | `sill_p = 0.7 - 0.3` | 0.4 |
| 4. Kernel _get_cov subtracts nugget | `partial = 0.4 - 0.3` | **0.1** |

Final partial sill used: **0.1** instead of **0.7** — a factor of 7 error.

This causes:
- Covariance matrix massively underestimated
- Kriging weights concentrated on nearest sample only
- Variance grossly underestimated (overconfident)
- Secondary variable influence distorted

**Fix (multi-step):**
1. Panel should store and send TOTAL sill, not partial
2. Engine should NOT subtract nugget (sill_p_user is already partial from panel, or should be total)
3. Standardize: all engines receive total sill, all `_get_cov` functions expect total sill

---

### BUG A-07: `sk_interpolate_secondary` sill mismatch (HIGH)

**Location:** cokriging3d.py lines 437-440  
**Impact:** SK interpolation of secondary variable at targets uses wrong covariance

```python
sill = float(variogram_params.get('sill', 1.0))
nugget = float(variogram_params.get('nugget', 0.0))
total_sill = sill + nugget  # Assumes 'sill' is partial
```

Then calls `_get_cov_np(d, effective_range, sill, nugget, model_code)` — passing `sill` (partial) where `_get_cov_np` expects total sill.

The function correctly computes `total_sill = sill + nugget` for variance calculations, but passes the wrong value to `_get_cov_np`.

**Fix:** Pass `total_sill` to `_get_cov_np` instead of `sill`:

```python
cov = _get_cov_np(d, effective_range, total_sill, nugget, model_code)
```

---

### BUG A-08: Bayesian IK ravel(order='F') coordinate mismatch (MEDIUM)

**Location:** bayesian_kriging.py lines 444-446, 472, 484  
**Impact:** Grid coordinate lookup uses wrong order, compounding IK reshape bug

The Bayesian IK wrapper ravels IK result grids with `order='F'`:

```python
grid_coords = np.column_stack([
    grid_x.ravel(order='F'),  # F-order ravel
    grid_y.ravel(order='F'),
    grid_z.ravel(order='F')
])
```

This was written to match the IK result's `order='F'` reshape (Bug A-03). If A-03 is fixed to `order='C'`, this must also be changed to `order='C'` (or just `ravel()`). If A-03 is NOT fixed, these are "consistently wrong" — matching each other but both mapping to wrong blocks.

**Fix:** Change to `ravel()` (default C-order) when A-03 is fixed.

---

### BUG A-09: CoK panel assisted loading sill ambiguity (MEDIUM)

**Location:** cokriging_panel.py lines 731-738  
**Impact:** Assisted variogram may load partial sill as if it were total, or vice versa

```python
sill = assisted_model.get('sill', 1.0)
self.sill_primary.setValue(max(0.001, sill))
```

After variogram_assistant fixes (Bug V-01), `fitted_models` stores partial sill in 'sill' and total sill in 'total_sill'. This code doesn't check for `total_sill`, so it stores whatever 'sill' is — which may be partial or total depending on the source.

For the combined_3d_model path (line 664), 'sill' is total sill, so `prim_partial_sill = total - nugget` is correct. But for fitted_models, 'sill' may already be partial, causing double-subtraction.

**Fix:** Check for 'total_sill' key first:

```python
total_sill = assisted_model.get('total_sill', assisted_model.get('sill', 1.0))
partial_sill = max(total_sill - nugget, 0.001)
```

---

### BUG A-10: IK & CoK precompile non-deterministic (MEDIUM)

**Location:** indicator_kriging.py line 644; cokriging3d.py line 1916 (CoK is OK — uses seed)  
**Impact:** IK precompile uses unseeded random data

```python
# IK (line 644) - NO SEED
data_coords = np.random.rand(n_data, 3).astype(np.float64) * 100

# CoK (line 1916) - HAS SEED ✓
np.random.seed(42)
data_coords = np.random.rand(n_data, 3).astype(np.float64) * 100
```

**Fix:** Add `np.random.seed(42)` before IK warmup data generation.

---

## WHAT'S DONE WELL

**Indicator Kriging Engine:**
- Order relation correction (forward/backward averaging + GSLIB re-check) is textbook correct
- E-Type and median calculation from CDF are mathematically sound
- Numba kernel architecture (extracted `_solve_single_ik_point`) correctly handles prange constraints
- Fallback to sample proportion when matrix is singular is a good defensive pattern

**Co-Kriging Engine:**
- Markov-1 cross-covariance model is correctly formulated: Cps(h) = ρ·√(Cpp(0)·Css(0))·Cpp(h)/Cpp(0)
- Correlation validation with automatic OK fallback is professional-grade
- SK interpolation for secondary variable is the right approach (vs. naive nearest-neighbor)
- Variable scaling analysis catches unit mismatches before they corrupt results
- Minimum neighbor threshold with sectoring support is audit-compliant
- Secondary influence tracking (`ws/(|wp|+|ws|)`) provides essential diagnostic output

**Soft Data Module:**
- SoA (Structure-of-Arrays) design is correct for performance — float32 arrays with Numba-friendly layout
- Coordinate normalization to local origin before float32 conversion prevents precision loss with large UTM coords
- Numba-compiled `_numba_ik_to_moments` kernel avoids 20GB intermediate arrays for large block models
- Python fallback implementation ensures functionality without Numba

**Bayesian Kriging (algorithmic design):**
- Precision-weighted combination formula is mathematically correct (when code reaches it)
- IK Bayesian update using normal CDF for soft probabilities is the standard approach
- CDF monotonicity enforcement after Bayesian update is necessary and correct

---

## CROSS-SUBSYSTEM PATTERN: The Sill Convention Problem (Updated)

The sill convention confusion now spans **all kriging engines and panels**:

| Component | What 'sill' means | What it should be |
|-----------|-------------------|-------------------|
| variogram_model.py | Total sill | ✓ |
| OK Numba kernel (kriging_engine.py) | Partial sill | Should be Total |
| OK standard (kriging3d.py) | Partial sill | Should be Total |
| SK Numba (simple_kriging3d.py) | Partial sill | Should be Total |
| UK `_get_cov` (universal_kriging.py) | Total sill | ✓ (after K-01 fix) |
| IK `_get_cov` (indicator_kriging.py) | Total sill | ✓ (but receives partial) |
| CoK `_get_cov` (cokriging3d.py) | Total sill | ✓ (but receives partial) |
| CoK `_get_cov_np` (cokriging3d.py) | Total sill | ✓ (but receives partial) |
| IK panel (indicator_kriging_panel.py) | Fixed 1.0 | OK for standard IK |
| CoK panel (cokriging_panel.py) | Partial sill | Should send Total |
| Soft panel (soft_kriging_panel.py) | Ambiguous | Should be explicit |

**Total bug count from sill confusion: 8 bugs across variogram + kriging subsystems.**

---

## PRIORITY FIX ORDER

| Priority | Bug | Severity | Effort | Impact |
|----------|-----|----------|--------|--------|
| 1 | A-01 | CRITICAL | Low | Fixes: Bayesian module completely non-functional |
| 2 | A-02 | CRITICAL | Medium | Fixes: Job runner crashes on any Bayesian run |
| 3 | A-03 | CRITICAL | Low | Fixes: Every IK probability at wrong block |
| 4 | A-04 | CRITICAL | Low | Fixes: Every CoK estimate at wrong block |
| 5 | A-06 | HIGH | Medium | Fixes: CoK covariance off by 7x |
| 6 | A-05 | HIGH | Low | Fixes: IK covariance with nugget > 0 |
| 7 | A-07 | HIGH | Low | Fixes: SK secondary interpolation covariance |
| 8 | A-08 | MEDIUM | Low | Fixes: Bayesian IK grid coordinates (after A-03) |
| 9 | A-09 | MEDIUM | Low | Fixes: CoK assisted loading sill interpretation |
| 10 | A-10 | MEDIUM | Low | Fixes: Deterministic precompilation |

---

## BAYESIAN KRIGING PANEL

A new `bayesian_kriging_panel.py` has been created to provide the missing UI. Key design decisions:

1. **Sill convention:** Panel UI labels sill as "Sill (partial C₁)" but `gather_parameters` computes and sends `sill_total = partial + nugget` — matching the canonical convention expected by engines
2. **Base method support:** OK, UK, IK, CoK all supported via dropdown with method-specific UI elements (drift type for UK, secondary variable for CoK)
3. **Soft data sources:** CSV, IK results (auto-convert via `soft_from_ik_result`), block model property
4. **Validation workflow:** Pre-flight check loads and validates soft data before committing to expensive kriging run
5. **Controller integration:** Uses `controller.run_task('bayesian_kriging', ...)` pattern matching all other panels
6. **Project save/restore:** Full `get_panel_settings` / `apply_panel_settings` implementation
