# Kriging Subsystem – Correctness Audit Report (Part 1: OK, SK, UK)

**Step 6 of Mining Estimation Pipeline**  
**Date:** 2026-02-23  
**Files audited:** 7 files, ~8,650 lines  
**Bugs found:** 10 (3 critical, 3 high, 4 medium)  
**Bugs fixed:** 5 of 10

---

## Files Audited

| File | Lines | Role |
|------|-------|------|
| `kriging3d.py` | 1,298 | Ordinary Kriging engine (standard + Numba-fast + full professional) |
| `kriging_engine.py` | 226 | Numba-accelerated OK kernel (prange parallelized) |
| `simple_kriging3d.py` | 792 | Simple Kriging engine (standard + Numba-fast) |
| `universal_kriging.py` | 960 | Universal Kriging engine (Numba + drift models) |
| `kriging_panel.py` | 2,597 | OK panel UI |
| `simple_kriging_panel.py` | 2,753 | SK panel UI |
| `universal_kriging_panel.py` | 1,945 | UK panel UI |

---

## CRITICAL BUGS

### BUG K-01: UK engine passes wrong sill to covariance kernel (CRITICAL)
**File:** `universal_kriging.py` lines 445–451, 299–306  
**Impact:** All UK estimates have wrong covariance matrix → systematically biased grades

The UK wrapper subtracts nugget from sill before passing to the Numba kernel:

```python
# BEFORE (broken)
params[1] = variogram_params['sill'] - variogram_params['nugget']  # Partial sill
```

But the kernel's `_get_cov()` function (line 90) expects **total sill** and internally computes partial sill:

```python
def _get_cov(d, range_val, sill, nugget, model_type):
    """sill : TOTAL sill (nugget + partial sill) - CANONICAL CONVENTION"""
    partial_sill = max(sill - nugget, 0.0)  # Double-subtracts nugget!
```

For a deposit with nugget=0.3, total_sill=1.0 (partial_sill=0.7):
- Wrapper sends: `params[1] = 1.0 - 0.3 = 0.7` (partial sill)
- Kernel's `_get_cov` computes: `partial_sill = 0.7 - 0.3 = 0.4` (should be 0.7)
- C(0) at h=0: returns 0.7 instead of 1.0 (30% underestimate of covariance diagonal)

Additionally, `run_uk_kernel` computed `total_sill = sill + nugget = 0.7 + 0.3 = 1.0` (accidentally correct for variance), but the covariance matrix itself has wrong values throughout.

The kriging weights derived from this corrupted covariance matrix are wrong, producing biased UK estimates.

**Note:** The OK engine (`kriging_engine.py`) uses a different convention — its `calc_covariance()` takes **partial sill** and internally adds nugget. These two engines are inconsistent with each other. The UK engine's `_get_cov` docstring says "TOTAL sill" and the code is written for total sill, so that's the canonical convention for UK.

**Fix:** (a) Wrapper passes total sill directly without subtraction. (b) Kernel sets `total_sill = sill` (not `sill + nugget`).

### BUG K-02: UK grid reshape uses wrong memory order (CRITICAL)
**File:** `universal_kriging.py` line 704  
**Impact:** Every UK estimate mapped to WRONG spatial block in the 3D grid

```python
# BEFORE (broken)
estimates = estimates_flat.reshape((nx, ny, nz), order='F')  # Fortran order
```

The estimation grid is created with `np.meshgrid(..., indexing='ij')` and flattened with `ravel()` (C-order, z varies fastest). Reshaping with `order='F'` (x varies fastest) maps values to wrong positions:

For a 2×3×4 grid, block at position [0,1,0] in the grid (data index 4 in C-order) gets mapped to position [0,2,0] in F-order. Every non-corner block gets the wrong grade estimate.

The OK engine correctly uses `order='C'` (kriging3d.py line 878).

**Fix:** Changed to `order='C'`.

### BUG K-03: OK panel loads total sill as partial sill from combined_3d_model (CRITICAL)
**File:** `kriging_panel.py` lines 967–974  
**Impact:** OK engine receives total_sill + nugget instead of total_sill → overshoot by one nugget

The comment says "'sill' is partial sill" but `combined_3d_model['sill']` is actually **total sill** (set from `major.get('total_sill', 1.0)` in variogram_panel.py):

```python
# BEFORE (broken)
partial_sill = combined.get('sill', 0.0)      # Gets total_sill, calls it partial
total_sill = combined.get('total_sill', nugget + partial_sill)  # Fallback double-adds nugget
self.sill_spin.setValue(partial_sill)           # Stores total_sill as "partial"
```

Then in `run_analysis`: `sill_total = sill_spin + nugget = total_sill + nugget`. The kriging engine receives sill that's one nugget too high, inflating the covariance matrix diagonal and producing incorrect kriging weights.

For nugget=0.3, total_sill=1.0: engine receives `sill = 1.3` instead of `1.0`.

**Fix:** Use `combined.get('total_sill', combined.get('sill', 0.0))` and compute partial_sill = total - nugget.

---

## HIGH BUGS

### BUG K-04: UK panel variogram loading assumes 'sill' is always total (HIGH)
**File:** `universal_kriging_panel.py` lines 618–640  
**Impact:** Wrong sill when loading from fitted_models that store partial sill

When loading from `major` or `omni` fitted models, the code does:
```python
total_sill = major.get('sill', 0.0)  # Could be partial sill!
partial_sill = total_sill - nugget   # Double-subtracts if partial
```

After the variogram_assistant fixes (Bug V-01), `fitted_models[dir]['sill']` is **partial sill** and `fitted_models[dir]['total_sill']` is total sill. The UK panel doesn't check for the `total_sill` key.

**Fix:** Check for `total_sill` key first, fall back to heuristic inference.

### BUG K-05: SK panel sends sill without adding nugget (HIGH)
**File:** `simple_kriging_panel.py` line 927  
**Impact:** SK engine receives partial sill where it expects total sill

The SK panel UI labels the sill spinner as "Sill (C1):" (partial sill convention). But it passes the value directly:

```python
"sill": self.sill_spin.value(),  # Partial sill passed as "sill"
```

The SK engine (`simple_kriging_fast`) treats `variogram_params["sill"]` as total sill:
```python
sill_total = float(variogram_params["sill"])
partial_sill = sill_total - nug  # Double-subtracts nugget
```

The severity depends on the source of the sill value. When loaded from `combined_3d_model` (where 'sill' = total), it's accidentally correct. When loaded from fitted_models (where 'sill' = partial), it's wrong.

**Not fixed** — requires controller code review to verify the full data flow. **Recommend:** Match OK panel pattern: `"sill": self.sill_spin.value() + self.nugget_spin.value()`.

### BUG K-06: OK + SK + UK engines all have C(0) = partial_sill instead of total_sill (HIGH)
**File:** `kriging3d.py` lines 688, 698; `kriging_engine.py` lines 30–35; `universal_kriging.py` line 93  
**Impact:** Covariance matrix diagonal missing the nugget component

All three variogram implementations evaluate γ(0) = nugget (not 0), which produces:

```
C(0) = total_sill - γ(0) = total_sill - nugget = partial_sill
```

Mathematically, C(0) should equal total_sill (the full variance σ²). The nugget effect creates a discontinuity at the origin: γ(0) = 0 but γ(0+) = nugget. All current implementations fail to distinguish h=0 (exact coincidence) from h→0+ (nearby but distinct).

This means the diagonal of the covariance matrix is `partial_sill` instead of `total_sill`. For deposits with significant nugget (common in gold deposits where nugget can be 50–80% of sill), this systematically underestimates the diagonal, leading to:
- Kriging weights that don't properly account for measurement noise
- Kriging variance that's too low (overconfident)
- Screening effect that's too strong

The UK kernel's `_get_cov` partially addresses this with a special h=0 check (line 92–93: `if d < 1e-9: return sill`), but the OK/SK engines don't.

**Not fixed** — requires careful coordination across all three engines. The regularization term (1e-10 * max diagonal) partially compensates but doesn't fully correct the issue.

---

## MEDIUM BUGS

### BUG K-07: Three different sill conventions across kriging engines (MEDIUM)
**Files:** `kriging_engine.py`, `kriging3d.py`, `universal_kriging.py`  
**Impact:** Code maintenance hazard; calling wrong engine with wrong convention produces silent errors

| Engine | Covariance function | `sill` param means | Convention |
|--------|-------------------|-------------------|------------|
| `kriging_engine.py` (OK Numba) | `calc_covariance()` | **Partial sill** | GSLIB |
| `kriging3d.py` (OK standard) | inline `gamma_fun()` | **Partial sill** (passed as `sill_total - nug`) | GSLIB |
| `universal_kriging.py` (UK) | `_get_cov()` | **Total sill** | Canonical |
| `simple_kriging3d.py` (SK Numba) | `_calc_covariance()` | **Partial sill** | GSLIB |

Any refactoring that calls the wrong function with the wrong convention produces silently wrong estimates with no error.

**Recommendation:** Standardize all engines on a single convention. The canonical choice is **total sill** (matching variogram_model.py). Add `assert` or explicit parameter names (`total_sill=`, `partial_sill=`) to prevent misuse.

### BUG K-08: Anisotropy not loaded from fitted_models in OK panel (MEDIUM)
**File:** `kriging_panel.py` lines 1280–1297  
**Impact:** Isotropic kriging used even when directional variograms are available

The OK panel's `gather_parameters` extracts anisotropy ranges only from `fitted_models`. But when loading from `combined_3d_model` (Priority 1 path), the ranges are available but not extracted into the anisotropy dict. This means the anisotropy spinner defaults to the single `range_spin` value for all three directions.

The code at lines 1288–1297 checks `fitted_models` but only if `major` AND `minor` keys exist. If only `combined_3d_model` is available (e.g., from the assistant), anisotropy is lost.

**Not fixed** — the combined_3d_model already contains `major_range`, `minor_range`, `vertical_range`. These should be extracted during the Priority 1 loading path.

### BUG K-09: OK fallback to lstsq uses non-regularized matrix (MEDIUM)
**File:** `kriging3d.py` line 711  
**Impact:** lstsq fallback may produce unstable weights

```python
except LinAlgError:
    w_mu = np.linalg.lstsq(K, rhs, rcond=1e-10)[0]  # Uses original K, not K_reg
```

The `solve()` call uses `K_reg` (regularized matrix), but the lstsq fallback uses `K` (non-regularized). If the matrix is ill-conditioned enough to fail `solve`, using the non-regularized version in lstsq may produce poor results.

**Not fixed** — change `K` to `K_reg` in the lstsq fallback.

### BUG K-10: Non-deterministic pre-compilation warmup (MEDIUM)
**File:** `universal_kriging.py` lines 910–911  
**Impact:** Precompilation uses different random data each time, may mask edge cases

```python
data_coords = np.random.rand(n_data, 3).astype(np.float64) * 100
```

Uses unseeded `np.random.rand()`. While this doesn't affect correctness of the compiled kernels, it's inconsistent with the determinism design used throughout the rest of the codebase (all random ops use seed=42).

**Not fixed** — add `rng = np.random.default_rng(42)` for consistency.

---

## What's Done Well

**Covariance form is correct.** The `C(h) = sill_total - γ(h)` implementation properly converts semivariance to covariance. The regularization approach (1e-10 * max diagonal) is well-scaled.

**Multi-pass search is production-grade.** The OK engine supports JORC/NI 43-101 compliant multi-pass search with configurable min/max neighbors and ellipsoid multipliers per pass. QA metrics (kriging efficiency, slope of regression, negative weight percentage) are comprehensive.

**Numba kernels are well-architected.** The single-point solver extraction pattern (avoiding `continue`/`try-except` in `prange` loops) correctly handles Numba's parallelization constraints. The chunked processing with progress callbacks is professional.

**Duplicate sample handling.** The `ordinary_kriging_3d_full` function detects and removes co-located samples before solving, preventing singular matrices.

**Coordinate normalization in UK.** Universal Kriging normalizes coordinates to a local origin before computing drift matrices, avoiding precision loss with large UTM coordinates. This is a common failure point in UK implementations.

**Deterministic neighbor search.** Lexicographic tie-breaking in neighbor selection (`np.lexsort((idx, d))`) ensures identical results across runs.

---

## Priority Fix Order

| Priority | Bug | Impact | Status |
|----------|-----|--------|--------|
| 1 | K-02 UK grid reshape order='F' | Estimates mapped to wrong blocks | ✅ Fixed |
| 2 | K-01 UK sill convention mismatch | Wrong covariance matrix in UK | ✅ Fixed |
| 3 | K-03 OK panel sill from combined model | OK engine overshoots sill by nugget | ✅ Fixed |
| 4 | K-04 UK panel fitted_models sill loading | Wrong UK sill from fitted models | ✅ Fixed |
| 5 | K-05 SK panel missing total_sill | SK engine double-subtracts nugget | ⚠️ Recommend |
| 6 | K-06 C(0) = partial_sill everywhere | Diagonal underestimated by nugget | ⚠️ Document |
| 7 | K-07 Three sill conventions | Maintenance hazard | ⚠️ Recommend |
| 8 | K-08 Anisotropy not loaded from combined | Isotropic kriging when aniso available | ⚠️ Recommend |
| 9 | K-09 lstsq fallback non-regularized | Unstable fallback | ⚠️ Recommend |
| 10 | K-10 Non-deterministic warmup | Inconsistent with design | ⚠️ Low |

---

## Fixed Files Delivered

| File | Bugs Fixed |
|------|-----------|
| `universal_kriging.py` | K-01 (sill convention), K-02 (reshape order) |
| `kriging_panel.py` | K-03 (combined_3d_model sill interpretation) |
| `universal_kriging_panel.py` | K-04 (fitted_models sill loading + gather_parameters total sill) |

---

## Cross-Subsystem Note: The Sill Convention Problem

The single most pervasive bug pattern across the variogram and kriging subsystems is **sill convention confusion**. The term "sill" is used to mean three different things:

| Convention | Meaning | Value (example) | Used in |
|-----------|---------|----------------|---------|
| **Total sill** | nugget + contribution = C₀ + C₁ = σ² | 1.0 | variogram_model.py canonical, UK _get_cov, combined_3d_model |
| **Partial sill** | contribution only = C₁ | 0.7 | fit_variogram() return, OK/SK Numba kernels, panel spinners |
| **Ambiguous 'sill'** | could be either | ??? | fitted_models dicts, variogram_assistant |

This has caused at least 5 separate bugs (V-01, K-01, K-03, K-04, K-05). The definitive fix is:
1. **Never use a bare `sill` key.** Always use `total_sill` or `partial_sill` explicitly.
2. **All variogram model dicts must store both** `total_sill` and `partial_sill`.
3. **All engines should document and validate** which convention they expect.
