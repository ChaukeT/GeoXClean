# Simulation Subsystem Audit Report
## SGSIM · CoSGSIM · SIS · IK-SGSIM · Turning Bands · GRF · MPS · DBS

**Date:** 2026-02-23  
**Auditor:** Claude (Anthropic)  
**Files audited:** sgsim3d.py, sgsim_engine.py, sgsim_panel.py, cosgsim3d.py, cosgsim_panel.py, sis.py, sis_panel.py, ik_sgsim.py, ik_sgsim_panel.py, turning_bands.py, turning_bands_panel.py, grf.py, grf_panel.py, mps.py, mps_panel.py, direct_block_sim.py, dbs_panel.py  
**Continues from:** Variogram + Kriging + Advanced Kriging audits

---

## EXECUTIVE SUMMARY

The simulation subsystem is substantially better engineered than the kriging subsystem. The sill convention is handled correctly in most engines (variogram functions use total sill throughout), the SGSIM's `order='F'` usage is self-consistent (unlike kriging where it's broken), and the overall architecture with Gaussian gates, seed enforcement, and lineage metadata is professional-grade.

However, **3 CRITICAL and 4 HIGH** bugs were found. The most impactful is that the Numba-accelerated SGSIM kernel (`sgsim_engine.py`) receives total sill but interprets it as partial sill, inflating the covariance by the nugget amount. The GRF conditioning step has a flatten-order mismatch that assigns wrong unconditional values to grid nodes. And the SIS order-relation logic enforces monotonicity on probabilities but not on the actual Bernoulli draws, allowing physically impossible indicator states.

| Severity | Count | Impact |
|----------|-------|--------|
| CRITICAL | 3 | Wrong covariance in Numba SGSIM, broken GRF conditioning, impossible SIS indicators |
| HIGH | 4 | CoSGSIM grid offset, CoSGSIM sill passthrough, SIS grade reconstruction, GRF panel ignores sill |
| MEDIUM | 3 | IK-SGSIM sequential mode unimplemented, FFT-MA spherical spectral approx, CoSGSIM ravel truncation |

---

## BUG CATALOG

### BUG SIM-01: Numba SGSIM kernel sill inflation (CRITICAL)

**Location:** sgsim_engine.py `calc_covariance` line 39 vs sgsim3d.py line 827  
**Impact:** Covariance inflated by nugget amount for ALL Numba-accelerated SGSIM runs

**Root Cause:** The caller sends total sill but the kernel treats it as partial sill.

sgsim3d.py packs parameters:
```python
kern_params = np.array([
    1.0,          # Range (normalized)
    params.sill,  # ← This is TOTAL sill (e.g., 1.0)
    params.nugget, # e.g., 0.2
    ...
])
```

sgsim_engine.py `calc_covariance` docstring says `sill: Partial sill (total sill - nugget)` and reconstructs:
```python
total_sill = sill + nugget  # = 1.0 + 0.2 = 1.2 (WRONG, should be 1.0)
```

**Example:** total_sill=1.0, nugget=0.2:
- Kernel computes: `total_sill = 1.0 + 0.2 = 1.2`
- C(0) = 1.2 instead of 1.0 (20% inflation)
- Unconditional draws: std = √1.2 instead of √1.0
- SK variance inflated → over-dispersed realizations
- Effect compounds across sequential path: each node's conditioning is wrong

**Severity note:** For SGSIM on Gaussian data (nugget typically ≈ 0 for normal-score transforms), impact is minimal. For SGSIM with non-trivial nugget (possible with assisted variogram loading), realizations are over-dispersed.

**Fix:** Either change sgsim3d.py to send partial sill:
```python
kern_params[1] = params.sill - params.nugget  # Send partial sill
```
Or change sgsim_engine.py to expect total sill:
```python
# In calc_covariance:
total_sill = sill  # sill IS total sill
partial_sill = sill - nugget
```

The second fix is preferred for consistency with all other simulation engines.

---

### BUG SIM-02: GRF conditioning flatten order mismatch (CRITICAL)

**Location:** grf.py line 531-536  
**Impact:** Conditioning assigns wrong unconditional values to wrong grid nodes

**Root Cause:** `uncond_field.flatten()` uses C-order on shape `(nz, ny, nx)`, but `grid_coords` is C-order from `meshgrid(indexing='ij')` shape `(nx, ny, nz)`. These are different orderings.

```python
# Line 504-505: Grid coords are C-order of (nx, ny, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
grid_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

# Line 531: Unconditional field is (nz, ny, nx) from FFT
conditioned_values = _simple_kriging_conditioning(
    grid_coords=grid_coords,
    uncond_field=uncond_field.flatten(),  # C-order of (nz,ny,nx) ← WRONG ORDER
    ...
)
```

For flat index k=1:
- `grid_coords[1]` → spatial position (x=0, y=0, z=dz) (iz=1)
- `uncond_field.flatten()[1]` → value at (x=dx, y=0, z=0) (ix=1)

These are different spatial positions. The conditioning computes residuals using values from wrong locations, corrupting the entire conditional field.

**Fix:** Use `uncond_field.ravel(order='F')` which gives the same ordering as C-order of (nx,ny,nz):
```python
uncond_field=uncond_field.ravel(order='F'),
```

(This works because F-order of (nz,ny,nx) ≡ C-order of (nx,ny,nz) — iz varies fastest in both.)

---

### BUG SIM-03: SIS order-relation violation on Bernoulli draws (CRITICAL)

**Location:** sis.py lines 270-296  
**Impact:** Physically impossible indicator states — block can be "above high threshold but below low threshold"

The SIS loop iterates thresholds low-to-high and enforces monotonicity on the **kriged probabilities** (`prob = min(prob, prev_indicator)`), but the actual **Bernoulli draws** are independent:

```python
for thresh in sorted_thresholds:      # Low → High
    prob, var = _indicator_kriging_estimate(...)
    prob = min(prob, prev_indicator)    # ← Enforces P(t1) ≥ P(t2)
    sim_value = 1.0 if np.random.random() < prob else 0.0  # ← Independent draw!
    prev_indicator = prob               # ← Passes probability, NOT draw
```

**Example:** Thresholds [1.0, 2.0, 3.0]
- Threshold 1.0: prob=0.8, draw=0 (below threshold 1.0)
- Threshold 2.0: prob=0.6, draw=1 (above threshold 2.0)

Result: "block is above 2.0 g/t but below 1.0 g/t" — physically impossible.

**Fix:** After drawing all indicators, enforce order relations on the draws themselves:
```python
# Post-draw order correction: if indicator[high_thresh]=1, force indicator[low_thresh]=1
for j in range(len(sorted_thresholds) - 1, 0, -1):
    if sim_indicators[sorted_thresholds[j]][i_node] == 1.0:
        sim_indicators[sorted_thresholds[j-1]][i_node] = 1.0
```

Or, use a single uniform draw u~U(0,1) and set indicator[thresh] = 1 if u < prob[thresh] for all thresholds simultaneously (cascade method).

---

### BUG SIM-04: CoSGSIM grid-to-block half-cell offset (HIGH)

**Location:** cosgsim3d.py lines 495-581  
**Impact:** Every block gets the value from a grid cell shifted by ½ cell in each direction

The SGSIM grid origin is set to block model minimum coordinates:
```python
xmin = np.min(x)  # Minimum block centroid coordinate (line 495)
```

But SGSIM places grid cell CENTERS at `xmin + i*xinc + xinc/2`:
```python
gx = np.arange(nx) * xinc + xmin + xinc/2  # (sgsim3d.py line 1222)
```

So the first SGSIM cell center is at `xmin + xinc/2`, but the first block centroid is at `xmin`. The index mapping:
```python
ix = np.round((coords[:, 0] - xmin) / xinc)  # First centroid: round(0) = 0
```

Maps the first centroid (at xmin) to ix=0, whose SGSIM value is at `xmin + xinc/2`. This is a systematic half-cell shift.

**Fix:** Set grid origin to `xmin - xinc/2`:
```python
xmin_grid = np.min(x) - xinc / 2
ymin_grid = np.min(y) - yinc / 2
zmin_grid = np.min(z) - zinc / 2
```

---

### BUG SIM-05: CoSGSIM passes variogram sill without convention check (HIGH)

**Location:** cosgsim3d.py line 536  
**Impact:** If variogram assistant stores partial sill as 'sill', SGSIM receives wrong total sill

```python
sill=prim_vario['sill'],  # Direct passthrough — no 'total_sill' check
```

Unlike the SGSIM panel (which correctly checks for 'total_sill' key and adds nugget to partial sill), CoSGSIM bypasses the panel and passes variogram parameters directly to SGSIMParameters. If the variogram registry stores partial sill in the 'sill' key, the SGSIM engine receives partial sill as total sill, underestimating variance.

**Fix:** Add the same convention check used by the SGSIM panel:
```python
nugget = prim_vario.get('nugget', 0.0)
if 'total_sill' in prim_vario:
    sill = prim_vario['total_sill']
else:
    sill = prim_vario['sill'] + nugget  # Assume 'sill' is partial
```

---

### BUG SIM-06: SIS grade reconstruction is naive midpoint interpolation (HIGH)

**Location:** sis.py lines 302-324  
**Impact:** Reconstructed grades have artificial plateaus and wrong distribution

The grade reconstruction from indicator states uses crude midpoint logic:
```python
grade = sorted_thresholds[0] * 0.5  # Below lowest threshold
for j, thresh in enumerate(sorted_thresholds):
    if indicators[j] == 1:
        grade = (thresh + sorted_thresholds[j + 1]) / 2  # Midpoint
    else:
        break
```

Problems:
1. Below lowest threshold: grade = threshold × 0.5 (arbitrary)
2. Above highest threshold: grade = threshold × 1.5 (arbitrary extrapolation)
3. Between thresholds: always midpoint (ignores CDF shape)
4. Produces discrete plateaus, not continuous grades

**Fix:** Use proper CDF interpolation with order-corrected probabilities, sampling from within each bin proportional to the conditional probability. Or use E-type estimation from the indicator probabilities.

---

### BUG SIM-07: GRF panel doesn't load sill/nugget from variogram (HIGH)

**Location:** grf_panel.py `_apply_variogram_results` lines 83-126  
**Impact:** Users who load variogram get correct ranges but must manually set sill and nugget

The `_apply_variogram_results` method loads model_type and ranges from the variogram registry, but completely ignores sill and nugget:
```python
# Sets: cov_type, rx, ry, rz
# Does NOT set: sill, nugget
```

Users who load an assisted variogram get correct ranges but sill/nugget remain at defaults (sill=1.0, nugget=0.0), which may be wrong for their data.

**Fix:** Add sill/nugget loading:
```python
nugget = p.get('nugget', 0.0)
total_sill = p.get('total_sill', p.get('sill', 1.0) + nugget)
self.sill.setValue(total_sill)
self.nug.setValue(nugget)
```

---

### BUG SIM-08: IK-SGSIM sequential mode unimplemented (MEDIUM)

**Location:** ik_sgsim.py `run_ik_sgsim` lines 130-190  
**Impact:** API and docstring document sequential mode but only independent sampling exists

The function signature has `use_sequential: bool = True` and the docstring describes "Sequential Mode" vs "Independent Mode". But the implementation always samples independently per block:
```python
for iblock in range(n_blocks):
    sample = _sample_from_cdf(thresholds, block_probs, n_samples=1, random_state=rng)
```

No random path visitation, no neighbor-aware sampling, no previously-simulated nodes as conditioning. All blocks are sampled independently from their local CDF, regardless of the `use_sequential` flag.

**Impact:** Realizations lack spatial correlation between blocks. For screening this is acceptable, but the API misleads users into thinking spatial correlation is maintained.

**Fix:** Either implement proper sequential path (using simulated neighbors as additional conditioning), or remove the `use_sequential` parameter and document honestly that this is independent sampling.

---

### BUG SIM-09: FFT-MA spherical spectral density approximation (MEDIUM)

**Location:** sgsim3d.py line 290  
**Impact:** Unconditional FFT-MA fields have slightly wrong correlation structure for spherical model

```python
if params.variogram_type == 'spherical':
    S = partial_sill * a**3 / (1 + (a * K)**2)**2  # ← NOT the spherical spectral density
```

The spherical variogram's spectral density is analytically complex (involves Bessel functions). The formula used here is closer to a Matérn ν=2 spectral density. This means FFT-MA fields labeled "spherical" actually have a slightly different covariance structure than the target spherical variogram.

**Impact:** For most practical purposes, the approximation is adequate (the conditioning step corrects near data). But for unconditional simulations or regions far from data, variogram reproduction won't match the target perfectly.

**Fix:** Use the exact spherical spectral density (Bessel function based), or document that "spherical" in FFT-MA mode uses an approximation.

---

### BUG SIM-10: CoSGSIM structured residual ravel truncation (MEDIUM)

**Location:** cosgsim3d.py line 363  
**Impact:** If block model has fewer blocks than grid cells, residual field may have wrong spatial mapping

```python
return structured_field.ravel(order='F')[:n_blocks]
```

The FFT generates a field of shape (nz, ny, nx). It's raveled with F-order (which gives ix-slowest, iz-fastest ordering) then truncated to `n_blocks`. But if `n_blocks < nx*ny*nz`, the truncated indices don't map cleanly to block model coordinates. The first n_blocks values in F-order correspond to a specific spatial subregion, not necessarily matching the block model's coordinate ordering.

**Fix:** Instead of truncation, use proper grid-to-block mapping (same as primary variable uses at lines 573-581).

---

## WHAT'S DONE WELL

### Architecture & Design
- **Gaussian gates:** SGSIM enforces normal-score validation (mean ≈ 0, std ≈ 1) before simulation — prevents the most common user error
- **Seed enforcement:** JORC/SAMREC mandatory reproducibility gate rejects seedless simulation
- **Back-transform compliance:** W-002 gate ensures realizations are never used in Gaussian space for metal/tonnage
- **Lineage metadata:** Every engine records variogram hash, source data hash, execution timestamp, and audit version
- **Method documentation:** Every file includes references, use cases, and algorithm description

### SGSIM Engine
- **Dual-path architecture:** FFT-MA for speed (O(N log N)) + sequential for accuracy — user selects
- **Parallel realizations:** ThreadPoolExecutor with configurable n_jobs, throttled progress callbacks
- **Numba kernel:** Search template instead of KDTree gives 100x speedup for sequential path
- **Cholesky caching:** Cache key from rounded neighbor distances avoids redundant factorizations
- **Anisotropy:** Proper coordinate transformation before simulation, consistent across all paths
- **SGSIM panel sill handling:** Correctly checks for 'total_sill' key and reconstructs from partial + nugget

### Turning Bands
- **Fibonacci lattice directions:** Quasi-uniform sphere coverage, better than random for fixed n_bands
- **1D covariance derivation:** Correct turning bands covariance for spherical model (C₁(h) = C₃(h) - h·dC₃/dh)
- **SK conditioning:** Proper residual kriging instead of IDW, using transformed coordinates for anisotropy

### Co-Simulation (CoSGSIM)
- **Markov Model 1:** Correct implementation of Y_sec = ρ·Y_pri + √(1−ρ²)·R
- **Structured residual:** FFT-generated residual with correct spatial correlation (not white noise)
- **Hard data freezing:** Secondary hard data values preserved exactly at sample locations
- **PCHIP back-transform:** Preserves tail behavior vs. linear interpolation
- **Correlation gate:** Validates ρ from cross-variogram before simulation

### GRF
- **Multiple methods:** FFT (fast, large grids) + Cholesky (exact, small grids) + spectral
- **Eigendecomposition fallback:** When Cholesky fails, graceful fallback to eigendecomposition

### DBS
- **Block variance computation:** Numerical integration of within-block variogram (2×2×2 quadrature)
- **Block-support kriging:** Uses regularized variogram with correct block variance as sill

### MPS
- **Pattern database:** Efficient tree-based pattern matching with configurable template size
- **No variogram dependency:** Correctly documented as variogram-free method

---

## SGSIM `order='F'` — NOT A BUG (Clarification)

The SGSIM FFT-MA conditioning uses `ravel(order='F')` on the unconditional field of shape `(nz, ny, nx)`. Unlike the kriging engines (where `order='F'` was a bug), here it is **intentional and correct**:

- `uncond_field` shape: `(nz, ny, nx)`. F-order ravel: iz varies fastest → flat index = ix·(ny·nz) + iy·nz + iz
- `grid_coords` from `meshgrid(indexing='ij')` shape `(nx, ny, nz)`. C-order ravel: iz varies fastest → flat index = ix·(ny·nz) + iy·nz + iz

These are **identical** mappings. F-order of (nz, ny, nx) ≡ C-order of (nx, ny, nz). The SGSIM code deliberately uses this equivalence.

The reshape at line 425 (`conditioned.reshape(nz, ny, nx, order='F')`) is also correct for the same reason.

**This is in contrast to the kriging engines**, where results are computed in C-order from `meshgrid(indexing='ij')` but reshaped with `order='F'` into a DIFFERENT target shape `(nx, ny, nz)`, causing spatial misalignment.

---

## CROSS-SUBSYSTEM SILL CONVENTION STATUS

| Engine | What 'sill' means | Variogram func expects | Status |
|--------|-------------------|----------------------|--------|
| SGSIM variogram funcs | Total sill | Total sill | ✅ Correct |
| SGSIM FFT-MA | Total sill (computes partial internally) | — | ✅ Correct |
| SGSIM sequential | Total sill (C0 = params.sill) | Total sill | ✅ Correct |
| sgsim_engine.py Numba | **Expects partial, receives total** | Partial sill | ❌ **SIM-01** |
| SGSIM panel | Sends total sill (checks 'total_sill' key) | — | ✅ Correct |
| CoSGSIM | Passes prim_vario['sill'] directly | Total sill | ⚠️ **SIM-05** |
| SIS | Total sill (p·(1-p)) | Total sill | ✅ Correct |
| Turning Bands | Total sill (config.sill) | Total sill | ✅ Correct |
| GRF | Total sill (config.sill) | Total sill | ✅ Correct |
| DBS | Total sill (config.sill) | Total sill | ✅ Correct |
| TB panel | Correctly reconstructs total_sill | — | ✅ Correct |
| DBS panel | Correctly reconstructs total_sill | — | ✅ Correct |
| SIS panel | Correctly reconstructs total_sill | — | ✅ Correct |
| GRF panel | **Doesn't load sill at all** | — | ⚠️ **SIM-07** |

---

## PRIORITY FIX ORDER

| Priority | Bug | Severity | Effort | Rationale |
|----------|-----|----------|--------|-----------|
| 1 | SIM-02 | CRITICAL | Low | GRF conditioning broken — one-line fix |
| 2 | SIM-03 | CRITICAL | Medium | SIS physically impossible states |
| 3 | SIM-01 | CRITICAL | Low | Numba kernel sill inflation — one constant change |
| 4 | SIM-04 | HIGH | Low | CoSGSIM half-cell offset — arithmetic fix |
| 5 | SIM-05 | HIGH | Low | CoSGSIM sill passthrough — add convention check |
| 6 | SIM-06 | HIGH | Medium | SIS grade reconstruction — replace midpoint logic |
| 7 | SIM-07 | HIGH | Low | GRF panel sill loading — add 2 lines |
| 8 | SIM-08 | MEDIUM | High | IK-SGSIM sequential mode — either implement or remove claim |
| 9 | SIM-09 | MEDIUM | Medium | FFT-MA spherical spectral density — Bessel function |
| 10 | SIM-10 | MEDIUM | Medium | CoSGSIM residual mapping — replace truncation with grid mapping |

---

## CUMULATIVE BUG COUNT (ALL AUDITS)

| Subsystem | Critical | High | Medium | Total |
|-----------|----------|------|--------|-------|
| Variogram (V-series) | 1 | 2 | 3 | 6 |
| OK/SK/UK Kriging (K-series) | 2 | 3 | 2 | 7 |
| IK/CoK/Bayesian (A-series) | 4 | 3 | 3 | 10 |
| **Simulation (SIM-series)** | **3** | **4** | **3** | **10** |
| **TOTAL** | **10** | **12** | **11** | **33** |

The simulation subsystem is noticeably cleaner than the kriging subsystem. The architecture is more mature, with proper validation gates, correct sill convention in most engines, and self-consistent memory ordering. The bugs that exist are either edge-case (SIM-01 only matters with non-zero nugget), localized to a specific engine (SIM-02 in GRF only), or algorithmic rather than plumbing issues (SIM-03, SIM-06).
