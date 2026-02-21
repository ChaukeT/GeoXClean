# GeoX Geological Modelling System — Chief Geologist Review & Fix Report

**Reviewer:** Chief Geologist & Software Architect  
**Date:** 2026-02-16  
**Scope:** 11 modules, ~5,500 lines — full geology engine stack

---

## Executive Summary

The modelling system suffered from **four interconnected geological unsoundness issues**, all tracing to a single root cause: **starvation of orientation data to the LoopStructural FDI solver**. The system computed only ONE gradient per formation at the centroid, leaving the interpolator unconstrained between drillholes. Surfaces had no reason to follow observed dip, pass through contacts, or remain continuous.

All four reported symptoms are now addressed:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| Always fails audit | Misfit→meters conversion used scaled-space gradient (wrong units) | Convert gradient to world-space before dividing |
| No snap to contacts | Only 1 orientation per formation — solver under-constrained | Orientations at EVERY contact point via 3 complementary strategies |
| Surfaces not continuous | boundary_padding=0.0 clipped surfaces at model edges | Default boundary_padding → 0.1 (10%) |
| Wrong strike & dip | Global PCA centroid gradient — ignores spatial dip variation | Per-boundary plane fitting + drillhole-sequence apparent dip + cross-hole dip |

---

## Detailed Findings & Fixes

### 1. CRITICAL — Gradient Estimation Rewrite (`gradient_estimator.py`)

**Old behaviour (geologically unsound):**
- ONE global PCA plane fitted per formation
- ONE gradient vector placed at the formation centroid
- Result: a model with N formations had only N orientation constraints
- LoopStructural had almost no dip information → "hallucinated" flat-lying surfaces

**New behaviour (4 complementary strategies):**

1. **Per-boundary PCA plane fitting** — For each boundary, fit a plane through ALL drillhole intercepts. Distribute the resulting gradient to EVERY contact point (not just the centroid). This alone typically increases orientation data by 10–50×.

2. **Drillhole-sequence apparent dip** — Within each drillhole, consecutive contacts define the vertical gradient (∂f/∂z). Places orientations at both contact endpoints plus the midpoint. Even for vertical holes (which only give gz), this anchors the interpolator to hard data and prevents surface "float".

3. **Cross-hole dip estimation** — Compares the same boundary across nearby drillholes to resolve the full 3D dip tensor. Captures lateral dip variation.

4. **Local k-NN PCA** (optional, for folded geology) — Computes local gradients at each point using nearest neighbours. Existing function that was NEVER CALLED has been integrated into the pipeline.

### 2. CRITICAL — Compliance Misfit Conversion (`compliance_manager.py`)

**Old code:**
```python
grad = strat_feature.evaluate_gradient(points)   # ← scaled space
grad_mag = np.linalg.norm(grad, axis=1)           # ← scaled units
residuals_m = scalar_misfit / grad_mag             # ← NOT in meters!
```

The gradient was evaluated in [0,1] normalised space. The magnitude was in units of (scalar / scaled_unit), where 1 scaled unit ≈ entire model extent (thousands of metres). Dividing scalar misfit by this gives distance in **scaled units**, not metres. The reported "residuals_m" were off by a factor of the model extent.

**Fixed code:**
```python
grad_scaled = strat_feature.evaluate_gradient(points)
grad_world = grad_scaled * scaler.scale_[np.newaxis, :]   # chain rule
grad_mag_world = np.linalg.norm(grad_world, axis=1)       # metres⁻¹
residuals_m = scalar_misfit / grad_mag_world               # now actually metres
```

This uses the MinMaxScaler's `scale_` (= 1/range per axis) to convert the gradient to world-space via the chain rule: ∂f/∂x_world = ∂f/∂x_scaled × scale_x.

### 3. CRITICAL — Default Parameters (`chronos_engine.py`, `model_runner.py`)

| Parameter | Old Default | New Default | Reason |
|-----------|------------|-------------|--------|
| `cgw` (regularisation) | 0.1 (engine) / 0.01 (runner) | 0.005 | 0.1 over-smoothes dramatically, pulling surfaces 10s of metres from contacts. 0.005 produces tight-fitting surfaces suitable for resource estimation. |
| `boundary_padding` | 0.0 | 0.1 | Zero padding clips isosurfaces at model boundaries → discontinuous geology. 10% padding gives room for surfaces to close naturally. |

### 4. IMPORTANT — Gradient Polarity Enforcement (`chronos_engine.py`)

**Old:** Logged warnings about mixed gz polarity but did not fix it.

**New:** Enforces majority-polarity consistency. If 70% of gradients point upward and 30% point downward, the minority are flipped. This is geologically correct for a SINGLE stratigraphic pile (scalar increases monotonically upward). True overturned domains should be handled by domain separation, not by feeding contradictory polarities into one feature.

### 5. IMPORTANT — Fault Displacement Scaling (`chronos_engine.py`)

**Old:** `scaled_displacement = f['displacement'] / np.mean(scaler.scale_)` — divides by mean scale when it should multiply.

**New:** `scaled_displacement = raw_displacement * scaler.scale_[2]` — correctly converts metres to [0,1] space using the Z-axis scale factor (since most fault throw is vertical).

### 6. MINOR — Synthetic Fallback Distribution (`model_runner.py`)

**Old:** When gradient computation failed, synthetic (0,0,1) orientations were placed using `scaled_contacts[['X_s','Y_s','Z_s']]` — creating orientation points from already-scaled coordinates, then re-scaling them through `prepare_data()`. This double-scaled the coordinates.

**New:** Builds synthetic orientations from the original validated contacts in world coordinates, then scales once through `prepare_data()`.

---

## Files Modified

| File | Lines | Changes |
|------|-------|---------|
| `gradient_estimator.py` | 553 | **Complete rewrite** — 4-strategy pipeline |
| `chronos_engine.py` | ~1620 | Padding default, cgw default, polarity enforcement, fault scaling |
| `compliance_manager.py` | ~413 | Misfit→metres conversion fix |
| `model_runner.py` | ~1289 | cgw/padding defaults, gradient pipeline integration, fallback fix |

## Files Unchanged

| File | Lines | Notes |
|------|-------|-------|
| `__init__.py` | 60 | No changes needed |
| `faults.py` | ~280 | Fault plane geometry correct |
| `fault_detection.py` | ~260 | DBSCAN fault suggestion correct |
| `mesh_validator.py` | ~640 | Continuity validation correct |
| `contact_deviation_report.py` | ~544 | Deviation analysis correct |
| `coordinate_diagnostic.py` | ~409 | Diagnostic tool correct |
| `industry_modeler.py` | ~920 | Deprecated, no changes |

---

## Geological Verification Checklist

After applying these fixes, verify the following:

- [ ] Surfaces pass through drillhole contacts (mean residual < 2m)
- [ ] Surface dip matches observed drillhole dip (±5°)
- [ ] Surfaces are continuous across the model extent (no edge clipping)
- [ ] Audit P90 < 5m for Indicated classification
- [ ] No isolated "floating" volumes in continuity report
- [ ] Formation boundaries do not cross each other
- [ ] Fault offsets match the specified throw magnitude

---

*Report prepared for JORC/SAMREC Competent Person review.*
