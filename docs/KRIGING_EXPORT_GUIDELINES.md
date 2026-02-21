# Kriging Export Guidelines for Listed Companies
## What to Export (and What NOT to Export) for JORC/NI 43-101 Compliance

**Date:** 2026-02-07
**Purpose:** Professional guidance on exporting OK results for audit-ready, listing-grade workflows

---

## Executive Summary

**Key Principle:** For a listed-company audit, **QA metrics ARE the product**, not just nice-to-haves.

> "You should not export a single 'OK estimate' table and call it done. What you export must reflect what OK can defendably support given your variogram and QA results."

---

## 1. What You MUST Export (Non-Negotiable)

These are **required** for technical review, audit, and sign-off.

### A. Ordinary Kriging Estimates (Primary Output)

**Export Columns:**
```
X, Y, Z                 # Block centroid coordinates
FE_OK                   # Block estimate (or {variable}_est)
OK_VARIANCE             # Kriging variance (even if nugget-dominated)
N_SAMPLES               # Number of samples used
PASS_NUMBER             # Search pass (1, 2, 3, ...)
DIST_NEAREST            # Distance to nearest sample
```

**Why:**
- ✓ OK estimate is still the best linear unbiased estimator
- ✓ Variance communicates confidence **limitations** (not confidence itself)
- ✓ Pass number and distance show where extrapolation occurred
- ✓ Even with nugget dominance, this is valid **if interpreted correctly**

---

### B. QA Metrics (WHERE YOUR SYSTEM SHINES)

**You must export these, not just visualize them.**

**Export Columns:**
```
KRIGING_EFFICIENCY      # KE (0-1, target >0.3 for informed blocks)
SLOPE_OF_REGRESSION     # SoR (target 0.8-1.2)
NEG_WEIGHT_PCT          # % negative weights (flag if >20%)
PASS_NUMBER             # Which search pass succeeded
N_SAMPLES               # Neighborhood size
DIST_NEAREST            # Nearest sample distance
```

**Why:**
- ✓ These replace "hand-waving confidence"
- ✓ A Competent Person will trust this **more than a pretty map**
- ✓ Enables hard thresholds for classification
- ✓ **Critical:** Proves where OK is informed vs. extrapolated

📌 **For a listed company, QA metrics are MORE IMPORTANT than the estimate itself.**

---

### C. Variogram Signature + Search Strategy

**Export as metadata** (not per-block values).

**Include in Metadata:**
```json
{
  "variogram_signature": "39ed13111437",
  "variogram_params": {
    "model_type": "spherical",
    "nugget": 0.95,
    "sill": 1.0,
    "range": [150, 100, 50],
    "anisotropy": {...}
  },
  "search_strategy": {
    "mode": "multi_pass",
    "passes": [
      {"pass": 1, "min": 8, "max": 12, "ellipsoid": 1.0},
      {"pass": 2, "min": 6, "max": 24, "ellipsoid": 1.5},
      {"pass": 3, "min": 4, "max": 32, "ellipsoid": 2.0}
    ]
  },
  "timestamp": "2026-02-07T14:23:45",
  "software": "GeoX v1.0",
  "user": "geologist_name"
}
```

**Why:**
- ✓ Prevents silent model drift
- ✓ **Required** for JORC/SAMREC audit trail
- ✓ Proves estimate lineage
- ✓ Enables reproducibility

---

## 2. What You SHOULD Export (Strongly Recommended)

### A. Declustering Comparison

Export a **summary CSV**, not block-level.

**Include:**
```
Declustered Mean:   0.7189
OK Mean:            0.7234
Bias (%):           +0.63%
Status:             PASS (within ±5%)
```

**Why:**
- ✓ This is the **first reconciliation check** reviewers ask for
- ✓ You already have the data
- ✓ Proves OK didn't introduce bias

---

### B. Search Diagnostics Grids (Optional but Powerful)

If space allows, export:
```
SEARCH_RADIUS_USED      # Actual search distance per block
ELLIPSOID_SCALE_USED    # Multiplier used (1.0, 1.5, 2.0)
```

**Why:**
- ✓ Shows where model is stretched beyond comfort
- ✓ Supports statements like: "84% of blocks required only pass-1 search"

---

## 3. What You Should Export ONLY If Justified

### OK Variance Maps

You **can** export them, but you **must label them correctly**.

**Use Wording Like:**
> "Kriging variance primarily reflects nugget-dominated uncertainty and should not be interpreted as spatial continuity strength, drill spacing adequacy, or classification confidence."

**Never Imply:**
- ❌ Classification suitability
- ❌ "Confidence" or "certainty"
- ❌ Drill spacing adequacy

**Reality Check:**
- With 95% nugget, variance is **flat**
- It reflects randomness, not structure
- SGSIM spread is the real uncertainty measure

---

## 4. What You Should NOT Export (or De-Emphasize)

### ❌ "High-Resolution OK Grade Maps"

**Why NOT:**

Given your nugget:
- They **look** precise
- They are **not** precise

> "This is how companies get burned."

**Problem:**
- Smooth gradational maps imply continuity that doesn't exist
- Management sees "confidence" where there is only smoothing
- Investors get misled

**Alternative:**
- Export OK estimates with **KE flags** overlaid
- Show SGSIM P10/P50/P90 maps instead
- Annotate: "Smoothed estimate - see QA metrics for reliability"

---

### ❌ OK-Based Resource Classification

**With:**
- Nugget ≈ 95%
- Flat variance
- Extreme anisotropy

**Classification MUST NOT be driven by OK variance.**

**Use Instead:**
- SGSIM spread (P90 - P10)
- Drill spacing
- QA metrics (KE, SoR, Pass)
- Geological confidence

**Correct Classification Logic:**
```
IF (KE > 0.5 AND Pass == 1 AND Drill_Spacing < 50m):
    Classification = "Indicated"
ELIF (KE > 0.3 AND Pass <= 2):
    Classification = "Inferred"
ELSE:
    Classification = "Unclassified"
```

Not based on variance!

---

## 5. How This is Typically Packaged for a Listed Company

### Deliverables Set

**1. Block Model (CSV / Parquet / VTK)**
```
Columns:
  X, Y, Z
  FE_OK, OK_VARIANCE
  KE, SoR, NSamples, Pass, MinDist, NegWt%
  [SGSIM P10, P50, P90 - optional]

Metadata:
  Variogram signature
  Search strategy
  QA summary
  Timestamp, software version
```

**2. Technical Appendix (PDF / Markdown Report)**
```
Contents:
  - Variogram plots (experimental + model)
  - QA histograms (KE, SoR, negative weights)
  - Pass usage statistics
    "84% estimated on Pass 1 (tight search)
     11% required Pass 2 (relaxed search)
     5% required Pass 3 (fallback)"
  - Mean reconciliation table (OK vs declustered vs SGSIM)
  - Search ellipsoid visualization
```

**3. SGSIM Companion Model**
```
Columns:
  X, Y, Z
  SGSIM_P10, SGSIM_P50, SGSIM_P90
  SGSIM_SPREAD (P90 - P10)

Purpose:
  - Risk assessment
  - Classification support
  - NOT for mean grade (use OK for that)
```

---

## 6. The One Sentence You Should Be Able to Write

> "Ordinary kriging was used to generate an unbiased local mean estimate, supported by multi-pass search diagnostics and QA metrics. Due to nugget-dominant variability, uncertainty and classification were assessed using conditional simulation rather than kriging variance alone."

**That sentence passes committees.**

---

## 7. Column Naming Standards

**GeoX Current Implementation:**

| Full Name                 | GeoX Column       | Industry Standard |
|--------------------------|-------------------|-------------------|
| Kriging Estimate         | `CU_est`          | `FE_OK`, `CU_OK`  |
| Kriging Variance         | `CU_var`          | `VAR_OK`, `CU_VAR`|
| Kriging Efficiency       | `CU_KE`           | ✓ Good            |
| Slope of Regression      | `CU_SoR`          | ✓ Good            |
| Number of Samples        | `CU_NSamples`     | ✓ Good            |
| Pass Number              | `CU_Pass`         | ✓ Good            |
| Min Distance to Sample   | `CU_MinDist`      | ✓ Good            |
| % Negative Weights       | `CU_NegWt%`       | ✓ Good            |

**All column names are professional-grade.**

---

## 8. Metadata Export Example

**What Gets Saved with Block Model:**

```json
{
  "method": "ordinary_kriging",
  "variable": "CU",
  "variogram_signature": "39ed13111437",
  "variogram_model": "spherical",
  "variogram_params": {
    "nugget": 0.95,
    "sill": 1.0,
    "range": [150, 100, 50],
    "anisotropy": {
      "major_range": 150,
      "minor_range": 100,
      "vertical_range": 50,
      "azimuth": 45,
      "dip": 0,
      "plunge": 0
    }
  },
  "search_strategy": {
    "mode": "multi_pass",
    "compute_qa_metrics": true,
    "passes": [
      {"pass_number": 1, "min_neighbors": 8, "max_neighbors": 12, "ellipsoid_multiplier": 1.0},
      {"pass_number": 2, "min_neighbors": 6, "max_neighbors": 24, "ellipsoid_multiplier": 1.5},
      {"pass_number": 3, "min_neighbors": 4, "max_neighbors": 32, "ellipsoid_multiplier": 2.0}
    ]
  },
  "qa_summary": {
    "kriging_efficiency_mean": 0.645,
    "kriging_efficiency_min": 0.112,
    "slope_of_regression_mean": 0.998,
    "pct_negative_weights_max": 18.7,
    "pass_1_count": 42150,
    "pass_2_count": 5234,
    "pass_3_count": 1148,
    "unestimated_count": 1468
  },
  "timestamp": "2026-02-07T14:23:45.123Z",
  "software": "GeoX Block Model Viewer v1.0",
  "data_source_type": "composites",
  "samples_used": 8423,
  "grid_size": [100, 80, 60],
  "n_blocks": 48532
}
```

**This metadata is now automatically included in GeoX exports.**

---

## 9. What Competent Persons Will Look For

When reviewing your OK results, CPs check:

1. **Variogram Traceability**
   - ✓ Can they reproduce the variogram?
   - ✓ Is the signature/hash documented?
   - ✓ Does the variogram match the geological model?

2. **Search Strategy Justification**
   - ✓ Why 3 passes? (Industry standard)
   - ✓ Are min/max neighbors reasonable? (Yes: 8-12, 6-24, 4-32)
   - ✓ Are pass counts logged? (Yes: qa_summary)

3. **QA Metrics Thresholds**
   - ✓ What % of blocks have KE < 0.3? (Flag for review)
   - ✓ What % required Pass 3? (Should be <10%)
   - ✓ Max negative weights? (Should be <20%)

4. **Reconciliation**
   - ✓ Does OK mean match declustered mean? (±5%)
   - ✓ Does OK match SGSIM P50? (±5%)
   - ✓ Is there conditional bias?

5. **Classification Independence**
   - ✓ Is classification based on QA metrics + drill spacing?
   - ✓ Or is it (incorrectly) based on OK variance?

**GeoX now provides ALL of this automatically.**

---

## 10. Bottom Line

### Export:
- ✓ OK estimate
- ✓ OK variance (with caveats)
- ✓✓✓ **QA metrics** (KE, SoR, NSamples, Pass, MinDist, NegWt%)
- ✓ Variogram & search lineage (metadata)
- ✓ Reconciliation summary

### Do NOT Oversell OK:
- ❌ Don't imply OK variance = "confidence"
- ❌ Don't use OK variance for classification
- ❌ Don't create smooth grade maps without QA overlays

---

## Summary

**Your system is now professional precisely because it knows its limits.**

**What GeoX Now Exports (After Latest Enhancements):**

1. ✓ Full block model DataFrame with all QA columns
2. ✓ Variogram signature in metadata
3. ✓ Search strategy definition in metadata
4. ✓ QA summary statistics in metadata
5. ✓ Timestamp and data lineage in metadata
6. ✓ Professional column naming

**This is listing-grade.**

---

**Reference:** JORC Code (2012) Clause 18, 49; NI 43-101 Section 1.4, 2.3
