# Professional Kriging Usage Guide
## Enabling Multi-Pass Search & QA Metrics

---

## Quick Start: Enable Professional Mode

### Two Return Formats Available

**Pattern 1: Dict-Based (Current)**
```python
estimates, variances, qa_metrics_dict = ordinary_kriging_3d(...)
# qa_metrics is a dict with keys: kriging_efficiency, slope_of_regression, etc.
```

**Pattern 2: OrdinaryKrigingResults Object (Professional)**
```python
results = ordinary_kriging_3d_with_results(...)
# results is an OrdinaryKrigingResults dataclass
# results.kriging_efficiency, results.slope_of_regression, etc.
```

**Both patterns are fully supported. Pattern 2 matches industry-standard software.**

---

### Option 1: Using SearchConfig (Recommended)

```python
from block_model_viewer.geostats.kriging_job_params import SearchConfig
from block_model_viewer.models.kriging3d import ordinary_kriging_3d

# Create professional-standard 3-pass configuration
search_config = SearchConfig.create_professional_default(
    base_max_distance=500.0  # Optional: base search distance
)

# Extract passes list
search_passes = [
    {
        'min_neighbors': p.min_neighbors,
        'max_neighbors': p.max_neighbors,
        'ellipsoid_multiplier': p.ellipsoid_multiplier
    }
    for p in search_config.passes
]

# Run kriging with professional mode
estimates, variances, qa_metrics = ordinary_kriging_3d(
    data_coords=data_coords,
    data_values=data_values,
    target_coords=target_coords,
    variogram_params=variogram_params,
    n_neighbors=12,  # Ignored when search_passes provided
    max_distance=500.0,
    model_type='spherical',
    search_passes=search_passes,      # Enable multi-pass
    compute_qa_metrics=True           # Enable QA metrics
)

# Access QA metrics
print(f"Kriging Efficiency: {qa_metrics['kriging_efficiency']}")
print(f"Pass Numbers: {qa_metrics['pass_number']}")
print(f"Negative Weights %: {qa_metrics['pct_negative_weights']}")
```

---

### Option 2: Manual Pass Configuration

```python
# Define custom multi-pass strategy
search_passes = [
    # Pass 1: Strict (high quality)
    {
        'min_neighbors': 10,
        'max_neighbors': 16,
        'ellipsoid_multiplier': 1.0
    },
    # Pass 2: Moderate
    {
        'min_neighbors': 6,
        'max_neighbors': 24,
        'ellipsoid_multiplier': 1.5
    },
    # Pass 3: Permissive (fill gaps)
    {
        'min_neighbors': 4,
        'max_neighbors': 40,
        'ellipsoid_multiplier': 2.5
    },
]

estimates, variances, qa_metrics = ordinary_kriging_3d(
    data_coords=data_coords,
    data_values=data_values,
    target_coords=target_coords,
    variogram_params=variogram_params,
    max_distance=500.0,
    model_type='spherical',
    search_passes=search_passes,
    compute_qa_metrics=True
)
```

---

### Option 3: Legacy Single-Pass Mode (Backward Compatible)

```python
# Old API still works (no breaking changes)
estimates, variances, _ = ordinary_kriging_3d(
    data_coords=data_coords,
    data_values=data_values,
    target_coords=target_coords,
    variogram_params=variogram_params,
    n_neighbors=12,
    max_distance=500.0,
    model_type='spherical'
    # search_passes=None (default) → single-pass mode
    # compute_qa_metrics=False (default) → no QA metrics
)
```

---

## Analyzing QA Metrics

### Check Multi-Pass Performance

```python
if qa_metrics is not None:
    pass_nums = qa_metrics['pass_number']

    # Count blocks per pass
    pass_1_count = np.sum(pass_nums == 1)
    pass_2_count = np.sum(pass_nums == 2)
    pass_3_count = np.sum(pass_nums == 3)
    unestimated = np.sum(pass_nums == 0)

    total = len(pass_nums)
    print(f"Pass 1: {pass_1_count} ({pass_1_count*100/total:.1f}%)")
    print(f"Pass 2: {pass_2_count} ({pass_2_count*100/total:.1f}%)")
    print(f"Pass 3: {pass_3_count} ({pass_3_count*100/total:.1f}%)")
    print(f"Unestimated: {unestimated} ({unestimated*100/total:.1f}%)")

    # Ideal: >80% on Pass 1, <5% on Pass 3, <1% unestimated
```

### Flag Poor Quality Blocks

```python
# Get valid estimates
valid = ~np.isnan(estimates)

# Extract QA metrics for valid blocks
ke = qa_metrics['kriging_efficiency'][valid]
sor = qa_metrics['slope_of_regression'][valid]
neg_wt = qa_metrics['pct_negative_weights'][valid]

# Flag poor quality
low_ke = ke < 0.3
bad_sor = (sor < 0.8) | (sor > 1.2)
high_neg = neg_wt > 20.0

# Identify problematic blocks
problem_blocks = low_ke | bad_sor | high_neg

print(f"Blocks with low KE (<0.3): {np.sum(low_ke)} ({np.sum(low_ke)*100/len(ke):.1f}%)")
print(f"Blocks with biased SoR: {np.sum(bad_sor)} ({np.sum(bad_sor)*100/len(sor):.1f}%)")
print(f"Blocks with high negative weights: {np.sum(high_neg)} ({np.sum(high_neg)*100/len(neg_wt):.1f}%)")

# Export flagged blocks for review
flagged_indices = np.where(valid)[0][problem_blocks]
print(f"Total flagged blocks: {len(flagged_indices)}")
```

### Visualize QA Metrics in 3D

```python
import pyvista as pv

# Create structured grid
grid = pv.StructuredGrid(grid_x, grid_y, grid_z)

# Add estimates
nx, ny, nz = grid_x.shape
estimates_grid = estimates.reshape(nx, ny, nz, order='C')
grid['CU_estimate'] = estimates_grid.ravel(order='F')

# Add QA metrics
ke_grid = qa_metrics['kriging_efficiency'].reshape(nx, ny, nz, order='C')
grid['kriging_efficiency'] = ke_grid.ravel(order='F')

pass_grid = qa_metrics['pass_number'].reshape(nx, ny, nz, order='C')
grid['pass_number'] = pass_grid.ravel(order='F')

# Visualize
plotter = pv.Plotter()
plotter.add_mesh(
    grid.threshold([1, 3], scalars='pass_number'),  # Only show Pass 1-3
    scalars='kriging_efficiency',
    cmap='viridis',
    clim=[0, 1],
    show_edges=False
)
plotter.add_scalar_bar(title='Kriging Efficiency')
plotter.show()
```

---

## Variogram Signature Validation

### Enforce Variogram Consistency

```python
from block_model_viewer.geostats.variogram_model import (
    compute_variogram_signature,
    validate_variogram_consistency
)

# After variogram analysis, compute signature
approved_variogram = {
    'range': 380.0,
    'sill': 1.05,
    'nugget': 0.95,
    'model_type': 'spherical',
    'anisotropy': {
        'azimuth': 45.0,
        'dip': 0.0,
        'major_range': 380.0,
        'minor_range': 45.0,
        'vert_range': 21.0
    }
}

# Compute and store signature
approved_signature = compute_variogram_signature(approved_variogram)
print(f"Approved Variogram Signature: {approved_signature}")

# Later, before running OK:
ok_variogram = {
    'range': 380.0,
    'sill': 1.05,
    'nugget': 0.95,
    # ... (should match approved_variogram)
}

# Validate - will raise ValueError if mismatch
try:
    validate_variogram_consistency(
        variogram_params=ok_variogram,
        reference_signature=approved_signature,
        context='Ordinary Kriging',
        raise_on_mismatch=True
    )
    print("✓ Variogram validated - proceeding with OK")
except ValueError as e:
    print(f"✗ Variogram mismatch: {e}")
    # Block execution
```

---

## GeoStats Controller Integration

The controller automatically enables professional mode when `compute_qa_metrics=True` is passed:

```python
# In kriging_panel.py or similar:
params = {
    "data_df": data_df,
    "variable": "CU",
    "variogram_params": variogram_params,
    "grid_spacing": (10.0, 10.0, 5.0),
    "n_neighbors": 12,
    "max_distance": 500.0,
    "model_type": "spherical",
    "layer_name": "OK_CU",

    # Professional mode parameters:
    "search_passes": [
        {'min_neighbors': 8, 'max_neighbors': 12, 'ellipsoid_multiplier': 1.0},
        {'min_neighbors': 6, 'max_neighbors': 24, 'ellipsoid_multiplier': 1.5},
        {'min_neighbors': 4, 'max_neighbors': 32, 'ellipsoid_multiplier': 2.0},
    ],
    "compute_qa_metrics": True,  # Enable QA metrics
}

# Controller handles multi-pass + QA automatically
payload = geostats_controller._prepare_kriging_payload(params)

# QA metrics are now in the grid:
grid = payload['visualization']['mesh']
print(f"Grid properties: {grid.array_names}")
# Output: ['CU_OK_est', 'CU_OK_var', 'CU_OK_kriging_efficiency',
#          'CU_OK_slope_of_regression', 'CU_OK_n_samples', ...]
```

---

## Export QA Report

```python
import pandas as pd

# Create QA report DataFrame
qa_report = pd.DataFrame({
    'X': target_coords[:, 0],
    'Y': target_coords[:, 1],
    'Z': target_coords[:, 2],
    'ESTIMATE': estimates,
    'VARIANCE': variances,
    'KRIGING_EFFICIENCY': qa_metrics['kriging_efficiency'],
    'SLOPE_OF_REGRESSION': qa_metrics['slope_of_regression'],
    'N_SAMPLES': qa_metrics['n_samples'],
    'PASS_NUMBER': qa_metrics['pass_number'],
    'DISTANCE_TO_NEAREST': qa_metrics['distance_to_nearest'],
    'PCT_NEGATIVE_WEIGHTS': qa_metrics['pct_negative_weights'],
})

# Filter to valid estimates
qa_report = qa_report[~qa_report['ESTIMATE'].isna()]

# Flag poor quality
qa_report['FLAG_LOW_KE'] = qa_report['KRIGING_EFFICIENCY'] < 0.3
qa_report['FLAG_BAD_SOR'] = (qa_report['SLOPE_OF_REGRESSION'] < 0.8) | (qa_report['SLOPE_OF_REGRESSION'] > 1.2)
qa_report['FLAG_HIGH_NEG_WT'] = qa_report['PCT_NEGATIVE_WEIGHTS'] > 20.0
qa_report['FLAG_ANY'] = qa_report['FLAG_LOW_KE'] | qa_report['FLAG_BAD_SOR'] | qa_report['FLAG_HIGH_NEG_WT']

# Export
qa_report.to_csv('kriging_qa_report.csv', index=False)
print(f"QA report exported: {len(qa_report)} blocks")
print(f"Flagged blocks: {qa_report['FLAG_ANY'].sum()} ({qa_report['FLAG_ANY'].sum()*100/len(qa_report):.1f}%)")
```

---

## Professional Standards Checklist

Before submitting to JORC/NI 43-101 auditor:

- [ ] Multi-pass search enabled (3+ passes)
- [ ] QA metrics computed for all blocks
- [ ] Variogram signature validated
- [ ] Multi-pass performance logged (>80% Pass 1, <5% Pass 3)
- [ ] QA flags reviewed (<5% flagged blocks)
- [ ] Poor quality blocks investigated and justified
- [ ] QA report exported and included in technical report
- [ ] Reconciliation against declustered mean (±5%)
- [ ] Reconciliation against SGSIM P50 (±5%)

---

## Troubleshooting

### "Most blocks estimated on Pass 3"
**Symptom:** >20% of blocks require Pass 3 (fallback pass)

**Diagnosis:**
- Data too sparse for chosen grid resolution
- Search ellipsoid too small
- Min neighbors too high for Pass 1/2

**Solutions:**
1. Increase grid spacing (coarser grid)
2. Increase search_ellipsoid base size (max_distance)
3. Lower min_neighbors for Pass 1/2
4. Review data distribution (clustering issues?)

---

### "High percentage of negative weights"
**Symptom:** Many blocks show >20% negative weights

**Diagnosis:**
- Search neighborhood includes too many samples in clustered regions
- Screening effect: closer samples screen far samples
- Possible variogram model issue (nugget too low?)

**Solutions:**
1. Reduce max_neighbors (12 → 8)
2. Reduce search_ellipsoid size
3. Review variogram fit (especially nugget)
4. Use octant search (not yet implemented)

---

### "Low kriging efficiency"
**Symptom:** Many blocks show KE < 0.3

**Diagnosis:**
- High kriging variance relative to data variance
- Samples too far from block centers
- High nugget effect

**Solutions:**
1. This may be deposit-dependent (high-nugget deposits naturally have low KE)
2. Review variogram fit
3. Consider tighter search strategy (Pass 1 only)
4. Document and justify in technical report

---

## Performance Notes

**Multi-Pass Overhead:**
- Minimal (<5%) for deposits where most blocks estimated on Pass 1
- Worst case: ~15% slower if many blocks require Pass 3
- Trade-off: slightly slower execution for significantly better audit trail

**QA Metrics Overhead:**
- ~5% compute time increase
- Negligible compared to benefit for professional reporting

**Recommendation:**
Always enable multi-pass + QA metrics for production work.
Disable only for rapid prototyping/testing.

---

## Professional Results Object Pattern

### Using OrdinaryKrigingResults Dataclass

For professional-grade code, use the structured results object:

```python
from block_model_viewer.models.kriging_results_builder import ordinary_kriging_3d_with_results

# Returns professional OrdinaryKrigingResults object
results = ordinary_kriging_3d_with_results(
    data_coords=data_coords,
    data_values=data_values,
    target_coords=target_coords,
    variogram_params=variogram_params,
    max_distance=500.0,
    model_type='spherical',
    search_passes=[...],
    compute_qa_metrics=True
)

# Access all professional attributes
print("Core Outputs:")
print(f"  Estimates: {results.estimates}")
print(f"  Status: {results.status}")  # 0=unestimated, 1=estimated
print(f"  Kriging Variance: {results.kriging_variance}")

print("\nKriging System Attributes:")
print(f"  Kriging Mean: {results.kriging_mean}")
print(f"  Kriging Efficiency: {results.kriging_efficiency}")
print(f"  Slope of Regression: {results.slope_of_regression}")
print(f"  Lagrange Multiplier: {results.lagrange_multiplier}")

print("\nNeighbourhood Attributes:")
print(f"  Num Samples: {results.num_samples}")
print(f"  Sum Weights: {results.sum_weights}")
print(f"  Sum Negative Weights: {results.sum_negative_weights}")
print(f"  Min Distance: {results.min_distance}")
print(f"  Avg Distance: {results.avg_distance}")

print("\nSearch Attributes:")
print(f"  Search Pass: {results.search_pass}")  # 1, 2, 3
print(f"  Search Volume: {results.search_volume}")

print("\nMetadata:")
print(f"  {results.metadata}")
```

### Converting From Dict-Based Pattern

```python
from block_model_viewer.models.kriging_results_builder import build_ordinary_kriging_results

# Current pattern
estimates, variances, qa_metrics = ordinary_kriging_3d(...)

# Convert to professional object
results = build_ordinary_kriging_results(
    estimates=estimates,
    variances=variances,
    qa_metrics=qa_metrics,
    metadata={'variogram_params': variogram_params}
)

# Now use as OrdinaryKrigingResults
pass_1_count = np.sum(results.search_pass == 1)
mean_ke = np.nanmean(results.kriging_efficiency)
```

### Extracting QA Summary

```python
from block_model_viewer.models.kriging_results_builder import extract_qa_summary_from_results

# Get QA summary dict (matches controller metadata format)
qa_summary = extract_qa_summary_from_results(results)

print(qa_summary)
# {
#     'kriging_efficiency_mean': 0.645,
#     'kriging_efficiency_min': 0.112,
#     'slope_of_regression_mean': 0.998,
#     'pct_negative_weights_max': 18.7,
#     'pass_1_count': 42150,
#     'pass_2_count': 5234,
#     'pass_3_count': 1148,
#     'unestimated_count': 1468,
#     'n_valid': 48532
# }
```

### Why Use OrdinaryKrigingResults?

**Advantages:**
- ✓ Matches industry software (Leapfrog, Datamine, Isatis)
- ✓ Structured dataclass with validation
- ✓ IDE autocomplete and type hints
- ✓ Self-documenting code
- ✓ Easy to extend with new attributes
- ✓ Serializable for export

**When to Use:**
- Building professional reports
- Integrating with external software
- Creating audit-trail exports
- Long-term data persistence

**When Dict-Based is OK:**
- Quick prototyping
- Internal workflows
- Backward compatibility required

---

## Next Steps

See [KRIGING_PROFESSIONAL_ENHANCEMENTS.md](KRIGING_PROFESSIONAL_ENHANCEMENTS.md) for:
- Phase 2 features (UI updates, reconciliation)
- Professional validation checklist
- References and standards
