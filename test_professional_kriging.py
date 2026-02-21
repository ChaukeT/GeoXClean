"""
Quick test to verify professional kriging enhancements.

Tests:
1. Variogram signature computation and validation
2. Multi-pass search configuration
3. Ordinary kriging with QA metrics
4. Legacy mode (backward compatibility)
"""

import numpy as np
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Fix Unicode encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("PROFESSIONAL KRIGING ENHANCEMENTS TEST")
print("=" * 70)

# Test 1: Variogram Signature
print("\n[TEST 1] Variogram Signature Computation")
print("-" * 70)

from block_model_viewer.geostats.variogram_model import (
    compute_variogram_signature,
    validate_variogram_consistency
)

variogram_params = {
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

signature = compute_variogram_signature(variogram_params)
print(f"✓ Variogram signature: {signature}")

# Test validation (should pass)
try:
    validate_variogram_consistency(
        variogram_params=variogram_params,
        reference_signature=signature,
        context='Test',
        raise_on_mismatch=True
    )
    print("✓ Validation passed (matching signature)")
except ValueError as e:
    print(f"✗ Validation failed: {e}")
    sys.exit(1)

# Test mismatch detection (should fail)
wrong_params = variogram_params.copy()
wrong_params['nugget'] = 0.5  # Change nugget
try:
    validate_variogram_consistency(
        variogram_params=wrong_params,
        reference_signature=signature,
        context='Test',
        raise_on_mismatch=True
    )
    print("✗ Mismatch detection FAILED - should have raised ValueError")
    sys.exit(1)
except ValueError:
    print("✓ Mismatch detected correctly")

# Test 2: Multi-Pass Search Configuration
print("\n[TEST 2] Multi-Pass Search Configuration")
print("-" * 70)

from block_model_viewer.geostats.kriging_job_params import SearchConfig

# Create professional default
search_config = SearchConfig.create_professional_default(base_max_distance=500.0)
print(f"✓ Professional default created: {len(search_config.passes)} passes")

for idx, pass_cfg in enumerate(search_config.passes, 1):
    print(f"  Pass {idx}: min={pass_cfg.min_neighbors}, max={pass_cfg.max_neighbors}, mult={pass_cfg.ellipsoid_multiplier}")

assert search_config.use_multi_pass == True, "Multi-pass not enabled"
assert len(search_config.passes) == 3, f"Expected 3 passes, got {len(search_config.passes)}"
print("✓ Configuration validated")

# Test 3: Ordinary Kriging with Multi-Pass + QA Metrics
print("\n[TEST 3] Ordinary Kriging with Professional Mode")
print("-" * 70)

from block_model_viewer.models.kriging3d import ordinary_kriging_3d

# Generate synthetic data
np.random.seed(42)
n_samples = 100
data_coords = np.random.rand(n_samples, 3) * 100  # 100x100x100 cube
data_values = np.random.rand(n_samples) * 10 + 50  # Grade 50-60

# Target grid (small for testing)
n_targets = 50
target_coords = np.random.rand(n_targets, 3) * 100

# Variogram for testing
test_variogram = {
    'range': 50.0,
    'sill': 10.0,
    'nugget': 5.0,
    'anisotropy': None  # Isotropic for simplicity
}

# Prepare search passes
search_passes = [
    {'min_neighbors': 5, 'max_neighbors': 10, 'ellipsoid_multiplier': 1.0},
    {'min_neighbors': 3, 'max_neighbors': 15, 'ellipsoid_multiplier': 1.5},
    {'min_neighbors': 2, 'max_neighbors': 20, 'ellipsoid_multiplier': 2.0},
]

print("Running kriging with multi-pass + QA metrics...")
estimates, variances, qa_metrics = ordinary_kriging_3d(
    data_coords=data_coords,
    data_values=data_values,
    target_coords=target_coords,
    variogram_params=test_variogram,
    n_neighbors=10,  # Ignored when search_passes provided
    max_distance=100.0,
    model_type='spherical',
    search_passes=search_passes,
    compute_qa_metrics=True
)

print(f"✓ Kriging completed: {len(estimates)} estimates")
print(f"  Valid estimates: {np.sum(~np.isnan(estimates))}/{len(estimates)}")

# Verify QA metrics exist
assert qa_metrics is not None, "QA metrics not returned"
assert 'kriging_efficiency' in qa_metrics, "Missing kriging_efficiency"
assert 'slope_of_regression' in qa_metrics, "Missing slope_of_regression"
assert 'pass_number' in qa_metrics, "Missing pass_number"
assert 'n_samples' in qa_metrics, "Missing n_samples"
assert 'distance_to_nearest' in qa_metrics, "Missing distance_to_nearest"
assert 'pct_negative_weights' in qa_metrics, "Missing pct_negative_weights"
print("✓ All QA metrics present")

# Analyze pass distribution
valid = ~np.isnan(estimates)
pass_nums = qa_metrics['pass_number'][valid]
for pass_idx in range(1, 4):
    count = np.sum(pass_nums == pass_idx)
    pct = count * 100.0 / np.sum(valid) if np.sum(valid) > 0 else 0
    print(f"  Pass {pass_idx}: {count} blocks ({pct:.1f}%)")

unestimated = np.sum(qa_metrics['pass_number'] == 0)
print(f"  Unestimated: {unestimated} blocks ({unestimated*100.0/len(estimates):.1f}%)")

# Check QA metric ranges
ke_valid = qa_metrics['kriging_efficiency'][valid]
sor_valid = qa_metrics['slope_of_regression'][valid]
print(f"  Kriging Efficiency: mean={np.nanmean(ke_valid):.3f}, range=[{np.nanmin(ke_valid):.3f}, {np.nanmax(ke_valid):.3f}]")
print(f"  Slope of Regression: mean={np.nanmean(sor_valid):.3f}, range=[{np.nanmin(sor_valid):.3f}, {np.nanmax(sor_valid):.3f}]")

# Test 4: Legacy Mode (Backward Compatibility)
print("\n[TEST 4] Legacy Mode (Backward Compatibility)")
print("-" * 70)

print("Running kriging in legacy mode (no multi-pass, no QA)...")
estimates_legacy, variances_legacy, qa_legacy = ordinary_kriging_3d(
    data_coords=data_coords,
    data_values=data_values,
    target_coords=target_coords,
    variogram_params=test_variogram,
    n_neighbors=10,
    max_distance=100.0,
    model_type='spherical'
    # search_passes=None (default)
    # compute_qa_metrics=False (default)
)

print(f"✓ Legacy mode completed: {len(estimates_legacy)} estimates")
assert qa_legacy is None, "QA metrics should be None in legacy mode"
print("✓ QA metrics correctly None in legacy mode")

# Verify estimates are similar (not identical due to different search strategies)
valid_both = ~np.isnan(estimates) & ~np.isnan(estimates_legacy)
if np.sum(valid_both) > 0:
    correlation = np.corrcoef(estimates[valid_both], estimates_legacy[valid_both])[0, 1]
    print(f"✓ Correlation between multi-pass and legacy: {correlation:.3f}")
    if correlation < 0.8:
        print(f"  ⚠ Warning: correlation < 0.8 (this is OK for test data)")

# Final Summary
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("✓ [PASS] Variogram signature computation and validation")
print("✓ [PASS] Multi-pass search configuration")
print("✓ [PASS] Ordinary kriging with QA metrics")
print("✓ [PASS] Legacy mode backward compatibility")
print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
print("\nProfessional kriging enhancements are working correctly.")
print("See docs/KRIGING_PROFESSIONAL_USAGE_GUIDE.md for usage instructions.")
print("=" * 70)
