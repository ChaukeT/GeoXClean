# ARBF Technical Audit

## Post-Fix Addendum

Code changes were applied after the original audit to remove the hidden accuracy floor, make subdomain factorisation retries explicit, treat discretized block estimation as the primary block-support path, and separate returned posterior variance from PUM stitching variance.

Re-run status after the fix set:

- Reproducibility is exact across repeated runs with different global NumPy seeds.
- Coverage is `1.00` across all synthetic cases.
- The returned `variances` are now posterior variances only; stitching disagreement is exposed separately in diagnostics and result fields.
- `change_of_support` no longer silently rescales uncertainty when discretized block support is already active.
- Spatial CV remains materially more honest than the original workflow, but difficult clustered and sparse cases still perform poorly, so the workflow is improved rather than fully solved.

Current verdict: partially sound but still flawed. The implementation is materially more defensible than the audited baseline, but it is still not equivalent to a fully coherent kriging workflow with a single covariance model, explicit support model, and rigorously calibrated uncertainty.

Second refactor tranche:

- Default estimation is now `local_neighbourhood_gpr`, not PUM blending.
- The estimator now uses a single global covariance model with anisotropic moving-neighbourhood solves.
- Classification geometry and spatial CV folds now operate in global anisotropic search space instead of isotropic Euclidean space.
- Legacy PUM and affine CoS remain available only as explicit legacy modes for comparison.

Third refactor tranche:

- The estimator now exposes explicit domain policy guardrails. `domain_policy='require'` fails fast when hard domains are missing, and hard-domain state is written into diagnostics and the audit record.
- Drift handling is no longer a blind static knob. `drift_type='auto'` compares constant versus linear drift by spatial LOO-CV and records both the requested and effective drift model.
- Local linear drift is now neighbourhood-aware. Where a local neighbourhood is too small or rank-poor for a stable linear solve, the estimator falls back locally to constant drift instead of producing singular systems.
- The synthetic audit now includes panel-support reproduction metrics and a hard-boundary domaining stress test.

Re-run status after the third tranche:

- Coverage remains `1.00` across the main synthetic cases.
- Reproducibility remains exact.
- Discretized block support remains materially better than affine legacy support handling.
- Hard domaining produces a large contact improvement: contact-zone RMSE dropped from `0.4354` without domains to `0.0279` with hard domains in the synthetic contact test.
- Panel-support reproduction is now explicit in the audit output. Core stationary and trend cases remain strong at panel scale, but the boundary case is still weak (`panel R² ≈ 0.2931`), so the workflow is not yet decision-grade for aggressive extrapolation.

Current verdict after the third tranche: materially more defensible, still partially sound rather than fully sign-off geostatistics. The biggest remaining weaknesses are boundary/extrapolation behaviour, weak clustered/sparse CV performance, and the absence of a recoverable-resource / conditional-simulation workflow.

## Executive Summary

The ARBF implementation is not a simple RBF interpolator. It is a full workflow that attempts to do local GPR/RBF estimation, partition-of-unity blending, local variogram fitting, uncertainty, cross-validation, change-of-support, and JORC-style classification.

The core predictor can recover a smooth stationary field reasonably well when the covariance parameters are correct. That is the strongest result in its favour.

The full workflow is not scientifically defensible as implemented. The main reasons are:

- The internal CV does not validate the same model that is used for estimation.
- Anisotropy is ignored in CV, support correction, and geometric classification.
- The estimator silently censors blocks through a hard pre-filter instead of returning high-uncertainty estimates.
- Partitioning is not reproducible because the subdomain k-means path ignores the configured seed.
- Several workflow controls are inconsistent or dead: `pum_threshold` has no effect in the default `n_subdomains=0` path, and `change_of_support=True` is effectively bypassed when discretisation is active.
- There are confirmed software defects in the domain path and diagnostics.

Verdict: the core kernel regression is partially sound, but the ARBF estimation workflow as a whole is not geostatistically defensible.

## Overview of the ARBF Method

Based on `geostats/arbf/engine.py`, the intended workflow is:

1. Optional ILR and normal-score transforms.
2. Optional locally varying anisotropy field.
3. Domain partitioning into overlapping subdomains.
4. Local variogram fitting per subdomain, or reuse of a supplied global variogram.
5. Local kernel matrix assembly and factorisation.
6. Block estimation by partition-of-unity blending of local GPR predictions.
7. Cross-validation.
8. Change-of-support correction.
9. Classification from posterior variance plus geometric criteria.
10. Audit record generation.

Inputs:

- Composite coordinates and values.
- Block centroids and block sizes.
- Kernel type, alpha, nugget, sill, anisotropy angles, anisotropy ranges.
- Partition controls such as `n_subdomains`, `max_samples`, `min_samples`, `overlap_factor`.
- Optional normal-score, ILR, LVA, classification thresholds, declustering weights, and domains.

Outputs:

- `grades`
- `variances`
- `classifications`
- `classification_names`
- optional `cv_result`, `cos_result`, `swath_data`, `audit_record`, and diagnostics

What it is mathematically doing:

- It builds covariance-style RBF kernels, not a generic deterministic thin-plate interpolation.
- In each local or global solve it augments the covariance matrix with drift terms.
- Posterior means come from `k(x)^T w + p(x)^T c`.
- Posterior variances come from `phi(0) - k^T K^{-1} k`.
- In PUM mode, local means and variances are blended by Wendland weights, and a between-model term is added as a law-of-total-variance style correction.

## Geostatistical Assessment

### What is sound

- The local/global predictor is recognisably a Gaussian-process / kriging-style covariance interpolation engine.
- The implementation avoids explicit matrix inversion and uses Cholesky/LU solves.
- The code at least attempts to separate point-support uncertainty from block-support handling.
- Domain-separated estimation is conceptually correct.

### What is not sound enough

- PUM blending breaks any single coherent covariance model. The code itself later treats the between-subdomain variance as a blending artefact rather than a real geostatistical variance.
- CV is not run through the same estimator that produces the block model.
- Anisotropy handling is inconsistent across workflow stages.
- Support correction is not consistent with the anisotropic model and is skipped in the default discretised workflow.
- Classification geometry is based on isotropic distance counts rather than the actual anisotropic neighbourhood used for estimation.
- Several hard thresholds in the normal-score path are heuristic and undocumented in terms of calibration.

### Standard geostatistical principles

- Stationarity: only partially handled. The engine offers `drift_type="linear"`, but the default workflow is effectively ordinary/simple-style stationary GPR. Non-stationarity is not diagnosed or modelled automatically.
- Spatial autocorrelation: handled in the mean predictor, but not consistently in validation and support correction.
- Variogram/covariance consistency: weak at workflow level because local subdomain models are stitched by PUM rather than embedded in one valid global covariance.
- Anisotropy: used in the main kernel, but ignored in CV (`cross_validation.py:107`), support correction (`change_of_support.py:91-98`), and geometry stats (`engine.py:2486-2512`).
- Support effects: treated heuristically and inconsistently.
- Bias and smoothing: the method is smoothing, as expected. The workflow adds extra smoothing by pre-filtering and optional affine support shrinkage.
- Uncertainty quantification: only partly meaningful. Returned variances mix local posterior uncertainty with a blending disagreement term.
- Validation methodology: not adequate for auditing the actual estimator.

## Synthetic Data Design

Audit runner: `scripts/arbf_method_audit.py`

Saved results: `audit_logs/arbf_method_audit_20260319.json`

Design choices:

- I generated synthetic fields with known truth using the same spheroidal covariance family the code uses, so the tests do not blame the method for variogram misspecification.
- Core spatial tests used the actual ARBF workflow path: `n_subdomains=0`, `variogram_mode='global'`, known true parameters, `change_of_support=False`, `discretisation_density=1`.
- A separate support experiment used coarse blocks and `discretisation_density=27`.

Cases:

1. Smooth stationary field.
2. Anisotropic field.
3. Non-stationary trend plus residual.
4. Clustered sampling.
5. Sparse sampling.
6. Noisy observations.
7. Boundary / extrapolation stress.

## Experimental Results

### Core cases

| Case | Coverage | RMSE | Bias | True-field R2 | Engine CV R2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth_stationary | 1.00 | 0.2703 | 0.0161 | 0.8903 | 0.9413 |
| anisotropic_field | 0.75 | 0.4238 | -0.0586 | 0.7982 | 0.5687 |
| trend_plus_residual | 0.97 | 0.2385 | -0.0290 | 0.8984 | 0.8909 |
| clustered_sampling | 0.97 | 0.3170 | 0.0288 | 0.8494 | 0.8444 |
| sparse_sampling | 0.76 | 0.5689 | 0.2444 | 0.6324 | 0.6166 |
| noisy_observations | 1.00 | 0.4975 | -0.1409 | 0.7254 | 0.6385 |
| boundary_extrapolation | 0.86 | 1.1210 | -0.6330 | -0.6237 | 0.9840 |

Interpretation:

- The smooth stationary case shows the kernel predictor itself is capable of working.
- The anisotropic case shows the workflow loses 25% of evaluation blocks and reports a CV R2 that is much worse than the actual field reconstruction because CV ignores anisotropy.
- The boundary case is the most damaging result. Actual truth recovery is bad (`R2 = -0.6237`) while the built-in CV reports near-perfect performance (`R2 = 0.9840`). That is an audit failure.
- Sparse and clustered sampling both show error increasing strongly with distance from data.

### Error versus support from data

- Smooth stationary: far-zone RMSE `0.4513` vs near-zone `0.1905`
- Anisotropic: far-zone RMSE `0.5673` vs near-zone `0.4026`
- Sparse: far-zone RMSE `0.7762` vs near-zone `0.3998`
- Boundary: far-zone RMSE `1.4664` vs near-zone `0.6385`

The estimator is doing what any smoother does near data, but the workflow does not honestly represent how fast quality degrades away from data.

### Sensitivity and controls

- Forced global solve on the smooth stationary case gave `R2 = 0.9048`.
- Auto-PUM on the same case gave `R2 = 0.8903`.
- `pum_threshold` check: changing it from `1` to `999999` under `n_subdomains=0` produced `max_abs_diff = 0.0`. The parameter is effectively dead in that path.

### Reproducibility

Same data, same estimator seed, different global NumPy seeds:

- seed 0: `R2 = 0.8898`
- seed 1: `R2 = 0.8960`
- seed 2: `R2 = 0.8885`

Prediction differences between runs:

- max absolute difference up to `0.3942`
- mean absolute difference up to `0.0353`

That is not acceptable for an auditable estimation workflow.

### Support handling

Support experiment with coarse blocks:

- `change_of_support=False`: RMSE `0.3534`, predicted std `0.7292`
- `change_of_support=True`: RMSE `0.3534`, predicted std `0.7292`

There was no effect because the code explicitly skips affine CoS whenever block discretisation is active (`engine.py:1184-1190`). Since the default discretisation density is `27`, the advertised CoS step is mostly bypassed.

## Bugs and Process Issues Found

### Confirmed bugs

1. Dead / misleading `pum_threshold` control.
   Evidence: `engine.py:146-151`, `engine.py:754`; experiment showed zero prediction difference when changing `pum_threshold` under `n_subdomains=0`.
   Consequence: the workflow does not behave as documented.
   Fix: implement the threshold in the actual mode selection logic or remove it.

2. `n_subdomains=1` does not reliably mean single-domain.
   Evidence: `engine.py:717-726` silently rewrites `n_subdomains` when `N > max_samples`.
   Consequence: the user can request a global solve and get PUM instead.
   Fix: explicit user request must override the performance heuristic.

3. Cross-validation validates the wrong model and ignores anisotropy.
   Evidence: `engine.py:1092-1106` uses one global variogram summary; `cross_validation.py:107` uses `scale_matrix(range_, range_, range_)`.
   Consequence: CV can materially understate or overstate quality. Boundary stress produced `CV R2 = 0.9840` while actual `R2 = -0.6237`.
   Fix: run CV through the same subdomain, anisotropy, masking, and support path used for estimation.

4. Hard block pre-filter removes blocks based on isotropic geometric-mean radius and minimum neighbour count.
   Evidence: `engine.py:1956-1979`.
   Consequence: valid blocks are censored rather than estimated with large uncertainty. Coverage fell to `0.75` on the anisotropic case and `0.86` on the boundary case.
   Fix: remove the hard filter or replace it with explicit search / domain logic and high-variance outputs.

5. Reproducibility bug in subdomain creation.
   Evidence: `partition.py:264` calls `kmeans2(...)` without using the configured estimator seed (`engine.py:231`).
   Consequence: repeated runs are not stable.
   Fix: pass a deterministic RNG seeded from `self.seed`.

6. Domain classification code is wrong for uncovered blocks.
   Evidence: `classification.py:20-23` defines `UNCLASSIFIED = 0`, but `engine.py:1674` initialises uncovered blocks with code `3`.
   Consequence: numeric code says "Measured" while name says "Unclassified".
   Fix: initialise with `UNCLASSIFIED`, not `3`.

7. `n_blocks_estimated` is counted incorrectly.
   Evidence: `engine.py:1621` and `engine.py:2640` use `np.sum(grades != 0)`.
   Consequence: zero-grade blocks are treated as unestimated, while `NaN != 0` is counted as estimated.
   Fix: count finite grades with `np.isfinite(grades)`.

8. Silent accuracy floor alters the fitted covariance model.
   Evidence: `engine.py:734-738`.
   Consequence: every zero-nugget audit run is silently given an extra diagonal term of `0.01 * sill`.
   Fix: make this explicit, opt-in, and user-visible as a nugget / regularisation parameter.

9. Change-of-support is effectively bypassed in the default discretised workflow.
   Evidence: `engine.py:1184-1190`.
   Consequence: the advertised CoS step is not actually applied in the common case.
   Fix: define one support strategy and execute it consistently.

### Likely issues

1. Support correction ignores anisotropy and orientation.
   Evidence: `change_of_support.py:91-98` uses Euclidean distance and one scalar range.
   Consequence: block-support variance is wrong whenever the fitted structure is anisotropic.

2. Geometric classification ignores anisotropy and domains.
   Evidence: `engine.py:2486-2512` uses isotropic radii based on `range_max`.
   Consequence: Measured / Indicated / Inferred categories are not tied to the actual search geometry.

3. Normal-score workflow uses hard-coded uncertainty thresholds.
   Evidence: `engine.py:1166` masks blocks at `sigma^2 / sill > 0.50`; `transforms.py:250` switches GH back-transform at `sigma^2 / sill < 0.30`.
   Consequence: discontinuities and undocumented behaviour.

4. Automatic block-coordinate shifting is too aggressive.
   Evidence: `engine.py:509-519`.
   Consequence: a coordinate system problem can be silently converted into a translated model.

5. Domain assignment by nearest composite is unsafe across faults or sharp contacts.
   Evidence: `engine.py:1664-1669`.
   Consequence: blocks can be assigned to the wrong hard domain when block domain labels are absent.

### Open questions

1. The local variogram fit is omnidirectional. The code does not estimate directional anisotropy locally.
2. Returned `variances` still include a PUM disagreement term even though the code later treats that term as non-physical for classification.
3. The local variogram objective is custom rather than a standard fully documented kriging fit workflow.

## Risk Assessment

- High scientific risk: CV, uncertainty, support, and classification outputs can look plausible while being statistically invalid.
- High audit risk: reproducibility is not guaranteed.
- High operational risk: boundary and sparse zones can be systematically misrepresented.
- Medium algorithmic risk: the core mean predictor is usable on simple stationary problems with known parameters, but the surrounding workflow degrades credibility.

## Recommended Fixes and Improvements

1. Make the model-selection logic honest.
   Implement `pum_threshold` properly.
   Respect explicit single-domain requests.

2. Replace the current CV with estimator-faithful CV.
   Use the same anisotropy, subdomaining, masking, and domain rules as production estimation.

3. Remove the hard block pre-filter.
   If blocks are weakly informed, estimate them and return high uncertainty.

4. Make partitioning reproducible.
   Seed k-means from the estimator seed and persist centres in the audit record.

5. Fix the domain path.
   Correct classification codes, propagate full result metadata, and avoid nearest-composite domain assignment when hard domains matter.

6. Rework support correction.
   Use anisotropic support integration.
   Do not advertise an affine CoS step that is skipped under the default block-kriging path.

7. Make uncertainty outputs physically meaningful.
   Separate true posterior variance from PUM stitching diagnostics.

8. Remove silent parameter rewriting.
   Accuracy floors and other regularisation changes must be explicit in both UI and audit output.

## Final Verdict

The ARBF workflow is **not geostatistically defensible** in its current form.

More precise statement:

- The underlying covariance-based RBF / GPR predictor is **partially sound**.
- The implemented workflow around it, especially validation, uncertainty, support handling, reproducibility, and domain bookkeeping, is **flawed enough to invalidate the end-to-end process** for scientific or reporting use.
