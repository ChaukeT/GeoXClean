**Determinism Rules**

Purpose: Define how reproducibility and determinism are enforced across GeoX computations.

Rules
- All stochastic operations must accept a `random_state` or `seed` parameter and persist that value in run metadata.
- Deterministic defaults: When `random_state` is not provided, use a documented default seed and mark the run as non-reproducible.
- Floating point determinism: Document that bitwise identical results may vary across platforms or BLAS implementations; full reproducibility requires pinned runtime environment.
- Algorithm versioning: Every algorithm must report a version identifier so identical code and parameters can be matched.

Testing
- Unit tests for deterministic components must set explicit seeds and assert stable output across runs.
