**GeoX Invariants**

Purpose: Record the core invariants (rules that must always hold) across the system.

Invariants
- Block model authoritative: the authoritative block model is the single source of truth for resource quantities (see `block_model_authoritative.md`).
- Drillholes immutable: original imported drillhole source data must be treated as immutable; changes produce derived datasets (see `drillholes_immutable.md`).
- No silent transformations: every change to data must be recorded in provenance with parameters and user.
- Deterministic metadata: all algorithm runs must store seed, software and algorithm version, and input dataset versions.
- Explicit ML: machine learning models or automated classification may only be used if explicitly enabled and documented (see `no_ml_unless_explicit.md`).

Enforcement
- Automated checks in the DataRegistry and CI should flag any operation that would violate these invariants.
