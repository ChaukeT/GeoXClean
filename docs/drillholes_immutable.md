**Drillholes Immutable**

Purpose: Policy and recommended patterns to treat drillhole source data as immutable.

Policy
- Never edit imported drillhole source files in-place. Preserve originals in read-only storage and register them in DataRegistry with a checksum.
- Any cleaning, normalization or correction must produce a new derived dataset and reference the original dataset version in provenance.

Correction workflow
1. Import original file and register as `draft`.
2. Create a derived dataset applying fixes (coordinate correction, depth fixes) and register with metadata describing changes.
3. Validation tests run; if acceptable, steward promotes derived dataset to `validated` or `authoritative` as appropriate.

Rationale
- Immutability preserves auditability, legal defensibility and reproducibility of resource estimates.
