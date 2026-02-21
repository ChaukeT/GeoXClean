**JORC / SAMREC Constraints**

Purpose: Describe reporting constraints and minimum requirements to align GeoX outputs with JORC / SAMREC reporting standards.

High-level guidance
- Use authoritative datasets for any reporting claims: only datasets promoted to `authoritative` may be used for resource statements.
- Maintain full provenance for every dataset used in reporting: inputs, transformations, algorithm parameters, and responsible users.

Required checks before reporting
- Data completeness: Ensure mandatory fields (e.g., coordinates, sample intervals) are present and pass validation.
- Sample representativity: Document sampling density, spatial coverage, and any declustering applied.
- Confidence & classification: Resource classification must be justified by data quality, density, and geostatistical outcomes—see `classification_philosophy.md`.

Auditability
- Reports must include the exact model version, software version, dataset versions, and parameter files used to generate numbers.

Legal & professional responsibility
- GeoX produces technical outputs; final JORC/SAMREC statements require qualified person review and sign-off.
