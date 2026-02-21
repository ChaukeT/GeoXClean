**DataRegistry Rules**

Purpose: Define governance rules and best practices for the Data Registry used in GeoX.

1. Single Source of Truth
- The DataRegistry is the authoritative index of loaded datasets for a session. Each dataset entry must include: id, source path, import timestamp, schema, checksum, owner (user), and provenance information.

2. Immutable Source Records
- Source files are not altered in-place by importers. Any normalization or cleaning operations produce new derived datasets with links to their source(s).

3. Validation Gates
- All imports must pass an automatic validation pipeline (schema checks, critical column presence, coordinate ranges). Failing validation produces a report and blocks promotion to 'authoritative' state.

4. Versioning
- Dataset entries should carry a version UUID. Re-importing the same source file produces a new version; the old version remains discoverable.

5. Provenance & Audit
- Every transformation must be recorded in the registry with: operation name, parameters, user, timestamp, input versions, output version(s), and any warnings/errors.

6. Promotion to Authoritative
- Datasets are explicitly promoted to 'authoritative' state by a steward. Only authoritative datasets should be used for final block model builds and exports for reporting.

7. Access & Lifecycle
- Lifecycle states: `draft` (session only), `validated`, `authoritative`, `deprecated`.
- Deprecated datasets remain read-only and labelled with reason and superseding dataset.

8. Category Codes vs. Display Labels
- **Codes identify geology. Labels describe it. Never confuse the two.**
- Raw data must use stable codes (e.g., "BIF", "MGT") for category identification.
- Display labels (e.g., "Banded Iron Formation") are stored separately as per-project aliases via `category_label_maps`.
- Labels are for human readability only and must never alter the underlying category codes in data tables.
- All filtering, coloring, and processing must operate on codes, not labels.