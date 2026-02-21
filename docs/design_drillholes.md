**GeoX — Drillholes & Data Pipeline Design**

Purpose: Describe data ingestion, validation, compositing, audit and storage flows for drillhole data found under `drillholes/`.

Overview
- Location: `drillholes/`
- Responsibilities: import/parse, validation, compositing, auditing, exporting and DB interactions.

Key Steps in Pipeline
- Ingest: parse supported file formats (CSV, DB exports, other geospatial formats).
- Validate: check coordinate consistency, nulls, unrealistic depths, duplicate samples.
- Compositing: transform raw samples into composite intervals per policy (see `compositing_engine.py`).
- Audit: record each transformation and user action in audit trail (`audit_trail.py`, `drillhole_audit_trail.py`).
- Persist/Export: store intermediate and final datasets to file or DB (`database.py`), with provenance metadata.

Public Interfaces (recommended)
- `import_drillholes(path, fmt=None) -> DrillholeCollection`
- `validate_drillholes(collection, rules) -> ValidationReport`
- `composite_drillholes(collection, params) -> CompositeCollection`
- `export_drillholes(collection, path, fmt)`

Data Model
- `Sample` records: `id`, `drillhole_id`, `from`, `to`, `value(s)`, `easting`, `northing`, `elevation`, `metadata`.
- `Drillhole` container with coordinate path and samples list.

Audit & Provenance
- Use `core/data_provenance.py` and `drillholes/audit_trail.py` to record source file, user, timestamp and transformation details.

Error Handling
- Fail early for corrupted files with clear messages; allow partial import with warnings for recoverable issues.

Testing
- Unit tests for parsers with typical malformed inputs and edge cases.
- End-to-end tests: sample file -> import -> validate -> composite -> export.

Security
- Sanitize file names and limit resource usage when parsing large files. Prefer streaming parsing for large inputs.
