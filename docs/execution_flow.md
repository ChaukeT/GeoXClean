**Execution Flow**

Purpose: Describe the typical execution flows inside GeoX, from data ingest to reporting.

Typical end-to-end flow
1. Import: User imports drillhole/collar/assay files via UI -> DataIO performs normalization and registers dataset in DataRegistry.
2. Validation: Automatic validation runs; user reviews validation report and fixes or accepts warnings.
3. Compositing / Preprocessing: Data cleaning and compositing produce derived datasets (versions recorded).
4. Model Build: User configures block model grid and estimation parameters and submits a job.
   - Short jobs run synchronously in UI thread (quick preview).
   - Long jobs enqueue via `job_registry` and run in `job_worker` background processes.
5. Post-processing: Diagnostics, uncertainty realizations, and exports are produced and stored as dataset versions.
6. Visualization: Rendering consumes produced grids/meshes; UI provides interactive exploration and snapshot/export.
7. Promotion & Reporting: Steward promotes datasets to authoritative state; reporting and export pipelines use authoritative datasets only.

Job control
- Jobs have metadata: id, owner, parameters, priority, seeds for RNG, start/end times, status, and outputs.
- Jobs are cancelable; partial results must be recorded and flagged as incomplete.
