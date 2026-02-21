**GeoX Architecture**

Purpose: Provide a concise architecture overview for GeoX, defining the main layers, responsibilities, and integration points.

1. Layers
- Presentation / UI: `block_model_viewer/ui/`, `block_model_viewer/controllers/` — handles user interactions, visualization, and orchestration of user-driven workflows.
- Application / Orchestration: Controllers and app-level services (`app_controller.py`, `data_controller.py`) — coordinate domain engines and UI, manage jobs and user sessions.
- Domain / Engines: `block_model_viewer/geostats/`, `geomet/`, `geomet_chain/` — pure computation: interpolation, simulation, optimization, modelling.
- Persistence & Data: `block_model_viewer/drillholes/`, `core/data_registry.py`, `core/data_provenance.py` — import/export, validation, audit trails, in-memory/dataframe containers.
- Packaging / Ops: `GeoX.spec`, `installer/` — packaging, installers, deployment and release automation.

2. Key Principles
- Separation of concerns: domain engines must not contain UI code; UI uses well-defined APIs to request work and consume results.
- Data-first design: core data models (block models, drillholes) are canonical and serializable; transformations are recorded with provenance.
- Determinism and auditability: algorithm parameters and RNG seeds are captured on every run; results must be reproducible given the same inputs.

3. Integration Patterns
- Controller -> Engine: synchronous (blocking) for small jobs, asynchronous (job queue/worker) for long-running jobs (`job_registry.py`, `job_worker.py`).
- Data flow: files -> `DataIO` -> `DrillholeDatabase` / block model -> engine -> results -> renderer/export.
- Persistence: in-memory DataFrames for interactive sessions; explicit export to files or DB for authoritative storage.

4. Security & Deployment
- Desktop-first: no required network services by default; telemetry / crash reporting must be opt-in.
- Release packaging: signed installers for Windows/macOS; use CI for reproducible builds.
