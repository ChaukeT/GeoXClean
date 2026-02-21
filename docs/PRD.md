**GeoX — Product Requirements Document (PRD)**

**Document Version:** 1.0

**Last Updated:** 2026-01-24

**Purpose**
Define the product requirements for the GeoX desktop application, aligned with component design documents in `docs/` and the current codebase. This PRD is intended for product, engineering, QA, and release owners.

**Scope**
- Core desktop app in `block_model_viewer/` with controllers, UI, and rendering.
- Domain engines in `geostats/`, `geology/`, `geomet/`, `geomet_chain/`, `geotech/`, `mine_planning/`, and related modules.
- Data pipeline in `drillholes/` and system services in `core/`.
- Packaging and installers in `GeoX.spec` and `installer/`.

**Out of Scope**
- Cloud services, web UIs, or server-side APIs.
- Schema changes to existing data models (unless explicitly requested).
- Any cross-layer coupling between UI and engine code.

**Goals**
- Provide a deterministic, auditable desktop workflow for geoscience modeling and visualization.
- Enable drillhole ingestion and validation, geostatistical estimation, and 3D visualization.
- Maintain clear separation of UI, controllers, and domain engines.

**Personas**
- **Geostatistician:** Builds variograms, runs interpolation/simulation, assesses uncertainty.
- **Data Steward:** Ensures data validation, compositing rules, and audit trails.
- **Mine Planner:** Visualizes block models and examines scenarios for planning.

**Primary Use Cases**
- Import drillhole data and validate/composite samples.
- Fit variograms and estimate block model attributes.
- Visualize block models and drillholes in 3D with LOD controls.
- Export results and audit trails for reporting.

**Functional Requirements**
1) Drillholes & Data Pipeline (`drillholes/`)
   - Import supported file formats (CSV and DB exports).
   - Validate coordinates, sample ranges, nulls, duplicates, and depth consistency.
   - Composite samples according to configurable compositing policies.
   - Record every transformation in audit trails and provenance logs.
   - Export derived datasets with provenance metadata.

2) Geostatistics (`geostats/`)
   - Fit experimental variograms and return model parameters.
   - Estimate block model values via interpolation (e.g., kriging, RBF).
   - Support declustering weights for clustered sample correction.
   - Provide optional stochastic simulation for uncertainty.
   - Keep algorithms isolated from UI and file I/O.

3) Rendering & Visualization (`block_model_viewer/`, `geox/rendering/`)
   - Render block models, drillholes, and wireframes in 3D.
   - Enable attribute-based coloring, slicing, clipping, and visibility toggles.
   - Provide LOD or decimation for large datasets and maintain interactive FPS.
   - Run heavy geometry building in background threads and update UI safely.

4) System Services (`core/`)
   - Track provenance and audit events for all transformations.
   - Enforce determinism and record process history for reproducibility.
   - Handle crashes safely and emit local logs.

**Non-Functional Requirements**
- **Determinism:** Same inputs and parameters yield identical outputs.
- **Auditability:** All data transformations are logged with timestamp, user, and source.
- **Performance:** Interactive rendering for medium-size models with LOD/tiling.
- **Reliability:** Fail fast on corrupted inputs with clear diagnostics.
- **Security:** Local file handling only; sanitize inputs and avoid unsafe parsing.
- **Maintainability:** Clear UI/controller/engine boundaries and minimal coupling.

**Architecture Overview**
- **UI Layer:** Panels and widgets in `block_model_viewer/ui/`.
- **Controller Layer:** Orchestrates flows in `block_model_viewer/controllers/`.
- **Domain Engines:** `geostats/`, `geology/`, `geomet/`, `geomet_chain/`.
- **Data Pipeline:** `drillholes/` parsers, validators, compositors.
- **System Services:** `core/` audit, provenance, crash handling.
- **Rendering Adapter:** Rendering integration in `block_model_viewer/` and `geox/rendering/`.

**Data Flow (High Level)**
1. Import -> Validate -> Composite (Drillholes)
2. Declustering -> Variogram -> Estimation/Simulation (Geostats)
3. Build/Update Block Model -> Render -> Export (Viewer)

**Interface Contracts**
- Controllers call domain engines through explicit APIs (see design docs).
- No engine code should depend on UI classes or UI state.
- Panels should not query data registries directly; controllers own data flow.

**Testing Requirements**
- Unit tests for variogram fitting, declustering, and compositing.
- Parser tests for malformed or edge-case input files.
- Integration tests for import -> validate -> composite -> estimate -> render.
- Performance baselines for rendering and estimation on standard datasets.

**Packaging & Deployment**
- Local run: `python run_app.py` (see `README.md`).
- Build: `pyinstaller GeoX.spec`.
- Installers: `installer/windows/` and `installer/macos/`.

**Risks & Mitigations**
- Large dataset performance: LOD, tiling, background processing.
- Invalid data imports: strong validation and warnings with partial import.
- Cross-layer coupling: enforce controller-driven data flow.

**References**
- `docs/design_geostats.md`
- `docs/design_drillholes.md`
- `docs/design_rendering.md`
- `docs/determinism_rules.md`
- `docs/data_registry_rules.md`
