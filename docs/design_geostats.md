**GeoX — Geostats Component Design**

Purpose: Describe architecture, interfaces, data flow and test points for the `geostats/` engines used for interpolation, variography and uncertainty estimation.

Overview
- Location: `geostats/`
- Responsibilities: variogram modelling, interpolation (e.g., ordinary kriging, RBF interpolation), simulation/uncertainty, declustering, variography tools.
- Consumers: controllers in `block_model_viewer/` and batch scripts that run estimation pipelines.

Core Concepts
- Data units: samples (drillhole assays), sample coordinates (X,Y,Z), attributes (grades, densities), search neighbourhood definitions, variogram models.
- Execution modes: interactive (UI-driven) and batch (scripted pipelines).

Public Interfaces (recommended)
- `estimate_block_model(block_model, samples, params) -> BlockModel` — run interpolation and return updated block model values.
- `fit_variogram(samples, attrs, bins, model_type) -> VariogramModel` — compute experimental variogram and fit parameters.
- `simulate_realizations(block_model, samples, params, n_realizations) -> Iterator[BlockModel]` — generate stochastic realizations.
- `decluster(samples, method='cell', grid=None) -> weights` — compute sample weights to correct spatial clustering.

Implementation Notes
- Keep pure algorithmic code separated from file I/O and UI glue.
- Use NumPy/SciPy for linear algebra and vectorized operations.
- For heavy compute, consider optional Cython/Numba acceleration with fallback pure-Python implementations.

Performance & LOD
- For large datasets, offer streaming and tiling: process block model in tiles and combine results.
- Allow lower-resolution previews (coarser grid) for interactive exploration.

Testing & Validation
- Unit tests for variogram fitting on synthetic datasets with known parameters.
- Round-trip tests: generate synthetic field -> sample -> estimate -> compare against truth distributions.
- Performance benchmarks for typical dataset sizes; record times in CI artifacts.

Extensibility
- Plugin points for new interpolators and variogram model types.
- Clear API for adding new search neighbourhood strategies and anisotropy corrections.

Security / Data Integrity
- Validate inputs (finite numeric values, coordinate ranges) and fail fast with informative errors.

Related: Geology Engines

The `geology/` subsystem provides implicit geological modeling capabilities:

**Standard Mode (Multi-Field)**
- Location: `geology/implicit_multidomain.py`
- Builds one SDF per domain, uses CSG boolean logic
- Good for discordant or complex geology

**Sedimentary Mode (Single-Field)**
- Location: `geology/sedimentary_solver.py`
- Builds ONE monotonic scalar field for entire stratigraphic series
- Extracts units as isosurfaces (level sets)
- Prevents numerical artifacts (blobs, islands) in layered sequences
- See `docs/sedimentary_modeling_guide.md` for full technical details

**Validation**
- Location: `geology/geological_model_validator.py`
- Pre-build, build-stage, and post-build validation gates
- `SedimentaryModelValidator` class for sedimentary-specific checks

References
- Files: `geostats/`, `geology/`, controller callers in `block_model_viewer/controllers/`.
