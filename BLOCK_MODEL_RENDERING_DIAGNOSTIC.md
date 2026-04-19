# GeoX Block Model Rendering Diagnostic

**Date:** 2026-04-12
**Type:** Static code audit (read-only, no code changes)
**Branch:** fix/drillhole-import-bugs

---

## 1. Block Model Data Structure

### 1.1 Where Defined

| File | Class | Purpose |
|------|-------|---------|
| `block_model_viewer/models/block_model.py` | `BlockModel` | Core data structure for all block model data |
| `block_model_viewer/models/block_model.py` | `BlockMetadata` | Provenance/source metadata dataclass |
| `block_model_viewer/models/block_model_definition.py` | `BlockModelDefinition` | Geometry-only grid definition (no grade data) |
| `block_model_viewer/models/blockmodel_advanced.py` | (functions) | IDW and NN property assignment utilities |

### 1.2 Fields Stored

**BlockModel core fields (private):**

| Field | Type | Shape | Description |
|-------|------|-------|-------------|
| `_positions` | `np.ndarray` or `None` | (N, 3) float64 | Block center X, Y, Z coordinates |
| `_dimensions` | `np.ndarray` or `None` | (N, 3) float64 | Block DX, DY, DZ sizes |
| `_properties` | `Dict[str, np.ndarray]` | each (N,) | Grade/property arrays keyed by name |
| `_rotation_matrix` | `np.ndarray` or `None` | (3, 3) | Rotation matrix for anisotropic grids |
| `_bounds` | `Tuple[float x6]` or `None` | - | (xmin, xmax, ymin, ymax, zmin, zmax), lazily computed |
| `_block_count` | `int` | - | Total number of blocks |
| `_is_orthogonal_cache` | `Optional[Tuple]` | - | Caches (bool, Optional[grid_info]) for ImageData detection |

**BlockMetadata fields:**
- `coordinate_system`, `units`, `source_file`, `file_format`, `creation_date`, `description`
- `file_checksum`, `checksum_algorithm`, `import_timestamp`
- `parser_version`, `parser_framework_version`
- `column_mapping: Optional[Dict[str, str]]`
- `inferred_dimensions: bool`

### 1.3 How Blocks Are Stored

**Columnar flat arrays.** All N blocks stored as rows in numpy arrays:
- Positions: (N, 3) array of centroids
- Dimensions: (N, 3) array of block sizes
- Properties: Dictionary of (N,) arrays, one per property name
- No explicit block IDs; array index IS the block identifier

### 1.4 How Grade/Property Arrays Are Attached

**Via `add_property(name: str, values: np.ndarray)` method:**
- Validates `len(values) == block_count`
- Auto-optimizes dtype: float64 preserved for precision; integers compressed to int8/16/32/64 based on value range
- Stored in `self._properties[name] = values`

**Specialized result methods:**
- `add_universal_kriging_result(property_name, estimates, variance)` - adds `uk_*_var`
- `add_cokriging_result(primary_name, result)` - adds `cok_*` and `cok_*_var`
- `add_indicator_kriging_result(property_name, ik_result)` - adds `ik_*_p_le_*` per threshold

### 1.5 How Domains Are Stored

Domains are stored as **regular properties** with categorical values (e.g., `'LITH'`, `'DOMAIN'`, `'MINERALIZATION'`). There is no special domain class. Domain filtering is done at query time via `get_engine_payload(grade_field, domain_field)` which returns the domain array in the payload.

### 1.6 Block Model Definition vs Block Model with Results

**Yes, there is a distinction:**
- `BlockModelDefinition` (in `block_model_definition.py`): Geometry-only. Stores `name`, `origin`, `dims` (nx,ny,nz), `block_size` (dx,dy,dz), `centres`, `ijk_indices`, `rotation_matrix`, `metadata`. No property arrays. Used as shared grid template across multiple estimation runs.
- `BlockModel`: Full data. Stores geometry + all properties. Can start empty (just positions/dimensions) and accumulate properties from estimation runs.

### 1.7 How Block Model Knows Spatial Extent and Block Sizes

- **Extent:** `bounds` property computes `(xmin, xmax, ymin, ymax, zmin, zmax)` from `min/max(position - dimension/2)` to `max(position + dimension/2)`. Accounts for rotation. Lazily cached in `_bounds`, invalidated on geometry change.
- **Block sizes:** Stored per-block in `_dimensions` (N, 3). May be uniform or varying.
- **Orthogonal detection:** `is_orthogonal(tolerance)` checks if grid is regular (constant spacing, axis-aligned, no rotation). Returns `(bool, Optional[grid_info])` where `grid_info = (origin, spacing, dimensions)`.

---

## 2. System-by-System Trace

### 2.1 Ordinary Kriging (OK)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/models/kriging3d.py` : `OrdinaryKriging3D` | `ordinary_kriging_3d()` calls Numba JIT kernel `run_kriging_kernel()` |
| **Output** | `OrdinaryKrigingResults` dataclass (`models/geostat_results.py:23-62`) | `estimates`, `kriging_variance`, `kriging_efficiency`, `slope_of_regression`, `num_samples`, `min_distance`, etc. |
| **Payload prep** | `controllers/geostats_controller.py:766-1100+` | `_prepare_kriging_payload()` |
| **Grid creation** | Same payload method | Creates `pv.RectilinearGrid` (or StructuredGrid), assigns: `grid[property_name]`, `grid[variance_property]`, `grid[f'OK_{var}_{qa_name}']` |
| **Property naming** | Convention | `OK_{variable}_estimate`, `OK_{variable}_variance`, `OK_{variable}_{qa_metric}` |
| **Signal** | `controllers/job_worker.py` | `JobWorker.finished.emit(payload)` -> `AppController._on_task_complete("kriging", payload)` -> `signals.task_finished.emit("kriging")` |
| **Domain filtering** | `geostats_controller.py:911-959` | `get_domain_mask()` pre-filters active blocks, `scatter_to_full_grid()` maps back with `fill=np.nan` |
| **Visualization** | Via shared path | `VisController.apply_results_to_model(payload)` -> `renderer.add_mesh()` |

### 2.2 Simple Kriging (SK)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/models/simple_kriging3d.py` : `SimpleKriging3D` | `simple_kriging_3d()` — same Numba kernel as OK but with global mean |
| **Output** | `SimpleKrigingResults` dataclass (`models/geostat_results.py:66-102`) | Same as OK + `global_mean` field |
| **Payload prep** | `controllers/geostats_controller.py:290-760` | `_prepare_simple_kriging_payload()` |
| **Grid creation** | Same | `grid[property_name]`, `grid[variance_property]`, `grid["SK_NN"]`, `grid["SK_StabilityFlag"]`. Arrays raveled with `order="F"` |
| **Property naming** | Convention | `SK_{variable}`, variance property, `SK_NN`, `SK_StabilityFlag` |
| **Signal** | Same as OK | `task_finished.emit("simple_kriging")` |
| **Domain filtering** | `geostats_controller.py:541-578` | Same pattern: `get_domain_mask()` -> filter -> `scatter_to_full_grid()` |
| **Differences** | Ravel order | Uses `order="F"` (Fortran order) for grid property assignment |

### 2.3 Universal Kriging (UK)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/universal_kriging.py` : `UniversalKriging3D` | `solve_uk()` with Numba kernel `_solve_single_uk_point()` using `prange` |
| **Output** | `UniversalKrigingResults` dataclass (`models/geostat_results.py:106-138`) | OK attributes + `drift_value`, `residual_estimate`, `trend_coefficients` |
| **Payload prep** | `controllers/geostats_controller.py:1108-1150+` | `_prepare_universal_kriging_payload()` |
| **Property naming** | Convention | `UK_{variable}`, `uk_*_var` |
| **Signal** | Same as OK | `task_finished.emit("universal_kriging")` |
| **Domain filtering** | Same pattern as OK | |

### 2.4 Indicator Kriging (IK)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/indicator_kriging.py` | `run_indicator_kriging_job()` function + Numba kernel `_solve_single_ik_point()` — separate OK system per threshold |
| **Output** | `IndicatorKrigingResults` dataclass (`models/geostat_results.py:142-163`) | `indicator_probability` (M, T), `local_conditional_variance` (M, T), `etype_estimate`, `median_estimate` |
| **Payload prep** | `controllers/geostats_controller.py:1206-1268+` | `_prepare_indicator_kriging_payload()` |
| **Property naming** | Convention | `ik_*_p_le_{threshold}` per threshold |
| **Signal** | Same as OK | `task_finished.emit("indicator_kriging")` |
| **Domain filtering** | Same pattern as OK | |
| **Differences** | Multiple properties per run | One probability array per threshold, stored as separate grid properties |

### 2.5 Co-Kriging (CoK)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/cokriging3d.py` : `CoKriging3D` | Markov Model 1 approximation with secondary variable interpolation |
| **Output** | `CoKrigingResults` dataclass (`models/geostat_results.py:167-198`) | Primary estimate, secondary influence weight `ws/(|wp|+|ws|)`, correlation metrics, standard OK attributes |
| **Payload prep** | `controllers/geostats_controller.py:1157-1205+` | `_prepare_cokriging_payload()` |
| **Property naming** | Convention | `cok_*`, `cok_*_var` |
| **Signal** | Same as OK | `task_finished.emit("cokriging")` |
| **Domain filtering** | Same pattern as OK | |

### 2.6 Bayesian/Soft Kriging (BayK)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/bayesian_kriging.py` : `BayesianKriging` | Function-based wrapper; modifies covariance matrix diagonal for soft data uncertainty, then delegates to base kriging method (OK/UK/IK/CoK) |
| **Output** | Same as base method + soft data weights | |
| **Payload prep** | `controllers/geostats_controller.py` | `_prepare_bayesian_kriging_payload()` |
| **Signal** | Same as OK | `task_finished.emit("bayesian_kriging")` |
| **Domain filtering** | Same pattern as OK (inherited from base method) | |

### 2.7 Sequential Gaussian Simulation (SGSIM)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/models/sgsim3d.py` : `SGSIM3D` + `block_model_viewer/models/sgsim_engine.py` | `run_realizations()` executes Numba kernel `run_sgsim_kernel()` in parallel threads |
| **Output** | `SGSIMResults` dataclass (`models/geostat_results.py:232-258`) | `realizations: List[np.ndarray]` each (nz, ny, nx), `mean_realization`, `variance_realization`, percentiles (P10, P50, P90, P95), `exceedance_volume` |
| **Payload prep** | `controllers/geostats_controller.py:1440-1600+` | `_prepare_sgsim_payload()` |
| **Grid creation** | Creates `pv.RectilinearGrid` | Mean, variance, and individual realizations stored as cell_data arrays |
| **Property naming** | Convention | `{variable}_SGSIM_MEAN`, `{variable}_SGSIM_VARIANCE`, `{variable}_SGSIM_R{n}` per realization |
| **Signal** | Same | `task_finished.emit("sgsim")` |
| **Domain filtering** | `geostats_controller.py:1521-1539` | Same `get_domain_mask()` pattern |
| **Differences** | Multiple realizations | Stores N+2 properties per run (mean + variance + N realizations). Grid is always RectilinearGrid since SGSIM works on regular grids. Back-transform from Gaussian space if normal-scored. |

### 2.8 Co-Sequential Gaussian Simulation (CoSGSIM)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/cosgsim3d.py` : `CoSGSIM3D` | Similar to SGSIM with collocated secondary variable |
| **Output** | `CoSimulationResults` dataclass (`models/geostat_results.py:353-371`) | Same as SGSIM + secondary influence metrics |
| **Payload prep** | `controllers/geostats_controller.py:2493+` | `_prepare_cosgsim_payload()` |
| **Signal** | Same | `task_finished.emit("cosgsim")` |
| **Differences** | Includes secondary variable realizations | |

### 2.9 Adaptive/Fast RBF (ARBF)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/arbf_adapter.py` : `ARBFAdapter` | Domain filter -> neighborhood search per block -> local RBF matrix assembly + LDLt factorization -> batched evaluation -> uncertainty from residual spread |
| **Supporting files** | `arbf_batch_solver.py`, `arbf_modes.py`, `arbf_estimator_definition.py`, `arbf_evaluation_job.py`, `arbf_output_writer.py` | |
| **Output** | `BatchBlockResult` per block: `sub_means`, `sub_vars` (residual variance, NOT kriging variance), `condnum`, `neff`, `n_used`, `fail_flag` | |
| **Storage** | `ARBFOutputWriter.write_to_block_model(block_model, results)` -> `block_model.add_property(name, values)` | |
| **Payload prep** | `controllers/geostats_controller.py:4638-5484+` | `_prepare_arbf_payload()` |
| **Signal** | Same | `task_finished.emit("arbf")` |
| **Differences** | Per-block local RBF solve, residual-based uncertainty (not kriging variance), PUM engine auto-dispatch for large datasets (N>1500) |

### 2.10 Inverse Distance Weighting (IDW)

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/models/blockmodel_advanced.py:365-429` | `assign_properties_idw()` function — uses `cKDTree` for neighbor search, standard IDW weights `1/d^p` |
| **Output** | Tuple `(values: np.ndarray, variances: np.ndarray)` both shape (N,) | |
| **Storage** | Returned to caller, which calls `block_model.add_property()` | |
| **Signal** | Unclear — IDW is a utility function, not a registered job. Likely called directly from a panel, not through the JobWorker system |
| **Domain filtering** | None built-in; caller must filter | |
| **Differences** | Not a registered job in `job_registry.py`. Direct function call, not queued through `JobWorker`. No `_prepare_idw_payload()` found in geostats_controller. Simpler output (no QA metrics). |

### 2.11 RBF Interpolation

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/rbf_interpolation.py` : `RBFInterpolator3D` | Uses SciPy `RBFInterpolator` with polynomial drift (constant/linear/quadratic) |
| **Output** | `estimates: np.ndarray`, `diagnostics: Dict` with MAE, RMSE, R^2 | |
| **Payload prep** | `controllers/geostats_controller.py:3895-4077+` | `_prepare_rbf_payload()` |
| **Signal** | Same | `task_finished.emit("rbf")` |

### 2.12 FastRBF

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/fastrbf_engine_v2.py` : `FastRBFEngine` / `FastRBFSolver` | N<=5000: direct LDLt; N>5000: preconditioned GMRES. Augmented system `[Phi+lambdaI P; P' 0][w;c]=[d;0]` |
| **Output** | `FittedRBF` dataclass: `weights`, `polynomial_coefficients`, `estimates`, `condition_number`, `solve_method` | |
| **Payload prep** | `controllers/geostats_controller.py:4078-4637+` | `_prepare_fastrbf_payload()` |
| **Signal** | Same | `task_finished.emit("fastrbf")` |

### 2.13 Indicator RBF

| Step | Location | Detail |
|------|----------|--------|
| **Computation** | `block_model_viewer/geostats/indicator_rbf_engine.py` : `IndicatorRBFDomainEstimator` | RBF on binary indicator -> probability field -> isosurface extraction |
| **Output** | `IndicatorRBFResult`: `prob` (nz,ny,nx) probability volume [0,1], `verts`/`faces` for isosurface, `inside_mask` boolean grid | |
| **Payload prep** | `controllers/geostats_controller.py` | `_prepare_indicator_rbf_payload()` |
| **Signal** | Same | `task_finished.emit("indicator_rbf")` |
| **Differences** | Produces both a probability volume AND a mesh isosurface | |

### 2.14 Additional Systems Found

| System | Location | Payload Method | Signal |
|--------|----------|----------------|--------|
| **Turning Bands** | `geostats_controller.py:2087-2186+` | `_prepare_turning_bands_payload()` | `task_finished.emit("turning_bands")` |
| **Direct Block Simulation (DBS)** | `geostats_controller.py:2218+` | `_prepare_dbs_payload()` | `task_finished.emit("dbs")` |
| **Gaussian Random Field (GRF)** | `geostats_controller.py:2358+` | `_prepare_grf_payload()` | `task_finished.emit("grf")` |
| **Sequential Indicator Simulation (SIS)** | `geostats_controller.py:1828-1964+` | `_prepare_sis_payload()` | `task_finished.emit("sis")` |
| **Multiple-Point Simulation (MPS)** | `geostats_controller.py:3754+` | `_prepare_mps_payload()` | `task_finished.emit("mps")` |
| **Nearest Neighbor** | `blockmodel_advanced.py:327-362` | None (utility function) | None (not a registered job) |

---

## 3. Rendering Path

### 3.1 Block Model Renderer

| Item | Detail |
|------|--------|
| **File** | `block_model_viewer/visualization/renderer/renderers/block_model_renderer.py` |
| **Class** | `BlockModelRenderer` |
| **Orchestrator** | `block_model_viewer/visualization/renderer/render_orchestrator.py` : `Renderer` class (delegates to `self._block_renderer`) |
| **Mesh builder** | `block_model_viewer/visualization/block_model_mesh_builder.py` |

### 3.2 Mesh Creation Pipeline

**Entry point:** `Renderer.load_block_model()` (line 479) -> delegates to `BlockModelRenderer.load_block_model()` (line 103)

**Step 1: Auto-detect grid type** (`block_model_mesh_builder.py`)

`is_uniform_grid()` checks:
1. Constant spacing (unique diffs equal within tolerance 1e-6)
2. Axis-aligned (no rotation matrix or identity)
3. Grid completeness (fill ratio > 1%)
4. Uniform block dimensions (std < tolerance)

**Step 2: Build mesh** (`generate_block_model_mesh()` in `block_model_mesh_builder.py:382`)

| Path | Condition | Mesh Type | Method |
|------|-----------|-----------|--------|
| **Uniform** | Regular, axis-aligned, >1% fill | `pv.ImageData` | `build_uniform_grid()` (lines 187-290) |
| **Non-uniform** | Rotated, sparse, or variable-sized | `pv.UnstructuredGrid` | `build_unstructured_grid()` (lines 293-379) |

**ImageData creation:**
```python
grid = pv.ImageData()
grid.origin = origin  # Corner of cell (0,0,0), NOT centroid
grid.spacing = (dx, dy, dz)
grid.dimensions = (nx+1, ny+1, nz+1)  # Points, not cells
# Properties: 3D arrays (nz,ny,nx) raveled to 1D in C-order -> cell_data
```

**UnstructuredGrid creation:**
- Computes 8 corners per block using half-dimensions
- Applies rotation matrix if present
- Creates VTK_HEXAHEDRON cells (type 12): `[8, v0, v1, ..., v7]`
- Properties assigned directly to cell_data

### 3.3 Scalar Array Attachment

| Stage | How |
|-------|-----|
| During mesh creation | Properties stored in `mesh.cell_data[prop_name]` — ImageData uses 3D->ravel, UnstructuredGrid uses direct assignment |
| Initial rendering | `plotter.add_mesh(mesh, scalars=property_name, ...)` — PyVista sets active scalar via `SetScalarModeToUseCellData()` |
| Property switching | `set_property_coloring()` (line 1172) — updates mapper active scalars, LUT, and scalar range WITHOUT recreating mesh |

### 3.4 Colormap / Lookup Table

**Color limits:** `_compute_clim()` (line 1115-1158) — P2/P98 percentiles, excludes NaN and sentinel values (`dtype.min` for integers)

**LUT setup:**
```python
lut = mapper.GetLookupTable()
lut.SetNumberOfTableValues(256)
lut.Build()
for i in range(256):
    rgba = cmap(i / 255.0)
    lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], rgba[3])
mapper.SetLookupTable(lut)
```

**Special cases:**
- NSR fields: divergent colormap, zero-centered `[-max_abs, +max_abs]`
- Classification: discrete 4-value LUT (Measured=0, Indicated=1, Inferred=2, Unclassified=3)
- Discrete mode: pre-generates colors by category INDEX (not value position) — fixes legend mismatch issue

### 3.5 plotter.add_mesh() Call

**Location:** `BlockModelRenderer._add_meshes_to_plotter()` (line 813)

```python
actor = plotter.add_mesh(
    mesh,
    scalars=initial_property,
    cmap=colormap_name,
    clim=color_limits,       # P2/P98
    show_edges=True,         # ImageData only; False for UnstructuredGrid
    lighting=True,
    smooth_shading=False,    # Flat shading for mining blocks
    interpolate_before_map=False,
    opacity=current_opacity
)
```

### 3.6 Actor Tracking

| Storage | Content |
|---------|---------|
| `renderer.mesh_actor` | Currently active block model VTK actor |
| `renderer.block_meshes` | Dict: `{'imagedata': pv.ImageData, 'unstructured_grid': pv.UnstructuredGrid, '_is_imagedata': bool}` |
| `renderer.active_layers[layer_name]` | `{'actor': actor, 'data': mesh, 'visible': bool, 'opacity': float, 'layer_type': 'blocks', 'current_property': str, 'current_colormap': str}` |

**Actor lifecycle:**
1. Remove previous: `plotter.remove_actor(mesh_actor)` (line 833)
2. Add new: `plotter.add_mesh()` (line 977) -> returns actor
3. Mark pickable: `mesh_actor.SetPickable(1)` (line 1047)
4. Force flat shading: `prop.SetInterpolationToFlat()` (line 1053)
5. Apply domain mask: `apply_domain_mask_transparency()` (line 1069)

### 3.7 Coordinate Transformation

**Location:** `block_model_renderer.py:375-453` (`_apply_coordinate_transform_to_meshes()`)

**Purpose:** Shift UTM coordinates to local for GPU float32 precision.

**Double-shift guard (lines 410-429):**
```
already_local = (shift_magnitude > 10,000) AND (model_center_magnitude < shift_magnitude * 0.1)
```
If already local, marks `_coordinate_shifted=True` without applying shift.

**Per mesh type:**
- `RectilinearGrid`: shift x, y, z edge arrays
- `ImageData`: shift origin
- `UnstructuredGrid`: shift points

---

## 4. Signal Chain

### 4.1 Estimation Completion to Blocks in Viewport

```
PHASE 1: Estimation runs in worker thread
  EstimationPanel._on_run() -> AppController.run_task("kriging", params)
  -> JobRegistry.get("kriging") returns _prepare_kriging_payload
  -> JobWorker(QThread).run() executes payload function
  -> Engine computes, creates PyVista grid with properties
  -> JobWorker.finished.emit(payload_dict)

PHASE 2: Main thread receives result
  AppController._on_task_complete(task, payload, callback)
  -> callback(payload)  [UI closes progress dialog — called FIRST]
  -> _result_has_visual_output(payload) checks for mesh/grid/estimated_values keys
  -> VisController.apply_results_to_model(payload)
  -> signals.task_finished.emit(task_name)

PHASE 3: Visualization applied
  VisController.apply_results_to_model(payload)
  -> Extracts mesh from payload["visualization"]["mesh"]
  -> Extracts property_name from payload["visualization"]["property"]
  -> Extracts layer_name from payload["visualization"]["layer_name"]
  -> renderer.add_mesh(mesh, scalars=property_name, name=layer_name, layer_type="analysis")
  -> renderer.set_active_layer_for_controls(layer_name)
  -> legend_manager.set_visibility(True)

PHASE 4: Renderer processes mesh
  Renderer.load_block_model() [if full block model load]
  OR
  Renderer.add_mesh() [if analysis result overlay]
  -> plotter.add_mesh(mesh, **kwargs) -> returns VTK actor
  -> Stores in active_layers dict
  -> BlockModelRenderer.apply_domain_mask_transparency(actor, mesh)
  -> plotter.render()

PHASE 5: UI updates
  Controller emits scene_updated and block_model_changed signals
  -> MainWindow._on_scene_updated() -> PropertyPanel._on_scene_changed()
  -> PropertyPanel repopulates layer combo and property combo
  -> Auto-selects newly added layer
  -> BLOCKS VISIBLE IN VIEWPORT
```

### 4.2 Two Distinct Visualization Paths

**Path A: Full block model load** (file import, SGSIM grid creation)
```
DataRegistry.blockModelLoaded.emit(BlockModel)
  -> AppController.load_block_model(block_model)
  -> Renderer.load_block_model(block_model)
  -> BlockModelRenderer._generate_block_meshes() [auto-detects ImageData vs UnstructuredGrid]
  -> _apply_coordinate_transform_to_meshes()
  -> _add_meshes_to_plotter()
  -> add_layer()
```

**Path B: Analysis result overlay** (kriging, ARBF, etc.)
```
JobWorker.finished.emit(payload)
  -> AppController._on_task_complete()
  -> VisController.apply_results_to_model(payload)
  -> renderer.add_mesh() [adds as separate layer, does NOT go through _generate_block_meshes()]
```

**This is a critical difference.** Path A builds the mesh from the `BlockModel` object via the mesh builder (ImageData/UnstructuredGrid auto-detection). Path B receives a pre-built PyVista mesh from the payload and adds it directly. The pre-built mesh type depends on what the `_prepare_*_payload()` method created (typically RectilinearGrid or StructuredGrid).

### 4.3 Intermediate Steps

| Step | Description |
|------|-------------|
| Property panel refresh | `PropertyPanel._on_scene_changed()` repopulates dropdowns from `renderer.active_layers` |
| Layer toggle refresh | Quick layer buttons enable/disable based on layer existence |
| Legend update | `legend_manager.update_for_layer()` called within `renderer.update_layer_property()` |
| Cache invalidation | `renderer._fixed_bounds = None` on new block model load |

### 4.4 Single Shared Path or Separate?

**Shared path for visualization delivery:** All estimation methods go through `VisController.apply_results_to_model()` -> `renderer.add_mesh()`. There is no per-engine rendering code.

**However:** The mesh creation happens in the `_prepare_*_payload()` methods, which are engine-specific. Each payload method creates its own PyVista grid. The renderer just adds whatever mesh it receives.

---

## 5. Domain Rendering

### 5.1 Domain Filter in Rendering Path

**Location:** `block_model_renderer.py:~1140-1200` : `apply_domain_mask_transparency()`

**Mechanism:** Per-cell alpha transparency. Domain filtering happens AFTER mesh creation, at the VTK mapper level.

### 5.2 How It Works

```python
def apply_domain_mask_transparency(actor, mesh, opacity=1.0):
    # 1. Find domain_mask in mesh.cell_data or mesh.point_data
    mask = mesh.cell_data['domain_mask']   # 0=hidden, 1=visible
    
    # 2. Identify hidden blocks
    hidden = (mask == 0)
    
    # 3. Get mapped RGBA colors from VTK mapper
    mapper = actor.GetMapper()
    mapper.Update()
    mapped_colors = mapper.GetColorMapColors()
    rgba = vtk_to_numpy(mapped_colors).copy()   # (N, 4) uint8
    
    # 4. Set alpha=0 for hidden blocks
    rgba[hidden, 3] = 0
    
    # 5. Write back to mapper as DirectScalars
    vtk_arr = numpy_to_vtk(rgba, deep=True)
    mapper.SetColorModeToDirectScalars()
```

### 5.3 How Renderer Knows Active Domain

The domain mask is a `'domain_mask'` array stored directly in the PyVista mesh's cell_data. It is set during mesh generation or applied post-creation via `apply_domain_mask_to_grid()` from `domain_block_classifier.py`.

### 5.4 No Domain Selected / All Domains

- **No domain:** `'domain_mask'` array not present in mesh -> `apply_domain_mask_transparency()` returns `False`, all blocks visible
- **All domains:** `domain_mask` all 1s -> no hidden blocks -> all visible

### 5.5 Domain Mask Data Type

Numeric array in cell_data. Values: 0 = hidden, 1 = visible. Integer type.

### 5.6 Domain Filtering During Estimation

**Location:** `geostats/domain_block_classifier.py`

`get_domain_mask(centroids, registry, domain_value="Inside", iso_value=0.5)`:
1. **Primary method:** IRBF probability field interpolation via `RegularGridInterpolator` -> `mask = (probability >= iso_value)`
2. **Fallback:** PyVista `select_enclosed_points()` against isosurface mesh

`scatter_to_full_grid(reduced, mask, n_total, fill=np.nan)`: Expands reduced-length array back to full grid, filling non-domain blocks with NaN.

---

## 6. Layer Visibility

### 6.1 Toggle Buttons

**Location:** `block_model_viewer/ui/property_panel.py:466-501`

Three toggle buttons in a `CollapsibleGroup` labeled "Toggle layer visibility:":

| Button | Line | Behavior |
|--------|------|----------|
| **Drillholes** | 473-477 | Toggles all layers with 'drillhole' in name |
| **Block Model** | 480-484 | Toggles ALL block-type layers together (Block Model + SGSIM + Kriging + Classification) |
| **Geology** | 487-491 | Toggles all geology/surface layers |

Buttons are checkable (`style="toggle"`) and initially disabled until layers exist.

### 6.2 Signal Emitted

Button `.toggled(bool)` -> handler method. No custom signal; handlers call renderer directly.

### 6.3 Visibility Handler

**Block Model toggle:** `_on_block_model_toggle()` (line 534-569)
```python
for layer_name in block_type_layers:
    self.renderer.set_layer_visibility(layer_name, checked)
self.renderer.plotter.render()   # Single forced render at end
```

### 6.4 Actor Visibility Mechanism

**Location:** `render_orchestrator.py:3092-3150` : `set_layer_visibility()`

```python
def set_layer_visibility(self, layer_name: str, visible: bool):
    layer = self.active_layers[layer_name]
    layer['visible'] = visible
    actor = layer.get('actor')
    if visible:
        actor.VisibilityOn()
    else:
        actor.VisibilityOff()
    # Special handling for drillholes: propagates to hole + collar actors
```

**The actor is hidden/shown, not removed/recreated.** `VisibilityOn/Off` is a VTK actor property that controls rendering without touching geometry or mapper state.

### 6.5 Toggle Back ON

Shows the existing actor via `actor.VisibilityOn()`. Does NOT recreate mesh. The mesh data and scalar arrays remain intact in memory.

---

## 7. Property Visualization Panel

### 7.1 "Update Visualization" Button

**Location:** `property_panel.py:755-756`

```python
apply_btn = action_button("Update Visualization", style="secondary")
apply_btn.clicked.connect(lambda: self._on_property_changed(self.property_combo.currentText()))
```

It simply re-triggers `_on_property_changed()` with the currently selected property. This is effectively a "refresh" — useful if the user changed colormap settings or the data was updated externally.

### 7.2 What Update Visualization Does

`_on_property_changed(property_name)` (line 2073-2164):

1. Validates property exists in active layer
2. Auto-selects colormap via `get_default_colormap(property_name)`
3. Auto-detects discrete vs continuous mode via `is_discrete_property(property_name)`
4. Calls `renderer.update_layer_property(layer, property_name, cmap, mode, custom_colors)`
5. Updates clim spinners
6. Emits `signals.propertySelected.emit(property_name)` to sync session state

### 7.3 Does It Recreate the Mesh?

**No.** It updates the existing actor's mapper:

In `render_orchestrator.py:update_layer_property()` (line 3284+):
```python
actual_mesh.set_active_scalars(property_name, preference='cell')
mapper.SetScalarModeToUseCellData()
# Updates LUT, scalar range, color mode
# Re-applies domain mask transparency
# Does NOT remove/re-add mesh
```

### 7.4 Property Switching

When switching from e.g. `Cu_SGSIM_MEAN` to `Cu_SGSIM_VARIANCE`:
1. Property panel dropdown emits `currentTextChanged(new_property)`
2. `_on_property_changed(new_property)` called
3. `renderer.update_layer_property()` changes active scalars on existing mesh
4. VTK mapper recalculates colors from new scalar array
5. Domain mask re-applied (because `apply_domain_mask_transparency()` sets `ColorModeToDirectScalars`)
6. Legend updated

**Critical note:** `apply_domain_mask_transparency()` sets `ColorModeToDirectScalars`. The `set_property_coloring()` method must reset this to `ColorModeToMapScalars` before updating the LUT. This is marked as "CRITICAL FIX #3" in the code.

### 7.5 Same Path or Separate?

**Same rendering path.** The property panel calls `renderer.update_layer_property()` which is the same method used by `vis_controller.set_active_property()`. Both converge on the same mapper update logic.

---

## 8. Comparison Table

| System | Output stored how | Property naming | Signal emitted | Uses shared renderer | Domain-aware | Known differences |
|--------|-------------------|-----------------|----------------|----------------------|-------------|-------------------|
| **Ordinary Kriging** | `OrdinaryKrigingResults` dataclass -> grid cell_data | `OK_{var}_estimate`, `OK_{var}_variance`, `OK_{var}_{qa}` | `task_finished("kriging")` | Yes (via `apply_results_to_model`) | Yes (`get_domain_mask` + `scatter_to_full_grid`) | Standard reference pattern |
| **Simple Kriging** | `SimpleKrigingResults` dataclass -> grid cell_data | `SK_{var}`, variance prop, `SK_NN`, `SK_StabilityFlag` | `task_finished("simple_kriging")` | Yes | Yes | Uses `order="F"` for ravel (Fortran order) |
| **Universal Kriging** | `UniversalKrigingResults` dataclass -> grid cell_data | `UK_{var}`, `uk_*_var` | `task_finished("universal_kriging")` | Yes | Yes | Extra drift/trend properties |
| **Indicator Kriging** | `IndicatorKrigingResults` dataclass -> grid cell_data | `ik_*_p_le_{threshold}` per threshold | `task_finished("indicator_kriging")` | Yes | Yes | Multiple properties per run (one per threshold) |
| **Co-Kriging** | `CoKrigingResults` dataclass -> grid cell_data | `cok_*`, `cok_*_var` | `task_finished("cokriging")` | Yes | Yes | Includes secondary influence weight |
| **Bayesian Kriging** | Base method results -> grid cell_data | Inherits from base method | `task_finished("bayesian_kriging")` | Yes | Yes (inherited) | Wrapper; modifies covariance diagonal |
| **SGSIM** | `SGSIMResults` dataclass -> grid cell_data | `{var}_SGSIM_MEAN`, `{var}_SGSIM_VARIANCE`, `{var}_SGSIM_R{n}` | `task_finished("sgsim")` | Yes | Yes | N+2 properties per run; always RectilinearGrid |
| **CoSGSIM** | `CoSimulationResults` dataclass -> grid cell_data | Similar to SGSIM + secondary | `task_finished("cosgsim")` | Yes | Yes | Secondary variable realizations |
| **IK-SGSIM / SIS** | `SISResults` dataclass -> grid cell_data | Probability and category arrays | `task_finished("sis")` | Yes | Yes | Categorical simulation |
| **ARBF** | `BatchBlockResult` per block -> `ARBFOutputWriter` -> `block_model.add_property()` | ARBF-specific names | `task_finished("arbf")` | Yes | Yes | Per-block local solve; residual uncertainty (not kriging variance); has its own output writer |
| **RBF** | `estimates` array + `diagnostics` dict -> grid cell_data | RBF-specific | `task_finished("rbf")` | Yes | Yes | Uses SciPy RBFInterpolator |
| **FastRBF** | `FittedRBF` dataclass -> grid cell_data | FastRBF-specific | `task_finished("fastrbf")` | Yes | Yes | Direct LDLt or GMRES solver |
| **Indicator RBF** | `IndicatorRBFResult` -> grid cell_data + isosurface mesh | Probability volume + mesh | `task_finished("indicator_rbf")` | Yes | Yes | Produces BOTH volume and mesh |
| **IDW** | Tuple `(values, variances)` np.ndarrays | Caller-defined | **NOT FOUND** (not a registered job) | **NOT FOUND** | No | Utility function, not in JobWorker pipeline |
| **Nearest Neighbor** | Tuple `(values, variances)` np.ndarrays | Caller-defined | **NOT FOUND** (not a registered job) | **NOT FOUND** | No | Utility function, not in JobWorker pipeline |
| **Turning Bands** | Via `execute_standardized_simulation_workflow()` | Simulation-standard | `task_finished("turning_bands")` | Yes | Yes | 1D line processes; `n_bands=1000` |
| **DBS** | `DBSResults` dataclass | DBS-specific | `task_finished("dbs")` | Yes | Unclear | Direct block simulation |
| **GRF** | Via `execute_standardized_simulation_workflow()` | Simulation-standard | `task_finished("grf")` | Yes | Yes | Gaussian random field |
| **MPS** | `MPSResults` dataclass | MPS-specific | `task_finished("mps")` | Yes | Unclear | Multiple-point simulation |

---

## 9. Issues Found

### 9.1 Systems That Store Results Differently

1. **IDW and Nearest Neighbor** are utility functions in `blockmodel_advanced.py` that return raw numpy tuples. They are NOT registered in `JobRegistry`, do NOT go through `JobWorker`, do NOT emit `task_finished`, and do NOT create visualization payloads. They must be called directly and the caller is responsible for storing results and creating visualization. This is architecturally inconsistent with all other estimation methods.

2. **ARBF** has its own `ARBFOutputWriter` that writes results to the block model via `block_model.add_property()`, while other systems create a PyVista grid in the payload and let the renderer handle storage. ARBF writes to the BlockModel object directly, then separately creates a visualization mesh.

3. **Indicator RBF** produces both a volumetric probability field AND a mesh isosurface, unlike all other systems that produce only scalar arrays on a grid.

### 9.2 Signal Differences

1. **IDW and Nearest Neighbor** emit NO signal after completion (not registered jobs).
2. All other systems emit `task_finished` with their task name string, but the task name varies per system. Any code that checks for a specific task name (e.g., `"kriging"` vs `"simple_kriging"`) could miss some systems.

### 9.3 Separate Rendering Code

1. **No system has its own rendering code.** All go through `VisController.apply_results_to_model()` -> `renderer.add_mesh()`. This is good.
2. **However:** The mesh creation in `_prepare_*_payload()` methods is per-system. Each payload method creates its own PyVista grid type. Some create RectilinearGrid, some StructuredGrid. The renderer receives the pre-built mesh and adds it without type checking. If a payload method creates a grid with incorrect dimensions or properties, the renderer will not catch it.

### 9.4 Broken Signal Chains

1. **`_result_has_visual_output()` heuristic** (app_controller.py:668-690): Uses key-name matching (`"estimated_values"`, `"block_model"`, `"grid"`, `"pit_shells"`, `"schedule_blocks"`) to decide whether to visualize. If a new estimation method uses a different key name, visualization will silently not happen.

2. **Property panel refresh timing:** `signals.scene_updated` triggers `PropertyPanel._on_scene_changed()` which repopulates dropdowns. If this signal fires before the renderer has fully registered the new layer in `active_layers`, the dropdown may not include the new layer.

### 9.5 Silent Exception Swallowing

1. **`update_layer_property()`** in render_orchestrator.py: Has broad try/except blocks that log warnings but don't raise. A malformed property array will be logged but the user sees no error — the display simply doesn't update.

2. **Domain mask application:** `apply_domain_mask_transparency()` returns `False` if domain_mask not found or mapped_colors is None. The caller doesn't check this return value in all cases.

3. **`vis_controller.set_colormap()`** and **`vis_controller.set_active_property()`** had missing `LegendManager` methods that were swallowed by try/except (fixed per memory notes, but pattern may exist elsewhere).

### 9.6 Silent Mesh Creation Failures

1. **Empty property array:** If a property has all NaN values, `_compute_clim()` returns `[nan, nan]` which would set an invalid scalar range. The P2/P98 percentile computation filters NaN, but if ALL values are NaN (e.g., entire grid outside domain), `np.nanpercentile` on empty array returns NaN.

2. **Zero blocks after domain filtering:** If `scatter_to_full_grid()` fills everything with NaN (domain mask is all-False), the grid has valid geometry but all-NaN properties. The renderer will add this mesh but it will render as a uniform single color (sentinel handling in `_compute_clim`).

3. **ImageData creation with wrong dimensions:** `build_uniform_grid()` computes `grid.dimensions = (nx+1, ny+1, nz+1)`. If the input positions don't form a complete grid (missing blocks), some cells will have sentinel values but the mesh dimensions are based on the bounding grid. No error is raised.

### 9.7 Domain Filtering Edge Cases

1. **All blocks filtered:** If `get_domain_mask()` returns an all-False mask, `scatter_to_full_grid()` returns an all-NaN array. The grid renders but with no meaningful colors. The user sees a block model with uniform color and no indication that domain filtering eliminated all blocks.

2. **Domain mask + property switching interaction:** `apply_domain_mask_transparency()` sets `ColorModeToDirectScalars`. When switching properties, `set_property_coloring()` must reset to `ColorModeToMapScalars` first (marked as CRITICAL FIX #3). If this reset is missed or another code path sets property colors without resetting, the mesh will display incorrect colors (the previously baked RGBA instead of the new property's mapped colors).

### 9.8 First Creation vs Property Switching Differences

1. **First creation (Path A via `load_block_model`):** Goes through `_generate_block_meshes()` -> auto-detects ImageData/UnstructuredGrid -> full coordinate transform -> `_add_meshes_to_plotter()` with initial `plotter.add_mesh()`.

2. **Analysis result (Path B via `apply_results_to_model`):** Receives pre-built mesh from payload -> direct `renderer.add_mesh()` call. Does NOT go through `_generate_block_meshes()` or auto-detection. The mesh type is whatever the payload method created.

3. **Property switching:** Updates existing actor in-place via `update_layer_property()`. Changes active scalars and LUT, does not recreate mesh.

4. **Critical gap:** Path B does not apply the same coordinate transform as Path A. The payload method is responsible for creating the mesh with correct coordinates. If a payload method creates a mesh in UTM coordinates while the existing block model was shifted to local, the new analysis layer will be ~500km from the existing model.

### 9.9 Hardcoded Values, Magic Numbers, and Assumptions

1. **Double-shift threshold:** `shift_mag > 10_000` and `bound_center_mag < shift_mag * 0.1` (block_model_renderer.py:410-429). Magic numbers 10000 and 0.1.

2. **Grid fill threshold:** `fill_ratio > 0.01` (1%) in `is_uniform_grid()` to accept sparse grids as ImageData.

3. **P2/P98 percentiles** for color limits — hardcoded in `_compute_clim()`.

4. **Sentinel value:** `dtype.min` for integer arrays, `np.nan` for float arrays — used in ImageData mesh builder for empty cells.

5. **Edge auto-disable threshold:** Edges automatically disabled for models >= 50,000 cells (renderer.py:5877-5889).

6. **Classification discrete LUT:** Hardcoded 4 categories (Measured=0, Indicated=1, Inferred=2, Unclassified=3).

7. **Ravel order inconsistency:** SK payload uses `order="F"` (Fortran), while ImageData mesh builder uses `order="C"`. If SK results are loaded into an ImageData mesh, the array ordering may be incompatible.

8. **NSR field detection:** Hardcoded list `property_name.upper() in ['NSR', 'NSR_TOTAL', ...]` for divergent colormap.

---

## 10. Files Read

### Core Data Model
- `block_model_viewer/models/block_model.py` — BlockModel class, BlockMetadata dataclass
- `block_model_viewer/models/block_model_definition.py` — BlockModelDefinition class
- `block_model_viewer/models/blockmodel_advanced.py` — IDW, Nearest Neighbor utility functions
- `block_model_viewer/models/geostat_results.py` — All result dataclasses (OK, SK, UK, IK, CoK, SGSIM, SIS, DBS, TB, MPS, CoSim)

### Estimation Engines
- `block_model_viewer/models/kriging3d.py` — OrdinaryKriging3D
- `block_model_viewer/models/kriging_engine.py` — Numba JIT kernel
- `block_model_viewer/models/simple_kriging3d.py` — SimpleKriging3D
- `block_model_viewer/models/sgsim3d.py` — SGSIM3D orchestrator
- `block_model_viewer/models/sgsim_engine.py` — Numba SGSIM kernel
- `block_model_viewer/geostats/universal_kriging.py` — UniversalKriging3D
- `block_model_viewer/geostats/indicator_kriging.py` — Indicator Kriging
- `block_model_viewer/geostats/cokriging3d.py` — CoKriging3D
- `block_model_viewer/geostats/bayesian_kriging.py` — Bayesian Kriging wrapper
- `block_model_viewer/geostats/cosgsim3d.py` — Co-SGSIM
- `block_model_viewer/geostats/ik_sgsim.py` — IK-SGSIM
- `block_model_viewer/geostats/rbf_interpolation.py` — RBF Interpolator
- `block_model_viewer/geostats/fastrbf_engine_v2.py` — FastRBF Engine
- `block_model_viewer/geostats/arbf_adapter.py` — ARBF Adapter
- `block_model_viewer/geostats/arbf_batch_solver.py` — ARBF batch solver
- `block_model_viewer/geostats/arbf_output_writer.py` — ARBF result writer
- `block_model_viewer/geostats/arbf_modes.py` — ARBF operation modes
- `block_model_viewer/geostats/arbf_estimator_definition.py` — ARBF config
- `block_model_viewer/geostats/arbf_evaluation_job.py` — ARBF job orchestration
- `block_model_viewer/geostats/indicator_rbf_engine.py` — Indicator RBF
- `block_model_viewer/geostats/domain_block_classifier.py` — Domain mask and scatter functions

### Controllers
- `block_model_viewer/controllers/controller_signals.py` — Signal definitions
- `block_model_viewer/controllers/app_controller.py` — Main controller, task completion
- `block_model_viewer/controllers/vis_controller.py` — Visualization bridge
- `block_model_viewer/controllers/geostats_controller.py` — All `_prepare_*_payload()` methods
- `block_model_viewer/controllers/job_registry.py` — Task registration (75+ tasks)
- `block_model_viewer/controllers/job_worker.py` — QThread worker, signals

### Rendering
- `block_model_viewer/visualization/renderer/render_orchestrator.py` — Main renderer, `active_layers`, `update_layer_property()`, `set_layer_visibility()`
- `block_model_viewer/visualization/renderer/renderers/block_model_renderer.py` — Mesh generation, scalar coloring, domain mask, coordinate transform
- `block_model_viewer/visualization/block_model_mesh_builder.py` — Auto-detect grid type, build ImageData/UnstructuredGrid
- `block_model_viewer/visualization/grid_adapter.py` — GridPayload to PyVista conversion
- `block_model_viewer/visualization/mesh_adapter.py` — MeshPayload to PyVista conversion
- `block_model_viewer/visualization/filters.py` — Domain/property filtering

### UI
- `block_model_viewer/ui/property_panel.py` — Property panel, quick layers, Update Visualization
- `block_model_viewer/ui/display_settings_panel.py` — Display rendering controls
- `block_model_viewer/ui/main_window.py` — Signal wiring, scene update handlers
- `block_model_viewer/ui/coordinators/signal_coordinator.py` — Central signal hub
- `block_model_viewer/ui/arbf_estimation_panel.py` — ARBF estimation panel (example)
- `block_model_viewer/core/data_registry.py` — Block model signals and registration
