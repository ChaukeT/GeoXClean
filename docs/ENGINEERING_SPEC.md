# Engineering Specification

**Product:** GeoX Desktop
**Version:** 1.0.0
**Date:** March 2026

---

## 1. System Architecture

### 1.1 Architecture Overview

GeoX Desktop follows a layered architecture with strict dependency direction (upper layers depend on lower layers, never the reverse):

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 4 — UI (PyQt6)                                      │
│  100+ analysis panels, 3D viewer, dialogs, menus            │
│  block_model_viewer/ui/                                     │
└────────────────────┬────────────────────────────────────────┘
                     │ Qt signals, controller method calls
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 3 — CONTROLLERS (Orchestration)                      │
│  AppController, GeostatsController, MiningController,       │
│  VisController, DataController, JobRegistry                 │
│  block_model_viewer/controllers/                            │
└────────────────────┬────────────────────────────────────────┘
                     │ Pure function calls (no Qt dependency)
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 2 — DOMAIN ENGINES (Computation)                     │
│  geostats/, models/, drillholes/, geology/,                 │
│  mine_planning/, irr_engine/, geomet/, geotech/             │
│  Pure NumPy/SciPy/Numba — no Qt imports                     │
└────────────────────┬────────────────────────────────────────┘
                     │ Data registry read/write
┌────────────────────▼────────────────────────────────────────┐
│  LAYER 1 — DATA / PERSISTENCE                               │
│  DataRegistry, DataProvenance, AuditManager, Parsers        │
│  block_model_viewer/core/, block_model_viewer/parsers/      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Rules

1. **No UI imports in domain engines.** All computation modules (`geostats/`, `models/`, `drillholes/engines`) use only NumPy, SciPy, Numba, and Pandas. They never import PyQt6 or PyVista.
2. **Deterministic computation.** Every stochastic function accepts an explicit `seed` parameter. Same data + parameters + seed = identical output.
3. **Typed results.** Engines return dataclasses (with `__getitem__` fallback for dict-style access) rather than raw dicts or tuples.
4. **Async by default for long operations.** Kriging, simulation, ARBF, and pit optimization run on `QThread` via the `JobRegistry`/`JobWorker` system. UI remains responsive.
5. **Signal-driven coordination.** Controllers and panels communicate through Qt signals defined in `controller_signals.py` and `ui/signals.py`. No direct panel-to-panel calls.

### 1.3 Dependency Injection

The `DataRegistry` is created once at application startup and passed by reference to all controllers and panels. Controllers are not singletons — they are instantiated by `AppController` and hold references to the registry and to each other where needed.

---

## 2. Technology Stack

### 2.1 Core Dependencies

| Category | Package | Version | Purpose |
|----------|---------|---------|---------|
| GUI Framework | PyQt6 | ≥ 6.5 | Widgets, signals/slots, event loop |
| 3D Rendering | PyVista | ≥ 0.42 | High-level VTK wrapper |
| 3D Rendering | VTK | ≥ 9.2 | Low-level GPU rendering |
| Qt-VTK Bridge | PyVistaQt | ≥ 0.10 | Embed VTK in Qt widgets |
| Numerical | NumPy | ≥ 1.24 | Array operations |
| Scientific | SciPy | ≥ 1.11 | Spatial, linear algebra, optimization |
| JIT Compiler | Numba | ≥ 0.57 | Kriging/simulation kernel acceleration |
| Data Frames | Pandas | ≥ 2.0 | Tabular data manipulation |
| Charting | Matplotlib | ≥ 3.7 | 2D plotting and chart export |
| ML | scikit-learn | ≥ 1.3 | Clustering (K-means, DBSCAN), transforms |
| Geometry | Trimesh | ≥ 3.23 | Mesh I/O and processing |
| Point Cloud | Open3D | ≥ 0.17 | Point cloud processing |
| Point Cloud | laspy | ≥ 2.5 | LAS file I/O |
| Geological | LoopStructural | ≥ 1.6 | Implicit geological modelling |
| Geological | GemPy | ≥ 2024.0 | Stratigraphic modelling |
| Optimization | PuLP | ≥ 2.7 | Linear programming (scheduling) |
| Config | Pydantic | ≥ 2.0 | Parameter validation |
| Config | PyYAML | ≥ 6.0 | YAML configuration files |
| File I/O | openpyxl | ≥ 3.1 | Excel export |
| File I/O | ReportLab | ≥ 4.0 | PDF report generation |
| File I/O | ezdxf | ≥ 1.0 | DXF wireframe import |
| Packaging | PyInstaller | ≥ 6.0 | Windows installer |

### 2.2 Python Version

- **Minimum:** Python 3.10
- **Tested:** 3.10, 3.11, 3.12, 3.13
- **Target for packaging:** 3.12 (PyInstaller compatibility)

---

## 3. Module Specifications

### 3.1 Entry Point and Application Bootstrap

**File:** `block_model_viewer/main.py`

Startup sequence:
1. Parse command-line arguments
2. Create `QApplication` with platform-specific settings
3. Initialize `DataRegistry` (singleton for the session)
4. Initialize `AppController` (creates sub-controllers)
5. Create `MainWindow` (creates viewer widget, panel manager, menus)
6. Register all panels via `panel_registration.py`
7. Restore window state and layout
8. Enter Qt event loop

### 3.2 Controller Layer

#### 3.2.1 AppController (`controllers/app_controller.py`)

Central orchestrator. Delegates domain operations to sub-controllers:

| Sub-Controller | Responsibility |
|---------------|---------------|
| `GeostatsController` | Kriging, simulation, variogram, declustering dispatch |
| `MiningController` | Resource reporting, pit optimization, scheduling, ESG |
| `VisController` | Rendering, layer management, legend, colormap, transparency |
| `DataController` | Drillhole import, geology data, structural data |
| `JobRegistry` | Background job queue (submit, cancel, progress tracking) |

**Key methods:**
- `load_block_model(path)` → parse CSV → register in DataRegistry → trigger render
- `run_estimation(method, params)` → submit to JobRegistry → on completion, add result to block model → re-render
- `set_active_property(name)` → update renderer + legend + session state
- `set_edge_visibility(visible)` → propagate to VisController → renderer
- `export_data(format, path)` → delegate to export dialog

#### 3.2.2 JobRegistry (`controllers/job_registry.py`) and JobWorker (`controllers/job_worker.py`)

Asynchronous job execution system:

```
Panel submits job
    → JobRegistry.submit(job_id, callable, params)
        → Creates JobWorker(QThread)
        → Emits: started, progress(int), finished(result), error(exception)
    → Panel connects to signals for progress bar and result handling
```

- Jobs are cancellable via `JobRegistry.cancel(job_id)`
- Maximum concurrent jobs: configurable (default 1 to prevent memory pressure)
- Progress callback passed to engine functions as `progress_fn(percent: int)`

#### 3.2.3 Controller Signals (`controllers/controller_signals.py`)

Centralized signal definitions for inter-controller communication:

| Signal | Emitted By | Consumed By |
|--------|-----------|-------------|
| `blockModelLoaded` | DataController | VisController, Panels |
| `estimationComplete` | GeostatsController | VisController, Panels |
| `propertySelected` | PropertyPanel / AppController | VisController, LegendManager |
| `colormapChanged` | PropertyPanel | VisController, LegendManager |
| `coordinateMismatchDetected` | DataRegistry | MainWindow (toast) |
| `classificationComplete` | MiningController | JORCClassificationPanel |

### 3.3 Data Layer

#### 3.3.1 DataRegistry (`core/data_registry.py`)

Central in-memory store for all loaded and computed data. ~68KB, the largest core module.

**Stored data types:**
- Drillhole DataFrames (collar, survey, assay, lithology, composites)
- Block model objects (BlockModel dataclass + PyVista grids)
- Estimation results (kriging grades, variances, simulation realizations)
- Variogram models (fitted parameters)
- Classification results
- Declustering weights
- Surfaces and meshes

**Key patterns:**
- `register_block_model(model)` → stores model, emits `blockModelRegistered` signal
- `get_block_model()` → returns current active block model
- `register_estimation_result(name, result)` → adds property to block model
- `_check_coordinate_alignment()` → auto-runs on data load, emits `coordinateMismatchDetected` if UTM vs local mismatch detected

#### 3.3.2 DataProvenance (`core/data_provenance.py`)

Audit infrastructure:
- `compute_hash(data: pd.DataFrame) → str` — SHA-256 of sorted, serialized DataFrame
- `log_operation(operation, params, input_hash, output_hash)` — append to daily JSONL
- File checksums for imported files

#### 3.3.3 AuditManager (`core/audit_manager.py`)

JSONL log writer:
- One file per day: `audit_logs/audit_YYYY-MM-DD.jsonl`
- Each line: `{"timestamp": "...", "operator": "...", "operation": "...", "params": {...}, "input_hash": "...", "output_hash": "..."}`
- Thread-safe (lock on write)

### 3.4 Parsers (`parsers/`)

| Parser | File | Input | Output |
|--------|------|-------|--------|
| CSV Block Model | `csv_parser.py` | CSV with X/Y/Z + attributes | `BlockModel` + `pv.ImageData` |
| DXF Wireframe | `dxf_parser.py` | DXF polylines/3D faces | `pv.PolyData` |
| Topography | `topo_parser.py` | DTM/elevation CSV/grid | `pv.StructuredGrid` |
| VTK | `vtk_parser.py` | VTK legacy/XML files | `pv.UnstructuredGrid` |
| Mesh | `mesh_parser.py` | OBJ, STL | `pv.PolyData` |
| Structural CSV | `structural_csv_parser.py` | Dip/azimuth measurements | Structural DataFrame |
| Mining formats | `mining_parser.py` | Industry-specific formats | DataFrame |

**CSV Parser features:**
- Column mapping dialog (auto-detect X, Y, Z, grade columns)
- Dimension inference from coordinate spacing
- Leapfrog CSV header parsing
- Checksum computation on import

---

## 4. Domain Engine Specifications

### 4.1 Variography Engine (`geostats/variogram3d.py`, `variogram_model.py`)

#### Experimental Variogram

**Function:** `run_variogram_pipeline(data, params, sample_weights=None) → VariogramResult`

**Algorithm:**
1. Extract coordinate and value arrays from DataFrame
2. If `sample_weights` provided (from declustering), apply pair weighting: `w_pair = w_i × w_j`
3. For each lag distance `h_k`:
   - Find all pairs `(i, j)` where `|d(i,j) - h_k| < lag_tolerance`
   - Apply angular bandwidth filter for directional variograms
   - Compute weighted semivariance: `γ(h_k) = Σ w_pair × (z_i - z_j)² / (2 × Σ w_pair)`
4. Return lag distances, semivariances, pair counts

**Directional support:** Azimuth (0–360°), dip (-90° to +90°), bandwidth tolerance

#### Model Fitting

**Supported models:** Spherical, Exponential, Gaussian, Power

**Nested structures:** Up to 3 nested structures + nugget

**Fitting:** Weighted least-squares minimization (SciPy `minimize`) with pair-count weighting

**Anisotropy:** Three principal axes with rotation angles (azimuth, dip, pitch) and range ratios

### 4.2 Kriging Engines

#### Ordinary Kriging (`geostats/kriging3d.py`)

**Function:** `run_ordinary_kriging(data, grid, variogram, search_params, progress_fn) → KrigingResult`

**Algorithm per block:**
1. Search for samples within anisotropic neighborhood (KD-tree with rotation)
2. Enforce min/max sample count and octant constraints
3. Build kriging matrix `C` (sample-to-sample covariances) + Lagrange multiplier row/col
4. Build RHS vector `c₀` (sample-to-block covariances)
5. Solve `C × λ = c₀` for kriging weights λ
6. Estimate: `z* = Σ λ_i × z_i`
7. Variance: `σ²_OK = C(0) - Σ λ_i × c₀_i - μ` (Lagrange multiplier)

**Numba acceleration:** Inner loop (matrix assembly + solve) compiled with `@njit`

**Output:** `KrigingResult` dataclass with `estimates`, `variances`, `sample_counts` arrays

#### Simple Kriging (`geostats/simple_kriging3d.py`)

Same as OK but without Lagrange multiplier. Requires known global mean `m`:
- Estimate: `z* = m + Σ λ_i × (z_i - m)`
- Variance: `σ²_SK = C(0) - Σ λ_i × c₀_i`

#### Universal Kriging (`geostats/universal_kriging.py`)

Extends OK with polynomial trend functions:
- Linear: `f(x,y,z) = [1, x, y, z]`
- Quadratic: `f(x,y,z) = [1, x, y, z, x², y², z², xy, xz, yz]`
- Additional Lagrange constraints for each trend function

#### Indicator Kriging (`geostats/indicator_kriging.py`)

Binary indicator transform at threshold `z_c`:
- `I_i = 1 if z_i ≤ z_c, else 0`
- Estimate proportion via OK on indicators
- Order relation correction: monotonic adjustment of cumulative proportions across thresholds

#### Co-Kriging (`geostats/cokriging3d.py`)

Multi-variable estimation with cross-variograms:
- Primary variable `Z₁` estimated using both `Z₁` and secondary `Z₂` samples
- Requires auto-variograms for each variable + cross-variogram
- Block kriging matrix includes all variable pairs

#### Bayesian Kriging (`geostats/bayesian_kriging.py`)

Incorporates soft (secondary) data with precision weighting:
- Soft data points have associated precision `p_i` (inverse variance)
- Modified kriging system adds soft data with reduced weight
- Posterior variance accounts for soft data information

### 4.3 ARBF Engine

**Location:** `geostats/rbf_interpolation.py` (31KB)

#### 10-Step Workflow

```
Step 1: Data Transform
    Normal-score or ILR transform → Gaussian domain
    ↓
Step 2: Orientation Field (LVA)
    Data-driven (Boisvert 2009) | Structural | Identity
    ↓
Step 3: Sub-domain Decomposition
    PUM auto-dispatch: N > 1,500 → k-means domains with Wendland C2 blend
    N ≤ 1,500 → single-domain bypass
    ↓
Step 4: Kernel Matrix Assembly
    Build Φ matrix per domain using selected kernel
    Kernels: Spheroidal, Gaussian, Matérn 3/2, Matérn 5/2, Cubic, Wendland C2
    Regularization: ε ≥ 1e-7 on diagonal
    ↓
Step 5: Factorization
    Cholesky (if positive-definite) or LU decomposition
    ↓
Step 6: Block Estimation
    Per block: evaluate kernel at block center → weight combination
    PUM: Wendland C2 partition-of-unity blending across domain boundaries
    Output: estimated value + posterior variance
    ↓
Step 7: LOO Cross-Validation
    Bartlett virtual formula (fast): CV_i = z_i - (Φ⁻¹z)_i / (Φ⁻¹)_ii
    No matrix re-inversion per sample
    Output: R², RMSE, MAE, slope, NRMSE
    ↓
Step 8: Change-of-Support Correction
    Matheron affine correction: z_block = m + (z_point - m) × (σ_block/σ_point)
    Ratio constraint: 0 < σ_block/σ_point ≤ 1.0
    ↓
Step 9: Back-Transform
    Normal-score → original scale with ratio-preserving sill rescale
    Fix: sill mismatch from transform (nugget/sill ratio preserved)
    ↓
Step 10: JORC Classification
    Dual-criteria:
    - Variance: T1 (Measured), T2 (Indicated), T3 (Inferred) as fractions of sill
    - Geometric: min samples + octant coverage
    Auto-scale if median variance > T3
    Output: classification array + JSON audit record
```

### 4.4 Simulation Engines

#### SGSIM (`models/sgsim3d.py`, 96KB)

**Function:** `run_sgsim(data, grid, variogram, n_realizations, seed, progress_fn) → SGSIMResult`

**Algorithm per realization:**
1. Normal-score transform data values
2. Build search template (pre-computed neighbor offsets sorted by distance)
3. Generate random path through all grid nodes
4. For each node in path:
   - Find nearest conditioning data + previously simulated nodes (via search template)
   - Solve Simple Kriging system → get SK estimate and SK variance
   - Draw from `N(SK_estimate, SK_variance)` using realization-specific RNG
   - Add simulated value to conditioning set
5. Back-transform to original scale

**Multi-realization output:**
- Individual realization arrays
- E-type (mean across realizations)
- P10, P50, P90 quantile grids
- Variogram reproduction statistics

**Numba acceleration:** Inner simulation loop and SK solver compiled with `@njit`

#### Turning Bands (`geostats/turning_bands.py`)

1D line process simulation:
1. Generate `L` random lines through origin
2. Simulate 1D Gaussian process on each line
3. For each grid node: project onto each line, interpolate, sum contributions
4. Scale to target variance

#### Sequential Indicator Simulation (`geostats/sis.py`)

Multi-threshold indicator simulation:
1. For each threshold `z_k`: transform data to indicator `I(z ≤ z_k)`
2. Simulate each indicator field independently (via IK-based sequential simulation)
3. Apply order relation corrections to ensure monotonic CDF
4. Draw category from simulated local CDF

### 4.5 Pit Optimization Engine (`models/pit_optimizer.py`, 77KB)

**Algorithm:** Pseudoflow maximum-closure (Lerchs-Grossmann equivalent)

**Function:** `nested_shells_optimize(block_model, params) → (shell_array, cashflow_df)`

**Computation per revenue factor `f`:**
1. Compute block economic value: `V = (grade × recovery × price × f) - mining_cost - processing_cost`
2. Build precedence graph (slope angle constraints)
3. Run pseudoflow algorithm → find maximum-closure subset
4. Mark blocks in optimal pit shell
5. Compute grade-tonnage and NPV for shell

**Output:**
- Shell index per block (integer array added as block model property `PIT_SHELL`)
- Cashflow DataFrame (per-shell NPV, IRR, tonnes, grade, metal)

### 4.6 JORC Classification Engine (`models/jorc_classification_engine.py`, 77KB)

**Dual-criteria classification:**

| Category | Variance Criterion | Geometric Criterion |
|----------|-------------------|-------------------|
| Measured | σ² ≤ T1 × sill | samples ≥ N1 AND octants ≥ O1 |
| Indicated | σ² ≤ T2 × sill | samples ≥ N2 AND octants ≥ O2 |
| Inferred | σ² ≤ T3 × sill | samples ≥ N3 |
| Unclassified | σ² > T3 × sill | (anything else) |

**Auto-scaling:** If median block variance > T3 × sill, multiply all thresholds by `median_variance / (T3 × sill)` to ensure at least 50% of blocks reach Inferred.

**Output:** Classification integer array (1=Measured, 2=Indicated, 3=Inferred, 4=Unclassified) + JSON audit record with all parameters and checksums.

### 4.7 Compositing Engine (`drillholes/compositing_engine.py`, 107KB)

**Length-weighted compositing:**

For target interval length `L`:
1. Walk down each drillhole from collar
2. Accumulate assay grades weighted by interval length: `z_comp = Σ(z_i × l_i) / Σ(l_i)`
3. If domain-based: reset accumulator at domain boundaries
4. Output composite DataFrame with: HOLEID, FROM, TO, composite grade, composite length

**Minimum threshold:** Discard composites shorter than user-specified minimum fraction of target length.

### 4.8 Declustering Engine (`drillholes/declustering.py`, 57KB)

**Cell-based method:**
1. Overlay regular grid with cell size `s` over sample locations
2. Count samples per cell `n_j`
3. Weight for sample `i` in cell `j`: `w_i = 1 / n_j`
4. Normalize: `w_i → w_i × N / Σ w_i` (sum-to-N, scale-invariant)

**Origin-offset method (Deutsch 1989):**
1. For each of `K` origin offsets (grid shift):
   - Recompute cell assignments and weights
2. Average weights across all offsets
3. Select cell size that minimizes declustered mean (for positively skewed distributions)

**Output:** Weight column added to DataFrame, N_eff (effective sample count), weighted quantiles.

---

## 5. Visualization Engine

### 5.1 Render Orchestrator (`visualization/renderer/render_orchestrator.py`, 304KB)

The largest single module. Central rendering coordinator managing all VTK actors and visual state.

**Responsibilities:**
- Block model mesh creation and property coloring
- Drillhole tube rendering
- Surface/wireframe overlay
- Colormap application (discrete and continuous)
- Legend synchronization
- Coordinate transform (UTM → local) with double-shift guards
- Camera clipping range management
- Level-of-detail downsampling
- Actor registry and visibility management

#### Coordinate Transform System

**Problem:** Block models may be in UTM coordinates (X ~500,000) while the renderer works in local coordinates (X ~0).

**Solution:**
1. On first block model load, compute `_global_shift` from centroid
2. Apply shift to all mesh origins/points
3. **Double-shift guard:** Before applying shift, check if `center_magnitude < shift_magnitude × 0.5`. If true → data already local → skip shift, mark `_coordinate_shifted = True`
4. Actor bounds (post-transform) are always in local coordinates → preferred for scene bounds

#### Camera Clipping

**Problem:** VTK's default `ResetCameraClippingRange()` uses 1:100 near/far ratio → geometry clips when zoomed in.

**Solution:** Custom `_maintain_clipping_range()`:
- Guard flag `_in_clipping_update` prevents infinite loop from `camera.SetClippingRange()` firing `ModifiedEvent`
- Near: `max(0.001, distance × 0.0001)` — scales with zoom
- Far: `max(max_projection × 2.0, scene_size × 10.0, distance × 10.0)`
- Fires on `InteractionEvent` (after VTK's internal reset)

#### Discrete Color Synchronization

**Problem:** Renderer samples colormap by scalar VALUE position; legend samples by CATEGORY INDEX → mismatch with uneven values.

**Solution:** Pre-generate categorical colors by index (`t = i / max(1, n-1)`) once. Pass same RGBA dict to VTK LUT, matplotlib ListedColormap, and legend_manager.

### 5.2 Drillhole GPU Renderer (`visualization/drillhole_gpu_renderer.py`, 79KB)

Renders drillholes as 3D tubes colored by attribute:
1. Desurvey: convert collar + survey (depth/azimuth/dip) → XYZ trace
2. Create tube geometry per interval
3. Color by selected attribute (grade, lithology code)
4. Apply coordinate shift consistent with block model

### 5.3 Level-of-Detail (`visualization/lod_manager.py`)

For models exceeding performance thresholds:
- **Threshold:** 50,000 cells → disable edges automatically
- **Threshold:** 200,000 cells → enable LOD downsampling
- **Downsampling:** Stride-based decimation preserving spatial distribution
- **Camera distance LOD:** Reduce detail when camera is far from model

### 5.4 Scene Bounds (`visualization/renderer/scene_bounds.py`)

Priority order for scene bounds computation:
1. Locked `_fixed_bounds` (prevents drift during interaction)
2. Actor-based bounds (`GetBounds()` — always local coords post-transform)
3. `current_model.bounds` fallback with double-shift guard
4. Legacy `_fixed_scene_bounds`

---

## 6. UI Architecture

### 6.1 Panel Hierarchy

```
QWidget
└── BasePanel
    ├── BaseAnalysisPanel (scroll area, progress bar, status)
    │   ├── KrigingPanel
    │   ├── SGSIMPanel
    │   ├── ARBFPanel
    │   ├── PitOptimisationPanel
    │   └── ... (80+ analysis panels)
    ├── BaseDisplayPanel (rendering, legend, property controls)
    │   ├── PropertyPanel
    │   ├── DisplaySettingsPanel
    │   └── SceneInspectorPanel
    └── BaseDialogPanel (dialog-style panels)
```

### 6.2 BaseAnalysisPanel Common Infrastructure

Every analysis panel inherits:
- `_get_block_model()` — retrieves classified block model first, raw fallback; sets `_using_classified` flag
- `_build_unclassified_notice()` — yellow QFrame warning when using unclassified data
- `_connect_registry_notice()` — auto-wires DataRegistry signals (via `QTimer.singleShot`) to refresh notice bar
- `setup_ui()` — must call `super().setup_ui()` then add panel-specific widgets
- `validate_inputs() → bool` — validate before submission
- `gather_parameters() → dict` — collect all widget values into parameter dict
- `_on_run()` — submit job to JobRegistry
- `on_results(result)` — handle completed job, update UI, register results

### 6.3 Panel Manager (`ui/panel_manager.py`)

Dynamic panel loading and docking:
- Panels registered at startup via `panel_registration.py`
- Lazy instantiation: panels created on first show
- Docking: panels can be docked, floated, tabbed, or closed
- State persistence: dock layout saved/restored between sessions
- Panel state file: `panel_states.json`

### 6.4 Menu System (`ui/menus/`)

| Menu | Key Items |
|------|----------|
| File | New, Open Project, Save, Import Block Model, Import Drillholes, Export, Screenshot |
| Edit | Undo, Redo, Preferences |
| Data | Block Model Builder, Data Viewer, Data Registry Status |
| Tools | Compositing, Declustering, Grade Transform, Block Property Calculator, Statistics |
| View | Panels submenu (toggle all panels), Theme, Axes/Scale Bar, Camera modes |
| Resources | Grade-Tonnage, Cutoff Optimization, JORC Classification, Resource Reporting |
| Survey | InSAR, Deformation Monitoring |
| Help | Documentation, About, Version Info |

### 6.5 Signal Architecture

**UI → Controller flow:**
```
Panel widget change
  → Panel method (validate + gather params)
    → Controller method call
      → Engine function (on QThread via JobRegistry)
        → Result signal
          → Controller handler
            → DataRegistry update
              → Registry signal
                → All listening panels refresh
```

**Key signal chains:**
- `propertySelected` → `AppController.set_active_property()` → `VisController` → `LegendManager.update_from_property()` → renderer re-color
- `colormapChanged` → `AppController.set_colormap()` → `VisController` → `LegendManager.set_colormap()` → renderer re-color
- `blockModelRegistered` → all panels with `_connect_registry_notice()` refresh their data source
- `estimationComplete` → PropertyPanel adds new property to dropdown → renderer re-colors

### 6.6 Theme System (`ui/theme_manager.py`)

Two themes: light and dark.
- Stylesheets: `assets/themes/light.qss`, `assets/themes/dark.qss`
- Design tokens in `ui/design_tokens.py` for programmatic color access
- `modern_styles.py` — CSS generation utilities
- `modern_widgets.py` — themed custom widgets (cards, badges, progress indicators)
- Theme switch: `ThemeManager.set_theme(name)` → reload stylesheet → emit `themeChanged` signal

---

## 7. Data Flow Diagrams

### 7.1 End-to-End Resource Estimation

```
CSV Files (collar, survey, assay)
    │
    ▼
┌──────────────────────────────┐
│  Drillhole Import Panel      │  FR-01
│  Column mapping + validation │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  Compositing Engine          │  FR-02
│  Length-weighted, domain-based│
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  Declustering Engine         │  FR-03
│  Cell-based + origin-offset  │
│  Output: weights column      │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  Variogram Engine            │  FR-06, FR-07
│  Experimental + model fit    │
│  Accepts declustering weights│
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  Estimation Engine           │  FR-08, FR-09, FR-10
│  Kriging / ARBF / Simulation │
│  Background job (QThread)    │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  JORC Classification         │  FR-13
│  Variance + geometric dual   │
│  Output: M/I/U categories    │
└──────────┬───────────────────┘
           ▼
┌──────────────────────────────┐
│  Resource Reporting          │  FR-14
│  Grade-tonnage + export      │
│  JORC audit record (JSON)    │
└──────────┬───────────────────┘
           ▼
    CSV / Excel / VTK Export
```

### 7.2 Mine Planning Flow

```
Classified Block Model (from estimation)
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌─────────────────┐          ┌──────────────────────┐
│  Grade-Tonnage   │          │  Pit Optimization     │
│  Cutoff Analysis │          │  Nested shells        │
│  (Lane's method) │          │  PIT_SHELL property   │
└─────────────────┘          └──────────┬───────────┘
                                        ▼
                             ┌──────────────────────┐
                             │  Pushback Designer    │
                             │  Shell → pushback     │
                             └──────────┬───────────┘
                                        ▼
                    ┌───────────────────┬────────────────────┐
                    ▼                   ▼                    ▼
           ┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
           │  Strategic    │  │  Tactical         │  │  Short-Term  │
           │  Schedule     │  │  Schedule          │  │  Schedule    │
           │  (annual)     │  │  (quarterly)       │  │  (daily)     │
           └──────┬───────┘  └──────────────────┘  └──────────────┘
                  ▼
           ┌──────────────┐
           │  NPV / IRR   │
           │  Analysis     │
           └──────────────┘
```

---

## 8. File and Directory Structure

```
block_model_viewer/
├── main.py                          # Entry point
├── assets/
│   ├── icons/                       # SVG icons by category
│   └── themes/                      # light.qss, dark.qss
├── controllers/
│   ├── app_controller.py            # Central orchestrator
│   ├── geostats_controller.py       # Estimation dispatch
│   ├── mining_controller.py         # Mine planning dispatch
│   ├── vis_controller.py            # Rendering coordination
│   ├── data_controller.py           # Data import coordination
│   ├── job_registry.py              # Async job queue
│   ├── job_worker.py                # QThread worker
│   ├── controller_signals.py        # Inter-controller signals
│   └── undo/                        # Undo/redo command system
├── core/
│   ├── data_registry.py             # Central data store (68KB)
│   ├── data_registry_simple.py      # Lightweight store (94KB)
│   ├── data_provenance.py           # SHA-256 hashing
│   ├── audit_manager.py             # JSONL audit logging
│   ├── crash_handler.py             # Exception capture
│   ├── security.py                  # Input validation
│   └── user_auth.py                 # Operator tracking
├── drillholes/
│   ├── compositing_engine.py        # Compositing (107KB)
│   ├── declustering.py              # Cell/origin-offset (57KB)
│   ├── drillhole_validation.py      # QC validation
│   ├── data_io.py                   # Import/export
│   └── drillhole_layer.py           # 3D rendering
├── geostats/
│   ├── variogram3d.py               # Experimental variogram (88KB)
│   ├── variogram_model.py           # Model fitting (30KB)
│   ├── kriging3d.py                 # Ordinary Kriging (52KB)
│   ├── simple_kriging3d.py          # Simple Kriging (28KB)
│   ├── universal_kriging.py         # Universal Kriging (37KB)
│   ├── indicator_kriging.py         # Indicator Kriging (24KB)
│   ├── cokriging3d.py               # Co-Kriging (80KB)
│   ├── bayesian_kriging.py          # Bayesian Kriging (37KB)
│   ├── rbf_interpolation.py         # ARBF engine (31KB)
│   ├── sgsim3d.py                   # SGSIM (96KB)
│   ├── turning_bands.py             # Turning Bands (24KB)
│   ├── sis.py                       # SIS (31KB)
│   ├── cosgsim3d.py                 # CoSGSIM (34KB)
│   ├── grf.py                       # GRF (21KB)
│   ├── mps.py                       # MPS (21KB)
│   └── ik_sgsim.py                  # IK-SGSIM (12KB)
├── geology/
│   ├── geological_model_engine.py   # Implicit surfaces
│   └── lithological_continuity.py   # Vein pinch-out
├── models/
│   ├── block_model.py               # BlockModel dataclass (51KB)
│   ├── pit_optimizer.py             # Pit optimization (77KB)
│   ├── jorc_classification_engine.py # JORC engine (77KB)
│   ├── sgsim3d.py                   # SGSIM engine (96KB)
│   ├── resource_reporting_engine.py # Reporting (28KB)
│   ├── transform.py                 # Data transforms (21KB)
│   └── stochastic_pit_optimizer.py  # Monte Carlo pit (7KB)
├── mine_planning/
│   ├── cutoff/                      # Cutoff optimization
│   └── scheduling/                  # Strategic + tactical
├── irr_engine/
│   └── lerchs_grossmann.py          # LG pit algorithm
├── geomet/                          # Geometallurgy
├── geotech/                         # Geotechnical
├── seismic/                         # Seismic hazard
├── esg/                             # ESG reporting
├── uncertainty_engine/              # Uncertainty propagation
├── reconciliation/                  # Mine reconciliation
├── grade_control/                   # Grade control
├── parsers/
│   ├── csv_parser.py                # Block model CSV (22KB)
│   ├── dxf_parser.py                # DXF import (12KB)
│   ├── topo_parser.py               # Topography (12KB)
│   └── structural_csv_parser.py     # Structural data (35KB)
├── visualization/
│   ├── renderer/
│   │   ├── render_orchestrator.py   # Central renderer (304KB)
│   │   ├── viewer_core.py           # VTK interactor (27KB)
│   │   └── scene_bounds.py          # Bounds management (7KB)
│   ├── block_model_mesh_builder.py  # Mesh construction (13KB)
│   ├── drillhole_gpu_renderer.py    # Drillhole rendering (79KB)
│   ├── color_mapper.py              # Colormap engine (15KB)
│   ├── filters.py                   # Clipping/slicing (17KB)
│   ├── lod_manager.py               # LOD downsampling (9KB)
│   ├── picking_controller.py        # Object picking (18KB)
│   └── overlay_manager.py           # Axes/scalebar (26KB)
├── ui/
│   ├── main_window.py               # Main application window
│   ├── viewer_widget.py             # 3D viewer widget
│   ├── panel_manager.py             # Panel lifecycle
│   ├── panel_registration.py        # Panel registry
│   ├── signals.py                   # UI signal definitions
│   ├── menus/                       # Menu bar definitions
│   ├── coordinators/                # Signal/menu coordination
│   ├── mixins/                      # Panel code reuse
│   ├── layout/                      # Dock layout persistence
│   ├── dialogs/                     # Dialog windows
│   ├── status/                      # Status bar management
│   └── ... (100+ panel files)
├── utils/
│   ├── coordinate_manager.py        # Coordinate system handling
│   ├── coordinate_utils.py          # UTM/local alignment
│   └── variogram_functions.py       # Variogram utilities
├── structural/                      # Fault/fold data models
├── risk/                            # Schedule risk
├── research/                        # Experiment tracking
├── ug/                              # Underground mining
├── haulage/                         # Haulage modelling
├── scans/                           # Point cloud processing
└── survey_deformation/              # Survey monitoring
```

---

## 9. Concurrency Model

### 9.1 Threading Strategy

| Component | Thread | Reason |
|-----------|--------|--------|
| UI widgets, signals | Main (Qt event loop) | Qt requires all widget access on main thread |
| Kriging / SGSIM / ARBF | QThread via JobWorker | CPU-intensive; blocks for seconds to minutes |
| Pit optimization | QThread via JobWorker | CPU-intensive; blocks for seconds |
| VTK rendering | Main thread | VTK is not thread-safe for actor modification |
| Audit log writes | Main thread (with file lock) | Low frequency, fast I/O |
| File parsing | QThread via JobWorker | I/O + validation can be slow for large files |

### 9.2 JobWorker Pattern

```python
class JobWorker(QThread):
    started = Signal()
    progress = Signal(int)        # 0-100
    finished = Signal(object)     # Result dataclass
    error = Signal(Exception)

    def run(self):
        try:
            self.started.emit()
            result = self.callable(
                *self.args,
                progress_fn=self.progress.emit,
                **self.kwargs
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(e)
```

### 9.3 Thread Safety Rules

1. Never modify VTK actors from a background thread. Use `QTimer.singleShot(0, ...)` to schedule actor updates on the main thread.
2. DataRegistry writes are protected by `QMutex`. Reads are lock-free (Python GIL provides sufficient protection for dict reads).
3. Progress signals are queued connections (default for cross-thread Qt signals).
4. Audit log writes use a file lock (`threading.Lock`).

---

## 10. Error Handling Strategy

### 10.1 Crash Handler (`core/crash_handler.py`)

Global exception handler installed at startup:
```python
sys.excepthook = crash_handler.handle_exception
```

On unhandled exception:
1. Log full traceback to audit log
2. Show error dialog with traceback summary
3. Attempt to save current session state
4. Do NOT terminate — allow user to continue or close gracefully

### 10.2 Engine Error Handling

Domain engines raise typed exceptions:
- `KrigingError` — singular matrix, insufficient samples
- `VariogramError` — invalid model parameters
- `CompositeError` — empty interval, missing data
- `ClassificationError` — no blocks above threshold

Controllers catch engine exceptions and:
1. Log to audit trail
2. Emit error signal
3. Panel shows user-friendly error message (not raw traceback)

### 10.3 Validation Boundaries

Validate at system boundaries only:
- **Parser input:** File existence, CSV format, column presence, numeric ranges
- **Panel input:** Widget values within bounds, required fields non-empty
- **Engine contracts:** Precondition checks (e.g., variogram sill > 0, min_samples ≥ 1)

Internal code trusts validated inputs — no redundant re-validation.

---

## 11. Testing Strategy

### 11.1 Test Organization

```
tests/
├── panel_debugger/
│   ├── core/
│   │   ├── fixtures.py           # Shared test fixtures
│   │   └── mock_factory.py       # Mock registry, controller, signals
│   └── tests/
│       ├── test_all_python_errors.py    # Syntax/import validation
│       ├── test_blank_panel_fixes.py    # Panel initialization
│       ├── test_drillhole_data_flow.py  # End-to-end drillhole pipeline
│       └── test_recent_fixes.py         # Regression tests
├── arbf/                         # ARBF engine tests
├── geostats/                     # Geostatistics engine tests
├── test_property_panel_signal_fix.py
└── test_estimations.py
```

### 11.2 Test Categories

| Category | Scope | Tools |
|----------|-------|-------|
| Unit | Individual engine functions | pytest, NumPy assertions |
| Integration | Panel → Controller → Engine → Registry | pytest, mock Qt app |
| Numerical accuracy | Engine output vs reference datasets | pytest, tolerance assertions |
| Regression | Previously fixed bugs | pytest, specific input data |
| Import validation | All .py files parse without error | AST parse check |

### 11.3 Numerical Validation Criteria

| Engine | Validation | Tolerance |
|--------|-----------|-----------|
| Kriging | LOO-CV R² vs brute-force | ±0.005 |
| ARBF | LOO-CV R² via Bartlett | ±0.005 |
| SGSIM | Variogram reproduction | Within 10% of model for 95% of lags |
| Declustering | Weighted mean vs known declustered mean | ±0.01 |
| Pit optimizer | NPV vs hand-calculated small example | ±$1 |

---

## 12. Build and Deployment

### 12.1 Development Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m block_model_viewer.main
```

### 12.2 PyInstaller Packaging

```bash
pyinstaller --onedir --windowed \
    --name GeoX \
    --icon assets/icons/app_icon.ico \
    --add-data "assets;assets" \
    block_model_viewer/main.py
```

**Output:** `dist/GeoX/` directory with `GeoX.exe` and all dependencies bundled.

### 12.3 Dependencies Frozen

`requirements.txt` pins all direct dependencies with minimum versions. Transitive dependencies are not pinned to allow compatible resolution.

---

## 13. Security Considerations

### 13.1 Input Validation (`core/security.py`)

- File path sanitization: reject paths containing `..`, null bytes, or system directories
- CSV injection prevention: strip leading `=`, `+`, `-`, `@` from cell values on import
- Maximum file size check before parsing (configurable, default 2 GB)
- Column count sanity check (reject files with >1,000 columns)

### 13.2 No Network Access

GeoX Desktop is fully offline. No telemetry, no license server, no update checks, no external API calls. All computation is local.

### 13.3 Operator Tracking

`user_auth.py` records an operator name (entered on first launch) in audit logs. This is not authentication — it is an audit trail identifier for JORC compliance.

---

## 14. Performance Optimization Notes

### 14.1 Numba JIT Compilation

Critical inner loops are compiled with Numba `@njit`:
- Kriging matrix assembly and solve
- SGSIM sequential simulation loop
- Variogram pair computation
- Declustering weight calculation

**First-call overhead:** ~2–5 seconds for JIT compilation (cached after first run).

### 14.2 Memory Management

- Large arrays (block model grids, simulation realizations) stored as NumPy arrays (contiguous memory)
- Simulation realizations computed and summarized one at a time when possible (not all held in memory simultaneously)
- PyVista grid objects share memory with NumPy arrays (zero-copy when possible)
- LOD manager reduces memory footprint for visualization of large models

### 14.3 VTK Rendering Optimization

- Edge visibility auto-disabled for >50K cells (massive GPU savings)
- LOD downsampling for >200K cells
- Actor visibility culling (off-screen actors not rendered)
- Batch property updates (single render call after all changes)
- Camera clipping range maintained dynamically (avoids Z-fighting and geometry loss)
