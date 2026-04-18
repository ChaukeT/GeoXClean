# Block Model Rendering: Framework Comparison

## Raw VTK vs PyVista vs ParaView vs GeoX Desktop

A systematic technical comparison of how mining block models are constructed, rendered, filtered, and interacted with across four frameworks.

---

# Framework 1: Raw VTK (C++/Python bindings)

## 1.1 Construction

Raw VTK provides three grid types relevant to block models. The programmer must choose and wire every stage manually.

**Grid type selection:**
- `vtkImageData` — uniform orthogonal blocks (implicit geometry, zero vertex storage)
- `vtkRectilinearGrid` — axis-aligned but variable spacing per axis
- `vtkUnstructuredGrid` — arbitrary hexahedra (rotated, sub-blocked, irregular)

**Grade/domain assignment:** Always via `vtkCellData`. VTK's data model distinguishes point data (at vertices) and cell data (per cell). Block model properties are per-block, therefore cell data.

**Active cell mask:** VTK has no built-in "active mask" concept. You either use `vtkThreshold` to extract cells, or inject NaN into scalar arrays and set the mapper's NaN color to transparent.

### Construction Code Example

```python
import vtk
import numpy as np

nx, ny, nz = 100, 100, 50
dx, dy, dz = 10.0, 10.0, 5.0
n_cells = nx * ny * nz

# --- Grid Definition ---
grid = vtk.vtkImageData()
grid.SetDimensions(nx + 1, ny + 1, nz + 1)   # Point dims = cell dims + 1
grid.SetOrigin(0.0, 0.0, 0.0)
grid.SetSpacing(dx, dy, dz)

# --- Fe Grade (continuous, cell data) ---
fe = vtk.vtkFloatArray()
fe.SetName("Fe")
fe.SetNumberOfTuples(n_cells)
rng = np.random.default_rng(42)
fe_values = rng.uniform(0, 65, n_cells).astype(np.float32)
for i in range(n_cells):
    fe.SetValue(i, fe_values[i])
grid.GetCellData().AddArray(fe)

# --- Domain Code (categorical, cell data) ---
domain = vtk.vtkIntArray()
domain.SetName("Domain")
domain.SetNumberOfTuples(n_cells)
domain_values = rng.integers(1, 5, n_cells)
for i in range(n_cells):
    domain.SetValue(i, int(domain_values[i]))
grid.GetCellData().AddArray(domain)

# --- Active Cell Mask (top 20% = air) ---
# Compute Z-center for each cell: cell_k * dz + dz/2
active = vtk.vtkUnsignedCharArray()
active.SetName("vtkGhostType")          # VTK's built-in ghost mechanism
active.SetNumberOfTuples(n_cells)
max_z = nz * dz
cutoff_z = max_z * 0.8
for i in range(n_cells):
    k = i // (nx * ny)
    z_center = k * dz + dz / 2.0
    if z_center >= cutoff_z:
        active.SetValue(i, vtk.vtkDataSetAttributes.HIDDENCELL)
    else:
        active.SetValue(i, 0)
grid.GetCellData().AddArray(active)

grid.GetCellData().SetActiveScalars("Fe")
```

### Rendering Code Example

```python
# --- Continuous colormap (Fe grade) ---
lut = vtk.vtkLookupTable()
lut.SetNumberOfTableValues(256)
lut.SetRange(0, 65)
lut.SetHueRange(0.667, 0.0)    # Blue → Red
lut.Build()

mapper = vtk.vtkDataSetMapper()
mapper.SetInputData(grid)
mapper.SetScalarModeToUseCellData()
mapper.SelectColorArray("Fe")
mapper.SetScalarVisibility(True)
mapper.SetLookupTable(lut)
mapper.SetScalarRange(0, 65)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.1, 0.1, 0.1)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(1200, 800)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)
interactor.Initialize()
window.Render()

# --- Switch to categorical domain colormap ---
mapper.SelectColorArray("Domain")
cat_lut = vtk.vtkLookupTable()
cat_lut.SetNumberOfTableValues(4)
cat_lut.SetRange(1, 4)
cat_lut.SetTableValue(0, 0.122, 0.467, 0.706, 1.0)   # Domain 1
cat_lut.SetTableValue(1, 1.000, 0.498, 0.055, 1.0)   # Domain 2
cat_lut.SetTableValue(2, 0.173, 0.627, 0.173, 1.0)   # Domain 3
cat_lut.SetTableValue(3, 0.839, 0.153, 0.157, 1.0)   # Domain 4
cat_lut.Build()
mapper.SetLookupTable(cat_lut)
mapper.SetScalarRange(1, 4)
window.Render()
```

### Threshold + Clip Example

```python
# --- Show only Fe > 45% AND Domain == 2 ---
# Step 1: Threshold by domain
domain_thresh = vtk.vtkThreshold()
domain_thresh.SetInputData(grid)
domain_thresh.SetInputArrayToProcess(0, 0, 0,
    vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Domain")
domain_thresh.SetLowerThreshold(2)
domain_thresh.SetUpperThreshold(2)
domain_thresh.Update()

# Step 2: Threshold by Fe grade
fe_thresh = vtk.vtkThreshold()
fe_thresh.SetInputConnection(domain_thresh.GetOutputPort())
fe_thresh.SetInputArrayToProcess(0, 0, 0,
    vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Fe")
fe_thresh.SetLowerThreshold(45.0)
fe_thresh.Update()

mapper2 = vtk.vtkDataSetMapper()
mapper2.SetInputConnection(fe_thresh.GetOutputPort())
mapper2.SetScalarModeToUseCellData()
mapper2.SelectColorArray("Fe")
mapper2.SetLookupTable(lut)
mapper2.SetScalarRange(0, 65)
```

### Pipeline Diagram

```
vtkImageData (origin + spacing + dims)
    │
    ├── cell_data["Fe"]        (vtkFloatArray)
    ├── cell_data["Domain"]    (vtkIntArray)
    └── cell_data["vtkGhostType"] (hidden cell mask)
            │
            ▼
     vtkThreshold (Domain == 2)
            │
            ▼
     vtkThreshold (Fe > 45)
            │
            ▼
     vtkDataSetMapper
       ├── SetScalarModeToUseCellData()
       ├── SelectColorArray("Fe")
       ├── SetLookupTable(lut)
       └── SetScalarRange(0, 65)
            │
            ▼
       vtkActor
            │
            ▼
       vtkRenderer → vtkRenderWindow → GPU
```

### Strengths and Weaknesses

| Aspect | Assessment |
|--------|------------|
| **Strengths** | Full control over every pipeline stage; zero abstraction overhead; vtkGhostType for native cell masking; maximum performance with ImageData |
| **Weaknesses** | Extreme verbosity (50+ lines for basic render); manual LUT construction; no Pythonic API; error-prone array index loops; no built-in colormap library |

---

# Framework 2: PyVista

## 2.1 Construction

PyVista wraps VTK with a NumPy-native API. The same three grid types are available but creation is 5–10× less code.

### Construction Code Example

```python
import pyvista as pv
import numpy as np

nx, ny, nz = 100, 100, 50
dx, dy, dz = 10.0, 10.0, 5.0
rng = np.random.default_rng(42)

# --- Grid Definition (one line) ---
grid = pv.ImageData(dimensions=(nx+1, ny+1, nz+1),
                    spacing=(dx, dy, dz),
                    origin=(0.0, 0.0, 0.0))

# --- Properties as numpy arrays directly into cell_data ---
n_cells = grid.n_cells
grid.cell_data["Fe"] = rng.uniform(0, 65, n_cells).astype(np.float32)
grid.cell_data["Domain"] = rng.integers(1, 5, n_cells).astype(np.int32)

# --- Active cell mask (top 20% = air → NaN) ---
# Compute Z-center per cell using cell_centers()
centers = grid.cell_centers().points
z_cutoff = nz * dz * 0.8
air_mask = centers[:, 2] >= z_cutoff

# Mask Fe values — air cells become NaN (rendered transparent)
fe = grid.cell_data["Fe"].copy()
fe[air_mask] = np.nan
grid.cell_data["Fe"] = fe
```

### Rendering Code Example

```python
# --- Continuous colourmap (Fe grade) ---
plotter = pv.Plotter()
plotter.add_mesh(grid, scalars="Fe", cmap="turbo",
                 clim=[0, 65], nan_color=(0,0,0,0), nan_opacity=0.0,
                 show_edges=False, preference="cell",
                 smooth_shading=False, interpolate_before_map=False)
plotter.add_scalar_bar("Fe (%)", vertical=True)
plotter.show()

# --- Categorical colourmap (Domain codes) ---
plotter2 = pv.Plotter()
plotter2.add_mesh(grid, scalars="Domain", cmap="tab10",
                  clim=[1, 4], n_colors=4,
                  show_edges=False, preference="cell",
                  smooth_shading=False, interpolate_before_map=False)
plotter2.add_scalar_bar("Domain", vertical=True)
plotter2.show()
```

### Threshold + Clip Example

```python
# --- Show only Fe > 45% within Domain 2 ---
# Step 1: Threshold by domain (range [2,2] for exact match)
domain2 = grid.threshold(value=[2, 2], scalars="Domain",
                         preference="cell")

# Step 2: Threshold by Fe grade
high_fe = domain2.threshold(value=45, scalars="Fe")

# Step 3: Render
plotter3 = pv.Plotter()
plotter3.add_mesh(high_fe, scalars="Fe", cmap="turbo",
                  clim=[0, 65], show_edges=True, preference="cell",
                  smooth_shading=False, interpolate_before_map=False)
plotter3.show()

# --- Cross-section clip ---
clipped = grid.clip(normal="x", origin=grid.center)
plotter4 = pv.Plotter()
plotter4.add_mesh(clipped, scalars="Fe", cmap="turbo",
                  clim=[0, 65], preference="cell")
plotter4.show()
```

### Pipeline Diagram

```
pv.ImageData(dims, spacing, origin)
    │
    ├── cell_data["Fe"]     = np.ndarray (float32)
    ├── cell_data["Domain"] = np.ndarray (int32)
    └── NaN injection for air mask
            │
            ▼
     .threshold(value=[2,2], scalars="Domain") → pv.UnstructuredGrid
            │
            ▼
     .threshold(value=45, scalars="Fe")      → pv.UnstructuredGrid
            │
            ▼
     Plotter.add_mesh(scalars="Fe", cmap="turbo", clim=[0,65])
       └── internally: vtkDataSetMapper + vtkLookupTable + vtkActor
            │
            ▼
     pv.Plotter.show()  →  vtkRenderWindow → GPU
```

### Strengths and Weaknesses

| Aspect | Assessment |
|--------|------------|
| **Strengths** | NumPy-native (no loops for array assignment); one-line grid creation; built-in threshold/clip/slice; matplotlib colormaps via string names; NaN transparency built-in |
| **Weaknesses** | threshold() on ImageData returns UnstructuredGrid (loses memory advantage); no built-in actor registry or lifecycle management; no ghost cell API (must use NaN workaround); limited LUT update without rebuilding actor |

---

# Framework 3: ParaView

## 3.1 Construction

ParaView operates as a GUI application backed by the ParaView Server (VTK pipeline engine). Block models are imported via readers or programmatic sources. The Python scripting API (`paraview.simple`) mirrors GUI operations.

ParaView natively supports vtkImageData, vtkRectilinearGrid, and vtkUnstructuredGrid. Block models from Leapfrog/Vulcan are typically imported as vtkUnstructuredGrid via custom readers or CSV-to-TableToStructuredGrid pipelines.

### Construction Code Example (pvpython)

```python
from paraview.simple import *
import numpy as np

nx, ny, nz = 100, 100, 50
dx, dy, dz = 10.0, 10.0, 5.0
n_cells = nx * ny * nz

# --- Programmatic Source ---
source = ProgrammableSource()
source.OutputDataSetType = "vtkImageData"
source.Script = f"""
import vtk
import numpy as np

output = self.GetOutput()
output.SetDimensions({nx+1}, {ny+1}, {nz+1})
output.SetOrigin(0.0, 0.0, 0.0)
output.SetSpacing({dx}, {dy}, {dz})

rng = np.random.default_rng(42)
n = {n_cells}

fe = vtk.vtkFloatArray()
fe.SetName("Fe")
fe.SetNumberOfTuples(n)
fe_vals = rng.uniform(0, 65, n).astype('float32')
for i in range(n):
    fe.SetValue(i, float(fe_vals[i]))
output.GetCellData().AddArray(fe)

domain = vtk.vtkIntArray()
domain.SetName("Domain")
domain.SetNumberOfTuples(n)
dom_vals = rng.integers(1, 5, n)
for i in range(n):
    domain.SetValue(i, int(dom_vals[i]))
output.GetCellData().AddArray(domain)

# Ghost cells for top 20%
ghost = vtk.vtkUnsignedCharArray()
ghost.SetName(vtk.vtkDataSetAttributes.GhostArrayName())
ghost.SetNumberOfTuples(n)
max_z = {nz * dz}
for i in range(n):
    k = i // ({nx} * {ny})
    z = k * {dz} + {dz} / 2.0
    if z >= max_z * 0.8:
        ghost.SetValue(i, vtk.vtkDataSetAttributes.HIDDENCELL)
    else:
        ghost.SetValue(i, 0)
output.GetCellData().AddArray(ghost)
"""
source.UpdatePipeline()

# --- Render ---
Show(source)
display = GetDisplayProperties(source)
display.Representation = "Surface"
ColorBy(display, ("CELLS", "Fe"))
feLUT = GetColorTransferFunction("Fe")
feLUT.RescaleTransferFunction(0, 65)
feLUT.ApplyPreset("Turbo", True)
GetScalarBar(feLUT).Visibility = 1
Render()
```

### Threshold + Clip Example (pvpython)

```python
# --- Threshold: Domain == 2 ---
thresh1 = Threshold(Input=source)
thresh1.Scalars = ["CELLS", "Domain"]
thresh1.LowerThreshold = 2
thresh1.UpperThreshold = 2
thresh1.UpdatePipeline()

# --- Threshold: Fe > 45 ---
thresh2 = Threshold(Input=thresh1)
thresh2.Scalars = ["CELLS", "Fe"]
thresh2.LowerThreshold = 45
thresh2.UpdatePipeline()

Show(thresh2)
ColorBy(GetDisplayProperties(thresh2), ("CELLS", "Fe"))
Render()

# --- Cross-section clip ---
clip = Clip(Input=source)
clip.ClipType = "Plane"
clip.ClipType.Origin = [500, 500, 125]
clip.ClipType.Normal = [1, 0, 0]
clip.UpdatePipeline()
Show(clip)
Render()
```

### Pipeline Diagram

```
ProgrammableSource (vtkImageData)
    │
    ├── CellData["Fe"]
    ├── CellData["Domain"]
    └── CellData[GhostArray] (HIDDENCELL mask)
            │
            ▼
     Threshold (Domain == 2)     ← GUI filter or Python
            │
            ▼
     Threshold (Fe > 45)
            │
            ▼
     Representation: Surface
     ColorBy: ("CELLS", "Fe")
     ColorTransferFunction: Turbo preset, range [0, 65]
     ScalarBar: visible
            │
            ▼
     ParaView RenderView → VTK Server → GPU
```

### Strengths and Weaknesses

| Aspect | Assessment |
|--------|------------|
| **Strengths** | Native ghost cell support; built-in scalar bar/colormap GUI; threshold/clip as pipeline objects (lazy, cached); distributed rendering for massive models; GUI inspection of every pipeline stage; undo/redo for filter operations |
| **Weaknesses** | Heavy process (server/client architecture); scripting API is verbose and differs from VTK; no mining-specific semantics (JORC, NSR, composites); programmable sources require VTK-level code; limited Qt integration for custom desktop apps |

---

# Framework 4: GeoX Desktop (PyQt6 + PyVista + pyvistaqt)

## 4.1 Construction

GeoX uses a multi-layer architecture purpose-built for mining block models. The data model (`BlockModel`), mesh builder (`block_model_mesh_builder`), and renderer (`BlockModelRenderer`) are separated.

**Grid type:** Auto-detected by `is_uniform_grid()`. Uniform orthogonal models use `pv.ImageData`; rotated or sub-blocked models use `pv.UnstructuredGrid` with vectorized hexahedron generation.

**Property assignment:** All properties are cell data. Float properties stay float64 (never downcast — RES-05 fix preserves kriging variance precision for JORC thresholds). Integer properties are optimally packed (uint8/uint16/int16).

**Active cell mask:** Dual mechanism:
1. `domain_mask` cell_data array (boolean: 1=active, 0=inactive) — authoritative geometry mask
2. NaN injection into scalar arrays for transparent rendering — reversible, no topology rebuild

**Coordinate handling:** Global UTM→local shift locked by first-dataset authority, with a double-shift guard for SGSIM output already in local coordinates.

### Construction Code Example

```python
import numpy as np
from block_model_viewer.models.block_model import BlockModel, BlockMetadata
from block_model_viewer.visualization.block_model_mesh_builder import (
    generate_block_model_mesh, is_uniform_grid
)

nx, ny, nz = 100, 100, 50
dx, dy, dz = 10.0, 10.0, 5.0
rng = np.random.default_rng(42)

# --- Data Model (framework-specific abstraction) ---
bm = BlockModel(metadata=BlockMetadata(
    source_file="synthetic.csv",
    units="meters",
    coordinate_system="WGS84 UTM Zone 35S"
))

# Generate grid positions (centroids)
xs = np.arange(nx) * dx + dx / 2
ys = np.arange(ny) * dy + dy / 2
zs = np.arange(nz) * dz + dz / 2
xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
dimensions = np.tile([dx, dy, dz], (len(positions), 1))

bm.set_geometry(positions, dimensions)

# --- Properties ---
n = bm.block_count
bm.add_property("Fe", rng.uniform(0, 65, n).astype(np.float64))
bm.add_property("Domain", rng.integers(1, 5, n).astype(np.int32))

# --- Active Cell Mask (top 20% = air) ---
z_cutoff = nz * dz * 0.8
active = (positions[:, 2] < z_cutoff).astype(np.int32)
bm.add_property("domain_mask", active)

# --- Mesh Generation (auto-detects ImageData) ---
mesh = generate_block_model_mesh(bm)
# Returns pv.ImageData for this uniform grid
# Properties assigned as cell_data with sentinel-aware indexing
# Original_ID mapping for O(1) picking included automatically
```

### Rendering Code Example

```python
# Inside GeoX, the BlockModelRenderer handles this:
# renderer.block_model_renderer.load_block_model(bm)
#
# Which internally calls:
#   1. _generate_block_meshes()      → auto ImageData/UnstructuredGrid
#   2. _apply_coordinate_transform() → UTM→local shift
#   3. _add_meshes_to_plotter()      → add_mesh with mining-specific kwargs

# The actual add_mesh call (from _add_meshes_to_plotter):
import pyvista as pv

# This is what GeoX does internally:
plotter = pv.Plotter()
actor = plotter.add_mesh(
    mesh,
    scalars="Fe",
    cmap="turbo",
    clim=[0, 65],
    nan_color=(0, 0, 0, 0),            # Transparent air/inactive
    nan_opacity=0.0,
    show_edges=False,                   # Disabled for >50k cells
    smooth_shading=False,               # CRITICAL: flat shading for blocks
    interpolate_before_map=False,       # CRITICAL: no color interpolation
    preference="cell",                  # Cell data, not point data
    style="surface",
    ambient=0.55, diffuse=0.45, specular=0.05,
    pbr=False,
    pickable=True,
)

# --- Switch to categorical domain colouring ---
# GeoX does this via in-place LUT update (no geometry rebuild):
# renderer.block_model_renderer.set_property_coloring("Domain", "tab10", discrete=True)
#
# Internally:
mapper = actor.GetMapper()
cd = mapper.GetInputAsDataSet().GetCellData()
cd.SetActiveScalars("Domain")
mapper.SetScalarModeToUseCellData()
mapper.SetColorModeToMapScalars()

lut = mapper.GetLookupTable()
lut.SetRange(1, 4)
lut.SetNumberOfTableValues(4)
import matplotlib
cmap = matplotlib.colormaps["tab10"]
for i in range(4):
    rgba = cmap(i / 3.0)
    lut.SetTableValue(i, *rgba)
lut.Build()
mapper.Modified()
plotter.render()
```

### Threshold + Clip Example

```python
# GeoX uses extract_cells with boolean masks (not chained threshold filters):
import numpy as np

fe = np.asarray(mesh.cell_data["Fe"])
domain = np.asarray(mesh.cell_data["Domain"])

mask = (domain == 2) & (fe > 45) & np.isfinite(fe)
filtered = mesh.extract_cells(np.nonzero(mask)[0])

plotter_filtered = pv.Plotter()
plotter_filtered.add_mesh(filtered, scalars="Fe", cmap="turbo",
                          clim=[0, 65], preference="cell",
                          smooth_shading=False,
                          interpolate_before_map=False)
plotter_filtered.show()

# For domain masking without topology rebuild (GeoX's NaN injection):
# BlockModelRenderer.apply_domain_mask_transparency(actor, mesh, opacity=1.0)
# This injects NaN into hidden cells' active scalars → transparent via nan_color
```

### Pipeline Diagram

```
BlockModel (positions, dimensions, properties, metadata)
    │
    ▼
is_uniform_grid()  [auto-detection: 4 checks]
    │
    ├──YES──→ build_uniform_grid() → pv.ImageData
    │           ├── cell_data props via 3D array indexing
    │           ├── sentinel: NaN (float) / iinfo.min (int)
    │           └── Original_ID for O(1) picking
    │
    └──NO───→ build_unstructured_grid() → pv.UnstructuredGrid
                ├── vectorized hexahedron generation
                ├── rotation matrix applied to offsets
                └── Original_ID = arange(N)
                        │
                        ▼
          _apply_coordinate_transform_to_meshes()
                ├── UTM → local shift (locked by first dataset)
                ├── Double-shift guard for SGSIM output
                └── ImageData: shift origin; UG: shift points
                        │
                        ▼
          _add_meshes_to_plotter()
                ├── Optional: extract_cells(visible_blocks)
                ├── add_mesh(smooth_shading=False,
                │            interpolate_before_map=False,
                │            preference="cell", pickable=True)
                └── apply_domain_mask_transparency()
                        │
                        ▼
          set_property_coloring()  [on user interaction]
                ├── In-place LUT update (no geometry rebuild)
                ├── Active scalar switch via mapper
                ├── 256-entry LUT from matplotlib cmap
                ├── NSR auto-detection → divergent colormap
                └── Re-apply domain_mask NaN injection
                        │
                        ▼
          PickingController + BlockModelPickAdapter
                ├── LOD-gated (P0/P1/P2/P3)
                ├── vtkPropPicker (actor-level hover)
                ├── vtkCellPicker (cell-level click)
                └── Original_ID → O(1) block data lookup
                        │
                        ▼
          ActorRegistry (category: block_model)
                └── Explicit lifecycle: add/remove/clear_category
```

---

# GeoX-Specific Architectural Evaluation

## Geometry Generation vs Actor Presentation Separation

**Assessment: Well separated.**

Geometry generation happens in `block_model_mesh_builder.py` — a pure module with no PyVista plotter references. It receives a `BlockModel` and returns a `pv.DataSet`. The renderer coordination layer (`BlockModelRenderer`) then adds this mesh to the plotter. The two concerns are in separate files with a clean data boundary (mesh object).

However, the `_add_meshes_to_plotter` method also performs visibility filtering (`extract_cells`) which is a geometry operation inside the presentation layer. This could be refactored so filtering happens in the mesh builder and the plotter method receives only ready-to-render geometry.

## Actor Registry and Lifecycle Management

**Assessment: Partially explicit, with dual-tracking debt.**

The `ActorRegistry` class provides explicit `add/remove/get_all/clear_category` with category-based organisation. Block model actors are registered under category `"block_model"`.

However, the `mesh_actor` reference is also stored directly on the `Renderer` object (`self._renderer.mesh_actor`), and `_add_meshes_to_plotter` manually calls `plotter.remove_actor(self._renderer.mesh_actor)` instead of going through the registry. This dual-tracking creates a risk of desynchronisation. The registry knows about the actor, but removal bypasses it.

## Scalar Array Binding Pipeline Compliance

**Assessment: Follows the pipeline model correctly with one shortcut.**

The GeoX pipeline is:
```
raw data (BlockModel)
  → structured/unstructured grid (pv.ImageData / pv.UnstructuredGrid)
    → filter (extract_cells, domain_mask)
      → mapper (vtkDataSetMapper via add_mesh)
        → actor (vtkActor)
          → renderer (pv.Plotter → vtkRenderer)
```

No levels are skipped. The `set_property_coloring` method correctly goes through the mapper to update active scalars and LUT without bypassing to direct actor color manipulation.

The one shortcut: domain mask transparency uses NaN injection into cell_data scalars (modifying the grid-level data) rather than a filter-level operation. This is an intentional design trade-off — NaN injection is O(1) to apply and reversible, while a filter would require topology rebuild. This is a valid deviation from pure pipeline orthodoxy.

## Active Cell Mask Handling

**Assessment: Robust, with explicit authoritative source.**

The `_compute_active_cell_mask` static method establishes a clear hierarchy:
1. `domain_mask` array is the authoritative geometry mask (if present)
2. Finite-value check is a secondary filter for truly invalid cells
3. The composite mask is: `(domain_mask > 0) AND isfinite(values)`

This means the rendering layer never does its own classification logic — it respects whatever mask the data layer provides. Sentinel values (`iinfo.min` for integers, NaN for floats) are handled consistently.

One risk: `extract_cells(visible_blocks)` in `_add_meshes_to_plotter` can also hide cells based on UI-driven visibility, which operates in parallel with `domain_mask`. These two masking systems could theoretically conflict. In practice they compose correctly (extract_cells runs first, then domain_mask NaN injection runs on the result).

## Classification Panel Support

**Assessment: Clean separation — renderer does not classify.**

The `BlockModel` stores classification properties (e.g., `jorc_class`) as cell data arrays just like any other property. The classification panel writes values to `BlockModel._properties["jorc_class"]`, and the renderer simply colours by that array using `set_property_coloring("jorc_class", "Set1", discrete=True)`. The renderer has no knowledge of JORC categories (Measured/Indicated/Inferred) — it treats them as categorical integers.

The only exception is NSR auto-detection in `set_property_coloring`, where the method checks if the property name is "NSR" and applies a divergent colormap. This is a presentation concern (how to display NSR) rather than classification logic, so it is acceptable.

---

# Cross-Framework Comparison Tables

## Construction Comparison

| Aspect | Raw VTK | PyVista | ParaView | GeoX Desktop |
|--------|---------|---------|----------|--------------|
| Grid creation | Manual `SetDimensions/Spacing/Origin` | One-line `pv.ImageData(...)` | ProgrammableSource with VTK code | `generate_block_model_mesh()` auto-detects |
| Sub-blocks | Manual hexahedra in UnstructuredGrid | Same, but numpy-native | Same via programmatic source | Vectorized hex generation with rotation support |
| Property assignment | Loop with `SetValue(i, v)` | `grid.cell_data["Fe"] = array` | Loop with `SetValue(i, v)` inside source | `bm.add_property("Fe", array)` with dtype optimization |
| Active cell mask | `vtkGhostType` with `HIDDENCELL` | NaN injection (no ghost API) | Ghost array (native VTK support) | `domain_mask` array + NaN injection (dual mechanism) |
| Coordinate handling | Manual | Manual | Manual | Auto UTM→local shift with double-shift guard |
| Code lines (100³×50) | ~60 | ~15 | ~50 (pvpython) | ~20 (data model) + 0 (renderer auto) |

## Rendering Pipeline Comparison

| Aspect | Raw VTK | PyVista | ParaView | GeoX Desktop |
|--------|---------|---------|----------|--------------|
| Pipeline stages | All manual (mapper, actor, renderer, window) | `add_mesh()` wraps all stages | GUI representation + display properties | `_add_meshes_to_plotter()` with mining-specific defaults |
| Colormap binding | Manual `vtkLookupTable` (per-entry) | `cmap="turbo"` string → matplotlib | Preset name via GUI or `ApplyPreset()` | In-place 256-entry LUT from matplotlib; NSR auto-divergent |
| Categorical colours | Manual LUT entries | `n_colors` + categorical cmap | GUI categorical mapping | `_is_categorical()` auto-detection with dtype-aware rules |
| Threshold | `vtkThreshold` filter chain | `.threshold(value, scalars)` | Threshold filter (lazy, cached) | `extract_cells(boolean_mask)` or NaN injection |
| Clip | `vtkClipDataSet` | `.clip(normal, origin)` | Clip filter (plane, box, sphere) | `.clip()` via PyVista |
| Opacity/transparency | Actor property + depth sorting | `opacity=` parameter + `nan_opacity` | Representation opacity + volume rendering | `nan_color=(0,0,0,0)` + domain_mask NaN injection |
| LUT update | Rebuild `vtkLookupTable` | Rebuild actor (`add_mesh` again) | `RescaleTransferFunction` | In-place mapper LUT update (no rebuild) |

## Performance Comparison

| Aspect | Raw VTK | PyVista | ParaView | GeoX Desktop |
|--------|---------|---------|----------|--------------|
| >1M cell ImageData | 40–120 FPS (implicit geometry) | Same (wraps VTK) | Same + distributed rendering option | Same + sentinel-based sparse grid support |
| >1M cell UnstructuredGrid | 2–8 FPS (explicit vertices) | Same | Distributed server can split across nodes | Same + Numba-accelerated corner generation |
| Caching | Manual (programmer responsibility) | No built-in cache | Pipeline caching (lazy execution) | `_drillhole_polylines_cache`, LUT in-place update, `_current_grid_signature` cache key |
| LOD | Manual | None built-in | Hierarchical LOD in distributed mode | PickingController LOD (P0–P3) for hover/click; edge display gated at 50k cells |
| Bottleneck: property switch | Rebuild LUT only | Rebuild entire actor | Update transfer function only | In-place LUT update (no actor rebuild) |
| Bottleneck: visibility change | Manual extract_cells | Same | Hide representation | extract_cells OR NaN injection (reversible) |
| Deep copy avoidance | Manual | `threshold()` creates copies | Pipeline caching avoids copies | `_coordinate_shifted` flag prevents double-shift; domain mask modifies scalars in-place |

## Interaction Comparison

| Aspect | Raw VTK | PyVista | ParaView | GeoX Desktop |
|--------|---------|---------|----------|--------------|
| Cell picking | `vtkCellPicker` (manual setup) | `plotter.enable_cell_picking()` | Built-in cell inspector | `BlockModelPickAdapter` with `Original_ID` O(1) lookup |
| Hover tooltip | Manual interactor style | Callback-based | Built-in hover tooltip | LOD-gated hover (disabled >200k cells or >5ms) |
| Legend/scalar bar | `vtkScalarBarActor` (manual) | `add_scalar_bar()` | Built-in scalar bar widget | Custom `LegendManager` (decoupled from PyVista scalar bar) |
| Orientation axes | `vtkAxesActor` + widget | `add_axes()` | Built-in orientation axes | Custom overlay via `OverlayBridge` |
| Camera control | Manual `vtkCamera` | `plotter.camera_position` | Built-in camera controls | Explicit clipping range from model bounds; mining-standard views |

## Geological Fidelity Comparison

| Aspect | Raw VTK | PyVista | ParaView | GeoX Desktop |
|--------|---------|---------|----------|--------------|
| Grade cutoff rendering | Manual threshold | `.threshold()` | Threshold filter | Boolean mask + extract_cells; preserves exact boundary |
| Domain boundary display | Manual LUT | `n_colors` param | Categorical colormap | `_is_categorical()` auto-detection; dtype-aware (Issue #4 fix) |
| JORC classification overlay | Not built-in | Not built-in | Not built-in | Classification as property array; renderer colours by it without interpreting it |
| NaN/NoData display | `vtkGhostType` hides; or manual | `nan_color` + `nan_opacity` | Ghost cells; or colormap NaN handling | `nan_color=(0,0,0,0)` + sentinel-aware (`iinfo.min` for int, NaN for float) |
| NSR divergent display | Manual divergent LUT | `cmap="seismic"` (manual choice) | Manual color transfer function | Auto-detected by property name; zero-centered divergent map |
| Precision at classification boundaries | float64 if programmed | float64 if numpy array is float64 | Depends on source | float64 enforced (RES-04/05); no downcast to float32 |

---

# Final Ranking

| Criterion | Raw VTK | PyVista | ParaView | GeoX Desktop |
|-----------|---------|---------|----------|--------------|
| **Ease of use** | 2/10 | 8/10 | 6/10 | 7/10 |
| **Performance at scale** | 9/10 | 8/10 | 10/10 | 8/10 |
| **Geological fidelity** | 4/10 | 5/10 | 5/10 | 9/10 |
| **Interactivity** | 3/10 | 6/10 | 9/10 | 8/10 |
| **Production mining suitability** | 2/10 | 4/10 | 5/10 | 9/10 |
| **Overall** | **4.0** | **6.2** | **7.0** | **8.2** |

### Rationale

**Raw VTK** scores highest on raw performance control but is impractical for mining software development — the verbosity and lack of geological semantics make it a foundation, not a product.

**PyVista** is the right abstraction for scripting and prototyping. Its weakness is the lack of in-place LUT updates (property switches rebuild actors) and no built-in lifecycle management for multi-layer scenes.

**ParaView** excels at interactive exploration and massive datasets (distributed rendering). Its scripting API is awkward, it has no mining-specific features (JORC, NSR, composites), and embedding it in a custom desktop app requires paraview server architecture.

**GeoX Desktop** is purpose-built for production mining. It auto-detects ImageData vs UnstructuredGrid, handles coordinate precision for JORC compliance, provides in-place LUT updates for responsive property switching, and has LOD-gated picking tuned for large block models. Its weaknesses are the dual actor-tracking pattern (registry + direct reference) and the extract_cells bottleneck when toggling visibility on ImageData grids.

---

# Appendix: Key VTK Pipeline Rules (from geox-pyvista-vtk-engine)

These rules apply to all four frameworks, since they all ultimately execute VTK pipelines:

1. **Always reason in order**: raw data → grid → filter → mapper → actor → renderer. Never skip levels.
2. **Know if data is point data or cell data**. Block model properties are cell data. Do not mix without explicit conversion.
3. **Threshold removes cells** — it can break topology. Use for selection, not for smooth surfaces.
4. **Clip preserves topology** better than threshold.
5. **Avoid repeated mesh creation inside loops** — cache processed grids.
6. **Avoid deep copies unless required** — NaN injection modifies in-place; extract_cells creates a copy.
7. **Reduce actor count** — one merged mesh is better than N individual actors.
8. **Validate coordinate systems and units before blaming rendering.**
