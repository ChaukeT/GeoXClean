# GeoX Block Model Rendering Architecture

## Complete Pipeline: From Raw Data to Screen Pixels

This document traces every step of how a block model goes from imported file data to a rendered 3D voxel grid in the PyVista viewport.

---

## 1. The Data Layer

**File:** `models/block_model.py`

The `BlockModel` class is the canonical store for all block model data. It uses numpy arrays (not Python objects) for vectorized performance.

### Core geometry:

| Attribute | Type | Description |
|-----------|------|-------------|
| `_positions` | (N, 3) float64 | Block center coordinates |
| `_dimensions` | (N, 3) float64 | Block sizes (dx, dy, dz) per block |
| `_rotation_matrix` | 3×3 float64 or None | Rotation for Leapfrog-style rotated grids |
| `_properties` | Dict[str, ndarray] | Grade, domain, density, kriging variance, etc. |

### Key design decisions:

**Coordinates stay in float64** (RES-04 fix). Float32 has ~7 significant digits — for UTM coordinates like X=537,250.125m that gives ≤0.0625m precision, enough to misclassify blocks near JORC classification threshold boundaries. The extra 12 bytes/block (12MB for 1M blocks) is negligible.

**Integer property dtype optimization**: Domain codes in uint8 (0–255), uint16 (0–65535) etc. But float properties are **never** downcast from float64 to float32 (RES-05 fix) because kriging variance values like KV=0.0000123 lose relative precision at float32.

### Metadata tracking (`BlockMetadata`):

```python
@dataclass
class BlockMetadata:
    coordinate_system: str       # CRS identifier
    units: str                   # "meters"
    source_file: str             # Original file path
    file_checksum: str           # SHA-256 for integrity
    parser_version: str          # Reproducibility
    column_mapping: Dict         # How columns were mapped
    inferred_dimensions: bool    # Whether block sizes were guessed
```

### Kriging result storage:

The model supports UK, CoK, and IK results with standardized naming:
- `uk_Fe`, `uk_Fe_var` — Universal Kriging estimate + variance
- `cok_Fe`, `cok_Fe_var` — Co-Kriging
- `ik_Fe_p_le_60`, `ik_Fe_median` — Indicator Kriging probabilities per threshold

---

## 2. Mesh Generation: The Fork Between ImageData and UnstructuredGrid

**File:** `visualization/block_model_mesh_builder.py`

This is the single most impactful architectural decision in the renderer. The choice between `pv.ImageData` and `pv.UnstructuredGrid` determines memory usage by 40–100× and FPS by 5–15×.

### The detection algorithm: `is_uniform_grid()`

Four checks determine whether a block model qualifies for the memory-efficient ImageData path:

**Check 1 — Constant spacing**: All blocks must have the same dx, dy, dz. For single-value axes (e.g., a 100×100×1 2D model), spacing is derived from block dimensions. Floating-point jitter is tolerated via `np.allclose(diffs, median, rtol=1e-6)` (FP-09 fix).

**Check 2 — Axis-aligned**: The rotation matrix must be None or identity. Rotated Leapfrog grids fail this check and go to UnstructuredGrid.

**Check 3 — Grid completeness**: Sparse grids are allowed if `fill_ratio > 1%`. This is critical for mining — block models are commonly sparse (only blocks within the estimation domain are exported). Rejecting sparse grids would force them into UnstructuredGrid, creating "island" artefacts instead of the continuous orebody that Leapfrog, Surpac, and Datamine display.

**Check 4 — Uniform block dimensions**: `np.std(dimensions, axis=0) ≤ tolerance`.

### Path A: ImageData (uniform grids)

**Memory**: ~24 bytes for geometry (origin + spacing + dims) + cell_data arrays. No vertex storage — geometry is **implicit**.

**Construction**:
```python
grid = pv.ImageData()
grid.origin = (x_min - dx/2, y_min - dy/2, z_min - dz/2)  # Corner, not centroid
grid.spacing = (dx, dy, dz)
grid.dimensions = (nx+1, ny+1, nz+1)  # VTK uses point count = cells + 1
```

**Block index mapping** (centroid → cell index):
```python
x_idx = floor((position_x - origin_x) / spacing_x)
y_idx = floor((position_y - origin_y) / spacing_y)
z_idx = floor((position_z - origin_z) / spacing_z)
```

Using `floor` with corner-based origin is mathematically robust at boundaries. The old `round` with centroid-based origin had edge-case bugs from banker's rounding (Issue #19 fix).

**Property assignment** into 3D arrays:
```python
# Initialize with appropriate sentinel
if integer_dtype:
    sentinel = np.iinfo(dtype).min    # NOT -1, which collides with domain codes
else:
    sentinel = np.nan

prop_array = np.full((nz, ny, nx), sentinel, dtype=...)
prop_array[z_idx, y_idx, x_idx] = values
grid.cell_data[name] = prop_array.ravel(order='C')
```

**Original_ID mapping** (critical for picking):
```python
original_id_array = np.full((nz, ny, nx), np.iinfo(np.int64).min, dtype=np.int64)
original_id_array[z_idx, y_idx, x_idx] = np.arange(N)
grid.cell_data['Original_ID'] = original_id_array.ravel(order='C')
```

This enables O(1) block lookup during hover/click. The sentinel (`np.iinfo(np.int64).min ≈ -9.2×10¹⁸`) is safely outside any realistic block count.

### Path B: UnstructuredGrid (rotated/irregular)

**Memory**: 8 vertices × N blocks × 3 coords × 8 bytes = ~192 bytes/block for geometry alone. For 10M blocks that's ~1.9GB.

**Hexahedron generation** (fully vectorized):
```python
signs = [[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
         [-1,-1,1],  [1,-1,1],  [1,1,1],  [-1,1,1]]

local_offsets = signs[None,:,:] * half_dims[:,None,:]   # (N, 8, 3)

# Apply rotation for Leapfrog-style rotated grids
if rotation_matrix is not None:
    flat_rotated = local_offsets.reshape(-1, 3) @ rotation_matrix.T
    local_offsets = flat_rotated.reshape(N, 8, 3)

corners = positions[:,None,:] + local_offsets             # (N, 8, 3)
all_points = corners.reshape(-1, 3)                       # (N*8, 3)
```

**VTK cells**: Each block is a `VTK_HEXAHEDRON` (type 12) with 8 vertices:
```python
cells = [8, v0, v1, ..., v7, 8, v8, v9, ...]  # VTK connectivity format
cell_types = [12, 12, 12, ...]
grid = pv.UnstructuredGrid(cells, cell_types, all_points)
```

There is also a **Numba-accelerated** fallback (`_compute_block_corners_numba`) for environments where Numba is available, though the NumPy broadcasting path is the primary one.

### The entry point: `generate_block_model_mesh()`

```python
def generate_block_model_mesh(block_model):
    is_uniform, grid_info = is_uniform_grid(block_model)
    if is_uniform:
        return build_uniform_grid(block_model, grid_info)
    else:
        return build_unstructured_grid(block_model)
```

If ImageData detection fails with an exception, it falls back to UnstructuredGrid. This never silently drops data.

---

## 3. The Renderer Coordination Layer

**File:** `visualization/renderer/renderers/block_model_renderer.py` — `BlockModelRenderer`

This class orchestrates the entire load-to-screen pipeline. It holds a reference to the parent `Renderer` orchestrator for shared state access.

### `load_block_model(block_model)` — The Master Pipeline

```
1. _clear_block_meshes()
   └─ Remove old actors, free GPU memory

2. _generate_block_meshes()
   └─ Auto-detect grid type → ImageData or UnstructuredGrid
   └─ Store in renderer.block_meshes['imagedata'] or ['unstructured_grid']

3. _apply_coordinate_transform_to_meshes()
   └─ Apply global shift (UTM → local) for GPU float32 precision
   └─ Double-shift guard for SGSIM output already in local coords

4. _add_meshes_to_plotter()
   └─ Visibility filtering via extract_cells()
   └─ plotter.add_mesh() with specific kwargs
   └─ Domain mask transparency

5. Camera positioning
   └─ Compute from transformed mesh bounds (not original model bounds!)
   └─ Set explicit clipping range

6. Register as active layer
   └─ renderer.add_layer(layer_name, actor, mesh, 'blocks', opacity)

7. Re-apply scene effects (SSAO/EDL)
8. Update floating axes, scene bounds, overlays
```

### Coordinate Transformation: The Double-Shift Guard

The coordinate shift converts UTM (e.g., 500,000m Easting) to local coordinates (~0,0,0) for GPU precision. But SGSIM-generated block models may already be in local coordinates (kriging runs in shifted space). Applying the shift again would push the mesh hundreds of kilometres away.

**Detection logic** (Bug #13 improved):
```python
shift_mag = max(abs(shift))
bounds_center_mag = max(abs(center_point))

already_local = (
    shift_mag > 10_000          # shift is UTM-scale (> 10 km)
    and bounds_center_mag < shift_mag * 0.1  # centre is tiny relative to shift
)
```

**Shift application** differs by grid type:
- **ImageData**: Shift the `origin` tuple (6 float subtractions)
- **RectilinearGrid**: Shift the x, y, z edge arrays
- **UnstructuredGrid/PolyData**: Shift all `points` directly

Each mesh is tagged `_coordinate_shifted = True` to prevent double-shifting.

---

## 4. Adding Meshes to the Plotter

**File:** `block_model_renderer.py` — `_add_meshes_to_plotter()`

### The critical `add_mesh` call:

```python
actor = plotter.add_mesh(
    grid,
    scalars=initial_property,          # Active scalar for coloring
    cmap=initial_colormap,             # Matplotlib colormap name
    clim=(vmin, vmax),                 # Color scale limits
    nan_color=(0, 0, 0, 0),           # Fully transparent NaN cells
    nan_opacity=0.0,                   # Matches NaN transparency
    show_edges=show_edges_enabled,     # Only for small models (<50k cells)
    smooth_shading=False,              # CRITICAL: Flat shading, no gradients
    interpolate_before_map=False,      # CRITICAL: No color interpolation
    preference='cell',                 # Cell-based coloring (not point-based)
    style='surface',
    ambient=0.55, diffuse=0.45, specular=0.05,  # Balanced lighting
    pbr=False,                         # No PBR for blocks
    pickable=True,                     # Essential for hover/click
)
```

**Why `smooth_shading=False`?** Block models must show hard edges between cells. Smooth shading would create misleading gradients suggesting a continuous field where there are discrete blocks.

**Why `interpolate_before_map=False`?** Without this, VTK interpolates scalar values across cell faces before applying the colormap, creating false gradient artefacts at block boundaries. Each block must display a single uniform color.

**Why `preference='cell'`?** Block model properties (grade, domain) are cell data, not point data. Using point data would require cell-to-point conversion, which averages values at shared vertices and creates blurry boundaries.

### Visibility filtering:

If only a subset of blocks should be visible (e.g., after domain filtering), `extract_cells()` is used:
```python
cell_mask = np.isin(filtered_indices, visible_block_ids)
ids = np.nonzero(cell_mask)[0]
grid = grid.extract_cells(ids)
```

**Important**: `extract_cells()` on an `ImageData` returns an `UnstructuredGrid`. This is a VTK limitation — ImageData has no mechanism for hiding individual cells. The conversion preserves cell data but loses the memory efficiency of implicit geometry.

### Edge display logic:

```python
show_edges = mesh.n_cells < 50_000
```

Cell edges on UnstructuredGrid are expensive to render. Above 50k cells, edges are disabled for performance.

---

## 5. Property Coloring and LUT Updates

**File:** `block_model_renderer.py` — `set_property_coloring()`

Colors are applied after loading. When the user switches the active property or colormap, this method runs.

### The fast path: In-place LUT update

If the grid structure hasn't changed (same mesh, same actor), only the VTK lookup table is updated — no geometry rebuild:

**Step 1 — Reset color mode:**
```python
mapper.SetColorModeToMapScalars()
```
This recovers from `DirectScalars` mode that domain masking sets (Issue #3 fix).

**Step 2 — Set active scalars:**
```python
cell_data = mapper.GetInputAsDataSet().GetCellData()
cell_data.SetActiveScalars(property_name)
mapper.SetScalarModeToUseCellData()
mapper.SetScalarVisibility(1)
```

**Step 3 — Compute color limits:**
Uses `_compute_clim()` for consistent scaling. Special handling for NSR (Net Smelter Return) fields: zero-centered divergent coloring with `max_abs = max(|vmin|, |vmax|)`.

**Step 4 — Build the 256-entry LUT:**
```python
lut = mapper.GetLookupTable()
lut.SetRange(scalar_range)
lut.SetNumberOfTableValues(256)
for i in range(256):
    rgba = cmap_matplotlib(i / 255.0)
    lut.SetTableValue(i, rgba[0], rgba[1], rgba[2], rgba[3])
mapper.Modified()
```

**Step 5 — Re-apply domain mask:**
After LUT update, domain mask transparency must be reapplied because `SetActiveScalars` may have cleared the NaN injection.

### Categorical detection (`ColorMapper._is_categorical`):

| Dtype | Rule | Example |
|-------|------|---------|
| String | Always categorical | Rock type names |
| Integer | Categorical if ≤20 unique values | Domain codes (1, 2, 3, ...) |
| Float | Categorical only if ≤10 unique AND all integer-valued | Float-encoded domains (1.0, 2.0) |

The float rule (Issue #4 fix) prevents misclassifying kriging variance or estimation pass as categorical when they happen to have few unique float values.

---

## 6. Domain Mask Transparency

**File:** `block_model_renderer.py` — `apply_domain_mask_transparency()`

This allows hiding blocks outside a geological domain without removing them from the mesh:

1. Find `domain_mask` array in cell_data (boolean: 1 = visible, 0 = hidden)
2. Back up original scalar values in `_original_{scalar_name}`
3. Inject `np.nan` into hidden cells' scalar values
4. VTK's `nan_color=(0,0,0,0)` makes NaN cells fully transparent

This is faster than `extract_cells()` (no topology rebuild) and reversible (restore from backup).

---

## 7. Picking and Interaction

### BlockModelPickAdapter (`ui/hover_inspector.py`)

The adapter sits between VTK's cell picker and the UI tooltip system. When a cell is picked:

1. **Read `Original_ID`** from the picked cell: O(1) lookup into the cached array.
2. **Sentinel check**: If `Original_ID < -2,000,000,000`, the cell is empty (sparse grid padding). Return `PickResult.miss()`.
3. **Extract properties**: Read all property arrays at the cell index, filtering out NaN, sentinels, and multi-component arrays.
4. **Return `PickResult`** with `layer_type="block_model"`, block identifier, and property dict.

**Caching**: The adapter caches `Original_ID` and property arrays. Cache is invalidated when `id(actor)` changes or cell count doesn't match.

### PickingController (`visualization/picking_controller.py`)

Controls when picking is allowed, using two VTK picker types:

| Picker | VTK Class | Cost | Use Case |
|--------|-----------|------|----------|
| Prop picker | `vtkPropPicker` | Very cheap | Actor-level identification (hover LOD-P1) |
| Cell picker | `vtkCellPicker` | Moderate | Cell-level data extraction (click LOD-P2) |

**LOD gating based on cell count:**

| Cell Count | LOD | Hover | Click |
|------------|-----|-------|-------|
| >200k | P1 | Actor-level only | Actor-level only |
| 50k–200k | P2 | Disabled | Cell-level |
| <50k | P2 | Cell-level | Cell-level |

**Performance targets:**
- Hover: <2ms (disable if >5ms)
- Click: <50ms (warn if >150ms, degrade if >250ms)

**Navigation suppression**: During camera rotation/zoom, `_is_navigating=True` disables all picking to prevent expensive cell lookups during interactive rendering.

---

## 8. The Active Layer System

**File:** `render_orchestrator.py`

Each rendered block model becomes a named layer:

```python
active_layers[layer_name] = {
    'actor': vtkActor,              # The VTK actor in the scene
    'data': pv.DataSet,             # The mesh (ImageData or UnstructuredGrid)
    'layer_type': 'blocks',
    'opacity': 1.0,
    'current_property': 'Fe',       # Currently displayed property
    'current_colormap': 'turbo',    # Currently applied colormap
    'visible': True
}
```

Layer names are derived from source filenames: `"Block Model: production_2024"`. This supports multi-model scenes.

---

## 9. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  USER DATA (CSV/Vulcan/Leapfrog/Datamine)                       │
│  block centroids + dimensions + grade/domain properties          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  BlockModel                                                      │
│  _positions (N,3) float64 + _dimensions (N,3) float64            │
│  _properties: {Fe: float64[], domain: uint8[], kv: float64[]}    │
│  _rotation_matrix: 3×3 or None                                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  is_uniform_grid()  [Detection]                                  │
│                                                                  │
│  Check 1: Constant spacing? (dx, dy, dz same for all blocks)   │
│  Check 2: Axis-aligned? (rotation_matrix ≈ identity)            │
│  Check 3: Fill ratio > 1%? (sparse but regular OK)              │
│  Check 4: Uniform dimensions? (std ≤ tolerance)                 │
│                                                                  │
│  ┌────────YES────────┐     ┌────────NO──────────┐              │
│  │                   │     │                     │              │
│  ▼                   │     ▼                     │              │
│  build_uniform_grid  │     build_unstructured    │              │
│                      │     _grid                 │              │
│  pv.ImageData()      │     pv.UnstructuredGrid() │              │
│  origin + spacing    │     N×8 vertices          │              │
│  + dims → implicit   │     VTK_HEXAHEDRON cells  │              │
│  geometry            │     explicit geometry     │              │
│                      │                           │              │
│  ~24 bytes geometry  │     ~192 bytes/block      │              │
│  40-120 FPS          │     2-8 FPS               │              │
│                      │                           │              │
│  Sparse: sentinel    │     Direct: cell order    │              │
│  filled NaN/iinfo    │     = block order         │              │
│  Original_ID via     │     Original_ID =         │              │
│  3D index mapping    │     arange(N)             │              │
│                      │                           │              │
│  └───────────────────┴─────┬─────────────────────┘              │
└──────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│  _apply_coordinate_transform_to_meshes()                         │
│                                                                  │
│  UTM (500,000 E) → local (~0,0,0)                               │
│                                                                  │
│  Double-shift guard:                                             │
│  if shift > 10km AND model_center < 10% of shift:               │
│    → skip (SGSIM output already local)                           │
│                                                                  │
│  ImageData: shift origin only                                    │
│  UnstructuredGrid: shift all points                              │
│  Tag: mesh._coordinate_shifted = True                            │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  _add_meshes_to_plotter()                                        │
│                                                                  │
│  1. Visibility filter: extract_cells(visible_block_ids)          │
│     (converts ImageData → UnstructuredGrid)                      │
│                                                                  │
│  2. plotter.add_mesh(grid,                                       │
│       scalars=property, cmap=colormap, clim=(min,max),           │
│       nan_color=(0,0,0,0),     # transparent NaN                 │
│       smooth_shading=False,    # hard block edges                │
│       interpolate_before_map=False,  # no color blending         │
│       preference='cell',       # cell data, not point data       │
│       pickable=True,                                             │
│       ambient=0.55, diffuse=0.45, specular=0.05)                 │
│                                                                  │
│  3. apply_domain_mask_transparency()                             │
│     → NaN injection for hidden-domain cells                      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  Camera + Clipping                                               │
│                                                                  │
│  Position: center + 2.0 × model diagonal                         │
│  View up: Z                                                      │
│  Near clip: min(cam_dist×0.0001, model_size×0.00001)            │
│  Far clip: max(model_size×50, cam_dist×100)                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  VTK Rendering Pipeline                                          │
│                                                                  │
│  1 vtkActor (block mesh)                                         │
│  mapper → LUT (256 entries from matplotlib colormap)             │
│  Cell-based scalar mapping (no point interpolation)              │
│  Flat shading (each cell face = uniform color)                   │
│  SSAO + EDL post-processing reapplied                            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 10. Property Update Flow (No Geometry Rebuild)

```
User selects new property in UI
        │
        ▼
set_property_coloring(property_name, colormap, discrete)
        │
        ├─ Same grid? ──YES──→ In-place LUT update:
        │                       1. mapper.SetColorModeToMapScalars()
        │                       2. cell_data.SetActiveScalars(property_name)
        │                       3. Compute clim from data
        │                       4. Build 256-entry LUT from matplotlib cmap
        │                       5. mapper.Modified()
        │                       6. Re-apply domain_mask NaN injection
        │
        └─ Grid changed? ──→ Full rebuild via _add_meshes_to_plotter()
```

---

## 11. Key Design Decisions Explained

**Why ImageData instead of always using UnstructuredGrid?**
For a 10M-block model: ImageData uses ~40–120MB (just cell_data arrays, geometry is implicit). UnstructuredGrid uses ~1.9GB (8 vertices × 10M × 24 bytes). This is 40–100× memory reduction. Performance jumps from 2–8 FPS to 40–120 FPS because VTK can exploit the regular structure for rendering, clipping, and picking.

**Why sentinel = `dtype.min` instead of -1 for integer arrays?**
Domain codes commonly use -1 (e.g., "outside estimation domain"). Using -1 as the empty-cell sentinel creates ambiguity. `np.iinfo(np.int32).min = -2,147,483,648` is safely outside any realistic domain code.

**Why NaN injection instead of extract_cells for domain masking?**
`extract_cells()` rebuilds mesh topology (O(N) copy), changes cell indices (breaks picking), and converts ImageData to UnstructuredGrid (loses memory advantage). NaN injection modifies only the scalar array, preserves topology and indices, and is reversible by restoring the backup.

**Why flat shading and no interpolation?**
Block models represent discrete geological domains and grade estimates. Smooth shading creates gradients between blocks that suggest a continuous field, which is geologically misleading. Each block must display a single uniform color matching its grade value.

**Why corner-based origin with floor for index computation?**
VTK's ImageData `origin` is the corner of cell (0,0,0), not the centroid. Using `floor((position - origin) / spacing)` with corner origin gives deterministic cell indices. The previous approach with `round` and centroid-based origin had banker's rounding edge cases (Issue #19).

**Why the double-shift guard for SGSIM models?**
Sequential Gaussian Simulation runs in local coordinate space (after the global shift has been applied to input data). The resulting block model already has local coordinates. Applying the UTM→local shift again would push it ~500km away. The guard checks if the model center is suspiciously small relative to a large UTM-scale shift.

---

## 12. File Reference

| Component | File | Class/Function |
|-----------|------|----------------|
| Data model | `models/block_model.py` | `BlockModel`, `BlockMetadata` |
| Mesh detection | `visualization/block_model_mesh_builder.py` | `is_uniform_grid()` |
| ImageData builder | `visualization/block_model_mesh_builder.py` | `build_uniform_grid()` |
| UnstructuredGrid builder | `visualization/block_model_mesh_builder.py` | `build_unstructured_grid()` |
| Auto-detect entry point | `visualization/block_model_mesh_builder.py` | `generate_block_model_mesh()` |
| Renderer coordination | `visualization/renderer/renderers/block_model_renderer.py` | `BlockModelRenderer` |
| Load pipeline | same | `load_block_model()` |
| Mesh-to-plotter | same | `_add_meshes_to_plotter()` |
| Property coloring | same | `set_property_coloring()` |
| Domain masking | same | `apply_domain_mask_transparency()` |
| Coordinate transform | same | `_apply_coordinate_transform_to_meshes()` |
| Color mapping | `visualization/color_mapper.py` | `ColorMapper` |
| Picking adapter | `ui/hover_inspector.py` | `BlockModelPickAdapter` |
| LOD picking | `visualization/picking_controller.py` | `PickingController` |
| Scene orchestrator | `visualization/renderer/render_orchestrator.py` | `Renderer` |
| Actor registry | `visualization/renderer/actor_registry.py` | `ActorRegistry` |
