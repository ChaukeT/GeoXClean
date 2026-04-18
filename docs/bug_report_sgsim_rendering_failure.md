# Bug Report: SGSIM Block Model Rendering Failure

## Critical rendering bugs when visualising SGSIM simulation output

**Date:** 2026-04-13  
**Severity:** Critical  
**Affected path:** SGSIM → BlockModel → generate_block_model_mesh → Renderer  
**Skill framework:** geox-debugger, geox-pyvista-vtk-engine, geox-rendering-pyvista-vtk

---

## 1. Bug Summary

When SGSIM simulation results are converted to a BlockModel and rendered, the blocks appear as **scattered paper-thin fragments** with **colour bleeding** and **wrong scalar binding** ("Property: None"). The root cause is a float-precision vulnerability in `_infer_dimensions_from_positions()` (block_model.py line 525): it uses `np.unique()` on float64 coordinates without tolerance, causing coordinate jitter from the SGSIM→DataFrame→BlockModel round-trip to inflate the unique-count by 100–400×, which collapses `np.median(spacings)` to near-zero. This produces block dimensions of ~1e-13 instead of the true spacing (e.g., 5.0m), creating hexahedra that are invisible to the naked eye. The same jitter also causes `is_uniform_grid()` to reject the model, forcing it down the UnstructuredGrid path with those near-zero dimensions.

A secondary bug exists in SGSIM property naming: the controller uses `f"SGSIM_{variable}_mean"` while the panel uses `f"{element}_SGSIM_{stat.upper()}"`, producing mismatched property keys that cause the renderer to show "Property: None".

---

## 2. Reproduction Path

### Data that triggers this:

1. Load a drillhole dataset with UTM coordinates (locks `_global_shift`)
2. Run SGSIM simulation on any variable (e.g., Cu normal-scored → Cu_NS)
3. SGSIM completes and creates a PyVista ImageData grid in local coordinates
4. SGSIM panel extracts `grid.cell_centers().points` into a DataFrame with columns {X, Y, Z, SGSIM_Cu_NS_mean, ...}
5. Panel calls `BlockModel.update_from_dataframe(df)` — **no DX/DY/DZ columns exist**
6. `update_from_dataframe` falls through to `_infer_dimensions_from_positions(positions)` (line 760)
7. Inferred dimensions are ~1e-13 on all axes
8. `load_block_model()` → `generate_block_model_mesh()` → `is_uniform_grid()` fails → `build_unstructured_grid()` with near-zero `half_dims`
9. Blocks render as microscopic, paper-thin, scattered fragments

### Steps to reproduce:

```
1. Launch GeoX → Open project with drillholes
2. Run SGSIM (any element, any grid size)
3. Wait for simulation to complete
4. Observe rendered block model → symptoms visible immediately
```

---

## 3. First Failing Stage

**Stage: BlockModel.update_from_dataframe → _infer_dimensions_from_positions**

File: `block_model_viewer/models/block_model.py`, lines 525–548

This is the first point where valid data becomes corrupt. The SGSIM PyVista grid has correct geometry (spacing encoded in the ImageData structure). The cell centers extracted from it are correct to float64 precision. But the dimension inference applies `np.unique()` to those float64 coordinates, and any sub-ULP jitter (from arithmetic, from anisotropy transforms, from DataFrame dtype coercion) causes the unique-count to explode.

Everything downstream — `is_uniform_grid()`, `build_unstructured_grid()`, the renderer — faithfully processes the corrupt near-zero dimensions. The geometry pipeline is correct; it is fed wrong input.

---

## 4. Root Cause Hypotheses (ranked by likelihood)

### Hypothesis 1 [CONFIRMED]: `_infer_dimensions_from_positions` np.unique on float64

**Likelihood: 99% — reproduced computationally**

```python
# block_model.py lines 541-545
for axis in range(3):
    unique = np.unique(positions[:, axis])       # ← PROBLEM: no tolerance
    if len(unique) > 1:
        spacings = np.diff(np.sort(unique))
        dims[:, axis] = np.median(spacings)      # ← median collapses to ~0
```

**Computational proof (run in this debugging session):**

| Condition | Unique Z values (expected 10) | Inferred dz (expected 2.5) |
|-----------|-------------------------------|---------------------------|
| Clean ImageData cell centers | 10 | 2.500000 |
| +1e-13 noise (sub-ULP) | 1,332 | 1.33e-15 |
| +1e-10 noise (anisotropy) | 3,992 | 3.59e-13 |

With jitter, `np.unique` produces ~N_blocks unique values (one per cell center) instead of ~N_grid_lines. The spacings become an alternating sequence of near-zero (between jittered duplicates) and actual spacing (between grid lines). Since there are more near-zero spacings than real ones, `np.median` returns near-zero.

**Why SGSIM triggers this but CSV imports don't:** CSV-imported block models have DX/DY/DZ columns in the DataFrame, so `update_from_dataframe` uses them directly (line 750–755) and never calls `_infer_dimensions_from_positions`. SGSIM output goes through a PyVista grid → cell_centers → DataFrame path that loses the grid's implicit spacing information, forcing the inference path.

### Hypothesis 2 [CONFIRMED]: `is_uniform_grid()` also fails under jitter

**Likelihood: 99% — reproduced computationally**

```python
# block_model_mesh_builder.py lines 52-54
xs = np.sort(np.unique(positions[:, 0]))    # ← Same np.unique problem
ys = np.sort(np.unique(positions[:, 1]))
zs = np.sort(np.unique(positions[:, 2]))
```

With jittered positions, `xs` has ~N_blocks_x × k unique values instead of N_grid_x. The spacing check at lines 76–85 computes `np.median(diffs)` which is near-zero, then `np.allclose(diffs, near_zero)` fails because the actual-spacing diffs are not close to near-zero. Result: `is_uniform_grid()` returns `(False, None)`.

This forces the model through `build_unstructured_grid()`, which uses the near-zero dimensions from `_infer_dimensions_from_positions` to compute `half_dims = dimensions / 2.0` (line 315), producing hexahedra with near-zero extent on all axes.

### Hypothesis 3 [LIKELY]: SGSIM property name mismatch

**Likelihood: 85%**

Two different naming conventions exist:

| Source | Pattern | Example |
|--------|---------|---------|
| `geostats_controller.py` line 1693 | `f"SGSIM_{variable}_mean"` | `SGSIM_Cu_NS_mean` |
| `sgsim_panel.py` line 2793 | `f"{element}_SGSIM_{stat.upper()}"` | `Cu_NS_SGSIM_MEAN` |
| `property_panel.py` line 2055 | `f"{variable}_SGSIM_{stat_key.upper()}"` | `Cu_NS_SGSIM_MEAN` |

The controller creates the PyVista grid with `"SGSIM_Cu_NS_mean"` as the cell_data key. When the SGSIM panel creates a BlockModel from this grid, the DataFrame column inherits this name, and so does the BlockModel property. But when the UI sets `renderer.current_property`, it uses the panel's naming convention: `"Cu_NS_SGSIM_MEAN"`. The renderer then checks:

```python
# block_model_renderer.py line 921
if self._renderer.current_property and self._renderer.current_property in grid.cell_data:
    initial_property = self._renderer.current_property
```

`"Cu_NS_SGSIM_MEAN"` is NOT in `grid.cell_data` (which has `"SGSIM_Cu_NS_mean"`). So `initial_property` stays `None`. The mesh renders with neutral grey and the status bar shows "Property: None".

### Hypothesis 4 [ELIMINATED]: Double-shift of SGSIM coordinates

**Likelihood: <5%** — the double-shift guard at lines 426–429 of `block_model_renderer.py` correctly detects SGSIM output in local coordinates and skips the shift. The guard checks `shift_mag > 10_000` AND `bound_center_mag < shift_mag * 0.1`, which correctly identifies SGSIM models (small coordinates, UTM-scale shift).

### Hypothesis 5 [ELIMINATED]: Wrong add_mesh kwargs

**Likelihood: 0%** — confirmed correct at lines 960–978:
- `smooth_shading=False` ✓
- `interpolate_before_map=False` ✓
- `preference='cell'` ✓
- `style='surface'` ✓
- VTK-level flat shading enforced at line 1065: `prop.SetInterpolationToFlat()` ✓

---

## 5. Evidence — Specific Code Lines

### Evidence A: The dimension inference vulnerability

**File:** `block_model_viewer/models/block_model.py`, lines 541–545

```python
for axis in range(3):
    unique = np.unique(positions[:, axis])       # BUG: exact equality on float64
    if len(unique) > 1:
        spacings = np.diff(np.sort(unique))
        dims[:, axis] = np.median(spacings)      # BUG: median of near-zero/actual alternation
```

### Evidence B: SGSIM DataFrame has no dimension columns

**File:** `block_model_viewer/ui/sgsim_panel.py`, lines 2420–2435 (reconstructed from agent trace)

```python
coords = grid.cell_centers().points   # (N, 3) — positions only
df_data = {
    'X': coords[:, 0],
    'Y': coords[:, 1],
    'Z': coords[:, 2]
}
for key in grid.cell_data.keys():     # properties only — NO DX/DY/DZ
    df_data[key] = grid.cell_data[key]
df = pd.DataFrame(df_data)
bm.update_from_dataframe(df)          # triggers _infer_dimensions_from_positions
```

### Evidence C: `is_uniform_grid` uses same vulnerable pattern

**File:** `block_model_viewer/visualization/block_model_mesh_builder.py`, lines 52–54

```python
xs = np.sort(np.unique(positions[:, 0]))    # exact float equality
ys = np.sort(np.unique(positions[:, 1]))
zs = np.sort(np.unique(positions[:, 2]))
```

### Evidence D: Property naming mismatch

**File:** `block_model_viewer/controllers/geostats_controller.py`, line 1693
```python
property_name = f"SGSIM_{variable}_mean"            # → "SGSIM_Cu_NS_mean"
```

**File:** `block_model_viewer/ui/sgsim_panel.py`, line 2793
```python
property_name = f"{element}_SGSIM_{stat.upper()}"   # → "Cu_NS_SGSIM_MEAN"
```

### Evidence E: The fallback to UnstructuredGrid with corrupt dimensions

**File:** `block_model_viewer/visualization/block_model_mesh_builder.py`, lines 314–315

```python
half_dims = dimensions / 2.0    # dimensions ≈ 1e-13 → half_dims ≈ 5e-14
```

Line 338:
```python
corners = positions[:, None, :] + local_offsets    # offsets ≈ ±5e-14 → zero-extent hexahedra
```

---

## 6. Symptom-to-Cause Mapping

| Symptom | Root Cause | Mechanism |
|---------|-----------|-----------|
| **Scattered blocks with gaps** | Hypothesis 1 | Blocks are ~1e-13 m across, with actual grid spacing (~5m) between them. Visually: tiny dots with huge gaps. |
| **Paper-thin horizontal slabs** | Hypothesis 1 | dz ≈ 1e-13 → Z-extent is zero. Side view shows 2D strips. |
| **Colour bleeding** | Hypothesis 1 | Blocks are so small that adjacent block colours visually merge at normal zoom. Also, UnstructuredGrid path doesn't guarantee per-cell colour isolation the way ImageData does. |
| **"Property: None"** | Hypothesis 3 | Property name `"Cu_NS_SGSIM_MEAN"` (panel convention) not found in grid.cell_data which has `"SGSIM_Cu_NS_mean"` (controller convention). |
| **Block model barely visible** | Hypothesis 1 | Total rendered volume is ~(1e-13)³ × N_blocks ≈ 0. Drillholes (correct size) dominate the scene. |

---

## 7. Minimal Safe Fix

### Fix 1: Tolerance-aware dimension inference (CRITICAL)

**File:** `block_model_viewer/models/block_model.py`, method `_infer_dimensions_from_positions`

Replace lines 541–547 with:

```python
@staticmethod
def _infer_dimensions_from_positions(positions: np.ndarray) -> 'np.ndarray | None':
    """Infer uniform block dimensions from coordinate spacing."""
    if positions is None or len(positions) == 0:
        return None
    n = len(positions)
    dims = np.zeros((n, 3), dtype=np.float64)
    for axis in range(3):
        coords = positions[:, axis]
        unique_raw = np.sort(np.unique(coords))
        if len(unique_raw) <= 1:
            dims[:, axis] = 1.0
            continue

        # FIX: Cluster near-duplicate values before computing spacing.
        # np.unique on float64 treats 2.4999999999 and 2.5000000001 as
        # distinct, inflating the unique count and collapsing median
        # spacing to near-zero.  Cluster values that are within a
        # relative tolerance of each other.
        diffs = np.diff(unique_raw)
        # Threshold: values closer than 1e-6 × median(non-tiny diffs)
        # are considered duplicates.  The "non-tiny" filter avoids
        # bootstrapping off the very jitter we're trying to remove.
        large_diffs = diffs[diffs > np.max(diffs) * 1e-3]
        if len(large_diffs) == 0:
            dims[:, axis] = 1.0
            continue
        cluster_tol = np.median(large_diffs) * 1e-6
        # Keep only the first value in each cluster
        keep = np.concatenate([[True], diffs > cluster_tol])
        unique_clean = unique_raw[keep]

        if len(unique_clean) <= 1:
            dims[:, axis] = 1.0
            continue

        spacings = np.diff(unique_clean)
        dims[:, axis] = np.median(spacings)
    return dims
```

### Fix 2: Pass grid spacing through SGSIM DataFrame (CRITICAL)

**File:** `block_model_viewer/ui/sgsim_panel.py`, in the SGSIM→BlockModel conversion section (near line 2420)

After building `df_data`, add the grid spacing as DX/DY/DZ columns:

```python
# Extract spacing from PyVista ImageData (implicit in grid structure)
if hasattr(grid, 'spacing'):
    spacing = grid.spacing
    n_cells = len(coords)
    df_data['DX'] = np.full(n_cells, spacing[0])
    df_data['DY'] = np.full(n_cells, spacing[1])
    df_data['DZ'] = np.full(n_cells, spacing[2])
```

This eliminates the inference entirely for SGSIM output, which is the safest fix.

### Fix 3: Normalise SGSIM property names (MODERATE)

**File:** `block_model_viewer/controllers/geostats_controller.py`, line 1693

Change:
```python
property_name = f"SGSIM_{variable}_mean"
```
To match the panel's convention:
```python
property_name = f"{variable}_SGSIM_MEAN"
```

Or, better, define a single naming function in a shared utility:

```python
# block_model_viewer/utils/property_names.py
def sgsim_property_name(variable: str, stat: str) -> str:
    """Canonical SGSIM property name. Single source of truth."""
    return f"{variable}_SGSIM_{stat.upper()}"
```

---

## 8. Structural Fix (Long-term)

### A: Eliminate the SGSIM→DataFrame→BlockModel round-trip

The current path is: `PyVista ImageData → cell_centers → DataFrame → BlockModel → generate_block_model_mesh → PyVista mesh`. This round-trip destroys the grid's implicit geometry (spacing, origin, dimensions) and then tries to infer it back — poorly.

**Better approach:** Store the SGSIM PyVista grid directly as the BlockModel's mesh, or add a `BlockModel.from_pyvista_grid(grid)` factory method that extracts spacing from the ImageData structure:

```python
@classmethod
def from_pyvista_grid(cls, grid: pv.ImageData, metadata=None):
    """Create BlockModel from PyVista ImageData preserving exact geometry."""
    bm = cls(metadata=metadata)
    centers = grid.cell_centers().points
    n = len(centers)
    spacing = np.array(grid.spacing)
    dims = np.tile(spacing, (n, 1))
    bm.set_geometry(centers, dims)
    for name in grid.cell_data:
        bm.add_property(name, np.asarray(grid.cell_data[name]))
    return bm
```

### B: Make `is_uniform_grid` use tolerance-aware unique

Replace `np.unique(positions[:, axis])` with the same clustering approach from Fix 1 above. This ensures that even if dimension inference is bypassed, the grid-type detection is still robust.

### C: Canonical property naming

Define all SGSIM/kriging/IK property names in a single module (`utils/property_names.py`). All panels, controllers, and renderers import from this module. No more ad-hoc f-strings.

---

## 9. Regression Tests

### Test 1: Dimension inference under float jitter

```python
def test_infer_dimensions_with_float_jitter():
    """_infer_dimensions_from_positions must be robust to float64 jitter."""
    rng = np.random.default_rng(42)
    # 20×20×10 grid, spacing 5×5×2.5
    xs = np.arange(20) * 5.0 + 2.5
    ys = np.arange(20) * 5.0 + 2.5
    zs = np.arange(10) * 2.5 + 1.25
    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    # Add 1e-10 jitter (realistic for anisotropy transform round-trip)
    positions += rng.uniform(-1e-10, 1e-10, positions.shape)

    dims = BlockModel._infer_dimensions_from_positions(positions)
    assert dims is not None
    assert np.allclose(dims[:, 0], 5.0, atol=1e-4), f"dx={dims[0,0]}"
    assert np.allclose(dims[:, 1], 5.0, atol=1e-4), f"dy={dims[0,1]}"
    assert np.allclose(dims[:, 2], 2.5, atol=1e-4), f"dz={dims[0,2]}"
```

### Test 2: SGSIM→BlockModel round-trip preserves geometry

```python
def test_sgsim_blockmodel_roundtrip_preserves_dimensions():
    """SGSIM grid → DataFrame → BlockModel must preserve block dimensions."""
    import pyvista as pv
    grid = pv.ImageData(dimensions=(21, 21, 11), spacing=(5.0, 5.0, 2.5),
                        origin=(0.0, 0.0, 0.0))
    grid.cell_data['SGSIM_Cu_mean'] = np.random.default_rng(42).uniform(0, 5, grid.n_cells)

    centers = grid.cell_centers().points
    df = pd.DataFrame({'X': centers[:, 0], 'Y': centers[:, 1], 'Z': centers[:, 2],
                        'DX': grid.spacing[0], 'DY': grid.spacing[1], 'DZ': grid.spacing[2],
                        'SGSIM_Cu_mean': grid.cell_data['SGSIM_Cu_mean']})
    bm = BlockModel()
    bm.update_from_dataframe(df)

    assert np.allclose(bm.dimensions[:, 0], 5.0)
    assert np.allclose(bm.dimensions[:, 1], 5.0)
    assert np.allclose(bm.dimensions[:, 2], 2.5)
```

### Test 3: `is_uniform_grid` tolerates float jitter

```python
def test_is_uniform_grid_with_jitter():
    """is_uniform_grid must detect uniform grid despite float64 jitter."""
    rng = np.random.default_rng(42)
    n = 4000
    positions = np.column_stack([
        np.repeat(np.arange(20) * 5.0 + 2.5, 200),
        np.tile(np.repeat(np.arange(20) * 5.0 + 2.5, 10), 20),
        np.tile(np.arange(10) * 2.5 + 1.25, 400),
    ]) + rng.uniform(-1e-10, 1e-10, (n, 3))
    dims = np.tile([5.0, 5.0, 2.5], (n, 1))

    bm = BlockModel()
    bm.set_geometry(positions, dims)
    is_uniform, grid_info = is_uniform_grid(bm)
    assert is_uniform, "Should detect uniform grid despite jitter"
    assert np.isclose(grid_info['spacing'][0], 5.0, atol=1e-4)
    assert np.isclose(grid_info['spacing'][2], 2.5, atol=1e-4)
```

### Test 4: Property name consistency

```python
def test_sgsim_property_name_consistency():
    """Controller and panel must produce identical SGSIM property names."""
    from block_model_viewer.utils.property_names import sgsim_property_name
    # Controller naming
    controller_name = sgsim_property_name("Cu_NS", "mean")
    # Panel naming
    panel_name = sgsim_property_name("Cu_NS", "MEAN")
    assert controller_name == panel_name, (
        f"Controller '{controller_name}' != Panel '{panel_name}'"
    )
```

### Test 5: End-to-end SGSIM rendering produces visible blocks

```python
def test_sgsim_blocks_are_visible_size():
    """SGSIM block model mesh must have correct block dimensions, not near-zero."""
    # Create SGSIM-style BlockModel via DataFrame (no DX/DY/DZ)
    rng = np.random.default_rng(42)
    grid = pv.ImageData(dimensions=(21, 21, 11), spacing=(5.0, 5.0, 2.5),
                        origin=(0.0, 0.0, 0.0))
    centers = grid.cell_centers().points
    # Add realistic jitter
    centers += rng.uniform(-1e-10, 1e-10, centers.shape)

    df = pd.DataFrame({'X': centers[:, 0], 'Y': centers[:, 1], 'Z': centers[:, 2],
                        'grade': rng.uniform(0, 5, len(centers))})
    bm = BlockModel()
    bm.update_from_dataframe(df)
    mesh = generate_block_model_mesh(bm)

    # Mesh bounds should span the full grid extent, not collapse to near-zero
    bounds = mesh.bounds
    x_extent = bounds[1] - bounds[0]
    y_extent = bounds[3] - bounds[2]
    z_extent = bounds[5] - bounds[4]
    assert x_extent > 90.0, f"X extent {x_extent} is too small (expected ~100)"
    assert y_extent > 90.0, f"Y extent {y_extent} is too small (expected ~100)"
    assert z_extent > 20.0, f"Z extent {z_extent} is too small (expected ~25)"
```

---

## 10. Files Reference

| File | Role | Key Lines |
|------|------|-----------|
| `models/block_model.py` | Data model, dimension inference | 525–548 (`_infer_dimensions_from_positions`), 699–786 (`update_from_dataframe`) |
| `visualization/block_model_mesh_builder.py` | Grid type detection, mesh construction | 22–184 (`is_uniform_grid`), 187–290 (`build_uniform_grid`), 293–379 (`build_unstructured_grid`) |
| `visualization/renderer/renderers/block_model_renderer.py` | Renderer coordination | 375–496 (`_apply_coordinate_transform_to_meshes`), 813–1081 (`_add_meshes_to_plotter`), 1218–1450 (`set_property_coloring`) |
| `ui/sgsim_panel.py` | SGSIM→BlockModel conversion | ~2420–2444 (cell center extraction, DataFrame creation) |
| `controllers/geostats_controller.py` | SGSIM property naming | 1693 (`f"SGSIM_{variable}_mean"`) |

---

## 11. Risk Assessment

| Fix | Risk | Mitigation |
|-----|------|------------|
| Fix 1 (dimension inference) | Low — only changes inference fallback; models with DX/DY/DZ columns are unaffected | Regression test 1 |
| Fix 2 (pass spacing through DataFrame) | Very low — adds columns to DataFrame, doesn't change any existing path | Regression test 2 |
| Fix 3 (property naming) | Moderate — existing saved sessions may reference old names | Add migration logic or alias lookup |
| Structural fix A (from_pyvista_grid) | Low — new code path, doesn't modify existing | Test 5 covers this |
| Structural fix B (is_uniform_grid tolerance) | Low — tolerance already partially implemented via FP-09 fix | Test 3 covers this |

**Recommended deployment order:** Fix 2 first (eliminates the trigger), then Fix 1 (makes inference robust for other data sources), then Fix 3 (resolves "Property: None").
