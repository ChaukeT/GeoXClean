# GeoX Drillhole Rendering Architecture

## Complete Pipeline: From Raw Data to Screen Pixels

This document traces every step of how a drillhole goes from CSV/Excel data to a rendered 3D tube in the PyVista viewport.

---

## 1. The Data Layer

**Files:** `drillholes/datamodel.py`, `drillholes/database.py`

The `DrillholeDatabase` stores everything in Pandas DataFrames (not Python objects) for vectorized performance. Four core tables:

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `collars` | hole_id, x, y, z, azimuth, dip, length | Where the drill entered the ground |
| `surveys` | hole_id, depth_from, depth_to, azimuth, dip | Direction changes downhole |
| `lithology` | hole_id, depth_from, depth_to, lith_code | Rock type per interval |
| `assays` | hole_id, depth_from, depth_to, + element columns | Grade values (Fe, SiO2, etc.) |

There's also a `structural_measurements` table for oriented discontinuities, stored as unit normal vectors with full provenance.

**Helper dataclasses** (`Collar`, `SurveyInterval`, `AssayInterval`, `LithologyInterval`) exist only for single-row UI interaction — the DataFrames are the canonical store.

---

## 2. Desurveying: Turning Survey Data into 3D Paths

**File:** `utils/desurvey.py`

This is the mathematical core. The **Minimum Curvature** algorithm converts survey measurements (depth, azimuth, dip) into 3D world coordinates (X, Y, Z). This is the industry-standard method (Saari, 1977).

### How it works:

1. **Input validation**: Auto-detects column names from aliases (DEPTH/MD/MEASURED_DEPTH, AZI/AZIMUTH, DIP/INC). Removes NaN rows, duplicate depths, and non-monotonic entries.

2. **Dip convention normalization**: GeoX uses negative-down (−90° = vertical down). If the data appears to be positive-down (>70% positive dips, median >30°), it auto-negates. A post-negation guard (DIA-DS03) reverts if the negation was a false positive.

3. **Force depth-zero**: If the first survey isn't at depth 0, it inserts one using the first station's angles.

4. **Dogleg angle calculation**:
   ```
   cos(α) = cos(I₂ − I₁) − sin(I₁)·sin(I₂)·(1 − cos(A₂ − A₁))
   ```

5. **Ratio factor** (approaches 1.0 for straight holes):
   ```
   F = tan(α/2) / (α/2)
   ```

6. **Displacement per interval** (Minimum Curvature formula):
   ```
   dx = (dr/2) · (cos(I₁)·sin(A₁) + cos(I₂)·sin(A₂)) · F
   dy = (dr/2) · (cos(I₁)·cos(A₁) + cos(I₂)·cos(A₂)) · F
   dz = (dr/2) · (sin(I₁) + sin(I₂)) · F
   ```
   Where dx = Easting, dy = Northing, dz = Elevation (positive up).

7. **Cumulative sum** from collar coordinates gives the 3D trajectory.

**Output**: Arrays of (depths, xs, ys, zs) — the desurvey stations along the hole.

---

## 3. Building Polylines: Segmenting the Hole for Coloring

**File:** `drillholes/drillhole_layer.py` — `build_drillhole_polylines()`

This function bridges the data model and the renderer. It takes the database and produces PyVista `PolyData` polylines with per-segment scalar values.

### Step-by-step:

1. **Extract DataFrames** into dictionaries keyed by hole_id: collars, surveys, lithology intervals, assay intervals.

2. **Composite override**: If a composite DataFrame is provided (e.g., composited grades), it replaces raw assays. Case-insensitive hole-ID matching handles HOLEID/hole_id/BHID variants. Diagnostic logging detects mismatches.

3. **Per-hole processing**:
   - Call `minimum_curvature_path_from_surveys()` to get 3D coordinates at survey stations.
   - Compute **break depths** from assay and lithology boundaries (NOT survey stations — including survey stations creates tiny uncolored segments).
   - **Interpolate** 3D coordinates at break depths that fall between survey stations (linear interpolation along the minimum curvature path).
   - Build a `pv.PolyData` polyline: points are the 3D coordinates, lines connect consecutive points (VTK format: `[2, i0, i1, 2, i1, i2, ...]`).

4. **Per-segment scalar assignment**:
   - For each segment (between consecutive break depths), look up the **lithology code** and **assay value** at the midpoint depth.
   - Lithology: half-open interval `[from, to)` except for the last interval which uses closed `[from, to]` (DH-07 fix).
   - Assay: exact field match with case-insensitive fallback. Auto-selects a default field if none specified.

5. **Assay range calculation**:
   - Uses data-driven lower bound (not hardcoded 0.0) for geophysical logs that can be negative (DH-03 fix).
   - Computes 98th percentile for auto color-limit (prevents outlier compression).

6. **Color palette**: 20 professional geological colors (red for sandstone/ore, green for shale, etc.), with HSV cycling fallback for additional lithology codes.

### Output dictionary:
```python
{
    "hole_polys": {hole_id: pv.PolyData},       # 3D polylines
    "hole_segment_lith": {hole_id: [codes]},     # Lithology per segment
    "hole_segment_assay": {hole_id: [values]},   # Assay per segment
    "hole_segment_from_depth": {hole_id: [depths]},
    "hole_segment_to_depth": {hole_id: [depths]},
    "lith_colors": {code: hex_color},
    "lith_to_index": {code: int},
    "assay_field": str,
    "assay_min": float, "assay_max": float, "assay_p98": float,
    "hole_ids": [str],
    "collar_coords": {hole_id: (x, y, z)},
    "_registry": DataRegistry,  # For GPU picking stability
}
```

---

## 4. The Standard Renderer: Spline Tubes and Per-Hole Actors

**File:** `visualization/renderer/renderers/drillhole_renderer.py` — `DrillholeRenderer`

This is the coordination layer. It doesn't call `plotter.render()` directly (that's the orchestrator's job, except for immediate-feedback scenarios). It has two rendering paths:

### Path A: Standard Renderer (default)

Used for datasets up to ~5000 holes. Creates **individual VTK actors per hole** for instant visibility toggling.

#### Phase 1: Coordinate Shift
```
UTM coordinates (e.g., 500,000 E) → local coordinates (~0,0,0)
```
This is **critical**. GPU float32 precision breaks at large UTM offsets. The shift is locked by whichever dataset loads first ("first dataset authority"). Both drillholes and geological surfaces must use the same shift, otherwise they appear in different locations and camera clipping hides everything.

```python
shifted_origin = self._renderer._to_local_precision(all_points[:1])  # Lock shift
for hid, poly in hole_polys.items():
    poly.points = self._renderer._to_local_precision(poly.points.copy())
# Collar coordinates also shifted
```

#### Phase 2: Spline Tube Construction (`_build_spline_tube`)

This is where polylines become smooth 3D tubes:

1. **Spline interpolation**: `pv.Spline(points, n_points=max(len*3, 6))` — 3× densification through desurvey points.
2. **Tube extrusion**: `spline.tube(radius=radius, capping=False, n_sides=n_sides)` — creates the cylindrical mesh.
3. **Ring-index scalar mapping**: The tube mesh has `n_sides` points per ring, `n_spline_pts` rings total. Point `i` belongs to ring `i // n_sides`. Each ring's fractional position `t` maps to a drillhole interval via `np.searchsorted(boundaries, t)`. This gives per-interval coloring with sub-ring transitions at boundaries.

```python
ring_idx = np.arange(n_tube_pts) // n_sides
t_pts = ring_idx / max(1, n_spline_pts - 1)
seg_idx = np.searchsorted(boundaries[1:], t_pts, side='right')
tube.point_data[scalar_name] = vals[seg_idx]
```

Adaptive quality based on dataset size: 16 sides (default) → 14 (>200 holes) → 12 (>500 holes).

#### Phase 3: Actor Creation

Each tube becomes an individual PyVista actor with PBR materials:
```python
actor = plotter.add_mesh(tube,
    color='lightgray',       # Default before color assignment
    smooth_shading=True,
    pbr=True,
    metallic=0.1,
    roughness=0.5,
    nan_color="gray",
    pickable=True,
)
```

Visibility is set per-hole via `actor.VisibilityOn()/Off()`.

Stored in `_drillhole_hole_actors: Dict[str, vtkActor]` keyed by hole_id.

#### Phase 4: Collar Markers

All visible collars are batched into a **single glyph actor**:
```python
collar_cloud = pv.PolyData(collar_points)
collar_glyphs = collar_cloud.glyph(geom=pv.Sphere(radius=radius*0.6))
collar_actor = plotter.add_mesh(collar_glyphs, color="white", pbr=True, ...)
```

One VTK actor for all collars = minimal rendering overhead.

#### Phase 5: Camera Positioning

Computes combined bounds of all tubes, sets camera at 1.5× the scene diagonal with view-up = Z. Sets explicit clipping range (`near = size * 0.001`, `far = size * 100.0`) instead of using `ResetCameraClippingRange()` which can include overlay actors at wrong coordinates.

---

### Path B: GPU Renderer (experimental, >5000 holes)

**File:** `visualization/drillhole_gpu_renderer.py` — `DrillholeGPURenderer`

Activated only when `_use_gpu_drillholes=True` AND dataset exceeds 5000 holes or 100k intervals.

#### Key differences from standard renderer:

1. **Batched geometry**: All intervals merged into a **single mesh** via `pv.merge(meshes)` — O(n) merge, not O(n²) sequential. Falls back to chunked merge (groups of 500) if memory fails.

2. **Per-interval cylinders**: Each interval is a standalone `pv.Cylinder(center, direction, radius, height, resolution)` rather than a spline tube. Simpler geometry but no smooth curvature between intervals.

3. **Cell data arrays** on the merged mesh:
   - `color_id` (int32): Unique per interval for GPU picking
   - `lith_idx` (int32): Lithology index for discrete coloring
   - `assay` (float32): Assay value for continuous coloring
   - `selection_state` (int8): NONE/HOVERED/SELECTED/HIDDEN

4. **LOD switching**: RenderQuality degrades based on interval count:
   - >10k intervals → LOW (4 sides)
   - >5k intervals → MEDIUM (8 sides)
   - Otherwise → HIGH (16 sides)

5. **Chunked construction with Qt event processing**: Every 500 intervals, calls `QApplication.processEvents()` to prevent UI freeze.

---

## 5. Color Updates: How Colors Are Applied and Changed

**File:** `render_orchestrator.py` — `_update_drillhole_colors()`

Colors are NOT applied during initial loading (tubes start as `lightgray`). They're applied when the user selects a property.

### Fast path: LUT-only update
If only the colormap changes (same property), `_update_drillhole_lut_only()` updates the VTK lookup table without rebuilding geometry.

### Full path: Property switch
If the property changes (e.g., lithology → Fe grade), this triggers:
1. Rebuild polylines via `build_drillhole_polylines()` with new assay field
2. Apply coordinate shift to new polylines
3. Remove old actors
4. Create new spline tubes with new scalar data
5. Create new actors
6. Update cache and `active_layers`
7. Reconnect interaction handlers

### Color mapping pipeline:
- **Lithology**: `lith_code → lith_to_index → colormap_index → RGB` (discrete, tab10)
- **Assay**: `value → normalize(min, p98) → [0,1] → colormap → RGB` (continuous, turbo)
- **Custom colors**: Override individual lithology codes via legend manager

---

## 6. Interaction: Picking, Hover, and Selection

**File:** `render_orchestrator.py` — `_setup_standard_drillhole_interaction()`

### Standard renderer:
- **Hover**: Uses VTK **PROP picker** (actor-level, not cell-level). Identifies which hole the cursor is over by mapping `actor → hole_id`. Visual feedback: adjusts ambient/diffuse properties.
- **Click**: Same PROP picker. Highlights selected hole with yellow edges. Emits `drillhole_selected` callback with hole_id and interval data.

### GPU renderer:
- Uses `color_id` buffer for **GPU picking**: each interval has a unique integer. Screen coordinates map to the closest interval via point cloud search.
- Hover is throttled (<2ms target, disabled if >5ms).
- Click emits via `DrillholeEventBus` (PyQt6 signals).

### Picking LOD levels:
| Level | When | Capability |
|-------|------|------------|
| P0 | During navigation | No picking |
| P1 | >200k cells | Actor-level hover only |
| P2 | <50k cells | Cell-level click picking |
| P3 | Debug mode | Full metadata inspection |

---

## 7. State Management

**File:** `visualization/drillhole_state.py` — `DrillholeStateManager`

Centralized state with undo/redo history. Tracks:

- **Visibility**: per-hole show/hide, bulk set_all_visible, toggle
- **Selection**: set of selected interval color_ids, hover state
- **Filters**: depth range, lithology codes, assay range
- **Camera**: position, focal point, view up
- **Scene**: color property, colormap, tube radius, render quality, clip plane

Uses publish/subscribe pattern (`on()`, `off()`, `_notify()`) to decouple UI panels from renderer state.

`SceneState` dataclass captures the complete snapshot:
```python
@dataclass
class SceneState:
    visible_holes: Set[str]
    selected_interval_ids: Set[int]
    color_property: str
    colormap: str
    tube_radius: float
    render_quality: str
    camera: CameraState
    clip_plane: Optional[ClipPlaneState]
    depth_filter: Optional[Tuple[float, float]]
    lith_filter: Optional[Set[str]]
    assay_filter: Optional[Tuple[float, float]]
    hole_count: int
    interval_count: int
    data_bounds: Tuple[float, ...]
```

---

## 8. Actor Registry and Scene Management

**File:** `visualization/renderer/actor_registry.py` — `ActorRegistry`

Categories: `block_model`, `drillhole`, `surface`, `overlay`, `legend`, `debug`.

Methods:
- `add(actor_id, actor, category, metadata)` — register actor
- `remove(actor_id)` — unregister (returns actor for plotter removal)
- `get_all(category)` — retrieve by category
- `clear_category(category)` — bulk cleanup
- `get_bounds(category)` — spatial bounds per category

Drillhole actors are stored in **two places** (historical pattern, not ideal):
1. `ActorRegistry` under category `"drillhole"`
2. `Renderer._drillhole_hole_actors` dict (direct access for fast visibility toggling)

---

## 9. Visual Density Controller

Registered after actor creation. Provides automatic LOD switching based on camera distance. Controls:
- Drillhole tube actors (show/hide based on distance)
- Collar actors (show/hide)
- Label actors (show/hide)

---

## 10. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  USER DATA (CSV/Excel)                                          │
│  collars.csv + surveys.csv + lithology.csv + assays.csv        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  DrillholeDatabase (DataFrames)                                  │
│  collars_df, surveys_df, lithology_df, assays_df                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  build_drillhole_polylines()                                     │
│                                                                  │
│  For each hole:                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. minimum_curvature_path_from_surveys()                   │ │
│  │    survey (depth,azi,dip) → 3D stations (x,y,z)           │ │
│  │                                                            │ │
│  │ 2. Compute break depths from assay/lith boundaries         │ │
│  │    (NOT survey stations — avoids tiny gray segments)        │ │
│  │                                                            │ │
│  │ 3. Interpolate 3D coords at break depths                   │ │
│  │                                                            │ │
│  │ 4. Build pv.PolyData polyline:                             │ │
│  │    points = [3D coords], lines = [2, i0, i1, 2, i1, i2]   │ │
│  │                                                            │ │
│  │ 5. Per-segment scalars: lith_code + assay_value at midpt   │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Output: {hole_polys, segment_lith, segment_assay, colors, ...} │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  DrillholeRenderer.add_drillhole_layer()                         │
│                                                                  │
│  1. COORDINATE SHIFT                                             │
│     UTM (500,000 E) → local (~0,0,0)                            │
│     Same shift as geological surfaces (first-dataset authority)  │
│                                                                  │
│  2. SPLINE TUBE CONSTRUCTION (_build_spline_tube)                │
│     polyline → pv.Spline(3× densification) → .tube(n_sides)     │
│     Ring-index mapping: ring_idx = pt_idx // n_sides             │
│     t = ring_idx / (n_spline_pts - 1)                           │
│     seg_idx = searchsorted(boundaries, t)                        │
│     tube.point_data["assay"] = vals[seg_idx]                     │
│                                                                  │
│  3. PER-HOLE ACTORS                                              │
│     plotter.add_mesh(tube, pbr=True, smooth_shading=True)        │
│     actor.VisibilityOn/Off() per visible_holes set               │
│                                                                  │
│  4. COLLAR GLYPHS                                                │
│     pv.PolyData(collars).glyph(pv.Sphere) → single actor        │
│                                                                  │
│  5. CAMERA SETUP                                                 │
│     Position = center + 1.5× diagonal                            │
│     Clipping = [size*0.001, size*100.0]                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  VTK/PyVista Rendering Engine                                    │
│                                                                  │
│  vtkRenderer → vtkRenderWindow → GPU                             │
│  Each hole = 1 vtkActor with PolyData mapper                    │
│  Collars = 1 vtkActor (glyphed spheres)                         │
│  PBR shading: metallic=0.1, roughness=0.5                       │
│  SSAO + EDL reapplied after add_mesh                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## 11. Key Design Decisions Explained

**Why individual actors instead of one merged mesh?**
Individual actors allow `actor.SetVisibility(True/False)` per hole — instant toggling without geometry rebuild. The merged mesh path (GPU renderer) is faster for initial render but requires a full rebuild when visibility changes.

**Why spline tubes instead of raw polylines?**
Polylines are 1-pixel lines. Spline tubes give volume, smooth curvature, and PBR shading — matching Leapfrog/Vulcan quality. The 3× densification smooths the angular segments from minimum curvature desurveying.

**Why ring-index scalar mapping?**
The tube mesh has a regular ring structure (n_sides points per cross-section). Rather than per-vertex color assignment (which requires knowing tube topology), the fractional position along the spline maps directly to a drillhole interval via `searchsorted`. This gives per-interval coloring with minimal code.

**Why coordinate shift?**
GPU float32 has ~7 significant digits. UTM Easting 500,123.456 loses sub-meter precision. Shifting to local coordinates (~0,0,0) preserves precision. The shift is locked by the first dataset loaded to ensure all data aligns.

**Why break depths exclude survey stations?**
Survey stations mark where direction was measured, not where lithology or grade changes. Including them as segment boundaries creates many tiny segments between surveys that have no assay data and render as gray (the NaN color). Break depths come only from assay/lithology boundaries, the collar (depth 0), and total depth.

**Why collar glyphs instead of individual spheres?**
A single glyph actor for all collars uses one VTK draw call. Individual sphere actors (one per collar) would each be a separate draw call — expensive for >100 holes.
