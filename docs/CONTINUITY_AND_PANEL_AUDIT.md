# Lithological Continuity & Panel Rework Audit

## Executive Summary

Two critical gaps in the geological modelling subsystem have been addressed:

1. **BUG GEO-07: No Lithological Continuity Enforcement (CRITICAL)**
2. **Missing Panel Rework for New Engine Workflow**

---

## Part 1: Lithological Continuity (GEO-07)

### The Problem

Implicit geological models (LoopStructural, Leapfrog, GemPy) use scalar fields
to define lithological boundaries. Where drillhole data is sparse, the scalar
field can create **closed isosurfaces** — small disconnected blobs of one
lithology floating inside another.

**Example:** A sandstone unit that should be a continuous sheet appears as:
- Main body: 95% of volume (correct)
- 12-cell island at depth, floating inside shale, 50m from any drillhole
- 3-cell island near surface edge

These are **geologically impossible** for sedimentary deposits. They indicate
interpolation failure in data-sparse regions. Every commercial package
(Leapfrog, Vulcan, GOCAD, Surpac) includes island detection.

### The Fix: LithologicalContinuityEnforcer

**File:** `lithological_continuity.py` (693 lines)

**Algorithm:**
1. **3D Connected Component Labeling** — BFS flood fill on regular voxel grid
   with 6-connectivity (face-adjacent only, no diagonals)
2. **Primary Body Identification** — largest component per formation
3. **Island Classification** — components with <5% of formation's cells OR
   <10 cells absolute (both configurable)
4. **Reassignment** — island cells reassigned to correct neighbour:
   - Priority 1: Neighbour majority (>60% of adjacent cells)
   - Priority 2: Enclosing formation (completely surrounded)
   - Priority 3: Nearest drillhole contact (tiebreaker)
5. **Validation** — confirm no significant islands remain after reassignment
6. **Audit Report** — per-formation component counts, island volumes,
   reassignment log

**Additional Checks:**
- **Lateral Continuity** — verifies each formation appears in >80% of
  vertical columns (expected for sedimentary sheets)
- **Vertical Stacking** — detects stratigraphic inversions (younger
  formation below older = strong island indicator)

**Key Classes:**
| Class | Purpose |
|-------|---------|
| `ConnectedComponentLabeler` | O(N) BFS flood fill on 3D grid |
| `LithologicalContinuityEnforcer` | Full analysis + reassignment pipeline |
| `ConnectedComponent` | Data class for each body |
| `IslandReassignment` | Audit record for each reassignment |
| `ContinuityReport` | Complete analysis results |

### Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `min_component_fraction` | 0.05 (5%) | Components below this fraction flagged |
| `min_component_cells` | 10 | Absolute minimum cell count |
| `max_islands_per_formation` | 20 | Warning threshold for model instability |
| `enable_reassignment` | True | Whether to actually reassign cells |

---

## Part 2: Panel Rework

### The Problem

The old panel had 7 tabs designed for the OLD engine workflow:
1. Input Validation
2. Stratigraphy
3. Domain
4. Build
5. Audit
6. Advisory
7. Export

This did NOT match the new GeologicalModelEngine workflow. Critical new
features (contact extraction, orientation computation, structure detection,
continuity QC) had NO UI representation.

### The Fix: Complete 8-Tab Workflow Panel

**File:** `loopstructural_panel.py` (1,816 lines)

**New Tab Structure:**

| Tab | Name | NEW? | Purpose |
|-----|------|------|---------|
| 1 | Input Data | Updated | Load drillhole data, column validation |
| 2 | Contacts | **NEW** | Display extracted lithological contacts |
| 3 | Orientations | **NEW** | Computed dip/dip-direction with confidence |
| 4 | Structures | **NEW** | Detected faults/folds with ACCEPT/REJECT |
| 5 | Stratigraphy | Updated | Sequence, domain, parameters, continuity settings |
| 6 | Build | Updated | Full pipeline execution with 5 phase indicators |
| 7 | Continuity QC | **NEW** | Island detection results, per-formation stats |
| 8 | Audit & Export | Updated | JORC compliance with continuity status |

### Tab 2: Contacts (NEW)

Shows extracted contacts in a table:
- Hole ID, X, Y, Z, Formation Above, Formation Below, Scalar Value
- Metric cards: total contacts, unique drillholes, unique boundaries
- Extract button triggers DrillholeContactExtractor

### Tab 3: Orientations (NEW)

Shows computed orientations:
- Boundary name, X, Y, Z, Dip, Dip Direction, Confidence, Method
- Method selector: Plane Fit (PCA) / Three-Point / Default Horizontal
- Warning banner when synthetic orientations are used
- Metric cards: count, average dip, average confidence

### Tab 4: Structures (NEW)

Interactive structure management:
- Each detected structure shown with Accept/Reject buttons
- Name, Type, Confidence, Evidence, Status columns
- Bulk actions: "Accept All >0.7 confidence", "Reject All <0.3 confidence"
- Metric cards: total detected, accepted, rejected

### Tab 7: Continuity QC (NEW)

Full island analysis results:
- Summary metrics: status, islands, cells, volume fraction
- Per-formation table: components, primary %, islands, island cells, continuous flag
- Collapsible reassignment log: original → new, cells, method, reason

### Build Tab Integration

The Build tab now runs the full pipeline through ModelBuildWorker:
1. Extract contacts from drillholes
2. Compute orientations from geometry
3. Build LoopStructural model
4. Extract surfaces and unified mesh
5. Enforce lithological continuity

Five phase indicators with real-time progress.

---

## Part 3: Updated Bug Count

| Subsystem | Critical | High | Medium | Total |
|-----------|----------|------|--------|-------|
| Variogram | 1 | 2 | 3 | 6 |
| Kriging | 2 | 3 | 2 | 7 |
| IK/CoK/Bayesian | 4 | 3 | 3 | 10 |
| Simulation | 3 | 4 | 3 | 10 |
| Resource/Reporting | 2 | 3 | 3 | 8 |
| Geological Modelling | 2 | 2 | 2 | 6 |
| **Continuity (NEW)** | **1** | **0** | **0** | **1** |
| **TOTAL** | **15** | **17** | **16** | **48** |

---

## Part 4: Files Delivered

| File | Lines | Purpose |
|------|-------|---------|
| `lithological_continuity.py` | 693 | 3D connected component analysis + island elimination |
| `loopstructural_panel.py` | 1,816 | Complete 8-tab panel rework |
| `CONTINUITY_AND_PANEL_AUDIT.md` | this file | Audit documentation |

### Integration

```python
# In geological_model_engine.py, after extract_unified_mesh():
from .lithological_continuity import enforce_lithological_continuity

unified_mesh = engine.extract_unified_mesh(stratigraphy)
report = enforce_lithological_continuity(
    unified_mesh=unified_mesh,
    min_island_fraction=0.05,
    min_island_cells=10,
    enable_reassignment=True,
)
# formation_ids in unified_mesh are modified IN PLACE
```

### Panel drops in as replacement for existing loopstructural_panel.py
- Same class name: `LoopStructuralModelPanel`
- Same signals: `model_built`, `surfaces_extracted`, `compliance_validated`, `geology_package_ready`
- Same public API: `set_contacts_data()`, `set_stratigraphy()`, `bind_controller()`
- `task_name = "loopstructural_model"` (unchanged)

---

## Part 5: What Changed Since Last Delivery

| Previously Delivered | Now Added |
|---------------------|-----------|
| geological_model_engine.py (contacts, orientations, structures) | lithological_continuity.py (island detection) |
| model_runner.py (pipeline coordinator) | loopstructural_panel.py (full UI rework) |
| GEOLOGICAL_MODELLING_AUDIT.md | CONTINUITY_AND_PANEL_AUDIT.md |

The previous delivery fixed the ENGINE but not the UI. This delivery completes
both the continuity enforcement (engine gap) and the panel (UI gap).
