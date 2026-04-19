# GeoX Desktop — Product Requirements Document (Revised)

**Version:** 2.0  
**Date:** 2 April 2026  
**Status:** Active Development (Beta)  
**Product:** GeoX Desktop v1.0.0  
**License:** MIT (Open Source)

---

## 1. Executive Summary

GeoX Desktop is an integrated 3D geostatistical workbench and mine-planning platform that delivers the full mineral resource estimation workflow—drillhole import, compositing, variography, spatial estimation, resource classification, geological modelling, pit optimisation, production scheduling, and financial analysis—in a single desktop application.

It targets a market segment where commercial packages (Vulcan, Leapfrog Geo, Datamine) cost \$20,000–\$50,000+ per seat per year, while open-source alternatives are fragmented CLI tools with no 3D visualisation, no integrated audit trail, and no mine-planning capability. GeoX eliminates the need to switch between three or more applications by unifying data management, estimation, visualisation, and planning under one roof with full JORC/SAMREC audit compliance.

---

## 2. Target Users

| Persona | Background | Primary Goals |
|---------|-----------|---------------|
| **Resource Estimator** | MSc/PhD Geoscience, 2–10 yrs experience | Defensible, reproducible estimates; JORC-compliant audit trails |
| **Geostatistician** | Advanced modelling focus | Fine-grained variogram control, cross-validation, LVA support |
| **Data Steward** | Database/geology specialist | Chain of custody, QC validation, composite parameter tracking |
| **Mine Planner** | Mining engineer | Cutoff analysis, pit optimisation, scheduling, NPV/IRR |
| **Uncertainty Analyst** | PhD-level researcher | Multi-realisation simulations, P10/P50/P90 distributions |
| **Geotechnical Engineer** | Rock mechanics specialist | Slope stability, stope design, rockburst risk assessment |
| **ESG/Compliance Officer** | Sustainability specialist | GHG tracking, water balance, GRI/TCFD/SASB reporting |

---

## 3. Platform & Technology

| Attribute | Specification |
|-----------|--------------|
| **Language** | Python 3.10+ (tested on 3.10–3.13) |
| **GUI Framework** | PyQt6 ≥ 6.5 |
| **3D Rendering** | PyVista ≥ 0.42 / VTK ≥ 9.2 via PyVistaQt |
| **Numerical Core** | NumPy, SciPy, Numba (JIT-compiled kernels) |
| **Data Handling** | Pandas ≥ 2.0 |
| **2D Charting** | Matplotlib ≥ 3.7 |
| **Geological Modelling** | LoopStructural ≥ 1.6, GemPy ≥ 2024.0 |
| **Optimisation** | PuLP ≥ 2.7, custom Pseudoflow (Lerchs-Grossmann) |
| **Packaging** | PyInstaller ≥ 6.0 → standalone .exe (no runtime dependencies) |
| **OS Target** | Windows (primary), macOS installer scripts available |
| **Configuration** | TOML / JSON, stored in `~/.geox/config.toml` |

---

## 4. Architecture

GeoX follows a strict **four-layer architecture** with no upward dependency leakage:

```
┌─────────────────────────────────────────────────────┐
│  Layer 4 — UI  (PyQt6 widgets, 150+ panels)         │
│  Signals/slots; no domain logic                      │
├─────────────────────────────────────────────────────┤
│  Layer 3 — Controllers  (orchestration)              │
│  AppController, GeostatsController, MiningController │
│  VisController, DataController, JobRegistry          │
│  No Qt imports                                       │
├─────────────────────────────────────────────────────┤
│  Layer 2 — Domain Engines  (pure computation)        │
│  geostats/, models/, mine_planning/, geology/,       │
│  geotech/, esg/                                     │
│  Pure NumPy/SciPy/Pandas — no UI dependencies       │
├─────────────────────────────────────────────────────┤
│  Layer 1 — Data & Persistence                        │
│  DataRegistry, DataProvenance (SHA-256),             │
│  AuditManager (JSONL), Parsers (CSV, Excel, VTK…)   │
└─────────────────────────────────────────────────────┘
```

**Key architectural invariants:**
- Domain engines must never import PyQt6.
- Controllers never access VTK actors directly.
- Every data transformation is logged with input/output SHA-256 checksums.
- Background computation runs through `JobRegistry` / `JobWorker` to keep the UI responsive.

---

## 5. Feature Requirements

### 5.1 Data Management

#### 5.1.1 Drillhole Import & Processing

| ID | Requirement | Priority |
|----|------------|----------|
| DM-01 | Import drillhole data from CSV, Excel, and SQLite | Must |
| DM-02 | Import Leapfrog Geo format drillhole files | Should |
| DM-03 | Flexible column-mapping dialog with auto-detection of common names (HOLEID, X, Y, Z, FROM, TO) | Must |
| DM-04 | Validate: duplicate hole IDs, missing coordinates, depth violations, null grades | Must |
| DM-05 | Auto-fix: depth reordering, whitespace trimming, case normalisation | Must |
| DM-06 | Manual editing with undo/redo support | Must |
| DM-07 | QC sample handling: CRM, blanks, duplicates control | Should |

#### 5.1.2 Block Model Import

| ID | Requirement | Priority |
|----|------------|----------|
| DM-10 | Import block models from CSV with column mapping | Must |
| DM-11 | Auto-detect coordinate system (UTM, local) | Must |
| DM-12 | Support up to 500K blocks at import | Must |
| DM-13 | Block model definition: origin, dimensions (nx, ny, nz), rotation | Must |

#### 5.1.3 Compositing

| ID | Requirement | Priority |
|----|------------|----------|
| DM-20 | Length-weighted compositing with user-defined interval | Must |
| DM-21 | Domain-based compositing (separate within geological domains) | Must |
| DM-22 | Downhole compositing with minimum sample threshold | Should |
| DM-23 | Statistics output: count, min, max, mean, std, histogram | Must |
| DM-24 | Full parameter logging to DataRegistry | Must |

#### 5.1.4 Declustering

| ID | Requirement | Priority |
|----|------------|----------|
| DM-30 | Cell-based declustering with uniform cell sizes | Must |
| DM-31 | Origin-offset method (Deutsch 1989) | Should |
| DM-32 | Effective sample count (N_eff) calculation | Must |

#### 5.1.5 Data Provenance & Audit

| ID | Requirement | Priority |
|----|------------|----------|
| DM-40 | SHA-256 checksums for all data inputs and outputs | Must |
| DM-41 | JSONL audit logs (daily files) recording every transformation | Must |
| DM-42 | Crash handler with full traceback capture | Must |
| DM-43 | Process history tracking for undo/redo chains | Should |

---

### 5.2 Geostatistics

#### 5.2.1 Variography

| ID | Requirement | Priority |
|----|------------|----------|
| GS-01 | Omnidirectional and directional experimental variograms | Must |
| GS-02 | Configurable: distance lags, lag tolerance, angular bandwidth | Must |
| GS-03 | Minimum pair count thresholds per lag | Must |
| GS-04 | Variogram cloud visualisation | Should |
| GS-05 | Directional variogram surface (azimuth × dip) | Should |
| GS-06 | Downhole variogram with pair weighting | Should |
| GS-07 | Accept external declustering weights | Must |

#### 5.2.2 Variogram Model Fitting

| ID | Requirement | Priority |
|----|------------|----------|
| GS-10 | Nested structures: Spherical, Exponential, Gaussian, Power | Must |
| GS-11 | Interactive graphical fitting with real-time preview | Must |
| GS-12 | Nugget + partial sill + range parameterisation | Must |
| GS-13 | Anisotropy angles (azimuth, dip, pitch) with anisotropy ratios | Must |
| GS-14 | Variogram assistant wizard for guided fitting | Should |
| GS-15 | Export fitted models to DataRegistry | Must |

#### 5.2.3 Kriging (8 Methods)

| ID | Method | Description | Priority |
|----|--------|-------------|----------|
| GS-20 | Ordinary Kriging (OK) | Stationary deposits, local mean estimation | Must |
| GS-21 | Simple Kriging (SK) | Known global mean, residual field estimation | Must |
| GS-22 | Universal Kriging (UK) | Linear & quadratic polynomial trend removal | Must |
| GS-23 | Indicator Kriging (IK) | Threshold-based proportion estimation, order relation correction | Must |
| GS-24 | Co-Kriging | Multi-variable estimation with cross-variograms | Should |
| GS-25 | Bayesian Kriging | Soft data integration with precision weighting | Should |
| GS-26 | Soft Kriging | Bayesian-style uncertainty incorporation | Could |
| GS-27 | Leave-One-Out CV | R², RMSE, MAE, slope, normalised RMSE | Must |

**Common kriging features:**
- Anisotropic search neighbourhoods (azimuth, dip, pitch, ranges)
- Octant/sector search control
- Min/max sample count constraints
- Multi-pass search strategies
- Output: estimated grade, kriging variance, sample count per block
- Numba-compiled solver kernels
- Background execution via JobRegistry

#### 5.2.4 Radial Basis Functions (2 Methods)

| ID | Method | Description | Priority |
|----|--------|-------------|----------|
| GS-30 | ARBF (Adaptive RBF with LVA) | 10-step PUM workflow with JORC audit | Must |
| GS-31 | FastRBF | Rapid gridding (thin-plate, multiquadric, Gaussian, inverse quadratic, linear) | Should |

**ARBF capabilities:**
- Partition-of-Unity Method (PUM) with Wendland C2 blending
- 6 kernel types: Spheroidal, Gaussian, Matérn 3/2, Matérn 5/2, Cubic, Wendland C2
- Locally Varying Anisotropy (LVA): data-driven, structural, or identity
- Change-of-support (Matheron affine correction)
- Leave-one-out CV via Bartlett virtual formula (fast, no recomputation)
- JORC dual-criteria classification (variance + geometric)
- JORC Table 1 audit record generation (JSON)
- Normal-score transform with ratio-preserving rescale

#### 5.2.5 Stochastic Simulation (7 Methods)

| ID | Method | Description | Priority |
|----|--------|-------------|----------|
| GS-40 | SGSIM | Sequential Gaussian Simulation, multi-realisations | Must |
| GS-41 | Turning Bands | 1D line–process simulation | Should |
| GS-42 | IK-SGSIM | Indicator kriging–based simulation for categorical variables | Should |
| GS-43 | SIS | Sequential Indicator Simulation | Should |
| GS-44 | GRF | Gaussian Random Field generator | Could |
| GS-45 | MPS | Multiple Point Statistics from training images | Could |
| GS-46 | CoSGSIM | Co-spatial simulation for multi-element estimation | Should |

**Common simulation features:**
- Explicit seed parameter for deterministic reproducibility
- Normal-score transform and back-transform
- E-type (mean), P10, P50, P90 summary grids
- Variogram reproduction validation statistics
- Numba-accelerated kernels

---

### 5.3 Geological Modelling

| ID | Requirement | Priority |
|----|------------|----------|
| GM-01 | Implicit surface modelling with PUM auto-dispatch for large datasets | Must |
| GM-02 | Vein model with scalar-field barren-hole pinch-out | Should |
| GM-03 | Fold frame with curvilinear S2 (local fold axis per point) | Should |
| GM-04 | Cross-section widget for 2D slice visualisation | Must |
| GM-05 | LoopStructural integration for implicit folding and faulting | Must |
| GM-06 | GemPy integration for stratigraphic modelling | Should |
| GM-07 | Lithology classification and grouping | Must |
| GM-08 | DXF wireframe import (faults, folds) | Should |
| GM-09 | Structural CSV import (dip, dip direction, plunge, trend) | Must |
| GM-10 | Fault definition panel with displacement vectors | Must |
| GM-11 | Fold definition panel with fold axis and limb geometry | Should |
| GM-12 | Vein definition panel with boundary constraints | Should |
| GM-13 | Rose diagram and stereonet visualisation | Must |
| GM-14 | Domain mask support for domain-separated estimation | Must |
| GM-15 | Domain-aware declustering and compositing | Must |

---

### 5.4 Resource Classification & Reporting

| ID | Requirement | Priority |
|----|------------|----------|
| RC-01 | Dual-criteria JORC classification: variance-based AND geometric (sample count, octant coverage) | Must |
| RC-02 | Variance thresholds T1, T2, T3 as fractions of variogram sill | Must |
| RC-03 | Geometric criteria: minimum samples + minimum octant coverage | Must |
| RC-04 | Auto-scaling when median variance exceeds T3 | Should |
| RC-05 | JORC Table 1 audit record (JSON) with all parameters and checksums | Must |
| RC-06 | Classification visualisation co-located with block model | Must |
| RC-07 | Grade-tonnage curves at variable cutoff grades | Must |
| RC-08 | Contained metal calculations | Must |
| RC-09 | Domain-level resource breakdown | Must |
| RC-10 | Classification category breakdown (Measured, Indicated, Inferred) | Must |
| RC-11 | Export to CSV and formatted Excel workbooks | Must |

---

### 5.5 Mine Planning

#### 5.5.1 Pit Optimisation

| ID | Requirement | Priority |
|----|------------|----------|
| MP-01 | Pseudoflow maximum-closure algorithm (Lerchs-Grossmann) | Must |
| MP-02 | Nested pit shells at variable revenue factors | Must |
| MP-03 | User-defined: slope angles, mining/processing costs, recovery rates, commodity price | Must |
| MP-04 | NPV and IRR calculation per pit shell | Must |
| MP-05 | Sensitivity analysis (price, cost, recovery) | Must |
| MP-06 | Stochastic pit optimisation (Monte Carlo on simulation realisations) | Should |
| MP-07 | Pit shell visualisation as block model property | Must |

#### 5.5.2 Production Scheduling

| ID | Requirement | Priority |
|----|------------|----------|
| MP-10 | Strategic (long-term): annual targets, pit phasing, domain blending | Must |
| MP-11 | Tactical (short-term): quarterly/monthly schedules, bench extraction, pushback design | Should |
| MP-12 | Short-term (daily/weekly): truck dispatch, fleet requirements, haulage optimisation | Could |
| MP-13 | Gantt chart visualisation | Should |
| MP-14 | Schedule export to CSV and Excel | Must |

#### 5.5.3 Financial Analysis

| ID | Requirement | Priority |
|----|------------|----------|
| MP-20 | NPV calculation with user-defined discount rates | Must |
| MP-21 | IRR calculation and sensitivity analysis | Must |
| MP-22 | Scenario planning: multi-pit shells, scheduling alternatives | Should |
| MP-23 | Planning and production dashboards | Should |
| MP-24 | Pushback designer with NPV integration | Should |

#### 5.5.4 Grade Control & Reconciliation

| ID | Requirement | Priority |
|----|------------|----------|
| MP-30 | Grade control simulation and ore/waste marking | Should |
| MP-31 | Model-to-mine reconciliation | Should |
| MP-32 | Mine-to-mill recovery reconciliation | Could |
| MP-33 | Tonnage and grade balance checks | Should |

#### 5.5.5 Cutoff Optimisation

| ID | Requirement | Priority |
|----|------------|----------|
| MP-40 | Lane's method: mine/mill/market balance for optimal cutoff | Must |
| MP-41 | Dynamic cutoff as function of remaining resource | Should |
| MP-42 | Economic sensitivity (price and cost variation) | Must |

---

### 5.6 Geotechnical Analysis

| ID | Requirement | Priority |
|----|------------|----------|
| GT-01 | Rock mass classification: GSI/RMR → UCS/friction angle conversions | Should |
| GT-02 | Limit equilibrium slope stability (2D) | Should |
| GT-03 | Probabilistic slope failure risk assessment (Monte Carlo) | Could |
| GT-04 | Underground stope stability design | Could |
| GT-05 | Seismic hazard volume and rockburst risk index | Could |
| GT-06 | Spatial interpolation of rock mass properties | Should |

---

### 5.7 Geometallurgy

| ID | Requirement | Priority |
|----|------------|----------|
| GM-20 | Multi-element block model with recovery modelling | Should |
| GM-21 | Comminution and liberation models | Could |
| GM-22 | Plant response and separation curves | Could |
| GM-23 | Domain-to-recovery chain mapping | Should |

---

### 5.8 ESG & Sustainability

| ID | Requirement | Priority |
|----|------------|----------|
| ESG-01 | GHG emissions tracking (Scope 1, 2, 3) with equipment-specific factors | Should |
| ESG-02 | Water balance and consumption tracking | Should |
| ESG-03 | Waste rock and tailings volume estimation | Should |
| ESG-04 | Compliance reporting: GRI, TCFD, SASB standards | Could |
| ESG-05 | ESG dashboard with metrics visualisation | Should |

---

### 5.9 Remote Sensing & Point Clouds

| ID | Requirement | Priority |
|----|------------|----------|
| RS-01 | LAS/PLY/XYZ point cloud import and visualisation | Should |
| RS-02 | InSAR displacement grid import and visualisation | Could |
| RS-03 | Multi-temporal coherence analysis | Could |
| RS-04 | Pit wall stability monitoring integration | Could |
| RS-05 | Survey deformation tracking | Could |

---

## 6. 3D Visualisation Requirements

| ID | Requirement | Priority |
|----|------------|----------|
| VIS-01 | Render 200K blocks at ≥ 30 fps | Must |
| VIS-02 | Render 500K blocks at ≥ 20 fps with LOD | Must |
| VIS-03 | Colour by any numeric or categorical attribute | Must |
| VIS-04 | Discrete and continuous colormaps (viridis, jet, plasma, custom) | Must |
| VIS-05 | Transparency and per-layer opacity control | Must |
| VIS-06 | Edge visibility toggle (auto-disabled > 50K cells) | Must |
| VIS-07 | Orthographic and perspective camera modes | Must |
| VIS-08 | 3D drillhole tube rendering coloured by grade or lithology | Must |
| VIS-09 | Interactive clip planes (X, Y, Z, oblique) | Must |
| VIS-10 | Object picking with O(1) cell lookup via Original_ID | Must |
| VIS-11 | Screenshot export (PNG, PDF) with scale bar and legend | Must |
| VIS-12 | Scene inspector: show/hide/opacity per object | Must |
| VIS-13 | Automatic coordinate transform (UTM ↔ local) with double-shift guard | Should |
| VIS-14 | Floating axes, scale bar, north arrow, grid overlays | Should |
| VIS-15 | Light/dark theme switching | Must |

---

## 7. Data Export Requirements

| ID | Format | Content | Priority |
|----|--------|---------|----------|
| EX-01 | CSV | Block models, drillhole data, schedules | Must |
| EX-02 | Excel (.xlsx) | Formatted resource reports, schedules | Must |
| EX-03 | VTK | Block models for external 3D tools | Should |
| EX-04 | PNG / PDF | Charts, screenshots | Must |
| EX-05 | JSON | JORC audit records | Must |
| EX-06 | JSONL | Audit trail logs | Must |

---

## 8. Performance Requirements

| Metric | Target | Rationale |
|--------|--------|-----------|
| Block model import (500K blocks) | < 30 s | Typical large deposit |
| Ordinary Kriging (100K blocks) | < 5 min | Standard estimation run |
| SGSIM 50 realisations (50K blocks) | < 30 min | Uncertainty analysis |
| ARBF estimation (50K blocks) | < 1 min | Rapid screening |
| 3D render (200K blocks) | ≥ 30 fps | Smooth interactive exploration |
| 3D render (500K blocks, LOD) | ≥ 20 fps | Large model inspection |
| UI responsiveness during estimation | No freeze | Background job execution |
| Application startup | < 10 s | Cold start to interactive |

---

## 9. Quality & Compliance Requirements

### 9.1 Audit Trail

- Every data transformation logged: timestamp, operator, parameters, input/output SHA-256 hashes.
- Daily JSONL audit files in `audit_logs/`.
- JORC Table 1 audit record auto-generated for every estimation run.
- Crash handler captures exceptions with full traceback and persists to audit log.

### 9.2 Determinism & Reproducibility

- All stochastic methods accept an explicit RNG seed parameter.
- Given identical inputs and seed, results must be bit-for-bit identical across runs.
- Determinism rules enforced via `determinism.py` module.

### 9.3 Data Integrity

- No silent data loss: every import/export verified by SHA-256 checksum.
- Sentinel values use `np.iinfo().min` (not -1) to avoid collision with legitimate domain codes.
- Coordinate transforms guarded against double-shift application.

### 9.4 Regulatory Compliance

| Standard | Coverage |
|----------|----------|
| JORC 2012 | Table 1 audit records, dual-criteria classification, variance + geometric |
| SAMREC | Classification criteria mapped to JORC equivalents |
| GRI | ESG reporting module |
| TCFD | Climate-related financial disclosure |
| SASB | Industry-specific sustainability metrics |

---

## 10. UI/UX Requirements

### 10.1 Layout

- Default window: 1200 × 800 px, resizable.
- Central 3D viewport with dockable analysis panels on left/right/bottom.
- Panels persist state (expanded/collapsed, parameter values) across sessions.
- Recent files list (max 10) in File menu.

### 10.2 Panel Architecture

| Base Class | Purpose |
|------------|---------|
| `BasePanel(QWidget)` | Signal emission, error handling, controller access |
| `BaseDockPanel(BasePanel)` | Persistent docking, state persistence |
| `BaseAnalysisPanel(BaseDockPanel)` | Grid layouts, parameter inputs, execute buttons |
| `BaseDisplayPanel(BaseDockPanel)` | 3D controls, overlays, scale/legend |

### 10.3 Panel Count (150+)

| Category | Approximate Count |
|----------|-------------------|
| Data Management | 20 |
| Geostatistics | 35 |
| Geological Modelling | 15 |
| Mine Planning | 25 |
| Geometallurgy | 7 |
| Geotechnical | 5 |
| ESG & Sustainability | 3 |
| 3D Visualisation | 20 |
| Advanced Analysis | 15 |
| Utilities & Preferences | 20 |

### 10.4 Themes

- Light and dark themes supported.
- Design token system for consistent styling.
- Theme switch at runtime without restart.

---

## 11. Packaging & Distribution

| Attribute | Value |
|-----------|-------|
| **Windows** | PyInstaller → standalone `GeoX.exe`; Inno Setup `.exe` installer; WiX `.msi` installer |
| **macOS** | DMG/Package scripts |
| **Runtime** | Bundled Python 3.12 — no user-visible Python installation required |
| **Size** | Standalone package includes all dependencies |
| **Updates** | Manual download (no auto-updater in v1) |

---

## 12. Testing Strategy

### 12.1 Test Suites

| Suite | Focus | Count |
|-------|-------|-------|
| Validation Framework | Block model, compositing, geostatistics, data integrity, workflows | 8 suites |
| Geostatistics Tests | Variogram pipelines, kriging nugget handling, domain masks | 7+ files |
| ARBF Tests | Production-scale (200K–500K blocks), quality gates, engine correctness | 11 files |
| Integration Tests | End-to-end workflows, panel-to-engine integration | Multiple |
| Performance Tests | Benchmark rendering and estimation at scale | Dedicated suite |

### 12.2 Validation References

- Kriging validated against GSLIB reference outputs.
- ARBF validated via method audit with documented results.
- Variogram engines validated against theoretical models.

---

## 13. Known Constraints & Limitations (v1.0)

| Area | Constraint |
|------|-----------|
| **GPU** | CPU-based rendering via VTK; no direct CUDA/GPU compute for estimation (Numba JIT only) |
| **Scale** | Tested to 500K blocks; performance above this threshold is not guaranteed |
| **Networking** | Standalone desktop; no multi-user collaboration or cloud deployment |
| **Auto-update** | No built-in update mechanism; manual download required |
| **macOS** | Installer scripts present but not production-validated |
| **Linux** | No dedicated packaging or testing |
| **Undo** | Undo/redo for drillhole edits; not all estimation operations are reversible |

---

## 14. Roadmap Considerations (Post-v1)

The following items are not committed but represent natural evolution paths based on the current architecture:

1. **Cloud/multi-user deployment** — web-based 3D viewer for collaborative review.
2. **GPU-accelerated estimation** — CuPy/CUDA kernels for kriging and simulation.
3. **Machine learning integration** — gradient-boosted estimation, domain boundary detection.
4. **Auto-updater** — in-app update checking and installation.
5. **Linux packaging** — AppImage or Flatpak distribution.
6. **Database backends** — PostgreSQL/PostGIS for project-level data management.
7. **Plugin architecture** — third-party estimation or planning modules.
8. **Multi-language UI** — internationalisation framework.

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **ARBF** | Adaptive Radial Basis Function — RBF interpolation with Partition-of-Unity, LVA, and change-of-support |
| **CoS** | Change of Support — volume-variance correction (Matheron affine) |
| **CV** | Cross-Validation — leave-one-out error assessment |
| **E-type** | Expected type — mean of multiple simulation realisations |
| **IK** | Indicator Kriging — threshold-based probability estimation |
| **JORC** | Joint Ore Reserves Committee — Australasian reporting code for mineral resources |
| **LOD** | Level of Detail — progressive mesh simplification for rendering performance |
| **LVA** | Locally Varying Anisotropy — spatially variable anisotropy ellipsoid |
| **NPV** | Net Present Value |
| **IRR** | Internal Rate of Return |
| **OK** | Ordinary Kriging |
| **PUM** | Partition of Unity Method — overlapping domain blending |
| **SAMREC** | South African Mineral Resource Committee — SA reporting code |
| **SGSIM** | Sequential Gaussian Simulation |
| **SK** | Simple Kriging |
| **UK** | Universal Kriging |

---

*End of document.*
