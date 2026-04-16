# Product Requirements Document

**Product:** GeoX Desktop
**Version:** 1.0.0
**Status:** Active Development
**Date:** March 2026
**License:** MIT

---

## 1. Executive Summary

GeoX Desktop is an integrated 3D geostatistical workbench and mine planning platform for the mining industry. It delivers a complete, auditable resource estimation workflow — from raw drillhole data through compositing, variography, spatial estimation, resource classification, geological modelling, and mine planning — within a single desktop application.

The software addresses a critical gap: commercial geostatistical packages cost $20,000–$50,000+ per seat per year, while open-source alternatives are fragmented CLI tools with no 3D visualization, no audit trail, and no integrated mine planning. GeoX unifies 15+ estimation methods, interactive 3D rendering of 500K+ blocks, JORC-compliant audit logging, and full mine-to-mill planning under one roof.

---

## 2. Problem Statement

### 2.1 User Pain Points

| Persona | Pain Point |
|---------|-----------|
| Resource Estimator | Must switch between 3+ tools (Vulcan, Supervisor, R/Python scripts) for one estimation workflow; no single audit trail |
| Geostatistician | Commercial software hides variogram internals, cannot run LOO-CV automatically, and lacks Locally Varying Anisotropy (LVA) support |
| Data Steward | No provable chain of custody from raw assays → composites → estimates; changes to composite parameters are unlogged |
| Mine Planner | 3D block model viewers are separate from estimation and planning tools; NPV/pit optimization requires yet another package |
| Uncertainty Analyst | Running multi-realization simulations requires external scripting; no integrated P10/P50/P90 workflow |

### 2.2 Market Context

- **Commercial desktop products** (Leapfrog Geo, Vulcan, Micromine, Surpac) cost $20,000–$50,000+ per seat per year with proprietary file formats and mandatory license servers
- **Open-source geostatistics** (GSLIB, GeostatsPy, SGeMS) are CLI-only or Python-library-only — no 3D visualization, no integrated workflow
- **Open-source 3D viewers** (Paraview, PyVista standalone) have no built-in estimation, compositing, or resource classification
- No single open-source solution integrates: data management + estimation + 3D visualization + mine planning + JORC compliance

---

## 3. Product Vision

> GeoX Desktop is the definitive open-source geostatistical workbench for professional resource estimators — combining rigorous, publication-quality estimation algorithms with interactive 3D visualization, complete JORC audit trail compliance, and integrated mine planning, accessible to any mining operation regardless of budget.

---

## 4. Goals and Non-Goals

### 4.1 Goals (Version 1.0)

| ID | Goal | Success Metric |
|----|------|---------------|
| G-01 | Complete workflow from raw drillhole CSV to JORC-classified block model export | A new user completes end-to-end workflow in ≤2 hours |
| G-02 | 15+ estimation methods covering kriging family, RBF, and stochastic simulation | All methods produce correct output validated against GSLIB reference datasets |
| G-03 | Deterministic reproducibility | Same data + parameters + seed = byte-identical output |
| G-04 | Full audit trail for every data transformation | Every operation logged with timestamp, operator, parameters, input/output SHA-256 hash |
| G-05 | Interactive 3D visualization at scale | 200K blocks at ≥30 fps; 500K blocks at ≥20 fps with LOD |
| G-06 | JORC Table 1 Section 3 compliant audit records | Automatic audit record generation for ARBF, kriging, and simulation runs |
| G-07 | Integrated mine planning | Pit optimization, scheduling, NPV/IRR, grade-tonnage, and ESG reporting in one application |
| G-08 | Standalone Windows installer | Ship as single installer with no license server, no external dependencies |

### 4.2 Non-Goals (Version 1.0)

- Cloud or web deployment — desktop only
- Real-time streaming sensor data ingestion
- Multi-user concurrent collaboration or version-controlled project sharing
- Native macOS/Linux packaged installers (scripts exist but are untested)
- Replace dedicated geological wireframe modelling tools (users import domains from Leapfrog/LoopStructural)

---

## 5. Target Users

### 5.1 Persona A — Resource Estimator / Geostatistician

- **Background:** MSc or PhD in geoscience or applied mathematics; 2–10 years in mining
- **Goals:** Run a defensible, reproducible mineral resource estimate; export for JORC/SAMREC/NI 43-101 reporting
- **Frustrations:** Cannot verify what software did internally; must write external scripts for cross-validation; no fine-grained variogram control
- **Key features:** Variography, Kriging (OK/SK/UK/IK/Co-kriging/Bayesian), ARBF with LVA, LOO-CV, swath plots, JORC classification, audit export

### 5.2 Persona B — Data Steward / Database Manager

- **Background:** Geologist or database specialist maintaining drill data libraries
- **Goals:** Import raw CSV data, composite, validate, detect QC failures, deliver clean datasets to estimators with provenance
- **Frustrations:** No provable chain of custody; composite parameter changes are invisible
- **Key features:** Drillhole import, compositing engine, validation rules, QC window, data provenance (SHA-256 hashing), JSONL audit logs

### 5.3 Persona C — Mine Planner

- **Background:** Mining engineer; familiar with tonnes and grades but not geostatistics
- **Goals:** Examine block model results, evaluate cutoffs, run pit optimization, build production schedules, assess NPV/IRR
- **Frustrations:** Receives block model CSVs via email and opens in separate tools; no interactive 3D access to estimation results
- **Key features:** 3D viewer, grade-tonnage curves, cutoff optimization (Lane's method), nested pit shell optimization, production scheduling (strategic/tactical/short-term), fleet sizing, NPV/IRR analysis, ESG dashboards

### 5.4 Persona D — Uncertainty Analyst / Researcher

- **Background:** PhD-level; focused on simulation uncertainty and probabilistic classification
- **Goals:** Run multiple SGSIM realizations, assess P10/P50/P90 distributions, compare uncertainty models, propagate uncertainty into economics
- **Key features:** SGSIM, Turning Bands, CoSGSIM, MPS, uncertainty propagation panels, realization export, probabilistic pit shells

---

## 6. Features and Requirements

### 6.1 Data Management

#### F-01: Drillhole Data Import and Validation
- **FR-01.1** Import drillhole data from CSV files (collar, survey, assay, lithology tables) with flexible column mapping dialog
- **FR-01.2** Import from Excel workbooks and SQLite databases
- **FR-01.3** Import from Leapfrog Geo export format
- **FR-01.4** Auto-detect common column names (HOLEID, X, Y, Z, FROM, TO, etc.)
- **FR-01.5** Validate: duplicate hole IDs, missing coordinates, depth ordering violations, null grades, coordinate range sanity
- **FR-01.6** Auto-fix common data issues (depth reordering, whitespace trimming, case normalization)
- **FR-01.7** Manual data editing with undo/redo support
- **FR-01.8** Mark holes/intervals for exclusion without deletion
- **FR-01.9** Control sample handling (CRM, blanks, duplicates)

#### F-02: Compositing Engine
- **FR-02.1** Length-weighted compositing with user-specified interval length
- **FR-02.2** Domain-based compositing (separate composites within geological domains)
- **FR-02.3** Downhole compositing with minimum sample threshold
- **FR-02.4** Composite statistics output (count, min, max, mean, std, histogram)
- **FR-02.5** Output composites to DataRegistry with full parameter logging

#### F-03: Declustering
- **FR-03.1** Cell-based declustering with uniform cell sizes
- **FR-03.2** Origin-offset method (Deutsch 1989) with user-specified grid search
- **FR-03.3** Output weights to DataFrame for downstream variography
- **FR-03.4** Effective sample count (N_eff) calculation
- **FR-03.5** Weighted quantile computation
- **FR-03.6** Weight normalization (scale-invariant, sum-to-N)

#### F-04: Data Provenance and Audit
- **FR-04.1** SHA-256 checksums of all input/output data
- **FR-04.2** JSONL audit logs per day in `audit_logs/` directory
- **FR-04.3** Every transformation logged with: timestamp, operator name, parameters, input hash, output hash
- **FR-04.4** Crash handler captures unhandled exceptions with full traceback to audit log
- **FR-04.5** Process history tracking for undo/redo chain

#### F-05: Data Registry
- **FR-05.1** Central in-memory registry storing drillholes, block models, estimation results, surfaces
- **FR-05.2** Qt signals for panel synchronization on data load/update/remove
- **FR-05.3** Coordinate alignment detection: auto-detect UTM vs local mismatch between datasets
- **FR-05.4** `get_aligned_data()` convenience method for panels requiring matched coordinate systems

### 6.2 Geostatistics — Variography

#### F-06: Experimental Variogram
- **FR-06.1** Omnidirectional and directional experimental variogram calculation
- **FR-06.2** Configurable distance lags, lag tolerance, angular bandwidth
- **FR-06.3** Minimum pair count thresholds per lag
- **FR-06.4** Variogram cloud visualization
- **FR-06.5** Directional variogram surface (azimuth × dip)
- **FR-06.6** Accept external declustering weights (`sample_weights` parameter)
- **FR-06.7** Downhole variogram with pair weighting (w = w_i × w_j)

#### F-07: Variogram Model Fitting
- **FR-07.1** Nested structures: Spherical, Exponential, Gaussian, Power
- **FR-07.2** Interactive graphical fitting with real-time preview
- **FR-07.3** Nugget + partial sill + range parameterization
- **FR-07.4** Anisotropy angles (azimuth, dip, pitch) with anisotropy ratios
- **FR-07.5** Variogram assistant wizard for guided fitting
- **FR-07.6** Export fitted model to DataRegistry for use in estimation

### 6.3 Geostatistics — Estimation Methods

#### F-08: Kriging Family (8 methods)

| Method | ID | Key Capability |
|--------|----|---------------|
| Ordinary Kriging (OK) | FR-08.1 | Stationary deposits, local mean estimation |
| Simple Kriging (SK) | FR-08.2 | Known global mean, residual field estimation |
| Universal Kriging (UK) | FR-08.3 | Linear and quadratic polynomial trend removal |
| Indicator Kriging (IK) | FR-08.4 | Threshold-based proportion estimation with order relation correction |
| Co-Kriging | FR-08.5 | Multi-variable estimation with cross-variograms |
| Bayesian Kriging | FR-08.6 | Soft data integration with precision weighting |
| Soft Kriging | FR-08.7 | Bayesian-style uncertainty incorporation |
| LOO Cross-Validation | FR-08.8 | Leave-one-out CV with R², RMSE, MAE, slope, normalized RMSE |

**Common kriging requirements:**
- **FR-08.9** Anisotropic search neighborhoods (azimuth, dip, pitch, ranges)
- **FR-08.10** Octant/sector search control
- **FR-08.11** Minimum/maximum sample count constraints
- **FR-08.12** Output: estimated grade, kriging variance, sample count per block
- **FR-08.13** Numba-compiled solver kernels for performance
- **FR-08.14** Background execution via JobRegistry with progress bar and cancellation

#### F-09: RBF Methods (2 methods)

| Method | ID | Key Capability |
|--------|----|---------------|
| ARBF (Adaptive RBF) | FR-09.1 | 10-step workflow: PUM, LVA, CoS, JORC audit |
| FastRBF | FR-09.2 | Rapid gridding with thin-plate spline, multiquadric, Gaussian kernels |

**ARBF-specific requirements:**
- **FR-09.3** Partition-of-Unity Method (PUM) with Wendland C2 blending (auto-dispatch at N_v + N_g > 1,500)
- **FR-09.4** 6 kernel types: Spheroidal, Gaussian, Matérn 3/2, Matérn 5/2, Cubic, Wendland C2
- **FR-09.5** Locally Varying Anisotropy (LVA) from data-driven (Boisvert 2009), structural, or identity
- **FR-09.6** Change-of-support correction (Matheron affine correction)
- **FR-09.7** Leave-one-out CV via Bartlett virtual formula (fast, no recomputation)
- **FR-09.8** JORC dual-criteria classification (variance + geometric)
- **FR-09.9** JORC Table 1 audit record generation (JSON)
- **FR-09.10** Normal-score transform with ratio-preserving rescale back to original scale

#### F-10: Stochastic Simulation (7 methods)

| Method | ID | Key Capability |
|--------|----|---------------|
| SGSIM | FR-10.1 | Sequential Gaussian Simulation with multiple realizations, P10/P50/P90 |
| Turning Bands | FR-10.2 | 1D line process simulation |
| IK-SGSIM | FR-10.3 | Indicator kriging-based simulation for categorical variables |
| SIS | FR-10.4 | Sequential Indicator Simulation for multi-threshold proportion grids |
| GRF | FR-10.5 | Gaussian Random Field generator |
| MPS | FR-10.6 | Multiple Point Statistics simulation from training images |
| CoSGSIM | FR-10.7 | Co-spatial Gaussian Simulation for correlated multi-element estimation |

**Common simulation requirements:**
- **FR-10.8** Explicit seed parameter for deterministic reproducibility
- **FR-10.9** Normal-score transform and back-transform
- **FR-10.10** E-type (mean), P10, P50, P90 summary grids from multi-realization sets
- **FR-10.11** Variogram reproduction validation statistics
- **FR-10.12** Numba-accelerated kernels

### 6.4 Geological Modelling

#### F-11: Implicit Surface Modelling
- **FR-11.1** Core scalar field engine with PUM auto-dispatch for large datasets
- **FR-11.2** Vein model with scalar-field barren-hole pinch-out
- **FR-11.3** Fold frame with true curvilinear S2 (local fold axis computed per point)
- **FR-11.4** 8-tab geological model panel
- **FR-11.5** Cross-section widget for 2D slice visualization
- **FR-11.6** LoopStructural integration for implicit folding and faulting
- **FR-11.7** GemPy integration for stratigraphic modelling
- **FR-11.8** Fault and fold definition panels with structural data import
- **FR-11.9** Lithology classification and grouping dialog
- **FR-11.10** Domain mask support for domain-separated estimation

#### F-12: Structural Geology
- **FR-12.1** DXF wireframe import for faults and folds
- **FR-12.2** Structural CSV import (dip, dip direction, plunge, trend)
- **FR-12.3** Fault definition panel with displacement vectors
- **FR-12.4** Fold definition panel with fold axis and limb geometry
- **FR-12.5** Vein definition panel with boundary constraints
- **FR-12.6** Rose diagram and stereonet visualization

### 6.5 Resource Classification and Reporting

#### F-13: JORC Classification
- **FR-13.1** Dual-criteria classification: variance-based AND geometric (sample count, octant coverage)
- **FR-13.2** Variance thresholds T1 (Measured), T2 (Indicated), T3 (Inferred) as fractions of variogram sill
- **FR-13.3** Geometric criteria: minimum samples per block + minimum octant coverage
- **FR-13.4** Auto-scaling when median variance exceeds T3 threshold
- **FR-13.5** JORC Table 1 audit record output (JSON) with all parameters and checksums
- **FR-13.6** Classification visualization co-located with estimation block model

#### F-14: Resource Reporting
- **FR-14.1** Grade-tonnage curves at variable cutoff grades
- **FR-14.2** Contained metal calculations
- **FR-14.3** Domain-level resource breakdown
- **FR-14.4** Classification category breakdown (Measured, Indicated, Inferred)
- **FR-14.5** Export to CSV and formatted Excel workbooks

### 6.6 Mine Planning

#### F-15: Pit Optimization
- **FR-15.1** Pseudoflow maximum-closure algorithm (Lerchs-Grossmann)
- **FR-15.2** Nested pit shells at variable revenue factors
- **FR-15.3** User-defined slope angles, mining/processing costs, recovery rates, commodity price
- **FR-15.4** NPV and IRR calculation per pit shell
- **FR-15.5** Sensitivity analysis (price, cost, recovery)
- **FR-15.6** Stochastic pit optimization (Monte Carlo on simulation realizations)
- **FR-15.7** Pit shell visualization as block model property (not mesh overlay)

#### F-16: Production Scheduling
- **FR-16.1** Strategic (long-term): annual production targets, pit phase sequencing, domain blending
- **FR-16.2** Tactical (short-term): quarterly/monthly schedules, bench extraction, pushback design
- **FR-16.3** Short-term (daily/weekly): truck dispatch, fleet requirements, haulage route optimization
- **FR-16.4** Gantt chart visualization for schedules
- **FR-16.5** Export schedules to CSV and Excel

#### F-17: Financial Analysis
- **FR-17.1** NPV calculation with user-defined discount rates
- **FR-17.2** IRR calculation and sensitivity analysis
- **FR-17.3** Scenario planning: compare multiple pit shells, schedules, processing alternatives
- **FR-17.4** Planning and production dashboards with KPI tracking
- **FR-17.5** Pushback designer with NPV integration

#### F-18: Grade Control and Reconciliation
- **FR-18.1** Grade control simulation and ore/waste block marking
- **FR-18.2** Grade control decision support panel
- **FR-18.3** Model-to-mine reconciliation
- **FR-18.4** Mine-to-mill recovery reconciliation
- **FR-18.5** Tonnage and grade balance checks

#### F-19: Cutoff Grade Optimization
- **FR-19.1** Lane's method: mine/mill/market balance for optimal cutoff
- **FR-19.2** Dynamic cutoff as function of remaining resource
- **FR-19.3** Economic sensitivity (price and cost variation)

### 6.7 Visualization

#### F-20: 3D Block Model Rendering
- **FR-20.1** Render up to 500,000 blocks at ≥20 fps with Level-of-Detail (LOD) downsampling
- **FR-20.2** Color by any numeric or categorical attribute
- **FR-20.3** Discrete and continuous colormaps (viridis, jet, plasma, custom)
- **FR-20.4** Transparency and per-layer opacity control
- **FR-20.5** Edge visibility toggle with automatic disable for large models (>50K cells)
- **FR-20.6** Orthographic and perspective camera modes

#### F-21: Drillhole Rendering
- **FR-21.1** 3D tube rendering colored by grade or lithology
- **FR-21.2** Collar, trace, and sample interval labels
- **FR-21.3** Filter by domain, grade range, or hole ID
- **FR-21.4** GPU-accelerated rendering for large datasets

#### F-22: Interactive Controls
- **FR-22.1** Orbit, pan, zoom with configurable mouse modes
- **FR-22.2** Interactive clip planes (X, Y, Z, oblique)
- **FR-22.3** Real-time cursor coordinate display
- **FR-22.4** Object picking and value inspection on click
- **FR-22.5** Screenshot export (PNG, PDF) with scale bar and legend

#### F-23: Scene Management
- **FR-23.1** Scene inspector listing all visible objects with show/hide/opacity controls
- **FR-23.2** Automatic coordinate transform (UTM ↔ local) with double-shift guard
- **FR-23.3** Floating axes, scale bar, north arrow, grid overlays
- **FR-23.4** Legend synchronized with renderer (discrete and continuous modes)
- **FR-23.5** Light/dark theme switching

### 6.8 Geotechnical and Specialized Modules

#### F-24: Geotechnical Analysis
- **FR-24.1** Rock mass classification (RMR, Q-system)
- **FR-24.2** Limit equilibrium slope stability analysis (2D and 3D)
- **FR-24.3** Probabilistic slope failure risk assessment
- **FR-24.4** Underground stope stability design
- **FR-24.5** Seismic hazard and rockburst risk assessment

#### F-25: Geometallurgy
- **FR-25.1** Multi-element block model with recovery modelling
- **FR-25.2** Comminution and liberation models
- **FR-25.3** Plant response and separation curves
- **FR-25.4** Domain-to-recovery chain mapping

#### F-26: Environmental, Social, Governance (ESG)
- **FR-26.1** GHG emissions tracking (Scope 1 + Scope 2)
- **FR-26.2** Water balance and consumption tracking
- **FR-26.3** Waste rock and tailings volume estimation
- **FR-26.4** Compliance reporting: GRI, TCFD, SASB standards
- **FR-26.5** ESG dashboard with metrics visualization

#### F-27: Uncertainty Propagation
- **FR-27.1** Latin Hypercube Sampling for parameter uncertainty
- **FR-27.2** Probabilistic pit shells from simulation realizations
- **FR-27.3** Economic uncertainty propagation (grade → revenue → NPV)
- **FR-27.4** P10/P50/P90 uncertainty quantification panels

#### F-28: Point Cloud and Remote Sensing
- **FR-28.1** LAS/PLY/XYZ point cloud import (via laspy, Open3D)
- **FR-28.2** Point cloud visualization in 3D viewer
- **FR-28.3** InSAR displacement grid import and visualization
- **FR-28.4** Survey deformation tracking

### 6.9 Data Export

#### F-29: Export Capabilities
- **FR-29.1** Block model export to CSV, Excel, VTK
- **FR-29.2** Drillhole data export to CSV
- **FR-29.3** Chart/figure export to PNG, PDF
- **FR-29.4** Resource reports to formatted Excel workbooks
- **FR-29.5** JORC audit records to JSON
- **FR-29.6** Audit trails to JSONL

---

## 7. Performance Requirements

| Metric | Target |
|--------|--------|
| ARBF estimation (N=2K samples, B=10K blocks) | < 30 seconds |
| ARBF estimation (N=2K samples, B=259K blocks) | < 180 seconds |
| Ordinary Kriging (N=1K samples, B=50K blocks) | < 60 seconds |
| SGSIM (100 realizations, 64K nodes) | < 120 seconds |
| 3D render (200K blocks) | ≥ 30 fps |
| 3D render (500K blocks with LOD) | ≥ 20 fps |
| Application startup to interactive | < 10 seconds |
| Drillhole import (10,000 intervals) | < 5 seconds |

---

## 8. Quality and Accuracy Requirements

| Requirement | Acceptance Criteria |
|-------------|-------------------|
| Kriging variance | Non-negative (clamped for numerical noise) |
| LOO-CV R² | Matches brute-force within ±0.005 (Bartlett formula) |
| Change-of-support ratio | 0 < σ_block/σ_point ≤ 1.0 |
| Classification auto-scaling | Preserves median block in Inferred category |
| Normal-score back-transform | Original-scale sill matches data variance (ratio-preserving rescale) |
| Simulation reproduction | Variogram reproduction within 10% of model for 95% of lags |
| Determinism | Identical output for identical inputs + seed across runs |

---

## 9. Platform and Deployment

| Attribute | Specification |
|-----------|--------------|
| Operating System | Windows 10/11 (primary); macOS/Linux (scripts available, untested) |
| Python | 3.10, 3.11, 3.12, 3.13 |
| Packaging | PyInstaller standalone installer (no license server) |
| License | MIT |
| Minimum RAM | 8 GB (16 GB recommended for large models) |
| GPU | Any GPU with OpenGL 3.2+ support (for VTK rendering) |

---

## 10. Roadmap

### Version 1.1
- Full ILR compositional multi-element estimation (currently first component only)
- P50 realization-based JORC classification from simulation
- macOS native installer (tested)

### Version 1.2
- Cloud project sync (read-only sharing)
- Real-time grade control integration with mine dispatch systems
- Multi-GPU LOD rendering for >1M blocks

### Version 2.0
- Web-based viewer companion (read-only 3D viewing)
- Multi-user project collaboration
- Plugin SDK for third-party estimation methods

---

## 11. Known Limitations (Version 1.0)

| Limitation | Workaround |
|-----------|-----------|
| ILR multi-element: only first component | Use individual element estimation |
| ARBF PUM boundary artifacts for small datasets | Use single-domain mode (n_subdomains=1) for N ≤ 3,000 |
| ARBF LOO-CV uses global RBF, not PUM | Trust swath plots over CV R² for multi-domain deposits |
| Max tested block count: ~500,000 | Thin large models or use LOD |
| Windows-only packaged installer | Run from source on macOS/Linux |
| Simulation not integrated with JORC classification | Use kriging variance for classification, simulation for uncertainty |

---

## 12. Success Metrics

| Metric | Target |
|--------|--------|
| Time to first completed resource estimate (new user) | ≤ 2 hours |
| Estimation methods available | ≥ 15 |
| JORC audit fields covered | 100% of Table 1 Section 3 |
| Test coverage on core engines | ≥ 80% |
| Panel count (functional UI modules) | ≥ 100 |
| Zero-cost barrier to entry | MIT license, no license server |
