#!/usr/bin/env python3
"""
GeoX Data Flow & Parameter Analyzer
====================================
Systematic diagnostic that traces data flow, parameter usage, and signal
propagation across every panel in the GeoX mining software suite.

Generates a structured report covering:
  - Preprocessing chain (import -> compositing -> declustering -> transform)
  - Variogram parameter lifecycle (fitting -> storage -> consumption)
  - Estimation / simulation data flow
  - Resource classification workflow
  - Signal / registry dependency graph
  - Potential issues & inconsistencies

Usage:
    python scripts/data_flow_analyzer.py [--output report.txt] [--verbose]

Author : GeoX Audit Tooling
Date   : 2026-03-12
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKG = PROJECT_ROOT / "block_model_viewer"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalInfo:
    """Describes a Qt signal connection."""
    signal_name: str
    emitter: str         # class that emits
    receiver: str        # class/method that connects
    payload: str = ""    # rough payload description


@dataclass
class PanelReport:
    """Analysis results for one panel."""
    panel_name: str
    file_path: str
    category: str = ""
    signals_emitted: List[SignalInfo] = field(default_factory=list)
    signals_consumed: List[SignalInfo] = field(default_factory=list)
    registry_reads: List[str] = field(default_factory=list)
    registry_writes: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class WorkflowStage:
    """One step in the end-to-end resource-estimation workflow."""
    name: str
    panel: str
    engine: str
    input_keys: List[str]
    output_keys: List[str]
    description: str = ""


# ---------------------------------------------------------------------------
# Static code scanner helpers
# ---------------------------------------------------------------------------

def _read_source(path: Path, max_bytes: int = 500_000) -> str:
    """Read source file, capped to avoid memory issues on huge files."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except Exception:
        return ""


def _extract_signal_definitions(source: str) -> List[str]:
    """Find pyqtSignal / Signal definitions."""
    return re.findall(r"(\w+)\s*=\s*(?:pyqtSignal|Signal)\(", source)


def _extract_registry_calls(source: str) -> Tuple[List[str], List[str]]:
    """Return (reads, writes) list of registry method names called."""
    reads = re.findall(r"registry\.get_(\w+)", source)
    writes = re.findall(r"registry\.register_(\w+)", source)
    return reads, writes


def _extract_signal_connects(source: str) -> List[Tuple[str, str]]:
    """Return list of (signal_name, slot_or_method) from .connect() calls."""
    return re.findall(r"\.(\w+)\.connect\(\s*(?:self\.)?(\w+)", source)


def _extract_signal_emits(source: str) -> List[str]:
    """Return signal names that are .emit()'d."""
    return re.findall(r"(?:self\.)?\w*\.?(\w+)\.emit\(", source)


def _extract_variogram_params(source: str) -> List[str]:
    """Find references to key variogram parameter names."""
    keywords = [
        "nugget", "sill", "partial_sill", "total_sill", "range_major",
        "range_minor", "range_vertical", "range_", "azimuth", "dip",
        "plunge", "model_type", "contribution", "cone_tolerance",
        "lag_distance", "n_lags", "bandwidth", "sample_weights",
    ]
    found = []
    for kw in keywords:
        pattern = rf"\b{kw}\b"
        if re.search(pattern, source):
            found.append(kw)
    return found


# ---------------------------------------------------------------------------
# Panel inventory (mirrors panel_registration.py categories)
# ---------------------------------------------------------------------------

PANEL_CATEGORIES = {
    "PREPROCESSING": [
        ("DrillholeImportPanel",     "ui/drillhole_import_panel.py"),
        ("CompositingWindow",        "ui/compositing_window.py"),
        ("DeclusteringPanel",        "ui/declustering_panel.py"),
        ("GradeTransformationPanel", "ui/grade_transformation_panel.py"),
        ("StatisticsPanel",          "ui/statistics_panel.py"),
        ("DataViewerPanel",          "ui/data_viewer_panel.py"),
        ("BlockModelImportPanel",    "ui/block_model_import_panel.py"),
    ],
    "VARIOGRAM": [
        ("VariogramPanel",           "ui/variogram_panel.py"),
        ("VariogramAssistantPanel",  "ui/variogram_assistant_panel.py"),
    ],
    "ESTIMATION": [
        ("KrigingPanel",             "ui/kriging_panel.py"),
        ("SimpleKrigingPanel",       "ui/simple_kriging_panel.py"),
        ("UniversalKrigingPanel",    "ui/universal_kriging_panel.py"),
        ("CoKrigingPanel",           "ui/cokriging_panel.py"),
        ("IndicatorKrigingPanel",    "ui/indicator_kriging_panel.py"),
        ("SoftKrigingPanel",         "ui/soft_kriging_panel.py"),
        ("BayesianKrigingPanel",     "ui/bayesian_kriging_panel.py"),
        ("ARBFPanel",                "ui/arbf_panel.py"),
        ("FastRBFPanel",             "ui/fastrbf_panel.py"),
        ("RBFPanel",                 "ui/rbf_panel.py"),
    ],
    "SIMULATION": [
        ("SGSIMPanel",               "ui/sgsim_panel.py"),
        ("CoSGSIMPanel",             "ui/cosgsim_panel.py"),
        ("IKSGSIMPanel",             "ui/ik_sgsim_panel.py"),
        ("SISPanel",                 "ui/sis_panel.py"),
        ("GRFPanel",                 "ui/grf_panel.py"),
        ("MPSPanel",                 "ui/mps_panel.py"),
        ("TurningBandsPanel",        "ui/turning_bands_panel.py"),
        ("DBSPanel",                 "ui/dbs_panel.py"),
    ],
    "RESOURCE_CLASSIFICATION": [
        ("JORCClassificationPanel",  "ui/jorc_classification_panel.py"),
        ("ResourceReportingPanel",   "ui/resource_reporting_panel.py"),
        ("GradeTonnagePanel",        "ui/grade_tonnage_panel.py"),
        ("GradeTonnageBasicPanel",   "ui/grade_tonnage_basic_panel.py"),
    ],
    "VALIDATION": [
        ("SwathPanel",               "ui/swath_panel.py"),
        ("SwathAnalysis3DPanel",     "ui/swath_analysis_3d_panel.py"),
        ("QCWindow",                 "ui/qc_window.py"),
    ],
    "MINE_PLANNING": [
        ("PitOptimisationPanel",     "ui/pit_optimisation_panel.py"),
        ("NPVSPanel",                "ui/npvs_panel.py"),
        ("StrategicSchedulePanel",   "ui/strategic_schedule_panel.py"),
        ("TacticalSchedulePanel",    "ui/tactical_schedule_panel.py"),
        ("ShortTermSchedulePanel",   "ui/short_term_schedule_panel.py"),
        ("FleetPanel",               "ui/fleet_panel.py"),
        ("PushbackDesignerPanel",    "ui/pushback_designer_panel.py"),
        ("BenchDesignPanel",         "ui/bench_design_panel.py"),
        ("PlanningDashboardPanel",   "ui/planning_dashboard_panel.py"),
        ("ProductionDashboardPanel", "ui/production_dashboard_panel.py"),
        ("UndergroundPanel",         "ui/underground_panel.py"),
    ],
    "ANALYSIS": [
        ("ESGDashboardPanel",        "ui/esg_dashboard_panel.py"),
        ("CutoffOptimizationPanel",  "ui/cutoff_optimization_panel.py"),
        ("KMeansClusteringPanel",    "ui/kmeans_clustering_panel.py"),
        ("ChartsPanel",              "ui/charts_panel.py"),
        ("ReconciliationPanel",      "ui/reconciliation_panel.py"),
    ],
    "GEOLOGY": [
        ("GeologicalExplorerPanel",  "ui/geological_explorer_panel.py"),
        ("GeologicalModelPanel",     "ui/geological_model_panel.py"),
        ("LoopStructuralPanel",      "ui/loopstructural_panel.py"),
        ("FaultDefinitionPanel",     "ui/fault_definition_panel.py"),
        ("FoldDefinitionPanel",      "ui/fold_definition_panel.py"),
        ("VeinDefinitionPanel",      "ui/vein_definition_panel.py"),
        ("LithologyManagerPanel",    "ui/lithology_manager_panel.py"),
    ],
}

# Engines referenced by panels
ENGINE_FILES = {
    "compositing_engine":       "drillholes/compositing_engine.py",
    "compositing_utils":        "drillholes/compositing_utils.py",
    "declustering":             "drillholes/declustering.py",
    "data_io":                  "drillholes/data_io.py",
    "transform":                "models/transform.py",
    "variogram3d":              "models/variogram3d.py",
    "variogram_functions":      "models/variogram_functions.py",
    "variogram_model":          "geostats/variogram_model.py",
    "variogram_gates":          "geostats/variogram_gates.py",
    "variogram_assistant":      "geostats/variogram_assistant.py",
    "kriging3d":                "models/kriging3d.py",
    "kriging_engine":           "models/kriging_engine.py",
    "kriging_job_params":       "geostats/kriging_job_params.py",
    "universal_kriging":        "geostats/universal_kriging.py",
    "cokriging3d":              "geostats/cokriging3d.py",
    "indicator_kriging":        "geostats/indicator_kriging.py",
    "bayesian_kriging":         "geostats/bayesian_kriging.py",
    "sgsim3d":                  "models/sgsim3d.py",
    "sgsim_engine":             "models/sgsim_engine.py",
    "cosgsim3d":                "geostats/cosgsim3d.py",
    "ik_sgsim":                 "geostats/ik_sgsim.py",
    "sis":                      "geostats/sis.py",
    "grf":                      "geostats/grf.py",
    "mps":                      "geostats/mps.py",
    "turning_bands":            "geostats/turning_bands.py",
    "jorc_classification":      "models/jorc_classification_engine.py",
    "resource_reporting":       "models/resource_reporting_engine.py",
    "grade_tonnage":            "mine_planning/cutoff/geostats_grade_tonnage.py",
    "sk_cross_validation":      "geostats/sk_cross_validation.py",
    "sk_stationarity":          "geostats/sk_stationarity.py",
    "domain_mask":              "geostats/domain_mask.py",
    "block_model":              "models/block_model.py",
    "pit_optimizer":            "models/pit_optimizer.py",
}

CONTROLLER_FILES = {
    "app_controller":         "controllers/app_controller.py",
    "data_controller":        "controllers/data_controller.py",
    "geostats_controller":    "controllers/geostats_controller.py",
    "vis_controller":         "controllers/vis_controller.py",
    "mining_controller":      "controllers/mining_controller.py",
    "controller_signals":     "controllers/controller_signals.py",
    "job_registry":           "controllers/job_registry.py",
}

CORE_FILES = {
    "data_registry":          "core/data_registry.py",
    "ui_signals":             "ui/signals.py",
}


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

class DataFlowAnalyzer:
    """Scans the GeoX codebase and builds a data-flow model."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.panel_reports: Dict[str, PanelReport] = {}
        self.engine_params: Dict[str, List[str]] = {}  # engine_name -> variogram params found
        self.registry_signals: List[str] = []
        self.controller_signals: List[str] = []
        self.ui_signals: List[str] = []
        self.sill_conventions: List[Tuple[str, str]] = []  # (file, convention_description)
        self.issues: List[Tuple[str, str]] = []  # (location, description)

    # -----------------------------------------------------------------------
    # Phase 1: Scan panels
    # -----------------------------------------------------------------------
    def scan_panels(self):
        """Scan every registered panel file for data-flow artifacts."""
        for category, panels in PANEL_CATEGORIES.items():
            for panel_name, rel_path in panels:
                fp = PKG / rel_path
                if not fp.exists():
                    self.issues.append((rel_path, f"Panel file missing: {fp}"))
                    continue
                src = _read_source(fp)
                report = PanelReport(
                    panel_name=panel_name,
                    file_path=rel_path,
                    category=category,
                )
                reads, writes = _extract_registry_calls(src)
                report.registry_reads = reads
                report.registry_writes = writes

                connects = _extract_signal_connects(src)
                for sig, slot in connects:
                    report.signals_consumed.append(
                        SignalInfo(signal_name=sig, emitter="registry/controller",
                                  receiver=f"{panel_name}.{slot}")
                    )
                emits = _extract_signal_emits(src)
                for sig in emits:
                    report.signals_emitted.append(
                        SignalInfo(signal_name=sig, emitter=panel_name, receiver="?")
                    )

                vario_params = _extract_variogram_params(src)
                if vario_params:
                    report.notes.append(f"Variogram params referenced: {', '.join(vario_params)}")

                # Check for sill convention issues
                if "partial_sill" in src and "total_sill" in src:
                    report.notes.append("Uses BOTH partial_sill and total_sill -- verify convention")
                if re.search(r"sill\s*-\s*nugget", src):
                    report.notes.append("Computes partial = sill - nugget (total->partial conversion)")
                if re.search(r"nugget\s*\+\s*sill", src) or re.search(r"sill\s*\+\s*nugget", src):
                    report.notes.append("Computes total = nugget + sill (partial->total conversion)")

                # Detect gather_parameters / run_analysis pattern
                if "gather_parameters" in src:
                    report.notes.append("Implements gather_parameters() for job dispatch")
                if "run_analysis" in src:
                    report.notes.append("Implements run_analysis() entry point")
                if "_get_block_model" in src:
                    report.notes.append("Uses _get_block_model() (classified-first, raw fallback)")
                if "_build_unclassified_notice" in src:
                    report.notes.append("Has unclassified notice bar")

                self.panel_reports[panel_name] = report

    # -----------------------------------------------------------------------
    # Phase 2: Scan engines
    # -----------------------------------------------------------------------
    def scan_engines(self):
        """Scan engine files for variogram parameter usage and sill conventions."""
        for engine_name, rel_path in ENGINE_FILES.items():
            fp = PKG / rel_path
            if not fp.exists():
                self.issues.append((rel_path, f"Engine file missing: {fp}"))
                continue
            src = _read_source(fp)
            params = _extract_variogram_params(src)
            self.engine_params[engine_name] = params

            # Detect sill convention
            if re.search(r"partial_sill\s*=\s*sill\s*-\s*nugget", src):
                self.sill_conventions.append(
                    (rel_path, "TOTAL->PARTIAL: partial_sill = sill - nugget"))
            if re.search(r"sill\s*=\s*nugget\s*\+", src):
                self.sill_conventions.append(
                    (rel_path, "PARTIAL->TOTAL: sill = nugget + contribution"))
            # Check for model function signatures
            for match in re.finditer(
                r"def\s+(spherical|exponential|gaussian)_(?:model|variogram)\s*\(([^)]+)\)", src
            ):
                func_name = match.group(1)
                args = match.group(2)
                arg_list = [a.strip().split(":")[0].split("=")[0].strip()
                            for a in args.split(",")]
                order_str = " -> ".join(arg_list)
                self.sill_conventions.append(
                    (rel_path, f"{func_name} function signature: ({order_str})"))

    # -----------------------------------------------------------------------
    # Phase 3: Scan controllers & signals infrastructure
    # -----------------------------------------------------------------------
    def scan_controllers(self):
        """Scan controller files for signal definitions and job dispatch."""
        for name, rel_path in CONTROLLER_FILES.items():
            fp = PKG / rel_path
            if not fp.exists():
                continue
            src = _read_source(fp)
            signals = _extract_signal_definitions(src)
            if name == "controller_signals":
                self.controller_signals = signals
            if self.verbose:
                print(f"  [{name}] signals defined: {len(signals)}")

        for name, rel_path in CORE_FILES.items():
            fp = PKG / rel_path
            if not fp.exists():
                continue
            src = _read_source(fp)
            signals = _extract_signal_definitions(src)
            if name == "data_registry":
                self.registry_signals = signals
            elif name == "ui_signals":
                self.ui_signals = signals

    # -----------------------------------------------------------------------
    # Phase 5 (internal): Build dependency graph -- called during report generation
    # -----------------------------------------------------------------------
    def build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Return dict mapping panel_name -> set of upstream panel names."""
        # Heuristic: if panel A writes 'X_results' and panel B reads 'X_results',
        # then B depends on A.
        writers: Dict[str, str] = {}   # registry_key -> panel_name
        for pname, report in self.panel_reports.items():
            for w in report.registry_writes:
                writers[w] = pname
        deps: Dict[str, Set[str]] = defaultdict(set)
        for pname, report in self.panel_reports.items():
            for r in report.registry_reads:
                if r in writers and writers[r] != pname:
                    deps[pname].add(writers[r])
        return dict(deps)

    # -----------------------------------------------------------------------
    # Phase 4: Check for issues
    # -----------------------------------------------------------------------
    def check_issues(self):
        """Run consistency checks."""
        # 1. Panels that read variogram but don't read drillhole
        for pname, report in self.panel_reports.items():
            if "variogram_results" in report.registry_reads:
                if not any("drillhole" in r for r in report.registry_reads):
                    if report.category not in ("RESOURCE_CLASSIFICATION", "MINE_PLANNING"):
                        report.issues.append(
                            "Reads variogram_results but does NOT read drillhole data directly "
                            "(may rely on registry-cached data -- verify lineage)."
                        )

        # 2. Engines using legacy variogram functions
        for ename, params in self.engine_params.items():
            if "range_" in params and "sill" in params and "nugget" in params:
                pass  # normal
            if ename in ("variogram_functions",) and "nugget" in params:
                # Check if it's the reversed-signature legacy file
                fp = PKG / ENGINE_FILES.get(ename, "")
                if fp.exists():
                    src = _read_source(fp, 2000)
                    if re.search(r"def\s+\w+_variogram\s*\(\s*h\s*,\s*nugget", src):
                        self.issues.append((
                            ENGINE_FILES[ename],
                            "LEGACY reversed parameter order (h, nugget, sill, range_) -- "
                            "risk of silent mis-parameterisation if called from new code."
                        ))

        # 3. Sill convention mismatches
        total_files = set()
        partial_files = set()
        for fpath, desc in self.sill_conventions:
            if "TOTAL" in desc:
                total_files.add(fpath)
            if "PARTIAL" in desc:
                partial_files.add(fpath)
        both = total_files & partial_files
        if both:
            for f in both:
                self.issues.append((
                    f,
                    "File uses BOTH total-sill and partial-sill conventions -- "
                    "ensure correct conversion at every handoff."
                ))

    # -----------------------------------------------------------------------
    # Report generation
    # -----------------------------------------------------------------------
    def generate_report(self) -> str:
        """Produce the full text report."""
        lines: List[str] = []
        w = lines.append
        hr = lambda: w("=" * 90)
        hr2 = lambda: w("-" * 90)

        hr()
        w("GeoX DATA FLOW & PARAMETER ANALYSIS REPORT")
        w("Generated by data_flow_analyzer.py")
        hr()
        w("")

        # ===================================================================
        # SECTION 1: Executive Summary
        # ===================================================================
        w("1. EXECUTIVE SUMMARY")
        hr2()
        total_panels = sum(len(v) for v in PANEL_CATEGORIES.values())
        scanned = len(self.panel_reports)
        w(f"   Panels defined in inventory : {total_panels}")
        w(f"   Panels successfully scanned : {scanned}")
        w(f"   Engine modules scanned      : {len(self.engine_params)}")
        w(f"   Registry signals found      : {len(self.registry_signals)}")
        w(f"   Controller signals found    : {len(self.controller_signals)}")
        w(f"   UI signals found            : {len(self.ui_signals)}")
        w(f"   Sill-convention references  : {len(self.sill_conventions)}")
        w(f"   Issues / warnings detected  : {len(self.issues)}")
        w("")

        # ===================================================================
        # SECTION 2: End-to-End Resource Estimation Workflow
        # ===================================================================
        w("2. END-TO-END RESOURCE ESTIMATION WORKFLOW")
        hr2()
        workflow = [
            WorkflowStage(
                name="1. Drillhole Import",
                panel="DrillholeImportPanel",
                engine="data_io",
                input_keys=["CSV files (collars, surveys, assays, lithology)"],
                output_keys=["drillhole_data -> DataRegistry"],
                description=(
                    "Load raw CSV files. Desurvey trajectories to compute 3D coordinates "
                    "(X, Y, Z) for each assay interval. Auto-detect column aliases. "
                    "Normalize DIP convention (positive-down -> negative-down). "
                    "Emits: drillholeDataLoaded."
                ),
            ),
            WorkflowStage(
                name="2. Validation / QC",
                panel="QCWindow",
                engine="drillhole_validation",
                input_keys=["drillhole_data from registry"],
                output_keys=["validation_state (PASS/FAIL/WARN)"],
                description=(
                    "Check for gaps, overlaps, missing depths, negative intervals. "
                    "Gate for compositing -- if FAIL, compositing blocked unless override."
                ),
            ),
            WorkflowStage(
                name="3. Compositing",
                panel="CompositingWindow",
                engine="compositing_engine",
                input_keys=["drillhole_data", "validation_state"],
                output_keys=["composites -> drillhole_data['composites'] in registry"],
                description=(
                    "Group assay intervals into regular-length composites. Methods: "
                    "Fixed Length (most common), Equal Mass, Rolling Window, True Thickness. "
                    "Weighting: length-weighted (default) or density-weighted. "
                    "treat_null_as_zero=True ensures NaN grades counted as 0 (industry standard). "
                    "Hard/soft breaks at lithology or domain boundaries. "
                    "Partial composites: Discard / Merge / Keep / Auto. "
                    "Emits: drillholeDataLoaded (updated payload)."
                ),
            ),
            WorkflowStage(
                name="4. Declustering",
                panel="DeclusteringPanel",
                engine="declustering",
                input_keys=["composites (or raw assays) from registry"],
                output_keys=["declust_weight column", "declustering_results in registry"],
                description=(
                    "Compute per-sample weights to correct for preferential sampling. "
                    "Algorithm: cell-based (Deutsch 1989) with origin-offset optimisation. "
                    "Weight = 1 / (samples in cell).  Output: declust_weight column appended "
                    "to DataFrame. Also produces N_eff (effective sample count) and "
                    "weighted quantiles. Emits: declusteringResultsLoaded."
                ),
            ),
            WorkflowStage(
                name="5. Grade Transformation",
                panel="GradeTransformationPanel",
                engine="transform (NormalScoreTransformer)",
                input_keys=["composites/declustered data", "declust_weight (optional)"],
                output_keys=["transformed column (e.g. Au_ppm_NS)", "transformer object in registry"],
                description=(
                    "Normal-score transform maps grades to Gaussian space (required for SGSIM). "
                    "Uses weighted CDF if declustering weights available. "
                    "Deterministic via lexsort tie-breaking. "
                    "Back-transform via PCHIP interpolation (monotonic). "
                    "Other options: Log, Box-Cox, Square Root. "
                    "Lineage gate: warns if raw assays used instead of composites. "
                    "Emits: transformationMetadataLoaded."
                ),
            ),
            WorkflowStage(
                name="6. Experimental Variogram",
                panel="VariogramPanel",
                engine="variogram3d (Variogram3D)",
                input_keys=[
                    "composites/declustered data",
                    "sample_weights (external declustering preferred)",
                    "n_lags, lag_distance, lag_tolerance",
                    "major_azimuth, major_dip, cone_tolerance",
                ],
                output_keys=[
                    "experimental variograms (omni, downhole, major, minor, vertical)",
                    "fitted models per direction per model_type",
                    "VariogramModel dataclass -> registry",
                ],
                description=(
                    "Compute experimental semivariance in 5 directions:\n"
                    "  * Omnidirectional -- all pairs, no angular filter.\n"
                    "  * Downhole -- within-hole pairs (best nugget estimate).\n"
                    "  * Major -- along azimuth direction (cone tolerance filter).\n"
                    "  * Minor -- perpendicular to major (azimuth + 90 deg).\n"
                    "  * Vertical -- dip = 90 deg (vertical structure).\n\n"
                    "Fit theoretical model per direction via curve_fit with geostatistical "
                    "constraints: nugget floor (0.1 x first-lag gamma), sill cap (1.3 x omni sill), "
                    "range bounds. Supports nested structures (2+). "
                    "Data lineage: SHA-256 hash of source data stored with model."
                ),
            ),
            WorkflowStage(
                name="7. Estimation (Kriging / ARBF)",
                panel="KrigingPanel / ARBFPanel / etc.",
                engine="kriging3d / kriging_engine / universal_kriging / cokriging3d / ...",
                input_keys=[
                    "composites/declustered data",
                    "VariogramModel (range, sill, nugget, model_type, anisotropy)",
                    "grid definition (origin, spacing, counts)",
                    "search config (n_neighbors, max_distance)",
                ],
                output_keys=[
                    "estimates array (block grades)",
                    "variances array (kriging variance)",
                    "block model DataFrame -> registry",
                    "pyvista grid -> 3D visualization",
                ],
                description=(
                    "Ordinary Kriging (OK): solves gamma-based covariance system per block.\n"
                    "Simple Kriging (SK): adds known global mean assumption.\n"
                    "Universal Kriging (UK): adds polynomial drift (constant/linear/quadratic).\n"
                    "Co-Kriging: primary + secondary variable (Markov Model 1).\n"
                    "Indicator Kriging: indicator transforms per threshold -> CDF.\n"
                    "ARBF: Adaptive RBF with PUM dispatch for large datasets.\n\n"
                    "CRITICAL SILL CONVENTION:\n"
                    "  * OK kernel (kriging_engine.py): expects PARTIAL sill\n"
                    "  * UK kernel (universal_kriging.py): expects TOTAL sill\n"
                    "  * IK kernel (indicator_kriging.py): expects TOTAL sill\n"
                    "  Panels must convert correctly or estimates will be wrong.\n\n"
                    "Domain mask applied post-estimation: blocks outside informing volume "
                    "set to NaN (prevents classification of unestimated blocks)."
                ),
            ),
            WorkflowStage(
                name="8. Simulation (SGSIM / CoSGSIM / etc.)",
                panel="SGSIMPanel / CoSGSIMPanel / etc.",
                engine="sgsim3d / sgsim_engine / cosgsim3d / ...",
                input_keys=[
                    "normal-scored data",
                    "VariogramModel",
                    "grid definition",
                    "n_realizations, seed",
                ],
                output_keys=[
                    "realizations array (n_real, nz, ny, nx)",
                    "mean / variance grids",
                    "pyvista grid -> 3D visualization",
                ],
                description=(
                    "SGSIM: sequential path -> local kriging -> conditional draw.\n"
                    "CoSGSIM: multi-variable via Markov Model 1.\n"
                    "IK-SGSIM: sample CDF from IK probabilities.\n"
                    "SIS: sequential indicator simulation per threshold.\n"
                    "GRF: FFT or Cholesky unconditional field (optional SK conditioning).\n"
                    "MPS: training image patterns (no variogram needed).\n"
                    "Turning Bands: 1D processes superimposed in 3D.\n\n"
                    "All work in Gaussian space; back-transform after simulation."
                ),
            ),
            WorkflowStage(
                name="9. Cross-Validation & Stationarity",
                panel="SwathPanel / VariogramPanel",
                engine="sk_cross_validation / sk_stationarity",
                input_keys=["estimates", "composites", "variogram model"],
                output_keys=["CV metrics (ME, MAE, RMSE, R^2)", "stationarity status"],
                description=(
                    "Leave-one-out CV: remove sample -> estimate -> compare.\n"
                    "Swath analysis: compare block estimates vs composites in spatial bins.\n"
                    "Stationarity: domain-wise mean comparison, K-S test, spatial trends."
                ),
            ),
            WorkflowStage(
                name="10. JORC Classification",
                panel="JORCClassificationPanel",
                engine="jorc_classification_engine",
                input_keys=[
                    "block model (estimated or simulated)",
                    "drillhole data",
                    "VariogramModel (ranges for isotropic transform)",
                    "classification ruleset (distance %, min holes, KV/SoR gates)",
                ],
                output_keys=[
                    "CLASS_FINAL column (Measured / Indicated / Inferred / Unclassified)",
                    "DIST_REAL_1ST, N_HOLES_* columns",
                    "classified block model -> registry",
                ],
                description=(
                    "JORC/SAMREC-compliant resource classification.\n"
                    "Algorithm:\n"
                    "  1. Build IsotropicTransformer from variogram ranges + anisotropy.\n"
                    "  2. For each block, compute distance to nearest drillholes in\n"
                    "     isotropic (range-normalised) space.\n"
                    "  3. Count holes within Measured/Indicated/Inferred thresholds\n"
                    "     (e.g. 25%/75%/150% of variogram range).\n"
                    "  4. Apply optional gates:\n"
                    "     * KV gate: kriging variance > threshold -> downgrade.\n"
                    "     * SoR gate: slope-of-regression < threshold -> downgrade.\n"
                    "  5. Write CLASS_FINAL, CLASS_REASON, distances, hole counts.\n"
                    "Emits: blockModelClassified."
                ),
            ),
            WorkflowStage(
                name="11. Resource Reporting",
                panel="ResourceReportingPanel",
                engine="resource_reporting_engine",
                input_keys=[
                    "classified block model (CLASS_FINAL)",
                    "density (constant / domain / per-block)",
                    "volume (field / constant dx*dy*dz)",
                    "grade field + units",
                ],
                output_keys=[
                    "ResourceSummaryResult per category",
                    "tonnage, weighted grade, contained metal",
                    "M+I and All totals",
                ],
                description=(
                    "Mass-weighted resource statement by JORC category.\n"
                    "For each category:\n"
                    "  tonnage = SUM(volume * density)\n"
                    "  weighted_grade = SUM(grade * tonnage) / SUM(tonnage)\n"
                    "  contained_metal = tonnage * (grade / 100%)\n"
                    "JORC Audit Gate: independent validation of tonnage conservation.\n"
                    "Export: CSV / Excel with formatting."
                ),
            ),
            WorkflowStage(
                name="12. Grade-Tonnage Curves",
                panel="GradeTonnagePanel",
                engine="grade_tonnage (geostats_grade_tonnage)",
                input_keys=[
                    "block model or composites",
                    "cutoff range (min, max, step)",
                    "optional economic parameters (price, costs, recovery)",
                ],
                output_keys=[
                    "GT curve points (cutoff -> tonnage, avg grade, metal)",
                    "optimal cutoff grade",
                    "NPV / IRR at each cutoff",
                ],
                description=(
                    "Grade-tonnage curve: for each cutoff grade g, compute tonnage, "
                    "average grade, and contained metal above g. "
                    "Supports both block model mode (kriged estimates) and composite mode "
                    "(with declustering weights). "
                    "Economic overlay: net value, NPV, IRR at each cutoff."
                ),
            ),
        ]

        for stage in workflow:
            w(f"\n   {stage.name}")
            w(f"   Panel  : {stage.panel}")
            w(f"   Engine : {stage.engine}")
            w(f"   Inputs : {', '.join(stage.input_keys)}")
            w(f"   Outputs: {', '.join(stage.output_keys)}")
            w(f"   {stage.description}")
        w("")

        # ===================================================================
        # SECTION 3: Variogram Parameter Glossary
        # ===================================================================
        w("3. VARIOGRAM PARAMETER GLOSSARY")
        hr2()
        glossary = [
            ("nugget (C0)", "float >= 0", "m^2/%^2",
             "Discontinuity at lag h = 0.  Represents measurement error plus micro-scale "
             "variability below the sampling resolution.  Estimated from downhole variogram "
             "(short-distance pairs within the same borehole).  A nugget floor of 0.1 x gamma(first lag) "
             "is enforced to prevent zero-lock during optimisation."),

            ("sill (total, C0 + C1)", "float > nugget", "m^2/%^2",
             "Plateau value of the variogram -- the total variance captured by the model. "
             "Equals nugget + sum of all structure contributions.  Estimated from the mean "
             "of the last third of the experimental variogram.  A sill cap of 1.3 x omni_sill "
             "is applied to directional fits to prevent wild overshoot.  THIS IS THE VALUE "
             "PASSED TO KRIGING ENGINES (except kriging_engine.py OK kernel which expects "
             "PARTIAL sill)."),

            ("partial_sill (C1, contribution)", "float > 0", "m^2/%^2",
             "Sill minus nugget for a single-structure model.  For nested models, each "
             "structure has its own 'contribution' (partial sill).  total_sill = C0 + SUM(Ci).  "
             "NEVER pass partial_sill where total_sill is expected -- this is the #1 "
             "source of estimation bugs (flat estimates, R^2 ~ 0)."),

            ("range_major (a_major)", "float > 0", "metres",
             "Practical range in the direction of maximum continuity (the major axis of "
             "the search ellipsoid).  At this distance the model reaches ~95% of the sill.  "
             "Azimuth and dip define the orientation of this axis.  Used directly for "
             "anisotropic distance transforms in kriging and JORC classification."),

            ("range_minor (a_minor)", "float > 0", "metres",
             "Practical range perpendicular to major in the horizontal plane.  Often "
             "smaller than range_major for deposits with directional continuity.  Defaults "
             "to range_major if not fitted separately."),

            ("range_vertical (a_vert)", "float > 0", "metres",
             "Practical range in the vertical direction.  Typically the shortest range "
             "in stratiform deposits.  Estimated from the vertical variogram.  Controls "
             "vertical smoothing in kriging."),

            ("azimuth", "float [0, 360)", "degrees",
             "Clockwise angle from North defining the direction of range_major.  "
             "Mining convention: 0 deg=N, 90 deg=E, 180 deg=S, 270 deg=W.  Used to build the "
             "3x3 rotation matrix for anisotropic distance calculation."),

            ("dip", "float [-90, 90]", "degrees",
             "Downward angle from horizontal defining the plunge of the major axis.  "
             "GeoX convention: negative = downward (mining standard).  Combined with "
             "azimuth to orient the search ellipsoid."),

            ("model_type", "str", "--",
             "Theoretical model function: 'spherical' (most common in mining, finite range), "
             "'exponential' (asymptotic approach to sill), 'gaussian' (very smooth, parabolic "
             "near origin -- use with caution as it causes numerical instability in kriging).  "
             "Selected by best-fit to experimental variogram."),

            ("n_lags", "int > 0", "--",
             "Number of lag bins for the experimental variogram.  Default 12.  "
             "Auto-lags mode calculates optimal lags per direction based on drill spacing "
             "and composite length."),

            ("lag_distance", "float > 0", "metres",
             "Spacing between lag bins.  Default 25 m.  Should be approximately half the "
             "average drill spacing for horizontal variograms.  For downhole: equals "
             "composite length."),

            ("cone_tolerance", "float [5, 45]", "degrees",
             "Angular half-width of the directional search cone.  Pairs within this angle "
             "of the target direction are included.  Smaller = purer direction but fewer "
             "pairs.  Default 15 deg (sometimes 22.5 deg)."),

            ("bandwidth", "float > 0", "metres",
             "Maximum perpendicular distance from the search plane for directional "
             "variograms.  Limits off-plane pairs.  Optional -- only used when very "
             "precise directional control is needed."),

            ("sample_weights", "array (N,)", "--",
             "Per-sample declustering weights from the declustering engine.  Preferred "
             "over internal cell-based weights.  Applied as pair weights: "
             "gamma_weighted = 0.5 * SUM(wi*wj*(Zi-Zj)^2) / SUM(wi*wj)."),
        ]
        for name, dtype, unit, desc in glossary:
            w(f"\n   {name}")
            w(f"     Type : {dtype}   Unit : {unit}")
            w(f"     {desc}")
        w("")

        # ===================================================================
        # SECTION 4: Sill Convention Audit
        # ===================================================================
        w("4. SILL CONVENTION AUDIT")
        hr2()
        w("   The total-sill vs partial-sill convention is the most error-prone")
        w("   aspect of the variogram -> kriging handoff.  This audit tracks which")
        w("   convention each engine expects.\n")

        conv_table = [
            ("kriging_engine.py (OK kernel)", "PARTIAL sill (subtract nugget before passing)"),
            ("universal_kriging.py (UK kernel)", "TOTAL sill"),
            ("indicator_kriging.py (IK kernel)", "TOTAL sill"),
            ("bayesian_kriging.py", "TOTAL sill"),
            ("cokriging3d.py", "TOTAL sill"),
            ("sgsim_engine.py (SGSIM kernel)", "TOTAL sill"),
            ("variogram_functions.py (fit output)", "TOTAL sill = nugget + partial"),
            ("variogram_model.py (VariogramModel.total_sill)", "TOTAL sill property"),
            ("variogram_model.py (VariogramStructure.contribution)", "PARTIAL sill per structure"),
            ("utils/variogram_functions.py (LEGACY)", "REVERSED signature: (h, nugget, sill, range_)"),
        ]
        w(f"   {'Engine / File':<50} {'Convention'}")
        w(f"   {'-'*50} {'-'*40}")
        for eng, conv in conv_table:
            w(f"   {eng:<50} {conv}")

        if self.sill_conventions:
            w(f"\n   Additional conventions detected by scanner:")
            for fpath, desc in self.sill_conventions:
                w(f"     {fpath}: {desc}")
        w("")

        # ===================================================================
        # SECTION 5: Per-Panel Data Flow Summaries
        # ===================================================================
        w("5. PER-PANEL DATA FLOW SUMMARIES")
        hr2()
        for category, panels in PANEL_CATEGORIES.items():
            w(f"\n   -- {category} {'-' * (70 - len(category))}")
            for panel_name, rel_path in panels:
                report = self.panel_reports.get(panel_name)
                if report is None:
                    w(f"\n   {panel_name} ({rel_path})")
                    w(f"     [!] File not found / not scanned")
                    continue
                w(f"\n   {panel_name} ({rel_path})")
                if report.registry_reads:
                    w(f"     Registry READS  : {', '.join(sorted(set(report.registry_reads)))}")
                if report.registry_writes:
                    w(f"     Registry WRITES : {', '.join(sorted(set(report.registry_writes)))}")
                sig_consumed = sorted(set(s.signal_name for s in report.signals_consumed))
                sig_emitted = sorted(set(s.signal_name for s in report.signals_emitted))
                if sig_consumed:
                    w(f"     Signals IN      : {', '.join(sig_consumed[:15])}")
                if sig_emitted:
                    w(f"     Signals OUT     : {', '.join(sig_emitted[:15])}")
                for note in report.notes:
                    w(f"     * {note}")
                for issue in report.issues:
                    w(f"     [!] {issue}")
        w("")

        # ===================================================================
        # SECTION 6: Engine Variogram Parameter Coverage
        # ===================================================================
        w("6. ENGINE VARIOGRAM PARAMETER COVERAGE")
        hr2()
        all_params = sorted({p for ps in self.engine_params.values() for p in ps})
        header = f"   {'Engine':<30}" + "".join(f" {p[:8]:>8}" for p in all_params)
        w(header)
        w(f"   {'-'*30}" + "-"*9*len(all_params))
        for ename in sorted(self.engine_params):
            params = self.engine_params[ename]
            row = f"   {ename:<30}"
            for p in all_params:
                row += f" {'  Y':>8}" if p in params else f" {'  .':>8}"
            w(row)
        w("")

        # ===================================================================
        # SECTION 7: Registry Signal Inventory
        # ===================================================================
        w("7. REGISTRY & CONTROLLER SIGNAL INVENTORY")
        hr2()
        w("   Registry signals (DataRegistry):")
        for s in sorted(self.registry_signals):
            w(f"     * {s}")
        w(f"\n   Controller signals (ControllerSignals):")
        for s in sorted(self.controller_signals):
            w(f"     * {s}")
        w(f"\n   UI signals (UISignals):")
        for s in sorted(self.ui_signals):
            w(f"     * {s}")
        w("")

        # ===================================================================
        # SECTION 8: Dependency Graph
        # ===================================================================
        w("8. PANEL DEPENDENCY GRAPH (via registry reads/writes)")
        hr2()
        deps = self.build_dependency_graph()
        if deps:
            for panel, upstreams in sorted(deps.items()):
                w(f"   {panel}")
                for u in sorted(upstreams):
                    w(f"     <- depends on: {u}")
        else:
            w("   (no cross-panel registry dependencies detected -- "
              "panels may use controller dispatch instead)")
        w("")

        # ===================================================================
        # SECTION 9: Issues & Inconsistencies
        # ===================================================================
        w("9. ISSUES & INCONSISTENCIES")
        hr2()
        if not self.issues:
            w("   No issues detected.")
        for loc, desc in self.issues:
            w(f"   [{loc}]")
            w(f"     {desc}")
            w("")

        # Additional per-panel issues
        panel_issues = [(pn, iss) for pn, r in self.panel_reports.items()
                        for iss in r.issues]
        if panel_issues:
            w("   Per-panel issues:")
            for pn, iss in panel_issues:
                w(f"   [{pn}] {iss}")
        w("")

        # ===================================================================
        # SECTION 10: Data Flow Diagram (ASCII)
        # ===================================================================
        w("10. DATA FLOW DIAGRAM (ASCII)")
        hr2()
        w(r"""
   +-----------------------------------------------------------------------+
   |                       RAW CSV FILES                                   |
   |  (collars.csv, surveys.csv, assays.csv, lithology.csv)               |
   +------------------------------+----------------------------------------+
                                  |
                                  v
   +----------------------------------------------------------------------+
   |  1. DRILLHOLE IMPORT  (DrillholeImportPanel -> data_io.py)           |
   |     * Desurvey -> (X, Y, Z)     * Column alias detection             |
   |     * DIP normalisation          * File checksums (audit)             |
   +------------------------------+---------------------------------------+
                                  |  drillholeDataLoaded signal
                                  v
   +----------------------------------------------------------------------+
   |  2. QC / VALIDATION  (QCWindow -> drillhole_validation)              |
   |     * Gap/overlap checks         * Missing depth detection            |
   |     * Validation gate for compositing                                 |
   +------------------------------+---------------------------------------+
                                  |
                                  v
   +----------------------------------------------------------------------+
   |  3. COMPOSITING  (CompositingWindow -> compositing_engine.py)        |
   |     * Fixed length / equal mass / rolling window / true thickness     |
   |     * Length- or density-weighted averaging                           |
   |     * Hard/soft breaks at lithology boundaries                        |
   |     * treat_null_as_zero = True (industry standard)                  |
   +------------------------------+---------------------------------------+
                                  |  drillholeDataLoaded (updated)
                    +-------------+-------------+
                    v                           v
   +------------------------+ +--------------------------------------------+
   | 4. DECLUSTERING        | | 5. GRADE TRANSFORMATION                    |
   | (DeclusteringPanel)    | | (GradeTransformationPanel)                 |
   | * Cell-based weights   | | * Normal-score (for SGSIM)                 |
   | * Origin-offset optim  | | * Log / Box-Cox / Sqrt                     |
   | * Deutsch 1989         | | * Weighted CDF if declust weights avail    |
   | Output: declust_weight | | Output: *_NS column + transformer obj      |
   +-----------+------------+ +--------------------+-----------------------+
               |                                   |
               +--------------+--------------------+
                              v
   +----------------------------------------------------------------------+
   |  6. VARIOGRAM ANALYSIS  (VariogramPanel -> variogram3d.py)           |
   |     * Omni / downhole / major / minor / vertical                      |
   |     * Fit: nugget floor, sill cap, range bounds                       |
   |     * Nested structures (2+ contributions)                            |
   |     * Data lineage: SHA-256 hash + fit timestamp                      |
   |     Output: VariogramModel -> registry                                |
   +--------------+-------------------------+-----------------------------+
                  |                         |
        +---------+                         +-----------+
        v                                               v
   +--------------------------+    +--------------------------------------+
   | 7. ESTIMATION            |    | 8. SIMULATION                        |
   | * OK / SK / UK / CoK     |    | * SGSIM / CoSGSIM                    |
   | * IK / Bayesian / ARBF   |    | * IK-SGSIM / SIS / GRF              |
   | Uses: variogram params,  |    | * MPS / Turning Bands / DBS          |
   |   grid, search config    |    | Uses: variogram, grid, n_realizations|
   | Output: estimates +      |    | Output: realizations array,          |
   |   variances -> registry  |    |   mean/variance grids -> registry    |
   +--------------+-----------+    +--------------+-----------------------+
                  |                                |
                  +-----------+--------------------+
                              v
   +----------------------------------------------------------------------+
   |  9. CROSS-VALIDATION & STATIONARITY                                   |
   |     * LOO-CV: ME, MAE, RMSE, R^2                                     |
   |     * Swath analysis: spatial bias detection                          |
   |     * Stationarity: K-S test, domain means                            |
   +------------------------------+---------------------------------------+
                                  v
   +----------------------------------------------------------------------+
   | 10. JORC CLASSIFICATION  (JORCClassificationPanel)                    |
   |     * IsotropicTransformer from variogram ranges                      |
   |     * Distance to nearest drillholes (range-normalised)               |
   |     * Threshold rules: Measured < 25%, Indicated < 75%, etc.          |
   |     * Optional KV & SoR gates                                         |
   |     Output: CLASS_FINAL column -> registry                            |
   +------------------------------+---------------------------------------+
                                  v
   +----------------------------------------------------------------------+
   | 11. RESOURCE REPORTING  (ResourceReportingPanel)                      |
   |     * Mass-weighted stats per JORC category                           |
   |     * tonnage = SUM(vol * density), grade = SUM(g*t) / SUM(t)        |
   |     * Contained metal, M+I totals, All totals                         |
   |     * JORC audit gate: independent tonnage conservation check         |
   +------------------------------+---------------------------------------+
                                  v
   +----------------------------------------------------------------------+
   | 12. GRADE-TONNAGE CURVES  (GradeTonnagePanel)                        |
   |     * Cutoff sweep: tonnage, avg grade, metal at each cutoff          |
   |     * Economic overlay: NPV, IRR per cutoff                           |
   |     * Optimal cutoff determination                                    |
   +----------------------------------------------------------------------+

   ======================================================================
   REGISTRY SIGNAL BUS (DataRegistry singleton)
   ======================================================================
     drillholeDataLoaded -> [all preprocessing + estimation panels]
     compositesLoaded -> [variogram, estimation, simulation panels]
     declusteringResultsLoaded -> [variogram, transform panels]
     variogramResultsLoaded -> [all estimation + simulation panels]
     blockModelLoaded / blockModelGenerated -> [classification, reporting]
     blockModelClassified -> [reporting, grade-tonnage, planning panels]
     sgsimResultsLoaded -> [classification, uncertainty, planning panels]

   ======================================================================
   CONTROLLER DISPATCH (AppController -> JobRegistry -> JobWorker)
   ======================================================================
     Panel.run_analysis() -> controller.run_task(task_name, params, callback)
       -> JobRegistry.get(task_name) -> SubController._prepare_payload()
       -> JobWorker(func, params).start() [new QThread]
       -> worker.finished -> controller._on_task_complete -> callback(result)
       -> vis_controller.apply_results_to_model(result) -> renderer + legend
""")
        w("")

        # ===================================================================
        # SECTION 11: Variogram Calculation Deep-Dive
        # ===================================================================
        w("11. VARIOGRAM CALCULATION DEEP-DIVE")
        hr2()
        w(r"""
   EXPERIMENTAL VARIOGRAM COMPUTATION
   -----------------------------------
   gamma_hat(h) = (1 / 2N(h)) * SUM [Z(xi) - Z(xi + h)]^2

   Where:
     h     = lag distance (bin centre)
     N(h)  = number of pairs in the lag bin
     Z(x)  = grade value at location x

   With declustering weights:
     gamma_hat_w(h) = SUM wi*wj*[Z(xi) - Z(xj)]^2 / (2 * SUM wi*wj)

   DIRECTIONAL FILTERING
   ---------------------
   For direction (azimuth, dip):
     Include pair (i, j) if:
       cos(angle(xj-xi, direction)) >= cos(cone_tolerance)
     AND (if bandwidth specified):
       perpendicular_distance(xj-xi, plane) <= bandwidth

   THEORETICAL MODEL FITTING
   -------------------------
   Spherical:     gamma(h) = C0 + C1[1.5(h/a) - 0.5(h/a)^3]  for h <= a
                  gamma(h) = C0 + C1                            for h > a

   Exponential:   gamma(h) = C0 + C1[1 - exp(-3h/a)]

   Gaussian:      gamma(h) = C0 + C1[1 - exp(-3(h/a)^2)]

   Where:
     C0 = nugget          (fitted, floor = 0.1 * gamma(first lag))
     C1 = partial sill    (fitted, total = C0 + C1)
     a  = practical range  (fitted, 80% sill point initial guess)

   NESTED MODEL (2 structures):
     gamma(h) = C0 + C1*g1(h/a1) + C2*g2(h/a2)
     where a1 < a2 (short-range + long-range)

   FITTING ALGORITHM (4 phases):
     Phase 1: Nugget estimation    -- extrapolation to h=0 + floor
     Phase 2: Sill estimation      -- plateau + cap (1.3x omni)
     Phase 3: Range estimation     -- 80% sill crossover point
     Phase 4: scipy.curve_fit      -- bounded optimisation
              Fallback: grid search if curve_fit fails

   ISOTROPIC TRANSFORM (for JORC classification):
     d_iso = ||R * diag(1/a_major, 1/a_minor, 1/a_vert) * (x_block - x_drillhole)||
     where R = rotation matrix from (azimuth, dip)
     d_iso = 1.0 means "one full variogram range away"
""")
        w("")

        # ===================================================================
        # SECTION 12: Known Issues & Recommendations
        # ===================================================================
        w("12. KNOWN ISSUES & RECOMMENDATIONS")
        hr2()
        known = [
            ("SILL-01", "Sill convention mismatch",
             "OK kernel expects PARTIAL sill; UK/IK/SGSIM kernels expect TOTAL sill.  "
             "Panels must convert correctly.  Previous bug: ARBF sill 22x too high after "
             "normal-score transform (ratio-preserving rescale fix in engine.py:356-378)."),
            ("SILL-02", "Sill import total vs partial",
             "_import_variogram() in arbf_panel.py loaded total_sill into partial_sill field. "
             "Fix: subtract nugget before storing as contribution."),
            ("NUG-01", "Nugget = 0 lock",
             "curve_fit lower bound = 0.0 allowed optimizer to lock nugget at zero.  "
             "Fix: lower bound = 1e-6 + nugget floor = 0.1 x first_lag_gamma (3 files)."),
            ("LEGACY-01", "Legacy variogram function signatures",
             "utils/variogram_functions.py uses REVERSED parameter order (h, nugget, sill, range_) "
             "vs canonical (h, range_, sill, nugget).  Risk of silent mis-parameterisation.  "
             "Recommendation: deprecate and redirect to geostats/variogram_model.py."),
            ("LINEAGE-01", "Data lineage enforcement",
             "Variogram model stores SHA-256 hash of source data.  Kriging should verify hash "
             "matches current data before estimation (variogram_gates.validate_pre_kriging).  "
             "Not all panels enforce this gate -- audit and enable globally."),
            ("COORD-01", "Coordinate system mismatch",
             "SGSIM models may already be in local coords; applying global_shift again -> double-shift "
             "-> invisible blocks.  Fix: magnitude guard in block_model_renderer.py and scene_bounds.py."),
            ("DOMAIN-01", "Domain mask application",
             "Post-estimation domain mask sets NaN for blocks outside informing volume.  "
             "Ensure all panels apply mask before JORC classification."),
            ("CV-01", "LOO-CV limitation",
             "ARBF LOO-CV uses global RBF, not PUM.  Multi-domain deposits need domain-separated "
             "estimation or hybrid variogram mode.  Trust swath plots over CV R^2."),
            ("WEAK-01", "Weak direction handling",
             "Directions with <30 pairs/lag trigger warnings.  Omni sill inherited for weak "
             "directions.  Panels should display warning prominently."),
        ]
        for code, title, desc in known:
            w(f"   {code}: {title}")
            w(f"     {desc}")
            w("")
        w("")

        hr()
        w("END OF REPORT")
        hr()

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GeoX Data Flow & Parameter Analyzer"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write report to file (default: print to stdout)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print progress messages"
    )
    args = parser.parse_args()

    analyzer = DataFlowAnalyzer(verbose=args.verbose)

    if args.verbose:
        print("Phase 1: Scanning panels...")
    analyzer.scan_panels()

    if args.verbose:
        print("Phase 2: Scanning engines...")
    analyzer.scan_engines()

    if args.verbose:
        print("Phase 3: Scanning controllers & signals...")
    analyzer.scan_controllers()

    if args.verbose:
        print("Phase 4: Checking issues...")
    analyzer.check_issues()

    if args.verbose:
        print("Phase 5: Building dependency graph + generating report...")
    report = analyzer.generate_report()

    if args.output:
        out = Path(args.output)
        out.write_text(report, encoding="utf-8")
        print(f"Report written to {out}")
    else:
        # Use utf-8 for stdout on Windows to avoid cp1252 encoding errors
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
