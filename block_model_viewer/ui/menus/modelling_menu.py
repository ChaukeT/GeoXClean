"""
Modelling menu — GeoX consolidated menu system.

Position: 5th.
Handles: geological modelling, geotechnical (with slope analysis),
         variograms, estimation, simulations, machine learning.
Workflow: Geological Model → Variogram → Estimate → Simulate.
"""

from typing import TYPE_CHECKING
from PyQt6.QtWidgets import QMenuBar, QMenu
from PyQt6.QtGui import QAction

if TYPE_CHECKING:
    from ..main_window import MainWindow

try:
    from ...assets.icons.icon_loader import get_menu_icon
except ImportError:
    def get_menu_icon(category, name):
        return None


def build_modelling_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    menu = menubar.addMenu("&Modelling")

    # ═══════════════════════════════════════════════════════════════
    # GEOLOGICAL MODELLING
    # ═══════════════════════════════════════════════════════════════
    geo_menu = menu.addMenu("&Geological Modelling")

    act = QAction(get_menu_icon("geology", "loopstructural"), "&LoopStructural Modeler...", main_window)
    act.setStatusTip("Industry-grade geological modeling with JORC/SAMREC compliance")
    act.triggered.connect(main_window.open_loopstructural_panel)
    geo_menu.addAction(act)

    act = QAction(get_menu_icon("geology", "explorer"), "Geological &Explorer", main_window)
    act.setStatusTip("Geological model visualization and exploration")
    act.triggered.connect(main_window.open_geological_explorer_panel)
    geo_menu.addAction(act)

    menu.addSeparator()

    # ═══════════════════════════════════════════════════════════════
    # GEOTECHNICAL — grouped with slope analysis
    # ═══════════════════════════════════════════════════════════════
    geotech_menu = menu.addMenu("Geo&technical")

    act = QAction(get_menu_icon("geotech", "geotech"), "&Geotechnical Dashboard", main_window)
    act.setStatusTip("Rock-mass property interpolation and geotechnical analysis")
    act.triggered.connect(main_window.open_geotech_panel)
    geotech_menu.addAction(act)

    geotech_menu.addSeparator()

    act = QAction(get_menu_icon("geotech", "slope"), "Slope &Risk Assessment", main_window)
    act.setStatusTip("Evaluate slope failure risk using probabilistic methods")
    act.triggered.connect(main_window.open_slope_risk_panel)
    geotech_menu.addAction(act)

    act = QAction(get_menu_icon("geotech", "slope"), "Slope &Stability Analysis", main_window)
    act.setStatusTip("Factor of safety, kinematic, and limit equilibrium analysis")
    act.triggered.connect(main_window.open_slope_stability_panel)
    geotech_menu.addAction(act)

    geotech_menu.addSeparator()

    act = QAction(get_menu_icon("geotech", "geotech_summary"), "Geotech S&ummary", main_window)
    act.setStatusTip("Rock-mass parameter summary (RMR, Q, GSI)")
    if hasattr(main_window, 'open_geotech_summary_panel'):
        act.triggered.connect(main_window.open_geotech_summary_panel)
    else:
        act.setEnabled(False)
    geotech_menu.addAction(act)

    menu.addSeparator()

    # ═══════════════════════════════════════════════════════════════
    # VARIOGRAM TOOLS
    # ═══════════════════════════════════════════════════════════════
    vario_menu = menu.addMenu("3D &Variogram Tools")

    act = QAction(get_menu_icon("estimations", "variogram"), "&Compute 3D Variogram", main_window)
    act.setStatusTip("Calculate experimental and modelled 3D variograms from drillhole data")
    act.triggered.connect(main_window.open_variogram_analysis)
    vario_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "variogram_cloud"), "&Show 3D Variogram Cloud", main_window)
    act.setStatusTip("Display variogram cloud (distance vs semivariance pairs)")
    act.triggered.connect(main_window.show_variogram_cloud)
    vario_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "variogram_fit"), "&Fit Variogram Model", main_window)
    act.setStatusTip("Fit theoretical variogram models (spherical, exponential, gaussian)")
    act.triggered.connect(main_window.fit_variogram_model)
    vario_menu.addAction(act)

    vario_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "variogram_assistant"), "Modelling &Assistant", main_window)
    act.setStatusTip("Semi-automatic variogram fitting with model selection and cross-validation")
    act.triggered.connect(main_window.open_variogram_assistant_panel)
    vario_menu.addAction(act)

    vario_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "variogram_export"), "&Export Variogram Table (CSV)", main_window)
    act.setStatusTip("Export variogram tables to CSV files")
    act.triggered.connect(main_window.export_variogram_tables)
    vario_menu.addAction(act)

    menu.addSeparator()

    # ═══════════════════════════════════════════════════════════════
    # ESTIMATION
    # ═══════════════════════════════════════════════════════════════
    est_menu = menu.addMenu("&Estimation")

    act = QAction(get_menu_icon("estimations", "kriging"), "&Ordinary Kriging (3D)", main_window)
    act.setStatusTip("Perform 3D Ordinary Kriging grade estimation")
    act.triggered.connect(main_window.open_kriging_panel)
    est_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "kriging"), "&Simple Kriging (3D)", main_window)
    act.setStatusTip("Perform 3D Simple Kriging with known global mean")
    act.triggered.connect(main_window.open_simple_kriging)
    est_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "kriging"), "&Universal Kriging", main_window)
    act.setStatusTip("Universal Kriging with configurable drift")
    act.triggered.connect(main_window.open_universal_kriging_panel)
    est_menu.addAction(act)

    est_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "cokriging"), "&Co-Kriging", main_window)
    act.setStatusTip("Co-Kriging using primary and secondary variables")
    act.triggered.connect(main_window.open_cokriging_panel)
    est_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "indicator_kriging"), "&Indicator Kriging", main_window)
    act.setStatusTip("Non-parametric estimation and CDF construction")
    act.triggered.connect(main_window.open_indicator_kriging_panel)
    est_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "bayesian"), "&Bayesian / Soft Kriging", main_window)
    act.setStatusTip("Bayesian kriging with soft data priors")
    act.triggered.connect(main_window.open_soft_kriging_panel)
    est_menu.addAction(act)

    est_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "rbf"), "&RBF Interpolation", main_window)
    act.setStatusTip("Radial Basis Function interpolation with anisotropy and GPU")
    act.triggered.connect(main_window.open_rbf_panel)
    est_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "rbf"), "&FastRBF Interpolation", main_window)
    act.setStatusTip("Leapfrog-style RBF interpolation with JORC classification")
    act.triggered.connect(main_window.open_fastrbf_panel)
    est_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "rbf"), "&ARBF Estimation", main_window)
    act.setStatusTip("Adaptive RBF with GPR posterior variance, PUM, LVA, and JORC audit")
    act.triggered.connect(main_window.open_arbf_panel)
    est_menu.addAction(act)

    est_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "uncertainty"), "Uncertainty &Propagation", main_window)
    act.setStatusTip("Propagate grade realisations through economic models")
    act.triggered.connect(main_window.open_uncertainty_propagation_panel)
    est_menu.addAction(act)

    menu.addSeparator()

    # ═══════════════════════════════════════════════════════════════
    # SIMULATIONS
    # ═══════════════════════════════════════════════════════════════
    sim_menu = menu.addMenu("&Simulations")

    act = QAction(get_menu_icon("estimations", "sgsim"), "&SGSIM (Sequential Gaussian)", main_window)
    act.setStatusTip("Sequential Gaussian Simulation for uncertainty quantification")
    act.triggered.connect(main_window.open_sgsim_panel)
    sim_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "indicator_kriging"), "&IK-SGSIM (Indicator-based)", main_window)
    act.setStatusTip("Indicator Kriging based Sequential Gaussian Simulation")
    act.triggered.connect(main_window.open_ik_sgsim_panel)
    sim_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "cokriging"), "&CoSGSIM (Co-Simulation)", main_window)
    act.setStatusTip("Sequential Gaussian Co-Simulation for correlated variables")
    act.triggered.connect(main_window.open_cosgsim_panel)
    sim_menu.addAction(act)

    sim_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "sis"), "S&IS (Sequential Indicator)", main_window)
    act.setStatusTip("Category-based simulation for lithology, ore/waste")
    act.triggered.connect(main_window.open_sis_panel)
    sim_menu.addAction(act)

    sim_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "turning_bands"), "&Turning Bands", main_window)
    act.setStatusTip("Fast Gaussian field simulation for large domains")
    act.triggered.connect(main_window.open_turning_bands_panel)
    sim_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "grf"), "&Gaussian Random Fields (GRF)", main_window)
    act.setStatusTip("Fast unconditional/conditional Gaussian fields")
    act.triggered.connect(main_window.open_grf_panel)
    sim_menu.addAction(act)

    sim_menu.addSeparator()

    act = QAction(get_menu_icon("estimations", "dbs"), "&Direct Block Simulation (DBS)", main_window)
    act.setStatusTip("Simulation directly at block support")
    act.triggered.connect(main_window.open_dbs_panel)
    sim_menu.addAction(act)

    act = QAction(get_menu_icon("estimations", "mps"), "&Multiple-Point Simulation (MPS)", main_window)
    act.setStatusTip("Pattern-based simulation using training images")
    act.triggered.connect(main_window.open_mps_panel)
    sim_menu.addAction(act)

    menu.addSeparator()

    # ═══════════════════════════════════════════════════════════════
    # MACHINE LEARNING
    # ═══════════════════════════════════════════════════════════════
    ml_menu = menu.addMenu("Machine &Learning")

    act = QAction(get_menu_icon("machine_learning", "kmeans"), "&K-Means Clustering", main_window)
    act.setStatusTip("Unsupervised K-Means clustering for domain classification")
    act.triggered.connect(main_window.open_kmeans_panel)
    ml_menu.addAction(act)

    menu.addSeparator()

    # ═══════════════════════════════════════════════════════════════
    # GEOMETALLURGY
    # ═══════════════════════════════════════════════════════════════
    geomet_menu = menu.addMenu("Geometallur&gy")

    for label, method, tip in [
        ("&Geomet Dashboard", 'open_geomet_panel',
         "Domain, plant response, and recovery modelling"),
        ("Geomet &Domains", 'open_geomet_domain_panel',
         "Cluster drillholes into geometallurgical domains"),
        ("Geomet &Plant Model", 'open_geomet_plant_panel',
         "Plant recovery and throughput modelling"),
        ("Value &Chain", 'open_geomet_chain_panel',
         "End-to-end mine-to-mill geometallurgy chain"),
    ]:
        act = QAction(get_menu_icon("geology", "geomet"), label, main_window)
        act.setStatusTip(tip)
        if hasattr(main_window, method):
            act.triggered.connect(getattr(main_window, method))
        else:
            act.setEnabled(False)
        geomet_menu.addAction(act)

    menu.addSeparator()

    # ═══════════════════════════════════════════════════════════════
    # FRAGMENTATION
    # ═══════════════════════════════════════════════════════════════
    frag_menu = menu.addMenu("&Fragmentation")

    for label, method, tip in [
        ("&Import Blast Images...", 'open_frag_import_panel',
         "Import images/videos of blast muck pile"),
        ("&Preprocessing...", 'open_frag_preprocessing_panel',
         "Scale calibration, ROI selection, enhancement"),
        ("&Segmentation...", 'open_frag_segmentation_panel',
         "Detect and segment rock fragments"),
        ("&Results / Analysis...", 'open_frag_results_panel',
         "Size distribution, P80, Kuz-Ram comparisons"),
        ("Manual &Editor...", 'open_frag_editor_panel',
         "Manually edit segmented fragments"),
    ]:
        act = QAction(get_menu_icon("tools", "fragmentation"), label, main_window)
        act.setStatusTip(tip)
        if hasattr(main_window, method):
            act.triggered.connect(getattr(main_window, method))
        else:
            act.setEnabled(False)
        frag_menu.addAction(act)

    return menu
