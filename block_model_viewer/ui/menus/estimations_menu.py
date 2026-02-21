"""
Estimations menu construction for GeoX.

Extracted from MainWindow to improve maintainability.
Handles variograms, kriging, simulations, and resource modeling.
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


def build_estimations_menu(main_window: 'MainWindow', menubar: QMenuBar) -> QMenu:
    """Build and return the Estimations menu."""
    estimations_menu = menubar.addMenu("&Estimations")
    
    # ================== VARIOGRAM TOOLS SUBMENU ==================
    variogram_menu = estimations_menu.addMenu("3D Variogram Tools")
    
    # Compute 3D Variogram
    compute_vario_action = QAction(get_menu_icon("estimations", "variogram"), "Compute 3D Variogram", main_window)
    compute_vario_action.setStatusTip("Calculate experimental and modelled 3D variograms from drillhole data")
    compute_vario_action.triggered.connect(main_window.open_variogram_analysis)
    variogram_menu.addAction(compute_vario_action)
    
    # Show Variogram Cloud
    vario_cloud_action = QAction(get_menu_icon("estimations", "variogram_cloud"), "Show 3D Variogram Cloud", main_window)
    vario_cloud_action.setStatusTip("Display variogram cloud (distance vs semivariance pairs)")
    vario_cloud_action.triggered.connect(main_window.show_variogram_cloud)
    variogram_menu.addAction(vario_cloud_action)
    
    # Fit Variogram Model
    fit_vario_action = QAction(get_menu_icon("estimations", "variogram_fit"), "Fit Variogram Model", main_window)
    fit_vario_action.setStatusTip("Fit theoretical variogram models (spherical, exponential, gaussian)")
    fit_vario_action.triggered.connect(main_window.fit_variogram_model)
    variogram_menu.addAction(fit_vario_action)
    
    variogram_menu.addSeparator()

    # Variogram Modelling Assistant
    variogram_assistant_action = QAction(get_menu_icon("estimations", "variogram_assistant"), "Variogram Modelling Assistant", main_window)
    variogram_assistant_action.setStatusTip(
        "Semi-automatic variogram fitting with model selection and cross-validation"
    )
    variogram_assistant_action.triggered.connect(main_window.open_variogram_assistant_panel)
    variogram_menu.addAction(variogram_assistant_action)

    # Export Variogram Table
    export_vario_action = QAction(get_menu_icon("estimations", "variogram_export"), "Export Variogram Table (CSV)", main_window)
    export_vario_action.setStatusTip("Export variogram tables to CSV files")
    export_vario_action.triggered.connect(main_window.export_variogram_tables)
    variogram_menu.addAction(export_vario_action)

    estimations_menu.addSeparator()

    # ================== KRIGING & ESTIMATION SUBMENU ==================
    kriging_menu = estimations_menu.addMenu("Kriging && Estimation")
    
    # Run 3D Ordinary Kriging
    kriging_action = QAction(get_menu_icon("estimations", "kriging"), "Run 3D Ordinary Kriging", main_window)
    kriging_action.setStatusTip("Perform 3D Ordinary Kriging grade estimation using drillhole data and variogram models")
    kriging_action.triggered.connect(main_window.open_kriging_panel)
    kriging_menu.addAction(kriging_action)
    
    kriging_menu.addSeparator()
    
    # Run 3D Simple Kriging
    simple_kriging_action = QAction(get_menu_icon("estimations", "kriging"), "Run 3D Simple Kriging", main_window)
    simple_kriging_action.setStatusTip("Perform 3D Simple Kriging grade estimation with known global mean")
    simple_kriging_action.triggered.connect(main_window.open_simple_kriging)
    kriging_menu.addAction(simple_kriging_action)
    
    kriging_menu.addSeparator()
    
    # Universal Kriging
    universal_kriging_action = QAction(get_menu_icon("estimations", "kriging"), "Universal Kriging", main_window)
    universal_kriging_action.setStatusTip("Universal Kriging with configurable drift (constant, linear, quadratic)")
    universal_kriging_action.triggered.connect(main_window.open_universal_kriging_panel)
    kriging_menu.addAction(universal_kriging_action)
    
    # Co-Kriging
    cokriging_action = QAction(get_menu_icon("estimations", "cokriging"), "Co-Kriging", main_window)
    cokriging_action.setStatusTip("Co-Kriging estimation using primary and secondary variables")
    cokriging_action.triggered.connect(main_window.open_cokriging_panel)
    kriging_menu.addAction(cokriging_action)
    
    # Indicator Kriging
    indicator_kriging_action = QAction(get_menu_icon("estimations", "indicator_kriging"), "Indicator Kriging", main_window)
    indicator_kriging_action.setStatusTip("Indicator Kriging for non-parametric estimation and CDF construction")
    indicator_kriging_action.triggered.connect(main_window.open_indicator_kriging_panel)
    kriging_menu.addAction(indicator_kriging_action)
    
    kriging_menu.addSeparator()
    
    # Soft / Bayesian Kriging
    soft_kriging_action = QAction(get_menu_icon("estimations", "bayesian"), "Soft / Bayesian Kriging", main_window)
    soft_kriging_action.setStatusTip("Bayesian kriging with soft data priors for OK/UK/IK/CoK")
    soft_kriging_action.triggered.connect(main_window.open_soft_kriging_panel)
    kriging_menu.addAction(soft_kriging_action)
    
    kriging_menu.addSeparator()

    # RBF Interpolation
    rbf_action = QAction(get_menu_icon("estimations", "rbf"), "RBF Interpolation", main_window)
    rbf_action.setStatusTip("Radial Basis Function interpolation with anisotropy, polynomial drift, and GPU acceleration")
    rbf_action.triggered.connect(main_window.open_rbf_panel)
    kriging_menu.addAction(rbf_action)

    kriging_menu.addSeparator()

    # Uncertainty Propagation
    uncertainty_prop_action = QAction(get_menu_icon("estimations", "uncertainty"), "Uncertainty Propagation", main_window)
    uncertainty_prop_action.setStatusTip(
        "Propagate grade realisations through economic models (NPV/IRR/Pit/Schedule)"
    )
    uncertainty_prop_action.triggered.connect(main_window.open_uncertainty_propagation_panel)
    kriging_menu.addAction(uncertainty_prop_action)
    
    kriging_menu.addSeparator()

    estimations_menu.addSeparator()
    
    # Simulations submenu
    simulations_menu = estimations_menu.addMenu("Simulations")

    # Classical SGSIM
    sgsim_action = QAction(get_menu_icon("estimations", "sgsim"), "Sequential Gaussian Simulation (SGSIM)", main_window)
    sgsim_action.setStatusTip(
        "Sequential Gaussian Simulation for uncertainty quantification (multiple realizations, P10/P50/P90)"
    )
    sgsim_action.triggered.connect(main_window.open_sgsim_panel)
    simulations_menu.addAction(sgsim_action)
    
    simulations_menu.addSeparator()
    
    # IK-based SGSIM
    ik_sgsim_action = QAction(get_menu_icon("estimations", "indicator_kriging"), "IK-based Simulation (IK-SGSIM)", main_window)
    ik_sgsim_action.setStatusTip(
        "Indicator Kriging based Sequential Gaussian Simulation using local CDFs"
    )
    ik_sgsim_action.triggered.connect(main_window.open_ik_sgsim_panel)
    simulations_menu.addAction(ik_sgsim_action)
    
    # Co-Simulation (CoSGSIM)
    cosgsim_action = QAction(get_menu_icon("estimations", "cokriging"), "Co-Simulation (CoSGSIM)", main_window)
    cosgsim_action.setStatusTip(
        "Sequential Gaussian Co-Simulation for multiple correlated variables"
    )
    cosgsim_action.triggered.connect(main_window.open_cosgsim_panel)
    simulations_menu.addAction(cosgsim_action)
    
    simulations_menu.addSeparator()
    
    # Sequential Indicator Simulation (SIS)
    sis_action = QAction(get_menu_icon("estimations", "sis"), "Sequential Indicator Simulation (SIS)", main_window)
    sis_action.setStatusTip(
        "Category-based simulation using indicator transforms - best for lithology, ore/waste, multi-modal data"
    )
    sis_action.triggered.connect(main_window.open_sis_panel)
    simulations_menu.addAction(sis_action)
    
    # Turning Bands Simulation
    turning_bands_action = QAction(get_menu_icon("estimations", "turning_bands"), "Turning Bands Simulation", main_window)
    turning_bands_action.setStatusTip(
        "Fast Gaussian field simulation using 1D line processes - best for large domains (>100M cells)"
    )
    turning_bands_action.triggered.connect(main_window.open_turning_bands_panel)
    simulations_menu.addAction(turning_bands_action)
    
    # Direct Block Simulation (DBS)
    dbs_action = QAction(get_menu_icon("estimations", "dbs"), "Direct Block Simulation (DBS)", main_window)
    dbs_action.setStatusTip(
        "Simulation directly at block support - best for iron ore, coal, recoverable resources"
    )
    dbs_action.triggered.connect(main_window.open_dbs_panel)
    simulations_menu.addAction(dbs_action)
    
    simulations_menu.addSeparator()
    
    # Multiple-Point Simulation (MPS)
    mps_action = QAction(get_menu_icon("estimations", "mps"), "Multiple-Point Simulation (MPS)", main_window)
    mps_action.setStatusTip(
        "Pattern-based simulation using training images - best for complex geology, channels, veins"
    )
    mps_action.triggered.connect(main_window.open_mps_panel)
    simulations_menu.addAction(mps_action)
    
    # Gaussian Random Fields (GRF)
    grf_action = QAction(get_menu_icon("estimations", "grf"), "Gaussian Random Fields (GRF)", main_window)
    grf_action.setStatusTip(
        "Fast unconditional/conditional Gaussian fields - best for geomechanics, hydrogeology, porosity"
    )
    grf_action.triggered.connect(main_window.open_grf_panel)
    simulations_menu.addAction(grf_action)
    
    simulations_menu.addSeparator()
    
    # Geomet Chain
    geomet_chain_action = QAction(get_menu_icon("estimations", "geomet_chain"), "Geomet Chain", main_window)
    geomet_chain_action.setStatusTip("Full geomet chain from ore types to plant response to NPV")
    geomet_chain_action.triggered.connect(main_window.open_geomet_chain_panel)
    estimations_menu.addAction(geomet_chain_action)
    
    estimations_menu.addSeparator()
    
    # Resource Modelling submenu
    resource_menu = estimations_menu.addMenu("Resource Modelling")
    
    # Build Block Model
    build_block_model_action = QAction(get_menu_icon("estimations", "resource_model"), "Build Block Model", main_window)
    build_block_model_action.setStatusTip("Generate regular 3D block model from kriging/estimation results")
    build_block_model_action.triggered.connect(main_window.open_block_model_builder)
    resource_menu.addAction(build_block_model_action)
    
    resource_menu.addSeparator()
    
    # Resource Classification (Drillhole-Based)
    resource_class_action = QAction(get_menu_icon("estimations", "classification"), "Resource Classification (Drillhole-Based)", main_window)
    resource_class_action.setStatusTip(
        "JORC/SAMREC-compliant resource classification using drillhole proximity and variance"
    )
    resource_class_action.triggered.connect(main_window.open_resource_classification_panel)
    resource_menu.addAction(resource_class_action)

    # Resource Reporting
    resource_reporting_action = QAction(get_menu_icon("drillholes", "reporting"), "Resource Reporting", main_window)
    resource_reporting_action.setStatusTip(
        "Generate geological tonnage resource summaries with mass-weighted grade calculations"
    )
    resource_reporting_action.triggered.connect(main_window.open_resource_reporting_panel)
    resource_menu.addAction(resource_reporting_action)

    resource_menu.addSeparator()

    # Block Property Calculator
    block_prop_calc_action = QAction(get_menu_icon("estimations", "block_properties"), "Block Property Calculator", main_window)
    block_prop_calc_action.setStatusTip(
        "Calculate and permanently add tonnage/volume properties to block models"
    )
    block_prop_calc_action.triggered.connect(main_window.open_block_property_calculator_panel)
    resource_menu.addAction(block_prop_calc_action)

    resource_menu.addSeparator()

    # Grade-Tonnage Analysis Submenu
    grade_tonnage_menu = resource_menu.addMenu("Grade-Tonnage Analysis")

    # Basic Grade-Tonnage (simplified)
    gt_basic_action = QAction(get_menu_icon("estimations", "grade_tonnage"), "Basic Grade-Tonnage Curves", main_window)
    gt_basic_action.setStatusTip("Simple grade-tonnage curves without economic optimization")
    gt_basic_action.triggered.connect(main_window.open_grade_tonnage_basic_panel)
    grade_tonnage_menu.addAction(gt_basic_action)

    grade_tonnage_menu.addSeparator()

    # Full Grade-Tonnage & Cut-off Sensitivity (comprehensive)
    gt_full_action = QAction(get_menu_icon("estimations", "grade_tonnage"), "Grade-Tonnage && Cut-off Sensitivity", main_window)
    gt_full_action.setStatusTip("Comprehensive grade-tonnage analysis with cut-off sensitivity and domain analysis")
    gt_full_action.triggered.connect(main_window.open_grade_tonnage_panel)
    grade_tonnage_menu.addAction(gt_full_action)

    grade_tonnage_menu.addSeparator()

    # Cut-off Optimization (economic analysis)
    cutoff_opt_action = QAction(get_menu_icon("estimations", "npv"), "Cut-off Grade Optimization", main_window)
    cutoff_opt_action.setStatusTip("Economic cut-off optimization with NPV, IRR, and payback analysis")
    cutoff_opt_action.triggered.connect(main_window.open_cutoff_optimization_panel)
    grade_tonnage_menu.addAction(cutoff_opt_action)

    return estimations_menu

