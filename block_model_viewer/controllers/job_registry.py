"""
Job Registry - Unified Task Dispatch

Maps task names to engine functions via sub-controllers. All analysis, resource,
and planning operations are registered here for centralized dispatch.
"""

import logging
from typing import Dict, Callable, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .geostats_controller import GeostatsController
    from .mining_controller import MiningController
    from .vis_controller import VisController
    from .data_controller import DataController
    from .scan_controller import ScanController
    from .survey_deformation_controller import SurveyDeformationController
    from .insar_controller import InsarController

logger = logging.getLogger(__name__)


class JobRegistry:
    """
    Registry mapping task names to engine functions.
    
    All engine functions must:
    - Accept a single params dict
    - Return a result object or dict
    - Be pure Python (no Qt dependencies)
    """
    
    REGISTRY: Dict[str, Callable] = {}
    _initialized = False
    
    @classmethod
    def initialize(cls, controller: Any) -> None:
        """
        Initialize the registry with controller's prepare functions.
        
        DEPRECATED: Use initialize_with_subcontrollers() instead.
        
        Args:
            controller: AppController instance (needed to access _prepare_* methods)
        """
        if cls._initialized:
            return
        
        # Fall back to old-style initialization if sub-controllers aren't available
        if hasattr(controller, '_geostats'):
            cls.initialize_with_subcontrollers(
                controller._geostats,
                controller._mining,
                controller._vis,
                controller._data
            )
            return
        
        # Legacy initialization - kept for backward compatibility
        logger.warning("Using legacy JobRegistry initialization")
        cls._initialize_legacy(controller)
    
    @classmethod
    def _initialize_legacy(cls, controller: Any) -> None:
        """Legacy initialization using AppController methods directly."""
        cls.REGISTRY = {
            "simple_kriging": lambda params: controller._prepare_simple_kriging_payload(params, params.get("_progress_callback")),
            "kriging": lambda params: controller._prepare_kriging_payload(params, params.get("_progress_callback")),
            "sgsim": lambda params: controller._prepare_sgsim_payload(params, params.get("_progress_callback")),
        }
        cls._initialized = True
        logger.info("JobRegistry initialized (legacy mode)")
    
    @classmethod
    def initialize_with_subcontrollers(
        cls,
        geostats: "GeostatsController",
        mining: "MiningController",
        vis: "VisController",
        data: "DataController",
        scan: "ScanController",
        survey: Optional["SurveyDeformationController"] = None,
        insar: Optional["InsarController"] = None
    ) -> None:
        """
        Initialize the registry with sub-controller prepare functions.
        
        Args:
            geostats: GeostatsController instance
            mining: MiningController instance
            vis: VisController instance
            data: DataController instance
            survey: SurveyDeformationController instance (survey deformation module)
        """
        if cls._initialized:
            return
        
        cls.REGISTRY = {
            # ==================================================================
            # GEOSTATISTICS (GeostatsController)
            # ==================================================================
            "simple_kriging": lambda params: geostats._prepare_simple_kriging_payload(params, params.get("_progress_callback")),
            "kriging": lambda params: geostats._prepare_kriging_payload(params, params.get("_progress_callback")),
            "sgsim": lambda params: geostats._prepare_sgsim_payload(params, params.get("_progress_callback")),
            "variogram": lambda params: geostats._prepare_variogram_payload(params),
            "uncertainty": lambda params: geostats._prepare_uncertainty_payload(params),
            "grade_stats": lambda params: geostats._prepare_grade_stats_payload(params),
            "grade_transform": lambda params: geostats._prepare_grade_transform_payload(params),
            "swath": lambda params: geostats._prepare_swath_payload(params),
            "kmeans": lambda params: geostats._prepare_kmeans_payload(params),
            "universal_kriging": lambda params: geostats._prepare_universal_kriging_payload(params, params.get("_progress_callback")),
            "cokriging": lambda params: geostats._prepare_cokriging_payload(params, params.get("_progress_callback")),
            "indicator_kriging": lambda params: geostats._prepare_indicator_kriging_payload(params, params.get("_progress_callback")),
            "variogram_assistant": lambda params: geostats._prepare_variogram_assistant_payload(params),
            "bayesian_kriging": lambda params: geostats._prepare_bayesian_kriging_payload(params),
            "soft_kriging": lambda params: geostats._prepare_bayesian_kriging_payload(params),  # Alias
            "ik_sgsim": lambda params: geostats._prepare_ik_sgsim_payload(params),
            "mps": lambda params: geostats._prepare_mps_payload(params, params.get("_progress_callback")),
            "sis": lambda params: geostats._prepare_sis_payload(params, params.get("_progress_callback")),
            "turning_bands": lambda params: geostats._prepare_turning_bands_payload(params, params.get("_progress_callback")),
            "dbs": lambda params: geostats._prepare_dbs_payload(params, params.get("_progress_callback")),
            "grf": lambda params: geostats._prepare_grf_payload(params, params.get("_progress_callback")),
            "cosgsim": lambda params: geostats._prepare_cosgsim_payload(params, params.get("_progress_callback")),
            "rbf": lambda params: geostats._prepare_rbf_payload(params, params.get("_progress_callback")),
            "economic_uncert": lambda params: geostats._prepare_economic_uncertainty_payload(params),

            # ==================================================================
            # MINING & PLANNING (MiningController)
            # ==================================================================
            "resources": lambda params: mining._prepare_resource_calculation_payload(params),
            "classify": lambda params: mining._prepare_resource_classification_payload(params, params.get("_progress_callback")),
            "resource_classification": lambda params: mining._prepare_resource_classification_payload(params, params.get("_progress_callback")),
            "irr": lambda params: mining._prepare_irr_payload(params),
            "npv": lambda params: mining._prepare_npv_payload(params),
            "pit_opt": lambda params: mining._prepare_pit_optimisation_payload(params),
            "underground": lambda params: mining._prepare_underground_payload(params),
            "esg": lambda params: mining._prepare_esg_payload(params),
            
            # Geometallurgy
            "geomet_assign_ore_types": lambda params: mining._prepare_geomet_assign_ore_types_payload(params),
            "geomet_compute_block_attrs": lambda params: mining._prepare_geomet_compute_block_attrs_payload(params),
            "geomet_plant_response": lambda params: mining._prepare_geomet_plant_response_payload(params),
            "geomet_compute_values": lambda params: mining._prepare_geomet_compute_values_payload(params),
            
            # Grade Control
            "gc_build_support_model": lambda params: mining._prepare_gc_build_support_model_payload(params),
            "gc_ok": lambda params: mining._prepare_gc_ok_payload(params),
            "gc_sgsim": lambda params: mining._prepare_gc_sgsim_payload(params),
            "gc_classify_ore_waste": lambda params: mining._prepare_gc_classify_ore_waste_payload(params),
            "gc_summarise_digpolys": lambda params: mining._prepare_gc_summarise_digpolys_payload(params),
            
            # Reconciliation
            "recon_model_mine": lambda params: mining._prepare_recon_model_mine_payload(params),
            "recon_mine_mill": lambda params: mining._prepare_recon_mine_mill_payload(params),
            "recon_metrics": lambda params: mining._prepare_recon_metrics_payload(params),
            
            # Strategic Scheduling
            "strategic_milp_schedule": lambda params: mining._prepare_strategic_milp_schedule_payload(params),
            "nested_shell_schedule": lambda params: mining._prepare_nested_shell_schedule_payload(params),
            "cutoff_schedule_opt": lambda params: mining._prepare_cutoff_schedule_opt_payload(params),
            
            # Tactical Scheduling
            "tactical_pushback_schedule": lambda params: mining._prepare_tactical_pushback_schedule_payload(params),
            "tactical_bench_schedule": lambda params: mining._prepare_tactical_bench_schedule_payload(params),
            "tactical_dev_schedule": lambda params: mining._prepare_tactical_dev_schedule_payload(params),
            
            # Short-Term Scheduling
            "short_term_digline_schedule": lambda params: mining._prepare_short_term_digline_schedule_payload(params),
            "short_term_blend": lambda params: mining._prepare_short_term_blend_payload(params),
            "shift_plan": lambda params: mining._prepare_shift_plan_payload(params),
            
            # Fleet & Haulage
            "fleet_cycle_time": lambda params: mining._prepare_fleet_cycle_time_payload(params),
            "fleet_dispatch": lambda params: mining._prepare_fleet_dispatch_payload(params),
            "haulage_evaluate": lambda params: mining._prepare_haulage_evaluate_payload(params),
            
            # Planning Dashboard & Scenarios
            "planning_run_scenario": lambda params: mining._prepare_planning_run_scenario_payload(params),
            "planning_compare_scenarios": lambda params: mining._prepare_planning_compare_scenarios_payload(params),
            
            # NPVS
            "npvs_run": lambda params: mining._prepare_npvs_run_payload(params),
            
            # Pushback Designer
            "pushback_build_plan": lambda params: mining._prepare_pushback_build_plan_payload(params),
            
            # Cutoff Optimiser
            "cutoff_optimise": lambda params: mining._prepare_cutoff_optimise_payload(params),
            
            # Production Alignment
            "production_align": lambda params: mining._prepare_production_align_payload(params),
            
            # Underground Planning
            "ug_slos_generate_stopes": lambda params: mining._prepare_ug_slos_generate_stopes_payload(params),
            "ug_slos_optimise": lambda params: mining._prepare_ug_slos_optimise_payload(params),
            "ug_cave_build_footprint": lambda params: mining._prepare_ug_cave_build_footprint_payload(params),
            "ug_cave_simulate_draw": lambda params: mining._prepare_ug_cave_simulate_draw_payload(params),
            "ug_apply_dilution": lambda params: mining._prepare_ug_apply_dilution_payload(params),
            
            # ==================================================================
            # DATA (DataController) - Drillholes, Geology, Structural, Geotech
            # ==================================================================
            # Drillholes
            "load_drillholes": lambda params: data._prepare_load_drillholes_payload(params),
            "drillhole_qaqc": lambda params: data._prepare_drillhole_qaqc_payload(params),
            "drillhole_import": lambda params: data._prepare_drillhole_import_payload(params, params.get("_progress_callback")),
            "drillhole_database": lambda params: data._prepare_drillhole_database_payload(params, params.get("_progress_callback")),
            "load_file": lambda params: data._prepare_load_file_payload(params, params.get("_progress_callback")),
            "build_block_model": lambda params: data._prepare_build_block_model_payload(params, params.get("_progress_callback")),
            
            # Geology (Legacy)
            "implicit_geology": lambda params: data._prepare_implicit_geology_payload(params),
            "build_wireframes": lambda params: data._prepare_build_wireframes_payload(params),
            
            # Geology Wizard Workflow
            "geology_compositing": lambda params: data._prepare_geology_compositing_payload(params, params.get("_progress_callback")),
            "geology_validate": lambda params: data._prepare_geology_validate_payload(params, params.get("_progress_callback")),
            "geology_domains": lambda params: data._prepare_geology_domains_payload(params, params.get("_progress_callback")),
            "geology_build_surfaces": lambda params: data._prepare_geology_build_surfaces_payload(params, params.get("_progress_callback")),
            "geology_build_solids": lambda params: data._prepare_geology_build_solids_payload(params, params.get("_progress_callback")),
            "geology_misfit_qc": lambda params: data._prepare_geology_misfit_qc_payload(params, params.get("_progress_callback")),
            "geology_export": lambda params: data._prepare_geology_export_payload(params, params.get("_progress_callback")),
            
            # LoopStructural Geological Modeling (Industry-grade JORC/SAMREC compliant)
            "loopstructural_model": lambda params: data._prepare_loopstructural_model_payload(params, params.get("_progress_callback")),
            "loopstructural_compliance": lambda params: data._prepare_loopstructural_compliance_payload(params, params.get("_progress_callback")),
            "loopstructural_fault_detection": lambda params: data._prepare_loopstructural_fault_detection_payload(params, params.get("_progress_callback")),
            "loopstructural_extract_surfaces": lambda params: data._prepare_loopstructural_extract_surfaces_payload(params, params.get("_progress_callback")),
            
            # Structural
            "structural_clusters": lambda params: data._prepare_structural_clusters_payload(params),
            "kinematic_analysis": lambda params: data._prepare_kinematic_analysis_payload(params),
            
            # Geotechnical Interpolation
            "geotech_interpolation": lambda params: data._prepare_geotech_interpolation_payload(params),
            "stope_stability": lambda params: data._prepare_stope_stability_payload(params),
            "stope_stability_mc": lambda params: data._prepare_stope_stability_mc_payload(params),
            "slope_risk": lambda params: data._prepare_slope_risk_payload(params),
            "slope_risk_mc": lambda params: data._prepare_slope_risk_mc_payload(params),
            
            # Slope Stability
            "slope_lem_2d": lambda params: data._prepare_slope_lem_2d_payload(params),
            "slope_lem_3d": lambda params: data._prepare_slope_lem_3d_payload(params),
            "slope_probabilistic": lambda params: data._prepare_slope_probabilistic_payload(params),
            "bench_design_suggest": lambda params: data._prepare_bench_design_suggest_payload(params),
            
            # Seismic
            "seismic_hazard": lambda params: data._prepare_seismic_hazard_payload(params),
            "seismic_hazard_mc": lambda params: data._prepare_seismic_hazard_mc_payload(params),
            "rockburst_index": lambda params: data._prepare_rockburst_index_payload(params),
            "rockburst_index_mc": lambda params: data._prepare_rockburst_index_mc_payload(params),
            
            # Schedule Risk
            "schedule_risk_profile": lambda params: data._prepare_schedule_risk_profile_payload(params),
            "schedule_risk_timeline": lambda params: data._prepare_schedule_risk_timeline_payload(params),
            "schedule_risk_compare": lambda params: data._prepare_schedule_risk_compare_payload(params),
            
            # Research Mode
            "research_run_grid": lambda params: data._prepare_research_grid_payload(params),

            # ==================================================================
            # SCAN ANALYSIS (ScanController)
            # ==================================================================
            "scan_validate": lambda params: scan._prepare_scan_validate_payload(params, params.get("_progress_callback")),
            "scan_clean": lambda params: scan._prepare_scan_clean_payload(params, params.get("_progress_callback")),
            "scan_segment": lambda params: scan._prepare_scan_segment_payload(params, params.get("_progress_callback")),
            "scan_compute_metrics": lambda params: scan._prepare_scan_compute_metrics_payload(params, params.get("_progress_callback")),
        }

        # ======================================================================
        # SURVEY DEFORMATION (SurveyDeformationController) - conditional
        # ======================================================================
        if survey is not None:
            cls.REGISTRY.update({
                "subsidence_time_series": lambda params: survey._prepare_subsidence_time_series_payload(params, params.get("_progress_callback")),
                "control_stability": lambda params: survey._prepare_control_stability_payload(params, params.get("_progress_callback")),
                "groundwater_time_series": lambda params: survey._prepare_groundwater_time_series_payload(params, params.get("_progress_callback")),
                "subsidence_groundwater_coupling": lambda params: survey._prepare_coupling_payload(params, params.get("_progress_callback")),
                "deformation_index": lambda params: survey._prepare_deformation_index_payload(params, params.get("_progress_callback")),
            })

        if insar is not None:
            cls.REGISTRY.update({
                "insar_run": lambda params: insar._prepare_insar_run_payload(params),
                "insar_ingest": lambda params: insar._prepare_insar_ingest_payload(params),
            })
        
        cls._initialized = True
        logger.info(f"JobRegistry initialized with {len(cls.REGISTRY)} tasks (sub-controller mode)")
    
    @classmethod
    def get(cls, task: str) -> Optional[Callable]:
        """
        Get engine function for a task.
        
        Args:
            task: Task name (e.g., "kriging", "sgsim", "irr")
            
        Returns:
            Function that accepts params dict, or None if task not found
        """
        func = cls.REGISTRY.get(task)
        if func is None:
            logger.warning(f"Task '{task}' not found in registry")
        return func
    
    @classmethod
    def register(cls, task: str, func: Callable) -> None:
        """
        Register a new task function.
        
        Args:
            task: Task name
            func: Function that accepts params dict and returns result
        """
        cls.REGISTRY[task] = func
        logger.debug(f"Registered task '{task}'")
    
    @classmethod
    def list_tasks(cls) -> list:
        """Get list of all registered task names."""
        return list(cls.REGISTRY.keys())
    
    @classmethod
    def reset(cls) -> None:
        """Reset the registry (useful for testing)."""
        cls.REGISTRY = {}
        cls._initialized = False
        logger.debug("JobRegistry reset")
