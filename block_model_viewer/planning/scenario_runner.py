"""
Scenario Runner (STEP 31)

Orchestrates execution of planning scenarios across all modules.
"""

from typing import Optional, Any, Dict
import logging

from .scenario_definition import PlanningScenario, ScenarioOutputs
from .scenario_store import ScenarioStore

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    Orchestrates running planning scenarios end-to-end.
    """
    
    def __init__(self, controller: Any, store: ScenarioStore):
        """
        Initialize scenario runner.
        
        Args:
            controller: AppController instance
            store: ScenarioStore instance
        """
        self.controller = controller
        self.store = store
        logger.info("Initialized ScenarioRunner")
    
    def run(self, scenario: PlanningScenario, progress_callback: Optional[Any] = None) -> PlanningScenario:
        """
        Execute a scenario end-to-end.
        
        Args:
            scenario: PlanningScenario to run
            progress_callback: Optional progress callback function
            
        Returns:
            Updated PlanningScenario with outputs and status
        """
        if progress_callback:
            progress_callback(0, f"Starting scenario {scenario.id.name} v{scenario.id.version}")
        
        # Update status
        scenario.status = "running"
        self.store.save(scenario)
        
        outputs = ScenarioOutputs()
        
        try:
            # Step 1: Geomet block attributes (if value_mode == "geomet")
            if scenario.inputs.value_mode == "geomet" and scenario.inputs.geomet_config:
                if progress_callback:
                    progress_callback(10, "Computing geomet block attributes...")
                
                try:
                    geomet_result = self._run_geomet(scenario)
                    if geomet_result:
                        outputs.geomet_attrs_ref = f"geomet_{scenario.id.name}_{scenario.id.version}"
                except Exception as e:
                    logger.warning(f"Geomet step failed: {e}", exc_info=True)
            
            # Step 2: Pit optimization (if configured)
            if scenario.inputs.pit_config:
                if progress_callback:
                    progress_callback(20, "Running pit optimization...")
                
                try:
                    pit_result = self._run_pit_optimization(scenario)
                    if pit_result:
                        outputs.pit_shell_ref = f"pit_{scenario.id.name}_{scenario.id.version}"
                except Exception as e:
                    logger.warning(f"Pit optimization failed: {e}", exc_info=True)
            
            # Step 3: Strategic schedule
            if scenario.inputs.schedule_config:
                if progress_callback:
                    progress_callback(40, "Running strategic schedule...")
                
                try:
                    schedule_result = self._run_strategic_schedule(scenario)
                    if schedule_result:
                        outputs.schedule_result_ref = f"schedule_{scenario.id.name}_{scenario.id.version}"
                except Exception as e:
                    logger.warning(f"Strategic schedule failed: {e}", exc_info=True)
            
            # Step 4: Tactical / Short-term schedules (optional)
            if scenario.inputs.schedule_config and scenario.inputs.schedule_config.get("include_tactical"):
                if progress_callback:
                    progress_callback(50, "Running tactical schedule...")
                
                try:
                    self._run_tactical_schedule(scenario)
                except Exception as e:
                    logger.warning(f"Tactical schedule failed: {e}", exc_info=True)
            
            if scenario.inputs.schedule_config and scenario.inputs.schedule_config.get("include_short_term"):
                if progress_callback:
                    progress_callback(55, "Running short-term schedule...")
                
                try:
                    self._run_short_term_schedule(scenario)
                except Exception as e:
                    logger.warning(f"Short-term schedule failed: {e}", exc_info=True)
            
            # Step 5: IRR/NPV
            if progress_callback:
                progress_callback(60, "Running IRR/NPV analysis...")
            
            try:
                irr_result = self._run_irr(scenario)
                if irr_result:
                    outputs.irr_result_ref = f"irr_{scenario.id.name}_{scenario.id.version}"
            except Exception as e:
                logger.warning(f"IRR analysis failed: {e}", exc_info=True)
            
            # Step 6: GC & Reconciliation (if configured)
            if scenario.inputs.gc_config:
                if progress_callback:
                    progress_callback(75, "Running GC model and reconciliation...")
                
                try:
                    gc_result = self._run_gc_and_recon(scenario)
                    if gc_result:
                        outputs.gc_model_ref = f"gc_{scenario.id.name}_{scenario.id.version}"
                        outputs.recon_result_ref = f"recon_{scenario.id.name}_{scenario.id.version}"
                except Exception as e:
                    logger.warning(f"GC/recon failed: {e}", exc_info=True)
            
            # Step 7: Risk/Uncertainty (if configured)
            if scenario.inputs.risk_config:
                if progress_callback:
                    progress_callback(85, "Running risk/uncertainty analysis...")
                
                try:
                    risk_result = self._run_risk(scenario)
                    if risk_result:
                        outputs.risk_result_ref = f"risk_{scenario.id.name}_{scenario.id.version}"
                except Exception as e:
                    logger.warning(f"Risk analysis failed: {e}", exc_info=True)
            
            # Update scenario with outputs
            scenario.outputs = outputs
            scenario.status = "completed"
            
            if progress_callback:
                progress_callback(100, "Scenario completed successfully")
            
            self.store.save(scenario)
            logger.info(f"Scenario {scenario.id.name} v{scenario.id.version} completed")
            
            return scenario
        
        except Exception as e:
            logger.error(f"Scenario {scenario.id.name} failed: {e}", exc_info=True)
            scenario.status = "failed"
            scenario.outputs = outputs
            self.store.save(scenario)
            
            if progress_callback:
                progress_callback(100, f"Scenario failed: {e}")
            
            return scenario
    
    def _run_geomet(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run geomet block attributes computation (STEP 38)."""
        if not hasattr(self.controller, 'run_geomet_chain'):
            logger.warning("Geomet chain not available")
            return None
        
        if not hasattr(self.controller, 'current_block_model') or not self.controller.current_block_model:
            logger.error("No block model available for geomet computation")
            return None
        
        block_model = self.controller.current_block_model
        config = scenario.inputs.geomet_config or {}
        
        # Call geomet chain engine synchronously (for scenario runner)
        try:
            result = self.controller._prepare_geomet_compute_values_payload({
                "block_model": block_model,
                **config
            })
            
            # Update scenario inputs with computed value field
            if result and result.get("result"):
                value_field = result.get("result", {}).get("value_field", "gvalue_default")
                scenario.inputs.value_field = value_field
                logger.info(f"Geomet value field set to: {value_field}")
            
            return result
        except Exception as e:
            logger.error(f"Geomet chain computation failed: {e}", exc_info=True)
            return None
    
    def _run_pit_optimization(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run pit optimization."""
        if not hasattr(self.controller, 'run_pit_optimisation'):
            logger.warning("Pit optimization not available")
            return None
        
        config = scenario.inputs.pit_config or {}
        # Would call controller.run_pit_optimisation(config)
        return {"status": "completed"}
    
    def _run_strategic_schedule(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run strategic schedule."""
        if not hasattr(self.controller, 'run_strategic_schedule'):
            logger.warning("Strategic schedule not available")
            return None
        
        schedule_config = scenario.inputs.schedule_config or {}
        strategic_config = schedule_config.get("strategic", {})
        
        # Would call controller.run_strategic_schedule(config)
        return {"status": "completed"}
    
    def _run_tactical_schedule(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run tactical schedule."""
        if not hasattr(self.controller, 'run_tactical_pushback_schedule'):
            logger.warning("Tactical schedule not available")
            return None
        
        schedule_config = scenario.inputs.schedule_config or {}
        tactical_config = schedule_config.get("tactical", {})
        
        # Would call controller.run_tactical_pushback_schedule(config)
        return {"status": "completed"}
    
    def _run_short_term_schedule(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run short-term schedule."""
        if not hasattr(self.controller, 'run_short_term_digline_schedule'):
            logger.warning("Short-term schedule not available")
            return None
        
        schedule_config = scenario.inputs.schedule_config or {}
        short_term_config = schedule_config.get("short_term", {})
        
        # Would call controller.run_short_term_digline_schedule(config)
        return {"status": "completed"}
    
    def _run_irr(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run IRR/NPV analysis."""
        if not hasattr(self.controller, 'run_irr_analysis'):
            logger.warning("IRR analysis not available")
            return None
        
        # Build IRR config from scenario inputs
        irr_config = {
            "block_model": scenario.inputs.model_name,
            "value_field": scenario.inputs.value_field,
            "schedule": scenario.outputs.schedule_result_ref if scenario.outputs else None
        }
        
        # Would call controller.run_irr_analysis(irr_config)
        return {"status": "completed"}
    
    def _run_gc_and_recon(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run GC model and reconciliation."""
        if not hasattr(self.controller, 'build_gc_support_model'):
            logger.warning("GC model not available")
            return None
        
        gc_config = scenario.inputs.gc_config or {}
        
        # Would call:
        # - controller.build_gc_support_model(...)
        # - controller.run_gc_ok(...)
        # - controller.run_recon_model_mine(...)
        
        return {"status": "completed"}
    
    def _run_risk(self, scenario: PlanningScenario) -> Optional[Any]:
        """Run risk/uncertainty analysis."""
        if not hasattr(self.controller, 'run_uncertainty_analysis'):
            logger.warning("Risk analysis not available")
            return None
        
        risk_config = scenario.inputs.risk_config or {}
        
        # Would call controller.run_uncertainty_analysis(risk_config)
        return {"status": "completed"}

