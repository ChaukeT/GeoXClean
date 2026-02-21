"""
Scenario Comparison Engine (STEP 31)

Compare multiple completed scenarios on key KPIs.
"""

from typing import List, Dict, Any, Callable, Optional
import logging

from .scenario_definition import PlanningScenario

logger = logging.getLogger(__name__)


def compare_scenarios(
    scenarios: List[PlanningScenario],
    irr_results_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    schedule_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    recon_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
    risk_loader: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple completed scenarios on key KPIs.
    
    Args:
        scenarios: List of PlanningScenario to compare
        irr_results_loader: Function to load IRR results by reference
        schedule_loader: Function to load schedule results by reference
        recon_loader: Function to load reconciliation results by reference
        risk_loader: Function to load risk results by reference
        
    Returns:
        Dictionary of comparison metrics
    """
    comparison = {
        "scenarios": [],
        "metrics": {}
    }
    
    # Default loaders (return empty dict if not provided)
    irr_loader = irr_results_loader or (lambda ref: {})
    schedule_loader_fn = schedule_loader or (lambda ref: {})
    recon_loader_fn = recon_loader or (lambda ref: {})
    risk_loader_fn = risk_loader or (lambda ref: {})
    
    # Collect metrics per scenario
    scenario_metrics = []
    
    for scenario in scenarios:
        if scenario.status != "completed":
            logger.warning(f"Skipping incomplete scenario {scenario.id.name}")
            continue
        
        metrics = {
            "name": scenario.id.name,
            "version": scenario.id.version,
            "description": scenario.description,
            "tags": scenario.tags,
        }
        
        # Load IRR results
        if scenario.outputs and scenario.outputs.irr_result_ref:
            irr_results = irr_loader(scenario.outputs.irr_result_ref)
            metrics["npv"] = irr_results.get("npv", 0.0)
            metrics["irr"] = irr_results.get("irr", 0.0)
            metrics["payback_period"] = irr_results.get("payback_period", None)
        else:
            metrics["npv"] = None
            metrics["irr"] = None
            metrics["payback_period"] = None
        
        # Load schedule results
        if scenario.outputs and scenario.outputs.schedule_result_ref:
            schedule_results = schedule_loader_fn(scenario.outputs.schedule_result_ref)
            metrics["lom_years"] = schedule_results.get("lom_years", None)
            metrics["peak_annual_production"] = schedule_results.get("peak_annual_production", None)
            metrics["total_tonnes"] = schedule_results.get("total_tonnes", None)
        else:
            metrics["lom_years"] = None
            metrics["peak_annual_production"] = None
            metrics["total_tonnes"] = None
        
        # Load reconciliation results
        if scenario.outputs and scenario.outputs.recon_result_ref:
            recon_results = recon_loader_fn(scenario.outputs.recon_result_ref)
            metrics["model_mine_bias"] = recon_results.get("model_mine_bias", {})
            metrics["mine_mill_bias"] = recon_results.get("mine_mill_bias", {})
        else:
            metrics["model_mine_bias"] = {}
            metrics["mine_mill_bias"] = {}
        
        # Load risk results
        if scenario.outputs and scenario.outputs.risk_result_ref:
            risk_results = risk_loader_fn(scenario.outputs.risk_result_ref)
            metrics["npv_p10"] = risk_results.get("npv_p10", None)
            metrics["npv_p50"] = risk_results.get("npv_p50", None)
            metrics["npv_p90"] = risk_results.get("npv_p90", None)
        else:
            metrics["npv_p10"] = None
            metrics["npv_p50"] = None
            metrics["npv_p90"] = None
        
        scenario_metrics.append(metrics)
    
    comparison["scenarios"] = scenario_metrics
    
    # Compute aggregate metrics
    npvs = [m["npv"] for m in scenario_metrics if m["npv"] is not None]
    irrs = [m["irr"] for m in scenario_metrics if m["irr"] is not None]
    
    if npvs:
        comparison["metrics"]["best_npv"] = max(npvs)
        comparison["metrics"]["worst_npv"] = min(npvs)
        comparison["metrics"]["avg_npv"] = sum(npvs) / len(npvs)
    
    if irrs:
        comparison["metrics"]["best_irr"] = max(irrs)
        comparison["metrics"]["worst_irr"] = min(irrs)
        comparison["metrics"]["avg_irr"] = sum(irrs) / len(irrs)
    
    logger.info(f"Compared {len(scenario_metrics)} scenarios")
    
    return comparison

