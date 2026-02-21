"""
Haulage Evaluator (STEP 34)

Evaluates schedule feasibility against fleet capacity.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from ..mine_planning.scheduling.types import ScheduleResult, ScheduleDecision, TimePeriod
from .fleet_model import FleetConfig, Truck, Shovel
from .cycle_time_model import Route, compute_cycle_time, CycleTimeResult

logger = logging.getLogger(__name__)


@dataclass
class HaulageEvalConfig:
    """
    Configuration for haulage capacity evaluation.
    
    Attributes:
        schedule: ScheduleResult from NPVS or tactical schedule
        fleet_config: FleetConfig with trucks and shovels
        routes: List of Route objects
        period_mapping: Optional mapping from schedule period IDs to haulage periods
    """
    schedule: ScheduleResult
    fleet_config: FleetConfig
    routes: List[Route]
    period_mapping: Optional[Dict[str, str]] = None


@dataclass
class HaulageEvalResult:
    """
    Result from haulage capacity evaluation.
    
    Attributes:
        period_metrics: List of dicts with per-period metrics
        route_metrics: List of dicts with per-route, per-period metrics
        overall_metrics: Summary KPIs
    """
    period_metrics: List[Dict[str, Any]] = field(default_factory=list)
    route_metrics: List[Dict[str, Any]] = field(default_factory=list)
    overall_metrics: Dict[str, Any] = field(default_factory=dict)


def evaluate_haulage_capacity(config: HaulageEvalConfig) -> HaulageEvalResult:
    """
    Evaluate haulage capacity against schedule requirements.
    
    Args:
        config: HaulageEvalConfig with schedule, fleet, and routes
    
    Returns:
        HaulageEvalResult with metrics
    """
    schedule = config.schedule
    fleet = config.fleet_config
    routes = config.routes
    
    # Create route lookup by (source, destination)
    route_lookup = {}
    for route in routes:
        key = (route.source, route.destination)
        route_lookup[key] = route
    
    period_metrics = []
    route_metrics = []
    
    # Aggregate tonnes by period and (source, destination)
    tonnes_by_period_route = {}
    
    for decision in schedule.decisions:
        # Handle both dataclass and dict formats
        if hasattr(decision, 'period_id'):
            period_id = decision.period_id
            source = decision.unit_id
            destination = decision.destination
            tonnes = decision.tonnes
        else:
            period_id = decision.get('period_id', '')
            source = decision.get('unit_id', '')
            destination = decision.get('destination', 'plant')
            tonnes = decision.get('tonnes', 0.0)
        
        key = (period_id, source, destination)
        if key not in tonnes_by_period_route:
            tonnes_by_period_route[key] = 0.0
        tonnes_by_period_route[key] += tonnes
    
    # Calculate available truck-hours
    total_trucks = len(fleet.trucks)
    shift_hours = fleet.shift_hours
    availability = np.mean([t.availability for t in fleet.trucks]) if fleet.trucks else 0.8
    utilisation = np.mean([t.utilisation for t in fleet.trucks]) if fleet.trucks else 0.85
    
    available_truck_hours_per_period = total_trucks * shift_hours * availability * utilisation
    
    # Process each period
    for period in schedule.periods:
        # Handle both dataclass and dict formats
        if hasattr(period, 'id'):
            period_id = period.id
        else:
            period_id = period.get('id', '')
        
        # Aggregate tonnes for this period
        period_tonnes_by_route = {}
        total_period_tonnes = 0.0
        
        for (p_id, source, dest), tonnes in tonnes_by_period_route.items():
            if p_id == period_id:
                route_key = (source, dest)
                if route_key not in period_tonnes_by_route:
                    period_tonnes_by_route[route_key] = 0.0
                period_tonnes_by_route[route_key] += tonnes
                total_period_tonnes += tonnes
        
        # Calculate required truck-hours per route
        total_required_hours = 0.0
        route_details = []
        
        for (source, dest), tonnes in period_tonnes_by_route.items():
            if tonnes <= 0:
                continue
            
            # Find route
            route = route_lookup.get((source, dest))
            if route is None:
                logger.warning(f"No route found for {source} -> {dest}")
                continue
            
            # Compute cycle time (use first truck as representative)
            if fleet.trucks:
                truck = fleet.trucks[0]
                cycle_result = compute_cycle_time(truck, route, {})
                
                # Calculate truck-hours needed
                cycles_needed = tonnes / truck.payload_tonnes
                cycle_time_hours = cycle_result.truck_cycle_minutes / 60.0
                required_hours = cycles_needed * cycle_time_hours
                
                total_required_hours += required_hours
                
                route_details.append({
                    "route_id": route.id,
                    "source": source,
                    "destination": dest,
                    "tonnes": tonnes,
                    "required_truck_hours": required_hours,
                    "cycle_time_minutes": cycle_result.truck_cycle_minutes,
                    "tonnes_per_hour": cycle_result.tonnes_per_hour
                })
        
        # Calculate utilisation and shortfall
        utilisation_pct = (total_required_hours / available_truck_hours_per_period * 100) if available_truck_hours_per_period > 0 else 0.0
        shortfall_hours = max(0, total_required_hours - available_truck_hours_per_period)
        
        # Calculate effective tonnes if shortfall exists
        effective_tonnes = total_period_tonnes
        if shortfall_hours > 0 and total_required_hours > 0:
            # Linear scaling
            scale_factor = available_truck_hours_per_period / total_required_hours
            effective_tonnes = total_period_tonnes * scale_factor
        
        shortfall_tonnes = total_period_tonnes - effective_tonnes
        
        period_metrics.append({
            "period_id": period_id,
            "scheduled_tonnes": total_period_tonnes,
            "effective_tonnes": effective_tonnes,
            "required_truck_hours": total_required_hours,
            "available_truck_hours": available_truck_hours_per_period,
            "utilisation_pct": utilisation_pct,
            "shortfall_hours": shortfall_hours,
            "shortfall_tonnes": shortfall_tonnes,
            "route_count": len(route_details)
        })
        
        # Add route metrics
        for route_detail in route_details:
            route_metrics.append({
                "period_id": period_id,
                **route_detail
            })
    
    # Calculate overall metrics
    total_scheduled_tonnes = sum(pm["scheduled_tonnes"] for pm in period_metrics)
    total_effective_tonnes = sum(pm["effective_tonnes"] for pm in period_metrics)
    avg_utilisation = np.mean([pm["utilisation_pct"] for pm in period_metrics]) if period_metrics else 0.0
    max_utilisation = max([pm["utilisation_pct"] for pm in period_metrics], default=0.0)
    total_shortfall_tonnes = sum(pm["shortfall_tonnes"] for pm in period_metrics)
    
    overall_metrics = {
        "total_scheduled_tonnes": total_scheduled_tonnes,
        "total_effective_tonnes": total_effective_tonnes,
        "avg_utilisation_pct": avg_utilisation,
        "max_utilisation_pct": max_utilisation,
        "total_shortfall_tonnes": total_shortfall_tonnes,
        "fleet_limited": total_shortfall_tonnes > 0,
        "period_count": len(period_metrics)
    }
    
    logger.info(f"Haulage evaluation complete: {len(period_metrics)} periods, "
                f"avg utilisation {avg_utilisation:.1f}%, "
                f"shortfall {total_shortfall_tonnes:,.0f} tonnes")
    
    return HaulageEvalResult(
        period_metrics=period_metrics,
        route_metrics=route_metrics,
        overall_metrics=overall_metrics
    )

