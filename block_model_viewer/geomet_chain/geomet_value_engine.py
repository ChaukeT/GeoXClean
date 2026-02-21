"""
Geomet Value Engine (STEP 38)

Compute geomet-adjusted block values and attach to block model.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

from .ore_type_model import OreTypeModel
from .route_selector import RouteSelectorConfig, choose_route_for_block
from .plant_response import PlantRoute, compute_recovery, compute_throughput

logger = logging.getLogger(__name__)


@dataclass
class GeometValueConfig:
    """
    Configuration for geomet value computation.
    
    Attributes:
        ore_type_model: OreTypeModel
        route_selector_config: RouteSelectorConfig
        plant_routes: List of PlantRoute objects
        prices: Prices by element
        mining_cost_per_t: Mining cost per tonne
        value_column_prefix: Prefix for value columns (e.g., "gvalue_")
    """
    ore_type_model: OreTypeModel
    route_selector_config: RouteSelectorConfig
    plant_routes: List[PlantRoute] = field(default_factory=list)
    prices: Dict[str, float] = field(default_factory=dict)
    mining_cost_per_t: float = 5.0
    value_column_prefix: str = "gvalue_"


def compute_geomet_block_values(
    block_model: Any,
    config: GeometValueConfig
) -> str:
    """
    Compute geomet-adjusted block values and attach to block model.
    
    Args:
        block_model: BlockModel instance
        config: GeometValueConfig
    
    Returns:
        Primary value column name to be used in NPVS
    """
    logger.info("Computing geomet block values")
    
    # Get block model DataFrame
    if hasattr(block_model, 'to_dataframe'):
        df = block_model.to_dataframe()
    else:
        logger.error("Block model does not support to_dataframe()")
        return ""
    
    if df.empty:
        logger.warning("Block model is empty")
        return ""
    
    # Ensure ore_domain exists
    if 'ore_domain' not in df.columns:
        logger.warning("ore_domain not found in block model, using default")
        df['ore_domain'] = 'UNKNOWN'
    
    if 'hardness_index' not in df.columns:
        logger.warning("hardness_index not found in block model, using default")
        df['hardness_index'] = 5.0
    
    # Get grade properties
    grade_properties = {}
    for col in df.columns:
        if col not in ['X', 'Y', 'Z', 'x', 'y', 'z', 'tonnage', 'TONNAGE', 'ore_domain', 'hardness_index']:
            if df[col].dtype in [np.float64, np.float32]:
                grade_properties[col] = df[col].values
    
    # Get tonnage
    if 'tonnage' in df.columns:
        tonnages = df['tonnage'].values
    elif 'TONNAGE' in df.columns:
        tonnages = df['TONNAGE'].values
    else:
        tonnages = np.ones(len(df)) * 1000.0
    
    # Compute values for each block
    block_values = []
    chosen_routes = []
    
    for idx in range(len(df)):
        row = df.iloc[idx]
        
        # Get ore domain and hardness
        ore_domain_id = str(row.get('ore_domain', 'UNKNOWN'))
        hardness_index = float(row.get('hardness_index', 5.0))
        
        # Get head grades
        head_grades = {}
        for element, values in grade_properties.items():
            grade = values[idx]
            if not np.isnan(grade):
                head_grades[element] = float(grade)
        
        # Choose route
        route_id = choose_route_for_block(
            ore_domain_id, head_grades, hardness_index,
            config.route_selector_config
        )
        
        chosen_routes.append(route_id)
        
        # Find route
        route = None
        for r in config.plant_routes:
            if r.id == route_id:
                route = r
                break
        
        if not route:
            block_values.append(0.0)
            continue
        
        # Compute recoveries
        recoveries = compute_recovery(route, ore_domain_id, head_grades)
        
        # Compute throughput factor
        effective_tph = compute_throughput(
            route, ore_domain_id, hardness_index,
            config.route_selector_config.plant_nominal_tph
        )
        throughput_factor = effective_tph / config.route_selector_config.plant_nominal_tph if config.route_selector_config.plant_nominal_tph > 0 else 1.0
        
        # Compute revenue
        revenue = 0.0
        for element, grade in head_grades.items():
            recovery = recoveries.get(element, 0.0)
            price = config.prices.get(element, 0.0)
            revenue += grade * recovery * price
        
        # Compute costs
        # Base cost
        cost = route.base_cost_per_t
        
        # Energy cost
        specific_energy = 15.0 + (hardness_index - 5.0) * 2.0  # kWh/t
        energy_cost = specific_energy * route.variable_cost_per_kWh
        
        # Total processing cost
        processing_cost = cost + energy_cost
        
        # Net value per tonne (adjusted for throughput)
        net_value_per_t = (revenue - processing_cost) * throughput_factor
        
        # Subtract mining cost
        net_value_per_t -= config.mining_cost_per_t
        
        block_values.append(net_value_per_t)
    
    # Add value columns to block model
    primary_column = f"{config.value_column_prefix}default"
    
    try:
        if hasattr(block_model, 'add_property'):
            block_model.add_property(primary_column, np.array(block_values, dtype=np.float32))
            block_model.add_property('geomet_route', np.array(chosen_routes, dtype=object))
        else:
            # Fallback: add to DataFrame
            df[primary_column] = block_values
            df['geomet_route'] = chosen_routes
            # Try to update block model if it has a method to sync from DataFrame
            if hasattr(block_model, '_sync_from_dataframe'):
                block_model._sync_from_dataframe(df)
    except Exception as e:
        logger.error(f"Error adding geomet values to block model: {e}", exc_info=True)
        raise
    
    logger.info(f"Computed geomet values for {len(block_values)} blocks")
    logger.info(f"Primary value column: {primary_column}")
    
    return primary_column

