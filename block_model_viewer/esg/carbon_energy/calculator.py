"""
Carbon & Energy Module

Activity-based CO2e calculation and energy consumption tracking.

Formula: CO2e = Σ (Activity_k × EmissionFactor_k)

Activities tracked:
- Diesel consumption (L/t mined)
- Electricity (MWh for ventilation, hoist, mill)
- Explosives (kg)
- Cement/backfill materials (t)

Author: BlockModelViewer Team
Date: 2025-11-06
"""

import logging
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

from ...esg.dataclasses import EmissionFactor, EmissionSource

logger = logging.getLogger(__name__)


def calc_co2e(period_kpis: List, ef_df: pd.DataFrame, energy_mix: Dict) -> pd.DataFrame:
    """
    Calculate CO2e emissions for each period based on activities.
    
    Args:
        period_kpis: List of PeriodKPI objects with activity data
        ef_df: DataFrame with columns [source, fuel_or_energy, unit, ef_kgco2e_per_unit, scope]
        energy_mix: Dictionary with energy source percentages and factors
        
    Returns:
        DataFrame with columns [period, co2e_total, co2e_diesel, co2e_electricity, ...]
    """
    logger.info(f"Calculating CO2e for {len(period_kpis)} periods")
    
    # Convert emission factors to dict for fast lookup
    ef_dict = {}
    for _, row in ef_df.iterrows():
        key = (row['source'].lower(), row['fuel_or_energy'].lower())
        ef_dict[key] = row['ef_kgco2e_per_unit']
    
    results = []
    
    for kpi in period_kpis:
        co2e_breakdown = {}
        
        # Diesel emissions (mining equipment)
        # Assume 0.05 L/t ore mined as default
        diesel_l_per_t = energy_mix.get('diesel_l_per_t', 0.05)
        diesel_l = kpi.ore_mined * diesel_l_per_t
        co2e_diesel = diesel_l * ef_dict.get(('diesel', 'diesel'), 2.68)
        co2e_breakdown['diesel'] = co2e_diesel / 1000.0  # Convert kg to tonnes
        
        # Electricity emissions (ventilation, hoist, mill)
        # Assume 50 kWh/t ore processed as default
        kwh_per_t = energy_mix.get('kwh_per_t', 50.0)
        electricity_mwh = (kpi.ore_proc * kwh_per_t) / 1000.0
        if getattr(kpi, 'energy_mwh', 0) > 0:
            electricity_mwh = getattr(kpi, 'energy_mwh', 0)
        
        grid_ef = ef_dict.get(('electricity', 'grid'), 915.0)  # kg CO2e/MWh (default South Africa)
        co2e_electricity = electricity_mwh * grid_ef
        co2e_breakdown['electricity'] = co2e_electricity / 1000.0  # Convert kg to tonnes
        
        # Explosives emissions
        # Assume 0.15 kg explosives/t ore
        explosives_kg_per_t = energy_mix.get('explosives_kg_per_t', 0.15)
        explosives_kg = kpi.ore_mined * explosives_kg_per_t
        co2e_explosives = explosives_kg * ef_dict.get(('explosives', 'anfo'), 1.01)
        co2e_breakdown['explosives'] = co2e_explosives / 1000.0  # Convert kg to tonnes
        
        # Cement/backfill emissions
        # Assume cement in backfill
        fill_placed = getattr(kpi, 'fill_placed', 0)
        if fill_placed > 0:
            cement_fraction = energy_mix.get('cement_fraction_in_fill', 0.10)
            cement_t = fill_placed * cement_fraction
            co2e_cement = cement_t * ef_dict.get(('cement', 'gbfs'), 650.0)
            co2e_breakdown['cement'] = co2e_cement / 1000.0  # Convert kg to tonnes
        else:
            co2e_breakdown['cement'] = 0.0
        
        # Total CO2e
        total_co2e = sum(co2e_breakdown.values())
        
        # CO2e intensities
        co2e_per_t_ore = (total_co2e * 1000.0) / kpi.ore_mined if kpi.ore_mined > 0 else 0.0
        co2e_per_t_product = (total_co2e * 1000.0) / kpi.ore_proc if kpi.ore_proc > 0 else 0.0
        
        results.append({
            'period': kpi.t,
            'activity': getattr(kpi, 'activity', 'total'),
            'co2e_tonnes': total_co2e,
            'co2e_total_t': total_co2e,
            'co2e_per_t_ore': co2e_per_t_ore,
            'co2e_per_t_product': co2e_per_t_product,
            **{f'co2e_{k}_t': v for k, v in co2e_breakdown.items()}
        })
    
    logger.info(f"Total CO2e: {sum(r['co2e_total_t'] for r in results):.0f} tonnes")
    return pd.DataFrame(results)


def create_default_emission_factors() -> pd.DataFrame:
    """
    Create default emission factors DataFrame.
    
    Returns:
        DataFrame with standard emission factors
    """
    factors = [
        {'source': 'diesel', 'fuel_or_energy': 'Diesel', 'unit': 'L', 'ef_kgco2e_per_unit': 2.68, 'scope': 1},
        {'source': 'electricity', 'fuel_or_energy': 'Grid', 'unit': 'MWh', 'ef_kgco2e_per_unit': 915.0, 'scope': 2},  # South Africa grid
        {'source': 'electricity', 'fuel_or_energy': 'Solar', 'unit': 'MWh', 'ef_kgco2e_per_unit': 45.0, 'scope': 2},
        {'source': 'explosives', 'fuel_or_energy': 'ANFO', 'unit': 'kg', 'ef_kgco2e_per_unit': 1.01, 'scope': 3},
        {'source': 'cement', 'fuel_or_energy': 'OPC', 'unit': 't', 'ef_kgco2e_per_unit': 900.0, 'scope': 3},
        {'source': 'cement', 'fuel_or_energy': 'GBFS', 'unit': 't', 'ef_kgco2e_per_unit': 650.0, 'scope': 3},
        {'source': 'lime', 'fuel_or_energy': 'Quicklime', 'unit': 't', 'ef_kgco2e_per_unit': 785.0, 'scope': 3},
    ]
    return pd.DataFrame(factors)
