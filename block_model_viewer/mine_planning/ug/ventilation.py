"""Placeholder ventilation design utilities.

Provides a simple main fan sizing function.
"""
from __future__ import annotations
from typing import Dict


def design_main_fan(
    required_airflow: float,
    total_resistance: float,
    efficiency: float = 0.75,
    pressure_margin: float = 1.10,
) -> Dict:
    """Return a simple fan design dictionary.

    Uses fan power ~ (Flow * Pressure) / efficiency. Pressure estimated from
    total resistance * flow^2 (placeholder relationship) and then inflated by
    a margin.
    """
    flow = max(required_airflow, 1.0)  # m^3/s
    resistance = max(total_resistance, 0.001)
    pressure_pa = resistance * (flow ** 2) * pressure_margin * 100.0  # placeholder scaling
    power_kw = (flow * pressure_pa) / (efficiency * 3600.0)

    return {
        "design_flow_m3_s": flow,
        "static_pressure_pa": pressure_pa,
        "fan_power_kw": power_kw,
        "efficiency": efficiency,
        "margin": pressure_margin,
    }
