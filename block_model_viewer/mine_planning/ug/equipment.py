"""Placeholder equipment calculations for underground mining.

This module provides a simple estimator for equipment fleet size based on
throughput, days, and hauling distance. This is a placeholder and should be
replaced with a calibrated model.
"""
from __future__ import annotations
from typing import List, Dict


def calculate_equipment_requirements(
    total_tonnes: float,
    total_days: int,
    avg_haul_distance_m: float = 500.0,
    period_hours_per_day: float = 16.0,
) -> List[Dict]:
    """Return a rough set of equipment requirements.

    The logic uses very simple productivity assumptions:
    - Loader productivity ~ 250 tph (effective)
    - Truck productivity ~ 80 tph baseline adjusted by haul distance
    - Bolter/Drill counts are minimal placeholders
    """
    hours = max(total_days, 1) * max(period_hours_per_day, 1.0)
    required_tph = total_tonnes / max(hours, 1.0)

    loaders_tph = 250.0
    trucks_tph = 80.0 * (500.0 / max(avg_haul_distance_m, 1.0)) ** 0.3

    n_loaders = int(max(1, round(required_tph / loaders_tph)))
    n_trucks = int(max(2, round(required_tph / trucks_tph * 1.5)))

    equipment = [
        {"type": "Loader (LHD)", "count": n_loaders, "utilization": 0.75},
        {"type": "Truck", "count": n_trucks, "utilization": 0.7},
        {"type": "Jumbo Drill", "count": 2, "utilization": 0.6},
        {"type": "Bolter", "count": 1, "utilization": 0.5},
    ]
    return equipment
