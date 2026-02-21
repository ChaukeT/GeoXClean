"""
Underground Equipment Scheduling Module

Resource-constrained scheduling for underground mining equipment.
"""

from .scheduler import (
    Equipment,
    EquipmentType,
    MaintenanceSchedule,
    EquipmentAssignment,
    schedule_equipment,
    calculate_equipment_requirements,
    optimize_fleet_size
)

__all__ = [
    'Equipment',
    'EquipmentType',
    'MaintenanceSchedule',
    'EquipmentAssignment',
    'schedule_equipment',
    'calculate_equipment_requirements',
    'optimize_fleet_size'
]
