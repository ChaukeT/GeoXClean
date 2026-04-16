"""
Variable unit utilities — format labels, axes, and values with units.

Usage:
    from ..utils.unit_utils import var_label, val_with_unit

    label = var_label(registry, "Cu")          # "Cu (%)" or "Cu"
    text  = val_with_unit(1.23, registry, "Cu") # "1.23 %" or "1.23"
"""

from __future__ import annotations

from typing import Optional


def var_label(registry, variable: str) -> str:
    """Return 'Variable (unit)' or just 'Variable'."""
    if registry is not None and hasattr(registry, "format_variable_label"):
        return registry.format_variable_label(variable)
    return variable


def val_with_unit(value, registry, variable: str, fmt: str = ".4f") -> str:
    """Format a numeric value with the variable's unit appended."""
    unit = ""
    if registry is not None and hasattr(registry, "get_variable_unit"):
        unit = registry.get_variable_unit(variable)
    formatted = f"{value:{fmt}}" if isinstance(value, (int, float)) else str(value)
    if unit:
        return f"{formatted} {unit}"
    return formatted


# Common mining variable units (auto-detect hints)
_COMMON_UNITS = {
    "au": "g/t",
    "ag": "g/t",
    "pt": "g/t",
    "pd": "g/t",
    "cu": "%",
    "zn": "%",
    "pb": "%",
    "ni": "%",
    "co": "%",
    "fe": "%",
    "al2o3": "%",
    "sio2": "%",
    "mgo": "%",
    "cao": "%",
    "s": "%",
    "p": "%",
    "mn": "%",
    "tio2": "%",
    "k2o": "%",
    "na2o": "%",
    "loi": "%",
    "density": "t/m³",
    "sg": "t/m³",
    "rd": "t/m³",
}


def guess_unit(variable: str) -> str:
    """Guess the unit from common mining variable names.

    Returns empty string if unknown.
    """
    key = variable.lower().replace("_pct", "").replace("_ppm", "").replace("_pct_ns", "").replace("_ns", "").strip()
    if "_ppm" in variable.lower():
        return "ppm"
    if "_pct" in variable.lower():
        return "%"
    return _COMMON_UNITS.get(key, "")
