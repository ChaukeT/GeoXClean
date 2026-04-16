"""
Mining-convention rounding for resource statements.

Applies industry-standard rounding to tonnes, grade, and metal content
so that reported figures are appropriate for public disclosure.

Metal is ALWAYS recalculated from rounded tonnes and grade to avoid
reconciliation errors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ── Element classification ──────────────────────────────────────────

# Precious metals: grade in g/t, metal in troy ounces (koz)
PRECIOUS_METALS = {"Au", "Ag", "Pt", "Pd", "Rh", "Ir", "Os", "Ru"}

# Base metals: grade in %, metal in tonnes
BASE_METALS = {"Cu", "Zn", "Pb", "Ni", "Co", "Sn", "Mo", "W", "Bi"}

# Bulk commodities: grade in %, metal in tonnes
BULK_METALS = {"Fe", "Mn", "Cr", "Al", "Ti", "V", "P", "S"}

# Lithium / rare: ppm or %, user-specified
RARE_METALS = {"Li", "Nb", "Ta", "REE", "U", "Th", "Be"}

TROY_OZ_PER_GRAM = 1.0 / 31.1035


def detect_grade_unit(element_name: str) -> str:
    """Return the most likely grade unit for an element.

    Returns one of: "pct", "g/t", "ppm".
    """
    name = element_name.strip().split("_")[0]  # handle "Cu_pct", "Au_gpt" etc.
    if name in PRECIOUS_METALS:
        return "g/t"
    if name in BASE_METALS or name in BULK_METALS:
        return "pct"
    if name in RARE_METALS:
        return "ppm"
    return "pct"  # default


def detect_metal_unit(element_name: str) -> str:
    """Return the reporting unit for contained metal.

    Returns one of: "t", "kt", "koz", "kg".
    """
    name = element_name.strip().split("_")[0]
    if name in PRECIOUS_METALS:
        return "koz"
    return "t"


# ── Rounding functions ──────────────────────────────────────────────

def round_tonnes(tonnes: float) -> float:
    """Round tonnes to mining-convention precision.

    - <100 kt: nearest 1,000 t
    - 100 kt to 10 Mt: nearest 10,000 t
    - >10 Mt: nearest 100,000 t
    """
    if tonnes <= 0:
        return 0.0
    if tonnes < 100_000:
        return round(tonnes / 1_000) * 1_000
    if tonnes < 10_000_000:
        return round(tonnes / 10_000) * 10_000
    return round(tonnes / 100_000) * 100_000


def round_grade(grade: float, unit: str = "pct") -> float:
    """Round grade to mining-convention precision.

    - pct commodities: 2 decimal places
    - g/t: 2 dp for <10 g/t, 1 dp for >=10 g/t
    - ppm: 0 decimal places
    """
    if unit == "pct":
        return round(grade, 2)
    elif unit == "g/t":
        return round(grade, 2) if grade < 10.0 else round(grade, 1)
    elif unit == "ppm":
        return round(grade, 0)
    return round(grade, 2)


def compute_metal(
    tonnes_t: float,
    grade: float,
    grade_unit: str = "pct",
    element_name: str = "",
) -> float:
    """Compute contained metal from (possibly rounded) tonnes and grade.

    Returns metal in the natural unit for the element:
    - pct: tonnes of metal
    - g/t precious: troy ounces
    - g/t base: kg
    - ppm: tonnes of metal
    """
    if grade_unit == "pct":
        return tonnes_t * grade / 100.0
    elif grade_unit == "g/t":
        grams = tonnes_t * grade  # total grams
        if element_name.strip().split("_")[0] in PRECIOUS_METALS:
            return grams * TROY_OZ_PER_GRAM  # troy ounces
        return grams / 1000.0  # kg
    elif grade_unit == "ppm":
        return tonnes_t * grade / 1e6
    return tonnes_t * grade / 100.0


def round_metal(metal: float, metal_unit: str = "t") -> float:
    """Round contained metal to mining-convention precision."""
    if metal_unit == "koz":
        # Troy ounces: report in koz, round to nearest 1,000 oz for large deposits
        if metal > 100_000:
            return round(metal / 1_000) * 1_000
        return round(metal, 0)
    # Tonnes of metal — match tonnage rounding
    return round_tonnes(metal)


@dataclass
class RoundedRow:
    """A resource row with mining-convention rounding applied."""
    classification: str
    tonnes_raw: float
    grade_raw: float
    metal_raw: float
    tonnes_rounded: float
    grade_rounded: float
    metal_rounded: float  # computed FROM rounded figures
    grade_unit: str
    metal_unit: str


def apply_mining_rounding(
    classification: str,
    tonnes_t: float,
    grade: float,
    grade_unit: str = "pct",
    element_name: str = "",
) -> RoundedRow:
    """Apply full mining-convention rounding to a single row.

    Metal is ALWAYS computed from rounded tonnes × rounded grade.
    """
    metal_unit = detect_metal_unit(element_name)

    t_rounded = round_tonnes(tonnes_t)
    g_rounded = round_grade(grade, grade_unit)
    m_rounded = compute_metal(t_rounded, g_rounded, grade_unit, element_name)

    # Also compute raw metal for comparison
    m_raw = compute_metal(tonnes_t, grade, grade_unit, element_name)

    return RoundedRow(
        classification=classification,
        tonnes_raw=tonnes_t,
        grade_raw=grade,
        metal_raw=m_raw,
        tonnes_rounded=t_rounded,
        grade_rounded=g_rounded,
        metal_rounded=m_rounded,
        grade_unit=grade_unit,
        metal_unit=metal_unit,
    )


def format_tonnes(tonnes: float) -> str:
    """Format tonnes with appropriate units."""
    if tonnes < 100_000:
        return f"{tonnes:,.0f} t"
    if tonnes < 10_000_000:
        return f"{tonnes / 1_000:,.0f} kt"
    return f"{tonnes / 1_000_000:,.1f} Mt"


def format_metal(metal: float, metal_unit: str = "t") -> str:
    """Format metal content with appropriate units."""
    if metal_unit == "koz":
        if metal >= 1_000_000:
            return f"{metal / 1_000:,.0f} koz"
        return f"{metal:,.0f} oz"
    return format_tonnes(metal)
