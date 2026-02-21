"""
DRILLHOLE IGNORE ENGINE (GeoX)

Suppresses minor issues & warnings based on user-defined rules.

This engine does NOT delete violations.
It hides them from QCWindow and AutoFix processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from .drillhole_validation import ValidationViolation


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class IgnoreRule:
    """
    A rule describing what to ignore.
    Any field = None means wildcard.
    """
    rule_code: Optional[str] = None
    hole_id: Optional[str] = None
    severity: Optional[str] = None  # ERROR / WARNING / INFO
    row_index: Optional[int] = None


@dataclass
class IgnoreResult:
    visible: List[ValidationViolation]
    ignored: List[ValidationViolation]


# =========================================================
# MATCHING LOGIC
# =========================================================

def _matches(rule: IgnoreRule, v: ValidationViolation) -> bool:
    """
    A violation matches a rule if all specified (non-None)
    fields match exactly.
    """
    if rule.rule_code is not None and rule.rule_code != v.rule_code:
        return False
    if rule.hole_id is not None and rule.hole_id != v.hole_id:
        return False
    if rule.severity is not None and rule.severity != v.severity:
        return False
    if rule.row_index is not None and rule.row_index != v.row_index:
        return False
    return True


# =========================================================
# IGNORE ENGINE
# =========================================================

def apply_ignore_rules(
    violations: List[ValidationViolation],
    ignore_rules: List[IgnoreRule],
    ignore_all_minor: bool = False,
    ignore_all_warnings: bool = False,
) -> IgnoreResult:
    """
    Filters violations by suppression rules.

    ignore_all_minor:
        Removes all INFO + WARNING automatically.

    ignore_all_warnings:
        Removes all WARNING automatically.

    ignore_rules:
        User-defined ignore entries.
    """
    visible = []
    ignored = []

    for v in violations:

        # 1) Global ignore-all-minor mode
        if ignore_all_minor and v.severity in ("INFO", "WARNING"):
            ignored.append(v)
            continue

        # 2) Global ignore-all-warnings mode
        if ignore_all_warnings and v.severity == "WARNING":
            ignored.append(v)
            continue

        # 3) Specific ignore rules
        suppressed = False
        for r in ignore_rules:
            if _matches(r, v):
                ignored.append(v)
                suppressed = True
                break

        if not suppressed:
            visible.append(v)

    return IgnoreResult(visible=visible, ignored=ignored)

