"""
Helper utilities for detecting and normalizing coordinate columns.

The goal is to provide consistent X/Y/Z column names regardless of how the
source data labels them (Easting/Northing/RL, lowercase variants, columns with
units appended, etc.).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)


COORD_ALIAS_MAP: Dict[str, list[str]] = {
    "X": ["x", "xc", "xcenter", "xcentre", "xcentroid", "xmid", "xorig", "e", "east", "easting", "eastings"],
    "Y": ["y", "yc", "ycenter", "ycentre", "ycentroid", "ymid", "yorig", "n", "north", "northing", "northings"],
    "Z": [
        "z", "zc", "zcenter", "zcentre", "zcentroid", "zmid", "rl", "reducedlevel",
        "elev", "elevation", "height", "altitude"
    ],
}


def _sanitize_column_name(name: str) -> str:
    """Return a lowercase string containing only alphanumeric characters."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _alias_matches(normalized_col: str, normalized_alias: str) -> bool:
    """Return True when the alias appears to represent the column name."""
    if not normalized_alias:
        return False
    # Single-letter aliases (e/n/x/y/z) must match exactly to avoid false positives (e.g., FE -> X)
    if len(normalized_alias) == 1:
        return normalized_col == normalized_alias
    if normalized_col == normalized_alias:
        return True
    if normalized_col.startswith(normalized_alias):
        return True
    if normalized_col.endswith(normalized_alias):
        return True
    if len(normalized_alias) > 2 and normalized_alias in normalized_col:
        return True
    return False


def detect_coordinate_columns(columns: Iterable[str]) -> Dict[str, Optional[str]]:
    """
    Analyze a list of column names and map them to canonical X/Y/Z names.

    Returns a dictionary with keys "X", "Y", and "Z" whose values are either the
    detected original column name or None if no match was found.
    """
    columns = list(columns)
    normalized_map = {col: _sanitize_column_name(str(col)) for col in columns}
    detected: Dict[str, Optional[str]] = {}
    used_columns: set[str] = set()
    uppercase_map = {col.upper(): col for col in columns}

    for target, aliases in COORD_ALIAS_MAP.items():
        candidate: Optional[str] = None
        if target in uppercase_map:
            candidate = uppercase_map[target]
            used_columns.add(candidate)
            detected[target] = candidate
            continue

        for alias in aliases:
            alias_norm = _sanitize_column_name(alias)
            for col, col_norm in normalized_map.items():
                if col in used_columns:
                    continue
                if _alias_matches(col_norm, alias_norm):
                    candidate = col
                    used_columns.add(col)
                    break
            if candidate:
                break

        detected[target] = candidate

    return detected


def ensure_xyz_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns to X/Y/Z when recognizable coordinate aliases are present.

    Returns a new DataFrame when renaming occurs; otherwise returns the original
    DataFrame.
    """
    if df is None:
        return df

    rename_map: Dict[str, str] = {}
    detected = detect_coordinate_columns(df.columns)

    for target, column in detected.items():
        if column and column != target:
            rename_map[column] = target

    if rename_map:
        logger.debug("Renaming coordinate columns: %s", rename_map)
        return df.rename(columns=rename_map)

    return df

