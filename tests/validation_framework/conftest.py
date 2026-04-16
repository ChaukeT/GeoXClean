"""
Shared fixtures for GeoX Validation Framework tests.

Provides synthetic drillhole data, registry instances, and
helper functions used across all test domains.
"""
import sys
import os
import copy
import logging
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

logger = logging.getLogger("geox.tests")


def pytest_configure(config):
    for marker, desc in [
        ("smoke", "Quick smoke tests for CI (<5 min total)"),
        ("unit", "Unit tests for individual components"),
        ("integration", "Multi-module integration tests"),
        ("numerical", "Numerical accuracy / golden-dataset tests"),
        ("blocker", "BLOCKER severity — must pass for any release"),
        ("critical", "CRITICAL severity — blocks release unless waived"),
        ("major", "MAJOR severity — fix within next cycle"),
        ("minor", "MINOR severity — optional"),
        ("data_integrity", "DI-xx checks"),
        ("drillholes", "DH-xx checks"),
        ("block_model", "BM-xx checks"),
        ("geostatistics", "GS-xx checks"),
        ("compositing", "CV-xx checks"),
        ("plotting", "PL-xx checks"),
        ("system_integrity", "SI-xx checks"),
        ("workflow", "WF-xx checks"),
        ("export", "ER-xx checks"),
    ]:
        config.addinivalue_line("markers", f"{marker}: {desc}")


# ───────────────────────── Synthetic Drillhole Data ─────────────────────────

@pytest.fixture
def sample_collars():
    return pd.DataFrame({
        "hole_id": ["DH001", "DH002", "DH003"],
        "x": [1000.0, 1050.0, 1100.0],
        "y": [2000.0, 2000.0, 2000.0],
        "z": [500.0, 510.0, 505.0],
        "total_depth": [200.0, 150.0, 180.0],
    })


@pytest.fixture
def sample_surveys():
    rows = []
    for hid, td in [("DH001", 200), ("DH002", 150), ("DH003", 180)]:
        for d in range(0, td + 1, 30):
            rows.append({"hole_id": hid, "depth": min(d, td), "azimuth": 90.0, "dip": -60.0})
    return pd.DataFrame(rows).drop_duplicates(subset=["hole_id", "depth"])


@pytest.fixture
def sample_assays():
    rows = []
    for hid, td in [("DH001", 200), ("DH002", 150), ("DH003", 180)]:
        for start in range(0, td, 2):
            end = min(start + 2, td)
            rows.append({
                "hole_id": hid, "depth_from": float(start), "depth_to": float(end),
                "Fe": np.random.uniform(30, 65), "SiO2": np.random.uniform(2, 12),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_lithology():
    rows = []
    for hid, td in [("DH001", 200), ("DH002", 150), ("DH003", 180)]:
        for start in range(0, td, 10):
            end = min(start + 10, td)
            rows.append({
                "hole_id": hid, "depth_from": float(start), "depth_to": float(end),
                "lith_code": np.random.choice(["BIF", "SHALE", "DOLERITE"]),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def overlapping_assays():
    """Assays with deliberate overlaps for testing DI/CV overlap detection."""
    return pd.DataFrame({
        "hole_id": ["DH001"] * 5,
        "depth_from": [0.0, 2.0, 3.5, 6.0, 8.0],
        "depth_to": [2.0, 4.0, 6.0, 8.0, 10.0],
        "Fe": [55.0, 60.0, 58.0, 52.0, 50.0],
    })


@pytest.fixture
def negative_grade_assays():
    """Assays with negative grades (possible missing-value codes)."""
    return pd.DataFrame({
        "hole_id": ["DH001"] * 3,
        "depth_from": [0.0, 2.0, 4.0],
        "depth_to": [2.0, 4.0, 6.0],
        "Fe": [55.0, -999.0, 50.0],
    })
