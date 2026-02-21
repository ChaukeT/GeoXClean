"""
Pytest configuration for GeoX Panel Debugger

This module configures pytest for testing all panels in GeoX Clean.
It sets up:
- Python path to include block_model_viewer
- Qt environment for headless testing
- Logging configuration
- Test markers for categorization
"""

import sys
import os
import logging
from pathlib import Path

import pytest

# Add block_model_viewer to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging to capture errors during tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'tests' / 'panel_debugger' / 'reports' / 'test.log'),
        logging.StreamHandler()
    ]
)

# Configure Qt for headless testing
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "panel_init: Tests for panel instantiation"
    )
    config.addinivalue_line(
        "markers", "signals: Tests for signal connections"
    )
    config.addinivalue_line(
        "markers", "data_flow: Tests for data flow through registry"
    )
    config.addinivalue_line(
        "markers", "renderer: Tests for renderer integration"
    )
    config.addinivalue_line(
        "markers", "coordinates: Tests for coordinate system alignment"
    )
    config.addinivalue_line(
        "markers", "performance: Tests for performance and memory"
    )
    config.addinivalue_line(
        "markers", "integration: End-to-end integration tests"
    )
    config.addinivalue_line(
        "markers", "critical: Critical tests that must pass"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file names"""
    for item in items:
        # Add marker based on test file name
        if "test_panel_init" in item.nodeid:
            item.add_marker(pytest.mark.panel_init)
        elif "test_signals" in item.nodeid:
            item.add_marker(pytest.mark.signals)
            item.add_marker(pytest.mark.critical)  # Signal tests are critical
        elif "test_data_flow" in item.nodeid:
            item.add_marker(pytest.mark.data_flow)
        elif "test_renderer" in item.nodeid:
            item.add_marker(pytest.mark.renderer)
        elif "test_coordinates" in item.nodeid:
            item.add_marker(pytest.mark.coordinates)
            item.add_marker(pytest.mark.critical)  # Coordinate tests are critical
        elif "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
            item.add_marker(pytest.mark.critical)  # Integration tests are critical


@pytest.fixture(scope="session")
def project_root():
    """Provide path to project root directory"""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def panel_debugger_root():
    """Provide path to panel_debugger directory"""
    return PROJECT_ROOT / 'tests' / 'panel_debugger'


# Import fixtures from panel_debugger
pytest_plugins = ['tests.panel_debugger.core.fixtures']
