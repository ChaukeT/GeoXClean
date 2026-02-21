"""
Pytest fixtures for panel testing

This module provides reusable fixtures for testing GeoX panels:
- mock_qapp: QApplication instance for Qt tests
- mock_registry: Mock DataRegistry with signal tracking
- mock_renderer: Mock renderer without GPU (headless)
- mock_signals: Mock UISignals bus
- panel_manifest: Panel metadata from JSON
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, MagicMock
import json

import pytest
import numpy as np

# Import Qt components
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, pyqtSignal
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False

# Import PyVista for mock renderer
try:
    import pyvista as pv
    pv.OFF_SCREEN = True  # Enable headless mode
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False


# ============================================================================
# Qt Application Fixture
# ============================================================================

@pytest.fixture(scope="session")
def mock_qapp():
    """
    Provide QApplication instance for Qt tests.

    Scope: session (created once, reused across all tests)
    Cleanup: Proper Qt shutdown after all tests complete
    """
    if not QT_AVAILABLE:
        pytest.skip("PyQt6 not available")

    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    yield app

    # Cleanup (session scope - runs after all tests)
    # Note: QApplication cleanup is handled automatically


# ============================================================================
# Mock DataRegistry
# ============================================================================

class MockSignalEmitter(QObject if QT_AVAILABLE else object):
    """Mock version of DataRegistry's _SignalEmitter"""

    if QT_AVAILABLE:
        # Drillhole signals
        drillholeDataLoaded = pyqtSignal(object)
        drillholeDataCleared = pyqtSignal()

        # Block model signals
        blockModelLoaded = pyqtSignal(object)
        blockModelGenerated = pyqtSignal(object)
        blockModelClassified = pyqtSignal(object)
        blockModelCleared = pyqtSignal()
        blockModelLoadedEx = pyqtSignal(str, object)
        blockModelGeneratedEx = pyqtSignal(str, object)
        currentBlockModelChanged = pyqtSignal(str)

        # Domain/geology signals
        domainModelLoaded = pyqtSignal(object)
        geologicalModelUpdated = pyqtSignal(object)
        geologicalSurfacesLoaded = pyqtSignal(object)
        geologicalSolidsLoaded = pyqtSignal(object)
        loopstructuralModelLoaded = pyqtSignal(object)

        # Estimation signals
        variogramResultsLoaded = pyqtSignal(object)
        declusteringResultsLoaded = pyqtSignal(object)
        transformationMetadataLoaded = pyqtSignal(object)
        krigingResultsLoaded = pyqtSignal(object)
        simpleKrigingResultsLoaded = pyqtSignal(object)
        cokrigingResultsLoaded = pyqtSignal(object)
        indicatorKrigingResultsLoaded = pyqtSignal(object)
        universalKrigingResultsLoaded = pyqtSignal(object)
        softKrigingResultsLoaded = pyqtSignal(object)
        rbfResultsLoaded = pyqtSignal(object)
        sgsimResultsLoaded = pyqtSignal(object)

        # Other signals
        geometResultsLoaded = pyqtSignal(object)
        resourceCalculated = pyqtSignal(object)
        pitOptimizationResultsLoaded = pyqtSignal(object)
        scheduleGenerated = pyqtSignal(object)


class MockDataRegistry:
    """
    Mock DataRegistry for testing.

    Features:
    - Tracks all signal connections
    - Records signal emissions
    - Provides data storage/retrieval
    - No actual database or file I/O
    """

    _instance = None

    def __init__(self):
        self._data_store = {}
        self._signal_connections = []
        self._signal_emissions = []

        if QT_AVAILABLE:
            self.signals = MockSignalEmitter()
        else:
            self.signals = Mock()

        # Track method calls
        self.method_calls = []

    @classmethod
    def instance(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton for testing"""
        cls._instance = None

    # Data storage methods
    def set_drillhole_data(self, data: Dict[str, Any]):
        """Store drillhole data and emit signal"""
        self._data_store['drillhole_data'] = data
        self.method_calls.append(('set_drillhole_data', data))
        if QT_AVAILABLE and hasattr(self.signals, 'drillholeDataLoaded'):
            self.signals.drillholeDataLoaded.emit(data)
            self._signal_emissions.append(('drillholeDataLoaded', data))

    def get_drillhole_data(self, copy_data=True):
        """Retrieve drillhole data"""
        self.method_calls.append(('get_drillhole_data', copy_data))
        return self._data_store.get('drillhole_data')

    def register_block_model(self, block_model, source_panel=None, model_id=None, set_as_current=True):
        """Store block model and emit signal"""
        if model_id is None:
            model_id = f"block_model_{len(self._data_store)}"

        self._data_store[f'block_model_{model_id}'] = block_model
        self._data_store['current_block_model_id'] = model_id
        self.method_calls.append(('register_block_model', block_model, source_panel, model_id))

        if QT_AVAILABLE and hasattr(self.signals, 'blockModelLoaded'):
            self.signals.blockModelLoaded.emit(block_model)
            self.signals.blockModelLoadedEx.emit(model_id, block_model)
            self._signal_emissions.append(('blockModelLoaded', block_model))

    def get_block_model(self, model_id=None):
        """Retrieve block model"""
        if model_id is None:
            model_id = self._data_store.get('current_block_model_id')

        self.method_calls.append(('get_block_model', model_id))
        return self._data_store.get(f'block_model_{model_id}')

    def register_variogram_results(self, results, source_panel=None, metadata=None):
        """Store variogram results"""
        self._data_store['variogram_results'] = results
        self.method_calls.append(('register_variogram_results', results))
        if QT_AVAILABLE and hasattr(self.signals, 'variogramResultsLoaded'):
            self.signals.variogramResultsLoaded.emit(results)

    def get_variogram_results(self):
        """Retrieve variogram results"""
        return self._data_store.get('variogram_results')

    def get(self, key, default=None):
        """Generic get method"""
        return self._data_store.get(key, default)

    def store(self, key, value):
        """Generic store method"""
        self._data_store[key] = value

    # Signal tracking
    def track_connection(self, signal_name, slot):
        """Track signal connection (called by tests)"""
        self._signal_connections.append((signal_name, slot))

    def was_signal_emitted(self, signal_name):
        """Check if signal was emitted"""
        return any(s[0] == signal_name for s in self._signal_emissions)

    def get_signal_emissions(self, signal_name=None):
        """Get all emissions for a signal"""
        if signal_name:
            return [s for s in self._signal_emissions if s[0] == signal_name]
        return self._signal_emissions


@pytest.fixture
def mock_registry(mock_qapp):
    """
    Provide mock DataRegistry for testing.

    Features:
    - Pre-populated with test signals
    - Tracks signal connections and emissions
    - Cleans up between tests
    """
    # Reset singleton
    MockDataRegistry.reset()

    # Create new instance
    registry = MockDataRegistry.instance()

    yield registry

    # Cleanup
    MockDataRegistry.reset()


# ============================================================================
# Mock Renderer
# ============================================================================

class MockRenderer:
    """
    Mock renderer for testing without GPU.

    Features:
    - Uses PyVista off-screen mode
    - Tracks method calls (add_drillhole_layer, add_block_model_layer)
    - Simulates _to_local_precision() coordinate transformation
    - No actual rendering (headless)
    - Maintains active_layers dict like real renderer
    """

    def __init__(self):
        self.active_layers = {}
        self.scene_layers = {}
        self.method_calls = []
        self._has_large_model = False
        self._coordinate_offset = None

        # Create plotter in off-screen mode
        if PYVISTA_AVAILABLE:
            self.plotter = pv.Plotter(off_screen=True)
        else:
            self.plotter = Mock()

    def add_drillhole_layer(self, drillhole_data, *args, **kwargs):
        """Mock drillhole layer addition"""
        self.method_calls.append(('add_drillhole_layer', args, kwargs))

        # Extract coordinates for transformation
        coords = None
        if isinstance(drillhole_data, dict):
            if 'trajectories' in drillhole_data:
                traj = drillhole_data['trajectories']
                if hasattr(traj, 'points'):
                    coords = traj.points
                elif isinstance(traj, np.ndarray):
                    coords = traj

        # Simulate layer creation
        self.active_layers['drillholes'] = {
            'data': drillhole_data,
            'layer_type': 'drillholes',
            'visible': True,
            'coords': coords
        }

        return True

    def add_block_model_layer(self, block_model, *args, **kwargs):
        """Mock block model layer addition"""
        self.method_calls.append(('add_block_model_layer', args, kwargs))

        # Check model size
        n_cells = 0
        if hasattr(block_model, '__len__'):
            n_cells = len(block_model)
        elif hasattr(block_model, 'n_cells'):
            n_cells = block_model.n_cells

        if n_cells > 30000:
            self._has_large_model = True

        # Simulate layer creation
        self.active_layers['blocks'] = {
            'data': block_model,
            'layer_type': 'blocks',
            'visible': True
        }

        return True

    def add_geological_layer(self, geo_data, layer_name, *args, **kwargs):
        """Mock geological layer addition"""
        self.method_calls.append(('add_geological_layer', layer_name, args, kwargs))

        self.active_layers[layer_name] = {
            'data': geo_data,
            'layer_type': 'geology',
            'visible': True
        }

        return True

    def _to_local_precision(self, coords):
        """
        Mock coordinate transformation.

        Simulates centering coordinates around origin by subtracting centroid.
        This matches the real renderer's behavior for UTM → local conversion.
        """
        if coords is None or len(coords) == 0:
            return coords

        coords = np.asarray(coords)

        # Calculate and store offset on first call
        if self._coordinate_offset is None:
            centroid = np.mean(coords, axis=0)
            self._coordinate_offset = centroid

        # Apply offset
        return coords - self._coordinate_offset

    def was_method_called(self, method_name):
        """Check if method was called"""
        return any(call[0] == method_name for call in self.method_calls)

    def get_method_call_args(self, method_name):
        """Get arguments from method call"""
        for call in self.method_calls:
            if call[0] == method_name:
                return call[1], call[2]  # args, kwargs
        return None, None

    def get_layer_coordinates(self, layer_name):
        """Get coordinates for a layer"""
        if layer_name not in self.active_layers:
            return None

        layer = self.active_layers[layer_name]
        data = layer['data']

        # Extract coordinates based on data type
        if hasattr(data, 'points'):
            return data.points
        elif isinstance(data, dict) and 'coords' in layer:
            return layer['coords']

        return None


@pytest.fixture
def mock_renderer(mock_qapp):
    """
    Provide mock renderer for testing.

    Features:
    - Off-screen PyVista rendering
    - Method call tracking
    - Coordinate transformation simulation
    - No GPU required
    """
    if not PYVISTA_AVAILABLE:
        pytest.skip("PyVista not available")

    renderer = MockRenderer()
    yield renderer

    # Cleanup
    if hasattr(renderer.plotter, 'close'):
        renderer.plotter.close()


# ============================================================================
# Mock UISignals
# ============================================================================

@pytest.fixture
def mock_signals(mock_qapp):
    """
    Provide mock UISignals bus for testing.

    All signals defined but not connected to actual components.
    """
    if not QT_AVAILABLE:
        pytest.skip("PyQt6 not available")

    signals = Mock()

    # Add signal attributes
    signals.drillholeDataLoaded = Mock()
    signals.blockModelLoaded = Mock()
    signals.variogramResultsLoaded = Mock()

    return signals


# ============================================================================
# Panel Manifest Fixture
# ============================================================================

@pytest.fixture(scope="session")
def panel_manifest(panel_debugger_root):
    """
    Load panel manifest from JSON.

    Scope: session (load once, reuse across all tests)
    """
    manifest_path = panel_debugger_root / 'config' / 'panel_manifest.json'

    if not manifest_path.exists():
        pytest.skip(f"Panel manifest not found: {manifest_path}")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    return manifest


# ============================================================================
# Helper Fixtures
# ============================================================================

@pytest.fixture
def capture_qt_signals():
    """
    Fixture to capture Qt signals during tests.

    Usage:
        def test_signal(capture_qt_signals):
            with capture_qt_signals(my_object.my_signal) as captured:
                # Trigger signal
                my_object.do_something()

            assert len(captured) == 1
            assert captured[0] == expected_value
    """
    from contextlib import contextmanager

    @contextmanager
    def _capture(signal):
        received = []

        def handler(*args):
            received.append(args if len(args) > 1 else args[0] if args else None)

        signal.connect(handler)
        try:
            yield received
        finally:
            signal.disconnect(handler)

    return _capture
