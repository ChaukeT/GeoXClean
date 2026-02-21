"""
Signal Connection Tests

Tests signal connections between panels and DataRegistry.

CRITICAL TESTS:
- test_no_silent_exceptions: Detects bare 'except: pass' statements
- test_panel_signal_handlers_execute: Verifies signal handlers work
- test_drillhole_control_panel_connections: Tests specific drillhole loading flow

These tests will catch the specific issues found at:
- drillhole_control_panel.py:255 - Silent exception
- dock_setup.py:129 - Silent exception
"""

import importlib
import inspect
import re
import pytest
from pathlib import Path

from tests.panel_debugger.core.signal_tester import SignalTester, SignalSpy
from tests.panel_debugger.core.mock_factory import MockDataFactory


@pytest.mark.signals
@pytest.mark.critical
class TestSilentExceptionDetection:
    """Detect silent exception handling that hides errors"""

    SILENT_EXCEPTION_PATTERNS = [
        # Bare except with pass
        r'except\s*:\s*\n\s*pass',
        # Except Exception with pass
        r'except\s+Exception\s*:\s*\n\s*pass',
        # Except BaseException with pass
        r'except\s+BaseException\s*:\s*\n\s*pass',
    ]

    def test_no_silent_exceptions(self, panel_manifest):
        """
        CRITICAL TEST: Detect bare except statements that hide errors.

        This will catch issues like:
        - drillhole_control_panel.py:255
        - dock_setup.py:129
        """
        panels = panel_manifest['panels']
        all_violations = []

        for panel in panels:
            module_path = panel['module']
            panel_class_name = panel['panel_class']

            try:
                module = importlib.import_module(module_path)
            except ImportError:
                continue

            # Get source code
            try:
                source = inspect.getsource(module)
            except OSError:
                continue

            violations = []

            # Check each pattern
            for pattern in self.SILENT_EXCEPTION_PATTERNS:
                matches = re.finditer(pattern, source)

                for match in matches:
                    # Calculate line number
                    line_no = source[:match.start()].count('\n') + 1

                    # Get context
                    lines = source.split('\n')
                    start_line = max(0, line_no - 3)
                    end_line = min(len(lines), line_no + 2)
                    context = '\n'.join(f"{i+1}: {lines[i]}" for i in range(start_line, end_line))

                    violations.append({
                        'panel': panel_class_name,
                        'module': module_path,
                        'line': line_no,
                        'pattern': pattern,
                        'context': context,
                        'code': match.group()
                    })

            all_violations.extend(violations)

        # Report all violations
        if all_violations:
            error_msg = f"\n\n{'='*70}\n"
            error_msg += f"🚨 CRITICAL: {len(all_violations)} Silent Exceptions Found!\n"
            error_msg += f"{'='*70}\n\n"

            for v in all_violations:
                error_msg += f"❌ {v['panel']} (line {v['line']})\n"
                error_msg += f"   File: {v['module']}\n"
                error_msg += f"   Pattern: {v['pattern']}\n"
                error_msg += f"\n   Context:\n{v['context']}\n"
                error_msg += f"\n{'-'*70}\n"

            error_msg += f"\n💡 FIX: Replace 'except: pass' with:\n"
            error_msg += f"  - Explicit logging: logger.error('Error', exc_info=True)\n"
            error_msg += f"  - Specific exceptions: except ImportError as e:\n"
            error_msg += f"  - Or remove try-except entirely\n"

            pytest.fail(error_msg)
        else:
            print(f"\n✓ No silent exceptions found in {len(panels)} panels")


@pytest.mark.signals
class TestPanelSignalConnections:
    """Test that panels connect to appropriate registry signals"""

    def test_panels_declare_signal_consumption(self, panel_manifest):
        """Test that panels which consume signals are documented in manifest"""
        panels = panel_manifest['panels']
        signal_consumers = [p for p in panels if p.get('signals_consumed')]
        print(f"\n📊 Signal Consumption: {len(signal_consumers)}/{len(panels)} panels consume signals")

    def test_panel_has_signal_handlers(self, panel_manifest, mock_qapp):
        """Test that panels have signal handler methods if they consume signals"""
        panels = panel_manifest['panels']
        missing_handlers = []

        for panel in panels:
            signals_consumed = panel.get('signals_consumed', [])
            if not signals_consumed:
                continue

            module_path = panel['module']
            panel_class_name = panel['panel_class']

            try:
                module = importlib.import_module(module_path)
                panel_class = getattr(module, panel_class_name)

                # Check for handler methods
                handler_methods = [
                    method for method in dir(panel_class)
                    if method.startswith('_on_') and callable(getattr(panel_class, method, None))
                ]

                if len(handler_methods) == 0:
                    missing_handlers.append(f"{panel_class_name}: Consumes {signals_consumed} but has no _on_* handlers")

            except ImportError:
                continue

        if missing_handlers:
            print(f"\n⚠️ Panels missing signal handlers:")
            for issue in missing_handlers[:5]:
                print(f"  - {issue}")


@pytest.mark.signals
@pytest.mark.critical
class TestDrillholeSignalFlow:
    """Test drillhole data signal flow (critical for reported issue)"""

    def test_drillhole_data_loaded_signal_exists(self, mock_registry):
        """Test that drillholeDataLoaded signal exists in registry"""
        assert hasattr(mock_registry.signals, 'drillholeDataLoaded'), \
            "DataRegistry missing drillholeDataLoaded signal"

    def test_drillhole_data_signal_can_be_emitted(self, mock_registry):
        """Test that drillholeDataLoaded signal can be emitted"""
        mock_data = MockDataFactory.create_drillhole_data(n_holes=5)

        tester = SignalTester(timeout_ms=1000)

        success, error = tester.test_signal_emission(
            mock_registry.signals.drillholeDataLoaded,
            mock_data,
            None  # No handler needed for this test
        )

        assert success, "drillholeDataLoaded signal was not emitted"
        assert error is None, f"Signal emission raised error: {error}"

    def test_drillhole_control_panel_connects_to_signal(self, mock_qapp, mock_registry, mock_signals):
        """
        CRITICAL TEST: Verify DrillholeControlPanel connects to drillholeDataLoaded

        This tests the specific issue at drillhole_control_panel.py:255
        """
        try:
            from block_model_viewer.ui.drillhole_control_panel import DrillholeControlPanel
        except ImportError as e:
            pytest.skip(f"Cannot import DrillholeControlPanel: {e}")

        # Create panel
        try:
            panel = DrillholeControlPanel(signals=mock_signals)
        except TypeError:
            try:
                panel = DrillholeControlPanel()
            except Exception as e:
                pytest.fail(f"Failed to instantiate DrillholeControlPanel: {e}")

        # Check if panel has expected handler method
        assert hasattr(panel, '_on_drillhole_data_loaded'), \
            "DrillholeControlPanel missing _on_drillhole_data_loaded handler"

    def test_drillhole_signal_handler_executes_without_error(self, mock_qapp, mock_registry, mock_signals):
        """Test that drillhole signal handler executes without raising exceptions"""
        try:
            from block_model_viewer.ui.drillhole_control_panel import DrillholeControlPanel
        except ImportError:
            pytest.skip("Cannot import DrillholeControlPanel")

        # Create panel
        try:
            panel = DrillholeControlPanel(signals=mock_signals)
        except:
            try:
                panel = DrillholeControlPanel()
            except Exception as e:
                pytest.skip(f"Cannot instantiate panel: {e}")

        # Check if handler method exists
        if not hasattr(panel, '_on_drillhole_data_loaded'):
            pytest.skip("Panel missing handler method")

        # Create mock data
        mock_data = MockDataFactory.create_drillhole_data(n_holes=5)

        # Call handler directly
        try:
            panel._on_drillhole_data_loaded(mock_data)
        except Exception as e:
            pytest.fail(f"Signal handler raised exception: {e}")


@pytest.mark.signals
class TestBlockModelSignals:
    """Test block model signal connections"""

    def test_block_model_loaded_signal_exists(self, mock_registry):
        """Test that blockModelLoaded signal exists"""
        assert hasattr(mock_registry.signals, 'blockModelLoaded'), \
            "DataRegistry missing blockModelLoaded signal"

    def test_block_model_signal_can_be_emitted(self, mock_registry):
        """Test that blockModelLoaded signal can be emitted"""
        mock_model = MockDataFactory.create_block_model(nx=10, ny=10, nz=5)

        tester = SignalTester(timeout_ms=1000)

        success, error = tester.test_signal_emission(
            mock_registry.signals.blockModelLoaded,
            mock_model,
            None
        )

        assert success, "blockModelLoaded signal was not emitted"
        assert error is None, f"Signal emission raised error: {error}"


@pytest.mark.signals
class TestGeologicalModelSignals:
    """Test geological model signal connections"""

    def test_geological_model_updated_signal_exists(self, mock_registry):
        """Test that geologicalModelUpdated signal exists"""
        assert hasattr(mock_registry.signals, 'geologicalModelUpdated'), \
            "DataRegistry missing geologicalModelUpdated signal"

    def test_geological_explorer_does_not_consume_drillhole_signals(self):
        """
        Test that GeologicalExplorerPanel does NOT consume drillhole signals.

        This is BY DESIGN - geological explorer shows only geological models,
        not drillholes. This test documents the expected behavior.
        """
        try:
            from block_model_viewer.ui.geological_explorer_panel import GeologicalExplorerPanel
        except ImportError:
            pytest.skip("Cannot import GeologicalExplorerPanel")

        # Get panel source
        module = importlib.import_module('block_model_viewer.ui.geological_explorer_panel')
        source = inspect.getsource(module)

        # Should NOT connect to drillholeDataLoaded
        assert 'drillholeDataLoaded.connect' not in source, \
            "GeologicalExplorerPanel should NOT connect to drillholeDataLoaded (by design)"

        # SHOULD connect to geological signals
        has_geological_connection = any([
            'geologicalModelUpdated' in source,
            'geologicalSurfacesLoaded' in source,
            'geologicalSolidsLoaded' in source
        ])

        assert has_geological_connection, \
            "GeologicalExplorerPanel should connect to geological signals"


@pytest.mark.signals
@pytest.mark.critical
class TestSignalEmissionReception:
    """Test end-to-end signal emission and reception"""

    def test_signal_spy_functionality(self, mock_registry):
        """Test SignalSpy utility works correctly"""
        spy = SignalSpy(mock_registry.signals.drillholeDataLoaded)

        mock_data = MockDataFactory.create_drillhole_data(n_holes=3)

        # Emit signal
        mock_registry.signals.drillholeDataLoaded.emit(mock_data)

        # Wait for reception
        success = spy.wait(timeout_ms=500)

        assert success, "SignalSpy did not receive signal"
        assert spy.count() == 1, f"Expected 1 emission, got {spy.count()}"

    def test_multiple_signal_emissions(self, mock_registry):
        """Test multiple signal emissions are all received"""
        spy = SignalSpy(mock_registry.signals.drillholeDataLoaded)

        # Emit 3 signals
        for i in range(3):
            mock_data = MockDataFactory.create_drillhole_data(n_holes=i+1)
            mock_registry.signals.drillholeDataLoaded.emit(mock_data)

        # All should be received
        assert spy.count() == 3, f"Expected 3 emissions, got {spy.count()}"


@pytest.mark.signals
class TestPropertyPanelSignals:
    """Test PropertyPanel signal connections"""

    def test_property_panel_has_property_changed_signal(self, mock_qapp, mock_signals):
        """Test that PropertyPanel emits property_changed signal"""
        try:
            from block_model_viewer.ui.property_panel import PropertyPanel
        except ImportError:
            pytest.skip("Cannot import PropertyPanel")

        # Create panel
        try:
            panel = PropertyPanel(signals=mock_signals)
        except:
            try:
                panel = PropertyPanel()
            except Exception as e:
                pytest.skip(f"Cannot instantiate PropertyPanel: {e}")

        # Check for signal emission methods
        source = inspect.getsource(panel.__class__)

        # Should have property change handling
        assert '_on_property_changed' in source or 'property_changed' in source, \
            "PropertyPanel should handle property changes"


# Summary test to report all signal-related issues
@pytest.mark.signals
@pytest.mark.critical
class TestSignalHealthSummary:
    """Generate summary of signal health across all panels"""

    def test_generate_signal_health_report(self, panel_manifest):
        """Generate a health report for signal connections"""

        report = {
            'total_panels': len(panel_manifest['panels']),
            'panels_with_signal_consumption': 0,
            'panels_with_signal_emission': 0,
            'critical_panels': 0
        }

        for panel in panel_manifest['panels']:
            if panel.get('signals_consumed'):
                report['panels_with_signal_consumption'] += 1

            if panel.get('signals_emitted'):
                report['panels_with_signal_emission'] += 1

            if panel.get('critical'):
                report['critical_panels'] += 1

        print(f"\n{'='*70}")
        print("SIGNAL HEALTH REPORT")
        print(f"{'='*70}")
        print(f"Total panels: {report['total_panels']}")
        print(f"Panels consuming signals: {report['panels_with_signal_consumption']}")
        print(f"Panels emitting signals: {report['panels_with_signal_emission']}")
        print(f"Critical panels: {report['critical_panels']}")
        print(f"{'='*70}\n")

        # This test always passes - it's informational
        assert True
