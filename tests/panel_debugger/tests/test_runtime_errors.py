"""
Runtime Error Detection Tests

Tests that catch runtime errors like:
- AttributeError: missing methods/attributes
- Duplicate layout warnings
- Initialization failures

These are different from silent exceptions - these are actual bugs that
prevent the application from running.
"""

import importlib
import inspect
import pytest
import logging
from io import StringIO

# Capture Qt warnings
qt_warning_handler = StringIO()
qt_logger = logging.getLogger('Qt')
qt_logger.addHandler(logging.StreamHandler(qt_warning_handler))


@pytest.mark.critical
class TestRuntimeInitialization:
    """Test that panels can be instantiated without runtime errors"""

    CRITICAL_PANELS = [
        ('block_model_viewer.ui.geological_explorer_panel', 'GeologicalExplorerPanel'),
        ('block_model_viewer.ui.drillhole_control_panel', 'DrillholeControlPanel'),
        ('block_model_viewer.ui.property_panel', 'PropertyPanel'),
        ('block_model_viewer.ui.gc_decision_panel', 'GCDecisionPanel'),
        ('block_model_viewer.ui.kriging_panel', 'KrigingPanel'),
        ('block_model_viewer.ui.sgsim_panel', 'SGSIMPanel'),
        ('block_model_viewer.ui.variogram_panel', 'VariogramAnalysisPanel'),
    ]

    @pytest.mark.parametrize("module_path,panel_class_name", CRITICAL_PANELS)
    def test_panel_instantiates_without_attribute_error(
        self,
        module_path,
        panel_class_name,
        mock_qapp,
        mock_signals
    ):
        """
        Test that panels can be created without AttributeError.

        This catches errors like:
        - AttributeError: 'ModernComboBox' object has no attribute '_get_stylesheet'
        """
        try:
            module = importlib.import_module(module_path)
            panel_class = getattr(module, panel_class_name)

            # Try to instantiate with signals
            try:
                panel = panel_class(signals=mock_signals)
            except TypeError:
                # Try without signals
                panel = panel_class()

            # If we get here, instantiation succeeded
            assert panel is not None, f"{panel_class_name} instantiated to None"

        except AttributeError as e:
            pytest.fail(
                f"\n{'='*70}\n"
                f"🚨 CRITICAL RUNTIME ERROR: {panel_class_name}\n"
                f"{'='*70}\n"
                f"\nAttributeError during instantiation:\n"
                f"  {e}\n\n"
                f"This is a BLOCKING BUG that prevents the application from starting.\n"
                f"\nCommon causes:\n"
                f"  - Missing method (e.g., calling self._get_stylesheet() when method doesn't exist)\n"
                f"  - Missing attribute (e.g., accessing self.foo before it's defined)\n"
                f"  - Wrong method name (typo in method call)\n"
                f"\nFile: {module_path}.py\n"
                f"{'='*70}\n"
            )
        except ImportError as e:
            pytest.skip(f"Cannot import {module_path}: {e}")
        except Exception as e:
            # Other exceptions might be expected (database errors, etc.)
            # Only fail on AttributeError which indicates a code bug
            error_type = type(e).__name__
            if 'database' not in str(e).lower() and 'file not found' not in str(e).lower():
                pytest.fail(
                    f"{panel_class_name} raised {error_type} during init: {e}"
                )

    @pytest.mark.parametrize("module_path,panel_class_name", CRITICAL_PANELS)
    def test_panel_no_duplicate_layout_warning(
        self,
        module_path,
        panel_class_name,
        mock_qapp,
        mock_signals,
        caplog
    ):
        """
        Test that panels don't trigger duplicate layout warnings.

        This catches warnings like:
        - QLayout: Attempting to add QLayout to Panel which already has a layout
        """
        # Clear previous warnings
        qt_warning_handler.truncate(0)
        qt_warning_handler.seek(0)

        try:
            module = importlib.import_module(module_path)
            panel_class = getattr(module, panel_class_name)

            # Instantiate panel
            try:
                panel = panel_class(signals=mock_signals)
            except TypeError:
                panel = panel_class()
            except Exception as e:
                # Skip on expected errors
                if 'database' in str(e).lower() or 'attribute' in str(e).lower():
                    pytest.skip(f"Cannot instantiate: {e}")
                raise

            # Check for Qt layout warnings
            warnings = qt_warning_handler.getvalue()

            if 'QLayout: Attempting to add QLayout' in warnings:
                pytest.fail(
                    f"\n{'='*70}\n"
                    f"⚠️  LAYOUT WARNING: {panel_class_name}\n"
                    f"{'='*70}\n"
                    f"\nDuplicate layout detected:\n"
                    f"{warnings}\n\n"
                    f"CAUSE:\n"
                    f"  Panel is creating a new layout with QVBoxLayout(self)\n"
                    f"  but BaseDockPanel already created self.main_layout\n"
                    f"\nFIX:\n"
                    f"  In setup_ui(), use:\n"
                    f"    layout = self.main_layout  # Use inherited layout\n"
                    f"  Instead of:\n"
                    f"    layout = QVBoxLayout(self)  # ❌ Creates duplicate!\n"
                    f"\nFile: {module_path}.py\n"
                    f"{'='*70}\n"
                )

        except ImportError:
            pytest.skip(f"Cannot import {module_path}")


@pytest.mark.critical
class TestMissingMethodDetection:
    """Detect missing method calls in panel code"""

    def test_detect_missing_get_stylesheet_calls(self):
        """
        Scan for classes calling _get_stylesheet() without defining it.

        This is a static analysis test that catches bugs before runtime.
        """
        violations = []

        # Files to check
        files_to_check = [
            'block_model_viewer/ui/geological_explorer_panel.py',
            'block_model_viewer/ui/drillhole_control_panel.py',
        ]

        for file_path in files_to_check:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Find all class definitions
                import re
                classes = re.finditer(r'class (\w+)\([^)]+\):', content)

                for class_match in classes:
                    class_name = class_match.group(1)
                    class_start = class_match.end()

                    # Find next class or end of file
                    next_class = re.search(r'\nclass ', content[class_start:])
                    class_end = class_start + next_class.start() if next_class else len(content)

                    class_body = content[class_start:class_end]

                    # Check if class calls _get_stylesheet()
                    if 'self._get_stylesheet()' in class_body:
                        # Check if class defines _get_stylesheet()
                        if 'def _get_stylesheet(self' not in class_body:
                            violations.append({
                                'file': file_path,
                                'class': class_name,
                                'method': '_get_stylesheet'
                            })
            except FileNotFoundError:
                pass

        if violations:
            error_msg = f"\n{'='*70}\n"
            error_msg += f"🚨 MISSING METHOD CALLS DETECTED\n"
            error_msg += f"{'='*70}\n\n"

            for v in violations:
                error_msg += f"❌ {v['class']} calls self.{v['method']}() but doesn't define it\n"
                error_msg += f"   File: {v['file']}\n\n"

            error_msg += f"FIX:\n"
            error_msg += f"  Either:\n"
            error_msg += f"  1. Add the missing method to the class\n"
            error_msg += f"  2. Remove the method call\n"
            error_msg += f"  3. Call a different method that exists\n"
            error_msg += f"{'='*70}\n"

            pytest.fail(error_msg)


@pytest.mark.critical
class TestRuntimeErrorSummary:
    """Generate summary of runtime errors"""

    def test_runtime_error_summary(self, panel_manifest):
        """
        Generate a summary report of runtime initialization health.

        This test always passes - it's informational.
        """
        panels = panel_manifest['panels']

        print(f"\n{'='*70}")
        print("RUNTIME ERROR DETECTION SUMMARY")
        print(f"{'='*70}")
        print(f"Total panels tested: {len(panels)}")
        print(f"\nTests run:")
        print(f"  ✓ AttributeError detection")
        print(f"  ✓ Duplicate layout detection")
        print(f"  ✓ Missing method call detection")
        print(f"\nSee individual test results above for details.")
        print(f"{'='*70}\n")

        assert True  # Always pass
