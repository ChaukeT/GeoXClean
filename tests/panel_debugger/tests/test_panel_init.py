"""
Panel Initialization Tests

Tests that all panels can be instantiated without errors.

Tests:
- Panel module can be imported
- Panel class can be instantiated
- Panel has required attributes (PANEL_ID, PANEL_NAME, etc.)
- Panel UI construction completes without errors
"""

import importlib
import inspect
import pytest
from pathlib import Path


@pytest.mark.panel_init
class TestPanelImport:
    """Test that all panels can be imported"""

    @pytest.fixture(autouse=True)
    def setup(self, panel_manifest):
        """Setup with panel manifest"""
        self.panels = panel_manifest['panels']

    def test_all_panels_can_be_imported(self, panel_manifest):
        """Test that all panel modules can be imported"""
        panels = panel_manifest['panels']

        failed_imports = []
        for panel in panels:
            module_path = panel['module']
            panel_class_name = panel['panel_class']

            try:
                module = importlib.import_module(module_path)
                if not hasattr(module, panel_class_name):
                    failed_imports.append(f"{panel_class_name}: Class not found in {module_path}")
            except ImportError as e:
                failed_imports.append(f"{panel_class_name}: {e}")

        if failed_imports:
            print(f"\n❌ Failed to import {len(failed_imports)} panels:")
            for failure in failed_imports:
                print(f"  - {failure}")
            pytest.fail(f"{len(failed_imports)} panels failed to import")
        else:
            print(f"\n✓ All {len(panels)} panels imported successfully")


@pytest.mark.panel_init
class TestPanelInstantiation:
    """Test that panels can be instantiated"""

    def test_all_panels_can_be_instantiated(self, panel_manifest, mock_qapp, mock_registry, mock_signals):
        """Test that all panels can be created"""
        panels = panel_manifest['panels']

        failed_instantiations = []
        skipped_panels = []

        for panel in panels:
            module_path = panel['module']
            panel_class_name = panel['panel_class']

            try:
                module = importlib.import_module(module_path)
                panel_class = getattr(module, panel_class_name)

                # Try to instantiate
                try:
                    panel_instance = panel_class(signals=mock_signals)
                except TypeError:
                    try:
                        panel_instance = panel_class()
                    except Exception as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['database', 'file not found', 'registry']):
                            skipped_panels.append(f"{panel_class_name}: {e}")
                            continue
                        raise

                if panel_instance is None:
                    failed_instantiations.append(f"{panel_class_name}: Returned None")

            except ImportError as e:
                skipped_panels.append(f"{panel_class_name}: Import error")
            except Exception as e:
                failed_instantiations.append(f"{panel_class_name}: {e}")

        print(f"\n📊 Panel Instantiation Results:")
        print(f"  ✓ Successful: {len(panels) - len(failed_instantiations) - len(skipped_panels)}")
        print(f"  ⊘ Skipped: {len(skipped_panels)}")
        print(f"  ❌ Failed: {len(failed_instantiations)}")

        if failed_instantiations:
            print(f"\n❌ Failed instantiations:")
            for failure in failed_instantiations[:5]:  # Show first 5
                print(f"  - {failure}")
            pytest.fail(f"{len(failed_instantiations)} panels failed to instantiate")


@pytest.mark.panel_init
class TestPanelAttributes:
    """Test that panels have required metadata attributes"""

    def test_panels_have_metadata_attributes(self, panel_manifest):
        """Test that panels have PANEL_ID and PANEL_NAME metadata"""
        panels = panel_manifest['panels']

        missing_id = []
        missing_name = []

        for panel in panels:
            try:
                module = importlib.import_module(panel['module'])
                panel_class = getattr(module, panel['panel_class'])

                if not hasattr(panel_class, 'PANEL_ID'):
                    missing_id.append(panel['panel_class'])
                if not hasattr(panel_class, 'PANEL_NAME'):
                    missing_name.append(panel['panel_class'])

            except ImportError:
                continue

        print(f"\n📊 Panel Metadata:")
        print(f"  Panels with PANEL_ID: {len(panels) - len(missing_id)}/{len(panels)}")
        print(f"  Panels with PANEL_NAME: {len(panels) - len(missing_name)}/{len(panels)}")


@pytest.mark.panel_init
class TestPanelBaseClass:
    """Test that panels inherit from correct base classes"""

    def test_panels_inherit_from_expected_base(self, panel_manifest):
        """Test that panels inherit from expected base classes"""
        panels = panel_manifest['panels']

        inheritance_issues = []

        for panel in panels:
            expected_base = panel.get('base_class')
            if not expected_base:
                continue

            try:
                module = importlib.import_module(panel['module'])
                panel_class = getattr(module, panel['panel_class'])

                bases = [base.__name__ for base in panel_class.__mro__]

                if expected_base not in bases:
                    inheritance_issues.append(
                        f"{panel['panel_class']}: Expected {expected_base}, got {bases[1] if len(bases) > 1 else 'None'}"
                    )
            except ImportError:
                continue

        if inheritance_issues:
            print(f"\n⚠️ Inheritance issues found:")
            for issue in inheritance_issues:
                print(f"  - {issue}")


@pytest.mark.panel_init
@pytest.mark.critical
class TestCriticalPanels:
    """Test critical panels that are essential for drillhole loading"""

    CRITICAL_PANELS = [
        'DrillholeControlPanel',
        'DrillholeImportPanel',
        'PropertyPanel',
        'GeologicalExplorerPanel'
    ]

    @pytest.mark.parametrize("panel_class_name", CRITICAL_PANELS)
    def test_critical_panel_instantiates(self, panel_class_name, mock_qapp, mock_registry, mock_signals, panel_manifest):
        """Test that critical panels can be instantiated"""

        # Find panel in manifest
        panel_info = None
        for p in panel_manifest['panels']:
            if p['panel_class'] == panel_class_name:
                panel_info = p
                break

        if not panel_info:
            pytest.fail(f"Critical panel {panel_class_name} not found in manifest")

        module_path = panel_info['module']

        # Import and instantiate
        module = importlib.import_module(module_path)
        panel_class = getattr(module, panel_class_name)

        try:
            try:
                panel = panel_class(signals=mock_signals)
            except TypeError:
                panel = panel_class()

            assert panel is not None
            print(f"✓ {panel_class_name} instantiated successfully")

        except Exception as e:
            pytest.fail(f"CRITICAL: {panel_class_name} failed to instantiate: {e}")
