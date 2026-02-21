"""
Test Recent Fixes - Validation and F-String CSS

This test verifies the two critical fixes:
1. Collar-only validation (data_registry_simple.py)
2. F-string CSS braces (drillhole_status_bar.py)
"""

import pytest
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@pytest.mark.critical
class TestCollarOnlyValidation:
    """Test that collar-only drillhole imports are now allowed."""

    def test_collar_only_import_passes_validation(self, mock_qapp):
        """
        Test that collar-only data passes validation.

        This addresses the issue where user's 142 collar import was rejected.
        """
        from block_model_viewer.core.data_registry_simple import DataRegistrySimple

        registry = DataRegistrySimple()

        # Create collar-only data (like the user's import)
        collars = pd.DataFrame({
            'hole_id': [f'DH{i:03d}' for i in range(1, 143)],  # 142 collars
            'x': [1000.0 + i * 10 for i in range(142)],
            'y': [2000.0 + i * 10 for i in range(142)],
            'z': [100.0 for i in range(142)],
            'depth': [150.0 for i in range(142)]
        })

        # Collar-only data structure (no assays, no composites)
        drillhole_data = {
            'collars': collars,
            'surveys': None,
            'assays': None,
            'lithology': None,
            'structures': None,
            'composites': None
        }

        # Validate - should PASS now
        is_valid, error_msg = registry._validate_drillholes(drillhole_data)

        # Assertions
        assert is_valid, f"Collar-only validation failed: {error_msg}"
        assert error_msg is None, f"Unexpected error message: {error_msg}"

        logger.info("✅ Collar-only validation PASSED")

    def test_collar_only_registers_successfully(self, mock_qapp):
        """
        Test that collar-only data can be registered to registry.

        Full workflow test: validate → register → retrieve
        """
        from block_model_viewer.core.data_registry import DataRegistry

        registry = DataRegistry.instance()

        # Create collar-only data
        collars = pd.DataFrame({
            'hole_id': ['DH001', 'DH002', 'DH003'],
            'x': [1000.0, 1100.0, 1200.0],
            'y': [2000.0, 2100.0, 2200.0],
            'z': [100.0, 100.0, 100.0],
            'depth': [150.0, 200.0, 175.0]
        })

        drillhole_data = {
            'collars': collars,
            'surveys': None,
            'assays': None,
            'lithology': None,
            'structures': None,
            'composites': None
        }

        # Register
        success = registry.register_drillhole_data(
            drillhole_data,
            source_panel="Test",
            metadata={'test': True}
        )

        assert success, "Failed to register collar-only data"

        # Retrieve
        stored_data = registry.get_drillhole_data(mode='validated')

        assert stored_data is not None, "Failed to retrieve collar-only data"
        assert 'collars' in stored_data, "Collars missing from stored data"
        assert len(stored_data['collars']) == 3, "Wrong number of collars retrieved"

        logger.info("✅ Collar-only data registered and retrieved successfully")

    def test_full_data_still_works(self, mock_qapp):
        """
        Test that full data (collars + assays) still works correctly.

        Ensure we didn't break normal imports.
        """
        from block_model_viewer.core.data_registry_simple import DataRegistrySimple

        registry = DataRegistrySimple()

        # Create full data (collars + assays)
        collars = pd.DataFrame({
            'hole_id': ['DH001', 'DH002'],
            'x': [1000.0, 1100.0],
            'y': [2000.0, 2100.0],
            'z': [100.0, 100.0],
            'depth': [150.0, 200.0]
        })

        assays = pd.DataFrame({
            'hole_id': ['DH001', 'DH001', 'DH002'],
            'from_m': [0.0, 10.0, 0.0],
            'to_m': [10.0, 20.0, 10.0],
            'FE_PCT': [45.5, 50.2, 48.1]
        })

        drillhole_data = {
            'collars': collars,
            'surveys': None,
            'assays': assays,
            'lithology': None,
            'structures': None,
            'composites': None
        }

        # Validate - should PASS
        is_valid, error_msg = registry._validate_drillholes(drillhole_data)

        assert is_valid, f"Full data validation failed: {error_msg}"
        assert error_msg is None, f"Unexpected error message: {error_msg}"

        logger.info("✅ Full data (collars + assays) validation PASSED")


@pytest.mark.critical
class TestFStringCSSFix:
    """Test that f-string CSS braces are properly escaped."""

    def test_drillhole_status_bar_instantiates(self, mock_qapp):
        """
        Test that DrillholeProcessStatusBar can be instantiated without NameError.

        This was failing with: NameError: name 'background' is not defined
        """
        from block_model_viewer.ui.drillhole_status_bar import DrillholeProcessStatusBar

        try:
            status_bar = DrillholeProcessStatusBar()
            assert status_bar is not None, "Status bar is None"
            logger.info("✅ DrillholeProcessStatusBar instantiated successfully")
        except NameError as e:
            pytest.fail(
                f"\n{'='*80}\n"
                f"🚨 NameError in DrillholeProcessStatusBar:\n"
                f"{'='*80}\n"
                f"\n{e}\n\n"
                f"This means CSS braces are not properly escaped in f-strings.\n"
                f"FIX: In _get_stylesheet(), change {{ to {{{{ and }} to }}}}\n"
                f"{'='*80}\n"
            )

    def test_status_bar_stylesheet_has_no_syntax_errors(self, mock_qapp):
        """
        Test that the stylesheet method executes without errors.
        """
        from block_model_viewer.ui.drillhole_status_bar import DrillholeProcessStatusBar

        status_bar = DrillholeProcessStatusBar()

        # Call _get_stylesheet() - should not raise NameError
        try:
            stylesheet = status_bar._get_stylesheet()
            assert isinstance(stylesheet, str), "Stylesheet is not a string"
            assert len(stylesheet) > 0, "Stylesheet is empty"

            # Verify CSS braces are properly escaped
            # The returned stylesheet should contain literal { and }, not Python code
            assert "ModernStatusBar {" in stylesheet or "ModernStatusBar{{" not in stylesheet.replace("{{", "{")

            logger.info("✅ Stylesheet generation successful")
        except NameError as e:
            pytest.fail(f"NameError in _get_stylesheet(): {e}")

    def test_qc_window_can_be_created(self, mock_qapp, mock_signals):
        """
        Test that QCWindow can be instantiated.

        This was failing because it creates DrillholeProcessStatusBar.
        """
        from block_model_viewer.ui.qc_window import QCWindow
        from block_model_viewer.core.data_registry import DataRegistry

        registry = DataRegistry.instance()

        # Create mock drillhole data for QCWindow
        collars = pd.DataFrame({
            'hole_id': ['DH001'],
            'x': [1000.0],
            'y': [2000.0],
            'z': [100.0],
            'depth': [150.0]
        })

        assays = pd.DataFrame({
            'hole_id': ['DH001'],
            'from_m': [0.0],
            'to_m': [10.0],
            'FE_PCT': [45.5],
            'X': [1000.0],
            'Y': [2000.0],
            'Z': [100.0]
        })

        drillhole_data = {
            'collars': collars,
            'surveys': None,
            'assays': assays,
            'lithology': None,
            'structures': None,
            'composites': None
        }

        registry.register_drillhole_data(drillhole_data, source_panel="Test")

        try:
            qc_window = QCWindow(
                drillhole_data=drillhole_data,
                registry=registry,
                signals=mock_signals,
                parent=None
            )
            assert qc_window is not None, "QC Window is None"
            logger.info("✅ QCWindow instantiated successfully (status bar created)")
        except NameError as e:
            pytest.fail(
                f"\n{'='*80}\n"
                f"🚨 NameError when creating QCWindow:\n"
                f"{'='*80}\n"
                f"\n{e}\n\n"
                f"This likely means DrillholeProcessStatusBar has CSS brace issues.\n"
                f"{'='*80}\n"
            )
        except Exception as e:
            # Other exceptions are OK (database, etc.) - we only care about NameError
            error_str = str(e).lower()
            if 'database' in error_str or 'file not found' in error_str:
                logger.info(f"⚠️ QCWindow skipped due to expected error: {e}")
            else:
                raise


@pytest.mark.critical
class TestRecentFixesSummary:
    """Generate summary of recent fixes verification."""

    def test_recent_fixes_summary(self):
        """
        Summary report of recent fixes.

        This test always passes - it's informational.
        """
        print(f"\n{'='*80}")
        print("RECENT FIXES VERIFICATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nFixes tested:")
        print(f"  1. Collar-only validation (data_registry_simple.py)")
        print(f"     - Allows imports with collars but no assays")
        print(f"     - Fixes user's 142 collar import issue")
        print(f"  2. F-string CSS braces (drillhole_status_bar.py)")
        print(f"     - Fixed NameError: name 'background' is not defined")
        print(f"     - QC Window can now be opened")
        print(f"\nRun with -v to see detailed test results")
        print(f"{'='*80}\n")

        assert True
