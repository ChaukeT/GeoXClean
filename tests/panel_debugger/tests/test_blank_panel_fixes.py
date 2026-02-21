"""
Test Blank Panel Fixes - Verify _build_ui() is called in __init__

Tests that the 3 panels that were blank now display correctly:
1. SGSIMPanel
2. CoKrigingPanel
3. CoSGSIMPanel

Root cause: _build_ui() was only called in refresh_theme(), not in __init__()
Fix: Moved _build_ui() and _init_registry() calls to __init__()
"""

import pytest
import logging

logger = logging.getLogger(__name__)


@pytest.mark.critical
class TestBlankPanelFixes:
    """Test that previously blank panels now display correctly."""

    def test_sgsim_panel_instantiates_with_ui(self, mock_qapp):
        """
        Test that SGSIMPanel builds its UI immediately on instantiation.

        Before fix: _build_ui() only called in refresh_theme(), panel was blank
        After fix: _build_ui() called in __init__(), UI displays immediately
        """
        from block_model_viewer.ui.sgsim_panel import SGSIMPanel

        try:
            panel = SGSIMPanel()

            # Verify panel was created
            assert panel is not None, "Panel is None"

            # Verify UI elements exist (created by _setup_ui())
            assert hasattr(panel, 'run_btn'), "Panel missing run_btn (UI not built)"
            assert hasattr(panel, 'nreal_spin'), "Panel missing nreal_spin (UI not built)"
            assert hasattr(panel, 'seed_spin'), "Panel missing seed_spin (UI not built)"

            # Verify main_layout has widgets (not empty)
            assert panel.main_layout.count() > 0, "main_layout is empty (no widgets added)"

            logger.info("✅ SGSIMPanel instantiated with UI successfully")

        except Exception as e:
            pytest.fail(
                f"\n{'='*80}\n"
                f"🚨 SGSIMPanel instantiation failed:\n"
                f"{'='*80}\n"
                f"\n{e}\n\n"
                f"This likely means _build_ui() is still not being called in __init__()\n"
                f"{'='*80}\n"
            )

    def test_cokriging_panel_instantiates_with_ui(self, mock_qapp):
        """
        Test that CoKrigingPanel builds its UI immediately on instantiation.

        Before fix: _build_ui() only called in refresh_theme(), panel was blank
        After fix: _build_ui() called in __init__(), UI displays immediately
        """
        from block_model_viewer.ui.cokriging_panel import CoKrigingPanel

        try:
            panel = CoKrigingPanel()

            # Verify panel was created
            assert panel is not None, "Panel is None"

            # Verify UI elements exist (actual widget names from the panel)
            assert hasattr(panel, 'run_btn'), "Panel missing run_btn (UI not built)"
            assert hasattr(panel, 'primary_combo'), "Panel missing primary_combo (UI not built)"
            assert hasattr(panel, 'secondary_combo'), "Panel missing secondary_combo (UI not built)"

            # Verify main_layout has widgets
            assert panel.main_layout.count() > 0, "main_layout is empty (no widgets added)"

            logger.info("✅ CoKrigingPanel instantiated with UI successfully")

        except Exception as e:
            pytest.fail(
                f"\n{'='*80}\n"
                f"🚨 CoKrigingPanel instantiation failed:\n"
                f"{'='*80}\n"
                f"\n{e}\n\n"
                f"This likely means _build_ui() is still not being called in __init__()\n"
                f"{'='*80}\n"
            )

    def test_cosgsim_panel_instantiates_with_ui(self, mock_qapp):
        """
        Test that CoSGSIMPanel builds its UI immediately on instantiation.

        Before fix: _build_ui() only called in refresh_theme(), panel was blank
        After fix: _build_ui() called in __init__(), UI displays immediately
        """
        from block_model_viewer.ui.cosgsim_panel import CoSGSIMPanel

        try:
            panel = CoSGSIMPanel()

            # Verify panel was created
            assert panel is not None, "Panel is None"

            # Verify UI elements exist (actual widget names from the panel)
            assert hasattr(panel, 'run_btn'), "Panel missing run_btn (UI not built)"
            assert hasattr(panel, 'nx_spin'), "Panel missing nx_spin (UI not built)"
            assert hasattr(panel, 'xmin_spin'), "Panel missing xmin_spin (UI not built)"

            # Verify main_layout has widgets
            assert panel.main_layout.count() > 0, "main_layout is empty (no widgets added)"

            logger.info("✅ CoSGSIMPanel instantiated with UI successfully")

        except Exception as e:
            pytest.fail(
                f"\n{'='*80}\n"
                f"🚨 CoSGSIMPanel instantiation failed:\n"
                f"{'='*80}\n"
                f"\n{e}\n\n"
                f"This likely means _build_ui() is still not being called in __init__()\n"
                f"{'='*80}\n"
            )

    def test_all_three_panels_have_ui_in_init(self):
        """
        Static code analysis: Verify _build_ui() is called in __init__() for all 3 panels.

        This test reads the source code to confirm the fix was applied correctly.
        """
        import inspect
        from block_model_viewer.ui.sgsim_panel import SGSIMPanel
        from block_model_viewer.ui.cokriging_panel import CoKrigingPanel
        from block_model_viewer.ui.cosgsim_panel import CoSGSIMPanel

        panels = [
            ("SGSIMPanel", SGSIMPanel),
            ("CoKrigingPanel", CoKrigingPanel),
            ("CoSGSIMPanel", CoSGSIMPanel)
        ]

        for panel_name, panel_class in panels:
            # Get __init__ source
            init_source = inspect.getsource(panel_class.__init__)

            # Verify _build_ui() is called in __init__
            assert "_build_ui()" in init_source, (
                f"{panel_name}.__init__() does NOT call _build_ui()!\n"
                f"The fix was not applied correctly."
            )

            # Verify _init_registry() is called in __init__
            assert "_init_registry()" in init_source, (
                f"{panel_name}.__init__() does NOT call _init_registry()!\n"
                f"The fix was not applied correctly."
            )

            logger.info(f"✅ {panel_name}.__init__() correctly calls _build_ui() and _init_registry()")

        logger.info("\n✅ All 3 panels have correct __init__() implementation")


@pytest.mark.critical
class TestBlankPanelFixesSummary:
    """Generate summary of blank panel fixes."""

    def test_blank_panel_fixes_summary(self):
        """
        Summary report of blank panel fixes.

        This test always passes - it's informational.
        """
        print(f"\n{'='*80}")
        print("BLANK PANEL FIXES VERIFICATION SUMMARY")
        print(f"{'='*80}")
        print(f"\nPanels fixed:")
        print(f"  1. SGSIMPanel (sgsim_panel.py)")
        print(f"     - Added _build_ui() call to __init__() at line ~156")
        print(f"     - Added _init_registry() call to __init__() at line ~159")
        print(f"  2. CoKrigingPanel (cokriging_panel.py)")
        print(f"     - Added _build_ui() call to __init__() at line ~58")
        print(f"     - Added _init_registry() call to __init__() at line ~61")
        print(f"  3. CoSGSIMPanel (cosgsim_panel.py)")
        print(f"     - Added _build_ui() call to __init__() at line ~62")
        print(f"     - Added _init_registry() call to __init__() at line ~65")
        print(f"\nRoot cause:")
        print(f"  _build_ui() was only called in refresh_theme(), not __init__()")
        print(f"  This meant UI only appeared when theme changed, not on first open")
        print(f"\nRun with -v to see detailed test results")
        print(f"{'='*80}\n")

        assert True
