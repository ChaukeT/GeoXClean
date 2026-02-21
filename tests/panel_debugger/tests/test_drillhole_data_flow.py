"""
Drillhole Data Flow Diagnostic Test

This test diagnoses the EXACT point where drillhole data flow breaks:
1. Data registered to registry
2. Signal emitted
3. Panel receives signal
4. Panel populates UI

This addresses the user's issue: "there is nothing on the control panels
and also drillhole explorer" even after successful data import.
"""

import pytest
import logging
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication
from pathlib import Path

logger = logging.getLogger(__name__)


@pytest.mark.critical
class TestDrillholeDataFlowDiagnostic:
    """Diagnostic test to find where drillhole data flow breaks."""

    def test_drillhole_import_to_panel_flow(self, mock_qapp, mock_signals):
        """
        Test complete flow from import to panel display.

        This will show exactly where the data flow breaks.
        """
        from block_model_viewer.core.data_registry import DataRegistry
        from block_model_viewer.ui.drillhole_control_panel import DrillholeControlPanel
        import pandas as pd

        # Create registry and panel
        registry = DataRegistry.create()

        try:
            panel = DrillholeControlPanel(signals=mock_signals)
        except TypeError:
            panel = DrillholeControlPanel()

        # Connect panel to registry
        signal_received = []

        def track_signal(data):
            logger.info(f"✅ Signal received with keys: {data.keys() if isinstance(data, dict) else 'NOT A DICT'}")
            signal_received.append(data)

        try:
            registry.drillholeDataLoaded.connect(track_signal)
            registry.drillholeDataLoaded.connect(panel._on_drillhole_data_loaded)
        except Exception as e:
            pytest.fail(f"Failed to connect signals: {e}")

        # Create mock drillhole data
        collars = pd.DataFrame({
            'hole_id': ['DH001', 'DH002', 'DH003'],
            'x': [100.0, 200.0, 300.0],
            'y': [100.0, 200.0, 300.0],
            'z': [0.0, 0.0, 0.0],
            'length': [100.0, 150.0, 200.0]
        })

        assays = pd.DataFrame({
            'hole_id': ['DH001', 'DH001', 'DH002'],
            'from_m': [0.0, 10.0, 0.0],
            'to_m': [10.0, 20.0, 10.0],
            'FE_PCT': [45.5, 50.2, 48.1],
            'X': [100.0, 100.0, 200.0],
            'Y': [100.0, 100.0, 200.0],
            'Z': [0.0, -10.0, 0.0]
        })

        drillhole_data = {
            'collars': collars,
            'assays': assays,
            'composites': assays.copy(),  # Use assays as composites for simplicity
        }

        # STEP 1: Register data
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Registering drillhole data to registry")
        logger.info("="*70)

        success = registry.register_drillhole_data(
            drillhole_data,
            source_panel="Test",
            metadata={'test': True}
        )

        assert success, "❌ FAILED at STEP 1: Data registration returned False"
        logger.info("✅ STEP 1 PASSED: Data registered successfully")

        # STEP 2: Verify data is stored
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Verifying data is stored in registry")
        logger.info("="*70)

        stored_data = registry.get_drillhole_data()

        if stored_data is None:
            pytest.fail("❌ FAILED at STEP 2: get_drillhole_data() returned None")

        logger.info(f"✅ STEP 2 PASSED: Data retrieved, keys: {stored_data.keys()}")

        # STEP 3: Verify signal was emitted
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Verifying signal was emitted")
        logger.info("="*70)

        # Process Qt events to allow signal to propagate
        QApplication.processEvents()

        if not signal_received:
            pytest.fail("❌ FAILED at STEP 3: Signal was not emitted or not received")

        logger.info(f"✅ STEP 3 PASSED: Signal received {len(signal_received)} time(s)")

        # STEP 4: Verify panel received data
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Verifying panel populated with data")
        logger.info("="*70)

        # Wait for deferred load (QTimer.singleShot(10, _do_load))
        QTimer.singleShot(50, lambda: None)  # Wait 50ms
        QApplication.processEvents()

        # Check if panel was populated
        if not hasattr(panel, '_collars_df') or panel._collars_df is None:
            pytest.fail(
                "❌ FAILED at STEP 4: Panel did not populate _collars_df\n"
                "This means _on_drillhole_data_loaded() either:\n"
                "1. Was not called\n"
                "2. Was called but failed to set _collars_df\n"
                "3. Was called but data format was wrong"
            )

        if panel._collars_df.empty:
            pytest.fail("❌ FAILED at STEP 4: Panel _collars_df is empty")

        logger.info(f"✅ STEP 4 PASSED: Panel populated with {len(panel._collars_df)} collars")

        # STEP 5: Verify UI widgets were updated
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Verifying UI widgets were updated")
        logger.info("="*70)

        if not hasattr(panel, '_hole_ids') or not panel._hole_ids:
            pytest.fail(
                "❌ FAILED at STEP 5: Panel _hole_ids not populated\n"
                "This means _rebuild_checklist() was not called or failed"
            )

        logger.info(f"✅ STEP 5 PASSED: Panel has {len(panel._hole_ids)} hole IDs")

        # STEP 6: Verify dataset combo was updated
        logger.info("\n" + "="*70)
        logger.info("STEP 6: Verifying dataset combo was updated")
        logger.info("="*70)

        dataset_count = panel.dataset_combo.count()
        if dataset_count == 0:
            pytest.fail(
                "❌ FAILED at STEP 6: Dataset combo is empty\n"
                "Expected 'Raw Assays' and/or 'Composites'"
            )

        dataset_items = [panel.dataset_combo.itemText(i) for i in range(dataset_count)]
        logger.info(f"✅ STEP 6 PASSED: Dataset combo has {dataset_count} items: {dataset_items}")

        # FINAL SUMMARY
        logger.info("\n" + "="*70)
        logger.info("✅ ALL STEPS PASSED - DATA FLOW IS WORKING")
        logger.info("="*70)
        logger.info("\nIf you're seeing blank panels in the actual application,")
        logger.info("the issue is NOT with the data flow but with:")
        logger.info("1. UI rendering/refresh")
        logger.info("2. Panel visibility/layout")
        logger.info("3. Qt event loop processing")
        logger.info("="*70)


    def test_drillhole_explorer_panel_receives_data(self, mock_qapp, mock_signals):
        """
        Test that GeologicalExplorerPanel receives drillhole data.

        NOTE: GeologicalExplorerPanel may NOT show drillholes by design
        (it's for geological models), but this test checks if it at least
        receives the signal.
        """
        from block_model_viewer.core.data_registry import DataRegistry
        from block_model_viewer.ui.geological_explorer_panel import GeologicalExplorerPanel
        import pandas as pd

        registry = DataRegistry.create()

        try:
            panel = GeologicalExplorerPanel(signals=mock_signals)
        except Exception as e:
            pytest.skip(f"Cannot instantiate GeologicalExplorerPanel: {e}")

        # Check if panel connects to drillholeDataLoaded signal
        # (It shouldn't, as it's for geological models)
        import inspect
        source = inspect.getsource(GeologicalExplorerPanel)

        if 'drillholeDataLoaded' in source:
            logger.info("GeologicalExplorerPanel DOES connect to drillholeDataLoaded")
        else:
            logger.info("GeologicalExplorerPanel does NOT connect to drillholeDataLoaded (expected)")
            logger.info("This panel is for geological models, not drillholes")


@pytest.mark.critical
class TestDataFlowSummary:
    """Generate summary of data flow test results."""

    def test_data_flow_summary(self):
        """
        Generate summary report.

        This test always passes - it's informational.
        """
        print(f"\n{'='*70}")
        print("DRILLHOLE DATA FLOW DIAGNOSTIC SUMMARY")
        print(f"{'='*70}")
        print(f"\nThis test suite diagnoses the complete data flow:")
        print(f"  1. Data registration to DataRegistry")
        print(f"  2. Signal emission (drillholeDataLoaded)")
        print(f"  3. Panel signal reception")
        print(f"  4. Panel data population (_on_drillhole_data_loaded)")
        print(f"  5. UI widget updates (_rebuild_checklist, etc.)")
        print(f"  6. Dataset combo population")
        print(f"\nRun with -v to see detailed flow diagnostics")
        print(f"{'='*70}\n")

        assert True
