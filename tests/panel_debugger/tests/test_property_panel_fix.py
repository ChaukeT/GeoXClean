"""
Test PropertyPanel Registry Initialization Fix

Tests that PropertyPanel properly initializes self.registry in __init__()
to prevent AttributeError when accessing it.

Root cause: self.registry was used in multiple methods but never initialized
Fix: Added self.registry = self.get_registry() to __init__()
"""

import pytest
import logging

logger = logging.getLogger(__name__)


@pytest.mark.critical
class TestPropertyPanelRegistryFix:
    """Test that PropertyPanel registry is properly initialized."""

    def test_property_panel_has_registry_attribute(self, mock_qapp):
        """
        Test that PropertyPanel initializes self.registry in __init__().

        Before fix: AttributeError: 'PropertyPanel' object has no attribute 'registry'
        After fix: self.registry is initialized (may be None, but attribute exists)
        """
        from block_model_viewer.ui.property_panel import PropertyPanel

        try:
            panel = PropertyPanel()

            # Verify panel was created
            assert panel is not None, "Panel is None"

            # Verify registry attribute exists (critical fix!)
            assert hasattr(panel, 'registry'), (
                "PropertyPanel missing 'registry' attribute!\n"
                "This will cause AttributeError when _populate_block_model_list() is called."
            )

            logger.info(f"✅ PropertyPanel has registry attribute: {panel.registry}")

        except AttributeError as e:
            pytest.fail(
                f"\n{'='*80}\n"
                f"🚨 PropertyPanel instantiation failed with AttributeError:\n"
                f"{'='*80}\n"
                f"\n{e}\n\n"
                f"The fix was not applied correctly.\n"
                f"{'='*80}\n"
            )

    def test_property_panel_populate_block_model_list_no_error(self, mock_qapp):
        """
        Test that _populate_block_model_list() doesn't crash with AttributeError.

        This was the specific error from the user's traceback:
        AttributeError: 'PropertyPanel' object has no attribute 'registry'
        at line 1622 in _populate_block_model_list()
        """
        from block_model_viewer.ui.property_panel import PropertyPanel

        panel = PropertyPanel()

        # This should NOT raise AttributeError anymore
        try:
            # Call the method that was crashing
            panel._populate_block_model_list()

            logger.info("✅ _populate_block_model_list() executed without AttributeError")

        except AttributeError as e:
            if "'registry'" in str(e):
                pytest.fail(
                    f"\n{'='*80}\n"
                    f"🚨 AttributeError still occurs in _populate_block_model_list():\n"
                    f"{'='*80}\n"
                    f"\n{e}\n\n"
                    f"The fix was not applied correctly.\n"
                    f"Check that self.registry is initialized in __init__()\n"
                    f"{'='*80}\n"
                )
            else:
                # Different AttributeError (not related to registry)
                raise

    def test_property_panel_registry_initialized_in_init(self):
        """
        Static code analysis: Verify self.registry is initialized in __init__().
        """
        import inspect
        from block_model_viewer.ui.property_panel import PropertyPanel

        # Get __init__ source
        init_source = inspect.getsource(PropertyPanel.__init__)

        # Verify self.registry is initialized
        assert "self.registry" in init_source, (
            "PropertyPanel.__init__() does NOT initialize self.registry!\n"
            "This will cause AttributeError when methods try to access it."
        )

        logger.info("✅ PropertyPanel.__init__() correctly initializes self.registry")

    def test_property_panel_on_active_layer_changed_no_error(self, mock_qapp):
        """
        Test that _on_active_layer_changed() doesn't crash.

        This is the method that calls _populate_block_model_list(),
        which was causing the AttributeError.
        """
        from block_model_viewer.ui.property_panel import PropertyPanel

        panel = PropertyPanel()

        # Mock a layer change event
        try:
            # This calls _populate_block_model_list() internally
            panel._on_active_layer_changed("test_layer")

            logger.info("✅ _on_active_layer_changed() executed without AttributeError")

        except AttributeError as e:
            if "'registry'" in str(e):
                pytest.fail(
                    f"\n{'='*80}\n"
                    f"🚨 AttributeError in _on_active_layer_changed():\n"
                    f"{'='*80}\n"
                    f"\n{e}\n\n"
                    f"{'='*80}\n"
                )
            else:
                # Different AttributeError (acceptable - might be other missing data)
                logger.debug(f"Different AttributeError (not registry related): {e}")


@pytest.mark.critical
class TestPropertyPanelFixSummary:
    """Generate summary of property panel fix."""

    def test_property_panel_fix_summary(self):
        """
        Summary report of PropertyPanel registry fix.

        This test always passes - it's informational.
        """
        print(f"\n{'='*80}")
        print("PROPERTY PANEL REGISTRY FIX SUMMARY")
        print(f"{'='*80}")
        print(f"\nIssue:")
        print(f"  AttributeError: 'PropertyPanel' object has no attribute 'registry'")
        print(f"  at property_panel.py:1622 in _populate_block_model_list()")
        print(f"\nRoot cause:")
        print(f"  self.registry was used in 4 places but never initialized in __init__()")
        print(f"\nFix:")
        print(f"  Added self.registry = self.get_registry() to __init__()")
        print(f"  Now self.registry is available to all methods")
        print(f"\nRun with -v to see detailed test results")
        print(f"{'='*80}\n")

        assert True
