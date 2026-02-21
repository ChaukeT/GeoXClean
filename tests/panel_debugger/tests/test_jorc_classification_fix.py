"""
Test JORC Classification Panel Fix

Tests that ClassificationCategoryCard properly has _get_stylesheet() method.

Root cause: _get_stylesheet() and refresh_theme() were defined in ClassificationWorker
instead of ClassificationCategoryCard, causing AttributeError.

Fix: Moved methods to ClassificationCategoryCard where they're actually called.
"""

import pytest
import logging

logger = logging.getLogger(__name__)


@pytest.mark.critical
class TestJORCClassificationFix:
    """Test that JORC Classification Panel components work correctly."""

    def test_classification_category_card_has_get_stylesheet(self, mock_qapp):
        """
        Test that ClassificationCategoryCard has _get_stylesheet() method.

        Before fix: AttributeError: 'ClassificationCategoryCard' object has no attribute '_get_stylesheet'
        After fix: Method exists and works correctly
        """
        from block_model_viewer.ui.jorc_classification_panel import ClassificationCategoryCard

        try:
            card = ClassificationCategoryCard(
                category="Measured",
                color="#4CAF50",
                default_dist_pct=100,
                default_min_holes=8
            )

            # Verify card was created
            assert card is not None, "Card is None"

            # Verify _get_stylesheet() method exists
            assert hasattr(card, '_get_stylesheet'), (
                "ClassificationCategoryCard missing '_get_stylesheet' method!\n"
                "This will cause AttributeError during __init__()"
            )

            # Verify the method works
            stylesheet = card._get_stylesheet()
            assert isinstance(stylesheet, str), "Stylesheet is not a string"
            assert len(stylesheet) > 0, "Stylesheet is empty"
            assert "#4CAF50" in stylesheet, "Color not in stylesheet"

            logger.info("✅ ClassificationCategoryCard has working _get_stylesheet() method")

        except AttributeError as e:
            if "'_get_stylesheet'" in str(e):
                pytest.fail(
                    f"\n{'='*80}\n"
                    f"🚨 ClassificationCategoryCard instantiation failed:\n"
                    f"{'='*80}\n"
                    f"\n{e}\n\n"
                    f"The fix was not applied correctly.\n"
                    f"_get_stylesheet() method must be in ClassificationCategoryCard,\n"
                    f"not in ClassificationWorker!\n"
                    f"{'='*80}\n"
                )
            else:
                raise

    def test_classification_category_card_has_color_attribute(self, mock_qapp):
        """
        Test that ClassificationCategoryCard stores the color parameter.

        The color is needed for _get_stylesheet() to work.
        """
        from block_model_viewer.ui.jorc_classification_panel import ClassificationCategoryCard

        card = ClassificationCategoryCard(
            category="Indicated",
            color="#FF9800",
            default_dist_pct=150,
            default_min_holes=6
        )

        # Verify color attribute exists
        assert hasattr(card, 'color'), (
            "ClassificationCategoryCard missing 'color' attribute!\n"
            "This is needed for _get_stylesheet() to reference self.color"
        )

        # Verify it's the correct color
        assert card.color == "#FF9800", f"Wrong color: {card.color}"

        logger.info("✅ ClassificationCategoryCard stores color correctly")

    def test_classification_category_card_refresh_theme(self, mock_qapp):
        """
        Test that ClassificationCategoryCard has refresh_theme() method.
        """
        from block_model_viewer.ui.jorc_classification_panel import ClassificationCategoryCard

        card = ClassificationCategoryCard(
            category="Inferred",
            color="#2196F3",
            default_dist_pct=200,
            default_min_holes=4
        )

        # Verify refresh_theme() method exists
        assert hasattr(card, 'refresh_theme'), (
            "ClassificationCategoryCard missing 'refresh_theme' method!"
        )

        # Call it (should not crash)
        try:
            card.refresh_theme()
            logger.info("✅ ClassificationCategoryCard.refresh_theme() works")
        except Exception as e:
            pytest.fail(f"refresh_theme() failed: {e}")

    def test_classification_worker_no_stylesheet_methods(self):
        """
        Test that ClassificationWorker does NOT have stylesheet methods.

        These methods were incorrectly placed in ClassificationWorker.
        They should only be in ClassificationCategoryCard.
        """
        from block_model_viewer.ui.jorc_classification_panel import ClassificationWorker

        # ClassificationWorker should NOT have these UI methods
        assert not hasattr(ClassificationWorker, '_get_stylesheet'), (
            "ClassificationWorker still has _get_stylesheet()!\n"
            "This method should only be in ClassificationCategoryCard."
        )

        assert not hasattr(ClassificationWorker, 'refresh_theme'), (
            "ClassificationWorker still has refresh_theme()!\n"
            "This method should only be in ClassificationCategoryCard."
        )

        logger.info("✅ ClassificationWorker correctly does NOT have stylesheet methods")

    def test_jorc_classification_panel_instantiates(self, mock_qapp):
        """
        Test that JORCClassificationPanel can be instantiated.

        This is the full integration test - the panel creates ClassificationCategoryCard
        instances which call _get_stylesheet() in their __init__().
        """
        from block_model_viewer.ui.jorc_classification_panel import JORCClassificationPanel

        try:
            panel = JORCClassificationPanel()

            assert panel is not None, "Panel is None"
            logger.info("✅ JORCClassificationPanel instantiated successfully")

        except AttributeError as e:
            if "'_get_stylesheet'" in str(e):
                pytest.fail(
                    f"\n{'='*80}\n"
                    f"🚨 JORCClassificationPanel instantiation failed:\n"
                    f"{'='*80}\n"
                    f"\n{e}\n\n"
                    f"This means ClassificationCategoryCard still doesn't have\n"
                    f"_get_stylesheet() method in the right place.\n"
                    f"{'='*80}\n"
                )
            else:
                raise


@pytest.mark.critical
class TestJORCClassificationFixSummary:
    """Generate summary of JORC Classification fix."""

    def test_jorc_classification_fix_summary(self):
        """
        Summary report of JORC Classification Panel fix.

        This test always passes - it's informational.
        """
        print(f"\n{'='*80}")
        print("JORC CLASSIFICATION PANEL FIX SUMMARY")
        print(f"{'='*80}")
        print(f"\nIssue:")
        print(f"  AttributeError: 'ClassificationCategoryCard' object has no")
        print(f"  attribute '_get_stylesheet'")
        print(f"  at jorc_classification_panel.py:231")
        print(f"\nRoot cause:")
        print(f"  _get_stylesheet() and refresh_theme() were defined in")
        print(f"  ClassificationWorker (wrong class!) instead of")
        print(f"  ClassificationCategoryCard (correct class)")
        print(f"\nFix:")
        print(f"  1. Added self.color = color to ClassificationCategoryCard.__init__()")
        print(f"  2. Moved _get_stylesheet() from ClassificationWorker to")
        print(f"     ClassificationCategoryCard")
        print(f"  3. Moved refresh_theme() from ClassificationWorker to")
        print(f"     ClassificationCategoryCard")
        print(f"\nRun with -v to see detailed test results")
        print(f"{'='*80}\n")

        assert True
