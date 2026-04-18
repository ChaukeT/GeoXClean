"""
Menu Coordinator — extracted from MainWindow.

Owns the entire menu bar lifecycle:
  1. Build all 11 menus via the consolidated menu modules
  2. Run shortcut validation at startup
  3. Manage the Command Palette singleton
  4. Enable/disable actions based on application state
  5. Handle hover-to-open behavior
  6. Rebuild menus when needed (e.g. plugin load)

Integration:
    # In MainWindow.__init__:
    self._menu_coordinator = MenuCoordinator(self)

    # In MainWindow._setup_menus:
    self._menu_coordinator.setup_menus()

    # When model loads/unloads:
    self._menu_coordinator.on_model_loaded()
    self._menu_coordinator.on_model_unloaded()

    # When Ctrl+Shift+P is triggered:
    self._menu_coordinator.show_command_palette()
"""

import logging
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QObject, QEvent, QTimer
from PyQt6.QtWidgets import QMenuBar

if TYPE_CHECKING:
    from ..main_window import MainWindow

from .shortcut_registry import ShortcutRegistry
from .command_palette import CommandPalette

logger = logging.getLogger(__name__)


class MenuCoordinator(QObject):
    """
    Owns the menu bar lifecycle for MainWindow.

    Extracted from MainWindow to reduce its size and isolate
    menu-related concerns.
    """

    def __init__(self, main_window: 'MainWindow'):
        super().__init__(main_window)
        self._mw = main_window
        self._shortcut_registry: Optional[ShortcutRegistry] = None
        self._command_palette: Optional[CommandPalette] = None
        self._hover_filter: Optional['_MenuBarHoverFilter'] = None

    # ═══════════════════════════════════════════════════════════════
    # SETUP
    # ═══════════════════════════════════════════════════════════════

    def setup_menus(self):
        """
        Build all menus, validate shortcuts, initialize command palette.

        Called once during MainWindow initialization.
        """
        logger.info("MenuCoordinator: building menus")

        # 1. Build all 11 menus
        self._build_all_menus()

        # 2. Install hover-to-open behavior
        self._install_hover_filter()

        # 3. Validate shortcuts (deferred to avoid startup delay)
        QTimer.singleShot(500, self._validate_shortcuts)

        # 4. Wire Ctrl+Shift+P to command palette
        self._wire_command_palette()

        # 5. Set initial action states
        self._set_initial_states()

        logger.info("MenuCoordinator: setup complete")

    def rebuild_menus(self):
        """
        Full rebuild (e.g. after plugin load or language change).
        """
        logger.info("MenuCoordinator: rebuilding menus")
        self._build_all_menus()
        self._validate_shortcuts()
        if self._command_palette:
            self._command_palette.invalidate()

    # ═══════════════════════════════════════════════════════════════
    # MENU BUILDING
    # ═══════════════════════════════════════════════════════════════

    def _build_all_menus(self):
        """Build the consolidated menu bar (matches March Pictures layout + Drillholes).

        Order: File | Edit | View | Data | Drillholes | Modelling | Survey |
               Planning | Resources | Tools | Window | Help
        """
        from ..menus import (
            build_file_menu,
            build_edit_menu,
            build_view_menu,
            build_data_menu,
            build_drillholes_menu,
            build_modelling_menu,
            build_survey_menu,
            build_planning_menu,
            build_resources_menu,
            build_tools_menu,
            build_window_menu,
            build_help_menu,
        )

        mw = self._mw
        menubar = mw.menuBar()
        menubar.clear()

        builders = [
            ("file_menu", build_file_menu),
            ("edit_menu", build_edit_menu),
            ("view_menu", build_view_menu),
            ("data_menu", build_data_menu),
            ("drillholes_menu", build_drillholes_menu),
            ("modelling_menu", build_modelling_menu),
            ("survey_menu", build_survey_menu),
            ("planning_menu", build_planning_menu),
            ("resources_menu", build_resources_menu),
            ("tools_menu", build_tools_menu),
            ("window_menu", build_window_menu),
            ("help_menu", build_help_menu),
        ]
        for attr, builder in builders:
            try:
                setattr(mw, attr, builder(mw, menubar))
            except Exception as exc:
                logger.error("Failed to build %s: %s", attr, exc)
                setattr(mw, attr, menubar.addMenu(attr.replace("_", " ").title()))

    # ═══════════════════════════════════════════════════════════════
    # SHORTCUT VALIDATION
    # ═══════════════════════════════════════════════════════════════

    def _validate_shortcuts(self):
        """Run shortcut collision detection."""
        self._shortcut_registry = ShortcutRegistry(self._mw)
        collisions = self._shortcut_registry.validate()

        if collisions:
            logger.warning(
                "MenuCoordinator: %d shortcut collisions detected! "
                "Check logs for details.", len(collisions)
            )
        else:
            logger.info(
                "MenuCoordinator: all %d shortcuts validated, no collisions",
                self._shortcut_registry.count
            )

    @property
    def shortcut_registry(self) -> Optional[ShortcutRegistry]:
        return self._shortcut_registry

    # ═══════════════════════════════════════════════════════════════
    # COMMAND PALETTE
    # ═══════════════════════════════════════════════════════════════

    def _wire_command_palette(self):
        """
        Connect the _search_modules method to the command palette.

        The tools_menu already wires Ctrl+Shift+P to
        main_window._search_modules. We replace that method
        with our command palette.
        """
        self._command_palette = CommandPalette(self._mw)
        # Override the main_window method to use our palette
        self._mw._search_modules = self.show_command_palette

    def show_command_palette(self):
        """Show the command palette dialog."""
        if self._command_palette is None:
            self._command_palette = CommandPalette(self._mw)
        self._command_palette.show()

    # ═══════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════

    def _set_initial_states(self):
        """Disable actions that require a loaded model."""
        mw = self._mw
        has_model = getattr(mw, 'current_model', None) is not None

        # Data table requires a model
        if hasattr(mw, 'view_data_action'):
            mw.view_data_action.setEnabled(has_model)

    def on_model_loaded(self):
        """
        Called when a block model is loaded. Enables model-dependent actions.
        """
        mw = self._mw

        if hasattr(mw, 'view_data_action'):
            mw.view_data_action.setEnabled(True)

        logger.debug("MenuCoordinator: model loaded — actions enabled")

    def on_model_unloaded(self):
        """
        Called when the scene is cleared. Disables model-dependent actions.
        """
        mw = self._mw

        if hasattr(mw, 'view_data_action'):
            mw.view_data_action.setEnabled(False)

        logger.debug("MenuCoordinator: model unloaded — actions disabled")

    def on_drillhole_loaded(self):
        """Enable drillhole-dependent actions."""
        logger.debug("MenuCoordinator: drillhole data loaded")

    def on_estimation_complete(self):
        """Enable resource actions after estimation completes."""
        logger.debug("MenuCoordinator: estimation complete — resource actions available")

    # ═══════════════════════════════════════════════════════════════
    # HOVER-TO-OPEN
    # ═══════════════════════════════════════════════════════════════

    def _install_hover_filter(self):
        """Install hover-to-open event filter on the menu bar."""
        menubar = self._mw.menuBar()
        if not menubar:
            return

        self._hover_filter = _MenuBarHoverFilter(menubar)
        menubar.installEventFilter(self._hover_filter)

    # ═══════════════════════════════════════════════════════════════
    # KEYBOARD SHORTCUTS DIALOG
    # ═══════════════════════════════════════════════════════════════

    def show_shortcuts_dialog(self):
        """
        Show the keyboard shortcuts reference dialog.

        Can be wired to main_window.show_shortcuts if desired.
        """
        if not self._shortcut_registry:
            self._validate_shortcuts()

        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

        dialog = QDialog(self._mw)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setMinimumSize(500, 600)

        layout = QVBoxLayout(dialog)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setFont(_monospace_font())
        text.setPlainText(self._shortcut_registry.format_reference())
        layout.addWidget(text)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)

        dialog.exec()

    # ═══════════════════════════════════════════════════════════════
    # ACCESSORS
    # ═══════════════════════════════════════════════════════════════

    @property
    def command_palette(self) -> Optional[CommandPalette]:
        return self._command_palette


# ─── Helpers ─────────────────────────────────────────────────────

class _MenuBarHoverFilter(QObject):
    """
    Event filter that enables hover-to-open on the menu bar.

    When any menu is already open and the user hovers over another
    top-level item, the new menu opens instantly. When no menu is open,
    hover activates after a brief delay.
    """

    def __init__(self, menubar: QMenuBar):
        super().__init__(menubar)
        self._menubar = menubar
        self._hover_timer = QTimer()
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(150)
        self._hover_timer.timeout.connect(self._open_hovered_menu)
        self._pending_action = None

    def _open_hovered_menu(self):
        if self._pending_action and self._pending_action.menu():
            self._menubar.setActiveAction(self._pending_action)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.MouseMove:
            action = self._menubar.actionAt(event.pos())
            if action and action.menu():
                if self._menubar.activeAction():
                    # A menu is already open — switch instantly
                    self._menubar.setActiveAction(action)
                    self._hover_timer.stop()
                else:
                    # No menu open — delay before opening
                    if action != self._pending_action:
                        self._pending_action = action
                        self._hover_timer.start()
            else:
                self._hover_timer.stop()
                self._pending_action = None

        return super().eventFilter(obj, event)


def _monospace_font():
    """Return a monospace font suitable for the shortcuts dialog."""
    from PyQt6.QtGui import QFont
    font = QFont("Consolas")
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPointSize(10)
    return font
