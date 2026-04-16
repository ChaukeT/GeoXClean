"""
LoopStructural Geological Modeling Panel — sidebar navigation layout.

This is the main entry point. It combines:
- LoopStructuralTabsMixin (all 7 tab builders)
- LoopStructuralBusinessLogicMixin (all data/model/export logic)
- BaseAnalysisPanel (scroll area, registry access, signals)

Layout: fixed-width sidebar with nav buttons + QStackedWidget content area.
All styling is via objectName-based QSS — zero inline setStyleSheet calls.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QStackedWidget, QSizePolicy,
)

from ..base_analysis_panel import BaseAnalysisPanel
from ..design_tokens import tokens
from ._tabs import LoopStructuralTabsMixin
from ._business_logic import LoopStructuralBusinessLogicMixin
from ._worker import ModelBuildWorker

if TYPE_CHECKING:
    from ...controllers.app_controller import AppController
    from ...geology.chronos_engine import ChronosEngine
    from ...geology.industry_modeler import GeoXIndustryModeler
    from ...geology.compliance_manager import AuditReport
    from ...geology.fault_detection import SuggestedFault

logger = logging.getLogger(__name__)

# Step labels used in the sidebar navigation
_NAV_LABELS = [
    "Input Data",
    "Stratigraphy",
    "Domain",
    "Build",
    "Audit",
    "Advisory",
    "Export",
]


class LoopStructuralModelPanel(
    LoopStructuralTabsMixin,
    LoopStructuralBusinessLogicMixin,
    BaseAnalysisPanel,
):
    """
    Main UI Panel for LoopStructural-based Geological Modeling.

    Uses a sidebar navigation pattern with a QStackedWidget content area.
    No toolbar, no breadcrumb banner, no QTabWidget.

    Features:
    - Data input configuration (contacts, orientations, faults)
    - Model building with progress tracking
    - JORC/SAMREC compliance validation
    - Automatic fault suggestion from errors
    - Surface extraction and visualization
    - Export to mining software formats

    Architecture:
    - Uses ChronosEngine for coordinate handling and event stacking
    - Uses GeoXIndustryModeler for industry-grade FDI interpolation
    - Uses ComplianceManager for audit reporting
    - Uses FaultDetectionEngine for structural suggestions
    """

    task_name = "loopstructural_model"

    # Signals
    model_built = pyqtSignal(object)
    surfaces_extracted = pyqtSignal(list)
    compliance_validated = pyqtSignal(object)
    geology_package_ready = pyqtSignal(dict)

    def __init__(self, parent: Optional[QWidget] = None):
        # Private state (GeoX Panel Safety Rules)
        self._engine: Optional[ChronosEngine] = None
        self._modeler: Optional[GeoXIndustryModeler] = None
        self._model = None
        self._contacts_df: Optional[pd.DataFrame] = None
        self._orientations_df: Optional[pd.DataFrame] = None
        self._fault_list: List[Dict[str, Any]] = []
        self._stratigraphy: List[str] = []
        self._surfaces: List[Dict[str, Any]] = []
        self._solids: List[Dict[str, Any]] = []
        self._current_report: Optional[AuditReport] = None

        # UI widget references (initialized in setup_ui via tab builders)
        self._strat_list = None
        self._strat_input = None
        self._resolution_spin = None
        self._cgw_spin = None
        self._fault_table = None
        self._build_btn = None
        self._xmin_spin = None
        self._xmax_spin = None
        self._ymin_spin = None
        self._ymax_spin = None
        self._zmin_spin = None
        self._zmax_spin = None
        self._tabs = None

        # Model runner and results
        self._runner = None
        self._model_result = None
        self._unified_mesh = None

        # Build worker
        self._build_worker: Optional[ModelBuildWorker] = None
        self._build_start_time = None

        # Lithology grouping
        self._lith_grouping_widget = None
        self._lithology_mapping: Dict[str, str] = {}

        # Formation scalar values mapping
        self._formation_values: Dict[str, float] = {}

        # Stratigraphy validation result (set by business logic)
        self._strat_validation_result = None

        # Workflow state
        self._workflow_state = 0
        self._nav_buttons: List[QPushButton] = []

        # This calls _setup_base_ui() -> setup_ui() via base class chain
        super().__init__(parent=parent, panel_id="loopstructural_model_panel")

        self.setWindowTitle("Geological Modeling (LoopStructural)")
        self.setMinimumSize(900, 700)

    # ------------------------------------------------------------------
    # UI Setup (called by BaseAnalysisPanel._setup_base_ui)
    # ------------------------------------------------------------------

    def setup_ui(self) -> None:
        """Build the complete panel UI with sidebar navigation layout."""
        self.main_layout.setSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        # Root panel objectName for QSS
        self.setObjectName("LoopPanel")

        # Horizontal split: sidebar | content
        body = QHBoxLayout()
        body.setSpacing(0)
        body.setContentsMargins(0, 0, 0, 0)

        # --- Sidebar ---
        sidebar = self._build_sidebar()
        body.addWidget(sidebar)

        # --- Stacked content area ---
        self._tabs = QStackedWidget()
        self._tabs.setObjectName("LoopContent")
        self._tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding,
        )

        # Create all pages (from _tabs.py mixin methods)
        self._tabs.addWidget(self._create_data_tab())        # 0: Input Data
        self._tabs.addWidget(self._create_config_tab())       # 1: Stratigraphy
        self._tabs.addWidget(self._create_domain_tab())       # 2: Domain
        self._tabs.addWidget(self._create_build_tab())        # 3: Build
        self._tabs.addWidget(self._create_compliance_tab())   # 4: Audit
        self._tabs.addWidget(self._create_advisory_tab())     # 5: Advisory
        self._tabs.addWidget(self._create_export_tab())       # 6: Export

        body.addWidget(self._tabs, stretch=1)

        self.main_layout.addLayout(body)

        # Initialize sidebar visual state
        self._navigate_to(0)
        self._update_workflow_banner()

    # ------------------------------------------------------------------
    # Sidebar construction
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> QFrame:
        """Build the fixed-width sidebar with title, nav buttons, and actions."""
        sidebar = QFrame()
        sidebar.setObjectName("LoopSidebar")
        sidebar.setFixedWidth(200)
        sidebar.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(
            tokens.SPACING_MD, tokens.SPACING_LG,
            tokens.SPACING_MD, tokens.SPACING_MD,
        )
        layout.setSpacing(tokens.SPACING_XS)

        # --- Title block ---
        title = QLabel("Geological\nModeling")
        title.setObjectName("LoopSidebarTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel("LoopStructural")
        subtitle.setObjectName("LoopSidebarSubtitle")
        layout.addWidget(subtitle)

        layout.addSpacing(tokens.SPACING_MD)

        # --- Navigation buttons ---
        self._nav_buttons = []
        for idx, label in enumerate(_NAV_LABELS):
            btn = QPushButton(f"  {label}")
            btn.setObjectName("LoopNavButton")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda checked, i=idx: self._navigate_to(i))
            layout.addWidget(btn)
            self._nav_buttons.append(btn)

        layout.addStretch()

        # --- Separator ---
        divider = QFrame()
        divider.setObjectName("LoopDivider")
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        layout.addSpacing(tokens.SPACING_SM)

        # --- Action buttons ---
        self._build_btn = QPushButton("Build Model")
        self._build_btn.setObjectName("PrimaryButton")
        self._build_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._build_btn.clicked.connect(self._on_quick_build)
        layout.addWidget(self._build_btn)

        close_btn = QPushButton("Close")
        close_btn.setObjectName("LoopActionButton")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        return sidebar

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _navigate_to(self, idx: int) -> None:
        """Switch the stacked widget page and update sidebar active state."""
        if self._tabs is None or not self._nav_buttons:
            return

        count = self._tabs.count()
        if idx < 0 or idx >= count:
            return

        self._tabs.setCurrentIndex(idx)

        # Update active property on all nav buttons
        for i, btn in enumerate(self._nav_buttons):
            is_active = (i == idx)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ------------------------------------------------------------------
    # Workflow banner (redirected to sidebar nav state)
    # ------------------------------------------------------------------

    def _update_workflow_banner(self) -> None:
        """
        Update sidebar nav buttons to reflect workflow progress.

        - Steps before self._workflow_state: show checkmark prefix (completed)
        - Step equal to self._workflow_state: mark as active
        - Steps after self._workflow_state: normal/default state

        Also preserves the current page active highlight.
        """
        if not self._nav_buttons:
            return

        current_page = self._tabs.currentIndex() if self._tabs else 0

        for i, btn in enumerate(self._nav_buttons):
            label = _NAV_LABELS[i]

            if i < self._workflow_state:
                # Completed step — checkmark prefix
                btn.setText(f"  \u2713 {label}")
                btn.setProperty("completed", "true")
            elif i == self._workflow_state:
                # Current workflow step
                btn.setText(f"  \u25cf {label}")
                btn.setProperty("completed", "false")
            else:
                # Future step
                btn.setText(f"  {label}")
                btn.setProperty("completed", "false")

            # Active state reflects which page is currently shown
            is_active = (i == current_page)
            btn.setProperty("active", "true" if is_active else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    # ------------------------------------------------------------------
    # Panel lifecycle
    # ------------------------------------------------------------------

    def clear_panel(self):
        """Clear all panel UI and state to initial defaults."""
        self._model = None
        self._contacts_df = None
        self._orientations_df = None
        self._fault_list = []
        self._stratigraphy = []
        self._surfaces = []
        self._solids = []
        self._current_report = None
        self._formation_values = {}
        self._lithology_mapping = {}

        if self._strat_list:
            self._strat_list.clear()
        if self._fault_table:
            self._fault_table.setRowCount(0)

        if hasattr(self, '_build_status_widget') and self._build_status_widget:
            self._build_status_widget.reset()

        super().clear_panel()
        logger.info("LoopStructuralModelPanel: Panel fully cleared")

        self._workflow_state = 0
        self._navigate_to(0)
        self._update_workflow_banner()

    def refresh_theme(self) -> None:
        """
        No-op. Theme changes are handled by the application-level QSS.

        All widgets use objectName-based styling, so theme switches
        are automatic when the application QSS is regenerated.
        """
        pass
