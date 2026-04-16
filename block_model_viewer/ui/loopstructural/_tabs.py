"""
LoopStructural tab builder mixin -- modern app-screen page designs.

Each page feels like a distinct screen in a modern application (think Figma,
VS Code settings, or macOS System Preferences). Pages go into a
QStackedWidget; no tab titles are generated here.

Widget trees are constructed from objectName + tokens.SPACING_* for layout
margins and spacing.  Business-logic signal connections reference
``self._method_name`` stubs that live in the business-logic mixin.

CRITICAL: This file contains ZERO calls to setStyleSheet().
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QScrollArea,
    QListWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QSizePolicy,
    QSplitter,
    QGridLayout,
)

from ..design_tokens import tokens
from ..collapsible_group import CollapsibleGroup
from ._widgets import (
    InputValidationChecklist,
    GeologicalDomainPanel,
    LithologyGroupingWidget,
    GeologicalAuditVerdictTable,
    ModelBuildExecutionPanel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers (no styling -- layout shortcuts only)
# ---------------------------------------------------------------------------

def _make_card(object_name: str = "Card") -> tuple[QFrame, QVBoxLayout]:
    """Return (frame, layout) for a card container with standard padding."""
    card = QFrame()
    card.setObjectName(object_name)
    lay = QVBoxLayout(card)
    lay.setContentsMargins(
        tokens.SPACING_LG, tokens.SPACING_MD,
        tokens.SPACING_LG, tokens.SPACING_MD,
    )
    lay.setSpacing(tokens.SPACING_SM)
    return card, lay


def _make_section_header(text: str) -> QLabel:
    """Return an uppercase section header label with the standard objectName."""
    lbl = QLabel(text)
    lbl.setObjectName("LoopSectionHeader")
    return lbl


def _make_divider() -> QFrame:
    """Return a 1 px horizontal divider."""
    div = QFrame()
    div.setObjectName("LoopDivider")
    return div


def _make_action_bar() -> tuple[QFrame, QHBoxLayout]:
    """Return (frame, layout) for a flat action-bar row."""
    bar = QFrame()
    bar.setObjectName("LoopActionBar")
    lay = QHBoxLayout(bar)
    lay.setContentsMargins(
        tokens.SPACING_SM, tokens.SPACING_XS,
        tokens.SPACING_SM, tokens.SPACING_XS,
    )
    lay.setSpacing(tokens.SPACING_SM)
    return bar, lay


def _make_action_button(text: str, *, danger: bool = False) -> QPushButton:
    """Return a flat text button for action bars."""
    btn = QPushButton(text)
    btn.setObjectName("LoopActionButtonDanger" if danger else "LoopActionButton")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def _make_muted_button(text: str) -> QPushButton:
    """Return a small muted icon-style button (arrows, etc.)."""
    btn = QPushButton(text)
    btn.setObjectName("LoopMutedButton")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setFixedSize(36, 36)
    return btn


def _make_form_row(label_text: str) -> tuple[QWidget, QHBoxLayout]:
    """Return (widget, hlayout) for a label-left / input-right parameter row."""
    row = QWidget()
    row.setObjectName("LoopFormRow")
    lay = QHBoxLayout(row)
    lay.setContentsMargins(
        tokens.SPACING_LG, tokens.SPACING_SM,
        tokens.SPACING_LG, tokens.SPACING_SM,
    )
    lay.setSpacing(tokens.SPACING_SM)
    lbl = QLabel(label_text)
    lay.addWidget(lbl)
    lay.addStretch()
    return row, lay


def _make_primary_button(text: str) -> QPushButton:
    """Return a prominent primary-action button."""
    btn = QPushButton(text)
    btn.setObjectName("PrimaryButton")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


def _make_info_text(text: str) -> QLabel:
    """Return a secondary info text label."""
    lbl = QLabel(text)
    lbl.setObjectName("LoopInfoText")
    lbl.setWordWrap(True)
    return lbl


def _make_scrollable(content: QWidget) -> QWidget:
    """Wrap *content* in a frameless QScrollArea and return a tab-root widget."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)

    tab = QWidget()
    tab_layout = QVBoxLayout(tab)
    tab_layout.setContentsMargins(0, 0, 0, 0)
    tab_layout.addWidget(scroll)
    return tab


def _tab_margins(layout) -> None:
    """Apply standard tab-level margins and spacing to a layout."""
    layout.setContentsMargins(
        tokens.SPACING_LG, tokens.SPACING_MD,
        tokens.SPACING_LG, tokens.SPACING_MD,
    )
    layout.setSpacing(tokens.SPACING_MD)


def _make_option_card(icon: str, title: str, description: str) -> QPushButton:
    """Return a clickable LoopOptionCard button with icon, title, and subtitle.

    The button is styled entirely via objectName 'LoopOptionCard'.
    Internal layout: icon on the left, title+description stacked on the right.
    """
    btn = QPushButton()
    btn.setObjectName("LoopOptionCard")
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    btn.setMinimumHeight(80)

    # Build internal layout inside the button
    card_layout = QHBoxLayout(btn)
    card_layout.setContentsMargins(
        tokens.SPACING_LG, tokens.SPACING_MD,
        tokens.SPACING_LG, tokens.SPACING_MD,
    )
    card_layout.setSpacing(tokens.SPACING_MD)

    icon_label = QLabel(icon)
    icon_label.setObjectName("LoopWelcomeTitle")
    icon_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    icon_label.setFixedWidth(40)
    card_layout.addWidget(icon_label)

    text_col = QVBoxLayout()
    text_col.setSpacing(tokens.SPACING_XS)

    title_lbl = QLabel(title)
    title_lbl.setObjectName("LoopSectionHeader")
    title_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    text_col.addWidget(title_lbl)

    desc_lbl = QLabel(description)
    desc_lbl.setObjectName("LoopInfoText")
    desc_lbl.setWordWrap(True)
    desc_lbl.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    text_col.addWidget(desc_lbl)

    card_layout.addLayout(text_col, stretch=1)

    return btn


# ============================================================================
# Mixin
# ============================================================================

class LoopStructuralTabsMixin:
    """Mixin providing all 7 page builder methods.

    Every method creates a complete widget tree that is styled exclusively
    through ``objectName``-based QSS rules (zero ``setStyleSheet`` calls).
    Business-logic attributes (``self._xxx``) referenced elsewhere in the
    panel are assigned here.

    Pages are returned as QWidget for insertion into a QStackedWidget.
    """

    # ------------------------------------------------------------------
    # 1. Input Data page  (landing / welcome screen)
    # ------------------------------------------------------------------
    def _create_data_tab(self) -> QWidget:
        """Input Data page -- welcome header, two-column option cards,
        requirements grid below."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        _tab_margins(layout)

        # ============================================================
        # Welcome header area (full width)
        # ============================================================
        welcome_card = QFrame()
        welcome_card.setObjectName("LoopWelcomeCard")
        welcome_inner = QVBoxLayout(welcome_card)
        welcome_inner.setContentsMargins(
            tokens.SPACING_XL, tokens.SPACING_LG,
            tokens.SPACING_XL, tokens.SPACING_LG,
        )
        welcome_inner.setSpacing(tokens.SPACING_SM)

        welcome_title = QLabel("\U0001f52c  Geological Model Setup")
        welcome_title.setObjectName("LoopWelcomeTitle")
        welcome_inner.addWidget(welcome_title)

        welcome_desc = _make_info_text(
            "Load your drillhole data to begin building a geological model. "
            "Choose a source below to get started."
        )
        welcome_inner.addWidget(welcome_desc)

        layout.addWidget(welcome_card)

        # ============================================================
        # Two-column option cards (side by side)
        # ============================================================
        cards_row = QHBoxLayout()
        cards_row.setSpacing(tokens.SPACING_MD)

        self._load_registry_btn = _make_option_card(
            "\U0001f4c1", "FROM REGISTRY",
            "Load from your data registry",
        )
        self._load_registry_btn.clicked.connect(self._on_load_from_registry)
        cards_row.addWidget(self._load_registry_btn)

        self._load_file_btn = _make_option_card(
            "\U0001f4c2", "FROM FILE",
            "Import CSV / Excel from disk",
        )
        self._load_file_btn.clicked.connect(self._on_load_from_file)
        cards_row.addWidget(self._load_file_btn)

        layout.addLayout(cards_row)

        # ============================================================
        # Data Requirements card (full width below)
        # ============================================================
        req_card, req_layout = _make_card()

        req_header = _make_section_header("DATA REQUIREMENTS")
        req_layout.addWidget(req_header)

        self._validation_checklist = InputValidationChecklist()
        req_layout.addWidget(self._validation_checklist)

        layout.addWidget(req_card, stretch=1)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # 2. Stratigraphy / Config page  (two-column split)
    # ------------------------------------------------------------------
    def _create_config_tab(self) -> QWidget:
        """Stratigraphy page -- left column: strat list + faults;
        right column: parameters + orientation + lithology grouping.
        Scrollable."""
        content = QWidget()
        root_layout = QVBoxLayout(content)
        _tab_margins(root_layout)

        # Two-column splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ==============================================================
        # LEFT COLUMN
        # ==============================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(tokens.SPACING_MD)

        # --- Stratigraphic Sequence Card ---
        strat_card, strat_card_layout = _make_card()

        strat_header_row = QHBoxLayout()
        strat_header_row.setSpacing(tokens.SPACING_SM)
        strat_title = _make_section_header("STRATIGRAPHIC SEQUENCE")
        strat_header_row.addWidget(strat_title)

        self._strat_validation_label = QLabel("Pending")
        self._strat_validation_label.setObjectName("LoopStatusPill")
        strat_header_row.addWidget(self._strat_validation_label)
        strat_header_row.addStretch()
        strat_card_layout.addLayout(strat_header_row)

        strat_card_layout.addWidget(_make_divider())

        self._strat_list = QListWidget()
        self._strat_list.setObjectName("LoopFlatList")
        self._strat_list.setMinimumHeight(120)
        self._strat_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection,
        )
        self._strat_list.setDragDropMode(
            QListWidget.DragDropMode.InternalMove,
        )
        strat_card_layout.addWidget(self._strat_list)

        strat_card_layout.addWidget(_make_divider())

        # Action bar: move arrows + Add / Remove
        strat_bar, strat_bar_layout = _make_action_bar()

        strat_up_btn = _make_muted_button("\u25b2")
        strat_up_btn.setToolTip("Move up (younger)")
        strat_up_btn.clicked.connect(self._move_strat_up)
        strat_bar_layout.addWidget(strat_up_btn)

        strat_down_btn = _make_muted_button("\u25bc")
        strat_down_btn.setToolTip("Move down (older)")
        strat_down_btn.clicked.connect(self._move_strat_down)
        strat_bar_layout.addWidget(strat_down_btn)

        strat_bar_layout.addStretch()

        strat_add_btn = _make_action_button("+ Add")
        strat_add_btn.setToolTip("Add a new formation")
        strat_add_btn.clicked.connect(self._add_strat_unit)
        strat_bar_layout.addWidget(strat_add_btn)

        strat_remove_btn = _make_action_button("Remove", danger=True)
        strat_remove_btn.setToolTip("Remove selected formation")
        strat_remove_btn.clicked.connect(self._remove_strat_unit)
        strat_bar_layout.addWidget(strat_remove_btn)

        strat_card_layout.addWidget(strat_bar)

        # Hidden text input for back-compat (stores newline-separated list)
        self._strat_input = QTextEdit()
        self._strat_input.setVisible(False)
        strat_card_layout.addWidget(self._strat_input)

        # Sync list changes to hidden text
        self._strat_list.model().rowsMoved.connect(self._sync_strat_to_text)

        left_layout.addWidget(strat_card, stretch=1)

        # --- Fault Events Card ---
        fault_card, fault_card_layout = _make_card()

        fault_header_row = QHBoxLayout()
        fault_header_row.setSpacing(tokens.SPACING_SM)
        fault_title = _make_section_header("FAULT EVENTS")
        fault_header_row.addWidget(fault_title)
        fault_hint = QLabel("Applied before stratigraphy")
        fault_hint.setObjectName("LoopParamHint")
        fault_header_row.addWidget(fault_hint)
        fault_header_row.addStretch()
        fault_card_layout.addLayout(fault_header_row)

        fault_card_layout.addWidget(_make_divider())

        self._fault_table = QTableWidget(0, 3)
        self._fault_table.setObjectName("LoopFlatTable")
        self._fault_table.setHorizontalHeaderLabels(
            ["Name", "Displacement (m)", "Type"],
        )
        self._fault_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch,
        )
        self._fault_table.setMaximumHeight(150)
        self._fault_table.verticalHeader().setVisible(False)
        self._fault_table.setShowGrid(False)
        fault_card_layout.addWidget(self._fault_table)

        fault_card_layout.addWidget(_make_divider())

        fault_bar, fault_bar_layout = _make_action_bar()
        fault_bar_layout.addStretch()

        add_fault_btn = _make_action_button("+ Add Fault")
        add_fault_btn.clicked.connect(self._on_add_fault)
        fault_bar_layout.addWidget(add_fault_btn)

        remove_fault_btn = _make_action_button("Remove", danger=True)
        remove_fault_btn.clicked.connect(self._on_remove_fault)
        fault_bar_layout.addWidget(remove_fault_btn)

        fault_card_layout.addWidget(fault_bar)

        left_layout.addWidget(fault_card)

        splitter.addWidget(left_widget)

        # ==============================================================
        # RIGHT COLUMN
        # ==============================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(tokens.SPACING_MD)

        # --- Parameters Card ---
        params_card, params_card_layout = _make_card()

        params_card_layout.addWidget(_make_section_header("PARAMETERS"))

        # Resolution
        res_row, res_lay = _make_form_row("Resolution")
        self._resolution_spin = QSpinBox()
        self._resolution_spin.setRange(10, 200)
        self._resolution_spin.setValue(100)
        self._resolution_spin.setToolTip(
            "Grid cells per axis (higher = more detail)",
        )
        res_lay.addWidget(self._resolution_spin)
        unit_lbl = QLabel("cells/axis")
        unit_lbl.setObjectName("LoopParamHint")
        res_lay.addWidget(unit_lbl)
        params_card_layout.addWidget(res_row)

        # Smoothing (CGW)
        cgw_row, cgw_lay = _make_form_row("Smoothing (CGW)")
        self._cgw_spin = QDoubleSpinBox()
        self._cgw_spin.setRange(0.001, 1.0)
        self._cgw_spin.setValue(0.005)
        self._cgw_spin.setSingleStep(0.005)
        self._cgw_spin.setDecimals(3)
        self._cgw_spin.setToolTip(
            "Regularization weight (lower = tighter fit to data)\n"
            "0.005 = tight fit (recommended)\n"
            "0.01-0.03 = moderate\n"
            "0.1+ = heavy smoothing"
        )
        cgw_lay.addWidget(self._cgw_spin)
        params_card_layout.addWidget(cgw_row)

        # Interpolator
        interp_row, interp_lay = _make_form_row("Interpolator")
        self._interp_combo = QComboBox()
        self._interp_combo.addItems([
            "FDI (Finite Difference)",
            "PLI (Piece-wise Linear)",
        ])
        self._interp_combo.setToolTip("FDI is best for layered rocks")
        interp_lay.addWidget(self._interp_combo)
        params_card_layout.addWidget(interp_row)

        # Taubin smoothing
        smooth_row, smooth_lay = _make_form_row("")
        # Remove the empty label that _make_form_row added
        item = smooth_lay.takeAt(0)
        if item and item.widget():
            item.widget().deleteLater()
        self._smooth_check = QCheckBox("Taubin Smoothing")
        self._smooth_check.setChecked(True)
        self._smooth_check.setToolTip(
            "Apply Taubin smoothing to remove voxel artifacts (recommended)",
        )
        smooth_lay.insertWidget(0, self._smooth_check)
        iter_lbl = QLabel("Iterations")
        iter_lbl.setObjectName("LoopParamHint")
        smooth_lay.addWidget(iter_lbl)
        self._smooth_iter_spin = QSpinBox()
        self._smooth_iter_spin.setRange(0, 100)
        self._smooth_iter_spin.setValue(20)
        self._smooth_iter_spin.setToolTip(
            "Number of smoothing passes (20 = industry standard)",
        )
        smooth_lay.addWidget(self._smooth_iter_spin)
        params_card_layout.addWidget(smooth_row)

        # Smart parameter hint (hidden until business logic shows it)
        self._param_hint_label = QLabel("")
        self._param_hint_label.setObjectName("LoopParamHint")
        self._param_hint_label.setWordWrap(True)
        self._param_hint_label.hide()
        params_card_layout.addWidget(self._param_hint_label)

        params_card_layout.addWidget(_make_divider())

        # Orientation Computation sub-section
        orient_header = _make_section_header("ORIENTATION")
        params_card_layout.addWidget(orient_header)

        self._compute_gradients_check = QCheckBox(
            "Compute gradients from contact geometry (PCA)",
        )
        self._compute_gradients_check.setChecked(True)
        self._compute_gradients_check.setToolTip(
            "Derives gradient vectors from contact point clouds using PCA.\n"
            "Produces geologically realistic orientations.\n"
            "RECOMMENDED: Leave enabled."
        )
        params_card_layout.addWidget(self._compute_gradients_check)

        self._allow_synthetic_check = QCheckBox(
            "Allow synthetic fallback if computation fails",
        )
        self._allow_synthetic_check.setChecked(True)
        self._allow_synthetic_check.setToolTip(
            "Fall back to synthetic horizontal orientations (0,0,1)\n"
            "if gradient computation fails."
        )
        params_card_layout.addWidget(self._allow_synthetic_check)

        right_layout.addWidget(params_card, stretch=1)

        # --- Lithology Grouping (collapsible) ---
        lith_group = CollapsibleGroup(
            "Lithology Grouping (Optional)", collapsed=True,
        )
        self._lith_grouping_widget = LithologyGroupingWidget()
        self._lith_grouping_widget.grouping_changed.connect(
            self._on_lithology_grouping_changed,
        )
        lith_group.add_widget(self._lith_grouping_widget)
        right_layout.addWidget(lith_group)

        splitter.addWidget(right_widget)

        # Even 50/50 stretch
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter, stretch=1)

        return _make_scrollable(content)

    # ------------------------------------------------------------------
    # 3. Domain page
    # ------------------------------------------------------------------
    def _create_domain_tab(self) -> QWidget:
        """Domain page -- geological domain card + data summary card."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        _tab_margins(layout)

        # ============================================================
        # Geological Domain Panel Card
        # ============================================================
        domain_card, domain_card_layout = _make_card()

        domain_card_layout.addWidget(
            _make_section_header("GEOLOGICAL DOMAIN"),
        )

        self._domain_panel = GeologicalDomainPanel()
        domain_card_layout.addWidget(self._domain_panel)

        # Alias internal spinboxes for backward compatibility
        self._xmin_spin = self._domain_panel._spinboxes['xmin']
        self._xmax_spin = self._domain_panel._spinboxes['xmax']
        self._ymin_spin = self._domain_panel._spinboxes['ymin']
        self._ymax_spin = self._domain_panel._spinboxes['ymax']
        self._zmin_spin = self._domain_panel._spinboxes['zmin']
        self._zmax_spin = self._domain_panel._spinboxes['zmax']

        layout.addWidget(domain_card)

        # ============================================================
        # Data Summary Card
        # ============================================================
        summary_card, summary_card_layout = _make_card()

        summary_card_layout.addWidget(
            _make_section_header("DATA SUMMARY"),
        )

        self._data_summary = QTextEdit()
        self._data_summary.setObjectName("LoopDataSummary")
        self._data_summary.setReadOnly(True)
        self._data_summary.setMinimumHeight(80)
        self._data_summary.setPlainText("No data loaded.")
        summary_card_layout.addWidget(self._data_summary)

        layout.addWidget(summary_card)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # 4. Build page
    # ------------------------------------------------------------------
    def _create_build_tab(self) -> QWidget:
        """Build page -- execution panel card + post-build actions card."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        _tab_margins(layout)

        # ============================================================
        # Model Build Execution Panel Card
        # ============================================================
        build_card, build_card_layout = _make_card()

        self._build_panel = ModelBuildExecutionPanel()
        self._build_panel.build_requested.connect(self._on_build_model)
        self._build_panel.cancel_requested.connect(self._on_cancel_build)
        build_card_layout.addWidget(self._build_panel)

        layout.addWidget(build_card)

        # ============================================================
        # Post-build Actions Card
        # ============================================================
        actions_card, actions_card_layout = _make_card()

        actions_card_layout.addWidget(
            _make_section_header("POST-BUILD ACTIONS"),
        )

        actions_row = QHBoxLayout()
        actions_row.setSpacing(tokens.SPACING_MD)

        self._extract_btn = _make_primary_button("Extract Geology")
        self._extract_btn.clicked.connect(self._on_extract_geology)
        self._extract_btn.setEnabled(False)
        self._extract_btn.setToolTip("Build model first")
        actions_row.addWidget(self._extract_btn)

        self._validate_btn = _make_primary_button("Run Audit")
        self._validate_btn.clicked.connect(self._on_validate_compliance)
        self._validate_btn.setEnabled(False)
        self._validate_btn.setToolTip("Build model first")
        actions_row.addWidget(self._validate_btn)

        actions_row.addStretch()
        actions_card_layout.addLayout(actions_row)

        layout.addWidget(actions_card)

        layout.addStretch()
        return tab

    # ------------------------------------------------------------------
    # 5. Audit / Compliance page
    # ------------------------------------------------------------------
    def _create_compliance_tab(self) -> QWidget:
        """Audit page -- scrollable: banner, verdict table, JORC thresholds,
        detailed compliance collapsible."""
        from ...geology.compliance_manager import (
            JORCThresholds,
            DEFAULT_JORC_THRESHOLDS,
        )

        content = QWidget()
        layout = QVBoxLayout(content)
        _tab_margins(layout)

        # ============================================================
        # Audit summary banner
        # ============================================================
        self._audit_summary_banner = QFrame()
        self._audit_summary_banner.setObjectName("LoopAuditBanner")
        banner_layout = QHBoxLayout(self._audit_summary_banner)
        banner_layout.setContentsMargins(
            tokens.SPACING_LG, tokens.SPACING_MD,
            tokens.SPACING_LG, tokens.SPACING_MD,
        )
        banner_layout.setSpacing(tokens.SPACING_MD)

        self._audit_status_icon = QLabel("--")
        self._audit_status_icon.setFixedWidth(28)
        self._audit_status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner_layout.addWidget(self._audit_status_icon)

        audit_text_layout = QVBoxLayout()
        audit_text_layout.setSpacing(tokens.SPACING_2XS)

        self._audit_status_label = QLabel("Audit Not Run")
        audit_text_layout.addWidget(self._audit_status_label)

        self._audit_classification_label = QLabel(
            "Build model and run audit",
        )
        self._audit_classification_label.setObjectName("LoopParamHint")
        audit_text_layout.addWidget(self._audit_classification_label)

        banner_layout.addLayout(audit_text_layout, stretch=1)

        self._audit_metrics_label = QLabel("")
        self._audit_metrics_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._audit_metrics_label.setObjectName("LoopDataSummary")
        banner_layout.addWidget(self._audit_metrics_label)

        layout.addWidget(self._audit_summary_banner)

        # ============================================================
        # Verdict Table Card
        # ============================================================
        verdict_card, verdict_card_layout = _make_card()

        verdict_card_layout.addWidget(
            _make_section_header("VERDICT TABLE"),
        )

        self._audit_verdict_table = GeologicalAuditVerdictTable()
        verdict_card_layout.addWidget(self._audit_verdict_table)

        layout.addWidget(verdict_card)

        # ============================================================
        # JORC Thresholds Card
        # ============================================================
        thresh_card, thresh_card_layout = _make_card()

        thresh_card_layout.addWidget(
            _make_section_header("JORC THRESHOLDS"),
        )

        # -- Measured --
        m_row, m_lay = _make_form_row("Measured")
        self._measured_p90_spin = QDoubleSpinBox()
        self._measured_p90_spin.setRange(0.1, 10.0)
        self._measured_p90_spin.setValue(DEFAULT_JORC_THRESHOLDS.measured_p90)
        self._measured_p90_spin.setSuffix(" m")
        self._measured_p90_spin.setDecimals(1)
        m_lay.addWidget(QLabel("P90"))
        m_lay.addWidget(self._measured_p90_spin)
        m_lay.addWidget(QLabel("Mean"))
        self._measured_mean_spin = QDoubleSpinBox()
        self._measured_mean_spin.setRange(0.1, 10.0)
        self._measured_mean_spin.setValue(
            DEFAULT_JORC_THRESHOLDS.measured_mean,
        )
        self._measured_mean_spin.setSuffix(" m")
        self._measured_mean_spin.setDecimals(1)
        m_lay.addWidget(self._measured_mean_spin)
        thresh_card_layout.addWidget(m_row)

        # -- Indicated --
        i_row, i_lay = _make_form_row("Indicated")
        self._indicated_p90_spin = QDoubleSpinBox()
        self._indicated_p90_spin.setRange(0.5, 20.0)
        self._indicated_p90_spin.setValue(
            DEFAULT_JORC_THRESHOLDS.indicated_p90,
        )
        self._indicated_p90_spin.setSuffix(" m")
        self._indicated_p90_spin.setDecimals(1)
        i_lay.addWidget(QLabel("P90"))
        i_lay.addWidget(self._indicated_p90_spin)
        i_lay.addWidget(QLabel("Mean"))
        self._indicated_mean_spin = QDoubleSpinBox()
        self._indicated_mean_spin.setRange(0.5, 10.0)
        self._indicated_mean_spin.setValue(
            DEFAULT_JORC_THRESHOLDS.indicated_mean,
        )
        self._indicated_mean_spin.setSuffix(" m")
        self._indicated_mean_spin.setDecimals(1)
        i_lay.addWidget(self._indicated_mean_spin)
        thresh_card_layout.addWidget(i_row)

        # -- Inferred --
        inf_row, inf_lay = _make_form_row("Inferred")
        self._inferred_p90_spin = QDoubleSpinBox()
        self._inferred_p90_spin.setRange(1.0, 50.0)
        self._inferred_p90_spin.setValue(
            DEFAULT_JORC_THRESHOLDS.inferred_p90,
        )
        self._inferred_p90_spin.setSuffix(" m")
        self._inferred_p90_spin.setDecimals(1)
        inf_lay.addWidget(QLabel("P90"))
        inf_lay.addWidget(self._inferred_p90_spin)
        inf_lay.addWidget(QLabel("Mean"))
        self._inferred_mean_spin = QDoubleSpinBox()
        self._inferred_mean_spin.setRange(0.5, 20.0)
        self._inferred_mean_spin.setValue(
            DEFAULT_JORC_THRESHOLDS.inferred_mean,
        )
        self._inferred_mean_spin.setSuffix(" m")
        self._inferred_mean_spin.setDecimals(1)
        inf_lay.addWidget(self._inferred_mean_spin)
        thresh_card_layout.addWidget(inf_row)

        thresh_card_layout.addWidget(_make_divider())

        # Reset defaults row
        reset_bar, reset_lay = _make_action_bar()
        reset_lay.addStretch()
        reset_btn = _make_action_button("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_jorc_thresholds)
        reset_lay.addWidget(reset_btn)
        thresh_card_layout.addWidget(reset_bar)

        layout.addWidget(thresh_card)

        # ============================================================
        # Detailed Compliance (collapsible)
        # ============================================================
        from ..loopstructural_compliance_panel import (
            ComplianceValidationPanel,
        )

        details_group = CollapsibleGroup(
            "Detailed Compliance View", collapsed=True,
        )
        self._compliance_panel = ComplianceValidationPanel()
        details_group.add_widget(self._compliance_panel)
        layout.addWidget(details_group)

        layout.addStretch()

        return _make_scrollable(content)

    # ------------------------------------------------------------------
    # 6. Advisory page  (unchanged)
    # ------------------------------------------------------------------
    def _create_advisory_tab(self) -> QWidget:
        """Advisory page -- structural advisory widget."""
        from ..loopstructural_advisory_panel import StructuralAdvisoryWidget

        self._advisory_panel = StructuralAdvisoryWidget(
            apply_callback=self._on_apply_suggested_fault,
        )
        return self._advisory_panel

    # ------------------------------------------------------------------
    # 7. Export page  (two-column split)
    # ------------------------------------------------------------------
    def _create_export_tab(self) -> QWidget:
        """Export page -- two-column layout: surfaces on left,
        audit data on right."""
        tab = QWidget()
        root_layout = QVBoxLayout(tab)
        _tab_margins(root_layout)

        # Two-column splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # ==============================================================
        # LEFT COLUMN -- Surfaces
        # ==============================================================
        surf_card, surf_card_layout = _make_card()

        surf_card_layout.addWidget(_make_section_header("SURFACES"))

        self._surface_list = QListWidget()
        self._surface_list.setObjectName("LoopFlatList")
        self._surface_list.setMinimumHeight(140)
        surf_card_layout.addWidget(self._surface_list, stretch=1)

        surf_card_layout.addWidget(_make_divider())

        # Export format buttons row
        export_row = QHBoxLayout()
        export_row.setSpacing(tokens.SPACING_SM)

        self._export_obj_btn = _make_primary_button("OBJ")
        self._export_obj_btn.clicked.connect(
            lambda: self._on_export_surfaces("obj"),
        )
        self._export_obj_btn.setEnabled(False)
        self._export_obj_btn.setToolTip("Export as Wavefront OBJ")
        export_row.addWidget(self._export_obj_btn)

        self._export_stl_btn = _make_primary_button("STL")
        self._export_stl_btn.clicked.connect(
            lambda: self._on_export_surfaces("stl"),
        )
        self._export_stl_btn.setEnabled(False)
        self._export_stl_btn.setToolTip("Export as STL mesh")
        export_row.addWidget(self._export_stl_btn)

        self._export_vtk_btn = _make_primary_button("VTK")
        self._export_vtk_btn.clicked.connect(
            lambda: self._on_export_surfaces("vtk"),
        )
        self._export_vtk_btn.setEnabled(False)
        self._export_vtk_btn.setToolTip("Export as VTK")
        export_row.addWidget(self._export_vtk_btn)

        export_row.addStretch()
        surf_card_layout.addLayout(export_row)

        splitter.addWidget(surf_card)

        # ==============================================================
        # RIGHT COLUMN -- Audit Data
        # ==============================================================
        audit_card, audit_card_layout = _make_card()

        audit_card_layout.addWidget(_make_section_header("AUDIT DATA"))

        self._export_audit_btn = _make_primary_button("Build Log (JSON)")
        self._export_audit_btn.clicked.connect(self._on_export_audit)
        self._export_audit_btn.setEnabled(False)
        self._export_audit_btn.setToolTip("Export build log as JSON")
        audit_card_layout.addWidget(self._export_audit_btn)

        self._export_compliance_btn = _make_primary_button(
            "Compliance Report",
        )
        self._export_compliance_btn.clicked.connect(
            self._on_export_compliance,
        )
        self._export_compliance_btn.setEnabled(False)
        self._export_compliance_btn.setToolTip("Export compliance report")
        audit_card_layout.addWidget(self._export_compliance_btn)

        audit_card_layout.addStretch()

        splitter.addWidget(audit_card)

        # Even 50/50 stretch
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter, stretch=1)

        return tab
