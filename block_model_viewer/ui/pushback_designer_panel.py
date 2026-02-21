"""
Pushback Visual Designer Panel (STEP 33)

UI panel for designing pushbacks and integrating with NPVS optimization.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging

from PyQt6.QtWidgets import (
QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QComboBox, QSpinBox, QPushButton, QDoubleSpinBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QMessageBox, QListWidget, QListWidgetItem
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QColor

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


def _has_model_data(model: Any) -> bool:
    """Safe model presence check that handles pandas DataFrames."""
    if model is None:
        return False
    try:
        return not bool(getattr(model, "empty"))
    except Exception:
        return True


class PushbackDesignerPanel(BaseAnalysisPanel):
    """
    Pushback Visual Designer Panel.
    
    - View LG/nested shells and phases
    - Group shells into pushbacks
    - Visualize phase/pushback sequence in 3D
    - Run NPVS using these pushbacks and compare NPVs
    """
    # PanelManager metadata
    PANEL_ID = "PushbackDesignerPanel"
    PANEL_NAME = "PushbackDesigner Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "pushback_build_plan"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="pushback_designer")
        self.current_plan = None
        self.shells = []
        self._block_model = None  # Use _block_model instead of block_model property
        self.pit_results = None
        
        # Subscribe to block model and pit optimization from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_loaded)
            self.registry.pitOptimizationResultsLoaded.connect(self._on_pit_optimization_results_loaded)
            
            # Prefer classified block model when available.
            existing_block_model = self.registry.get_classified_block_model()
            if not _has_model_data(existing_block_model):
                existing_block_model = self.registry.get_block_model()
            if _has_model_data(existing_block_model):
                self._on_block_model_loaded(existing_block_model)
            
            existing_pit = self.registry.get_pit_optimization_results()
            if existing_pit:
                self._on_pit_optimization_results_loaded(existing_pit)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self._build_ui()
    


    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, "setStyleSheet"):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, "refresh_theme"):
                child.refresh_theme()
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("Pushback Designer Panel received block model from DataRegistry")
        self._block_model = block_model  # Use _block_model instead of block_model property
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_pit_optimization_results_loaded(self, pit_results):
        """
        Automatically receive pit optimization results when they're loaded.
        
        Args:
            pit_results: Pit optimization results (shells/phases) from DataRegistry
        """
        logger.info("Pushback Designer Panel received pit optimization results from DataRegistry")
        self.pit_results = pit_results
        # Extract shells if available
        if isinstance(pit_results, dict):
            shells = pit_results.get('shells') or pit_results.get('phases')
            if shells:
                self.shells = shells
                logger.info(f"Pushback Designer Panel loaded {len(shells)} shells from pit results")
    
    def _build_ui(self):
        """Build the UI."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Pushback Visual Designer: Group shells/phases into pushbacks, "
            "visualize in 3D, and run NPVS optimization with pushback constraints."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Main split: Shells (left) and Pushbacks (right)
        main_split = QHBoxLayout()
        
        # Left: Shell/phase list
        left_panel = self._create_shell_list_panel()
        main_split.addWidget(left_panel, 1)
        
        # Right: Pushback editor
        right_panel = self._create_pushback_editor_panel()
        main_split.addWidget(right_panel, 2)
        
        layout.addLayout(main_split)
    
    def _create_shell_list_panel(self) -> QWidget:
        """Create shell/phase list panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("<b>Available Shells/Phases</b>"))
        
        # Shell table
        self.shell_table = QTableWidget()
        self.shell_table.setColumnCount(4)
        self.shell_table.setHorizontalHeaderLabels([
            "ID", "Tonnes", "Value", "Precedence"
        ])
        self.shell_table.horizontalHeader().setStretchLastSection(True)
        self.shell_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.shell_table)
        
        # Load shells button
        load_btn = QPushButton("Load Shells from Pit Optimization")
        load_btn.clicked.connect(self._on_load_shells)
        layout.addWidget(load_btn)
        
        return widget
    
    def _create_pushback_editor_panel(self) -> QWidget:
        """Create pushback editor panel."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Pushback definition group
        def_group = QGroupBox("Pushback Definition")
        def_layout = QFormLayout(def_group)
        
        self.grouping_mode = QComboBox()
        self.grouping_mode.addItems(["by depth", "by value", "manual"])
        def_layout.addRow("Grouping Mode:", self.grouping_mode)
        
        self.num_pushbacks = QSpinBox()
        self.num_pushbacks.setMinimum(1)
        self.num_pushbacks.setMaximum(50)
        self.num_pushbacks.setValue(5)
        def_layout.addRow("Number of Pushbacks:", self.num_pushbacks)
        
        auto_group_btn = QPushButton("Auto-Group Shells")
        auto_group_btn.clicked.connect(self._on_auto_group)
        def_layout.addRow(auto_group_btn)
        
        layout.addWidget(def_group)
        
        # Pushback list
        layout.addWidget(QLabel("<b>Pushbacks</b>"))
        
        self.pushback_list = QListWidget()
        self.pushback_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.pushback_list.itemChanged.connect(self._on_pushback_order_changed)
        layout.addWidget(self.pushback_list)
        
        # Pushback buttons
        btn_layout = QHBoxLayout()
        
        up_btn = QPushButton("↑")
        up_btn.clicked.connect(self._on_move_pushback_up)
        btn_layout.addWidget(up_btn)
        
        down_btn = QPushButton("↓")
        down_btn.clicked.connect(self._on_move_pushback_down)
        btn_layout.addWidget(down_btn)
        
        btn_layout.addStretch()
        
        send_npvs_btn = QPushButton("Send to NPVS")
        send_npvs_btn.clicked.connect(self._on_send_to_npvs)
        btn_layout.addWidget(send_npvs_btn)
        
        highlight_btn = QPushButton("Highlight in View")
        highlight_btn.clicked.connect(self._on_highlight_pushback)
        btn_layout.addWidget(highlight_btn)
        
        layout.addLayout(btn_layout)
        
        return widget
    
    def _on_load_shells(self):
        """Load shells from pit optimization."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        # Try to get shells from pit optimization results
        # For now, create sample shells
        self.show_info("Load Shells", "Loading shells from pit optimization...")
        
        # Sample shells (would come from actual pit optimization)
        from ..mine_planning.pushbacks.pushback_model import ShellPhase
        
        sample_shells = [
            ShellPhase(id="S_30", tonnes=1_000_000, value=50_000_000, precedence_ids=[]),
            ShellPhase(id="S_35", tonnes=1_200_000, value=60_000_000, precedence_ids=["S_30"]),
            ShellPhase(id="S_40", tonnes=1_500_000, value=75_000_000, precedence_ids=["S_35"]),
            ShellPhase(id="S_45", tonnes=1_800_000, value=90_000_000, precedence_ids=["S_40"]),
            ShellPhase(id="S_50", tonnes=2_000_000, value=100_000_000, precedence_ids=["S_45"]),
        ]
        
        self.shells = sample_shells
        self._refresh_shell_table()
    
    def _refresh_shell_table(self):
        """Refresh shell table."""
        self.shell_table.setRowCount(len(self.shells))
        
        for row, shell in enumerate(self.shells):
            self.shell_table.setItem(row, 0, QTableWidgetItem(shell.id))
            self.shell_table.setItem(row, 1, QTableWidgetItem(f"{shell.tonnes:,.0f}"))
            self.shell_table.setItem(row, 2, QTableWidgetItem(f"${shell.value:,.0f}"))
            self.shell_table.setItem(row, 3, QTableWidgetItem(str(len(shell.precedence_ids))))
        
        self.shell_table.resizeColumnsToContents()
    
    def _on_auto_group(self):
        """Auto-group shells into pushbacks."""
        if not self.shells:
            self.show_warning("No Shells", "Please load shells first.")
            return
        
        mode = self.grouping_mode.currentText()
        num_pushbacks = self.num_pushbacks.value()
        
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        try:
            from ..mine_planning.pushbacks.pushback_builder import (
                auto_group_shells_by_depth,
                auto_group_shells_by_value
            )
            
            if mode == "by depth":
                plan = auto_group_shells_by_depth(self.shells, num_pushbacks)
            elif mode == "by value":
                plan = auto_group_shells_by_value(self.shells, num_pushbacks)
            else:
                self.show_warning("Manual Mode", "Manual grouping not yet implemented.")
                return
            
            self.current_plan = plan
            self._refresh_pushback_list()
            
            self.show_info("Grouping Complete", f"Created {len(plan.pushbacks)} pushbacks.")
        
        except Exception as e:
            logger.error(f"Failed to auto-group shells: {e}", exc_info=True)
            self.show_error("Grouping Failed", f"Failed to group shells:\n{e}")
    
    def _refresh_pushback_list(self):
        """Refresh pushback list."""
        self.pushback_list.clear()
        
        if not self.current_plan:
            return
        
        for pushback in self.current_plan.pushbacks:
            item = QListWidgetItem(
                f"{pushback.name} ({len(pushback.shell_ids)} shells, "
                f"{pushback.tonnes:,.0f} t, ${pushback.value:,.0f})"
            )
            item.setData(Qt.ItemDataRole.UserRole, pushback.id)
            self.pushback_list.addItem(item)
    
    def _on_move_pushback_up(self):
        """Move selected pushback up."""
        current_row = self.pushback_list.currentRow()
        if current_row > 0:
            item = self.pushback_list.takeItem(current_row)
            self.pushback_list.insertItem(current_row - 1, item)
            self.pushback_list.setCurrentRow(current_row - 1)
            self._on_pushback_order_changed()
    
    def _on_move_pushback_down(self):
        """Move selected pushback down."""
        current_row = self.pushback_list.currentRow()
        if current_row < self.pushback_list.count() - 1:
            item = self.pushback_list.takeItem(current_row)
            self.pushback_list.insertItem(current_row + 1, item)
            self.pushback_list.setCurrentRow(current_row + 1)
            self._on_pushback_order_changed()
    
    def _on_pushback_order_changed(self):
        """Handle pushback order change."""
        if not self.current_plan:
            return
        
        # Get new order from list
        new_order = []
        for i in range(self.pushback_list.count()):
            item = self.pushback_list.item(i)
            if item:
                pushback_id = item.data(Qt.ItemDataRole.UserRole)
                if pushback_id:
                    new_order.append(pushback_id)
        
        if new_order:
            try:
                from ..mine_planning.pushbacks.pushback_builder import reorder_pushbacks
                self.current_plan = reorder_pushbacks(self.current_plan, new_order)
                logger.info("Reordered pushbacks")
            except Exception as e:
                logger.error(f"Failed to reorder pushbacks: {e}", exc_info=True)
    
    def _on_send_to_npvs(self):
        """Send pushback plan to NPVS optimization."""
        if not self.current_plan:
            self.show_warning("No Pushback Plan", "Please create a pushback plan first.")
            return
        
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        # Open NPVS panel with pushback configuration
        self.show_info("Send to NPVS", "Opening NPVS panel with pushback configuration...")
        
        # Would integrate with NPVS panel to pass pushback phases
        # For now, just show message
        self.show_info("NPVS Integration", 
                      f"Pushback plan with {len(self.current_plan.pushbacks)} pushbacks ready for NPVS.")
    
    def _on_highlight_pushback(self):
        """Highlight pushbacks in 3D view."""
        if not self.current_plan:
            self.show_warning("No Pushback Plan", "Please create a pushback plan first.")
            return
        
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        try:
            # Call controller to render pushbacks
            self.controller.render_pushback_layer(self.current_plan, {})
            self.show_info("Visualization", "Pushbacks highlighted in 3D view.")
        except Exception as e:
            logger.error(f"Failed to highlight pushbacks: {e}", exc_info=True)
            self.show_error("Visualization Failed", f"Failed to highlight pushbacks:\n{e}")
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        # Publish pushback plan to DataRegistry
        if hasattr(self, 'registry') and self.registry:
            try:
                pushback_plan = {
                    'current_plan': self.current_plan,
                    'shells': self.shells,
                    'pushback_groups': payload.get('pushback_groups', []),
                    'source': 'pushback_designer'
                }
                # Store as pit optimization results extension or separate storage
                # For now, update pit optimization results if available
                existing_pit = self.registry.get_pit_optimization_results()
                if existing_pit and isinstance(existing_pit, dict):
                    updated_pit = existing_pit.copy()
                    updated_pit['pushback_plan'] = pushback_plan
                    self.registry.register_pit_optimization_results(updated_pit, source_panel="PushbackDesignerPanel")
                    logger.info("Pushback Designer Panel published pushback plan to DataRegistry")
            except Exception as e:
                logger.warning(f"Failed to register pushback plan: {e}")

