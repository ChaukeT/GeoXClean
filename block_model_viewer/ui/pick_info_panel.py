"""
Pick Info Panel - Displays information about clicked objects in the 3D scene.

This panel shows:
- Layer name
- Cell/Point ID
- All property values at the picked location

Author: Block Model Viewer Team
"""

from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTreeWidget, 
                              QTreeWidgetItem, QFrame, QPushButton)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PickInfoPanel(QWidget):
    """
    A Qt-based panel for displaying information about picked objects.
    
    This panel updates in real-time as the user clicks on different objects
    in the 3D scene.
    """
    
    # Signals
    clear_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.current_info = {}
        
        self._setup_ui()
        
        logger.info("Initialized PickInfoPanel")
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Title
        title_label = QLabel("Pick Information")
        title_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)
        
        # Info tree
        self.info_tree = QTreeWidget()
        self.info_tree.setHeaderLabels(["Property", "Value"])
        self.info_tree.setAlternatingRowColors(True)
        self.info_tree.setRootIsDecorated(False)
        layout.addWidget(self.info_tree)
        
        # Clear button
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear)
        layout.addWidget(self.clear_btn)
        
        # Initial state
        self.clear()
    
    def update_info(self, info_dict: Dict[str, Any]):
        """
        Update the displayed information.
        
        Args:
            info_dict: Dictionary of property names to values
        """
        self.current_info = info_dict
        self.info_tree.clear()
        
        if not info_dict:
            item = QTreeWidgetItem(["No data", "--"])
            self.info_tree.addTopLevelItem(item)
            return
        
        # Add each property-value pair
        for key, value in info_dict.items():
            # Format value based on type
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            elif isinstance(value, int):
                value_str = str(value)
            elif isinstance(value, (list, tuple)):
                value_str = f"[{', '.join(f'{v:.3f}' if isinstance(v, float) else str(v) for v in value)}]"
            else:
                value_str = str(value)
            
            item = QTreeWidgetItem([str(key), value_str])
            self.info_tree.addTopLevelItem(item)
        
        # Resize columns to content
        self.info_tree.resizeColumnToContents(0)
        self.info_tree.resizeColumnToContents(1)
        
        logger.debug(f"Pick info updated with {len(info_dict)} properties")
    
    def clear(self):
        """Clear the pick information display."""
        self.current_info = {}
        self.info_tree.clear()
        
        item = QTreeWidgetItem(["Click on an object", "to inspect it"])
        item.setForeground(0, Qt.GlobalColor.gray)
        item.setForeground(1, Qt.GlobalColor.gray)
        self.info_tree.addTopLevelItem(item)
        
        self.clear_requested.emit()


















