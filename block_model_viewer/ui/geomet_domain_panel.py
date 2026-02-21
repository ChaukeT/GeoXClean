"""
Geomet Domain Panel (STEP 28)

Define and manage GeometOreType and GeometDomainMap.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QGroupBox, QFormLayout,
    QLineEdit, QComboBox, QDoubleSpinBox, QTextEdit, QFileDialog,
    QMessageBox, QHeaderView
)
from PyQt6.QtCore import Qt

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class GeometDomainPanel(BaseAnalysisPanel):
    """
    Panel for defining and managing geometallurgical ore types and domain mapping.
    """
    
    task_name = "geomet_assign_ore_types"
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent=parent, panel_id="geomet_domain")
        self.ore_types: Dict[str, Any] = {}
        self._block_model = None  # Use _block_model instead of block_model property
        
        # Subscribe to block model updates from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            self.registry.blockModelClassified.connect(self._on_block_model_classified)
            
            # Load existing block model if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
            
            existing_classified = self.registry.get_classified_block_model()
            if existing_classified:
                self._on_block_model_classified(existing_classified)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        logger.info("Initialized Geomet Domain Panel")
    


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
    def setup_ui(self):
        """Setup the user interface."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Define geometallurgical ore types and map them to geological domains.\n"
            "Ore types link geology, texture, hardness, and plant response."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Ore type definition section
        ore_type_group = self._create_ore_type_group()
        layout.addWidget(ore_type_group)
        
        # Domain mapping table
        mapping_group = self._create_mapping_table()
        layout.addWidget(mapping_group)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        load_btn = QPushButton("📂 Load from CSV")
        load_btn.clicked.connect(self._load_from_csv)
        button_layout.addWidget(load_btn)
        
        export_btn = QPushButton("💾 Export to CSV")
        export_btn.clicked.connect(self._export_to_csv)
        button_layout.addWidget(export_btn)
        
        button_layout.addStretch()
        
        assign_btn = QPushButton("▶️ Assign Ore Types to Blocks")
        assign_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        assign_btn.clicked.connect(self._assign_ore_types)
        button_layout.addWidget(assign_btn)
        
        layout.addLayout(button_layout)
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        try:
            # Convert to BlockModel if needed
            if hasattr(block_model, 'to_dataframe'):
                self._block_model = block_model  # Use _block_model
            elif isinstance(block_model, pd.DataFrame):
                # Store DataFrame - will be used when assigning ore types
                logger.info(f"Geomet Domain Panel received block model DataFrame: {len(block_model)} blocks")
            else:
                logger.warning(f"Unexpected block model type: {type(block_model)}")
                return
            
            logger.info(f"Geomet Domain Panel auto-received block model")
        except Exception as e:
            logger.error(f"Error processing block model in Geomet Domain Panel: {e}", exc_info=True)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def _on_block_model_classified(self, block_model):
        """
        Automatically receive classified block model when it's classified.
        
        Args:
            block_model: Classified BlockModel from DataRegistry
        """
        # Use same handler as generated
        self._on_block_model_generated(block_model)
    
    def _create_ore_type_group(self) -> QGroupBox:
        """Create ore type definition group."""
        group = QGroupBox("Ore Type Definition")
        layout = QVBoxLayout()
        
        form = QFormLayout()
        
        self.ore_type_code = QLineEdit()
        self.ore_type_code.setPlaceholderText("e.g., OT1, MASSIVE_FE")
        form.addRow("Ore Type Code:", self.ore_type_code)
        
        self.ore_type_name = QLineEdit()
        self.ore_type_name.setPlaceholderText("e.g., Massive Magnetite")
        form.addRow("Name:", self.ore_type_name)
        
        self.texture_combo = QComboBox()
        self.texture_combo.addItems(["", "massive", "banded", "friable", "disseminated", "brecciated"])
        form.addRow("Texture Class:", self.texture_combo)
        
        self.hardness_combo = QComboBox()
        self.hardness_combo.addItems(["", "soft", "medium", "hard"])
        form.addRow("Hardness Class:", self.hardness_combo)
        
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.1, 10.0)
        self.density_spin.setDecimals(2)
        self.density_spin.setSuffix(" t/m³")
        self.density_spin.setSpecialValueText("Auto")
        form.addRow("Density:", self.density_spin)
        
        self.notes_text = QTextEdit()
        self.notes_text.setMaximumHeight(60)
        self.notes_text.setPlaceholderText("Additional notes...")
        form.addRow("Notes:", self.notes_text)
        
        layout.addLayout(form)
        
        # Add/Update button
        btn_layout = QHBoxLayout()
        self.add_ore_type_btn = QPushButton("➕ Add/Update Ore Type")
        self.add_ore_type_btn.clicked.connect(self._add_ore_type)
        btn_layout.addWidget(self.add_ore_type_btn)
        
        self.remove_ore_type_btn = QPushButton("➖ Remove Selected")
        self.remove_ore_type_btn.clicked.connect(self._remove_ore_type)
        btn_layout.addWidget(self.remove_ore_type_btn)
        
        layout.addLayout(btn_layout)
        
        group.setLayout(layout)
        return group
    
    def _create_mapping_table(self) -> QGroupBox:
        """Create domain mapping table."""
        group = QGroupBox("Ore Types & Domain Mapping")
        layout = QVBoxLayout()
        
        self.mapping_table = QTableWidget()
        self.mapping_table.setColumnCount(6)
        self.mapping_table.setHorizontalHeaderLabels([
            "Ore Type Code", "Name", "Geology Domains", "Texture", "Hardness", "Density"
        ])
        self.mapping_table.horizontalHeader().setStretchLastSection(True)
        self.mapping_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.mapping_table)
        
        group.setLayout(layout)
        return group
    
    def _add_ore_type(self):
        """Add or update ore type."""
        code = self.ore_type_code.text().strip()
        name = self.ore_type_name.text().strip()
        
        if not code:
            self.show_error("Invalid Input", "Ore type code is required.")
            return
        
        # Import here to avoid circular imports
        from ..geomet.domains_links import GeometOreType
        
        ore_type = GeometOreType(
            code=code,
            name=name or code,
            geology_domains=[],  # Will be populated from domain model
            texture_class=self.texture_combo.currentText() or None,
            hardness_class=self.hardness_combo.currentText() or None,
            density=self.density_spin.value() if self.density_spin.value() > 0 else None,
            notes=self.notes_text.toPlainText()
        )
        
        self.ore_types[code] = ore_type
        self._update_mapping_table()
        
        # Clear form
        self.ore_type_code.clear()
        self.ore_type_name.clear()
        self.texture_combo.setCurrentIndex(0)
        self.hardness_combo.setCurrentIndex(0)
        self.density_spin.setValue(0.0)
        self.notes_text.clear()
        
        self.emit_status(f"Added ore type: {code}")
    
    def _remove_ore_type(self):
        """Remove selected ore type."""
        row = self.mapping_table.currentRow()
        if row < 0:
            self.show_warning("No Selection", "Please select an ore type to remove.")
            return
        
        code_item = self.mapping_table.item(row, 0)
        if code_item:
            code = code_item.text()
            if code in self.ore_types:
                del self.ore_types[code]
                self._update_mapping_table()
                self.emit_status(f"Removed ore type: {code}")
    
    def _update_mapping_table(self):
        """Update the mapping table."""
        self.mapping_table.setRowCount(len(self.ore_types))
        
        for row, (code, ore_type) in enumerate(self.ore_types.items()):
            self.mapping_table.setItem(row, 0, QTableWidgetItem(code))
            self.mapping_table.setItem(row, 1, QTableWidgetItem(ore_type.name))
            self.mapping_table.setItem(row, 2, QTableWidgetItem(", ".join(ore_type.geology_domains)))
            self.mapping_table.setItem(row, 3, QTableWidgetItem(ore_type.texture_class or ""))
            self.mapping_table.setItem(row, 4, QTableWidgetItem(ore_type.hardness_class or ""))
            density_str = f"{ore_type.density:.2f}" if ore_type.density else ""
            self.mapping_table.setItem(row, 5, QTableWidgetItem(density_str))
    
    def _load_from_csv(self):
        """Load ore types from CSV file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Ore Types from CSV", "", "CSV Files (*.csv)"
        )
        if not filename:
            return
        
        try:
            df = pd.read_csv(filename)
            
            # Import here to avoid circular imports
            from ..geomet.domains_links import GeometOreType
            
            for _, row in df.iterrows():
                code = str(row.get("code", "")).strip()
                if not code:
                    continue
                
                ore_type = GeometOreType(
                    code=code,
                    name=str(row.get("name", code)),
                    geology_domains=str(row.get("geology_domains", "")).split(",") if pd.notna(row.get("geology_domains")) else [],
                    texture_class=str(row.get("texture_class", "")) if pd.notna(row.get("texture_class")) else None,
                    hardness_class=str(row.get("hardness_class", "")) if pd.notna(row.get("hardness_class")) else None,
                    density=float(row.get("density", 0)) if pd.notna(row.get("density")) else None,
                    notes=str(row.get("notes", ""))
                )
                self.ore_types[code] = ore_type
            
            self._update_mapping_table()
            self.show_info("Success", f"Loaded {len(self.ore_types)} ore types from CSV.")
        except Exception as e:
            self.show_error("Load Error", f"Failed to load CSV: {str(e)}")
    
    def _export_to_csv(self):
        """Export ore types to CSV file."""
        if not self.ore_types:
            self.show_warning("No Data", "No ore types to export.")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Ore Types to CSV", "", "CSV Files (*.csv)"
        )
        if not filename:
            return
        
        try:
            data = []
            for code, ore_type in self.ore_types.items():
                data.append({
                    "code": code,
                    "name": ore_type.name,
                    "geology_domains": ",".join(ore_type.geology_domains),
                    "texture_class": ore_type.texture_class or "",
                    "hardness_class": ore_type.hardness_class or "",
                    "density": ore_type.density or "",
                    "notes": ore_type.notes
                })
            
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            self.show_info("Success", f"Exported {len(self.ore_types)} ore types to CSV.")
        except Exception as e:
            self.show_error("Export Error", f"Failed to export CSV: {str(e)}")
    
    def _assign_ore_types(self):
        """Assign ore types to blocks."""
        if not self.ore_types:
            self.show_error("No Ore Types", "Please define at least one ore type.")
            return
        
        if not self.validate_block_model_loaded():
            return
        
        # Build domain map
        from ..geomet.domains_links import GeometDomainMap
        
        domain_map = GeometDomainMap(ore_types=self.ore_types.copy())
        
        # Gather parameters
        params = {
            "block_model": self.block_model,
            "geomet_domain_map": domain_map,
            "rules": {"domain_property": "domain"}
        }
        
        # Run via controller
        if self.controller:
            self.controller.assign_geomet_ore_types(params, self._on_assign_complete)
        else:
            self.show_error("No Controller", "Controller not available.")
    
    def _on_assign_complete(self, result: Dict[str, Any]):
        """Handle ore type assignment completion."""
        if result.get("error"):
            self.show_error("Assignment Error", result["error"])
        else:
            self.show_info("Success", "Ore types assigned to blocks successfully.")
            self.emit_status("Ore types assigned. Check block model properties.")
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Gather parameters for ore type assignment."""
        from ..geomet.domains_links import GeometDomainMap
        
        domain_map = GeometDomainMap(ore_types=self.ore_types.copy())
        
        return {
            "geomet_domain_map": domain_map,
            "rules": {"domain_property": "domain"}
        }
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        if not self.ore_types:
            self.show_error("No Ore Types", "Please define at least one ore type.")
            return False
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        if payload.get("error"):
            self.show_error("Error", payload["error"])
        else:
            self.show_info("Success", "Ore types assigned successfully.")

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            settings = {}
            
            # Save ore types
            if self.ore_types:
                settings['ore_types'] = {}
                for code, ore_type in self.ore_types.items():
                    settings['ore_types'][code] = {
                        'name': ore_type.name,
                        'geology_domains': list(ore_type.geology_domains),
                        'texture_class': ore_type.texture_class,
                        'hardness_class': ore_type.hardness_class,
                        'density': ore_type.density,
                        'notes': ore_type.notes
                    }
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save geomet domain panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            # Restore ore types
            if 'ore_types' in settings:
                from ..geomet.domains_links import OreType
                
                self.ore_types = {}
                for code, ore_data in settings['ore_types'].items():
                    self.ore_types[code] = OreType(
                        code=code,
                        name=ore_data.get('name', ''),
                        geology_domains=set(ore_data.get('geology_domains', [])),
                        texture_class=ore_data.get('texture_class'),
                        hardness_class=ore_data.get('hardness_class'),
                        density=ore_data.get('density'),
                        notes=ore_data.get('notes', '')
                    )
                
                # Update mapping table
                if hasattr(self, '_update_mapping_table'):
                    self._update_mapping_table()
                
            logger.info("Restored geomet domain panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore geomet domain panel settings: {e}")
