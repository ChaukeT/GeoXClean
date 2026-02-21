"""
Geomet Chain Panel (STEP 38)

Full geomet chain from ore types to plant response to NPV.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List
import logging
import json
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QTabWidget, QDoubleSpinBox, QSpinBox, QLineEdit,
    QTextEdit, QMessageBox, QFileDialog
)
from PyQt6.QtCore import pyqtSlot

from .base_analysis_panel import BaseAnalysisPanel

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class GeometChainPanel(BaseAnalysisPanel):
    """
    Geomet Chain Panel for ore types, plant routes, routing, and value computation.
    """
    PANEL_ID = "GeometChainPanel"  # STEP 40
    task_name = "geomet_compute_values"
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="geomet_chain")
        self.ore_type_model = None
        self.plant_routes = []
        self.route_selector_config = None
        self.geomet_value_config = None
        # block_model is a read-only property from BasePanel - use _block_model instead
        self._block_model = None
        
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
        
        self.setup_ui()
    


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
        """Build the UI (overrides BaseAnalysisPanel.setup_ui)."""
        layout = self.main_layout
        
        # Info label
        info = QLabel(
            "Geometallurgical Chain: Connect ore types, plant response surfaces, "
            "route selection, and compute geomet-adjusted block values for NPVS."
        )
        info.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._build_ore_types_tab(), "Ore Types")
        tabs.addTab(self._build_plant_routes_tab(), "Plant Routes")
        tabs.addTab(self._build_routing_tab(), "Routing & Value")
        tabs.addTab(self._build_preview_tab(), "Preview")
        layout.addWidget(tabs)
    
    def _build_ore_types_tab(self) -> QWidget:
        """Build Ore Types tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Ore domains table
        self.ore_domains_table = QTableWidget()
        self.ore_domains_table.setColumnCount(5)
        self.ore_domains_table.setHorizontalHeaderLabels([
            "Domain ID", "Description", "Hardness Index", "Density (t/m³)", "Alteration Code"
        ])
        layout.addWidget(QLabel("Ore Domains:"))
        layout.addWidget(self.ore_domains_table)
        
        # Add/Edit buttons
        btn_layout = QHBoxLayout()
        add_domain_btn = QPushButton("Add Domain")
        add_domain_btn.clicked.connect(self._on_add_domain)
        btn_layout.addWidget(add_domain_btn)
        
        apply_ore_types_btn = QPushButton("Apply Ore Types to Block Model")
        apply_ore_types_btn.clicked.connect(self._on_apply_ore_types)
        btn_layout.addWidget(apply_ore_types_btn)
        
        layout.addLayout(btn_layout)
        
        layout.addStretch()
        return widget
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        try:
            # Convert to BlockModel if needed
            if hasattr(block_model, 'to_dataframe'):
                self._block_model = block_model
            elif isinstance(block_model, pd.DataFrame):
                # Store DataFrame
                logger.info(f"Geomet Chain Panel received block model DataFrame: {len(block_model)} blocks")
            else:
                logger.warning(f"Unexpected block model type: {type(block_model)}")
                return
            
            logger.info(f"Geomet Chain Panel auto-received block model")
        except Exception as e:
            logger.error(f"Error processing block model in Geomet Chain Panel: {e}", exc_info=True)
    
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
    
    def _build_plant_routes_tab(self) -> QWidget:
        """Build Plant Routes tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Routes table
        self.routes_table = QTableWidget()
        self.routes_table.setColumnCount(5)
        self.routes_table.setHorizontalHeaderLabels([
            "Route ID", "Name", "Base Cost ($/t)", "Energy Cost ($/kWh)", "Recovery Surfaces"
        ])
        layout.addWidget(QLabel("Plant Routes:"))
        layout.addWidget(self.routes_table)
        
        # Add route button
        add_route_btn = QPushButton("Add Plant Route")
        add_route_btn.clicked.connect(self._on_add_route)
        layout.addWidget(add_route_btn)
        
        layout.addStretch()
        return widget
    
    def _build_routing_tab(self) -> QWidget:
        """Build Routing & Value tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Route selector configuration
        selector_group = QGroupBox("Route Selector Configuration")
        selector_form = QFormLayout(selector_group)
        
        self.selector_rule = QComboBox()
        self.selector_rule.addItems(["highest_value", "fixed_route", "blend_target"])
        selector_form.addRow("Selection Rule:", self.selector_rule)
        
        self.energy_cost = QDoubleSpinBox()
        self.energy_cost.setRange(0.0, 1.0)
        self.energy_cost.setValue(0.10)
        self.energy_cost.setDecimals(3)
        self.energy_cost.setPrefix("$")
        selector_form.addRow("Energy Cost per kWh:", self.energy_cost)
        
        self.plant_nominal_tph = QDoubleSpinBox()
        self.plant_nominal_tph.setRange(100.0, 10_000.0)
        self.plant_nominal_tph.setValue(1000.0)
        self.plant_nominal_tph.setSuffix(" tph")
        selector_form.addRow("Plant Nominal Throughput:", self.plant_nominal_tph)
        
        layout.addWidget(selector_group)
        
        # Value computation configuration
        value_group = QGroupBox("Value Computation")
        value_form = QFormLayout(value_group)
        
        self.mining_cost = QDoubleSpinBox()
        self.mining_cost.setRange(0.0, 100.0)
        self.mining_cost.setValue(5.0)
        self.mining_cost.setPrefix("$")
        self.mining_cost.setSuffix("/t")
        value_form.addRow("Mining Cost per Tonne:", self.mining_cost)
        
        # Prices (simplified - would have per-element inputs)
        self.price_fe = QDoubleSpinBox()
        self.price_fe.setRange(0.0, 1000.0)
        self.price_fe.setValue(100.0)
        self.price_fe.setPrefix("$")
        value_form.addRow("Fe Price:", self.price_fe)
        
        compute_btn = QPushButton("Compute Geomet Block Values")
        compute_btn.clicked.connect(self._on_compute_values)
        value_form.addRow(compute_btn)
        
        layout.addWidget(value_group)
        
        layout.addStretch()
        return widget
    
    def _build_preview_tab(self) -> QWidget:
        """Build Preview tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Summary table
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(6)
        self.summary_table.setHorizontalHeaderLabels([
            "Ore Domain", "Tonnes", "Head Grade", "Chosen Route", "Value/t", "Total Value"
        ])
        layout.addWidget(QLabel("Per-Ore-Type Summary:"))
        layout.addWidget(self.summary_table)
        
        layout.addStretch()
        return widget
    
    @pyqtSlot()
    def _on_add_domain(self):
        """Add ore domain."""
        # Simplified - would open dialog
        self.show_info("Add Domain", "Domain editor would open here")
    
    @pyqtSlot()
    def _on_apply_ore_types(self):
        """Apply ore types to block model."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        if not self.controller.current_block_model:
            self.show_error("No Block Model", "Please load a block model first.")
            return
        
        # Create ore type model from table
        from ..geomet_chain.ore_type_model import OreTypeModel, OreDomain
        
        domains = []
        for r in range(self.ore_domains_table.rowCount()):
            domain_id = self.ore_domains_table.item(r, 0).text() if self.ore_domains_table.item(r, 0) else ""
            if domain_id:
                description = self.ore_domains_table.item(r, 1).text() if self.ore_domains_table.item(r, 1) else ""
                hardness = float(self.ore_domains_table.item(r, 2).text()) if self.ore_domains_table.item(r, 2) else 5.0
                density = float(self.ore_domains_table.item(r, 3).text()) if self.ore_domains_table.item(r, 3) else 2.7
                alteration = self.ore_domains_table.item(r, 4).text() if self.ore_domains_table.item(r, 4) else None
                
                domains.append(OreDomain(
                    id=domain_id,
                    description=description,
                    hardness_index=hardness,
                    density_t_m3=density,
                    alteration_code=alteration
                ))
        
        if not domains:
            self.show_warning("No Domains", "Please add at least one ore domain.")
            return
        
        ore_type_model = OreTypeModel(domains=domains)
        
        # Apply to block model
        from ..geomet_chain.ore_type_model import attach_ore_type_to_block_model
        
        config = {
            "domain_property": None,  # Would be set from UI
            "default_domain": domains[0].id if domains else "UNKNOWN"
        }
        
        try:
            attach_ore_type_to_block_model(
                self.controller.current_block_model,
                ore_type_model,
                config
            )
            
            self.ore_type_model = ore_type_model
            self.show_info("Applied", f"Applied {len(domains)} ore domains to block model")
        except Exception as e:
            logger.error(f"Error applying ore types: {e}", exc_info=True)
            self.show_error("Application Error", f"Failed to apply ore types:\n{e}")
    
    @pyqtSlot()
    def _on_add_route(self):
        """Add plant route."""
        # Simplified - would open dialog
        self.show_info("Add Route", "Route editor would open here")
    
    @pyqtSlot()
    def _on_compute_values(self):
        """Compute geomet block values."""
        if not self.controller:
            self.show_error("No Controller", "Controller not available.")
            return
        
        if not self.controller.current_block_model:
            self.show_error("No Block Model", "Please load a block model first.")
            return
        
        if not self.ore_type_model:
            self.show_warning("No Ore Types", "Please apply ore types to block model first.")
            return
        
        # Build configuration
        from ..geomet_chain.geomet_value_engine import GeometValueConfig
        from ..geomet_chain.route_selector import RouteSelectorConfig
        from ..geomet_chain.plant_response import PlantRoute
        
        # Get plant routes (simplified - would build from table)
        plant_routes = self.plant_routes if self.plant_routes else []
        
        route_selector_config = RouteSelectorConfig(
            routes=plant_routes,
            rule=self.selector_rule.currentText(),
            energy_cost_per_kWh=self.energy_cost.value(),
            plant_nominal_tph=self.plant_nominal_tph.value(),
            prices={"Fe": self.price_fe.value()}  # Simplified
        )
        
        try:
            geomet_config = GeometValueConfig(
                ore_type_model=self.ore_type_model,
                route_selector_config=route_selector_config,
                plant_routes=plant_routes,
                prices={"Fe": self.price_fe.value()},
                mining_cost_per_t=self.mining_cost.value()
            )
            
            config = {
                "block_model": self.controller.current_block_model,
                "geomet_config": geomet_config
            }
            
            self.show_progress("Computing geomet block values...")
            self.controller.run_geomet_chain(config, self._on_values_computed)
        except Exception as e:
            logger.error(f"Error computing geomet values: {e}", exc_info=True)
            self.hide_progress()
            self.show_error("Computation Error", f"Failed to compute geomet values:\n{e}")
    
    @pyqtSlot(dict)
    def _on_values_computed(self, result: dict):
        """Handle value computation completion."""
        self.hide_progress()
        
        if result.get("error"):
            self.show_error("Computation Error", result["error"])
            return
        
        value_field = result.get("result", {}).get("value_field", "gvalue_default")
        self.show_info("Values Computed", f"Geomet block values computed. Value field: {value_field}")
        
        # Update preview
        self._update_preview()
    
    def _update_preview(self):
        """Update preview tab."""
        if not self.controller or not self.controller.current_block_model:
            return
        
        # Get block model DataFrame
        df = self.controller.current_block_model.to_dataframe()
        
        if 'ore_domain' not in df.columns or 'gvalue_default' not in df.columns:
            return
        
        # Aggregate by ore domain
        summary = df.groupby('ore_domain').agg({
            'tonnage': 'sum',
            'gvalue_default': ['mean', 'sum']
        }).reset_index()
        
        self.summary_table.setRowCount(len(summary))
        for r, row in summary.iterrows():
            self.summary_table.setItem(r, 0, QTableWidgetItem(str(row['ore_domain'])))
            self.summary_table.setItem(r, 1, QTableWidgetItem(f"{row[('tonnage', 'sum')]:,.0f}"))
            self.summary_table.setItem(r, 2, QTableWidgetItem("N/A"))  # Would compute head grade
            self.summary_table.setItem(r, 3, QTableWidgetItem("N/A"))  # Would get from geomet_route
            self.summary_table.setItem(r, 4, QTableWidgetItem(f"${row[('gvalue_default', 'mean')]:,.2f}"))
            self.summary_table.setItem(r, 5, QTableWidgetItem(f"${row[('gvalue_default', 'sum')]:,.0f}"))
        
        self.summary_table.resizeColumnsToContents()
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect parameters."""
        return {}
    
    def validate_inputs(self) -> bool:
        """Validate inputs."""
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Handle results."""
        pass

    # =========================================================
    # PROJECT SAVE/RESTORE
    # =========================================================
    def get_panel_settings(self) -> Optional[Dict[str, Any]]:
        """Get panel settings for project save."""
        try:
            from .panel_settings_utils import get_safe_widget_value
            
            settings = {}
            
            # Value computation settings
            settings['selector_rule'] = get_safe_widget_value(self, 'selector_rule')
            settings['energy_cost'] = get_safe_widget_value(self, 'energy_cost')
            settings['plant_nominal_tph'] = get_safe_widget_value(self, 'plant_nominal_tph')
            settings['price_fe'] = get_safe_widget_value(self, 'price_fe')
            
            # Filter out None values
            settings = {k: v for k, v in settings.items() if v is not None}
            
            return settings if settings else None
            
        except Exception as e:
            logger.warning(f"Could not save geomet chain panel settings: {e}")
            return None

    def apply_panel_settings(self, settings: Dict[str, Any]) -> None:
        """Apply panel settings from project load."""
        if not settings:
            return
            
        try:
            from .panel_settings_utils import set_safe_widget_value
            
            # Value computation settings
            set_safe_widget_value(self, 'selector_rule', settings.get('selector_rule'))
            set_safe_widget_value(self, 'energy_cost', settings.get('energy_cost'))
            set_safe_widget_value(self, 'plant_nominal_tph', settings.get('plant_nominal_tph'))
            set_safe_widget_value(self, 'price_fe', settings.get('price_fe'))
                
            logger.info("Restored geomet chain panel settings from project")
            
        except Exception as e:
            logger.warning(f"Could not restore geomet chain panel settings: {e}")

