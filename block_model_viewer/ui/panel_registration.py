"""
Panel Registration Script for GeoX Application.

Automatically analyzes and registers all UI panels with the PanelManager.
Provides intelligent defaults based on panel names, locations, and functionality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Type, Any
import importlib
import inspect

from .panel_manager import PanelManager, PanelCategory, DockArea

logger = logging.getLogger(__name__)


class PanelRegistrar:
    """
    Automatically registers all panels in the GeoX application with the PanelManager.

    Analyzes panel classes to determine appropriate categories, shortcuts, and settings.
    """

    def __init__(self, panel_manager: PanelManager):
        self.panel_manager = panel_manager
        self._registered_panels = set()

    def register_all_panels(self):
        """Register all discoverable panels in the application."""
        logger.info("Starting automatic panel registration...")

        # Core panels (explicitly defined)
        self._register_core_panels()

        # Analysis and geostats panels
        self._register_analysis_panels()

        # Drillhole panels
        self._register_drillhole_panels()

        # ESG and optimization panels
        self._register_esg_panels()

        # Planning and scheduling panels
        self._register_planning_panels()

        # Underground mining panels
        self._register_underground_panels()

        # Geotechnical panels
        self._register_geotech_panels()

        # Resource modeling panels
        self._register_resource_panels()

        # Chart and visualization panels
        self._register_chart_panels()

        # Configuration and utility panels
        self._register_config_panels()

        logger.info(f"Registered {len(self._registered_panels)} panels with PanelManager")

    def _register_core_panels(self):
        """Register core viewer and property panels."""
        from .property_panel import PropertyPanel
        from .scene_inspector_panel import SceneInspectorPanel
        from .pick_info_panel import PickInfoPanel
        from .block_info_panel import BlockInfoPanel
        from .selection_panel import SelectionPanel
        from .cross_section_manager_panel import CrossSectionManagerPanel
        from .interactive_slicer_panel import InteractiveSlicerPanel
        from .display_settings_panel import DisplaySettingsPanel
        from .data_viewer_panel import DataViewerPanel
        from .table_viewer_panel import TableViewerPanel
        from .statistics_panel import StatisticsPanel
        from .charts_panel import ChartsPanel
        from .swath_panel import SwathPanel
        from .process_history_panel import ProcessHistoryPanel

        core_panels = [
            # PropertyPanel and SceneInspectorPanel are created manually in main_window.py
            # and added to the left tab widget, so they should not be registered with PanelManager to avoid duplicates
            # (PropertyPanel, PanelCategory.PROPERTY, "property", "Ctrl+1", DockArea.LEFT, True, "Property selection and filtering"),
            # (SceneInspectorPanel, PanelCategory.SCENE, "pit", "Ctrl+3", DockArea.LEFT, True, "Scene settings and overlays"),
            (PickInfoPanel, PanelCategory.INFO, "block", "Ctrl+P", DockArea.RIGHT, True, "Picked element information"),
            (BlockInfoPanel, PanelCategory.INFO, "info", "Ctrl+Shift+I", DockArea.RIGHT, True, "View information about selected blocks, drillholes, or geology"),
            (SelectionPanel, PanelCategory.SELECTION, "block", "Ctrl+S", DockArea.RIGHT, False, "Multi-block selection management"),
            (CrossSectionManagerPanel, PanelCategory.CROSS_SECTION, "block", "Ctrl+X", DockArea.RIGHT, False, "Cross-section creation and management"),
            (InteractiveSlicerPanel, PanelCategory.CROSS_SECTION, "block", "Ctrl+Shift+X", DockArea.RIGHT, False, "Interactive slicing with draggable widgets"),
            (DisplaySettingsPanel, PanelCategory.DISPLAY, "layers", "Ctrl+D", DockArea.RIGHT, False, "Display and rendering settings"),
            (DataViewerPanel, PanelCategory.RESOURCE, "table", None, DockArea.RIGHT, False, "Data table viewer"),
            (TableViewerPanel, PanelCategory.RESOURCE, "table", None, DockArea.RIGHT, False, "Advanced table viewer"),
            (StatisticsPanel, PanelCategory.ANALYSIS, "chart", None, DockArea.RIGHT, False, "Statistical analysis"),
            (ChartsPanel, PanelCategory.CHART, "chart", None, DockArea.RIGHT, False, "Chart visualization"),
            (ProcessHistoryPanel, PanelCategory.OTHER, "chart", None, DockArea.RIGHT, False, "Process execution history"),
            (SwathPanel, PanelCategory.ANALYSIS, "chart", None, DockArea.RIGHT, False, "Swath plot analysis"),
        ]

        for panel_class, category, icon, shortcut, dock_area, visible, tooltip in core_panels:
            self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

    def _register_analysis_panels(self):
        """Register geostatistics and analysis panels."""
        try:
            from .insar_panel import InSARPanel
            self._register_single_panel(
                InSARPanel,
                PanelCategory.ANALYSIS,
                None,
                None,
                DockArea.RIGHT,
                False,
                "InSAR deformation via ISCE-2",
            )
        except ImportError as e:
            logger.warning(f"InSAR panel not available: {e}")

        try:
            from .kriging_panel import KrigingPanel
            from .simple_kriging_panel import SimpleKrigingPanel
            from .universal_kriging_panel import UniversalKrigingPanel
            from .indicator_kriging_panel import IndicatorKrigingPanel
            from .cokriging_panel import CoKrigingPanel
            from .soft_kriging_panel import SoftKrigingPanel
            from .variogram_panel import VariogramAnalysisPanel as VariogramPanel
            from .variogram_assistant_panel import VariogramAssistantPanel
            from .grf_panel import GRFPanel
            from .sis_panel import SISPanel
            from .ik_sgsim_panel import IKSGSIMPanel
            from .sgsim_panel import SGSIMPanel
            from .cosgsim_panel import CoSGSIMPanel
            from .turning_bands_panel import TurningBandsPanel
            from .mps_panel import MPSPanel
            from .direct_block_sim import DirectBlockSimPanel
            from .dbs_panel import DBSPanel
            from .declustering_panel import DeclusteringPanel
            from .scan_panel import ScanPanel
            from .survey_deformation_panel import SurveyDeformationPanel

            analysis_panels = [
                (KrigingPanel, PanelCategory.GEOSTATS, "kriging", "Ctrl+K", DockArea.LEFT, False, "Ordinary kriging interpolation"),
                (SimpleKrigingPanel, PanelCategory.GEOSTATS, "kriging", None, DockArea.LEFT, False, "Simple kriging interpolation"),
                (UniversalKrigingPanel, PanelCategory.GEOSTATS, "kriging", None, DockArea.LEFT, False, "Universal kriging interpolation"),
                (IndicatorKrigingPanel, PanelCategory.GEOSTATS, "kriging", None, DockArea.LEFT, False, "Indicator kriging"),
                (CoKrigingPanel, PanelCategory.GEOSTATS, "kriging", None, DockArea.LEFT, False, "Co-kriging with secondary variables"),
                (SoftKrigingPanel, PanelCategory.GEOSTATS, "kriging", None, DockArea.LEFT, False, "Soft kriging with uncertainty"),
                (VariogramPanel, PanelCategory.GEOSTATS, "variogram", "Ctrl+V", DockArea.LEFT, False, "Variogram analysis and modeling"),
                (VariogramAssistantPanel, PanelCategory.GEOSTATS, "variogram", None, DockArea.LEFT, False, "Automated variogram fitting"),
                (GRFPanel, PanelCategory.GEOSTATS, "grf", None, DockArea.LEFT, False, "Gaussian Random Field simulation"),
                (SISPanel, PanelCategory.GEOSTATS, "sis", None, DockArea.LEFT, False, "Sequential Indicator Simulation"),
                (IKSGSIMPanel, PanelCategory.GEOSTATS, "sgsim", None, DockArea.LEFT, False, "Indicator Kriging Sequential Gaussian Simulation"),
                (SGSIMPanel, PanelCategory.GEOSTATS, "sgsim", None, DockArea.LEFT, False, "Sequential Gaussian Simulation"),
                (CoSGSIMPanel, PanelCategory.GEOSTATS, "sgsim", None, DockArea.LEFT, False, "Co-Sequential Gaussian Simulation"),
                (TurningBandsPanel, PanelCategory.GEOSTATS, "turning_bands", None, DockArea.LEFT, False, "Turning Bands simulation"),
                (MPSPanel, PanelCategory.GEOSTATS, "mps", None, DockArea.LEFT, False, "Multiple Point Statistics"),
                (DirectBlockSimPanel, PanelCategory.GEOSTATS, "block", None, DockArea.LEFT, False, "Direct block simulation"),
                (DBSPanel, PanelCategory.GEOSTATS, "block", None, DockArea.LEFT, False, "Distance Based Simulation"),
                (DeclusteringPanel, PanelCategory.GEOSTATS, "decluster", None, DockArea.LEFT, False, "Sample declustering"),
                (ScanPanel, PanelCategory.ANALYSIS, "scan", "Ctrl+Shift+S", DockArea.RIGHT, False, "Scan analysis and fragmentation"),
                (SurveyDeformationPanel, PanelCategory.ANALYSIS, None, "Ctrl+Shift+D", DockArea.RIGHT, True, "Survey deformation & subsidence analysis"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in analysis_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Some geostats panels not available: {e}")


    def _register_drillhole_panels(self):
        """Register drillhole-related panels."""
        try:
            from .drillhole_panel import DrillholePanel
            from .drillhole_import_panel import DrillholeImportPanel
            from .drillhole_control_panel import DrillholeControlPanel
            from .drillhole_info_panel import DrillholeInfoPanel
            from .drillhole_plotting_panel import DrillholePlottingPanel
            from .drillhole_reporting_panel import DrillholeReportingPanel
            from .qc_window import QCWindow
            from .compositing_window import CompositingWindow

            drillhole_panels = [
                (DrillholePanel, PanelCategory.DRILLHOLE, "drillhole", "Ctrl+H", DockArea.LEFT, False, "Drillhole visualization"),
                (DrillholeImportPanel, PanelCategory.DRILLHOLE, "import", None, DockArea.LEFT, False, "Import drillhole data"),
                (DrillholeControlPanel, PanelCategory.DRILLHOLE, "drillhole", None, DockArea.LEFT, False, "Drillhole display controls"),
                (DrillholeInfoPanel, PanelCategory.DRILLHOLE, "info", None, DockArea.RIGHT, False, "Drillhole information"),
                (DrillholePlottingPanel, PanelCategory.DRILLHOLE, "chart", None, DockArea.RIGHT, False, "Drillhole plotting"),
                (DrillholeReportingPanel, PanelCategory.DRILLHOLE, "report", None, DockArea.RIGHT, False, "Drillhole reporting"),
                (QCWindow, PanelCategory.DRILLHOLE, "qc", "Ctrl+Q", DockArea.RIGHT, False, "Quality control and validation"),
                (CompositingWindow, PanelCategory.DRILLHOLE, "composite", None, DockArea.RIGHT, False, "Sample compositing"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in drillhole_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Some drillhole panels not available: {e}")

    def _register_esg_panels(self):
        """Register ESG and sustainability panels."""
        try:
            from .esg_dashboard_panel import ESGDashboardPanel

            esg_panels = [
                (ESGDashboardPanel, PanelCategory.ESG, "esg", None, DockArea.RIGHT, False, "ESG and sustainability dashboard"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in esg_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"ESG panels not available: {e}")

    def _register_planning_panels(self):
        """Register mine planning and scheduling panels."""
        try:
            from .mine_planning.npvs.npvs_panel import NPVSPanel
            from .mine_planning.scheduling.short_term_schedule_panel import ShortTermSchedulePanel
            from .mine_planning.scheduling.strategic_schedule_panel import StrategicSchedulePanel
            from .mine_planning.scheduling.tactical_schedule_panel import TacticalSchedulePanel
            from .planning_dashboard_panel import PlanningDashboardPanel
            from .production_dashboard_panel import ProductionDashboardPanel
            from .research_dashboard_panel import ResearchDashboardPanel
            from .pit_optimisation_panel import PitOptimisationPanel
            from .pit_optimizer_panel import PitOptimizerPanel
            from .pushback_designer_panel import PushbackDesignerPanel
            from .bench_design_panel import BenchDesignPanel
            from .grade_control_panel import GradeControlPanel
            from .gc_decision_panel import GCDecisionPanel
            planning_panels = [
                (NPVSPanel, PanelCategory.PLANNING, "npv", None, DockArea.RIGHT, False, "Net Present Value analysis"),
                (ShortTermSchedulePanel, PanelCategory.PLANNING, "schedule", None, DockArea.RIGHT, False, "Short-term scheduling"),
                (StrategicSchedulePanel, PanelCategory.PLANNING, "schedule", None, DockArea.RIGHT, False, "Strategic scheduling"),
                (TacticalSchedulePanel, PanelCategory.PLANNING, "schedule", None, DockArea.RIGHT, False, "Tactical scheduling"),
                (PlanningDashboardPanel, PanelCategory.PLANNING, "dashboard", None, DockArea.RIGHT, False, "Planning dashboard"),
                (ProductionDashboardPanel, PanelCategory.PLANNING, "dashboard", None, DockArea.RIGHT, False, "Production dashboard"),
                (ResearchDashboardPanel, PanelCategory.PLANNING, "research", None, DockArea.RIGHT, False, "Research dashboard"),
                (PitOptimisationPanel, PanelCategory.OPTIMIZATION, "pit", None, DockArea.LEFT, False, "Pit optimization"),
                (PitOptimizerPanel, PanelCategory.OPTIMIZATION, "pit", None, DockArea.LEFT, False, "Advanced pit optimization"),
                (PushbackDesignerPanel, PanelCategory.PLANNING, "pushback", None, DockArea.LEFT, False, "Pushback design"),
                (BenchDesignPanel, PanelCategory.PLANNING, "bench", None, DockArea.LEFT, False, "Bench design"),
                (GradeControlPanel, PanelCategory.RESOURCE, "grade", None, DockArea.LEFT, False, "Grade control"),
                (GCDecisionPanel, PanelCategory.RESOURCE, "decision", None, DockArea.RIGHT, False, "Grade control decisions"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in planning_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Some planning panels not available: {e}")

    def _register_underground_panels(self):
        """Register underground mining panels."""
        try:
            from .underground_panel import UndergroundPanel
            from .ug_advanced_panel import UGAdvancedPanel
            from .stope_stability_panel import StopeStabilityPanel

            ug_panels = [
                # UndergroundPanel removed: opened as floating dialog via main_window.open_underground_panel()
                # (UndergroundPanel, PanelCategory.PLANNING, "underground", None, DockArea.LEFT, False, "Underground mining"),
                (UGAdvancedPanel, PanelCategory.PLANNING, "underground", None, DockArea.LEFT, False, "Advanced underground design"),
                (StopeStabilityPanel, PanelCategory.PLANNING, "stope", None, DockArea.LEFT, False, "Stope stability analysis"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in ug_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Underground panels not available: {e}")

    def _register_geotech_panels(self):
        """Register geotechnical panels."""
        try:
            from .geotech_panel import GeotechPanel
            from .geotech_summary_panel import GeotechSummaryPanel
            from .slope_risk_panel import SlopeRiskPanel
            from .slope_stability_panel import SlopeStabilityPanel

            geotech_panels = [
                (GeotechPanel, PanelCategory.RESOURCE, "geotech", None, DockArea.LEFT, False, "Geotechnical analysis"),
                (GeotechSummaryPanel, PanelCategory.RESOURCE, "geotech", None, DockArea.RIGHT, False, "Geotechnical summary"),
                (SlopeRiskPanel, PanelCategory.RESOURCE, "slope", None, DockArea.LEFT, False, "Slope risk assessment"),
                (SlopeStabilityPanel, PanelCategory.RESOURCE, "slope", None, DockArea.LEFT, False, "Slope stability analysis"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in geotech_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Geotech panels not available: {e}")

    def _register_resource_panels(self):
        """Register resource modeling panels."""
        try:
            from .resource_classification_panel import JORCClassificationPanel
            from .jorc_classification_panel import JORCClassificationPanel as JORCClassificationPanel2
            from .resource_reporting_panel import ResourceReportingPanel
            from .block_resource_panel import BlockModelResourcePanel
            from .grade_transformation_panel import GradeTransformationPanel
            from .kmeans_clustering_panel import KMeansClusteringPanel
            from .block_property_calculator_panel import BlockPropertyCalculatorPanel

            resource_panels = [
                (JORCClassificationPanel, PanelCategory.RESOURCE, "jorc", None, DockArea.RIGHT, False, "JORC resource classification"),
                (JORCClassificationPanel2, PanelCategory.RESOURCE, "jorc", None, DockArea.RIGHT, False, "JORC resource classification"),
                (ResourceReportingPanel, PanelCategory.RESOURCE, "report", None, DockArea.RIGHT, False, "Resource reporting"),
                (BlockModelResourcePanel, PanelCategory.RESOURCE, "block", None, DockArea.RIGHT, False, "Block model resources"),
                (BlockPropertyCalculatorPanel, PanelCategory.RESOURCE, "block", None, DockArea.RIGHT, False, "Calculate and add tonnage/volume properties to block models"),
                (GradeTransformationPanel, PanelCategory.RESOURCE, "transform", None, DockArea.RIGHT, False, "Grade transformation"),
                (KMeansClusteringPanel, PanelCategory.RESOURCE, "cluster", None, DockArea.RIGHT, False, "K-means clustering"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in resource_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Some resource panels not available: {e}")

        # Geological Explorer Panel
        try:
            from .geological_explorer_panel import GeologicalExplorerPanel
            self._register_single_panel(
                GeologicalExplorerPanel,
                PanelCategory.RESOURCE,
                "geology",
                None,
                DockArea.RIGHT,
                True,  # Make visible by default so it appears at startup
                "Control geological model visualization"
            )
        except ImportError as e:
            logger.warning(f"Geological Explorer panel not available: {e}")

    def _register_chart_panels(self):
        """Register chart and visualization panels."""
        try:
            from .separated_charts import SeparatedChartsPanel

            chart_panels = [
                (SeparatedChartsPanel, PanelCategory.CHART, "chart", None, DockArea.RIGHT, False, "Separated charts"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in chart_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Chart panels not available: {e}")

    def _register_config_panels(self):
        """Register configuration and utility panels."""
        try:
            from .preferences_dialog import PreferencesDialog
            from .theme_manager import ThemeManager
            from .data_registry_status_panel import DataRegistryStatusPanel
            from .irr_panel import IRRPanel

            config_panels = [
                (DataRegistryStatusPanel, PanelCategory.CONFIG, "registry", None, DockArea.RIGHT, False, "Data registry status"),
                (IRRPanel, PanelCategory.RESOURCE, "irr", None, DockArea.RIGHT, False, "Internal Rate of Return analysis"),
            ]

            for panel_class, category, icon, shortcut, dock_area, visible, tooltip in config_panels:
                self._register_single_panel(panel_class, category, icon, shortcut, dock_area, visible, tooltip)

        except ImportError as e:
            logger.warning(f"Some config panels not available: {e}")

    def _register_single_panel(self, panel_class: Type, category: PanelCategory,
                              icon: str = None, shortcut: str = None,
                              dock_area: DockArea = DockArea.LEFT,
                              default_visible: bool = False, tooltip: str = None):
        """
        Register a single panel with intelligent defaults.

        Args:
            panel_class: The panel class to register
            category: Panel category
            icon: Icon name
            shortcut: Keyboard shortcut
            dock_area: Default dock area
            default_visible: Whether visible by default
            tooltip: Help tooltip
        """
        try:
            # Get panel ID from class
            panel_id = getattr(panel_class, 'PANEL_ID', panel_class.__name__)

            if panel_id in self._registered_panels:
                logger.debug(f"Panel {panel_id} already registered, skipping")
                return

            # Register with PanelManager
            self.panel_manager.register_panel_class(
                panel_class=panel_class,
                category=category,
                icon_name=icon,
                shortcut=shortcut,
                default_dock_area=dock_area,
                default_visible=default_visible,
                tooltip=tooltip
            )

            self._registered_panels.add(panel_id)
            logger.debug(f"Registered panel: {panel_id}")

        except Exception as e:
            logger.warning(f"Failed to register panel {panel_class.__name__}: {e}")


def register_all_panels(panel_manager: PanelManager):
    """
    Convenience function to register all panels with a PanelManager.

    Args:
        panel_manager: The PanelManager instance to register panels with
    """
    registrar = PanelRegistrar(panel_manager)
    registrar.register_all_panels()
