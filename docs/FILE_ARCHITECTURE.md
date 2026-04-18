# GeoX File Architecture Documentation

**Generated:** February 21, 2026  
**Application:** GeoX Block Model Viewer  
**Version:** v2.0+

---

## Table of Contents

1. [Overview](#overview)
2. [Entry Points](#entry-points)
3. [Core Systems](#core-systems)
4. [Controllers Layer](#controllers-layer)
5. [Models Layer](#models-layer)
6. [UI Layer](#ui-layer)
7. [Visualization Layer](#visualization-layer)
8. [Domain-Specific Modules](#domain-specific-modules)
9. [Utility Modules](#utility-modules)
10. [Import Relationships](#import-relationships)
11. [Data Flow Architecture](#data-flow-architecture)
12. [Panel System Architecture](#panel-system-architecture)
13. [Menu System Architecture](#menu-system-architecture)
14. [Theme System Architecture](#theme-system-architecture)
15. [Patterns and Best Practices](#patterns-and-best-practices)

---

## Overview

GeoX follows a **layered architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points (__main__, main.py)         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│               UI Layer (MainWindow, Panels, Menus)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Controllers (AppController + Sub-Controllers)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Models (BlockModel, DataFrames, etc.)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Visualization (Renderer, PyVista)               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Domain Modules (Geostats, Mining, IRR, etc.)       │
└─────────────────────────────────────────────────────────────┘
```

**Key Design Principles:**
- **Single Responsibility**: Each module has one clear purpose
- **Dependency Injection**: Controllers and dependencies are passed to panels
- **Signal-Based Communication**: PyQt signals for loose coupling
- **Layered Architecture**: Clear separation UI → Controller → Model → Core
- **Plugin-Style Panels**: Panels registered via central registry
- **Menu-Driven UI**: Modular menu system

---

## Entry Points

### `__main__.py`
**Purpose:** Application entry point for `python -m block_model_viewer`

**Key Functions:**
- Sets up session logging before any imports
- Delegates to `main.py` via `from .main import main`

**Imports:**
```python
from block_model_viewer.utils.session_logger import setup_session_logging
from .main import main
```

### `main.py` (532 lines)
**Purpose:** Main application initialization and launch

**Key Functions:**
- Configures environment variables (Qt, VTK, matplotlib)
- Sets up logging infrastructure (console + file handlers)
- Initializes QApplication
- Creates and shows MainWindow
- Handles crash recovery and exception handling

**Key Environment Setup:**
```python
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['VTK_SILENCE_GET_VOID_POINTER_WARNINGS'] = '1'
matplotlib.use('QtAgg')  # PyQt6 compatibility
```

**Imports:**
- `PyQt6` for GUI
- `ui.main_window.MainWindow`
- `config.Config`
- Logging and system utilities

---

## Core Systems

Directory: `block_model_viewer/core/`

### Core Architecture
The core systems provide fundamental infrastructure for the entire application.

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| **`state_manager.py`** | Application state persistence | StateManager |
| **`data_registry.py`** | Central data source registry | DataRegistry |
| **`data_registry_simple.py`** | Simplified data registry | SimpleDataRegistry |
| **`process_history_tracker.py`** | Process provenance tracking | ProcessHistoryTracker |
| **`audit_manager.py`** | Audit trail for compliance | AuditManager |
| **`data_provenance.py`** | Data lineage tracking | ProvenanceTracker |
| **`crash_handler.py`** | Exception handling and recovery | CrashHandler |
| **`errors.py`** | Custom exception classes | GeoXError, ValidationError |
| **`worker.py`** | Background task execution | Worker |
| **`scan_registry.py`** | Scan data management | ScanRegistry |
| **`economic_rule.py`** | Economic calculation rules | EconomicRule |
| **`thread_safe_cache.py`** | Thread-safe caching | ThreadSafeCache |

### Key Import Relationships (Core)

```
core/
│
├── data_registry.py
│   └── imports: data_provenance, process_history_tracker
│
├── state_manager.py
│   └── imports: PyQt6.QtCore (QSettings)
│
├── audit_manager.py
│   └── imports: process_history_tracker, data_provenance
│
└── worker.py
    └── imports: PyQt6.QtCore (QThread, pyqtSignal)
```

---

## Controllers Layer

Directory: `block_model_viewer/controllers/`

### Controller Architecture

The application uses a **main controller with specialized sub-controllers** pattern:

```
AppController (Orchestrator)
    │
    ├── GeostatsController (Kriging, Simulation, Variograms)
    ├── MiningController (Resources, IRR/NPV, Planning)
    ├── VisController (Rendering, Layers, Overlays)
    ├── DataController (Drillholes, Geology, Structural)
    ├── ScanController (3D Scan Management)
    ├── SurveyDeformationController (Survey Data)
    └── InsarController (InSAR Remote Sensing)
```

### Controllers

| File | Class | Purpose |
|------|-------|---------|
| **`app_controller.py`** | `AppController` | Main orchestration layer between UI and Renderer |
| **`geostats_controller.py`** | `GeostatsController` | Geostatistics operations (kriging, simulation, variogram) |
| **`mining_controller.py`** | `MiningController` | Resource classification, IRR/NPV, scheduling |
| **`vis_controller.py`** | `VisController` | Rendering, layers, legend management |
| **`data_controller.py`** | `DataController` | Data import/export, drillholes, geology |
| **`scan_controller.py`** | `ScanController` | 3D scan data processing |
| **`survey_deformation_controller.py`** | `SurveyDeformationController` | Survey deformation analysis |
| **`insar_controller.py`** | `InsarController` | InSAR remote sensing operations |
| **`app_state.py`** | `AppState`, `SessionState` | Application state management |
| **`controller_signals.py`** | `ControllerSignals` | Signal definitions for controllers |
| **`job_worker.py`** | `JobWorker` | Background job execution |
| **`job_registry.py`** | `JobRegistry` | Job tracking and management |
| **`grid_builder.py`** | `GridBuilder` | Block model grid construction |

### AppController Key Responsibilities

From `app_controller.py`:

```python
class AppController:
    """
    Thin orchestration layer between UI panels and the Renderer.
    
    Manages shared state and provides unified interface for visualization.
    Delegates domain-specific work to specialized sub-controllers.
    """
    
    def __init__(self, renderer, config=None, registry=None):
        # Initialize sub-controllers
        self.geostats = GeostatsController(self)
        self.mining = MiningController(self)
        self.vis = VisController(self)
        self.data = DataController(self)
        # ... more sub-controllers
```

**Key Methods:**
- `add_block_model()` - Add block model to scene
- `update_property()` - Update displayed property
- `apply_filter()` - Apply data filters
- `export_results()` - Export analysis results
- Signal routing between components

### Import Relationships (Controllers)

```
controllers/
│
├── app_controller.py
│   ├── imports: PyQt6.QtCore (QObject, pyqtSignal)
│   ├── imports: visualization.render_payloads
│   ├── imports: core.process_history_tracker
│   └── imports: All sub-controllers
│
├── geostats_controller.py
│   ├── imports: geostats/* (kriging, simulation engines)
│   └── imports: models.geostat_results
│
├── mining_controller.py
│   ├── imports: irr_engine/*
│   ├── imports: models.resource_classification
│   └── imports: models.pit_optimizer
│
└── vis_controller.py
    ├── imports: visualization.renderer
    └── imports: visualization.render_payloads
```

---

## Models Layer

Directory: `block_model_viewer/models/`

### Model Classes

| File | Purpose | Key Classes |
|------|---------|-------------|
| **`block_model.py`** | Core 3D block model structure | `BlockModel`, `BlockMetadata` |
| **`blockmodel_builder.py`** | Block model construction | `BlockModelBuilder` |
| **`blockmodel_advanced.py`** | Advanced operations | Advanced queries and transformations |
| **`kriging3d.py`** | 3D kriging implementation | Kriging algorithms |
| **`kriging_engine.py`** | Kriging execution engine | `KrigingEngine` |
| **`kriging_results_builder.py`** | Kriging result handling | Result formatters |
| **`sgsim3d.py`** | 3D sequential Gaussian simulation | SGSIM algorithms |
| **`sgsim_engine.py`** | SGSIM execution engine | `SGSIMEngine` |
| **`simple_kriging3d.py`** | Simple kriging implementation | Simple kriging |
| **`variogram3d.py`** | 3D variogram modeling | `Variogram3D` |
| **`variogram_functions.py`** | Variogram functions | Spherical, exponential, gaussian |
| **`pit_optimizer.py`** | Pit optimization algorithms | Lerchs-Grossmann |
| **`compositor.py`** | Drillhole compositing | Compositing engine |
| **`resource_classification.py`** | JORC resource classification | Classification engine |
| **`resource_reporting_engine.py`** | Resource reporting | Report generation |
| **`jorc_classification_engine.py`** | JORC compliance | JORC-specific logic |
| **`transform.py`** | Grade transformations | Box-Cox, log transforms |
| **`post_processing.py`** | Post-processing utilities | Smoothing, filtering |
| **`geostat_export.py`** | Export geostatistics results | Export formatters |
| **`geostat_results.py`** | Geostatistics result storage | Result containers |
| **`workflow_manager.py`** | Workflow orchestration | Workflow execution |
| **`simulation_workflow_manager.py`** | Simulation workflows | Simulation orchestration |
| **`visualization.py`** | Visualization helpers | Plot generation |

### BlockModel Structure

From `block_model.py`:

```python
@dataclass
class BlockMetadata:
    """Metadata for a block model."""
    coordinate_system: str = "unknown"
    units: str = "meters"
    source_file: str = ""
    file_format: str = ""
    creation_date: str = ""
    # Provenance fields
    file_checksum: str = ""
    import_timestamp: str = ""
    parser_version: str = ""

class BlockModel:
    """Core data structure for 3D block model information."""
    
    def __init__(self, metadata: Optional[BlockMetadata] = None):
        self.metadata = metadata or BlockMetadata()
        self.data = pd.DataFrame()  # Block properties
        self.positions = None  # numpy array of (x, y, z)
        self.dimensions = None  # numpy array of (dx, dy, dz)
```

### Import Relationships (Models)

```
models/
│
├── block_model.py
│   ├── imports: pandas, numpy
│   └── imports: numba (optional, for performance)
│
├── kriging_engine.py
│   ├── imports: models.kriging3d
│   ├── imports: models.variogram3d
│   └── imports: models.kriging_results_builder
│
└── resource_classification.py
    ├── imports: models.block_model
    └── imports: models.variogram3d
```

---

## UI Layer

Directory: `block_model_viewer/ui/`

### UI Architecture Overview

The UI layer follows a **panel-based architecture** where:
- **MainWindow** is the primary container
- **Panels** are dockable widgets that provide specific functionality
- **Menus** provide access to panels and actions
- **Dialogs** handle specific user interactions

### Main Window

**File:** `main_window.py` (15,713 lines)

**Purpose:** Primary application window and UI orchestrator

**Key Responsibilities:**
- Creates and manages the PyVista renderer
- Initializes AppController
- Builds menu system
- Manages panel lifecycle
- Handles file I/O operations
- Coordinates between UI components

**Key Components:**
```python
class MainWindow(QMainWindow):
    def __init__(self):
        # Initialize renderer
        self.renderer = Renderer()
        
        # Initialize controller
        self.controller = AppController(self.renderer, config, registry)
        
        # Setup menus
        self._build_menus()
        
        # Setup panels
        self.panel_manager = PanelManager(self)
        register_all_panels(self.panel_manager)
```

### Panel Base Classes

Located in `ui/base_panel.py`:

| Base Class | Purpose | Inheritance |
|-----------|---------|-------------|
| **`BasePanel`** | Base for all panels | QWidget |
| **`BaseDockPanel`** | Base for dockable panels | BasePanel |
| **`BaseAnalysisPanel`** | Base for analysis panels | BaseDockPanel |
| **`BaseDisplayPanel`** | Base for display panels | BaseDockPanel |
| **`BaseDialogPanel`** | Base for dialog panels | QDialog |

### BasePanel Architecture

From `base_panel.py`:

```python
class BasePanel(QWidget):
    """Base class for all panel widgets."""
    
    # Diagnostic identifier
    PANEL_ID: str = "BasePanel"
    
    # Common signals
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, parent=None, panel_id=None):
        super().__init__(parent)
        self.controller = None  # Set by MainWindow
        self._block_model = None
        self._is_initialized = False
        
        # Template method pattern
        self.setup_ui()  # Must be implemented by subclasses
        self.connect_signals()  # Optional override
    
    def setup_ui(self):
        """Build widget layout. Subclasses must override."""
        raise NotImplementedError()
    
    def connect_signals(self):
        """Wire Qt signals. Subclasses can override."""
        pass
    
    def refresh(self):
        """Refresh panel state from latest data."""
        pass
    
    def set_controller(self, controller):
        """Inject controller dependency."""
        self.controller = controller
```

### Panel Categories

Panels are organized by functionality:

#### 1. Data Import/Management Panels

| Panel | File | Purpose |
|-------|------|---------|
| DrillholeImportPanel | `drillhole_import_panel.py` | Import drillhole data |
| DrillholeDatabasePanel | `drillhole_database_panel.py` | Manage drillhole database |
| BlockModelImportPanel | `block_model_import_panel.py` | Import block models |
| StructuralImportPanel | `structural_import_panel.py` | Import structural geology |
| DataViewerPanel | `data_viewer_panel.py` | View tabular data |
| DataRegistryStatusPanel | `data_registry_status_panel.py` | Data source status |

#### 2. Visualization/Display Panels

| Panel | File | Purpose |
|-------|------|---------|
| PropertyPanel | `property_panel.py` | Select/display properties |
| DisplaySettingsPanel | `display_settings_panel.py` | Display controls |
| AxesScaleBarPanel | `axes_scalebar_panel.py` | Axes and scale bar |
| LegendWidget | `legend_widget.py` | Color legend display |
| MousePanel | `mouse_panel.py` | Mouse interaction controls |
| SelectionPanel | `selection_panel.py` | Selection tools |
| SceneInspectorPanel | `scene_inspector_panel.py` | Scene hierarchy |

#### 3. Drillhole Analysis Panels

| Panel | File | Purpose |
|-------|------|---------|
| DrillholeControlPanel | `drillhole_control_panel.py` | Drillhole display controls |
| DrillholePlottingPanel | `drillhole_plotting_panel.py` | Drillhole plotting |
| DrillholeReportingPanel | `drillhole_reporting_panel.py` | Drillhole reports |
| DrillholeInfoPanel | `drillhole_info_panel.py` | Drillhole information |
| CompositingWindow | `compositing_window.py` | Drillhole compositing |
| DeclusteringPanel | `declustering_panel.py` | Statistical declustering |

#### 4. Geostatistics Panels

| Panel | File | Purpose |
|-------|------|---------|
| VariogramAnalysisPanel | `variogram_panel.py` | Variogram modeling |
| VariogramAssistantPanel | `variogram_assistant_panel.py` | Variogram wizard |
| KrigingPanel | `kriging_panel.py` | Ordinary kriging |
| SimpleKrigingPanel | `simple_kriging_panel.py` | Simple kriging |
| UniversalKrigingPanel | `universal_kriging_panel.py` | Universal kriging |
| CokrigingPanel | `cokriging_panel.py` | Co-kriging |
| IndicatorKrigingPanel | `indicator_kriging_panel.py` | Indicator kriging |
| SGSIMPanel | `sgsim_panel.py` | Sequential Gaussian simulation |
| IKSGSIMPanel | `ik_sgsim_panel.py` | Indicator kriging SGSIM |
| SISPanel | `sis_panel.py` | Sequential indicator simulation |
| COSGSIMPanel | `cosgsim_panel.py` | Co-SGSIM |
| TurningBandsPanel | `turning_bands_panel.py` | Turning bands simulation |
| MPSPanel | `mps_panel.py` | Multiple-point statistics |
| DBSPanel | `dbs_panel.py` | Direct block simulation |
| GRFPanel | `grf_panel.py` | Gaussian random fields |
| RBFPanel | `rbf_panel.py` | Radial basis functions |
| SoftKrigingPanel | `soft_kriging_panel.py` | Soft kriging (with soft data) |

#### 5. Resource/Classification Panels

| Panel | File | Purpose |
|-------|------|---------|
| JORCClassificationPanel | `jorc_classification_panel.py` | JORC resource classification |
| ResourceClassificationPanel | `resource_classification_panel.py` | Resource classification |
| ResourceReportingPanel | `resource_reporting_panel.py` | Resource reporting |
| BlockModelResourcePanel | `block_resource_panel.py` | Block model resources |
| GradeTonnagePanel | `grade_tonnage_panel.py` | Grade-tonnage curves |
| GradeTonnageBasicPanel | `grade_tonnage_basic_panel.py` | Simple grade-tonnage |
| CutoffOptimizationPanel | `cutoff_optimization_panel.py` | Cutoff grade optimization |

#### 6. Mine Planning Panels

| Panel | File | Purpose |
|-------|------|---------|
| PitOptimisationPanel | `pit_optimisation_panel.py` | Pit optimization |
| PitOptimizerPanel | `pit_optimizer_panel.py` | Pit shell generation |
| PushbackDesignerPanel | `pushback_designer_panel.py` | Pushback design |
| IRRPanel | `irr_panel.py` | IRR/NPV analysis |
| NPVSPanel | `npvs_panel.py` | NPV scenarios |
| StrategicSchedulePanel | `strategic_schedule_panel.py` | Long-term scheduling |
| TacticalSchedulePanel | `tactical_schedule_panel.py` | Mid-term scheduling |
| ShortTermSchedulePanel | `short_term_schedule_panel.py` | Short-term scheduling |
| FleetPanel | `fleet_panel.py` | Fleet management |

#### 7. Underground Mining Panels

| Panel | File | Purpose |
|-------|------|---------|
| UndergroundPanel | `underground_panel.py` | Underground design |
| UGAdvancedPanel | `ug_advanced_panel.py` | Advanced UG tools |
| StopeStabilityPanel | `stope_stability_panel.py` | Stope stability analysis |

#### 8. Geology/Structural Panels

| Panel | File | Purpose |
|-------|------|---------|
| LoopStructuralModelPanel | `loopstructural_panel.py` | LoopStructural modeling |
| LoopStructuralAdvisoryPanel | `loopstructural_advisory_panel.py` | Geological advisory |
| LoopStructuralCompliancePanel | `loopstructural_compliance_panel.py` | Compliance checking |
| FaultDefinitionPanel | `fault_definition_panel.py` | Fault definition |
| FoldDefinitionPanel | `fold_definition_panel.py` | Fold definition |
| VeinDefinitionPanel | `vein_definition_panel.py` | Vein definition |
| StructuralPanel | `structural_panel.py` | Structural geology |
| GeologicalExplorerPanel | `geological_explorer_panel.py` | Geological exploration |

#### 9. Geotech Panels

| Panel | File | Purpose |
|-------|------|---------|
| GeotechPanel | `geotech_panel.py` | Geotechnical analysis |
| GeotechSummaryPanel | `geotech_summary_panel.py` | Geotech summary |
| SlopeStabilityPanel | `slope_stability_panel.py` | Slope stability |
| SlopeRiskPanel | `slope_risk_panel.py` | Slope risk assessment |
| RockburstPanel | `rockburst_panel.py` | Rockburst analysis |

#### 10. Cross-Section/Swath Panels

| Panel | File | Purpose |
|-------|------|---------|
| CrossSectionPanel | `cross_section_panel.py` | Cross-section creation |
| CrossSectionManagerPanel | `cross_section_manager_panel.py` | Cross-section management |
| SwathPanel | `swath_panel.py` | Swath plots |
| SwathAnalysis3DPanel | `swath_analysis_3d_panel.py` | 3D swath analysis |

#### 11. Data Analysis Panels

| Panel | File | Purpose |
|-------|------|---------|
| StatisticsPanel | `statistics_panel.py` | Statistical analysis |
| ChartsPanel | `charts_panel.py` | Chart generation |
| BlockPropertyCalculatorPanel | `block_property_calculator_panel.py` | Property calculations |
| GradeTransformationPanel | `grade_transformation_panel.py` | Grade transformations |
| KMeansClusteringPanel | `kmeans_clustering_panel.py` | K-means clustering |

#### 12. Geometallurgy Panels

| Panel | File | Purpose |
|-------|------|---------|
| GeometPanel | `geomet_panel.py` | Geometallurgy |
| GeometDomainPanel | `geomet_domain_panel.py` | Geomet domains |
| GeometPlantPanel | `geomet_plant_panel.py` | Plant simulation |
| GeometChainPanel | `geomet_chain_panel.py` | Value chain |

#### 13. Grade Control Panels

| Panel | File | Purpose |
|-------|------|---------|
| GradeControlPanel | `grade_control_panel.py` | Grade control |
| GCDecisionPanel | `gc_decision_panel.py` | GC decisions |

#### 14. ESG/Risk Panels

| Panel | File | Purpose |
|-------|------|---------|
| ESGDashboardPanel | `esg_dashboard_panel.py` | ESG metrics |
| RiskTimelinePanel | `risk_timeline_panel.py` | Risk timeline |
| ScheduleRiskPanel | `schedule_risk_panel.py` | Schedule risk |

#### 15. Remote Sensing Panels

| Panel | File | Purpose |
|-------|------|---------|
| ScanPanel | `scan_panel.py` | 3D scan management |
| InSARPanel | `insar_panel.py` | InSAR analysis |
| SurveyDeformationPanel | `survey_deformation_panel.py` | Survey deformation |
| SeismicPanel | `seismic_panel.py` | Seismic data |

#### 16. Uncertainty/Research Panels

| Panel | File | Purpose |
|-------|------|---------|
| UncertaintyPanel | `uncertainty_panel.py` | Uncertainty quantification |
| UncertaintyPropagationPanel | `uncertainty_propagation_panel.py` | Uncertainty propagation |
| ResearchDashboardPanel | `research_dashboard_panel.py` | Research tools |
| ExperimentConfigPanel | `experiment_config_panel.py` | Experiment setup |
| ExperimentResultsPanel | `experiment_results_panel.py` | Experiment results |

#### 17. Planning/Dashboard Panels

| Panel | File | Purpose |
|-------|------|---------|
| PlanningDashboardPanel | `planning_dashboard_panel.py` | Planning overview |
| ProductionDashboardPanel | `production_dashboard_panel.py` | Production metrics |

#### 18. Administrative Panels

| Panel | File | Purpose |
|-------|------|---------|
| ProcessHistoryPanel | `process_history_panel.py` | Process audit trail |
| ReconciliationPanel | `reconciliation_panel.py` | Reconciliation |
| BlockInfoPanel | `block_info_panel.py` | Block information |
| PickInfoPanel | `pick_info_panel.py` | Pick information |

#### 19. Workflow/Design Panels

| Panel | File | Purpose |
|-------|------|---------|
| WorkflowWizard | `workflow_wizard.py` | Workflow automation |
| BenchDesignPanel | `bench_design_panel.py` | Bench design |
| BlockmodelBuilderPanel | `blockmodel_builder_panel.py` | Block model builder |
| DiglinePanel | `digline_panel.py` | Digline tools |

#### 20. Special Purpose Panels

| Panel | File | Purpose |
|-------|------|---------|
| InteractiveSlicer Panel | `interactive_slicer_panel.py` | Interactive slicing |
| SamplingControlsPanel | `sampling_controls_panel.py` | Sampling controls |
| TableViewerPanel | `table_viewer_panel.py` | Table viewer |
| QCWindow | `qc_window.py` | QA/QC tools |

### Panel Registration System

**File:** `panel_registry.py`

```python
class PanelCategory(Enum):
    """Panel categories for organization."""
    PROPERTY = "property"
    SCENE = "scene"
    LAYER = "layer"
    INFO = "info"
    ANALYSIS = "analysis"
    SELECTION = "selection"
    CROSS_SECTION = "cross_section"
    DISPLAY = "display"
    RESOURCE = "resource"
    PLANNING = "planning"

@dataclass
class PanelMetadata:
    """Metadata for a UI panel."""
    name: str
    category: PanelCategory
    icon_name: Optional[str] = None
    shortcut: Optional[str] = None
    default_dock_area: DockArea = DockArea.LEFT
    default_visible: bool = True
    factory: Optional[Callable[[], QWidget]] = None

class PanelRegistry:
    """Central registry for all UI panels."""
    
    def register_panel(self, panel_id: str, metadata: PanelMetadata):
        """Register a panel."""
        self._panels[panel_id] = metadata
    
    def get_panel_factory(self, panel_id: str):
        """Get factory function for panel."""
        return self._panels[panel_id].factory
```

**File:** `panel_registration.py`

```python
def register_all_panels(panel_manager):
    """Register all panels with the panel manager."""
    registry = get_panel_registry()
    
    # Register each panel
    registry.register_panel("property", PanelMetadata(
        name="Property",
        category=PanelCategory.PROPERTY,
        factory=lambda: PropertyPanel()
    ))
    # ... more registrations
```

### Panel Manager

**File:** `panel_manager.py`

```python
class PanelManager:
    """Manages panel lifecycle and docking."""
    
    def __init__(self, main_window):
        self.main_window = main_window
        self.panels = {}  # panel_id -> panel instance
        self.docks = {}   # panel_id -> QDockWidget
    
    def create_panel(self, panel_id: str):
        """Create and dock a panel."""
        registry = get_panel_registry()
        metadata = registry.get_metadata(panel_id)
        
        # Create panel using factory
        panel = metadata.factory()
        panel.set_controller(self.main_window.controller)
        
        # Create dock widget
        dock = PersistentDockWidget(metadata.name, self.main_window)
        dock.setWidget(panel)
        
        # Add to main window
        self.main_window.addDockWidget(metadata.default_dock_area, dock)
        
        # Store references
        self.panels[panel_id] = panel
        self.docks[panel_id] = dock
        
        return panel
```

### UI Supporting Files

| File | Purpose |
|------|---------|
| **`modern_styles.py`** | Modern UI styling and colors |
| **`modern_widgets.py`** | Custom modern widgets |
| **`collapsible_group.py`** | Collapsible group box widget |
| **`panel_header.py`** | Panel header widget |
| **`persistent_dock.py`** | Persistent dock widget |
| **`toast.py`** | Toast notification widget |
| **`splash_screen.py`** | Application splash screen |
| **`toolbar.py`** | Main toolbar |
| **`shortcuts.py`** | Keyboard shortcuts |
| **`signals.py`** | UI signal definitions |
| **`data_source_mixin.py`** | Data source selection mixin |
| **`data_source_selector.py`** | Data source selector widget |
| **`comparison_utils.py`** | Comparison utilities |

### UI Subdirectories

#### `ui/menus/`
Menu construction modules (see [Menu System Architecture](#menu-system-architecture))

#### `ui/dialogs/`
Dialog windows for specific interactions

#### `ui/actions/`
Action definitions for menu items and toolbar

#### `ui/bookmarks/`
Bookmark management for saved views

#### `ui/icons/`
UI icon assets

#### `ui/layout/`
Layout management tools

#### `ui/interaction/`
Interaction controllers

#### `ui/status/`
Status bar components

#### `ui/utils/`
UI utility functions

### Import Relationships (UI)

```
ui/
│
├── main_window.py
│   ├── imports: PyQt6.QtWidgets (QMainWindow, QDockWidget, etc.)
│   ├── imports: controllers.app_controller.AppController
│   ├── imports: visualization.renderer.Renderer
│   ├── imports: All panel classes
│   ├── imports: All menu builders
│   ├── imports: panel_manager.PanelManager
│   ├── imports: panel_registration.register_all_panels
│   └── imports: theme_manager.ThemeManager
│
├── base_panel.py
│   ├── imports: PyQt6.QtWidgets (QWidget, layouts, etc.)
│   ├── imports: PyQt6.QtCore (pyqtSignal, Qt)
│   └── imports: models.block_model.BlockModel
│
├── base_analysis_panel.py
│   ├── imports: base_panel.BaseDockPanel
│   └── imports: Additional analysis widgets
│
└── [specific_panel].py
    ├── imports: base_panel.BasePanel (or BaseAnalysisPanel)
    ├── imports: controllers.app_controller.AppController
    ├── imports: models/* (specific models needed)
    └── imports: PyQt6.QtWidgets (specific widgets)
```

---

## Visualization Layer

Directory: `block_model_viewer/visualization/`

### Visualization Architecture

The visualization layer is built on **PyVista** (VTK wrapper) and follows a **renderer-centric architecture**:

```
MainWindow
    │
    ├── Creates Renderer
    │
    └── Passes Renderer to AppController
            │
            └── AppController uses Renderer for all visualization
                    │
                    ├── Adds meshes
                    ├── Updates properties
                    ├── Manages layers
                    └── Controls camera
```

### Visualization Components

| File | Purpose |
|------|---------|
| **`renderer/render_orchestrator.py`** | Main Renderer class (public API) |
| **`renderer/actor_registry.py`** | Actor tracking and management |
| **`renderer/scene_bounds.py`** | Scene boundary management |
| **`renderer/pyvista_suppression.py`** | PyVista warning suppression |
| **`render_payloads.py`** | Data payload classes for rendering |
| **`block_model_mesh_builder.py`** | Build PyVista meshes from block models |
| **`drillhole_gpu_renderer.py`** | GPU-accelerated drillhole rendering |
| **`drillhole_state.py`** | Drillhole display state |
| **`color_mapper.py`** | Color mapping for properties |
| **`axis_manager.py`** | Axis and annotation management |
| **`overlay_manager.py`** | Overlay rendering |
| **`picking_controller.py`** | 3D picking interaction |
| **`cross_section.py`** | Cross-section generation |
| **`cross_section_adapter.py`** | Cross-section rendering adapter |
| **`grid_adapter.py`** | Grid rendering adapter |
| **`mesh_adapter.py`** | Mesh rendering adapter |
| **`pit_adapter.py`** | Pit shell rendering adapter |
| **`secondary_view_adapter.py`** | Secondary view handling |
| **`filters.py`** | Data filtering for visualization |
| **`decimation.py`** | Mesh decimation for performance |
| **`lod_manager.py`** | Level-of-detail management |
| **`visual_density_controller.py`** | Visual density optimization |
| **`scene_layer.py`** | Layer management |
| **`legend_manager.py`** | Legend rendering |
| **`gantt_chart.py`** | Gantt chart visualization |
| **`sankey_diagram.py`** | Sankey diagram visualization |
| **`gc_spider_chart.py`** | Spider chart for grade control |
| **`stope_visualizer.py`** | Stope visualization |
| **`renderer_voxels.py`** | Voxel rendering |
| **`pyvista_axes_scalebar.py`** | PyVista axes and scale bar |
| **`_batch_update.py`** | Batch rendering updates |

#### `visualization/hud/`
Heads-up display components

#### `visualization/renderer/`
Core renderer implementation

**File:** `renderer/__init__.py`
```python
"""
renderer package — 3D rendering engine for GeoX.

The Renderer class is the public API.
"""
from .render_orchestrator import Renderer

__all__ = ['Renderer']
```

### Render Payloads

**File:** `render_payloads.py`

Defines data classes for passing rendering information:

```python
@dataclass
class MeshPayload:
    """Payload for mesh rendering."""
    mesh: pv.PolyData
    name: str
    scalars: Optional[np.ndarray] = None
    cmap: str = "viridis"
    opacity: float = 1.0
    show_edges: bool = False

@dataclass
class GridPayload:
    """Payload for grid/block model rendering."""
    positions: np.ndarray
    dimensions: np.ndarray
    scalars: np.ndarray
    name: str
    cmap: str = "viridis"

@dataclass
class CrossSectionPayload:
    """Payload for cross-section rendering."""
    origin: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    # ... more fields

# More payload classes...
```

### Import Relationships (Visualization)

```
visualization/
│
├── renderer/
│   ├── __init__.py
│   │   └── imports: render_orchestrator.Renderer
│   │
│   ├── render_orchestrator.py
│   │   ├── imports: pyvista
│   │   ├── imports: actor_registry.ActorRegistry
│   │   ├── imports: scene_bounds.SceneBounds
│   │   └── imports: render_payloads.*
│   │
│   └── actor_registry.py
│       └── imports: pyvista
│
├── render_payloads.py
│   ├── imports: dataclasses
│   ├── imports: numpy
│   └── imports: pyvista
│
├── block_model_mesh_builder.py
│   ├── imports: pyvista
│   ├── imports: numpy
│   ├── imports: models.block_model.BlockModel
│   └── imports: color_mapper.ColorMapper
│
└── color_mapper.py
    ├── imports: numpy
    └── imports: matplotlib.cm (colormaps)
```

---

## Domain-Specific Modules

### Geostatistics

Directory: `block_model_viewer/geostats/`

| File | Purpose |
|------|---------|
| **`geostats_utils.py`** | Utility functions |
| **`determinism.py`** | Deterministic controls |
| **`variogram_model.py`** | Variogram modeling |
| **`variogram_gates.py`** | Variogram gate constraints |
| **`variogram_assistant.py`** | Variogram fitting assistant |
| **`anisotropy_utils.py`** | Anisotropy calculations |
| **`cokriging3d.py`** | Co-kriging implementation |
| **`cosgsim3d.py`** | Co-SGSIM implementation |
| **`bayesian_kriging.py`** | Bayesian kriging |
| **`universal_kriging.py`** | Universal kriging |
| **`indicator_kriging.py`** | Indicator kriging |
| **`ik_sgsim.py`** | IK-SGSIM |
| **`sis.py`** | Sequential indicator simulation |
| **`turning_bands.py`** | Turning bands simulation |
| **`mps.py`** | Multiple-point statistics |
| **`grf.py`** | Gaussian random fields |
| **`rbf_interpolation.py`** | Radial basis function interpolation |
| **`direct_block_sim.py`** | Direct block simulation |
| **`soft_data.py`** | Soft data integration |
| **`simulation_interface.py`** | Simulation interface |
| **`reconciliation.py`** | Reconciliation tools |
| **`kriging_job_params.py`** | Kriging job parameters |
| **`uk_validation_utils.py`** | UK validation |
| **`ik_audit_utils.py`** | IK audit utilities |
| **`sk_cross_validation.py`** | SK cross-validation |
| **`sk_debugger.py`** | SK debugging tools |
| **`sk_stationarity.py`** | SK stationarity checks |

### Drillholes

Directory: `block_model_viewer/drillholes/`

| File | Purpose |
|------|---------|
| **`datamodel.py`** | Drillhole data model |
| **`database.py`** | Drillhole database |
| **`data_io.py`** | Import/export |
| **`compositing_engine.py`** | Compositing algorithms |
| **`compositing_ui_engines.py`** | Compositing UI integration |
| **`compositing_utils.py`** | Compositing utilities |
| **`declustering.py`** | Statistical declustering |
| **`plotting.py`** | Plotting utilities |
| **`reporting.py`** | Report generation |
| **`drillhole_layer.py`** | Layer management |
| **`drillhole_validation.py`** | Data validation |
| **`drillhole_autofix.py`** | Auto-fix validation errors |
| **`drill hole_ignore.py`** | Ignore flagged data |
| **`drillhole_manual_edit.py`** | Manual editing |
| **`drillhole_audit_trail.py`** | Audit trail |
| **`audit_trail.py`** | Generic audit trail |
| **`audit_export.py`** | Audit export |
| **`approval_workflow.py`** | Approval workflow |
| **`control_samples.py`** | Control samples |
| **`backup_recovery.py`** | Backup and recovery |
| **`data_migration.py`** | Data migration |
| **`performance.py`** | Performance optimizations |
| **`security.py`** | Security features |
| **`user_auth.py`** | User authentication |
| **`documentation.py`** | Documentation generation |
| **`templates.py`** | Report templates |
| **`system_config.py`** | System configuration |
| **`registry_utils.py`** | Registry utilities |

### Geology

Directory: `block_model_viewer/geology/`

| File | Purpose |
|------|---------|
| **`faults.py`** | Fault modeling |
| **`fault_detection.py`** | Fault detection algorithms |
| **`industry_modeler.py`** | Industry-standard geological modeler |
| **`chronos_engine.py`** | Chronological modeling |
| **`compliance_manager.py`** | Compliance checking |
| **`gradient_estimator.py`** | Gradient estimation |
| **`mesh_validator.py`** | Mesh validation |
| **`model_runner.py`** | Model execution |
| **`contact_deviation_report.py`** | Contact deviation reporting |
| **`coordinate_diagnostic.py`** | Coordinate diagnostics |

### IRR Engine (Investment Analysis)

Directory: `block_model_viewer/irr_engine/`

| File | Purpose |
|------|---------|
| **`__init__.py`** | Public API exports |
| **`engine_api.py`** | Main engine API |
| **`irr_bisection.py`** | IRR calculation (bisection method) |
| **`npv_calc.py`** | NPV calculation |
| **`lerchs_grossmann.py`** | Lerchs-Grossmann pit optimization |
| **`milp_optimizer.py`** | MILP optimizer |
| **`dynamic_shell_selector.py`** | Dynamic shell selection |
| **`fast_scheduler.py`** | Fast scheduling |
| **`pit_phases.py`** | Pit phase design |
| **`scenario_generator.py`** | Scenario generation |
| **`config_loader.py`** | Configuration loading |
| **`results_model.py`** | Results data model |
| **`provenance.py`** | Provenance tracking |
| **`validation.py`** | Input validation |

### Uncertainty Engine

Directory: `block_model_viewer/uncertainty_engine/`

| File | Purpose |
|------|---------|
| **`monte_carlo.py`** | Monte Carlo simulation |
| **`bootstrap.py`** | Bootstrap methods |
| **`lhs_sampler.py`** | Latin Hypercube Sampling |
| **`grade_realisations.py`** | Grade realizations |
| **`economic_propagation.py`** | Economic uncertainty propagation |
| **`prob_shells.py`** | Probabilistic pit shells |
| **`dashboard_generator.py`** | Uncertainty dashboards |
| **`examples.py`** | Example workflows |

### Mine Planning

Directory: `block_model_viewer/mine_planning/`

Subdirectories:
- **`cutoff/`** - Cutoff grade optimization
- **`npvs/`** - NPV scenarios
- **`pushbacks/`** - Pushback design
- **`scheduling/`** - Scheduling algorithms
- **`ug/`** - Underground planning

### Geotech

Directory: `block_model_viewer/geotech/`

| File | Purpose |
|------|---------|
| **`dataclasses.py`** | Geotech data structures |
| **`rock_mass_model.py`** | Rock mass characterization |
| **`slope_risk.py`** | Slope risk assessment |
| **`stope_stability.py`** | Stope stability analysis |
| **`probabilistic_geotech.py`** | Probabilistic geotech |
| **`interpolation.py`** | Geotech interpolation |

### Additional Domain Modules

- **`esg/`** - ESG (Environmental, Social, Governance) metrics
- **`geomet/`** - Geometallurgy
- **`geomet_chain/`** - Geomet value chain
- **`geotech_common/`** - Common geotech utilities
- **`geotech_pit/`** - Pit-specific geotech
- **`grade_control/`** - Grade control
- **`haulage/`** - Haulage optimization
- **`scans/`** - 3D scan processing
- **`seismic/`** - Seismic data
- **`structural/`** - Structural geology
- **`survey_deformation/`** - Survey deformation
- **`ug/`** - Underground mining
- **`risk/`** - Risk assessment
- **`research/`** - Research tools
- **`planning/`** - Mine planning
- **`reconciliation/`** - Production reconciliation

---

## Utility Modules

Directory: `block_model_viewer/utils/`

| File | Purpose |
|------|---------|
| **`coordinate_manager.py`** | Coordinate system management |
| **`coordinate_utils.py`** | Coordinate utilities |
| **`desurvey.py`** | Drillhole desurvey algorithms |
| **`selection_manager.py`** | Selection state management |
| **`cross_section_manager.py`** | Cross-section management |
| **`screenshot_manager.py`** | Screenshot utilities |
| **`export_helpers.py`** | Export utilities |
| **`plotting_helpers.py`** | Plotting utilities |
| **`statistics_helpers.py`** | Statistical utilities |
| **`variable_utils.py`** | Variable manipulation |
| **`variogram_functions.py`** | Variogram function library |
| **`data_bridge.py`** | Data bridging between formats |
| **`chunk_loader.py`** | Chunked data loading |
| **`profiling.py`** | Performance profiling |
| **`session_logger.py`** | Session logging |
| **`audit_logging.py`** | Audit logging |
| **`security.py`** | Security utilities |

---

## Parsers

Directory: `block_model_viewer/parsers/`

| File | Purpose |
|------|---------|
| **`base_parser.py`** | Base parser class |
| **`csv_parser.py`** | CSV file parser |
| **`vtk_parser.py`** | VTK file parser |
| **`mesh_parser.py`** | Mesh file parser |
| **`mining_parser.py`** | Mining software format parser |
| **`structural_csv_parser.py`** | Structural data CSV parser |

**Parser Pattern:**
```python
class BaseParser:
    """Base class for all parsers."""
    
    def parse(self, file_path: Path) -> Any:
        raise NotImplementedError()
    
    def validate(self, data: Any) -> bool:
        raise NotImplementedError()

class CSVParser(BaseParser):
    """CSV file parser."""
    
    def parse(self, file_path: Path) -> pd.DataFrame:
        # Parse CSV and return DataFrame
        pass
```

---

## Import Relationships

### High-Level Import Flow

```
Entry Point (__main__.py, main.py)
    │
    ├─→ ui.main_window.MainWindow
    │       │
    │       ├─→ visualization.renderer.Renderer
    │       ├─→ controllers.app_controller.AppController
    │       ├─→ ui.panel_*.py (all panels)
    │       ├─→ ui.menus.*.py (all menus)
    │       └─→ ui.theme_manager.ThemeManager
    │
    ├─→ controllers.app_controller.AppController
    │       │
    │       ├─→ controllers.geostats_controller.GeostatsController
    │       ├─→ controllers.mining_controller.MiningController
    │       ├─→ controllers.vis_controller.VisController
    │       ├─→ controllers.data_controller.DataController
    │       └─→ Other sub-controllers
    │
    └─→ config.Config
```

### Panel Import Pattern

Most panels follow this pattern:

```python
# Panel file: ui/some_panel.py

from PyQt6.QtWidgets import (widgets...)
from PyQt6.QtCore import (core...)

from .base_panel import BasePanel  # or BaseAnalysisPanel
from ..models.block_model import BlockModel
from ..controllers.app_controller import AppController

class SomePanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller: Optional[AppController] = None
    
    def set_controller(self, controller: AppController):
        self.controller = controller
        self.controller.signals.data_loaded.connect(self.on_data_loaded)
```

### Controller Import Pattern

```python
# Controller file: controllers/some_controller.py

import numpy as np
import pandas as pd

from ..models.some_model import SomeModel
from ..geostats.some_algorithm import some_function
from ..visualization.render_payloads import SomePayload

class SomeController:
    def __init__(self, app_controller):
        self.app = app_controller
        self.renderer = app_controller.renderer
```

### Model Import Pattern

```python
# Model file: models/some_model.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SomeModel:
    """Data model for..."""
    data: pd.DataFrame
    metadata: Dict[str, Any]
```

---

## Data Flow Architecture

### Complete Data Flow

```
1. DATA IMPORT
   User Action → Menu → MainWindow → DataController → Parser → DataFrame
        │
        └─→ DataRegistry (register data source)
        
2. DATA PROCESSING
   Panel UI → Controller → Model/Engine → Result
        │
        ├─→ BlockModel construction
        ├─→ Geostatistics (kriging, simulation)
        ├─→ Resource classification
        └─→ Mine planning
        
3. VISUALIZATION
   Controller → RenderPayload → Renderer → PyVista → Screen
        │
        └─→ ActorRegistry (track actors)
        
4. RESULTS EXPORT
   Panel → Controller → Export Helper → File
        │
        └─→ Audit Trail (log export)
```

### Example: Kriging Workflow Data Flow

```
1. User opens KrigingPanel
   └─→ KrigingPanel.setup_ui() creates UI

2. User selects data source
   └─→ DataSourceSelector → DataRegistry → Available datasets

3. User configures kriging parameters
   └─→ UI widgets → Panel state

4. User clicks "Run Kriging"
   └─→ KrigingPanel.on_run_clicked()
       └─→ controller.geostats.run_kriging(params)
           └─→ GeostatsController.run_kriging()
               └─→ JobWorker (background thread)
                   └─→ models.kriging_engine.KrigingEngine.run()
                       └─→ models.kriging3d (core algorithm)
                           └─→ Return KrigingResults

5. Kriging complete
   └─→ JobWorker emits signal
       └─→ GeostatsController receives result
           └─→ Store in DataRegistry
           └─→ Emit controller signal
               └─→ KrigingPanel updates UI
               └─→ VisController auto-visualize
                   └─→ Renderer.add_grid(GridPayload(...))
                       └─→ PyVista rendering

6. User exports results
   └─→ KrigingPanel → Export menu
       └─→ ExportHelper.export_to_csv()
           └─→ AuditManager.log_export()
```

### Signal Flow

```
AppController Signals:
├─→ data_loaded → All panels refresh
├─→ property_changed → Update visualization
├─→ filter_applied → Re-render filtered data
├─→ analysis_complete → Update result panels
└─→ selection_changed → Update selection displays

Panel Signals:
├─→ status_message → MainWindow status bar
├─→ error_occurred → MainWindow error dialog
└─→ request_action → Controller handles action
```

---

## Panel System Architecture

### Panel Lifecycle

```
1. REGISTRATION
   panel_registration.py → register_all_panels()
        │
        └─→ PanelRegistry.register_panel(id, metadata)

2. CREATION (when user opens panel from menu)
   Menu Action → MainWindow.show_panel(panel_id)
        │
        └─→ PanelManager.create_panel(panel_id)
            ├─→ Get factory from registry
            ├─→ panel = factory()
            ├─→ panel.set_controller(controller)
            ├─→ dock = PersistentDockWidget()
            ├─→ dock.setWidget(panel)
            └─→ MainWindow.addDockWidget(dock)

3. INITIALIZATION
   Panel.__init__()
        │
        ├─→ setup_ui() [subclass implements]
        └─→ connect_signals() [subclass implements]

4. CONTROLLER INJECTION
   panel.set_controller(controller)
        │
        └─→ Connect to controller signals
            ├─→ controller.signals.data_loaded.connect(...)
            └─→ controller.signals.property_changed.connect(...)

5. USER INTERACTION
   User interacts with panel
        │
        └─→ Panel calls controller methods
            └─→ controller.geostats.run_something(...)

6. REFRESH
   Controller emits signal
        │
        └─→ Panel.refresh() or specific slot
            └─→ Update UI from latest data

7. DESTRUCTION (when user closes panel)
   User closes dock → PersistentDockWidget stores state
        │
        └─→ Panel.deleteLater()
```

### Panel-Controller Communication Pattern

**Panel Side:**
```python
class MyPanel(BaseAnalysisPanel):
    def setup_ui(self):
        # Build UI
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.on_run)
    
    def connect_signals(self):
        # Connect to controller signals
        if self.controller:
            self.controller.signals.analysis_complete.connect(
                self.on_analysis_complete
            )
    
    def on_run(self):
        # Call controller
        params = self.get_parameters()
        self.controller.geostats.run_analysis(params)
    
    def on_analysis_complete(self, results):
        # Update UI with results
        self.display_results(results)
```

**Controller Side:**
```python
class GeostatsController:
    def run_analysis(self, params):
        # Start background job
        job = JobWorker(self._do_analysis, params)
        job.finished.connect(self._on_analysis_finished)
        job.start()
    
    def _do_analysis(self, params):
        # Heavy computation
        result = some_algorithm(params)
        return result
    
    def _on_analysis_finished(self, result):
        # Store result
        self.app.data_registry.register("analysis_result", result)
        
        # Emit signal
        self.app.signals.analysis_complete.emit(result)
```

---

## Menu System Architecture

Directory: `block_model_viewer/ui/menus/`

### Menu Module Pattern

Each menu module exports a single build function:

**File:** `menus/some_menu.py`
```python
def build_some_menu(main_window, menubar):
    """Build Some Menu."""
    menu = menubar.addMenu("&Some")
    
    # Add actions
    action = QAction("Do Something", main_window)
    action.triggered.connect(main_window.on_do_something)
    menu.addAction(action)
    
    # Add more actions...
    
    return menu
```

### Menu Modules

| Module | Purpose |
|--------|---------|
| **`file_menu.py`** | File operations (Open, Save, Import, Export) |
| **`edit_menu.py`** | Edit operations (Undo, Redo, Preferences) |
| **`view_menu.py`** | View controls (Camera, Display, Layout) |
| **`search_menu.py`** | Search functionality |
| **`data_menu.py`** | Data management |
| **`drillholes_menu.py`** | Drillhole operations |
| **`geology_menu.py`** | Geological modeling |
| **`resources_menu.py`** | Resource estimation |
| **`estimations_menu.py`** | Estimation methods |
| **`geotech_menu.py`** | Geotechnical analysis |
| **`mine_planning_menu.py`** | Mine planning tools |
| **`ml_menu.py`** | Machine learning |
| **`dashboards_menu.py`** | Dashboard views |
| **`panels_menu.py`** | Panel visibility controls |
| **`mouse_menu.py`** | Mouse interaction modes |
| **`tools_menu.py`** | Utility tools |
| **`layout_menu.py`** | Layout management |
| **`survey_menu.py`** | Survey data |
| **`scan_menu.py`** | 3D scans |
| **`remote_sensing_menu.py`** | Remote sensing (InSAR) |
| **`workbench_menu.py`** | Workbench features |
| **`workflows_menu.py`** | Workflow automation |
| **`help_menu.py`** | Help and about |

### Menu Build Process

From `main_window.py`:

```python
def _build_menus(self):
    """Build menu bar."""
    menubar = self.menuBar()
    
    # Import all menu builders
    from .menus import (
        build_file_menu,
        build_edit_menu,
        build_view_menu,
        # ... all menu builders
    )
    
    # Build menus
    self.file_menu = build_file_menu(self, menubar)
    self.edit_menu = build_edit_menu(self, menubar)
    self.view_menu = build_view_menu(self, menubar)
    # ... build all menus
```

### Menu-to-Panel Connection

Menus typically show/hide panels:

**In menu builder:**
```python
def build_panels_menu(main_window, menubar):
    menu = menubar.addMenu("&Panels")
    
    # Get panel registry
    registry = get_panel_registry()
    
    # Create action for each panel
    for panel_id, metadata in registry.get_all():
        action = QAction(metadata.name, main_window)
        action.setCheckable(True)
        action.triggered.connect(
            lambda checked, pid=panel_id: 
                main_window.toggle_panel(pid, checked)
        )
        
        if metadata.shortcut:
            action.setShortcut(metadata.shortcut)
        
        menu.addAction(action)
    
    return menu
```

**In MainWindow:**
```python
def toggle_panel(self, panel_id, show):
    """Show or hide a panel."""
    if show:
        if panel_id not in self.panel_manager.panels:
            # Create panel if not exists
            panel = self.panel_manager.create_panel(panel_id)
        else:
            # Show existing panel
            dock = self.panel_manager.docks[panel_id]
            dock.show()
    else:
        # Hide panel
        dock = self.panel_manager.docks[panel_id]
        dock.hide()
```

---

## Theme System Architecture

### Theme Manager

**File:** `ui/theme_manager.py`

```python
class ThemeManager(QObject):
    """Manages application themes and color palettes."""
    
    theme_changed = pyqtSignal(str)
    
    def __init__(self, app=None):
        super().__init__()
        self.app = app
        self._current_theme = "light"
        self._color_palette = {}
        
        self._load_color_palette()
    
    def load_theme(self, name: str):
        """Load a theme by name ('light' or 'dark')."""
        self._current_theme = name
        
        # Load QSS stylesheet
        theme_file = self._themes_dir / f"{name}.qss"
        if theme_file.exists():
            with open(theme_file, 'r') as f:
                stylesheet = f.read()
            
            if self.app:
                self.app.setStyleSheet(stylesheet)
        
        self.theme_changed.emit(name)
    
    def get_color_palette(self, category: str) -> List[str]:
        """Get color palette for a category."""
        return self._color_palette.get(category, [])
```

### Theme Application

Themes are applied in layers:

1. **Application-wide QSS**
   - Loaded via `QApplication.setStyleSheet()`
   - Defined in `assets/themes/light.qss` or `dark.qss`

2. **Panel-specific styles**
   - Applied in panel `setup_ui()` methods
   - Uses `modern_styles.py` helper functions

3. **Widget-specific styles**
   - Custom widgets in `modern_widgets.py`
   - Apply theme-aware styling

### Theme Files Location

```
block_model_viewer/
├── assets/
│   ├── themes/
│   │   ├── light.qss
│   │   ├── dark.qss
│   │   └── color_palette.json
│   └── branding/
│       └── (logos, icons)
└── ui/
    ├── theme_manager.py
    ├── modern_styles.py
    └── modern_widgets.py
```

### Modern Styles

**File:** `ui/modern_styles.py`

```python
@dataclass
class ModernColors:
    """Modern color palette."""
    primary: str = "#2196F3"
    secondary: str = "#FFC107"
    background: str = "#FFFFFF"
    surface: str = "#F5F5F5"
    text: str = "#212121"
    # ... more colors

def get_theme_colors(theme: str = "light") -> ModernColors:
    """Get colors for theme."""
    if theme == "dark":
        return ModernColors(
            primary="#1976D2",
            background="#121212",
            # ... dark colors
        )
    else:
        return ModernColors()  # Light colors

def apply_modern_style(widget, theme="light"):
    """Apply modern styling to a widget."""
    colors = get_theme_colors(theme)
    style = f"""
    QWidget {{
        background-color: {colors.background};
        color: {colors.text};
    }}
    QPushButton {{
        background-color: {colors.primary};
        color: white;
        border-radius: 4px;
        padding: 8px 16px;
    }}
    """
    widget.setStyleSheet(style)
```

### Theme Propagation

When theme changes:

```
1. User selects theme
   └─→ MainWindow → ThemeManager.load_theme(name)

2. ThemeManager loads QSS
   └─→ QApplication.setStyleSheet(qss)

3. ThemeManager emits signal
   └─→ theme_changed.emit(name)

4. Panels listen to signal
   └─→ Panel.on_theme_changed(name)
       └─→ Panel refreshes theme-dependent elements
           ├─→ Update chart colors
           ├─→ Update widget styles
           └─→ Re-render visualizations if needed
```

---

## Patterns and Best Practices

### 1. Dependency Injection Pattern

All panels receive controller via dependency injection:

```python
class MyPanel(BasePanel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.controller = None  # Set later
    
    def set_controller(self, controller):
        """Inject controller dependency."""
        self.controller = controller
        self.connect_signals()  # Now we can connect
```

**Benefits:**
- Loose coupling
- Easier testing (can inject mock controller)
- Clear dependencies

### 2. Template Method Pattern

BasePanel uses template method pattern:

```python
class BasePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()  # Template method - subclass implements
        self.connect_signals()  # Template method - subclass implements
    
    def setup_ui(self):
        raise NotImplementedError()  # Force override
    
    def connect_signals(self):
        pass  # Optional override
```

### 3. Registry Pattern

Panels, menus, and other components use registries:

```python
# Singleton registry
class PanelRegistry:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register_panel(self, panel_id, metadata):
        self._panels[panel_id] = metadata

# Usage
registry = get_panel_registry()  # Always returns same instance
registry.register_panel("my_panel", metadata)
```

### 4. Command Pattern (via Signals)

Commands are encapsulated as signals:

```python
# Controller defines command signals
class ControllerSignals(QObject):
    run_analysis = pyqtSignal(dict)  # Command with parameters
    cancel_analysis = pyqtSignal()   # Command without parameters

# Panel emits command
self.controller.signals.run_analysis.emit({
    'method': 'kriging',
    'params': params
})

# Controller handles command
self.signals.run_analysis.connect(self._handle_run_analysis)
```

### 5. Observer Pattern (via Qt Signals)

Components observe state changes via signals:

```python
# Subject (Controller)
class AppController:
    def __init__(self):
        self.signals = ControllerSignals()
    
    def load_data(self, data):
        self._data = data
        self.signals.data_loaded.emit(data)  # Notify observers

# Observers (Panels)
class Panel1(BasePanel):
    def connect_signals(self):
        self.controller.signals.data_loaded.connect(self.on_data_loaded)
    
    def on_data_loaded(self, data):
        # React to data change
        self.refresh()

class Panel2(BasePanel):
    def connect_signals(self):
        self.controller.signals.data_loaded.connect(self.on_data_loaded)
    
    def on_data_loaded(self, data):
        # React to data change (different behavior)
        self.update_list()
```

### 6. Factory Pattern

Panel creation uses factories:

```python
# Factory function
def create_kriging_panel():
    return KrigingPanel()

# Registry stores factories
registry.register_panel("kriging", PanelMetadata(
    name="Kriging",
    factory=create_kriging_panel
))

# Usage
panel = registry.get_panel_factory("kriging")()
```

### 7. Adapter Pattern

Visualization adapters adapt data for rendering:

```python
class GridAdapter:
    """Adapts BlockModel to PyVista grid."""
    
    @staticmethod
    def adapt(block_model: BlockModel) -> pv.UnstructuredGrid:
        # Convert BlockModel to PyVista format
        positions = block_model.positions
        dimensions = block_model.dimensions
        
        # Create PyVista grid
        grid = create_uniform_grid(positions, dimensions)
        return grid

# Usage
adapter = GridAdapter()
grid = adapter.adapt(block_model)
renderer.add_mesh(grid)
```

### 8. Strategy Pattern

Different algorithms implement common interface:

```python
class KrigingStrategy(ABC):
    @abstractmethod
    def estimate(self, data, params):
        pass

class SimpleKriging(KrigingStrategy):
    def estimate(self, data, params):
        # Simple kriging algorithm
        pass

class OrdinaryKriging(KrigingStrategy):
    def estimate(self, data, params):
        # Ordinary kriging algorithm
        pass

# Usage
strategy = SimpleKriging() if params['type'] == 'simple' else OrdinaryKriging()
result = strategy.estimate(data, params)
```

### 9. Builder Pattern

Complex objects use builders:

```python
class BlockModelBuilder:
    """Builder for BlockModel."""
    
    def __init__(self):
        self._model = BlockModel()
    
    def with_positions(self, positions):
        self._model.positions = positions
        return self
    
    def with_dimensions(self, dimensions):
        self._model.dimensions = dimensions
        return self
    
    def with_property(self, name, values):
        self._model.data[name] = values
        return self
    
    def build(self):
        return self._model

# Usage
model = (BlockModelBuilder()
    .with_positions(positions)
    .with_dimensions(dimensions)
    .with_property("grade", grades)
    .build())
```

### 10. Decorator Pattern

Audit logging decorates functions:

```python
def audit_log(operation: str):
    """Decorator to log operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Log before
            audit_manager.log_start(operation, args, kwargs)
            
            try:
                result = func(*args, **kwargs)
                # Log success
                audit_manager.log_success(operation, result)
                return result
            except Exception as e:
                # Log error
                audit_manager.log_error(operation, e)
                raise
        return wrapper
    return decorator

# Usage
@audit_log("import_drillhole_data")
def import_drillholes(file_path):
    # Import logic
    pass
```

### 11. Lazy Loading Pattern

Panels and modules loaded on-demand:

```python
class PanelManager:
    def create_panel(self, panel_id):
        # Only create panel when first requested
        if panel_id not in self.panels:
            factory = self.registry.get_panel_factory(panel_id)
            panel = factory()
            self.panels[panel_id] = panel
        
        return self.panels[panel_id]
```

### 12. Error Handling Pattern

Consistent error handling across application:

```python
class BasePanel(QWidget):
    error_occurred = pyqtSignal(str)
    
    def handle_error(self, error: Exception):
        """Handle errors consistently."""
        error_msg = str(error)
        logger.error(f"Error in {self.panel_id}: {error_msg}", exc_info=True)
        self.error_occurred.emit(error_msg)
        QMessageBox.critical(self, "Error", error_msg)

# Usage in panels
try:
    result = self.controller.do_something()
except Exception as e:
    self.handle_error(e)
```

---

## Key Architectural Decisions

### 1. Why Separate Controllers from UI?

**Decision:** Controllers are separate from panels (not embedded in MainWindow)

**Rationale:**
- **Testability**: Controllers can be tested without GUI
- **Reusability**: Same controller logic for different UIs (CLI, web, etc.)
- **Maintainability**: Business logic separate from presentation
- **Scalability**: Multiple panels can share one controller

### 2. Why Sub-Controllers?

**Decision:** AppController delegates to domain-specific sub-controllers

**Rationale:**
- **Single Responsibility**: Each controller has one domain
- **Manageable Size**: Prevents AppController from becoming a "god object"
- **Team Parallelization**: Different developers can work on different controllers
- **Clear Boundaries**: Domain boundaries are explicit

### 3. Why Panel Registry?

**Decision:** Central panel registry instead of hardcoded panel references

**Rationale:**
- **Discoverability**: All panels in one place
- **Metadata**: Store panel metadata (icons, shortcuts, categories)
- **Dynamic Loading**: Panels created on-demand
- **Extensibility**: Easy to add new panels via plugin system

### 4. Why Menu Modules?

**Decision:** Each menu in separate module instead of all in MainWindow

**Rationale:**
- **Maintainability**: Small, focused files
- **Readability**: Easy to find menu-related code
- **Team Parallelization**: Multiple developers can work on different menus
- **Code Organization**: Clear structure

### 5. Why PyVista/VTK?

**Decision:** PyVista (VTK wrapper) for 3D visualization

**Rationale:**
- **Industry Standard**: VTK is the standard for scientific visualization
- **Performance**: Hardware-accelerated rendering
- **Features**: Rich visualization capabilities (volume rendering, picking, etc.)
- **Python Integration**: PyVista provides clean Python API

### 6. Why DataRegistry?

**Decision:** Central data registry for all loaded data

**Rationale:**
- **Single Source of Truth**: All panels see the same data
- **Memory Management**: Avoid duplicating large datasets
- **Provenance Tracking**: Track data lineage
- **State Management**: Save/restore application state

### 7. Why Background Workers?

**Decision:** Heavy computations run in background threads via JobWorker

**Rationale:**
- **Responsiveness**: UI remains responsive during computation
- **Cancellation**: Users can cancel long-running operations
- **Progress**: Show progress indicators
- **Stability**: Isolated errors don't crash UI

### 8. Why Payload Classes?

**Decision:** RenderPayload dataclasses for renderer communication

**Rationale:**
- **Type Safety**: Clear contract for rendering operations
- **Extensibility**: Easy to add new payload types
- **Documentation**: Self-documenting via dataclass fields
- **Validation**: Can validate payloads before rendering

---

## File Count Summary

| Category | Approximate File Count |
|----------|----------------------|
| UI Panels | ~120 files |
| Menus | ~24 files |
| Controllers | ~14 files |
| Models | ~25 files |
| Visualization | ~25 files |
| Geostats | ~30 files |
| Drillholes | ~28 files |
| Core | ~13 files |
| Utilities | ~18 files |
| Parsers | ~6 files |
| Domain Modules | ~100+ files across geology, geotech, mine_planning, etc. |
| **Total** | **~400+ Python files** |

---

## Critical Files Reference

For quick navigation, these are the most critical files:

### Essential Entry Points
1. `__main__.py` - Application entry
2. `main.py` - Initialization
3. `ui/main_window.py` - Main window

### Core Architecture
4. `controllers/app_controller.py` - Main controller
5. `ui/base_panel.py` - Panel base class
6. `ui/panel_registry.py` - Panel registry
7. `core/data_registry.py` - Data registry

### Visualization
8. `visualization/renderer/render_orchestrator.py` - Renderer
9. `visualization/render_payloads.py` - Render payloads

### Models
10. `models/block_model.py` - Block model
11. `models/kriging_engine.py` - Kriging engine
12. `models/sgsim_engine.py` - SGSIM engine

### Configuration
13. `config.py` - Configuration management

---

## Navigation Tips

### Finding Panel Implementation
```
ui/[panel_name]_panel.py
```

### Finding Menu Implementation
```
ui/menus/[menu_name]_menu.py
```

### Finding Controller for Domain
```
controllers/[domain]_controller.py
```

### Finding Model/Algorithm
```
models/[algorithm].py  OR  geostats/[algorithm].py
```

### Finding Parser
```
parsers/[format]_parser.py
```

---

## Conclusion

GeoX follows a **clean, layered architecture** with clear separation of concerns:

- **UI Layer**: Panels provide user interaction
- **Controller Layer**: Controllers orchestrate business logic
- **Model Layer**: Models encapsulate data and algorithms
- **Visualization Layer**: Renderer handles 3D graphics
- **Domain Layer**: Specialized modules for geostatistics, mining, geology, etc.

This architecture provides:
- ✅ **Maintainability**: Clear structure, small focused files
- ✅ **Testability**: Layers can be tested independently
- ✅ **Scalability**: Easy to add new features
- ✅ **Team Collaboration**: Multiple developers can work in parallel
- ✅ **Extensibility**: Plugin-style panel system

The codebase is professionally organized, following industry-standard design patterns and best practices for large-scale Qt applications.

---

**Document Version:** 1.0  
**Last Updated:** February 21, 2026  
**Maintained By:** GeoX Development Team
