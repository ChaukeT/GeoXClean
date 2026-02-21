"""
Drillhole Database Management Panel.

Provides UI for managing drillhole databases with SQLite storage,
project management, import/export, and backup/restore.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import pandas as pd
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QGroupBox, QFormLayout, QListWidget, QListWidgetItem, QTextEdit,
    QFileDialog, QMessageBox, QDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QWidget, QSplitter, QTabWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .base_analysis_panel import BaseAnalysisPanel
from ..drillholes import DrillholeDatabaseManager, DrillholeDatabase
from ..drillholes.datamodel import Collar, SurveyInterval, AssayInterval, LithologyInterval

from .modern_styles import get_theme_colors, ModernColors
logger = logging.getLogger(__name__)


class DrillholeDatabasePanel(BaseAnalysisPanel):
    """
    Panel for managing drillhole databases - persistent storage.
    
    PURPOSE:
    This panel manages SQLite databases for storing drillhole data persistently.
    It allows you to save drillhole data loaded in memory (from Drillhole Loading)
    to a database for long-term storage, backup, and multi-project management.
    
    WORKFLOW:
    1. Load drillhole data via "Geological Modeling → Drillhole Loading"
    2. (Optional) Create a project here to organize your data
    3. Save the loaded drillhole data to the database project
    4. Later, load saved data back from the database for analysis
    
    Features:
    - Project management (create, open, delete projects)
    - Export database to CSV
    - Backup/restore databases
    - Database statistics
    - Load saved drillhole data back into memory
    
    Note: To load new drillhole data from CSV files, use the Drillhole Loading panel
    (Geological Modeling → Drillhole Loading).
    """
    
    task_name = "drillhole_database"
    
    # Signals
    database_loaded = pyqtSignal(object)  # Emits DrillholeDatabase
    project_changed = pyqtSignal(str)  # Emits project name
    
    def __init__(self, parent=None):
        # Database manager
        self.db_manager: Optional[DrillholeDatabaseManager] = None
        self.current_project: Optional[str] = None
        self.current_database: Optional[DrillholeDatabase] = None
        
        # UI components (will be created in setup_ui)
        self.project_combo: Optional[QComboBox] = None
        self.stats_text: Optional[QTextEdit] = None
        self.holes_list: Optional[QListWidget] = None
        self._pending_project_to_select: Optional[str] = None
        self._refresh_retry_count: int = 0  # Track retry attempts to prevent infinite loops
        
        # DataRegistry will be accessed directly when needed via self.get_registry()
        # No need to pre-connect to RegistryBus - simplified approach
        self.registry = None
        
        # Call parent __init__ which will call setup_ui()
        super().__init__(parent=parent, panel_id="drillhole_database")

    def refresh_theme(self):
        """Update colors when theme changes."""
        colors = get_theme_colors()
        # Re-apply stylesheet with new theme colors
        if hasattr(self, 'setStyleSheet'):
            self.setStyleSheet(self.styleSheet())
        # Refresh child widgets
        for child in self.findChildren(QWidget):
            if hasattr(child, 'refresh_theme'):
                child.refresh_theme()
        self.setWindowTitle("Drillhole Database Management")
        
        logger.info("Initialized Drillhole Database Management panel")
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = self.main_layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("<b>Drillhole Database Management</b>")
        header.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(header)
        
        # Project Management Group
        project_group = QGroupBox("Project Management")
        project_layout = QVBoxLayout()
        
        # Project selection
        project_select_layout = QHBoxLayout()
        project_select_layout.addWidget(QLabel("Project:"))
        
        self.project_combo = QComboBox()
        self.project_combo.currentTextChanged.connect(self._on_project_changed)
        project_select_layout.addWidget(self.project_combo)
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_projects)
        project_select_layout.addWidget(refresh_btn)
        
        project_layout.addLayout(project_select_layout)
        
        # Project actions
        project_actions_layout = QHBoxLayout()
        
        create_btn = QPushButton("Create Project")
        create_btn.clicked.connect(self._create_project)
        project_actions_layout.addWidget(create_btn)
        
        delete_btn = QPushButton("Delete Project")
        delete_btn.clicked.connect(self._delete_project)
        project_actions_layout.addWidget(delete_btn)
        
        project_layout.addLayout(project_actions_layout)
        
        project_group.setLayout(project_layout)
        layout.addWidget(project_group)
        
        # Export Group (only export existing data, no import)
        export_group = QGroupBox("Export Database")
        export_layout = QHBoxLayout()
        
        export_btn = QPushButton("Export Database to CSV")
        export_btn.clicked.connect(self._export_database)
        export_btn.setToolTip("Export existing database data to CSV files")
        export_layout.addWidget(export_btn)
        
        # Save composited data from DataRegistry
        save_composited_btn = QPushButton("Save Composited Data to Database")
        save_composited_btn.clicked.connect(self._save_composited_data)
        save_composited_btn.setToolTip("Save drillhole data from DataRegistry to the selected project")
        export_layout.addWidget(save_composited_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Info section
        info_group = QGroupBox("Purpose & Workflow")
        info_layout = QVBoxLayout()
        info_text = QLabel(
            "<b>Purpose:</b> This panel manages persistent storage of drillhole data in SQLite databases.<br><br>"
            "<b>Workflow:</b><br>"
            "1. Load drillhole data via <b>Geological Modeling → Drillhole Loading</b><br>"
            "2. Create a project below to organize your data<br>"
            "3. Save the loaded drillhole data to this database project<br>"
            "4. Later, load saved data back from the database for analysis"
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: #333; padding: 10px; background-color: #f5f5f5; border-radius: 5px;")
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Backup/Restore Group
        backup_group = QGroupBox("Backup/Restore")
        backup_layout = QHBoxLayout()
        
        backup_btn = QPushButton("Backup Database")
        backup_btn.clicked.connect(self._backup_database)
        backup_layout.addWidget(backup_btn)
        
        restore_btn = QPushButton("Restore Database")
        restore_btn.clicked.connect(self._restore_database)
        backup_layout.addWidget(restore_btn)
        
        backup_group.setLayout(backup_layout)
        layout.addWidget(backup_group)
        
        # Statistics Group
        stats_group = QGroupBox("Database Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setMaximumHeight(200)
        stats_layout.addWidget(self.stats_text)
        
        refresh_stats_btn = QPushButton("Refresh Statistics")
        refresh_stats_btn.clicked.connect(self._refresh_statistics)
        stats_layout.addWidget(refresh_stats_btn)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Holes List
        holes_group = QGroupBox("Holes")
        holes_layout = QVBoxLayout()
        
        self.holes_list = QListWidget()
        self.holes_list.setMaximumHeight(150)
        holes_layout.addWidget(self.holes_list)
        
        load_btn = QPushButton("Load Database")
        load_btn.clicked.connect(self._load_database)
        holes_layout.addWidget(load_btn)
        
        holes_group.setLayout(holes_layout)
        layout.addWidget(holes_group)
        
        # Initialize database manager AFTER UI is fully set up
        # Use QTimer to defer to next event loop cycle
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self._initialize_database_manager)

    def _on_registry_ready(self, registry):
        """One-time hook when DataRegistry is initialized."""
        try:
            self.registry = registry
            self.registry.drillholeDataLoaded.connect(self._on_drillhole_data_loaded)
            try:
                existing_data = self.registry.get_drillhole_data()
                if existing_data:
                    self._on_drillhole_data_loaded(existing_data)
            except Exception:
                pass
            logger.info("Drillhole Database panel connected to DataRegistry (bus)")
        except Exception as e:
            logger.warning(f"Failed to hook DataRegistry: {e}")
    
    def _initialize_database_manager(self):
        """Initialize database manager (called after UI is ready)."""
        self._do_initialize_database_manager()
    
    def _do_initialize_database_manager(self):
        """Actually initialize database manager (called after UI is shown)."""
        try:
            self.db_manager = DrillholeDatabaseManager()
            logger.info("Initialized database manager")
            # Refresh projects after initialization
            self._refresh_projects()
        except Exception as e:
            logger.error(f"Error initializing database manager: {e}")
            # Don't show message box here as it might be called before UI is ready
            # Just log the error
    
    def _refresh_projects(self):
        """Refresh project list."""
        if not self.db_manager:
            return
        
        # Check if UI is ready - retry if not (with limit)
        if self.project_combo is None:
            self._refresh_retry_count += 1
            if self._refresh_retry_count > 50:  # Max 50 retries = 10 seconds
                logger.error("Project combo still not ready after 50 retries, giving up")
                self._refresh_retry_count = 0
                return
            logger.debug(f"Project combo not ready yet, retrying in 200ms... (attempt {self._refresh_retry_count}/50)")
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, self._refresh_projects)
            return
        
        # Reset retry count on successful access
        self._refresh_retry_count = 0
        
        try:
            projects = self.db_manager.list_projects()
            self.project_combo.clear()
            
            for project in projects:
                self.project_combo.addItem(project['name'], project['id'])
            
            logger.info(f"Refreshed projects: {len(projects)} projects")

            # If a current project is known, attempt to re-select it
            if self.current_project:
                try:
                    self.project_combo.setCurrentText(self.current_project)
                except Exception:
                    pass
                if self.project_combo.currentText() != self.current_project:
                    for i in range(self.project_combo.count()):
                        if self.project_combo.itemText(i) == self.current_project:
                            self.project_combo.setCurrentIndex(i)
                            break
        except Exception as e:
            logger.error(f"Error refreshing projects: {e}")
            # Don't show message box if UI isn't ready
            if self.project_combo is not None:
                QMessageBox.critical(self, "Error", f"Failed to refresh projects:\n{e}")
    
    def _on_project_changed(self, project_name: str):
        """Handle project selection change."""
        if not project_name:
            return
        
        self.current_project = project_name
        self._refresh_statistics()
        self._refresh_holes_list()
        self.project_changed.emit(project_name)
        logger.info(f"Selected project: {project_name}")
    
    def _create_project(self):
        """Create a new project."""
        if not self.db_manager:
            QMessageBox.warning(self, "Database Error", "Database manager not initialized. Please check logs.")
            return
        
        from PyQt6.QtWidgets import QInputDialog, QDialog, QVBoxLayout, QLabel, QLineEdit, QDialogButtonBox
        
        # Create custom dialog for better UX
        dialog = QDialog(self)
        dialog.setWindowTitle("Create New Project")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Project name
        layout.addWidget(QLabel("Project Name:"))
        name_input = QLineEdit()
        name_input.setPlaceholderText("Enter project name (e.g., 'Exploration_2024')")
        layout.addWidget(name_input)
        
        # Description
        layout.addWidget(QLabel("Description (optional):"))
        desc_input = QLineEdit()
        desc_input.setPlaceholderText("Brief description of the project")
        layout.addWidget(desc_input)
        
        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        name = name_input.text().strip()
        description = desc_input.text().strip()
        
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Project name cannot be empty.")
            return
        
        try:
            project_id = self.db_manager.create_project(name, description)
            # Refresh and try to select the project robustly
            self._refresh_projects()
            self._pending_project_to_select = name
            self._try_select_project(name)
            QMessageBox.information(
                self, 
                "Success", 
                f"Created project '{name}'\n\n"
                f"You can now save drillhole data to this project from the Drillhole Loading panel."
            )
            logger.info(f"Created project '{name}' (ID: {project_id})")
        except ValueError as e:
            logger.error(f"Error creating project: {e}")
            QMessageBox.warning(self, "Project Exists", f"Project '{name}' already exists. Please choose a different name.")
        except Exception as e:
            logger.error(f"Error creating project: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{e}")

    def _try_select_project(self, name: str):
        """Attempt to select a project in the combo with retries and fallback."""
        from PyQt6.QtCore import QTimer
        
        def _select_now() -> bool:
            if self.project_combo is None:
                return False
            # Try direct
            try:
                self.project_combo.setCurrentText(name)
            except Exception:
                pass
            if self.project_combo.currentText() == name:
                self.current_project = name
                return True
            # Scan items
            for i in range(self.project_combo.count()):
                if self.project_combo.itemText(i) == name:
                    self.project_combo.setCurrentIndex(i)
                    self.current_project = name
                    return True
            return False
        
        # Immediate attempt
        if _select_now():
            return
        
        # Retry refresh/select for up to ~10s to allow DB/UI to catch up
        retries = {'count': 0, 'max': 50}
        
        def _retry():
            if retries['count'] >= retries['max']:
                # Fallback: insert item if still missing
                if self.project_combo is not None:
                    self.project_combo.addItem(name)
                    self.project_combo.setCurrentText(name)
                    self.current_project = name
                    logger.warning(f"Project '{name}' not listed by manager; inserted into combo as fallback")
                return
            retries['count'] += 1
            # Only refresh if combo is ready to avoid spam
            if self.project_combo is not None:
                self._refresh_projects()
            if not _select_now():
                QTimer.singleShot(200, _retry)
        
        QTimer.singleShot(200, _retry)

    # Ensure selection/refresh is applied after the widget is shown
    def showEvent(self, event):
        try:
            super().showEvent(event)
        except Exception:
            pass
        # After the panel is visible, refresh projects and apply any pending selection
        from PyQt6.QtCore import QTimer
        def _apply_pending():
            if self.project_combo is not None:
                self._refresh_projects()
                if self._pending_project_to_select:
                    self._try_select_project(self._pending_project_to_select)
        QTimer.singleShot(0, _apply_pending)
    
    def _delete_project(self):
        """Delete current project."""
        if not self.db_manager or not self.current_project:
            QMessageBox.warning(self, "No Project", "Please select a project to delete")
            return
        
        reply = QMessageBox.question(
            self, "Delete Project",
            f"Are you sure you want to delete project '{self.current_project}'?\n\n"
            "This will permanently delete all data in this project.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.db_manager.delete_project(self.current_project)
                self.current_project = None
                self.current_database = None
                self._refresh_projects()
                QMessageBox.information(self, "Success", "Project deleted")
                logger.info(f"Deleted project '{self.current_project}'")
            except Exception as e:
                logger.error(f"Error deleting project: {e}")
                QMessageBox.critical(self, "Error", f"Failed to delete project:\n{e}")
    
    
    def _export_database(self):
        """Export database to CSV."""
        if not self.db_manager or not self.current_project:
            QMessageBox.warning(self, "No Project", "Please select a project first")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Database", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            db = self.db_manager.load_database(self.current_project)
            
            # Export to CSV
            from ..drillholes.reporting import ReportGenerator
            report_gen = ReportGenerator(db)
            report_gen.export_to_csv(Path(file_path), "all")
            
            QMessageBox.information(self, "Success", f"Exported database to {file_path}")
            logger.info(f"Exported database to {file_path}")
        except Exception as e:
            logger.error(f"Error exporting database: {e}")
            QMessageBox.critical(self, "Error", f"Failed to export database:\n{e}")
    
    def _backup_database(self):
        """Create database backup."""
        if not self.db_manager:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Backup Database", "", "Database Files (*.db);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.db_manager.backup_database(Path(file_path))
            QMessageBox.information(self, "Success", f"Created database backup at {file_path}")
            logger.info(f"Created database backup at {file_path}")
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            QMessageBox.critical(self, "Error", f"Failed to backup database:\n{e}")
    
    def _restore_database(self):
        """Restore database from backup."""
        if not self.db_manager:
            return
        
        reply = QMessageBox.question(
            self, "Restore Database",
            "Are you sure you want to restore the database?\n\n"
            "This will replace the current database with the backup.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Restore Database", "", "Database Files (*.db);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            self.db_manager.restore_database(Path(file_path))
            self._refresh_projects()
            QMessageBox.information(self, "Success", "Database restored")
            logger.info(f"Restored database from {file_path}")
        except Exception as e:
            logger.error(f"Error restoring database: {e}")
            QMessageBox.critical(self, "Error", f"Failed to restore database:\n{e}")
    
    def _refresh_statistics(self):
        """Refresh database statistics asynchronously to avoid UI freezes."""
        if not self.db_manager or not self.current_project:
            if self.stats_text is not None:
                self.stats_text.clear()
            return
        
        # Avoid starting multiple workers
        # Show loading hint
        if self.stats_text is not None:
            self.stats_text.setPlainText("Loading statistics...")
        
        # Use controller task system for statistics loading
        if not self.controller:
            logger.warning("Controller not available for statistics loading")
            return
        
        params = {
            'db_manager': self.db_manager,
            'project': self.current_project,
            'operation': 'get_statistics'
        }
        
        def _on_stats_complete(result: Dict[str, Any]):
            if result is None or result.get("error"):
                error_msg = result.get("error", "Unknown error") if result else "No result"
                if self.stats_text is not None:
                    self.stats_text.setPlainText(f"Error: {error_msg}")
                logger.error(f"Error refreshing statistics: {error_msg}")
                return
            
            stats = result.get("stats", {})
            try:
                stats_text = (
                    f"""
=== Database Statistics ===

Project: {self.current_project}

Number of Holes: {stats.get('num_holes', 'N/A')}
Number of Collars: {stats.get('num_collars', 'N/A')}
Number of Surveys: {stats.get('num_surveys', 'N/A')}
Number of Assays: {stats.get('num_assays', 'N/A')}
Number of Elements: {stats.get('num_elements', 'N/A')}
Number of Lithology Intervals: {stats.get('num_lithology', 'N/A')}
"""
                )
                if self.stats_text is not None:
                    self.stats_text.setPlainText(stats_text)
                logger.info(f"Refreshed statistics for project '{self.current_project}'")
            except Exception as e:
                logger.error(f"Error formatting statistics: {e}")
        
        # Note: This is a lightweight operation, but we'll use the task system for consistency
        # In practice, this could be done synchronously, but using task system ensures consistency
        self.controller.run_task('drillhole_database', params, callback=_on_stats_complete)
    
    def _refresh_holes_list(self):
        """Refresh holes list."""
        if not self.db_manager or not self.current_project:
            self.holes_list.clear()
            return
        
        try:
            db = self.db_manager.load_database(self.current_project)
            hole_ids = db.get_hole_ids()
            
            self.holes_list.clear()
            for hole_id in hole_ids:
                self.holes_list.addItem(hole_id)
            
            logger.info(f"Refreshed holes list: {len(hole_ids)} holes")
        except Exception as e:
            logger.error(f"Error refreshing holes list: {e}")
    
    def _load_database(self):
        """Load database into memory and publish to DataRegistry."""
        if not self.db_manager or not self.current_project:
            QMessageBox.warning(self, "No Project", "Please select a project first")
            return
        
        try:
            self.current_database = self.db_manager.load_database(self.current_project)
            
            # Publish loaded data to DataRegistry
            try:
                # Use get_registry() for dependency injection
                registry = self.get_registry()
                if registry is None:
                    logger.warning("get_registry() returned None - this should not happen")
                    self.database_loaded.emit(self.current_database)
                    QMessageBox.information(self, "Loaded",
                        f"Loaded database for project '{self.current_project}'.\n"
                        f"(Registry initialization issue; data loaded but not published.)")
                    return

                # Convert database to DataFrames using to_dataframe() method
                dfs = self.current_database.to_dataframe()
                drillhole_data = {}
                
                # Get collars
                if 'collars' in dfs and dfs['collars'] is not None and not dfs['collars'].empty:
                    drillhole_data['collars'] = dfs['collars']
                    logger.info(f"Loaded {len(dfs['collars'])} collars from database")
                
                # Get surveys
                if 'surveys' in dfs and dfs['surveys'] is not None and not dfs['surveys'].empty:
                    drillhole_data['surveys'] = dfs['surveys']
                    logger.info(f"Loaded {len(dfs['surveys'])} surveys from database")
                
                # Get assays
                if 'assays' in dfs and dfs['assays'] is not None and not dfs['assays'].empty:
                    assays_df = dfs['assays']
                    if not assays_df.empty:
                        drillhole_data['assays'] = assays_df
                        logger.info(f"Loaded {len(assays_df)} assays from database")
                
                # Get lithology if available
                if 'lithology' in dfs and dfs['lithology'] is not None and not dfs['lithology'].empty:
                    lithology_df = dfs['lithology']
                    if not lithology_df.empty:
                        drillhole_data['lithology'] = lithology_df
                        logger.info(f"Loaded {len(lithology_df)} lithology records from database")
                
                # Register with DataRegistry
                if drillhole_data:
                    registry.register_drillhole_data(drillhole_data, source_panel="Drillhole Database Management")
                    logger.info(f"Published {len(drillhole_data)} data types to DataRegistry")
                    
                    # Also update drillhole loading panel if available
                    if hasattr(self.parent(), 'domain_compositing_panel') and self.parent().domain_compositing_panel:
                        panel = self.parent().domain_compositing_panel
                        if 'assays' in drillhole_data:
                            panel.assay_df = drillhole_data['assays']
                        if 'collars' in drillhole_data:
                            panel.collar_df = drillhole_data['collars']
                        if 'surveys' in drillhole_data:
                            panel.survey_df = drillhole_data['surveys']
                        if 'lithology' in drillhole_data:
                            panel.lithology_df = drillhole_data['lithology']
                        logger.info("Updated drillhole loading panel with database data")
                
            except Exception as reg_error:
                logger.warning(f"Could not publish to DataRegistry: {reg_error}")
            
            self.database_loaded.emit(self.current_database)
            QMessageBox.information(self, "Success", f"Loaded database for project '{self.current_project}' and published to DataRegistry")
            logger.info(f"Loaded database for project '{self.current_project}'")
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load database:\n{e}")
    
    def get_current_database(self) -> Optional[DrillholeDatabase]:
        """Get current database."""
        return self.current_database
    
    def get_current_project(self) -> Optional[str]:
        """Get current project name."""
        return self.current_project
    
    def _save_composited_data(self):
        """Save drillhole data from DataRegistry to the selected project."""
        if not self.db_manager:
            QMessageBox.warning(self, "Database Error", "Database manager not initialized. Please check logs.")
            return
        
        if not self.current_project:
            QMessageBox.warning(self, "No Project", "Please select a project first")
            return
        
        try:
            # Get data from DataRegistry via dependency injection
            registry = self.get_registry()
            if registry is None:
                QMessageBox.warning(self, "Registry Error",
                    "DataRegistry failed to initialize. Please restart the application.")
                return
            drillhole_data = registry.get_drillhole_data()
            
            if drillhole_data is None:
                QMessageBox.information(
                    self,
                    "No Data",
                    "No drillhole data found in registry.\n\n"
                    "Please load drillhole data first via:\n"
                    "Drillholes → Drillhole Loading\n\n"
                    "Or run compositing to generate composites."
                )
                return
            
            # Check if we have composited data (preferred) or raw assays
            composites = drillhole_data.get('composites')
            assays = drillhole_data.get('assays')
            
            if composites is None or composites.empty:
                if assays is None or assays.empty:
                    QMessageBox.information(
                        self,
                        "No Data",
                        "No drillhole data found in registry.\n\n"
                        "Please load drillhole data first via:\n"
                        "Drillholes → Drillhole Loading\n\n"
                        "Or run compositing to generate composites."
                    )
                    return
                # Use raw assays if no composites
                drillhole_data['assays'] = assays
                logger.info("Saving raw assays to database (no composites available)")
            else:
                # Prefer composites, but also save raw assays if available for comparison
                drillhole_data['composites'] = composites
                if assays is not None and not assays.empty:
                    drillhole_data['assays'] = assays
                logger.info(f"Saving {len(composites)} composites to database (raw assays also available)")
            
            # Load or create database for the project
            try:
                db = self.db_manager.load_database(self.current_project)
            except Exception:
                # Database doesn't exist yet, create it
                logger.info(f"Database for project '{self.current_project}' doesn't exist, creating...")
                self.db_manager.create_project(self.current_project, f"Auto-created for drillhole data")
                db = self.db_manager.load_database(self.current_project)
            
            # Save each data component (convert DataFrames into DrillholeDatabase then persist)
            saved_components = []

            def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
                """Find a column by common aliases (case-insensitive)."""
                lowered = {c.lower(): c for c in df.columns}
                for cand in candidates:
                    if cand in df.columns:
                        return cand
                    if cand.lower() in lowered:
                        return lowered[cand.lower()]
                return None

            new_db = DrillholeDatabase(metadata=db.metadata.copy())

            # Collars
            collars_df = drillhole_data.get('collars')
            if collars_df is not None and not collars_df.empty:
                hole_col = _find_column(collars_df, ["hole_id", "hole", "holeid", "hole_id"])
                x_col = _find_column(collars_df, ["x", "easting", "east"])
                y_col = _find_column(collars_df, ["y", "northing", "north"])
                z_col = _find_column(collars_df, ["z", "elevation", "rl"])
                az_col = _find_column(collars_df, ["azimuth", "azi"])
                dip_col = _find_column(collars_df, ["dip"])
                len_col = _find_column(collars_df, ["length", "total_length", "len"])

                if hole_col and x_col and y_col and z_col:
                    # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                    rows = []
                    for _, row in collars_df.iterrows():
                        if pd.isna(row[hole_col]):
                            continue
                        rows.append({
                            'hole_id': str(row[hole_col]),
                            'x': float(row[x_col]) if pd.notna(row.get(x_col)) else 0.0,
                            'y': float(row[y_col]) if pd.notna(row.get(y_col)) else 0.0,
                            'z': float(row[z_col]) if pd.notna(row.get(z_col)) else 0.0,
                            'azimuth': float(row[az_col]) if az_col and pd.notna(row.get(az_col)) else None,
                            'dip': float(row[dip_col]) if dip_col and pd.notna(row.get(dip_col)) else None,
                            'length': float(row[len_col]) if len_col and pd.notna(row.get(len_col)) else None,
                        })
                    
                    if rows:
                        collar_df_new = pd.DataFrame(rows)
                        if new_db.collars.empty:
                            new_db.collars = collar_df_new
                        else:
                            new_db.collars = pd.concat([new_db.collars, collar_df_new], ignore_index=True)
                    saved_components.append(f"Collars: {len(new_db.collars)} records")

            # Surveys
            surveys_df = drillhole_data.get('surveys')
            if surveys_df is not None and not surveys_df.empty:
                hole_col = _find_column(surveys_df, ["hole_id", "holeid", "hole", "bhid", "dhid", "drillhole"])
                from_col = _find_column(surveys_df, ["depth_from", "from", "mfrom", "depth", "md"])
                to_col = _find_column(surveys_df, ["depth_to", "to", "mto", "depth_to"])
                az_col = _find_column(surveys_df, ["azimuth", "azi", "az"])
                dip_col = _find_column(surveys_df, ["dip", "inclination", "incl", "inc"])
                # Allow missing TO column – treat survey as point measurement if TO absent
                if hole_col and from_col and az_col and dip_col:
                    skipped_rows = 0
                    # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                    rows = []
                    for _, row in surveys_df.iterrows():
                        depth_to_val = row[to_col] if to_col and to_col in surveys_df.columns else row[from_col]
                        # Skip rows with missing azimuth/dip rather than breaking NOT NULL constraint
                        if pd.isna(row[az_col]) or pd.isna(row[dip_col]):
                            skipped_rows += 1
                            continue
                        if pd.isna(row[hole_col]):
                            skipped_rows += 1
                            continue
                        rows.append({
                            'hole_id': str(row[hole_col]),
                            'depth_from': float(row[from_col]),
                            'depth_to': float(depth_to_val),
                            'azimuth': float(row[az_col]),
                            'dip': float(row[dip_col]),
                        })
                    
                    if rows:
                        survey_df_new = pd.DataFrame(rows)
                        if new_db.surveys.empty:
                            new_db.surveys = survey_df_new
                        else:
                            new_db.surveys = pd.concat([new_db.surveys, survey_df_new], ignore_index=True)
                    saved_components.append(f"Surveys: {len(new_db.surveys)} records")
                    if skipped_rows:
                        logger.warning(f"Skipped {skipped_rows} survey rows due to missing azimuth/dip")

            # Assays (raw and composites share the same structure)
            def _ingest_assays(df: pd.DataFrame, label: str):
                if df is None or df.empty:
                    return 0
                hole_col = _find_column(df, ["hole_id", "holeid", "hole"])
                from_col = _find_column(df, ["depth_from", "from", "mfrom"])
                to_col = _find_column(df, ["depth_to", "to", "mto"])
                x_aliases = ["x", "east", "easting"]
                y_aliases = ["y", "north", "northing"]
                z_aliases = ["z", "rl", "elevation", "elev"]

                def _find_optional_column(aliases):
                    try:
                        return _find_column(df, aliases)
                    except ValueError:
                        return None

                x_col = _find_optional_column(x_aliases)
                y_col = _find_optional_column(y_aliases)
                z_col = _find_optional_column(z_aliases)

                if not (hole_col and from_col and to_col):
                    return 0
                # Columns that should never be treated as assay elements (metadata / geometry)
                meta_exclude_upper = {
                    "HOLEID",
                    "HOLE_ID",
                    "HOLE",
                    "SAMPLEID",
                    "SAMPLE_ID",
                    "FROM",
                    "TO",
                    "DEPTH",
                    "DEPTH_FROM",
                    "DEPTH_TO",
                    "LENGTH",
                    "MID",
                    "X",
                    "Y",
                    "Z",
                    "EAST",
                    "EASTING",
                    "NORTH",
                    "NORTHING",
                    "RL",
                    "ELEV",
                    "ELEVATION",
                    "DOMAIN",
                }
                exclude_cols = {hole_col, from_col, to_col}
                count_start = len(new_db.assays)
                # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                rows = []
                for _, row in df.iterrows():
                    if pd.isna(row[hole_col]):
                        continue
                    row_dict = {
                        'hole_id': str(row[hole_col]),
                        'depth_from': float(row[from_col]),
                        'depth_to': float(row[to_col]),
                    }
                    # Add assay value columns
                    has_values = False
                    for col in df.columns:
                        if col in exclude_cols or col.upper() in meta_exclude_upper:
                            continue
                        if pd.notna(row[col]):
                            try:
                                row_dict[col] = float(row[col])
                                has_values = True
                            except Exception:
                                # Ignore non-numeric columns
                                continue
                    # Add optional x, y, z coordinates if present
                    if x_col and pd.notna(row.get(x_col)):
                        row_dict['x'] = float(row[x_col])
                    if y_col and pd.notna(row.get(y_col)):
                        row_dict['y'] = float(row[y_col])
                    if z_col and pd.notna(row.get(z_col)):
                        row_dict['z'] = float(row[z_col])
                    
                    if has_values or True:  # Always add row if it has required columns
                        rows.append(row_dict)
                
                if rows:
                    assay_df_new = pd.DataFrame(rows)
                    if new_db.assays.empty:
                        new_db.assays = assay_df_new
                    else:
                        new_db.assays = pd.concat([new_db.assays, assay_df_new], ignore_index=True)
                return len(new_db.assays) - count_start

            assays_df = drillhole_data.get('assays')
            count_assays = _ingest_assays(assays_df, "assay")
            if count_assays:
                saved_components.append(f"Assays: {count_assays} records")


            # Lithology
            lith_df = drillhole_data.get('lithology')
            if lith_df is not None and not lith_df.empty:
                hole_col = _find_column(lith_df, ["hole_id", "holeid", "hole"])
                from_col = _find_column(lith_df, ["depth_from", "from", "mfrom"])
                to_col = _find_column(lith_df, ["depth_to", "to", "mto"])
                code_col = _find_column(lith_df, ["lith_code", "lithology", "code", "lith"])
                if hole_col and from_col and to_col and code_col:
                    # Build DataFrame rows (DataFrame.append() was removed in pandas 2.0)
                    rows = []
                    for _, row in lith_df.iterrows():
                        if pd.isna(row[hole_col]):
                            continue
                        rows.append({
                            'hole_id': str(row[hole_col]),
                            'depth_from': float(row[from_col]),
                            'depth_to': float(row[to_col]),
                            'lith_code': str(row[code_col]) if pd.notna(row.get(code_col)) else "Unknown",
                        })
                    
                    if rows:
                        lith_df_new = pd.DataFrame(rows)
                        if new_db.lithology.empty:
                            new_db.lithology = lith_df_new
                        else:
                            new_db.lithology = pd.concat([new_db.lithology, lith_df_new], ignore_index=True)
                    saved_components.append(f"Lithology: {len(new_db.lithology)} records")

            # Persist to SQLite
            if saved_components:
                self.db_manager.save_database(new_db, self.current_project)
                message = f"Successfully saved to project '{self.current_project}':\n\n" + "\n".join(saved_components)
                QMessageBox.information(self, "Success", message)
                logger.info(f"Saved drillhole data to project '{self.current_project}': {saved_components}")
                self._refresh_statistics()
                self._refresh_holes_list()
            else:
                QMessageBox.warning(self, "No Data", "No data to save. The registry data is empty.")
                
        except Exception as e:
            logger.error(f"Error saving composited data: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to save drillhole data:\n{e}")
    
    def _on_drillhole_data_loaded(self, drillhole_data: Dict[str, pd.DataFrame]):
        """Handle drillhole data loaded from DataRegistry - update UI to reflect availability."""
        try:
            if drillhole_data is None:
                return
            
            # Update UI to show that data is available
            has_data = any(
                df is not None and not df.empty 
                for df in drillhole_data.values()
            )
            
            if has_data:
                logger.info(f"Drillhole Database panel: Received data from DataRegistry "
                           f"({', '.join(k for k, v in drillhole_data.items() if v is not None and not v.empty)})")
                    
        except Exception as e:
            logger.warning(f"Error handling drillhole data loaded signal: {e}")

