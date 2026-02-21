"""
K-Means Clustering Panel for Block Model Viewer
Implements unsupervised clustering for geological domain classification
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QMessageBox,
    QListWidget, QTextEdit
)
from .panel_manager import PanelCategory, DockArea

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from .base_analysis_panel import BaseAnalysisPanel

logger = logging.getLogger(__name__)


class KMeansClusteringPanel(BaseAnalysisPanel):
    """Panel for K-means clustering analysis."""
    # PanelManager metadata
    PANEL_ID = "KMeansClusteringPanel"
    PANEL_NAME = "KMeansClustering Panel"
    PANEL_CATEGORY = PanelCategory.OTHER
    PANEL_DEFAULT_VISIBLE = False
    PANEL_DEFAULT_DOCK_AREA = DockArea.RIGHT




    
    task_name = "kmeans"
    
    # Signals
    clustering_complete = pyqtSignal(str, object)  # Emits property_name, labels
    
    def __init__(self, parent=None):
        super().__init__(parent=parent, panel_id="kmeans")
        
        # block_model is a read-only property from BasePanel - use _block_model instead
        self._block_model = None
        self.results = None
        
        # Subscribe to block model from DataRegistry
        try:
            self.registry = self.get_registry()
            self.registry.blockModelGenerated.connect(self._on_block_model_generated)
            self.registry.blockModelLoaded.connect(self._on_block_model_loaded)
            
            # Load existing block model if available
            existing_block_model = self.registry.get_block_model()
            if existing_block_model:
                self._on_block_model_loaded(existing_block_model)
        except Exception as e:
            logger.warning(f"Failed to connect to DataRegistry: {e}")
            self.registry = None
        
        self.setup_ui()
        logger.info("K-means clustering panel initialized")
    
    def _on_block_model_generated(self, block_model):
        """
        Automatically receive block model when it's generated.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        logger.info("K-means Clustering Panel received block model from DataRegistry")
        # Use base class method to properly set the block model and trigger callbacks
        super().set_block_model(block_model)
    
    def _on_block_model_loaded(self, block_model):
        """
        Automatically receive block model when it's loaded.
        
        Args:
            block_model: BlockModel instance or DataFrame from DataRegistry
        """
        self._on_block_model_generated(block_model)
    
    def setup_ui(self):
        """Initialize the user interface."""
        layout = self.main_layout
        
        # Header
        header = QLabel("K-Means Clustering Analysis")
        header.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Feature Selection
        feature_group = QGroupBox("1. Select Features for Clustering")
        feature_layout = QVBoxLayout(feature_group)
        
        feature_layout.addWidget(QLabel("Select properties to use as clustering features:"))
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.feature_list.setMaximumHeight(150)
        feature_layout.addWidget(self.feature_list)
        
        # Quick selection buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self.feature_list.selectAll)
        clear_selection_btn = QPushButton("Clear Selection")
        clear_selection_btn.clicked.connect(self.feature_list.clearSelection)
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(clear_selection_btn)
        btn_layout.addStretch()
        feature_layout.addLayout(btn_layout)
        
        layout.addWidget(feature_group)
        
        # Clustering Parameters
        params_group = QGroupBox("2. Clustering Parameters")
        params_layout = QVBoxLayout(params_group)
        
        # Number of clusters
        clusters_layout = QHBoxLayout()
        clusters_layout.addWidget(QLabel("Number of Clusters (k):"))
        self.n_clusters_spin = QSpinBox()
        self.n_clusters_spin.setRange(2, 20)
        self.n_clusters_spin.setValue(3)
        self.n_clusters_spin.setToolTip("Number of clusters to identify")
        clusters_layout.addWidget(self.n_clusters_spin)
        clusters_layout.addStretch()
        params_layout.addLayout(clusters_layout)
        
        # Standardization
        self.standardize_check = QCheckBox("Standardize Features (Recommended)")
        self.standardize_check.setChecked(True)
        self.standardize_check.setToolTip("Normalize features to zero mean and unit variance")
        params_layout.addWidget(self.standardize_check)
        
        # Advanced parameters (collapsible)
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(advanced_group)
        
        # N_init
        ninit_layout = QHBoxLayout()
        ninit_layout.addWidget(QLabel("Number of Initializations:"))
        self.n_init_spin = QSpinBox()
        self.n_init_spin.setRange(1, 50)
        self.n_init_spin.setValue(10)
        self.n_init_spin.setToolTip("Number of times to run with different seeds")
        ninit_layout.addWidget(self.n_init_spin)
        ninit_layout.addStretch()
        advanced_layout.addLayout(ninit_layout)
        
        # Max iterations
        maxiter_layout = QHBoxLayout()
        maxiter_layout.addWidget(QLabel("Maximum Iterations:"))
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 1000)
        self.max_iter_spin.setValue(300)
        self.max_iter_spin.setToolTip("Maximum number of iterations per run")
        maxiter_layout.addWidget(self.max_iter_spin)
        maxiter_layout.addStretch()
        advanced_layout.addLayout(maxiter_layout)
        
        # Random state
        random_layout = QHBoxLayout()
        self.use_random_state = QCheckBox("Use Random Seed:")
        self.use_random_state.setChecked(True)
        random_layout.addWidget(self.use_random_state)
        self.random_state_spin = QSpinBox()
        self.random_state_spin.setRange(0, 99999)
        self.random_state_spin.setValue(42)
        self.random_state_spin.setToolTip("Seed for reproducibility")
        random_layout.addWidget(self.random_state_spin)
        random_layout.addStretch()
        advanced_layout.addLayout(random_layout)
        
        params_layout.addWidget(advanced_group)
        layout.addWidget(params_group)
        
        # Run button
        self.run_btn = QPushButton("Run K-Means Clustering")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("font-size: 12pt; font-weight: bold;")
        layout.addWidget(self.run_btn)
        
        # Results
        results_group = QGroupBox("3. Clustering Results")
        results_layout = QVBoxLayout(results_group)
        
        # Summary statistics
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(150)
        self.results_text.setFont(QFont("Consolas", 9))
        results_layout.addWidget(self.results_text)
        
        # Cluster statistics table
        self.cluster_table = QTableWidget()
        self.cluster_table.setMaximumHeight(200)
        results_layout.addWidget(self.cluster_table)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.apply_btn = QPushButton("Apply to Block Model")
        self.apply_btn.clicked.connect(self._apply_to_model)
        self.apply_btn.setEnabled(False)
        action_layout.addWidget(self.apply_btn)
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.clicked.connect(self._export_results)
        self.export_btn.setEnabled(False)
        action_layout.addWidget(self.export_btn)
        
        action_layout.addStretch()
        results_layout.addLayout(action_layout)
        
        layout.addWidget(results_group)
        
        layout.addStretch()
    
    def set_block_model(self, block_model):
        """Set the block model for clustering."""
        # Use base class method to properly set the block model
        super().set_block_model(block_model)
        
        # Populate feature list with numeric properties
        self.feature_list.clear()
        
        if block_model is not None:
            df = block_model.to_dataframe()
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Exclude coordinate columns
            exclude = ['XMORIG', 'YMORIG', 'ZMORIG', 'XINC', 'YINC', 'ZINC', 
                      'NX', 'NY', 'NZ', 'BLOCK_ID']
            
            features = [col for col in numeric_cols if col not in exclude]
            self.feature_list.addItems(features)
            
            logger.info(f"Loaded {len(features)} features for clustering")
    
    # ------------------------------------------------------------------
    # BaseAnalysisPanel overrides
    # ------------------------------------------------------------------
    
    def gather_parameters(self) -> Dict[str, Any]:
        """Collect all parameters from the UI."""
        if self.block_model is None:
            raise ValueError("No block model loaded.")
        
        # Get selected features
        selected_items = self.feature_list.selectedItems()
        if not selected_items:
            raise ValueError("Please select at least one feature for clustering.")
        
        features = [item.text() for item in selected_items]
        
        # Get parameters
        n_clusters = self.n_clusters_spin.value()
        n_init = self.n_init_spin.value()
        max_iter = self.max_iter_spin.value()
        random_state = self.random_state_spin.value() if self.use_random_state.isChecked() else None
        standardize = self.standardize_check.isChecked()
        
        # Prepare data
        df = self.block_model.to_dataframe()
        
        return {
            "data_df": df,
            "features": features,
            "n_clusters": n_clusters,
            "n_init": n_init,
            "max_iter": max_iter,
            "random_state": random_state,
            "standardize": standardize,
        }
    
    def validate_inputs(self) -> bool:
        """Validate collected parameters."""
        if not super().validate_inputs():
            return False
        
        if self.block_model is None:
            self.show_error("No Data", "Please load a block model first.")
            return False
        
        selected_items = self.feature_list.selectedItems()
        if not selected_items:
            self.show_error("No Features Selected", "Please select at least one feature for clustering.")
            return False
        
        return True
    
    def on_results(self, payload: Dict[str, Any]) -> None:
        """Process and display clustering results."""
        results = payload.get("results")
        if results is None:
            return
        
        self.results = results
        
        # Publish clustering results to DataRegistry
        if hasattr(self, 'registry') and self.registry:
            try:
                clustering_results = {
                    'results': results,
                    'n_clusters': results.get('n_clusters'),
                    'features': results.get('features', []),
                    'cluster_labels': results.get('cluster_labels'),
                    'source': 'kmeans_clustering'
                }
                # Store as block model property if block model available
                existing_block_model = self.registry.get_block_model()
                if existing_block_model is not None:
                    # Add cluster labels to block model
                    if hasattr(existing_block_model, 'df') and 'cluster_labels' in results:
                        existing_block_model.df['cluster_label'] = results['cluster_labels']
                        self.registry.register_block_model(existing_block_model, source_panel="KMeansClusteringPanel")
                        logger.info("KMeans Clustering Panel published cluster labels to block model in DataRegistry")
            except Exception as e:
                logger.warning(f"Failed to register clustering results: {e}")
        
        # Display summary
        summary = f"""
K-Means Clustering Results
{'='*50}

Parameters:
  Number of Clusters (k): {results['n_clusters']}
  Features Used: {', '.join(results['features'])}
  Iterations: {results['n_iterations']}

Quality Metrics:
  Within-Cluster Sum of Squares (Inertia): {results['inertia']:.2f}
"""
        
        if results['silhouette'] is not None:
            summary += f"  Silhouette Score: {results['silhouette']:.4f} (Range: -1 to 1, higher is better)\n"
        
        if results['calinski_harabasz'] is not None:
            summary += f"  Calinski-Harabasz Index: {results['calinski_harabasz']:.2f} (Higher is better)\n"
        
        if results['davies_bouldin'] is not None:
            summary += f"  Davies-Bouldin Index: {results['davies_bouldin']:.4f} (Lower is better)\n"
        
        # valid_mask might be a list or numpy array, use sum() for compatibility
        valid_mask = results.get('valid_mask', [])
        if isinstance(valid_mask, (list, tuple)):
            valid_count = sum(valid_mask)
        else:
            # numpy array or similar
            valid_count = valid_mask.sum() if hasattr(valid_mask, 'sum') else sum(valid_mask)
        summary += f"\nValid Blocks: {valid_count} / {len(results['labels'])}\n"
        
        if hasattr(self, 'results_text'):
            self.results_text.setPlainText(summary)
        
        # Populate cluster statistics table
        stats = results['cluster_stats']
        features = results['features']
        
        if hasattr(self, 'cluster_table'):
            self.cluster_table.setRowCount(len(stats))
            self.cluster_table.setColumnCount(3 + len(features) * 2)
            
            headers = ['Cluster', 'Count', 'Percentage (%)']
            for feat in features:
                headers.append(f'{feat} (Mean)')
                headers.append(f'{feat} (Std)')
            
            self.cluster_table.setHorizontalHeaderLabels(headers)
            
            for i, stat in enumerate(stats):
                self.cluster_table.setItem(i, 0, QTableWidgetItem(str(stat['cluster_id'])))
                self.cluster_table.setItem(i, 1, QTableWidgetItem(f"{stat['count']:,}"))
                self.cluster_table.setItem(i, 2, QTableWidgetItem(f"{stat['percentage']:.2f}"))
                
                col_idx = 3
                for feat in features:
                    self.cluster_table.setItem(i, col_idx, QTableWidgetItem(f"{stat[f'{feat}_mean']:.3f}"))
                    self.cluster_table.setItem(i, col_idx + 1, QTableWidgetItem(f"{stat[f'{feat}_std']:.3f}"))
                    col_idx += 2
            
            self.cluster_table.resizeColumnsToContents()
        
        # Enable action buttons
        if hasattr(self, 'apply_btn'):
            self.apply_btn.setEnabled(True)
        if hasattr(self, 'export_btn'):
            self.export_btn.setEnabled(True)
        
        # valid_mask might be a list or numpy array, use sum() for compatibility
        valid_mask = results.get('valid_mask', [])
        if isinstance(valid_mask, (list, tuple)):
            valid_count = sum(valid_mask)
        else:
            # numpy array or similar
            valid_count = valid_mask.sum() if hasattr(valid_mask, 'sum') else sum(valid_mask)
        
        self.show_info("Clustering Complete", f"K-means clustering completed successfully!\n\nIdentified {results['n_clusters']} clusters from {valid_count} blocks.")
    
    def _apply_to_model(self):
        """Apply clustering results to the block model."""
        if self.results is None:
            return
        
        property_name = f"CLUSTER_K{self.results['n_clusters']}"
        
        # Add cluster labels as a new property
        self.block_model.add_property(property_name, self.results['labels'])
        
        # Emit signal
        self.clustering_complete.emit(property_name, self.results['labels'])
        
        QMessageBox.information(
            self,
            "Applied to Model",
            f"Cluster labels added as property: '{property_name}'\n\n"
            f"You can now visualize this property in the 3D viewer."
        )
        
        logger.info(f"Applied clustering results as property: {property_name}")
    
    def _export_results(self):
        """Export clustering results to CSV."""
        if self.results is None:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Clustering Results",
            f"kmeans_k{self.results['n_clusters']}_results.csv",
            "CSV Files (*.csv)"
        )
        
        if not file_path:
            return
        
        try:
            # Create DataFrame with results
            export_df = pd.DataFrame({
                'BLOCK_ID': range(len(self.results['labels'])),
                'CLUSTER': self.results['labels']
            })
            
            # Add original features
            df = self.block_model.to_dataframe()
            for feat in self.results['features']:
                export_df[feat] = df[feat].values
            
            # Export
            # Step 10: Use ExportHelpers
            from ..utils.export_helpers import export_dataframe_to_csv
            export_dataframe_to_csv(export_df, file_path)
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Clustering results exported to:\n{file_path}"
            )
            
            logger.info(f"Exported clustering results to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export results:\n{str(e)}"
            )
            logger.error(f"Export failed: {e}", exc_info=True)

