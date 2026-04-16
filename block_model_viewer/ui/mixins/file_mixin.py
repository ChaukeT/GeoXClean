"""
FileMixin — project save/load, session persistence, and file I/O methods
for MainWindow.

These methods were extracted from the 15 000-line MainWindow god-object to
keep each file maintainable.  MainWindow inherits from this mixin so every
call still works identically through the normal `self.<method>()` call path.

DO NOT import anything from main_window here to avoid circular imports.
All attributes accessed as `self.<attr>` resolve on the MainWindow instance
at runtime via Python's MRO.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

# Qt imports needed by file I/O methods
from PyQt6.QtCore import Qt, QTimer, QSettings, QEventLoop
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QInputDialog,
    QMessageBox,
    QProgressDialog,
    QVBoxLayout,
)

from ..project_loading_dialog import ProjectLoadingDialog
from ...core.process_history_tracker import get_process_history_tracker

logger = logging.getLogger(__name__)


class FileMixin:
    """
    Mixin that provides all project save/load and file I/O methods for
    MainWindow.  Inherit BEFORE QMainWindow in the MRO:
        class MainWindow(PanelMixin, FileMixin, QMainWindow): ...
    """

    def _save_session(self):
        """Persist last file and renderer state for next launch."""
        try:
            settings = QSettings("GeoX", "Session")
            if self.current_file_path:
                settings.setValue("last_file", str(self.current_file_path))
            # Persist renderer state
            try:
                if self.viewer_widget and self.viewer_widget.renderer:
                    state = self.viewer_widget.renderer.get_session_state()
                    settings.setValue("renderer_state", json.dumps(state))
            except Exception:
                pass
            # Also persist the toggle state, in case it changed programmatically
            try:
                if hasattr(self, 'restore_session_action') and self.restore_session_action is not None:
                    settings.setValue("restore_on_startup", bool(self.restore_session_action.isChecked()))
            except Exception:
                pass
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Failed to save session: {e_msg}")
            except Exception:
                logger.debug("Failed to save session: <unprintable error>")

    # ============================================================================
    # UNDO / REDO
    # ============================================================================
    def _on_renderer_state_change(self, snapshot: Dict[str, Any], reason: str = ""):
        """Receive pre-change snapshots from renderer and push onto undo stack."""
        try:
            if not hasattr(self, '_undo_stack'):
                self._undo_stack = []
                self._redo_stack = []
                self._max_history = 50
            # Push snapshot and trim
            self._undo_stack.append(snapshot)
            if len(self._undo_stack) > getattr(self, '_max_history', 50):
                self._undo_stack = self._undo_stack[-self._max_history:]
            # Clear redo on new action
            self._redo_stack.clear()
            # Update UI state
            if hasattr(self, 'undo_action'):
                self.undo_action.setEnabled(len(self._undo_stack) > 0)
            if hasattr(self, 'redo_action'):
                self.redo_action.setEnabled(len(self._redo_stack) > 0)
            # Optional status
            self.statusBar().showMessage("Change captured for undo", 1500)
            # Mark project dirty
            self._mark_dirty()
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Undo snapshot push failed: {e_msg}")
            except Exception:
                logger.debug("Undo snapshot push failed: <unprintable error>")

    def _mark_dirty(self):
        """Mark current project as having unsaved changes and update title."""
        try:
            self._dirty = True
            base = "GeoX"
            if self.current_project_path:
                base += f" - {self.current_project_path.stem}"
            self.setWindowTitle(base + "*")
        except Exception:
            pass

    def _clear_dirty(self):
        try:
            self._dirty = False
            base = "GeoX"
            if self.current_project_path:
                base += f" - {self.current_project_path.stem}"
            self.setWindowTitle(base)
        except Exception:
            pass

    def _autosave_if_dirty(self):
        """Periodically write an autosave backup if there are unsaved changes."""
        try:
            if not self._dirty:
                return
            if not self.current_project_path:
                return
            # Write to sidecar autosave file next to project
            autosave_path = self.current_project_path.with_suffix(self.current_project_path.suffix + ".autosave")
            state = self._collect_project_state()
            autosave_path.parent.mkdir(parents=True, exist_ok=True)
            with open(autosave_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            logger.info(f"Autosaved project backup: {autosave_path}")
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"Autosave skipped: {e_msg}")
            except Exception:
                logger.debug("Autosave skipped: <unprintable error>")

    # ============================================================================
    # PROJECT: Save / Load
    # ============================================================================
    def _collect_project_state(self) -> Dict[str, Any]:
        """Collect the current application state for project serialization."""
        data_files: Dict[str, Any] = {}
        if self.current_file_path:
            data_files['block_model'] = str(self.current_file_path)

        # Save drillhole data if present - check multiple sources
        drillhole_state = None
        try:
            drillhole_data = None
            composites_df = None
            assays_df = None
            collars_df = None
            surveys_df = None
            trajectories_df = None
            color_by = ''
            radius = 5.0

            # Method 1: Check compositing panel for composited data
            if hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel:
                if hasattr(self.domain_compositing_panel, 'comp_domain_df') and self.domain_compositing_panel.comp_domain_df is not None:
                    composites_df = self.domain_compositing_panel.comp_domain_df
                    logger.info(f"Found {len(composites_df)} composites in compositing panel")

                if hasattr(self.domain_compositing_panel, 'assay_df') and self.domain_compositing_panel.assay_df is not None:
                    assays_df = self.domain_compositing_panel.assay_df
                    logger.info(f"Found {len(assays_df)} assays in drillhole loading panel")

                if hasattr(self.domain_compositing_panel, 'collar_df') and self.domain_compositing_panel.collar_df is not None:
                    collars_df = self.domain_compositing_panel.collar_df
                    logger.info(f"Found {len(collars_df)} collars in drillhole loading panel")

                if hasattr(self.domain_compositing_panel, 'survey_df') and self.domain_compositing_panel.survey_df is not None:
                    surveys_df = self.domain_compositing_panel.survey_df
                    logger.info(f"Found {len(surveys_df)} surveys in drillhole loading panel")

            # Method 2: Check DataRegistry for drillhole data (using proper getter method)
            try:
                from ...core.data_registry import DataRegistry
                if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                    registry = DataRegistry._instance
                    # Use proper getter method - get_drillhole_data() returns data from _data_store
                    reg_data = registry.get_drillhole_data(copy_data=False)
                    if reg_data is not None:
                        if assays_df is None and 'assays' in reg_data and reg_data['assays'] is not None:
                            assays_df = reg_data['assays']
                            logger.info(f"Found {len(assays_df)} assays in DataRegistry")
                        if collars_df is None and 'collars' in reg_data and reg_data['collars'] is not None:
                            collars_df = reg_data['collars']
                            logger.info(f"Found {len(collars_df)} collars in DataRegistry")
                        if surveys_df is None and 'surveys' in reg_data and reg_data['surveys'] is not None:
                            surveys_df = reg_data['surveys']
                            logger.info(f"Found {len(surveys_df)} surveys in DataRegistry")
                        if trajectories_df is None and 'trajectories' in reg_data and reg_data['trajectories'] is not None:
                            trajectories_df = reg_data['trajectories']
                            logger.info(f"Found {len(trajectories_df)} trajectories in DataRegistry")
                        if composites_df is None and 'composites' in reg_data and reg_data['composites'] is not None:
                            composites_df = reg_data['composites']
                            logger.info(f"Found {len(composites_df)} composites in DataRegistry")
                        if 'lithology' in reg_data and reg_data['lithology'] is not None:
                            # Save lithology as well
                            pass  # Will be handled in save logic below
            except Exception as reg_error:
                try:
                    reg_msg = str(reg_error)
                    logger.debug(f"Could not check DataRegistry: {reg_msg}")
                except Exception:
                    logger.debug("Could not check DataRegistry: <unprintable error>")

            # Method 3: Check renderer for drillhole data (existing method)
            if (hasattr(self.viewer_widget.renderer, 'drillhole_data') and
                self.viewer_widget.renderer.drillhole_data is not None):
                drillhole_data = self.viewer_widget.renderer.drillhole_data

                if composites_df is None and 'composites_df' in drillhole_data:
                    composites_df = drillhole_data['composites_df']
                    logger.info(f"Found {len(composites_df)} composites in renderer")

                # Get visualization settings from renderer
                color_by = drillhole_data.get('color_by', '')
                radius = float(drillhole_data.get('radius', 5.0))

            # Save all found drillhole data to CSV files (only if files don't exist)
            if self.current_project_path and (composites_df is not None or assays_df is not None or
                                             collars_df is not None or surveys_df is not None or
                                             trajectories_df is not None):
                # Create a drillhole_data subfolder next to the project file
                drillhole_folder = self.current_project_path.parent / f"{self.current_project_path.stem}_drillholes"
                drillhole_folder.mkdir(exist_ok=True)

                saved_files = {}

                # Save drillhole CSVs (always overwrite so edits are persisted)
                from ...utils.export_helpers import export_dataframe_to_csv

                if composites_df is not None and not composites_df.empty:
                    composites_path = drillhole_folder / "composites.csv"
                    export_dataframe_to_csv(composites_df, composites_path, process_events=False)
                    saved_files['composites'] = str(composites_path)
                    logger.info(f"Exported {len(composites_df)} composites to {composites_path}")

                if assays_df is not None and not assays_df.empty:
                    assays_path = drillhole_folder / "assays.csv"
                    export_dataframe_to_csv(assays_df, assays_path, process_events=False)
                    saved_files['assays'] = str(assays_path)
                    logger.info(f"Exported {len(assays_df)} assays to {assays_path}")

                if collars_df is not None and not collars_df.empty:
                    collars_path = drillhole_folder / "collars.csv"
                    export_dataframe_to_csv(collars_df, collars_path, process_events=False)
                    saved_files['collars'] = str(collars_path)
                    logger.info(f"Exported {len(collars_df)} collars to {collars_path}")

                if surveys_df is not None and not surveys_df.empty:
                    surveys_path = drillhole_folder / "surveys.csv"
                    export_dataframe_to_csv(surveys_df, surveys_path, process_events=False)
                    saved_files['surveys'] = str(surveys_path)
                    logger.info(f"Exported {len(surveys_df)} surveys to {surveys_path}")

                if trajectories_df is not None and not trajectories_df.empty:
                    trajectories_path = drillhole_folder / "trajectories.csv"
                    export_dataframe_to_csv(trajectories_df, trajectories_path, process_events=False)
                    saved_files['trajectories'] = str(trajectories_path)
                    logger.info(f"Exported {len(trajectories_df)} trajectories to {trajectories_path}")

                # Save lithology (from DataRegistry)
                try:
                    from ...core.data_registry import DataRegistry
                    if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                        registry = DataRegistry._instance
                        reg_data = registry.get_drillhole_data(copy_data=False)
                        if reg_data is not None and 'lithology' in reg_data and reg_data['lithology'] is not None:
                            lithology_df = reg_data['lithology']
                            if not lithology_df.empty:
                                lithology_path = drillhole_folder / "lithology.csv"
                                export_dataframe_to_csv(lithology_df, lithology_path, process_events=False)
                                saved_files['lithology'] = str(lithology_path)
                                logger.info(f"Exported {len(lithology_df)} lithology records to {lithology_path}")
                except Exception as e:
                    logger.debug(f"Could not save lithology: {e}")

                # Create drillhole state dict
                if saved_files:
                    drillhole_state = saved_files.copy()
                    # Also save visualization settings if available
                    if color_by:
                        drillhole_state['color_by'] = color_by
                    drillhole_state['radius'] = radius

                    logger.info(f"Saved drillhole data: {', '.join(saved_files.keys())}")
        except Exception as e:
            logger.error(f"Error saving drillhole data: {e}", exc_info=True)

        # Fallback: if no drillhole data was found in memory but a companion
        # folder already exists from a previous save, reference those CSV files
        # so the project JSON keeps pointing at the data.
        if drillhole_state is None and self.current_project_path:
            try:
                fallback_folder = (
                    self.current_project_path.parent
                    / f"{self.current_project_path.stem}_drillholes"
                )
                if fallback_folder.exists():
                    fallback_files: Dict[str, Any] = {}
                    for key in ('assays', 'collars', 'surveys', 'composites',
                                'trajectories', 'lithology'):
                        p = fallback_folder / f"{key}.csv"
                        if p.exists():
                            fallback_files[key] = str(p)
                    if fallback_files:
                        fallback_files['radius'] = 5.0
                        drillhole_state = fallback_files
                        logger.info(
                            f"Drillhole data not in memory — referencing "
                            f"existing CSVs from companion folder: "
                            f"{list(fallback_files.keys())}"
                        )
            except Exception as e:
                logger.debug(f"Drillhole companion folder fallback failed: {e}")

        # Save all DataRegistry models and results
        registry_models_state = None
        try:
            import pickle

            import pandas as pd

            from ...core.data_registry import DataRegistry

            if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                registry = DataRegistry._instance

                if self.current_project_path:
                    models_folder = self.current_project_path.parent / f"{self.current_project_path.stem}_models"
                    models_folder.mkdir(exist_ok=True)

                    saved_models = {}

                    def save_registry_item(data_key: str, filename: str, display_name: str):
                        """Save a registry data item (always overwrites to persist edits)."""
                        try:
                            data = registry.get_data(data_key, copy_data=False)
                            if data is not None:
                                save_path = models_folder / filename
                                if isinstance(data, pd.DataFrame):
                                    csv_path = save_path.with_suffix('.csv')
                                    from ...utils.export_helpers import export_dataframe_to_csv
                                    export_dataframe_to_csv(data, csv_path, process_events=False)
                                    logger.info(f"Exported {display_name} ({len(data)} rows) to project")
                                    saved_models[data_key] = str(csv_path)
                                else:
                                    pkl_path = save_path.with_suffix('.pkl')
                                    with open(pkl_path, 'wb') as f:
                                        pickle.dump(data, f)
                                    logger.info(f"Exported {display_name} to project")
                                    saved_models[data_key] = str(pkl_path)
                        except Exception as e:
                            logger.warning(f"Could not save {display_name}: {e}")

                    # Helper: save data to pickle (always overwrites)
                    def _save_pkl(data_obj, key: str, filename: str, label: str):
                        if data_obj is None:
                            return
                        try:
                            p = models_folder / filename
                            with open(p, 'wb') as f:
                                pickle.dump(data_obj, f)
                            saved_models[key] = str(p)
                            logger.info(f"Exported {label} to project")
                        except Exception as e:
                            logger.warning(f"Could not save {label}: {e}")

                    _save_pkl(registry.get_variogram_results(copy_data=False),
                              'variogram_results', 'variogram_results.pkl', 'variogram results')
                    _save_pkl(registry.get_drillholes_validation_state(),
                              'drillholes_validation_state', 'drillholes_validation_state.pkl', 'drillhole validation state')
                    _save_pkl(registry.get_declustering_results(copy_data=False),
                              'declustering_results', 'declustering_results.pkl', 'declustering results')
                    _save_pkl(registry.get_transformation_metadata(copy_data=False),
                              'transformation_metadata', 'transformation_metadata.pkl', 'transformation metadata')
                    _save_pkl(registry.get_transformers(),
                              'transformers', 'transformers.pkl', 'transformers')
                    _save_pkl(registry.get_kriging_results(copy_data=False),
                              'kriging_results', 'kriging_results.pkl', 'kriging results')
                    _save_pkl(registry.get_sgsim_results(copy_data=False),
                              'sgsim_results', 'sgsim_results.pkl', 'SGSIM results')
                    _save_pkl(registry.get_simple_kriging_results(copy_data=False),
                              'simple_kriging_results', 'simple_kriging_results.pkl', 'simple kriging results')
                    _save_pkl(registry.get_cokriging_results(copy_data=False),
                              'cokriging_results', 'cokriging_results.pkl', 'co-kriging results')
                    _save_pkl(registry.get_indicator_kriging_results(copy_data=False),
                              'indicator_kriging_results', 'indicator_kriging_results.pkl', 'indicator kriging results')
                    _save_pkl(registry.get_universal_kriging_results(copy_data=False),
                              'universal_kriging_results', 'universal_kriging_results.pkl', 'universal kriging results')
                    _save_pkl(registry.get_soft_kriging_results(copy_data=False),
                              'soft_kriging_results', 'soft_kriging_results.pkl', 'soft kriging results')
                    _save_pkl(registry.get_domain_model(copy_data=False),
                              'domain_model', 'domain_model.pkl', 'domain model')
                    _save_pkl(registry.get_contact_set(copy_data=False),
                              'contact_set', 'contact_set.pkl', 'contact set')

                    # Save all registered block models (multi-model support)
                    from ...utils.export_helpers import export_dataframe_to_csv as _export_csv

                    all_models = registry.get_block_model_list()
                    for model_info in all_models:
                        model_id = model_info['model_id']
                        model = registry.get_block_model(model_id=model_id, copy_data=False)
                        if model is not None:
                            try:
                                if hasattr(model, 'to_dataframe'):
                                    df = model.to_dataframe()
                                elif isinstance(model, pd.DataFrame):
                                    df = model
                                else:
                                    _save_pkl(model, f'block_model_{model_id}',
                                              f'block_model_{model_id}.pkl', f"block model '{model_id}'")
                                    continue
                                model_path = models_folder / f"block_model_{model_id}.csv"
                                _export_csv(df, model_path, process_events=False)
                                saved_models[f'block_model_{model_id}'] = str(model_path)
                                logger.info(f"Exported block model '{model_id}' to project")
                            except Exception as e:
                                logger.warning(f"Could not save block model '{model_id}': {e}")

                    # Save current model pointer
                    current_id = registry.get_current_block_model_id()
                    if current_id:
                        saved_models['current_block_model_id'] = current_id

                    # Backward compat: Also save current model as "block_model.csv"
                    if current_id:
                        current_model = registry.get_block_model(copy_data=False)
                        if current_model is not None:
                            try:
                                if hasattr(current_model, 'to_dataframe'):
                                    df = current_model.to_dataframe()
                                elif isinstance(current_model, pd.DataFrame):
                                    df = current_model
                                else:
                                    df = None
                                if df is not None:
                                    legacy_path = models_folder / "block_model.csv"
                                    _export_csv(df, legacy_path, process_events=False)
                                    saved_models['block_model'] = str(legacy_path)
                            except Exception:
                                pass

                    # Save classified block model
                    classified_model = registry.get_classified_block_model(copy_data=False)
                    if classified_model is not None:
                        try:
                            if hasattr(classified_model, 'to_dataframe'):
                                df = classified_model.to_dataframe()
                            elif isinstance(classified_model, pd.DataFrame):
                                df = classified_model
                            else:
                                df = None
                            if df is not None:
                                classified_path = models_folder / "classified_block_model.csv"
                                _export_csv(df, classified_path, process_events=False)
                                saved_models['classified_block_model'] = str(classified_path)
                                logger.info("Exported classified block model to project")
                            else:
                                _save_pkl(classified_model, 'classified_block_model',
                                          'classified_block_model.pkl', 'classified block model')
                        except Exception as e:
                            logger.warning(f"Could not save classified block model: {e}")

                    # Save remaining pickle-serialised results (always overwrite)
                    _save_pkl(registry.get_resource_summary(copy_data=False),
                              'resource_summary', 'resource_summary.pkl', 'resource summary')
                    _save_pkl(registry.get_geomet_results(copy_data=False),
                              'geomet_results', 'geomet_results.pkl', 'geomet results')
                    _save_pkl(registry.get_geomet_ore_types(copy_data=False),
                              'geomet_ore_types', 'geomet_ore_types.pkl', 'geomet ore types')
                    _save_pkl(registry.get_pit_optimization_results(copy_data=False),
                              'pit_optimization_results', 'pit_optimization_results.pkl', 'pit optimization results')
                    _save_pkl(registry.get_schedule(copy_data=False),
                              'schedule', 'schedule.pkl', 'schedule')
                    _save_pkl(registry.get_irr_results(copy_data=False),
                              'irr_results', 'irr_results.pkl', 'IRR results')
                    _save_pkl(registry.get_reconciliation_results(copy_data=False),
                              'reconciliation_results', 'reconciliation_results.pkl', 'reconciliation results')
                    _save_pkl(registry.get_haulage_evaluation(copy_data=False),
                              'haulage_evaluation', 'haulage_evaluation.pkl', 'haulage evaluation')
                    _save_pkl(registry.get_experiment_results(copy_data=False),
                              'experiment_results', 'experiment_results.pkl', 'experiment results')
                    _save_pkl(registry.get_category_label_maps(),
                              'category_label_maps', 'category_label_maps.pkl', 'category label maps')

                    # RBF / FastRBF / ARBF interpolation results
                    _save_pkl(registry.get_rbf_results(copy_data=False),
                              'rbf_results', 'rbf_results.pkl', 'RBF results')
                    _save_pkl(registry.get_fastrbf_results(copy_data=False),
                              'fastrbf_results', 'fastrbf_results.pkl', 'FastRBF results')
                    _save_pkl(registry.get_arbf_results(copy_data=False),
                              'arbf_results', 'arbf_results.pkl', 'ARBF results')

                    # Implicit geological modelling results
                    _save_pkl(registry.get_geological_surfaces(copy_data=False),
                              'implicit_surfaces', 'implicit_surfaces.pkl', 'implicit geological surfaces')
                    _save_pkl(registry.get_geological_solids(copy_data=False),
                              'voxel_solids', 'voxel_solids.pkl', 'voxel geological solids')

                    # LoopStructural geological model
                    _save_pkl(registry.get_loopstructural_model(copy_data=False),
                              'loopstructural_model', 'loopstructural_model.pkl', 'LoopStructural model')
                    _save_pkl(registry.get_loopstructural_compliance(copy_data=False),
                              'loopstructural_compliance', 'loopstructural_compliance.pkl', 'LoopStructural compliance')

                    # ── CATCH-ALL ──────────────────────────────────────────
                    # Save ANY remaining registry keys not already handled
                    # above.  This future-proofs the save so that new
                    # register_model() calls (e.g. InSAR, survey deformation,
                    # classification audits) are automatically persisted.
                    _already_saved = set(saved_models.keys())
                    # Keys that are handled via dedicated code above or are
                    # transient / internal and should NOT be persisted.
                    _skip_prefixes = ('block_model_',)
                    _skip_keys = {
                        'drillhole_data', 'drillhole_data_raw',  # saved separately via drillhole CSVs
                    }
                    try:
                        all_store_keys = list(registry._data_store.keys())
                        for store_key in all_store_keys:
                            if store_key in _already_saved:
                                continue
                            if store_key in _skip_keys:
                                continue
                            if any(store_key.startswith(pfx) for pfx in _skip_prefixes):
                                continue
                            # Try to save this previously-unknown key
                            try:
                                data = registry.get_data(store_key, copy_data=False)
                                if data is not None:
                                    safe_name = store_key.replace('/', '_').replace('\\', '_')
                                    if isinstance(data, pd.DataFrame):
                                        csv_path = models_folder / f"{safe_name}.csv"
                                        from ...utils.export_helpers import export_dataframe_to_csv as _exp_csv
                                        _exp_csv(data, csv_path, process_events=False)
                                        saved_models[store_key] = str(csv_path)
                                    else:
                                        pkl_path = models_folder / f"{safe_name}.pkl"
                                        with open(pkl_path, 'wb') as f:
                                            pickle.dump(data, f)
                                        saved_models[store_key] = str(pkl_path)
                                    logger.info(f"Catch-all saved registry key '{store_key}' to project")
                            except Exception as e:
                                logger.debug(f"Could not save registry key '{store_key}': {e}")
                    except Exception as e:
                        logger.debug(f"Catch-all registry save failed: {e}")

                    if saved_models:
                        registry_models_state = saved_models
                        logger.info(f"Saved {len(saved_models)} models/results to project: {', '.join(saved_models.keys())}")
        except Exception as e:
            logger.error(f"Error saving DataRegistry models: {e}", exc_info=True)

        # Fallback: if no registry models were found in memory but a companion
        # _models folder already exists, reference those files.
        if registry_models_state is None and self.current_project_path:
            try:
                fallback_models_folder = (
                    self.current_project_path.parent
                    / f"{self.current_project_path.stem}_models"
                )
                if fallback_models_folder.exists():
                    fallback_models: Dict[str, Any] = {}
                    known_keys = [
                        'variogram_results', 'kriging_results', 'sgsim_results',
                        'simple_kriging_results', 'cokriging_results',
                        'indicator_kriging_results', 'universal_kriging_results',
                        'soft_kriging_results', 'declustering_results',
                        'transformation_metadata', 'transformers',
                        'domain_model', 'contact_set', 'classified_block_model',
                        'classification_grid',
                        'resource_summary', 'geomet_results', 'geomet_ore_types',
                        'pit_optimization_results', 'schedule', 'irr_results',
                        'reconciliation_results', 'haulage_evaluation',
                        'experiment_results', 'category_label_maps',
                        'drillholes_validation_state', 'block_model',
                        'rbf_results', 'fastrbf_results', 'arbf_results',
                        'implicit_surfaces', 'voxel_solids',
                        'loopstructural_model', 'loopstructural_compliance',
                    ]
                    for key in known_keys:
                        for ext in ('.pkl', '.csv'):
                            p = fallback_models_folder / f"{key}{ext}"
                            if p.exists():
                                fallback_models[key] = str(p)
                                break
                    # Also pick up block_model_<id> files
                    for p in fallback_models_folder.glob('block_model_*.csv'):
                        stem = p.stem  # e.g. 'block_model_default'
                        fallback_models[stem] = str(p)
                    for p in fallback_models_folder.glob('block_model_*.pkl'):
                        stem = p.stem
                        if stem not in fallback_models:
                            fallback_models[stem] = str(p)
                    if fallback_models:
                        registry_models_state = fallback_models
                        logger.info(
                            f"Registry models not in memory — referencing "
                            f"existing files from companion folder: "
                            f"{list(fallback_models.keys())}"
                        )
            except Exception as e:
                logger.debug(f"Registry models companion folder fallback failed: {e}")

        try:
            renderer_state = self.viewer_widget.renderer.get_session_state() if (self.viewer_widget and self.viewer_widget.renderer) else {}
        except Exception:
            renderer_state = {}

        # Save process history
        process_history_state = None
        try:
            process_tracker = get_process_history_tracker()
            process_history_state = process_tracker.to_dict()
            logger.info(f"Saved process history: {process_tracker.get_process_count()} processes")
        except Exception as e:
            logger.warning(f"Could not save process history: {e}")

        # Save compositing settings (if compositing window is open/initialized)
        compositing_settings = None
        try:
            if hasattr(self, 'compositing_window') and self.compositing_window is not None:
                if hasattr(self.compositing_window, 'get_settings_state'):
                    compositing_settings = self.compositing_window.get_settings_state()
                    if compositing_settings:
                        logger.info("Saved compositing settings to project")
        except Exception as e:
            logger.warning(f"Could not save compositing settings: {e}")

        # Save panel settings from all registered panels
        panel_settings = {}
        try:
            panel_manager = getattr(self, 'panel_manager', None)
            if panel_manager:
                for panel_info in panel_manager.get_all_panels():
                    panel_id = panel_info.panel_id
                    panel_instance = panel_manager.get_panel_instance(panel_id)
                    if panel_instance and hasattr(panel_instance, 'get_panel_settings'):
                        try:
                            settings = panel_instance.get_panel_settings()
                            if settings:
                                panel_settings[panel_id] = settings
                                logger.debug(f"Saved settings for panel: {panel_id}")
                        except Exception as e:
                            logger.warning(f"Could not save settings for panel {panel_id}: {e}")
        except Exception as e:
            logger.warning(f"Could not collect panel settings: {e}")

        return {
            'version': 1,
            'app': 'BlockModelViewer',
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'data': data_files,
            'drillhole_data': drillhole_state,
            'registry_models': registry_models_state,
            'renderer_state': renderer_state,
            'bookmarks': self.bookmarks.get_bookmarks() if self.bookmarks is not None else self.view_bookmarks,
            'process_history': process_history_state,
            'compositing_settings': compositing_settings,
            'panel_settings': panel_settings if panel_settings else None,
        }

    def _restore_drillhole_data(self, drillhole_state: Dict[str, Any]):
        """Restore drillhole data from saved project state."""
        try:
            if not drillhole_state:
                return

            # Handle old format (composites_file) and new format (composites, assays, etc.)
            composites_file = drillhole_state.get('composites_file') or drillhole_state.get('composites')
            assays_file = drillhole_state.get('assays')
            collars_file = drillhole_state.get('collars')
            surveys_file = drillhole_state.get('surveys')
            trajectories_file = drillhole_state.get('trajectories')

            # Restore composites (optional — missing file is a warning, not a fatal error)
            composites_df = None
            if composites_file:
                if not Path(composites_file).exists():
                    logger.warning(
                        f"Drillhole composites file not found: {composites_file} "
                        f"(skipping composites, other data will still load)"
                    )
                else:
                    import pandas as pd
                    composites_df = pd.read_csv(composites_file)
                    logger.info(f"Loaded {len(composites_df)} composites from {composites_file}")

            # Load other drillhole data files
            assays_df = None
            if assays_file and Path(assays_file).exists():
                import pandas as pd
                assays_df = pd.read_csv(assays_file)
                logger.info(f"Loaded {len(assays_df)} assays from {assays_file}")

            collars_df = None
            if collars_file and Path(collars_file).exists():
                import pandas as pd
                collars_df = pd.read_csv(collars_file)
                logger.info(f"Loaded {len(collars_df)} collars from {collars_file}")

            surveys_df = None
            if surveys_file and Path(surveys_file).exists():
                import pandas as pd
                surveys_df = pd.read_csv(surveys_file)
                logger.info(f"Loaded {len(surveys_df)} surveys from {surveys_file}")

            trajectories_df = None
            if trajectories_file and Path(trajectories_file).exists():
                import pandas as pd
                trajectories_df = pd.read_csv(trajectories_file)
                logger.info(f"Loaded {len(trajectories_df)} trajectories from {trajectories_file}")

            # Load lithology if present (before panel restore so it's available)
            lithology_file = drillhole_state.get('lithology')
            lithology_df = None
            if lithology_file and Path(lithology_file).exists():
                import pandas as pd
                lithology_df = pd.read_csv(lithology_file)
                logger.info(f"Loaded {len(lithology_df)} lithology records from {lithology_file}")

            # Store loaded data in drillhole import panel (domain_compositing_panel)
            if hasattr(self, 'domain_compositing_panel') and self.domain_compositing_panel:
                panel = self.domain_compositing_panel
                if composites_df is not None:
                    panel.comp_domain_df = composites_df
                    logger.info("Restored composites to import panel")
                if assays_df is not None:
                    panel.assay_df = assays_df
                    logger.info("Restored assays to import panel")
                if collars_df is not None:
                    panel.collar_df = collars_df
                    logger.info("Restored collars to import panel")
                if surveys_df is not None:
                    panel.survey_df = surveys_df
                    logger.info("Restored surveys to import panel")
                if lithology_df is not None:
                    panel.lithology_df = lithology_df
                    logger.info("Restored lithology to import panel")

            # Register drillhole data with DataRegistry using proper method
            try:
                from ...core.data_registry import DataRegistry
                if hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None:
                    registry = DataRegistry._instance
                    drillhole_data = {}
                    if assays_df is not None:
                        drillhole_data['assays'] = assays_df
                    if collars_df is not None:
                        drillhole_data['collars'] = collars_df
                    if surveys_df is not None:
                        drillhole_data['surveys'] = surveys_df
                    if trajectories_df is not None:
                        drillhole_data['trajectories'] = trajectories_df
                    if composites_df is not None:
                        drillhole_data['composites'] = composites_df
                    if lithology_df is not None:
                        drillhole_data['lithology'] = lithology_df

                    if drillhole_data:
                        # Use proper registration method to store in _data_store
                        registry.register_drillhole_data(drillhole_data, source_panel="ProjectRestore")
                        logger.info("Restored drillhole data to DataRegistry")
            except Exception as reg_error:
                try:
                    reg_msg = str(reg_error)
                    logger.debug(f"Could not restore to DataRegistry: {reg_msg}")
                except Exception:
                    logger.debug("Could not restore to DataRegistry: <unprintable error>")

            # Data has been restored to DataRegistry above.
            # Auto-render drillholes so the project looks the same as when saved.
            n_composites = len(composites_df) if composites_df is not None else 0
            n_assays = len(assays_df) if assays_df is not None else 0
            logger.info(f"Drillhole data restored to registry: {n_composites} composites, {n_assays} assays")

            # Schedule deferred render so the viewer is fully ready first.
            _radius = float(drillhole_state.get('radius', 5.0))
            _color_by = drillhole_state.get('color_by', '')
            QTimer.singleShot(
                800,
                lambda: self._auto_render_drillholes_from_project(_radius, _color_by)
            )
            self.statusBar().showMessage("Drillhole data loaded from project", 3000)

        except Exception as e:
            logger.error(f"Error restoring drillhole data: {e}", exc_info=True)

    def _auto_render_drillholes_from_project(self, radius: float = 5.0, color_by: str = ''):
        """Render drillholes automatically after a project is loaded.

        Called via QTimer.singleShot so the viewer is fully initialised before
        we try to add layers.  Uses the Drillhole Control Panel if it is open,
        otherwise falls back to the viewer's drillhole layer API.
        """
        try:
            pm = getattr(self, 'panel_manager', None)
            if pm is None:
                return

            # If the DrillholeControlPanel is already open, just trigger its
            # plot button — it uses the settings already in the UI.
            dh_panel = pm.get_panel_instance("DrillholeControlPanel")
            if dh_panel is not None and hasattr(dh_panel, '_on_plot_clicked'):
                dh_panel._on_plot_clicked()
                logger.info("Auto-rendered drillholes via DrillholeControlPanel on project open")
                return

            # Panel not open — render directly via the viewer widget.
            if not (self.viewer_widget and self.viewer_widget.renderer):
                return

            from ...core.data_registry import DataRegistry
            if not (hasattr(DataRegistry, '_instance') and DataRegistry._instance is not None):
                return
            registry = DataRegistry._instance
            dh_data = registry.get_drillhole_data(copy_data=False)
            if not dh_data:
                return

            renderer = self.viewer_widget.renderer
            if hasattr(renderer, 'add_drillhole_layer'):
                renderer.add_drillhole_layer(dh_data, radius=radius, color_by=color_by or None)
                logger.info(
                    f"Auto-rendered drillholes on project open "
                    f"(radius={radius}, color_by={color_by!r})"
                )
                self.statusBar().showMessage("Drillholes restored from project", 3000)
        except Exception as e:
            logger.warning(f"Could not auto-render drillholes on project open: {e}")

    def _restore_compositing_settings(self, compositing_settings: Dict[str, Any]):
        """Restore compositing window settings from saved project state.
        
        This method initializes the compositing window (if needed) and applies
        the saved settings so that compositing parameters are preserved across
        project save/load cycles.
        
        Args:
            compositing_settings: Dictionary containing compositing tab settings
        """
        try:
            if not compositing_settings:
                logger.debug("No compositing settings to restore")
                return

            # If the compositing window is already open, apply settings directly
            if hasattr(self, 'compositing_window') and self.compositing_window is not None:
                if hasattr(self.compositing_window, 'apply_settings_state'):
                    self.compositing_window.apply_settings_state(compositing_settings)
                    logger.info("Applied compositing settings to existing window")
                return

            # Store settings for later application when compositing window is opened
            self._saved_compositing_settings = compositing_settings
            logger.info("Stored compositing settings for deferred application")

        except Exception as e:
            logger.warning(f"Error restoring compositing settings: {e}")

    def _restore_panel_settings(self, panel_settings: Dict[str, Dict[str, Any]]):
        """Restore settings for all panels from saved project state.
        
        This method applies saved settings to panels when they are opened.
        Settings are stored and applied when panels are initialized.
        
        Args:
            panel_settings: Dictionary mapping panel_id to settings dict
        """
        try:
            if not panel_settings:
                return

            # Store settings for deferred application (panels may not be open yet)
            if not hasattr(self, '_saved_panel_settings'):
                self._saved_panel_settings = {}

            self._saved_panel_settings.update(panel_settings)
            logger.info(f"Stored settings for {len(panel_settings)} panels for deferred application")

            # Try to apply settings to panels that are already open
            panel_manager = getattr(self, 'panel_manager', None)
            if panel_manager:
                for panel_id, settings in panel_settings.items():
                    panel_instance = panel_manager.get_panel_instance(panel_id)
                    if panel_instance and hasattr(panel_instance, 'apply_panel_settings'):
                        try:
                            panel_instance.apply_panel_settings(settings)
                            logger.debug(f"Applied settings to open panel: {panel_id}")
                        except Exception as e:
                            logger.warning(f"Could not apply settings to panel {panel_id}: {e}")

        except Exception as e:
            logger.warning(f"Error restoring panel settings: {e}")

    def _rebroadcast_registry_after_restore(self):
        """Re-emit key registry signals after project restore.

        During restore, ``register_*`` calls emit signals immediately, but
        lazy-loaded panels may not exist yet.  This method re-emits the
        signals after a short delay so panels created during the load
        sequence discover the restored data.
        """
        from PyQt6.QtCore import QTimer

        def _broadcast():
            try:
                registry = None
                if hasattr(self, 'controller') and self.controller:
                    registry = getattr(self.controller, 'registry', None)
                if registry is None:
                    return

                n = 0

                # Block models (all registered)
                bm = registry.get_block_model(copy_data=False)
                if bm is not None:
                    registry._emit("blockModelLoaded", bm)
                    n += 1

                # Classified block model
                cbm = registry.get_classified_block_model(copy_data=False)
                if cbm is not None:
                    registry._emit("blockModelClassified", cbm)
                    n += 1

                # Drillhole data
                dh = registry.get_drillhole_data(copy_data=False)
                if dh is not None:
                    registry._emit("drillholeDataLoaded", dh)
                    n += 1

                # Variogram
                vario = registry.get_data("variogram_results", copy_data=False)
                if vario is not None:
                    registry._emit("variogramResultsLoaded", vario)
                    n += 1

                # Estimation results — re-emit so panels like JORC, GT,
                # Resource Reporting discover restored models.
                for key, signal_name in [
                    ("kriging_results", "krigingResultsLoaded"),
                    ("simple_kriging_results", "simpleKrigingResultsLoaded"),
                    ("sgsim_results", "sgsimResultsLoaded"),
                    ("arbf_results", "arbfResultsLoaded"),
                    ("declustering_results", "declusteringResultsLoaded"),
                    ("grade_tonnage_results", "gradeTonnageResultsLoaded"),
                ]:
                    data = registry.get_data(key, copy_data=False)
                    if data is not None:
                        registry._emit(signal_name, data)
                        n += 1

                # IRBF domains
                if hasattr(registry, 'list_indicator_rbf_domains'):
                    names = registry.list_indicator_rbf_domains() or []
                    if names:
                        irbf = registry.get_indicator_rbf_domain()
                        if irbf is not None:
                            registry._emit("indicatorRBFDomainLoaded", irbf)
                            n += 1

                logger.info(
                    "Re-broadcast %d registry signals after project restore", n
                )
            except Exception as exc:
                logger.debug("Post-restore signal broadcast failed: %s", exc)

        QTimer.singleShot(800, _broadcast)

    def _restore_registry_models(self, registry_models_state: Dict[str, Any]):
        """Restore DataRegistry models from saved project state.

        This method restores all saved models (variogram, kriging, SGSIM, resource,
        classification, etc.) back into the DataRegistry when a project is loaded.
        """
        try:
            if not registry_models_state:
                logger.debug("No registry models to restore")
                return

            import pickle

            import pandas as pd

            from ...core.data_registry import DataRegistry

            if not hasattr(DataRegistry, '_instance') or DataRegistry._instance is None:
                logger.warning("DataRegistry not available for restoring models")
                return

            registry = DataRegistry._instance
            restored_count = 0
            restore_errors = []  # Track failures for user-visible reporting

            # Helper function to load a pickle file
            def load_pickle(file_path: str) -> Any:
                """Load data from a pickle file with security validation.
                
                SECURITY: Validates path and file size before loading.
                WARNING: Pickle can execute arbitrary code. Only load trusted files.
                """
                from ...utils.security import SecurityError, validate_pickle_file

                try:
                    validated_path, file_size = validate_pickle_file(
                        Path(file_path),
                        allowed_base=None  # Project files are in project directory
                    )
                    logger.debug(f"Loading pickle file: {validated_path} ({file_size} bytes)")
                    with open(validated_path, 'rb') as f:
                        return pickle.load(f)
                except SecurityError as e:
                    logger.error(f"Security error loading pickle file {file_path}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Error loading pickle file {file_path}: {e}")
                    return None

            # Helper function to load a CSV file
            def load_csv(file_path: str) -> Optional[pd.DataFrame]:
                """Load data from a CSV file."""
                if not Path(file_path).exists():
                    logger.warning(f"File not found: {file_path}")
                    return None
                return pd.read_csv(file_path)

            # Restore variogram results
            if 'variogram_results' in registry_models_state:
                data = load_pickle(registry_models_state['variogram_results'])
                if data is not None:
                    registry.register_variogram_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored variogram results from project")

            # Restore drillhole validation state
            if 'drillholes_validation_state' in registry_models_state:
                data = load_pickle(registry_models_state['drillholes_validation_state'])
                if data is not None and isinstance(data, dict):
                    # Unpack the validation state dict and pass as kwargs
                    registry.set_drillholes_validation_state(
                        status=data.get('status', 'NOT_RUN'),
                        timestamp=data.get('timestamp', ''),
                        config_hash=data.get('config_hash', ''),
                        fatal_count=data.get('fatal_count', 0),
                        warn_count=data.get('warn_count', 0),
                        info_count=data.get('info_count', 0),
                        violations_summary=data.get('violations_summary'),
                        tables_validated=data.get('tables_validated'),
                        schema_errors=data.get('schema_errors'),
                        excluded_rows=data.get('excluded_rows')
                    )
                    restored_count += 1
                    logger.info(f"Restored drillhole validation state from project (status: {data.get('status', 'UNKNOWN')})")

            # Restore declustering results
            if 'declustering_results' in registry_models_state:
                data = load_pickle(registry_models_state['declustering_results'])
                if data is not None:
                    # Declustering results are stored as dict with 'weighted_dataframe' and 'summary'
                    if isinstance(data, dict) and 'weighted_dataframe' in data and 'summary' in data:
                        registry.register_declustering_results(
                            (data['weighted_dataframe'], data['summary']),
                            source_panel="ProjectRestore"
                        )
                    else:
                        # Legacy format
                        registry.register_model('declustering_results', data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored declustering results from project")

            # Restore transformation metadata
            if 'transformation_metadata' in registry_models_state:
                data = load_pickle(registry_models_state['transformation_metadata'])
                if data is not None:
                    registry.register_transformation_metadata(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored transformation metadata from project")

            # Restore transformers (special case)
            if 'transformers' in registry_models_state:
                data = load_pickle(registry_models_state['transformers'])
                if data is not None:
                    registry.register_transformers(data)
                    restored_count += 1
                    logger.info("Restored transformers from project")

            # Restore kriging results
            if 'kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['kriging_results'])
                if data is not None:
                    registry.register_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored kriging results from project")

            # Restore SGSIM results
            if 'sgsim_results' in registry_models_state:
                data = load_pickle(registry_models_state['sgsim_results'])
                if data is not None:
                    registry.register_sgsim_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored SGSIM results from project")

            # Restore simple kriging results
            if 'simple_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['simple_kriging_results'])
                if data is not None:
                    registry.register_simple_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored simple kriging results from project")

            # Restore co-kriging results
            if 'cokriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['cokriging_results'])
                if data is not None:
                    registry.register_cokriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored co-kriging results from project")

            # Restore indicator kriging results
            if 'indicator_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['indicator_kriging_results'])
                if data is not None:
                    registry.register_indicator_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored indicator kriging results from project")

            # Restore universal kriging results
            if 'universal_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['universal_kriging_results'])
                if data is not None:
                    registry.register_universal_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored universal kriging results from project")

            # Restore soft kriging results
            if 'soft_kriging_results' in registry_models_state:
                data = load_pickle(registry_models_state['soft_kriging_results'])
                if data is not None:
                    registry.register_soft_kriging_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored soft kriging results from project")

            # Restore domain model
            if 'domain_model' in registry_models_state:
                data = load_pickle(registry_models_state['domain_model'])
                if data is not None:
                    registry.register_domain_model(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored domain model from project")

            # Restore contact set
            if 'contact_set' in registry_models_state:
                data = load_pickle(registry_models_state['contact_set'])
                if data is not None:
                    registry.register_contact_set(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored contact set from project")

            # Helper: convert a DataFrame to a proper BlockModel object so that
            # the viewer and panels receive a real BlockModel, not a raw DF.
            def _df_to_block_model(df: pd.DataFrame) -> Any:
                """Convert a DataFrame (from saved CSV) into a BlockModel."""
                try:
                    from ...models.block_model import BlockModel as BM
                    bm = BM()
                    bm.update_from_dataframe(df)
                    return bm
                except Exception as conv_err:
                    logger.warning(f"Could not convert DataFrame to BlockModel: {conv_err}")
                    return df  # fallback: return the raw DF

            # Restore all block models (multi-model support)
            block_models_restored = 0
            current_model_id = None

            # Check if this is a legacy project (single block_model.csv)
            has_legacy = 'block_model' in registry_models_state
            has_multi = any(k.startswith('block_model_') and k != 'current_block_model_id'
                            for k in registry_models_state)

            if has_multi:
                # New multi-model project
                for key, file_path in registry_models_state.items():
                    if key.startswith('block_model_') and key != 'current_block_model_id':
                        # Extract model_id: "block_model_sgsim_FE_mean" -> "sgsim_FE_mean"
                        model_id = key[len('block_model_'):]

                        # Load data
                        try:
                            if file_path.endswith('.csv'):
                                data = load_csv(file_path)
                                if data is not None:
                                    data = _df_to_block_model(data)
                            else:
                                data = load_pickle(file_path)

                            if data is not None:
                                registry.register_block_model(
                                    data,
                                    source_panel="ProjectRestore",
                                    model_id=model_id,
                                    set_as_current=False
                                )
                                block_models_restored += 1
                                logger.info(f"Restored block model '{model_id}' from project")
                        except Exception as e:
                            logger.warning(f"Could not restore block model '{model_id}': {e}")

                # Restore current model pointer
                if 'current_block_model_id' in registry_models_state:
                    current_model_id = registry_models_state['current_block_model_id']
                    if registry.set_current_block_model(current_model_id):
                        logger.info(f"Set current block model to '{current_model_id}'")
                elif block_models_restored > 0:
                    # No saved current model - set first restored model as current
                    first_model_id = registry.get_block_model_list()[0]['model_id']
                    registry.set_current_block_model(first_model_id)
                    logger.info(f"Set first restored model '{first_model_id}' as current (no saved current)")

            elif has_legacy and not registry.has_block_model():
                # Legacy project - auto-migrate as "default" model
                try:
                    file_path = registry_models_state['block_model']
                    if file_path.endswith('.csv'):
                        data = load_csv(file_path)
                        if data is not None:
                            data = _df_to_block_model(data)
                    else:
                        data = load_pickle(file_path)

                    if data is not None:
                        registry.register_block_model(
                            data,
                            source_panel="ProjectRestore",
                            model_id="default",
                            set_as_current=True
                        )
                        block_models_restored = 1
                        logger.info("Restored legacy block model as 'default'")
                except Exception as e:
                    logger.warning(f"Could not restore legacy block model: {e}")

            if block_models_restored > 0:
                restored_count += block_models_restored
                logger.info(f"Restored {block_models_restored} block model(s) from project")

            # Restore classified block model (check for both CSV and pickle)
            if 'classified_block_model' in registry_models_state:
                file_path = registry_models_state['classified_block_model']
                if file_path.endswith('.csv'):
                    data = load_csv(file_path)
                    if data is not None:
                        data = _df_to_block_model(data)
                else:
                    data = load_pickle(file_path)
                if data is not None:
                    registry.register_classified_block_model(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored classified block model from project")

            # Restore resource summary
            if 'resource_summary' in registry_models_state:
                data = load_pickle(registry_models_state['resource_summary'])
                if data is not None:
                    registry.register_resource_summary(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored resource summary from project")

            # Restore geomet results
            if 'geomet_results' in registry_models_state:
                data = load_pickle(registry_models_state['geomet_results'])
                if data is not None:
                    registry.register_geomet_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored geomet results from project")

            # Restore geomet ore types
            if 'geomet_ore_types' in registry_models_state:
                data = load_pickle(registry_models_state['geomet_ore_types'])
                if data is not None:
                    registry.register_geomet_ore_types(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored geomet ore types from project")

            # Restore pit optimization results
            if 'pit_optimization_results' in registry_models_state:
                data = load_pickle(registry_models_state['pit_optimization_results'])
                if data is not None:
                    registry.register_pit_optimization_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored pit optimization results from project")

            # Restore schedule
            if 'schedule' in registry_models_state:
                data = load_pickle(registry_models_state['schedule'])
                if data is not None:
                    registry.register_schedule(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored schedule from project")

            # Restore IRR results
            if 'irr_results' in registry_models_state:
                data = load_pickle(registry_models_state['irr_results'])
                if data is not None:
                    registry.register_irr_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored IRR results from project")

            # Restore reconciliation results
            if 'reconciliation_results' in registry_models_state:
                data = load_pickle(registry_models_state['reconciliation_results'])
                if data is not None:
                    registry.register_reconciliation_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored reconciliation results from project")

            # Restore haulage evaluation
            if 'haulage_evaluation' in registry_models_state:
                data = load_pickle(registry_models_state['haulage_evaluation'])
                if data is not None:
                    registry.register_haulage_evaluation(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored haulage evaluation from project")

            # Restore experiment results
            if 'experiment_results' in registry_models_state:
                data = load_pickle(registry_models_state['experiment_results'])
                if data is not None:
                    registry.register_experiment_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored experiment results from project")

            # Restore category label maps
            if 'category_label_maps' in registry_models_state:
                data = load_pickle(registry_models_state['category_label_maps'])
                if data is not None:
                    registry.register_model("category_label_maps", data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info(f"Restored {len(data)} category label namespaces from project")

            # Restore RBF interpolation results
            if 'rbf_results' in registry_models_state:
                data = load_pickle(registry_models_state['rbf_results'])
                if data is not None:
                    registry.register_rbf_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored RBF results from project")

            # Restore FastRBF interpolation results
            if 'fastrbf_results' in registry_models_state:
                data = load_pickle(registry_models_state['fastrbf_results'])
                if data is not None:
                    registry.register_fastrbf_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored FastRBF results from project")

            # Restore ARBF interpolation results
            if 'arbf_results' in registry_models_state:
                data = load_pickle(registry_models_state['arbf_results'])
                if data is not None:
                    registry.register_arbf_results(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored ARBF results from project")

            # Restore implicit geological surfaces
            if 'implicit_surfaces' in registry_models_state:
                data = load_pickle(registry_models_state['implicit_surfaces'])
                if data is not None:
                    registry.register_geological_surfaces(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored implicit geological surfaces from project")

            # Restore voxel geological solids
            if 'voxel_solids' in registry_models_state:
                data = load_pickle(registry_models_state['voxel_solids'])
                if data is not None:
                    registry.register_geological_solids(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored voxel geological solids from project")

            # Restore LoopStructural geological model
            if 'loopstructural_model' in registry_models_state:
                data = load_pickle(registry_models_state['loopstructural_model'])
                if data is not None:
                    registry.register_loopstructural_model(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored LoopStructural model from project")

            # Restore LoopStructural compliance report
            if 'loopstructural_compliance' in registry_models_state:
                data = load_pickle(registry_models_state['loopstructural_compliance'])
                if data is not None:
                    registry.register_loopstructural_compliance(data, source_panel="ProjectRestore")
                    restored_count += 1
                    logger.info("Restored LoopStructural compliance report from project")

            # ── CATCH-ALL RESTORE ─────────────────────────────────
            # Restore any remaining keys not handled by explicit code
            # above (e.g. InSAR, survey deformation, classification
            # audits, or any future register_model() keys).
            _explicitly_handled = {
                'variogram_results', 'drillholes_validation_state',
                'declustering_results', 'transformation_metadata',
                'transformers', 'kriging_results', 'sgsim_results',
                'simple_kriging_results', 'cokriging_results',
                'indicator_kriging_results', 'universal_kriging_results',
                'soft_kriging_results', 'domain_model', 'contact_set',
                'classified_block_model', 'resource_summary',
                'geomet_results', 'geomet_ore_types',
                'pit_optimization_results', 'schedule', 'irr_results',
                'reconciliation_results', 'haulage_evaluation',
                'experiment_results', 'category_label_maps',
                'rbf_results', 'fastrbf_results', 'arbf_results',
                'implicit_surfaces', 'voxel_solids',
                'loopstructural_model', 'loopstructural_compliance',
                'block_model', 'current_block_model_id',
            }
            for key, file_path in registry_models_state.items():
                if key in _explicitly_handled:
                    continue
                if key.startswith('block_model_'):
                    continue  # handled by multi-model restore above
                if not isinstance(file_path, str):
                    continue  # skip non-path values
                try:
                    if file_path.endswith('.csv'):
                        data = load_csv(file_path)
                    elif file_path.endswith('.pkl'):
                        data = load_pickle(file_path)
                    else:
                        continue
                    if data is not None:
                        registry.register_model(key, data, source_panel="ProjectRestore")
                        restored_count += 1
                        logger.info(f"Catch-all restored registry key '{key}' from project")
                except Exception as e:
                    logger.debug(f"Could not restore registry key '{key}': {e}")

            if restored_count > 0:
                logger.info(f"Successfully restored {restored_count} models/results from project")
                if restore_errors:
                    error_summary = f"Restored {restored_count} items, {len(restore_errors)} failed"
                    self.statusBar().showMessage(error_summary, 5000)
                    logger.warning(f"Project restore errors: {'; '.join(restore_errors)}")
                else:
                    self.statusBar().showMessage(f"Restored {restored_count} project items", 3000)
            elif restore_errors:
                error_msg = f"Failed to restore project data: {'; '.join(restore_errors[:3])}"
                self.statusBar().showMessage(error_msg, 8000)
                logger.error(f"Project restore failures: {'; '.join(restore_errors)}")

        except Exception as e:
            logger.error(f"Error restoring registry models: {e}", exc_info=True)
            try:
                self.statusBar().showMessage(f"Error restoring project data: {e}", 8000)
            except Exception:
                pass

    def save_project(self):
        """Save the current project to disk. Uses existing path or prompts if none."""
        try:
            if not self.current_project_path:
                return self.save_project_as()

            # Show progress dialog for large saves
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import QProgressDialog
            progress = QProgressDialog("Saving project...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModality.WindowModal)
            progress.setMinimumDuration(500)  # Only show if takes > 500ms
            progress.setValue(0)
            QApplication.processEvents()

            try:
                # Collect project state (this does CSV exports)
                progress.setLabelText("Collecting project data...")
                progress.setValue(10)
                QApplication.processEvents()

                state = self._collect_project_state()

                progress.setLabelText("Writing project file...")
                progress.setValue(90)
                QApplication.processEvents()

                # Write JSON project file
                self.current_project_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.current_project_path, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2)

                progress.setValue(100)
                QApplication.processEvents()

            finally:
                progress.close()

            self.statusBar().showMessage(f"Project saved: {self.current_project_path.name}", 3000)
            logger.info(f"Saved project to {self.current_project_path}")
            # Clear dirty flag after successful save
            try:
                self._clear_dirty()
            except Exception:
                pass
            # Remember last project
            try:
                settings = QSettings("GeoX", "Project")
                settings.setValue("last_project", str(self.current_project_path))
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            QMessageBox.warning(self, "Save Project", f"Failed to save project:\n{e}")

    def save_project_as(self):
        """Prompt for a project filename and save the project."""
        try:
            # Prefer the current project path for defaults; fall back to block model path, then home
            if self.current_project_path:
                default_dir = str(self.current_project_path.parent)
                default_name = self.current_project_path.name
            elif self.current_file_path:
                default_dir = str(self.current_file_path.parent)
                default_name = self.current_file_path.stem + ".bmvproj"
            else:
                default_dir = str(Path.home())
                default_name = "project.bmvproj"
            path_str, _ = QFileDialog.getSaveFileName(self, "Save Project As", str(Path(default_dir) / default_name), "Block Model Viewer Project (*.bmvproj);;JSON (*.json)")
            if not path_str:
                return

            new_path = Path(path_str)
            old_path = self.current_project_path  # may be None

            # Migrate companion folders when the project name changes so that
            # drillhole/model CSVs are not orphaned under the old name.
            if old_path and old_path != new_path:
                self._migrate_companion_folders(old_path, new_path)

            self.current_project_path = new_path
            self.save_project()
            # Reflect in window title
            try:
                self.setWindowTitle(f"GeoX - {self.current_project_path.stem}")
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Error in Save Project As: {e}")
            QMessageBox.warning(self, "Save Project As", f"Failed to save project:\n{e}")

    @staticmethod
    def _migrate_companion_folders(old_project: Path, new_project: Path):
        """Copy companion folders (_drillholes, _models) when saving a project under a new name.

        Only copies if the old folders exist and the new ones don't yet.
        """
        import shutil

        for suffix in ('_drillholes', '_models'):
            old_folder = old_project.parent / f"{old_project.stem}{suffix}"
            new_folder = new_project.parent / f"{new_project.stem}{suffix}"
            if old_folder.exists() and not new_folder.exists():
                try:
                    shutil.copytree(old_folder, new_folder)
                    logger.info(f"Copied companion folder {old_folder.name} → {new_folder.name}")
                except Exception as e:
                    logger.warning(f"Could not copy companion folder {old_folder.name}: {e}")

    def open_project(self):
        """Open a project file, load data and reapply renderer state."""
        try:
            path_str, _ = QFileDialog.getOpenFileName(self, "Open Project", str(Path.home()), "Block Model Viewer Project (*.bmvproj);;JSON (*.json)")
            if not path_str:
                return
            self._load_project_from_path(Path(path_str))
        except Exception as e:
            logger.error(f"Error opening project: {e}")
            QMessageBox.warning(self, "Open Project", f"Failed to open project:\n{e}")

    def _load_block_model_with_progress(self, file_path: Path, dialog, current_step: int, total_steps: int):
        """
        Load a block model with detailed progress tracking in the loading dialog.

        Blocks until the async load completes using a local QEventLoop so that
        subsequent steps (renderer state, compositing, panel settings) are only
        applied *after* the block model data is available.

        Args:
            file_path: Path to the block model file
            dialog: ProjectLoadingDialog instance
            current_step: Current step counter
            total_steps: Total steps for progress calculation
        """
        # Add to recent files
        self._add_recent_file(file_path)

        # Use controller task system for file loading with progress callback
        if not self.controller:
            QMessageBox.critical(self, "Error", "Controller not available for file loading")
            return

        params = {
            "file_path": file_path
        }

        # Event loop to block until the async task completes
        wait_loop = QEventLoop()

        def progress_callback(progress_percent: int, message: str):
            """Update loading dialog with file loading progress."""
            dialog.set_detailed_progress(
                current_step,
                total_steps,
                f"Loading block model: {message}"
            )

        def on_load_complete(result: Dict[str, Any]):
            """
            Handle file load completion.

            This callback is already called on the main thread via Qt signals,
            so we can update UI directly without QTimer.singleShot.
            """
            try:
                logger.info(f"Block model load complete. Result keys: {list(result.keys()) if result else 'None'}")

                if result is None or result.get("error"):
                    error_msg = result.get("error", "Unknown error") if result else "No result"
                    logger.error(f"Block model load error: {error_msg}")
                    dialog.set_detailed_progress(current_step + 1, total_steps, f"Block model loading failed: {error_msg}")
                    return

                block_model = result.get("block_model")
                if block_model is not None:
                    block_count = getattr(block_model, "block_count", None)
                    if block_count is None:
                        try:
                            block_count = len(block_model)
                        except Exception:
                            block_count = 0
                    logger.info(f"Block model loaded: {block_count} blocks")
                    # Store file path before calling on_file_loaded
                    result_file_path = result.get("file_path")
                    if result_file_path:
                        self.current_file_path = Path(result_file_path)
                    try:
                        self.on_file_loaded(block_model)
                        dialog.set_detailed_progress(current_step + 1, total_steps, f"Block model loaded: {int(block_count):,} blocks")
                    except Exception as e:
                        # Avoid potential recursion in logging by not using exc_info=True
                        logger.error(f"Error in on_file_loaded: {type(e).__name__}: {str(e)}")
                        dialog.set_detailed_progress(current_step + 1, total_steps, f"Block model processing failed: {e}")
                else:
                    logger.error("No block model in result")
                    dialog.set_detailed_progress(current_step + 1, total_steps, "Block model loading failed: No data")
            finally:
                # Always exit the wait loop so the project loading continues
                wait_loop.quit()

        # If the user cancels the dialog, stop waiting
        dialog.cancelled.connect(wait_loop.quit)

        # Run the task with progress callback
        self.controller.run_task('load_file', params, callback=on_load_complete, progress_callback=progress_callback)

        # Block here until on_load_complete fires (keeps UI responsive via event loop)
        wait_loop.exec_()

    def _restore_project_data_without_block_model(self, dialog=None, current_step=None, total_steps=None):
        """Restore project data (drillholes, registry models, renderer state) when no block model is loaded.

        This is called when a project is loaded that either has no block model or the block model file is missing.
        It ensures drillhole data and all registry models are still restored.

        Args:
            dialog: Optional ProjectLoadingDialog for progress updates
            current_step: Current step counter for progress
            total_steps: Total steps for progress calculation
        """
        try:
            # Restore drillhole data FIRST
            if getattr(self, '_pending_drillhole_state', None):
                if dialog and current_step is not None and total_steps is not None:
                    dialog.set_detailed_progress(current_step, total_steps, "Restoring drillhole data...")
                try:
                    self._restore_drillhole_data(self._pending_drillhole_state)
                    self._pending_drillhole_state = None
                    logger.info("Restored drillhole data from project")
                except Exception as e:
                    logger.warning(f"Failed to restore drillhole data: {e}")

            # Restore DataRegistry models (block models, estimation results, etc.)
            if getattr(self, '_pending_registry_models_state', None):
                if dialog and current_step is not None and total_steps is not None:
                    dialog.set_detailed_progress(current_step, total_steps, "Restoring analysis results...")
                try:
                    self._restoring_registry_models = True
                    try:
                        self._restore_registry_models(self._pending_registry_models_state)
                    finally:
                        self._restoring_registry_models = False
                    self._pending_registry_models_state = None
                    logger.info("Restored registry models from project")

                    # Re-broadcast key signals so panels created during load
                    # (or about to be created) discover the restored data.
                    self._rebroadcast_registry_after_restore()
                except Exception as e:
                    logger.warning(f"Failed to restore registry models: {e}")

            # Refresh viewer with the correct current model from registry
            try:
                registry_inst = None
                if hasattr(self, 'controller') and self.controller:
                    registry_inst = getattr(self.controller, 'registry', None)
                if registry_inst is not None:
                    current_model = registry_inst.get_block_model(copy_data=False)
                    if current_model is not None and self.viewer_widget:
                        self.viewer_widget.refresh_scene(current_model)
                        if hasattr(self, 'property_panel') and self.property_panel:
                            self.property_panel.set_block_model(current_model)
                        logger.info("Refreshed viewer with restored block model")
            except Exception as e:
                logger.warning(f"Failed to refresh viewer with restored model: {e}")

            # Rebuild every tagged scene layer from the freshly restored
            # registry (ARBF/kriging/SGSIM/etc).  Runs BEFORE apply_session_state
            # so that layer visibility and active-property restoration land on
            # layers that actually exist in the scene.
            try:
                pending_state = getattr(self, '_pending_session_state', None) or {}
                manifest = pending_state.get('scene_manifest')
                if manifest and self.viewer_widget and self.viewer_widget.renderer:
                    registry_inst = None
                    if hasattr(self, 'controller') and self.controller:
                        registry_inst = getattr(self.controller, 'registry', None)
                    if registry_inst is not None:
                        kind_handlers = {}
                        if hasattr(self, '_handle_classification_visualization'):
                            kind_handlers['classification'] = self._handle_classification_visualization
                        n = self.viewer_widget.renderer.rebuild_scene_from_manifest(
                            manifest, registry_inst, kind_handlers=kind_handlers,
                        )
                        logger.info(f"Scene rebuilder reconstructed {n} tagged layers")
            except Exception as e:
                logger.warning(f"Scene rebuild from manifest failed: {e}", exc_info=True)

            # Apply renderer state LAST — after all data is loaded so that
            # saved properties (kr_Cu, PIT_SHELL, etc.) exist on the model.
            if getattr(self, '_pending_session_state', None):
                if dialog and current_step is not None and total_steps is not None:
                    dialog.set_detailed_progress(current_step, total_steps, "Applying visual settings...")
                try:
                    if self.viewer_widget and self.viewer_widget.renderer:
                        self.viewer_widget.renderer.apply_session_state(self._pending_session_state)
                        self._pending_session_state = None
                        logger.info("Applied renderer session state from project")
                except Exception as e:
                    logger.warning(f"Failed to apply pending session state: {e}")

        except Exception as e:
            logger.error(f"Error restoring project data without block model: {e}", exc_info=True)

    @staticmethod
    def _resolve_project_paths(data: Dict[str, Any], project_path: Path) -> Dict[str, Any]:
        """Resolve file paths stored in a project file relative to the project location.

        When a .bmvproj and its companion folders (_models/, _drillholes/) are
        moved together, the absolute paths inside the JSON become stale.  This
        method detects broken paths and tries to find the file next to the
        current project location using the companion-folder naming convention.

        Mutates *data* in-place and returns it for convenience.
        """
        project_dir = project_path.parent
        project_stem = project_path.stem

        def _try_resolve(abs_path_str: str) -> str:
            """Return *abs_path_str* unchanged if it exists, or a corrected
            path relative to the project file when possible."""
            p = Path(abs_path_str)
            if p.exists():
                return abs_path_str

            # Try to locate the file in the companion folder next to the project
            filename = p.name
            parent_name = p.parent.name  # e.g. "project_models"

            # Determine which companion suffix this belongs to
            for suffix in ('_models', '_drillholes'):
                if parent_name.endswith(suffix):
                    companion = project_dir / f"{project_stem}{suffix}"
                    candidate = companion / filename
                    if candidate.exists():
                        resolved = str(candidate)
                        logger.info(f"Resolved moved project path: {abs_path_str} → {resolved}")
                        return resolved

            # Fallback: just try same filename in the known companion folders
            for suffix in ('_models', '_drillholes'):
                candidate = project_dir / f"{project_stem}{suffix}" / filename
                if candidate.exists():
                    resolved = str(candidate)
                    logger.info(f"Resolved moved project path: {abs_path_str} → {resolved}")
                    return resolved

            return abs_path_str  # unchanged — will be reported as missing later

        def _resolve_dict(d: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            if not d:
                return d
            for key, val in d.items():
                if isinstance(val, str) and ('\\' in val or '/' in val):
                    d[key] = _try_resolve(val)
            return d

        # Resolve paths in each section that contains file references
        if 'data' in data and isinstance(data['data'], dict):
            _resolve_dict(data['data'])
        if 'drillhole_data' in data and isinstance(data['drillhole_data'], dict):
            _resolve_dict(data['drillhole_data'])
        if 'registry_models' in data and isinstance(data['registry_models'], dict):
            _resolve_dict(data['registry_models'])

        return data

    def _load_project_from_path(self, project_path: Path):
        """
        Load a project from file with detailed progress indication.

        Shows a progress dialog with loading steps and displays the processes
        that were run on the project.
        """

        # Clear existing scene, registry, and panels so old project data
        # doesn't bleed into the newly loaded project.
        if hasattr(self, '_clear_for_project_load'):
            self._clear_for_project_load()

        # Create and show progress dialog
        dialog = ProjectLoadingDialog(project_path.stem, self)
        dialog.show()

        # Set up detailed progress tracking
        total_steps = 8  # More granular steps for better progress indication
        dialog.set_total_steps(total_steps)
        current_step = 0

        try:
            # Step 1: Reading project file
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Reading project file...")

            with open(project_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Resolve stale absolute paths when project was moved
            self._resolve_project_paths(data, project_path)

            self.current_project_path = project_path

            # Step 2: Loading project metadata
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Loading project metadata...")

            # Update window title
            try:
                self.setWindowTitle(f"GeoX - {project_path.stem}")
            except Exception:
                pass

            # Restore bookmarks
            try:
                bookmarks = data.get('bookmarks') or {}
                if isinstance(bookmarks, dict):
                    if self.bookmarks is not None:
                        self.bookmarks.set_bookmarks(bookmarks)
                        self.view_bookmarks = self.bookmarks.bookmarks
                    else:
                        self.view_bookmarks = bookmarks
                        self._persist_bookmarks()
            except Exception:
                pass

            # Step 3: Loading process history
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Loading process history...")

            try:
                # Restore process history
                process_history_data = data.get('process_history')
                if process_history_data:
                    process_tracker = get_process_history_tracker()
                    process_tracker.from_dict(process_history_data)
                    logger.info("Restored process history from project")

                    # Display processes in the dialog
                    history = process_tracker.get_history()
                    dialog.set_process_history([p.to_dict() for p in history])
                else:
                    # Clear process history for new/legacy projects
                    process_tracker = get_process_history_tracker()
                    process_tracker.clear_history()
                    dialog.set_process_history([])
            except Exception as e:
                logger.warning(f"Could not restore process history: {e}")
                dialog.set_process_history([])

            # Prepare renderer state to apply after data load
            self._pending_session_state = data.get('renderer_state') or None
            self._pending_drillhole_state = data.get('drillhole_data') or None
            self._pending_registry_models_state = data.get('registry_models') or None
            self._pending_compositing_settings = data.get('compositing_settings') or None
            self._pending_panel_settings = data.get('panel_settings') or {}

            # Step 4-6: Loading data components
            #
            # PRIORITY: Prefer the SAVED block model CSV from the _models/
            # companion folder over the original import file.  The saved CSV
            # contains ALL properties accumulated during the session (kriging,
            # SGSIM, classification, PIT_SHELL, etc.) whereas the original
            # import file only has the raw columns.
            data_section = data.get('data') or {}
            reg_section = data.get('registry_models') or {}

            # 1st choice: saved block model in _models/ (has all properties)
            bm_path = reg_section.get('block_model')
            # 2nd choice: original import file (raw columns only)
            if not bm_path:
                bm_path = data_section.get('block_model')

            if bm_path:
                current_step += 1
                dialog.set_detailed_progress(current_step, total_steps, "Preparing block model data...")

                p = Path(bm_path)
                if p.exists():
                    current_step += 1
                    dialog.set_detailed_progress(current_step, total_steps, "Loading block model file...")

                    # Load block model with progress tracking
                    self._load_block_model_with_progress(p, dialog, current_step, total_steps)
                    current_step += 1  # Increment after block model loading completes
                else:
                    QMessageBox.warning(self, "Open Project", f"Block model file not found:\n{p}")
                    # Still restore other data even if block model not found
                    current_step += 2
                    dialog.set_detailed_progress(current_step, total_steps, "Loading project data (block model missing)...")
                    self._restore_project_data_without_block_model(dialog, current_step, total_steps)
            else:
                # No block model - restore drillhole data, registry models, and renderer state directly
                current_step += 1
                dialog.set_detailed_progress(current_step, total_steps, "Loading drillhole data...")
                current_step += 1
                dialog.set_detailed_progress(current_step, total_steps, "Loading registry models...")
                self._restore_project_data_without_block_model(dialog, current_step, total_steps)

            # Step 7: Applying renderer state (safety fallback — the signal
            # handler or _restore_project_data_without_block_model should have
            # already applied it after all data was restored).
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Applying visual settings...")

            if self._pending_session_state:
                try:
                    if self.viewer_widget and self.viewer_widget.renderer:
                        self.viewer_widget.renderer.apply_session_state(self._pending_session_state)
                        logger.info("Applied saved renderer state (fallback)")
                except Exception as e:
                    logger.warning(f"Could not apply renderer state: {e}")
                finally:
                    self._pending_session_state = None

            # Step 8: Finalizing project load
            current_step += 1
            dialog.set_detailed_progress(current_step, total_steps, "Finalizing project load...")

            # Restore compositing settings if saved
            if self._pending_compositing_settings:
                try:
                    self._restore_compositing_settings(self._pending_compositing_settings)
                    logger.info("Restored compositing settings from project")
                except Exception as e:
                    logger.warning(f"Could not restore compositing settings: {e}")
                finally:
                    self._pending_compositing_settings = None

            # Restore panel settings from all panels
            if self._pending_panel_settings:
                try:
                    self._restore_panel_settings(self._pending_panel_settings)
                    logger.info(f"Restored settings for {len(self._pending_panel_settings)} panels")
                except Exception as e:
                    logger.warning(f"Could not restore panel settings: {e}")
                finally:
                    self._pending_panel_settings = {}

            # Remember last project
            try:
                settings = QSettings("GeoX", "Project")
                settings.setValue("last_project", str(project_path))
            except Exception:
                pass

            self.statusBar().showMessage(f"Project loaded: {project_path.name}", 3000)
            logger.info(f"Loaded project from {project_path}")

            # Loaded project should start clean
            try:
                self._clear_dirty()
            except Exception:
                pass

            # Complete the loading
            dialog.complete_loading()

        except Exception as e:
            logger.error(f"Failed to load project: {e}")
            dialog.close()
            QMessageBox.warning(self, "Open Project", f"Failed to load project:\n{e}")

    def _new_project(self):
        """Start a new project by clearing scene and resetting project path."""
        try:
            self.clear_scene()
            self.current_project_path = None
            self.statusBar().showMessage("New project created", 2000)
            self.setWindowTitle("GeoX")
            try:
                self._clear_dirty()
            except Exception:
                pass
        except Exception as e:
            try:
                e_msg = str(e)
                logger.debug(f"New project error: {e_msg}")
            except Exception:
                logger.debug("New project error: <unprintable error>")

    def _undo(self):
        try:
            if not hasattr(self, '_undo_stack') or not self._undo_stack:
                return
            if not self.viewer_widget or not self.viewer_widget.renderer:
                return
            current = self.viewer_widget.renderer.get_session_state()
            prev = self._undo_stack.pop()
            # Apply previous state and push current to redo
            self.viewer_widget.renderer.apply_session_state(prev)
            if not hasattr(self, '_redo_stack'):
                self._redo_stack = []
            self._redo_stack.append(current)
            # Update UI
            self.undo_action.setEnabled(len(self._undo_stack) > 0)
            self.redo_action.setEnabled(len(self._redo_stack) > 0)
            self.statusBar().showMessage("Undid last change", 1500)
        except Exception as e:
            logger.warning(f"Undo failed: {e}")

    def _redo(self):
        try:
            if not hasattr(self, '_redo_stack') or not self._redo_stack:
                return
            if not self.viewer_widget or not self.viewer_widget.renderer:
                return
            current = self.viewer_widget.renderer.get_session_state()
            nxt = self._redo_stack.pop()
            # Apply next state and push current to undo
            self.viewer_widget.renderer.apply_session_state(nxt)
            if not hasattr(self, '_undo_stack'):
                self._undo_stack = []
            self._undo_stack.append(current)
            # Update UI
            self.undo_action.setEnabled(len(self._undo_stack) > 0)
            self.redo_action.setEnabled(len(self._redo_stack) > 0)
            self.statusBar().showMessage("Redid last change", 1500)
        except Exception as e:
            logger.warning(f"Redo failed: {e}")

