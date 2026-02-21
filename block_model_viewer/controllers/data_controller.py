"""
Data Controller - Handles drillhole, geology, structural, and geotechnical operations.

This controller manages drillhole loading/QAQC, implicit geology modeling,
wireframe generation, structural analysis, and geotechnical computations.
"""

from typing import Optional, Dict, Any, Callable, TYPE_CHECKING
from pathlib import Path
import logging

import numpy as np

if TYPE_CHECKING:
    from .app_controller import AppController

logger = logging.getLogger(__name__)


# Geology module removed - exception classes defined locally
class DataGateError(Exception):
    pass
class ImplicitModellingError(Exception):
    pass


class DataController:
    """
    Controller for data-related operations.
    
    Handles drillhole loading and QAQC, implicit geology modeling, wireframe
    generation, structural analysis (clustering, kinematic), and geotechnical
    analysis (slope stability, stope stability, seismic, rockburst).
    """
    
    def __init__(self, app_controller: "AppController"):
        """
        Initialize data controller.
        
        Args:
            app_controller: Parent AppController instance for shared state access
        """
        self._app = app_controller
    
    @property
    def block_model(self):
        """Return the currently loaded block model."""
        return self._app.block_model
    
    # =========================================================================
    # Drillhole Loading
    # =========================================================================
    
    def _prepare_load_drillholes_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Load Drillholes payload.
        
        SECURITY: Validates all file paths and sizes before loading.
        """
        from ..drillholes.data_io import load_from_csv
        from ..utils.security import validate_file_path, validate_file_size, SecurityError
        
        if progress_callback:
            progress_callback(10, "Validating files...")
        
        # SECURITY: Validate all file paths
        def validate_drillhole_file(file_path: Optional[Path], file_type: str) -> Optional[Path]:
            if file_path is None:
                return None
            try:
                validated = validate_file_path(file_path, must_exist=True)
                validate_file_size(validated, file_type='csv')
                return validated
            except SecurityError as e:
                logger.error(f"Security validation failed for {file_type} file {file_path}: {e}")
                raise ValueError(f"Security validation failed for {file_type} file: {e}")
        
        collar_file = validate_drillhole_file(
            Path(params.get("collar_file", "")) if params.get("collar_file") else None,
            "collar"
        )
        survey_file = validate_drillhole_file(
            Path(params.get("survey_file", "")) if params.get("survey_file") else None,
            "survey"
        )
        assay_file = validate_drillhole_file(
            Path(params.get("assay_file", "")) if params.get("assay_file") else None,
            "assay"
        )
        lithology_file = validate_drillhole_file(
            Path(params.get("lithology_file", "")) if params.get("lithology_file") else None,
            "lithology"
        )
        
        if progress_callback:
            progress_callback(30, "Loading drillhole data...")
        
        db = load_from_csv(
            collar_file=collar_file,
            survey_file=survey_file,
            assay_file=assay_file,
            lithology_file=lithology_file,
            collar_mapping=params.get("collar_mapping"),
            survey_mapping=params.get("survey_mapping"),
            assay_mapping=params.get("assay_mapping"),
            lithology_mapping=params.get("lithology_mapping"),
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "load_drillholes",
            "database": db,
            "hole_count": len(db.get_hole_ids()),
        }
    
    def _prepare_drillhole_import_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare drillhole import payload - pure computation.
        
        Processes drillhole data (assays, collars, surveys, lithology) by:
        1. Standardizing column names
        2. Computing 3D coordinates using Minimum Curvature desurveying
        3. Packaging into drillhole_data structure
        
        This is a pure function with no access to DataRegistry, controller, or globals.
        All data is passed in params and returned in result.
        """
        import pandas as pd
        import numpy as np
        import traceback
        from ..utils.desurvey import minimum_curvature_desurvey, interpolate_at_depths
        
        try:
            logger.info("=" * 80)
            logger.info("DRILLHOLE IMPORT: Starting payload preparation")
            logger.info("=" * 80)
            
            if progress_callback:
                progress_callback(10, "Validating data structures...")
            
            # Extract data from params
            logger.debug("STEP 1: Extracting DataFrames from params")
            assay_df = params.get('assay_df')
            collar_df = params.get('collar_df')
            survey_df = params.get('survey_df')
            lithology_df = params.get('lithology_df')
            structures_df = params.get('structures_df')
            
            logger.info(f"Data received - Assays: {assay_df is not None and not assay_df.empty if assay_df is not None else False}, "
                       f"Collars: {collar_df is not None and not collar_df.empty if collar_df is not None else False}, "
                       f"Surveys: {survey_df is not None and not survey_df.empty if survey_df is not None else False}, "
                       f"Lithology: {lithology_df is not None and not lithology_df.empty if lithology_df is not None else False}, "
                       f"Structures: {structures_df is not None and not structures_df.empty if structures_df is not None else False}")
            
            if assay_df is not None:
                logger.debug(f"Assay DataFrame shape: {assay_df.shape}, columns: {list(assay_df.columns)}")
            if collar_df is not None:
                logger.debug(f"Collar DataFrame shape: {collar_df.shape}, columns: {list(collar_df.columns)}")
            if survey_df is not None:
                logger.debug(f"Survey DataFrame shape: {survey_df.shape}, columns: {list(survey_df.columns)}")
            if lithology_df is not None:
                logger.debug(f"Lithology DataFrame shape: {lithology_df.shape}, columns: {list(lithology_df.columns)}")
        
            # Standardize column names
            logger.debug("STEP 2: Standardizing column names")
            def clean_cols(df, df_name="DataFrame"):
                """Standardize column names to uppercase and normalize."""
                try:
                    logger.debug(f"  Cleaning {df_name}: {df.shape[0]} rows, {df.shape[1]} columns")
                    df = df.copy()
                    original_cols = list(df.columns)
                    df.columns = [c.upper().strip() for c in df.columns]
                    rename_map = {
                        'HOLE_ID': 'HOLEID', 'BHID': 'HOLEID', 
                        'DEPTH_FROM': 'FROM', 'DEPTH_TO': 'TO',
                        'X': 'X', 'EAST': 'X', 'EASTING': 'X',
                        'Y': 'Y', 'NORTH': 'Y', 'NORTHING': 'Y',
                        'Z': 'Z', 'ELEV': 'Z', 'RL': 'Z'
                    }
                    df.rename(columns=rename_map, inplace=True)
                    logger.debug(f"  {df_name} cleaned: {list(df.columns)}")
                    return df
                except Exception as e:
                    logger.error(f"ERROR cleaning {df_name}: {e}", exc_info=True)
                    raise
            
            if collar_df is not None:
                collar_df = clean_cols(collar_df, "Collars")
            if survey_df is not None:
                survey_df = clean_cols(survey_df, "Surveys")
            if lithology_df is not None:
                lithology_df = clean_cols(lithology_df, "Lithology")
        
            # Process assays with coordinates
            logger.debug("STEP 3: Processing assays with coordinates")
            assays_with_coords = None
            if assay_df is not None and not assay_df.empty:
                try:
                    if progress_callback:
                        progress_callback(30, "Desurveying drillholes (Minimum Curvature)...")
                    
                    logger.info(f"Processing {len(assay_df)} assay rows")
                    assay_df = clean_cols(assay_df.copy(), "Assays")
                    
                    # Validate required columns
                    if 'HOLEID' not in assay_df.columns:
                        raise ValueError("Assay DataFrame missing HOLEID column after cleaning")
                    if 'FROM' not in assay_df.columns or 'TO' not in assay_df.columns:
                        raise ValueError("Assay DataFrame missing FROM/TO columns after cleaning")
                    
                    logger.debug("  Adding coordinate columns")
                    assay_df['X'] = 0.0
                    assay_df['Y'] = 0.0
                    assay_df['Z'] = 0.0
                    assay_df['_MID'] = (assay_df['FROM'] + assay_df['TO']) / 2.0
                    
                    # Process each hole
                    logger.debug("  Grouping assays by HOLEID")
                    grouped_assays = assay_df.groupby('HOLEID')
                    num_holes = len(grouped_assays)
                    logger.info(f"  Found {num_holes} unique holes")
                    
                    collar_idx = collar_df.set_index('HOLEID') if collar_df is not None and not collar_df.empty else pd.DataFrame()
                    logger.debug(f"  Collar index: {len(collar_idx)} entries")

                    # BUG FIX #14: Ensure collar index uses string type for consistent lookup
                    if not collar_idx.empty:
                        collar_idx.index = collar_idx.index.astype(str)

                    results = []
                    hole_count = 0
                    failed_holes = []  # BUG FIX #5: Track failed holes for reporting

                    for hid, group in grouped_assays:
                        hole_count += 1
                        if hole_count % 100 == 0:
                            logger.debug(f"  Processing hole {hole_count}/{num_holes}: {hid}")
                        try:
                            # Make a copy of the group to avoid SettingWithCopyWarning
                            group = group.copy()

                            # BUG FIX #14: Convert hid to string for consistent lookup
                            hid_str = str(hid)

                            if collar_df is None or collar_df.empty or collar_idx.empty or hid_str not in collar_idx.index:
                                # No collar - keep default coords (0, 0, 0)
                                logger.debug(f"    Hole {hid}: No collar found, using default coordinates")
                                failed_holes.append({'hole_id': hid, 'reason': 'no_collar', 'rows': len(group)})
                                results.append(group)
                                continue

                            collar = collar_idx.loc[hid_str]  # BUG FIX #14: Use hid_str
                            x0 = float(collar['X'])
                            y0 = float(collar['Y'])
                            z0 = float(collar['Z'])
                            logger.debug(f"    Hole {hid}: Collar at ({x0}, {y0}, {z0}), {len(group)} intervals")

                            # Desurvey using Minimum Curvature
                            # BUG FIX #14: Convert survey HOLEID to string for consistent comparison
                            survey_hole_ids = survey_df['HOLEID'].astype(str).values if survey_df is not None and not survey_df.empty else []
                            if survey_df is not None and not survey_df.empty and hid_str in survey_hole_ids:
                                h_surv = survey_df[survey_df['HOLEID'].astype(str) == hid_str]
                                logger.debug(f"    Hole {hid}: Found {len(h_surv)} survey points")
                                depths, xs, ys, zs = minimum_curvature_desurvey(x0, y0, z0, h_surv)
                                
                                if depths is not None:
                                    logger.debug(f"    Hole {hid}: Desurvey successful, interpolating coordinates")
                                    # Interpolate coordinates at midpoints
                                    mids = group['_MID'].values
                                    interp_xs, interp_ys, interp_zs = interpolate_at_depths(
                                        depths, xs, ys, zs, mids
                                    )
                                    group['X'] = interp_xs
                                    group['Y'] = interp_ys
                                    group['Z'] = interp_zs
                                else:
                                    logger.warning(f"    Hole {hid}: Desurvey returned None, using vertical fallback")
                                    # Fallback to vertical
                                    group['X'] = x0
                                    group['Y'] = y0
                                    group['Z'] = z0 - group['_MID']
                            else:
                                logger.debug(f"    Hole {hid}: No survey data, using vertical assumption")
                                # No survey - assume vertical
                                group['X'] = x0
                                group['Y'] = y0
                                group['Z'] = z0 - group['_MID']
                            
                            results.append(group)
                        except Exception as e:
                            logger.error(f"ERROR processing hole {hid}: {e}", exc_info=True)
                            # BUG FIX #5: Track failed holes instead of silently using default coords
                            failed_holes.append({'hole_id': hid, 'reason': str(e), 'rows': len(group)})
                            # Still append group to maintain row count, but with default coords
                            results.append(group)

                    # BUG FIX #5: Log summary of failed holes
                    if failed_holes:
                        total_failed_rows = sum(h['rows'] for h in failed_holes)
                        logger.warning(f"  {len(failed_holes)} holes failed coordinate calculation ({total_failed_rows} rows affected)")
                        for fh in failed_holes[:5]:  # Log first 5
                            logger.warning(f"    - Hole {fh['hole_id']}: {fh['reason']} ({fh['rows']} rows)")
                        if len(failed_holes) > 5:
                            logger.warning(f"    ... and {len(failed_holes) - 5} more")

                    # Only concat if we have results
                    logger.debug("STEP 4: Concatenating results")
                    if results:
                        logger.info(f"  Concatenating {len(results)} hole groups")
                        assays_with_coords = pd.concat(results, ignore_index=True)
                        logger.info(f"  Final assays DataFrame: {len(assays_with_coords)} rows")

                        # BUG FIX #13: Validate that coordinates were actually computed
                        zero_coord_count = ((assays_with_coords['X'] == 0) &
                                           (assays_with_coords['Y'] == 0) &
                                           (assays_with_coords['Z'] == 0)).sum()
                        if zero_coord_count > 0:
                            pct = zero_coord_count / len(assays_with_coords) * 100
                            logger.warning(f"  WARNING: {zero_coord_count} rows ({pct:.1f}%) have (0,0,0) coordinates")
                            if pct > 50:
                                logger.error(f"  CRITICAL: More than 50% of rows have default coordinates - check collar/survey data!")
                        if '_MID' in assays_with_coords.columns:
                            assays_with_coords.drop(columns=['_MID'], inplace=True)
                    else:
                        logger.warning("  No results to concatenate, using original DataFrame")
                        # No results - return empty DataFrame with same structure
                        assays_with_coords = assay_df.copy()
                        if '_MID' in assays_with_coords.columns:
                            assays_with_coords.drop(columns=['_MID'], inplace=True)
                except Exception as e:
                    logger.error(f"ERROR processing assays: {e}", exc_info=True)
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    raise
            else:
                logger.info("No assay data to process")
            
            if progress_callback:
                progress_callback(80, "Packaging datasets...")
            
            logger.debug("STEP 5: Packaging data")
            # Package data
            drillhole_data = {
                "collars": collar_df,
                "surveys": survey_df,
                "assays": assays_with_coords,
                "lithology": lithology_df,
                "structures": structures_df,
                "composites": None  # Placeholder for future compositing
            }
            
            metadata = {
                "collar_count": len(collar_df) if collar_df is not None else 0,
                "assay_rows": len(assay_df) if assay_df is not None else 0,
                "has_surveys": survey_df is not None and not survey_df.empty
            }
            
            logger.info(f"Packaged data - Collars: {metadata['collar_count']}, Assays: {metadata['assay_rows']}, Has Surveys: {metadata['has_surveys']}")
            
            if progress_callback:
                progress_callback(100, "Done.")
            
            logger.info("=" * 80)
            logger.info("DRILLHOLE IMPORT: Payload preparation completed successfully")
            logger.info("=" * 80)
            
            return {
                "name": "drillhole_import",
                "drillhole_data": drillhole_data,
                "metadata": metadata
            }
        except Exception as e:
            error_msg = f"Drillhole import failed: {str(e)}"
            error_traceback = traceback.format_exc()
            logger.error("=" * 80)
            logger.error("DRILLHOLE IMPORT: FAILED")
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error(error_traceback)
            logger.error("=" * 80)
            
            if progress_callback:
                progress_callback(100, f"ERROR: {error_msg}")
            
            return {
                "name": "drillhole_import",
                "error": error_msg,
                "traceback": error_traceback,
                "drillhole_data": None,
                "metadata": {}
            }
    
    def _prepare_build_block_model_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Build Block Model payload - pure computation.
        
        Builds a regular 3D block model from estimation/kriging results.
        This is a pure function with no access to DataRegistry, controller, or globals.
        """
        from ..models import blockmodel_builder as bmb
        
        if progress_callback:
            progress_callback(10, "Initializing...")
        
        estimation_df = params.get('estimation_df')
        if estimation_df is None:
            raise ValueError("estimation_df is required")
        
        xinc = params.get('xinc', 25.0)
        yinc = params.get('yinc', 25.0)
        zinc = params.get('zinc', 10.0)
        grade_col = params.get('grade_col', 'Fe_est')
        var_col = params.get('var_col')
        max_blocks = params.get('max_blocks', 100000)
        extents = params.get('extents')  # Optional: (xmin, xmax, ymin, ymax, zmin, zmax)
        
        block_df, grid_def, info = bmb.build_block_model(
            estimation_df, 'X', 'Y', 'Z', grade_col, var_col,
            xinc, yinc, zinc,
            extents=extents,
            max_blocks=max_blocks
        )
        
        if progress_callback:
            progress_callback(100, "Done.")
        
        return {
            "name": "build_block_model",
            "block_df": block_df,
            "grid_def": grid_def,  # Grid definition for PyVista creation
            "info": info,  # Contains cell_data and _create_grid_in_main_thread flag
            "_create_grid_in_main_thread": True  # Flag to create PyVista grid in main thread
        }
    
    def _prepare_drillhole_database_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Drillhole Database payload - pure computation.
        
        Handles database statistics and other database operations.
        """
        operation = params.get("operation")
        
        if operation == "get_statistics":
            if progress_callback:
                progress_callback(10, "Loading database statistics...")
            
            db_manager = params.get("db_manager")
            project = params.get("project")
            
            if db_manager is None or project is None:
                raise ValueError("db_manager and project are required")
            
            # Get statistics (pure operation - no registry access)
            stats = db_manager.get_statistics(project)
            
            if progress_callback:
                progress_callback(100, "Statistics loaded")
            
            return {
                "name": "drillhole_database_statistics",
                "operation": "get_statistics",
                "stats": stats
            }
        else:
            raise ValueError(f"Unknown drillhole database operation: {operation}")
    
    def _prepare_load_file_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare File Load payload - pure computation.
        
        Loads block model files using parser registry.
        
        SECURITY: Validates file path and size before loading.
        """
        from ..parsers import parser_registry
        from ..utils.security import validate_file_path, validate_file_size, SecurityError
        
        # FREEZE PROTECTION: Log worker start
        logger.debug("WORKER START: _prepare_load_file_payload")
        
        if progress_callback:
            progress_callback(10, "Validating file...")
        
        file_path = params.get("file_path")
        if file_path is None:
            raise ValueError("file_path is required")
        
        # Convert to Path if needed
        file_path = Path(file_path)
        
        # SECURITY: Validate path and file size
        try:
            validated_path = validate_file_path(file_path, must_exist=True)
            file_size = validate_file_size(validated_path, file_type='csv')
            logger.info(f"Loading file: {validated_path} ({file_size / (1024*1024):.1f} MB)")
        except SecurityError as e:
            logger.error(f"Security validation failed: {e}")
            raise ValueError(f"File security validation failed: {e}")
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        
        if progress_callback:
            progress_callback(20, "Parsing file...")
        
        # Parse file
        block_model = parser_registry.parse_file(validated_path)
        
        logger.info(f"Parsed file: {block_model.block_count} blocks")
        
        if progress_callback:
            progress_callback(90, "Validating model...")
        
        # Validate
        errors = block_model.validate()
        if errors:
            raise ValueError(f"Validation errors: {', '.join(errors)}")
        
        if progress_callback:
            progress_callback(100, "File loaded")
        
        # FREEZE PROTECTION: Log worker end
        logger.debug("WORKER END: _prepare_load_file_payload")
        
        return {
            "name": "load_file",
            "block_model": block_model,
            "file_path": str(file_path)
        }
    
    # =========================================================================
    # Drillhole QAQC
    # =========================================================================
    
    def _prepare_drillhole_qaqc_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Drillhole QAQC payload."""
        if progress_callback:
            progress_callback(10, "Running QAQC checks...")
        
        db = params.get("database")
        if db is None:
            raise ValueError("Database required for QAQC")
        
        # QAQC functionality placeholder
        results = []
        messages = []
        warning_count = 0
        error_count = 0
        quality_score = None
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "drillhole_qaqc",
            "messages": messages,
            "warning_count": warning_count,
            "error_count": error_count,
            "quality_score": quality_score,
            "qc_results": results,
        }
    
    # =========================================================================
    # Implicit Geology
    # =========================================================================
    
    def _prepare_implicit_geology_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Implicit Geology payload."""
        from ..geology.implicit_model import ImplicitGeologyModel
        
        if progress_callback:
            progress_callback(10, "Building implicit geology model...")
        
        contact_set = params.get("contact_set")
        domain_model = params.get("domain_model")
        config = params.get("config", {})
        
        if contact_set is None or domain_model is None:
            raise ValueError("ContactSet and DomainModel required")
        
        model = ImplicitGeologyModel(contact_set, domain_model, config)
        
        grid_definition = params.get("grid_definition")
        scalar_field = None
        if grid_definition:
            try:
                scalar_field = model.build_scalar_field(grid_definition)
            except Exception as e:
                logger.warning(f"Failed to build scalar field: {e}")
                scalar_field = None
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "implicit_geology",
            "model": model,
            "scalar_field": scalar_field,
            "grid_definition": grid_definition,
        }
    
    # =========================================================================
    # Wireframes
    # =========================================================================
    
    def _prepare_build_wireframes_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Build Wireframes payload."""
        from ..geology.wireframes import extract_isosurface_from_scalar_field
        
        if progress_callback:
            progress_callback(10, "Extracting wireframes...")
        
        scalar_field = params.get("scalar_field")
        grid_definition = params.get("grid_definition")
        level = params.get("level", 0.0)
        
        if scalar_field is None or grid_definition is None:
            raise ValueError("Scalar field and grid definition required")
        
        wireframe = extract_isosurface_from_scalar_field(scalar_field, grid_definition, level)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "build_wireframes",
            "wireframe": wireframe,
            "vertex_count": len(wireframe.vertices),
            "face_count": len(wireframe.faces),
        }
    
    # =========================================================================
    # Structural Clustering (using geox.structural.core engine)
    # =========================================================================
    
    def _prepare_structural_clusters_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Structural Clusters payload using geox.structural.core engine.
        
        Returns auditable result bundle with set statistics.
        """
        import time
        import uuid
        from datetime import datetime
        
        start_time = time.time()
        
        if progress_callback:
            progress_callback(10, "Preparing structural data for clustering...")
        
        dataset = params.get("dataset")
        normals = params.get("normals")  # Alternative: direct normals input
        n_sets = params.get("n_sets")
        method = params.get("method", "auto")
        min_cluster_size = params.get("min_cluster_size", 5)
        random_seed = params.get("random_seed", 42)
        
        # Convert dataset planes to normals if provided
        if normals is None and dataset is not None:
            from ..structural.datasets import PlaneMeasurement
            try:
                from geox.structural.core import dip_dipdir_to_normal
                dips = np.array([p.dip for p in dataset.planes])
                dip_dirs = np.array([p.dip_direction for p in dataset.planes])
                normals = dip_dipdir_to_normal(dips, dip_dirs)
            except ImportError:
                # Fallback to old implementation
                from ..structural.stereonet import cluster_planes
                set_ids = cluster_planes(dataset.planes, n_sets or 3)
                for i, plane in enumerate(dataset.planes):
                    plane.set_id = set_ids[i]
                return {
                    "name": "structural_clusters",
                    "dataset": dataset,
                    "n_sets": n_sets,
                    "set_ids": set_ids,
                }
        
        if normals is None:
            raise ValueError("StructuralDataset or normals array required")
        
        if progress_callback:
            progress_callback(30, "Running clustering algorithm...")
        
        # Use new engine
        try:
            from geox.structural.core import (
                OrientationData, 
                cluster_orientations,
                AnalysisBundle,
            )
            
            # Create OrientationData
            orientation_data = OrientationData(normals=normals)
            
            # Run clustering
            labels, structural_sets = cluster_orientations(
                orientation_data,
                method=method,
                n_clusters=n_sets,
                min_cluster_size=min_cluster_size,
                random_seed=random_seed,
            )
            
            if progress_callback:
                progress_callback(80, "Computing set statistics...")
            
            # Update dataset if provided
            if dataset is not None:
                for i, label in enumerate(labels):
                    if label >= 0 and i < len(dataset.planes):
                        dataset.planes[i].set_id = f"set_{label}"
            
            # Build audit bundle
            execution_time_ms = (time.time() - start_time) * 1000
            
            bundle = AnalysisBundle(
                analysis_id=str(uuid.uuid4()),
                analysis_type="clustering",
                timestamp=datetime.now(),
                n_measurements=len(normals),
                parameters={
                    "method": method,
                    "n_sets": n_sets,
                    "min_cluster_size": min_cluster_size,
                    "random_seed": random_seed,
                },
                structural_sets=structural_sets,
                engine_version="1.0.0",
                random_seed=random_seed,
                execution_time_ms=execution_time_ms,
            )
            
            if progress_callback:
                progress_callback(100, "Complete")
            
            return {
                "name": "structural_clusters",
                "dataset": dataset,
                "n_sets": len(structural_sets),
                "set_ids": [f"set_{l}" for l in labels],
                "labels": labels.tolist(),
                "structural_sets": [s.to_dict() for s in structural_sets],
                "audit_bundle": bundle.to_dict(),
            }
            
        except ImportError:
            # Fallback to old implementation
            logger.warning("geox.structural.core not available, using legacy clustering")
            from ..structural.stereonet import cluster_planes
            set_ids = cluster_planes(dataset.planes, n_sets or 3)
            for i, plane in enumerate(dataset.planes):
                plane.set_id = set_ids[i]
            return {
                "name": "structural_clusters",
                "dataset": dataset,
                "n_sets": n_sets,
                "set_ids": set_ids,
            }
    
    # =========================================================================
    # Kinematic Analysis (using geox.structural.core engine)
    # =========================================================================
    
    def _prepare_kinematic_analysis_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Kinematic Analysis payload using geox.structural.core engine.
        
        Returns auditable result bundle with detailed feasibility checks.
        """
        import time
        import uuid
        from datetime import datetime
        
        start_time = time.time()
        
        if progress_callback:
            progress_callback(10, "Preparing kinematic analysis...")
        
        dataset = params.get("dataset")
        normals = params.get("normals")  # Alternative: direct normals input
        slope_dip = params.get("slope_dip", 45.0)
        slope_dip_direction = params.get("slope_dip_direction", 0.0)
        phi = params.get("phi", 35.0)
        analysis_type = params.get("analysis_type", "plane")  # plane, wedge, toppling, all
        lateral_limits = params.get("lateral_limits", 20.0)
        
        # Convert dataset planes to normals if provided
        if normals is None and dataset is not None:
            try:
                from geox.structural.core import dip_dipdir_to_normal
                dips = np.array([p.dip for p in dataset.planes])
                dip_dirs = np.array([p.dip_direction for p in dataset.planes])
                normals = dip_dipdir_to_normal(dips, dip_dirs)
            except ImportError:
                # Fallback to old implementation
                from ..structural.kinematic_analysis import kinematic_plane_slope_feasibility, kinematic_wedge_feasibility
                if analysis_type == "plane":
                    result = kinematic_plane_slope_feasibility(dataset.planes, slope_dip, slope_dip_direction, phi)
                elif analysis_type == "wedge":
                    result = kinematic_wedge_feasibility(dataset.planes, slope_dip, slope_dip_direction, phi)
                else:
                    raise ValueError(f"Unknown analysis type: {analysis_type}")
                return {
                    "name": "kinematic_analysis",
                    "result": result,
                    "analysis_type": analysis_type,
                }
        
        if normals is None:
            raise ValueError("StructuralDataset or normals array required")
        
        if progress_callback:
            progress_callback(30, f"Running {analysis_type} analysis...")
        
        # Use new engine
        try:
            from geox.structural.core import (
                OrientationData,
                kinematic_analysis,
                planar_sliding_feasibility,
                wedge_sliding_feasibility,
                toppling_feasibility,
                AnalysisBundle,
                summarize_kinematic_results,
            )
            
            orientation_data = OrientationData(normals=normals)
            
            # Run appropriate analysis
            if analysis_type == "all":
                result = kinematic_analysis(
                    orientation_data,
                    slope_dip=slope_dip,
                    slope_dip_direction=slope_dip_direction,
                    friction_angle=phi,
                    lateral_limits=lateral_limits,
                    analyze_planar=True,
                    analyze_wedge=True,
                    analyze_toppling=True,
                )
            else:
                result = kinematic_analysis(
                    orientation_data,
                    slope_dip=slope_dip,
                    slope_dip_direction=slope_dip_direction,
                    friction_angle=phi,
                    lateral_limits=lateral_limits,
                    analyze_planar=(analysis_type == "plane"),
                    analyze_wedge=(analysis_type == "wedge"),
                    analyze_toppling=(analysis_type == "toppling"),
                )
            
            if progress_callback:
                progress_callback(80, "Computing summary statistics...")
            
            # Get summary
            summary = summarize_kinematic_results(result)
            
            # Build audit bundle
            execution_time_ms = (time.time() - start_time) * 1000
            
            bundle = AnalysisBundle(
                analysis_id=str(uuid.uuid4()),
                analysis_type=f"kinematic_{analysis_type}",
                timestamp=datetime.now(),
                n_measurements=len(normals),
                parameters={
                    "slope_dip": slope_dip,
                    "slope_dip_direction": slope_dip_direction,
                    "friction_angle": phi,
                    "lateral_limits": lateral_limits,
                    "analysis_type": analysis_type,
                },
                kinematic_result=result,
                engine_version="1.0.0",
                execution_time_ms=execution_time_ms,
            )
            
            if progress_callback:
                progress_callback(100, "Complete")
            
            # Build backward-compatible result format
            legacy_result = {
                "feasible_count": result.n_planar_feasible if analysis_type == "plane" else (
                    result.n_wedge_feasible if analysis_type == "wedge" else result.n_toppling_feasible
                ),
                "total_count": result.n_total_measurements,
                "feasible_fraction": (
                    result.planar_feasible_fraction if analysis_type == "plane" else (
                        result.wedge_feasible_fraction if analysis_type == "wedge" else result.toppling_feasible_fraction
                    )
                ),
            }
            
            return {
                "name": "kinematic_analysis",
                "result": legacy_result,  # Backward compatible
                "kinematic_result": result.to_dict(),  # Full result
                "summary": summary,
                "analysis_type": analysis_type,
                "audit_bundle": bundle.to_dict(),
            }
            
        except ImportError:
            # Fallback to old implementation
            logger.warning("geox.structural.core not available, using legacy kinematic analysis")
            from ..structural.kinematic_analysis import kinematic_plane_slope_feasibility, kinematic_wedge_feasibility
            
            if analysis_type == "plane":
                result = kinematic_plane_slope_feasibility(dataset.planes, slope_dip, slope_dip_direction, phi)
            elif analysis_type == "wedge":
                result = kinematic_wedge_feasibility(dataset.planes, slope_dip, slope_dip_direction, phi)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            if progress_callback:
                progress_callback(100, "Complete")
            
            return {
                "name": "kinematic_analysis",
                "result": result,
                "analysis_type": analysis_type,
            }
    
    # =========================================================================
    # Stereonet Analysis (using geox.structural.core engine)
    # =========================================================================
    
    def _prepare_stereonet_analysis_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Stereonet Analysis payload using geox.structural.core engine.
        
        Returns projected coordinates, density grid, and audit bundle.
        """
        import time
        import uuid
        from datetime import datetime
        
        start_time = time.time()
        
        if progress_callback:
            progress_callback(10, "Preparing stereonet projection...")
        
        dataset = params.get("dataset")
        normals = params.get("normals")
        net_type = params.get("net_type", "schmidt")
        hemisphere = params.get("hemisphere", "lower")
        compute_density = params.get("compute_density", True)
        density_bandwidth = params.get("density_bandwidth")
        show_planes = params.get("show_planes", False)
        
        # Convert dataset planes to normals if provided
        if normals is None and dataset is not None:
            try:
                from geox.structural.core import dip_dipdir_to_normal
                dips = np.array([p.dip for p in dataset.planes])
                dip_dirs = np.array([p.dip_direction for p in dataset.planes])
                normals = dip_dipdir_to_normal(dips, dip_dirs)
            except ImportError:
                raise ImportError("geox.structural.core not available")
        
        if normals is None:
            raise ValueError("StructuralDataset or normals array required")
        
        try:
            from geox.structural.core import (
                project_schmidt,
                project_wulff,
                spherical_kde_grid,
                compute_great_circle,
                NetType,
                Hemisphere,
                StereonetResult,
                AnalysisBundle,
            )
            
            if progress_callback:
                progress_callback(30, "Projecting poles...")
            
            # Select projection
            net_type_enum = NetType.SCHMIDT if net_type.lower() == "schmidt" else NetType.WULFF
            hemisphere_enum = Hemisphere.LOWER if hemisphere.lower() == "lower" else Hemisphere.UPPER
            
            if net_type_enum == NetType.SCHMIDT:
                x, y = project_schmidt(normals, hemisphere_enum)
            else:
                x, y = project_wulff(normals, hemisphere_enum)
            
            # Compute density if requested
            density_grid = None
            density_x = None
            density_y = None
            
            if compute_density and len(normals) > 2:
                if progress_callback:
                    progress_callback(50, "Computing density contours...")
                
                lat_grid, lon_grid, density = spherical_kde_grid(
                    normals, 
                    bandwidth=density_bandwidth
                )
                density_grid = density
                
                # Convert lat/lon to stereonet coordinates for plotting
                from geox.structural.core import dip_dipdir_to_normal
                grid_normals = []
                for i in range(lat_grid.shape[0]):
                    for j in range(lat_grid.shape[1]):
                        dip = lat_grid[i, j]
                        dd = lon_grid[i, j]
                        grid_normals.append(dip_dipdir_to_normal(dip, dd).flatten())
                grid_normals = np.array(grid_normals)
                
                if net_type_enum == NetType.SCHMIDT:
                    density_x, density_y = project_schmidt(grid_normals, hemisphere_enum)
                else:
                    density_x, density_y = project_wulff(grid_normals, hemisphere_enum)
                
                density_x = density_x.reshape(lat_grid.shape)
                density_y = density_y.reshape(lat_grid.shape)
            
            # Compute great circles if requested
            great_circles = None
            if show_planes:
                if progress_callback:
                    progress_callback(70, "Computing plane traces...")
                great_circles = []
                for normal in normals:
                    gx, gy = compute_great_circle(normal, net_type=net_type_enum, hemisphere=hemisphere_enum)
                    great_circles.append((gx.tolist(), gy.tolist()))
            
            if progress_callback:
                progress_callback(90, "Building result...")
            
            # Build result
            stereonet_result = StereonetResult(
                x=x,
                y=y,
                net_type=net_type_enum,
                hemisphere=hemisphere_enum,
                show_planes=show_planes,
                density_grid=density_grid,
                density_x=density_x,
                density_y=density_y,
                great_circles=great_circles,
                n_points=len(normals),
                parameters={
                    "net_type": net_type,
                    "hemisphere": hemisphere,
                    "compute_density": compute_density,
                    "density_bandwidth": density_bandwidth,
                }
            )
            
            # Build audit bundle
            execution_time_ms = (time.time() - start_time) * 1000
            
            bundle = AnalysisBundle(
                analysis_id=str(uuid.uuid4()),
                analysis_type="stereonet",
                timestamp=datetime.now(),
                n_measurements=len(normals),
                parameters=stereonet_result.parameters,
                stereonet_result=stereonet_result,
                engine_version="1.0.0",
                execution_time_ms=execution_time_ms,
            )
            
            if progress_callback:
                progress_callback(100, "Complete")
            
            return {
                "name": "stereonet_analysis",
                "x": x.tolist(),
                "y": y.tolist(),
                "n_points": len(normals),
                "net_type": net_type,
                "hemisphere": hemisphere,
                "density_grid": density_grid.tolist() if density_grid is not None else None,
                "great_circles": great_circles,
                "stereonet_result": stereonet_result.to_dict(),
                "audit_bundle": bundle.to_dict(),
            }
            
        except ImportError as e:
            raise ImportError(f"geox.structural.core not available: {e}")
    
    # =========================================================================
    # Geotechnical Interpolation
    # =========================================================================
    
    def _prepare_geotech_interpolation_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare geotechnical interpolation payload."""
        from ..geotech.interpolation import interpolate_to_block_model
        from ..geotech.rock_mass_model import RockMassModel
        
        if not self.block_model:
            raise ValueError("No block model loaded")
        
        rock_mass_model = params.get('rock_mass_model')
        if rock_mass_model is None:
            data_path = params.get('data_path')
            if data_path:
                rock_mass_model = RockMassModel()
                rock_mass_model.load_from_csv(data_path)
            else:
                raise ValueError("No rock mass model or data path provided")
        
        if progress_callback:
            progress_callback(50, "Interpolating geotechnical properties...")
        
        grid = interpolate_to_block_model(rock_mass_model, self.block_model, params)
        
        variable = params.get('variable', 'RMR')
        property_array = grid.get_property(variable)
        
        if property_array is not None:
            property_name = f"Geotech_{variable}"
            self.block_model.add_property(property_name, property_array)
        else:
            property_name = None
        
        visualization = {
            "property": property_name,
            "layer_name": f"Geotech_{variable}"
        }
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "geotech_interpolation",
            "property_name": property_name,
            "grid": grid,
            "summary_stats": rock_mass_model.get_summary_statistics(),
            "visualization": visualization
        }
    
    # =========================================================================
    # Stope Stability
    # =========================================================================
    
    def _prepare_stope_stability_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare stope stability analysis payload."""
        from ..geotech.stope_stability import evaluate_stope
        from ..geotech.dataclasses import StopeStabilityInput
        
        input_data = StopeStabilityInput(
            span=params['span'],
            height=params['height'],
            q_prime=params.get('q_prime', 1.0),
            stress_factor=params.get('stress_factor', 1.0),
            joint_orientation_factor=params.get('joint_orientation_factor', 1.0),
            gravity_factor=params.get('gravity_factor', 1.0),
            dilution_allowance=params.get('dilution_allowance', 0.0),
            rock_mass_properties=params.get('rock_mass_properties')
        )
        
        if progress_callback:
            progress_callback(50, "Evaluating stope stability...")
        
        result = evaluate_stope(input_data)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "stope_stability",
            "result": {
                "stability_number": result.stability_number,
                "factor_of_safety": result.factor_of_safety,
                "probability_of_instability": result.probability_of_instability,
                "stability_class": result.stability_class,
                "recommended_support_class": result.recommended_support_class,
                "notes": result.notes
            },
            "input": {
                "span": input_data.span,
                "height": input_data.height,
                "hydraulic_radius": input_data.hydraulic_radius
            }
        }
    
    def _prepare_stope_stability_mc_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare probabilistic stope stability analysis payload."""
        from ..geotech.probabilistic_geotech import run_stope_stability_monte_carlo
        from ..geotech.dataclasses import StopeStabilityInput
        
        input_data = StopeStabilityInput(
            span=params['span'],
            height=params['height'],
            q_prime=params.get('q_prime', 1.0),
            stress_factor=params.get('stress_factor', 1.0),
            joint_orientation_factor=params.get('joint_orientation_factor', 1.0),
            gravity_factor=params.get('gravity_factor', 1.0),
            dilution_allowance=params.get('dilution_allowance', 0.0),
            rock_mass_properties=params.get('rock_mass_properties')
        )
        
        n_realizations = params.get('n_realizations', 100)
        
        if progress_callback:
            progress_callback(10, f"Running {n_realizations} Monte Carlo realizations...")
        
        mc_result = run_stope_stability_monte_carlo(
            input_data,
            n_realizations,
            sampler=None,
            q_prime_dist=params.get('q_prime_dist'),
            stress_factor_dist=params.get('stress_factor_dist'),
            span_dist=params.get('span_dist')
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "stope_stability_mc",
            "result": {
                "stability_numbers": mc_result.stability_numbers.tolist() if mc_result.stability_numbers is not None else [],
                "stability_classes": mc_result.stability_classes or [],
                "summary_stats": mc_result.summary_stats,
                "exceedance_curves": {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                     for k, v in mc_result.exceedance_curves.items()}
            },
            "n_realizations": n_realizations
        }
    
    # =========================================================================
    # Slope Risk
    # =========================================================================
    
    def _prepare_slope_risk_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare slope risk analysis payload."""
        from ..geotech.slope_risk import evaluate_slope
        from ..geotech.dataclasses import SlopeRiskInput
        
        input_data = SlopeRiskInput(
            bench_height=params['bench_height'],
            overall_slope_angle=params['overall_slope_angle'],
            gsm=params.get('gsm', 50.0),
            jrc=params.get('jrc', 10.0),
            jcs=params.get('jcs', 100.0),
            water_condition=params.get('water_condition', 'dry'),
            seismic_zone=params.get('seismic_zone', 'low'),
            rock_mass_properties=params.get('rock_mass_properties')
        )
        
        if progress_callback:
            progress_callback(50, "Evaluating slope risk...")
        
        result = evaluate_slope(input_data)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "slope_risk",
            "result": {
                "stability_number": result.stability_number,
                "factor_of_safety": result.factor_of_safety,
                "probability_of_failure": result.probability_of_failure,
                "risk_class": result.risk_class,
                "recommended_action": result.recommended_action,
                "notes": result.notes
            },
            "input": {
                "bench_height": input_data.bench_height,
                "overall_slope_angle": input_data.overall_slope_angle
            }
        }
    
    def _prepare_slope_risk_mc_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare probabilistic slope risk payload."""
        from ..geotech.probabilistic_geotech import run_slope_risk_monte_carlo
        from ..geotech.dataclasses import SlopeRiskInput
        
        input_data = SlopeRiskInput(
            bench_height=params['bench_height'],
            overall_slope_angle=params['overall_slope_angle'],
            gsm=params.get('gsm', 50.0),
            jrc=params.get('jrc', 10.0),
            jcs=params.get('jcs', 100.0),
            water_condition=params.get('water_condition', 'dry'),
            seismic_zone=params.get('seismic_zone', 'low'),
            rock_mass_properties=params.get('rock_mass_properties')
        )
        
        n_realizations = params.get('n_realizations', 100)
        
        if progress_callback:
            progress_callback(10, f"Running {n_realizations} Monte Carlo realizations...")
        
        mc_result = run_slope_risk_monte_carlo(
            input_data,
            n_realizations,
            sampler=None,
            gsm_dist=params.get('gsm_dist'),
            jrc_dist=params.get('jrc_dist'),
            jcs_dist=params.get('jcs_dist')
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "slope_risk_mc",
            "mc_result": mc_result,
            "n_realizations": n_realizations,
            "summary_stats": mc_result.summary_stats,
            "exceedance_curve": mc_result.exceedance_curve
        }
    
    # =========================================================================
    # Slope LEM 2D/3D
    # =========================================================================
    
    def _prepare_slope_lem_2d_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Slope LEM 2D payload."""
        from ..geotech_pit.limit_equilibrium_2d import search_critical_surface_2d, SlopeLEM2DConfig
        
        if progress_callback:
            progress_callback(10, "Searching critical 2D failure surfaces...")
        
        slope_sector = params.get("slope_sector")
        material = params.get("material")
        search_params = params.get("search_params", {"n_surfaces": 50})
        lem_config = params.get("lem_config")
        
        if slope_sector is None or material is None:
            raise ValueError("SlopeSector and GeotechMaterial required")
        
        if lem_config is None:
            lem_config = SlopeLEM2DConfig(
                method=params.get("method", "Bishop"),
                pore_pressure_mode=params.get("pore_pressure_mode", "none"),
                ru=params.get("ru"),
                n_slices=params.get("n_slices", 20)
            )
        
        results = search_critical_surface_2d(slope_sector, material, search_params, lem_config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "slope_lem_2d",
            "results": results,
            "critical_fos": results[0].fos if results else None,
            "n_surfaces": len(results)
        }
    
    def _prepare_slope_lem_3d_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Slope LEM 3D payload."""
        from ..geotech_pit.limit_equilibrium_3d import search_critical_surface_3d, SlopeLEM3DConfig
        
        if progress_callback:
            progress_callback(10, "Searching critical 3D failure surfaces...")
        
        slope_sector = params.get("slope_sector")
        material = params.get("material")
        search_params = params.get("search_params", {"n_surfaces": 20})
        lem_config = params.get("lem_config")
        
        if slope_sector is None or material is None:
            raise ValueError("SlopeSector and GeotechMaterial required")
        
        if lem_config is None:
            lem_config = SlopeLEM3DConfig(
                method=params.get("method", "ellipsoid"),
                n_columns=params.get("n_columns", 10),
                pore_pressure_mode=params.get("pore_pressure_mode", "none"),
                ru=params.get("ru")
            )
        
        results = search_critical_surface_3d(slope_sector, material, search_params, lem_config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "slope_lem_3d",
            "results": results,
            "critical_fos": results[0].fos if results else None,
            "n_surfaces": len(results)
        }
    
    def _prepare_slope_probabilistic_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Probabilistic Slope payload."""
        from ..geotech_pit.slope_probabilistic import run_probabilistic_slope, ProbSlopeConfig
        
        if progress_callback:
            progress_callback(10, "Running probabilistic slope analysis...")
        
        slope_sector = params.get("slope_sector")
        method_2d_or_3d = params.get("method_2d_or_3d", "2D")
        base_material = params.get("base_material")
        cov = params.get("cov", {})
        n_realizations = params.get("n_realizations", 100)
        
        if slope_sector is None or base_material is None:
            raise ValueError("SlopeSector and GeotechMaterial required")
        
        config = ProbSlopeConfig(
            slope_sector=slope_sector,
            method_2d_or_3d=method_2d_or_3d,
            base_material=base_material,
            cov=cov,
            n_realizations=n_realizations,
            surface=params.get("surface"),
            lem_config_2d=params.get("lem_config_2d"),
            lem_config_3d=params.get("lem_config_3d"),
            search_params=params.get("search_params")
        )
        
        result = run_probabilistic_slope(config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "slope_probabilistic",
            "result": result,
            "probability_of_failure": result.probability_of_failure,
            "mean_fos": result.fos_stats.get("mean", 0.0)
        }
    
    def _prepare_bench_design_suggest_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Bench Design Suggest payload."""
        from ..geotech_pit.slope_design import suggest_bench_design
        
        if progress_callback:
            progress_callback(10, "Suggesting bench design...")
        
        domain_code = params.get("domain_code")
        material = params.get("material")
        rock_mass_class = params.get("rock_mass_class", "Fair")
        constraints = params.get("constraints", {})
        
        if domain_code is None or material is None:
            raise ValueError("domain_code and GeotechMaterial required")
        
        rule = suggest_bench_design(domain_code, material, rock_mass_class, constraints)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "bench_design_suggest",
            "rule": rule,
            "bench_height": rule.bench_height,
            "berm_width": rule.berm_width,
            "overall_slope_angle": rule.overall_slope_angle
        }
    
    # =========================================================================
    # Seismic Analysis
    # =========================================================================
    
    def _prepare_seismic_hazard_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare seismic hazard analysis payload."""
        from ..seismic.hazard_model import compute_seismic_hazard_volume
        
        events = params.get('events')
        if events is None:
            raise ValueError("No seismic events provided")
        
        grid_spec = params.get('grid_spec')
        parameters = params.get('parameters', {})
        
        if progress_callback:
            progress_callback(50, "Computing seismic hazard volume...")
        
        hazard_volume = compute_seismic_hazard_volume(events, grid_spec, parameters)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "seismic_hazard",
            "hazard_volume": hazard_volume,
            "grid_spec": grid_spec,
            "event_count": len(events)
        }
    
    def _prepare_seismic_hazard_mc_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare probabilistic seismic hazard payload."""
        from ..seismic.probabilistic_seismic import run_hazard_monte_carlo
        
        events = params.get('events')
        if events is None:
            raise ValueError("No seismic events provided")
        
        grid_spec = params.get('grid_spec')
        parameters = params.get('parameters', {})
        n_realizations = params.get('n_realizations', 100)
        
        if progress_callback:
            progress_callback(10, f"Running {n_realizations} Monte Carlo realizations...")
        
        mc_result = run_hazard_monte_carlo(events, grid_spec, n_realizations, parameters)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "seismic_hazard_mc",
            "mc_result": mc_result,
            "n_realizations": n_realizations,
            "summary_stats": mc_result.summary_stats,
            "exceedance_curve": mc_result.exceedance_curve
        }
    
    # =========================================================================
    # Rockburst Analysis
    # =========================================================================
    
    def _prepare_rockburst_index_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare rockburst index analysis payload."""
        from ..seismic.rockburst_index import compute_rockburst_index_at_points
        
        hazard_volume = params.get('hazard_volume')
        if hazard_volume is None:
            raise ValueError("No hazard volume provided")
        
        target_type = params.get('target_type', 'Stopes')
        use_existing_stopes = params.get('use_existing_stopes', True)
        
        if use_existing_stopes and target_type == 'Stopes':
            if self.block_model and self.block_model.positions is not None:
                points = self.block_model.positions[:100]
            else:
                raise ValueError("No block model available")
        else:
            if self.block_model and self.block_model.positions is not None:
                points = self.block_model.positions[:100]
            else:
                raise ValueError("No block model available")
        
        rock_mass_grid = params.get('rock_mass_grid')
        
        if progress_callback:
            progress_callback(50, "Computing rockburst indices...")
        
        results = compute_rockburst_index_at_points(
            hazard_volume, rock_mass_grid, points, params
        )
        
        results_dict = [
            {
                'location': r.location,
                'index_value': r.index_value,
                'index_class': r.index_class,
                'contributing_events': r.contributing_events,
                'notes': r.notes
            }
            for r in results
        ]
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "rockburst_index",
            "results": results_dict,
            "n_points": len(results)
        }
    
    def _prepare_rockburst_index_mc_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare probabilistic rockburst index payload."""
        from ..seismic.probabilistic_seismic import run_rockburst_monte_carlo
        
        hazard_volume = params.get('hazard_volume')
        if hazard_volume is None:
            raise ValueError("No hazard volume provided")
        
        if self.block_model and self.block_model.positions is not None:
            points = self.block_model.positions[:100]
        else:
            raise ValueError("No block model available")
        
        rock_mass_grid = params.get('rock_mass_grid')
        n_realizations = params.get('n_realizations', 100)
        
        if progress_callback:
            progress_callback(10, f"Running {n_realizations} Monte Carlo realizations...")
        
        mc_result = run_rockburst_monte_carlo(
            hazard_volume, rock_mass_grid, points, n_realizations, params
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "rockburst_index_mc",
            "mc_result": mc_result,
            "n_realizations": n_realizations,
            "summary_stats": mc_result.summary_stats,
            "exceedance_curve": mc_result.exceedance_curve
        }
    
    # =========================================================================
    # Schedule Risk Analysis
    # =========================================================================
    
    def _prepare_schedule_risk_profile_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare schedule risk profile payload."""
        from ..risk.schedule_risk_linker import build_period_risk_profile
        
        schedule = params.get('schedule')
        if schedule is None:
            raise ValueError("No schedule provided")
        
        hazard_volume = params.get('hazard_volume')
        rockburst_results = params.get('rockburst_results')
        slope_results = params.get('slope_results')
        
        if progress_callback:
            progress_callback(50, "Building risk profile...")
        
        profile = build_period_risk_profile(
            schedule,
            hazard_volume,
            rockburst_results,
            slope_results,
            params
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "schedule_risk_profile",
            "profile": profile,
            "schedule_id": profile.schedule_id,
            "n_periods": len(profile.periods)
        }
    
    def _prepare_schedule_risk_timeline_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare risk timeline payload."""
        from ..risk.risk_timeline import build_risk_time_series, compute_exposure_curve
        
        profile = params.get('profile')
        if profile is None:
            raise ValueError("No risk profile provided")
        
        metric = params.get('metric', 'combined_risk_score')
        use_time = params.get('use_time', False)
        threshold = params.get('threshold', None)
        
        if progress_callback:
            progress_callback(50, "Building risk timeline...")
        
        time_series = build_risk_time_series(profile, metric, use_time)
        
        exposure_stats = None
        if threshold is not None:
            exposure_stats = compute_exposure_curve(profile, threshold, metric)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "schedule_risk_timeline",
            "time_series": time_series,
            "exposure_stats": exposure_stats,
            "metric": metric
        }
    
    def _prepare_schedule_risk_compare_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare schedule risk comparison payload."""
        from ..risk.risk_scenarios import compare_risk_scenarios
        
        base_profile = params.get('base_profile')
        if base_profile is None:
            raise ValueError("No base profile provided")
        
        alternative_profiles = params.get('alternative_profiles', [])
        if not alternative_profiles:
            raise ValueError("No alternative profiles provided")
        
        metric = params.get('metric', 'combined_risk_score')
        
        if progress_callback:
            progress_callback(50, "Comparing risk scenarios...")
        
        comparison = compare_risk_scenarios(
            base_profile,
            alternative_profiles,
            metric
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "schedule_risk_compare",
            "comparison": comparison,
            "n_alternatives": len(alternative_profiles)
        }
    
    # =========================================================================
    # Research Mode
    # =========================================================================
    
    def _prepare_research_grid_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Research Grid payload."""
        from ..research.runner import run_scenario_grid_job
        
        params['controller'] = self._app
        
        if progress_callback:
            progress_callback(10, "Running research experiment grid...")
        
        result = run_scenario_grid_job(params)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "research_run_grid",
            **result
        }
    
    # =========================================================================
    # Geology Wizard Workflow Tasks
    # =========================================================================
    
    def _prepare_geology_compositing_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Geology Compositing payload - generates composites with sensible defaults.
        
        This task uses the existing compositing engine with auto-detected parameters.
        Registers composites to drillhole_data with full provenance.
        """
        import pandas as pd
        from ..drillholes.compositing_engine import (
            CompositingMethod, CompositeConfig, WeightingMode,
            PartialStrategy, Composite
        )
        from ..drillholes.compositing_utils import get_intervals_from_registry
        from ..drillholes.compositing_ui_engines import CompositingMethodEngine
        
        if progress_callback:
            progress_callback(5, "Loading drillhole data...")
        
        # Get drillhole data from params (passed by controller)
        drillhole_data = params.get("drillhole_data")
        if drillhole_data is None:
            return {
                "name": "geology_compositing",
                "error": "No drillhole data provided",
                "composite_count": 0
            }
        
        assays_df = drillhole_data.get('assays')
        if assays_df is None or assays_df.empty:
            return {
                "name": "geology_compositing",
                "error": "No assay data available",
                "composite_count": 0
            }
        
        if progress_callback:
            progress_callback(15, "Converting to intervals...")
        
        # Get intervals
        intervals = get_intervals_from_registry(drillhole_data)
        if not intervals:
            return {
                "name": "geology_compositing",
                "error": "No intervals available for compositing",
                "composite_count": 0
            }
        
        if progress_callback:
            progress_callback(25, "Auto-detecting composite length...")
        
        # Auto-detect composite length if not specified
        auto_length = params.get("auto_length", True)
        composite_length = params.get("composite_length")
        
        if auto_length or composite_length is None:
            # Compute median interval length
            lengths = [iv.length for iv in intervals if iv.length > 0]
            if lengths:
                median_length = sorted(lengths)[len(lengths) // 2]
                # Clamp to sensible bounds
                composite_length = max(0.5, min(median_length * 2, 10.0))
            else:
                composite_length = 2.0  # Default fallback
        
        if progress_callback:
            progress_callback(35, f"Compositing with length {composite_length:.1f}m...")
        
        # Create config
        weighting_mode = WeightingMode.LENGTH
        weighting_str = params.get("weighting", "length")
        if weighting_str == "equal":
            # For equal weight, we still use length but with equal intervals
            weighting_mode = WeightingMode.LENGTH
        
        config = CompositeConfig(
            method=CompositingMethod.FIXED_LENGTH,
            composite_length=composite_length,
            weighting_mode=weighting_mode,
            partial_strategy=PartialStrategy.KEEP,
            treat_null_as_zero=True,
        )
        
        # Run compositing
        engine = CompositingMethodEngine()
        try:
            composites = engine.run(intervals, config)
        except Exception as e:
            logger.error(f"Compositing failed: {e}", exc_info=True)
            return {
                "name": "geology_compositing",
                "error": str(e),
                "composite_count": 0
            }
        
        if progress_callback:
            progress_callback(75, "Building composites DataFrame...")
        
        # Convert composites to DataFrame
        records = []
        for comp in composites:
            record = {
                'HOLEID': comp.hole_id,
                'FROM': comp.from_depth,
                'TO': comp.to_depth,
            }
            record.update(comp.grades)
            if comp.metadata:
                for k, v in comp.metadata.items():
                    if k not in record:
                        record[k] = v
            records.append(record)
        
        composites_df = pd.DataFrame(records)
        
        if progress_callback:
            progress_callback(95, "Finalizing...")
        
        # Add midpoint coordinates if assays have them
        if 'X' in assays_df.columns and 'Y' in assays_df.columns and 'Z' in assays_df.columns:
            # Compute midpoint coordinates for composites (simplified)
            if not composites_df.empty:
                composites_df['_MID'] = (composites_df['FROM'] + composites_df['TO']) / 2.0
                # This is a simplified approach - ideally we'd interpolate properly
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "geology_compositing",
            "composites_df": composites_df,
            "composite_count": len(composites_df),
            "composite_length": composite_length,
            "params": {
                "composite_length": composite_length,
                "weighting": weighting_str,
                "method": "fixed_length",
            }
        }
    
    def _prepare_geology_validate_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Geology Validate payload - runs drillhole validation checks.
        
        Returns validation summary with error/warning counts.
        """
        from ..drillholes.drillhole_validation import run_drillhole_validation
        
        if progress_callback:
            progress_callback(10, "Loading data for validation...")
        
        drillhole_data = params.get("drillhole_data")
        if drillhole_data is None:
            return {
                "name": "geology_validate",
                "error": "No drillhole data provided",
                "error_count": 0,
                "warning_count": 0,
            }
        
        if progress_callback:
            progress_callback(30, "Running validation checks...")
        
        try:
            validation_result = run_drillhole_validation(
                collars_df=drillhole_data.get('collars'),
                assays_df=drillhole_data.get('assays'),
                surveys_df=drillhole_data.get('surveys'),
                lithology_df=drillhole_data.get('lithology'),
            )
            
            if progress_callback:
                progress_callback(80, "Counting issues...")
            
            error_count = sum(1 for v in validation_result.violations if v.severity == "ERROR")
            warning_count = sum(1 for v in validation_result.violations if v.severity == "WARNING")
            
            if progress_callback:
                progress_callback(100, "Complete")
            
            return {
                "name": "geology_validate",
                "validation_result": validation_result,
                "error_count": error_count,
                "warning_count": warning_count,
                "is_valid": error_count == 0,
            }
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return {
                "name": "geology_validate",
                "error": str(e),
                "error_count": -1,
                "warning_count": 0,
            }
    
    def _prepare_geology_domains_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Geology Domains payload - applies domain assignments.
        
        Creates a DomainModel with mapping and chronology.
        """
        if progress_callback:
            progress_callback(10, "Loading domain mapping...")
        
        domain_mapping = params.get("domain_mapping", {})
        chronology = params.get("chronology", [])
        
        if not domain_mapping:
            return {
                "name": "geology_domains",
                "error": "No domain mapping provided",
            }
        
        if progress_callback:
            progress_callback(50, "Building domain model...")
        
        # Build domain model structure
        domain_model = {
            "mapping": domain_mapping,
            "chronology": chronology,
            "unique_domains": list(set(domain_mapping.values())),
        }
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "geology_domains",
            "domain_model": domain_model,
            "domain_count": len(domain_model["unique_domains"]),
        }
    
    def _prepare_geology_build_surfaces_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Geology Build Surfaces payload - builds implicit multi-domain surfaces.
        
        Uses one-vs-rest RBF classification to generate contact surfaces,
        OR sedimentary mode with single monotonic scalar field.
        """
        import time
        import sys
        import psutil
        import os
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory_mb = process.memory_info().rss / 1024 / 1024
        
        logger.info("=" * 80)
        logger.info("DATA_CONTROLLER: _prepare_geology_build_surfaces_payload STARTED")
        logger.info("=" * 80)
        logger.info(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Process PID: {os.getpid()}")
        logger.info(f"Initial memory usage: {start_memory_mb:.1f} MB")
        logger.info(f"CPU count: {psutil.cpu_count()}")
        
        try:
            if progress_callback:
                progress_callback(5, "Loading domain model...")
            
            logger.info("Step 1: Extracting parameters...")
            drillhole_data = params.get("drillhole_data")
            domain_model = params.get("domain_model")
            
            logger.info(f"  - drillhole_data type: {type(drillhole_data)}")
            logger.info(f"  - domain_model type: {type(domain_model)}")
            
            if drillhole_data is None:
                logger.error("No drillhole data provided!")
                return {
                    "name": "geology_build_surfaces",
                    "error": "No drillhole data provided",
                }
            
            if domain_model is None:
                logger.error("No domain model provided!")
                return {
                    "name": "geology_build_surfaces",
                    "error": "No domain model provided. Please define domains first.",
                }
            
            # =====================================================================
            # SEDIMENTARY MODE CHECK
            # =====================================================================
            mode = domain_model.get("mode", "default") if isinstance(domain_model, dict) else "default"
            
            # Use new modeling dispatcher that routes to appropriate engine
            logger.info("Step 2: Importing modeling dispatcher...")
            try:
                from ..geology.modeling_dispatcher import build_geological_model
                logger.info("  - Dispatcher imported successfully")
            except ImportError as ie:
                logger.error(f"modeling_dispatcher import failed: {ie}", exc_info=True)
                # Fallback to old direct import
                if mode == "sedimentary":
                    logger.info("SEDIMENTARY MODE ENABLED - using single scalar field solver")
                    return self._build_sedimentary_surfaces(
                        drillhole_data=drillhole_data,
                        domain_model=domain_model,
                        params=params,
                        progress_callback=progress_callback,
                    )
                else:
                    from ..geology.implicit_multidomain import build_multidomain_surfaces
                    logger.info("Using implicit_multidomain fallback")
                    result = build_multidomain_surfaces(
                        drillhole_data=drillhole_data,
                        domain_model=domain_model,
                        resolution=resolution if not auto_resolution else None,
                        kernel=kernel,
                        progress_callback=progress_callback,
                    )
                    return {
                        "name": "geology_build_surfaces",
                        **result
                    }
            
            # =====================================================================
            # CALL DISPATCHER WITH ENGINE SELECTION
            # =====================================================================
            # =====================================================================
            # CALL DISPATCHER WITH ENGINE SELECTION
            # =====================================================================
            
            if progress_callback:
                progress_callback(20, "Building geological model...")
            
            logger.info("Step 3: Extracting method parameters...")
            engine = params.get("engine", "auto")  # New: engine selection
            method = params.get("method", "rbf")
            auto_resolution = params.get("auto_resolution", True)
            resolution = params.get("resolution", 50)
            kernel = params.get("kernel", "linear")
            smooth_sigma = params.get("smooth_sigma", 1.5)
            surface_method = params.get("surface_method", "gempy")  # Legacy parameter (now ignored)
            
            logger.info(f"  - engine: {engine}")
            logger.info(f"  - method: {method}")
            logger.info(f"  - kernel: {kernel}")
            logger.info(f"  - auto_resolution: {auto_resolution}")
            logger.info(f"  - resolution: {resolution}")
            
            logger.info("Step 4: Calling modeling dispatcher...")
            result = build_geological_model(
                drillhole_data=drillhole_data,
                domain_model=domain_model,
                engine=engine,
                resolution=resolution if not auto_resolution else None,
                kernel=kernel,
                use_hermite=True,
                progress_callback=progress_callback,
                smooth_sigma=smooth_sigma,
            )
            
            logger.info("Step 5: Modeling completed successfully")
            
            end_time = time.time()
            end_memory_mb = process.memory_info().rss / 1024 / 1024
            total_time = end_time - start_time
            
            logger.info("=" * 80)
            logger.info("GEOLOGICAL MODELING COMPLETED - PERFORMANCE SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Total elapsed time: {total_time:.2f} seconds ({total_time / 60:.1f} minutes)")
            logger.info(f"Memory usage: {start_memory_mb:.1f} MB → {end_memory_mb:.1f} MB (Δ {end_memory_mb - start_memory_mb:.1f} MB)")
            logger.info(f"Peak memory: {end_memory_mb:.1f} MB")
            logger.info(f"Engine used: {result.get('metadata', {}).get('method', 'unknown')}")
            
            if 'surfaces' in result:
                logger.info(f"Surfaces generated: {len(result.get('surfaces', []))}")
            if 'diagnostics' in result:
                logger.info(f"Diagnostics: {result['diagnostics']}")
            
            logger.info("=" * 80)
            
            return {
                "name": "geology_build_surfaces",
                **result,
                "performance": {
                    "total_seconds": total_time,
                    "start_memory_mb": start_memory_mb,
                    "end_memory_mb": end_memory_mb,
                    "memory_delta_mb": end_memory_mb - start_memory_mb,
                }
            }
        
        # AUDIT FIX: Handle specific exceptions from implicit_multidomain
        except (DataGateError, ImplicitModellingError) as e:
            # These are expected validation/modelling errors with user-friendly messages
            end_time = time.time()
            logger.error("=" * 80)
            logger.error(f"DATA_CONTROLLER: Surface building FAILED - {type(e).__name__}")
            logger.error(f"Error: {e}")
            logger.error(f"Elapsed time before failure: {end_time - start_time:.2f} seconds")
            logger.error("=" * 80)
            return {
                "name": "geology_build_surfaces",
                "error": str(e),
                "error_type": type(e).__name__,
                "surface_count": 0,
                "timing_seconds": end_time - start_time,
            }
            
        except Exception as e:
            import traceback
            end_time = time.time()
            logger.error("=" * 80)
            logger.error("DATA_CONTROLLER: Surface building FAILED - Unexpected error")
            logger.error(f"Error: {e}")
            logger.error(f"Elapsed time before failure: {end_time - start_time:.2f} seconds")
            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            return {
                "name": "geology_build_surfaces",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "surface_count": 0,
                "timing_seconds": end_time - start_time,
            }
    
    def _build_sedimentary_surfaces(
        self,
        drillhole_data: Dict[str, Any],
        domain_model: Dict[str, Any],
        params: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Build surfaces using sedimentary mode (single monotonic scalar field).
        
        This is the correct approach for layered sedimentary sequences.
        It prevents numerical artifacts (blobs/islands) by construction.
        
        Args:
            drillhole_data: Dict with 'composites' DataFrame
            domain_model: Dict with 'sedimentary' config and 'chronology'
            params: Additional parameters
            progress_callback: Progress callback function
            
        Returns:
            Dict with surfaces, voxel_labels, diagnostics, etc.
        """
        logger.info("=" * 80)
        logger.info("SEDIMENTARY MODE: Building single scalar field model")
        logger.info("=" * 80)
        
        try:
            # Import sedimentary solver and validator
            from ..geology.sedimentary_solver import (
                SedimentaryScalarFieldSolver,
                normalize_to_boundary_events,
                SedimentaryDataGateError,
            )
            from ..geology.geological_model_validator import SedimentaryModelValidator
            
            if progress_callback:
                progress_callback(5, "Validating sedimentary configuration...")
            
            # Extract sedimentary config
            sed_config = domain_model.get("sedimentary", {})
            chronology = sed_config.get("chronology", domain_model.get("chronology", []))
            gradient_type = sed_config.get("gradient_type", "vertical")
            gradient_polarity = sed_config.get("gradient_polarity", 1)
            
            logger.info(f"Chronology: {chronology}")
            logger.info(f"Gradient: {gradient_type}, polarity={gradient_polarity}")
            
            # Initialize validator
            validator = SedimentaryModelValidator()
            
            # PRE-BUILD VALIDATION
            if progress_callback:
                progress_callback(10, "Pre-build validation...")
            
            # Validate chronology
            chrono_result = validator.validate_chronology_table(chronology)
            if not chrono_result.passed:
                logger.error(f"Chronology validation failed: {chrono_result.message}")
                return {
                    "name": "geology_build_surfaces",
                    "error": chrono_result.message,
                    "error_type": "SedimentaryValidationError",
                    "blocker": "pre-build",
                    "surface_count": 0,
                }
            
            if progress_callback:
                progress_callback(15, "Normalizing contacts to boundary events...")
            
            # Normalize contacts to boundary events
            composites_df = drillhole_data.get("composites")
            if composites_df is None or composites_df.empty:
                return {
                    "name": "geology_build_surfaces",
                    "error": "SEDI-CTRL-001: No composite data available for sedimentary mode",
                    "error_type": "SedimentaryDataGateError",
                    "surface_count": 0,
                }
            
            try:
                boundary_events = normalize_to_boundary_events(
                    composites_df=composites_df,
                    chronology=chronology,
                    polarity=gradient_polarity,
                )
            except SedimentaryDataGateError as e:
                logger.error(f"Contact normalization failed: {e}")
                return {
                    "name": "geology_build_surfaces",
                    "error": str(e),
                    "error_type": "SedimentaryDataGateError",
                    "blocker": "pre-build",
                    "surface_count": 0,
                }
            
            # Convert boundary events to dict format for validator
            contacts_dicts = [
                {
                    "x": e.x, "y": e.y, "z": e.z,
                    "boundary_id": e.boundary_id,
                    "above_unit": e.above_unit,
                    "below_unit": e.below_unit,
                    "polarity": e.polarity,
                    "hole_id": e.hole_id,
                }
                for e in boundary_events
            ]
            
            # Validate contacts
            contact_result = validator.validate_contacts(contacts_dicts, chronology)
            if not contact_result.passed:
                logger.error(f"Contact validation failed: {contact_result.message}")
                return {
                    "name": "geology_build_surfaces",
                    "error": contact_result.message,
                    "error_type": "SedimentaryValidationError",
                    "blocker": "pre-build",
                    "details": {
                        "total_events": contact_result.total_events,
                        "valid_events": contact_result.valid_events,
                        "events_per_boundary": contact_result.events_per_boundary,
                        "insufficient_boundaries": contact_result.insufficient_boundaries,
                    },
                    "surface_count": 0,
                }
            
            logger.info(f"Pre-build validation passed: {len(boundary_events)} boundary events")
            
            if progress_callback:
                progress_callback(20, "Building sedimentary scalar field...")
            
            # Run sedimentary solver
            resolution = params.get("resolution")
            kernel = params.get("kernel", "linear")
            
            solver = SedimentaryScalarFieldSolver(
                kernel=kernel,
                progress_callback=progress_callback,
            )
            
            result = solver.solve(
                boundary_events=boundary_events,
                chronology=chronology,
                gradient_type=gradient_type,
                gradient_polarity=gradient_polarity,
                resolution=resolution,
            )
            
            if progress_callback:
                progress_callback(80, "Build-stage validation...")
            
            # BUILD-STAGE VALIDATION
            diagnostics = result.get("diagnostics", {})
            
            # Check monotonicity
            mono_diag = diagnostics.get("monotonicity", {})
            if not mono_diag.get("passed", True):
                logger.error(f"Monotonicity validation failed: {mono_diag.get('summary')}")
                return {
                    "name": "geology_build_surfaces",
                    "error": mono_diag.get("summary", "Monotonicity validation failed"),
                    "error_type": "SedimentaryValidationError",
                    "blocker": "build-stage",
                    "diagnostics": diagnostics,
                    "surface_count": 0,
                }
            
            # Check residuals
            resid_diag = diagnostics.get("residuals", {})
            if not resid_diag.get("passed", True):
                logger.error(f"Residual validation failed: {resid_diag.get('summary')}")
                return {
                    "name": "geology_build_surfaces",
                    "error": resid_diag.get("summary", "Contact residual validation failed"),
                    "error_type": "SedimentaryValidationError",
                    "blocker": "build-stage",
                    "diagnostics": diagnostics,
                    "surface_count": 0,
                }
            
            if progress_callback:
                progress_callback(90, "Post-build quality checks...")
            
            # POST-BUILD QUALITY FLAGS
            topo_diag = diagnostics.get("topology", {})
            if topo_diag.get("has_extreme_violations", False):
                # Extreme topology violations are blocking
                logger.error(f"Extreme topology violations detected")
                return {
                    "name": "geology_build_surfaces",
                    "error": "Extreme topology violations: multiple units have disconnected islands",
                    "error_type": "SedimentaryValidationError",
                    "blocker": "post-build",
                    "diagnostics": diagnostics,
                    "surface_count": 0,
                }
            
            # Attach any warnings
            warnings = []
            if topo_diag.get("total_islands", 0) > len(chronology):
                warnings.append(f"Topology: {topo_diag.get('total_islands')} total islands detected")
            
            result["validation_warnings"] = warnings
            result["mode"] = "sedimentary"
            
            if progress_callback:
                progress_callback(100, "Sedimentary model complete")
            
            logger.info("=" * 80)
            logger.info("SEDIMENTARY MODE: Build completed successfully")
            logger.info(f"Surfaces: {len(result.get('surfaces', []))}")
            logger.info(f"Diagnostics: monotonicity={mono_diag.get('passed')}, residuals={resid_diag.get('passed')}")
            logger.info("=" * 80)
            
            return {
                "name": "geology_build_surfaces",
                **result,
                "surface_count": len(result.get("surfaces", [])),
            }
            
        except Exception as e:
            import traceback
            logger.error("=" * 80)
            logger.error("SEDIMENTARY MODE: Build FAILED - Unexpected error")
            logger.error(f"Error: {e}")
            logger.error(traceback.format_exc())
            logger.error("=" * 80)
            return {
                "name": "geology_build_surfaces",
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "surface_count": 0,
            }
    
    def _prepare_geology_build_solids_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Geology Build Solids payload - generates voxel solids from surfaces.
        
        Creates a label grid (domain code per cell) with optional clipping.
        """
        logger.info("=" * 80)
        logger.info("DATA_CONTROLLER: _prepare_geology_build_solids_payload STARTED")
        logger.info("=" * 80)
        
        try:
            if progress_callback:
                progress_callback(10, "Loading surfaces...")
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
        
        logger.info("Step 1: Extracting parameters...")
        surfaces = params.get("surfaces")
        domain_model = params.get("domain_model")
        logger.info(f"  - surfaces type: {type(surfaces)}, is None: {surfaces is None}")
        logger.info(f"  - domain_model type: {type(domain_model)}, is None: {domain_model is None}")
        
        if surfaces is None or domain_model is None:
            logger.error("Missing surfaces or domain_model!")
            return {
                "name": "geology_build_solids",
                "error": "Surfaces and domain model required. Run 'Build Surfaces' first.",
            }
        
        logger.info("Step 2: Importing voxel_solids module...")
        # Import the voxel solids engine (will be created)
        try:
            from ..geology.voxel_solids import build_voxel_solids
            logger.info("  - Module imported successfully")
        except ImportError:
            logger.warning("voxel_solids not yet implemented, using stub")
            
            if progress_callback:
                progress_callback(100, "Complete (stub)")
            
            return {
                "name": "geology_build_solids",
                "solids": None,
                "topology_report": "voxel_solids module not implemented - stub result",
            }
        
        try:
            if progress_callback:
                progress_callback(30, "Building voxel grid...")
        except Exception as e:
            logger.warning(f"Progress callback failed: {e}")
        
        clip_to_topography = params.get("clip_to_topography", False)
        logger.info(f"Step 3: clip_to_topography={clip_to_topography}")
        
        logger.info("Step 4: Calling build_voxel_solids...")
        try:
            result = build_voxel_solids(
                surfaces=surfaces,
                domain_model=domain_model,
                clip_to_topography=clip_to_topography,
                progress_callback=progress_callback,
            )
            logger.info("Step 5: build_voxel_solids returned successfully")
            
            return {
                "name": "geology_build_solids",
                **result
            }
        except Exception as e:
            logger.error(f"Solids building failed: {e}", exc_info=True)
            return {
                "name": "geology_build_solids",
                "error": str(e),
            }
    
    def _prepare_geology_misfit_qc_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Geology Misfit QC payload - computes quality metrics.
        
        Returns classification accuracy, confusion matrix, etc.
        """
        if progress_callback:
            progress_callback(10, "Loading model data...")
        
        surfaces = params.get("surfaces")
        solids = params.get("solids")
        drillhole_data = params.get("drillhole_data")
        
        # Import the misfit QC module (will be created)
        try:
            from ..geology.misfit_qc import compute_geology_qc
        except ImportError:
            logger.warning("misfit_qc not yet implemented, using stub")
            
            if progress_callback:
                progress_callback(100, "Complete (stub)")
            
            return {
                "name": "geology_misfit_qc",
                "metrics": {
                    "accuracy": "N/A (module not implemented)",
                    "contact_count": 0,
                    "surface_count": len(surfaces) if surfaces else 0,
                    "solid_volume": "N/A",
                },
            }
        
        if progress_callback:
            progress_callback(50, "Computing QC metrics...")
        
        try:
            result = compute_geology_qc(
                surfaces=surfaces,
                solids=solids,
                drillhole_data=drillhole_data,
                progress_callback=progress_callback,
            )
            
            return {
                "name": "geology_misfit_qc",
                **result
            }
        except Exception as e:
            logger.error(f"QC computation failed: {e}", exc_info=True)
            return {
                "name": "geology_misfit_qc",
                "error": str(e),
            }
    
    def _prepare_geology_export_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Geology Export payload - exports surfaces and solids.
        
        Writes files to the specified directory with JSON metadata sidecar.
        """
        if progress_callback:
            progress_callback(10, "Preparing export...")
        
        export_dir = params.get("export_dir")
        if not export_dir:
            return {
                "name": "geology_export",
                "error": "No export directory specified",
            }
        
        # Import the export module (will be created)
        try:
            from ..geology.export_geology import export_geology_model
        except ImportError:
            logger.warning("export_geology not yet implemented, using stub")
            
            if progress_callback:
                progress_callback(100, "Complete (stub)")
            
            return {
                "name": "geology_export",
                "export_dir": export_dir,
                "note": "export_geology module not implemented - stub result",
            }
        
        surfaces = params.get("surfaces")
        solids = params.get("solids")
        domain_model = params.get("domain_model")
        
        export_surfaces = params.get("export_surfaces", True)
        surface_format = params.get("surface_format", "vtk")
        export_solids = params.get("export_solids", True)
        solid_format = params.get("solid_format", "vtk")
        include_metadata = params.get("include_metadata", True)
        
        if progress_callback:
            progress_callback(30, "Exporting files...")
        
        try:
            result = export_geology_model(
                export_dir=export_dir,
                surfaces=surfaces,
                solids=solids,
                domain_model=domain_model,
                export_surfaces=export_surfaces,
                surface_format=surface_format,
                export_solids=export_solids,
                solid_format=solid_format,
                include_metadata=include_metadata,
                progress_callback=progress_callback,
            )
            
            return {
                "name": "geology_export",
                **result
            }
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)
            return {
                "name": "geology_export",
                "error": str(e),
            }

    # =========================================================================
    # LoopStructural Geological Modeling
    # =========================================================================
    
    def _prepare_loopstructural_model_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare LoopStructural geological model payload.
        
        Industry-grade implicit modeling with JORC/SAMREC compliance.
        Uses FDI interpolation and handles fault displacement fields.
        """
        from ..geology.chronos_engine import ChronosEngine
        
        if progress_callback:
            progress_callback(10, "Initializing LoopStructural engine...")
        
        # Extract parameters
        extent = params.get("extent")
        if not extent:
            raise ValueError("extent parameter required (xmin, xmax, ymin, ymax, zmin, zmax)")
        
        resolution = params.get("resolution", 50)
        cgw = params.get("cgw", 0.1)
        interpolator_type = params.get("interpolator_type", "FDI")
        
        contacts = params.get("contacts")
        orientations = params.get("orientations")
        stratigraphy = params.get("stratigraphy", [])
        faults = params.get("faults", [])
        
        if contacts is None:
            raise ValueError("contacts DataFrame required")
        
        # Initialize engine
        engine = ChronosEngine(extent, resolution=resolution)
        
        if progress_callback:
            progress_callback(20, "Preparing contact data...")
        
        # Prepare data (scale to [0,1])
        contacts_prepared = engine.prepare_data(contacts)
        
        if orientations is not None:
            orientations_prepared = engine.prepare_data(orientations)
        else:
            # Generate synthetic orientations (horizontal layers)
            orientations_prepared = contacts_prepared[['X_s', 'Y_s', 'Z_s']].copy()
            orientations_prepared['gx'] = 0.0
            orientations_prepared['gy'] = 0.0
            orientations_prepared['gz'] = 1.0
        
        if progress_callback:
            progress_callback(40, "Building geological model...")
        
        # Build model
        engine.build_model(
            stratigraphy=stratigraphy,
            contacts=contacts_prepared,
            orientations=orientations_prepared,
            faults=faults if faults else None,
            cgw=cgw,
            interpolator_type=interpolator_type
        )
        
        if progress_callback:
            progress_callback(90, "Finalizing...")
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "loopstructural_model",
            "model": engine.model,
            "engine": engine,
            "build_log": engine.get_build_log(),
            "resolution": resolution,
            "n_contacts": len(contacts),
            "n_faults": len(faults) if faults else 0,
        }
    
    def _prepare_loopstructural_compliance_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare LoopStructural compliance validation payload.
        
        Calculates spatial misfit and generates JORC/SAMREC audit report.
        """
        from ..geology.compliance_manager import ComplianceManager
        
        if progress_callback:
            progress_callback(10, "Validating model compliance...")
        
        model = params.get("model")
        engine = params.get("engine")
        contacts = params.get("contacts")
        
        if model is None or engine is None or contacts is None:
            raise ValueError("model, engine, and contacts required")
        
        # Prepare contacts for validation
        contacts_prepared = engine.prepare_data(contacts)
        contacts_scaled = contacts_prepared.rename(columns={
            'X_s': 'X', 'Y_s': 'Y', 'Z_s': 'Z'
        })
        
        if progress_callback:
            progress_callback(50, "Computing misfit metrics...")
        
        # Generate compliance report
        report = ComplianceManager.generate_misfit_report(
            model,
            contacts_scaled,
            engine.scaler,
            feature_name="Stratigraphy"
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "loopstructural_compliance",
            "report": report,
            "mean_residual": report.mean_residual,
            "p90_error": report.p90_error,
            "status": report.status,
            "classification": report.classification_recommendation,
            "jorc_compliant": report.is_jorc_compliant,
        }
    
    def _prepare_loopstructural_fault_detection_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare LoopStructural fault detection payload.
        
        Analyzes model misfit to suggest missing fault planes.
        """
        from ..geology.fault_detection import FaultDetectionEngine
        
        if progress_callback:
            progress_callback(10, "Initializing fault detection...")
        
        misfit_data = params.get("misfit_data")
        if misfit_data is None:
            raise ValueError("misfit_data DataFrame required")
        
        error_threshold = params.get("error_threshold", 3.0)
        cluster_eps = params.get("cluster_eps", 50.0)
        cluster_min_samples = params.get("cluster_min_samples", 4)
        
        if progress_callback:
            progress_callback(30, "Clustering high-error regions...")
        
        # Run fault detection
        detector = FaultDetectionEngine(
            error_threshold_m=error_threshold,
            cluster_eps=cluster_eps,
            cluster_min_samples=cluster_min_samples,
        )
        
        suggestions = detector.detect_potential_faults(misfit_data)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "loopstructural_fault_detection",
            "suggestions": suggestions,
            "n_suggestions": len(suggestions),
            "detection_log": detector.get_detection_log(),
        }
    
    def _prepare_loopstructural_extract_surfaces_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare LoopStructural surface extraction payload.
        
        Extracts watertight meshes for visualization and volume calculation.
        """
        from ..geology.mesh_validator import validate_mesh_detailed
        
        if progress_callback:
            progress_callback(10, "Extracting surfaces...")
        
        engine = params.get("engine")
        if engine is None:
            raise ValueError("engine (ChronosEngine) required")
        
        n_surfaces = params.get("n_surfaces")
        
        # Extract meshes
        surfaces = engine.extract_meshes(n_surfaces=n_surfaces)
        
        if progress_callback:
            progress_callback(70, "Validating mesh integrity...")
        
        # Validate each surface
        validated_surfaces = []
        for surface in surfaces:
            validation = validate_mesh_detailed(
                surface['vertices'],
                surface['faces']
            )
            surface['validation'] = validation.to_dict()
            validated_surfaces.append(surface)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "loopstructural_extract_surfaces",
            "surfaces": validated_surfaces,
            "n_surfaces": len(validated_surfaces),
        }

    # =========================================================================
    # Structural Features
    # =========================================================================
    
    def _prepare_load_structural_features_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Load Structural Features payload - loads faults, folds, unconformities from CSV.
        
        Pure computation - parses CSV files and returns structural feature collection.
        """
        from pathlib import Path
        from ..parsers.structural_csv_parser import (
            StructuralCSVParser,
            ColumnMapping,
            CSVFormat,
        )
        
        if progress_callback:
            progress_callback(10, "Initializing parser...")
        
        file_path = params.get("file_path")
        if file_path is None:
            raise ValueError("file_path is required")
        
        file_path = Path(file_path)
        logger.info(f"Loading structural features from: {file_path}")
        
        # Get optional parameters
        column_mapping = params.get("column_mapping")
        expected_format = params.get("expected_format")
        validate = params.get("validate", True)
        
        # Convert column_mapping dict to ColumnMapping object if needed
        if isinstance(column_mapping, dict):
            from ..parsers.structural_csv_parser import ColumnMapping as CM
            mapping = CM(**column_mapping)
        else:
            mapping = column_mapping
        
        # Convert expected_format string to enum if needed
        if isinstance(expected_format, str):
            expected_format = CSVFormat(expected_format)
        
        if progress_callback:
            progress_callback(30, "Parsing CSV file...")
        
        # Parse the file
        parser = StructuralCSVParser()
        result = parser.parse(
            file_path=file_path,
            column_mapping=mapping,
            expected_format=expected_format,
            validate=validate,
        )
        
        if progress_callback:
            progress_callback(80, "Processing features...")
        
        # Log results
        fc = result.collection.feature_count
        logger.info(f"Parsed structural features: {fc['faults']} faults, {fc['folds']} folds, {fc['unconformities']} unconformities")
        
        if result.validation_errors:
            logger.warning(f"Validation errors: {result.validation_errors}")
        if result.validation_warnings:
            logger.debug(f"Validation warnings: {result.validation_warnings[:5]}...")  # Only log first 5
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "load_structural_features",
            "collection": result.collection,
            "format_detected": result.format_detected.value,
            "column_mapping": result.column_mapping.to_dict(),
            "validation_errors": result.validation_errors,
            "validation_warnings": result.validation_warnings,
            "rows_processed": result.rows_processed,
            "rows_skipped": result.rows_skipped,
            "metadata": result.metadata,
            "feature_count": result.collection.feature_count,
        }
    
    def register_structural_features(self, collection) -> None:
        """
        Register structural features in the data registry.
        
        Emits appropriate signals for UI updates.
        
        Args:
            collection: StructuralFeatureCollection to register
        """
        from ..structural.feature_types import (
            StructuralFeatureCollection,
            FaultFeature,
            FoldFeature,
            UnconformityFeature,
        )
        
        if not hasattr(self._app, '_structural_features'):
            self._app._structural_features = StructuralFeatureCollection()
        
        # Merge with existing features
        self._app._structural_features.merge(collection)
        
        # Emit signals
        if hasattr(self._app, 'signals') and self._app.signals:
            self._app.signals.structural_features_loaded.emit(collection)
            
            # Emit individual feature signals
            for fault in collection.faults:
                self._app.signals.fault_added.emit(fault)
            for fold in collection.folds:
                self._app.signals.fold_added.emit(fold)
            for unconformity in collection.unconformities:
                self._app.signals.unconformity_added.emit(unconformity)
        
        fc = collection.feature_count
        logger.info(f"Registered structural features: {fc['total']} total ({fc['faults']} faults, {fc['folds']} folds, {fc['unconformities']} unconformities)")
    
    def get_structural_features(self):
        """
        Get all registered structural features.
        
        Returns:
            StructuralFeatureCollection or None if none registered
        """
        return getattr(self._app, '_structural_features', None)
    
    def get_faults(self):
        """Get all registered fault features."""
        features = self.get_structural_features()
        return features.faults if features else []
    
    def get_folds(self):
        """Get all registered fold features."""
        features = self.get_structural_features()
        return features.folds if features else []
    
    def get_unconformities(self):
        """Get all registered unconformity features."""
        features = self.get_structural_features()
        return features.unconformities if features else []
    
    def remove_structural_feature(self, feature_id: str) -> bool:
        """
        Remove a structural feature by ID.
        
        Args:
            feature_id: ID of feature to remove
            
        Returns:
            True if feature was found and removed
        """
        features = self.get_structural_features()
        if features is None:
            return False
        
        removed = features.remove_feature(feature_id)
        
        if removed and hasattr(self._app, 'signals') and self._app.signals:
            self._app.signals.structural_feature_removed.emit(feature_id)
        
        return removed
    
    def clear_structural_features(self) -> None:
        """Clear all structural features."""
        from ..structural.feature_types import StructuralFeatureCollection
        
        self._app._structural_features = StructuralFeatureCollection()
        
        if hasattr(self._app, 'signals') and self._app.signals:
            self._app.signals.structural_features_cleared.emit()
        
        logger.info("Cleared all structural features")
    
    # =========================================================================
    # Public API Methods (delegated from AppController)
    # =========================================================================
    
    def load_drillholes(self, config: Dict[str, Any], callback=None) -> None:
        """Load drillholes via task system."""
        self._app.run_task("load_drillholes", config, callback)
    
    def run_drillhole_qaqc(self, db_config: Dict[str, Any], callback=None) -> None:
        """Run drillhole QAQC via task system."""
        self._app.run_task("drillhole_qaqc", db_config, callback)
    
    def run_implicit_geology(self, config: Dict[str, Any], callback=None) -> None:
        """Run implicit geology modelling via task system."""
        self._app.run_task("implicit_geology", config, callback)
    
    def run_loopstructural_model(self, config: Dict[str, Any], callback=None) -> None:
        """Run LoopStructural geological modelling via task system."""
        self._app.run_task("loopstructural_model", config, callback)
    
    def run_loopstructural_compliance(self, config: Dict[str, Any], callback=None) -> None:
        """Run LoopStructural compliance validation via task system."""
        self._app.run_task("loopstructural_compliance", config, callback)
    
    def run_loopstructural_fault_detection(self, config: Dict[str, Any], callback=None) -> None:
        """Run LoopStructural fault detection via task system."""
        self._app.run_task("loopstructural_fault_detection", config, callback)
    
    def build_wireframes(self, config: Dict[str, Any], callback=None) -> None:
        """Build wireframes via task system."""
        self._app.run_task("build_wireframes", config, callback)
    
    def run_structural_analysis(self, config: Dict[str, Any], callback=None) -> None:
        """Run structural analysis via task system."""
        analysis_type = config.get("analysis_type", "clusters")
        if analysis_type == "clusters":
            self._app.run_task("structural_clusters", config, callback)
        elif analysis_type == "kinematic":
            self._app.run_task("kinematic_analysis", config, callback)
        else:
            raise ValueError(f"Unknown structural analysis type: {analysis_type}")
    
    def run_slope_lem_2d(self, config: Dict[str, Any], callback=None) -> None:
        """Run 2D slope limit equilibrium analysis via task system."""
        self._app.run_task("slope_lem_2d", config, callback)
    
    def run_slope_lem_3d(self, config: Dict[str, Any], callback=None) -> None:
        """Run 3D slope limit equilibrium analysis via task system."""
        self._app.run_task("slope_lem_3d", config, callback)
    
    def run_slope_probabilistic(self, config: Dict[str, Any], callback=None) -> None:
        """Run probabilistic slope stability analysis via task system."""
        self._app.run_task("slope_probabilistic", config, callback)
    
    def suggest_bench_design(self, config: Dict[str, Any], callback=None) -> None:
        """Suggest bench design via task system."""
        self._app.run_task("bench_design_suggest", config, callback)
    
    def run_research_grid(self, grid_config: Dict[str, Any], callback=None) -> None:
        """Run Research Grid via task system."""
        self._app.run_task("research_run_grid", grid_config, callback)
    
    def load_structural_features(self, config: Dict[str, Any], callback=None) -> None:
        """Load structural features from CSV via task system."""
        self._app.run_task("load_structural_features", config, callback)

