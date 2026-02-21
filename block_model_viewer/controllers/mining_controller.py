"""
Mining Controller - Handles all mining, planning, and scheduling operations.

This controller manages resource calculation, IRR/NPV analysis, pit optimization,
scheduling (strategic, tactical, short-term), grade control, reconciliation,
underground planning, haulage, and scenario management.
"""

from typing import Optional, Dict, Any, Callable, Union, List, TYPE_CHECKING
from pathlib import Path
import logging

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .app_controller import AppController

logger = logging.getLogger(__name__)


class MiningController:
    """
    Controller for mining and planning operations.
    
    Handles resource calculation/classification, IRR/NPV analysis, pit optimization,
    strategic/tactical/short-term scheduling, grade control, reconciliation,
    underground planning, haulage, pushback design, and scenario management.
    """
    
    def __init__(self, app_controller: "AppController"):
        """
        Initialize mining controller.
        
        Args:
            app_controller: Parent AppController instance for shared state access
        """
        self._app = app_controller
    
    @property
    def block_model(self):
        """Return the currently loaded block model."""
        return self._app.block_model
    
    @property
    def scenario_store(self):
        """Return the scenario store."""
        return self._app.scenario_store
    
    @property
    def scenario_runner(self):
        """Return the scenario runner."""
        return self._app.scenario_runner
    
    # =========================================================================
    # Resource Calculation
    # =========================================================================
    
    def _prepare_resource_calculation_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Resource Calculation payload."""
        data_df = params["data_df"]
        grade_col = params["grade_col"]
        density_col = params["density_col"]
        cutoff = params["cutoff"]
        comparator = params["comparator"]
        default_density = params["default_density"]
        dx = params["dx"]
        dy = params["dy"]
        dz = params["dz"]
        
        df = data_df.copy()
        
        # Calculate volume (m³)
        df['VOLUME'] = dx * dy * dz
        
        # Get or calculate density (t/m³)
        if density_col == '<Use Default>' or density_col not in df.columns:
            df['DENSITY'] = default_density
        else:
            df['DENSITY'] = df[density_col].fillna(default_density)
        
        # Calculate tonnes
        df['TONNES'] = df['DENSITY'] * df['VOLUME']
        
        # Apply cut-off filter
        comparators = {
            '>=': lambda x, y: x >= y,
            '<=': lambda x, y: x <= y,
            '>': lambda x, y: x > y,
            '<': lambda x, y: x < y,
            '==': lambda x, y: x == y
        }
        
        comp_func = comparators[comparator]
        selected = df[comp_func(df[grade_col], cutoff)]
        
        # Calculate summaries
        total_blocks = len(selected)
        total_tonnes = selected['TONNES'].sum()
        avg_grade = selected[grade_col].mean()
        
        # Contained metal (t) = (tonnes * grade_g/t) / 1,000,000
        contained_metal = (total_tonnes * avg_grade) / 1e6
        
        result_df = pd.DataFrame([{
            'Cut-off': f"{comparator} {cutoff} g/t",
            'Blocks': total_blocks,
            'Tonnes': total_tonnes,
            'Grade': avg_grade,
            'ContainedMetal': contained_metal
        }])
        
        mask = comp_func(df[grade_col], cutoff).values
        
        metadata = {
            "operation": "calculate",
            "cutoff": cutoff,
            "comparator": comparator,
            "total_blocks": int(total_blocks),
            "total_tonnes": float(total_tonnes),
            "avg_grade": float(avg_grade),
            "contained_metal": float(contained_metal),
            "message": f"Resource calculation completed with cut-off {comparator} {cutoff}.",
        }
        
        payload = {
            "name": "resource_calculation",
            "operation": "calculate",
            "result_df": result_df,
            "mask": mask.tolist(),
            "metadata": metadata,
        }
        return payload

    # =========================================================================
    # Resource Classification
    # =========================================================================
    
    def _prepare_resource_classification_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Resource Classification payload.

        Supports multiple classification modes:
        - Mode 4: Audit-grade block-based classification (uses estimation outputs)
        - Mode 3: JORC drillhole proximity classification
        - Mode 2: Drillhole spacing classification
        - Mode 1: Legacy estimation-based classification
        """
        # =====================================================================
        # Mode 4: Audit-Grade Block-Based Classification (JORC/SAMREC/CIM)
        # Uses estimation outputs already stored in block model (dist1, ncomp,
        # nholes, pass, kv). Per-domain parameters with hard gates.
        # =====================================================================
        if "params_by_domain" in params:
            from geox.resource_classification import (
                ResourceClassificationEngine,
                DomainClassificationParams,
                ClassificationError,
                InvalidParameterError,
            )
            from ..models.block_model import BlockModel

            if progress_callback:
                progress_callback(0, "Initializing audit-grade classification...")

            # Get block model
            block_model = params.get("block_model")
            if block_model is None:
                # Try to get from registry
                registry = self._app.registry if hasattr(self._app, 'registry') else None
                if registry is not None:
                    block_model = registry.get_block_model(copy_data=False)

            if block_model is None:
                raise ValueError("No block model provided for classification.")

            # Convert to DataFrame
            if isinstance(block_model, BlockModel):
                blocks_df = block_model.to_dataframe()
            elif isinstance(block_model, pd.DataFrame):
                blocks_df = block_model
            else:
                raise ValueError(f"Unsupported block model type: {type(block_model)}")

            if blocks_df.empty:
                raise ValueError("Block model is empty")

            if progress_callback:
                progress_callback(5, f"Loaded {len(blocks_df):,} blocks")

            # Convert params to DomainClassificationParams if needed
            params_by_domain = params["params_by_domain"]
            converted_params: Dict[str, DomainClassificationParams] = {}

            for domain, domain_params in params_by_domain.items():
                if isinstance(domain_params, DomainClassificationParams):
                    converted_params[domain] = domain_params
                elif isinstance(domain_params, dict):
                    try:
                        converted_params[domain] = DomainClassificationParams.from_dict(domain_params)
                    except Exception as e:
                        raise InvalidParameterError(f"Invalid parameters for domain '{domain}': {e}")
                else:
                    raise InvalidParameterError(
                        f"Parameters for domain '{domain}' must be DomainClassificationParams or dict"
                    )

            # Create and run engine
            engine = ResourceClassificationEngine(
                params_by_domain=converted_params,
                estimation_run_id=params.get("estimation_run_id"),
                drillhole_database_version=params.get("drillhole_database_version"),
                user=params.get("user", "unknown"),
            )

            if progress_callback:
                progress_callback(10, "Running classification engine...")

            result = engine.classify(blocks_df, progress_callback=progress_callback)

            if not result.success:
                raise ClassificationError(f"Classification failed: {result.errors}")

            # Write results back to block model if requested
            write_back = params.get("write_back", True)
            if write_back and isinstance(block_model, BlockModel) and result.classified_df is not None:
                block_model.add_property("resource_class", result.classified_df["resource_class"].values)
                block_model.add_property("class_reason", result.classified_df["class_reason"].values)
                block_model.add_property("kvn", result.classified_df["kvn"].values)
                block_model.add_property(
                    "classification_run_id",
                    np.full(len(result.classified_df), result.audit_metadata.classification_run_id)
                )

                # Register in registry if available
                registry = self._app.registry if hasattr(self._app, 'registry') else None
                if registry is not None:
                    registry.register_classified_block_model(
                        block_model,
                        source_panel="ResourceClassification",
                        metadata={
                            "run_id": result.audit_metadata.classification_run_id,
                            "summary": result.summary,
                        }
                    )

                    # Store audit metadata
                    audit_key = f"classification_audit_{result.audit_metadata.classification_run_id}"
                    registry.register_model(
                        key=audit_key,
                        data=result.audit_metadata.to_dict(),
                        metadata={"type": "classification_audit"},
                        source_panel="ResourceClassification",
                    )
                    registry.register_model(
                        key="latest_classification_audit",
                        data=result.audit_metadata.to_dict(),
                        metadata={"type": "classification_audit"},
                        source_panel="ResourceClassification",
                    )

                logger.info(f"Classification results written to block model (run_id={result.audit_metadata.classification_run_id})")

            if progress_callback:
                progress_callback(100, "Classification complete")

            # Log summary
            summary = result.summary
            logger.info(
                f"Audit-grade classification complete: "
                f"Measured={summary['Measured']['count']}, "
                f"Indicated={summary['Indicated']['count']}, "
                f"Inferred={summary['Inferred']['count']}, "
                f"Unclassified={summary['Unclassified']['count']}"
            )

            return {
                "name": "resource_classification",
                "operation": "audit_grade",
                "classified_df": result.classified_df,
                "block_model": block_model if isinstance(block_model, BlockModel) else None,
                "summary": result.summary,
                "audit_metadata": result.audit_metadata.to_dict(),
                "run_id": result.audit_metadata.classification_run_id,
                "signature": result.audit_metadata.classification_signature,
                "warnings": result.warnings,
                "metadata": {
                    "operation": "audit_grade",
                    "total_blocks": result.audit_metadata.total_blocks,
                    "domains_processed": result.audit_metadata.domains_processed,
                    "module_version": result.audit_metadata.module_version,
                    "timestamp": result.audit_metadata.timestamp,
                    "user": result.audit_metadata.user,
                },
            }

        # =====================================================================
        # Mode 3: JORC Classification (drillhole proximity based)
        # =====================================================================
        if "jorc_variogram" in params and "jorc_rules" in params:
            from ..models.jorc_classification_engine import (
                JORCClassificationEngine,
                VariogramModel,
                ClassificationRuleset,
            )
            
            if progress_callback:
                progress_callback(10, "Initializing JORC Classification Engine...")
            
            block_model = params.get("block_model")
            drillhole_df = params.get("drillhole_df")
            variogram_dict = params["jorc_variogram"]
            rules_dict = params["jorc_rules"]
            domain_value = params.get("domain_value")
            
            if block_model is None:
                raise ValueError("JORC classification: no block model provided.")
            if drillhole_df is None or len(drillhole_df) == 0:
                raise ValueError("JORC classification: no drillhole data provided.")
            
            # Create variogram model
            var = VariogramModel(
                range_major=variogram_dict.get("range_major", 100.0),
                sill=variogram_dict.get("sill", 1.0)
            )
            
            # Create ruleset
            rules = ClassificationRuleset.from_dict(rules_dict)
            
            if progress_callback:
                progress_callback(30, "Running JORC classification...")
            
            # Create engine and classify
            engine = JORCClassificationEngine(var, rules, domain_value=domain_value)
            result = engine.classify(block_model, drillhole_df)
            
            if progress_callback:
                progress_callback(90, "Adding classification results to BlockModel...")
            
            # --- NEW: Add classification results to BlockModel (standard API) ---
            from ..models.block_model import BlockModel
            
            # Check if input was BlockModel
            if isinstance(block_model, BlockModel):
                # Extract category from ClassificationResult
                if hasattr(result, 'classified_df') and 'CLASS_FINAL' in result.classified_df.columns:
                    categories = result.classified_df['CLASS_FINAL'].values
                    block_model.add_property('Category', categories)
                    block_model.add_property('CLASS_FINAL', categories)
                    
                    # Add reason if available
                    if 'CLASS_REASON' in result.classified_df.columns:
                        reasons = result.classified_df['CLASS_REASON'].values
                        block_model.add_property('CLASS_REASON', reasons)
                    
                    logger.info("✅ Added classification results to BlockModel")
            
            if progress_callback:
                progress_callback(100, "Classification complete.")
            
            return {
                "name": "resource_classification",
                "operation": "jorc",
                "classified_df": result,  # Keep for backward compatibility
                "block_model": block_model if isinstance(block_model, BlockModel) else None,  # ✅ NEW: Return BlockModel
                "summary": result.summary if hasattr(result, 'summary') else {},
                "ruleset": rules,
                "variogram": var,
                "domain_name": domain_value,
                "audit_records": result.audit_records if hasattr(result, 'audit_records') else [],
                "execution_time_seconds": result.execution_time_seconds if hasattr(result, 'execution_time_seconds') else 0.0,
                "metadata": {
                    "operation": "jorc",
                    "variogram": variogram_dict,
                    "rules": rules_dict,
                    "domain": domain_value or "Full Model Extent"
                }
            }
        
        # Mode 2: Drillhole spacing classification
        if "block_model" in params and "drillhole_df" in params:
            from ..models.resource_classification import (
                classify_by_spacing,
                get_classification_summary,
            )
            
            block_model = params["block_model"]
            drillhole_df = params["drillhole_df"]
            rc_params = params.get("params")
            
            if block_model is None:
                raise ValueError("Resource classification: no block model provided.")
            if drillhole_df is None or len(drillhole_df) == 0:
                raise ValueError("Resource classification: no drillhole data provided.")
            
            if hasattr(block_model, "to_dataframe"):
                block_df = block_model.to_dataframe()
            else:
                block_df = block_model
            
            if block_df is None or len(block_df) == 0:
                raise ValueError("Resource classification: block model is empty.")
            
            classified_df = classify_by_spacing(
                block_df=block_df,
                drillhole_df=drillhole_df,
                params=rc_params,
            )
            summary = get_classification_summary(classified_df)
            
            metadata = {
                "operation": "spacing",
                "total_blocks": int(len(classified_df)),
                "total_drillholes": int(len(drillhole_df)),
                "message": "Blocks classified by drillhole spacing.",
            }
            
            payload = {
                "name": "resource_classification",
                "operation": "spacing",
                "classified_df": classified_df,
                "summary": summary,
                "metadata": metadata,
            }
            return payload
        
        # Mode 1: Legacy estimation-based classification
        data_df = params["data_df"]
        dist_col = params["dist_col"]
        var_col = params["var_col"]
        samples_col = params["samples_col"]
        
        df = data_df.copy()
        
        missing_fields = []
        if dist_col == '<Not Available>' or dist_col not in df.columns:
            missing_fields.append("Distance")
        if var_col == '<Not Available>' or var_col not in df.columns:
            missing_fields.append("Variance")
        if samples_col == '<Not Available>' or samples_col not in df.columns:
            missing_fields.append("Sample Count")
        
        if missing_fields:
            raise ValueError(f"Missing fields: {', '.join(missing_fields)}")
        
        # Apply classification criteria
        cond_measured = (
            (df[dist_col] <= 25) &
            (df[var_col] <= 0.05) &
            (df[samples_col] >= 8)
        )
        
        cond_indicated = (
            (df[dist_col] <= 50) &
            (df[var_col] <= 0.15) &
            (df[samples_col] >= 4)
        )
        
        cond_inferred = (df[dist_col] > 50)
        
        df['CLASS'] = np.select(
            [cond_measured, cond_indicated, cond_inferred],
            ['Measured', 'Indicated', 'Inferred'],
            default='Unclassified'
        )
        
        class_summary = []
        for class_name in ['Measured', 'Indicated', 'Inferred', 'Unclassified']:
            class_mask = df['CLASS'] == class_name
            class_count = class_mask.sum()
            if class_count > 0:
                class_summary.append({
                    'Class': class_name,
                    'Blocks': int(class_count),
                    'Percentage': float((class_count / len(df)) * 100)
                })
        
        class_summary_df = pd.DataFrame(class_summary)
        
        metadata = {
            "operation": "classify",
            "classes": class_summary_df['Class'].tolist(),
            "total_blocks": int(len(df)),
            "message": "Resources classified into Measured/Indicated/Inferred.",
        }
        
        payload = {
            "name": "resource_classification",
            "operation": "classify",
            "data_df": df,
            "class_summary_df": class_summary_df,
            "metadata": metadata,
        }
        return payload

    # =========================================================================
    # Drillhole Resources - REMOVED (depended on ResourceCalculator which has been deleted)
    # =========================================================================

    # =========================================================================
    # IRR Analysis
    # =========================================================================
    
    def _prepare_irr_payload(self, params: Any, progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare IRR Analysis payload."""
        from ..irr_engine import run_irr
        from ..irr_engine.config_loader import IRRConfig
        
        # Extract classification filter parameters (if present)
        classification_filter = None
        classification_column = 'CLASSIFICATION'
        
        if isinstance(params, dict):
            # MP-018 FIX: Copy dict to avoid mutating the original
            params_copy = params.copy()
            # Extract classification params before creating IRRConfig
            classification_filter = params_copy.pop('classification_filter', None)
            classification_column = params_copy.pop('classification_column', 'CLASSIFICATION')
            config = IRRConfig(**params_copy)
        else:
            config = params
        
        def progress_wrapper(iteration, r_trial, satisfaction_rate):
            if progress_callback:
                progress_callback(int(iteration * 100 / config.max_iterations), f"Iteration {iteration}: r={r_trial:.4f}")
        
        # Run IRR analysis with classification filter support
        # MP-015 FIX: Changed strict_classification to True for JORC/SAMREC compliance
        result = run_irr(
            config, 
            progress_callback=progress_wrapper,
            classification_filter=classification_filter,
            strict_classification=True,  # Enforce classification for JORC/SAMREC compliance
            store_all_cashflows=False,    # Don't store all cashflows by default
            validate_inputs=True
        )
        
        return {
            "name": "irr",
            "irr_result": result,
            "metadata": {
                "irr_alpha": result.irr_alpha,
                "satisfaction_rate": result.satisfaction_rate,
                "num_scenarios": result.num_scenarios,
                "classification_filter_applied": result.classification_filter_applied,
                "blocks_before_filter": result.blocks_before_filter,
                "blocks_after_filter": result.blocks_after_filter
            }
        }

    # =========================================================================
    # NPV Analysis
    # =========================================================================
    
    def _prepare_npv_payload(self, config: Any, progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare NPV Calculation payload."""
        from ..irr_engine import run_npv
        from ..irr_engine.config_loader import ScheduleConfig
        
        if isinstance(config, dict):
            config = ScheduleConfig(**config)
        
        if progress_callback:
            progress_callback(50, "Optimizing schedule...")
        
        result = run_npv(config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "npv",
            "npv_result": result,
            "metadata": {
                "npv": result.get("npv", 0),
                "revenue": result.get("revenue", 0),
                "operating_cost": result.get("operating_cost", 0)
            }
        }

    # =========================================================================
    # NPVS Optimization
    # =========================================================================
    
    def _prepare_npvs_run_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare NPVS Run payload."""
        from ..mine_planning.npvs.npvs_solver import run_npvs
        
        block_model = params.get("block_model")
        if block_model is None:
            raise ValueError("block_model is required")
        
        result = run_npvs(params)
        
        if progress_callback:
            progress_callback(100, "NPVS optimization complete")
        
        return {
            "name": "npvs_run",
            "result": result
        }

    # =========================================================================
    # Pit Optimisation / Underground
    # =========================================================================
    
    def _prepare_pit_optimisation_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Pit Optimisation payload - pure computation.
        
        Calculates block values based on economics and runs Lerchs-Grossmann optimization.
        This is a pure function with no access to DataRegistry, controller, or globals.
        
        MP-002 FIX: No silent defaults for economic parameters.
        All required parameters must be explicitly provided.
        """
        import pandas as pd
        import numpy as np
        from ..models.pit_optimizer import lerchs_grossmann_optimize_fast, is_fast_solver_available
        
        if progress_callback:
            progress_callback(10, "Starting Pit Optimization...")
        
        # Extract parameters - MP-002 FIX: No silent defaults for critical parameters
        block_model = params.get('block_model')
        if block_model is None:
            raise ValueError("block_model is required")
        
        # Required economic parameters - MUST be explicitly provided
        REQUIRED_PARAMS = ['price', 'mining_cost', 'proc_cost', 'recovery', 'grade_col']
        missing_params = [p for p in REQUIRED_PARAMS if p not in params or params[p] is None]
        if missing_params:
            raise ValueError(
                f"Missing required pit optimization parameters: {missing_params}. "
                "All economic parameters must be explicitly provided - no defaults allowed for safety."
            )
        
        price = params['price']
        cost_m = params['mining_cost']
        cost_p = params['proc_cost']
        rec = params['recovery']
        grade_col = params['grade_col']
        use_fast_solver = params.get('use_fast_solver', True)
        
        # Validate parameter values
        if price <= 0:
            raise ValueError(f"Metal price must be positive, got {price}")
        if rec <= 0 or rec > 1:
            raise ValueError(f"Recovery must be between 0 and 1, got {rec}")
        if cost_m < 0:
            raise ValueError(f"Mining cost cannot be negative, got {cost_m}")
        if cost_p < 0:
            raise ValueError(f"Processing cost cannot be negative, got {cost_p}")
        
        # --- NEW: Handle BlockModel input (standard API) ---
        from ..models.block_model import BlockModel
        
        if isinstance(block_model, BlockModel):
            # ✅ STANDARD API: Use BlockModel methods
            if progress_callback:
                progress_callback(15, f"Using BlockModel API ({block_model.block_count} blocks)")
            
            # Convert to DataFrame for processing (temporary - will migrate optimizer later)
            df = block_model.to_dataframe()
            
            # Validate grade field exists
            if grade_col not in block_model.get_property_names():
                available = ', '.join(block_model.get_property_names())
                raise ValueError(f"Grade column '{grade_col}' not found in BlockModel. Available: {available}")
        else:
            # Legacy DataFrame input
            df = block_model.copy()
            if progress_callback:
                progress_callback(15, f"⚠️ Using DataFrame API (legacy) - {len(df)} blocks")
        
        if progress_callback:
            progress_callback(20, f"Using grade column: {grade_col}")
        
        # Check if grade column exists
        if grade_col not in df.columns:
            raise ValueError(f"Grade column '{grade_col}' not found in block model. Available columns: {', '.join(df.columns)}")
        
        # Auto-detect if grade is %
        if df[grade_col].mean() > 1.0:
            grade_factor = 0.01
            if progress_callback:
                progress_callback(25, "Detected grade in percentage format, converting to decimal")
        else:
            grade_factor = 1.0
        
        # Get tonnage column (try common names)
        tonnage_col = None
        for col in ['TONNAGE', 'TONNES', 'T', 'MASS', 'WEIGHT']:
            if col in df.columns:
                tonnage_col = col
                break
        
        if tonnage_col is None:
            # Try to calculate from volume and density
            volume_col = None
            density_col = None
            for col in ['VOLUME', 'VOL', 'V']:
                if col in df.columns:
                    volume_col = col
                    break
            for col in ['DENSITY', 'DENS', 'RHO']:
                if col in df.columns:
                    density_col = col
                    break
            
            if volume_col and density_col:
                df['TONNAGE'] = df[volume_col] * df[density_col]
                tonnage_col = 'TONNAGE'
                if progress_callback:
                    progress_callback(25, f"Calculated tonnage from {volume_col} × {density_col}")
            else:
                # MP-010 FIX: Warn about missing tonnage instead of silently defaulting
                logger.warning(
                    "TONNAGE WARNING: No tonnage column found and cannot calculate from volume×density. "
                    "Using 1.0 tonne per block - THIS WILL PRODUCE INCORRECT ECONOMICS. "
                    "Ensure block model has TONNAGE, or VOLUME+DENSITY columns."
                )
                df['TONNAGE'] = 1.0
                tonnage_col = 'TONNAGE'
                if progress_callback:
                    progress_callback(25, "⚠ WARNING: No tonnage column found, assuming 1.0 tonne per block - RESULTS MAY BE INCORRECT")
        
        if progress_callback:
            progress_callback(30, "Calculating block values...")
        
        # Simple Revenue Model
        # Revenue = Tonnage × Grade × Recovery × Price
        # Cost = Tonnage × (Mining_Cost + Processing_Cost)
        # Value = Revenue - Cost
        
        rev = df[tonnage_col] * (df[grade_col] * grade_factor) * rec * price
        cost = df[tonnage_col] * (cost_m + cost_p)
        df['LG_VALUE'] = rev - cost
        
        positive_blocks = (df['LG_VALUE'] > 0).sum()
        negative_blocks = (df['LG_VALUE'] < 0).sum()
        if progress_callback:
            progress_callback(40, f"Calculated block values. Positive blocks: {positive_blocks}, Negative: {negative_blocks}")
        
        # Run Solver
        if progress_callback:
            progress_callback(60, "Building graph and solving maximum flow problem...")
        
        if use_fast_solver and is_fast_solver_available():
            if progress_callback:
                progress_callback(70, "Using fast SciPy-based solver")
            in_pit_mask = lerchs_grossmann_optimize_fast(df, value_col='LG_VALUE')
        else:
            if progress_callback:
                progress_callback(70, "Warning: Fast solver not available, using fallback")
            raise ImportError("Fast solver requires scipy. Install with: pip install scipy")
        
        if progress_callback:
            progress_callback(90, "Tagging results...")
        
        # Tag Result
        df['IN_PIT'] = in_pit_mask.astype(int)
        
        if progress_callback:
            progress_callback(100, "Optimization complete!")
        
        # Return results compatible with BlockModel API
        in_pit_array = in_pit_mask.astype(int)
        lg_value_array = df['LG_VALUE'].values
        
        result = {
            "name": "pit_opt",
            "in_pit": in_pit_array,  # numpy array (can be added to BlockModel)
            "lg_value": lg_value_array,  # numpy array
            "total_value": float(df[df['IN_PIT'] == 1]['LG_VALUE'].sum()),
            "blocks_in_pit": int((df['IN_PIT'] == 1).sum()),
            # Keep DataFrame for backward compatibility (UI may still expect it)
            "result_df": df
        }
        
        # If input was BlockModel, add results to it
        if isinstance(block_model, BlockModel):
            block_model.add_property('IN_PIT', in_pit_array)
            block_model.add_property('LG_VALUE', lg_value_array)
            result["block_model"] = block_model  # Return updated BlockModel
        
        return result

    def _prepare_underground_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare Underground Planning payload - routes to appropriate operation.
        
        Supported operations:
        - optimize_stopes: Stope optimization using maximum closure algorithm
        - schedule_production: MILP-based production scheduling
        - analyze_ground_control: Rock mass classification (RMR, Q-system)
        - schedule_equipment: Equipment requirements calculation
        - design_ventilation: Ventilation system design
        """
        operation = params.get("operation")
        
        if progress_callback:
            progress_callback(5, f"Starting underground operation: {operation}")
        
        if operation == "optimize_stopes":
            return self._run_stope_optimization(params, progress_callback)
        elif operation == "schedule_production":
            return self._run_ug_schedule_production(params, progress_callback)
        elif operation == "analyze_ground_control":
            return self._run_ground_control_analysis(params, progress_callback)
        elif operation == "schedule_equipment":
            return self._run_equipment_scheduling(params, progress_callback)
        elif operation == "design_ventilation":
            return self._run_ventilation_design(params, progress_callback)
        else:
            raise ValueError(f"Unknown underground operation: {operation}")
    
    def _run_stope_optimization(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Run stope optimization using maximum closure algorithm."""
        from ..ug.stope_opt.optimizer import optimize_stopes
        import pandas as pd
        
        blocks_df = params.get("blocks_df")
        if blocks_df is None:
            raise ValueError("blocks_df is required for stope optimization")
        
        if progress_callback:
            progress_callback(10, "Preparing block model for stope optimization...")
        
        # Ensure blocks_df has required columns
        if not isinstance(blocks_df, pd.DataFrame):
            raise ValueError("blocks_df must be a pandas DataFrame")
        
        # Calculate NSR if not present
        if 'nsr' not in blocks_df.columns:
            if progress_callback:
                progress_callback(20, "Calculating Net Smelter Return (NSR)...")
            
            # Get grade column
            grade_cols = [c for c in blocks_df.columns if 'grade' in c.lower() or c in ['Fe', 'Cu', 'Au', 'Ag']]
            if grade_cols:
                grade_col = grade_cols[0]
                metal_price = params.get("metal_price", 60.0)
                recovery = params.get("recovery", 0.85)
                processing_cost = params.get("processing_cost", 25.0)
                mining_cost = params.get("mining_cost", 3.0)
                
                # Simple NSR calculation
                blocks_df['nsr'] = (blocks_df[grade_col] * metal_price * recovery) - (processing_cost + mining_cost)
            else:
                logger.warning("No grade column found - using zero NSR")
                blocks_df['nsr'] = 0.0
        
        if progress_callback:
            progress_callback(30, "Running maximum closure stope optimization...")
        
        # Build params dict for optimizer
        opt_params = {
            'min_nsr_diluted': params.get("min_nsr", 0.0),
            'dilation_skin_m': params.get("dilution_skin", 0.5),
            'min_length': params.get("stope_length", 30.0) * 0.5,
            'max_length': params.get("stope_length", 30.0) * 2.0,
            'min_width': params.get("stope_width", 15.0) * 0.5,
            'max_width': params.get("stope_width", 15.0) * 2.0,
            'min_height': params.get("stope_height", 30.0) * 0.5,
            'max_height': params.get("stope_height", 30.0) * 2.0,
            'algorithm': 'max_closure'
        }
        
        stopes = optimize_stopes(blocks_df, opt_params)
        
        if progress_callback:
            progress_callback(90, f"Found {len(stopes)} stopes")
        
        # Build diagnostics
        diagnostics = {
            'blocks_above_threshold': int((blocks_df['nsr'] >= params.get("min_nsr", 0.0)).sum()),
            'nsr_stats_pre_filter': {
                'min': float(blocks_df['nsr'].min()),
                'mean': float(blocks_df['nsr'].mean()),
                'max': float(blocks_df['nsr'].max())
            },
            'min_nsr_threshold': params.get("min_nsr", 0.0)
        }
        
        if progress_callback:
            progress_callback(100, "Stope optimization complete")
        
        return {
            "name": "underground", 
            "operation": "optimize_stopes", 
            "results": {
                "stopes": stopes,
                "blocks_df": blocks_df,
                "diagnostics": diagnostics
            }
        }
    
    def _run_ug_schedule_production(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Run underground production scheduling.
        
        Note: This uses a simplified greedy scheduler since the SLOS optimizer
        requires additional infrastructure. A full MILP scheduler would be
        implemented in a future iteration.
        """
        from dataclasses import dataclass, field
        
        @dataclass
        class SchedulePeriod:
            """Simple schedule period for UG scheduling results."""
            t: int
            ore_mined: float = 0.0
            ore_proc: float = 0.0
            fill_placed: float = 0.0
            stockpile: float = 0.0
            cashflow: float = 0.0
            dcf: float = 0.0
            stopes_mined: list = field(default_factory=list)
        
        stopes = params.get("stopes")
        if not stopes:
            raise ValueError("stopes are required for production scheduling")
        
        if progress_callback:
            progress_callback(10, "Running production scheduling...")
        
        n_periods = params.get("n_periods", 24)
        mine_capacity = params.get("mine_capacity", 10000.0)
        mill_capacity = params.get("mill_capacity", 10000.0)
        fill_capacity = params.get("fill_capacity", 8000.0)
        discount_rate = params.get("discount_rate", 0.10)
        curing_lag = params.get("curing_lag", 2)
        
        if progress_callback:
            progress_callback(20, f"Scheduling {len(stopes)} stopes over {n_periods} periods...")
        
        # Simple greedy scheduling - sort stopes by NSR and assign to periods
        sorted_stopes = sorted(stopes, key=lambda s: getattr(s, 'nsr_dil', getattr(s, 'nsr', 0)), reverse=True)
        
        schedule = []
        stope_idx = 0
        
        for t in range(n_periods):
            if progress_callback and t % 5 == 0:
                progress_callback(20 + int(70 * t / n_periods), f"Scheduling period {t+1}/{n_periods}...")
            
            period_tonnes = 0.0
            period_value = 0.0
            stopes_in_period = []
            
            # Fill period up to capacity
            while stope_idx < len(sorted_stopes) and period_tonnes < mine_capacity:
                stope = sorted_stopes[stope_idx]
                stope_tonnes = getattr(stope, 'tonnes_dil', getattr(stope, 'tonnes', 0))
                
                if period_tonnes + stope_tonnes <= mine_capacity:
                    period_tonnes += stope_tonnes
                    period_value += stope_tonnes * getattr(stope, 'nsr_dil', getattr(stope, 'nsr', 0))
                    stopes_in_period.append(stope.id)
                    stope_idx += 1
                else:
                    break
            
            # Calculate DCF
            discount_factor = 1.0 / ((1 + discount_rate) ** t)
            dcf = period_value * discount_factor
            
            # Create period record
            period = SchedulePeriod(
                t=t,
                ore_mined=period_tonnes,
                ore_proc=min(period_tonnes, mill_capacity),
                fill_placed=min(period_tonnes * 0.8, fill_capacity),  # Simplified fill
                stockpile=max(0, period_tonnes - mill_capacity),
                cashflow=period_value,
                dcf=dcf,
                stopes_mined=stopes_in_period
            )
            schedule.append(period)
        
        if progress_callback:
            progress_callback(100, "Production scheduling complete")
        
        return {"name": "underground", "operation": "schedule_production", "results": {"schedule": schedule}}
    
    def _run_ground_control_analysis(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Run ground control analysis (RMR, Q-system)."""
        from ..ug.ground_control.analyzer import (
            RockMassProperties, calculate_rmr, calculate_q_system, 
            estimate_pillar_strength, calculate_pillar_fos, select_support
        )
        
        if progress_callback:
            progress_callback(10, "Running ground control analysis...")
        
        # Create rock mass properties
        props = RockMassProperties(
            ucs=params.get("ucs", 80.0),
            rqd=params.get("rqd", 75.0),
            spacing=params.get("spacing", 0.3),
            condition=params.get("condition", 3),
            groundwater=params.get("groundwater", 10),
            orientation=params.get("orientation", -5)
        )
        
        if progress_callback:
            progress_callback(30, "Calculating Rock Mass Rating (RMR)...")
        
        # Calculate RMR
        rmr, rmr_class = calculate_rmr(props)
        
        if progress_callback:
            progress_callback(50, "Calculating Q-system...")
        
        # Calculate Q-system
        jn = params.get("jn", 9)
        jr = params.get("jr", 3)
        ja = params.get("ja", 2)
        jw = params.get("jw", 1.0)
        srf = 2.5  # Medium stress
        
        q_value = calculate_q_system(props.rqd, jn, jr, ja, jw, srf)
        
        # Pillar analysis if requested
        pillar_fos = None
        if params.get("calculate_pillar", False):
            if progress_callback:
                progress_callback(70, "Calculating pillar Factor of Safety...")
            
            pillar_width = params.get("pillar_width", 6.0)
            pillar_height = params.get("pillar_height", 3.0)
            stress = 10.0  # Assumed stress (MPa)
            
            pillar_strength = estimate_pillar_strength(props.ucs, pillar_width, pillar_height)
            pillar_fos = calculate_pillar_fos(pillar_strength, stress)
        
        if progress_callback:
            progress_callback(90, "Selecting support requirements...")
        
        # Get support recommendations
        support = select_support(rmr, span=10.0)  # Assume 10m span
        
        result = {
            'rmr': rmr,
            'q_value': q_value,
            'pillar_fos': pillar_fos,
            'support': support,
            'properties': props
        }
        
        if progress_callback:
            progress_callback(100, "Ground control analysis complete")
        
        return {"name": "underground", "operation": "analyze_ground_control", "results": result}
    
    def _run_equipment_scheduling(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Run equipment requirements calculation."""
        from ..ug.equipment.scheduler import calculate_equipment_requirements
        
        schedule = params.get("schedule")
        if not schedule:
            raise ValueError("schedule is required for equipment calculation")
        
        if progress_callback:
            progress_callback(10, "Calculating equipment requirements...")
        
        # Calculate annual production from schedule
        total_tonnes = sum(getattr(p, 'ore_mined', 0) for p in schedule)
        periods = len(schedule)
        period_days = params.get("period_days", 30)
        annual_production = total_tonnes * (365 / (periods * period_days)) if periods > 0 else 0
        
        haul_distance = params.get("haul_distance", 500.0)
        
        if progress_callback:
            progress_callback(50, f"Estimating fleet for {annual_production:,.0f} t/year production...")
        
        # Calculate equipment requirements
        equipment = calculate_equipment_requirements(
            annual_production=annual_production,
            mine_depth=500.0,  # Default depth
            haul_distance=haul_distance
        )
        
        # Convert to serializable format
        equipment_dict = {eq_type.value: count for eq_type, count in equipment.items()}
        
        if progress_callback:
            progress_callback(100, "Equipment calculation complete")
        
        return {"name": "underground", "operation": "schedule_equipment", "results": {"equipment": equipment_dict}}
    
    def _run_ventilation_design(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Run ventilation system design."""
        from ..ug.ventilation.network_solver import design_main_fan, calculate_fan_duty
        
        if progress_callback:
            progress_callback(10, "Designing ventilation system...")
        
        required_airflow = params.get("required_airflow", 250.0)
        total_resistance = params.get("total_resistance", 0.05)
        fan_efficiency = params.get("fan_efficiency", 0.75)
        
        if progress_callback:
            progress_callback(50, "Calculating main fan requirements...")
        
        # Design main fan
        fan_design = design_main_fan(
            total_airflow=required_airflow,
            total_resistance=total_resistance,
            efficiency=fan_efficiency,
            pressure_margin=1.15
        )
        
        # Calculate operating cost (simplified)
        power_kw = fan_design['rated_power_kw']
        hours_per_year = 8760
        electricity_cost = 0.10  # $/kWh
        operating_cost_per_year = power_kw * hours_per_year * electricity_cost
        
        result = {
            'ventilation': {
                'airflow': fan_design['duty_flow_m3s'],
                'pressure': fan_design['duty_pressure_pa'],
                'power': fan_design['rated_power_kw'],
                'operating_cost': operating_cost_per_year,
                'efficiency': fan_design['efficiency']
            }
        }
        
        if progress_callback:
            progress_callback(100, "Ventilation design complete")
        
        return {"name": "underground", "operation": "design_ventilation", "results": result}

    def _prepare_esg_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Prepare ESG Dashboard payload - pure computation.
        
        Handles carbon_footprint, water_balance, waste_tracking, and generate_reports operations.
        """
        operation = params.get("operation")
        
        if operation == "carbon_footprint":
            from ..esg.carbon_energy.calculator import calc_co2e, create_default_emission_factors
            from dataclasses import dataclass
            
            @dataclass
            class PeriodKPI:
                """Simple PeriodKPI for carbon calculation."""
                t: int
                ore_mined: float
                ore_proc: float
                activity: str = 'total'
            
            if progress_callback:
                progress_callback(10, "Calculating carbon footprint...")
            
            schedule = params.get("schedule")
            if schedule is None:
                raise ValueError("schedule is required for carbon_footprint")
            
            # Convert schedule to PeriodKPI objects
            period_kpis = []
            for item in schedule:
                if isinstance(item, dict):
                    kpi = PeriodKPI(
                        t=item.get('period', 0),
                        ore_mined=item.get('ore_mined', 0.0),
                        ore_proc=item.get('ore_proc', item.get('ore_mined', 0.0)),
                        activity=item.get('activity', 'total')
                    )
                    period_kpis.append(kpi)
                else:
                    # Assume it's already a PeriodKPI-like object
                    period_kpis.append(item)
            
            # Get emission factors
            ef_df = create_default_emission_factors()
            energy_mix = {}  # Use defaults
            
            # Calculate CO2e
            results_df = calc_co2e(period_kpis, ef_df, energy_mix)
            
            # Aggregate results
            total_co2e = results_df['co2e_total_t'].sum()
            intensity = (total_co2e * 1000.0) / results_df['co2e_per_t_ore'].sum() if len(results_df) > 0 else 0.0
            
            # Create summary by activity
            summary = results_df.groupby('activity')['co2e_total_t'].sum().reset_index()
            summary.columns = ['activity', 'co2e_tonnes']
            
            if progress_callback:
                progress_callback(100, "Carbon footprint calculated")
            
            return {
                "name": "esg_carbon_footprint",
                "operation": "carbon_footprint",
                "results": {
                    "total_co2e": total_co2e,
                    "intensity": intensity,
                    "summary": summary,
                    "period_data": results_df
                }
            }
        
        elif operation == "water_balance":
            from ..esg.water.water_balance import simulate_water_balance, calculate_water_footprint, WaterNode, NodeType, WaterLink
            
            if progress_callback:
                progress_callback(10, "Simulating water balance...")
            
            schedule = params.get("schedule")
            if schedule is None:
                raise ValueError("schedule is required for water_balance")
            
            # Extract water balance parameters
            pit_capacity = params.get("pit_capacity", 50000.0)
            pit_inflow = params.get("pit_inflow", 100.0)
            tailings_capacity = params.get("tailings_capacity", 500000.0)
            tailings_area = params.get("tailings_area", 50.0)
            period_days = params.get("period_days", 30)
            
            # Create water nodes
            nodes = [
                WaterNode(node_id="pit", node_type=NodeType.PIT, capacity_m3=pit_capacity, area_m2=10000.0),
                WaterNode(node_id="tailings", node_type=NodeType.TAILINGS, capacity_m3=tailings_capacity, area_m2=tailings_area * 10000.0),
                WaterNode(node_id="recycling", node_type=NodeType.RECYCLING_POND, capacity_m3=50000.0, area_m2=5000.0)
            ]
            
            # Create links (simplified)
            links = []
            
            # Process water demand from schedule
            process_water_demand = {}
            precipitation = {}
            for item in schedule:
                period = item.get('period', 0)
                ore_proc = item.get('ore_proc', item.get('ore_mined', 0.0))
                # Assume 0.5 m³/t water demand
                process_water_demand[period] = (ore_proc * 0.5) / period_days  # m³/day
                precipitation[period] = pit_inflow / period_days  # mm/day (simplified)
            
            n_periods = len(schedule)
            
            # Simulate water balance
            balances = simulate_water_balance(nodes, links, process_water_demand, precipitation, n_periods, period_days)
            
            # Calculate totals
            total_water_use = sum(b.water_use for b in balances)
            total_recycled = sum(b.recycled for b in balances)
            total_production = sum(item.get('ore_proc', item.get('ore_mined', 0.0)) for item in schedule)
            
            # Calculate footprint
            footprint = calculate_water_footprint(total_water_use, total_recycled, total_production)
            
            if progress_callback:
                progress_callback(100, "Water balance calculated")
            
            return {
                "name": "esg_water_balance",
                "operation": "water_balance",
                "results": {
                    "total_water_use": total_water_use,
                    "recycled_water": total_recycled,
                    "footprint": footprint,
                    "balances": balances
                }
            }
        
        elif operation == "waste_tracking":
            from ..esg.waste_land.waste_land import track_waste_rock, calculate_disturbance
            
            if progress_callback:
                progress_callback(10, "Tracking waste rock...")
            
            schedule = params.get("schedule")
            if schedule is None:
                raise ValueError("schedule is required for waste_tracking")
            
            strip_ratio = params.get("strip_ratio", 3.0)
            pag_percentage = params.get("pag_percentage", 0.05)
            
            # Convert schedule to format expected by track_waste_rock
            mining_schedule = []
            for item in schedule:
                mining_schedule.append({
                    'period': item.get('period', 0),
                    'ore_mined': item.get('ore_mined', 0.0)
                })
            
            # Track waste
            waste_reports = track_waste_rock(mining_schedule, strip_ratio=strip_ratio, pag_percentage=pag_percentage)
            
            # Calculate disturbance
            disturbance = calculate_disturbance(mining_schedule, strip_ratio=strip_ratio)
            
            # Aggregate totals
            total_waste = sum(r.waste_generated_t for r in waste_reports)
            total_disturbance = sum(disturbance.values())
            
            if progress_callback:
                progress_callback(100, "Waste tracking complete")
            
            return {
                "name": "esg_waste_tracking",
                "operation": "waste_tracking",
                "results": {
                    "total_waste": total_waste,
                    "total_disturbance": total_disturbance,
                    "waste_reports": waste_reports,
                    "disturbance": disturbance
                }
            }
        
        elif operation == "generate_reports":
            from ..esg.governance.reporting import generate_gri_report, ESGMetrics
            
            if progress_callback:
                progress_callback(10, "Generating ESG reports...")
            
            carbon_data = params.get("carbon_data", {})
            water_data = params.get("water_data", {})
            waste_data = params.get("waste_data", {})
            
            # Create metrics
            metrics = ESGMetrics(
                carbon_metrics={
                    "total_co2e": carbon_data.get("total_co2e", 0.0),
                    "intensity": carbon_data.get("intensity", 0.0)
                },
                water_metrics={
                    "total_water_use": water_data.get("total_water_use", 0.0),
                    "recycled_water": water_data.get("recycled_water", 0.0),
                    "water_intensity": water_data.get("footprint", {}).get("water_intensity_m3_per_t", 0.0)
                },
                waste_metrics={
                    "total_waste": waste_data.get("total_waste", 0.0),
                    "total_disturbance": waste_data.get("total_disturbance", 0.0)
                }
            )
            
            # Generate report
            report = generate_gri_report(metrics)
            
            if progress_callback:
                progress_callback(100, "Reports generated")
            
            return {
                "name": "esg_generate_reports",
                "operation": "generate_reports",
                "results": {
                    "report": report,
                    "metrics": metrics
                }
            }
        
        else:
            raise ValueError(f"Unknown ESG operation: {operation}")

    # =========================================================================
    # Geometallurgy
    # =========================================================================
    
    def _prepare_geomet_assign_ore_types_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Geomet Assign Ore Types payload."""
        from ..geomet.geomet_block_model import assign_ore_types_to_blocks
        
        if progress_callback:
            progress_callback(10, "Assigning ore types to blocks...")
        
        block_model = params.get("block_model")
        geomet_domain_map = params.get("geomet_domain_map")
        rules = params.get("rules", {})
        
        if block_model is None or geomet_domain_map is None:
            raise ValueError("block_model and geomet_domain_map required")
        
        ore_type_codes = assign_ore_types_to_blocks(block_model, geomet_domain_map, rules)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "geomet_assign_ore_types",
            "ore_type_codes": ore_type_codes,
            "n_blocks": len(ore_type_codes)
        }

    def _prepare_geomet_compute_block_attrs_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Geomet Compute Block Attributes payload."""
        from ..geomet.geomet_block_model import compute_geomet_attributes_for_blocks
        
        if progress_callback:
            progress_callback(10, "Computing geomet attributes...")
        
        block_model = params.get("block_model")
        plant_config = params.get("plant_config")
        domain_map = params.get("domain_map")
        liberation_models = params.get("liberation_models", {})
        comminution_props = params.get("comminution_props", {})
        
        if block_model is None or plant_config is None or domain_map is None:
            raise ValueError("block_model, plant_config, and domain_map required")
        
        geomet_attrs = compute_geomet_attributes_for_blocks(
            block_model, plant_config, domain_map, liberation_models, comminution_props
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "geomet_compute_block_attrs",
            "geomet_attrs": geomet_attrs,
            "n_blocks": len(geomet_attrs.ore_type_code)
        }

    def _prepare_geomet_plant_response_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Geomet Plant Response payload."""
        from ..geomet.plant_response import evaluate_ore_type_response
        
        if progress_callback:
            progress_callback(10, "Evaluating plant response...")
        
        ore_type_code = params.get("ore_type_code")
        chemistry = params.get("chemistry", {})
        plant_config = params.get("plant_config")
        liberation_models = params.get("liberation_models", {})
        comminution_props = params.get("comminution_props", {})
        
        if ore_type_code is None or plant_config is None:
            raise ValueError("ore_type_code and plant_config required")
        
        response = evaluate_ore_type_response(
            ore_type_code, chemistry, plant_config, liberation_models, comminution_props
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "geomet_plant_response",
            "response": response
        }

    def _prepare_geomet_compute_values_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Geomet Compute Values payload."""
        from ..geomet_chain.geomet_value_engine import compute_geomet_block_values
        
        block_model = params.get("block_model")
        geomet_config = params.get("geomet_config")
        
        if block_model is None:
            raise ValueError("block_model is required")
        if geomet_config is None:
            raise ValueError("geomet_config is required")
        
        value_field = compute_geomet_block_values(block_model, geomet_config)
        
        if progress_callback:
            progress_callback(100, "Geomet block values computed")
        
        return {
            "name": "geomet_compute_values",
            "result": {"value_field": value_field}
        }

    # =========================================================================
    # Grade Control
    # =========================================================================
    
    def _prepare_gc_build_support_model_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare GC Build Support Model payload."""
        from ..grade_control.support_model import derive_gc_grid_from_long_term, resample_long_term_to_gc
        
        if progress_callback:
            progress_callback(10, "Deriving GC grid...")
        
        long_model = params.get("long_model")
        smu_size = params.get("smu_size", (2.5, 2.5, 2.5))
        method = params.get("method", "volume_weighted")
        
        if long_model is None:
            raise ValueError("long_model required")
        
        gc_grid = derive_gc_grid_from_long_term(long_model, smu_size)
        
        if progress_callback:
            progress_callback(50, "Resampling to GC grid...")
        
        gc_model = resample_long_term_to_gc(long_model, gc_grid, method)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {
            "name": "gc_build_support_model",
            "gc_grid": gc_grid,
            "gc_model": gc_model
        }

    def _prepare_gc_ok_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare GC OK payload."""
        from ..grade_control.gc_kriging import run_gc_ok, GCKrigingConfig
        
        if progress_callback:
            progress_callback(10, "Running GC kriging...")
        
        samples = params.get("samples")
        gc_grid = params.get("gc_grid")
        config_dict = params.get("config")
        
        if samples is None or gc_grid is None or config_dict is None:
            raise ValueError("samples, gc_grid, and config required")
        
        config = GCKrigingConfig(**config_dict)
        result = run_gc_ok(samples, gc_grid, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "gc_ok", "result": result}

    def _prepare_gc_sgsim_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare GC SGSIM payload."""
        from ..grade_control.gc_simulation import run_gc_sgsim, GCSimulationConfig
        
        if progress_callback:
            progress_callback(10, "Running GC simulation...")
        
        samples = params.get("samples")
        gc_grid = params.get("gc_grid")
        config_dict = params.get("config")
        
        if samples is None or gc_grid is None or config_dict is None:
            raise ValueError("samples, gc_grid, and config required")
        
        config = GCSimulationConfig(**config_dict)
        config.gc_grid = gc_grid
        result = run_gc_sgsim(samples, gc_grid, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "gc_sgsim", "result": result}

    def _prepare_gc_classify_ore_waste_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare GC Classify Ore/Waste payload."""
        from ..grade_control.ore_waste_marking import classify_gc_blocks, OreWasteCutoffRule
        
        if progress_callback:
            progress_callback(10, "Classifying GC blocks...")
        
        gc_model = params.get("gc_model")
        cutoff_rules_dict = params.get("cutoff_rules", [])
        
        if gc_model is None:
            raise ValueError("gc_model required")
        
        cutoff_rules = [OreWasteCutoffRule(**rule) for rule in cutoff_rules_dict]
        result = classify_gc_blocks(gc_model, cutoff_rules)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "gc_classify_ore_waste", "result": result}

    def _prepare_gc_summarise_digpolys_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare GC Summarise Dig Polygons payload."""
        from ..grade_control.ore_waste_marking import summarise_by_digpolygon
        
        if progress_callback:
            progress_callback(10, "Summarizing dig polygons...")
        
        gc_model = params.get("gc_model")
        diglines = params.get("diglines")
        ore_waste_result = params.get("ore_waste_result")
        density_property = params.get("density_property", "density")
        
        if gc_model is None or diglines is None or ore_waste_result is None:
            raise ValueError("gc_model, diglines, and ore_waste_result required")
        
        summaries = summarise_by_digpolygon(gc_model, diglines, ore_waste_result, density_property)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "gc_summarise_digpolys", "summaries": summaries}

    # =========================================================================
    # Reconciliation
    # =========================================================================
    
    def _prepare_recon_model_mine_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Recon Model-Mine payload."""
        from ..reconciliation.model_to_mine import build_model_mine_series
        
        if progress_callback:
            progress_callback(10, "Building reconciliation series...")
        
        long_model = params.get("long_model")
        gc_model = params.get("gc_model")
        diglines = params.get("diglines")
        mined_table = params.get("mined_table")
        density_property = params.get("density_property", "density")
        grade_properties = params.get("grade_properties", [])
        
        if mined_table is None:
            raise ValueError("mined_table required")
        
        if isinstance(mined_table, dict):
            mined_table = pd.DataFrame(mined_table)
        
        series_dict = build_model_mine_series(
            long_model, gc_model, diglines, mined_table, density_property, grade_properties
        )
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "recon_model_mine", "series": series_dict}

    def _prepare_recon_mine_mill_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Recon Mine-Mill payload."""
        from ..reconciliation.mine_to_mill import build_mine_mill_series
        from ..reconciliation.tonnage_grade_balance import TonnageGradeSeries, TonnageGradeRecord
        
        if progress_callback:
            progress_callback(10, "Building mine-mill series...")
        
        mined_series_dict = params.get("mined_series")
        plant_feed_table = params.get("plant_feed_table")
        recovery_table = params.get("recovery_table")
        
        if mined_series_dict is None or plant_feed_table is None:
            raise ValueError("mined_series and plant_feed_table required")
        
        mined_series = TonnageGradeSeries()
        for record_dict in mined_series_dict.get("records", []):
            record = TonnageGradeRecord(**record_dict)
            mined_series.add_record(record)
        
        if isinstance(plant_feed_table, dict):
            plant_feed_table = pd.DataFrame(plant_feed_table)
        if recovery_table is not None and isinstance(recovery_table, dict):
            recovery_table = pd.DataFrame(recovery_table)
        
        series_dict = build_mine_mill_series(mined_series, plant_feed_table, recovery_table)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "recon_mine_mill", "series": series_dict}

    def _prepare_recon_metrics_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Recon Metrics payload."""
        from ..reconciliation.recon_metrics import compute_reconciliation_metrics
        from ..reconciliation.tonnage_grade_balance import TonnageGradeSeries, TonnageGradeRecord
        
        if progress_callback:
            progress_callback(10, "Computing reconciliation metrics...")
        
        model_series_dict = params.get("model_series")
        gc_series_dict = params.get("gc_series")
        mined_series_dict = params.get("mined_series")
        plant_series_dict = params.get("plant_series")
        
        def dict_to_series(d):
            if d is None:
                return None
            series = TonnageGradeSeries()
            for record_dict in d.get("records", []):
                series.add_record(TonnageGradeRecord(**record_dict))
            return series
        
        model_series = dict_to_series(model_series_dict)
        gc_series = dict_to_series(gc_series_dict)
        mined_series = dict_to_series(mined_series_dict)
        plant_series = dict_to_series(plant_series_dict)
        
        metrics = compute_reconciliation_metrics(model_series, gc_series, mined_series, plant_series)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "recon_metrics", "metrics": metrics}

    # =========================================================================
    # Strategic Scheduling
    # =========================================================================
    
    def _prepare_strategic_milp_schedule_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Strategic MILP Schedule payload."""
        from ..mine_planning.scheduling.strategic.strategic_milp import build_strategic_milp_model, solve_strategic_schedule, StrategicScheduleConfig
        
        if progress_callback:
            progress_callback(10, "Building strategic MILP model...")
        
        block_model = params.get("block_model")
        config_dict = params.get("config")
        
        if block_model is None or config_dict is None:
            raise ValueError("block_model and config required")
        
        if "periods" in config_dict and config_dict["periods"]:
            from ..mine_planning.scheduling.types import TimePeriod
            periods = []
            for p_dict in config_dict["periods"]:
                if isinstance(p_dict, dict):
                    periods.append(TimePeriod(**p_dict))
                else:
                    periods.append(p_dict)
            config_dict["periods"] = periods
        
        config = StrategicScheduleConfig(**config_dict)
        model = build_strategic_milp_model(block_model, config)
        
        if progress_callback:
            progress_callback(50, "Solving strategic schedule...")
        
        result = solve_strategic_schedule(model, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "strategic_milp_schedule", "result": result}

    def _prepare_nested_shell_schedule_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Nested Shell Schedule payload."""
        from ..mine_planning.scheduling.strategic.nested_shell_scheduler import allocate_shells_to_periods, NestedShellScheduleConfig
        
        if progress_callback:
            progress_callback(10, "Allocating shells to periods...")
        
        shell_tonnage = params.get("shell_tonnage", {})
        config_dict = params.get("config")
        
        if config_dict is None:
            raise ValueError("config required")
        
        # Remove 'periods' key if present - NestedShellScheduleConfig doesn't accept it
        # (periods are created internally based on target_years)
        config_dict_cleaned = {k: v for k, v in config_dict.items() if k != 'periods'}
        
        config = NestedShellScheduleConfig(**config_dict_cleaned)
        result = allocate_shells_to_periods(shell_tonnage, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "nested_shell_schedule", "result": result}

    def _prepare_cutoff_schedule_opt_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Cutoff Schedule Optimization payload.
        
        ⚠️ UPDATED: Now uses cutoff_engine.py instead of cutoff_scheduler.py.
        The cutoff_engine supports pattern-based optimization (flat, ramp up/down)
        which is more robust than period-by-period independent optimization.
        """
        from ..mine_planning.cutoff.cutoff_engine import optimise_cutoff_schedule, CutoffOptimiserConfig
        from ..mine_planning.npvs.npvs_solver import run_npvs
        
        if progress_callback:
            progress_callback(10, "Optimizing cutoff schedule...")
        
        block_model = params.get("block_model")
        config_dict = params.get("config")
        
        if block_model is None or config_dict is None:
            raise ValueError("block_model and config required")
        
        # Convert periods to list of period IDs (strings)
        periods = []
        if "periods" in config_dict and config_dict["periods"]:
            from ..mine_planning.scheduling.types import TimePeriod
            for p_dict in config_dict["periods"]:
                if isinstance(p_dict, dict):
                    period = TimePeriod(**p_dict)
                    periods.append(period.id)
                elif hasattr(p_dict, 'id'):
                    periods.append(p_dict.id)
                else:
                    periods.append(str(p_dict))
        
        # Build CutoffOptimiserConfig (different API than CutoffScheduleConfig)
        config = CutoffOptimiserConfig(
            periods=periods,
            candidate_cutoffs=config_dict.get("candidate_cutoffs", []),
            pattern_type=config_dict.get("pattern_type", "flat"),
            max_patterns=config_dict.get("max_patterns"),
            price_by_element=config_dict.get("prices", {}),
            recovery_by_element=config_dict.get("recovery_by_element", {}),
            mining_cost_per_t=config_dict.get("costs", {}).get("mining_cost_per_t", 0.0),
            processing_cost_per_t=config_dict.get("costs", {}).get("processing_cost_per_t", 0.0),
            element_name=config_dict.get("element_name", "Fe")
        )
        
        # NPVS runner function
        def npvs_runner(payload: Dict[str, Any]) -> Dict[str, Any]:
            return run_npvs(payload)
        
        result = optimise_cutoff_schedule(block_model, config, npvs_runner)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "cutoff_schedule_opt", "result": result}

    # =========================================================================
    # Tactical Scheduling
    # =========================================================================
    
    def _prepare_tactical_pushback_schedule_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Tactical Pushback Schedule payload."""
        from ..mine_planning.scheduling.tactical.pushback_scheduler import derive_pushback_schedule, TacticalScheduleConfig
        
        if progress_callback:
            progress_callback(10, "Deriving pushback schedule...")
        
        pit_phases = params.get("pit_phases")
        strategic_schedule = params.get("strategic_schedule")
        config_dict = params.get("config")
        
        if strategic_schedule is None or config_dict is None:
            raise ValueError("strategic_schedule and config required")
        
        config = TacticalScheduleConfig(**config_dict)
        config.strategic_schedule = strategic_schedule
        result = derive_pushback_schedule(pit_phases, strategic_schedule, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "tactical_pushback_schedule", "result": result}

    def _prepare_tactical_bench_schedule_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Tactical Bench Schedule payload."""
        from ..mine_planning.scheduling.tactical.bench_stope_progression import build_bench_schedule_from_pushbacks
        
        if progress_callback:
            progress_callback(10, "Building bench schedule...")
        
        block_model = params.get("block_model")
        pushback_schedule = params.get("pushback_schedule")
        bench_height = params.get("bench_height", 15.0)
        
        if block_model is None or pushback_schedule is None:
            raise ValueError("block_model and pushback_schedule required")
        
        result = build_bench_schedule_from_pushbacks(block_model, pushback_schedule, bench_height)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "tactical_bench_schedule", "result": result}

    def _prepare_tactical_dev_schedule_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Tactical Development Schedule payload."""
        from ..mine_planning.scheduling.tactical.development_scheduler import schedule_development, DevelopmentTask, DevelopmentScheduleConfig
        
        if progress_callback:
            progress_callback(10, "Scheduling development...")
        
        tasks_dict = params.get("tasks", [])
        config_dict = params.get("config")
        
        if config_dict is None:
            raise ValueError("config required")
        
        tasks = [DevelopmentTask(**task) for task in tasks_dict]
        
        if "periods" in config_dict and config_dict["periods"]:
            from ..mine_planning.scheduling.types import TimePeriod
            periods = []
            for p_dict in config_dict["periods"]:
                if isinstance(p_dict, dict):
                    periods.append(TimePeriod(**p_dict))
                else:
                    periods.append(p_dict)
            config_dict["periods"] = periods
        
        config = DevelopmentScheduleConfig(**config_dict)
        result = schedule_development(tasks, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "tactical_dev_schedule", "result": result}

    # =========================================================================
    # Short-Term Scheduling
    # =========================================================================
    
    def _prepare_short_term_digline_schedule_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Short-Term Digline Schedule payload."""
        from ..mine_planning.scheduling.short_term.digline_scheduler import build_short_term_schedule, ShortTermScheduleConfig
        
        if progress_callback:
            progress_callback(10, "Building short-term digline schedule...")
        
        gc_model = params.get("gc_model")
        diglines = params.get("diglines")
        config_dict = params.get("config")
        
        if gc_model is None or diglines is None or config_dict is None:
            raise ValueError("gc_model, diglines, and config required")
        
        config = ShortTermScheduleConfig(**config_dict)
        config.gc_model_ref = gc_model
        config.diglines_ref = diglines
        result = build_short_term_schedule(gc_model, diglines, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "short_term_digline_schedule", "result": result}

    def _prepare_short_term_blend_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Short-Term Blend payload."""
        from ..mine_planning.scheduling.short_term.short_term_blend import optimise_short_term_blend, ShortTermBlendConfig
        
        if progress_callback:
            progress_callback(10, "Optimizing short-term blend...")
        
        source_tonnage_grade = params.get("source_tonnage_grade", {})
        config_dict = params.get("config")
        
        if config_dict is None:
            raise ValueError("config required")
        
        if "periods" in config_dict and config_dict["periods"]:
            from ..mine_planning.scheduling.types import TimePeriod
            periods = []
            for p_dict in config_dict["periods"]:
                if isinstance(p_dict, dict):
                    periods.append(TimePeriod(**p_dict))
                else:
                    periods.append(p_dict)
            config_dict["periods"] = periods
        
        config = ShortTermBlendConfig(**config_dict)
        result = optimise_short_term_blend(source_tonnage_grade, config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "short_term_blend", "result": result}

    def _prepare_shift_plan_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Shift Plan payload."""
        from ..mine_planning.scheduling.short_term.shift_plan import generate_shift_plan, ShiftConfig
        
        if progress_callback:
            progress_callback(10, "Generating shift plan...")
        
        short_term_schedule = params.get("short_term_schedule")
        fleet = params.get("fleet")
        shift_config_dict = params.get("shift_config", [])
        
        if short_term_schedule is None or fleet is None:
            raise ValueError("short_term_schedule and fleet required")
        
        shift_config = [ShiftConfig(**shift) for shift in shift_config_dict]
        result = generate_shift_plan(short_term_schedule, fleet, shift_config)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "shift_plan", "result": result}

    # =========================================================================
    # Fleet & Haulage
    # =========================================================================
    
    def _prepare_fleet_cycle_time_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Fleet Cycle Time payload."""
        from ..haulage.cycle_time_model import compute_cycle_time, Route
        from ..haulage.fleet_model import Truck
        
        if progress_callback:
            progress_callback(10, "Computing cycle times...")
        
        truck_dict = params.get("truck")
        route_dict = params.get("route")
        parameters = params.get("parameters", {})
        
        if truck_dict is None or route_dict is None:
            raise ValueError("truck and route required")
        
        truck = Truck(**truck_dict) if isinstance(truck_dict, dict) else truck_dict
        route = Route(**route_dict) if isinstance(route_dict, dict) else route_dict
        
        result = compute_cycle_time(truck, route, parameters)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "fleet_cycle_time", "result": result}

    def _prepare_fleet_dispatch_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Fleet Dispatch payload."""
        from ..haulage.dispatch_rules import allocate_trucks_to_routes
        
        if progress_callback:
            progress_callback(10, "Allocating trucks to routes...")
        
        fleet = params.get("fleet")
        routes = params.get("routes", [])
        production_targets = params.get("production_targets", {})
        
        if fleet is None:
            raise ValueError("fleet required")
        
        result = allocate_trucks_to_routes(fleet, routes, production_targets)
        
        if progress_callback:
            progress_callback(100, "Complete")
        
        return {"name": "fleet_dispatch", "result": result}

    def _prepare_haulage_evaluate_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Haulage Evaluate payload."""
        from ..haulage.haulage_evaluator import evaluate_haulage_capacity, HaulageEvalConfig
        
        schedule = params.get("schedule")
        fleet_config = params.get("fleet_config")
        routes = params.get("routes", [])
        period_mapping = params.get("period_mapping")
        
        if schedule is None or fleet_config is None:
            raise ValueError("schedule and fleet_config required")
        
        config = HaulageEvalConfig(
            schedule=schedule, fleet_config=fleet_config, routes=routes, period_mapping=period_mapping
        )
        
        result = evaluate_haulage_capacity(config)
        
        if progress_callback:
            progress_callback(100, "Haulage evaluation complete")
        
        return {"name": "haulage_evaluate", "result": result}

    # =========================================================================
    # Pushback Designer
    # =========================================================================
    
    def _prepare_pushback_build_plan_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Pushback Build Plan payload."""
        from ..mine_planning.pushbacks.pushback_builder import auto_group_shells_by_depth, auto_group_shells_by_value
        from ..mine_planning.pushbacks.pushback_model import ShellPhase
        
        shells_data = params.get("shells", [])
        grouping_mode = params.get("grouping_mode", "by_depth")
        target_pushbacks = params.get("target_pushbacks", 5)
        
        shells = []
        for shell_data in shells_data:
            if isinstance(shell_data, dict):
                shells.append(ShellPhase(**shell_data))
            else:
                shells.append(shell_data)
        
        if grouping_mode == "by_depth":
            plan = auto_group_shells_by_depth(shells, target_pushbacks)
        elif grouping_mode == "by_value":
            plan = auto_group_shells_by_value(shells, target_pushbacks)
        else:
            raise ValueError(f"Unknown grouping mode: {grouping_mode}")
        
        if progress_callback:
            progress_callback(100, "Pushback plan built")
        
        return {"name": "pushback_build_plan", "result": plan}

    # =========================================================================
    # Cutoff Optimiser
    # =========================================================================
    
    def _prepare_cutoff_optimise_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Cutoff Optimise payload."""
        from ..mine_planning.cutoff.cutoff_engine import optimise_cutoff_schedule, CutoffOptimiserConfig
        from ..mine_planning.npvs.npvs_solver import run_npvs
        
        block_model = params.get("block_model")
        if block_model is None:
            raise ValueError("block_model is required")
        
        config = CutoffOptimiserConfig(
            periods=params.get("periods", []),
            candidate_cutoffs=params.get("candidate_cutoffs", []),
            pattern_type=params.get("pattern_type", "flat"),
            max_patterns=params.get("max_patterns"),
            price_by_element=params.get("price_by_element", {}),
            recovery_by_element=params.get("recovery_by_element", {}),
            mining_cost_per_t=params.get("mining_cost_per_t", 0.0),
            processing_cost_per_t=params.get("processing_cost_per_t", 0.0),
            element_name=params.get("element_name", "Fe")
        )
        
        def npvs_runner(payload: Dict[str, Any]) -> Dict[str, Any]:
            return run_npvs(payload)
        
        result = optimise_cutoff_schedule(block_model, config, npvs_runner)
        
        if progress_callback:
            progress_callback(100, "Cutoff optimisation complete")
        
        return {"name": "cutoff_optimise", "result": result}

    # =========================================================================
    # Production Alignment
    # =========================================================================
    
    def _prepare_production_align_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Production Align payload."""
        from ..planning.production_alignment import align_production_data
        
        schedule_result = params.get("schedule_result")
        haulage_eval_result = params.get("haulage_eval_result")
        recon_result = params.get("recon_result")
        
        if schedule_result is None:
            raise ValueError("schedule_result is required")
        
        result = align_production_data(
            schedule_result=schedule_result,
            haulage_eval_result=haulage_eval_result,
            recon_result=recon_result
        )
        
        if progress_callback:
            progress_callback(100, "Production alignment complete")
        
        return {"name": "production_align", "result": result}

    # =========================================================================
    # Underground Planning
    # =========================================================================
    
    def _prepare_ug_slos_generate_stopes_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare UG SLOS Generate Stopes payload."""
        from ..ug.slos.slos_geometry import generate_stopes_from_block_model
        
        block_model = params.get("block_model")
        template = params.get("template")
        level_spacing_m = params.get("level_spacing_m", 30.0)
        strike_panel_length_m = params.get("strike_panel_length_m", 50.0)
        
        if block_model is None or template is None:
            raise ValueError("block_model and template required")
        
        stopes = generate_stopes_from_block_model(
            block_model=block_model,
            template=template,
            level_spacing_m=level_spacing_m,
            strike_panel_length_m=strike_panel_length_m,
            ore_domain_property=params.get("ore_domain_property"),
            min_grade_cutoff=params.get("min_grade_cutoff")
        )
        
        if progress_callback:
            progress_callback(100, "SLOS stope generation complete")
        
        return {"name": "ug_slos_generate_stopes", "result": {"stopes": stopes}}

    def _prepare_ug_slos_optimise_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare UG SLOS Optimise payload."""
        from ..ug.slos.slos_optimizer import optimise_slos_schedule, SlosOptimiserConfig
        
        stopes = params.get("stopes")
        periods = params.get("periods", [])
        
        if stopes is None or not periods:
            raise ValueError("stopes and periods required")
        
        config = SlosOptimiserConfig(
            periods=periods,
            discount_rate=params.get("discount_rate", 0.10),
            target_tonnes_per_period=params.get("target_tonnes_per_period", 100_000.0),
            max_concurrent_stopes=params.get("max_concurrent_stopes", 5),
            development_lag_periods=params.get("development_lag_periods", 2),
            geotech_factors=params.get("geotech_factors", {})
        )
        
        result = optimise_slos_schedule(stopes, config)
        
        if progress_callback:
            progress_callback(100, "SLOS schedule optimization complete")
        
        return {"name": "ug_slos_optimise", "result": result}

    def _prepare_ug_cave_build_footprint_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare UG Cave Build Footprint payload."""
        from ..ug.caving.cave_footprint import build_cave_footprint_from_block_model
        
        block_model = params.get("block_model")
        if block_model is None:
            raise ValueError("block_model is required")
        
        result = build_cave_footprint_from_block_model(
            block_model=block_model,
            footprint_polygon=params.get("footprint_polygon"),
            levels=params.get("levels"),
            cell_size_m=params.get("cell_size_m", 25.0)
        )
        
        if progress_callback:
            progress_callback(100, "Cave footprint build complete")
        
        return {"name": "ug_cave_build_footprint", "result": result}

    def _prepare_ug_cave_simulate_draw_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare UG Cave Simulate Draw payload."""
        from ..ug.caving.cave_schedule import simulate_cave_draw, CaveScheduleConfig, CaveDrawRule
        
        footprint = params.get("footprint")
        periods = params.get("periods", [])
        
        if footprint is None or not periods:
            raise ValueError("footprint and periods required")
        
        rule_dict = params.get("rule", {})
        rule = CaveDrawRule(
            max_draw_rate_tpy=rule_dict.get("max_draw_rate_tpy", 10_000_000.0),
            max_draw_height_m=rule_dict.get("max_draw_height_m", 100.0),
            dilution_entry_height_ratio=rule_dict.get("dilution_entry_height_ratio", 0.7),
            secondary_break_fraction=rule_dict.get("secondary_break_fraction", 0.1)
        )
        
        config = CaveScheduleConfig(
            periods=periods,
            rule=rule,
            target_tonnes_per_period=params.get("target_tonnes_per_period", 100_000.0)
        )
        
        result = simulate_cave_draw(footprint, config)
        
        if progress_callback:
            progress_callback(100, "Cave draw simulation complete")
        
        return {"name": "ug_cave_simulate_draw", "result": result}

    def _prepare_ug_apply_dilution_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare UG Apply Dilution payload."""
        from ..ug.dilution.dilution_engine import apply_dilution, DilutionModel
        
        stope = params.get("stope")
        model_dict = params.get("model", {})
        
        if stope is None:
            raise ValueError("stope is required")
        
        model = DilutionModel(
            overbreak_m=model_dict.get("overbreak_m", 1.0),
            contact_grade=model_dict.get("contact_grade", {}),
            in_stope_recovery=model_dict.get("in_stope_recovery", 0.95)
        )
        
        result = apply_dilution(stope, model)
        
        if progress_callback:
            progress_callback(100, "Dilution calculation complete")
        
        return {"name": "ug_apply_dilution", "result": result}

    # =========================================================================
    # Scenario Management
    # =========================================================================
    
    def _prepare_planning_run_scenario_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Planning Run Scenario payload."""
        from ..planning.scenario_definition import PlanningScenario
        
        scenario_dict = params.get("scenario")
        if scenario_dict is None:
            raise ValueError("scenario required")
        
        if isinstance(scenario_dict, dict):
            scenario = PlanningScenario.from_dict(scenario_dict)
        else:
            scenario = scenario_dict
        
        result = self.scenario_runner.run(scenario, progress_callback)
        
        return {"name": "planning_run_scenario", "result": result}

    def _prepare_planning_compare_scenarios_payload(self, params: Dict[str, Any], progress_callback: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """Prepare Planning Compare Scenarios payload."""
        from ..planning.scenario_comparison import compare_scenarios
        from ..planning.scenario_definition import ScenarioID
        
        scenario_ids = params.get("scenario_ids", [])
        if not scenario_ids:
            raise ValueError("scenario_ids required")
        
        scenarios = []
        for sid_dict in scenario_ids:
            if isinstance(sid_dict, dict):
                sid = ScenarioID(**sid_dict)
            else:
                sid = sid_dict
            
            scenario = self.scenario_store.get(sid.name, sid.version)
            if scenario:
                scenarios.append(scenario)
        
        def irr_loader(ref: str) -> Dict[str, Any]:
            return {}
        
        def schedule_loader(ref: str) -> Dict[str, Any]:
            return {}
        
        def recon_loader(ref: str) -> Dict[str, Any]:
            return {}
        
        def risk_loader(ref: str) -> Dict[str, Any]:
            return {}
        
        comparison = compare_scenarios(
            scenarios,
            irr_results_loader=irr_loader,
            schedule_loader=schedule_loader,
            recon_loader=recon_loader,
            risk_loader=risk_loader
        )
        
        if progress_callback:
            progress_callback(100, "Comparison complete")
        
        return {"name": "planning_compare_scenarios", "result": comparison}

    # =========================================================================
    # Public API Methods (delegated from AppController)
    # =========================================================================
    
    def calculate_resources(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Calculate resources via task system."""
        self._app.run_task("resources", params, callback, progress_callback)
    
    def classify_resources(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Classify resources via task system."""
        self._app.run_task("classify", params, callback, progress_callback)
    
    def run_drillhole_resources(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run drillhole resources via task system - REMOVED (depended on ResourceCalculator which has been deleted)."""
        raise NotImplementedError("Drillhole resources functionality has been removed (depended on ResourceCalculator).")
    
    def run_irr(self, config: Any, callback=None, progress_callback=None) -> None:
        """Run IRR analysis via task system."""
        self._app.run_task("irr", config, callback, progress_callback)
    
    def run_npv(self, config: Any, callback=None, progress_callback=None) -> None:
        """Run NPV analysis via task system."""
        self._app.run_task("npv", config, callback, progress_callback)
    
    def run_npvs(self, payload: Dict[str, Any], callback=None) -> None:
        """Run NPVS optimization via task system."""
        self._app.run_task("npvs_run", payload, callback)
    
    def run_pit_optimisation(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run pit optimization via task system."""
        self._app.run_task("pit_opt", params, callback, progress_callback)
    
    def run_underground_planning(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run underground planning via task system."""
        self._app.run_task("underground", params, callback, progress_callback)
    
    def run_esg(self, params: Dict[str, Any], callback=None, progress_callback=None) -> None:
        """Run ESG analysis via task system."""
        self._app.run_task("esg", params, callback, progress_callback)
    
    # Scheduling methods
    def run_strategic_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run strategic schedule via task system."""
        self._app.run_task("strategic_milp_schedule", config, callback)
    
    def run_nested_shell_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run nested shell schedule via task system."""
        self._app.run_task("nested_shell_schedule", config, callback)
    
    def run_cutoff_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run cutoff schedule via task system."""
        self._app.run_task("cutoff_schedule_opt", config, callback)
    
    def run_tactical_pushback_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run tactical pushback schedule via task system."""
        self._app.run_task("tactical_pushback_schedule", config, callback)
    
    def run_tactical_bench_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run tactical bench schedule via task system."""
        self._app.run_task("tactical_bench_schedule", config, callback)
    
    def run_tactical_dev_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run tactical development schedule via task system."""
        self._app.run_task("tactical_dev_schedule", config, callback)
    
    def run_short_term_digline_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run short-term digline schedule via task system."""
        self._app.run_task("short_term_digline_schedule", config, callback)
    
    def run_short_term_blend(self, config: Dict[str, Any], callback=None) -> None:
        """Run short-term blend via task system."""
        self._app.run_task("short_term_blend", config, callback)
    
    def run_shift_plan(self, config: Dict[str, Any], callback=None) -> None:
        """Run shift plan via task system."""
        self._app.run_task("shift_plan", config, callback)
    
    # Fleet & Haulage methods
    def compute_fleet_cycle_time(self, config: Dict[str, Any], callback=None) -> None:
        """Compute fleet cycle time via task system."""
        self._app.run_task("fleet_cycle_time", config, callback)
    
    def run_fleet_dispatch(self, config: Dict[str, Any], callback=None) -> None:
        """Run fleet dispatch via task system."""
        self._app.run_task("fleet_dispatch", config, callback)
    
    def evaluate_haulage(self, config: Dict[str, Any], callback=None) -> None:
        """Evaluate haulage capacity via task system."""
        self._app.run_task("haulage_evaluate", config, callback)
    
    # Grade Control & Reconciliation methods
    def build_gc_support_model(self, config: Dict[str, Any], callback=None) -> None:
        """Build GC support model via task system."""
        self._app.run_task("gc_build_support_model", config, callback)
    
    def run_gc_ok(self, config: Dict[str, Any], callback=None) -> None:
        """Run GC OK via task system."""
        self._app.run_task("gc_ok", config, callback)
    
    def run_gc_sgsim(self, config: Dict[str, Any], callback=None) -> None:
        """Run GC SGSIM via task system."""
        self._app.run_task("gc_sgsim", config, callback)
    
    def classify_gc_ore_waste(self, config: Dict[str, Any], callback=None) -> None:
        """Classify GC ore/waste via task system."""
        self._app.run_task("gc_classify_ore_waste", config, callback)
    
    def summarise_gc_by_digpolygon(self, config: Dict[str, Any], callback=None) -> None:
        """Summarize GC by dig polygon via task system."""
        self._app.run_task("gc_summarise_digpolys", config, callback)
    
    def run_recon_model_mine(self, config: Dict[str, Any], callback=None) -> None:
        """Run model-mine reconciliation via task system."""
        self._app.run_task("recon_model_mine", config, callback)
    
    def run_recon_mine_mill(self, config: Dict[str, Any], callback=None) -> None:
        """Run mine-mill reconciliation via task system."""
        self._app.run_task("recon_mine_mill", config, callback)
    
    def run_recon_metrics(self, config: Dict[str, Any], callback=None) -> None:
        """Run reconciliation metrics via task system."""
        self._app.run_task("recon_metrics", config, callback)
    
    # Underground methods
    def ug_generate_slos_stopes(self, config: Dict[str, Any], callback=None) -> None:
        """Generate SLOS stopes via task system."""
        self._app.run_task("ug_slos_generate_stopes", config, callback)
    
    def ug_run_slos_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run SLOS schedule optimization via task system."""
        self._app.run_task("ug_slos_optimise", config, callback)
    
    def ug_build_cave_footprint(self, config: Dict[str, Any], callback=None) -> None:
        """Build cave footprint via task system."""
        self._app.run_task("ug_cave_build_footprint", config, callback)
    
    def ug_run_cave_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Run cave draw simulation via task system."""
        self._app.run_task("ug_cave_simulate_draw", config, callback)
    
    def ug_apply_dilution(self, config: Dict[str, Any], callback=None) -> None:
        """Apply dilution model via task system."""
        self._app.run_task("ug_apply_dilution", config, callback)
    
    # Geometallurgy methods
    def assign_geomet_ore_types(self, config: Dict[str, Any], callback=None) -> None:
        """Assign geomet ore types via task system."""
        self._app.run_task("geomet_assign_ore_types", config, callback)
    
    def compute_geomet_block_attributes(self, config: Dict[str, Any], callback=None) -> None:
        """Compute geomet block attributes via task system."""
        self._app.run_task("geomet_compute_block_attrs", config, callback)
    
    def evaluate_geomet_plant_response(self, config: Dict[str, Any], callback=None) -> None:
        """Evaluate geomet plant response via task system."""
        self._app.run_task("geomet_plant_response", config, callback)
    
    def run_geomet_chain(self, config: Dict[str, Any], callback=None) -> None:
        """Run geomet chain computation via task system."""
        self._app.run_task("geomet_compute_values", config, callback)
    
    # Scenario methods
    def create_scenario(self, scenario: Any) -> None:
        """Create a new scenario."""
        from ..planning.scenario_definition import PlanningScenario
        
        if isinstance(scenario, dict):
            scenario = PlanningScenario.from_dict(scenario)
        
        self.scenario_store.save(scenario)
    
    def list_scenarios(self) -> list:
        """List all scenarios."""
        return self.scenario_store.list_scenarios()
    
    def run_scenario(self, name: str, version: str, callback=None) -> None:
        """Run a scenario."""
        scenario = self.scenario_store.get(name, version)
        if scenario is None:
            raise ValueError(f"Scenario {name} v{version} not found")
        
        params = {"scenario": scenario.to_dict()}
        self._app.run_task("planning_run_scenario", params, callback)
    
    def compare_scenarios(self, scenario_ids: list, callback=None) -> None:
        """Compare multiple scenarios."""
        params = {"scenario_ids": [sid.to_dict() if hasattr(sid, 'to_dict') else sid for sid in scenario_ids]}
        self._app.run_task("planning_compare_scenarios", params, callback)
    
    def build_pushback_plan(self, config: Dict[str, Any], callback=None) -> None:
        """Build pushback plan via task system."""
        self._app.run_task("pushback_build_plan", config, callback)
    
    def optimise_cutoff_schedule(self, config: Dict[str, Any], callback=None) -> None:
        """Optimise cutoff schedule via task system."""
        self._app.run_task("cutoff_optimise", config, callback)
    
    def align_production(self, config: Dict[str, Any], callback=None) -> None:
        """Align production data via task system."""
        self._app.run_task("production_align", config, callback)

