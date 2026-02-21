"""
Model → Mine Reconciliation Engine (STEP 29)

Compare long-term model & GC model to mined tonnes/grades.
"""

from typing import Dict, Optional, Any, List
import numpy as np
import pandas as pd
import logging

from .tonnage_grade_balance import TonnageGradeSeries, TonnageGradeRecord

logger = logging.getLogger(__name__)


def build_model_mine_series(
    long_model: Any,
    gc_model: Optional[Any],
    diglines: Optional[Any],
    mined_tonnage_table: pd.DataFrame,
    density_property: str,
    grade_properties: List[str]
) -> Dict[str, TonnageGradeSeries]:
    """
    Build tonnage-grade series for model-to-mine reconciliation.
    
    Args:
        long_model: BlockModel (long-term model)
        gc_model: Optional GCModel
        diglines: Optional DiglineSet
        mined_tonnage_table: DataFrame with mined data (columns: period_id, tonnes, grade columns)
        density_property: Name of density property
        grade_properties: List of grade property names
        
    Returns:
        Dictionary with keys: "long_model", "gc_model" (if provided), "mined"
    """
    series_dict = {}
    
    # Build long-term model series
    if long_model and long_model.positions is not None:
        long_series = TonnageGradeSeries()
        
        # Calculate tonnage from long-term model
        if density_property in long_model.get_property_names():
            density_values = long_model.get_property(density_property)
        else:
            density_values = np.full(long_model.block_count, 2.7)
        
        dimensions = long_model.dimensions
        volumes = dimensions[:, 0] * dimensions[:, 1] * dimensions[:, 2]
        tonnes = volumes * density_values
        
        # Aggregate by period (if period property exists)
        period_prop = long_model.get_property("period")
        if period_prop is not None:
            unique_periods = np.unique(period_prop)
            for period_id in unique_periods:
                mask = period_prop == period_id
                period_tonnes = np.sum(tonnes[mask])
                
                grades = {}
                for grade_prop in grade_properties:
                    grade_values = long_model.get_property(grade_prop)
                    if grade_values is not None:
                        period_grades = grade_values[mask]
                        valid_grades = period_grades[~np.isnan(period_grades)]
                        if len(valid_grades) > 0:
                            grades[grade_prop] = np.mean(valid_grades)
                        else:
                            grades[grade_prop] = 0.0
                
                long_series.add_record(TonnageGradeRecord(
                    source="long_model",
                    period_id=str(period_id),
                    material_type="ore",
                    tonnes=period_tonnes,
                    grades=grades
                ))
        else:
            # Single record for entire model
            total_tonnes = np.sum(tonnes)
            grades = {}
            for grade_prop in grade_properties:
                grade_values = long_model.get_property(grade_prop)
                if grade_values is not None:
                    valid_grades = grade_values[~np.isnan(grade_values)]
                    if len(valid_grades) > 0:
                        grades[grade_prop] = np.mean(valid_grades)
                    else:
                        grades[grade_prop] = 0.0
            
            long_series.add_record(TonnageGradeRecord(
                source="long_model",
                period_id="all",
                material_type="ore",
                tonnes=total_tonnes,
                grades=grades
            ))
        
        series_dict["long_model"] = long_series
    
    # Build GC model series
    if gc_model:
        gc_series = TonnageGradeSeries()
        
        # Calculate tonnage from GC model
        density_values = gc_model.get_property(density_property)
        if density_values is None:
            density_values = np.full(gc_model.grid.get_block_count(), 2.7)
        
        block_volume = gc_model.grid.dx * gc_model.grid.dy * gc_model.grid.dz
        volumes = np.full(len(density_values), block_volume)
        tonnes = volumes * density_values
        
        # Aggregate by bench if diglines provided
        if diglines:
            from ..grade_control.digpolygons import blocks_within_polygon
            
            for polygon in diglines.polygons:
                block_mask = blocks_within_polygon(gc_model.grid, polygon)
                if not np.any(block_mask):
                    continue
                
                polygon_tonnes = np.sum(tonnes[block_mask])
                
                grades = {}
                for grade_prop in grade_properties:
                    grade_values = gc_model.get_property(grade_prop)
                    if grade_values is not None:
                        polygon_grades = grade_values[block_mask]
                        valid_grades = polygon_grades[~np.isnan(polygon_grades)]
                        if len(valid_grades) > 0:
                            grades[grade_prop] = np.mean(valid_grades)
                        else:
                            grades[grade_prop] = 0.0
                
                gc_series.add_record(TonnageGradeRecord(
                    source="gc_model",
                    period_id=polygon.id,
                    material_type="ore" if polygon.ore_flag else "waste",
                    tonnes=polygon_tonnes,
                    grades=grades
                ))
        else:
            # Single record for entire GC model
            total_tonnes = np.sum(tonnes)
            grades = {}
            for grade_prop in grade_properties:
                grade_values = gc_model.get_property(grade_prop)
                if grade_values is not None:
                    valid_grades = grade_values[~np.isnan(grade_values)]
                    if len(valid_grades) > 0:
                        grades[grade_prop] = np.mean(valid_grades)
                    else:
                        grades[grade_prop] = 0.0
            
            gc_series.add_record(TonnageGradeRecord(
                source="gc_model",
                period_id="all",
                material_type="ore",
                tonnes=total_tonnes,
                grades=grades
            ))
        
        series_dict["gc_model"] = gc_series
    
    # Build mined series from table
    mined_series = TonnageGradeSeries()
    
    if not mined_tonnage_table.empty:
        for _, row in mined_tonnage_table.iterrows():
            period_id = str(row.get("period_id", "unknown"))
            tonnes = float(row.get("tonnes", 0.0))
            
            grades = {}
            for grade_prop in grade_properties:
                if grade_prop in row:
                    grades[grade_prop] = float(row[grade_prop])
            
            mined_series.add_record(TonnageGradeRecord(
                source="mined",
                period_id=period_id,
                material_type=row.get("material_type", "ore"),
                tonnes=tonnes,
                grades=grades
            ))
    
    series_dict["mined"] = mined_series
    
    logger.info(f"Built reconciliation series: {list(series_dict.keys())}")
    return series_dict

