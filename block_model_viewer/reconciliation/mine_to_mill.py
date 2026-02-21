"""
Mine → Mill Reconciliation Engine (STEP 29)

Compare mined data to plant feed and recovered metal.
"""

from typing import Dict, Optional, Any
import pandas as pd
import logging

from .tonnage_grade_balance import TonnageGradeSeries, TonnageGradeRecord

logger = logging.getLogger(__name__)


def build_mine_mill_series(
    mined_series: TonnageGradeSeries,
    plant_feed_table: pd.DataFrame,
    recovery_table: Optional[pd.DataFrame] = None
) -> Dict[str, TonnageGradeSeries]:
    """
    Build tonnage-grade series for mine-to-mill reconciliation.
    
    Args:
        mined_series: TonnageGradeSeries from mined data
        plant_feed_table: DataFrame with plant feed data (columns: period_id, tonnes, grade columns)
        recovery_table: Optional DataFrame with recovery data (columns: period_id, element, recovery)
        
    Returns:
        Dictionary with keys: "mined", "plant_feed", "plant_recovered"
    """
    series_dict = {}
    
    # Mined series (already provided)
    series_dict["mined"] = mined_series
    
    # Build plant feed series
    plant_feed_series = TonnageGradeSeries()
    
    if not plant_feed_table.empty:
        for _, row in plant_feed_table.iterrows():
            period_id = str(row.get("period_id", "unknown"))
            tonnes = float(row.get("tonnes", 0.0))
            
            # Extract grade columns (all columns except period_id, tonnes, material_type)
            grades = {}
            for col in plant_feed_table.columns:
                if col not in ["period_id", "tonnes", "material_type"]:
                    try:
                        grades[col] = float(row[col])
                    except (ValueError, TypeError):
                        pass
            
            plant_feed_series.add_record(TonnageGradeRecord(
                source="plant_feed",
                period_id=period_id,
                material_type=row.get("material_type", "ore"),
                tonnes=tonnes,
                grades=grades
            ))
    
    series_dict["plant_feed"] = plant_feed_series
    
    # Build plant recovered series (apply recovery factors)
    plant_recovered_series = TonnageGradeSeries()
    
    # Create recovery lookup
    recovery_lookup = {}
    if recovery_table is not None and not recovery_table.empty:
        for _, row in recovery_table.iterrows():
            period_id = str(row.get("period_id", "all"))
            element = str(row.get("element", ""))
            recovery = float(row.get("recovery", 1.0))
            
            if period_id not in recovery_lookup:
                recovery_lookup[period_id] = {}
            recovery_lookup[period_id][element] = recovery
    
    # Apply recovery to plant feed
    for record in plant_feed_series.records:
        # Get recovery factors for this period
        period_recovery = recovery_lookup.get(record.period_id, recovery_lookup.get("all", {}))
        
        # Calculate recovered tonnes (assuming recovery applies to metal, not tonnes)
        recovered_tonnes = record.tonnes  # Tonnes don't change
        
        # Apply recovery to grades (metal recovery)
        recovered_grades = {}
        for element, grade in record.grades.items():
            recovery_factor = period_recovery.get(element, 1.0)
            # Recovered grade = feed grade * recovery
            recovered_grades[element] = grade * recovery_factor
        
        plant_recovered_series.add_record(TonnageGradeRecord(
            source="plant_recovered",
            period_id=record.period_id,
            material_type=record.material_type,
            tonnes=recovered_tonnes,
            grades=recovered_grades
        ))
    
    series_dict["plant_recovered"] = plant_recovered_series
    
    logger.info(f"Built mine-mill series: {list(series_dict.keys())}")
    return series_dict

