"""
Reconciliation Metrics & KPIs (STEP 29)

Compute standard reconciliation KPIs for each stage.
"""

from typing import Dict, Optional, Any
import numpy as np
import logging

from .tonnage_grade_balance import TonnageGradeSeries, compute_bias

logger = logging.getLogger(__name__)


def compute_reconciliation_metrics(
    model_series: Optional[TonnageGradeSeries],
    gc_series: Optional[TonnageGradeSeries],
    mined_series: TonnageGradeSeries,
    plant_series: Optional[TonnageGradeSeries] = None
) -> Dict[str, Any]:
    """
    Compute reconciliation metrics for all stages.
    
    Args:
        model_series: Optional long-term model series
        gc_series: Optional GC model series
        mined_series: Mined data series
        plant_series: Optional plant feed/recovered series
        
    Returns:
        Dictionary with stage-by-stage metrics and global indices
    """
    metrics = {
        "stages": {},
        "global_indices": {}
    }
    
    # Stage 1: Model → GC
    if model_series and gc_series:
        model_gc_bias = compute_bias(model_series, gc_series)
        metrics["stages"]["model_to_gc"] = {
            "tonnes_bias_pct": model_gc_bias["tonnes_bias_pct"],
            "grade_bias_pct": model_gc_bias["grade_bias_pct"],
            "metal_bias_pct": model_gc_bias["metal_bias_pct"]
        }
    
    # Stage 2: GC → Mined (or Model → Mined if no GC)
    if gc_series:
        gc_mined_bias = compute_bias(gc_series, mined_series)
        metrics["stages"]["gc_to_mined"] = {
            "tonnes_bias_pct": gc_mined_bias["tonnes_bias_pct"],
            "grade_bias_pct": gc_mined_bias["grade_bias_pct"],
            "metal_bias_pct": gc_mined_bias["metal_bias_pct"]
        }
    elif model_series:
        model_mined_bias = compute_bias(model_series, mined_series)
        metrics["stages"]["model_to_mined"] = {
            "tonnes_bias_pct": model_mined_bias["tonnes_bias_pct"],
            "grade_bias_pct": model_mined_bias["grade_bias_pct"],
            "metal_bias_pct": model_mined_bias["metal_bias_pct"]
        }
    
    # Stage 3: Mined → Plant Feed
    if plant_series:
        # Extract plant feed series (assuming it's in plant_series or separate)
        # For now, use plant_series as feed
        mined_plant_bias = compute_bias(mined_series, plant_series)
        metrics["stages"]["mined_to_plant"] = {
            "tonnes_bias_pct": mined_plant_bias["tonnes_bias_pct"],
            "grade_bias_pct": mined_plant_bias["grade_bias_pct"],
            "metal_bias_pct": mined_plant_bias["metal_bias_pct"]
        }
    
    # Compute global indices
    # Variance ratio (simplified)
    if model_series and mined_series:
        model_tonnes = model_series.get_total_tonnes()
        mined_tonnes = mined_series.get_total_tonnes()
        
        if model_tonnes > 0 and mined_tonnes > 0:
            variance_ratio = mined_tonnes / model_tonnes
            metrics["global_indices"]["variance_ratio"] = variance_ratio
    
    # Reconciliation factors (cumulative)
    reconciliation_factors = {}
    
    if model_series:
        model_tonnes = model_series.get_total_tonnes()
        if model_tonnes > 0:
            if gc_series:
                gc_tonnes = gc_series.get_total_tonnes()
                reconciliation_factors["model_to_gc"] = gc_tonnes / model_tonnes
            
            mined_tonnes = mined_series.get_total_tonnes()
            reconciliation_factors["model_to_mined"] = mined_tonnes / model_tonnes
            
            if plant_series:
                plant_tonnes = plant_series.get_total_tonnes()
                reconciliation_factors["model_to_plant"] = plant_tonnes / model_tonnes
    
    metrics["global_indices"]["reconciliation_factors"] = reconciliation_factors
    
    # GC model quality index (slope of regression vs mined)
    if gc_series and mined_series:
        # Match records by period_id and compute regression slope
        gc_periods = {r.period_id: r for r in gc_series.records}
        mined_periods = {r.period_id: r for r in mined_series.records}
        
        common_periods = set(gc_periods.keys()) & set(mined_periods.keys())
        
        if len(common_periods) >= 2:
            gc_values = []
            mined_values = []
            
            for period_id in common_periods:
                gc_record = gc_periods[period_id]
                mined_record = mined_periods[period_id]
                
                # Use primary element (first grade property)
                if gc_record.grades and mined_record.grades:
                    primary_element = list(gc_record.grades.keys())[0]
                    if primary_element in mined_record.grades:
                        gc_values.append(gc_record.grades[primary_element])
                        mined_values.append(mined_record.grades[primary_element])
            
            if len(gc_values) >= 2:
                # Simple linear regression slope
                gc_array = np.array(gc_values)
                mined_array = np.array(mined_values)
                
                # Remove NaN values
                valid_mask = ~(np.isnan(gc_array) | np.isnan(mined_array))
                if np.sum(valid_mask) >= 2:
                    gc_valid = gc_array[valid_mask]
                    mined_valid = mined_array[valid_mask]
                    
                    # Slope = covariance / variance
                    covariance = np.mean((gc_valid - np.mean(gc_valid)) * (mined_valid - np.mean(mined_valid)))
                    variance = np.var(gc_valid)
                    
                    if variance > 0:
                        slope = covariance / variance
                        metrics["global_indices"]["gc_quality_slope"] = slope
                    else:
                        metrics["global_indices"]["gc_quality_slope"] = 1.0
    
    logger.info(f"Computed reconciliation metrics for {len(metrics['stages'])} stages")
    return metrics

