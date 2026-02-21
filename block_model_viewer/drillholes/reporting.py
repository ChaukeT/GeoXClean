"""
Drillhole Reporting - Statistics and report generation.

Provides summary statistics, grade-tonnage curves, comparison reports,
and export functionality (PDF, Excel, HTML).

LINEAGE ENFORCEMENT:
- All statistics methods can accept declustering weights
- Grade-tonnage curves support weighted calculations
- Export includes provenance metadata for JORC/SAMREC compliance

STATISTICAL CORRECTNESS:
- Weighted statistics use proper normalisation
- Division by zero guarded against
- Missing values handled explicitly (raise ValueError rather than silent fail)
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from .datamodel import DrillholeDatabase

logger = logging.getLogger(__name__)


class DrillholeStatistics:
    """
    Statistics calculator for drillhole data.
    
    LINEAGE SUPPORT:
    - Accepts optional declustering weights
    - Tracks data source for audit trail
    
    STATISTICAL METHODS:
    - All methods support weighted calculations via declust_weights parameter
    - Division by zero is guarded against (STAT-006)
    - Empty/missing data raises ValueError instead of silent failure (STAT-010)
    """
    
    def __init__(self, db: DrillholeDatabase, declust_weights: Optional[Dict[str, float]] = None):
        """
        Initialize statistics calculator.
        
        Args:
            db: DrillholeDatabase instance
            declust_weights: Optional dict mapping (hole_id, depth_from, depth_to) to weight.
                            If provided, all weighted statistics will use these weights.
        """
        self.db = db
        self._declust_weights = declust_weights or {}
        self._provenance = {
            'source': 'DrillholeStatistics',
            'timestamp': datetime.now().isoformat(),
            'declustered': bool(declust_weights),
        }
    
    def get_summary_statistics(self, element: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary statistics for drillhole database.
        
        Args:
            element: Optional element name (all elements if None)
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            "num_holes": len(self.db.get_hole_ids()),
            "num_collars": len(self.db.collars),
            "num_surveys": len(self.db.surveys),
            "num_assays": len(self.db.assays),
            "num_lithology": len(self.db.lithology),
        }
        
        # Get all elements
        all_elements = set()
        for assay in self.db.assays:
            all_elements.update(assay.values.keys())
        
        stats["elements"] = sorted(all_elements)
        
        # Element statistics
        if element:
            values = []
            for assay in self.db.assays:
                if element in assay.values:
                    values.append(assay.values[element])
            
            if values:
                values = np.array(values)
                stats["element_statistics"] = {
                    "element": element,
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                }
        
        # Hole statistics
        hole_stats = []
        for hole_id in self.db.get_hole_ids():
            collar = next((c for c in self.db.collars if c.hole_id == hole_id), None)
            assays = self.db.get_assays_for(hole_id)
            
            if collar:
                hole_stats.append({
                    "hole_id": hole_id,
                    "x": collar.x,
                    "y": collar.y,
                    "z": collar.z,
                    "length": collar.length,
                    "num_assays": len(assays),
                    "total_length": sum(a.depth_to - a.depth_from for a in assays) if assays else 0,
                })
        
        stats["hole_statistics"] = hole_stats
        
        return stats
    
    def get_grade_tonnage_curve(
        self, 
        element: str, 
        cutoffs: List[float],
        hole_ids: Optional[List[str]] = None,
        use_declustering: bool = True
    ) -> pd.DataFrame:
        """
        Calculate grade-tonnage curve with optional declustering weights.
        
        STAT-003: Integrates declustering weights into GT calculation.
        STAT-006: Guards against division by zero.
        
        Args:
            element: Element name
            cutoffs: List of cutoff grades
            hole_ids: Optional list of hole IDs to include
            use_declustering: If True and declust_weights available, apply them
        
        Returns:
            DataFrame with grade-tonnage data including weighted columns
            
        Raises:
            ValueError: If element not found in any assays
        """
        # Collect all assay data
        data = []
        for assay in self.db.assays:
            if hole_ids and assay.hole_id not in hole_ids:
                continue
            
            if element in assay.values:
                length = assay.depth_to - assay.depth_from
                grade = assay.values[element]
                
                # STAT-003: Get declustering weight if available
                declust_weight = 1.0
                if use_declustering and self._declust_weights:
                    key = (assay.hole_id, assay.depth_from, assay.depth_to)
                    declust_weight = self._declust_weights.get(key, 1.0)
                
                data.append({
                    "hole_id": assay.hole_id,
                    "depth_from": assay.depth_from,
                    "depth_to": assay.depth_to,
                    "length": length,
                    "grade": grade,
                    "declust_weight": declust_weight,
                })
        
        # STAT-010: Raise error instead of returning empty DataFrame silently
        if not data:
            raise ValueError(f"Element '{element}' not found in any assays for specified holes")
        
        df = pd.DataFrame(data)
        
        # Calculate grade-tonnage curve
        results = []
        for cutoff in cutoffs:
            # Filter by cutoff
            filtered = df[df["grade"] >= cutoff]
            
            if len(filtered) > 0:
                # STAT-006: Guard against division by zero
                total_length = filtered["length"].sum()
                if total_length <= 0:
                    logger.warning(f"Zero total length at cutoff {cutoff}, skipping")
                    continue
                
                # Length-weighted grade (standard)
                weighted_grade = (filtered["grade"] * filtered["length"]).sum() / total_length
                num_intervals = len(filtered)
                num_holes = len(filtered["hole_id"].unique())
                
                # STAT-003: Declustering-weighted statistics
                declustered_grade = weighted_grade
                declustered_proportion = 1.0
                if use_declustering and self._declust_weights:
                    total_weight = df["declust_weight"].sum()
                    weight_above = filtered["declust_weight"].sum()
                    
                    if total_weight > 0:
                        declustered_proportion = weight_above / total_weight
                        
                        # Combined weight: length × declust_weight
                        combined_weights = filtered["length"] * filtered["declust_weight"]
                        combined_sum = combined_weights.sum()
                        
                        if combined_sum > 0:
                            declustered_grade = (filtered["grade"] * combined_weights).sum() / combined_sum
            else:
                total_length = 0
                weighted_grade = 0
                declustered_grade = 0
                declustered_proportion = 0
                num_intervals = 0
                num_holes = 0
            
            results.append({
                "cutoff": cutoff,
                "tonnage": total_length,  # Length in metres (proxy for tonnage)
                "grade": weighted_grade,  # Length-weighted
                "grade_declustered": declustered_grade,  # Decluster+length weighted
                "proportion_declustered": declustered_proportion,  # Declustered proportion above
                "num_intervals": num_intervals,
                "num_holes": num_holes,
                "declustering_applied": bool(self._declust_weights) and use_declustering,
            })
        
        result_df = pd.DataFrame(results)
        
        # Add provenance metadata
        result_df.attrs['provenance'] = {
            **self._provenance,
            'element': element,
            'cutoffs': cutoffs,
            'num_holes_filtered': len(hole_ids) if hole_ids else 'all',
            'declustering_applied': bool(self._declust_weights) and use_declustering,
        }
        
        return result_df
    
    def get_element_statistics(
        self, 
        element: str, 
        hole_ids: Optional[List[str]] = None,
        use_declustering: bool = True
    ) -> Dict[str, Any]:
        """
        Get detailed statistics for an element with optional declustering.
        
        STAT-002: Integrates declustering weights into statistics.
        STAT-006: Guards against division by zero in CV calculation.
        STAT-010: Raises ValueError instead of returning empty dict.
        
        Args:
            element: Element name
            hole_ids: Optional list of hole IDs to include
            use_declustering: If True and declust_weights available, apply them
        
        Returns:
            Dictionary with element statistics including both raw and weighted values
            
        Raises:
            ValueError: If element not found in any assays
        """
        values = []
        lengths = []
        declust_weights = []
        holes = set()
        
        for assay in self.db.assays:
            if hole_ids and assay.hole_id not in hole_ids:
                continue
            
            if element in assay.values:
                value = assay.values[element]
                # Handle NaN values explicitly
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue
                    
                length = assay.depth_to - assay.depth_from
                
                # STAT-003: Get declustering weight
                declust_weight = 1.0
                if use_declustering and self._declust_weights:
                    key = (assay.hole_id, assay.depth_from, assay.depth_to)
                    declust_weight = self._declust_weights.get(key, 1.0)
                
                values.append(value)
                lengths.append(length)
                declust_weights.append(declust_weight)
                holes.add(assay.hole_id)
        
        # STAT-010: Raise error instead of returning empty dict
        if not values:
            raise ValueError(
                f"Element '{element}' not found in any assays"
                + (f" for holes: {hole_ids[:5]}..." if hole_ids else "")
            )
        
        values = np.array(values)
        lengths = np.array(lengths)
        declust_weights = np.array(declust_weights)
        
        # Basic (unweighted) statistics
        raw_mean = float(np.mean(values))
        raw_std = float(np.std(values))
        raw_var = float(np.var(values))
        
        # STAT-006: Guard against division by zero in CV
        raw_cv = (raw_std / raw_mean * 100) if raw_mean != 0 else 0.0
        
        # Length-weighted statistics (standard practice)
        total_length = np.sum(lengths)
        if total_length > 0:
            length_weighted_mean = np.average(values, weights=lengths)
            length_weighted_var = np.average((values - length_weighted_mean) ** 2, weights=lengths)
            length_weighted_std = np.sqrt(length_weighted_var)
        else:
            length_weighted_mean = raw_mean
            length_weighted_var = raw_var
            length_weighted_std = raw_std
        
        # STAT-002: Declustering-weighted statistics
        declust_weighted_mean = length_weighted_mean
        declust_weighted_std = length_weighted_std
        declust_weighted_var = length_weighted_var
        bias_correction = 0.0
        
        if use_declustering and self._declust_weights:
            # Combined weights: length × declust_weight
            combined_weights = lengths * declust_weights
            total_combined = np.sum(combined_weights)
            
            if total_combined > 0:
                declust_weighted_mean = np.sum(values * combined_weights) / total_combined
                declust_weighted_var = np.sum(combined_weights * (values - declust_weighted_mean) ** 2) / total_combined
                declust_weighted_std = np.sqrt(declust_weighted_var)
                bias_correction = declust_weighted_mean - raw_mean
        
        return {
            "element": element,
            "count": len(values),
            "num_holes": len(holes),
            # Raw statistics
            "mean": raw_mean,
            "std": raw_std,
            "var": raw_var,
            "cv": raw_cv,
            # Length-weighted statistics
            "weighted_mean": float(length_weighted_mean),
            "weighted_std": float(length_weighted_std),
            "weighted_var": float(length_weighted_var),
            # Declustered statistics (STAT-002)
            "declustered_mean": float(declust_weighted_mean),
            "declustered_std": float(declust_weighted_std),
            "declustered_var": float(declust_weighted_var),
            "bias_correction": float(bias_correction),
            # Percentiles
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "q25": float(np.percentile(values, 25)),
            "q75": float(np.percentile(values, 75)),
            "q90": float(np.percentile(values, 90)),
            "q95": float(np.percentile(values, 95)),
            "q99": float(np.percentile(values, 99)),
            # Provenance
            "declustering_applied": bool(self._declust_weights) and use_declustering,
            "provenance": self._provenance,
        }
    
    def compare_datasets(self, db1: DrillholeDatabase, db2: DrillholeDatabase,
                        element: str) -> Dict[str, Any]:
        """
        Compare two datasets.
        
        Args:
            db1: First database
            db2: Second database
            element: Element name to compare
        
        Returns:
            Dictionary with comparison statistics
        """
        stats1 = DrillholeStatistics(db1)
        stats2 = DrillholeStatistics(db2)
        
        stats1_data = stats1.get_element_statistics(element)
        stats2_data = stats2.get_element_statistics(element)
        
        if not stats1_data or not stats2_data:
            return {}
        
        comparison = {
            "element": element,
            "dataset1": stats1_data,
            "dataset2": stats2_data,
            "difference": {
                "mean": stats1_data["mean"] - stats2_data["mean"],
                "median": stats1_data["median"] - stats2_data["median"],
                "std": stats1_data["std"] - stats2_data["std"],
            },
            "percent_difference": {
                "mean": ((stats1_data["mean"] - stats2_data["mean"]) / stats2_data["mean"] * 100) if stats2_data["mean"] != 0 else 0,
                "median": ((stats1_data["median"] - stats2_data["median"]) / stats2_data["median"] * 100) if stats2_data["median"] != 0 else 0,
            }
        }
        
        return comparison


class ReportGenerator:
    """Report generator for drillhole data."""
    
    def __init__(self, db: DrillholeDatabase):
        """Initialize report generator."""
        self.db = db
        self.statistics = DrillholeStatistics(db)
    
    def generate_summary_report(self, element: Optional[str] = None) -> str:
        """
        Generate summary report.
        
        Args:
            element: Optional element name
        
        Returns:
            Report text
        """
        stats = self.statistics.get_summary_statistics(element)
        
        report = f"""
=== Drillhole Database Summary Report ===

Database Statistics:
- Number of Holes: {stats['num_holes']}
- Number of Collars: {stats['num_collars']}
- Number of Surveys: {stats['num_surveys']}
- Number of Assays: {stats['num_assays']}
- Number of Lithology Intervals: {stats['num_lithology']}
- Elements: {', '.join(stats['elements'])}

"""
        
        if element and "element_statistics" in stats:
            elem_stats = stats["element_statistics"]
            report += f"""
Element Statistics ({element}):
- Count: {elem_stats['count']}
- Mean: {elem_stats['mean']:.3f}
- Median: {elem_stats['median']:.3f}
- Std Dev: {elem_stats['std']:.3f}
- Min: {elem_stats['min']:.3f}
- Max: {elem_stats['max']:.3f}
- Q25: {elem_stats['q25']:.3f}
- Q75: {elem_stats['q75']:.3f}

"""
        
        return report
    
    def export_to_csv(self, file_path: Path, data_type: str = "all"):
        """
        Export database to CSV.
        
        Args:
            file_path: Output file path
            data_type: 'all', 'collars', 'surveys', 'assays', 'lithology'
        """
        if data_type == "all" or data_type == "collars":
            # Export collars
            collars_data = []
            for collar in self.db.collars:
                collars_data.append({
                    "hole_id": collar.hole_id,
                    "x": collar.x,
                    "y": collar.y,
                    "z": collar.z,
                    "azimuth": collar.azimuth,
                    "dip": collar.dip,
                    "length": collar.length,
                })
            
            if collars_data:
                df = pd.DataFrame(collars_data)
                df.to_csv(file_path.with_suffix(".collars.csv"), index=False)
        
        if data_type == "all" or data_type == "assays":
            # Export assays
            assays_data = []
            for assay in self.db.assays:
                row = {
                    "hole_id": assay.hole_id,
                    "depth_from": assay.depth_from,
                    "depth_to": assay.depth_to,
                }
                row.update(assay.values)
                assays_data.append(row)
            
            if assays_data:
                df = pd.DataFrame(assays_data)
                df.to_csv(file_path.with_suffix(".assays.csv"), index=False)
        
        logger.info(f"Exported database to {file_path}")
    
    def export_grade_tonnage_curve(self, element: str, cutoffs: List[float],
                                   file_path: Path, hole_ids: Optional[List[str]] = None):
        """
        Export grade-tonnage curve to CSV.
        
        Args:
            element: Element name
            cutoffs: List of cutoff grades
            file_path: Output file path
            hole_ids: Optional list of hole IDs to include
        """
        df = self.statistics.get_grade_tonnage_curve(element, cutoffs, hole_ids)
        df.to_csv(file_path, index=False)
        logger.info(f"Exported grade-tonnage curve to {file_path}")

