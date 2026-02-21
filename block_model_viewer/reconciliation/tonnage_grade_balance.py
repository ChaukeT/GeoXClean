"""
Tonnage-Grade Balance (STEP 29)

Core tonnage/grade balancing operations for reconciliation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class TonnageGradeRecord:
    """
    Single tonnage-grade record for reconciliation.
    
    Attributes:
        source: Source identifier ("long_model", "gc_model", "mined", "plant_feed", etc.)
        period_id: Period identifier (e.g., "2024-Q1", "Bench_123", "Blast_456")
        material_type: Material type ("ore", "waste", "stockpile", etc.)
        tonnes: Tonnes
        grades: Dictionary mapping element name -> grade value
    """
    source: str
    period_id: str
    material_type: str
    tonnes: float
    grades: Dict[str, float] = field(default_factory=dict)
    
    def get_metal_content(self, element: str) -> float:
        """Get metal content for an element."""
        grade = self.grades.get(element, 0.0)
        return self.tonnes * grade / 100.0  # Assuming grade is in %


@dataclass
class TonnageGradeSeries:
    """
    Series of tonnage-grade records.
    
    Attributes:
        records: List of TonnageGradeRecord
    """
    records: List[TonnageGradeRecord] = field(default_factory=list)
    
    def add_record(self, record: TonnageGradeRecord) -> None:
        """Add a record to the series."""
        self.records.append(record)
    
    def get_total_tonnes(self, material_type: Optional[str] = None) -> float:
        """Get total tonnes, optionally filtered by material type."""
        if material_type:
            return sum(r.tonnes for r in self.records if r.material_type == material_type)
        return sum(r.tonnes for r in self.records)
    
    def get_average_grade(self, element: str, material_type: Optional[str] = None) -> float:
        """Get average grade for an element."""
        filtered = self.records
        if material_type:
            filtered = [r for r in self.records if r.material_type == material_type]
        
        if not filtered:
            return 0.0
        
        total_tonnes = sum(r.tonnes for r in filtered)
        if total_tonnes == 0:
            return 0.0
        
        weighted_sum = sum(r.tonnes * r.grades.get(element, 0.0) for r in filtered)
        return weighted_sum / total_tonnes
    
    def get_total_metal(self, element: str, material_type: Optional[str] = None) -> float:
        """Get total metal content for an element."""
        filtered = self.records
        if material_type:
            filtered = [r for r in self.records if r.material_type == material_type]
        
        return sum(r.get_metal_content(element) for r in filtered)


def aggregate_records(
    records: List[TonnageGradeRecord],
    by: List[str]
) -> TonnageGradeSeries:
    """
    Aggregate records by specified fields.
    
    Args:
        records: List of TonnageGradeRecord
        by: List of fields to group by (e.g., ["source", "period_id"])
        
    Returns:
        TonnageGradeSeries with aggregated records
    """
    if not records:
        return TonnageGradeSeries()
    
    # Group records by specified fields
    groups: Dict[tuple, List[TonnageGradeRecord]] = {}
    
    for record in records:
        key_parts = []
        for field in by:
            if field == "source":
                key_parts.append(record.source)
            elif field == "period_id":
                key_parts.append(record.period_id)
            elif field == "material_type":
                key_parts.append(record.material_type)
            else:
                key_parts.append("")
        
        key = tuple(key_parts)
        if key not in groups:
            groups[key] = []
        groups[key].append(record)
    
    # Aggregate each group
    aggregated = TonnageGradeSeries()
    
    for key, group_records in groups.items():
        # Sum tonnes
        total_tonnes = sum(r.tonnes for r in group_records)
        
        # Weighted average grades
        aggregated_grades = {}
        if group_records:
            # Get all element names
            all_elements = set()
            for r in group_records:
                all_elements.update(r.grades.keys())
            
            for element in all_elements:
                weighted_sum = sum(r.tonnes * r.grades.get(element, 0.0) for r in group_records)
                if total_tonnes > 0:
                    aggregated_grades[element] = weighted_sum / total_tonnes
                else:
                    aggregated_grades[element] = 0.0
        
        # Create aggregated record
        source = key[0] if len(key) > 0 else group_records[0].source
        period_id = key[1] if len(key) > 1 else group_records[0].period_id
        material_type = key[2] if len(key) > 2 else group_records[0].material_type
        
        aggregated.add_record(TonnageGradeRecord(
            source=source,
            period_id=period_id,
            material_type=material_type,
            tonnes=total_tonnes,
            grades=aggregated_grades
        ))
    
    return aggregated


def compute_bias(
    reference: TonnageGradeSeries,
    compare: TonnageGradeSeries
) -> Dict[str, Any]:
    """
    Compute bias between reference and comparison series.
    
    Args:
        reference: Reference TonnageGradeSeries
        compare: Comparison TonnageGradeSeries
        
    Returns:
        Dictionary with bias metrics (% bias in tonnes and grades)
    """
    ref_tonnes = reference.get_total_tonnes()
    comp_tonnes = compare.get_total_tonnes()
    
    # Tonnage bias
    if ref_tonnes > 0:
        tonnes_bias_pct = ((comp_tonnes - ref_tonnes) / ref_tonnes) * 100.0
    else:
        tonnes_bias_pct = 0.0 if comp_tonnes == 0 else np.inf
    
    # Grade bias per element
    grade_bias = {}
    
    # Get all elements from both series
    all_elements = set()
    for r in reference.records:
        all_elements.update(r.grades.keys())
    for r in compare.records:
        all_elements.update(r.grades.keys())
    
    for element in all_elements:
        ref_grade = reference.get_average_grade(element)
        comp_grade = compare.get_average_grade(element)
        
        if ref_grade > 0:
            grade_bias[element] = ((comp_grade - ref_grade) / ref_grade) * 100.0
        else:
            grade_bias[element] = 0.0 if comp_grade == 0 else np.inf
    
    # Metal bias
    metal_bias = {}
    for element in all_elements:
        ref_metal = reference.get_total_metal(element)
        comp_metal = compare.get_total_metal(element)
        
        if ref_metal > 0:
            metal_bias[element] = ((comp_metal - ref_metal) / ref_metal) * 100.0
        else:
            metal_bias[element] = 0.0 if comp_metal == 0 else np.inf
    
    return {
        "tonnes_bias_pct": tonnes_bias_pct,
        "grade_bias_pct": grade_bias,
        "metal_bias_pct": metal_bias,
        "reference_tonnes": ref_tonnes,
        "compare_tonnes": comp_tonnes
    }

