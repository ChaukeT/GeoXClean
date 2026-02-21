"""
Data Provenance & Lineage Tracking
==================================

This module provides comprehensive data provenance tracking for the geostatistical
workflow. It ensures users ALWAYS know:

1. WHERE their data came from (source file, source panel)
2. WHAT transformations have been applied (compositing, declustering, etc.)
3. WHEN each transformation occurred
4. WHO/WHAT panel performed each transformation

This eliminates the "black box" problem where users must blindly trust
that the software is using the correct data.

Example lineage chain:
    Raw Drillholes (drillholes.csv)
        → Composited (2m bench, CompositingWindow)
        → Declustered (50x50x10m cells, DeclusteringPanel)
        → Used in Variogram Analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import json

logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """
    Enumeration of data source types in the geostatistical workflow.
    
    This provides clear categorization so users know exactly what
    type of data they're working with.
    """
    # Raw/Original Data
    RAW_DRILLHOLES = auto()
    RAW_ASSAYS = auto()
    RAW_LITHOLOGY = auto()
    RAW_SURVEYS = auto()
    
    # Transformed Data
    COMPOSITED = auto()
    DECLUSTERED = auto()
    GRADE_TRANSFORMED = auto()  # Normal score transform, etc.
    
    # Estimated/Derived Data
    VARIOGRAM_MODEL = auto()
    KRIGING_ESTIMATE = auto()
    SIMPLE_KRIGING_ESTIMATE = auto()
    COKRIGING_ESTIMATE = auto()
    INDICATOR_KRIGING_ESTIMATE = auto()
    UNIVERSAL_KRIGING_ESTIMATE = auto()
    SOFT_KRIGING_ESTIMATE = auto()
    SGSIM_SIMULATION = auto()
    
    # Block Models
    IMPORTED_BLOCK_MODEL = auto()
    GENERATED_BLOCK_MODEL = auto()
    CLASSIFIED_BLOCK_MODEL = auto()
    
    # Planning Data
    RESOURCE_SUMMARY = auto()
    PIT_OPTIMIZATION = auto()
    SCHEDULE = auto()
    
    # Unknown/Other
    UNKNOWN = auto()
    
    def get_display_name(self) -> str:
        """Get human-readable display name for this data type."""
        display_names = {
            DataSourceType.RAW_DRILLHOLES: "Raw Drillholes",
            DataSourceType.RAW_ASSAYS: "Raw Assays",
            DataSourceType.RAW_LITHOLOGY: "Raw Lithology",
            DataSourceType.RAW_SURVEYS: "Raw Surveys",
            DataSourceType.COMPOSITED: "Composited Data",
            DataSourceType.DECLUSTERED: "Declustered Data",
            DataSourceType.GRADE_TRANSFORMED: "Grade Transformed",
            DataSourceType.VARIOGRAM_MODEL: "Variogram Model",
            DataSourceType.KRIGING_ESTIMATE: "Ordinary Kriging Estimate",
            DataSourceType.SIMPLE_KRIGING_ESTIMATE: "Simple Kriging Estimate",
            DataSourceType.COKRIGING_ESTIMATE: "Co-Kriging Estimate",
            DataSourceType.INDICATOR_KRIGING_ESTIMATE: "Indicator Kriging Estimate",
            DataSourceType.UNIVERSAL_KRIGING_ESTIMATE: "Universal Kriging Estimate",
            DataSourceType.SOFT_KRIGING_ESTIMATE: "Soft Kriging Estimate",
            DataSourceType.SGSIM_SIMULATION: "SGSIM Simulation",
            DataSourceType.IMPORTED_BLOCK_MODEL: "Imported Block Model",
            DataSourceType.GENERATED_BLOCK_MODEL: "Generated Block Model",
            DataSourceType.CLASSIFIED_BLOCK_MODEL: "Classified Block Model",
            DataSourceType.RESOURCE_SUMMARY: "Resource Summary",
            DataSourceType.PIT_OPTIMIZATION: "Pit Optimization",
            DataSourceType.SCHEDULE: "Schedule",
            DataSourceType.UNKNOWN: "Unknown",
        }
        return display_names.get(self, str(self.name))
    
    def get_icon_name(self) -> str:
        """Get icon name for this data type (for UI display)."""
        icon_map = {
            DataSourceType.RAW_DRILLHOLES: "raw_data",
            DataSourceType.RAW_ASSAYS: "raw_data",
            DataSourceType.COMPOSITED: "composite",
            DataSourceType.DECLUSTERED: "decluster",
            DataSourceType.GRADE_TRANSFORMED: "transform",
            DataSourceType.VARIOGRAM_MODEL: "variogram",
            DataSourceType.KRIGING_ESTIMATE: "kriging",
            DataSourceType.SGSIM_SIMULATION: "simulation",
            DataSourceType.IMPORTED_BLOCK_MODEL: "block_model",
            DataSourceType.GENERATED_BLOCK_MODEL: "block_model",
            DataSourceType.CLASSIFIED_BLOCK_MODEL: "classification",
        }
        return icon_map.get(self, "data")
    
    def get_color(self) -> str:
        """Get color code for this data type (for visual distinction)."""
        color_map = {
            # Raw data - neutral gray
            DataSourceType.RAW_DRILLHOLES: "#808080",
            DataSourceType.RAW_ASSAYS: "#808080",
            DataSourceType.RAW_LITHOLOGY: "#808080",
            DataSourceType.RAW_SURVEYS: "#808080",
            # Transformed - blue shades
            DataSourceType.COMPOSITED: "#2196F3",
            DataSourceType.DECLUSTERED: "#03A9F4",
            DataSourceType.GRADE_TRANSFORMED: "#00BCD4",
            # Estimation - green shades
            DataSourceType.VARIOGRAM_MODEL: "#8BC34A",
            DataSourceType.KRIGING_ESTIMATE: "#4CAF50",
            DataSourceType.SIMPLE_KRIGING_ESTIMATE: "#66BB6A",
            DataSourceType.COKRIGING_ESTIMATE: "#81C784",
            DataSourceType.INDICATOR_KRIGING_ESTIMATE: "#A5D6A7",
            DataSourceType.UNIVERSAL_KRIGING_ESTIMATE: "#43A047",
            DataSourceType.SOFT_KRIGING_ESTIMATE: "#388E3C",
            DataSourceType.SGSIM_SIMULATION: "#FFC107",
            # Block models - purple shades
            DataSourceType.IMPORTED_BLOCK_MODEL: "#9C27B0",
            DataSourceType.GENERATED_BLOCK_MODEL: "#AB47BC",
            DataSourceType.CLASSIFIED_BLOCK_MODEL: "#7B1FA2",
            # Planning - orange shades
            DataSourceType.RESOURCE_SUMMARY: "#FF9800",
            DataSourceType.PIT_OPTIMIZATION: "#FF5722",
            DataSourceType.SCHEDULE: "#F44336",
        }
        return color_map.get(self, "#666666")


@dataclass
class TransformationStep:
    """
    Represents a single transformation step in the data lineage.
    
    Each step records:
    - What transformation was applied
    - When it was applied
    - Which panel/component performed it
    - Parameters used for the transformation
    """
    transformation_type: str  # e.g., "compositing", "declustering", "kriging"
    source_panel: str  # e.g., "CompositingWindow", "DeclusteringPanel"
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""  # Human-readable description
    row_count_before: Optional[int] = None
    row_count_after: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "transformation_type": self.transformation_type,
            "source_panel": self.source_panel,
            "timestamp": self.timestamp.isoformat(),
            "parameters": self.parameters,
            "description": self.description,
            "row_count_before": self.row_count_before,
            "row_count_after": self.row_count_after,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformationStep":
        """Create from dictionary."""
        return cls(
            transformation_type=data["transformation_type"],
            source_panel=data["source_panel"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            parameters=data.get("parameters", {}),
            description=data.get("description", ""),
            row_count_before=data.get("row_count_before"),
            row_count_after=data.get("row_count_after"),
        )
    
    def get_summary(self) -> str:
        """Get a one-line summary of this transformation."""
        params_str = ""
        if self.parameters:
            key_params = []
            # Show most important parameters
            for key in ["length", "cell_size", "model_type", "variable"]:
                if key in self.parameters:
                    key_params.append(f"{key}={self.parameters[key]}")
            if key_params:
                params_str = f" ({', '.join(key_params[:3])})"
        
        return f"{self.transformation_type}{params_str}"


@dataclass
class DataProvenance:
    """
    Complete provenance information for a dataset.
    
    This is the core class that tracks the full lineage of any data
    in the system, ensuring users always know:
    
    1. Original source (file, panel, etc.)
    2. All transformations applied
    3. Current state of the data
    
    Example:
        provenance = DataProvenance(
            source_type=DataSourceType.COMPOSITED,
            source_file="drillholes.csv",
            source_panel="CompositingWindow",
            transformation_chain=[
                TransformationStep(
                    transformation_type="compositing",
                    source_panel="CompositingWindow",
                    timestamp=datetime.now(),
                    parameters={"length": 2.0, "method": "length_weighted"},
                    description="2m bench composites, length-weighted"
                )
            ]
        )
    """
    # Core identification
    source_type: DataSourceType
    source_file: Optional[str] = None
    source_panel: str = "Unknown"
    created_at: datetime = field(default_factory=datetime.now)
    
    # Lineage tracking
    transformation_chain: List[TransformationStep] = field(default_factory=list)
    parent_provenance_id: Optional[str] = None  # Links to parent data
    
    # Data statistics
    row_count: Optional[int] = None
    column_names: Optional[List[str]] = None
    
    # User-facing metadata
    user_notes: str = ""
    quality_flags: List[str] = field(default_factory=list)  # e.g., ["validated", "QC_passed"]
    
    # Unique identifier for this provenance record
    provenance_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    
    def add_transformation(
        self,
        transformation_type: str,
        source_panel: str,
        parameters: Optional[Dict[str, Any]] = None,
        description: str = "",
        row_count_before: Optional[int] = None,
        row_count_after: Optional[int] = None,
    ) -> None:
        """Add a transformation step to the lineage chain."""
        step = TransformationStep(
            transformation_type=transformation_type,
            source_panel=source_panel,
            timestamp=datetime.now(),
            parameters=parameters or {},
            description=description,
            row_count_before=row_count_before,
            row_count_after=row_count_after,
        )
        self.transformation_chain.append(step)
        logger.info(f"Added transformation: {step.get_summary()}")
    
    def get_lineage_summary(self) -> str:
        """
        Get a human-readable summary of the data lineage.
        
        Returns something like:
            "Raw Drillholes (drillholes.csv) → Composited (2m) → Declustered (50x50x10m)"
        """
        parts = []
        
        # Start with source
        source_name = self.source_type.get_display_name()
        if self.source_file:
            source_name += f" ({self.source_file})"
        parts.append(source_name)
        
        # Add each transformation
        for step in self.transformation_chain:
            parts.append(step.get_summary())
        
        return " → ".join(parts)
    
    def get_current_type(self) -> DataSourceType:
        """
        Get the current data type after all transformations.
        
        If data started as RAW_ASSAYS and was composited, 
        this returns COMPOSITED (not RAW_ASSAYS).
        """
        if not self.transformation_chain:
            return self.source_type
        
        # Map transformation types to resulting data types
        last_transform = self.transformation_chain[-1].transformation_type.lower()
        transform_to_type = {
            "compositing": DataSourceType.COMPOSITED,
            "composite": DataSourceType.COMPOSITED,
            "declustering": DataSourceType.DECLUSTERED,
            "decluster": DataSourceType.DECLUSTERED,
            "grade_transformation": DataSourceType.GRADE_TRANSFORMED,
            "normal_score": DataSourceType.GRADE_TRANSFORMED,
            "variogram": DataSourceType.VARIOGRAM_MODEL,
            "kriging": DataSourceType.KRIGING_ESTIMATE,
            "ordinary_kriging": DataSourceType.KRIGING_ESTIMATE,
            "simple_kriging": DataSourceType.SIMPLE_KRIGING_ESTIMATE,
            "cokriging": DataSourceType.COKRIGING_ESTIMATE,
            "indicator_kriging": DataSourceType.INDICATOR_KRIGING_ESTIMATE,
            "universal_kriging": DataSourceType.UNIVERSAL_KRIGING_ESTIMATE,
            "soft_kriging": DataSourceType.SOFT_KRIGING_ESTIMATE,
            "sgsim": DataSourceType.SGSIM_SIMULATION,
            "simulation": DataSourceType.SGSIM_SIMULATION,
            "classification": DataSourceType.CLASSIFIED_BLOCK_MODEL,
        }
        
        return transform_to_type.get(last_transform, self.source_type)
    
    def is_derived_from(self, source_type: DataSourceType) -> bool:
        """Check if this data was derived from a specific source type."""
        if self.source_type == source_type:
            return True
        # Check transformation chain
        for step in self.transformation_chain:
            if step.transformation_type.lower() in source_type.name.lower():
                return True
        return False
    
    def has_transformation(self, transformation_type: str) -> bool:
        """Check if a specific transformation has been applied."""
        for step in self.transformation_chain:
            if step.transformation_type.lower() == transformation_type.lower():
                return True
        return False
    
    def get_transformation_params(self, transformation_type: str) -> Optional[Dict[str, Any]]:
        """Get parameters for a specific transformation (if applied)."""
        for step in self.transformation_chain:
            if step.transformation_type.lower() == transformation_type.lower():
                return step.parameters
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provenance_id": self.provenance_id,
            "source_type": self.source_type.name,
            "source_file": self.source_file,
            "source_panel": self.source_panel,
            "created_at": self.created_at.isoformat(),
            "transformation_chain": [step.to_dict() for step in self.transformation_chain],
            "parent_provenance_id": self.parent_provenance_id,
            "row_count": self.row_count,
            "column_names": self.column_names,
            "user_notes": self.user_notes,
            "quality_flags": self.quality_flags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataProvenance":
        """Create from dictionary."""
        return cls(
            provenance_id=data.get("provenance_id", ""),
            source_type=DataSourceType[data["source_type"]],
            source_file=data.get("source_file"),
            source_panel=data.get("source_panel", "Unknown"),
            created_at=datetime.fromisoformat(data["created_at"]),
            transformation_chain=[
                TransformationStep.from_dict(s) 
                for s in data.get("transformation_chain", [])
            ],
            parent_provenance_id=data.get("parent_provenance_id"),
            row_count=data.get("row_count"),
            column_names=data.get("column_names"),
            user_notes=data.get("user_notes", ""),
            quality_flags=data.get("quality_flags", []),
        )
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "DataProvenance":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def clone(self) -> "DataProvenance":
        """Create a deep copy of this provenance."""
        return DataProvenance.from_dict(self.to_dict())


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_raw_data_provenance(
    source_file: str,
    source_panel: str = "DrillholeImportPanel",
    row_count: Optional[int] = None,
    column_names: Optional[List[str]] = None,
) -> DataProvenance:
    """Create provenance for raw imported data."""
    return DataProvenance(
        source_type=DataSourceType.RAW_DRILLHOLES,
        source_file=source_file,
        source_panel=source_panel,
        row_count=row_count,
        column_names=column_names,
    )


def create_composited_provenance(
    parent_provenance: DataProvenance,
    composite_length: float,
    composite_method: str = "length_weighted",
    source_panel: str = "CompositingWindow",
    row_count_after: Optional[int] = None,
    additional_params: Optional[Dict[str, Any]] = None,
) -> DataProvenance:
    """Create provenance for composited data."""
    new_prov = parent_provenance.clone()
    new_prov.parent_provenance_id = parent_provenance.provenance_id
    
    params = {
        "length": composite_length,
        "method": composite_method,
    }
    if additional_params:
        params.update(additional_params)
    
    new_prov.add_transformation(
        transformation_type="compositing",
        source_panel=source_panel,
        parameters=params,
        description=f"{composite_length}m composites ({composite_method})",
        row_count_before=parent_provenance.row_count,
        row_count_after=row_count_after,
    )
    
    new_prov.source_type = DataSourceType.COMPOSITED
    if row_count_after:
        new_prov.row_count = row_count_after
    
    return new_prov


def create_declustered_provenance(
    parent_provenance: DataProvenance,
    cell_size: Tuple[float, float, float],
    source_panel: str = "DeclusteringPanel",
    row_count_after: Optional[int] = None,
    additional_params: Optional[Dict[str, Any]] = None,
) -> DataProvenance:
    """Create provenance for declustered data."""
    new_prov = parent_provenance.clone()
    new_prov.parent_provenance_id = parent_provenance.provenance_id
    
    params = {
        "cell_size_x": cell_size[0],
        "cell_size_y": cell_size[1],
        "cell_size_z": cell_size[2],
        "cell_size": f"{cell_size[0]}x{cell_size[1]}x{cell_size[2]}m",
    }
    if additional_params:
        params.update(additional_params)
    
    new_prov.add_transformation(
        transformation_type="declustering",
        source_panel=source_panel,
        parameters=params,
        description=f"Declustered ({cell_size[0]}x{cell_size[1]}x{cell_size[2]}m cells)",
        row_count_before=parent_provenance.row_count,
        row_count_after=row_count_after,
    )
    
    new_prov.source_type = DataSourceType.DECLUSTERED
    if row_count_after:
        new_prov.row_count = row_count_after
    
    return new_prov


def create_estimation_provenance(
    parent_provenance: DataProvenance,
    estimation_type: str,  # "kriging", "simple_kriging", "sgsim", etc.
    source_panel: str,
    estimated_variable: str,
    variogram_params: Optional[Dict[str, Any]] = None,
    search_params: Optional[Dict[str, Any]] = None,
    block_count: Optional[int] = None,
) -> DataProvenance:
    """Create provenance for estimation results."""
    new_prov = parent_provenance.clone()
    new_prov.parent_provenance_id = parent_provenance.provenance_id
    
    # Map estimation type to DataSourceType
    estimation_type_map = {
        "kriging": DataSourceType.KRIGING_ESTIMATE,
        "ordinary_kriging": DataSourceType.KRIGING_ESTIMATE,
        "simple_kriging": DataSourceType.SIMPLE_KRIGING_ESTIMATE,
        "cokriging": DataSourceType.COKRIGING_ESTIMATE,
        "indicator_kriging": DataSourceType.INDICATOR_KRIGING_ESTIMATE,
        "universal_kriging": DataSourceType.UNIVERSAL_KRIGING_ESTIMATE,
        "soft_kriging": DataSourceType.SOFT_KRIGING_ESTIMATE,
        "sgsim": DataSourceType.SGSIM_SIMULATION,
    }
    
    params = {
        "variable": estimated_variable,
    }
    if variogram_params:
        params["variogram"] = variogram_params
    if search_params:
        params["search"] = search_params
    
    new_prov.add_transformation(
        transformation_type=estimation_type,
        source_panel=source_panel,
        parameters=params,
        description=f"{estimation_type} estimate of {estimated_variable}",
        row_count_after=block_count,
    )
    
    new_prov.source_type = estimation_type_map.get(
        estimation_type.lower(), 
        DataSourceType.KRIGING_ESTIMATE
    )
    if block_count:
        new_prov.row_count = block_count
    
    return new_prov


# =============================================================================
# LINEAGE DISPLAY HELPERS
# =============================================================================

def format_lineage_for_display(provenance: DataProvenance) -> List[Dict[str, Any]]:
    """
    Format provenance for UI display.
    
    Returns a list of dictionaries suitable for display in a tree view
    or list widget, with:
    - step_number: 1, 2, 3...
    - name: Human-readable name
    - color: Color for visual distinction
    - icon: Icon name
    - details: Additional details string
    - timestamp: When this step occurred
    """
    result = []
    
    # Add source as step 0
    result.append({
        "step_number": 0,
        "name": provenance.source_type.get_display_name(),
        "color": provenance.source_type.get_color(),
        "icon": provenance.source_type.get_icon_name(),
        "details": provenance.source_file or "No file",
        "timestamp": provenance.created_at.strftime("%Y-%m-%d %H:%M"),
        "is_current": len(provenance.transformation_chain) == 0,
    })
    
    # Add each transformation
    for i, step in enumerate(provenance.transformation_chain, start=1):
        current_type = DataSourceType.UNKNOWN
        # Try to determine the resulting type
        transform_lower = step.transformation_type.lower()
        if "composite" in transform_lower:
            current_type = DataSourceType.COMPOSITED
        elif "decluster" in transform_lower:
            current_type = DataSourceType.DECLUSTERED
        elif "kriging" in transform_lower:
            if "simple" in transform_lower:
                current_type = DataSourceType.SIMPLE_KRIGING_ESTIMATE
            elif "indicator" in transform_lower:
                current_type = DataSourceType.INDICATOR_KRIGING_ESTIMATE
            elif "universal" in transform_lower:
                current_type = DataSourceType.UNIVERSAL_KRIGING_ESTIMATE
            elif "co" in transform_lower:
                current_type = DataSourceType.COKRIGING_ESTIMATE
            else:
                current_type = DataSourceType.KRIGING_ESTIMATE
        elif "sgsim" in transform_lower or "simulation" in transform_lower:
            current_type = DataSourceType.SGSIM_SIMULATION
        
        result.append({
            "step_number": i,
            "name": step.transformation_type.replace("_", " ").title(),
            "color": current_type.get_color(),
            "icon": current_type.get_icon_name(),
            "details": step.description or step.get_summary(),
            "timestamp": step.timestamp.strftime("%Y-%m-%d %H:%M"),
            "parameters": step.parameters,
            "row_count_before": step.row_count_before,
            "row_count_after": step.row_count_after,
            "is_current": i == len(provenance.transformation_chain),
        })
    
    return result


def get_available_data_sources(registry) -> List[Dict[str, Any]]:
    """
    Get all available data sources from the registry.
    
    Returns a list of data sources with their provenance,
    suitable for populating a DataSourceSelector widget.
    """
    sources = []
    
    # Check drillhole data
    if registry.has_drillhole_data():
        data = registry.get_drillhole_data(copy_data=False)
        if data:
            # Check for raw assays
            if "assays" in data:
                df = data["assays"]
                sources.append({
                    "key": "raw_assays",
                    "name": "Raw Assays",
                    "type": DataSourceType.RAW_ASSAYS,
                    "row_count": len(df) if hasattr(df, "__len__") else 0,
                    "provenance": data.get("_provenance_assays"),
                })
            
            # Check for composites
            if "composites" in data:
                df = data["composites"]
                sources.append({
                    "key": "composites",
                    "name": "Composited Data",
                    "type": DataSourceType.COMPOSITED,
                    "row_count": len(df) if hasattr(df, "__len__") else 0,
                    "provenance": data.get("_provenance_composites"),
                })
    
    # Check declustering results
    if registry.has_data("declustering_results"):
        declust = registry.get_data("declustering_results", copy_data=False)
        if declust and "weighted_dataframe" in declust:
            df = declust["weighted_dataframe"]
            sources.append({
                "key": "declustered",
                "name": "Declustered Data",
                "type": DataSourceType.DECLUSTERED,
                "row_count": len(df) if hasattr(df, "__len__") else 0,
                "provenance": declust.get("_provenance"),
            })
    
    # Check variogram results
    if registry.has_data("variogram_results"):
        sources.append({
            "key": "variogram",
            "name": "Variogram Model",
            "type": DataSourceType.VARIOGRAM_MODEL,
            "provenance": None,
        })
    
    # Check kriging results
    if registry.has_data("kriging_results"):
        krig = registry.get_data("kriging_results", copy_data=False)
        block_count = 0
        if krig and "estimated_values" in krig:
            block_count = len(krig["estimated_values"])
        sources.append({
            "key": "kriging",
            "name": "Kriging Estimate",
            "type": DataSourceType.KRIGING_ESTIMATE,
            "row_count": block_count,
            "provenance": krig.get("_provenance") if krig else None,
        })
    
    return sources

