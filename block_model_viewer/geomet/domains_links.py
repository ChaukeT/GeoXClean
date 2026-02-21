"""
Geomet Domain / Ore Type Linking (STEP 28)

Links geological domains, mineralogy, and plant ore types.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class GeometOreType:
    """
    Definition of a geometallurgical ore type.
    
    Attributes:
        code: Unique code identifier (e.g., "OT1", "MASSIVE_FE")
        name: Human-readable name
        geology_domains: List of domain codes from geology.DomainModel
        texture_class: Textural classification (e.g., "massive", "banded", "friable")
        hardness_class: Hardness classification (e.g., "soft", "medium", "hard")
        density: Bulk density (t/m³)
        notes: Additional notes/metadata
    """
    code: str
    name: str
    geology_domains: List[str] = field(default_factory=list)
    texture_class: Optional[str] = None
    hardness_class: Optional[str] = None
    density: Optional[float] = None
    notes: str = ""


@dataclass
class GeometDomainMap:
    """
    Mapping between geological domains and geometallurgical ore types.
    
    Attributes:
        ore_types: Dictionary mapping ore type codes to GeometOreType
        metadata: Additional metadata (e.g., creation date, source)
    """
    ore_types: Dict[str, GeometOreType] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_ore_type(self, ore_type: GeometOreType) -> None:
        """Add an ore type to the mapping."""
        self.ore_types[ore_type.code] = ore_type
    
    def get_ore_type(self, code: str) -> Optional[GeometOreType]:
        """Get ore type by code."""
        return self.ore_types.get(code)
    
    def list_ore_types(self) -> List[str]:
        """Get list of all ore type codes."""
        return sorted(self.ore_types.keys())


def infer_ore_type(
    geology_domain_code: str,
    texture_features: Dict[str, Any],
    hardness_index: Optional[float],
    mapping: GeometDomainMap
) -> Optional[str]:
    """
    Infer ore type code from geology domain, texture, and hardness.
    
    Args:
        geology_domain_code: Code from geology.DomainModel
        texture_features: Dictionary with texture-related features
        hardness_index: Hardness index value (if available)
        mapping: GeometDomainMap containing ore type definitions
        
    Returns:
        Ore type code if match found, None otherwise
    """
    # First, try direct domain match
    for ore_type_code, ore_type in mapping.ore_types.items():
        if geology_domain_code in ore_type.geology_domains:
            # Check texture match if specified
            if ore_type.texture_class:
                texture_class = texture_features.get("texture_class")
                if texture_class and texture_class != ore_type.texture_class:
                    continue
            
            # Check hardness match if specified
            if ore_type.hardness_class and hardness_index is not None:
                hardness_class = _classify_hardness(hardness_index)
                if hardness_class != ore_type.hardness_class:
                    continue
            
            return ore_type_code
    
    return None


def _classify_hardness(hardness_index: float) -> str:
    """
    Classify hardness from index value.
    
    Args:
        hardness_index: Hardness index (e.g., Bond Work Index, A*b value)
        
    Returns:
        Hardness class: "soft", "medium", or "hard"
    """
    # Simple classification - can be refined based on project-specific thresholds
    if hardness_index < 10:
        return "soft"
    elif hardness_index < 15:
        return "medium"
    else:
        return "hard"

