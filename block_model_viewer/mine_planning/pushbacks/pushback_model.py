"""
Pushback Data Model (STEP 33)

Data structures for representing shells, phases, and pushbacks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class ShellPhase:
    """
    Represents a shell or phase from pit optimization.
    
    Attributes:
        id: Shell or phase identifier (e.g., "S_30", "Phase_3")
        tonnes: Total tonnes in this shell/phase
        value: Total value in this shell/phase
        precedence_ids: List of shell/phase IDs that must be mined before this one
    """
    id: str
    tonnes: float
    value: float
    precedence_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate shell phase."""
        if self.tonnes < 0:
            raise ValueError("tonnes must be non-negative")
        if not self.id:
            raise ValueError("id must be provided")


@dataclass
class Pushback:
    """
    Represents a pushback grouping multiple shells/phases.
    
    Attributes:
        id: Pushback identifier
        name: Human-readable name
        color: RGB color tuple (0-1 range) for visualization
        shell_ids: List of shell/phase IDs in this pushback
        tonnes: Total tonnes in this pushback
        value: Total value in this pushback
        order_index: Order index (0 = first pushback to mine)
    """
    id: str
    name: str
    color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    shell_ids: List[str] = field(default_factory=list)
    tonnes: float = 0.0
    value: float = 0.0
    order_index: int = 0
    
    def __post_init__(self):
        """Validate pushback."""
        if not self.id:
            raise ValueError("id must be provided")
        if not self.name:
            self.name = self.id
        if len(self.color) != 3:
            raise ValueError("color must be a 3-tuple")
        if not all(0 <= c <= 1 for c in self.color):
            raise ValueError("color values must be in [0, 1] range")


@dataclass
class PushbackPlan:
    """
    Complete pushback plan containing multiple pushbacks.
    
    Attributes:
        pushbacks: List of Pushback objects
        metadata: Additional metadata (source shells, creation method, etc.)
    """
    pushbacks: List[Pushback] = field(default_factory=list)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def get_pushback_by_id(self, pushback_id: str) -> Optional[Pushback]:
        """Get pushback by ID."""
        for pb in self.pushbacks:
            if pb.id == pushback_id:
                return pb
        return None
    
    def get_total_tonnes(self) -> float:
        """Get total tonnes across all pushbacks."""
        return sum(pb.tonnes for pb in self.pushbacks)
    
    def get_total_value(self) -> float:
        """Get total value across all pushbacks."""
        return sum(pb.value for pb in self.pushbacks)
    
    def get_pushback_count(self) -> int:
        """Get number of pushbacks."""
        return len(self.pushbacks)


def compute_pushback_stats(
    shells: List[ShellPhase],
    groups: Dict[str, List[str]]
) -> PushbackPlan:
    """
    Compute pushback statistics from shell groupings.
    
    Args:
        shells: List of ShellPhase objects
        groups: Dictionary mapping pushback_id -> list of shell_ids
    
    Returns:
        PushbackPlan with computed statistics
    """
    # Create shell lookup
    shell_lookup = {s.id: s for s in shells}
    
    pushbacks = []
    order_index = 0
    
    # Generate colors for pushbacks (distinct colors)
    import colorsys
    num_pushbacks = len(groups)
    
    for pushback_id, shell_ids in groups.items():
        # Compute aggregate stats
        total_tonnes = 0.0
        total_value = 0.0
        
        for shell_id in shell_ids:
            if shell_id in shell_lookup:
                shell = shell_lookup[shell_id]
                total_tonnes += shell.tonnes
                total_value += shell.value
        
        # Generate color (hue varies, saturation and value fixed)
        hue = order_index / max(num_pushbacks, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        
        pushback = Pushback(
            id=pushback_id,
            name=f"Pushback {order_index + 1}",
            color=rgb,
            shell_ids=shell_ids,
            tonnes=total_tonnes,
            value=total_value,
            order_index=order_index
        )
        
        pushbacks.append(pushback)
        order_index += 1
    
    metadata = {
        "source_shell_count": len(shells),
        "pushback_count": len(pushbacks),
        "grouping_method": "manual"
    }
    
    return PushbackPlan(pushbacks=pushbacks, metadata=metadata)

