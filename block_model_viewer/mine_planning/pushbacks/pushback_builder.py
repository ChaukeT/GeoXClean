"""
Pushback Builder (STEP 33)

Helper functions for grouping shells into pushbacks.
"""

import logging
from typing import List, Dict
from .pushback_model import ShellPhase, PushbackPlan, compute_pushback_stats

logger = logging.getLogger(__name__)


def auto_group_shells_by_depth(
    shells: List[ShellPhase],
    target_pushbacks: int
) -> PushbackPlan:
    """
    Automatically group shells into pushbacks based on depth/precedence order.
    
    Shells are sorted by precedence depth (shells with fewer predecessors come first),
    then divided into roughly equal groups.
    
    Args:
        shells: List of ShellPhase objects
        target_pushbacks: Target number of pushbacks
    
    Returns:
        PushbackPlan with auto-grouped pushbacks
    """
    if not shells:
        return PushbackPlan()
    
    if target_pushbacks < 1:
        target_pushbacks = 1
    
    # Sort shells by precedence depth (shells with fewer predecessors first)
    def get_precedence_depth(shell: ShellPhase) -> int:
        """Get precedence depth (number of predecessors)."""
        return len(shell.precedence_ids)
    
    sorted_shells = sorted(shells, key=get_precedence_depth)
    
    # Group shells into pushbacks
    shells_per_pushback = len(sorted_shells) / target_pushbacks
    groups: Dict[str, List[str]] = {}
    
    for i in range(target_pushbacks):
        pushback_id = f"PB_{i+1:02d}"
        start_idx = int(i * shells_per_pushback)
        end_idx = int((i + 1) * shells_per_pushback) if i < target_pushbacks - 1 else len(sorted_shells)
        
        shell_ids = [s.id for s in sorted_shells[start_idx:end_idx]]
        groups[pushback_id] = shell_ids
    
    logger.info(f"Auto-grouped {len(shells)} shells into {target_pushbacks} pushbacks by depth")
    
    return compute_pushback_stats(shells, groups)


def auto_group_shells_by_value(
    shells: List[ShellPhase],
    target_pushbacks: int
) -> PushbackPlan:
    """
    Automatically group shells into pushbacks based on value.
    
    Shells are sorted by value per tonne (descending), then divided into groups
    to balance total tonnes per pushback.
    
    Args:
        shells: List of ShellPhase objects
        target_pushbacks: Target number of pushbacks
    
    Returns:
        PushbackPlan with auto-grouped pushbacks
    """
    if not shells:
        return PushbackPlan()
    
    if target_pushbacks < 1:
        target_pushbacks = 1
    
    # Sort shells by value per tonne (descending)
    def get_value_per_tonne(shell: ShellPhase) -> float:
        """Get value per tonne."""
        if shell.tonnes > 0:
            return shell.value / shell.tonnes
        return 0.0
    
    sorted_shells = sorted(shells, key=get_value_per_tonne, reverse=True)
    
    # Calculate target tonnes per pushback
    total_tonnes = sum(s.tonnes for s in shells)
    target_tonnes_per_pushback = total_tonnes / target_pushbacks
    
    # Group shells to balance tonnes per pushback
    groups: Dict[str, List[str]] = {}
    current_pushback_id = f"PB_{1:02d}"
    current_pushback_tonnes = 0.0
    pushback_index = 1
    
    for shell in sorted_shells:
        # Start new pushback if current one is full
        if current_pushback_tonnes >= target_tonnes_per_pushback and pushback_index < target_pushbacks:
            pushback_index += 1
            current_pushback_id = f"PB_{pushback_index:02d}"
            current_pushback_tonnes = 0.0
        
        if current_pushback_id not in groups:
            groups[current_pushback_id] = []
        
        groups[current_pushback_id].append(shell.id)
        current_pushback_tonnes += shell.tonnes
    
    logger.info(f"Auto-grouped {len(shells)} shells into {len(groups)} pushbacks by value")
    
    return compute_pushback_stats(shells, groups)


def reorder_pushbacks(
    plan: PushbackPlan,
    new_order: List[str]
) -> PushbackPlan:
    """
    Reorder pushbacks according to new order list.
    
    Args:
        plan: Existing PushbackPlan
        new_order: List of pushback IDs in desired order
    
    Returns:
        New PushbackPlan with reordered pushbacks
    """
    if not plan.pushbacks:
        return plan
    
    # Create lookup
    pushback_lookup = {pb.id: pb for pb in plan.pushbacks}
    
    # Reorder pushbacks
    reordered = []
    for idx, pushback_id in enumerate(new_order):
        if pushback_id in pushback_lookup:
            pb = pushback_lookup[pushback_id]
            # Update order_index
            pb.order_index = idx
            reordered.append(pb)
    
    # Add any pushbacks not in new_order (shouldn't happen, but handle gracefully)
    for pb in plan.pushbacks:
        if pb.id not in new_order:
            pb.order_index = len(reordered)
            reordered.append(pb)
    
    logger.info(f"Reordered {len(reordered)} pushbacks")
    
    return PushbackPlan(
        pushbacks=reordered,
        metadata={**plan.metadata, "reordered": True}
    )

