"""
Ore Type Model (STEP 38)

Ore domain classification and attachment to block model.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OreDomain:
    """
    Ore domain definition.
    
    Attributes:
        id: Domain identifier (e.g., "OX", "TRANS", "FR")
        description: Description
        hardness_index: Hardness index
        density_t_m3: Density in tonnes per cubic meter
        alteration_code: Optional alteration code
    """
    id: str
    description: str = ""
    hardness_index: float = 5.0
    density_t_m3: float = 2.7
    alteration_code: Optional[str] = None


@dataclass
class OreTypeModel:
    """
    Ore type model containing domain definitions.
    
    Attributes:
        domains: List of OreDomain objects
    """
    domains: List[OreDomain] = field(default_factory=list)
    
    def get_domain(self, domain_id: str) -> Optional[OreDomain]:
        """Get domain by ID."""
        for domain in self.domains:
            if domain.id == domain_id:
                return domain
        return None


def infer_ore_domain_for_block(
    block_row: Any,
    config: Dict[str, Any]
) -> str:
    """
    Infer ore domain for a block based on properties.
    
    Args:
        block_row: Block row (dict or Series) with properties
        config: Configuration dict with rules:
            - domain_property: Property name to use directly
            - domain_rules: Dict mapping property values to domain IDs
            - default_domain: Default domain ID
    
    Returns:
        Domain ID string
    """
    # Check if domain property exists directly
    domain_property = config.get("domain_property")
    if domain_property:
        if isinstance(block_row, dict):
            domain_id = block_row.get(domain_property)
        else:
            domain_id = getattr(block_row, domain_property, None)
        
        if domain_id:
            return str(domain_id)
    
    # Check domain rules
    domain_rules = config.get("domain_rules", {})
    for prop_name, domain_map in domain_rules.items():
        if isinstance(block_row, dict):
            prop_value = block_row.get(prop_name)
        else:
            prop_value = getattr(block_row, prop_name, None)
        
        if prop_value is not None:
            # Check if value maps to a domain
            if isinstance(domain_map, dict):
                domain_id = domain_map.get(prop_value)
            elif callable(domain_map):
                domain_id = domain_map(prop_value)
            else:
                domain_id = domain_map
            
            if domain_id:
                return str(domain_id)
    
    # Return default
    default_domain = config.get("default_domain", "UNKNOWN")
    return str(default_domain)


def attach_ore_type_to_block_model(
    block_model: Any,
    ore_type_model: OreTypeModel,
    config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Attach ore type/domain attributes to block model.
    
    Args:
        block_model: BlockModel instance
        ore_type_model: OreTypeModel
        config: Optional configuration dict
    """
    logger.info("Attaching ore types to block model")
    
    if config is None:
        config = {}
    
    # Get block model DataFrame
    if hasattr(block_model, 'to_dataframe'):
        df = block_model.to_dataframe()
    else:
        logger.error("Block model does not support to_dataframe()")
        return
    
    if df.empty:
        logger.warning("Block model is empty")
        return
    
    # Infer domains for each block
    ore_domains = []
    hardness_indices = []
    densities = []
    
    for idx, row in df.iterrows():
        domain_id = infer_ore_domain_for_block(row, config)
        ore_domains.append(domain_id)
        
        # Get domain properties
        domain = ore_type_model.get_domain(domain_id)
        if domain:
            hardness_indices.append(domain.hardness_index)
            densities.append(domain.density_t_m3)
        else:
            hardness_indices.append(5.0)  # Default
            densities.append(2.7)  # Default
    
    # Add columns to block model
    try:
        if hasattr(block_model, 'add_property'):
            block_model.add_property('ore_domain', np.array(ore_domains, dtype=object))
            block_model.add_property('hardness_index', np.array(hardness_indices, dtype=np.float32))
            block_model.add_property('density_geomet', np.array(densities, dtype=np.float32))
        else:
            # Fallback: add to DataFrame if block model supports it
            df['ore_domain'] = ore_domains
            df['hardness_index'] = hardness_indices
            df['density_geomet'] = densities
            # Try to update block model if it has a method to sync from DataFrame
            if hasattr(block_model, '_sync_from_dataframe'):
                block_model._sync_from_dataframe(df)
    except Exception as e:
        logger.error(f"Error attaching ore types to block model: {e}", exc_info=True)
        raise
    
    logger.info(f"Attached ore types to {len(ore_domains)} blocks")

