"""
IRR Engine Provenance Module

Provides cryptographic hashing, timestamping, and lineage tracking for
reproducible and auditable IRR results.

This module addresses audit findings:
- Violation #4: Incomplete Provenance Metadata
- Violation #6: Stochastic Results Not Seeded Deterministically

Author: GeoX Mining Software
Date: 2025-12
"""

import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# VERSION TRACKING
# =============================================================================

IRR_ENGINE_VERSION = "1.2.0"  # Increment when algorithm changes
PROVENANCE_SCHEMA_VERSION = "1.0"


def get_system_info() -> Dict[str, str]:
    """Get system information for reproducibility."""
    return {
        'python_version': sys.version,
        'platform': platform.platform(),
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__
    }


# =============================================================================
# VIOLATION #4 FIX: Provenance Hashing
# =============================================================================

def compute_hash(data: Any, truncate: int = 16) -> str:
    """
    Compute SHA-256 hash of arbitrary data.
    
    Args:
        data: Data to hash (will be serialized to JSON)
        truncate: Number of characters to keep (default 16)
        
    Returns:
        Truncated hex hash string
    """
    if isinstance(data, np.ndarray):
        # For numpy arrays, use tobytes for efficiency
        content = data.tobytes()
    elif isinstance(data, pd.DataFrame):
        # For DataFrames, hash the values and column names
        content = (
            data.values.tobytes() + 
            '|'.join(data.columns.tolist()).encode() +
            str(data.index.tolist()).encode()
        )
    elif isinstance(data, dict):
        # For dicts, serialize to JSON with sorted keys
        content = json.dumps(data, sort_keys=True, default=str).encode()
    elif isinstance(data, (list, tuple)):
        content = json.dumps(list(data), default=str).encode()
    elif isinstance(data, (int, float, str)):
        content = str(data).encode()
    else:
        content = repr(data).encode()
    
    full_hash = hashlib.sha256(content).hexdigest()
    return full_hash[:truncate]


def compute_block_model_hash(block_model: pd.DataFrame) -> str:
    """
    Compute hash of block model for provenance tracking.
    
    Includes:
    - All numeric values
    - Column names
    - Row count
    - Key statistics (sum of TONNAGE, GRADE if present)
    
    Args:
        block_model: Block model DataFrame
        
    Returns:
        16-character hex hash
    """
    # Create signature including structure and key values
    signature = {
        'n_blocks': len(block_model),
        'columns': sorted(block_model.columns.tolist()),
        'dtypes': {k: str(v) for k, v in block_model.dtypes.items()}
    }
    
    # Add key statistics for numerical columns
    for col in ['TONNAGE', 'GRADE', 'VALUE', 'X', 'Y', 'Z', 'XC', 'YC', 'ZC']:
        if col in block_model.columns:
            try:
                signature[f'{col}_sum'] = float(block_model[col].sum())
                signature[f'{col}_mean'] = float(block_model[col].mean())
            except (TypeError, ValueError):
                pass
    
    # Add data hash
    try:
        signature['data_hash'] = compute_hash(block_model.values, truncate=32)
    except Exception:
        signature['data_hash'] = 'unavailable'
    
    return compute_hash(signature)


def compute_config_hash(config: Dict[str, Any]) -> str:
    """
    Compute hash of configuration for provenance tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        16-character hex hash
    """
    # Remove non-deterministic fields before hashing
    clean_config = {k: v for k, v in config.items() 
                    if k not in ['timestamp', 'random_seed', '_internal']}
    return compute_hash(clean_config)


def compute_economic_params_hash(params: Dict[str, Any]) -> str:
    """
    Compute hash of economic parameters.
    
    Args:
        params: Economic parameters dictionary
        
    Returns:
        16-character hex hash
    """
    # Extract only economically relevant parameters
    relevant_keys = [
        'metal_price', 'mining_cost', 'processing_cost', 'recovery',
        'selling_cost', 'capex', 'discount_rate', 'by_products'
    ]
    relevant_params = {k: v for k, v in params.items() if k in relevant_keys}
    return compute_hash(relevant_params)


# =============================================================================
# VIOLATION #6 FIX: Deterministic Random Seeding
# =============================================================================

DEFAULT_RANDOM_SEED = 42  # Standard reproducible seed


@dataclass
class RandomSeedManager:
    """
    Manages random seed for reproducible stochastic analysis.
    
    Ensures that:
    1. A seed is always set (no silent non-determinism)
    2. Seed is recorded in provenance
    3. Seed can be recovered from results
    """
    seed: int = DEFAULT_RANDOM_SEED
    seed_source: str = "explicit"  # "explicit", "default", "timestamp"
    _original_state: Optional[Dict] = field(default=None, repr=False)
    
    def __post_init__(self):
        if self.seed is None:
            # Use timestamp-based seed but LOG it clearly
            self.seed = int(datetime.now(timezone.utc).timestamp() * 1000) % (2**31)
            self.seed_source = "timestamp"
            logger.warning(
                f"No random seed provided - using timestamp-derived seed: {self.seed}. "
                "For reproducible results, explicitly set random_seed in configuration."
            )
    
    def apply(self) -> None:
        """Apply the seed to numpy random state."""
        self._original_state = np.random.get_state()
        np.random.seed(self.seed)
        logger.info(f"Random seed applied: {self.seed} (source: {self.seed_source})")
    
    def restore(self) -> None:
        """Restore original random state if saved."""
        if self._original_state is not None:
            np.random.set_state(self._original_state)
            logger.debug("Random state restored to original")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export seed info for provenance."""
        return {
            'seed': self.seed,
            'seed_source': self.seed_source
        }


def ensure_deterministic_seed(
    config: Dict[str, Any],
    key: str = 'random_seed'
) -> RandomSeedManager:
    """
    Ensure configuration has a deterministic random seed.
    
    Args:
        config: Configuration dictionary
        key: Key for random seed in config
        
    Returns:
        RandomSeedManager with applied seed
    """
    seed = config.get(key)
    
    if seed is None:
        seed = DEFAULT_RANDOM_SEED
        source = "default"
        logger.info(
            f"No random_seed in config - using default seed {seed} for reproducibility."
        )
    else:
        source = "explicit"
    
    manager = RandomSeedManager(seed=int(seed), seed_source=source)
    manager.apply()
    
    # Update config with actual seed used
    config[key] = manager.seed
    
    return manager


# =============================================================================
# COMPREHENSIVE PROVENANCE RECORD
# =============================================================================

@dataclass
class IRRProvenance:
    """
    Complete provenance record for IRR analysis.
    
    This record enables:
    - Full reproducibility of results
    - Audit trail for regulatory compliance
    - Version tracking of algorithms
    """
    # Timestamps
    analysis_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Version info
    engine_version: str = IRR_ENGINE_VERSION
    provenance_schema: str = PROVENANCE_SCHEMA_VERSION
    
    # Hashes for data integrity
    block_model_hash: Optional[str] = None
    config_hash: Optional[str] = None
    economic_params_hash: Optional[str] = None
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None
    random_seed_source: str = "unknown"
    
    # Input summary (for quick verification without rehashing)
    input_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Classification filter applied
    classification_filter: Optional[List[str]] = None
    blocks_before_filter: int = 0
    blocks_after_filter: int = 0
    
    # Solver info
    solver_type: str = "unknown"
    solver_version: Optional[str] = None
    
    # System info
    system_info: Dict[str, str] = field(default_factory=get_system_info)
    
    # Validation metadata
    validation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IRRProvenance':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def verify_block_model(self, block_model: pd.DataFrame) -> bool:
        """Verify block model matches recorded hash."""
        if self.block_model_hash is None:
            logger.warning("No block model hash recorded - cannot verify")
            return False
        
        current_hash = compute_block_model_hash(block_model)
        matches = current_hash == self.block_model_hash
        
        if not matches:
            logger.warning(
                f"Block model hash mismatch! "
                f"Recorded: {self.block_model_hash}, Current: {current_hash}"
            )
        
        return matches


def create_provenance(
    block_model: pd.DataFrame,
    config: Dict[str, Any],
    economic_params: Dict[str, Any],
    seed_manager: Optional[RandomSeedManager] = None,
    classification_filter: Optional[List[str]] = None,
    blocks_before_filter: int = 0,
    validation_metadata: Optional[Dict[str, Any]] = None,
    solver_type: str = "unknown"
) -> IRRProvenance:
    """
    Create comprehensive provenance record for IRR analysis.
    
    Args:
        block_model: Block model used (after filtering)
        config: Full configuration
        economic_params: Economic parameters used
        seed_manager: Random seed manager
        classification_filter: Classifications included
        blocks_before_filter: Block count before filtering
        validation_metadata: Any validation results
        solver_type: Type of solver used
        
    Returns:
        Complete IRRProvenance record
    """
    provenance = IRRProvenance(
        block_model_hash=compute_block_model_hash(block_model),
        config_hash=compute_config_hash(config),
        economic_params_hash=compute_economic_params_hash(economic_params),
        classification_filter=classification_filter,
        blocks_before_filter=blocks_before_filter,
        blocks_after_filter=len(block_model),
        solver_type=solver_type,
        validation_metadata=validation_metadata or {}
    )
    
    if seed_manager:
        provenance.random_seed = seed_manager.seed
        provenance.random_seed_source = seed_manager.seed_source
    
    # Create input summary
    provenance.input_summary = {
        'n_blocks': len(block_model),
        'metal_price': economic_params.get('metal_price'),
        'mining_cost': economic_params.get('mining_cost'),
        'processing_cost': economic_params.get('processing_cost'),
        'recovery': economic_params.get('recovery'),
        'num_scenarios': config.get('scenario_generation', {}).get('num_scenarios'),
        'num_periods': config.get('scenario_generation', {}).get('num_periods')
    }
    
    logger.info(
        f"Provenance record created: "
        f"block_model_hash={provenance.block_model_hash}, "
        f"config_hash={provenance.config_hash}, "
        f"seed={provenance.random_seed}"
    )
    
    return provenance


# =============================================================================
# EXPORT FOR AUDIT
# =============================================================================

def export_provenance_report(
    provenance: IRRProvenance,
    irr_result: Dict[str, Any],
    output_path: str
) -> None:
    """
    Export complete provenance report for audit purposes.
    
    Args:
        provenance: IRRProvenance record
        irr_result: IRR analysis results
        output_path: Path to save report
    """
    report = {
        'provenance': provenance.to_dict(),
        'results_summary': {
            'irr_alpha': irr_result.get('irr_alpha'),
            'mean_npv': irr_result.get('mean_npv'),
            'num_scenarios': irr_result.get('num_scenarios'),
            'satisfaction_rate': irr_result.get('satisfaction_rate')
        },
        'export_timestamp': datetime.now(timezone.utc).isoformat()
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Provenance report exported to {output_path}")

