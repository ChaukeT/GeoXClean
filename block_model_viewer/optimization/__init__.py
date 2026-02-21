"""
Optimization algorithms for mining applications.

High-performance implementations of pit optimization and related algorithms.

NOTE: The Lerchs-Grossmann solver has been consolidated into models/pit_optimizer.py
      to eliminate duplicate implementations. Use the unified API:
      
      from block_model_viewer.models.pit_optimizer import (
          lerchs_grossmann_optimize_fast,  # Fast scipy-based solver
          lerchs_grossmann_optimize,        # NetworkX-based solver (full features)
          is_fast_solver_available,
      )
"""

# Re-export from unified location for backward compatibility
from ..models.pit_optimizer import (
    lerchs_grossmann_optimize_fast,
    lerchs_grossmann_optimize,
    is_fast_solver_available,
)

# Legacy alias for backward compatibility
# Note: LerchsGrossmannSolver class has been replaced with function-based API
def LerchsGrossmannSolver(block_model, value_col='VALUE'):
    """
    Legacy wrapper for backward compatibility.
    
    DEPRECATED: Use lerchs_grossmann_optimize_fast() instead.
    """
    import warnings
    warnings.warn(
        "LerchsGrossmannSolver is deprecated. Use lerchs_grossmann_optimize_fast() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    class _LegacySolver:
        def __init__(self, block_model, value_col):
            self.block_model = block_model
            self.value_col = value_col
        
        def solve(self):
            return lerchs_grossmann_optimize_fast(self.block_model, self.value_col)
    
    return _LegacySolver(block_model, value_col)


__all__ = [
    'lerchs_grossmann_optimize_fast',
    'lerchs_grossmann_optimize',
    'is_fast_solver_available',
    'LerchsGrossmannSolver',  # Deprecated, for backward compatibility
]

