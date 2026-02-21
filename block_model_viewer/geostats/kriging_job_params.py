"""
Standardized Parameter Models for Kriging Job Runners.

Uses Pydantic for UI → Worker parameter validation (type conversion, validation, serialization).
Uses dataclasses for internal Numba-safe configs.

This ensures:
- Type safety before heavy calculations start
- Automatic type conversion (strings → floats, etc.)
- Clear error messages
- JSON/YAML serialization support
- IDE autocompletion
- Consistent API across all geostat engines
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Callable, Literal
import pandas as pd
import numpy as np

# Try to import Pydantic (required for proper parameter validation)
try:
    from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    import warnings
    warnings.warn(
        "Pydantic is not installed. Parameter validation will be limited. "
        "Install with: pip install pydantic>=2.0.0",
        ImportWarning
    )
    # Minimal fallback - will raise ImportError when trying to use Pydantic models
    BaseModel = None
    Field = None
    field_validator = None
    model_validator = None
    ValidationError = ValueError


# ============================================================================
# Internal Numba-Safe Configs (Dataclasses)
# ============================================================================

@dataclass
class KrigingConfig:
    """Internal Numba-safe kriging configuration."""
    max_neighbors: int
    kdtree_k: int
    search_radius: Optional[float]
    use_radius_search: bool
    drift_terms: int = 0  # for universal kriging


# ============================================================================
# Pydantic Models for Job Parameters (UI → Worker)
# ============================================================================

if PYDANTIC_AVAILABLE and BaseModel is not None:
    class VariogramParams(BaseModel):
        """Variogram parameters with validation."""
        range_: float = Field(..., gt=0, description="Variogram range (must be positive)")
        sill: float = Field(..., gt=0, description="Total sill (must be positive)")
        nugget: float = Field(0.0, ge=0, description="Nugget effect (non-negative)")
        model_type: Literal["spherical", "exponential", "gaussian"] = Field(
            "spherical",
            description="Variogram model type"
        )
        anisotropy: Optional[Dict[str, Any]] = Field(
            None,
            description="Anisotropy parameters: azimuth, dip, major_range, minor_range, vert_range"
        )
        
        @field_validator('anisotropy')
        @classmethod
        def validate_anisotropy(cls, v):
            if v is not None:
                required_keys = ['azimuth', 'dip', 'major_range', 'minor_range', 'vert_range']
                missing = [k for k in required_keys if k not in v]
                if missing:
                    raise ValueError(f"Anisotropy dict missing keys: {missing}")
                if v['major_range'] <= 0 or v['minor_range'] <= 0 or v['vert_range'] <= 0:
                    raise ValueError("Anisotropy ranges must be positive")
            return v
        
        def to_dict(self) -> Dict[str, Any]:
            """Convert to dictionary format expected by kriging functions."""
            result = {
                'range': self.range_,
                'sill': self.sill,
                'nugget': self.nugget,
                'model_type': self.model_type
            }
            if self.anisotropy:
                result['anisotropy'] = self.anisotropy
            return result


    class GridConfig(BaseModel):
        """Grid configuration for kriging estimation."""
        spacing: Tuple[float, float, float] = Field(
            (10.0, 10.0, 5.0),
            description="Grid spacing (dx, dy, dz)"
        )
        origin: Optional[Tuple[float, float, float]] = Field(
            None,
            description="Grid origin (x0, y0, z0)"
        )
        counts: Optional[Tuple[int, int, int]] = Field(
            None,
            description="Grid counts (nx, ny, nz)"
        )
        max_points: int = Field(50000, gt=0, description="Maximum grid points")
        
        @field_validator('spacing')
        @classmethod
        def validate_spacing(cls, v):
            if any(s <= 0 for s in v):
                raise ValueError("Grid spacing must be positive")
            return v
        
        @field_validator('counts')
        @classmethod
        def validate_counts(cls, v):
            if v is not None and any(c <= 0 for c in v):
                raise ValueError("Grid counts must be positive")
            return v


    class SearchPassConfig(BaseModel):
        """Configuration for a single search pass."""
        min_neighbors: int = Field(..., ge=1, description="Minimum neighbors required")
        max_neighbors: int = Field(..., gt=0, description="Maximum neighbors to use")
        ellipsoid_multiplier: float = Field(1.0, gt=0, description="Search ellipsoid size multiplier")

        @field_validator('max_neighbors')
        @classmethod
        def validate_max_gt_min(cls, v, info):
            if 'min_neighbors' in info.data and v < info.data['min_neighbors']:
                raise ValueError(f"max_neighbors ({v}) must be >= min_neighbors ({info.data['min_neighbors']})")
            return v


    class SearchConfig(BaseModel):
        """Neighbor search configuration with multi-pass support."""
        # Legacy single-pass parameters (for backward compatibility)
        n_neighbors: int = Field(12, gt=0, description="Maximum number of neighbors (legacy single-pass)")
        max_distance: Optional[float] = Field(None, gt=0, description="Maximum search distance")
        min_neighbors: int = Field(3, ge=0, description="Minimum neighbors required (legacy single-pass)")

        # Multi-pass configuration (professional standard)
        use_multi_pass: bool = Field(False, description="Enable multi-pass search strategy")
        passes: Optional[List[SearchPassConfig]] = Field(
            None,
            description="Multi-pass search configurations (Pass 1, 2, 3, ...)"
        )

        @field_validator('max_distance')
        @classmethod
        def validate_max_distance(cls, v):
            if v is not None and v <= 0:
                raise ValueError("max_distance must be positive if provided")
            return v

        @field_validator('passes')
        @classmethod
        def validate_passes(cls, v, info):
            if info.data.get('use_multi_pass') and not v:
                raise ValueError("use_multi_pass=True requires passes configuration")
            if v and len(v) < 1:
                raise ValueError("At least one pass required if passes are configured")
            return v

        @classmethod
        def create_professional_default(
            cls,
            base_max_distance: Optional[float] = None
        ) -> 'SearchConfig':
            """
            Create professional-standard 3-pass search configuration.

            JORC/NI 43-101 compliant defaults:
            - Pass 1: 8-12 neighbors, tight search
            - Pass 2: 6-24 neighbors, relaxed search (1.5x ellipsoid)
            - Pass 3: 4-32 neighbors, fallback (2.0x ellipsoid)

            Parameters
            ----------
            base_max_distance : float, optional
                Base maximum search distance (scaled by ellipsoid_multiplier per pass)

            Returns
            -------
            SearchConfig
                Professional-standard multi-pass configuration
            """
            return cls(
                n_neighbors=12,  # Legacy parameter (not used in multi-pass)
                min_neighbors=8,  # Legacy parameter (not used in multi-pass)
                max_distance=base_max_distance,
                use_multi_pass=True,
                passes=[
                    SearchPassConfig(min_neighbors=8, max_neighbors=12, ellipsoid_multiplier=1.0),
                    SearchPassConfig(min_neighbors=6, max_neighbors=24, ellipsoid_multiplier=1.5),
                    SearchPassConfig(min_neighbors=4, max_neighbors=32, ellipsoid_multiplier=2.0),
                ]
            )


    class UniversalKrigingJobParams(BaseModel):
        """Parameters for Universal Kriging job (Pydantic model)."""
        data_df: Any = Field(..., description="DataFrame with X, Y, Z, and variable columns")
        variable: str = Field(..., description="Variable name to krige")
        variogram_params: VariogramParams = Field(..., description="Variogram parameters")
        grid_config: GridConfig = Field(default_factory=GridConfig, description="Grid configuration")
        search_config: SearchConfig = Field(default_factory=SearchConfig, description="Search configuration")
        drift_type: Literal["constant", "linear", "quadratic"] = Field(
            "linear",
            description="Drift model type"
        )
        post_processing_config: Dict[str, Any] = Field(
            default_factory=dict,
            description="Post-processing filter configuration"
        )
        progress_callback: Optional[Callable] = Field(None, exclude=True, description="Progress callback (excluded from serialization)")
        
        @field_validator('data_df')
        @classmethod
        def validate_data_df(cls, v):
            if not isinstance(v, pd.DataFrame):
                raise ValueError("data_df must be a pandas DataFrame")
            required_cols = ['X', 'Y', 'Z']
            missing = [c for c in required_cols if c not in v.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return v
        
        @model_validator(mode='after')
        def validate_variable(self):
            if self.variable not in self.data_df.columns:
                raise ValueError(f"Variable '{self.variable}' not found in data_df")
            return self
        
        @classmethod
        def from_dict(cls, params: Dict[str, Any]) -> 'UniversalKrigingJobParams':
            """Create from dictionary (for backward compatibility)."""
            # Extract variogram params
            vario_dict = params.get('variogram_params', {})
            variogram_params = VariogramParams(
                range_=vario_dict.get('range', 100.0),
                sill=vario_dict.get('sill', 1.0),
                nugget=vario_dict.get('nugget', 0.0),
                model_type=vario_dict.get('model_type', 'spherical'),
                anisotropy=vario_dict.get('anisotropy')
            )
            
            # Extract grid config
            grid_spacing = params.get('grid_spacing', (10.0, 10.0, 5.0))
            grid_origin = params.get('grid_origin', None)
            grid_counts = params.get('grid_counts', None)
            grid_config = GridConfig(
                spacing=grid_spacing,
                origin=grid_origin,
                counts=grid_counts
            )
            
            # Extract search config
            search_config = SearchConfig(
                n_neighbors=params.get('n_neighbors', 12),
                max_distance=params.get('max_distance', None)
            )
            
            return cls(
                data_df=params['data_df'],
                variable=params['variable'],
                variogram_params=variogram_params,
                grid_config=grid_config,
                search_config=search_config,
                drift_type=params.get('drift_type', 'linear'),
                progress_callback=params.get('_progress_callback')
            )


    class CoKrigingJobParams(BaseModel):
        """Parameters for Co-Kriging job (Pydantic model)."""
        data_df: Any = Field(..., description="DataFrame with X, Y, Z, and variable columns")
        primary_name: str = Field(..., description="Primary variable name")
        secondary_name: str = Field(..., description="Secondary variable name")
        variogram_primary: VariogramParams = Field(..., description="Primary variogram parameters")
        variogram_secondary: VariogramParams = Field(..., description="Secondary variogram parameters")
        cross_variogram: Optional[VariogramParams] = Field(None, description="Cross-variogram parameters")
        grid_config: GridConfig = Field(default_factory=GridConfig, description="Grid configuration")
        search_config: SearchConfig = Field(default_factory=SearchConfig, description="Search configuration")
        method: Literal["collocated", "full"] = Field("collocated", description="Co-kriging method")
        progress_callback: Optional[Callable] = Field(None, exclude=True, description="Progress callback")
        
        @field_validator('data_df')
        @classmethod
        def validate_data_df(cls, v):
            if not isinstance(v, pd.DataFrame):
                raise ValueError("data_df must be a pandas DataFrame")
            required_cols = ['X', 'Y', 'Z']
            missing = [c for c in required_cols if c not in v.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return v
        
        @model_validator(mode='after')
        def validate_variables(self):
            if self.primary_name not in self.data_df.columns:
                raise ValueError(f"Primary variable '{self.primary_name}' not found in data_df")
            if self.secondary_name not in self.data_df.columns:
                raise ValueError(f"Secondary variable '{self.secondary_name}' not found in data_df")
            return self
        
        @classmethod
        def from_dict(cls, params: Dict[str, Any]) -> 'CoKrigingJobParams':
            """Create from dictionary (for backward compatibility)."""
            # Extract variogram params
            vario_prim_dict = params.get('variogram_primary') or {}
            variogram_primary = VariogramParams(
                range_=vario_prim_dict.get('range', 100.0),
                sill=vario_prim_dict.get('sill', 1.0),
                nugget=vario_prim_dict.get('nugget', 0.0),
                model_type=vario_prim_dict.get('model_type', 'spherical'),
                anisotropy=vario_prim_dict.get('anisotropy')
            )

            vario_sec_dict = params.get('variogram_secondary') or {}
            variogram_secondary = VariogramParams(
                range_=vario_sec_dict.get('range', 100.0),
                sill=vario_sec_dict.get('sill', 1.0),
                nugget=vario_sec_dict.get('nugget', 0.0),
                model_type=vario_sec_dict.get('model_type', 'spherical'),
                anisotropy=vario_sec_dict.get('anisotropy')
            )
            
            cross_vario_dict = params.get('cross_variogram')
            cross_variogram = None
            if cross_vario_dict and isinstance(cross_vario_dict, dict):
                cross_variogram = VariogramParams(
                    range_=cross_vario_dict.get('range', 100.0),
                    sill=cross_vario_dict.get('sill', 1.0),
                    nugget=cross_vario_dict.get('nugget', 0.0),
                    model_type=cross_vario_dict.get('model_type', 'spherical'),
                    anisotropy=cross_vario_dict.get('anisotropy')
                )
            
            # Extract grid config
            grid_spacing = params.get('grid_spacing', (10.0, 10.0, 5.0))
            grid_origin = params.get('grid_origin', None)
            grid_counts = params.get('grid_counts', None)
            grid_config = GridConfig(
                spacing=grid_spacing,
                origin=grid_origin,
                counts=grid_counts
            )
            
            # Extract search config with min_neighbors support
            search_config = SearchConfig(
                n_neighbors=params.get('n_neighbors', 12),
                max_distance=params.get('max_distance', None),
                min_neighbors=params.get('min_neighbors', 3)  # Professional audit setting
            )
            
            return cls(
                data_df=params['data_df'],
                primary_name=params['primary_name'],
                secondary_name=params['secondary_name'],
                variogram_primary=variogram_primary,
                variogram_secondary=variogram_secondary,
                cross_variogram=cross_variogram,
                grid_config=grid_config,
                search_config=search_config,
                method=params.get('method', 'collocated'),
                progress_callback=params.get('_progress_callback')
            )


    class IndicatorKrigingJobParams(BaseModel):
        """Parameters for Indicator Kriging job (Pydantic model)."""
        data_df: Any = Field(..., description="DataFrame with X, Y, Z, and variable columns")
        variable: str = Field(..., description="Variable name to krige")
        thresholds: List[float] = Field(..., min_length=2, description="Threshold values (must be sorted)")
        variogram_template: VariogramParams = Field(..., description="Variogram template parameters")
        grid_config: GridConfig = Field(default_factory=GridConfig, description="Grid configuration")
        search_config: SearchConfig = Field(default_factory=SearchConfig, description="Search configuration")
        compute_median: bool = Field(True, description="Compute median from CDF")
        compute_mean: bool = Field(False, description="Compute mean (E-type) from CDF")
        progress_callback: Optional[Callable] = Field(None, exclude=True, description="Progress callback")
        
        @field_validator('data_df')
        @classmethod
        def validate_data_df(cls, v):
            if not isinstance(v, pd.DataFrame):
                raise ValueError("data_df must be a pandas DataFrame")
            required_cols = ['X', 'Y', 'Z']
            missing = [c for c in required_cols if c not in v.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            return v
        
        @field_validator('thresholds')
        @classmethod
        def validate_thresholds(cls, v):
            if len(v) < 2:
                raise ValueError("At least 2 thresholds required for Indicator Kriging")
            # Check if sorted
            if v != sorted(v):
                raise ValueError("Thresholds must be sorted in ascending order")
            return v
        
        @model_validator(mode='after')
        def validate_variable(self):
            if self.variable not in self.data_df.columns:
                raise ValueError(f"Variable '{self.variable}' not found in data_df")
            return self
        
        @classmethod
        def from_dict(cls, params: Dict[str, Any]) -> 'IndicatorKrigingJobParams':
            """Create from dictionary (for backward compatibility)."""
            # Extract variogram params
            vario_dict = params.get('variogram_template', {})
            variogram_template = VariogramParams(
                range_=vario_dict.get('range', 100.0),
                sill=vario_dict.get('sill', 1.0),
                nugget=vario_dict.get('nugget', 0.0),
                model_type=vario_dict.get('model_type', 'spherical'),
                anisotropy=vario_dict.get('anisotropy')
            )
            
            # Extract grid config
            grid_spacing = params.get('grid_spacing', (10.0, 10.0, 5.0))
            grid_origin = params.get('grid_origin', None)
            grid_counts = params.get('grid_counts', None)
            grid_config = GridConfig(
                spacing=grid_spacing,
                origin=grid_origin,
                counts=grid_counts
            )
            
            # Extract search config
            search_config = SearchConfig(
                n_neighbors=params.get('n_neighbors', 12),
                max_distance=params.get('max_distance', None)
            )
            
            thresholds = params.get('thresholds', [])
            if isinstance(thresholds, (list, tuple)):
                thresholds = list(thresholds)
            else:
                thresholds = [thresholds]
            
            return cls(
                data_df=params['data_df'],
                variable=params['variable'],
                thresholds=thresholds,
                variogram_template=variogram_template,
                grid_config=grid_config,
                search_config=search_config,
                compute_median=params.get('compute_median', True),
                compute_mean=params.get('compute_mean', False),
                progress_callback=params.get('_progress_callback')
            )

else:
    # Fallback to dataclasses if Pydantic not available
    from dataclasses import dataclass, field
    
    @dataclass
    class VariogramParams:
        """Variogram parameters (fallback)."""
        range_: float
        sill: float
        nugget: float = 0.0
        model_type: str = "spherical"
        anisotropy: Optional[Dict[str, Any]] = None
        
        def __post_init__(self):
            if self.range_ <= 0:
                raise ValueError("Variogram range must be positive")
            if self.sill <= 0:
                raise ValueError("Variogram sill must be positive")
            if self.nugget < 0:
                raise ValueError("Variogram nugget must be non-negative")
        
        def to_dict(self) -> Dict[str, Any]:
            result = {'range': self.range_, 'sill': self.sill, 'nugget': self.nugget, 'model_type': self.model_type}
            if self.anisotropy:
                result['anisotropy'] = self.anisotropy
            return result
    
    @dataclass
    class GridConfig:
        """Grid configuration (fallback)."""
        spacing: Tuple[float, float, float] = (10.0, 10.0, 5.0)
        origin: Optional[Tuple[float, float, float]] = None
        counts: Optional[Tuple[int, int, int]] = None
        max_points: int = 50000
        
        def __post_init__(self):
            if any(s <= 0 for s in self.spacing):
                raise ValueError("Grid spacing must be positive")
            if self.counts is not None and any(c <= 0 for c in self.counts):
                raise ValueError("Grid counts must be positive")
    
    @dataclass
    class SearchConfig:
        """Search configuration (fallback)."""
        n_neighbors: int = 12
        max_distance: Optional[float] = None
        min_neighbors: int = 3
        
        def __post_init__(self):
            if self.n_neighbors <= 0:
                raise ValueError("n_neighbors must be positive")
            if self.min_neighbors < 0:
                raise ValueError("min_neighbors must be non-negative")
            if self.max_distance is not None and self.max_distance <= 0:
                raise ValueError("max_distance must be positive if provided")
    
    # Fallback job params classes would go here (similar structure but using dataclass)
    # For brevity, keeping the Pydantic versions as primary
