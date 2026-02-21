"""
Block Model data structure for storing 3D block model information.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except Exception:
    # Catch any exception (ImportError, OSError from DLL init failures, etc.)
    # so the application can still start when llvmlite/numba fail to load.
    _NUMBA_AVAILABLE = False
    njit = None  # type: ignore

logger = logging.getLogger(__name__)

if _NUMBA_AVAILABLE:

    @njit(cache=True)
    def _compute_block_corners_numba(positions, half_dims):
        n = positions.shape[0]
        corners = np.empty((n, 8, 3), dtype=np.float64)
        offsets = np.array([
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ], dtype=np.float64)
        for i in range(n):
            cx = positions[i, 0]
            cy = positions[i, 1]
            cz = positions[i, 2]
            hx = half_dims[i, 0]
            hy = half_dims[i, 1]
            hz = half_dims[i, 2]
            for j in range(8):
                corners[i, j, 0] = cx + hx * offsets[j, 0]
                corners[i, j, 1] = cy + hy * offsets[j, 1]
                corners[i, j, 2] = cz + hz * offsets[j, 2]
        return corners

else:

    def _compute_block_corners_numba(*args, **kwargs):
        raise RuntimeError("Numba is not available")


@dataclass
class BlockMetadata:
    """
    Metadata for a block model.
    
    GeoX Invariant Compliance:
    - Includes provenance tracking fields
    - File checksum for data integrity
    - Parser version for reproducibility
    """
    coordinate_system: str = "unknown"
    units: str = "meters"
    source_file: str = ""
    file_format: str = ""
    creation_date: str = ""
    description: str = ""
    # GeoX invariant: Provenance fields
    file_checksum: str = ""
    checksum_algorithm: str = "sha256"
    import_timestamp: str = ""
    parser_version: str = ""
    parser_framework_version: str = ""
    column_mapping: Optional[Dict[str, str]] = None
    inferred_dimensions: bool = False  # Track if dimensions were inferred


class BlockModel:
    """
    Core data structure for storing 3D block model information.
    
    Stores block geometry (positions, dimensions) and properties (grade, density, etc.)
    with metadata about coordinate system, units, and source information.
    """
    
    def __init__(self, metadata: Optional[BlockMetadata] = None):
        """
        Initialize a BlockModel.
        
        Args:
            metadata: Optional metadata about the block model
        """
        self.metadata = metadata or BlockMetadata()
        
        # Core geometry data
        self._positions: Optional[np.ndarray] = None  # (N, 3) array of block centers
        self._dimensions: Optional[np.ndarray] = None  # (N, 3) array of block sizes
        self._properties: Dict[str, np.ndarray] = {}  # Property name -> values array
        
        # Rotation and grid information
        self._rotation_matrix: Optional[np.ndarray] = None  # 3x3 rotation matrix (None = no rotation)
        
        # Computed properties
        self._bounds: Optional[Tuple[float, float, float, float, float, float]] = None
        self._block_count: int = 0
        self._is_orthogonal_cache: Optional[Tuple[bool, Optional[Tuple]]] = None  # Cache for is_orthogonal result
        
    @property
    def positions(self) -> Optional[np.ndarray]:
        """Get block center positions as (N, 3) array."""
        return self._positions
    
    @property
    def dimensions(self) -> Optional[np.ndarray]:
        """Get block dimensions as (N, 3) array."""
        return self._dimensions
    
    @property
    def properties(self) -> Dict[str, np.ndarray]:
        """Get all properties as a dictionary."""
        return self._properties.copy()
    
    @property
    def block_count(self) -> int:
        """Get number of blocks in the model."""
        return self._block_count

    @property
    def coordinates(self) -> Optional[np.ndarray]:
        """Get block center coordinates as (N, 3) array (alias for positions)."""
        return self._positions

    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        """
        Get model bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
        
        Returns:
            Tuple of bounds or None if no data
        """
        return self._bounds
    
    def set_geometry(self, positions: np.ndarray, dimensions: np.ndarray) -> None:
        """
        Set block geometry data.
        
        Args:
            positions: (N, 3) array of block center positions
            dimensions: (N, 3) array of block dimensions (dx, dy, dz)
            
        Raises:
            ValueError: If arrays have incompatible shapes
        """
        if positions.shape[0] != dimensions.shape[0]:
            raise ValueError("Positions and dimensions must have same number of blocks")
        
        if positions.shape[1] != 3 or dimensions.shape[1] != 3:
            raise ValueError("Positions and dimensions must have 3 columns (x, y, z)")
        
        # STEP 18: Optimize dtypes - use float32 for coordinates
        if positions.dtype != np.float32:
            self._positions = positions.astype(np.float32, copy=False)
        else:
            self._positions = positions.copy()
        
        if dimensions.dtype != np.float32:
            self._dimensions = dimensions.astype(np.float32, copy=False)
        else:
            self._dimensions = dimensions.copy()
        
        self._block_count = positions.shape[0]
        
        # Clear orthogonal cache when geometry changes
        self._is_orthogonal_cache = None
        
        # Update bounds
        self._update_bounds()
        
        logger.info(f"Set geometry for {self._block_count} blocks")
    
    def add_property(self, name: str, values: np.ndarray) -> None:
        """
        Add a property to the block model.
        
        Args:
            name: Property name
            values: Array of property values (length must match block count)
            
        Raises:
            ValueError: If values length doesn't match block count
        """
        if self._block_count == 0:
            raise ValueError("Must set geometry before adding properties")
        
        if len(values) != self._block_count:
            raise ValueError(f"Property values length ({len(values)}) must match block count ({self._block_count})")
        
        # STEP 18: Optimize dtype based on property type
        optimized_values = self._optimize_property_dtype(values)
        self._properties[name] = optimized_values
        logger.info(f"Added property '{name}' with {len(values)} values")
    
    # =========================================================================
    # STEP 22: Helper methods for UK/CoK/IK results
    # =========================================================================
    
    def add_universal_kriging_result(self, property_name: str, estimates: np.ndarray, variance: np.ndarray) -> None:
        """
        Add Universal Kriging results to block model.
        
        Args:
            property_name: Base property name (e.g., 'Fe')
            estimates: (N,) estimated values
            variance: (N,) kriging variance values
        """
        self.add_property(f"uk_{property_name}", estimates)
        self.add_property(f"uk_{property_name}_var", variance)
        logger.info(f"Added Universal Kriging results for '{property_name}'")
    
    def add_cokriging_result(self, primary_name: str, result: Any) -> None:
        """
        Add Co-Kriging results to block model.
        
        Args:
            primary_name: Primary variable name
            result: CoKrigingResult instance or dict with 'estimates' and 'variance'
        """
        if hasattr(result, 'estimates'):
            estimates = result.estimates
            variance = result.variance
        else:
            estimates = result['estimates']
            variance = result['variance']
        
        self.add_property(f"cok_{primary_name}", estimates)
        self.add_property(f"cok_{primary_name}_var", variance)
        logger.info(f"Added Co-Kriging results for '{primary_name}'")
    
    def add_indicator_kriging_result(self, property_name: str, ik_result: Any) -> None:
        """
        Add Indicator Kriging results to block model.
        
        Args:
            property_name: Base property name
            ik_result: IKResult instance or dict with 'probabilities', 'thresholds', etc.
        """
        if hasattr(ik_result, 'probabilities'):
            probabilities = ik_result.probabilities
            thresholds = ik_result.thresholds
            median = getattr(ik_result, 'median', None)
            mean = getattr(ik_result, 'mean', None)
        else:
            probabilities = ik_result['probabilities']
            thresholds = ik_result['thresholds']
            median = ik_result.get('median')
            mean = ik_result.get('mean')
        
        # Store probabilities per threshold
        n_blocks = probabilities.shape[0]
        for k, threshold in enumerate(thresholds):
            prop_name = f"ik_{property_name}_p_le_{threshold:g}"
            self.add_property(prop_name, probabilities[:, k])
        
        # Store median if available
        if median is not None:
            self.add_property(f"ik_{property_name}_median", median)
        
        # Store mean if available
        if mean is not None:
            self.add_property(f"ik_{property_name}_mean", mean)
        
        logger.info(f"Added Indicator Kriging results for '{property_name}': "
                   f"{len(thresholds)} thresholds, median={median is not None}, mean={mean is not None}")
    
    def _optimize_property_dtype(self, values: np.ndarray) -> np.ndarray:
        """
        Optimize property dtype for memory efficiency.
        
        Args:
            values: Property values array
            
        Returns:
            Optimized array with appropriate dtype
        """
        if not np.issubdtype(values.dtype, np.number):
            # Non-numeric (strings, etc.) - return as-is
            return values.copy()
        
        # Float values: use float32 instead of float64
        if np.issubdtype(values.dtype, np.floating):
            if values.dtype == np.float64:
                return values.astype(np.float32, copy=False)
            return values.copy()
        
        # Integer values: use smallest appropriate type
        if np.issubdtype(values.dtype, np.integer):
            min_val = np.min(values)
            max_val = np.max(values)
            
            if min_val >= 0:  # Unsigned
                if max_val <= 255:
                    return values.astype(np.uint8, copy=False)
                elif max_val <= 65535:
                    return values.astype(np.uint16, copy=False)
            else:  # Signed
                if min_val >= -128 and max_val <= 127:
                    return values.astype(np.int8, copy=False)
                elif min_val >= -32768 and max_val <= 32767:
                    return values.astype(np.int16, copy=False)
        
        return values.copy()
    
    def calculate_tonnage(self, density_field: Optional[str] = None, default_density: float = 2.7) -> np.ndarray:
        """
        Calculate tonnage for each block based on volume and density.
        
        Tonnage = Volume × Density
        Volume = DX × DY × DZ
        
        Args:
            density_field: Name of density property field (optional)
            default_density: Default density to use if density_field not specified (tonnes/m³)
                           Default is 2.7 t/m³ (typical for mixed rock)
            
        Returns:
            Array of tonnage values for each block
            
        Raises:
            ValueError: If geometry not set or density field doesn't exist
        """
        if self._dimensions is None:
            raise ValueError("Cannot calculate tonnage: block dimensions not set")
        
        # Calculate volume (DX × DY × DZ)
        volumes = self._dimensions[:, 0] * self._dimensions[:, 1] * self._dimensions[:, 2]
        
        # Get density values
        if density_field is not None:
            if density_field not in self._properties:
                raise ValueError(f"Density field '{density_field}' not found in properties")
            densities = self._properties[density_field]
        else:
            densities = np.full(self._block_count, default_density)
        
        # Calculate tonnage
        tonnage = volumes * densities
        
        logger.info(f"Calculated tonnage for {self._block_count} blocks (volume × density)")
        logger.info(f"Total tonnage: {tonnage.sum()/1e6:.2f} Mt, Average: {tonnage.mean():.2f} t/block")
        
        return tonnage
    
    def ensure_tonnage_property(self, tonnage_field: Optional[str] = None, 
                                density_field: Optional[str] = None,
                                default_density: float = 2.7) -> str:
        """
        Ensure a tonnage property exists in the block model.
        
        If tonnage_field exists, uses it. Otherwise, calculates tonnage from volume and density.
        
        Args:
            tonnage_field: Name of existing tonnage field (if any)
            density_field: Name of density field for calculation (optional)
            default_density: Default density if no density field specified
            
        Returns:
            Name of the tonnage field (either existing or newly created)
        """
        # Check if tonnage field already exists
        if tonnage_field and tonnage_field in self._properties:
            logger.info(f"Using existing tonnage field: {tonnage_field}")
            return tonnage_field
        
        # Calculate tonnage
        tonnage = self.calculate_tonnage(density_field, default_density)
        
        # Add as a property
        self.add_property('TONNAGE', tonnage)
        
        logger.info("Added calculated TONNAGE property to block model")
        return 'TONNAGE'
    
    def get_property(self, name: str) -> Optional[np.ndarray]:
        """
        Get a property by name.
        
        Args:
            name: Property name
            
        Returns:
            Property values array or None if not found
        """
        return self._properties.get(name)
    
    def get_property_names(self) -> List[str]:
        """Get list of all property names."""
        return list(self._properties.keys())
    
    def get_property_statistics(self, name: str) -> Optional[Dict[str, float]]:
        """
        Get statistics for a property.
        
        Args:
            name: Property name
            
        Returns:
            Dictionary with min, max, mean, std or None if property not found
        """
        values = self.get_property(name)
        if values is None:
            return None
        
        if not np.issubdtype(values.dtype, np.number):
            return {"type": "categorical", "unique_count": len(np.unique(values))}
        
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": len(values)
        }
    
    def filter_blocks(self, property_name: str, min_value: Optional[float] = None, 
                     max_value: Optional[float] = None) -> np.ndarray:
        """
        Get indices of blocks matching property filter criteria.
        
        Args:
            property_name: Name of property to filter on
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            
        Returns:
            Array of block indices matching criteria
        """
        if property_name not in self._properties:
            return np.array([])
        
        values = self._properties[property_name]
        mask = np.ones(len(values), dtype=bool)
        
        if min_value is not None:
            mask &= values >= min_value
        if max_value is not None:
            mask &= values <= max_value
        
        return np.where(mask)[0]
    
    def get_block_corners(self, indices: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get block corner coordinates.
        
        Args:
            indices: Optional array of block indices to get corners for
            
        Returns:
            Array of shape (N, 8, 3) containing 8 corner points for each block
        """
        if self._positions is None or self._dimensions is None:
            return np.array([])
        
        if indices is None:
            positions = self._positions
            dimensions = self._dimensions
        else:
            positions = self._positions[indices]
            dimensions = self._dimensions[indices]
        
        # Calculate half-dimensions
        half_dims = dimensions / 2.0

        if _NUMBA_AVAILABLE:
            try:
                pos64 = np.ascontiguousarray(positions, dtype=np.float64)
                half64 = np.ascontiguousarray(half_dims, dtype=np.float64)
                corners64 = _compute_block_corners_numba(pos64, half64)
                if positions.dtype != np.float64:
                    return corners64.astype(positions.dtype, copy=False)
                return corners64
            except Exception as exc:
                logger.debug(f"Numba corner generation failed, falling back to numpy: {exc}")

        signs = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ], dtype=positions.dtype if positions.dtype.kind in {"f"} else np.float64)
        corners = positions[:, None, :] + half_dims[:, None, :] * signs[None, :, :]
        return np.asarray(corners, dtype=positions.dtype)
    
    def _update_bounds(self) -> None:
        """Update model bounds from current geometry."""
        if self._positions is None or self._dimensions is None:
            self._bounds = None
            return
        
        # Calculate bounds from block positions and dimensions
        half_dims = self._dimensions / 2.0
        
        min_positions = self._positions - half_dims
        max_positions = self._positions + half_dims
        
        self._bounds = (
            float(np.min(min_positions[:, 0])),  # xmin
            float(np.max(max_positions[:, 0])),  # xmax
            float(np.min(min_positions[:, 1])),  # ymin
            float(np.max(max_positions[:, 1])),  # ymax
            float(np.min(min_positions[:, 2])),  # zmin
            float(np.max(max_positions[:, 2]))   # zmax
        )
    
    def validate(self) -> List[str]:
        """
        Validate the block model data.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if self._positions is None:
            errors.append("No position data")
        elif self._positions.shape[1] != 3:
            errors.append("Positions must have 3 columns (x, y, z)")
        
        if self._dimensions is None:
            errors.append("No dimension data")
        elif self._dimensions.shape[1] != 3:
            errors.append("Dimensions must have 3 columns (dx, dy, dz)")
        
        if self._positions is not None and self._dimensions is not None:
            if self._positions.shape[0] != self._dimensions.shape[0]:
                errors.append("Position and dimension arrays must have same length")
            
            # Check for negative dimensions
            if np.any(self._dimensions <= 0):
                errors.append("All block dimensions must be positive")
        
        # Validate properties
        for prop_name, prop_values in self._properties.items():
            if len(prop_values) != self._block_count:
                errors.append(f"Property '{prop_name}' length doesn't match block count")
        
        return errors
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert block model to pandas DataFrame.
        
        Returns:
            DataFrame with columns for positions, dimensions, and properties
        """
        if self._positions is None or self._dimensions is None:
            return pd.DataFrame()
        
        data = {
            'x': self._positions[:, 0],
            'y': self._positions[:, 1],
            'z': self._positions[:, 2],
            'dx': self._dimensions[:, 0],
            'dy': self._dimensions[:, 1],
            'dz': self._dimensions[:, 2],
        }
        
        # Add properties
        for prop_name, prop_values in self._properties.items():
            data[prop_name] = prop_values
        
        return pd.DataFrame(data)
    
    def update_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Update block model from pandas DataFrame (in-place modification).
        
        Useful for applying coordinate transformations or property updates.
        
        Args:
            df: DataFrame with required columns (XMORIG, YMORIG, ZMORIG or x, y, z or XC, YC, ZC) 
                and optional DX, DY, DZ columns
        """
        if df.empty:
            logger.warning("Cannot update from empty DataFrame")
            return
        
        # Update positions - try multiple column name formats
        positions_updated = False
        
        # Try centroid columns (XC, YC, ZC) first - most common in CSVs
        if all(col in df.columns for col in ['XC', 'YC', 'ZC']):
            # STEP 18: Optimize dtype to float32
            self._positions = df[['XC', 'YC', 'ZC']].values.astype(np.float32, copy=False)
            positions_updated = True
            logger.info("Updated positions from XC, YC, ZC columns")
        # Try origin columns (XMORIG, YMORIG, ZMORIG)
        elif all(col in df.columns for col in ['XMORIG', 'YMORIG', 'ZMORIG']):
            self._positions = df[['XMORIG', 'YMORIG', 'ZMORIG']].values.astype(np.float32, copy=False)
            # If dimensions available, convert origin to centroid for _positions
            if all(col in df.columns for col in ['DX', 'DY', 'DZ']):
                dims = df[['DX', 'DY', 'DZ']].values.astype(np.float32, copy=False)
                self._positions = self._positions + dims / 2
                logger.info("Updated positions from XMORIG/YMORIG/ZMORIG + DX/DY/DZ (calculated centroids)")
            else:
                logger.info("Updated positions from XMORIG, YMORIG, ZMORIG (origin)")
            positions_updated = True
        # Try lowercase (x, y, z)
        elif all(col in df.columns for col in ['x', 'y', 'z']):
            self._positions = df[['x', 'y', 'z']].values.astype(np.float32, copy=False)
            positions_updated = True
            logger.info("Updated positions from x, y, z columns")
        
        if not positions_updated:
            logger.warning("DataFrame must contain coordinate columns (XC/YC/ZC, XMORIG/YMORIG/ZMORIG, or x/y/z)")
            return
        
        # Update dimensions if present
        if all(col in df.columns for col in ['DX', 'DY', 'DZ']):
            self._dimensions = df[['DX', 'DY', 'DZ']].values.astype(np.float32, copy=False)
        elif all(col in df.columns for col in ['dx', 'dy', 'dz']):
            self._dimensions = df[['dx', 'dy', 'dz']].values.astype(np.float32, copy=False)
        elif all(col in df.columns for col in ['XINC', 'YINC', 'ZINC']):
            self._dimensions = df[['XINC', 'YINC', 'ZINC']].values.astype(np.float32, copy=False)
            logger.info("Updated dimensions from XINC, YINC, ZINC columns")
        
        # Update properties (skip coordinate and dimension columns)
        coord_cols = {'XMORIG', 'YMORIG', 'ZMORIG', 'x', 'y', 'z', 'XC', 'YC', 'ZC',
                     'DX', 'DY', 'DZ', 'dx', 'dy', 'dz', 'XINC', 'YINC', 'ZINC'}
        
        for col in df.columns:
            if col not in coord_cols:
                # STEP 18: Optimize property dtype
                values = df[col].values
                self._properties[col] = self._optimize_property_dtype(values)
        
        # Update block count and bounds
        self._block_count = len(df)
        
        # Clear orthogonal cache when geometry changes
        self._is_orthogonal_cache = None
        
        # Note: rotation_matrix is preserved (not cleared) when updating from DataFrame
        # If rotation information is in the DataFrame, it should be set explicitly via set_rotation_matrix()
        
        self._update_bounds()
        
        logger.info(f"✓ Updated block model from DataFrame: {self._block_count} blocks, {len(self._properties)} properties")
    
    def memory_usage(self) -> int:
        """
        Calculate approximate memory usage in bytes.
        
        Returns:
            Memory usage in bytes
        """
        total = 0
        
        # Positions
        if self._positions is not None:
            total += self._positions.nbytes
        
        # Dimensions
        if self._dimensions is not None:
            total += self._dimensions.nbytes
        
        # Properties
        for prop_values in self._properties.values():
            if prop_values is not None:
                total += prop_values.nbytes
        
        return total
    
    # =========================================================================
    # UNIFIED ENGINE API - Industry Standard Data Provider
    # =========================================================================
    
    def get_data_matrix(self, fields: List[str]) -> Dict[str, np.ndarray]:
        """
        Get multiple fields as numpy arrays for engine consumption.
        
        This is the standard method for engines to extract data from block models.
        Returns float64 arrays suitable for numerical computation.
        
        Args:
            fields: List of field names to extract (e.g., ['X', 'Y', 'Z', 'Fe'])
                   Coordinates can be specified as 'X', 'Y', 'Z' (case-insensitive)
        
        Returns:
            Dictionary mapping field name -> numpy array (float64)
        
        Raises:
            ValueError: If block model is empty or required field not found
            
        Example:
            >>> data = block_model.get_data_matrix(['X', 'Y', 'Z', 'Fe', 'Cu'])
            >>> coords = np.column_stack([data['X'], data['Y'], data['Z']])
            >>> grades = data['Fe']
        """
        if self._block_count == 0:
            raise ValueError("Cannot extract data from empty block model")
        
        result = {}
        
        for field in fields:
            field_upper = field.upper()
            
            # Handle coordinate fields
            if field_upper == 'X':
                if self._positions is None:
                    raise ValueError("Block positions not set")
                result[field] = self._positions[:, 0].astype(np.float64)
            elif field_upper == 'Y':
                if self._positions is None:
                    raise ValueError("Block positions not set")
                result[field] = self._positions[:, 1].astype(np.float64)
            elif field_upper == 'Z':
                if self._positions is None:
                    raise ValueError("Block positions not set")
                result[field] = self._positions[:, 2].astype(np.float64)
            else:
                # Property field - try exact match first, then case-insensitive
                if field in self._properties:
                    values = self._properties[field]
                else:
                    # Try case-insensitive match
                    found = False
                    for prop_name, prop_values in self._properties.items():
                        if prop_name.upper() == field_upper:
                            values = prop_values
                            found = True
                            break
                    if not found:
                        raise ValueError(f"Field '{field}' not found in block model properties. "
                                       f"Available: {list(self._properties.keys())}")
                
                # Convert to float64 for computation
                if not np.issubdtype(values.dtype, np.number):
                    raise ValueError(f"Field '{field}' is not numeric (dtype={values.dtype})")
                result[field] = values.astype(np.float64)
        
        return result
    
    def get_numpy_matrix(self, fields: List[str]) -> np.ndarray:
        """
        Get multiple fields as a single N×k float64 matrix.
        
        Args:
            fields: List of field names (e.g., ['X', 'Y', 'Z', 'Fe'])
        
        Returns:
            N×k numpy array (float64) where N=block_count, k=len(fields)
            
        Example:
            >>> coords_matrix = block_model.get_numpy_matrix(['X', 'Y', 'Z'])
            >>> # coords_matrix.shape = (N, 3)
        """
        data_dict = self.get_data_matrix(fields)
        columns = [data_dict[field] for field in fields]
        return np.column_stack(columns)
    
    def get_engine_payload(
        self,
        grade_field: Optional[str] = None,
        domain_field: Optional[str] = None,
        variance_field: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get standardized payload for geostatistical engines.
        
        This is the canonical method for engines (OK, UK, COK, SGSIM, IK, etc.)
        to extract all necessary data in a consistent format.
        
        Args:
            grade_field: Primary grade variable name
            domain_field: Optional domain/lithology field for domain kriging
            variance_field: Optional variance field for conditional simulation
        
        Returns:
            Dictionary with standardized engine inputs:
                - 'coords': N×3 float64 array (X, Y, Z)
                - 'grade': N float64 array (primary variable) if grade_field specified
                - 'domain': N array (domain codes) if domain_field specified
                - 'variance': N float64 array if variance_field specified
                - 'indices': N int64 array (block indices)
                - 'block_count': int
                - 'bounds': tuple (xmin, xmax, ymin, ymax, zmin, zmax)
                
        Example:
            >>> payload = block_model.get_engine_payload(grade_field='Fe', domain_field='LITH')
            >>> coords = payload['coords']  # Shape: (N, 3)
            >>> grade = payload['grade']    # Shape: (N,)
            >>> domain = payload['domain']  # Shape: (N,)
        """
        if self._block_count == 0:
            raise ValueError("Cannot create engine payload from empty block model")
        
        if self._positions is None:
            raise ValueError("Block positions not set")
        
        payload = {
            'coords': self._positions.astype(np.float64),  # N×3
            'indices': np.arange(self._block_count, dtype=np.int64),
            'block_count': self._block_count,
            'bounds': self._bounds
        }
        
        # Add grade if specified
        if grade_field is not None:
            data = self.get_data_matrix([grade_field])
            payload['grade'] = data[grade_field]
        
        # Add domain if specified
        if domain_field is not None:
            if domain_field in self._properties:
                payload['domain'] = self._properties[domain_field].copy()
            else:
                # Try case-insensitive match
                domain_field_upper = domain_field.upper()
                for prop_name, prop_values in self._properties.items():
                    if prop_name.upper() == domain_field_upper:
                        payload['domain'] = prop_values.copy()
                        break
                else:
                    logger.warning(f"Domain field '{domain_field}' not found, proceeding without domain filtering")
                    payload['domain'] = None
        
        # Add variance if specified
        if variance_field is not None:
            data = self.get_data_matrix([variance_field])
            payload['variance'] = data[variance_field]
        
        return payload
    
    def compact(self) -> None:
        """
        Compact memory usage by dropping caches and temporary arrays.
        
        This method can be called after heavy analysis sequences to free memory.
        """
        # Clear computed bounds (will be recomputed on next access)
        self._bounds = None
        
        # Note: We don't clear positions/dimensions/properties as they are core data
        # But we could optimize arrays in-place if needed
        
        logger.debug("Compacted block model memory")
    
    def to_npvs_blocks(self, value_field: str) -> List[Any]:
        """
        Convert internal block model to NpvsBlock list (STEP 32).
        
        Args:
            value_field: Name of the value property to use
        
        Returns:
            List of NpvsBlock instances
        """
        from ..mine_planning.npvs.npvs_data import NpvsBlock
        
        if self._block_count == 0:
            return []
        
        # Convert to DataFrame for easier access
        df = self.to_dataframe()
        
        blocks = []
        
        # Get value field
        if value_field not in df.columns:
            logger.warning(f"Value field '{value_field}' not found, using 0.0")
            value_values = np.zeros(self._block_count)
        else:
            value_values = df[value_field].values
        
        # Get tonnage (calculate if not present)
        if "tonnage" in df.columns:
            tonnage_values = df["tonnage"].values
        elif "TONNAGE" in df.columns:
            tonnage_values = df["TONNAGE"].values
        else:
            # Calculate from dimensions and density
            if self._dimensions is not None:
                volumes = np.prod(self._dimensions, axis=1)
                if "density" in df.columns:
                    density = df["density"].values
                elif "DENSITY" in df.columns:
                    density = df["DENSITY"].values
                else:
                    density = np.full(self._block_count, 2.7)
                tonnage_values = volumes * density
            else:
                tonnage_values = np.ones(self._block_count) * 1000.0  # Default
        
        # Get block IDs
        if "block_id" in df.columns:
            block_ids = df["block_id"].values
        elif "BLOCK_ID" in df.columns:
            block_ids = df["BLOCK_ID"].values
        else:
            block_ids = np.arange(self._block_count)
        
        # Get element columns (grade properties)
        exclude_cols = {'block_id', 'BLOCK_ID', 'tonnage', 'TONNAGE', 
                       'x', 'y', 'z', 'XC', 'YC', 'ZC', 'XMORIG', 'YMORIG', 'ZMORIG',
                       'DX', 'DY', 'DZ', 'dx', 'dy', 'dz', 'XINC', 'YINC', 'ZINC',
                       'density', 'DENSITY', value_field, 'precedence', 'PRECEDENCE',
                       'bench', 'BENCH', 'pit', 'PIT', 'phase', 'PHASE', 'ore_type', 'ORE_TYPE'}
        element_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Build precedence graph (if available)
        precedence_graph = {}
        if "precedence" in df.columns:
            prec_col = df["precedence"]
            for idx, block_id in enumerate(block_ids):
                prec_value = prec_col.iloc[idx]
                if pd.notna(prec_value) and prec_value:
                    # Parse precedence (could be comma-separated list of IDs)
                    if isinstance(prec_value, str):
                        parent_ids = [int(x.strip()) for x in prec_value.split(',') if x.strip().isdigit()]
                    elif isinstance(prec_value, (int, float)):
                        parent_ids = [int(prec_value)]
                    else:
                        parent_ids = []
                    precedence_graph[int(block_id)] = parent_ids
        elif "PRECEDENCE" in df.columns:
            prec_col = df["PRECEDENCE"]
            for idx, block_id in enumerate(block_ids):
                prec_value = prec_col.iloc[idx]
                if pd.notna(prec_value) and prec_value:
                    if isinstance(prec_value, str):
                        parent_ids = [int(x.strip()) for x in prec_value.split(',') if x.strip().isdigit()]
                    elif isinstance(prec_value, (int, float)):
                        parent_ids = [int(prec_value)]
                    else:
                        parent_ids = []
                    precedence_graph[int(block_id)] = parent_ids
        
        # Create NpvsBlock instances
        for i in range(self._block_count):
            block_id = int(block_ids[i])
            
            # Extract grades
            grade_by_element = {}
            for col in element_columns:
                val = df[col].iloc[i]
                if pd.notna(val) and isinstance(val, (int, float)):
                    grade_by_element[col] = float(val)
            
            # Get optional fields
            bench_val = None
            if "bench" in df.columns:
                bench_val = df["bench"].iloc[i] if pd.notna(df["bench"].iloc[i]) else None
            elif "BENCH" in df.columns:
                bench_val = df["BENCH"].iloc[i] if pd.notna(df["BENCH"].iloc[i]) else None
            
            pit_val = None
            if "pit" in df.columns:
                pit_val = df["pit"].iloc[i] if pd.notna(df["pit"].iloc[i]) else None
            elif "PIT" in df.columns:
                pit_val = df["PIT"].iloc[i] if pd.notna(df["PIT"].iloc[i]) else None
            
            phase_val = None
            if "phase" in df.columns:
                phase_val = df["phase"].iloc[i] if pd.notna(df["phase"].iloc[i]) else None
            elif "PHASE" in df.columns:
                phase_val = df["PHASE"].iloc[i] if pd.notna(df["PHASE"].iloc[i]) else None
            
            ore_type_val = None
            if "ore_type" in df.columns:
                ore_type_val = df["ore_type"].iloc[i] if pd.notna(df["ore_type"].iloc[i]) else None
            elif "ORE_TYPE" in df.columns:
                ore_type_val = df["ORE_TYPE"].iloc[i] if pd.notna(df["ORE_TYPE"].iloc[i]) else None
            
            blocks.append(NpvsBlock(
                id=block_id,
                tonnage=float(tonnage_values[i]),
                bench=str(bench_val) if bench_val is not None else None,
                pit=str(pit_val) if pit_val is not None else None,
                phase=str(phase_val) if phase_val is not None else None,
                ore_type=str(ore_type_val) if ore_type_val is not None else None,
                grade_by_element=grade_by_element,
                value_raw=float(value_values[i]) if i < len(value_values) else 0.0,
                precedence_parents=precedence_graph.get(block_id, [])
            ))
        
        logger.info(f"Converted {len(blocks)} blocks to NPVS format")
        return blocks
    
    def __repr__(self) -> str:
        """String representation of the block model."""
        if self._block_count == 0:
            return "BlockModel(empty)"
        
        bounds_str = f"bounds={self._bounds}" if self._bounds else "bounds=None"
        props_str = f"properties={list(self._properties.keys())}" if self._properties else "properties=[]"
        
        return f"BlockModel(blocks={self._block_count}, {bounds_str}, {props_str})"
    
    def set_rotation_matrix(self, rotation_matrix: Optional[np.ndarray]) -> None:
        """
        Set rotation matrix for the block model.
        
        Args:
            rotation_matrix: 3x3 numpy array representing rotation, or None for no rotation
        """
        if rotation_matrix is not None:
            rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
            if rotation_matrix.shape != (3, 3):
                raise ValueError("Rotation matrix must be 3x3")
            self._rotation_matrix = rotation_matrix
        else:
            self._rotation_matrix = None
        
        # Clear orthogonal cache when rotation changes
        self._is_orthogonal_cache = None
        
        logger.info(f"Set rotation matrix: {'identity' if rotation_matrix is None or np.allclose(rotation_matrix, np.eye(3)) else 'rotated'}")
    
    def is_orthogonal(self, tolerance: float = 1e-6) -> Tuple[bool, Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[int, int, int]]]]:
        """
        Check if block model is orthogonal (regular grid, axis-aligned, no rotation).
        
        CRITICAL MEMORY OPTIMIZATION: This detection enables ImageData (UniformGrid) usage,
        which uses near-zero geometry memory vs UnstructuredGrid's ~4GB for 10M blocks.
        
        Detection checks:
        1. No rotation (rotation_matrix is None or identity)
        2. Uniform block dimensions (all blocks have same dx, dy, dz)
        3. Regular grid spacing (unique X/Y/Z coordinates form uniform deltas)
        4. Complete grid alignment (positions align to grid pattern)
        
        Args:
            tolerance: Tolerance for floating point comparisons
            
        Returns:
            Tuple of (is_orthogonal: bool, grid_info: Optional[Tuple])
            If orthogonal, grid_info contains (origin, spacing, dimensions) for ImageData
            If not orthogonal, grid_info is None
        """
        # Use cache if available and geometry hasn't changed
        if self._is_orthogonal_cache is not None:
            return self._is_orthogonal_cache
        
        if self._positions is None or self._dimensions is None:
            result = (False, None)
            self._is_orthogonal_cache = result
            return result
        
        if len(self._positions) == 0:
            result = (False, None)
            self._is_orthogonal_cache = result
            return result
        
        # Check 1: No rotation (rotation_matrix is None or identity)
        if self._rotation_matrix is not None:
            if not np.allclose(self._rotation_matrix, np.eye(3), atol=tolerance):
                logger.debug("Block model has rotation - not orthogonal")
                result = (False, None)
                self._is_orthogonal_cache = result
                return result
        
        # Check 2: Uniform block dimensions (all blocks have same dx, dy, dz)
        unique_dims = np.unique(self._dimensions, axis=0)
        if len(unique_dims) > 1:
            # Check if dimensions are close enough (within tolerance)
            dim_std = np.std(self._dimensions, axis=0)
            if np.any(dim_std > tolerance):
                logger.debug(f"Non-uniform block dimensions detected (std: {dim_std})")
                result = (False, None)
                self._is_orthogonal_cache = result
                return result
        
        # Get uniform block size
        spacing = self._dimensions[0].copy()
        
        # Check 3: Regular grid spacing (unique X/Y/Z coordinates form uniform deltas)
        unique_x = np.unique(self._positions[:, 0])
        unique_y = np.unique(self._positions[:, 1])
        unique_z = np.unique(self._positions[:, 2])
        
        # Check if coordinates are evenly spaced
        if len(unique_x) > 1:
            x_diffs = np.diff(np.sort(unique_x))
            x_spacing = np.median(x_diffs)
            if np.any(np.abs(x_diffs - x_spacing) > tolerance):
                logger.debug(f"X coordinates not evenly spaced")
                result = (False, None)
                self._is_orthogonal_cache = result
                return result
        else:
            x_spacing = spacing[0]
        
        if len(unique_y) > 1:
            y_diffs = np.diff(np.sort(unique_y))
            y_spacing = np.median(y_diffs)
            if np.any(np.abs(y_diffs - y_spacing) > tolerance):
                logger.debug(f"Y coordinates not evenly spaced")
                result = (False, None)
                self._is_orthogonal_cache = result
                return result
        else:
            y_spacing = spacing[1]
        
        if len(unique_z) > 1:
            z_diffs = np.diff(np.sort(unique_z))
            z_spacing = np.median(z_diffs)
            if np.any(np.abs(z_diffs - z_spacing) > tolerance):
                logger.debug(f"Z coordinates not evenly spaced")
                result = (False, None)
                self._is_orthogonal_cache = result
                return result
        else:
            z_spacing = spacing[2]
        
        # Verify spacing matches block dimensions
        if (np.abs(x_spacing - spacing[0]) > tolerance or
            np.abs(y_spacing - spacing[1]) > tolerance or
            np.abs(z_spacing - spacing[2]) > tolerance):
            logger.debug(f"Spacing mismatch: expected {spacing}, got ({x_spacing}, {y_spacing}, {z_spacing})")
            result = (False, None)
            self._is_orthogonal_cache = result
            return result
        
        # Check 4: Complete grid alignment (positions align to grid pattern)
        nx, ny, nz = len(unique_x), len(unique_y), len(unique_z)
        expected_blocks = nx * ny * nz
        
        if len(self._positions) != expected_blocks:
            # Allow sparse grids (subset of full grid) but verify alignment
            logger.debug(f"Incomplete grid: expected {expected_blocks} blocks, got {len(self._positions)}")
            # Check if all positions align to the grid
            x_min_pos = self._positions[:, 0].min()
            y_min_pos = self._positions[:, 1].min()
            z_min_pos = self._positions[:, 2].min()
            
            # Sample check: verify positions align to grid
            sample_size = min(100, len(self._positions))
            sample_indices = np.random.choice(len(self._positions), sample_size, replace=False)
            for idx in sample_indices:
                pos = self._positions[idx]
                # Check if position aligns to grid
                x_offset = (pos[0] - x_min_pos) % x_spacing
                y_offset = (pos[1] - y_min_pos) % y_spacing
                z_offset = (pos[2] - z_min_pos) % z_spacing
                # Allow for small floating point errors
                if (x_offset > tolerance and abs(x_offset - x_spacing) > tolerance or
                    y_offset > tolerance and abs(y_offset - y_spacing) > tolerance or
                    z_offset > tolerance and abs(z_offset - z_spacing) > tolerance):
                    logger.debug(f"Position {pos} does not align to grid (offsets: {x_offset}, {y_offset}, {z_offset})")
                    result = (False, None)
                    self._is_orthogonal_cache = result
                    return result
        
        # All checks passed - this is an orthogonal grid
        # Calculate origin (bottom-left-back corner of first block)
        half_dims = spacing / 2.0
        origin = (
            float(self._positions[:, 0].min() - half_dims[0]),
            float(self._positions[:, 1].min() - half_dims[1]),
            float(self._positions[:, 2].min() - half_dims[2])
        )
        
        grid_dims = (nx, ny, nz)
        grid_info = (origin, tuple(spacing), grid_dims)
        
        logger.info(f"✅ Block model is orthogonal: origin={origin}, spacing={spacing}, dimensions={grid_dims}")
        result = (True, grid_info)
        self._is_orthogonal_cache = result
        return result
