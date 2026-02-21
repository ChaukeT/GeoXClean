"""
CSV parser for block model data.

GeoX Invariant Compliance:
- Parser versioning for reproducibility
- File checksums for data integrity
- Column detection audit trail
- Dimension inference tracking
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging

from .base_parser import BaseParser, compute_file_checksum, PARSER_FRAMEWORK_VERSION
from ..models.block_model import BlockModel, BlockMetadata

logger = logging.getLogger(__name__)

# CSV Parser version for reproducibility
CSV_PARSER_VERSION = "1.0.0"


class AmbiguousColumnsError(ValueError):
    """
    Exception raised when multiple column candidates are found.
    
    This prompts the UI to show a column mapping dialog instead of silently guessing.
    """
    pass


class CSVParser(BaseParser):
    """
    Parser for CSV/text format block model files.
    
    Supports automatic column detection for common block model formats.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.csv', '.txt', '.dat']
        self.format_name = "CSV"
        
        # Common column name patterns for automatic detection
        self.position_patterns = {
            'x': ['x', 'x_center', 'x_centre', 'xc', 'xpos', 'x_coord'],
            'y': ['y', 'y_center', 'y_centre', 'yc', 'ypos', 'y_coord'],
            'z': ['z', 'z_center', 'z_centre', 'zc', 'zpos', 'z_coord']
        }
        
        self.dimension_patterns = {
            'dx': ['dx', 'dx_size', 'size_x', 'block_size_x', 'width', 'xsize'],
            'dy': ['dy', 'dy_size', 'size_y', 'block_size_y', 'length', 'ysize'],
            'dz': ['dz', 'dz_size', 'size_z', 'block_size_z', 'height', 'zsize']
        }
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def parse(self, file_path: Path, **kwargs) -> BlockModel:
        """
        Parse a CSV file into a BlockModel object.
        
        GeoX Invariant Compliance:
        - File checksum computed for data integrity
        - Column detection recorded in metadata
        - Dimension inference tracked
        - Import timestamp recorded
        
        Args:
            file_path: Path to the CSV file
            **kwargs: Additional options:
                - delimiter: CSV delimiter (default: auto-detect)
                - header_row: Row number containing headers (default: 0)
                - encoding: File encoding (default: utf-8)
                - x_col, y_col, z_col: Explicit column names for positions
                - dx_col, dy_col, dz_col: Explicit column names for dimensions
                - skip_rows: Number of rows to skip at start
                - max_rows: Maximum number of rows to read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # GeoX invariant: Compute file checksum for data integrity
        file_checksum = compute_file_checksum(file_path)
        import_timestamp = datetime.now().isoformat()
        
        # Parse options
        delimiter = kwargs.get('delimiter', None)
        header_row = kwargs.get('header_row', 0)
        encoding = kwargs.get('encoding', 'utf-8')
        skip_rows = kwargs.get('skip_rows', 0)
        max_rows = kwargs.get('max_rows', None)
        
        # Explicit column mappings
        explicit_cols = {
            'x': kwargs.get('x_col'),
            'y': kwargs.get('y_col'),
            'z': kwargs.get('z_col'),
            'dx': kwargs.get('dx_col'),
            'dy': kwargs.get('dy_col'),
            'dz': kwargs.get('dz_col')
        }
        
        try:
            # Read CSV file
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                header=header_row,
                encoding=encoding,
                skiprows=skip_rows,
                nrows=max_rows
            )
            
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # GeoX invariant: Track original columns for audit
            original_columns = list(df.columns)
            
            # Detect or map columns
            column_mapping = self._detect_columns(df, explicit_cols)
            
            # Extract geometry data
            positions = df[column_mapping['positions']].values.astype(np.float64)
            
            # Handle dimensions - some may be None (missing)
            dim_cols = column_mapping['dimensions']
            dimensions_inferred = False
            inferred_dimensions_info = {}
            
            if None in dim_cols:
                # Some dimension columns are missing - infer from positions
                logger.info("Inferring block dimensions from coordinate spacing")
                dimensions = self._infer_dimensions(df, column_mapping['positions'])
                dimensions_inferred = True
                # Record what was inferred
                inferred_dimensions_info = {
                    'dx': float(dimensions[0, 0]) if len(dimensions) > 0 else None,
                    'dy': float(dimensions[0, 1]) if len(dimensions) > 0 else None,
                    'dz': float(dimensions[0, 2]) if len(dimensions) > 0 else None
                }
            else:
                dimensions = df[dim_cols].values.astype(np.float64)
            
            # Build column mapping dict for metadata (GeoX invariant: no silent transformations)
            detected_mapping = {
                'position_columns': column_mapping['positions'],
                'dimension_columns': [c for c in column_mapping['dimensions'] if c is not None],
                'explicit_mappings': {k: v for k, v in explicit_cols.items() if v is not None}
            }
            
            # Create block model with comprehensive metadata
            metadata = BlockMetadata(
                source_file=str(file_path),
                file_format="CSV",
                coordinate_system="unknown",
                units="unknown",
                # GeoX invariant: Provenance fields
                file_checksum=file_checksum,
                checksum_algorithm="sha256",
                import_timestamp=import_timestamp,
                parser_version=CSV_PARSER_VERSION,
                parser_framework_version=PARSER_FRAMEWORK_VERSION,
                column_mapping=detected_mapping,
                inferred_dimensions=dimensions_inferred
            )
            
            block_model = BlockModel(metadata)
            block_model.set_geometry(positions, dimensions)
            
            # Add properties (all columns except geometry)
            property_columns = [col for col in df.columns 
                              if col not in column_mapping['positions'] + column_mapping['dimensions']]
            
            # Track type conversions (GeoX invariant: no silent transformations)
            type_conversions = {}
            for prop_col in property_columns:
                values = df[prop_col].values
                original_dtype = str(values.dtype)
                # Convert to numeric if possible, otherwise keep as strings
                try:
                    converted_values = pd.to_numeric(values, errors='coerce')
                    # Check if conversion introduced NaNs (silent transformation)
                    original_nan_count = pd.isna(values).sum()
                    converted_nan_count = pd.isna(converted_values).sum()
                    if converted_nan_count > original_nan_count:
                        type_conversions[prop_col] = {
                            'original_dtype': original_dtype,
                            'converted_dtype': 'float64',
                            'values_converted_to_nan': int(converted_nan_count - original_nan_count)
                        }
                    values = converted_values
                except:
                    pass  # Keep as strings for categorical data
                
                block_model.add_property(prop_col, values)
            
            # Log type conversions if any
            if type_conversions:
                logger.info(f"Type conversions applied: {type_conversions}")
            
            # Log dimension inference if applicable
            if dimensions_inferred:
                logger.info(f"Dimensions inferred from data: {inferred_dimensions_info}")
            
            logger.info(f"Parsed CSV: {block_model.block_count} blocks, {len(property_columns)} properties")
            return block_model
            
        except Exception as e:
            logger.error(f"Error parsing CSV file {file_path}: {e}")
            raise ValueError(f"Failed to parse CSV file: {e}")
    
    def _detect_columns(self, df: pd.DataFrame, explicit_cols: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
        """
        Detect column mappings for positions and dimensions.
        
        Args:
            df: DataFrame to analyze
            explicit_cols: Explicit column mappings from user
            
        Returns:
            Dictionary with 'positions' and 'dimensions' column lists
        """
        columns = df.columns.tolist()
        columns_lower = {col.lower(): col for col in columns}
        column_mapping = {'positions': [], 'dimensions': []}
        
        # Use explicit mappings if provided
        if all(explicit_cols.values()):
            column_mapping['positions'] = [explicit_cols['x'], explicit_cols['y'], explicit_cols['z']]
            column_mapping['dimensions'] = [explicit_cols['dx'], explicit_cols['dy'], explicit_cols['dz']]
            return column_mapping
        
        # STEP 1: Detect dimension columns FIRST (they're more specific)
        # Common dimension/increment patterns in mining block models
        dimension_exact_patterns = {
            'dx': ['xinc', 'dx', 'dx_size', 'size_x', 'block_size_x', 'xsize', 'dimx'],
            'dy': ['yinc', 'dy', 'dy_size', 'size_y', 'block_size_y', 'ysize', 'dimy'],
            'dz': ['zinc', 'dz', 'dz_size', 'size_z', 'block_size_z', 'zsize', 'dimz']
        }
        
        dimension_columns = set()  # Track which columns are used for dimensions
        
        for dim in ['dx', 'dy', 'dz']:
            found_col = None
            
            # Try exact match first (case-insensitive)
            for pattern in dimension_exact_patterns.get(dim, []):
                if pattern.lower() in columns_lower:
                    found_col = columns_lower[pattern.lower()]
                    break
            
            # If no exact match, try patterns from instance variable
            if found_col is None:
                for pattern in self.dimension_patterns.get(dim, []):
                    for col in columns:
                        # Use exact match, not substring
                        if pattern.lower() == col.lower():
                            found_col = col
                            break
                    if found_col:
                        break
            
            if found_col:
                column_mapping['dimensions'].append(found_col)
                dimension_columns.add(found_col.lower())
                logger.info(f"Detected dimension column: {dim} -> {found_col}")
            else:
                # Dimension column not found - will handle later
                logger.debug(f"No dimension column found for {dim}")
        
        # STEP 2: Detect position columns (excluding already-detected dimension columns)
        # Priority: exact match > specific patterns > generic patterns
        position_exact_patterns = {
            'x': ['xc', 'x_center', 'x_centre', 'xpos', 'x_coord', 'x_centroid', 'xcent', 'x'],
            'y': ['yc', 'y_center', 'y_centre', 'ypos', 'y_coord', 'y_centroid', 'ycent', 'y'],
            'z': ['zc', 'z_center', 'z_centre', 'zpos', 'z_coord', 'z_centroid', 'zcent', 'z']
        }
        
        for coord in ['x', 'y', 'z']:
            found_col = None
            
            # Try exact match first (case-insensitive), excluding dimension columns
            for pattern in position_exact_patterns.get(coord, []):
                if pattern.lower() in columns_lower:
                    candidate = columns_lower[pattern.lower()]
                    # Skip if already used as dimension column
                    if candidate.lower() not in dimension_columns:
                        found_col = candidate
                        break
            
            if found_col:
                column_mapping['positions'].append(found_col)
                logger.info(f"Detected position column: {coord} -> {found_col}")
            else:
                # No exact match found - raise error with helpful message
                raise ValueError(
                    f"Could not detect {coord} coordinate column. "
                    f"Expected column names like: {position_exact_patterns[coord][:3]}. "
                    f"Available columns: {columns}"
                )
        
        # STEP 3: Handle missing dimension columns (use uniform block size)
        if len(column_mapping['dimensions']) < 3:
            logger.warning(f"Only found {len(column_mapping['dimensions'])} dimension columns. "
                          f"Missing dimensions will be inferred from data.")
            # Fill in missing dimensions with None (will be handled later)
            while len(column_mapping['dimensions']) < 3:
                column_mapping['dimensions'].append(None)
        
        return column_mapping
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get CSV file information."""
        info = super().get_file_info(file_path)
        
        try:
            # Try to read first few rows to get column info
            df_preview = pd.read_csv(file_path, nrows=5)
            info.update({
                "columns": df_preview.columns.tolist(),
                "column_count": len(df_preview.columns),
                "estimated_rows": self._estimate_row_count(file_path)
            })
        except Exception as e:
            logger.warning(f"Could not preview CSV file: {e}")
            info["columns"] = []
            info["column_count"] = 0
            info["estimated_rows"] = 0
        
        return info
    
    def _estimate_row_count(self, file_path: Path) -> int:
        """Estimate number of rows in CSV file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f) - 1  # Subtract header row
        except:
            return 0
    
    def _infer_dimensions(self, df: pd.DataFrame, position_cols: List[str]) -> np.ndarray:
        """
        Infer block dimensions from coordinate spacing.
        
        GeoX invariant: Logs inference details for auditability.
        
        Assumes regular grid with uniform block sizes.
        
        Args:
            df: DataFrame with position data
            position_cols: List of position column names [x, y, z]
            
        Returns:
            (N, 3) array of block dimensions
        """
        n_blocks = len(df)
        dimensions = np.zeros((n_blocks, 3))
        dim_names = ['dx', 'dy', 'dz']
        
        for i, col in enumerate(position_cols):
            coords = df[col].values
            unique_coords = np.sort(np.unique(coords))
            
            if len(unique_coords) > 1:
                # Calculate spacing between unique coordinates
                spacings = np.diff(unique_coords)
                # Use median spacing as block dimension
                block_size = np.median(spacings)
                inference_method = "median_spacing"
            else:
                # Only one unique coordinate - assume 1 unit (GeoX invariant: document default)
                block_size = 1.0
                inference_method = "default_single_coordinate"
                logger.warning(
                    f"Only one unique {col} coordinate found. Using default block size of 1.0. "
                    "This may not be correct for your data."
                )
            
            dimensions[:, i] = block_size
            logger.info(
                f"Inferred {dim_names[i]} = {block_size:.3f} from coordinate spacing "
                f"(method: {inference_method}, unique_coords: {len(unique_coords)})"
            )
        
        return dimensions
