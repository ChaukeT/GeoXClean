"""
Mining industry format parser for Datamine, Surpac, and similar formats.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
import re

from .base_parser import BaseParser
from ..models.block_model import BlockModel, BlockMetadata

logger = logging.getLogger(__name__)


class MiningParser(BaseParser):
    """
    Parser for mining industry block model formats.
    
    Supports Datamine, Surpac, and similar CSV-based formats with specific conventions.
    """
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.dm', '.dat', '.csv', '.txt']
        self.format_name = "Mining"
        
        # Mining-specific column patterns
        self.mining_patterns = {
            # Datamine format
            'datamine': {
                'x': ['x', 'xcentre', 'x_centre', 'xc'],
                'y': ['y', 'ycentre', 'y_centre', 'yc'],
                'z': ['z', 'zcentre', 'z_centre', 'zc'],
                'dx': ['dx', 'xsize', 'x_size'],
                'dy': ['dy', 'ysize', 'y_size'],
                'dz': ['dz', 'zsize', 'z_size']
            },
            # Surpac format
            'surpac': {
                'x': ['x', 'xcoord', 'x_coord'],
                'y': ['y', 'ycoord', 'y_coord'],
                'z': ['z', 'zcoord', 'z_coord'],
                'dx': ['dx', 'xsize', 'x_size'],
                'dy': ['dy', 'ysize', 'y_size'],
                'dz': ['dz', 'zsize', 'z_size']
            }
        }
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        if file_path.suffix.lower() not in self.supported_extensions:
            return False
        
        # Try to detect mining format by content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(10)]
            
            # Look for mining-specific keywords
            mining_keywords = ['datamine', 'surpac', 'block', 'grade', 'tonnage', 'volume']
            content = ' '.join(first_lines).lower()
            
            return any(keyword in content for keyword in mining_keywords)
        except:
            return False
    
    def parse(self, file_path: Path, **kwargs) -> BlockModel:
        """
        Parse a mining format file into a BlockModel object.
        
        Args:
            file_path: Path to the mining file
            **kwargs: Additional options:
                - format_type: Specific format ('datamine', 'surpac', 'auto')
                - delimiter: CSV delimiter (default: auto-detect)
                - header_row: Row number containing headers (default: 0)
                - skip_rows: Number of rows to skip at start
                - units: Units for coordinates ('meters', 'feet', 'unknown')
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Parse options
        format_type = kwargs.get('format_type', 'auto')
        delimiter = kwargs.get('delimiter', None)
        header_row = kwargs.get('header_row', 0)
        skip_rows = kwargs.get('skip_rows', 0)
        units = kwargs.get('units', 'unknown')
        
        try:
            # Detect format if auto
            if format_type == 'auto':
                format_type = self._detect_format(file_path)
            
            # Read file
            df = self._read_mining_file(file_path, delimiter, header_row, skip_rows)
            logger.info(f"Loaded mining file ({format_type}): {len(df)} rows")
            
            # Create metadata
            metadata = BlockMetadata(
                source_file=str(file_path),
                file_format=f"Mining ({format_type})",
                coordinate_system="unknown",
                units=units
            )
            
            block_model = BlockModel(metadata)
            
            # Extract geometry and properties
            self._extract_mining_data(block_model, df, format_type)
            
            logger.info(f"Parsed mining file: {block_model.block_count} blocks")
            return block_model
            
        except Exception as e:
            logger.error(f"Error parsing mining file {file_path}: {e}")
            raise ValueError(f"Failed to parse mining file: {e}")
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect mining format from file content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(20)]
            
            content = ' '.join(first_lines).lower()
            
            if 'datamine' in content or any('xcentre' in line for line in first_lines):
                return 'datamine'
            elif 'surpac' in content or any('xcoord' in line for line in first_lines):
                return 'surpac'
            else:
                return 'generic'
        except:
            return 'generic'
    
    def _read_mining_file(self, file_path: Path, delimiter: Optional[str], 
                         header_row: int, skip_rows: int) -> pd.DataFrame:
        """
        Read mining file with appropriate settings.
        
        ⚠️ PERFORMANCE OPTIMIZED: Uses nrows=5 for delimiter testing to prevent
        loading entire large files into RAM just to test delimiters.
        """
        # Try different delimiters if not specified
        if delimiter is None:
            delimiters = [',', '\t', ' ', ';']
        else:
            delimiters = [delimiter]
        
        # CRITICAL FIX: Test delimiters with only 5 rows to prevent OOM on large files
        # For a 2GB file, pd.read_csv() without nrows will load entire file into RAM
        # just to check if delimiter works - this causes crashes!
        test_rows = 5
        
        for delim in delimiters:
            try:
                # First, test delimiter with small sample
                df_test = pd.read_csv(
                    file_path,
                    delimiter=delim,
                    header=header_row,
                    skiprows=skip_rows,
                    encoding='utf-8',
                    on_bad_lines='skip',
                    nrows=test_rows  # CRITICAL: Only read 5 rows for testing
                )
                
                # Check if delimiter produces reasonable structure
                if len(df_test.columns) >= 3:
                    # Delimiter looks good, now read full file
                    logger.info(f"Delimiter '{delim}' validated, reading full file...")
                    df = pd.read_csv(
                        file_path,
                        delimiter=delim,
                        header=header_row,
                        skiprows=skip_rows,
                        encoding='utf-8',
                        on_bad_lines='skip'
                        # No nrows here - read full file now that delimiter is confirmed
                    )
                    
                    if len(df.columns) >= 3 and len(df) > 0:
                        logger.info(f"Successfully read file with delimiter '{delim}': {len(df)} rows")
                        return df
            except Exception as e:
                logger.debug(f"Delimiter '{delim}' failed: {e}")
                continue
        
        raise ValueError("Could not read file with any delimiter")
    
    def _extract_mining_data(self, block_model: BlockModel, df: pd.DataFrame, 
                           format_type: str) -> None:
        """Extract geometry and properties from mining data."""
        # Get column mapping for this format
        patterns = self.mining_patterns.get(format_type, self.mining_patterns['datamine'])
        
        # Find position columns
        pos_cols = self._find_columns(df, patterns, ['x', 'y', 'z'])
        if len(pos_cols) != 3:
            raise ValueError(f"Could not find all position columns. Found: {pos_cols}")
        
        # Find dimension columns
        dim_cols = self._find_columns(df, patterns, ['dx', 'dy', 'dz'])
        if len(dim_cols) != 3:
            logger.warning("Could not find all dimension columns, using defaults")
            # Create default dimensions
            df['dx_default'] = 1.0
            df['dy_default'] = 1.0
            df['dz_default'] = 1.0
            dim_cols = ['dx_default', 'dy_default', 'dz_default']
        
        # Extract geometry
        positions = df[pos_cols].values.astype(np.float64)
        dimensions = df[dim_cols].values.astype(np.float64)
        
        # Validate dimensions
        if np.any(dimensions <= 0):
            logger.warning("Found non-positive dimensions, setting to 1.0")
            dimensions[dimensions <= 0] = 1.0
        
        block_model.set_geometry(positions, dimensions)
        
        # Add all other columns as properties
        used_cols = set(pos_cols + dim_cols)
        property_cols = [col for col in df.columns if col not in used_cols]
        
        for prop_col in property_cols:
            values = df[prop_col].values
            
            # Convert to numeric if possible
            try:
                numeric_values = pd.to_numeric(values, errors='coerce')
                if not np.isnan(numeric_values).all():
                    values = numeric_values
            except:
                pass  # Keep as strings for categorical data
            
            block_model.add_property(prop_col, values)
        
        logger.info(f"Added {len(property_cols)} properties: {property_cols}")
    
    def _find_columns(self, df: pd.DataFrame, patterns: Dict[str, List[str]], 
                     keys: List[str]) -> List[str]:
        """Find columns matching patterns."""
        columns = df.columns.tolist()
        found_cols = []
        
        for key in keys:
            key_patterns = patterns.get(key, [])
            found_col = None
            
            for pattern in key_patterns:
                for col in columns:
                    if pattern.lower() in col.lower():
                        found_col = col
                        break
                if found_col:
                    break
            
            if found_col:
                found_cols.append(found_col)
        
        return found_cols
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get mining file information."""
        info = super().get_file_info(file_path)
        
        try:
            # Detect format
            format_type = self._detect_format(file_path)
            
            # Try to read first few rows
            df = self._read_mining_file(file_path, None, 0, 0)
            
            info.update({
                "format_type": format_type,
                "columns": df.columns.tolist(),
                "column_count": len(df.columns),
                "estimated_rows": len(df),
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
            })
        except Exception as e:
            logger.warning(f"Could not preview mining file: {e}")
            info.update({
                "format_type": "unknown",
                "columns": [],
                "column_count": 0,
                "estimated_rows": 0,
                "sample_data": []
            })
        
        return info
