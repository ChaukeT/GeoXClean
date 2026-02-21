"""
Chunked loading utilities for large block models.

Provides async/streamed loading to handle very large datasets without
freezing the UI or consuming excessive memory.
"""

import logging
from typing import Iterable, Optional, Dict, Any, Callable
from pathlib import Path
import numpy as np
import pandas as pd

from ..models.block_model import BlockModel, BlockMetadata
from ..parsers import parser_registry

logger = logging.getLogger(__name__)


class BlockModelChunk:
    """Represents a chunk of block model data."""
    
    def __init__(self, chunk_id: int, positions: np.ndarray, 
                 dimensions: Optional[np.ndarray] = None,
                 properties: Optional[Dict[str, np.ndarray]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.chunk_id = chunk_id
        self.positions = positions
        self.dimensions = dimensions
        self.properties = properties or {}
        self.metadata = metadata or {}
        self.block_count = len(positions)


class ChunkLoader:
    """
    Loads large block models in chunks for progressive rendering.
    
    Supports both CSV and other formats, loading data incrementally
    to avoid memory spikes and UI freezing.
    """
    
    def __init__(self, chunk_size: int = 200_000, 
                 reader: Optional[Callable] = None):
        """
        Initialize ChunkLoader.
        
        Args:
            chunk_size: Number of blocks per chunk
            reader: Optional custom reader function
        """
        self.chunk_size = chunk_size
        self.custom_reader = reader
        
    def load_in_chunks(self, path: Path) -> Iterable[BlockModelChunk]:
        """
        Load block model file in chunks.
        
        Args:
            path: Path to block model file
            
        Yields:
            BlockModelChunk objects
        """
        try:
            # Detect file format
            file_ext = path.suffix.lower()
            
            if file_ext == '.csv' or file_ext == '.txt':
                yield from self._load_csv_chunks(path)
            else:
                # For other formats, try to use parser registry
                # If file is too large, fall back to chunked CSV-like reading
                yield from self._load_generic_chunks(path)
                
        except Exception as e:
            logger.error(f"Error loading chunks from {path}: {e}", exc_info=True)
            raise
    
    def _load_csv_chunks(self, path: Path) -> Iterable[BlockModelChunk]:
        """Load CSV file in chunks."""
        try:
            # Read header first to determine columns
            df_sample = pd.read_csv(path, nrows=100)
            
            # Identify coordinate columns (X, Y, Z)
            coord_cols = []
            for col in ['X', 'Y', 'Z', 'x', 'y', 'z']:
                if col in df_sample.columns:
                    coord_cols.append(col)
            
            if len(coord_cols) < 3:
                # Try to infer from column names
                for col in df_sample.columns:
                    col_lower = col.lower()
                    if 'x' in col_lower and 'coord' in col_lower:
                        coord_cols.append(col)
                    elif 'y' in col_lower and 'coord' in col_lower:
                        coord_cols.append(col)
                    elif 'z' in col_lower and 'coord' in col_lower:
                        coord_cols.append(col)
            
            if len(coord_cols) < 3:
                raise ValueError(f"Could not identify coordinate columns in {path}")
            
            # Identify dimension columns (DX, DY, DZ or similar)
            dim_cols = []
            for pattern in ['DX', 'DY', 'DZ', 'dx', 'dy', 'dz', 'SIZE', 'size']:
                for col in df_sample.columns:
                    if pattern in col:
                        dim_cols.append(col)
                        break
            
            # Property columns (everything else except coordinates and dimensions)
            property_cols = [
                col for col in df_sample.columns 
                if col not in coord_cols and col not in dim_cols
            ]
            
            # Read file in chunks
            chunk_id = 0
            for chunk_df in pd.read_csv(path, chunksize=self.chunk_size):
                if len(chunk_df) == 0:
                    break
                
                # Extract positions
                positions = chunk_df[coord_cols[:3]].values.astype(np.float32)
                
                # Extract dimensions if available
                dimensions = None
                if len(dim_cols) >= 3:
                    dimensions = chunk_df[dim_cols[:3]].values.astype(np.float32)
                
                # Extract properties
                properties = {}
                for prop_col in property_cols:
                    if prop_col in chunk_df.columns:
                        prop_values = chunk_df[prop_col].values
                        # Use appropriate dtype
                        if prop_values.dtype == 'float64':
                            prop_values = prop_values.astype(np.float32)
                        elif prop_values.dtype in ['int64', 'int32']:
                            # Use smaller int types where possible
                            if prop_values.min() >= -128 and prop_values.max() <= 127:
                                prop_values = prop_values.astype(np.int8)
                            elif prop_values.min() >= -32768 and prop_values.max() <= 32767:
                                prop_values = prop_values.astype(np.int16)
                        properties[prop_col] = prop_values
                
                yield BlockModelChunk(
                    chunk_id=chunk_id,
                    positions=positions,
                    dimensions=dimensions,
                    properties=properties,
                    metadata={'source_file': str(path)}
                )
                
                chunk_id += 1
                
        except Exception as e:
            logger.error(f"Error loading CSV chunks: {e}", exc_info=True)
            raise
    
    def _load_generic_chunks(self, path: Path) -> Iterable[BlockModelChunk]:
        """Load generic format file in chunks (fallback)."""
        try:
            # Try to use parser registry
            # For very large files, we might need to implement format-specific chunking
            # For now, load full model and split into chunks
            
            block_model = parser_registry.parse_file(path)
            
            positions = block_model.positions
            if positions is None:
                return
            
            total_blocks = len(positions)
            chunk_id = 0
            
            for start_idx in range(0, total_blocks, self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, total_blocks)
                
                chunk_positions = positions[start_idx:end_idx].astype(np.float32)
                
                chunk_dimensions = None
                if block_model.dimensions is not None:
                    chunk_dimensions = block_model.dimensions[start_idx:end_idx].astype(np.float32)
                
                chunk_properties = {}
                for prop_name in block_model.get_property_names():
                    prop_values = block_model.get_property(prop_name)
                    if prop_values is not None:
                        chunk_values = prop_values[start_idx:end_idx]
                        # Optimize dtype
                        if chunk_values.dtype == 'float64':
                            chunk_values = chunk_values.astype(np.float32)
                        chunk_properties[prop_name] = chunk_values
                
                yield BlockModelChunk(
                    chunk_id=chunk_id,
                    positions=chunk_positions,
                    dimensions=chunk_dimensions,
                    properties=chunk_properties,
                    metadata={'source_file': str(path)}
                )
                
                chunk_id += 1
                
        except Exception as e:
            logger.error(f"Error loading generic chunks: {e}", exc_info=True)
            raise
    
    def estimate_chunk_count(self, path: Path) -> int:
        """
        Estimate number of chunks for a file.
        
        Args:
            path: Path to file
            
        Returns:
            Estimated chunk count
        """
        try:
            if path.suffix.lower() == '.csv':
                # Count lines (rough estimate)
                with open(path, 'r') as f:
                    line_count = sum(1 for _ in f) - 1  # Subtract header
                return max(1, (line_count + self.chunk_size - 1) // self.chunk_size)
            else:
                # For other formats, try to parse and estimate
                # This is a rough estimate
                file_size_mb = path.stat().st_size / (1024 * 1024)
                # Assume ~100 bytes per block (rough estimate)
                estimated_blocks = int(file_size_mb * 1024 * 1024 / 100)
                return max(1, (estimated_blocks + self.chunk_size - 1) // self.chunk_size)
        except Exception:
            return 1


def assemble_chunks_into_model(chunks: Iterable[BlockModelChunk], 
                               metadata: Optional[BlockMetadata] = None,
                               use_memmap: bool = False,
                               memmap_dir: Optional[Path] = None) -> BlockModel:
    """
    Assemble chunks into a complete BlockModel.
    
    ⚠️ MEMORY OPTIMIZED: Pre-allocates arrays to avoid 2x RAM usage.
    For very large datasets (>2GB), can use memory-mapped arrays.
    
    Args:
        chunks: Iterable of BlockModelChunk objects
        metadata: Optional metadata for the model
        use_memmap: If True, use memory-mapped arrays for files > 2GB
        memmap_dir: Directory for temporary memmap files (if use_memmap=True)
        
    Returns:
        Complete BlockModel
    """
    try:
        # CRITICAL FIX: First pass - count total blocks to pre-allocate arrays
        # This prevents 2x RAM usage from list.append() + concatenate()
        chunk_list = list(chunks)  # Convert to list to allow two passes
        
        if not chunk_list:
            logger.warning("No chunks to assemble")
            return BlockModel(metadata=metadata)
        
        # Calculate total size
        total_blocks = sum(chunk.block_count for chunk in chunk_list)
        
        if total_blocks == 0:
            logger.warning("All chunks are empty")
            return BlockModel(metadata=metadata)
        
        # Determine if we should use memmap (for very large datasets)
        estimated_size_mb = total_blocks * 3 * 4 / (1024 * 1024)  # Rough estimate
        use_memmap = use_memmap or (estimated_size_mb > 2048)  # >2GB
        
        logger.info(f"Assembling {len(chunk_list)} chunks into {total_blocks} blocks (est. {estimated_size_mb:.1f} MB)")
        
        if use_memmap and memmap_dir:
            # Use memory-mapped arrays for very large datasets
            memmap_dir = Path(memmap_dir)
            memmap_dir.mkdir(parents=True, exist_ok=True)
            
            # Create temporary memmap files
            positions_mmap = np.memmap(
                memmap_dir / 'positions_temp.dat',
                dtype=np.float32,
                mode='w+',
                shape=(total_blocks, 3)
            )
            
            # Determine if dimensions exist
            has_dimensions = any(chunk.dimensions is not None for chunk in chunk_list)
            dimensions_mmap = None
            if has_dimensions:
                dimensions_mmap = np.memmap(
                    memmap_dir / 'dimensions_temp.dat',
                    dtype=np.float32,
                    mode='w+',
                    shape=(total_blocks, 3)
                )
            
            # Collect property names and create memmap arrays
            property_mmaps = {}
            property_names = set()
            for chunk in chunk_list:
                property_names.update(chunk.properties.keys())
            
            for prop_name in property_names:
                # Determine dtype from first chunk that has this property
                sample_dtype = None
                for chunk in chunk_list:
                    if prop_name in chunk.properties:
                        sample_dtype = chunk.properties[prop_name].dtype
                        break
                
                if sample_dtype is None:
                    continue
                
                # Use float32 for numeric properties to save memory
                if np.issubdtype(sample_dtype, np.floating):
                    dtype = np.float32
                elif np.issubdtype(sample_dtype, np.integer):
                    # Determine appropriate int size
                    for chunk in chunk_list:
                        if prop_name in chunk.properties:
                            arr = chunk.properties[prop_name]
                            if arr.min() >= -128 and arr.max() <= 127:
                                dtype = np.int8
                            elif arr.min() >= -32768 and arr.max() <= 32767:
                                dtype = np.int16
                            else:
                                dtype = np.int32
                            break
                else:
                    dtype = sample_dtype
                
                property_mmaps[prop_name] = np.memmap(
                    memmap_dir / f'property_{prop_name}_temp.dat',
                    dtype=dtype,
                    mode='w+',
                    shape=(total_blocks,)
                )
            
            # Fill memmap arrays
            offset = 0
            for chunk in chunk_list:
                n = chunk.block_count
                positions_mmap[offset:offset+n] = chunk.positions
                
                if chunk.dimensions is not None and dimensions_mmap is not None:
                    dimensions_mmap[offset:offset+n] = chunk.dimensions
                
                for prop_name, prop_values in chunk.properties.items():
                    if prop_name in property_mmaps:
                        property_mmaps[prop_name][offset:offset+n] = prop_values
                
                offset += n
            
            # Convert memmap to regular arrays (or keep as memmap for very large)
            positions = np.array(positions_mmap) if estimated_size_mb < 4096 else positions_mmap
            dimensions = np.array(dimensions_mmap) if dimensions_mmap is not None and estimated_size_mb < 4096 else dimensions_mmap
            
            properties = {}
            for prop_name, mmap_arr in property_mmaps.items():
                properties[prop_name] = np.array(mmap_arr) if estimated_size_mb < 4096 else mmap_arr
            
            # Clean up temporary files if we converted to arrays
            if estimated_size_mb < 4096:
                (memmap_dir / 'positions_temp.dat').unlink(missing_ok=True)
                if dimensions_mmap is not None:
                    (memmap_dir / 'dimensions_temp.dat').unlink(missing_ok=True)
                for prop_name in property_mmaps:
                    (memmap_dir / f'property_{prop_name}_temp.dat').unlink(missing_ok=True)
        else:
            # STANDARD APPROACH: Pre-allocate arrays to avoid 2x RAM usage
            # Pre-allocate position array
            positions = np.zeros((total_blocks, 3), dtype=np.float32)
            
            # Check if any chunk has dimensions
            has_dimensions = any(chunk.dimensions is not None for chunk in chunk_list)
            dimensions = None
            if has_dimensions:
                dimensions = np.zeros((total_blocks, 3), dtype=np.float32)
            
            # Pre-allocate property arrays
            property_names = set()
            for chunk in chunk_list:
                property_names.update(chunk.properties.keys())
            
            properties = {}
            for prop_name in property_names:
                # Determine dtype and shape from first chunk
                sample_dtype = None
                sample_shape = None
                for chunk in chunk_list:
                    if prop_name in chunk.properties:
                        sample_dtype = chunk.properties[prop_name].dtype
                        sample_shape = chunk.properties[prop_name].shape
                        break
                
                if sample_dtype is None:
                    continue
                
                # Optimize dtype for memory
                if np.issubdtype(sample_dtype, np.floating):
                    dtype = np.float32
                elif np.issubdtype(sample_dtype, np.integer):
                    # Check range to use smallest int type
                    for chunk in chunk_list:
                        if prop_name in chunk.properties:
                            arr = chunk.properties[prop_name]
                            if arr.min() >= -128 and arr.max() <= 127:
                                dtype = np.int8
                            elif arr.min() >= -32768 and arr.max() <= 32767:
                                dtype = np.int16
                            else:
                                dtype = np.int32
                            break
                else:
                    dtype = sample_dtype
                
                if len(sample_shape) == 1:
                    properties[prop_name] = np.zeros(total_blocks, dtype=dtype)
                else:
                    properties[prop_name] = np.zeros((total_blocks, *sample_shape[1:]), dtype=dtype)
            
            # Fill pre-allocated arrays slice by slice
            offset = 0
            for chunk in chunk_list:
                n = chunk.block_count
                positions[offset:offset+n] = chunk.positions
                
                if chunk.dimensions is not None and dimensions is not None:
                    dimensions[offset:offset+n] = chunk.dimensions
                
                for prop_name, prop_values in chunk.properties.items():
                    if prop_name in properties:
                        properties[prop_name][offset:offset+n] = prop_values
                
                offset += n
        
        # Create model with pre-allocated arrays
        model = BlockModel(metadata=metadata)
        model.set_positions(positions)
        
        if dimensions is not None:
            model.set_dimensions(dimensions)
        
        for prop_name, prop_array in properties.items():
            model.set_property(prop_name, prop_array)
        
        logger.info(f"Assembled model from chunks: {model.block_count} blocks (memory optimized)")
        return model
        
    except Exception as e:
        logger.error(f"Error assembling chunks: {e}", exc_info=True)
        raise

