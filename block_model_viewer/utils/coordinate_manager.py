"""
Coordinate System Manager for aligning block models, drillholes, and other spatial data.

Automatically detects coordinate offsets and aligns all datasets to a common reference frame.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CoordinateBounds:
    """Bounding box for a dataset in 3D space."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    
    @property
    def center(self) -> Tuple[float, float, float]:
        """Return the center point of the bounds."""
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.zmax) / 2
        )
    
    @property
    def range_x(self) -> float:
        return self.xmax - self.xmin
    
    @property
    def range_y(self) -> float:
        return self.ymax - self.ymin
    
    @property
    def range_z(self) -> float:
        return self.zmax - self.zmin
    
    def __repr__(self) -> str:
        return (f"CoordinateBounds(X: [{self.xmin:.2f}, {self.xmax:.2f}], "
                f"Y: [{self.ymin:.2f}, {self.ymax:.2f}], "
                f"Z: [{self.zmin:.2f}, {self.zmax:.2f}])")


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    name: str
    bounds: CoordinateBounds
    point_count: int
    data_type: str  # 'block_model', 'drillhole', 'mesh', etc.
    is_aligned: bool = False
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)


class CoordinateManager:
    """
    Manages coordinate systems and alignment for all spatial datasets.
    
    Automatically detects coordinate mismatches and applies transformations
    to align all data to a common reference frame.
    """
    
    def __init__(self):
        self.datasets: Dict[str, DatasetInfo] = {}
        self.reference_dataset: Optional[str] = None
        self.global_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        
    def register_dataset(self, name: str, data: Any, data_type: str) -> DatasetInfo:
        """
        Register a new dataset and compute its bounds.
        
        Args:
            name: Unique name for the dataset
            data: The dataset (BlockModel, DataFrame, numpy array, etc.)
            data_type: Type of data ('block_model', 'drillhole', 'mesh')
        
        Returns:
            DatasetInfo object with bounds and metadata
        """
        try:
            # Extract coordinates based on data type
            coords = self._extract_coordinates(data, data_type)
            
            if coords is None or len(coords) == 0:
                raise ValueError(f"No coordinates found in dataset '{name}'")
            
            # Compute bounds
            bounds = CoordinateBounds(
                xmin=float(np.min(coords[:, 0])),
                xmax=float(np.max(coords[:, 0])),
                ymin=float(np.min(coords[:, 1])),
                ymax=float(np.max(coords[:, 1])),
                zmin=float(np.min(coords[:, 2])),
                zmax=float(np.max(coords[:, 2]))
            )
            
            # Create dataset info
            dataset_info = DatasetInfo(
                name=name,
                bounds=bounds,
                point_count=len(coords),
                data_type=data_type
            )
            
            self.datasets[name] = dataset_info
            
            # Set as reference if it's the first dataset
            if self.reference_dataset is None:
                self.reference_dataset = name
                dataset_info.is_aligned = True
                # Avoid Unicode symbols that may not render in some Windows consoles
                logger.info(f"Set '{name}' as coordinate reference frame")
                logger.info(f"  Reference bounds: {bounds}")
            else:
                # Check alignment with reference
                self._check_alignment(dataset_info)
            
            # Avoid Unicode symbols that may not render in some Windows consoles
            logger.info(f"Registered dataset '{name}' ({data_type}): {dataset_info.point_count} points")
            logger.info(f"  Bounds: {bounds}")
            
            return dataset_info
            
        except Exception as e:
            logger.error(f"Error registering dataset '{name}': {e}", exc_info=True)
            raise
    
    def _extract_coordinates(self, data: Any, data_type: str) -> Optional[np.ndarray]:
        """Extract (N, 3) coordinate array from various data types."""
        try:
            if data_type == 'block_model':
                # BlockModel object - try direct positions access first (most reliable)
                if hasattr(data, 'positions') and data.positions is not None:
                    logger.info(f"Extracting block model coordinates from internal positions attribute")
                    return data.positions
                
                # Fallback to DataFrame extraction
                elif hasattr(data, 'to_dataframe'):
                    df = data.to_dataframe()
                    
                    # Try lowercase variants (x, y, z) - standard to_dataframe() output
                    if all(col in df.columns for col in ['x', 'y', 'z']):
                        logger.info(f"Extracting block model coordinates from x, y, z columns")
                        return df[['x', 'y', 'z']].values
                    # Try centroid columns (XC, YC, ZC) - from properties
                    elif all(col in df.columns for col in ['XC', 'YC', 'ZC']):
                        logger.info(f"Extracting block model coordinates from XC, YC, ZC columns")
                        return df[['XC', 'YC', 'ZC']].values
                    # Try origin columns (XMORIG, YMORIG, ZMORIG)
                    elif all(col in df.columns for col in ['XMORIG', 'YMORIG', 'ZMORIG']):
                        coords = df[['XMORIG', 'YMORIG', 'ZMORIG']].values
                        # Calculate centroids if dimensions available
                        if all(col in df.columns for col in ['DX', 'DY', 'DZ']):
                            dims = df[['DX', 'DY', 'DZ']].values
                            coords = coords + dims / 2
                            logger.info(f"Calculated block model centroids from XMORIG/YMORIG/ZMORIG + DX/DY/DZ")
                        else:
                            logger.info(f"Using block model origin coordinates XMORIG, YMORIG, ZMORIG")
                        return coords
                    else:
                        logger.warning(f"Block model missing coordinate columns. Available: {df.columns.tolist()}")
                        return None
                        
            elif data_type == 'drillhole':
                # DataFrame with X, Y, Z or collar data
                if isinstance(data, pd.DataFrame):
                    if all(col in data.columns for col in ['X', 'Y', 'Z']):
                        return data[['X', 'Y', 'Z']].values
                    elif all(col in data.columns for col in ['EAST', 'NORTH', 'ELEV']):
                        return data[['EAST', 'NORTH', 'ELEV']].values
                        
            elif data_type == 'mesh':
                # PyVista mesh or numpy array
                if hasattr(data, 'points'):
                    return np.array(data.points)
                elif isinstance(data, np.ndarray) and data.shape[1] == 3:
                    return data
            
            # Generic fallback for DataFrame
            if isinstance(data, pd.DataFrame):
                # Try common coordinate column combinations
                for col_set in [['X', 'Y', 'Z'], ['EAST', 'NORTH', 'ELEV'], 
                               ['XC', 'YC', 'ZC'], ['x', 'y', 'z']]:
                    if all(col in data.columns for col in col_set):
                        return data[col_set].values
            
            # NumPy array
            if isinstance(data, np.ndarray) and len(data.shape) == 2 and data.shape[1] >= 3:
                return data[:, :3]
            
            logger.warning(f"Could not extract coordinates from {type(data)}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting coordinates: {e}", exc_info=True)
            return None
    
    def _check_alignment(self, dataset_info: DatasetInfo) -> None:
        """
        Check if dataset is aligned with reference frame.
        Detect large coordinate offsets that indicate different coordinate systems.
        """
        if self.reference_dataset is None:
            return
        
        ref_info = self.datasets[self.reference_dataset]
        ref_bounds = ref_info.bounds
        new_bounds = dataset_info.bounds
        
        # Calculate center offset
        ref_center = ref_bounds.center
        new_center = new_bounds.center
        
        offset_x = abs(ref_center[0] - new_center[0])
        offset_y = abs(ref_center[1] - new_center[1])
        offset_z = abs(ref_center[2] - new_center[2])
        
        # Check if offset is significant relative to data extents
        # Threshold: offset > 10x the data range indicates different coordinate systems
        range_threshold_x = max(ref_bounds.range_x, new_bounds.range_x) * 10
        range_threshold_y = max(ref_bounds.range_y, new_bounds.range_y) * 10
        range_threshold_z = max(ref_bounds.range_z, new_bounds.range_z) * 10
        
        misaligned_x = offset_x > range_threshold_x
        misaligned_y = offset_y > range_threshold_y
        misaligned_z = offset_z > range_threshold_z
        
        if misaligned_x or misaligned_y or misaligned_z:
            # Avoid non-ASCII symbols for Windows console compatibility
            logger.warning(f"[MISALIGNED] Dataset '{dataset_info.name}' appears MISALIGNED with reference '{self.reference_dataset}'")
            logger.warning("   Offset from reference center:")
            logger.warning(f"   - X: {offset_x:,.2f}m {'(LARGE!)' if misaligned_x else '(OK)'}")
            logger.warning(f"   - Y: {offset_y:,.2f}m {'(LARGE!)' if misaligned_y else '(OK)'}")
            logger.warning(f"   - Z: {offset_z:,.2f}m {'(LARGE!)' if misaligned_z else '(OK)'}")
            
            # Calculate required offset to align
            required_offset = (
                ref_center[0] - new_center[0],
                ref_center[1] - new_center[1],
                ref_center[2] - new_center[2]
            )
            
            dataset_info.is_aligned = False
            dataset_info.offset = required_offset
            
            logger.info(
                f"   Required offset to align: ({required_offset[0]:,.2f}, "
                f"{required_offset[1]:,.2f}, {required_offset[2]:,.2f})"
            )
        else:
            dataset_info.is_aligned = True
            logger.info(f"Dataset '{dataset_info.name}' is aligned with reference")
    
    def get_alignment_offset(self, dataset_name: str) -> Tuple[float, float, float]:
        """
        Get the offset needed to align a dataset with the reference frame.
        
        Returns:
            (dx, dy, dz) offset in meters
        """
        if dataset_name not in self.datasets:
            logger.warning(f"Dataset '{dataset_name}' not registered")
            return (0.0, 0.0, 0.0)
        
        dataset_info = self.datasets[dataset_name]
        return dataset_info.offset
    
    def apply_offset_to_dataframe(self, df: pd.DataFrame, offset: Tuple[float, float, float],
                                  coord_cols: Tuple[str, str, str] = ('X', 'Y', 'Z')) -> pd.DataFrame:
        """
        Apply coordinate offset to a DataFrame.
        
        Automatically detects and updates all coordinate columns (XC/YC/ZC, XMORIG/YMORIG/ZMORIG, X/Y/Z, etc.)
        
        Args:
            df: DataFrame with coordinate columns
            offset: (dx, dy, dz) offset to apply
            coord_cols: Primary coordinate column names (used if specific columns not found)
        
        Returns:
            DataFrame with updated coordinates
        """
        df = df.copy()
        
        # Track how many coordinate sets we updated
        coords_updated = []
        
        # Update all possible coordinate column combinations
        coord_sets = [
            ('XC', 'YC', 'ZC'),  # Centroids
            ('XMORIG', 'YMORIG', 'ZMORIG'),  # Origins
            ('X', 'Y', 'Z'),  # Standard
            ('x', 'y', 'z'),  # Lowercase
            ('EAST', 'NORTH', 'ELEV'),  # Survey coords
            coord_cols  # User-specified
        ]
        
        for x_col, y_col, z_col in coord_sets:
            if all(col in df.columns for col in (x_col, y_col, z_col)):
                df[x_col] = df[x_col] + offset[0]
                df[y_col] = df[y_col] + offset[1]
                df[z_col] = df[z_col] + offset[2]
                coords_updated.append(f"{x_col}/{y_col}/{z_col}")
        
        if coords_updated:
            logger.info(f"Applied offset ({offset[0]:,.2f}, {offset[1]:,.2f}, {offset[2]:,.2f}) to columns: {', '.join(coords_updated)}")
        else:
            logger.warning(f"No recognized coordinate columns found in DataFrame. Tried: {[set for set in coord_sets]}")
        
        return df
    
    def apply_offset_to_block_model(self, block_model, offset: Tuple[float, float, float]):
        """
        Apply coordinate offset to a BlockModel object.
        
        Modifies the block model in-place by updating XMORIG, YMORIG, ZMORIG.
        """
        try:
            df = block_model.to_dataframe()
            
            if all(col in df.columns for col in ['XMORIG', 'YMORIG', 'ZMORIG']):
                df['XMORIG'] = df['XMORIG'] + offset[0]
                df['YMORIG'] = df['YMORIG'] + offset[1]
                df['ZMORIG'] = df['ZMORIG'] + offset[2]
                
                # Update centroids if they exist
                if all(col in df.columns for col in ['XC', 'YC', 'ZC']):
                    df['XC'] = df['XC'] + offset[0]
                    df['YC'] = df['YC'] + offset[1]
                    df['ZC'] = df['ZC'] + offset[2]
                
                # Update the block model's internal data
                block_model.update_from_dataframe(df)
                
                logger.info(f"Applied offset ({offset[0]:,.2f}, {offset[1]:,.2f}, {offset[2]:,.2f}) to BlockModel")
            else:
                logger.warning("Block model missing origin coordinate columns (XMORIG, YMORIG, ZMORIG)")
                
        except Exception as e:
            logger.error(f"Error applying offset to block model: {e}", exc_info=True)
    
    def get_misaligned_datasets(self) -> List[DatasetInfo]:
        """Return list of datasets that are not aligned with reference frame."""
        return [info for info in self.datasets.values() if not info.is_aligned]
    
    def get_alignment_summary(self) -> str:
        """Generate a human-readable summary of dataset alignment status."""
        if not self.datasets:
            return "No datasets registered."
        
        lines = []
        lines.append("\n" + "="*70)
        lines.append("COORDINATE ALIGNMENT STATUS")
        lines.append("="*70)
        
        if self.reference_dataset:
            ref_info = self.datasets[self.reference_dataset]
            lines.append(f"\nReference Dataset: '{self.reference_dataset}' ({ref_info.data_type})")
            lines.append(f"   Bounds: {ref_info.bounds}")
        
        lines.append(f"\nRegistered Datasets: {len(self.datasets)}")
        
        for name, info in self.datasets.items():
            status = "ALIGNED" if info.is_aligned else "MISALIGNED"
            lines.append(f"\n   - {name} ({info.data_type}): {status}")
            lines.append(f"     Points: {info.point_count:,}")
            lines.append(f"     Bounds: {info.bounds}")
            
            if not info.is_aligned:
                lines.append(f"     Required Offset: ({info.offset[0]:,.2f}, "
                           f"{info.offset[1]:,.2f}, {info.offset[2]:,.2f}) meters")
        
        misaligned = self.get_misaligned_datasets()
        if misaligned:
            lines.append(f"\nATTENTION: {len(misaligned)} dataset(s) require alignment!")
        else:
            lines.append("\nAll datasets are aligned.")
        
        lines.append("="*70 + "\n")
        
        return "\n".join(lines)
    
    def reset(self):
        """Clear all registered datasets and reset reference frame."""
        self.datasets.clear()
        self.reference_dataset = None
        self.global_offset = (0.0, 0.0, 0.0)
        logger.info("Coordinate manager reset")

