"""
Block Model Table Model - Lazy-loading QAbstractTableModel for BlockModel data.

This model reads directly from BlockModel's numpy arrays on-demand, avoiding
the memory spike from creating a full DataFrame copy.
"""

from __future__ import annotations

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, Qt, QModelIndex

from ..models.block_model import BlockModel

logger = logging.getLogger(__name__)


class BlockModelTableModel(QAbstractTableModel):
    """
    Table model that reads directly from BlockModel's numpy arrays.
    
    Provides lazy loading - only reads data when requested by the view.
    This eliminates the memory spike from converting the entire BlockModel
    to a DataFrame upfront.
    """
    
    def __init__(self, block_model: Optional[BlockModel] = None, parent=None):
        """
        Initialize the table model.
        
        Args:
            block_model: BlockModel instance to display (can be None initially)
            parent: Parent QObject
        """
        super().__init__(parent)
        self._block_model: Optional[BlockModel] = None
        self._column_names: list[str] = []
        self._column_cache: Dict[str, np.ndarray] = {}
        self._filter_mask: Optional[np.ndarray] = None
        
        if block_model is not None:
            self.set_block_model(block_model)
    
    def set_block_model(self, block_model: Optional[BlockModel]):
        """Set the block model and refresh the model."""
        self.beginResetModel()
        self._block_model = block_model
        self._column_cache.clear()
        self._filter_mask = None
        self._update_column_names()
        self.endResetModel()
        logger.info(f"Set block model with {self.rowCount()} rows, {self.columnCount()} columns")
    
    def _update_column_names(self):
        """Update the list of column names from the block model."""
        if self._block_model is None:
            self._column_names = []
            return
        
        # Standard geometry columns
        self._column_names = ['x', 'y', 'z', 'dx', 'dy', 'dz']
        
        # Add property columns
        if self._block_model.properties:
            property_names = sorted(self._block_model.get_property_names())
            self._column_names.extend(property_names)
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of rows."""
        if self._block_model is None or parent.isValid():
            return 0
        
        if self._filter_mask is not None:
            return int(np.sum(self._filter_mask))
        
        return self._block_model.block_count
    
    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        """Return the number of columns."""
        if parent.isValid():
            return 0
        return len(self._column_names)
    
    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        """Return header data for the given section."""
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self._column_names):
                return self._column_names[section]
        else:
            # Row numbers
            if self._filter_mask is not None:
                # Return actual block index for filtered rows
                visible_indices = np.where(self._filter_mask)[0]
                if 0 <= section < len(visible_indices):
                    return int(visible_indices[section])
            return section
        
        return None
    
    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        """Return data for the given index."""
        if not index.isValid() or self._block_model is None:
            return None
        
        row = index.row()
        col = index.column()
        
        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            # Get the actual block index (accounting for filters)
            block_idx = self._get_block_index(row)
            if block_idx is None:
                return None
            
            # Get column name
            if col >= len(self._column_names):
                return None
            col_name = self._column_names[col]
            
            # Get value from block model
            value = self._get_value(block_idx, col_name)
            
            # Format for display
            if value is None:
                return ""
            elif isinstance(value, float):
                if np.isnan(value):
                    return ""
                return f"{value:.6g}"
            else:
                return str(value)
        
        return None
    
    def _get_block_index(self, row: int) -> Optional[int]:
        """Get the actual block index for a given table row (accounting for filters)."""
        if self._block_model is None:
            return None
        
        if self._filter_mask is not None:
            visible_indices = np.where(self._filter_mask)[0]
            if 0 <= row < len(visible_indices):
                return int(visible_indices[row])
            return None
        
        if 0 <= row < self._block_model.block_count:
            return row
        
        return None
    
    def _get_value(self, block_idx: int, col_name: str):
        """Get a value from the block model for a specific block and column."""
        if self._block_model is None:
            return None
        
        # Geometry columns
        if col_name == 'x':
            if self._block_model.positions is not None:
                return float(self._block_model.positions[block_idx, 0])
        elif col_name == 'y':
            if self._block_model.positions is not None:
                return float(self._block_model.positions[block_idx, 1])
        elif col_name == 'z':
            if self._block_model.positions is not None:
                return float(self._block_model.positions[block_idx, 2])
        elif col_name == 'dx':
            if self._block_model.dimensions is not None:
                return float(self._block_model.dimensions[block_idx, 0])
        elif col_name == 'dy':
            if self._block_model.dimensions is not None:
                return float(self._block_model.dimensions[block_idx, 1])
        elif col_name == 'dz':
            if self._block_model.dimensions is not None:
                return float(self._block_model.dimensions[block_idx, 2])
        else:
            # Property column
            prop_values = self._block_model.get_property(col_name)
            if prop_values is not None and 0 <= block_idx < len(prop_values):
                value = prop_values[block_idx]
                # Convert numpy scalar to Python type
                if isinstance(value, (np.integer, np.floating)):
                    return value.item()
                return value
        
        return None
    
    def apply_filter(self, mask: Optional[np.ndarray]):
        """
        Apply a filter mask to show only certain rows.
        
        Args:
            mask: Boolean array of length block_count, True for visible rows.
                  None to clear filter.
        """
        if self._block_model is None:
            logger.warning("Cannot apply filter: no block model is set")
            return

        if mask is not None and len(mask) != self._block_model.block_count:
            logger.warning(f"Filter mask length {len(mask)} doesn't match block count {self._block_model.block_count}")
            return
        
        self.beginResetModel()
        self._filter_mask = mask
        self.endResetModel()
        logger.info(f"Applied filter: {np.sum(mask) if mask is not None else 'all'} rows visible")
    
    def clear_filter(self):
        """Clear any active filter."""
        self.apply_filter(None)
    
    def get_block_index_for_row(self, row: int) -> Optional[int]:
        """Get the block index for a given table row."""
        return self._get_block_index(row)
    
    def to_dataframe(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Convert visible rows to a DataFrame (for export).
        
        This creates a DataFrame only when needed (e.g., for export),
        not during normal table display.
        
        Args:
            max_rows: Maximum number of rows to include (None for all)
        
        Returns:
            DataFrame with block model data
        """
        if self._block_model is None:
            return pd.DataFrame()
        
        # Determine which rows to include
        if self._filter_mask is not None:
            visible_indices = np.where(self._filter_mask)[0]
        else:
            visible_indices = np.arange(self._block_model.block_count)
        
        if max_rows is not None:
            visible_indices = visible_indices[:max_rows]
        
        # Build DataFrame columns
        data = {}
        
        # Geometry columns
        if self._block_model.positions is not None:
            data['x'] = self._block_model.positions[visible_indices, 0]
            data['y'] = self._block_model.positions[visible_indices, 1]
            data['z'] = self._block_model.positions[visible_indices, 2]
        
        if self._block_model.dimensions is not None:
            data['dx'] = self._block_model.dimensions[visible_indices, 0]
            data['dy'] = self._block_model.dimensions[visible_indices, 1]
            data['dz'] = self._block_model.dimensions[visible_indices, 2]
        
        # Property columns
        for prop_name in self._block_model.get_property_names():
            prop_values = self._block_model.get_property(prop_name)
            if prop_values is not None:
                data[prop_name] = prop_values[visible_indices]
        
        return pd.DataFrame(data, index=visible_indices)

