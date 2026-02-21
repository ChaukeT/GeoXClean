"""
Selection Manager

Manages multi-block selections, named selection sets, and exports.
Supports marquee selection, property-based queries, and set operations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Coordinate column name candidates (shared so panel/manager stay in sync)
X_CANDIDATES = ['X', 'XC', 'x', 'xc', 'EASTING', 'easting']
Y_CANDIDATES = ['Y', 'YC', 'y', 'yc', 'NORTHING', 'northing']
Z_CANDIDATES = ['Z', 'ZC', 'z', 'zc', 'RL', 'rl', 'ELEVATION', 'elevation']


class SelectionSet:
    """Represents a named set of selected block indices."""

    def __init__(self, name: str, indices: Set[int], description: str = ""):
        self.name = name
        self.indices = indices
        self.description = description
        self.created_at = pd.Timestamp.now()

    def __len__(self):
        return len(self.indices)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'indices': list(self.indices),
            'description': self.description,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SelectionSet':
        """Deserialize from dictionary."""
        obj = cls(
            name=data['name'],
            indices=set(data['indices']),
            description=data.get('description', '')
        )
        if 'created_at' in data:
            obj.created_at = pd.Timestamp(data['created_at'])
        return obj


class SelectionManager:
    """
    Manages block selections and named selection sets.

    Features:
    - Marquee (box) selection in 3D space
    - Property-based query selection (single and multi-criteria)
    - Named selection sets (save, load, combine)
    - Export selections to CSV/VTK
    - Set operations (union, intersection, difference)
    """

    # Tolerance for float equality comparisons
    FLOAT_TOLERANCE = 1e-6

    def __init__(self):
        self.block_df: Optional[pd.DataFrame] = None
        self.current_selection: Set[int] = set()
        self.named_sets: Dict[str, SelectionSet] = {}
        self.grid_spec: Optional[Dict] = None
        logger.info("Initialized SelectionManager")

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def set_block_model(self, block_df: pd.DataFrame, grid_spec: Optional[Dict] = None):
        """Set the block model data."""
        self.block_df = block_df.copy()
        self.grid_spec = grid_spec
        self.clear_selection()
        logger.info(f"Set block model: {len(self.block_df)} blocks, "
                     f"columns: {list(self.block_df.columns)}")

    @property
    def has_model(self) -> bool:
        """Whether a block model is loaded."""
        return self.block_df is not None and len(self.block_df) > 0

    @property
    def numeric_columns(self) -> List[str]:
        """Return numeric column names from the current model."""
        if self.block_df is None:
            return []
        return self.block_df.select_dtypes(include=['number']).columns.tolist()

    @property
    def model_bounds(self) -> Optional[Dict[str, Tuple[float, float]]]:
        """Return spatial bounds of the model as {axis: (min, max)}."""
        if not self.has_model:
            return None
        x_col = self._find_coordinate_column(X_CANDIDATES)
        y_col = self._find_coordinate_column(Y_CANDIDATES)
        z_col = self._find_coordinate_column(Z_CANDIDATES)
        if not all([x_col, y_col, z_col]):
            return None
        return {
            'x': (float(self.block_df[x_col].min()), float(self.block_df[x_col].max())),
            'y': (float(self.block_df[y_col].min()), float(self.block_df[y_col].max())),
            'z': (float(self.block_df[z_col].min()), float(self.block_df[z_col].max())),
        }

    # ------------------------------------------------------------------
    # Selection operations
    # ------------------------------------------------------------------

    def clear_selection(self):
        """Clear current selection."""
        self.current_selection = set()

    def select_all(self):
        """Select all blocks."""
        if self.has_model:
            self.current_selection = set(self.block_df.index)

    def _apply_mode(self, new_indices: Set[int], mode: str):
        """Apply set-operation mode to combine *new_indices* with the current selection."""
        if mode == 'new':
            self.current_selection = new_indices
        elif mode == 'add':
            self.current_selection |= new_indices
        elif mode == 'subtract':
            self.current_selection -= new_indices
        elif mode == 'intersect':
            self.current_selection &= new_indices
        else:
            logger.warning(f"Unknown selection mode '{mode}', treating as 'new'")
            self.current_selection = new_indices

    def select_by_marquee(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        z_range: Tuple[float, float],
        mode: str = 'new',
    ) -> int:
        """
        Select blocks within a 3D bounding box.

        Returns the total number of blocks in the resulting selection.
        """
        if not self.has_model:
            return 0

        x_col = self._find_coordinate_column(X_CANDIDATES)
        y_col = self._find_coordinate_column(Y_CANDIDATES)
        z_col = self._find_coordinate_column(Z_CANDIDATES)

        if not all([x_col, y_col, z_col]):
            logger.warning("Could not find coordinate columns for marquee selection")
            return 0

        mask = (
            (self.block_df[x_col] >= x_range[0]) & (self.block_df[x_col] <= x_range[1]) &
            (self.block_df[y_col] >= y_range[0]) & (self.block_df[y_col] <= y_range[1]) &
            (self.block_df[z_col] >= z_range[0]) & (self.block_df[z_col] <= z_range[1])
        )

        selected_indices = set(self.block_df[mask].index)
        self._apply_mode(selected_indices, mode)

        logger.info(f"Marquee selection: {len(selected_indices)} matched, "
                     f"mode={mode}, total={len(self.current_selection)}")
        return len(self.current_selection)

    def select_by_indices(self, indices: Set[int], mode: str = 'new') -> int:
        """
        Select blocks by explicit index set (e.g. from click selection).

        Returns the total number of blocks in the resulting selection.
        """
        if not self.has_model:
            return 0
        # Validate indices against actual DataFrame index
        valid = indices & set(self.block_df.index)
        self._apply_mode(valid, mode)
        logger.info(f"Index selection: {len(valid)} valid of {len(indices)}, "
                     f"mode={mode}, total={len(self.current_selection)}")
        return len(self.current_selection)

    def _build_comparison_mask(self, series: pd.Series, operator: str, value: float) -> pd.Series:
        """Build a boolean mask for a single comparison, with tolerance for == / !=."""
        if operator == '>':
            return series > value
        elif operator == '>=':
            return series >= value
        elif operator == '<':
            return series < value
        elif operator == '<=':
            return series <= value
        elif operator == '==':
            return np.isclose(series, value, atol=self.FLOAT_TOLERANCE)
        elif operator == '!=':
            return ~np.isclose(series, value, atol=self.FLOAT_TOLERANCE)
        else:
            raise ValueError(f"Unknown operator: {operator}")

    def select_by_property(
        self,
        property_name: str,
        operator: str,
        value: float,
        mode: str = 'new',
    ) -> int:
        """
        Select blocks based on a single property criterion.

        Returns the total number of blocks in the resulting selection.
        """
        if not self.has_model:
            return 0

        if property_name not in self.block_df.columns:
            logger.warning(f"Property '{property_name}' not found in block model")
            return len(self.current_selection)

        try:
            mask = self._build_comparison_mask(
                self.block_df[property_name], operator, value
            )
            selected_indices = set(self.block_df[mask].index)
            self._apply_mode(selected_indices, mode)

            logger.info(
                f"Property selection: {property_name} {operator} {value}, "
                f"{len(selected_indices)} matched, mode={mode}, "
                f"total={len(self.current_selection)}"
            )
            return len(self.current_selection)

        except Exception as e:
            logger.error(f"Error in property selection: {e}", exc_info=True)
            return len(self.current_selection)

    def select_by_multiple_criteria(
        self,
        criteria: List[Dict],
        logic: str = 'AND',
        mode: str = 'new',
    ) -> int:
        """
        Select blocks matching multiple property criteria.

        Args:
            criteria: List of dicts with keys 'property', 'operator', 'value'
            logic: 'AND' or 'OR'
            mode: Selection combination mode
        """
        if not self.has_model or not criteria:
            return 0

        masks = []
        for criterion in criteria:
            prop = criterion.get('property')
            op = criterion.get('operator')
            val = criterion.get('value')

            if prop not in self.block_df.columns:
                logger.warning(f"Skipping unknown column '{prop}'")
                continue

            try:
                masks.append(self._build_comparison_mask(self.block_df[prop], op, val))
            except Exception as e:
                logger.warning(f"Error evaluating criterion {criterion}: {e}")
                continue

        if not masks:
            return len(self.current_selection)

        if logic.upper() == 'AND':
            combined = masks[0]
            for m in masks[1:]:
                combined = combined & m
        else:
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m

        selected_indices = set(self.block_df[combined].index)
        self._apply_mode(selected_indices, mode)

        logger.info(
            f"Multi-criteria selection: {len(criteria)} criteria, logic={logic}, "
            f"{len(selected_indices)} matched, total={len(self.current_selection)}"
        )
        return len(self.current_selection)

    # ------------------------------------------------------------------
    # Named sets
    # ------------------------------------------------------------------

    def save_selection_as_set(self, name: str, description: str = "") -> bool:
        """Save current selection as a named set."""
        if not self.current_selection:
            logger.warning("Cannot save empty selection")
            return False

        self.named_sets[name] = SelectionSet(
            name=name,
            indices=self.current_selection.copy(),
            description=description,
        )
        logger.info(f"Saved selection set '{name}': {len(self.current_selection)} blocks")
        return True

    def load_selection_set(self, name: str, mode: str = 'new') -> bool:
        """Load a named selection set."""
        if name not in self.named_sets:
            logger.warning(f"Selection set '{name}' not found")
            return False

        self._apply_mode(self.named_sets[name].indices.copy(), mode)
        logger.info(
            f"Loaded selection set '{name}': mode={mode}, "
            f"total={len(self.current_selection)}"
        )
        return True

    def delete_selection_set(self, name: str) -> bool:
        """Delete a named selection set."""
        if name in self.named_sets:
            del self.named_sets[name]
            logger.info(f"Deleted selection set '{name}'")
            return True
        return False

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def get_selected_blocks(self, copy: bool = True) -> Optional[pd.DataFrame]:
        """
        Get DataFrame of currently selected blocks.

        Args:
            copy: If True return a copy (safe). If False return a view (fast).
        """
        if not self.has_model or not self.current_selection:
            return None

        idx = list(self.current_selection)
        return self.block_df.loc[idx].copy() if copy else self.block_df.loc[idx]

    def get_selected_indices(self) -> Set[int]:
        """Get a copy of the selected index set."""
        return self.current_selection.copy()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_selection_statistics(self) -> Dict:
        """Get statistics about current selection (uses view for efficiency)."""
        if not self.has_model or not self.current_selection:
            return {'count': 0, 'properties': {}}

        selected_df = self.get_selected_blocks(copy=False)
        stats: Dict = {'count': len(selected_df), 'properties': {}}

        for col in selected_df.select_dtypes(include=[np.number]).columns:
            try:
                col_data = selected_df[col]
                stats['properties'][col] = {
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'mean': float(col_data.mean()),
                    'sum': float(col_data.sum()),
                }
            except Exception as e:
                logger.debug(f"Could not compute stats for '{col}': {e}")

        return stats

    def get_selection_summary(self) -> str:
        """Return a short human-readable summary of the selection."""
        count = len(self.current_selection)
        if count == 0:
            return "No selection (0 blocks)"
        return f"Selected: {count:,} blocks"

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_selection_csv(self, filepath: Path, include_index: bool = True) -> bool:
        """Export current selection to CSV."""
        selected_df = self.get_selected_blocks(copy=True)
        if selected_df is None:
            logger.warning("No blocks selected for CSV export")
            return False

        try:
            selected_df.to_csv(filepath, index=include_index)
            logger.info(f"Exported {len(selected_df)} blocks to CSV: {filepath}")
            return True
        except Exception as e:
            logger.error(f"CSV export failed: {e}", exc_info=True)
            return False

    def export_selection_vtk(self, filepath: Path, grid_spec: Optional[Dict] = None) -> bool:
        """Export current selection to VTK point cloud."""
        selected_df = self.get_selected_blocks(copy=True)
        if selected_df is None:
            logger.warning("No blocks selected for VTK export")
            return False

        try:
            import pyvista as pv

            # Use the same coordinate candidates as marquee selection
            x_col = self._find_coordinate_column(X_CANDIDATES)
            y_col = self._find_coordinate_column(Y_CANDIDATES)
            z_col = self._find_coordinate_column(Z_CANDIDATES)

            if not all([x_col, y_col, z_col]):
                logger.error("Could not find coordinate columns for VTK export")
                return False

            points = np.column_stack([
                selected_df[x_col].values,
                selected_df[y_col].values,
                selected_df[z_col].values,
            ])

            mesh = pv.PolyData(points)

            for col in selected_df.columns:
                if col in (x_col, y_col, z_col):
                    continue
                try:
                    mesh[col] = selected_df[col].values
                except Exception as e:
                    logger.debug(f"Skipped non-numeric column '{col}' in VTK export: {e}")

            mesh.save(str(filepath))
            logger.info(f"Exported {len(selected_df)} blocks to VTK: {filepath}")
            return True

        except ImportError:
            logger.error("pyvista is required for VTK export but is not installed")
            return False
        except Exception as e:
            logger.error(f"VTK export failed: {e}", exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_coordinate_column(self, candidates: List[str]) -> Optional[str]:
        """Find first matching coordinate column name."""
        if self.block_df is None:
            return None
        for candidate in candidates:
            if candidate in self.block_df.columns:
                return candidate
        return None
