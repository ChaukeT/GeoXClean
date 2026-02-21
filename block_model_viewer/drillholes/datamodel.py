"""
Drillhole Data Model - High Performance (DataFrame-centric)

Stores drillhole data in Pandas DataFrames for O(1) access and vectorized math.
Eliminates the memory overhead of storing millions of Python objects.

GeoX Invariant Compliance:
- Provenance tracking in metadata
- Silent transformation audit (column mapping recorded)
- Default value documentation
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np
import logging

from ..utils.desurvey import minimum_curvature_path_from_surveys

logger = logging.getLogger(__name__)

# Data model version for schema tracking
DATAMODEL_VERSION = "1.1.0"  # Updated for structural measurements table

# Document default values used in dataclasses (GeoX invariant: no silent defaults)
COLLAR_DEFAULTS = {
    'azimuth': 0.0,  # Degrees, north-referenced
    'dip': -90.0,    # Degrees, negative = downward (vertical hole default)
    'length': 0.0    # Meters, unknown length
}


# --- Helper Dataclasses (For single-row UI interaction only) ---

@dataclass
class Collar:
    hole_id: str
    x: float
    y: float
    z: float
    azimuth: float = 0.0
    dip: float = -90.0
    length: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class SurveyInterval:
    """Survey measurement interval (azimuth/dip changes)."""
    hole_id: str
    depth_from: float
    depth_to: float
    azimuth: float
    dip: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssayInterval:
    """Assay sample interval with multiple element values."""
    hole_id: str
    depth_from: float
    depth_to: float
    values: Dict[str, float]  # {"Fe": 62.5, "SiO2": 4.2, ...}
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LithologyInterval:
    """Lithology/geology interval."""
    hole_id: str
    depth_from: float
    depth_to: float
    lith_code: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# Alias for backward compatibility (kept for imports)
Survey = SurveyInterval


@dataclass
class StructuralMeasurement:
    """
    Structural measurement interval (oriented discontinuity from drillhole).
    
    Stores orientation as a unit normal vector (vector-first design).
    Raw angles (alpha/beta or dip/dip-dir) preserved for provenance.
    
    GeoX Invariant Compliance:
    - Canonical storage is unit normal vector (computed on import)
    - Raw input angles preserved for audit trail
    - Source and confidence tracked
    """
    hole_id: str
    depth_from: float
    depth_to: float
    # Canonical representation: unit normal vector
    normal_x: float
    normal_y: float
    normal_z: float
    # Feature classification
    feature_type: str = "unknown"  # bedding, joint, fault, foliation, etc.
    # Provenance
    source: str = "unknown"  # csv_import, televiewer, oriented_core, mapping
    confidence: float = 1.0  # 0-1, data quality indicator
    # Optional raw input angles (for audit trail)
    raw_alpha: Optional[float] = None  # Original alpha if from alpha/beta
    raw_beta: Optional[float] = None   # Original beta if from alpha/beta
    raw_dip: Optional[float] = None    # Original dip if from dip/dip-dir
    raw_dip_direction: Optional[float] = None  # Original dip-dir
    # Additional metadata
    domain_id: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def normal(self) -> np.ndarray:
        """Get unit normal as numpy array."""
        return np.array([self.normal_x, self.normal_y, self.normal_z])
    
    @property
    def dip(self) -> float:
        """Derive dip from normal vector."""
        # Dip = arccos(-nz) for lower hemisphere normal
        nz = self.normal_z
        if nz > 0:  # Upper hemisphere - flip
            nz = -nz
        return float(np.degrees(np.arccos(-nz)))
    
    @property
    def dip_direction(self) -> float:
        """Derive dip direction from normal vector."""
        nx, ny, nz = self.normal_x, self.normal_y, self.normal_z
        if nz > 0:  # Flip to lower hemisphere
            nx, ny = -nx, -ny
        dd = float(np.degrees(np.arctan2(nx, ny))) % 360
        return dd


# Document default values for structural measurements (GeoX invariant)
STRUCTURAL_DEFAULTS = {
    'feature_type': 'unknown',
    'source': 'unknown',
    'confidence': 1.0,
}


# --- Main Database Class ---

class DrillholeDatabase:
    """
    High-Performance Container.
    Stores data as Pandas DataFrames.
    
    GeoX Invariant Compliance:
    - Metadata includes provenance tracking fields
    - Transformations are recorded in transformation_log
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None):
        # Initialize metadata with provenance fields (GeoX invariant)
        self.metadata = metadata or {}
        if 'created_at' not in self.metadata:
            self.metadata['created_at'] = datetime.now().isoformat()
        if 'datamodel_version' not in self.metadata:
            self.metadata['datamodel_version'] = DATAMODEL_VERSION
        
        # GeoX invariant: Track all transformations applied to data
        self._transformation_log: List[Dict[str, Any]] = []

        # Core Tables - Initialize as empty DataFrames with typed columns
        self._collars_df = pd.DataFrame(columns=['hole_id', 'x', 'y', 'z', 'azimuth', 'dip', 'length'])
        self._surveys_df = pd.DataFrame(columns=['hole_id', 'depth_from', 'depth_to', 'azimuth', 'dip'])
        self._assays_df = pd.DataFrame(columns=['hole_id', 'depth_from', 'depth_to'])
        self._lithology_df = pd.DataFrame(columns=['hole_id', 'depth_from', 'depth_to', 'lith_code'])
        
        # Structural measurements table (vector-first design)
        # Stores oriented discontinuities: joints, bedding, faults, foliation
        self._structures_df = pd.DataFrame(columns=[
            'hole_id', 'depth_from', 'depth_to',
            'normal_x', 'normal_y', 'normal_z',  # Unit normal vector (canonical)
            'feature_type',  # bedding, joint, fault, foliation, etc.
            'source', 'confidence', 'domain_id', 'timestamp',
            # Provenance: raw input angles preserved for audit
            'raw_alpha', 'raw_beta', 'raw_dip', 'raw_dip_direction',
            'source_file', 'source_checksum', 'row_index'
        ])

        # Enforce types for empty frames to avoid DtypeWarnings later
        if not self._collars_df.empty:
            self._collars_df['hole_id'] = self._collars_df['hole_id'].astype(str)
        if not self._surveys_df.empty:
            self._surveys_df['hole_id'] = self._surveys_df['hole_id'].astype(str)
        if not self._assays_df.empty:
            self._assays_df['hole_id'] = self._assays_df['hole_id'].astype(str)
        if not self._lithology_df.empty:
            self._lithology_df['hole_id'] = self._lithology_df['hole_id'].astype(str)
        if not self._structures_df.empty:
            self._structures_df['hole_id'] = self._structures_df['hole_id'].astype(str)
    
    def _log_transformation(self, operation: str, table_name: str, details: Dict[str, Any] = None):
        """
        Log a transformation operation for audit trail.
        
        GeoX invariant: No silent transformations.
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'table_name': table_name,
            'details': details or {}
        }
        self._transformation_log.append(entry)
        logger.debug(f"Transformation logged: {operation} on {table_name}")
    
    @property
    def transformation_log(self) -> List[Dict[str, Any]]:
        """Get the transformation log for audit purposes."""
        return list(self._transformation_log)

    @property
    def collars(self):
        """Access collars DataFrame or iterate as objects."""
        return self._collars_df
    
    @collars.setter
    def collars(self, value):
        self._collars_df = value
        if not self._collars_df.empty and 'hole_id' in self._collars_df.columns:
            self._collars_df['hole_id'] = self._collars_df['hole_id'].astype(str)
    
    @property
    def surveys(self):
        """Access surveys DataFrame or iterate as objects."""
        return self._surveys_df
    
    @surveys.setter
    def surveys(self, value):
        self._surveys_df = value
        if not self._surveys_df.empty and 'hole_id' in self._surveys_df.columns:
            self._surveys_df['hole_id'] = self._surveys_df['hole_id'].astype(str)
    
    @property
    def assays(self):
        """Access assays DataFrame or iterate as objects."""
        return self._assays_df
    
    @assays.setter
    def assays(self, value):
        self._assays_df = value
        if not self._assays_df.empty and 'hole_id' in self._assays_df.columns:
            self._assays_df['hole_id'] = self._assays_df['hole_id'].astype(str)
    
    @property
    def lithology(self):
        """Access lithology DataFrame or iterate as objects."""
        return self._lithology_df
    
    @lithology.setter
    def lithology(self, value):
        self._lithology_df = value
        if not self._lithology_df.empty and 'hole_id' in self._lithology_df.columns:
            self._lithology_df['hole_id'] = self._lithology_df['hole_id'].astype(str)
    
    @property
    def structures(self):
        """
        Access structural measurements DataFrame.
        
        Contains oriented discontinuity measurements from drillholes
        with vector-first storage (normal_x, normal_y, normal_z).
        """
        return self._structures_df
    
    @structures.setter
    def structures(self, value):
        self._structures_df = value
        if not self._structures_df.empty and 'hole_id' in self._structures_df.columns:
            self._structures_df['hole_id'] = self._structures_df['hole_id'].astype(str)

    def get_hole_ids(self) -> List[str]:
        """Fast unique hole ID lookup."""
        if self._collars_df.empty:
            return []
        return sorted(self._collars_df['hole_id'].unique().tolist())

    def get_collar(self, hole_id: str) -> Optional[pd.Series]:
        """Get single collar row as Series."""
        if self._collars_df.empty:
            return None
        row = self._collars_df[self._collars_df['hole_id'] == str(hole_id)]
        return row.iloc[0] if not row.empty else None

    def get_table(self, table_name: str) -> pd.DataFrame:
        """Generic accessor."""
        if table_name == 'collars':
            return self._collars_df
        if table_name == 'surveys':
            return self._surveys_df
        if table_name == 'assays':
            return self._assays_df
        if table_name == 'lithology':
            return self._lithology_df
        if table_name == 'structures':
            return self._structures_df
        raise ValueError(f"Unknown table: {table_name}")

    def set_table(self, table_name: str, df: pd.DataFrame):
        """
        Bulk update a table with column mapping audit trail.
        
        GeoX invariant: Column mappings are recorded (no silent transformations).
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Must provide a Pandas DataFrame")

        # Basic column validation
        required = {'hole_id'}
        if table_name == 'collars':
            required.update({'x', 'y', 'z'})
        elif table_name in ['assays', 'lithology']:
            required.update({'depth_from', 'depth_to'})
        elif table_name == 'surveys':
            required.update({'depth_from', 'depth_to', 'azimuth', 'dip'})
        elif table_name == 'structures':
            required.update({'depth_from', 'depth_to', 'normal_x', 'normal_y', 'normal_z'})

        # Track original columns for audit
        original_columns = list(df.columns)
        
        # Check headers - use case-insensitive column mapping
        cols = set(c.lower() for c in df.columns)
        applied_mappings = {}
        
        if not required.issubset(cols):
            # Map common aliases (case-insensitive)
            alias_map = {
                'holeid': 'hole_id', 'hole_id': 'hole_id', 'hole': 'hole_id', 'bhid': 'hole_id',
                'from': 'depth_from', 'to': 'depth_to',
                'depth_from': 'depth_from', 'depth_to': 'depth_to',
                'mfrom': 'depth_from', 'mto': 'depth_to',
                'easting': 'x', 'northing': 'y', 'elevation': 'z', 'rl': 'z',
            }
            # Build case-insensitive rename map for actual columns in the DataFrame
            rename_map = {}
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in alias_map:
                    target = alias_map[col_lower]
                    if target != col:  # Only rename if different
                        rename_map[col] = target
            
            if rename_map:
                df = df.rename(columns=rename_map)
                applied_mappings = rename_map
                # GeoX invariant: Log column mapping transformation
                logger.info(f"Column mapping applied for {table_name}: {rename_map}")
            
            cols = set(c.lower() for c in df.columns)
            if not required.issubset(cols):
                missing = required.difference(cols)
                raise ValueError(
                    f"Missing required columns for table '{table_name}': {sorted(missing)}. "
                    f"Available columns: {list(df.columns)}"
                )

        # Ensure hole_id is string
        if 'hole_id' in df.columns:
            df['hole_id'] = df['hole_id'].astype(str)

        # Log the transformation (GeoX invariant)
        self._log_transformation('SET_TABLE', table_name, {
            'original_columns': original_columns,
            'applied_mappings': applied_mappings,
            'row_count': len(df)
        })

        if table_name == 'collars':
            self._collars_df = df
        elif table_name == 'surveys':
            self._surveys_df = df
        elif table_name == 'assays':
            self._assays_df = df
        elif table_name == 'lithology':
            self._lithology_df = df
        elif table_name == 'structures':
            self._structures_df = df

    def validate(self) -> List[str]:
        """Fast vectorized validation checks."""
        issues = []

        # 1. Check for missing collars
        if not self._assays_df.empty and not self._collars_df.empty:
            assay_holes = set(self._assays_df['hole_id'].unique())
            collar_holes = set(self._collars_df['hole_id'].unique())
            missing = assay_holes - collar_holes
            if missing:
                issues.append(f"{len(missing)} holes have assays but no collar coordinates.")

        # 2. Check for overlapping intervals (Vectorized)
        for name, df in [('Assays', self._assays_df), ('Lithology', self._lithology_df)]:
            if df.empty:
                continue

            # Sort by ID and From
            df_sorted = df.sort_values(['hole_id', 'depth_from'])

            # Shifted values
            prev_hole = df_sorted['hole_id'].shift(1)
            prev_to = df_sorted['depth_to'].shift(1)

            # Logic: If same hole AND current_from < prev_to -> Overlap
            overlap_mask = (df_sorted['hole_id'] == prev_hole) & (df_sorted['depth_from'] < prev_to)

            count = overlap_mask.sum()
            if count > 0:
                issues.append(f"{name} table has {count} overlapping intervals.")

        return issues

    # --- DataFrame Access Methods (No backward compatibility - pure DataFrame) ---

    def get_assays_for(self, hole_id: str) -> pd.DataFrame:
        """Get all assay intervals for a specific hole as DataFrame."""
        if self._assays_df.empty:
            return pd.DataFrame()
        return self._assays_df[self._assays_df['hole_id'] == str(hole_id)].copy()

    def get_surveys_for(self, hole_id: str) -> pd.DataFrame:
        """Get all survey intervals for a specific hole as DataFrame."""
        if self._surveys_df.empty:
            return pd.DataFrame()
        return self._surveys_df[self._surveys_df['hole_id'] == str(hole_id)].copy()

    def get_lithology_for(self, hole_id: str) -> pd.DataFrame:
        """Get all lithology intervals for a specific hole as DataFrame."""
        if self._lithology_df.empty:
            return pd.DataFrame()
        return self._lithology_df[self._lithology_df['hole_id'] == str(hole_id)].copy()

    def get_collars_for(self, hole_id: str) -> pd.DataFrame:
        """Get all collars for a specific hole as DataFrame."""
        if self._collars_df.empty:
            return pd.DataFrame()
        return self._collars_df[self._collars_df['hole_id'] == str(hole_id)].copy()
    
    def get_structures_for(self, hole_id: str) -> pd.DataFrame:
        """Get all structural measurements for a specific hole as DataFrame."""
        if self._structures_df.empty:
            return pd.DataFrame()
        return self._structures_df[self._structures_df['hole_id'] == str(hole_id)].copy()

    def get_traces(self) -> np.ndarray:
        """
        Compute 3D traces for all drillholes using Minimum Curvature algorithm.
        
        This ensures mathematical consistency with the visualization layer.
        
        Returns:
            Array of shape (N, 3) with (x, y, z) coordinates along traces
        """
        traces = []

        for hole_id in self.get_hole_ids():
            collar_row = self.get_collar(hole_id)
            if collar_row is None:
                continue

            x, y, z = float(collar_row['x']), float(collar_row['y']), float(collar_row['z'])
            azimuth = float(collar_row.get('azimuth', 0.0))
            dip = float(collar_row.get('dip', -90.0))
            length = float(collar_row.get('length', 0.0))

            # Get surveys for this hole, sorted by depth
            hole_surveys_df = self.get_surveys_for(hole_id)

            if hole_surveys_df.empty:
                # No surveys - assume straight down from collar
                if length > 0:
                    traces.append([x, y, z])
                    traces.append([x, y, z - length])
                continue

            # Convert DataFrame surveys to list of dictionaries for Minimum Curvature
            surveys = []
            for _, survey_row in hole_surveys_df.sort_values('depth_from').iterrows():
                surveys.append({
                    'depth_from': float(survey_row['depth_from']),
                    'depth_to': float(survey_row['depth_to']),
                    'azimuth': float(survey_row['azimuth']),
                    'dip': float(survey_row['dip'])
                })

            # Use Minimum Curvature algorithm (same as visualization layer)
            coord_depths, station_coords = minimum_curvature_path_from_surveys(x, y, z, surveys)
            
            # Add all coordinates to traces
            for coord in station_coords:
                traces.append(coord.tolist())

        return np.array(traces) if traces else np.empty((0, 3))

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert database to dictionary of DataFrames (already DataFrames, just return).
        
        Returns:
            Dictionary with keys: 'collars', 'surveys', 'assays', 'lithology', 'structures'
        """
        return {
            'collars': self._collars_df.copy(),
            'surveys': self._surveys_df.copy(),
            'assays': self._assays_df.copy(),
            'lithology': self._lithology_df.copy(),
            'structures': self._structures_df.copy()
        }

    def get_assays_df(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """Get assays as DataFrame, optionally filtered by hole_id."""
        if hole_id:
            return self._assays_df[self._assays_df['hole_id'] == str(hole_id)].copy()
        return self._assays_df.copy()

    def get_surveys_df(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """Get surveys as DataFrame, optionally filtered by hole_id."""
        if hole_id:
            return self._surveys_df[self._surveys_df['hole_id'] == str(hole_id)].copy()
        return self._surveys_df.copy()

    def get_lithology_df(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """Get lithology as DataFrame, optionally filtered by hole_id."""
        if hole_id:
            return self._lithology_df[self._lithology_df['hole_id'] == str(hole_id)].copy()
        return self._lithology_df.copy()
    
    def get_structures_df(self, hole_id: Optional[str] = None) -> pd.DataFrame:
        """Get structural measurements as DataFrame, optionally filtered by hole_id."""
        if hole_id:
            return self._structures_df[self._structures_df['hole_id'] == str(hole_id)].copy()
        return self._structures_df.copy()
    
    def get_structures_as_normals(self, hole_id: Optional[str] = None) -> np.ndarray:
        """
        Get structural measurements as Nx3 array of unit normal vectors.
        
        Convenience method for feeding directly into structural analysis kernels.
        
        Args:
            hole_id: Optional filter by hole ID
        
        Returns:
            Nx3 numpy array of unit normal vectors
        """
        df = self.get_structures_df(hole_id)
        if df.empty:
            return np.empty((0, 3))
        
        normals = df[['normal_x', 'normal_y', 'normal_z']].values.astype(np.float64)
        return normals

    def save_assays(self, hole_id: str, df: pd.DataFrame):
        """
        Replace assays for a hole with rows from df.
        
        GeoX invariant: Operation logged to transformation log.
        """
        # Count existing for audit
        existing_count = len(self._assays_df[self._assays_df['hole_id'] == str(hole_id)])
        # Remove existing assays for this hole
        self._assays_df = self._assays_df[self._assays_df['hole_id'] != str(hole_id)]
        # Add new assays
        new_count = 0
        if not df.empty:
            df_copy = df.copy()
            if 'hole_id' not in df_copy.columns:
                df_copy['hole_id'] = hole_id
            else:
                df_copy['hole_id'] = df_copy['hole_id'].astype(str)
            # Ensure required columns
            if 'depth_from' not in df_copy.columns:
                raise ValueError("DataFrame must have 'depth_from' column")
            if 'depth_to' not in df_copy.columns:
                raise ValueError("DataFrame must have 'depth_to' column")
            self._assays_df = pd.concat([self._assays_df, df_copy], ignore_index=True)
            self._assays_df = self._assays_df.sort_values(['hole_id', 'depth_from', 'depth_to']).reset_index(drop=True)
            new_count = len(df_copy)
        
        # Log transformation (GeoX invariant)
        self._log_transformation('SAVE_ASSAYS', 'assays', {
            'hole_id': hole_id,
            'records_removed': existing_count,
            'records_added': new_count
        })

    def save_surveys(self, hole_id: str, df: pd.DataFrame):
        """
        Replace surveys for a hole with rows from df.
        
        GeoX invariant: Operation logged to transformation log.
        """
        existing_count = len(self._surveys_df[self._surveys_df['hole_id'] == str(hole_id)])
        self._surveys_df = self._surveys_df[self._surveys_df['hole_id'] != str(hole_id)]
        new_count = 0
        if not df.empty:
            df_copy = df.copy()
            if 'hole_id' not in df_copy.columns:
                df_copy['hole_id'] = hole_id
            else:
                df_copy['hole_id'] = df_copy['hole_id'].astype(str)
            if 'depth_from' not in df_copy.columns or 'depth_to' not in df_copy.columns:
                raise ValueError("DataFrame must have 'depth_from' and 'depth_to' columns")
            self._surveys_df = pd.concat([self._surveys_df, df_copy], ignore_index=True)
            self._surveys_df = self._surveys_df.sort_values(['hole_id', 'depth_from']).reset_index(drop=True)
            new_count = len(df_copy)
        
        self._log_transformation('SAVE_SURVEYS', 'surveys', {
            'hole_id': hole_id,
            'records_removed': existing_count,
            'records_added': new_count
        })

    def save_lithology(self, hole_id: str, df: pd.DataFrame):
        """
        Replace lithology for a hole with rows from df.
        
        GeoX invariant: Operation logged to transformation log.
        """
        existing_count = len(self._lithology_df[self._lithology_df['hole_id'] == str(hole_id)])
        self._lithology_df = self._lithology_df[self._lithology_df['hole_id'] != str(hole_id)]
        new_count = 0
        if not df.empty:
            df_copy = df.copy()
            if 'hole_id' not in df_copy.columns:
                df_copy['hole_id'] = hole_id
            else:
                df_copy['hole_id'] = df_copy['hole_id'].astype(str)
            if 'depth_from' not in df_copy.columns or 'depth_to' not in df_copy.columns:
                raise ValueError("DataFrame must have 'depth_from' and 'depth_to' columns")
            self._lithology_df = pd.concat([self._lithology_df, df_copy], ignore_index=True)
            self._lithology_df = self._lithology_df.sort_values(['hole_id', 'depth_from']).reset_index(drop=True)
            new_count = len(df_copy)
        
        self._log_transformation('SAVE_LITHOLOGY', 'lithology', {
            'hole_id': hole_id,
            'records_removed': existing_count,
            'records_added': new_count
        })
    
    def save_structures(self, hole_id: str, df: pd.DataFrame):
        """
        Replace structural measurements for a hole with rows from df.
        
        GeoX invariant: Operation logged to transformation log.
        
        Args:
            hole_id: Drillhole identifier
            df: DataFrame with required columns:
                - depth_from, depth_to
                - normal_x, normal_y, normal_z (unit vector)
                Optional columns:
                - feature_type, source, confidence, domain_id, timestamp
                - raw_alpha, raw_beta, raw_dip, raw_dip_direction (provenance)
        """
        existing_count = len(self._structures_df[self._structures_df['hole_id'] == str(hole_id)])
        self._structures_df = self._structures_df[self._structures_df['hole_id'] != str(hole_id)]
        new_count = 0
        if not df.empty:
            df_copy = df.copy()
            if 'hole_id' not in df_copy.columns:
                df_copy['hole_id'] = hole_id
            else:
                df_copy['hole_id'] = df_copy['hole_id'].astype(str)
            
            # Validate required columns
            required = {'depth_from', 'depth_to', 'normal_x', 'normal_y', 'normal_z'}
            if not required.issubset(set(df_copy.columns)):
                missing = required - set(df_copy.columns)
                raise ValueError(f"DataFrame missing required columns: {missing}")
            
            # Apply defaults for optional columns (GeoX invariant: document defaults)
            defaults_applied = {}
            if 'feature_type' not in df_copy.columns:
                df_copy['feature_type'] = STRUCTURAL_DEFAULTS['feature_type']
                defaults_applied['feature_type'] = STRUCTURAL_DEFAULTS['feature_type']
            if 'source' not in df_copy.columns:
                df_copy['source'] = STRUCTURAL_DEFAULTS['source']
                defaults_applied['source'] = STRUCTURAL_DEFAULTS['source']
            if 'confidence' not in df_copy.columns:
                df_copy['confidence'] = STRUCTURAL_DEFAULTS['confidence']
                defaults_applied['confidence'] = STRUCTURAL_DEFAULTS['confidence']
            
            # Normalize the normal vectors
            nx = df_copy['normal_x'].values
            ny = df_copy['normal_y'].values
            nz = df_copy['normal_z'].values
            norms = np.sqrt(nx**2 + ny**2 + nz**2)
            norms = np.where(norms > 1e-10, norms, 1.0)
            df_copy['normal_x'] = nx / norms
            df_copy['normal_y'] = ny / norms
            df_copy['normal_z'] = nz / norms
            
            self._structures_df = pd.concat([self._structures_df, df_copy], ignore_index=True)
            self._structures_df = self._structures_df.sort_values(['hole_id', 'depth_from']).reset_index(drop=True)
            new_count = len(df_copy)
        
        self._log_transformation('SAVE_STRUCTURES', 'structures', {
            'hole_id': hole_id,
            'records_removed': existing_count,
            'records_added': new_count,
            'defaults_applied': defaults_applied if df.empty else {}
        })
    
    def add_structures_from_dip_dipdir(
        self,
        hole_id: str,
        depth_from: np.ndarray,
        depth_to: np.ndarray,
        dip: np.ndarray,
        dip_direction: np.ndarray,
        feature_type: Optional[Union[str, List[str]]] = None,
        source: str = 'dip_dipdir_import',
        **kwargs
    ):
        """
        Add structural measurements from dip/dip-direction angles.
        
        Convenience method that converts to unit normals automatically.
        
        Args:
            hole_id: Drillhole identifier
            depth_from: Array of from depths
            depth_to: Array of to depths
            dip: Array of dip angles (degrees, 0-90)
            dip_direction: Array of dip direction angles (degrees, 0-360)
            feature_type: Feature type(s)
            source: Data source identifier
            **kwargs: Additional columns (domain_id, confidence, etc.)
        """
        # Import conversion function
        try:
            from geox.structural.core import dip_dipdir_to_normal
        except ImportError:
            # Fallback implementation
            dip_rad = np.radians(dip)
            dd_rad = np.radians(dip_direction)
            nx = np.sin(dip_rad) * np.sin(dd_rad)
            ny = np.sin(dip_rad) * np.cos(dd_rad)
            nz = -np.cos(dip_rad)
            normals = np.column_stack([nx, ny, nz])
        else:
            normals = dip_dipdir_to_normal(dip, dip_direction)
        
        n = len(depth_from)
        
        # Build DataFrame
        df = pd.DataFrame({
            'hole_id': str(hole_id),
            'depth_from': depth_from,
            'depth_to': depth_to,
            'normal_x': normals[:, 0],
            'normal_y': normals[:, 1],
            'normal_z': normals[:, 2],
            'source': source,
            'raw_dip': dip,
            'raw_dip_direction': dip_direction,
        })
        
        # Add feature_type
        if feature_type is None:
            df['feature_type'] = STRUCTURAL_DEFAULTS['feature_type']
        elif isinstance(feature_type, str):
            df['feature_type'] = feature_type
        else:
            df['feature_type'] = feature_type
        
        # Add kwargs
        for key, value in kwargs.items():
            df[key] = value
        
        self.save_structures(hole_id, df)
    
    def add_structures_from_alpha_beta(
        self,
        hole_id: str,
        depth_from: np.ndarray,
        depth_to: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray,
        feature_type: Optional[Union[str, List[str]]] = None,
        source: str = 'alpha_beta_import',
        **kwargs
    ):
        """
        Add structural measurements from alpha/beta angles (oriented core).
        
        Uses the oriented-core/televiewer convention:
        - Alpha: acute angle between plane and borehole axis (0-90)
        - Beta: clockwise angle around core from highside to plane trace (0-360)
        
        Requires hole survey data to convert to world coordinates.
        
        Args:
            hole_id: Drillhole identifier
            depth_from: Array of from depths
            depth_to: Array of to depths
            alpha: Array of alpha angles (degrees, 0-90)
            beta: Array of beta angles (degrees, 0-360)
            feature_type: Feature type(s)
            source: Data source identifier
            **kwargs: Additional columns
        """
        # Get hole orientation at each measurement depth
        collar = self.get_collar(hole_id)
        if collar is None:
            raise ValueError(f"No collar found for hole {hole_id}")
        
        surveys_df = self.get_surveys_for(hole_id)
        
        # Interpolate hole azimuth and dip at each measurement depth
        mid_depths = (np.asarray(depth_from) + np.asarray(depth_to)) / 2
        
        if surveys_df.empty:
            # Use collar orientation for entire hole
            hole_azimuths = np.full(len(mid_depths), float(collar.get('azimuth', 0.0)))
            hole_dips = np.full(len(mid_depths), float(collar.get('dip', -90.0)))
        else:
            # Interpolate from surveys
            surveys_df = surveys_df.sort_values('depth_from')
            survey_depths = surveys_df['depth_from'].values
            survey_az = surveys_df['azimuth'].values
            survey_dip = surveys_df['dip'].values
            
            hole_azimuths = np.interp(mid_depths, survey_depths, survey_az)
            hole_dips = np.interp(mid_depths, survey_depths, survey_dip)
        
        # Convert alpha/beta to world-space normals
        try:
            from geox.structural.core import alpha_beta_to_normal
            normals = alpha_beta_to_normal(alpha, beta, hole_azimuths, hole_dips)
        except ImportError:
            raise ImportError(
                "alpha_beta_to_normal not available. Install geox.structural.core "
                "or use add_structures_from_dip_dipdir instead."
            )
        
        n = len(depth_from)
        
        # Build DataFrame
        df = pd.DataFrame({
            'hole_id': str(hole_id),
            'depth_from': depth_from,
            'depth_to': depth_to,
            'normal_x': normals[:, 0],
            'normal_y': normals[:, 1],
            'normal_z': normals[:, 2],
            'source': source,
            'raw_alpha': alpha,
            'raw_beta': beta,
        })
        
        if feature_type is None:
            df['feature_type'] = STRUCTURAL_DEFAULTS['feature_type']
        elif isinstance(feature_type, str):
            df['feature_type'] = feature_type
        else:
            df['feature_type'] = feature_type
        
        for key, value in kwargs.items():
            df[key] = value
        
        self.save_structures(hole_id, df)

    def save_collar(self, hole_id: str, df: pd.DataFrame):
        """
        Update collar for a hole from DataFrame.
        
        GeoX invariant: Default values applied are documented in COLLAR_DEFAULTS.
        """
        if df.empty:
            return
        row = df.iloc[0]
        # Remove existing collar
        self._collars_df = self._collars_df[self._collars_df['hole_id'] != str(hole_id)]
        
        # Track which defaults are applied (GeoX invariant: no silent defaults)
        defaults_applied = {}
        azimuth = row.get('azimuth')
        dip = row.get('dip')
        length = row.get('length')
        
        if pd.isna(azimuth) or azimuth is None:
            azimuth = COLLAR_DEFAULTS['azimuth']
            defaults_applied['azimuth'] = COLLAR_DEFAULTS['azimuth']
        if pd.isna(dip) or dip is None:
            dip = COLLAR_DEFAULTS['dip']
            defaults_applied['dip'] = COLLAR_DEFAULTS['dip']
        if pd.isna(length) or length is None:
            length = COLLAR_DEFAULTS['length']
            defaults_applied['length'] = COLLAR_DEFAULTS['length']
        
        # Add new collar
        new_collar = pd.DataFrame([{
            'hole_id': str(hole_id),
            'x': float(row.get('x', 0)),
            'y': float(row.get('y', 0)),
            'z': float(row.get('z', 0)),
            'azimuth': float(azimuth),
            'dip': float(dip),
            'length': float(length)
        }])
        self._collars_df = pd.concat([self._collars_df, new_collar], ignore_index=True)
        
        # Log transformation with defaults applied
        self._log_transformation('SAVE_COLLAR', 'collars', {
            'hole_id': hole_id,
            'defaults_applied': defaults_applied
        })

    def replace_surveys_for(self, hole_id: str, surveys_df: pd.DataFrame) -> None:
        """Replace all surveys for a hole with new DataFrame."""
        self.save_surveys(hole_id, surveys_df)

    def replace_assays_for(self, hole_id: str, assays_df: pd.DataFrame) -> None:
        """Replace all assays for a hole with new DataFrame."""
        self.save_assays(hole_id, assays_df)

    def replace_lithology_for(self, hole_id: str, lithology_df: pd.DataFrame) -> None:
        """Replace all lithology for a hole with new DataFrame."""
        self.save_lithology(hole_id, lithology_df)

    def copy_from(self, other: 'DrillholeDatabase') -> None:
        """Copy all data from another database."""
        self._collars_df = other._collars_df.copy()
        self._surveys_df = other._surveys_df.copy()
        self._assays_df = other._assays_df.copy()
        self._lithology_df = other._lithology_df.copy()
        self._structures_df = other._structures_df.copy()
        self.metadata = dict(other.metadata)
    
    def replace_structures_for(self, hole_id: str, structures_df: pd.DataFrame) -> None:
        """Replace all structural measurements for a hole with new DataFrame."""
        self.save_structures(hole_id, structures_df)
    
