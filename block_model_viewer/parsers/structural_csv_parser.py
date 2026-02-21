"""
Structural CSV Parser - Flexible parser for folds, faults, and unconformities.

Supports multiple CSV formats:
1. Orientation Measurements: X, Y, Z, dip, dip_direction, feature_type, feature_name
2. Point Clouds: X, Y, Z, feature_type, feature_name
3. Vector Constraints (Hermite): X, Y, Z, dip, azimuth, G_x, G_y, G_z, feature_type
4. Fold Axis Data: X, Y, Z, plunge, trend, fold_type, feature_name

GeoX Invariant Compliance:
- Parser versioning for reproducibility
- File checksums for data integrity
- Column mapping audit trail
- Validation gates for data quality
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from .base_parser import compute_file_checksum, PARSER_FRAMEWORK_VERSION
from ..structural.feature_types import (
    FeatureType,
    FaultDisplacementType,
    FoldStyle,
    UnconformityType,
    StructuralOrientation,
    StructuralFeature,
    FaultFeature,
    FoldFeature,
    UnconformityFeature,
    StructuralFeatureCollection,
)
from ..structural.datasets import PlaneMeasurement, LineationMeasurement

logger = logging.getLogger(__name__)

# Parser version for audit trail
STRUCTURAL_CSV_PARSER_VERSION = "1.0.0"


# =============================================================================
# ENUMERATIONS
# =============================================================================

class CSVFormat(str, Enum):
    """Detected CSV format types."""
    ORIENTATION = "orientation"  # Has dip/dip_direction
    POINT_CLOUD = "point_cloud"  # Just X, Y, Z points
    VECTOR_CONSTRAINT = "vector_constraint"  # Has G_x, G_y, G_z gradient vectors
    FOLD_AXIS = "fold_axis"  # Has plunge/trend
    UNKNOWN = "unknown"


# =============================================================================
# COLUMN PATTERNS
# =============================================================================

# Column name patterns for auto-detection (case-insensitive)
COORDINATE_PATTERNS = {
    'x': ['x', 'x_coord', 'xc', 'xpos', 'easting', 'east', 'x_center'],
    'y': ['y', 'y_coord', 'yc', 'ypos', 'northing', 'north', 'y_center'],
    'z': ['z', 'z_coord', 'zc', 'zpos', 'elevation', 'elev', 'rl', 'z_center'],
}

ORIENTATION_PATTERNS = {
    'dip': ['dip', 'dip_angle', 'inclination', 'true_dip'],
    'dip_direction': ['dip_direction', 'dipdirection', 'dd', 'dip_dir', 'azimuth', 'strike'],
    'azimuth': ['azimuth', 'az', 'bearing', 'direction'],
}

LINEATION_PATTERNS = {
    'plunge': ['plunge', 'plunge_angle', 'plng'],
    'trend': ['trend', 'trend_azimuth', 'trd'],
}

GRADIENT_PATTERNS = {
    'g_x': ['g_x', 'gx', 'gradient_x', 'nx', 'normal_x'],
    'g_y': ['g_y', 'gy', 'gradient_y', 'ny', 'normal_y'],
    'g_z': ['g_z', 'gz', 'gradient_z', 'nz', 'normal_z'],
}

FEATURE_PATTERNS = {
    'feature_type': ['feature_type', 'type', 'structure_type', 'struct_type', 'category'],
    'feature_name': ['feature_name', 'name', 'structure_name', 'label', 'id'],
    'set_id': ['set_id', 'set', 'group', 'family'],
}

# Feature type aliases
FAULT_ALIASES = {'fault', 'flt', 'f', 'fault_surface', 'discontinuity'}
FOLD_ALIASES = {'fold', 'fld', 'anticline', 'syncline', 'monocline'}
UNCONFORMITY_ALIASES = {'unconformity', 'unc', 'u', 'erosion', 'unconform'}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ColumnMapping:
    """Mapping between CSV columns and expected fields."""
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    z_col: Optional[str] = None
    dip_col: Optional[str] = None
    dip_direction_col: Optional[str] = None
    plunge_col: Optional[str] = None
    trend_col: Optional[str] = None
    g_x_col: Optional[str] = None
    g_y_col: Optional[str] = None
    g_z_col: Optional[str] = None
    feature_type_col: Optional[str] = None
    feature_name_col: Optional[str] = None
    set_id_col: Optional[str] = None
    polarity_col: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Optional[str]]:
        """Convert to dictionary for serialization."""
        return {
            'x_col': self.x_col,
            'y_col': self.y_col,
            'z_col': self.z_col,
            'dip_col': self.dip_col,
            'dip_direction_col': self.dip_direction_col,
            'plunge_col': self.plunge_col,
            'trend_col': self.trend_col,
            'g_x_col': self.g_x_col,
            'g_y_col': self.g_y_col,
            'g_z_col': self.g_z_col,
            'feature_type_col': self.feature_type_col,
            'feature_name_col': self.feature_name_col,
            'set_id_col': self.set_id_col,
            'polarity_col': self.polarity_col,
        }


@dataclass
class ParseResult:
    """Result of parsing a structural CSV file."""
    collection: StructuralFeatureCollection
    format_detected: CSVFormat
    column_mapping: ColumnMapping
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    rows_processed: int = 0
    rows_skipped: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# STRUCTURAL CSV PARSER
# =============================================================================

class StructuralCSVParser:
    """
    Parser for structural geology CSV files.
    
    Supports faults, folds, and unconformities in multiple formats.
    """
    
    def __init__(self):
        self.supported_extensions = ['.csv', '.txt', '.dat']
        self.format_name = "Structural CSV"
        self.version = STRUCTURAL_CSV_PARSER_VERSION
    
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file."""
        return file_path.suffix.lower() in self.supported_extensions
    
    def parse(
        self,
        file_path: Path,
        column_mapping: Optional[ColumnMapping] = None,
        expected_format: Optional[CSVFormat] = None,
        delimiter: Optional[str] = None,
        encoding: str = 'utf-8',
        skip_rows: int = 0,
        max_rows: Optional[int] = None,
        validate: bool = True,
    ) -> ParseResult:
        """
        Parse a structural CSV file.
        
        Args:
            file_path: Path to the CSV file
            column_mapping: Optional explicit column mapping (auto-detect if None)
            expected_format: Optional expected format (auto-detect if None)
            delimiter: CSV delimiter (auto-detect if None)
            encoding: File encoding
            skip_rows: Number of rows to skip
            max_rows: Maximum rows to read (None for all)
            validate: Whether to validate data quality
            
        Returns:
            ParseResult with collection, mapping, and validation info
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Parsing structural CSV: {file_path}")
        
        # Compute file checksum for provenance
        checksum = compute_file_checksum(file_path)
        
        # Read CSV
        df = self._read_csv(file_path, delimiter, encoding, skip_rows, max_rows)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Auto-detect or use provided column mapping
        if column_mapping is None:
            column_mapping = self._detect_columns(df)
            logger.info(f"Auto-detected column mapping: {column_mapping.to_dict()}")
        
        # Detect format
        if expected_format is None:
            detected_format = self._detect_format(df, column_mapping)
        else:
            detected_format = expected_format
        logger.info(f"Detected format: {detected_format.value}")
        
        # Parse based on format
        collection, rows_processed, rows_skipped, errors, warnings = self._parse_by_format(
            df, column_mapping, detected_format, validate
        )
        
        # Add provenance metadata to all features
        timestamp = datetime.now()
        for feature in collection.all_features:
            feature.source_file = str(file_path)
            feature.source_checksum = checksum
            feature.import_timestamp = timestamp
            feature.parser_version = self.version
        
        # Build metadata
        metadata = {
            'source_file': str(file_path),
            'checksum': checksum,
            'checksum_algorithm': 'sha256',
            'parser_version': self.version,
            'parser_framework_version': PARSER_FRAMEWORK_VERSION,
            'format_detected': detected_format.value,
            'import_timestamp': timestamp.isoformat(),
            'rows_in_file': len(df),
            'columns_in_file': list(df.columns),
            'column_mapping': column_mapping.to_dict(),
        }
        
        return ParseResult(
            collection=collection,
            format_detected=detected_format,
            column_mapping=column_mapping,
            validation_errors=errors,
            validation_warnings=warnings,
            rows_processed=rows_processed,
            rows_skipped=rows_skipped,
            metadata=metadata,
        )
    
    def _read_csv(
        self,
        file_path: Path,
        delimiter: Optional[str],
        encoding: str,
        skip_rows: int,
        max_rows: Optional[int],
    ) -> pd.DataFrame:
        """Read CSV file with auto-detection of delimiter."""
        
        # Auto-detect delimiter if not provided
        if delimiter is None:
            delimiter = self._detect_delimiter(file_path, encoding)
        
        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=encoding,
            skiprows=skip_rows,
            nrows=max_rows,
            on_bad_lines='skip',
        )
        
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        
        return df
    
    def _detect_delimiter(self, file_path: Path, encoding: str) -> str:
        """Auto-detect CSV delimiter."""
        delimiters = [',', '\t', ';', '|', ' ']
        
        with open(file_path, 'r', encoding=encoding) as f:
            first_lines = [f.readline() for _ in range(5)]
        
        sample = ''.join(first_lines)
        
        # Count occurrences of each delimiter
        counts = {d: sample.count(d) for d in delimiters}
        
        # Return most common delimiter
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ','
    
    def _detect_columns(self, df: pd.DataFrame) -> ColumnMapping:
        """Auto-detect column mapping from column names."""
        columns = df.columns.tolist()
        columns_lower = {c.lower(): c for c in columns}
        
        mapping = ColumnMapping()
        
        # Detect coordinate columns
        for target, patterns in COORDINATE_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in columns_lower:
                    setattr(mapping, f'{target}_col', columns_lower[pattern.lower()])
                    break
        
        # Detect orientation columns
        for target, patterns in ORIENTATION_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in columns_lower:
                    setattr(mapping, f'{target}_col', columns_lower[pattern.lower()])
                    break
        
        # Detect lineation columns
        for target, patterns in LINEATION_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in columns_lower:
                    setattr(mapping, f'{target}_col', columns_lower[pattern.lower()])
                    break
        
        # Detect gradient columns
        for target, patterns in GRADIENT_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in columns_lower:
                    setattr(mapping, f'{target}_col', columns_lower[pattern.lower()])
                    break
        
        # Detect feature columns
        for target, patterns in FEATURE_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in columns_lower:
                    setattr(mapping, f'{target}_col', columns_lower[pattern.lower()])
                    break
        
        # Detect polarity column
        for pattern in ['polarity', 'pol', 'overturn', 'overturned']:
            if pattern.lower() in columns_lower:
                mapping.polarity_col = columns_lower[pattern.lower()]
                break
        
        return mapping
    
    def _detect_format(self, df: pd.DataFrame, mapping: ColumnMapping) -> CSVFormat:
        """Detect CSV format from columns and mapping."""
        
        has_coords = all([mapping.x_col, mapping.y_col, mapping.z_col])
        has_orientation = mapping.dip_col is not None
        has_lineation = mapping.plunge_col is not None and mapping.trend_col is not None
        has_gradient = all([mapping.g_x_col, mapping.g_y_col, mapping.g_z_col])
        
        if not has_coords:
            return CSVFormat.UNKNOWN
        
        if has_gradient:
            return CSVFormat.VECTOR_CONSTRAINT
        
        if has_lineation:
            return CSVFormat.FOLD_AXIS
        
        if has_orientation:
            return CSVFormat.ORIENTATION
        
        return CSVFormat.POINT_CLOUD
    
    def _parse_by_format(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        format_type: CSVFormat,
        validate: bool,
    ) -> Tuple[StructuralFeatureCollection, int, int, List[str], List[str]]:
        """Parse DataFrame based on detected format."""
        
        if format_type == CSVFormat.ORIENTATION:
            return self._parse_orientation_format(df, mapping, validate)
        elif format_type == CSVFormat.POINT_CLOUD:
            return self._parse_point_cloud_format(df, mapping, validate)
        elif format_type == CSVFormat.VECTOR_CONSTRAINT:
            return self._parse_vector_constraint_format(df, mapping, validate)
        elif format_type == CSVFormat.FOLD_AXIS:
            return self._parse_fold_axis_format(df, mapping, validate)
        else:
            errors = [f"Unknown format: {format_type}"]
            return StructuralFeatureCollection(), 0, len(df), errors, []
    
    def _parse_orientation_format(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        validate: bool,
    ) -> Tuple[StructuralFeatureCollection, int, int, List[str], List[str]]:
        """
        Parse orientation measurement format.
        
        Expected columns: X, Y, Z, dip, dip_direction, feature_type, feature_name
        """
        collection = StructuralFeatureCollection()
        errors = []
        warnings = []
        rows_processed = 0
        rows_skipped = 0
        
        # Group by feature name and type
        feature_groups = self._group_by_feature(df, mapping)
        
        for (feature_type, feature_name), group_df in feature_groups.items():
            try:
                # Extract coordinates
                coords = self._extract_coordinates(group_df, mapping)
                
                # Extract orientations
                orientations = []
                for idx, row in group_df.iterrows():
                    try:
                        x = float(row[mapping.x_col])
                        y = float(row[mapping.y_col])
                        z = float(row[mapping.z_col])
                        dip = float(row[mapping.dip_col])
                        
                        # Handle dip direction (may be labeled as azimuth)
                        dd_col = mapping.dip_direction_col or mapping.dip_col.replace('dip', 'azimuth') if mapping.dip_col else None
                        if dd_col and dd_col in row:
                            dip_dir = float(row[dd_col])
                        elif mapping.dip_direction_col:
                            dip_dir = float(row[mapping.dip_direction_col])
                        else:
                            dip_dir = 0.0
                            warnings.append(f"Row {idx}: No dip direction found, using 0")
                        
                        polarity = 1
                        if mapping.polarity_col and mapping.polarity_col in row:
                            pol_val = row[mapping.polarity_col]
                            if pol_val in (-1, '-1', 'overturned', 'reversed'):
                                polarity = -1
                        
                        # Validate angles if requested
                        if validate:
                            if not (0 <= dip <= 90):
                                warnings.append(f"Row {idx}: Dip {dip} outside 0-90, clamping")
                                dip = max(0, min(90, dip))
                            if not (0 <= dip_dir <= 360):
                                warnings.append(f"Row {idx}: Dip direction {dip_dir} outside 0-360, wrapping")
                                dip_dir = dip_dir % 360
                        
                        orientations.append(StructuralOrientation(
                            x=x, y=y, z=z,
                            dip=dip, azimuth=dip_dir,
                            polarity=polarity,
                            source='csv_import',
                            metadata={'row_index': idx},
                        ))
                        rows_processed += 1
                        
                    except (ValueError, KeyError) as e:
                        warnings.append(f"Row {idx}: Skipped - {str(e)}")
                        rows_skipped += 1
                
                # Create feature based on type
                feature = self._create_feature(
                    feature_type=feature_type,
                    feature_name=feature_name,
                    surface_points=coords,
                    orientations=orientations,
                )
                
                if feature:
                    collection.add_feature(feature)
                    
            except Exception as e:
                errors.append(f"Feature '{feature_name}': {str(e)}")
        
        return collection, rows_processed, rows_skipped, errors, warnings
    
    def _parse_point_cloud_format(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        validate: bool,
    ) -> Tuple[StructuralFeatureCollection, int, int, List[str], List[str]]:
        """
        Parse point cloud format.
        
        Expected columns: X, Y, Z, feature_type, feature_name
        """
        collection = StructuralFeatureCollection()
        errors = []
        warnings = []
        rows_processed = 0
        rows_skipped = 0
        
        # Group by feature name and type
        feature_groups = self._group_by_feature(df, mapping)
        
        for (feature_type, feature_name), group_df in feature_groups.items():
            try:
                # Extract coordinates
                coords = self._extract_coordinates(group_df, mapping)
                
                if validate and len(coords) < 3:
                    warnings.append(f"Feature '{feature_name}': Only {len(coords)} points, minimum 3 recommended")
                
                rows_processed += len(group_df)
                
                # Create feature based on type
                feature = self._create_feature(
                    feature_type=feature_type,
                    feature_name=feature_name,
                    surface_points=coords,
                    orientations=[],
                )
                
                if feature:
                    collection.add_feature(feature)
                    
            except Exception as e:
                errors.append(f"Feature '{feature_name}': {str(e)}")
                rows_skipped += len(group_df)
        
        return collection, rows_processed, rows_skipped, errors, warnings
    
    def _parse_vector_constraint_format(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        validate: bool,
    ) -> Tuple[StructuralFeatureCollection, int, int, List[str], List[str]]:
        """
        Parse vector constraint (Hermite) format.
        
        Expected columns: X, Y, Z, G_x, G_y, G_z, feature_type, feature_name
        Optional: dip, azimuth (if not provided, computed from gradient)
        """
        collection = StructuralFeatureCollection()
        errors = []
        warnings = []
        rows_processed = 0
        rows_skipped = 0
        
        # Group by feature name and type
        feature_groups = self._group_by_feature(df, mapping)
        
        for (feature_type, feature_name), group_df in feature_groups.items():
            try:
                # Extract coordinates
                coords = self._extract_coordinates(group_df, mapping)
                
                # Extract orientations from gradient vectors
                orientations = []
                for idx, row in group_df.iterrows():
                    try:
                        x = float(row[mapping.x_col])
                        y = float(row[mapping.y_col])
                        z = float(row[mapping.z_col])
                        g_x = float(row[mapping.g_x_col])
                        g_y = float(row[mapping.g_y_col])
                        g_z = float(row[mapping.g_z_col])
                        
                        # Convert gradient to dip/azimuth
                        dip, azimuth, polarity = self._gradient_to_orientation(g_x, g_y, g_z)
                        
                        # Override with explicit values if provided
                        if mapping.dip_col and mapping.dip_col in row:
                            dip = float(row[mapping.dip_col])
                        if mapping.dip_direction_col and mapping.dip_direction_col in row:
                            azimuth = float(row[mapping.dip_direction_col])
                        
                        orientations.append(StructuralOrientation(
                            x=x, y=y, z=z,
                            dip=dip, azimuth=azimuth,
                            polarity=polarity,
                            source='csv_import_gradient',
                            metadata={
                                'row_index': idx,
                                'gradient': (g_x, g_y, g_z),
                            },
                        ))
                        rows_processed += 1
                        
                    except (ValueError, KeyError) as e:
                        warnings.append(f"Row {idx}: Skipped - {str(e)}")
                        rows_skipped += 1
                
                # Create feature based on type
                feature = self._create_feature(
                    feature_type=feature_type,
                    feature_name=feature_name,
                    surface_points=coords,
                    orientations=orientations,
                )
                
                if feature:
                    collection.add_feature(feature)
                    
            except Exception as e:
                errors.append(f"Feature '{feature_name}': {str(e)}")
        
        return collection, rows_processed, rows_skipped, errors, warnings
    
    def _parse_fold_axis_format(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
        validate: bool,
    ) -> Tuple[StructuralFeatureCollection, int, int, List[str], List[str]]:
        """
        Parse fold axis format.
        
        Expected columns: X, Y, Z, plunge, trend, fold_type, feature_name
        """
        collection = StructuralFeatureCollection()
        errors = []
        warnings = []
        rows_processed = 0
        rows_skipped = 0
        
        # Group by feature name and type
        feature_groups = self._group_by_feature(df, mapping)
        
        for (feature_type, feature_name), group_df in feature_groups.items():
            try:
                # Extract coordinates
                coords = self._extract_coordinates(group_df, mapping)
                
                # Extract fold axes
                fold_axes = []
                for idx, row in group_df.iterrows():
                    try:
                        plunge = float(row[mapping.plunge_col])
                        trend = float(row[mapping.trend_col])
                        
                        # Validate angles if requested
                        if validate:
                            if not (0 <= plunge <= 90):
                                warnings.append(f"Row {idx}: Plunge {plunge} outside 0-90, clamping")
                                plunge = max(0, min(90, plunge))
                            if not (0 <= trend <= 360):
                                warnings.append(f"Row {idx}: Trend {trend} outside 0-360, wrapping")
                                trend = trend % 360
                        
                        set_id = None
                        if mapping.set_id_col and mapping.set_id_col in row:
                            set_id = str(row[mapping.set_id_col])
                        
                        fold_axes.append(LineationMeasurement(
                            plunge=plunge,
                            trend=trend,
                            set_id=set_id or feature_name,
                            metadata={'row_index': idx},
                        ))
                        rows_processed += 1
                        
                    except (ValueError, KeyError) as e:
                        warnings.append(f"Row {idx}: Skipped - {str(e)}")
                        rows_skipped += 1
                
                # Determine fold style
                fold_style = self._parse_fold_style(feature_type)
                
                # Create fold feature
                feature = FoldFeature(
                    feature_type=FeatureType.FOLD,
                    name=feature_name,
                    surface_points=coords,
                    fold_axes=fold_axes,
                    fold_style=fold_style,
                )
                
                collection.add_feature(feature)
                    
            except Exception as e:
                errors.append(f"Feature '{feature_name}': {str(e)}")
        
        return collection, rows_processed, rows_skipped, errors, warnings
    
    def _group_by_feature(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """Group DataFrame rows by feature type and name."""
        
        # Add default feature type if column not present
        if mapping.feature_type_col and mapping.feature_type_col in df.columns:
            df = df.copy()
            df['_feature_type'] = df[mapping.feature_type_col].fillna('fault').astype(str).str.lower()
        else:
            df = df.copy()
            df['_feature_type'] = 'fault'  # Default to fault
        
        # Add default feature name if column not present
        if mapping.feature_name_col and mapping.feature_name_col in df.columns:
            df['_feature_name'] = df[mapping.feature_name_col].fillna('unnamed').astype(str)
        else:
            df['_feature_name'] = 'unnamed'
        
        # Group by type and name
        groups = {}
        for (ftype, fname), group in df.groupby(['_feature_type', '_feature_name']):
            groups[(ftype, fname)] = group.drop(columns=['_feature_type', '_feature_name'])
        
        return groups
    
    def _extract_coordinates(
        self,
        df: pd.DataFrame,
        mapping: ColumnMapping,
    ) -> np.ndarray:
        """Extract X, Y, Z coordinates from DataFrame."""
        
        if not all([mapping.x_col, mapping.y_col, mapping.z_col]):
            raise ValueError("Missing coordinate columns")
        
        coords = df[[mapping.x_col, mapping.y_col, mapping.z_col]].values.astype(np.float64)
        
        # Filter out NaN values
        valid_mask = np.all(np.isfinite(coords), axis=1)
        coords = coords[valid_mask]
        
        return coords
    
    def _gradient_to_orientation(
        self,
        g_x: float,
        g_y: float,
        g_z: float,
    ) -> Tuple[float, float, int]:
        """
        Convert gradient vector to dip/azimuth.
        
        Returns:
            (dip, azimuth, polarity)
        """
        # Normalize gradient
        mag = np.sqrt(g_x**2 + g_y**2 + g_z**2)
        if mag < 1e-10:
            return (0.0, 0.0, 1)
        
        g_x /= mag
        g_y /= mag
        g_z /= mag
        
        # Determine polarity (gradient pointing up or down)
        polarity = 1 if g_z >= 0 else -1
        
        # Calculate dip (angle from horizontal)
        dip = np.degrees(np.arccos(abs(g_z)))
        
        # Calculate azimuth (dip direction)
        if abs(g_x) < 1e-10 and abs(g_y) < 1e-10:
            azimuth = 0.0
        else:
            azimuth = np.degrees(np.arctan2(g_x, g_y))
            if azimuth < 0:
                azimuth += 360.0
        
        return (dip, azimuth, polarity)
    
    def _create_feature(
        self,
        feature_type: str,
        feature_name: str,
        surface_points: np.ndarray,
        orientations: List[StructuralOrientation],
    ) -> Optional[StructuralFeature]:
        """Create appropriate feature based on type string."""
        
        ftype_lower = feature_type.lower()
        
        if ftype_lower in FAULT_ALIASES:
            return FaultFeature(
                feature_type=FeatureType.FAULT,
                name=feature_name,
                surface_points=surface_points,
                orientations=orientations,
            )
        elif ftype_lower in FOLD_ALIASES:
            # Convert orientations to limb orientations
            limb_orientations = [o.to_plane_measurement() for o in orientations]
            return FoldFeature(
                feature_type=FeatureType.FOLD,
                name=feature_name,
                surface_points=surface_points,
                limb_orientations=limb_orientations,
                fold_style=self._parse_fold_style(ftype_lower),
            )
        elif ftype_lower in UNCONFORMITY_ALIASES:
            return UnconformityFeature(
                feature_type=FeatureType.UNCONFORMITY,
                name=feature_name,
                surface_points=surface_points,
                orientations=orientations,
            )
        else:
            # Default to fault
            return FaultFeature(
                feature_type=FeatureType.FAULT,
                name=feature_name,
                surface_points=surface_points,
                orientations=orientations,
            )
    
    def _parse_fold_style(self, type_str: str) -> FoldStyle:
        """Parse fold style from type string."""
        type_lower = type_str.lower()
        
        if 'anticline' in type_lower:
            return FoldStyle.ANTICLINE
        elif 'syncline' in type_lower:
            return FoldStyle.SYNCLINE
        elif 'monocline' in type_lower:
            return FoldStyle.MONOCLINE
        elif 'dome' in type_lower:
            return FoldStyle.DOME
        elif 'basin' in type_lower:
            return FoldStyle.BASIN
        else:
            return FoldStyle.UNKNOWN
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get basic information about a structural CSV file."""
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        
        # Read first few rows for preview
        try:
            df = pd.read_csv(file_path, nrows=10)
            columns = list(df.columns)
            row_count_estimate = sum(1 for _ in open(file_path, 'r', encoding='utf-8')) - 1
        except Exception:
            columns = []
            row_count_estimate = 0
        
        # Detect column mapping
        if columns:
            try:
                df_full = pd.read_csv(file_path, nrows=100)
                mapping = self._detect_columns(df_full)
                detected_format = self._detect_format(df_full, mapping)
            except Exception:
                mapping = ColumnMapping()
                detected_format = CSVFormat.UNKNOWN
        else:
            mapping = ColumnMapping()
            detected_format = CSVFormat.UNKNOWN
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'format': self.format_name,
            'detected_csv_format': detected_format.value,
            'columns': columns,
            'row_count_estimate': row_count_estimate,
            'column_mapping': mapping.to_dict(),
            'parser_version': self.version,
            'checksum': compute_file_checksum(file_path),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_structural_csv(
    file_path: Path,
    column_mapping: Optional[ColumnMapping] = None,
    expected_format: Optional[CSVFormat] = None,
    validate: bool = True,
) -> ParseResult:
    """
    Convenience function to parse a structural CSV file.
    
    Args:
        file_path: Path to CSV file
        column_mapping: Optional explicit column mapping
        expected_format: Optional expected format
        validate: Whether to validate data quality
        
    Returns:
        ParseResult with collection and metadata
    """
    parser = StructuralCSVParser()
    return parser.parse(
        file_path=file_path,
        column_mapping=column_mapping,
        expected_format=expected_format,
        validate=validate,
    )


def detect_structural_csv_format(file_path: Path) -> CSVFormat:
    """
    Detect the format of a structural CSV file.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Detected CSVFormat
    """
    parser = StructuralCSVParser()
    df = pd.read_csv(file_path, nrows=100)
    mapping = parser._detect_columns(df)
    return parser._detect_format(df, mapping)

