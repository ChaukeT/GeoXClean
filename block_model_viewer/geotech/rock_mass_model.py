"""
Rock Mass Model for managing geotechnical logging data.

Stores and manages rock-mass properties (RQD, Q, RMR, GSI) from boreholes
and provides interfaces for loading and classification.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from pathlib import Path

from .dataclasses import RockMassPoint, RockMassGrid

logger = logging.getLogger(__name__)


class RockMassModel:
    """
    Manages rock-mass property data from logging.
    
    Stores measurements and provides interfaces for:
    - Loading from CSV/borehole data
    - Computing classification bins
    - Querying properties by location
    """
    
    def __init__(self):
        """Initialize empty rock mass model."""
        self.points: List[RockMassPoint] = []
        self.domains: Dict[str, List[int]] = {}  # domain -> list of point indices
        
    def add_point(self, point: RockMassPoint) -> None:
        """
        Add a rock mass measurement point.
        
        Args:
            point: RockMassPoint instance
        """
        idx = len(self.points)
        self.points.append(point)
        
        # Track by domain
        if point.domain:
            if point.domain not in self.domains:
                self.domains[point.domain] = []
            self.domains[point.domain].append(idx)
        
        logger.debug(f"Added rock mass point at ({point.x:.1f}, {point.y:.1f}, {point.z:.1f})")
    
    def load_from_dataframe(self, df: pd.DataFrame, 
                           x_col: str = 'X', y_col: str = 'Y', z_col: str = 'Z',
                           rqd_col: Optional[str] = None,
                           q_col: Optional[str] = None,
                           rmr_col: Optional[str] = None,
                           gsi_col: Optional[str] = None,
                           domain_col: Optional[str] = None,
                           confidence_col: Optional[str] = None) -> None:
        """
        Load rock mass data from DataFrame.
        
        Args:
            df: DataFrame with geotechnical data
            x_col, y_col, z_col: Coordinate column names
            rqd_col, q_col, rmr_col, gsi_col: Property column names (optional)
            domain_col: Domain identifier column (optional)
            confidence_col: Confidence column (optional)
        """
        required_cols = [x_col, y_col, z_col]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        for idx, row in df.iterrows():
            point = RockMassPoint(
                x=float(row[x_col]),
                y=float(row[y_col]),
                z=float(row[z_col]),
                rqd=float(row[rqd_col]) if rqd_col and rqd_col in df.columns else None,
                q=float(row[q_col]) if q_col and q_col in df.columns else None,
                rmr=float(row[rmr_col]) if rmr_col and rmr_col in df.columns else None,
                gsi=float(row[gsi_col]) if gsi_col and gsi_col in df.columns else None,
                domain=str(row[domain_col]) if domain_col and domain_col in df.columns else None,
                confidence=float(row[confidence_col]) if confidence_col and confidence_col in df.columns else 1.0
            )
            self.add_point(point)
        
        logger.info(f"Loaded {len(self.points)} rock mass points from DataFrame")
    
    def load_from_csv(self, path: Path, **kwargs) -> None:
        """
        Load rock mass data from CSV file.
        
        Args:
            path: Path to CSV file
            **kwargs: Arguments passed to load_from_dataframe
        """
        df = pd.read_csv(path)
        self.load_from_dataframe(df, **kwargs)
    
    def get_points_in_domain(self, domain: str) -> List[RockMassPoint]:
        """Get all points in a specific domain."""
        indices = self.domains.get(domain, [])
        return [self.points[i] for i in indices]
    
    def get_property_array(self, property_name: str) -> np.ndarray:
        """
        Get array of property values for all points.
        
        Args:
            property_name: Property name ('RQD', 'Q', 'RMR', 'GSI')
            
        Returns:
            Array of property values (NaN for missing values)
        """
        prop_map = {
            'RQD': 'rqd',
            'Q': 'q',
            'RMR': 'rmr',
            'GSI': 'gsi'
        }
        
        attr_name = prop_map.get(property_name.upper())
        if not attr_name:
            raise ValueError(f"Unknown property: {property_name}")
        
        values = []
        for point in self.points:
            val = getattr(point, attr_name)
            values.append(val if val is not None else np.nan)
        
        return np.array(values)
    
    def get_coordinates(self) -> np.ndarray:
        """Get array of coordinates for all points."""
        coords = np.array([[p.x, p.y, p.z] for p in self.points])
        return coords
    
    def compute_classification_bins(self, property_name: str = 'RMR') -> np.ndarray:
        """
        Compute quality classification bins based on property.
        
        Args:
            property_name: Property to use for classification ('RMR' or 'Q')
            
        Returns:
            Array of category codes (0-4: Very Poor to Very Good)
        """
        if property_name.upper() == 'RMR':
            values = self.get_property_array('RMR')
        elif property_name.upper() == 'Q':
            # Convert Q to approximate RMR for classification
            q_values = self.get_property_array('Q')
            # Approximate conversion: RMR ≈ 9*ln(Q) + 44 (simplified)
            values = 9 * np.log(np.maximum(q_values, 0.001)) + 44
        else:
            raise ValueError(f"Cannot classify using {property_name}")
        
        categories = np.zeros_like(values, dtype=np.int8)
        categories[values >= 80] = 4  # Very Good
        categories[(values >= 60) & (values < 80)] = 3  # Good
        categories[(values >= 40) & (values < 60)] = 2  # Fair
        categories[(values >= 20) & (values < 40)] = 1  # Poor
        categories[values < 20] = 0  # Very Poor
        
        return categories
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert all points to DataFrame."""
        data = []
        for point in self.points:
            data.append(point.to_dict())
        return pd.DataFrame(data)
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all properties."""
        stats = {}
        
        for prop_name in ['RQD', 'Q', 'RMR', 'GSI']:
            values = self.get_property_array(prop_name)
            valid_values = values[~np.isnan(values)]
            
            if len(valid_values) > 0:
                stats[prop_name] = {
                    'count': len(valid_values),
                    'mean': float(np.mean(valid_values)),
                    'std': float(np.std(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'p25': float(np.percentile(valid_values, 25)),
                    'p50': float(np.percentile(valid_values, 50)),
                    'p75': float(np.percentile(valid_values, 75))
                }
        
        return stats

