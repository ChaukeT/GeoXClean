"""
Seismic Catalogue Handling

Load, filter, and analyze seismic event catalogues.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime

from .dataclasses import SeismicEvent, SeismicCatalogue

logger = logging.getLogger(__name__)


def load_catalog(path: Path, **kwargs) -> SeismicCatalogue:
    """
    Load seismic catalogue from file.
    
    Args:
        path: Path to catalogue file (CSV, ASCII, etc.)
        **kwargs: Format-specific parameters:
            - time_col: Column name for time (default: 'time' or 'datetime')
            - x_col, y_col, z_col: Coordinate columns (default: 'X', 'Y', 'Z')
            - mag_col: Magnitude column (default: 'magnitude' or 'ML' or 'Mw')
            - energy_col: Energy column (optional)
            - id_col: Event ID column (optional, auto-generated if missing)
    
    Returns:
        SeismicCatalogue instance
    """
    logger.info(f"Loading seismic catalogue from {path}")
    
    # Read file
    if path.suffix.lower() == '.csv':
        df = pd.read_csv(path, **kwargs.get('read_kwargs', {}))
    else:
        # Try CSV first, fallback to space-delimited
        try:
            df = pd.read_csv(path, sep=r'\s+', **kwargs.get('read_kwargs', {}))  # Fixed: raw string for regex
        except Exception:
            df = pd.read_csv(path, **kwargs.get('read_kwargs', {}))
    
    # Column name mappings
    time_col = kwargs.get('time_col', None)
    if time_col is None:
        # Try common time column names
        for col in ['time', 'datetime', 'Time', 'DateTime', 'DATE_TIME']:
            if col in df.columns:
                time_col = col
                break
    
    x_col = kwargs.get('x_col', 'X')
    y_col = kwargs.get('y_col', 'Y')
    z_col = kwargs.get('z_col', 'Z')
    mag_col = kwargs.get('mag_col', None)
    if mag_col is None:
        for col in ['magnitude', 'ML', 'Mw', 'MAG', 'mag']:
            if col in df.columns:
                mag_col = col
                break
    
    energy_col = kwargs.get('energy_col', None)
    if energy_col is None:
        for col in ['energy', 'ENERGY', 'E']:
            if col in df.columns:
                energy_col = col
                break
    
    id_col = kwargs.get('id_col', None)
    if id_col is None:
        for col in ['id', 'ID', 'event_id', 'EVENT_ID']:
            if col in df.columns:
                id_col = col
                break
    
    # Parse events
    events = []
    for idx, row in df.iterrows():
        # Parse time
        if time_col and time_col in df.columns:
            try:
                if isinstance(row[time_col], str):
                    time_val = pd.to_datetime(row[time_col])
                else:
                    time_val = row[time_col]
            except Exception:
                logger.warning(f"Failed to parse time for row {idx}, using current time")
                time_val = datetime.now()
        else:
            time_val = datetime.now()
        
        # Get coordinates
        x = float(row[x_col]) if x_col in df.columns else 0.0
        y = float(row[y_col]) if y_col in df.columns else 0.0
        z = float(row[z_col]) if z_col in df.columns else 0.0
        
        # Get magnitude
        magnitude = float(row[mag_col]) if mag_col and mag_col in df.columns else 0.0
        
        # Get optional fields
        energy = float(row[energy_col]) if energy_col and energy_col in df.columns else None
        event_id = str(row[id_col]) if id_col and id_col in df.columns else f"EVENT_{idx}"
        
        event = SeismicEvent(
            id=event_id,
            time=time_val,
            x=x,
            y=y,
            z=z,
            magnitude=magnitude,
            energy=energy,
            mechanism=None,
            quality=None
        )
        events.append(event)
    
    metadata = {
        'source_file': str(path),
        'n_events': len(events),
        'time_range': (min(e.time for e in events), max(e.time for e in events)) if events else None,
        'magnitude_range': (min(e.magnitude for e in events), max(e.magnitude for e in events)) if events else None
    }
    
    catalogue = SeismicCatalogue(events=events, metadata=metadata)
    logger.info(f"Loaded {len(events)} seismic events")
    
    return catalogue


def filter_catalog(
    catalog: SeismicCatalogue,
    time_window: Optional[Tuple[datetime, datetime]] = None,
    mag_range: Optional[Tuple[float, float]] = None,
    bbox: Optional[Tuple[float, float, float, float, float, float]] = None
) -> SeismicCatalogue:
    """
    Filter seismic catalogue by time, magnitude, and spatial bounds.
    
    Args:
        catalog: Input SeismicCatalogue
        time_window: Time window tuple (start, end)
        mag_range: Magnitude range tuple (min, max)
        bbox: Bounding box tuple (xmin, xmax, ymin, ymax, zmin, zmax)
    
    Returns:
        Filtered SeismicCatalogue
    """
    filtered_events = catalog.events.copy()
    
    # Filter by time
    if time_window:
        start_time, end_time = time_window
        filtered_events = [e for e in filtered_events if start_time <= e.time <= end_time]
    
    # Filter by magnitude
    if mag_range:
        min_mag, max_mag = mag_range
        filtered_events = [e for e in filtered_events if min_mag <= e.magnitude <= max_mag]
    
    # Filter by spatial bounds
    if bbox:
        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        filtered_events = [
            e for e in filtered_events
            if xmin <= e.x <= xmax and ymin <= e.y <= ymax and zmin <= e.z <= zmax
        ]
    
    filtered_catalog = SeismicCatalogue(
        events=filtered_events,
        metadata={**catalog.metadata, 'filtered': True}
    )
    
    logger.info(f"Filtered catalogue: {len(catalog.events)} -> {len(filtered_events)} events")
    
    return filtered_catalog


def compute_b_value(catalog: SeismicCatalogue, mag_completeness: Optional[float] = None) -> float:
    """
    Compute b-value from Gutenberg-Richter relationship.
    
    b-value is the slope of log10(N) vs magnitude, where N is cumulative count.
    
    Args:
        catalog: SeismicCatalogue
        mag_completeness: Magnitude of completeness (optional, auto-detected if None)
    
    Returns:
        b-value
    """
    if len(catalog.events) < 10:
        logger.warning("Too few events for reliable b-value calculation")
        return 1.0  # Default b-value
    
    magnitudes = catalog.get_magnitudes()
    
    # Determine completeness magnitude if not provided
    if mag_completeness is None:
        # Simple method: use magnitude bin with highest count
        hist, bins = np.histogram(magnitudes, bins=20)
        mag_completeness = bins[np.argmax(hist)]
    
    # Filter by completeness
    complete_mags = magnitudes[magnitudes >= mag_completeness]
    
    if len(complete_mags) < 5:
        logger.warning("Too few complete events for b-value calculation")
        return 1.0
    
    # Compute cumulative counts
    mag_bins = np.arange(mag_completeness, np.max(complete_mags) + 0.1, 0.1)
    cumulative_counts = []
    
    for mag in mag_bins:
        count = np.sum(complete_mags >= mag)
        if count > 0:
            cumulative_counts.append((mag, count))
    
    if len(cumulative_counts) < 3:
        return 1.0
    
    # Linear regression: log10(N) = a - b*M
    mags = np.array([m for m, _ in cumulative_counts])
    log_counts = np.log10(np.array([c for _, c in cumulative_counts]))
    
    # Simple linear fit
    coeffs = np.polyfit(mags, log_counts, 1)
    b_value = -coeffs[0]  # Negative of slope
    
    logger.info(f"Computed b-value: {b_value:.2f} (completeness: {mag_completeness:.2f})")
    
    return max(0.5, min(2.0, b_value))  # Clamp to reasonable range


def compute_event_rate(catalog: SeismicCatalogue, time_bin_days: float = 1.0) -> pd.DataFrame:
    """
    Compute event rate vs time.
    
    Args:
        catalog: SeismicCatalogue
        time_bin_days: Time bin size in days
    
    Returns:
        DataFrame with columns: time, count, cumulative_count
    """
    if not catalog.events:
        return pd.DataFrame(columns=['time', 'count', 'cumulative_count'])
    
    times = catalog.get_times()
    start_time = min(times)
    end_time = max(times)
    
    # Create time bins
    from datetime import timedelta
    time_bins = []
    current_time = start_time
    while current_time <= end_time:
        time_bins.append(current_time)
        current_time += timedelta(days=time_bin_days)
    
    # Count events per bin
    counts = []
    for i in range(len(time_bins) - 1):
        bin_start = time_bins[i]
        bin_end = time_bins[i + 1]
        count = sum(1 for t in times if bin_start <= t < bin_end)
        counts.append(count)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': time_bins[:-1],
        'count': counts,
        'cumulative_count': np.cumsum(counts)
    })
    
    return df

