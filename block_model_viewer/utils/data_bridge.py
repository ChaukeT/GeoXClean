"""
Data Bridge for Inter-Panel Communication

This module provides a centralized data sharing mechanism for communication
between different panels in the application (e.g., Underground Mining panel
to ESG Dashboard).

Features:
- Shared data cache with event notifications
- Type-safe data transfer
- Automatic data validation
- Signal-based updates (Qt signals)
- Data versioning and timestamps
- Data conversion utilities (BlockModel <-> DataFrame, ResourceResult <-> DataFrame, etc.)

Author: Mining Software Team
Date: 2025
"""

import logging
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
try:
    from PyQt6.QtCore import QObject, pyqtSignal, QCoreApplication
    _PYQT_AVAILABLE = True
except Exception:
    _PYQT_AVAILABLE = False
    QObject = object  # type: ignore
    def pyqtSignal(*_args, **_kwargs):  # type: ignore
        return None

logger = logging.getLogger(__name__)


# ============================================================================
# Data Conversion Utilities - Step 10: Canonical API
# ============================================================================

def blockmodel_to_dataframe(block_model: Any, columns: Optional[List[str]] = None, copy: bool = True) -> pd.DataFrame:
    """
    Convert BlockModel to pandas DataFrame (canonical API).
    
    Args:
        block_model: BlockModel instance
        columns: Optional list of property columns to include
        copy: Whether to copy data (False for views when safe)
        
    Returns:
        DataFrame with block data (lower_snake_case column names)
    """
    if hasattr(block_model, 'to_dataframe'):
        df = block_model.to_dataframe()
        if columns:
            # Filter to requested columns
            available_cols = [c for c in columns if c in df.columns]
            # STEP 18: Use view when safe (no copy needed for column selection)
            df = df[available_cols] if copy else df.loc[:, available_cols]
        # Normalize column names to lower_snake_case
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df.copy() if copy else df
    elif isinstance(block_model, pd.DataFrame):
        # STEP 18: Only copy if requested
        df = block_model.copy() if copy else block_model
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    else:
        raise ValueError(f"Unsupported block_model type: {type(block_model)}")


def dataframe_to_blockmodel(df: pd.DataFrame, template: Any = None) -> Any:
    """
    Convert DataFrame to BlockModel (requires template for geometry).
    
    Args:
        df: DataFrame with block data
        template: Optional BlockModel template to copy geometry from
        
    Returns:
        BlockModel instance (if template provided) or DataFrame
    """
    if template is not None and hasattr(template, 'from_dataframe'):
        return template.from_dataframe(df)
    # Return DataFrame if no template
    return df.copy()


def resource_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert ResourceResult to DataFrame (canonical API).
    
    Args:
        result: ResourceResult instance
        
    Returns:
        DataFrame with resource summary (lower_snake_case column names)
    """
    if hasattr(result, 'summary_df'):
        df = result.summary_df.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif isinstance(result, dict) and 'summary_df' in result:
        df = result['summary_df'].copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    else:
        raise ValueError(f"Unsupported ResourceResult type: {type(result)}")


def classification_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert ClassificationResult to DataFrame (canonical API).
    
    Args:
        result: ClassificationResult instance
        
    Returns:
        DataFrame with classified blocks (lower_snake_case column names)
    """
    if hasattr(result, 'classified_df'):
        df = result.classified_df.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif isinstance(result, dict) and 'classified_df' in result:
        df = result['classified_df'].copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    else:
        raise ValueError(f"Unsupported ClassificationResult type: {type(result)}")


def variogram_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert Variogram3DResult to DataFrame (canonical API).
    
    Args:
        result: Variogram3DResult instance
        
    Returns:
        DataFrame with variogram data (lower_snake_case column names)
    """
    if hasattr(result, 'table'):
        df = result.table.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        # Add model parameters as metadata columns
        if hasattr(result, 'model'):
            df['model_type'] = result.model
        if hasattr(result, 'nugget'):
            df['nugget'] = result.nugget
        if hasattr(result, 'sill'):
            df['sill'] = result.sill
        if hasattr(result, 'prange'):
            df['range'] = result.prange
        return df
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    else:
        raise ValueError(f"Unsupported VariogramResult type: {type(result)}")


def kriging_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert KrigingResult to DataFrame (canonical API).
    
    Args:
        result: KrigingResult dict or object
        
    Returns:
        DataFrame with kriging estimates and variances
    """
    if isinstance(result, dict):
        # Extract grid data
        if 'estimates' in result and 'grid_x' in result:
            grid_x = result['grid_x']
            grid_y = result['grid_y']
            grid_z = result['grid_z']
            estimates = result['estimates']
            variances = result.get('variances', None)
            
            # Flatten arrays
            n_points = estimates.size
            df_data = {
                'x': grid_x.ravel(order='F')[:n_points],
                'y': grid_y.ravel(order='F')[:n_points],
                'z': grid_z.ravel(order='F')[:n_points],
                'estimate': estimates.ravel(order='F')
            }
            if variances is not None:
                df_data['variance'] = variances.ravel(order='F')[:n_points]
            
            df = pd.DataFrame(df_data)
            if 'variable' in result:
                df['variable'] = result['variable']
            return df
        else:
            # Try to convert dict directly
            return pd.DataFrame([result])
    elif hasattr(result, 'estimates'):
        # KrigingResult object
        df_data = {
            'x': result.grid_x.ravel(order='F'),
            'y': result.grid_y.ravel(order='F'),
            'z': result.grid_z.ravel(order='F'),
            'estimate': result.estimates.ravel(order='F')
        }
        if hasattr(result, 'variances'):
            df_data['variance'] = result.variances.ravel(order='F')
        return pd.DataFrame(df_data)
    else:
        raise ValueError(f"Unsupported KrigingResult type: {type(result)}")


def sgsim_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert SGSIMResult to DataFrame (canonical API).
    
    Args:
        result: SGSIMResult dict or object
        
    Returns:
        DataFrame with SGSIM realizations
    """
    if isinstance(result, dict):
        if 'realizations' in result:
            # Multiple realizations
            realizations = result['realizations']
            if isinstance(realizations, np.ndarray):
                # Shape: (n_realizations, nx, ny, nz)
                n_real, nx, ny, nz = realizations.shape
                df_data = {
                    'realization_id': np.repeat(np.arange(n_real), nx * ny * nz),
                    'x_index': np.tile(np.arange(nx), n_real * ny * nz),
                    'y_index': np.tile(np.repeat(np.arange(ny), nx), n_real * nz),
                    'z_index': np.tile(np.repeat(np.arange(nz), nx * ny), n_real),
                    'value': realizations.ravel()
                }
                return pd.DataFrame(df_data)
        # Fallback: try to convert dict
        return pd.DataFrame([result])
    elif hasattr(result, 'realizations'):
        realizations = result.realizations
        n_real, nx, ny, nz = realizations.shape
        df_data = {
            'realization_id': np.repeat(np.arange(n_real), nx * ny * nz),
            'x_index': np.tile(np.arange(nx), n_real * ny * nz),
            'y_index': np.tile(np.repeat(np.arange(ny), nx), n_real * nz),
            'z_index': np.tile(np.repeat(np.arange(nz), nx * ny), n_real),
            'value': realizations.ravel()
        }
        return pd.DataFrame(df_data)
    else:
        raise ValueError(f"Unsupported SGSIMResult type: {type(result)}")


def swath_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert SwathResult to DataFrame (canonical API).
    
    Args:
        result: SwathResult dict or object
        
    Returns:
        DataFrame with swath analysis data
    """
    if isinstance(result, dict):
        if 'swath_data' in result:
            df = result['swath_data'].copy()
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            return df
        # Try to convert dict directly
        return pd.DataFrame([result])
    elif hasattr(result, 'swath_data'):
        df = result.swath_data.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    else:
        raise ValueError(f"Unsupported SwathResult type: {type(result)}")


def kmeans_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert KMeansResult to DataFrame (canonical API).
    
    Args:
        result: KMeansResult dict or object
        
    Returns:
        DataFrame with cluster assignments and centroids
    """
    if isinstance(result, dict):
        if 'cluster_labels' in result:
            df_data = {
                'cluster_id': result['cluster_labels']
            }
            if 'centroids' in result:
                centroids = result['centroids']
                for i in range(centroids.shape[1]):
                    df_data[f'centroid_dim_{i}'] = centroids[:, i]
            return pd.DataFrame(df_data)
        return pd.DataFrame([result])
    elif hasattr(result, 'cluster_labels'):
        df_data = {'cluster_id': result.cluster_labels}
        if hasattr(result, 'centroids'):
            centroids = result.centroids
            for i in range(centroids.shape[1]):
                df_data[f'centroid_dim_{i}'] = centroids[:, i]
        return pd.DataFrame(df_data)
    else:
        raise ValueError(f"Unsupported KMeansResult type: {type(result)}")


def irr_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert IRRResult to DataFrame (canonical API).
    
    Args:
        result: IRRResult instance or dict
        
    Returns:
        DataFrame with IRR results (lower_snake_case column names)
    """
    if isinstance(result, dict):
        # Extract key metrics
        data = {
            'irr_alpha': [result.get('irr_alpha', 0)],
            'alpha_target': [result.get('alpha_target', 0)],
            'satisfaction_rate': [result.get('satisfaction_rate', 0)],
            'mean_npv': [result.get('mean_npv', 0)],
            'std_npv': [result.get('std_npv', 0)],
            'min_npv': [result.get('min_npv', 0)],
            'max_npv': [result.get('max_npv', 0)],
            'num_scenarios': [result.get('num_scenarios', 0)],
            'iterations': [result.get('iterations', 0)]
        }
        return pd.DataFrame(data)
    elif hasattr(result, 'irr_alpha'):
        # IRRResult dataclass
        data = {
            'irr_alpha': [result.irr_alpha],
            'alpha_target': [result.alpha_target],
            'satisfaction_rate': [result.satisfaction_rate],
            'mean_npv': [result.mean_npv],
            'std_npv': [result.std_npv],
            'min_npv': [result.min_npv],
            'max_npv': [result.max_npv],
            'num_scenarios': [result.num_scenarios],
            'iterations': [result.iterations]
        }
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported IRRResult type: {type(result)}")


def pit_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert PitResult to DataFrame (canonical API).
    
    Args:
        result: PitResult dict or object
        
    Returns:
        DataFrame with pit shell data
    """
    if isinstance(result, dict):
        if 'ultimate_pit' in result:
            df = result['ultimate_pit'].copy()
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            return df
        # Try to convert dict directly
        return pd.DataFrame([result])
    elif hasattr(result, 'ultimate_pit'):
        df = result.ultimate_pit.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    else:
        raise ValueError(f"Unsupported PitResult type: {type(result)}")


def schedule_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert ScheduleResult to DataFrame (canonical API).
    
    Args:
        result: ScheduleResult dict or object
        
    Returns:
        DataFrame with schedule data
    """
    if isinstance(result, dict):
        if 'schedule' in result:
            df = result['schedule'].copy()
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            return df
        elif 'best_schedule' in result:
            df = result['best_schedule'].copy()
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
            return df
        return pd.DataFrame([result])
    elif hasattr(result, 'schedule'):
        df = result.schedule.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif hasattr(result, 'best_schedule') and result.best_schedule is not None:
        df = result.best_schedule.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    elif isinstance(result, pd.DataFrame):
        df = result.copy()
        df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
        return df
    else:
        raise ValueError(f"Unsupported ScheduleResult type: {type(result)}")


def uncertainty_result_to_dataframe(result: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert UncertaintyResult to multiple DataFrames (canonical API).
    
    Args:
        result: UncertaintyResult dict or object (MonteCarloResults, BootstrapResult, etc.)
        
    Returns:
        Dictionary of DataFrames: {'summary': ..., 'realizations': ..., ...}
    """
    frames = {}
    
    if isinstance(result, dict):
        if 'summary_stats' in result:
            frames['summary'] = result['summary_stats'].copy()
        if 'simulations' in result:
            # Convert list of SimulationResult to DataFrame
            sims = result['simulations']
            if sims and len(sims) > 0:
                sim_data = []
                for sim in sims:
                    if isinstance(sim, dict):
                        sim_data.append(sim)
                    elif hasattr(sim, '__dict__'):
                        sim_data.append(sim.__dict__)
                if sim_data:
                    frames['realizations'] = pd.DataFrame(sim_data)
        if 'percentiles' in result:
            frames['percentiles'] = result['percentiles'].copy()
    elif hasattr(result, 'summary_stats'):
        frames['summary'] = result.summary_stats.copy() if result.summary_stats is not None else pd.DataFrame()
        if hasattr(result, 'simulations') and result.simulations:
            sim_data = []
            for sim in result.simulations:
                if hasattr(sim, '__dict__'):
                    sim_data.append(sim.__dict__)
            if sim_data:
                frames['realizations'] = pd.DataFrame(sim_data)
        if hasattr(result, 'percentiles') and result.percentiles is not None:
            frames['percentiles'] = result.percentiles.copy()
    
    # Normalize column names
    for key in frames:
        frames[key].columns = [col.lower().replace(' ', '_').replace('-', '_') for col in frames[key].columns]
    
    return frames if frames else {'summary': pd.DataFrame()}


# Backward compatibility aliases
to_dataframe = blockmodel_to_dataframe
resources_to_dataframe = resource_result_to_dataframe
irr_to_dataframe = irr_result_to_dataframe


class DataType(Enum):
    """Types of data that can be shared between panels."""
    PRODUCTION_SCHEDULE = "production_schedule"
    STOPE_LIST = "stope_list"
    BLOCK_MODEL = "block_model"
    EQUIPMENT_PLAN = "equipment_plan"
    WATER_BALANCE = "water_balance"
    CARBON_FOOTPRINT = "carbon_footprint"
    WASTE_TRACKING = "waste_tracking"
    COMPLIANCE_DATA = "compliance_data"
    CUSTOM = "custom"


@dataclass
class DataPackage:
    """Container for shared data with metadata."""
    data_type: DataType
    data: Any
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return (f"DataPackage(type={self.data_type.value}, source={self.source}, "
                f"timestamp={self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"version={self.version})")


class DataBridge(QObject):
    """
    Centralized data bridge for inter-panel communication.
    
    This singleton class manages shared data between different panels,
    providing event notifications when data is updated.
    """
    
    # Signals for data updates (use generic object types to avoid PyQt type registration issues)
    # Emitting: (data_type: DataType, data_package: DataPackage)
    if _PYQT_AVAILABLE:
        data_updated = pyqtSignal(object, object)
        schedule_updated = pyqtSignal(object)
        stopes_updated = pyqtSignal(object)
        block_model_updated = pyqtSignal(object)
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - only one DataBridge instance."""
        if cls._instance is None:
            logger.debug("Creating new DataBridge singleton instance (pre-init)")
            cls._instance = super(DataBridge, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize data bridge."""
        if self._initialized:
            return
        
        # If PyQt not available or no app instance yet, we will operate in "fallback" mode (no signals)
        if _PYQT_AVAILABLE and QCoreApplication.instance() is not None:
            super().__init__()
            self._signals_enabled = True
        else:
            self._signals_enabled = False
        self._data_cache: Dict[DataType, DataPackage] = {}
        self._subscribers: Dict[DataType, List[Callable]] = {}
        self._initialized = True
        
    logger.info("DataBridge initialized")
    
    def publish(
        self,
        data_type: DataType,
        data: Any,
        source: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Publish data to the bridge.
        
        Args:
            data_type: Type of data being published
            data: The actual data
            source: Source panel/module name
            metadata: Optional metadata dict
            
        Returns:
            True if published successfully
        """
        try:
            # Create data package
            package = DataPackage(
                data_type=data_type,
                data=data,
                source=source,
                metadata=metadata or {}
            )
            
            # Update version if data already exists
            if data_type in self._data_cache:
                package.version = self._data_cache[data_type].version + 1
            
            # Store in cache
            self._data_cache[data_type] = package
            
            # Emit signals
            if self._signals_enabled:
                try:
                    self.data_updated.emit(data_type, package)
                    if data_type == DataType.PRODUCTION_SCHEDULE:
                        self.schedule_updated.emit(data)
                    elif data_type == DataType.STOPE_LIST:
                        self.stopes_updated.emit(data)
                    elif data_type == DataType.BLOCK_MODEL:
                        self.block_model_updated.emit(data)
                except Exception as sig_err:
                    logger.warning(f"Signal emission failed (fallback mode engaged): {sig_err}")
                    self._signals_enabled = False
            
            # Notify subscribers
            if data_type in self._subscribers:
                for callback in self._subscribers[data_type]:
                    try:
                        callback(package)
                    except Exception as e:
                        logger.error(f"Error in subscriber callback: {e}")
            
            logger.info(f"Published {data_type.value} from {source} (v{package.version})")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing data: {e}", exc_info=True)
            return False
    
    def get(
        self,
        data_type: DataType,
        default: Any = None
    ) -> Optional[Any]:
        """
        Get data from the bridge.
        
        Args:
            data_type: Type of data to retrieve
            default: Default value if data not found
            
        Returns:
            The data, or default if not found
        """
        package = self._data_cache.get(data_type)
        if package:
            return package.data
        return default
    
    def get_package(
        self,
        data_type: DataType
    ) -> Optional[DataPackage]:
        """
        Get full data package with metadata.
        
        Args:
            data_type: Type of data to retrieve
            
        Returns:
            DataPackage or None
        """
        return self._data_cache.get(data_type)
    
    def has(self, data_type: DataType) -> bool:
        """
        Check if data type exists in cache.
        
        Args:
            data_type: Type to check
            
        Returns:
            True if data exists
        """
        return data_type in self._data_cache
    
    def subscribe(
        self,
        data_type: DataType,
        callback: Callable[[DataPackage], None]
    ):
        """
        Subscribe to data updates.
        
        Args:
            data_type: Type of data to subscribe to
            callback: Function to call when data updates (receives DataPackage)
        """
        if data_type not in self._subscribers:
            self._subscribers[data_type] = []
        
        self._subscribers[data_type].append(callback)
        logger.debug(f"New subscriber for {data_type.value}")
    
    def unsubscribe(
        self,
        data_type: DataType,
        callback: Callable[[DataPackage], None]
    ):
        """
        Unsubscribe from data updates.
        
        Args:
            data_type: Type of data
            callback: Callback function to remove
        """
        if data_type in self._subscribers:
            try:
                self._subscribers[data_type].remove(callback)
                logger.debug(f"Removed subscriber for {data_type.value}")
            except ValueError:
                pass
    
    def clear(self, data_type: Optional[DataType] = None):
        """
        Clear cached data.
        
        Args:
            data_type: Specific type to clear, or None to clear all
        """
        if data_type:
            if data_type in self._data_cache:
                del self._data_cache[data_type]
                logger.info(f"Cleared {data_type.value}")
        else:
            self._data_cache.clear()
            logger.info("Cleared all cached data")
    
    def get_all_types(self) -> List[DataType]:
        """
        Get list of all data types currently cached.
        
        Returns:
            List of DataType enums
        """
        return list(self._data_cache.keys())
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about cached data.
        
        Returns:
            Dict with cache statistics
        """
        info = {
            'total_items': len(self._data_cache),
            'data_types': [dt.value for dt in self._data_cache.keys()],
            'items': []
        }
        
        for data_type, package in self._data_cache.items():
            info['items'].append({
                'type': data_type.value,
                'source': package.source,
                'timestamp': package.timestamp.isoformat(),
                'version': package.version,
                'metadata': package.metadata
            })
        
        return info
    
    def export_schedule_for_esg(self) -> Optional[Dict[str, Any]]:
        """
        Export production schedule in format suitable for ESG Dashboard.
        
        Returns:
            Dict with schedule data or None
        """
        schedule_pkg = self.get_package(DataType.PRODUCTION_SCHEDULE)
        stopes_pkg = self.get_package(DataType.STOPE_LIST)
        
        if not schedule_pkg:
            return None
        
        export_data = {
            'schedule': schedule_pkg.data,
            'stopes': stopes_pkg.data if stopes_pkg else None,
            'timestamp': schedule_pkg.timestamp.isoformat(),
            'source': schedule_pkg.source,
            'metadata': schedule_pkg.metadata
        }
        
        return export_data
    
    def import_schedule_from_underground(
        self,
        schedule: List,
        stopes: Optional[List] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Import schedule data from Underground Mining panel.
        
        Args:
            schedule: Production schedule list
            stopes: Optional stope list
            metadata: Optional metadata
            
        Returns:
            True if imported successfully
        """
        success = True
        
        # Publish schedule
        success &= self.publish(
            DataType.PRODUCTION_SCHEDULE,
            schedule,
            "UndergroundMining",
            metadata
        )
        
        # Publish stopes if provided
        if stopes:
            success &= self.publish(
                DataType.STOPE_LIST,
                stopes,
                "UndergroundMining",
                metadata
            )
        
        return success


# Global bridge instance
_bridge = None


class FallbackDataBridge:
    """Minimal non-Qt fallback for environments where QObject signals crash.

    Provides the same API subset used by panels but without Qt dependency.
    """
    def __init__(self):
        self._cache: Dict[DataType, DataPackage] = {}
        self._subs: Dict[DataType, List[Callable]] = {}
        logger.info("FallbackDataBridge initialized (no Qt signals)")
        # Provide stub signal-like attributes for UI code expecting .connect
        class _SignalStub:
            def connect(self, *_args, **_kwargs):
                logger.debug("SignalStub.connect called - no-op")
        self.schedule_updated = _SignalStub()
        self.stopes_updated = _SignalStub()
        self.block_model_updated = _SignalStub()

    def publish(self, data_type: DataType, data: Any, source: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        pkg = DataPackage(data_type, data, source, metadata=metadata or {})
        if data_type in self._cache:
            pkg.version = self._cache[data_type].version + 1
        self._cache[data_type] = pkg
        for cb in self._subs.get(data_type, []):
            try:
                cb(pkg)
            except Exception as e:
                logger.error(f"Fallback subscriber error: {e}")
        logger.info(f"[Fallback] Published {data_type.value} from {source} (v{pkg.version})")
        return True

    def get(self, data_type: DataType, default: Any = None) -> Any:
        return self._cache.get(data_type, DataPackage(data_type, default, "fallback")).data if data_type in self._cache else default

    def get_package(self, data_type: DataType) -> Optional[DataPackage]:
        return self._cache.get(data_type)

    def has(self, data_type: DataType) -> bool:
        return data_type in self._cache

    def subscribe(self, data_type: DataType, callback: Callable[[DataPackage], None]):
        self._subs.setdefault(data_type, []).append(callback)
        logger.debug(f"[Fallback] New subscriber for {data_type.value}")

    def clear(self, data_type: Optional[DataType] = None):
        if data_type:
            self._cache.pop(data_type, None)
        else:
            self._cache.clear()

    def get_all_types(self) -> List[DataType]:
        return list(self._cache.keys())

    def get_info(self) -> Dict[str, Any]:
        return {
            'total_items': len(self._cache),
            'data_types': [dt.value for dt in self._cache.keys()],
        }

    # Compatibility helpers
    def import_schedule_from_underground(self, schedule: List, stopes: Optional[List] = None, metadata: Optional[Dict] = None) -> bool:
        ok = self.publish(DataType.PRODUCTION_SCHEDULE, schedule, "UndergroundMining", metadata)
        if stopes:
            ok &= self.publish(DataType.STOPE_LIST, stopes, "UndergroundMining", metadata)
        return ok


def get_data_bridge() -> DataBridge:
    """
    Get the global DataBridge instance.
    
    Returns:
        DataBridge singleton
    """
    global _bridge
    if _bridge is None:
        # Temporarily force fallback implementation due to PyQt QObject crash during instantiation
        logger.warning("Using FallbackDataBridge (Qt DataBridge disabled due to crash)")
        _bridge = FallbackDataBridge()
    return _bridge


# Convenience functions

def publish_schedule(
    schedule: List,
    stopes: Optional[List] = None,
    source: str = "Unknown",
    **metadata
) -> bool:
    """
    Quick function to publish production schedule.
    
    Args:
        schedule: Production schedule
        stopes: Optional stope list
        source: Source module name
        **metadata: Additional metadata
        
    Returns:
        True if successful
    """
    bridge = get_data_bridge()
    return bridge.import_schedule_from_underground(schedule, stopes, metadata)


def get_schedule() -> Optional[List]:
    """
    Quick function to get production schedule.
    
    Returns:
        Schedule list or None
    """
    bridge = get_data_bridge()
    return bridge.get(DataType.PRODUCTION_SCHEDULE)


def get_stopes() -> Optional[List]:
    """
    Quick function to get stope list.
    
    Returns:
        Stope list or None
    """
    bridge = get_data_bridge()
    return bridge.get(DataType.STOPE_LIST)


def clear_all_data():
    """Clear all cached data in the bridge."""
    bridge = get_data_bridge()
    bridge.clear()


# ============================================================================
# Geotechnical Data Conversions (STEP 19)
# ============================================================================

def rock_mass_grid_to_dataframe(grid: Any) -> pd.DataFrame:
    """
    Convert RockMassGrid to DataFrame.
    
    Args:
        grid: RockMassGrid instance
        
    Returns:
        DataFrame with geotechnical properties
    """
    from ..geotech.dataclasses import RockMassGrid
    
    if not isinstance(grid, RockMassGrid):
        raise ValueError(f"Expected RockMassGrid, got {type(grid)}")
    
    data = {}
    
    # Add coordinates if available
    if grid.grid_definition.get('n_blocks'):
        n_blocks = grid.grid_definition['n_blocks']
        # Create placeholder coordinates (would need actual block positions)
        data['block_id'] = np.arange(n_blocks)
    
    # Add properties
    if grid.rqd is not None:
        data['rqd'] = grid.rqd
    if grid.q is not None:
        data['q'] = grid.q
    if grid.rmr is not None:
        data['rmr'] = grid.rmr
    if grid.gsi is not None:
        data['gsi'] = grid.gsi
    if grid.quality_category is not None:
        data['quality_category'] = grid.quality_category
    
    return pd.DataFrame(data)


def stope_stability_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert StopeStabilityResult or result dict to DataFrame.
    
    Args:
        result: StopeStabilityResult instance or dict
        
    Returns:
        DataFrame with stability analysis results
    """
    from ..geotech.dataclasses import StopeStabilityResult
    
    if isinstance(result, dict):
        data = {
            'stability_number': [result.get('stability_number', 0.0)],
            'factor_of_safety': [result.get('factor_of_safety', 0.0)],
            'probability_of_instability': [result.get('probability_of_instability', 0.0)],
            'stability_class': [result.get('stability_class', 'Unknown')],
            'recommended_support_class': [result.get('recommended_support_class', 'Unknown')],
            'notes': [result.get('notes', '')]
        }
    elif isinstance(result, StopeStabilityResult):
        data = {
            'stability_number': [result.stability_number],
            'factor_of_safety': [result.factor_of_safety],
            'probability_of_instability': [result.probability_of_instability],
            'stability_class': [result.stability_class],
            'recommended_support_class': [result.recommended_support_class],
            'notes': [result.notes]
        }
    else:
        raise ValueError(f"Unsupported result type: {type(result)}")
    
    return pd.DataFrame(data)


def slope_risk_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert SlopeRiskResult or result dict to DataFrame.
    
    Args:
        result: SlopeRiskResult instance or dict
        
    Returns:
        DataFrame with slope risk analysis results
    """
    from ..geotech.dataclasses import SlopeRiskResult
    
    if isinstance(result, dict):
        data = {
            'risk_index': [result.get('risk_index', 0.0)],
            'qualitative_class': [result.get('qualitative_class', 'Unknown')],
            'probability_of_failure': [result.get('probability_of_failure', 0.0)],
            'notes': [result.get('notes', '')]
        }
    elif isinstance(result, SlopeRiskResult):
        data = {
            'risk_index': [result.risk_index],
            'qualitative_class': [result.qualitative_class],
            'probability_of_failure': [result.probability_of_failure],
            'notes': [result.notes]
        }
    else:
        raise ValueError(f"Unsupported result type: {type(result)}")
    
    return pd.DataFrame(data)


def geotech_mc_result_to_dataframes(mc_result: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert GeotechMCResult to dict of DataFrames.
    
    Args:
        mc_result: GeotechMCResult instance or dict
        
    Returns:
        Dict with 'summary' and 'distributions' DataFrames
    """
    from ..geotech.dataclasses import GeotechMCResult
    
    if isinstance(mc_result, dict):
        summary_stats = mc_result.get('summary_stats', {})
        stability_numbers = mc_result.get('stability_numbers')
        risk_indices = mc_result.get('risk_indices')
    elif isinstance(mc_result, GeotechMCResult):
        summary_stats = mc_result.summary_stats
        stability_numbers = mc_result.stability_numbers
        risk_indices = mc_result.risk_indices
    else:
        raise ValueError(f"Unsupported MC result type: {type(mc_result)}")
    
    # Summary DataFrame
    summary_data = {}
    if 'stability_number' in summary_stats:
        sn_stats = summary_stats['stability_number']
        summary_data.update({
            'statistic': ['mean', 'std', 'min', 'max', 'p10', 'p50', 'p90'],
            'stability_number': [
                sn_stats.get('mean', 0.0),
                sn_stats.get('std', 0.0),
                sn_stats.get('min', 0.0),
                sn_stats.get('max', 0.0),
                sn_stats.get('p10', 0.0),
                sn_stats.get('p50', 0.0),
                sn_stats.get('p90', 0.0)
            ]
        })
    
    if 'risk_index' in summary_stats:
        ri_stats = summary_stats['risk_index']
        if 'statistic' not in summary_data:
            summary_data['statistic'] = ['mean', 'std', 'min', 'max', 'p10', 'p50', 'p90']
        summary_data['risk_index'] = [
            ri_stats.get('mean', 0.0),
            ri_stats.get('std', 0.0),
            ri_stats.get('min', 0.0),
            ri_stats.get('max', 0.0),
            ri_stats.get('p10', 0.0),
            ri_stats.get('p50', 0.0),
            ri_stats.get('p90', 0.0)
        ]
    
    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
    
    # Distributions DataFrame
    distributions_data = {}
    if stability_numbers is not None:
        distributions_data['stability_number'] = stability_numbers if isinstance(stability_numbers, np.ndarray) else np.array(stability_numbers)
    if risk_indices is not None:
        distributions_data['risk_index'] = risk_indices if isinstance(risk_indices, np.ndarray) else np.array(risk_indices)
    
    distributions_df = pd.DataFrame(distributions_data) if distributions_data else pd.DataFrame()
    
    return {
        'summary': summary_df,
        'distributions': distributions_df
    }


# ============================================================================
# Seismic Data Conversions (STEP 20)
# ============================================================================

def seismic_catalogue_to_dataframe(catalog: Any) -> pd.DataFrame:
    """
    Convert SeismicCatalogue to DataFrame.
    
    Args:
        catalog: SeismicCatalogue instance
        
    Returns:
        DataFrame with seismic events
    """
    from ..seismic.dataclasses import SeismicCatalogue
    
    if not isinstance(catalog, SeismicCatalogue):
        raise ValueError(f"Expected SeismicCatalogue, got {type(catalog)}")
    
    data = []
    for event in catalog.events:
        data.append(event.to_dict())
    
    return pd.DataFrame(data)


def hazard_volume_to_dataframe(volume: Any) -> pd.DataFrame:
    """
    Convert HazardVolume to DataFrame.
    
    Args:
        volume: HazardVolume instance
        
    Returns:
        DataFrame with hazard indices and coordinates
    """
    from ..seismic.dataclasses import HazardVolume
    
    if not isinstance(volume, HazardVolume):
        raise ValueError(f"Expected HazardVolume, got {type(volume)}")
    
    coords = volume.grid_definition.get('coordinates')
    if coords is None:
        raise ValueError("HazardVolume grid_definition must contain 'coordinates'")
    
    coords_array = np.array(coords)
    
    data = {
        'x': coords_array[:, 0],
        'y': coords_array[:, 1],
        'z': coords_array[:, 2],
        'hazard_index': volume.hazard_index
    }
    
    return pd.DataFrame(data)


def rockburst_index_results_to_dataframe(results: List[Any]) -> pd.DataFrame:
    """
    Convert list of RockburstIndexResult to DataFrame.
    
    Args:
        results: List of RockburstIndexResult instances or dicts
        
    Returns:
        DataFrame with rockburst index results
    """
    from ..seismic.dataclasses import RockburstIndexResult
    
    data = []
    for result in results:
        if isinstance(result, dict):
            data.append({
                'x': result.get('location', [0, 0, 0])[0],
                'y': result.get('location', [0, 0, 0])[1],
                'z': result.get('location', [0, 0, 0])[2],
                'index_value': result.get('index_value', 0.0),
                'index_class': result.get('index_class', 'Unknown'),
                'contributing_events': result.get('contributing_events', 0),
                'notes': result.get('notes', '')
            })
        elif isinstance(result, RockburstIndexResult):
            data.append({
                'x': result.location[0],
                'y': result.location[1],
                'z': result.location[2],
                'index_value': result.index_value,
                'index_class': result.index_class,
                'contributing_events': result.contributing_events,
                'notes': result.notes
            })
        else:
            raise ValueError(f"Unsupported result type: {type(result)}")
    
    return pd.DataFrame(data)


def seismic_mc_result_to_dataframes(mc_result: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert SeismicMCResult to dict of DataFrames.
    
    Args:
        mc_result: SeismicMCResult instance or dict
        
    Returns:
        Dict with 'summary' and 'realisations' DataFrames
    """
    from ..seismic.dataclasses import SeismicMCResult
    
    if isinstance(mc_result, dict):
        summary_stats = mc_result.get('summary_stats', {})
        exceedance_curve = mc_result.get('exceedance_curve', {})
        realisations = mc_result.get('realisations', [])
    elif isinstance(mc_result, SeismicMCResult):
        summary_stats = mc_result.summary_stats
        exceedance_curve = mc_result.exceedance_curve
        realisations = mc_result.realisations
    else:
        raise ValueError(f"Unsupported MC result type: {type(mc_result)}")
    
    # Summary DataFrame
    summary_data = {}
    if 'hazard_index' in summary_stats:
        hi_stats = summary_stats['hazard_index']
        summary_data = {
            'statistic': ['mean', 'std', 'min', 'max', 'p10', 'p50', 'p90'],
            'hazard_index': [
                hi_stats.get('mean', 0.0),
                hi_stats.get('std', 0.0),
                hi_stats.get('min', 0.0),
                hi_stats.get('max', 0.0),
                hi_stats.get('p10', 0.0),
                hi_stats.get('p50', 0.0),
                hi_stats.get('p90', 0.0)
            ]
        }
    
    summary_df = pd.DataFrame(summary_data) if summary_data else pd.DataFrame()
    
    # Exceedance curve DataFrame
    if exceedance_curve:
        exceedance_df = pd.DataFrame({
            'threshold': list(exceedance_curve.keys()),
            'exceedance_probability': list(exceedance_curve.values())
        })
    else:
        exceedance_df = pd.DataFrame(columns=['threshold', 'exceedance_probability'])
    
    # Realisations DataFrame (flattened)
    realisations_data = []
    for i, vol in enumerate(realisations):
        vol_df = hazard_volume_to_dataframe(vol)
        vol_df['realization'] = i
        realisations_data.append(vol_df)
    
    realisations_df = pd.concat(realisations_data, ignore_index=True) if realisations_data else pd.DataFrame()
    
    return {
        'summary': summary_df,
        'exceedance_curve': exceedance_df,
        'realisations': realisations_df
    }


# ============================================================================
# Schedule Risk Data Conversions (STEP 21)
# ============================================================================

def schedule_risk_profile_to_dataframe(profile: Any) -> pd.DataFrame:
    """
    Convert ScheduleRiskProfile to DataFrame.
    
    Args:
        profile: ScheduleRiskProfile instance
        
    Returns:
        DataFrame with one row per period
    """
    from ..risk.risk_dataclasses import ScheduleRiskProfile
    
    if not isinstance(profile, ScheduleRiskProfile):
        raise ValueError(f"Expected ScheduleRiskProfile, got {type(profile)}")
    
    data = []
    for period in profile.periods:
        data.append(period.to_dict())
    
    return pd.DataFrame(data)


def risk_scenario_comparison_to_dataframes(comparison: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert RiskScenarioComparison to dict of DataFrames.
    
    Args:
        comparison: RiskScenarioComparison instance
        
    Returns:
        Dict with per-schedule metrics and comparison metrics
    """
    from ..risk.risk_dataclasses import RiskScenarioComparison
    
    if not isinstance(comparison, RiskScenarioComparison):
        raise ValueError(f"Expected RiskScenarioComparison, got {type(comparison)}")
    
    # Base profile DataFrame
    base_df = schedule_risk_profile_to_dataframe(comparison.base_profile)
    base_df['schedule_id'] = comparison.base_profile.schedule_id
    
    # Alternative profiles DataFrames
    alternative_dfs = {}
    for alt_profile in comparison.alternative_profiles:
        alt_df = schedule_risk_profile_to_dataframe(alt_profile)
        alt_df['schedule_id'] = alt_profile.schedule_id
        alternative_dfs[alt_profile.schedule_id] = alt_df
    
    # Comparison metrics DataFrame
    comparison_data = []
    for alt_id, metrics in comparison.metrics.items():
        delta_metrics = metrics.get('delta_metrics', {})
        period_deltas = metrics.get('period_deltas', [])
        
        # Summary row
        comparison_data.append({
            'alternative_id': alt_id,
            'delta_combined_risk_mean': delta_metrics.get('combined_risk_mean', 0.0),
            'delta_combined_risk_pct_change': delta_metrics.get('combined_risk_pct_change', 0.0),
            'delta_combined_risk_max': delta_metrics.get('combined_risk_max', 0.0),
            'n_periods': len(period_deltas)
        })
    
    comparison_df = pd.DataFrame(comparison_data) if comparison_data else pd.DataFrame()
    
    # Period-by-period deltas (flattened)
    period_delta_data = []
    for alt_id, metrics in comparison.metrics.items():
        period_deltas = metrics.get('period_deltas', [])
        for delta in period_deltas:
            period_delta_data.append({
                'alternative_id': alt_id,
                'period': delta.get('period'),
                'delta_risk': delta.get('delta_risk', 0.0),
                'base_risk': delta.get('base_risk', 0.0),
                'alt_risk': delta.get('alt_risk', 0.0)
            })
    
    period_deltas_df = pd.DataFrame(period_delta_data) if period_delta_data else pd.DataFrame()
    
    return {
        'base_profile': base_df,
        'alternative_profiles': alternative_dfs,
        'comparison_summary': comparison_df,
        'period_deltas': period_deltas_df
    }


# ============================================================================
# Geostatistics Data Conversions (STEP 22)
# ============================================================================

def uk_result_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Universal Kriging result to DataFrame.
    
    Args:
        result: Result dict with 'estimates', 'variances', 'grid_x', 'grid_y', 'grid_z'
    
    Returns:
        DataFrame with coordinates, estimates, and variances
    """
    grid_x = result.get('grid_x')
    grid_y = result.get('grid_y')
    grid_z = result.get('grid_z')
    estimates = result.get('estimates')
    variances = result.get('variances')
    
    if grid_x is None or estimates is None:
        raise ValueError("Missing required data in UK result")
    
    # Flatten grid coordinates
    coords = np.column_stack([
        grid_x.ravel(),
        grid_y.ravel(),
        grid_z.ravel()
    ])
    
    # Flatten estimates and variances
    estimates_flat = estimates.ravel()
    variances_flat = variances.ravel() if variances is not None else np.full_like(estimates_flat, np.nan)
    
    df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'Estimate': estimates_flat,
        'Variance': variances_flat
    })
    
    return df.dropna()


def cokriging_result_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Co-Kriging result to DataFrame.
    
    Args:
        result: Result dict with 'estimates', 'variances', 'grid_x', 'grid_y', 'grid_z', 'metadata'
    
    Returns:
        DataFrame with coordinates, estimates, and variances
    """
    grid_x = result.get('grid_x')
    grid_y = result.get('grid_y')
    grid_z = result.get('grid_z')
    estimates = result.get('estimates')
    variances = result.get('variances')
    metadata = result.get('metadata', {})
    
    if grid_x is None or estimates is None:
        raise ValueError("Missing required data in Co-Kriging result")
    
    # Flatten grid coordinates
    coords = np.column_stack([
        grid_x.ravel(),
        grid_y.ravel(),
        grid_z.ravel()
    ])
    
    # Flatten estimates and variances
    estimates_flat = estimates.ravel()
    variances_flat = variances.ravel() if variances is not None else np.full_like(estimates_flat, np.nan)
    
    primary_name = metadata.get('primary_name', 'Primary')
    secondary_name = metadata.get('secondary_name', 'Secondary')
    
    df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        f'{primary_name}_Estimate': estimates_flat,
        f'{primary_name}_Variance': variances_flat
    })
    
    return df.dropna()


def ik_result_to_dataframe(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert Indicator Kriging result to dict of DataFrames.
    
    Args:
        result: Result dict with 'probabilities', 'thresholds', 'grid_x', 'grid_y', 'grid_z', etc.
    
    Returns:
        Dict with:
        - 'probabilities': DataFrame with coordinates and probability columns per threshold
        - 'median': DataFrame with median estimates (if available)
        - 'mean': DataFrame with mean estimates (if available)
    """
    grid_x = result.get('grid_x')
    grid_y = result.get('grid_y')
    grid_z = result.get('grid_z')
    probabilities = result.get('probabilities')
    thresholds = result.get('thresholds')
    median = result.get('median')
    mean = result.get('mean')
    
    if grid_x is None or probabilities is None:
        raise ValueError("Missing required data in IK result")
    
    # Flatten grid coordinates
    coords = np.column_stack([
        grid_x.ravel(),
        grid_y.ravel(),
        grid_z.ravel()
    ])
    
    # Build probabilities DataFrame
    prob_data = {
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2]
    }
    
    # Add probability columns for each threshold
    if probabilities.ndim == 4:
        # Reshape from (nx, ny, nz, n_thresholds) to (n_blocks, n_thresholds)
        n_blocks = probabilities.shape[0] * probabilities.shape[1] * probabilities.shape[2]
        probabilities_flat = probabilities.reshape((n_blocks, probabilities.shape[3]))
    else:
        probabilities_flat = probabilities
    
    for k, threshold in enumerate(thresholds):
        prob_data[f'P_le_{threshold:g}'] = probabilities_flat[:, k]
    
    prob_df = pd.DataFrame(prob_data)
    
    result_dict = {
        'probabilities': prob_df.dropna()
    }
    
    # Add median if available
    if median is not None:
        median_flat = median.ravel()
        median_df = pd.DataFrame({
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2],
            'Median': median_flat
        })
        result_dict['median'] = median_df.dropna()
    
    # Add mean if available
    if mean is not None:
        mean_flat = mean.ravel()
        mean_df = pd.DataFrame({
            'X': coords[:, 0],
            'Y': coords[:, 1],
            'Z': coords[:, 2],
            'Mean': mean_flat
        })
        result_dict['mean'] = mean_df.dropna()
    
        return result_dict


# ============================================================================
# Variogram Assistant and Bayesian Kriging Data Conversions (STEP 23)
# ============================================================================

def variogram_assistant_result_to_dataframe(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert Variogram Assistant result to DataFrames.
    
    Args:
        result: VariogramAssistantResult dict with candidates, best_model, experimental_variogram
    
    Returns:
        Dict with 'candidates' DataFrame and 'best_model' DataFrame
    """
    candidates = result.get('candidates', [])
    best_model = result.get('best_model', {})
    experimental = result.get('experimental_variogram', {})
    
    # Candidates DataFrame
    candidates_data = []
    for i, cand in enumerate(candidates):
        candidates_data.append({
            'rank': i + 1,
            'model_type': cand.get('model_type', 'N/A'),
            'ranges': ', '.join([f"{r:.2f}" for r in cand.get('ranges', [])]),
            'sills': ', '.join([f"{s:.3f}" for s in cand.get('sills', [])]),
            'nugget': cand.get('nugget', 0.0),
            'score_sse': cand.get('score_sse', 0.0),
            'score_cv_rmse': cand.get('score_cv_rmse', 0.0),
            'score_total': cand.get('score_total', 0.0)
        })
    
    candidates_df = pd.DataFrame(candidates_data)
    
    # Best model DataFrame
    best_model_df = pd.DataFrame([{
        'model_type': best_model.get('model_type', 'N/A'),
        'ranges': ', '.join([f"{r:.2f}" for r in best_model.get('ranges', [])]),
        'sills': ', '.join([f"{s:.3f}" for s in best_model.get('sills', [])]),
        'nugget': best_model.get('nugget', 0.0),
        'score_sse': best_model.get('score_sse', 0.0),
        'score_cv_rmse': best_model.get('score_cv_rmse', 0.0),
        'score_total': best_model.get('score_total', 0.0)
    }])
    
    # Experimental variogram DataFrame
    exp_df = None
    if experimental:
        lag_distances = experimental.get('lag_distances', np.array([]))
        semivariances = experimental.get('semivariances', np.array([]))
        pair_counts = experimental.get('pair_counts', np.array([]))
        
        if len(lag_distances) > 0:
            exp_df = pd.DataFrame({
                'lag_distance': lag_distances,
                'semivariance': semivariances,
                'pair_count': pair_counts
            })
    
    result_dict = {
        'candidates': candidates_df,
        'best_model': best_model_df
    }
    
    if exp_df is not None:
        result_dict['experimental'] = exp_df
    
    return result_dict


def soft_dataset_to_dataframe(soft_data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert SoftDataSet to DataFrame.
    
    Args:
        soft_data: SoftDataSet dict with points list
    
    Returns:
        DataFrame with soft data points (X, Y, Z, mean, variance)
    """
    points = soft_data.get('points', [])
    
    if len(points) == 0:
        return pd.DataFrame(columns=['X', 'Y', 'Z', 'mean', 'variance'])
    
    data = []
    for p in points:
        data.append({
            'X': p.get('x', 0.0),
            'Y': p.get('y', 0.0),
            'Z': p.get('z', 0.0),
            'mean': p.get('mean', 0.0),
            'variance': p.get('variance', 0.0)
        })
    
    return pd.DataFrame(data)


def bayesian_kriging_result_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Bayesian Kriging result to DataFrame.
    
    Args:
        result: Bayesian kriging result dict with estimates, variances, grid coordinates
    
    Returns:
        DataFrame with coordinates, estimates, and variances
    """
    grid_x = result.get('grid_x')
    grid_y = result.get('grid_y')
    grid_z = result.get('grid_z')
    estimates = result.get('estimates')
    variances = result.get('variances')
    method = result.get('method', 'bayesian_kriging')
    
    if grid_x is None or estimates is None:
        raise ValueError("Missing required data in Bayesian Kriging result")
    
    # Flatten grid coordinates
    if grid_x.ndim == 3:
        coords = np.column_stack([
            grid_x.ravel(),
            grid_y.ravel(),
            grid_z.ravel()
        ])
    else:
        coords = np.column_stack([grid_x, grid_y, grid_z])
    
    estimates_flat = np.array(estimates).ravel()
    variances_flat = np.array(variances).ravel() if variances is not None else np.full_like(estimates_flat, np.nan)
    
    df = pd.DataFrame({
        'X': coords[:, 0],
        'Y': coords[:, 1],
        'Z': coords[:, 2],
        'estimate': estimates_flat,
        'variance': variances_flat,
        'method': method
    })
    
    return df.dropna()


# ============================================================================
# IK-SGSIM, Co-SGSIM, and Economic Uncertainty Data Conversions (STEP 24)
# ============================================================================

def ik_sgsim_result_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert IK-SGSIM result to DataFrame.
    
    Args:
        result: IK-SGSIM result dict with realization_names
    
    Returns:
        DataFrame with realisation mapping
    """
    realization_names = result.get('realization_names', [])
    
    df = pd.DataFrame({
        'realisation_index': range(len(realization_names)),
        'property_name': realization_names,
        'method': 'IK-SGSIM'
    })
    
    return df


def cosgsim_result_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert Co-SGSIM result to DataFrame.
    
    Args:
        result: Co-SGSIM result dict with realization_names per variable
    
    Returns:
        DataFrame with realisation mapping per variable
    """
    realization_names = result.get('realization_names', {})
    
    data = []
    for var_name, names in realization_names.items():
        for idx, prop_name in enumerate(names):
            data.append({
                'variable': var_name,
                'realisation_index': idx,
                'property_name': prop_name,
                'method': 'Co-SGSIM'
            })
    
    return pd.DataFrame(data)


def grade_realisation_set_to_dataframe(grade_set: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert GradeRealisationSet to DataFrame.
    
    Args:
        grade_set: GradeRealisationSet dict
    
    Returns:
        DataFrame with mapping realisation index → property name
    """
    property_name = grade_set.get('property_name', 'unknown')
    realisation_names = grade_set.get('realisation_names', [])
    
    df = pd.DataFrame({
        'property_name': property_name,
        'realisation_index': range(len(realisation_names)),
        'realisation_property': realisation_names
    })
    
    return df


def economic_uncertainty_result_to_dataframe(result: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert Economic Uncertainty result to DataFrames.
    
    Args:
        result: Economic uncertainty result dict
    
    Returns:
        Dict with 'samples', 'summary', 'schedule_summaries' DataFrames
    """
    npv_samples = np.array(result.get('npv_samples', []))
    irr_samples = np.array(result.get('irr_samples', []))
    pit_shell_ids = result.get('pit_shell_ids', [])
    schedule_profiles = result.get('schedule_profiles', [])
    summary_stats = result.get('summary_stats', {})
    
    # NPV/IRR samples DataFrame
    samples_df = pd.DataFrame({
        'realisation_index': range(len(npv_samples)),
        'npv': npv_samples,
        'irr': irr_samples,
        'pit_shell_id': pit_shell_ids
    })
    
    # Summary statistics DataFrame
    summary_data = []
    for metric_type in ['npv', 'irr']:
        stats = summary_stats.get(metric_type, {})
        for stat_name, stat_value in stats.items():
            summary_data.append({
                'metric_type': metric_type.upper(),
                'statistic': stat_name,
                'value': stat_value
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Schedule summaries DataFrame
    schedule_data = []
    for idx, profile in enumerate(schedule_profiles):
        if profile:
            schedule_id = profile.get('schedule_id', f'schedule_{idx}')
            stats = profile.get('summary_stats', {})
            schedule_data.append({
                'realisation_index': idx,
                'schedule_id': schedule_id,
                'n_periods': stats.get('n_periods', 0),
                'total_tonnage': stats.get('total_tonnage', 0.0),
                'total_metal': stats.get('total_metal', 0.0)
            })
    
    schedule_summaries_df = pd.DataFrame(schedule_data)
    
    return {
        'samples': samples_df,
        'summary': summary_df,
        'schedule_summaries': schedule_summaries_df
    }
    
    # ============================================================================
    # Research Mode Data Conversions (STEP 25)
    # ============================================================================
    
    def experiment_run_to_dataframe(run_result: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert ExperimentRunResult to DataFrame.
        
        Args:
            run_result: ExperimentRunResult dict
        
        Returns:
            DataFrame with one row per instance
        """
        from ..research.reporting import experiment_to_dataframe
        from ..research.runner import ExperimentRunResult
        
        # Reconstruct ExperimentRunResult
        exp_result = ExperimentRunResult(
            definition_id=run_result.get('definition_id', ''),
            results=run_result.get('results', []),
            metrics=run_result.get('metrics', []),
            metadata=run_result.get('metadata', {})
        )
        
        return experiment_to_dataframe(exp_result)


# ============================================================================
# STEP 26: Drillhole, Geology, Structural Data Conversions
# ============================================================================

def drillhole_database_to_dataframe(db: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert DrillholeDatabase to dictionary of DataFrames.
    
    Args:
        db: DrillholeDatabase instance
    
    Returns:
        Dictionary with keys: 'collars', 'surveys', 'assays', 'lithology'
    """
    try:
        from ..drillholes.datamodel import DrillholeDatabase
        
        if not isinstance(db, DrillholeDatabase):
            logger.warning(f"Expected DrillholeDatabase, got {type(db)}")
            return {}
        
        return db.to_dataframe()
    
    except Exception as e:
        logger.error(f"Error converting drillhole database to DataFrames: {e}")
        return {}


def structural_dataset_to_dataframe(dataset: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert StructuralDataset to DataFrames.
    
    Args:
        dataset: StructuralDataset instance
    
    Returns:
        Dictionary with 'planes' and 'lineations' DataFrames
    """
    try:
        from ..structural.datasets import StructuralDataset
        
        if not isinstance(dataset, StructuralDataset):
            logger.warning(f"Expected StructuralDataset, got {type(dataset)}")
            return {}
        
        result = {}
        
        # Planes DataFrame
        if dataset.planes:
            planes_data = []
            for plane in dataset.planes:
                planes_data.append({
                    'dip': plane.dip,
                    'dip_direction': plane.dip_direction,
                    'set_id': plane.set_id,
                })
            result['planes'] = pd.DataFrame(planes_data)
        
        # Lineations DataFrame
        if dataset.lineations:
            lineations_data = []
            for lineation in dataset.lineations:
                lineations_data.append({
                    'plunge': lineation.plunge,
                    'trend': lineation.trend,
                    'set_id': lineation.set_id,
                })
            result['lineations'] = pd.DataFrame(lineations_data)
        
        return result
    
    except Exception as e:
        logger.error(f"Error converting structural dataset to DataFrames: {e}")
        return {}


def domain_model_to_dataframe(domain_model: Any) -> pd.DataFrame:
    """
    Convert DomainModel to DataFrame.
    
    Args:
        domain_model: DomainModel instance
    
    Returns:
        DataFrame with domain definitions
    """
    try:
        from ..geology.domains import DomainModel
        
        if not isinstance(domain_model, DomainModel):
            logger.warning(f"Expected DomainModel, got {type(domain_model)}")
            return pd.DataFrame()
        
        domains_data = []
        for code, domain in domain_model.domains.items():
            domains_data.append({
                'code': domain.code,
                'name': domain.name,
                'colour_r': domain.colour[0],
                'colour_g': domain.colour[1],
                'colour_b': domain.colour[2],
                'domain_type': domain.domain_type,
            })
        
        return pd.DataFrame(domains_data)
    
    except Exception as e:
        logger.error(f"Error converting domain model to DataFrame: {e}")
        return pd.DataFrame()


def wireframe_to_dataframe(wireframe: Any) -> pd.DataFrame:
    """
    Convert Wireframe to DataFrame description.
    
    Args:
        wireframe: Wireframe instance
    
    Returns:
        DataFrame with wireframe metadata
    """
    try:
        from ..geology.wireframes import Wireframe
        
        if not isinstance(wireframe, Wireframe):
            logger.warning(f"Expected Wireframe, got {type(wireframe)}")
            return pd.DataFrame()
        
        return pd.DataFrame([{
            'name': wireframe.name or 'unnamed',
            'domain_code': wireframe.domain_code,
            'vertex_count': len(wireframe.vertices),
            'face_count': len(wireframe.faces),
        }])
    
    except Exception as e:
        logger.error(f"Error converting wireframe to DataFrame: {e}")
        return pd.DataFrame()


# ============================================================================
# STEP 27: Slope Stability & Geotechnical Data Conversions
# ============================================================================

def geotech_material_library_to_dataframe(library: Any) -> pd.DataFrame:
    """
    Convert GeotechMaterialLibrary to DataFrame.
    
    Args:
        library: GeotechMaterialLibrary instance
    
    Returns:
        DataFrame with material properties
    """
    try:
        from ..geotech_common.material_properties import GeotechMaterialLibrary
        
        if not isinstance(library, GeotechMaterialLibrary):
            logger.warning(f"Expected GeotechMaterialLibrary, got {type(library)}")
            return pd.DataFrame()
        
        materials_data = []
        for name, material in library.materials.items():
            materials_data.append({
                'name': material.name,
                'unit_weight': material.unit_weight,
                'friction_angle': material.friction_angle,
                'cohesion': material.cohesion,
                'tensile_strength': material.tensile_strength,
                'hoek_brown_mb': material.hoek_brown_mb,
                'hoek_brown_s': material.hoek_brown_s,
                'hoek_brown_a': material.hoek_brown_a,
                'water_condition': material.water_condition,
            })
        
        return pd.DataFrame(materials_data)
    
    except Exception as e:
        logger.error(f"Error converting material library to DataFrame: {e}")
        return pd.DataFrame()


def slope_set_to_dataframe(slope_set: Any) -> pd.DataFrame:
    """
    Convert SlopeSet to DataFrame.
    
    Args:
        slope_set: SlopeSet instance
    
    Returns:
        DataFrame with slope sector information
    """
    try:
        from ..geotech_common.slope_geometry import SlopeSet
        
        if not isinstance(slope_set, SlopeSet):
            logger.warning(f"Expected SlopeSet, got {type(slope_set)}")
            return pd.DataFrame()
        
        sectors_data = []
        for sector in slope_set.sectors:
            sectors_data.append({
                'id': sector.id,
                'toe_x': sector.toe_point[0],
                'toe_y': sector.toe_point[1],
                'toe_z': sector.toe_point[2],
                'crest_x': sector.crest_point[0],
                'crest_y': sector.crest_point[1],
                'crest_z': sector.crest_point[2],
                'height': sector.height,
                'dip': sector.dip,
                'dip_direction': sector.dip_direction,
                'bench_height': sector.bench_height,
                'berm_width': sector.berm_width,
                'overall_slope_angle': sector.overall_slope_angle,
                'domain_code': sector.domain_code,
                'material_name': sector.material_name,
            })
        
        return pd.DataFrame(sectors_data)
    
    except Exception as e:
        logger.error(f"Error converting slope set to DataFrame: {e}")
        return pd.DataFrame()


def lem_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert LEM2DResult or LEM3DResult to DataFrame.
    
    Args:
        result: LEM2DResult or LEM3DResult instance
    
    Returns:
        DataFrame with FOS and geometry summary
    """
    try:
        from ..geotech_pit.limit_equilibrium_2d import LEM2DResult
        from ..geotech_pit.limit_equilibrium_3d import LEM3DResult
        
        if isinstance(result, LEM2DResult):
            return pd.DataFrame([{
                'method': '2D',
                'fos': result.fos,
                'converged': result.converged,
                'iterations': result.iterations,
                'surface_type': result.surface.surface_type,
                'surface_length': result.surface.get_length(),
            }])
        elif isinstance(result, LEM3DResult):
            return pd.DataFrame([{
                'method': '3D',
                'fos': result.fos,
                'surface_type': result.surface.surface_type,
                'surface_area': result.surface.get_area(),
            }])
        else:
            logger.warning(f"Expected LEM2DResult or LEM3DResult, got {type(result)}")
            return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error converting LEM result to DataFrame: {e}")
        return pd.DataFrame()


def prob_slope_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert ProbSlopeResult to DataFrame.
    
    Args:
        result: ProbSlopeResult instance
    
    Returns:
        DataFrame with FOS samples and statistics
    """
    try:
        from ..geotech_pit.slope_probabilistic import ProbSlopeResult
        
        if not isinstance(result, ProbSlopeResult):
            logger.warning(f"Expected ProbSlopeResult, got {type(result)}")
            return pd.DataFrame()
        
        # Create DataFrame with FOS samples
        fos_df = pd.DataFrame({
            'fos': result.fos_samples
        })
        
        # Add statistics as metadata row
        stats_row = {
            'fos': result.fos_stats.get('mean', 0.0),
            'probability_of_failure': result.probability_of_failure,
            'fos_std': result.fos_stats.get('std', 0.0),
            'fos_min': result.fos_stats.get('min', 0.0),
            'fos_max': result.fos_stats.get('max', 0.0),
            'fos_p5': result.fos_stats.get('p5', 0.0),
            'fos_p95': result.fos_stats.get('p95', 0.0),
        }
        
        return pd.DataFrame([stats_row])
    
    except Exception as e:
        logger.error(f"Error converting probabilistic slope result to DataFrame: {e}")
        return pd.DataFrame()


# ============================================================================
# Geometallurgy conversions (STEP 28)
# ============================================================================

def geomet_domain_map_to_dataframe(domain_map: Any) -> pd.DataFrame:
    """
    Convert GeometDomainMap to DataFrame.
    
    Args:
        domain_map: GeometDomainMap instance
    
    Returns:
        DataFrame with ore type definitions
    """
    try:
        from ..geomet.domains_links import GeometDomainMap
        
        if not isinstance(domain_map, GeometDomainMap):
            logger.warning(f"Expected GeometDomainMap, got {type(domain_map)}")
            return pd.DataFrame()
        
        data = []
        for code, ore_type in domain_map.ore_types.items():
            data.append({
                'code': code,
                'name': ore_type.name,
                'geology_domains': ','.join(ore_type.geology_domains),
                'texture_class': ore_type.texture_class or '',
                'hardness_class': ore_type.hardness_class or '',
                'density': ore_type.density or np.nan,
                'notes': ore_type.notes
            })
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error converting GeometDomainMap to DataFrame: {e}")
        return pd.DataFrame()


def plant_config_to_dataframe(plant_config: Any) -> pd.DataFrame:
    """
    Convert PlantConfig to DataFrame/dict representation.
    
    Args:
        plant_config: PlantConfig instance
    
    Returns:
        DataFrame with plant configuration summary
    """
    try:
        from ..geomet.plant_response import PlantConfig
        
        if not isinstance(plant_config, PlantConfig):
            logger.warning(f"Expected PlantConfig, got {type(plant_config)}")
            return pd.DataFrame()
        
        data = [{
            'plant_name': plant_config.name,
            'circuit_type': plant_config.comminution_config.circuit_type,
            'target_p80': plant_config.comminution_config.target_p80,
            'f80': plant_config.comminution_config.f80 or np.nan,
            'max_throughput': plant_config.constraints.get('max_throughput', np.nan),
            'n_separation_stages': len(plant_config.separation_configs)
        }]
        
        return pd.DataFrame(data)
    
    except Exception as e:
        logger.error(f"Error converting PlantConfig to DataFrame: {e}")
        return pd.DataFrame()


def geomet_block_attributes_to_dataframe(geomet_attrs: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert GeometBlockAttributes to DataFrame(s) for export.
    
    Args:
        geomet_attrs: GeometBlockAttributes instance
    
    Returns:
        Dictionary with 'ore_types' and 'recoveries' DataFrames
    """
    try:
        from ..geomet.geomet_block_model import GeometBlockAttributes
        
        if not isinstance(geomet_attrs, GeometBlockAttributes):
            logger.warning(f"Expected GeometBlockAttributes, got {type(geomet_attrs)}")
            return {}
        
        n_blocks = len(geomet_attrs.ore_type_code)
        
        # Main attributes DataFrame
        main_df = pd.DataFrame({
            'ore_type': geomet_attrs.ore_type_code,
            'tonnage_factor': geomet_attrs.plant_tonnage_factor,
            'specific_energy': geomet_attrs.plant_specific_energy
        })
        
        # Recoveries DataFrame
        recovery_data = {'ore_type': geomet_attrs.ore_type_code}
        for element, recovery_array in geomet_attrs.recovery_by_element.items():
            recovery_data[f'rec_{element}'] = recovery_array
        
        recoveries_df = pd.DataFrame(recovery_data)
        
        # Concentrate grades DataFrame
        grade_data = {'ore_type': geomet_attrs.ore_type_code}
        for element, grade_array in geomet_attrs.concentrate_grade_by_element.items():
            grade_data[f'conc_{element}'] = grade_array
        
        grades_df = pd.DataFrame(grade_data)
        
        return {
            'main': main_df,
            'recoveries': recoveries_df,
            'concentrate_grades': grades_df
        }
    
    except Exception as e:
        logger.error(f"Error converting GeometBlockAttributes to DataFrame: {e}")
        return {}


# Grade Control & Reconciliation conversion functions (STEP 29)
def gc_model_to_dataframe(gc_model: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert GCModel to DataFrame(s) for export.
    
    Args:
        gc_model: GCModel instance
    
    Returns:
        Dictionary with 'grid' and 'properties' DataFrames
    """
    try:
        from ..grade_control.support_model import GCModel
        
        if not isinstance(gc_model, GCModel):
            logger.warning(f"Expected GCModel, got {type(gc_model)}")
            return {}
        
        # Grid definition DataFrame
        grid_df = pd.DataFrame({
            'origin_x': [gc_model.grid.origin[0]],
            'origin_y': [gc_model.grid.origin[1]],
            'origin_z': [gc_model.grid.origin[2]],
            'dx': [gc_model.grid.dx],
            'dy': [gc_model.grid.dy],
            'dz': [gc_model.grid.dz],
            'nx': [gc_model.grid.nx],
            'ny': [gc_model.grid.ny],
            'nz': [gc_model.grid.nz]
        })
        
        # Properties DataFrame
        block_centers = gc_model.grid.get_block_centers()
        n_blocks = gc_model.grid.get_block_count()
        
        props_data = {
            'x': block_centers[:, 0],
            'y': block_centers[:, 1],
            'z': block_centers[:, 2]
        }
        
        for prop_name, prop_values in gc_model.properties.items():
            props_data[prop_name] = prop_values
        
        props_df = pd.DataFrame(props_data)
        
        return {
            'grid': grid_df,
            'properties': props_df
        }
    
    except Exception as e:
        logger.error(f"Error converting GCModel to DataFrame: {e}")
        return {}


def digline_set_to_dataframe(diglines: Any) -> pd.DataFrame:
    """
    Convert DiglineSet to DataFrame for export.
    
    Args:
        diglines: DiglineSet instance
    
    Returns:
        DataFrame with polygon data
    """
    try:
        from ..grade_control.digpolygons import DiglineSet
        
        if not isinstance(diglines, DiglineSet):
            logger.warning(f"Expected DiglineSet, got {type(diglines)}")
            return pd.DataFrame()
        
        rows = []
        for polygon in diglines.polygons:
            rows.append({
                'id': polygon.id,
                'bench_code': polygon.bench_code,
                'elevation': polygon.elevation,
                'ore_flag': polygon.ore_flag,
                'material_type': polygon.material_type or '',
                'target_destination': polygon.target_destination or '',
                'n_vertices': len(polygon.vertices_xy)
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting DiglineSet to DataFrame: {e}")
        return pd.DataFrame()


def tonnage_grade_series_to_dataframe(series: Any) -> pd.DataFrame:
    """
    Convert TonnageGradeSeries to DataFrame for export.
    
    Args:
        series: TonnageGradeSeries instance
    
    Returns:
        DataFrame with records
    """
    try:
        from ..reconciliation.tonnage_grade_balance import TonnageGradeSeries
        
        if not isinstance(series, TonnageGradeSeries):
            logger.warning(f"Expected TonnageGradeSeries, got {type(series)}")
            return pd.DataFrame()
        
        rows = []
        for record in series.records:
            row = {
                'source': record.source,
                'period_id': record.period_id,
                'material_type': record.material_type,
                'tonnes': record.tonnes
            }
            # Add grades
            for element, grade in record.grades.items():
                row[f'grade_{element}'] = grade
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting TonnageGradeSeries to DataFrame: {e}")
        return pd.DataFrame()


def reconciliation_metrics_to_dataframe(metrics: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Convert reconciliation metrics to DataFrame(s) for export.
    
    Args:
        metrics: Dictionary with reconciliation metrics
    
    Returns:
        Dictionary with 'stages' and 'indices' DataFrames
    """
    try:
        stages_rows = []
        for stage_name, stage_data in metrics.get('stages', {}).items():
            row = {'stage': stage_name}
            row.update(stage_data)
            stages_rows.append(row)
        
        stages_df = pd.DataFrame(stages_rows) if stages_rows else pd.DataFrame()
        
        indices_rows = []
        for index_name, index_value in metrics.get('global_indices', {}).items():
            if isinstance(index_value, dict):
                for key, val in index_value.items():
                    indices_rows.append({
                        'index': index_name,
                        'key': key,
                        'value': val
                    })
            else:
                indices_rows.append({
                    'index': index_name,
                    'value': index_value
                })
        
        indices_df = pd.DataFrame(indices_rows) if indices_rows else pd.DataFrame()
        
        return {
            'stages': stages_df,
            'indices': indices_df
        }
    
    except Exception as e:
        logger.error(f"Error converting reconciliation metrics to DataFrame: {e}")
        return {}


# Scheduling conversion functions (STEP 30)
def schedule_result_to_dataframe(schedule_result: Any) -> pd.DataFrame:
    """
    Convert ScheduleResult to DataFrame for export.
    
    Args:
        schedule_result: ScheduleResult instance
    
    Returns:
        DataFrame with schedule decisions
    """
    try:
        from ..mine_planning.scheduling.types import ScheduleResult
        
        if not isinstance(schedule_result, ScheduleResult):
            logger.warning(f"Expected ScheduleResult, got {type(schedule_result)}")
            return pd.DataFrame()
        
        rows = []
        for decision in schedule_result.decisions:
            rows.append({
                'period_id': decision.period_id,
                'unit_id': decision.unit_id,
                'tonnes': decision.tonnes,
                'destination': decision.destination
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting ScheduleResult to DataFrame: {e}")
        return pd.DataFrame()


def shift_plan_result_to_dataframe(shift_plans: List[Any]) -> pd.DataFrame:
    """
    Convert ShiftPlanResult list to DataFrame for export.
    
    Args:
        shift_plans: List of ShiftPlanResult instances
    
    Returns:
        DataFrame with shift plan data
    """
    try:
        from ..mine_planning.scheduling.short_term.shift_plan import ShiftPlanResult
        
        rows = []
        for plan in shift_plans:
            if not isinstance(plan, ShiftPlanResult):
                continue
            
            for assignment in plan.assignments:
                rows.append({
                    'period_id': plan.period_id,
                    'shift_name': plan.shift_name,
                    'unit_id': assignment.get('unit_id', ''),
                    'tonnes': assignment.get('tonnes', 0.0),
                    'destination': assignment.get('destination', '')
                })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting ShiftPlanResult to DataFrame: {e}")
        return pd.DataFrame()


def fleet_config_to_dataframe(fleet_config: Any) -> Dict[str, pd.DataFrame]:
    """
    Convert FleetConfig to DataFrame(s) for export.
    
    Args:
        fleet_config: FleetConfig instance
    
    Returns:
        Dictionary with 'trucks' and 'shovels' DataFrames
    """
    try:
        from ..haulage.fleet_model import FleetConfig
        
        if not isinstance(fleet_config, FleetConfig):
            logger.warning(f"Expected FleetConfig, got {type(fleet_config)}")
            return {}
        
        # Trucks DataFrame
        trucks_rows = []
        for truck in fleet_config.trucks:
            trucks_rows.append({
                'id': truck.id,
                'payload_tonnes': truck.payload_tonnes,
                'speed_loaded_kmh': truck.speed_loaded_kmh,
                'speed_empty_kmh': truck.speed_empty_kmh,
                'availability': truck.availability,
                'utilisation': truck.utilisation
            })
        trucks_df = pd.DataFrame(trucks_rows)
        
        # Shovels DataFrame
        shovels_rows = []
        for shovel in fleet_config.shovels:
            shovels_rows.append({
                'id': shovel.id,
                'capacity_tonnes': shovel.capacity_tonnes,
                'cycle_time_sec': shovel.cycle_time_sec,
                'availability': shovel.availability,
                'utilisation': shovel.utilisation
            })
        shovels_df = pd.DataFrame(shovels_rows)
        
        return {
            'trucks': trucks_df,
            'shovels': shovels_df
        }
    
    except Exception as e:
        logger.error(f"Error converting FleetConfig to DataFrame: {e}")
        return {}


def route_to_dataframe(routes: List[Any]) -> pd.DataFrame:
    """
    Convert Route list to DataFrame for export.
    
    Args:
        routes: List of Route instances
    
    Returns:
        DataFrame with route data
    """
    try:
        from ..haulage.cycle_time_model import Route
        
        rows = []
        for route in routes:
            if not isinstance(route, Route):
                continue
            
            rows.append({
                'id': route.id,
                'source': route.source,
                'destination': route.destination,
                'distance_km': route.distance_km,
                'vertical_change_m': route.vertical_change_m,
                'congestion_factor': route.congestion_factor
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting Route to DataFrame: {e}")
        return pd.DataFrame()


def cutoff_schedule_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert CutoffScheduleResult or CutoffOptimiserResult to DataFrame for export.
    
    ⚠️ UPDATED: Now supports CutoffOptimiserResult from cutoff_engine.py.
    Falls back to old CutoffScheduleResult format for backward compatibility.
    
    Args:
        result: CutoffScheduleResult or CutoffOptimiserResult instance
    
    Returns:
        DataFrame with cutoff schedule data
    """
    try:
        # Try new cutoff_engine API first
        from ..mine_planning.cutoff.cutoff_engine import CutoffOptimiserResult
        
        if isinstance(result, CutoffOptimiserResult):
            rows = []
            if result.best_pattern:
                for period_id, cutoff in result.best_pattern.cutoffs_by_period.items():
                    rows.append({
                        'period_id': period_id,
                        'cutoff': cutoff,
                        'pattern_id': result.best_pattern.id,
                        'pattern_description': result.best_pattern.description
                    })
            # Also include all pattern results
            for pattern_result in result.pattern_results:
                for period_id, cutoff in pattern_result.get('cutoffs_by_period', {}).items():
                    rows.append({
                        'period_id': period_id,
                        'cutoff': cutoff,
                        'pattern_id': pattern_result.get('pattern_id', ''),
                        'pattern_description': pattern_result.get('description', ''),
                        'npv': pattern_result.get('npv', 0.0)
                    })
            return pd.DataFrame(rows)
        
        logger.warning(f"Expected CutoffOptimiserResult, got {type(result)}")
        return pd.DataFrame()
    
    except ImportError as e:
        logger.error(f"Failed to import CutoffOptimiserResult from cutoff_engine: {e}")
        return pd.DataFrame()
        
        logger.warning(f"Expected CutoffOptimiserResult, got {type(result)}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error converting cutoff result to DataFrame: {e}")
        return pd.DataFrame()


# Planning Dashboard conversion functions (STEP 31)
def planning_scenario_to_dataframe(scenario: Any) -> pd.DataFrame:
    """
    Convert PlanningScenario to DataFrame for export.
    
    Args:
        scenario: PlanningScenario instance
    
    Returns:
        DataFrame with scenario configuration
    """
    try:
        from ..planning.scenario_definition import PlanningScenario
        
        if not isinstance(scenario, PlanningScenario):
            logger.warning(f"Expected PlanningScenario, got {type(scenario)}")
            return pd.DataFrame()
        
        rows = [{
            'name': scenario.id.name,
            'version': scenario.id.version,
            'description': scenario.description,
            'tags': ', '.join(scenario.tags),
            'status': scenario.status,
            'model_name': scenario.inputs.model_name,
            'value_mode': scenario.inputs.value_mode,
            'value_field': scenario.inputs.value_field,
            'created_at': scenario.created_at.isoformat() if isinstance(scenario.created_at, datetime) else str(scenario.created_at),
            'modified_at': scenario.modified_at.isoformat() if isinstance(scenario.modified_at, datetime) else str(scenario.modified_at),
        }]
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting PlanningScenario to DataFrame: {e}")
        return pd.DataFrame()


def planning_scenario_to_dict(scenario: Any) -> Dict[str, Any]:
    """
    Convert PlanningScenario to dictionary for JSON export.
    
    Args:
        scenario: PlanningScenario instance
    
    Returns:
        Dictionary representation
    """
    try:
        from ..planning.scenario_definition import PlanningScenario
        
        if not isinstance(scenario, PlanningScenario):
            logger.warning(f"Expected PlanningScenario, got {type(scenario)}")
            return {}
        
        return scenario.to_dict()
    
    except Exception as e:
        logger.error(f"Error converting PlanningScenario to dict: {e}")
        return {}


def scenario_comparison_to_dataframe(comparison: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert scenario comparison metrics to DataFrame for export.
    
    Args:
        comparison: Comparison dictionary from compare_scenarios
    
    Returns:
        DataFrame with comparison metrics
    """
    try:
        scenarios = comparison.get("scenarios", [])
        if not scenarios:
            return pd.DataFrame()
        
        rows = []
        for scenario_metrics in scenarios:
            rows.append({
                'name': scenario_metrics.get('name', ''),
                'version': scenario_metrics.get('version', ''),
                'npv': scenario_metrics.get('npv'),
                'irr': scenario_metrics.get('irr'),
                'payback_period': scenario_metrics.get('payback_period'),
                'lom_years': scenario_metrics.get('lom_years'),
                'peak_annual_production': scenario_metrics.get('peak_annual_production'),
                'total_tonnes': scenario_metrics.get('total_tonnes'),
                'npv_p10': scenario_metrics.get('npv_p10'),
                'npv_p50': scenario_metrics.get('npv_p50'),
                'npv_p90': scenario_metrics.get('npv_p90'),
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting scenario comparison to DataFrame: {e}")
        return pd.DataFrame()


def npvs_config_to_dataframe(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert NPVS configuration to DataFrame for export (STEP 32).
    
    Args:
        config: NPVS configuration dictionary
    
    Returns:
        DataFrame with configuration details
    """
    try:
        rows = []
        
        # Periods
        for period in config.get("periods", []):
            rows.append({
                "type": "period",
                "id": period.get("id", ""),
                "index": period.get("index", 0),
                "duration_years": period.get("duration_years", 1.0),
                "discount_factor": period.get("discount_factor", 1.0),
            })
        
        # Destinations
        for dest in config.get("destinations", []):
            recovery = dest.get("recovery_by_element", {})
            rows.append({
                "type": "destination",
                "id": dest.get("id", ""),
                "dest_type": dest.get("type", ""),
                "capacity_tpy": dest.get("capacity_tpy", 0),
                "recovery_cu": recovery.get("Cu", 0),
                "recovery_au": recovery.get("Au", 0),
                "processing_cost_per_t": dest.get("processing_cost_per_t", 0),
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting NPVS config to DataFrame: {e}")
        return pd.DataFrame()


def npvs_result_to_dataframe(result: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert NPVS result to DataFrame for export (STEP 32).
    
    Args:
        result: NPVS result dictionary with schedule and npv
    
    Returns:
        DataFrame with schedule decisions
    """
    try:
        schedule = result.get("schedule", {})
        decisions = schedule.get("decisions", [])
        
        if not decisions:
            return pd.DataFrame()
        
        rows = []
        for decision in decisions:
            rows.append({
                "period_id": decision.get("period_id", ""),
                "unit_id": decision.get("unit_id", ""),
                "tonnes": decision.get("tonnes", 0),
                "destination": decision.get("destination", ""),
            })
        
        df = pd.DataFrame(rows)
        
        # Add NPV summary row
        npv = result.get("npv", 0)
        if npv:
            summary_row = pd.DataFrame([{
                "period_id": "SUMMARY",
                "unit_id": "NPV",
                "tonnes": npv,
                "destination": "Total"
            }])
            df = pd.concat([df, summary_row], ignore_index=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error converting NPVS result to DataFrame: {e}")
        return pd.DataFrame()


def pushback_plan_to_dataframe(plan: Any) -> pd.DataFrame:
    """
    Convert PushbackPlan to DataFrame for export (STEP 33).
    
    Args:
        plan: PushbackPlan instance
    
    Returns:
        DataFrame with pushback details
    """
    try:
        from ..mine_planning.pushbacks.pushback_model import PushbackPlan
        
        if not isinstance(plan, PushbackPlan):
            logger.warning(f"Expected PushbackPlan, got {type(plan)}")
            return pd.DataFrame()
        
        rows = []
        for pushback in plan.pushbacks:
            rows.append({
                "pushback_id": pushback.id,
                "name": pushback.name,
                "order_index": pushback.order_index,
                "tonnes": pushback.tonnes,
                "value": pushback.value,
                "shell_count": len(pushback.shell_ids),
                "shell_ids": ", ".join(pushback.shell_ids),
                "color_r": pushback.color[0],
                "color_g": pushback.color[1],
                "color_b": pushback.color[2],
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting PushbackPlan to DataFrame: {e}")
        return pd.DataFrame()


def shell_phases_to_dataframe(shells: List[Any]) -> pd.DataFrame:
    """
    Convert list of ShellPhase to DataFrame for export (STEP 33).
    
    Args:
        shells: List of ShellPhase instances
    
    Returns:
        DataFrame with shell/phase details
    """
    try:
        from ..mine_planning.pushbacks.pushback_model import ShellPhase
        
        rows = []
        for shell in shells:
            if not isinstance(shell, ShellPhase):
                continue
            
            rows.append({
                "shell_id": shell.id,
                "tonnes": shell.tonnes,
                "value": shell.value,
                "value_per_tonne": shell.value / shell.tonnes if shell.tonnes > 0 else 0,
                "precedence_count": len(shell.precedence_ids),
                "precedence_ids": ", ".join(shell.precedence_ids),
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting ShellPhase list to DataFrame: {e}")
        return pd.DataFrame()


def haulage_eval_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert HaulageEvalResult to DataFrame for export (STEP 34).
    
    Args:
        result: HaulageEvalResult instance
    
    Returns:
        DataFrame with period metrics
    """
    try:
        from ..haulage.haulage_evaluator import HaulageEvalResult
        
        if not isinstance(result, HaulageEvalResult):
            logger.warning(f"Expected HaulageEvalResult, got {type(result)}")
            return pd.DataFrame()
        
        # Convert period metrics to DataFrame
        if result.period_metrics:
            return pd.DataFrame(result.period_metrics)
        
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error converting HaulageEvalResult to DataFrame: {e}")
        return pd.DataFrame()


def cutoff_optimiser_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert CutoffOptimiserResult to DataFrame for export (STEP 35).
    
    Args:
        result: CutoffOptimiserResult instance
    
    Returns:
        DataFrame with pattern results
    """
    try:
        from ..mine_planning.cutoff.cutoff_engine import CutoffOptimiserResult
        
        if not isinstance(result, CutoffOptimiserResult):
            logger.warning(f"Expected CutoffOptimiserResult, got {type(result)}")
            return pd.DataFrame()
        
        if result.pattern_results:
            return pd.DataFrame(result.pattern_results)
        
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error converting CutoffOptimiserResult to DataFrame: {e}")
        return pd.DataFrame()


def cutoff_pattern_to_dict(pattern: Any) -> Dict[str, Any]:
    """
    Convert CutoffPattern to dictionary/JSON (STEP 35).
    
    Args:
        pattern: CutoffPattern instance
    
    Returns:
        Dictionary representation
    """
    try:
        from ..mine_planning.cutoff.cutoff_engine import CutoffPattern
        
        if not isinstance(pattern, CutoffPattern):
            logger.warning(f"Expected CutoffPattern, got {type(pattern)}")
            return {}
        
        return {
            "id": pattern.id,
            "description": pattern.description,
            "cutoffs_by_period": pattern.cutoffs_by_period,
            "avg_cutoff": pattern.get_avg_cutoff()
        }
    
    except Exception as e:
        logger.error(f"Error converting CutoffPattern to dict: {e}")
        return {}


def aligned_dashboard_result_to_dataframe(result: Any) -> pd.DataFrame:
    """
    Convert AlignedDashboardResult to DataFrame for export (STEP 36).
    
    Args:
        result: AlignedDashboardResult instance or dict
    
    Returns:
        DataFrame with period metrics
    """
    try:
        from ..planning.production_alignment import AlignedDashboardResult, AlignedPeriodMetrics
        
        # Handle dict or dataclass
        if isinstance(result, dict):
            periods = result.get("periods", [])
        elif isinstance(result, AlignedDashboardResult):
            periods = result.periods
        else:
            logger.warning(f"Expected AlignedDashboardResult or dict, got {type(result)}")
            return pd.DataFrame()
        
        if not periods:
            return pd.DataFrame()
        
        rows = []
        for period in periods:
            if isinstance(period, dict):
                p = period
            else:
                # Convert dataclass to dict
                p = {
                    'period_id': getattr(period, 'period_id', ''),
                    'index': getattr(period, 'index', 0),
                    'planned_mined_t': getattr(period, 'planned_mined_t', 0.0),
                    'planned_plant_t': getattr(period, 'planned_plant_t', 0.0),
                    'planned_value': getattr(period, 'planned_value', 0.0),
                    'hauled_t': getattr(period, 'hauled_t', 0.0),
                    'haulage_utilisation': getattr(period, 'haulage_utilisation', 0.0),
                    'haulage_shortfall_t': getattr(period, 'haulage_shortfall_t', 0.0),
                    'mined_actual_t': getattr(period, 'mined_actual_t', 0.0),
                    'mill_actual_t': getattr(period, 'mill_actual_t', 0.0),
                    'delta_mined_t': getattr(period, 'delta_mined_t', 0.0),
                    'delta_mill_t': getattr(period, 'delta_mill_t', 0.0),
                }
            
            rows.append({
                'period_id': p.get('period_id', ''),
                'index': p.get('index', 0),
                'planned_mined_t': p.get('planned_mined_t', 0.0),
                'planned_plant_t': p.get('planned_plant_t', 0.0),
                'planned_value': p.get('planned_value', 0.0),
                'hauled_t': p.get('hauled_t', 0.0),
                'haulage_utilisation': p.get('haulage_utilisation', 0.0),
                'haulage_shortfall_t': p.get('haulage_shortfall_t', 0.0),
                'mined_actual_t': p.get('mined_actual_t', 0.0),
                'mill_actual_t': p.get('mill_actual_t', 0.0),
                'delta_mined_t': p.get('delta_mined_t', 0.0),
                'delta_mill_t': p.get('delta_mill_t', 0.0),
                'delta_mined_pct': (p.get('delta_mined_t', 0.0) / p.get('planned_mined_t', 1.0) * 100) if p.get('planned_mined_t', 0.0) > 0 else 0.0,
                'delta_mill_pct': (p.get('delta_mill_t', 0.0) / p.get('planned_plant_t', 1.0) * 100) if p.get('planned_plant_t', 0.0) > 0 else 0.0,
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting AlignedDashboardResult to DataFrame: {e}")
        return pd.DataFrame()


def aligned_dashboard_result_to_dict(result: Any) -> Dict[str, Any]:
    """
    Convert AlignedDashboardResult to dictionary/JSON (STEP 36).
    
    Args:
        result: AlignedDashboardResult instance
    
    Returns:
        Dictionary representation
    """
    try:
        from ..planning.production_alignment import AlignedDashboardResult, AlignedPeriodMetrics
        
        if isinstance(result, dict):
            return result
        
        if not isinstance(result, AlignedDashboardResult):
            logger.warning(f"Expected AlignedDashboardResult, got {type(result)}")
            return {}
        
        periods_dict = []
        for period in result.periods:
            periods_dict.append({
                'period_id': period.period_id,
                'index': period.index,
                'planned_mined_t': period.planned_mined_t,
                'planned_plant_t': period.planned_plant_t,
                'planned_grade_by_element': period.planned_grade_by_element,
                'planned_value': period.planned_value,
                'hauled_t': period.hauled_t,
                'haulage_utilisation': period.haulage_utilisation,
                'haulage_shortfall_t': period.haulage_shortfall_t,
                'mined_actual_t': period.mined_actual_t,
                'mill_actual_t': period.mill_actual_t,
                'grade_mine_by_element': period.grade_mine_by_element,
                'grade_mill_by_element': period.grade_mill_by_element,
                'delta_mined_t': period.delta_mined_t,
                'delta_mill_t': period.delta_mill_t,
                'delta_grade_mine': period.delta_grade_mine,
                'delta_grade_mill': period.delta_grade_mill,
            })
        
        return {
            'periods': periods_dict,
            'overall_kpis': result.overall_kpis,
            'metadata': result.metadata
        }
    
    except Exception as e:
        logger.error(f"Error converting AlignedDashboardResult to dict: {e}")
        return {}


def slos_stopes_to_dataframe(stopes: List[Any]) -> pd.DataFrame:
    """
    Convert list of StopeInstance to DataFrame for export (STEP 37).
    
    Args:
        stopes: List of StopeInstance objects
    
    Returns:
        DataFrame with stope data
    """
    try:
        from ..ug.slos.slos_geometry import StopeInstance
        
        rows = []
        for stope in stopes:
            if isinstance(stope, dict):
                s = stope
            else:
                s = {
                    'id': getattr(stope, 'id', ''),
                    'template_id': getattr(stope, 'template_id', ''),
                    'level': getattr(stope, 'level', ''),
                    'center': getattr(stope, 'center', (0, 0, 0)),
                    'tonnes': getattr(stope, 'tonnes', 0.0),
                    'grade_by_element': getattr(stope, 'grade_by_element', {}),
                    'dilution_tonnes': getattr(stope, 'dilution_tonnes', 0.0),
                    'ore_loss_fraction': getattr(stope, 'ore_loss_fraction', 0.0)
                }
            
            center = s.get('center', (0, 0, 0))
            rows.append({
                'stope_id': s.get('id', ''),
                'template_id': s.get('template_id', ''),
                'level': s.get('level', ''),
                'center_x': center[0] if isinstance(center, (tuple, list)) else 0.0,
                'center_y': center[1] if isinstance(center, (tuple, list)) else 0.0,
                'center_z': center[2] if isinstance(center, (tuple, list)) else 0.0,
                'tonnes': s.get('tonnes', 0.0),
                'dilution_tonnes': s.get('dilution_tonnes', 0.0),
                'ore_loss_fraction': s.get('ore_loss_fraction', 0.0)
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting stopes to DataFrame: {e}")
        return pd.DataFrame()


def cave_footprint_to_dataframe(footprint: Any) -> pd.DataFrame:
    """
    Convert CaveFootprint to DataFrame for export (STEP 37).
    
    Args:
        footprint: CaveFootprint instance
    
    Returns:
        DataFrame with cell data
    """
    try:
        from ..ug.caving.cave_footprint import CaveFootprint
        
        if isinstance(footprint, dict):
            cells = footprint.get('cells', [])
        elif hasattr(footprint, 'cells'):
            cells = footprint.cells
        else:
            cells = []
        
        rows = []
        for cell in cells:
            if isinstance(cell, dict):
                c = cell
            else:
                c = {
                    'id': getattr(cell, 'id', ''),
                    'x': getattr(cell, 'x', 0.0),
                    'y': getattr(cell, 'y', 0.0),
                    'level': getattr(cell, 'level', 0.0),
                    'tonnage': getattr(cell, 'tonnage', 0.0),
                    'grade_by_element': getattr(cell, 'grade_by_element', {})
                }
            
            rows.append({
                'cell_id': c.get('id', ''),
                'x': c.get('x', 0.0),
                'y': c.get('y', 0.0),
                'level': c.get('level', 0.0),
                'tonnage': c.get('tonnage', 0.0)
            })
        
        return pd.DataFrame(rows)
    
    except Exception as e:
        logger.error(f"Error converting cave footprint to DataFrame: {e}")
        return pd.DataFrame()
