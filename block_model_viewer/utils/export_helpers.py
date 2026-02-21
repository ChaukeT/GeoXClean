"""
Shared export utility functions - Step 10: Canonical Export API.

Centralizes ALL CSV/Excel export logic. Pure I/O functions - no UI dialogs.
UI panels handle file dialogs and call these functions with paths.
"""

import logging
from pathlib import Path
from typing import Optional, Union, Dict
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Pure I/O Functions (Step 10: No UI dependencies)
# ============================================================================

def export_dataframe_to_csv(df: pd.DataFrame, path: Union[str, Path], chunk_size: int = 100000, process_events: bool = True) -> None:
    """
    Export pandas DataFrame to CSV file (optimized for large DataFrames).
    
    Uses chunked writing for large DataFrames to avoid memory issues and improve performance.
    Can process Qt events during export to keep UI responsive.
    
    Args:
        df: DataFrame to export
        path: Output file path
        chunk_size: Number of rows to write per chunk (default 100k)
        process_events: If True, process Qt events between chunks to keep UI responsive
        
    Raises:
        ValueError: If path is invalid
        IOError: If file write fails
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .csv extension
    if path.suffix.lower() != '.csv':
        path = path.with_suffix('.csv')
    
    # Try to import QApplication for event processing
    qapp = None
    if process_events:
        try:
            from PyQt6.QtWidgets import QApplication
            qapp = QApplication.instance()
        except ImportError:
            qapp = None
    
    # For large DataFrames, use chunked writing
    n_rows = len(df)
    if n_rows > chunk_size:
        # Write header first
        df.head(0).to_csv(path, index=False, mode='w')
        
        # Write data in chunks
        n_chunks = (n_rows + chunk_size - 1) // chunk_size
        for chunk_idx, start_idx in enumerate(range(0, n_rows, chunk_size)):
            end_idx = min(start_idx + chunk_size, n_rows)
            chunk = df.iloc[start_idx:end_idx]
            chunk.to_csv(path, index=False, mode='a', header=False)
            
            # Process events every few chunks to keep UI responsive
            if process_events and qapp is not None and chunk_idx % 5 == 0:
                qapp.processEvents()
        
        logger.info(f"Exported {n_rows} rows to {path} (chunked)")
    else:
        # Small DataFrame - write all at once
        df.to_csv(path, index=False)
        logger.info(f"Exported {n_rows} rows to {path}")


def export_dataframe_to_excel(
    df: pd.DataFrame,
    path: Union[str, Path],
    sheet_name: str = "Sheet1"
) -> None:
    """
    Export pandas DataFrame to Excel file (pure I/O, no UI).
    
    Args:
        df: DataFrame to export
        path: Output file path
        sheet_name: Excel sheet name
        
    Raises:
        ValueError: If path is invalid
        ImportError: If openpyxl not installed
        IOError: If file write fails
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl not installed - cannot export to Excel. Install with: pip install openpyxl")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .xlsx extension
    if path.suffix.lower() not in ['.xlsx', '.xls']:
        path = path.with_suffix('.xlsx')
    
    df.to_excel(path, index=False, sheet_name=sheet_name, engine='openpyxl')
    logger.info(f"Exported {len(df)} rows to {path} (sheet: {sheet_name})")


def export_multiple_sheets_to_excel(
    frames: Dict[str, pd.DataFrame],
    path: Union[str, Path]
) -> None:
    """
    Export multiple DataFrames to Excel file with multiple sheets (pure I/O, no UI).
    
    Args:
        frames: Dictionary mapping sheet names to DataFrames
        path: Output file path
        
    Raises:
        ValueError: If path is invalid or frames is empty
        ImportError: If openpyxl not installed
        IOError: If file write fails
    """
    if not frames:
        raise ValueError("frames dictionary cannot be empty")
    
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl not installed - cannot export to Excel. Install with: pip install openpyxl")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Ensure .xlsx extension
    if path.suffix.lower() not in ['.xlsx', '.xls']:
        path = path.with_suffix('.xlsx')
    
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, df in frames.items():
            # Excel sheet names have length limit and character restrictions
            safe_sheet_name = sheet_name[:31].replace('/', '_').replace('\\', '_').replace('?', '_').replace('*', '_').replace('[', '_').replace(']', '_')
            df.to_excel(writer, sheet_name=safe_sheet_name, index=False)
    
    logger.info(f"Exported {len(frames)} sheets to {path}")


def suggest_export_filename(base: str, ext: str = ".csv") -> Path:
    """
    Suggest an export filename with timestamp (pure utility, no UI).
    
    Args:
        base: Base filename (without extension)
        ext: File extension (e.g., ".csv", ".xlsx")
        
    Returns:
        Path with suggested filename
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base}_{timestamp}{ext}"
    return Path(filename)


def ensure_export_directory(path: Union[str, Path]) -> None:
    """
    Ensure the directory for an export path exists (pure utility, no UI).
    
    Args:
        path: File path (directory will be created if needed)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UI Helper Functions (for backward compatibility)
# ============================================================================

def export_dataframe_to_csv_with_dialog(
    df: pd.DataFrame,
    default_filename: str = "export.csv",
    parent: Optional['QWidget'] = None,
    title: str = "Export to CSV"
) -> Optional[Path]:
    """
    Export pandas DataFrame to CSV with file dialog (UI helper).
    
    DEPRECATED: Prefer using QFileDialog in UI, then calling export_dataframe_to_csv().
    
    Args:
        df: DataFrame to export
        default_filename: Default filename suggestion
        parent: Parent widget for dialog
        title: Dialog title
    
    Returns:
        Path to saved file, or None if cancelled
    """
    try:
        from PyQt6.QtWidgets import QFileDialog
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            title,
            default_filename,
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if not file_path:
            return None
        
        # Export using pure I/O function
        export_dataframe_to_csv(df, file_path)
        return Path(file_path)
    
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}", exc_info=True)
        return None


def export_dataframe_to_excel_with_dialog(
    df: pd.DataFrame,
    default_filename: str = "export.xlsx",
    parent: Optional['QWidget'] = None,
    title: str = "Export to Excel",
    sheet_name: str = "Sheet1"
) -> Optional[Path]:
    """
    Export pandas DataFrame to Excel with file dialog (UI helper).
    
    DEPRECATED: Prefer using QFileDialog in UI, then calling export_dataframe_to_excel().
    
    Args:
        df: DataFrame to export
        default_filename: Default filename suggestion
        parent: Parent widget for dialog
        title: Dialog title
        sheet_name: Excel sheet name
    
    Returns:
        Path to saved file, or None if cancelled
    """
    try:
        from PyQt6.QtWidgets import QFileDialog
        
        # Show file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            parent,
            title,
            default_filename,
            "Excel Files (*.xlsx);;All Files (*.*)"
        )
        
        if not file_path:
            return None
        
        # Export using pure I/O function
        export_dataframe_to_excel(df, file_path, sheet_name)
        return Path(file_path)
    
    except ImportError:
        logger.error("openpyxl not installed - cannot export to Excel")
        return None
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}", exc_info=True)
        return None


def export_to_vtk(
    points: pd.DataFrame,
    file_path: Union[str, Path],
    point_data: Optional[dict] = None
) -> bool:
    """
    Export point cloud to VTK format.
    
    Args:
        points: DataFrame with X, Y, Z columns
        file_path: Output file path
        point_data: Dictionary of arrays to attach as point data
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import pyvista as pv
        
        # Create point cloud
        cloud = pv.PolyData(points[['X', 'Y', 'Z']].values)
        
        # Add point data
        if point_data:
            for name, data in point_data.items():
                cloud[name] = data
        
        # Save
        cloud.save(str(file_path))
        logger.info(f"Exported {len(points)} points to VTK: {file_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error exporting to VTK: {e}", exc_info=True)
        return False


def format_number(value: float, precision: int = 2) -> str:
    """
    Format number with thousands separators and precision.
    
    Args:
        value: Number to format
        precision: Decimal places
    
    Returns:
        Formatted string
    """
    if abs(value) >= 1e6:
        return f"{value:,.{precision}f}"
    elif abs(value) >= 1000:
        return f"{value:,.{precision}f}"
    else:
        return f"{value:.{precision}f}"


def create_summary_dict(
    df: pd.DataFrame,
    numeric_cols: Optional[list] = None
) -> dict:
    """
    Create summary statistics dictionary from DataFrame.
    
    Args:
        df: DataFrame to summarize
        numeric_cols: List of numeric columns to summarize (None = all numeric)
    
    Returns:
        Dictionary with summary stats
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    summary = {
        'row_count': len(df),
        'column_count': len(df.columns),
        'numeric_columns': numeric_cols,
        'statistics': {}
    }
    
    for col in numeric_cols:
        if col in df.columns:
            summary['statistics'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
    
    return summary


def export_experiment_results(
    run_result: Dict,
    path: Union[str, Path],
    format: str = "excel"
) -> None:
    """
    Export experiment results to CSV or Excel (STEP 25).
    
    Args:
        run_result: ExperimentRunResult dict
        path: Output file path
        format: "excel" or "csv"
    """
    from ..research.reporting import ExperimentRunResult, to_excel, to_csv
    from pathlib import Path as PathType
    
    # Reconstruct ExperimentRunResult
    exp_result = ExperimentRunResult(
        definition_id=run_result.get('definition_id', ''),
        results=run_result.get('results', []),
        metrics=run_result.get('metrics', []),
        metadata=run_result.get('metadata', {})
    )
    
    path = PathType(path)
    
    if format.lower() == 'excel':
        to_excel(exp_result, path)
    else:
        to_csv(exp_result, path)
    
    logger.info(f"Exported experiment results to {path}")


def batch_export(
    dataframes: dict,
    output_dir: Union[str, Path],
    format: str = 'csv',
    prefix: str = ""
) -> list:
    """
    Export multiple DataFrames to files.
    
    Args:
        dataframes: Dictionary of {name: DataFrame}
        output_dir: Output directory
        format: Export format ('csv' or 'xlsx')
        prefix: Filename prefix
    
    Returns:
        List of exported file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    exported_files = []
    
    for name, df in dataframes.items():
        # Create filename
        filename = f"{prefix}{name}.{format}" if prefix else f"{name}.{format}"
        file_path = output_dir / filename
        
        try:
            if format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'xlsx':
                df.to_excel(file_path, index=False, engine='openpyxl')
            else:
                logger.warning(f"Unknown format '{format}', skipping {name}")
                continue
            
            exported_files.append(file_path)
            logger.info(f"Exported {name} to {file_path}")
        
        except Exception as e:
            logger.error(f"Failed to export {name}: {e}")
    
    return exported_files

















