"""
Reporting Module

Export experiment results to various formats (DataFrame, Excel, LaTeX).
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np

from .runner import ExperimentRunResult

logger = logging.getLogger(__name__)


def experiment_to_dataframe(run_result: ExperimentRunResult) -> pd.DataFrame:
    """
    Convert ExperimentRunResult to DataFrame.
    
    Args:
        run_result: ExperimentRunResult instance
    
    Returns:
        DataFrame with one row per instance, columns for parameters and metrics
    """
    rows = []
    
    for result in run_result.results:
        row = {}
        
        # Add parameter values
        param_values = result.get('parameter_values', {})
        row.update(param_values)
        
        # Add metrics
        metrics = result.get('metrics', {})
        row.update(metrics)
        
        # Add instance index
        row['instance_index'] = result.get('instance_index', -1)
        
        # Add error flag if present
        if 'error' in result:
            row['error'] = result['error']
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    return df


def to_excel(run_result: ExperimentRunResult, path: Path) -> None:
    """
    Export experiment results to Excel file.
    
    Args:
        run_result: ExperimentRunResult instance
        path: Output file path
    """
    df = experiment_to_dataframe(run_result)
    
    # Create Excel writer
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        # Main results sheet
        df.to_excel(writer, sheet_name='Results', index=False)
        
        # Summary sheet
        summary_data = []
        summary_data.append(['Definition ID', run_result.definition_id])
        summary_data.append(['Total Instances', len(run_result.results)])
        summary_data.append(['Successful', run_result.metadata.get('n_successful', 0)])
        summary_data.append(['Metrics', ', '.join(run_result.metrics)])
        
        summary_df = pd.DataFrame(summary_data, columns=['Field', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Metrics summary sheet (if metrics exist)
        if run_result.metrics:
            metrics_data = []
            for result in run_result.results:
                if 'metrics' in result:
                    metrics = result['metrics']
                    for metric_name, metric_value in metrics.items():
                        metrics_data.append({
                            'instance_index': result.get('instance_index', -1),
                            'metric': metric_name,
                            'value': metric_value
                        })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
    
    logger.info(f"Exported experiment results to Excel: {path}")


def to_latex_table(
    run_result: ExperimentRunResult,
    metric_subset: Optional[List[str]] = None,
    parameter_subset: Optional[List[str]] = None
) -> str:
    """
    Generate LaTeX table from experiment results.
    
    Args:
        run_result: ExperimentRunResult instance
        metric_subset: List of metric names to include (if None, include all)
        parameter_subset: List of parameter names to include (if None, include all)
    
    Returns:
        LaTeX table string
    """
    df = experiment_to_dataframe(run_result)
    
    # Filter columns
    columns_to_include = []
    
    # Add parameter columns
    if parameter_subset:
        columns_to_include.extend(parameter_subset)
    else:
        # Include all parameter columns (exclude metrics and special columns)
        param_cols = [col for col in df.columns 
                     if col not in ['instance_index', 'error'] 
                     and col not in run_result.metrics]
        columns_to_include.extend(param_cols)
    
    # Add metric columns
    if metric_subset:
        columns_to_include.extend(metric_subset)
    else:
        columns_to_include.extend(run_result.metrics)
    
    # Filter to available columns
    columns_to_include = [col for col in columns_to_include if col in df.columns]
    
    if not columns_to_include:
        return "% No columns to include\n"
    
    # Select columns
    df_subset = df[columns_to_include]
    
    # Remove rows with errors
    if 'error' in df.columns:
        df_subset = df_subset[df['error'].isna()]
    
    # Format numbers
    for col in df_subset.columns:
        if df_subset[col].dtype in [np.float64, np.float32]:
            # Format to 3 decimal places
            df_subset[col] = df_subset[col].apply(lambda x: f"{x:.3f}" if not np.isnan(x) else "---")
    
    # Generate LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Experiment Results}")
    latex_lines.append("\\label{tab:experiment_results}")
    
    # Column specification
    n_cols = len(columns_to_include)
    col_spec = "|" + "|".join(["c"] * n_cols) + "|"
    latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_lines.append("\\hline")
    
    # Header row
    header = " & ".join([col.replace("_", "\\_") for col in columns_to_include])
    latex_lines.append(f"{header} \\\\")
    latex_lines.append("\\hline")
    
    # Data rows
    for idx, row in df_subset.iterrows():
        row_data = " & ".join([str(val) for val in row.values])
        latex_lines.append(f"{row_data} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    latex_str = "\n".join(latex_lines)
    
    return latex_str


def to_csv(run_result: ExperimentRunResult, path: Path) -> None:
    """
    Export experiment results to CSV file.
    
    Args:
        run_result: ExperimentRunResult instance
        path: Output file path
    """
    df = experiment_to_dataframe(run_result)
    df.to_csv(path, index=False)
    logger.info(f"Exported experiment results to CSV: {path}")


def generate_summary_statistics(run_result: ExperimentRunResult) -> Dict[str, Any]:
    """
    Generate summary statistics for experiment results.
    
    Args:
        run_result: ExperimentRunResult instance
    
    Returns:
        Dict with summary statistics per metric
    """
    df = experiment_to_dataframe(run_result)
    
    summary = {
        'n_instances': len(run_result.results),
        'n_successful': run_result.metadata.get('n_successful', 0),
        'metrics_summary': {}
    }
    
    # Compute statistics for each metric
    for metric_name in run_result.metrics:
        if metric_name in df.columns:
            metric_values = df[metric_name].dropna()
            if len(metric_values) > 0:
                summary['metrics_summary'][metric_name] = {
                    'mean': float(metric_values.mean()),
                    'std': float(metric_values.std()),
                    'min': float(metric_values.min()),
                    'max': float(metric_values.max()),
                    'median': float(metric_values.median())
                }
    
    return summary

