"""Simplified underground stope optimization placeholders.

These are intentionally lightweight to unblock the UI and demos.
Replace with your production optimizer and dilution logic.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd

@dataclass
class Stope:
    id: str
    block_indices: List[int]
    tonnes: float
    grade: float
    nsr: float

    # optional fields (dilution results)
    diluted_tonnes: float | None = None
    diluted_grade: float | None = None
    
    # STEP 19: Geotechnical stability integration
    stability_result_id: str | None = None  # Reference to stope stability analysis result


def _infer_dimensions(df: pd.DataFrame) -> tuple[float, float, float]:
    for cols in (("DX","DY","DZ"),("dx","dy","dz"),("XINC","YINC","ZINC")):
        if all(c in df.columns for c in cols):
            vals = df[list(cols)].astype(float).values
            return float(np.nanmean(vals[:,0])), float(np.nanmean(vals[:,1])), float(np.nanmean(vals[:,2]))
    # fallback
    return 10.0, 10.0, 10.0


def generate_grid_stopes(blocks_df: pd.DataFrame, min_nsr: float = 0.0,
                         stope_width: float = 15.0, stope_height: float = 30.0,
                         stope_length: float = 30.0) -> List[Stope]:
    """
    Generate a simple grid of stopes by spatial binning (NOT an optimizer).
    
    ⚠️ WARNING: This is a DEMO/PLACEHOLDER function using rigid grid binning.
    It does NOT perform real stope optimization and should NOT be used for actual mine planning.
    
    CRITICAL LIMITATIONS:
    - Uses rigid grid alignment (bins aligned to fixed intervals)
    - High-grade veins on bin boundaries will be split and potentially classified as waste
    - Cannot rotate or shift stopes to capture ore optimally
    - Does not use floating stope algorithms (MSO-style)
    
    This function creates stope placeholders by grouping blocks into spatial bins.
    It does NOT optimize stope shapes - it simply grids the model into regular
    rectangular regions based on the specified dimensions.
    
    For actual stope optimization, use the algorithms in block_model_viewer.ug.stope_opt.optimizer.
    
    Args:
        blocks_df: DataFrame with block model data
        min_nsr: Minimum NSR threshold
        stope_width: Width of each stope bin (m)
        stope_height: Height of each stope bin (m)
        stope_length: Length of each stope bin (m)
    
    Returns:
        List of Stope objects representing the grid bins
    
    Note:
        - Computes NSR (if column missing) as grade*60*0.85-25 approx (consistent with UI defaults)
        - Filters positive-nsr blocks
        - Bins x,y,z by stope dims to create pseudo-stopes
    """
    df = blocks_df.copy()
    if "nsr" not in df.columns:
        grade = df.get("grade", df.get("au_grade", pd.Series(0, index=df.index)))
        df["nsr"] = grade.astype(float) * 60.0 * 0.85 - 25.0

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["x","y","z","nsr"])  # require coords & nsr
    df = df[df["nsr"] >= min_nsr]
    if df.empty:
        return []

    # ⚠️ RIGID GRID BINNING - This is the problematic binning heuristic
    # Blocks are assigned to bins based on floor division, creating rigid grid alignment
    # High-grade veins on boundaries will be split across bins, causing ore loss
    bins = (
        np.floor(df["x"].astype(float) / stope_length),
        np.floor(df["y"].astype(float) / stope_width),
        np.floor(df["z"].astype(float) / stope_height),
    )
    key = bins[0].astype(int).astype(str) + ":" + bins[1].astype(int).astype(str) + ":" + bins[2].astype(int).astype(str)
    df = df.assign(_bin=key.values)

    # density for tonnes estimation
    dx,dy,dz = _infer_dimensions(blocks_df)
    density = 2.7
    block_tonnes = dx*dy*dz*density

    stopes: List[Stope] = []
    for bin_key, g in df.groupby("_bin"):
        idx = g.index.to_list()
        tonnes = block_tonnes * len(g)
        # choose grade column heuristically
        if "grade" in g.columns:
            grade_vals = g["grade"].astype(float).values
        else:
            # use first numeric property as proxy
            num_cols = [c for c in g.columns if c not in ("x","y","z","dx","dy","dz","XINC","YINC","ZINC","nsr","_bin") and np.issubdtype(g[c].dtype, np.number)]
            grade_vals = g[num_cols[0]].astype(float).values if num_cols else np.zeros(len(g))
        avg_grade = float(np.nanmean(grade_vals)) if len(grade_vals) else 0.0
        avg_nsr = float(np.nanmean(g["nsr"].astype(float).values))
        stopes.append(Stope(id=f"STOPE-{len(stopes)+1}", block_indices=idx, tonnes=tonnes, grade=avg_grade, nsr=avg_nsr))

    return stopes


# Backward compatibility alias (deprecated)
def quick_stope_grid(blocks_df: pd.DataFrame, min_nsr: float = 0.0,
                     stope_width: float = 15.0, stope_height: float = 30.0,
                     stope_length: float = 30.0) -> List[Stope]:
    """
    Deprecated alias for generate_grid_stopes.
    
    ⚠️ DEPRECATED: Use generate_grid_stopes() instead.
    This alias is kept for backward compatibility only.
    """
    return generate_grid_stopes(blocks_df, min_nsr, stope_width, stope_height, stope_length)


def calculate_dilution_contact(stope: Stope, blocks_df: pd.DataFrame, dilution_skin_m: float = 0.5) -> Dict[str, Any]:
    """Placeholder dilution model: add 5% tonnes at 20% of stope grade."""
    dilution_tonnes = stope.tonnes * 0.05
    dilution_grade = stope.grade * 0.2
    return {"dilution_tonnes": dilution_tonnes, "dilution_grade": dilution_grade}
