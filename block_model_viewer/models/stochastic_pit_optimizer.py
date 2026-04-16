"""
Stochastic Pit Optimizer
========================
Monte Carlo simulation wrapper around the Lerchs-Grossmann solver.

Quantifies pit selection probability under grade and price uncertainty.
Each simulation samples grade (per-block) and price from probability
distributions, runs LG, and records which blocks were selected.  After
N simulations the selection frequency gives the *pit probability* — a
number between 0 and 1 telling the user how confidently each block
should be in the pit.

Typical use
-----------
>>> opt = StochasticPitOptimizer()
>>> result = opt.run(df, grid_spec, grade_col='au_gt', price=2000,
...                  recovery=0.88, mining_cost=4.5,
...                  processing_cost=18.0, n_sims=50,
...                  price_cv=0.15, grade_cv=0.10)
>>> pit_prob = result['pit_probability']   # float32 (0–1) per block
>>> p90_mask = result['p90_mask']          # blocks selected in ≥90 % of runs
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StochasticPitOptimizer:
    """Monte Carlo pit optimisation under grade and price uncertainty."""

    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        grid_spec: Dict[str, float],
        grade_col: str,
        price: float,
        recovery: float,
        mining_cost: float,
        processing_cost: float,
        geotech_sectors: Optional[List[Any]] = None,
        n_sims: int = 50,
        price_cv: float = 0.15,
        grade_cv: float = 0.10,
        use_kriging_variance: bool = False,
        variance_col: Optional[str] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Run *n_sims* Monte Carlo pit optimisations.

        Parameters
        ----------
        df : DataFrame with normalised x/y/z/tonnes columns.
        grid_spec : dict(nx, ny, nz, xmin, ymin, zmin, xinc, yinc, zinc).
        grade_col : name of grade column.
        price : base commodity price.
        recovery : metallurgical recovery (0–1).
        mining_cost : $/t mined.
        processing_cost : $/t ore.
        geotech_sectors : list of GeoTechSector objects (may be empty).
        n_sims : number of Monte Carlo simulations.
        price_cv : price coefficient of variation (0–1, e.g. 0.15 = 15 %).
        grade_cv : grade CV used when kriging variance is not available.
        use_kriging_variance : if True, use *variance_col* for per-block σ.
        variance_col : column name containing kriging variance values.
        progress_callback : callable(pct, message) or None.

        Returns
        -------
        dict with keys:
            pit_probability – float32 ndarray (0.0–1.0), one value per block
            p90_mask        – bool ndarray; True where pit_probability ≥ 0.90
            p50_mask        – bool ndarray; True where pit_probability ≥ 0.50
            p10_mask        – bool ndarray; True where pit_probability ≥ 0.10
            selection_count – int32 ndarray; raw selection counts
            n_sims          – number of simulations completed
        """
        # Lazy import so the module is usable standalone
        from .pit_optimizer import (
            _normalize_columns,
            build_azimuth_slope_precedence,
            lerchs_grossmann_optimize,
        )

        df = df.copy()

        # Normalise column names
        try:
            df = _normalize_columns(df)
        except Exception:
            pass

        # Tonnes
        if 'tonnes' not in df.columns:
            if 'volume' in df.columns and 'density' in df.columns:
                df['tonnes'] = df['volume'] * df['density']
            else:
                df['tonnes'] = np.ones(len(df))

        tonnes = df['tonnes'].values.astype(np.float64)
        base_grade = df[grade_col].values.astype(np.float64)

        # Grid dimensions
        nx = int(grid_spec['nx'])
        ny = int(grid_spec['ny'])
        nz = int(grid_spec['nz'])

        x_unique = np.sort(df['x'].unique())
        y_unique = np.sort(df['y'].unique())
        z_unique = np.sort(df['z'].unique())

        xi_arr = np.searchsorted(x_unique, df['x'].values).astype(np.intp)
        yi_arr = np.searchsorted(y_unique, df['y'].values).astype(np.intp)
        zi_arr = np.searchsorted(z_unique, df['z'].values).astype(np.intp)

        avg_tonnes = float(np.mean(tonnes)) if len(tonnes) else 1.0

        # Per-block grade standard deviations
        if use_kriging_variance and variance_col and variance_col in df.columns:
            grade_std = np.sqrt(df[variance_col].values.clip(0).astype(np.float64))
        else:
            grade_std = np.abs(base_grade) * grade_cv

        # Build precedence graph ONCE
        sectors = geotech_sectors or []
        default_slope = sectors[0].slope_angle if sectors else 45.0
        if progress_callback:
            progress_callback(0, "Building precedence graph…")

        precedence = build_azimuth_slope_precedence(
            grid_spec, sectors, default_slope=default_slope
        )

        # Monte Carlo loop
        n_blocks = len(df)
        selection_count = np.zeros(n_blocks, dtype=np.int32)

        rng = np.random.default_rng()

        for sim_i in range(n_sims):
            if progress_callback:
                pct = int(sim_i / n_sims * 95)
                progress_callback(pct, f"Simulation {sim_i + 1}/{n_sims}")

            # Sample price
            sim_price = float(
                rng.normal(price, price * max(price_cv, 1e-6))
            )
            sim_price = max(price * 0.05, sim_price)

            # Sample grade per block
            sim_grade = rng.normal(base_grade, grade_std).clip(0.0)

            # Block economic values at sampled price/grade
            nsr_per_t = sim_grade * sim_price * recovery
            block_value_1d = nsr_per_t * tonnes - (mining_cost + processing_cost) * tonnes

            # Fill 3-D grid
            block_values_3d = np.full(
                (nx, ny, nz), -mining_cost * avg_tonnes, dtype=np.float64
            )
            block_values_3d[xi_arr, yi_arr, zi_arr] = block_value_1d

            # Run LG
            selected_3d = lerchs_grossmann_optimize(block_values_3d, precedence)

            # Map back to 1-D and accumulate
            selected_1d = selected_3d[xi_arr, yi_arr, zi_arr]
            selection_count += selected_1d.astype(np.int32)

        if progress_callback:
            progress_callback(100, "Done")

        pit_probability = (selection_count / n_sims).astype(np.float32)

        return {
            'pit_probability': pit_probability,
            'p90_mask': pit_probability >= 0.90,
            'p50_mask': pit_probability >= 0.50,
            'p10_mask': pit_probability >= 0.10,
            'selection_count': selection_count,
            'n_sims': n_sims,
        }
