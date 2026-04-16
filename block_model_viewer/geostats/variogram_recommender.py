"""Variogram settings recommender — canonical public import path.

This module exposes ``recommend_variogram_settings`` as the canonical
``geostats`` entry point so that UI panels and the variogram assistant
don't need to reach into ``block_model_viewer/models/variogram3d.py``.

Current behaviour (consolidation pass B3):
    The body still lives in ``models/variogram3d.py`` because it depends
    on a handful of legacy-only helpers (``calculate_auto_lags``,
    ``_score_variogram_model_family``, ``_weighted_rmse``, …) that
    haven't been extracted yet, and it currently calls the legacy
    ``run_variogram_pipeline`` internally to run its probe fits.

    This module is deliberately a thin delegator so callers can switch
    their imports to ``geostats.variogram_recommender`` now. A follow-up
    commit after the Phase C caller redirects can move the
    implementation here proper and have it call
    ``run_variogram_pipeline_v2`` instead, at which point the legacy
    module can be deleted.
"""

from __future__ import annotations

from ..models.variogram3d import recommend_variogram_settings

__all__ = ["recommend_variogram_settings"]
