"""Shared coded-domain filtering helpers for geostatistical UI panels."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class CodedDomainFilterMixin:
    """Attach imported lithology codes and filter by explicit domain values."""

    _DOMAIN_IGNORE_COLUMNS = {
        "interval_id",
        "global_interval_id",
        "method",
        "weighting",
        "element_weights",
        "merged_partial",
        "merged_partial_auto",
        "source_type",
        "data_source_type",
    }
    _DOMAIN_NAME_HINTS = ("domain", "lith", "unit", "rock", "zone", "facies", "ore")

    def _get_domain_combo_widget(self):
        return getattr(self, "domain_combo", None)

    def _get_domain_all_label(self) -> str:
        return "All Data"

    def _copy_dataframe_attrs(self, target_df: pd.DataFrame, source_df: pd.DataFrame) -> pd.DataFrame:
        if hasattr(source_df, "attrs") and source_df.attrs:
            target_df.attrs = source_df.attrs.copy()
        return target_df

    def _get_domain_source_dataframe(self) -> Optional[pd.DataFrame]:
        for attr in ("data_df", "drillhole_data"):
            df = getattr(self, attr, None)
            if isinstance(df, pd.DataFrame):
                return df
        return None

    def _get_registry_dataframe_for_domain_filter(
        self,
        registry_payload: Any = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        payload = registry_payload if registry_payload is not None else getattr(self, "_registry_data", None)
        if isinstance(payload, pd.DataFrame):
            return payload, None
        if not isinstance(payload, dict):
            return None, None

        candidate_keys: list[str] = []
        preferred_getter = getattr(self, "_preferred_source_registry_keys", None)
        if callable(preferred_getter):
            try:
                candidate_keys.extend(str(key) for key in preferred_getter())
            except Exception:
                logger.debug("Failed to read preferred source registry keys", exc_info=True)

        candidate_keys.extend([
            "weighted_dataframe",
            "composites",
            "composites_df",
            "assays",
            "assays_df",
        ])

        seen: set[str] = set()
        for key in candidate_keys:
            if key in seen:
                continue
            seen.add(key)
            candidate = payload.get(key)
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                return candidate, key
        return None, None

    def _find_interval_column(self, df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
        if df is None or not isinstance(df, pd.DataFrame):
            return None
        lookup = {str(c).strip().lower() for c in candidates}
        for col in df.columns:
            if str(col).strip().lower() in lookup:
                return col
        return None

    def _get_registry_lithology_dataframe(self, registry_payload: Any = None) -> Optional[pd.DataFrame]:
        candidates = [registry_payload, getattr(self, "_registry_data", None)]
        for payload in candidates:
            if isinstance(payload, dict):
                lithology = payload.get("lithology")
                if isinstance(lithology, pd.DataFrame) and not lithology.empty:
                    return lithology

        registry_getter = getattr(self, "get_registry", None)
        if not callable(registry_getter):
            return None

        try:
            registry = registry_getter()
        except Exception:
            return None
        if registry is None or not hasattr(registry, "get_drillhole_data"):
            return None

        payload = None
        try:
            payload = registry.get_drillhole_data(copy_data=False)
        except TypeError:
            payload = registry.get_drillhole_data()
        except Exception:
            return None

        if isinstance(payload, dict):
            lithology = payload.get("lithology")
            if isinstance(lithology, pd.DataFrame) and not lithology.empty:
                return lithology
        return None

    def _get_registry_indicator_rbf_domain(self, registry_payload: Any = None) -> Optional[Dict[str, Any]]:
        candidates = [registry_payload, getattr(self, "_registry_data", None)]
        for payload in candidates:
            if isinstance(payload, dict):
                irbf = payload.get("indicator_rbf_domain")
                if isinstance(irbf, dict):
                    return irbf

        registry_getter = getattr(self, "get_registry", None)
        if not callable(registry_getter):
            return None

        try:
            registry = registry_getter()
        except Exception:
            return None
        if registry is None:
            return None

        getter = getattr(registry, "get_indicator_rbf_domain", None)
        if not callable(getter):
            return None
        try:
            irbf = getter()
        except Exception:
            return None
        return irbf if isinstance(irbf, dict) else None

    def _annotate_with_lithology(
        self,
        base_df: pd.DataFrame,
        lithology_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if (
            base_df is None
            or not isinstance(base_df, pd.DataFrame)
            or base_df.empty
            or lithology_df is None
            or not isinstance(lithology_df, pd.DataFrame)
            or lithology_df.empty
        ):
            return base_df

        hole_col = self._find_interval_column(base_df, ["HOLEID", "holeid", "hole_id", "BHID"])
        from_col = self._find_interval_column(base_df, ["FROM", "from", "depth_from", "start"])
        to_col = self._find_interval_column(base_df, ["TO", "to", "depth_to", "end"])

        lith_hole_col = self._find_interval_column(lithology_df, ["HOLEID", "holeid", "hole_id", "BHID"])
        lith_from_col = self._find_interval_column(lithology_df, ["FROM", "from", "depth_from", "start"])
        lith_to_col = self._find_interval_column(lithology_df, ["TO", "to", "depth_to", "end"])
        lith_code_col = self._find_interval_column(
            lithology_df,
            ["lith_code", "lithology", "LITHOLOGY", "LITH", "code", "unit", "rock_type"],
        )

        if not all([hole_col, from_col, to_col, lith_hole_col, lith_from_col, lith_to_col, lith_code_col]):
            return base_df

        if "lith_code" in base_df.columns and base_df["lith_code"].notna().any():
            return base_df

        enriched = base_df.copy()
        self._copy_dataframe_attrs(enriched, base_df)
        if "lith_code" not in enriched.columns:
            enriched["lith_code"] = None

        lith_groups = {
            str(hole_id): grp
            for hole_id, grp in lithology_df.groupby(lith_hole_col, sort=False)
        }

        for idx, row in enriched.iterrows():
            hole_id = str(row[hole_col])
            intervals = lith_groups.get(hole_id)
            if intervals is None or intervals.empty:
                continue

            midpoint = 0.5 * (float(row[from_col]) + float(row[to_col]))
            matches = intervals[
                (intervals[lith_from_col].astype(float) <= midpoint)
                & (intervals[lith_to_col].astype(float) >= midpoint)
            ]
            if not matches.empty:
                enriched.at[idx, "lith_code"] = str(matches.iloc[0][lith_code_col])

        return enriched

    def _annotate_with_indicator_rbf_domain(
        self,
        base_df: pd.DataFrame,
        irbf_domain: Optional[Dict[str, Any]],
    ) -> pd.DataFrame:
        if (
            base_df is None
            or not isinstance(base_df, pd.DataFrame)
            or base_df.empty
            or irbf_domain is None
            or not isinstance(irbf_domain, dict)
        ):
            return base_df

        labels = irbf_domain.get("sample_domain_labels")
        if labels is None:
            return base_df

        target_col = str(irbf_domain.get("sample_domain_column") or "IRBF_Domain")
        if target_col in base_df.columns:
            existing = base_df[target_col].dropna().astype(str).str.strip()
            informative = existing[existing.str.lower() != "unclassified"]
            if not informative.empty:
                return base_df

        def _assign_by_scalar_keys(series: pd.Series, key_values, df_col) -> bool:
            if key_values is None or df_col is None or len(key_values) != len(labels_list):
                return False
            assigned = pd.Series(
                labels_list,
                index=pd.Index([str(value).strip() for value in key_values]),
                dtype=object,
            )
            target_keys = base_df[df_col].astype(str).str.strip()
            common = pd.Index(target_keys).intersection(assigned.index)
            if common.empty:
                return False
            mask = target_keys.isin(common)
            if not mask.any():
                return False
            series.loc[mask] = target_keys.loc[mask].map(assigned).to_numpy(dtype=object)
            return True

        def _assign_by_interval_triples(series: pd.Series) -> bool:
            hole_values = irbf_domain.get("sample_hole_ids")
            from_values = irbf_domain.get("sample_from_values")
            to_values = irbf_domain.get("sample_to_values")
            if (
                hole_values is None
                or from_values is None
                or to_values is None
                or len(hole_values) != len(labels_list)
                or len(from_values) != len(labels_list)
                or len(to_values) != len(labels_list)
            ):
                return False

            hole_col = self._find_interval_column(base_df, ["HOLEID", "holeid", "hole_id", "BHID"])
            from_col = self._find_interval_column(base_df, ["FROM", "from", "depth_from", "start"])
            to_col = self._find_interval_column(base_df, ["TO", "to", "depth_to", "end"])
            if hole_col is None or from_col is None or to_col is None:
                return False

            key_map = {}
            for hole, from_val, to_val, label in zip(hole_values, from_values, to_values, labels_list):
                try:
                    key = (
                        str(hole).strip(),
                        round(float(from_val), 6),
                        round(float(to_val), 6),
                    )
                except (TypeError, ValueError):
                    continue
                key_map[key] = label

            if not key_map:
                return False

            matched = False
            hole_series = base_df[hole_col].astype(str).str.strip()
            from_series = pd.to_numeric(base_df[from_col], errors="coerce")
            to_series = pd.to_numeric(base_df[to_col], errors="coerce")
            for idx, hole, from_val, to_val in zip(series.index, hole_series, from_series, to_series):
                if pd.isna(from_val) or pd.isna(to_val):
                    continue
                label = key_map.get((hole, round(float(from_val), 6), round(float(to_val), 6)))
                if label is None:
                    continue
                series.at[idx] = label
                matched = True
            return matched

        labels_list = [str(v) if pd.notna(v) else "Unclassified" for v in labels]
        mapped = pd.Series("Unclassified", index=base_df.index, dtype=object)
        mapped_any = False

        sample_index = irbf_domain.get("sample_domain_index")
        if sample_index is not None and len(sample_index) == len(labels_list):
            assigned = pd.Series(labels_list, index=pd.Index(sample_index), dtype=object)
            common = mapped.index.intersection(assigned.index)
            if not common.empty:
                mapped.loc[common] = assigned.reindex(common).to_numpy(dtype=object)
                mapped_any = True
            elif len(labels_list) == len(base_df):
                mapped[:] = labels_list
                mapped_any = True
        elif len(labels_list) == len(base_df):
            mapped[:] = labels_list
            mapped_any = True

        if not mapped_any:
            mapped_any = _assign_by_scalar_keys(
                mapped,
                irbf_domain.get("sample_global_interval_ids"),
                self._find_interval_column(base_df, ["GLOBAL_INTERVAL_ID", "global_interval_id"]),
            )
        if not mapped_any:
            mapped_any = _assign_by_scalar_keys(
                mapped,
                irbf_domain.get("sample_interval_ids"),
                self._find_interval_column(base_df, ["INTERVAL_ID", "interval_id", "SAMPLE_ID", "sample_id"]),
            )
        if not mapped_any:
            mapped_any = _assign_by_interval_triples(mapped)

        # Spatial fallback: when no sample-identifier matching works, classify
        # each row by its (X, Y, Z) coordinates against the IRBF probability
        # field / inside_mask. This handles the common case where composites
        # were re-loaded from a project after IRBF ran, so the original sample
        # identifiers are gone but the spatial positions still match.
        if not mapped_any:
            try:
                _x_names = ("X", "x", "XC", "xc", "EAST", "east", "EASTING", "easting",
                            "mid_x", "MID_X", "midx", "x_mid", "X_MID", "xmid", "XMID",
                            "X_CENTROID", "x_centroid", "x_centre", "X_CENTRE")
                _y_names = ("Y", "y", "YC", "yc", "NORTH", "north", "NORTHING", "northing",
                            "mid_y", "MID_Y", "midy", "y_mid", "Y_MID", "ymid", "YMID",
                            "Y_CENTROID", "y_centroid", "y_centre", "Y_CENTRE")
                _z_names = ("Z", "z", "ZC", "zc", "RL", "rl", "ELEV", "elev", "ELEVATION",
                            "mid_z", "MID_Z", "midz", "z_mid", "Z_MID", "zmid", "ZMID",
                            "Z_CENTROID", "z_centroid", "z_centre", "Z_CENTRE")
                _xcol = next((c for c in _x_names if c in base_df.columns), None)
                _ycol = next((c for c in _y_names if c in base_df.columns), None)
                _zcol = next((c for c in _z_names if c in base_df.columns), None)
                if _xcol and _ycol and _zcol:
                    import numpy as _np
                    _coords = base_df[[_xcol, _ycol, _zcol]].to_numpy(dtype=float)
                    # Only classify rows with finite XYZ
                    _ok = _np.isfinite(_coords).all(axis=1)
                    if _ok.any():
                        from ...geostats.domain_mask import resample_irbf_mask_to_points
                        _mask = resample_irbf_mask_to_points(irbf_domain, _coords[_ok])
                        if _mask is not None and len(_mask) == int(_ok.sum()):
                            _labels = _np.full(len(base_df), "Unclassified", dtype=object)
                            _ok_idx = _np.where(_ok)[0]
                            _labels[_ok_idx[_mask]] = "Inside"
                            _labels[_ok_idx[~_mask]] = "Outside"
                            mapped = pd.Series(_labels, index=base_df.index, dtype=object)
                            mapped_any = True
            except Exception as _exc:
                logger.debug("IRBF spatial fallback failed: %s", _exc)

        if not mapped_any:
            return base_df

        enriched = base_df.copy()
        self._copy_dataframe_attrs(enriched, base_df)
        enriched[target_col] = pd.Categorical(mapped)
        return enriched

    def _is_domain_like_column(self, column_name: str) -> bool:
        name = str(column_name).strip().lower()
        return any(hint in name for hint in self._DOMAIN_NAME_HINTS)

    def _get_domain_candidate_columns(self, df: pd.DataFrame) -> list[str]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return []

        candidates: list[str] = []
        for col in df.columns:
            lower = str(col).strip().lower()
            if lower in self._DOMAIN_IGNORE_COLUMNS:
                continue

            series = df[col]
            dtype = series.dtype
            is_categorical = isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype)
            if not is_categorical:
                continue

            non_null = series.dropna()
            if non_null.empty:
                continue

            unique_values = sorted({str(v) for v in non_null.tolist()}, key=str.casefold)
            max_unique = 200 if self._is_domain_like_column(col) else 30
            if not 2 <= len(unique_values) <= max_unique:
                continue
            if not all(len(value) <= 64 for value in unique_values[:50]):
                continue

            candidates.append(str(col))

        return sorted(candidates, key=lambda name: (not self._is_domain_like_column(name), name.lower()))

    def _populate_domain_filter_combo(
        self,
        df: pd.DataFrame,
        combo=None,
        all_label: Optional[str] = None,
    ) -> None:
        combo = combo or self._get_domain_combo_widget()
        if combo is None:
            return

        all_label = all_label or self._get_domain_all_label()
        current = combo.currentText().strip() if combo.count() else all_label

        combo.blockSignals(True)
        combo.clear()
        combo.addItem(all_label)

        seen_entries: set[str] = set()
        if isinstance(df, pd.DataFrame) and not df.empty:
            for column in self._get_domain_candidate_columns(df):
                values = sorted({str(v) for v in df[column].dropna().tolist()}, key=str.casefold)
                for value in values:
                    entry = f"{column}: {value}"
                    if entry not in seen_entries:
                        combo.addItem(entry)
                        seen_entries.add(entry)

        # Unconditionally inject IRBF_Domain entries when registry has an
        # IRBF domain, regardless of whether the dataframe was successfully
        # enriched. _apply_domain_filter handles on-the-fly classification.
        try:
            _irbf = self._get_registry_indicator_rbf_domain(getattr(self, "_registry_data", None))
        except Exception:
            _irbf = None
        if isinstance(_irbf, dict):
            for entry in ("IRBF_Domain: Inside", "IRBF_Domain: Outside"):
                if entry not in seen_entries:
                    combo.addItem(entry)
                    seen_entries.add(entry)

        if combo.findText(current) >= 0:
            combo.setCurrentText(current)
        else:
            combo.setCurrentText(all_label)
        combo.blockSignals(False)

    def _on_composites_refreshed(self, data) -> None:
        """Slot for registry.compositesLoaded — refreshes domain filter combo.

        Connect this in any panel that uses CodedDomainFilterMixin so that
        Indicator RBF (and other tools that inject domain columns into composites)
        are immediately reflected in the domain filter without a full data reload.
        """
        df = data if isinstance(data, pd.DataFrame) else None
        if df is None and isinstance(data, dict):
            for key in ("composites", "composites_df", "assays"):
                v = data.get(key)
                if isinstance(v, pd.DataFrame) and not v.empty:
                    df = v
                    break
        if df is not None:
            enriched = self._prepare_domain_filter_dataframe(
                df,
                registry_payload=getattr(self, "_registry_data", None),
                populate_combo=False,
            )

            registry_payload = getattr(self, "_registry_data", None)
            if isinstance(registry_payload, dict):
                updated_payload = dict(registry_payload)
                if "composites" in updated_payload or "composites_df" in updated_payload:
                    updated_payload["composites"] = enriched
                self._registry_data = updated_payload

            using_raw_assays = False
            source_combo = getattr(self, "source_combo", None)
            if source_combo is not None and hasattr(source_combo, "currentText"):
                using_raw_assays = source_combo.currentText() == "Raw Assays"

            if not using_raw_assays:
                extractor = getattr(self, "_extract_dataframe", None)
                if callable(extractor):
                    try:
                        extractor(getattr(self, "_registry_data", None) or enriched)
                    except Exception:
                        logger.debug("Failed to refresh dataframe after composites update", exc_info=True)
                else:
                    for attr in ("drillhole_data", "data_df"):
                        current_df = getattr(self, attr, None)
                        if isinstance(current_df, pd.DataFrame):
                            if current_df.index.equals(enriched.index) or len(current_df) == len(enriched):
                                setattr(self, attr, self._copy_dataframe_attrs(enriched.copy(), enriched))
                                break

            self._populate_domain_filter_combo(enriched)

    def _on_indicator_rbf_domain_loaded(self, _domain_data=None) -> None:
        """Slot for registry.indicatorRBFDomainLoaded — re-enriches and repopulates combo.

        Connect this in any panel that uses CodedDomainFilterMixin so that
        a newly registered Indicator RBF domain is immediately available in
        the domain filter combo without a full data reload.
        """
        # Find the panel's current working dataframe
        df = None
        source_payload_key = None
        for attr in ("drillhole_data", "data_df"):
            candidate = getattr(self, attr, None)
            if isinstance(candidate, pd.DataFrame) and not candidate.empty:
                df = candidate
                break

        registry_payload = getattr(self, "_registry_data", None)
        if df is None:
            df, source_payload_key = self._get_registry_dataframe_for_domain_filter(registry_payload)

        if df is None:
            registry_getter = getattr(self, "get_registry", None)
            if callable(registry_getter):
                try:
                    registry = registry_getter()
                except Exception:
                    registry = None
                if registry is not None:
                    try:
                        registry_payload = registry.get_estimation_ready_data()
                    except Exception:
                        registry_payload = None
                    if registry_payload is None and hasattr(registry, "get_drillhole_data"):
                        try:
                            registry_payload = registry.get_drillhole_data()
                        except Exception:
                            registry_payload = None
                    if registry_payload is not None:
                        self._registry_data = registry_payload
                        df, source_payload_key = self._get_registry_dataframe_for_domain_filter(registry_payload)

        if df is None:
            return

        # Strip stale IRBF_Domain so re-enrichment picks up the new one
        irbf_col = "IRBF_Domain"
        if irbf_col in df.columns:
            df = df.drop(columns=[irbf_col])

        enriched = self._prepare_domain_filter_dataframe(
            df,
            registry_payload=getattr(self, "_registry_data", None),
            populate_combo=False,
        )

        # Push enriched frame back into the panel's data attribute
        for attr in ("drillhole_data", "data_df"):
            if getattr(self, attr, None) is not None:
                setattr(self, attr, enriched)
                break

        # Update registry payload cache if dict-based
        if isinstance(registry_payload, dict):
            updated = dict(registry_payload)
            if source_payload_key and source_payload_key in updated:
                updated[source_payload_key] = enriched
            elif "weighted_dataframe" in updated:
                updated["weighted_dataframe"] = enriched
            else:
                for key in ("composites", "composites_df", "assays", "assays_df"):
                    if key in updated:
                        updated[key] = enriched
            self._registry_data = updated
            registry_payload = updated

        extractor = getattr(self, "_extract_dataframe", None)
        if callable(extractor):
            try:
                extractor(registry_payload if registry_payload is not None else enriched)
            except Exception:
                logger.debug("Failed to refresh dataframe after IRBF domain update", exc_info=True)

        self._populate_domain_filter_combo(enriched)

    def _prepare_domain_filter_dataframe(
        self,
        df: pd.DataFrame,
        registry_payload: Any = None,
        *,
        populate_combo: bool = False,
        combo=None,
        all_label: Optional[str] = None,
    ) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame):
            return df

        enriched = self._annotate_with_lithology(
            df,
            self._get_registry_lithology_dataframe(registry_payload),
        )
        enriched = self._annotate_with_indicator_rbf_domain(
            enriched,
            self._get_registry_indicator_rbf_domain(registry_payload),
        )

        if populate_combo:
            self._populate_domain_filter_combo(
                enriched,
                combo=combo,
                all_label=all_label,
            )
        return enriched

    def _apply_domain_filter(
        self,
        df: pd.DataFrame,
        combo=None,
        all_label: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return df, {}

        combo = combo or self._get_domain_combo_widget()
        all_label = all_label or self._get_domain_all_label()
        selection = combo.currentText().strip() if combo is not None else all_label

        if selection in ("", all_label, "(none)", "All Domains"):
            metadata: Dict[str, Any] = {"domain_filter_selection": None}
            self._active_domain_filter_metadata = metadata
            return df, metadata

        if ":" in selection:
            column, raw_value = selection.split(":", 1)
            column = column.strip()
            domain_value = raw_value.strip()
            if column in df.columns:
                filtered = df[df[column].astype(str) == domain_value].copy()
                self._copy_dataframe_attrs(filtered, df)
                metadata = {
                    "domain_filter_selection": selection,
                    "domain_filter_column": column,
                    "domain_filter_value": domain_value,
                    "domain_filter_n_samples": len(filtered),
                }
                self._active_domain_filter_metadata = metadata
                return filtered, metadata

            # On-the-fly spatial classification for IRBF_Domain when the
            # column is missing from the dataframe but the registry has an
            # IRBF domain. Classifies each row by its XYZ coordinates.
            if column.lower() in ("irbf_domain", "irbfdomain", "irbf"):
                try:
                    irbf = self._get_registry_indicator_rbf_domain(getattr(self, "_registry_data", None))
                except Exception:
                    irbf = None
                if isinstance(irbf, dict):
                    _x_names = ("X", "x", "XC", "xc", "EAST", "east", "EASTING", "easting",
                                "mid_x", "MID_X", "midx", "x_mid", "X_MID", "xmid", "XMID")
                    _y_names = ("Y", "y", "YC", "yc", "NORTH", "north", "NORTHING", "northing",
                                "mid_y", "MID_Y", "midy", "y_mid", "Y_MID", "ymid", "YMID")
                    _z_names = ("Z", "z", "ZC", "zc", "RL", "rl", "ELEV", "elev",
                                "mid_z", "MID_Z", "midz", "z_mid", "Z_MID", "zmid", "ZMID")
                    _xcol = next((c for c in _x_names if c in df.columns), None)
                    _ycol = next((c for c in _y_names if c in df.columns), None)
                    _zcol = next((c for c in _z_names if c in df.columns), None)
                    if _xcol and _ycol and _zcol:
                        try:
                            import numpy as _np
                            from ...geostats.domain_mask import resample_irbf_mask_to_points
                            _coords = df[[_xcol, _ycol, _zcol]].to_numpy(dtype=float)
                            _ok = _np.isfinite(_coords).all(axis=1)
                            _mask = resample_irbf_mask_to_points(irbf, _coords[_ok])
                            if _mask is not None and len(_mask) == int(_ok.sum()):
                                want_inside = domain_value.lower() == "inside"
                                keep = _np.zeros(len(df), dtype=bool)
                                _ok_idx = _np.where(_ok)[0]
                                keep[_ok_idx[_mask if want_inside else ~_mask]] = True
                                filtered = df.iloc[keep].copy()
                                self._copy_dataframe_attrs(filtered, df)
                                metadata = {
                                    "domain_filter_selection": selection,
                                    "domain_filter_column": column,
                                    "domain_filter_value": domain_value,
                                    "domain_filter_n_samples": len(filtered),
                                    "domain_filter_source": "irbf_spatial_fallback",
                                }
                                self._active_domain_filter_metadata = metadata
                                return filtered, metadata
                        except Exception:
                            logger.debug("On-the-fly IRBF spatial classification failed", exc_info=True)

            metadata = {
                "domain_filter_selection": selection,
                "domain_filter_missing_column": column,
            }
            self._active_domain_filter_metadata = metadata
            return df, metadata

        if selection in df.columns:
            filtered = df.dropna(subset=[selection]).copy()
            self._copy_dataframe_attrs(filtered, df)
            metadata = {
                "domain_filter_selection": selection,
                "domain_filter_column": selection,
                "domain_filter_value": None,
                "domain_filter_n_samples": len(filtered),
            }
            self._active_domain_filter_metadata = metadata
            return filtered, metadata

        metadata = {"domain_filter_selection": selection}
        self._active_domain_filter_metadata = metadata
        return df, metadata

    def _get_current_domain_filtered_data(
        self,
        df: Optional[pd.DataFrame] = None,
        registry_payload: Any = None,
        *,
        combo=None,
        all_label: Optional[str] = None,
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        base_df = df if isinstance(df, pd.DataFrame) else self._get_domain_source_dataframe()
        if base_df is None or not isinstance(base_df, pd.DataFrame):
            return None, {}

        enriched = self._prepare_domain_filter_dataframe(
            base_df,
            registry_payload=registry_payload,
            populate_combo=False,
        )

        return self._apply_domain_filter(enriched, combo=combo, all_label=all_label)

    def _merge_domain_filter_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata = getattr(self, "_active_domain_filter_metadata", None) or {}
        if not metadata or not isinstance(payload, dict):
            return payload

        selection = metadata.get("domain_filter_selection")
        if not selection:
            return payload

        payload_metadata = payload.setdefault("metadata", {})
        payload_metadata.update(metadata)
        if (
            "domain_filter_n_samples" in metadata
            and "source_data_n_samples" not in payload_metadata
        ):
            payload_metadata["source_data_n_samples"] = metadata["domain_filter_n_samples"]
        return payload

    def _on_domain_filter_selection_changed(self, _selection: str) -> None:
        base_df = self._get_domain_source_dataframe()
        if base_df is None or not isinstance(base_df, pd.DataFrame) or base_df.empty:
            return

        filtered, metadata = self._get_current_domain_filtered_data(df=base_df)
        total = len(base_df)
        kept = len(filtered) if isinstance(filtered, pd.DataFrame) else total
        selection = metadata.get("domain_filter_selection") or self._get_domain_all_label()

        log_message = f"Domain filter: {selection} ({kept}/{total} samples)"
        log_fn = getattr(self, "_log_event", None) or getattr(self, "_log", None)
        if callable(log_fn):
            try:
                log_fn(log_message, "info")
            except TypeError:
                log_fn(log_message)
            except Exception:
                logger.debug("Failed to log domain filter change", exc_info=True)

        auto_fit = getattr(self, "auto_fit_grid_check", None)
        if auto_fit is not None and hasattr(auto_fit, "isChecked") and auto_fit.isChecked():
            for method_name in ("_auto_detect_grid", "_on_auto_detect_grid"):
                handler = getattr(self, method_name, None)
                if callable(handler):
                    try:
                        handler()
                    except Exception:
                        logger.debug("Domain-triggered auto-fit failed", exc_info=True)
                    break

        # Clear stale estimation/simulation results from previous domain
        _result_attrs = (
            "kriging_results", "arbf_results", "sgsim_results",
            "simulation_results", "fastrbf_results",
        )
        for attr in _result_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)
                logger.debug("Cleared stale %s on domain change", attr)

        # Remove stale visualization layer from 3D viewer
        _last_layer = getattr(self, "_last_estimation_layer_name", None)
        if _last_layer:
            try:
                mw = getattr(self, "main_window", None)
                _vc = getattr(mw, "vis_controller", None) if mw else None
                if _vc and hasattr(_vc, "renderer") and hasattr(_vc.renderer, "clear_layer"):
                    if _last_layer in _vc.renderer.active_layers:
                        _vc.renderer.clear_layer(_last_layer)
                        logger.info("Cleared stale 3D layer '%s' on domain change", _last_layer)
                self._last_estimation_layer_name = None
            except Exception:
                logger.debug("Failed to clear stale visualization layer", exc_info=True)

        # Check NS transform availability for new domain
        try:
            self._check_ns_transform_for_domain()
        except Exception as _ns_exc:
            logger.debug("NS transform domain check failed: %s", _ns_exc)

    # ------------------------------------------------------------------
    # Per-domain NS transform warning
    # ------------------------------------------------------------------

    def _check_ns_transform_for_domain(self) -> None:
        """Show a warning if NS transform is enabled but no per-domain transformer exists."""
        # Only relevant for panels that use NS transforms
        ns_enabled = False
        for attr in ("chk_normal_score", "use_ns_check", "ns_checkbox"):
            chk = getattr(self, attr, None)
            if chk is not None and hasattr(chk, "isChecked") and chk.isChecked():
                ns_enabled = True
                break
        if not ns_enabled:
            return

        domain_meta = getattr(self, "_active_domain_filter_metadata", {}) or {}
        domain_val = domain_meta.get("domain_filter_value") or "(global)"
        if domain_val == "(global)":
            return

        registry = getattr(self, "registry", None)
        if registry is None or not hasattr(registry, "list_transformer_domains"):
            return

        stored = registry.list_transformer_domains()
        if domain_val not in stored and "(global)" not in stored:
            log_fn = getattr(self, "_log_event", None) or getattr(self, "_log", None)
            msg = (
                f"No NS transformer for domain '{domain_val}'. "
                f"Back-transform will use global transformer."
            )
            if callable(log_fn):
                try:
                    log_fn(msg, "warning")
                except TypeError:
                    log_fn(msg)
            logger.warning("%s: %s", type(self).__name__, msg)

    # ------------------------------------------------------------------
    # Per-domain variogram auto-loading
    # ------------------------------------------------------------------

    def _reload_variogram_for_domain(self) -> None:
        """Look up a domain-specific variogram from the registry and load it.

        Called when the domain combo changes.  If a per-domain variogram
        exists for the current variable + domain, the panel's spinners are
        updated via ``load_variogram_parameters()``.

        Falls back to the global variogram if no domain-specific one exists,
        and leaves spinners unchanged if no variogram is found at all.

        Panels that use this must have:
        - ``self.registry`` (DataRegistry reference)
        - ``self.variogram_results`` (stores loaded variogram)
        - ``load_variogram_parameters()`` or ``set_variogram_results()``
        - A variable combo accessible via ``_get_current_variable()`` or
          ``self.variable_combo``
        """
        registry = getattr(self, "registry", None)
        if registry is None:
            return

        # Resolve current variable
        var_combo = getattr(self, "variable_combo", None) or getattr(self, "var_combo", None)
        selected_var = var_combo.currentText() if var_combo else None
        if not selected_var:
            return

        # Resolve current domain
        domain_meta = getattr(self, "_active_domain_filter_metadata", {}) or {}
        domain_val = domain_meta.get("domain_filter_value") or "(global)"

        from ..panel_utils import resolve_variogram_for_domain
        vario = resolve_variogram_for_domain(registry, selected_var, domain_val)
        if vario is None:
            logger.info(
                "%s: No variogram found for '%s' domain '%s' — keeping spinners.",
                type(self).__name__, selected_var, domain_val,
            )
            return

        # Store and load into spinners
        self.variogram_results = vario  # type: ignore[attr-defined]

        loader = getattr(self, "load_variogram_parameters", None)
        if callable(loader):
            try:
                loader()
            except Exception:
                logger.debug("load_variogram_parameters failed", exc_info=True)
        else:
            setter = getattr(self, "set_variogram_results", None)
            if callable(setter):
                try:
                    setter(vario)
                except Exception:
                    logger.debug("set_variogram_results failed", exc_info=True)

        logger.info(
            "%s: Loaded variogram for domain '%s' (variable '%s')",
            type(self).__name__, domain_val, selected_var,
        )

    # ------------------------------------------------------------------
    # Pre-run domain readiness validation
    # ------------------------------------------------------------------

    def _validate_domain_readiness(
        self,
        variable_name: Optional[str] = None,
        needs_variogram: bool = True,
        needs_ns_transform: bool = False,
    ) -> tuple:
        """Check that the selected domain has required prerequisites.

        Returns ``(is_ready, warnings, errors)`` where *errors* block
        execution and *warnings* are informational.

        Panels should call this before running and show the messages
        to the user.
        """
        warnings_list: list = []
        errors_list: list = []

        domain_meta = getattr(self, "_active_domain_filter_metadata", {}) or {}
        domain_val = domain_meta.get("domain_filter_value") or "(global)"

        if domain_val == "(global)":
            return True, warnings_list, errors_list

        registry = getattr(self, "registry", None)
        if registry is None:
            return True, warnings_list, errors_list

        if not variable_name:
            var_combo = (
                getattr(self, "variable_combo", None)
                or getattr(self, "var_combo", None)
            )
            variable_name = var_combo.currentText() if var_combo else None

        # Check variogram availability
        if needs_variogram and variable_name and hasattr(registry, "list_variogram_domains"):
            stored = registry.list_variogram_domains(variable_name)
            if domain_val not in stored and "(global)" not in stored:
                errors_list.append(
                    f"No variogram found for variable '{variable_name}' "
                    f"domain '{domain_val}'. Compute a variogram for this "
                    f"domain in the Variogram panel first."
                )
            elif domain_val not in stored:
                warnings_list.append(
                    f"No domain-specific variogram for '{domain_val}'. "
                    f"The global variogram will be used."
                )

        # Check NS transformer availability
        if needs_ns_transform and variable_name and hasattr(registry, "list_transformer_domains"):
            stored_domains = registry.list_transformer_domains()
            if domain_val not in stored_domains and "(global)" not in stored_domains:
                warnings_list.append(
                    f"No NS transformer found for domain '{domain_val}'. "
                    f"Back-transform will use the global transformer."
                )

        is_ready = len(errors_list) == 0
        return is_ready, warnings_list, errors_list
