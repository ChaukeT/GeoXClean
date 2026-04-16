"""
Leapfrog Geo (.aproj) Project File Importer

Reads Leapfrog Geo project files and extracts drillhole data (collars, surveys,
assays) and block models into pandas DataFrames compatible with the GeoX import
pipelines.

Format notes:
  - .aproj files are ZODB (Zope Object Database) SQLite databases
  - Contains a `data` table with (oid, serial, data) columns
  - Each blob has TWO concatenated pickle streams:
    1. The class reference (module.ClassName)
    2. The instance state dict (with ZODB persistent references to other objects)
  - Actual numerical data is stored as embedded Apache Parquet within ResultStore blobs
  - Data is stored column-by-column: each column is a separate database object
  - Category columns (e.g., HOLEID) use index-to-name mapping via a separate parquet

Supported table types:
  - CollarTableBlock   -> collars  (HOLEID, X, Y, Z, MAXDEPTH)
  - SurveyTableBlock   -> surveys  (HOLEID, DEPTH, DIP, AZIMUTH)
  - IntervalTableBlock -> assays   (HOLEID, FROM, TO, + grade columns)
  - Block models       -> discovered via heuristic class matching + orphan column detection
"""

from __future__ import annotations

import io
import logging
import sqlite3
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# Low-level blob parsing
# ════════════════════════════════════════════════════════════════════════════════

_PAR1 = b'PAR1'

# Column block class names that hold drillhole data
_COLUMN_CLASSES = frozenset({
    'NumericColumnBlock', 'CategoryColumnBlock', 'BooleanColumnBlock',
    'CollarZColumnBlock', 'FloatColumnBlock', 'IntColumnBlock',
    'DoubleColumnBlock', 'TextColumnBlock', 'StringColumnBlock',
})

# Table block class names (drillhole)
_TABLE_CLASSES = {
    'CollarTableBlock': 'collar',
    'SurveyTableBlock': 'survey',
    'IntervalTableBlock': 'interval',
    'NumericCompositeEntireDrillholeTableBlock': 'composite',
    'EconomicCompositedTableBlock': 'economic_composite',
}

# Block model root class names (definitions only — data is computed lazily
# by Leapfrog's engine and NOT stored in the .aproj ResultStores).
# Actual class names discovered from real .aproj files:
#   RegularSbmRootBlock  — regular sub-blocked model root
# Heuristic substrings for projects we haven't seen yet:
_BLOCK_MODEL_ROOT_CLASSES = (
    'RegularSbmRootBlock', 'SbmRootBlock',
)
_BLOCK_MODEL_CLASS_PATTERNS = (
    'SbmRoot', 'BlockModelRoot', 'RegularBlockModel', 'OctreeBlockModel',
)

# Internal column names to skip (applies to drillhole columns)
_SKIP_COLNAMES = frozenset({
    'id', '_is_manually_frozen', '_outputs', '_z_on_terrain', 'dogleg_severity',
})

# Leapfrog column name -> GeoX standard column name
_COLLAR_RENAMES = {
    'holeid': 'HOLEID', 'hole_id': 'HOLEID', 'bhid': 'HOLEID',
    'hole': 'HOLEID', 'name': 'HOLEID', 'id': 'HOLEID',
    'x': 'X', 'east': 'X', 'easting': 'X', 'x_collar': 'X',
    'y': 'Y', 'north': 'Y', 'northing': 'Y', 'y_collar': 'Y',
    'z': 'Z', 'elev': 'Z', 'elevation': 'Z', 'rl': 'Z',
    'z_collar': 'Z', 'altitude': 'Z',
    'maxdepth': 'LENGTH', 'max_depth': 'LENGTH', 'length': 'LENGTH',
    'total_depth': 'LENGTH', 'td': 'LENGTH',
    'azimuth': 'AZIMUTH', 'azi': 'AZIMUTH', 'bearing': 'AZIMUTH',
    'dip': 'DIP', 'inclination': 'DIP', 'incl': 'DIP',
}

_SURVEY_RENAMES = {
    'holeid': 'HOLEID', 'hole_id': 'HOLEID', 'bhid': 'HOLEID',
    'hole': 'HOLEID', 'name': 'HOLEID', 'id': 'HOLEID',
    'depth': 'DEPTH', 'md': 'DEPTH', 'measured_depth': 'DEPTH',
    'distance': 'DEPTH', 'at': 'DEPTH',
    'azimuth': 'AZIMUTH', 'azi': 'AZIMUTH', 'bearing': 'AZIMUTH',
    'dip': 'DIP', 'inclination': 'DIP', 'incl': 'DIP',
}

_INTERVAL_RENAMES = {
    'holeid': 'HOLEID', 'hole_id': 'HOLEID', 'bhid': 'HOLEID',
    'hole': 'HOLEID', 'name': 'HOLEID', 'id': 'HOLEID',
    'from': 'FROM', 'depth_from': 'FROM', 'start': 'FROM',
    'from_depth': 'FROM', 'top': 'FROM',
    'to': 'TO', 'depth_to': 'TO', 'end': 'TO',
    'to_depth': 'TO', 'bottom': 'TO',
}


def _get_classname(blob: bytes) -> Optional[str]:
    """
    Extract the class name from the first pickle stream in a blob.

    The first pickle stream contains just a class reference:
    PROTO 4 + FRAME + SHORT_BINUNICODE module + SHORT_BINUNICODE name + STACK_GLOBAL + STOP
    """
    if not blob or len(blob) < 10 or blob[0:2] != b'\x80\x04':
        return None
    strings = []
    i = 2
    while i < min(len(blob), 300):
        if blob[i] == 0x8c:  # SHORT_BINUNICODE
            if i + 1 >= len(blob):
                break
            slen = blob[i + 1]
            if i + 2 + slen > len(blob):
                break
            try:
                s = blob[i + 2:i + 2 + slen].decode('utf-8')
                strings.append(s)
            except UnicodeDecodeError:
                pass
            i += 2 + slen
        elif blob[i] == 0x2e:  # STOP
            break
        else:
            i += 1
    # The second string is the class name (first is the module)
    return strings[1] if len(strings) >= 2 else None


def _get_state_strings(blob: bytes) -> List[str]:
    """
    Extract all string literals from the second pickle stream (state dict).
    """
    second = blob.find(b'\x80\x04', 2)
    if second < 0:
        return []
    strings = []
    j = second
    while j < len(blob):
        if blob[j] == 0x8c and j + 1 < len(blob):
            slen = blob[j + 1]
            if j + 2 + slen <= len(blob):
                try:
                    strings.append(blob[j + 2:j + 2 + slen].decode('utf-8'))
                except UnicodeDecodeError:
                    pass
                j += 2 + slen
            else:
                j += 1
        else:
            j += 1
    return strings


def _find_string_after(strings: List[str], key: str) -> Optional[str]:
    """Find the string immediately following `key` in a list of strings."""
    for i, s in enumerate(strings):
        if s == key and i + 1 < len(strings):
            return strings[i + 1]
    return None


def _extract_persistent_refs(blob: bytes) -> List[int]:
    """
    Extract ZODB persistent reference OIDs from the state pickle.

    Persistent references are encoded as:
    C (SHORT_BINBYTES, opcode 0x43) + 8 (length byte) + 8-byte big-endian OID
    """
    second = blob.find(b'\x80\x04', 2)
    if second < 0:
        return []
    refs = []
    i = second
    while i < len(blob) - 9:
        if blob[i] == 0x43 and blob[i + 1] == 8:
            oid_bytes = blob[i + 2:i + 10]
            oid_val = struct.unpack('>q', oid_bytes)[0]
            refs.append(oid_val)
            i += 10
        else:
            i += 1
    return refs


def _extract_parquets(blob: bytes) -> List[pd.DataFrame]:
    """
    Find and extract all Parquet files embedded within a binary blob.

    Parquet files start and end with the magic bytes 'PAR1'.
    """
    positions = []
    offset = 0
    while True:
        idx = blob.find(_PAR1, offset)
        if idx == -1:
            break
        positions.append(idx)
        offset = idx + 4

    if len(positions) < 2:
        return []

    dfs = []
    for i in range(0, len(positions) - 1):
        start = positions[i]
        end = positions[i + 1] + 4
        chunk = blob[start:end]
        if len(chunk) < 12:
            continue
        try:
            df = pd.read_parquet(io.BytesIO(chunk))
            if not df.empty:
                dfs.append(df)
        except Exception:
            continue
    return dfs


# ════════════════════════════════════════════════════════════════════════════════
# Data structures
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class _ColumnInfo:
    """A detected drillhole column."""
    block_oid: int
    classname: str
    colname: str
    label: str
    parent_table_oid: int
    result_store_oid: int
    is_category: bool


@dataclass
class _TableInfo:
    """A detected drillhole table."""
    oid: int
    classname: str
    table_type: str  # 'collar', 'survey', 'interval', 'composite', ...
    label: str
    columns: List[_ColumnInfo] = field(default_factory=list)


@dataclass
class _BlockModelInfo:
    """A discovered block model candidate."""
    parent_oid: int
    parent_classname: str
    label: str
    columns: List[_ColumnInfo] = field(default_factory=list)
    grid_params: Dict[str, Any] = field(default_factory=dict)


def _has_xyz_columns(df: pd.DataFrame) -> bool:
    """Check if a DataFrame has X, Y, Z position columns."""
    cols_lower = {c.lower() for c in df.columns}
    return (
        ('x' in cols_lower and 'y' in cols_lower and 'z' in cols_lower)
        or ('xc' in cols_lower and 'yc' in cols_lower and 'zc' in cols_lower)
        or ('easting' in cols_lower and 'northing' in cols_lower
            and 'elevation' in cols_lower)
    )


# ════════════════════════════════════════════════════════════════════════════════
# Main importer
# ════════════════════════════════════════════════════════════════════════════════

class LeapfrogImporter:
    """
    Imports drillhole data and block models from Leapfrog Geo .aproj project files.

    Usage::

        importer = LeapfrogImporter(path)
        result = importer.read()
        # result['collars']       -> pd.DataFrame or None
        # result['surveys']       -> pd.DataFrame or None
        # result['assays']        -> pd.DataFrame or None
        # result['block_models']  -> list of dicts with 'name', 'dataframe', ...
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")

        self._blobs: Dict[int, bytes] = {}        # oid -> raw blob
        self._classnames: Dict[int, str] = {}     # oid -> class short name
        self._tables: List[_TableInfo] = []
        self._block_models: List[_BlockModelInfo] = []

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def read(self) -> Dict[str, Any]:
        """
        Read the .aproj file and return extracted drillhole + block model data.

        Returns dict with keys:
            'collars', 'surveys', 'assays', 'tables', 'block_models', 'metadata'
        """
        logger.info(f"Reading Leapfrog project: {self.path}")

        self._load_blobs()
        self._discover_structure()
        self._discover_block_models()
        result = self._build_dataframes()

        return result

    # ──────────────────────────────────────────────────────────────────────
    # Step 1: Load all blobs and extract class names
    # ──────────────────────────────────────────────────────────────────────

    def _load_blobs(self):
        """Load all blobs from the SQLite data table."""
        conn = sqlite3.connect(str(self.path))
        try:
            cursor = conn.execute("SELECT oid, data FROM data")
            for oid, blob in cursor:
                if not blob:
                    continue
                self._blobs[oid] = blob
                cn = _get_classname(blob)
                if cn:
                    self._classnames[oid] = cn
        finally:
            conn.close()

        logger.info(f"Loaded {len(self._blobs)} objects from project file")

        # Log class distribution (all classes — helps identify unknown BM types)
        class_counts: Dict[str, int] = {}
        for cn in self._classnames.values():
            class_counts[cn] = class_counts.get(cn, 0) + 1
        for cn, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
            logger.debug(f"  {cn}: {cnt}")

    # ──────────────────────────────────────────────────────────────────────
    # Step 2: Discover tables and columns via ZODB references
    # ──────────────────────────────────────────────────────────────────────

    def _discover_structure(self):
        """Find table blocks, then find their column blocks and data stores."""

        # 2a. Find table blocks
        for oid, cn in self._classnames.items():
            if cn in _TABLE_CLASSES:
                blob = self._blobs[oid]
                strings = _get_state_strings(blob)
                label = _find_string_after(strings, 'label') or cn
                table_type = _TABLE_CLASSES[cn]
                self._tables.append(_TableInfo(
                    oid=oid, classname=cn,
                    table_type=table_type, label=label,
                ))
                logger.info(f"Found {table_type} table: OID={oid} [{cn}] label='{label}'")

        if not self._tables:
            logger.warning("No drillhole table blocks found in project file")
            # Don't return early — block models may still be discoverable

        # 2b. Find column blocks and resolve their parent table + data store
        table_oids = {t.oid for t in self._tables}
        table_by_oid = {t.oid: t for t in self._tables}

        for oid, cn in self._classnames.items():
            if cn not in _COLUMN_CLASSES:
                continue

            blob = self._blobs[oid]
            strings = _get_state_strings(blob)
            colname = _find_string_after(strings, 'colname')
            label = _find_string_after(strings, 'label') or ''

            if not colname:
                # CollarZColumnBlock may not have colname, use label
                if label and label not in ('_outputs', '_z_on_terrain'):
                    colname = label
                else:
                    continue

            # Skip internal columns
            if colname in ('id', '_is_manually_frozen', '_outputs',
                           '_z_on_terrain', 'dogleg_severity'):
                continue

            # Extract persistent refs to find parent table and ResultStore
            refs = _extract_persistent_refs(blob)
            parent_table_oid = None
            result_store_oid = None

            for ref_oid in refs:
                if ref_oid in table_oids:
                    parent_table_oid = ref_oid
                ref_cn = self._classnames.get(ref_oid, '')
                if ref_cn == 'ResultStore' and result_store_oid is None:
                    result_store_oid = ref_oid

            if parent_table_oid is None or result_store_oid is None:
                logger.debug(
                    f"  Skipping column '{colname}' (OID={oid}): "
                    f"parent={parent_table_oid}, store={result_store_oid}"
                )
                continue

            is_category = cn in ('CategoryColumnBlock', 'TextColumnBlock',
                                 'StringColumnBlock')

            col_info = _ColumnInfo(
                block_oid=oid, classname=cn, colname=colname,
                label=label, parent_table_oid=parent_table_oid,
                result_store_oid=result_store_oid, is_category=is_category,
            )
            table_by_oid[parent_table_oid].columns.append(col_info)
            logger.debug(
                f"  Column: '{colname}' [{cn}] -> table={parent_table_oid}, "
                f"store={result_store_oid}"
            )

        # Log summary
        for table in self._tables:
            col_names = [c.colname for c in table.columns]
            logger.info(
                f"  {table.table_type} table (OID={table.oid}): "
                f"{len(table.columns)} columns: {col_names}"
            )

    # ──────────────────────────────────────────────────────────────────────
    # Step 2b: Discover block model definitions
    # ──────────────────────────────────────────────────────────────────────

    def _discover_block_models(self):
        """
        Discover block model definitions in the ZODB.

        Leapfrog stores block model *definitions* (grid params, rotation,
        evaluation references) in the .aproj file, but the actual cell data
        (grades, coordinates) is computed lazily by Leapfrog's engine and
        NOT stored in the ResultStore blobs.  We extract the definition
        names so the UI can prompt the user to provide exported CSV files.
        """
        for oid, cn in self._classnames.items():
            # Match known root classes exactly
            if cn in _BLOCK_MODEL_ROOT_CLASSES:
                self._add_block_model_definition(oid, cn)
                continue
            # Heuristic substring match
            cn_lower = cn.lower()
            for pattern in _BLOCK_MODEL_CLASS_PATTERNS:
                if pattern.lower() in cn_lower:
                    self._add_block_model_definition(oid, cn)
                    break

        if self._block_models:
            logger.info(
                f"Discovered {len(self._block_models)} block model "
                f"definition(s) in project file"
            )
        else:
            logger.info("No block model definitions found in project file")

    def _add_block_model_definition(self, oid: int, classname: str):
        """Register a block model definition from the ZODB."""
        blob = self._blobs.get(oid, b'')
        strings = _get_state_strings(blob) if blob else []
        label = _find_string_after(strings, 'label') or classname

        # Avoid duplicates (same OID)
        if any(bm.parent_oid == oid for bm in self._block_models):
            return

        bm_info = _BlockModelInfo(
            parent_oid=oid,
            parent_classname=classname,
            label=label,
        )
        self._block_models.append(bm_info)
        logger.info(
            f"Block model definition: '{label}' "
            f"(OID={oid} [{classname}])"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Step 3: Extract data and build DataFrames
    # ──────────────────────────────────────────────────────────────────────

    def _build_dataframes(self) -> Dict[str, Any]:
        """Build pandas DataFrames from the discovered structure."""

        collars_df = None
        surveys_df = None
        assays_df = None
        all_tables: Dict[str, pd.DataFrame] = {}

        # Track all interval-type tables for fallback selection
        interval_candidates: List[Tuple[str, pd.DataFrame]] = []

        for table in self._tables:
            if not table.columns:
                logger.warning(f"Table '{table.label}' has no columns, skipping")
                continue

            df = self._extract_table_data(table)
            if df is None or df.empty:
                logger.warning(f"Table '{table.label}' produced no data")
                continue

            logger.info(
                f"Extracted {table.table_type} table '{table.label}': "
                f"{len(df)} rows, columns={list(df.columns)}"
            )
            all_tables[table.label] = df

            if table.table_type == 'collar' and collars_df is None:
                collars_df = _rename_columns(df, _COLLAR_RENAMES)
            elif table.table_type == 'survey' and surveys_df is None:
                surveys_df = _rename_columns(df, _SURVEY_RENAMES)
                # Leapfrog stores DIP as positive-downward (90 = vertical down)
                # GeoX convention: negative-downward (-90 = vertical down)
                if 'DIP' in surveys_df.columns:
                    surveys_df['DIP'] = -surveys_df['DIP'].astype(float)
                    logger.info(
                        "Converted survey DIP: Leapfrog positive-down -> "
                        "GeoX negative-down (negated)"
                    )
            elif table.table_type in ('interval', 'composite', 'economic_composite'):
                renamed = _rename_columns(df, _INTERVAL_RENAMES)
                interval_candidates.append((table.table_type, renamed))

        # Select best assay table: prefer interval > composite > economic_composite
        # But only if the table has grade columns (not just HOLEID)
        type_priority = {'interval': 0, 'composite': 1, 'economic_composite': 2}
        interval_candidates.sort(key=lambda x: type_priority.get(x[0], 99))

        for ttype, df in interval_candidates:
            grade_cols = [c for c in df.columns
                          if c not in ('HOLEID', 'FROM', 'TO')]
            if grade_cols and 'FROM' in df.columns and 'TO' in df.columns:
                assays_df = df
                logger.info(
                    f"Selected '{ttype}' as assay table: {len(df)} rows, "
                    f"grades={grade_cols}"
                )
                break

        # Fallback: use any interval table even without grades
        if assays_df is None and interval_candidates:
            assays_df = interval_candidates[0][1]

        # Build metadata
        metadata = {
            'source_format': 'Leapfrog Geo (.aproj)',
            'source_file': str(self.path),
            'total_objects': len(self._blobs),
            'tables_found': len(self._tables),
        }
        if collars_df is not None:
            metadata['collar_count'] = len(collars_df)
        if surveys_df is not None:
            metadata['survey_count'] = len(surveys_df)
        if assays_df is not None:
            metadata['assay_count'] = len(assays_df)
            grade_cols = [c for c in assays_df.columns
                          if c not in ('HOLEID', 'FROM', 'TO')]
            metadata['grade_columns'] = grade_cols

        # ── Block model definitions (names only — data requires CSV export) ──
        bm_names = [bm.label for bm in self._block_models]
        metadata['block_model_count'] = len(bm_names)
        metadata['block_model_names'] = bm_names

        return {
            'collars': collars_df,
            'surveys': surveys_df,
            'assays': assays_df,
            'tables': all_tables,
            'block_model_names': bm_names,
            'metadata': metadata,
        }

    def _extract_table_data(self, table: _TableInfo) -> Optional[pd.DataFrame]:
        """Extract all column data for a table and combine into a DataFrame."""

        columns_data: Dict[str, np.ndarray] = {}
        max_rows = 0
        holeid_names: Optional[np.ndarray] = None  # Name map for category resolution

        for col in table.columns:
            store_blob = self._blobs.get(col.result_store_oid)
            if not store_blob:
                logger.debug(f"  No blob for store OID={col.result_store_oid}")
                continue

            parquets = _extract_parquets(store_blob)
            if not parquets:
                logger.debug(f"  No parquet in store OID={col.result_store_oid}")
                continue

            if col.is_category:
                # Category columns: extract name map and index
                str_pqs = [pq for pq in parquets
                           if pq.iloc[:, 0].dtype == object]
                int_pqs = [pq for pq in parquets
                           if pq.iloc[:, 0].dtype in (np.int32, np.int64, np.uint8)]

                if str_pqs:
                    names = str_pqs[0].iloc[:, 0].values
                    if col.colname.lower() == 'holeid':
                        holeid_names = names

                if str_pqs and int_pqs:
                    names = str_pqs[0].iloc[:, 0].values
                    indices = int_pqs[0].iloc[:, 0].values.astype(int)
                    resolved = _resolve_category(names, indices)
                    columns_data[col.colname] = resolved
                    max_rows = max(max_rows, len(resolved))
                elif str_pqs and not int_pqs:
                    # Name map only — will be resolved after all columns
                    logger.debug(
                        f"  Category '{col.colname}': name map only "
                        f"({len(str_pqs[0])} names), index deferred"
                    )
                else:
                    logger.debug(f"  Category '{col.colname}': no usable data")
            else:
                # Numeric column: prefer float data, skip booleans
                best_pq = None
                for pq in parquets:
                    dt = pq.iloc[:, 0].dtype
                    if dt.kind == 'f':  # float — best
                        if best_pq is None or len(pq) > len(best_pq):
                            best_pq = pq
                    elif dt.kind in ('i', 'u') and best_pq is None:
                        best_pq = pq

                if best_pq is not None:
                    values = best_pq.iloc[:, 0].values
                    # Skip boolean-like data for interval columns
                    if values.dtype == np.bool_:
                        logger.debug(f"  Skipping boolean '{col.colname}'")
                        continue
                    columns_data[col.colname] = values
                    max_rows = max(max_rows, len(values))

        # ── Holeid inference ──
        # If holeid column has names but no index, infer the mapping.
        if 'holeid' not in columns_data and holeid_names is not None and max_rows > 0:
            n_names = len(holeid_names)
            if n_names == max_rows:
                # One name per row (collar table pattern)
                columns_data['holeid'] = holeid_names.copy()
                logger.debug(f"  Holeid: 1:1 name mapping ({n_names} rows)")
            elif max_rows > n_names and max_rows % n_names == 0:
                # Multiple rows per name (survey table pattern)
                repeats = max_rows // n_names
                columns_data['holeid'] = np.repeat(holeid_names, repeats)
                logger.debug(
                    f"  Holeid: {repeats}:1 repeat mapping "
                    f"({n_names} names x {repeats} = {max_rows} rows)"
                )

        # ── Consolidated table fallback ──
        # For interval/composite tables, look for a consolidated ResultStore
        # that has a multi-column parquet with 'collar_id', 'from', 'to'.
        if table.table_type in ('interval', 'composite', 'economic_composite'):
            needs_consolidation = (
                'holeid' not in columns_data
                or 'from' not in columns_data
                or 'to' not in columns_data
            )
            if needs_consolidation:
                consolidated = self._find_consolidated_parquet(table)
                if consolidated is not None:
                    logger.info(
                        f"  Using consolidated table: {consolidated.shape} "
                        f"cols={list(consolidated.columns)}"
                    )
                    # Extract collar_id -> holeid mapping
                    if 'collar_id' in consolidated.columns and holeid_names is not None:
                        collar_ids = consolidated['collar_id'].values.astype(int)
                        resolved = _resolve_category(holeid_names, collar_ids)
                        columns_data['holeid'] = resolved
                        max_rows = max(max_rows, len(resolved))

                    # Extract from/to
                    for key in ('from', 'to'):
                        if key in consolidated.columns:
                            columns_data[key] = consolidated[key].values
                            max_rows = max(max_rows, len(consolidated))

        if not columns_data:
            return None

        # Pad shorter columns with NaN/None to match max_rows
        result = {}
        for name, values in columns_data.items():
            if len(values) < max_rows:
                if values.dtype == object:
                    padded = np.empty(max_rows, dtype=object)
                    padded[:] = None
                    padded[:len(values)] = values
                else:
                    padded = np.full(max_rows, np.nan)
                    padded[:len(values)] = values
                result[name] = padded
            else:
                result[name] = values[:max_rows]

        return pd.DataFrame(result)

    def _find_consolidated_parquet(
        self, table: _TableInfo
    ) -> Optional[pd.DataFrame]:
        """
        Find a consolidated ResultStore parquet for a table.

        Searches ALL ResultStore objects for a multi-column parquet that
        contains 'collar_id' plus 'from'/'to'. These are table-level data
        stores that Leapfrog creates for interval/composite tables.
        """
        # Scan ResultStore objects near the table's columns
        # (consolidated store is typically after the column blocks)
        min_oid = min(c.result_store_oid for c in table.columns) if table.columns else 0
        max_oid = max(c.result_store_oid for c in table.columns) if table.columns else 0
        # Also check OIDs well beyond the column stores
        scan_range = range(max(0, min_oid - 50), max_oid + 200)

        for oid in scan_range:
            cn = self._classnames.get(oid, '')
            if cn != 'ResultStore':
                continue
            blob = self._blobs.get(oid)
            if not blob:
                continue

            parquets = _extract_parquets(blob)
            for pq in parquets:
                if pq.shape[1] >= 3 and 'collar_id' in pq.columns:
                    return pq

        return None


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _resolve_category(names: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Resolve category indices to names using 1-based (then 0-based) indexing."""
    resolved = np.empty(len(indices), dtype=object)
    for i, idx_val in enumerate(indices):
        if 1 <= idx_val <= len(names):
            resolved[i] = names[idx_val - 1]
        elif 0 <= idx_val < len(names):
            resolved[i] = names[idx_val]
        else:
            resolved[i] = None
    return resolved


def _rename_columns(df: pd.DataFrame, rename_map: Dict[str, str]) -> pd.DataFrame:
    """Rename columns using case-insensitive matching."""
    renames = {}
    for col in df.columns:
        key = col.lower().strip().replace(' ', '_')
        if key in rename_map:
            renames[col] = rename_map[key]
        # Also try without underscores
        key2 = key.replace('_', '')
        if key2 in rename_map and col not in renames:
            renames[col] = rename_map[key2]
    if renames:
        df = df.rename(columns=renames)
    return df


# ════════════════════════════════════════════════════════════════════════════════
# Convenience function
# ════════════════════════════════════════════════════════════════════════════════

def read_leapfrog_project(path: str | Path) -> Dict[str, Any]:
    """
    Read a Leapfrog Geo .aproj project file and extract drillhole + block model data.

    Args:
        path: Path to the .aproj file

    Returns:
        dict with keys: 'collars', 'surveys', 'assays', 'tables',
        'block_models', 'metadata'
    """
    importer = LeapfrogImporter(path)
    return importer.read()
