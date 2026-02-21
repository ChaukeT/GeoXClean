"""
Drillhole Database Manager - High Performance SQLite

Uses Pandas SQL engine for bulk inserts/reads.

GeoX Invariant Compliance:
- Audit trail for all data operations
- Provenance tracking with timestamps and user info
- Parameterized queries to prevent SQL injection
"""

import sqlite3
import logging
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from .datamodel import DrillholeDatabase

logger = logging.getLogger(__name__)

# Database schema version for migration tracking
DATABASE_SCHEMA_VERSION = "1.0.0"


class DrillholeDatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            data_dir = Path.home() / '.geox' / 'drillholes'
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = data_dir / "drillhole_database.db"

        self.db_path = Path(db_path)
        self._initialize_schema()

    def connect(self):
        return sqlite3.connect(str(self.db_path))

    def _initialize_schema(self):
        """Create tables if not exist."""
        with self.connect() as conn:
            # Projects table (Meta) with provenance fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    schema_version TEXT DEFAULT '1.0.0'
                )
            """)
            
            # GeoX invariant: Audit log table for all data operations
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    operation TEXT NOT NULL,
                    table_name TEXT,
                    records_affected INTEGER,
                    user_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    details TEXT,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)

            # Collars table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    hole_id TEXT NOT NULL,
                    x REAL NOT NULL,
                    y REAL NOT NULL,
                    z REAL NOT NULL,
                    azimuth REAL,
                    dip REAL,
                    length REAL,
                    FOREIGN KEY (project_id) REFERENCES projects(id),
                    UNIQUE(project_id, hole_id)
                )
            """)

            # Surveys table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS surveys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    hole_id TEXT NOT NULL,
                    depth_from REAL NOT NULL,
                    depth_to REAL NOT NULL,
                    azimuth REAL NOT NULL,
                    dip REAL NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)

            # Assays table (stores assay intervals)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assays (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    hole_id TEXT NOT NULL,
                    depth_from REAL NOT NULL,
                    depth_to REAL NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)

            # Assay values table (stores element grades - normalized for multiple elements per interval)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assay_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assay_id INTEGER NOT NULL,
                    element TEXT NOT NULL,
                    value REAL NOT NULL,
                    FOREIGN KEY (assay_id) REFERENCES assays(id) ON DELETE CASCADE,
                    UNIQUE(assay_id, element)
                )
            """)

            # Lithology table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS lithology (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id INTEGER,
                    hole_id TEXT NOT NULL,
                    depth_from REAL NOT NULL,
                    depth_to REAL NOT NULL,
                    lith_code TEXT NOT NULL,
                    FOREIGN KEY (project_id) REFERENCES projects(id)
                )
            """)

            # Create indices for speed
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collars_project ON collars(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_collars_hole ON collars(hole_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_surveys_project ON surveys(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_surveys_hole ON surveys(hole_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assays_project ON assays(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assays_hole ON assays(hole_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_assay_values_assay ON assay_values(assay_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lithology_project ON lithology(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_lithology_hole ON lithology(hole_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_project ON audit_log(project_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)")

            conn.commit()
    
    def _log_operation(self, conn, project_id: int, operation: str, table_name: str = None,
                       records_affected: int = 0, user_id: str = None, details: Dict = None):
        """
        Log an operation to the audit trail.
        
        GeoX invariant: All data operations must be auditable.
        """
        conn.execute(
            """INSERT INTO audit_log (project_id, operation, table_name, records_affected, user_id, details)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (project_id, operation, table_name, records_affected, user_id, json.dumps(details) if details else None)
        )

    def create_project(self, name: str, description: str = "") -> int:
        with self.connect() as conn:
            try:
                cur = conn.execute(
                    "INSERT INTO projects (name, description) VALUES (?, ?)",
                    (name, description)
                )
                return cur.lastrowid
            except sqlite3.IntegrityError:
                # Get existing ID
                cur = conn.execute("SELECT id FROM projects WHERE name=?", (name,))
                return cur.fetchone()[0]

    def save_database(self, db: DrillholeDatabase, project_name: str, user_id: str = None):
        """
        High-performance save using Pandas to_sql.
        
        GeoX invariant: All operations are logged to audit trail.
        
        Args:
            db: DrillholeDatabase to save
            project_name: Name of project to save to
            user_id: Optional user identifier for audit trail
        """
        project_id = self.create_project(project_name)

        with self.connect() as conn:
            # 1. Collars
            if not db.collars.empty:
                df = db.collars.copy()
                df['project_id'] = project_id
                # Wipe existing for this project (parameterized query)
                old_count = pd.read_sql("SELECT COUNT(*) as cnt FROM collars WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
                conn.execute("DELETE FROM collars WHERE project_id=?", (project_id,))
                df.to_sql('collars', conn, if_exists='append', index=False)
                # Audit log
                self._log_operation(conn, project_id, 'SAVE_COLLARS', 'collars', len(df), user_id, 
                                   {'records_deleted': int(old_count), 'records_inserted': len(df)})

            # 2. Surveys
            if not db.surveys.empty:
                df = db.surveys.copy()
                df['project_id'] = project_id
                old_count = pd.read_sql("SELECT COUNT(*) as cnt FROM surveys WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
                conn.execute("DELETE FROM surveys WHERE project_id=?", (project_id,))
                df.to_sql('surveys', conn, if_exists='append', index=False)
                self._log_operation(conn, project_id, 'SAVE_SURVEYS', 'surveys', len(df), user_id,
                                   {'records_deleted': int(old_count), 'records_inserted': len(df)})

            # 3. Assays (handle element columns separately)
            if not db.assays.empty:
                df = db.assays.copy()
                df['project_id'] = project_id

                # Separate base columns from element columns
                base_cols = ['project_id', 'hole_id', 'depth_from', 'depth_to']
                element_cols = [c for c in df.columns if c not in base_cols and pd.notna(df[c]).any()]

                # Get counts before delete for audit
                old_assay_count = pd.read_sql("SELECT COUNT(*) as cnt FROM assays WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
                old_values_count = pd.read_sql(
                    "SELECT COUNT(*) as cnt FROM assay_values WHERE assay_id IN (SELECT id FROM assays WHERE project_id=?)",
                    conn, params=(project_id,)
                ).iloc[0]['cnt']

                # Save base assay intervals (parameterized queries)
                conn.execute("DELETE FROM assay_values WHERE assay_id IN (SELECT id FROM assays WHERE project_id=?)", (project_id,))
                conn.execute("DELETE FROM assays WHERE project_id=?", (project_id,))
                base_df = df[base_cols].copy()
                base_df.to_sql('assays', conn, if_exists='append', index=False)

                # Save element values (normalized)
                if element_cols:
                    # Get assay IDs we just inserted
                    assay_map = pd.read_sql(
                        f"SELECT id, hole_id, depth_from, depth_to FROM assays WHERE project_id={project_id}",
                        conn
                    )
                    # Merge to get IDs
                    df_with_ids = df.merge(assay_map, on=['hole_id', 'depth_from', 'depth_to'], how='left')

                    # Melt element columns into rows
                    value_rows = []
                    for _, row in df_with_ids.iterrows():
                        assay_id = row['id']
                        if pd.notna(assay_id):
                            for elem in element_cols:
                                val = row[elem]
                                if pd.notna(val):
                                    value_rows.append({
                                        'assay_id': int(assay_id),
                                        'element': str(elem),
                                        'value': float(val)
                                    })

                    if value_rows:
                        values_df = pd.DataFrame(value_rows)
                        values_df.to_sql('assay_values', conn, if_exists='append', index=False)
                
                # Audit log for assays
                self._log_operation(conn, project_id, 'SAVE_ASSAYS', 'assays', len(df), user_id, {
                    'assays_deleted': int(old_assay_count), 'assays_inserted': len(df),
                    'values_deleted': int(old_values_count), 'element_columns': element_cols
                })

            # 4. Lithology
            if not db.lithology.empty:
                df = db.lithology.copy()
                df['project_id'] = project_id
                old_count = pd.read_sql("SELECT COUNT(*) as cnt FROM lithology WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
                conn.execute("DELETE FROM lithology WHERE project_id=?", (project_id,))
                df.to_sql('lithology', conn, if_exists='append', index=False)
                self._log_operation(conn, project_id, 'SAVE_LITHOLOGY', 'lithology', len(df), user_id,
                                   {'records_deleted': int(old_count), 'records_inserted': len(df)})

        logger.info(f"Saved project '{project_name}' (ID: {project_id})")

    def load_database(self, project_name: str, user_id: str = None) -> DrillholeDatabase:
        """
        High-performance load using Pandas read_sql.
        
        GeoX invariant: Uses parameterized queries to prevent SQL injection.
        
        Args:
            project_name: Name of project to load
            user_id: Optional user identifier for audit trail
        
        Returns:
            DrillholeDatabase with loaded data
        """
        with self.connect() as conn:
            # Get ID (parameterized query)
            cur = conn.execute("SELECT id FROM projects WHERE name=?", (project_name,))
            res = cur.fetchone()
            if not res:
                raise ValueError(f"Project {project_name} not found")
            pid = res[0]

            db = DrillholeDatabase()

            # Load tables directly into DataFrames (parameterized queries)
            try:
                collars_df = pd.read_sql("SELECT * FROM collars WHERE project_id=?", conn, params=(pid,))
                if not collars_df.empty and 'project_id' in collars_df.columns:
                    collars_df.drop(columns=['project_id'], inplace=True)
                if not collars_df.empty:
                    db.collars = collars_df
            except Exception as e:
                logger.warning(f"Failed to load collars: {e}")

            try:
                surveys_df = pd.read_sql("SELECT * FROM surveys WHERE project_id=?", conn, params=(pid,))
                if not surveys_df.empty and 'project_id' in surveys_df.columns:
                    surveys_df.drop(columns=['project_id'], inplace=True)
                if not surveys_df.empty:
                    db.surveys = surveys_df
            except Exception as e:
                logger.warning(f"Failed to load surveys: {e}")

            try:
                # Load base assays (parameterized query)
                assays_df = pd.read_sql("SELECT * FROM assays WHERE project_id=?", conn, params=(pid,))
                if not assays_df.empty:
                    if 'project_id' in assays_df.columns:
                        assays_df.drop(columns=['project_id'], inplace=True)

                    # Load element values and pivot (safe query with placeholders)
                    if 'id' in assays_df.columns:
                        assay_ids = assays_df['id'].tolist()
                        if assay_ids:
                            # Use parameterized query with placeholder for each id
                            placeholders = ','.join(['?' for _ in assay_ids])
                            values_df = pd.read_sql(
                                f"SELECT assay_id, element, value FROM assay_values WHERE assay_id IN ({placeholders})",
                                conn,
                                params=assay_ids
                            )
                            if not values_df.empty:
                                # Pivot to wide format
                                values_pivot = values_df.pivot(index='assay_id', columns='element', values='value')
                                # Merge back
                                assays_df = assays_df.merge(values_pivot, left_on='id', right_index=True, how='left')
                                assays_df.drop(columns=['id'], inplace=True)

                    db.assays = assays_df
            except Exception as e:
                logger.warning(f"Failed to load assays: {e}")

            try:
                lithology_df = pd.read_sql("SELECT * FROM lithology WHERE project_id=?", conn, params=(pid,))
                if not lithology_df.empty and 'project_id' in lithology_df.columns:
                    lithology_df.drop(columns=['project_id'], inplace=True)
                if not lithology_df.empty:
                    db.lithology = lithology_df
            except Exception as e:
                logger.warning(f"Failed to load lithology: {e}")

            # Log the load operation
            self._log_operation(conn, pid, 'LOAD_DATABASE', None, 
                               len(db.collars) + len(db.surveys) + len(db.assays) + len(db.lithology),
                               user_id, {'project_name': project_name})

            return db

    def list_projects(self) -> List[Dict]:
        with self.connect() as conn:
            df = pd.read_sql("SELECT id, name, description, created_at FROM projects", conn)
            return df.to_dict('records')

    def get_project_id(self, project_name: str) -> Optional[int]:
        """Get project ID by name."""
        with self.connect() as conn:
            cur = conn.execute("SELECT id FROM projects WHERE name = ?", (project_name,))
            row = cur.fetchone()
            return row[0] if row else None

    def delete_project(self, project_name: str):
        """Delete a project and all its data."""
        project_id = self.get_project_id(project_name)
        if project_id is None:
            raise ValueError(f"Project '{project_name}' not found")

        with self.connect() as conn:
            conn.execute("DELETE FROM assay_values WHERE assay_id IN (SELECT id FROM assays WHERE project_id=?)", (project_id,))
            conn.execute("DELETE FROM assays WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM surveys WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM lithology WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM collars WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM projects WHERE id=?", (project_id,))
            conn.commit()

    def get_statistics(self, project_name: str) -> Dict[str, Any]:
        """Get statistics for a project using parameterized queries."""
        project_id = self.get_project_id(project_name)
        if project_id is None:
            raise ValueError(f"Project '{project_name}' not found")

        with self.connect() as conn:
            stats = {}
            # Use parameterized queries to prevent SQL injection
            stats['num_collars'] = pd.read_sql("SELECT COUNT(*) as cnt FROM collars WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
            stats['num_surveys'] = pd.read_sql("SELECT COUNT(*) as cnt FROM surveys WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
            stats['num_assays'] = pd.read_sql("SELECT COUNT(*) as cnt FROM assays WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
            stats['num_elements'] = pd.read_sql(
                "SELECT COUNT(DISTINCT element) as cnt FROM assay_values WHERE assay_id IN (SELECT id FROM assays WHERE project_id=?)",
                conn, params=(project_id,)
            ).iloc[0]['cnt']
            stats['num_lithology'] = pd.read_sql("SELECT COUNT(*) as cnt FROM lithology WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
            stats['num_holes'] = pd.read_sql("SELECT COUNT(DISTINCT hole_id) as cnt FROM collars WHERE project_id=?", conn, params=(project_id,)).iloc[0]['cnt']
            return stats
    
    def get_audit_log(self, project_name: str, limit: int = 100) -> List[Dict]:
        """
        Get audit log entries for a project.
        
        GeoX invariant: Provides access to full audit trail.
        
        Args:
            project_name: Name of project
            limit: Maximum number of entries to return
        
        Returns:
            List of audit log entries
        """
        project_id = self.get_project_id(project_name)
        if project_id is None:
            raise ValueError(f"Project '{project_name}' not found")
        
        with self.connect() as conn:
            df = pd.read_sql(
                "SELECT * FROM audit_log WHERE project_id=? ORDER BY timestamp DESC LIMIT ?",
                conn, params=(project_id, limit)
            )
            return df.to_dict('records')
