"""
Data Migration System

Provides data migration capabilities for version upgrades and schema changes.
Supports version tracking, migration scripts, and rollback capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging
import json

from .datamodel import DrillholeDatabase

logger = logging.getLogger(__name__)


class MigrationStatus(Enum):
    """Migration status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class Migration:
    """A data migration."""
    migration_id: str
    version_from: str
    version_to: str
    description: str
    migration_func: Callable[[DrillholeDatabase], bool]
    rollback_func: Optional[Callable[[DrillholeDatabase], bool]] = None
    dependencies: List[str] = field(default_factory=list)
    created_date: datetime = field(default_factory=datetime.now)


@dataclass
class MigrationRecord:
    """Record of a migration execution."""
    record_id: str
    migration_id: str
    timestamp: datetime
    status: MigrationStatus
    version_from: str
    version_to: str
    error_message: str = ""
    rollback_timestamp: Optional[datetime] = None


class DataMigrationManager:
    """
    Data migration manager.
    
    Manages database schema migrations and version upgrades.
    """
    
    def __init__(self):
        self.migrations: List[Migration] = []
        self.migration_history: List[MigrationRecord] = []
        self.current_version: str = "1.0.0"
        self._record_counter = 0
        self._register_default_migrations()
        
        logger.info("DataMigrationManager initialized")
    
    def _generate_record_id(self) -> str:
        """Generate a unique record ID."""
        self._record_counter += 1
        return f"MIG-{self._record_counter:08d}"
    
    def _register_default_migrations(self):
        """Register default migrations."""
        # Example: Migration from v1.0.0 to v1.1.0
        def migrate_1_0_to_1_1(db: DrillholeDatabase) -> bool:
            """Add new fields to existing data."""
            try:
                # Example: Add default values to collars
                for collar in db.collars:
                    if collar.azimuth is None:
                        collar.azimuth = 0.0
                    if collar.dip is None:
                        collar.dip = -90.0
                return True
            except Exception as e:
                logger.error(f"Migration 1.0->1.1 failed: {e}")
                return False
        
        migration = Migration(
            migration_id="mig_1_0_to_1_1",
            version_from="1.0.0",
            version_to="1.1.0",
            description="Add default azimuth and dip values",
            migration_func=migrate_1_0_to_1_1,
        )
        self.migrations.append(migration)
        
        logger.info(f"Registered {len(self.migrations)} default migrations")
    
    def register_migration(self, migration: Migration) -> None:
        """Register a migration."""
        self.migrations.append(migration)
        logger.info(f"Registered migration: {migration.migration_id}")
    
    def get_pending_migrations(self, current_version: str) -> List[Migration]:
        """
        Get migrations that need to be applied.
        
        Args:
            current_version: Current database version
        
        Returns:
            List of pending Migration objects
        """
        # Simple version comparison (in production, use proper version parsing)
        pending = []
        for migration in sorted(self.migrations, key=lambda m: m.version_to):
            if self._version_compare(current_version, migration.version_from) < 0:
                continue  # Migration is for a future version
            if self._version_compare(current_version, migration.version_to) >= 0:
                continue  # Migration already applied
            
            # Check if already applied
            if not self._is_migration_applied(migration.migration_id):
                pending.append(migration)
        
        return pending
    
    def _version_compare(self, v1: str, v2: str) -> int:
        """
        Compare two version strings.
        
        Returns:
            -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
        """
        try:
            parts1 = [int(x) for x in v1.split('.')]
            parts2 = [int(x) for x in v2.split('.')]
            
            # Pad to same length
            max_len = max(len(parts1), len(parts2))
            parts1.extend([0] * (max_len - len(parts1)))
            parts2.extend([0] * (max_len - len(parts2)))
            
            for p1, p2 in zip(parts1, parts2):
                if p1 < p2:
                    return -1
                elif p1 > p2:
                    return 1
            
            return 0
        except Exception:
            # Fallback: string comparison
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            return 0
    
    def _is_migration_applied(self, migration_id: str) -> bool:
        """Check if a migration has been applied."""
        return any(
            r.migration_id == migration_id and r.status == MigrationStatus.COMPLETED
            for r in self.migration_history
        )
    
    def apply_migration(
        self,
        database: DrillholeDatabase,
        migration: Migration,
    ) -> MigrationRecord:
        """
        Apply a migration to a database.
        
        Args:
            database: DrillholeDatabase to migrate
            migration: Migration to apply
        
        Returns:
            MigrationRecord with execution results
        """
        record_id = self._generate_record_id()
        
        record = MigrationRecord(
            record_id=record_id,
            migration_id=migration.migration_id,
            timestamp=datetime.now(),
            status=MigrationStatus.RUNNING,
            version_from=migration.version_from,
            version_to=migration.version_to,
        )
        
        try:
            logger.info(f"Applying migration: {migration.migration_id} ({migration.version_from} -> {migration.version_to})")
            
            # Check dependencies
            for dep_id in migration.dependencies:
                if not self._is_migration_applied(dep_id):
                    raise Exception(f"Dependency migration not applied: {dep_id}")
            
            # Apply migration
            success = migration.migration_func(database)
            
            if success:
                record.status = MigrationStatus.COMPLETED
                self.current_version = migration.version_to
                logger.info(f"Migration {migration.migration_id} completed successfully")
            else:
                record.status = MigrationStatus.FAILED
                record.error_message = "Migration function returned False"
                logger.error(f"Migration {migration.migration_id} failed")
            
        except Exception as e:
            record.status = MigrationStatus.FAILED
            record.error_message = str(e)
            logger.error(f"Migration {migration.migration_id} failed: {e}", exc_info=True)
        
        self.migration_history.append(record)
        return record
    
    def rollback_migration(
        self,
        database: DrillholeDatabase,
        migration_id: str,
    ) -> bool:
        """
        Rollback a migration.
        
        Args:
            database: DrillholeDatabase to rollback
            migration_id: ID of migration to rollback
        
        Returns:
            True if successful, False otherwise
        """
        # Find migration
        migration = None
        for m in self.migrations:
            if m.migration_id == migration_id:
                migration = m
                break
        
        if not migration:
            logger.error(f"Migration not found: {migration_id}")
            return False
        
        if not migration.rollback_func:
            logger.error(f"Migration {migration_id} has no rollback function")
            return False
        
        # Find migration record
        record = None
        for r in reversed(self.migration_history):
            if r.migration_id == migration_id and r.status == MigrationStatus.COMPLETED:
                record = r
                break
        
        if not record:
            logger.error(f"Migration record not found: {migration_id}")
            return False
        
        try:
            logger.info(f"Rolling back migration: {migration_id}")
            
            success = migration.rollback_func(database)
            
            if success:
                record.status = MigrationStatus.ROLLED_BACK
                record.rollback_timestamp = datetime.now()
                self.current_version = migration.version_from
                logger.info(f"Migration {migration_id} rolled back successfully")
            else:
                logger.error(f"Migration {migration_id} rollback failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}", exc_info=True)
            return False
    
    def migrate_database(
        self,
        database: DrillholeDatabase,
        target_version: Optional[str] = None,
    ) -> List[MigrationRecord]:
        """
        Migrate database to target version (or latest).
        
        Args:
            database: DrillholeDatabase to migrate
            target_version: Target version (latest if None)
        
        Returns:
            List of MigrationRecord objects
        """
        if target_version is None:
            # Find latest version
            if self.migrations:
                target_version = max(m.version_to for m in self.migrations)
            else:
                target_version = self.current_version
        
        pending = self.get_pending_migrations(self.current_version)
        results = []
        
        logger.info(f"Migrating database from {self.current_version} to {target_version}")
        
        for migration in pending:
            if self._version_compare(migration.version_to, target_version) > 0:
                break  # Stop if we've reached target version
            
            record = self.apply_migration(database, migration)
            results.append(record)
            
            if record.status == MigrationStatus.FAILED:
                logger.error(f"Migration failed, stopping: {migration.migration_id}")
                break
        
        return results


# Global migration manager instance
_migration_manager: Optional[DataMigrationManager] = None


def get_migration_manager() -> DataMigrationManager:
    """Get the global migration manager instance."""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = DataMigrationManager()
    return _migration_manager

