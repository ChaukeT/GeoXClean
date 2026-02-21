"""
Data Backup and Recovery System

Provides comprehensive backup and recovery capabilities for drillhole data.
Supports automated backups, versioning, and point-in-time recovery.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import logging
import json
import shutil
import zipfile

from .datamodel import DrillholeDatabase

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Types of backups."""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    MANUAL = "manual"


class BackupStatus(Enum):
    """Backup status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackupRecord:
    """Record of a backup operation."""
    backup_id: str
    backup_type: BackupType
    timestamp: datetime
    status: BackupStatus
    file_path: Path
    size_bytes: int = 0
    database_name: str = ""
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BackupRecoveryManager:
    """
    Backup and recovery manager.
    
    Provides automated backup, versioning, and recovery capabilities.
    """
    
    def __init__(self, backup_directory: Optional[Path] = None):
        if backup_directory is None:
            # Default backup directory
            backup_directory = Path.home() / ".geox" / "backups"
        
        self.backup_directory = Path(backup_directory)
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        
        self.backups: List[BackupRecord] = []
        self._backup_counter = 0
        self._load_backup_index()
        
        logger.info(f"BackupRecoveryManager initialized with backup directory: {self.backup_directory}")
    
    def _generate_backup_id(self) -> str:
        """Generate a unique backup ID."""
        self._backup_counter += 1
        return f"BACKUP-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{self._backup_counter:04d}"
    
    def create_backup(
        self,
        database: DrillholeDatabase,
        backup_type: BackupType = BackupType.FULL,
        description: str = "",
        database_name: str = "default",
    ) -> BackupRecord:
        """
        Create a backup of a database.
        
        Args:
            database: DrillholeDatabase to backup
            backup_type: Type of backup
            description: Optional description
            database_name: Name of the database
        
        Returns:
            BackupRecord with backup information
        """
        backup_id = self._generate_backup_id()
        timestamp = datetime.now()
        
        backup_record = BackupRecord(
            backup_id=backup_id,
            backup_type=backup_type,
            timestamp=timestamp,
            status=BackupStatus.IN_PROGRESS,
            file_path=self.backup_directory / f"{backup_id}.zip",
            database_name=database_name,
            description=description,
        )
        
        try:
            # Create backup file
            with zipfile.ZipFile(backup_record.file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Serialize database to JSON
                db_data = {
                    "database_name": database_name,
                    "backup_timestamp": timestamp.isoformat(),
                    "collars": [
                        {
                            "hole_id": c.hole_id,
                            "x": c.x,
                            "y": c.y,
                            "z": c.z,
                            "azimuth": c.azimuth,
                            "dip": c.dip,
                            "length": c.length,
                        }
                        for c in database.collars
                    ],
                    "surveys": [
                        {
                            "hole_id": s.hole_id,
                            "depth_from": s.depth_from,
                            "depth_to": s.depth_to,
                            "azimuth": s.azimuth,
                            "dip": s.dip,
                        }
                        for s in database.surveys
                    ],
                    "assays": [
                        {
                            "hole_id": a.hole_id,
                            "depth_from": a.depth_from,
                            "depth_to": a.depth_to,
                            "values": dict(a.values),
                        }
                        for a in database.assays
                    ],
                    "lithology": [
                        {
                            "hole_id": l.hole_id,
                            "depth_from": l.depth_from,
                            "depth_to": l.depth_to,
                            "lith_code": getattr(l, "lith_code", ""),
                        }
                        for l in database.lithology
                    ],
                }
                
                # Write database data
                zipf.writestr("database.json", json.dumps(db_data, indent=2))
                
                # Write metadata
                metadata = {
                    "backup_id": backup_id,
                    "backup_type": backup_type.value,
                    "timestamp": timestamp.isoformat(),
                    "database_name": database_name,
                    "description": description,
                    "record_count": {
                        "collars": len(database.collars),
                        "surveys": len(database.surveys),
                        "assays": len(database.assays),
                        "lithology": len(database.lithology),
                    },
                }
                zipf.writestr("metadata.json", json.dumps(metadata, indent=2))
            
            # Update backup record
            backup_record.status = BackupStatus.COMPLETED
            backup_record.size_bytes = backup_record.file_path.stat().st_size
            backup_record.metadata = metadata
            
            self.backups.append(backup_record)
            self._save_backup_index()
            
            logger.info(f"Created backup: {backup_id} ({backup_record.size_bytes / 1024:.1f} KB)")
            
        except Exception as e:
            backup_record.status = BackupStatus.FAILED
            backup_record.metadata = {"error": str(e)}
            logger.error(f"Backup failed: {e}", exc_info=True)
            
            # Clean up failed backup file
            if backup_record.file_path.exists():
                try:
                    backup_record.file_path.unlink()
                except Exception:
                    pass
        
        return backup_record
    
    def restore_backup(
        self,
        backup_id: str,
    ) -> Optional[DrillholeDatabase]:
        """
        Restore a database from backup.
        
        Args:
            backup_id: ID of backup to restore
        
        Returns:
            Restored DrillholeDatabase or None if failed
        """
        # Find backup record
        backup_record = None
        for backup in self.backups:
            if backup.backup_id == backup_id:
                backup_record = backup
                break
        
        if not backup_record:
            logger.error(f"Backup not found: {backup_id}")
            return None
        
        if not backup_record.file_path.exists():
            logger.error(f"Backup file not found: {backup_record.file_path}")
            return None
        
        try:
            # Extract and restore
            with zipfile.ZipFile(backup_record.file_path, 'r') as zipf:
                # Read database data
                db_data_str = zipf.read("database.json").decode('utf-8')
                db_data = json.loads(db_data_str)
            
            # Reconstruct database
            database = DrillholeDatabase()
            
            from .datamodel import Collar, SurveyInterval, AssayInterval, LithologyInterval
            
            # Restore collars
            for c_data in db_data.get("collars", []):
                database.collars.append(Collar(
                    hole_id=c_data["hole_id"],
                    x=c_data["x"],
                    y=c_data["y"],
                    z=c_data["z"],
                    azimuth=c_data.get("azimuth"),
                    dip=c_data.get("dip"),
                    length=c_data.get("length"),
                ))
            
            # Restore surveys
            for s_data in db_data.get("surveys", []):
                database.surveys.append(SurveyInterval(
                    hole_id=s_data["hole_id"],
                    depth_from=s_data["depth_from"],
                    depth_to=s_data["depth_to"],
                    azimuth=s_data["azimuth"],
                    dip=s_data["dip"],
                ))
            
            # Restore assays
            for a_data in db_data.get("assays", []):
                database.assays.append(AssayInterval(
                    hole_id=a_data["hole_id"],
                    depth_from=a_data["depth_from"],
                    depth_to=a_data["depth_to"],
                    values=a_data.get("values", {}),
                ))
            
            # Restore lithology
            for l_data in db_data.get("lithology", []):
                database.lithology.append(LithologyInterval(
                    hole_id=l_data["hole_id"],
                    depth_from=l_data["depth_from"],
                    depth_to=l_data["depth_to"],
                    lith_code=l_data.get("lith_code", ""),
                ))
            
            logger.info(f"Restored backup: {backup_id} ({len(database.collars)} collars, {len(database.assays)} assays)")
            return database
            
        except Exception as e:
            logger.error(f"Restore failed: {e}", exc_info=True)
            return None
    
    def list_backups(
        self,
        database_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[BackupRecord]:
        """
        List available backups.
        
        Args:
            database_name: Filter by database name
            start_date: Filter by start date
            end_date: Filter by end date
        
        Returns:
            List of BackupRecord objects
        """
        filtered = self.backups
        
        if database_name:
            filtered = [b for b in filtered if b.database_name == database_name]
        
        if start_date:
            filtered = [b for b in filtered if b.timestamp >= start_date]
        
        if end_date:
            filtered = [b for b in filtered if b.timestamp <= end_date]
        
        return sorted(filtered, key=lambda b: b.timestamp, reverse=True)
    
    def delete_backup(self, backup_id: str) -> bool:
        """
        Delete a backup.
        
        Args:
            backup_id: ID of backup to delete
        
        Returns:
            True if successful, False otherwise
        """
        for i, backup in enumerate(self.backups):
            if backup.backup_id == backup_id:
                try:
                    if backup.file_path.exists():
                        backup.file_path.unlink()
                    self.backups.pop(i)
                    self._save_backup_index()
                    logger.info(f"Deleted backup: {backup_id}")
                    return True
                except Exception as e:
                    logger.error(f"Error deleting backup: {e}")
                    return False
        
        return False
    
    def _save_backup_index(self):
        """Save backup index to file."""
        index_path = self.backup_directory / "backup_index.json"
        try:
            index_data = {
                "backups": [
                    {
                        "backup_id": b.backup_id,
                        "backup_type": b.backup_type.value,
                        "timestamp": b.timestamp.isoformat(),
                        "status": b.status.value,
                        "file_path": str(b.file_path.relative_to(self.backup_directory)),
                        "size_bytes": b.size_bytes,
                        "database_name": b.database_name,
                        "description": b.description,
                    }
                    for b in self.backups
                ],
            }
            
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error saving backup index: {e}")
    
    def _load_backup_index(self):
        """Load backup index from file."""
        index_path = self.backup_directory / "backup_index.json"
        if not index_path.exists():
            return
        
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            self.backups = []
            for b_data in index_data.get("backups", []):
                backup = BackupRecord(
                    backup_id=b_data["backup_id"],
                    backup_type=BackupType(b_data["backup_type"]),
                    timestamp=datetime.fromisoformat(b_data["timestamp"]),
                    status=BackupStatus(b_data["status"]),
                    file_path=self.backup_directory / b_data["file_path"],
                    size_bytes=b_data.get("size_bytes", 0),
                    database_name=b_data.get("database_name", ""),
                    description=b_data.get("description", ""),
                )
                
                # Verify file exists
                if backup.file_path.exists():
                    self.backups.append(backup)
                else:
                    logger.warning(f"Backup file not found: {backup.file_path}")
            
            logger.info(f"Loaded {len(self.backups)} backups from index")
            
        except Exception as e:
            logger.warning(f"Error loading backup index: {e}")


# Global backup manager instance
_backup_manager: Optional[BackupRecoveryManager] = None


def get_backup_manager(backup_directory: Optional[Path] = None) -> BackupRecoveryManager:
    """Get the global backup manager instance."""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupRecoveryManager(backup_directory)
    return _backup_manager

