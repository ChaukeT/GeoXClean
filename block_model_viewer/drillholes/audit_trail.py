"""
Audit Trail System

Provides comprehensive audit trail for all data changes.
Tracks who, what, when, and why for complete data history.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditAction(Enum):
    """Types of audit actions."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    IMPORT = "import"
    EXPORT = "export"
    QC_RUN = "qc_run"
    FIX_APPLIED = "fix_applied"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_APPROVED = "approval_approved"
    APPROVAL_REJECTED = "approval_rejected"
    CONFIG_CHANGED = "config_changed"


@dataclass
class AuditRecord:
    """
    A single audit record.
    
    Tracks a single action with complete context.
    """
    record_id: str
    timestamp: datetime
    user: str
    action: AuditAction
    entity_type: str  # e.g., "drillhole", "assay", "qc_result"
    entity_id: str
    description: str
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit record to dictionary."""
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp.isoformat(),
            "user": self.user,
            "action": self.action.value,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "description": self.description,
            "before_state": self.before_state,
            "after_state": self.after_state,
            "metadata": self.metadata,
        }


class AuditTrail:
    """
    Comprehensive audit trail system.
    
    Tracks all data changes and system actions for complete history.
    """
    
    def __init__(self):
        self.records: List[AuditRecord] = []
        self._record_counter = 0
        self._max_records = 100000  # Keep last 100k records
        logger.info("AuditTrail initialized")
    
    def _generate_record_id(self) -> str:
        """Generate a unique record ID."""
        self._record_counter += 1
        return f"AUDIT-{self._record_counter:08d}"
    
    def log_action(
        self,
        user: str,
        action: AuditAction,
        entity_type: str,
        entity_id: str,
        description: str,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditRecord:
        """
        Log an audit action.
        
        Args:
            user: Username who performed the action
            action: Type of action
            entity_type: Type of entity (e.g., "drillhole", "assay")
            entity_id: ID of the entity
            description: Human-readable description
            before_state: Optional state before the action
            after_state: Optional state after the action
            metadata: Optional additional metadata
        
        Returns:
            Created AuditRecord
        """
        record = AuditRecord(
            record_id=self._generate_record_id(),
            timestamp=datetime.now(),
            user=user,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            description=description,
            before_state=before_state,
            after_state=after_state,
            metadata=metadata or {},
        )
        
        self.records.append(record)
        
        # Trim old records if we exceed max
        if len(self.records) > self._max_records:
            self.records = self.records[-self._max_records:]
        
        logger.debug(f"Audit log: {action.value} on {entity_type}/{entity_id} by {user}")
        return record
    
    def get_records(
        self,
        user: Optional[str] = None,
        action: Optional[AuditAction] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[AuditRecord]:
        """
        Query audit records with filters.
        
        Args:
            user: Filter by user
            action: Filter by action type
            entity_type: Filter by entity type
            entity_id: Filter by entity ID
            start_date: Filter by start date
            end_date: Filter by end date
        
        Returns:
            List of matching AuditRecord objects
        """
        filtered = self.records
        
        if user:
            filtered = [r for r in filtered if r.user == user]
        
        if action:
            filtered = [r for r in filtered if r.action == action]
        
        if entity_type:
            filtered = [r for r in filtered if r.entity_type == entity_type]
        
        if entity_id:
            filtered = [r for r in filtered if r.entity_id == entity_id]
        
        if start_date:
            filtered = [r for r in filtered if r.timestamp >= start_date]
        
        if end_date:
            filtered = [r for r in filtered if r.timestamp <= end_date]
        
        return sorted(filtered, key=lambda r: r.timestamp, reverse=True)
    
    def get_entity_history(
        self,
        entity_type: str,
        entity_id: str,
    ) -> List[AuditRecord]:
        """
        Get complete history for a specific entity.
        
        Args:
            entity_type: Type of entity
            entity_id: ID of entity
        
        Returns:
            List of AuditRecord objects in chronological order
        """
        records = self.get_records(entity_type=entity_type, entity_id=entity_id)
        return sorted(records, key=lambda r: r.timestamp)
    
    def export_audit_trail(
        self,
        output_path: Path,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> bool:
        """
        Export audit trail to file.
        
        Args:
            output_path: Path to output file
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            True if successful, False otherwise
        """
        try:
            records = self.get_records(start_date=start_date, end_date=end_date)
            
            export_data = {
                "export_date": datetime.now().isoformat(),
                "total_records": len(records),
                "start_date": start_date.isoformat() if start_date else None,
                "end_date": end_date.isoformat() if end_date else None,
                "records": [r.to_dict() for r in records],
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Exported {len(records)} audit records to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting audit trail: {e}", exc_info=True)
            return False
    
    def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get audit trail statistics.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            Dictionary with statistics
        """
        records = self.get_records(start_date=start_date, end_date=end_date)
        
        stats = {
            "total_records": len(records),
            "by_action": {},
            "by_user": {},
            "by_entity_type": {},
        }
        
        for record in records:
            # Count by action
            action_key = record.action.value
            stats["by_action"][action_key] = stats["by_action"].get(action_key, 0) + 1
            
            # Count by user
            stats["by_user"][record.user] = stats["by_user"].get(record.user, 0) + 1
            
            # Count by entity type
            stats["by_entity_type"][record.entity_type] = stats["by_entity_type"].get(record.entity_type, 0) + 1
        
        return stats


# Global audit trail instance
_audit_trail: Optional[AuditTrail] = None


def get_audit_trail() -> AuditTrail:
    """Get the global audit trail instance."""
    global _audit_trail
    if _audit_trail is None:
        _audit_trail = AuditTrail()
    return _audit_trail

