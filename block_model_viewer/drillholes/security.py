"""
Simplified Security System for Local Desktop Application

Provides role-based access control (Viewer Mode vs Editor Mode) without
unnecessary server-side features like password hashing, IP logging, or session timeouts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
import logging

from .user_auth import User, UserManager, get_current_user

logger = logging.getLogger(__name__)


class AccessType(Enum):
    """Types of access operations."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXPORT = "export"
    IMPORT = "import"
    ADMIN = "admin"


@dataclass
class AccessLog:
    """Simple log entry for access tracking (audit trail only)."""
    log_id: str
    timestamp: datetime
    user: str
    access_type: AccessType
    resource: str
    allowed: bool
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """
    Simplified security manager for local desktop application.
    
    Provides role-based access control without server-side security features.
    """
    
    def __init__(self):
        self.access_logs: List[AccessLog] = []
        self._log_counter = 0
        self._max_logs = 10000
        
        logger.info("SecurityManager initialized (simplified for local desktop app)")
    
    def _generate_log_id(self) -> str:
        """Generate a unique log ID."""
        self._log_counter += 1
        return f"SEC-{self._log_counter:08d}"
    
    def check_access(
        self,
        user: Optional[User],
        access_type: AccessType,
        resource: str,
    ) -> bool:
        """
        Check if user has access to a resource.
        
        Args:
            user: User requesting access
            access_type: Type of access requested
            resource: Resource identifier
            
        Returns:
            True if access allowed, False otherwise
        """
        if user is None:
            self._log_access(None, access_type, resource, False, "No user provided")
            return False
        
        # Simple role-based access control
        if access_type == AccessType.READ:
            allowed = True  # Read access generally allowed
        elif access_type == AccessType.WRITE:
            allowed = user.can_edit_data()
        elif access_type == AccessType.DELETE:
            allowed = user.can_delete_data()
        elif access_type == AccessType.EXPORT:
            allowed = user.can_export_data()
        elif access_type == AccessType.IMPORT:
            allowed = user.can_import_data()
        elif access_type == AccessType.ADMIN:
            allowed = user.can_administer()
        else:
            allowed = False
        
        reason = "Access granted" if allowed else "Insufficient permissions"
        self._log_access(user.username, access_type, resource, allowed, reason)
        
        return allowed
    
    def _log_access(
        self,
        user: str,
        access_type: AccessType,
        resource: str,
        allowed: bool,
        reason: str,
    ):
        """Log an access attempt (for audit trail only)."""
        log = AccessLog(
            log_id=self._generate_log_id(),
            timestamp=datetime.now(),
            user=user or "anonymous",
            access_type=access_type,
            resource=resource,
            allowed=allowed,
            reason=reason,
        )
        
        self.access_logs.append(log)
        
        # Trim old logs to prevent memory growth
        if len(self.access_logs) > self._max_logs:
            self.access_logs = self.access_logs[-self._max_logs:]
        
        if not allowed:
            logger.warning(f"Access denied: {user} -> {access_type.value} on {resource}: {reason}")
    
    def get_access_logs(
        self,
        user: Optional[str] = None,
        access_type: Optional[AccessType] = None,
        resource: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        denied_only: bool = False,
    ) -> List[AccessLog]:
        """
        Get access logs with filters (for audit trail).
        
        Args:
            user: Filter by user
            access_type: Filter by access type
            resource: Filter by resource
            start_date: Filter by start date
            end_date: Filter by end date
            denied_only: Only return denied access attempts
            
        Returns:
            List of AccessLog objects
        """
        filtered = self.access_logs
        
        if user:
            filtered = [log for log in filtered if log.user == user]
        
        if access_type:
            filtered = [log for log in filtered if log.access_type == access_type]
        
        if resource:
            filtered = [log for log in filtered if log.resource == resource]
        
        if start_date:
            filtered = [log for log in filtered if log.timestamp >= start_date]
        
        if end_date:
            filtered = [log for log in filtered if log.timestamp <= end_date]
        
        if denied_only:
            filtered = [log for log in filtered if not log.allowed]
        
        return sorted(filtered, key=lambda log: log.timestamp, reverse=True)
    
    def get_security_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get security statistics (for audit reporting).
        
        Args:
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Dictionary with security statistics
        """
        logs = self.get_access_logs(start_date=start_date, end_date=end_date)
        
        stats = {
            "total_access_attempts": len(logs),
            "allowed": len([log for log in logs if log.allowed]),
            "denied": len([log for log in logs if not log.allowed]),
            "by_access_type": {},
            "by_user": {},
            "denied_resources": {},
        }
        
        for log in logs:
            # Count by access type
            key = log.access_type.value
            stats["by_access_type"][key] = stats["by_access_type"].get(key, 0) + 1
            
            # Count by user
            stats["by_user"][log.user] = stats["by_user"].get(log.user, 0) + 1
            
            # Track denied resources
            if not log.allowed:
                stats["denied_resources"][log.resource] = stats["denied_resources"].get(log.resource, 0) + 1
        
        return stats


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get the global security manager instance."""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager
