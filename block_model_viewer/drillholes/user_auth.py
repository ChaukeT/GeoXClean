"""
User Authentication and Authorization System

Provides user management, roles, and permissions for JORC/SAMREC compliance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Permission(Enum):
    """User permissions for QC/Editor operations."""
    VIEW_QC_RESULTS = "view_qc_results"
    RUN_QC = "run_qc"
    VIEW_EDITOR = "view_editor"
    APPLY_AUTO_FIX = "apply_auto_fix"
    APPLY_MANUAL_FIX = "apply_manual_fix"
    APPROVE_MANUAL_FIX = "approve_manual_fix"
    EXPORT_REPORTS = "export_reports"
    CONFIGURE_QC = "configure_qc"
    MANAGE_CONTROL_SAMPLES = "manage_control_samples"
    COMPETENT_PERSON = "competent_person"  # JORC requirement


class Role(Enum):
    """User roles with predefined permission sets."""
    VIEWER = "viewer"
    TECHNICIAN = "technician"
    GEOLOGIST = "geologist"
    SENIOR_GEOLOGIST = "senior_geologist"
    COMPETENT_PERSON = "competent_person"  # JORC requirement
    ADMIN = "admin"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.VIEW_QC_RESULTS,
        Permission.VIEW_EDITOR,
    },
    Role.TECHNICIAN: {
        Permission.VIEW_QC_RESULTS,
        Permission.VIEW_EDITOR,
        Permission.RUN_QC,
        Permission.APPLY_AUTO_FIX,
    },
    Role.GEOLOGIST: {
        Permission.VIEW_QC_RESULTS,
        Permission.VIEW_EDITOR,
        Permission.RUN_QC,
        Permission.APPLY_AUTO_FIX,
        Permission.APPLY_MANUAL_FIX,
        Permission.EXPORT_REPORTS,
    },
    Role.SENIOR_GEOLOGIST: {
        Permission.VIEW_QC_RESULTS,
        Permission.VIEW_EDITOR,
        Permission.RUN_QC,
        Permission.APPLY_AUTO_FIX,
        Permission.APPLY_MANUAL_FIX,
        Permission.APPROVE_MANUAL_FIX,
        Permission.EXPORT_REPORTS,
        Permission.CONFIGURE_QC,
    },
    Role.COMPETENT_PERSON: {
        Permission.VIEW_QC_RESULTS,
        Permission.VIEW_EDITOR,
        Permission.RUN_QC,
        Permission.APPLY_AUTO_FIX,
        Permission.APPLY_MANUAL_FIX,
        Permission.APPROVE_MANUAL_FIX,
        Permission.EXPORT_REPORTS,
        Permission.CONFIGURE_QC,
        Permission.MANAGE_CONTROL_SAMPLES,
        Permission.COMPETENT_PERSON,
    },
    Role.ADMIN: set(Permission),  # All permissions
}


@dataclass
class User:
    """
    User account with role and permissions.
    
    JORC/SAMREC compliant user management.
    """
    username: str
    full_name: str
    email: str
    role: Role
    permissions: Set[Permission] = field(default_factory=set)
    created_date: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize permissions from role if not explicitly set."""
        if not self.permissions:
            self.permissions = ROLE_PERMISSIONS.get(self.role, set()).copy()
        logger.debug(f"User {self.username} initialized with role {self.role.value}, {len(self.permissions)} permissions")
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.permissions
    
    def can_approve_fixes(self) -> bool:
        """Check if user can approve manual fixes (JORC requirement)."""
        return (
            self.has_permission(Permission.APPROVE_MANUAL_FIX) or
            self.has_permission(Permission.COMPETENT_PERSON)
        )
    
    def is_competent_person(self) -> bool:
        """Check if user is a Competent Person (JORC requirement)."""
        return self.has_permission(Permission.COMPETENT_PERSON)
    
    # Simple role-based access control methods for local desktop app
    def can_edit_data(self) -> bool:
        """Check if user can edit data (Editor Mode)."""
        return self.role != Role.VIEWER
    
    def can_delete_data(self) -> bool:
        """Check if user can delete data."""
        return self.role in (Role.ADMIN, Role.SENIOR_GEOLOGIST, Role.COMPETENT_PERSON)
    
    def can_export_data(self) -> bool:
        """Check if user can export data."""
        return self.role != Role.VIEWER
    
    def can_import_data(self) -> bool:
        """Check if user can import data."""
        return self.role != Role.VIEWER
    
    def can_administer(self) -> bool:
        """Check if user has admin privileges."""
        return self.role == Role.ADMIN


class UserManager:
    """
    Manages users and authentication.
    
    Singleton pattern for global user management.
    """
    _instance: Optional['UserManager'] = None
    _current_user: Optional[User] = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._users: Dict[str, User] = {}
            cls._instance._initialize_default_users()
        return cls._instance
    
    def _initialize_default_users(self):
        """Initialize default users for testing/development."""
        default_users = [
            User(
                username="admin",
                full_name="System Administrator",
                email="admin@example.com",
                role=Role.ADMIN,
            ),
            User(
                username="geologist",
                full_name="Senior Geologist",
                email="geologist@example.com",
                role=Role.SENIOR_GEOLOGIST,
            ),
            User(
                username="technician",
                full_name="Data Technician",
                email="technician@example.com",
                role=Role.TECHNICIAN,
            ),
        ]
        for user in default_users:
            self._users[user.username] = user
        logger.info(f"UserManager initialized with {len(self._users)} default users")
    
    def add_user(self, user: User) -> None:
        """Add a new user."""
        self._users[user.username] = user
        logger.info(f"Added user: {user.username} ({user.role.value})")
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self._users.get(username)
    
    def list_users(self) -> List[User]:
        """List all users."""
        return list(self._users.values())
    
    def set_current_user(self, username: str) -> bool:
        """
        Set the current active user.
        
        Args:
            username: Username to set as current
        
        Returns:
            True if user exists and was set, False otherwise
        """
        user = self.get_user(username)
        if user and user.is_active:
            self._current_user = user
            user.last_login = datetime.now()
            logger.info(f"Current user set to: {user.username} ({user.role.value})")
            return True
        logger.warning(f"Failed to set current user: {username} (not found or inactive)")
        return False
    
    def get_current_user(self) -> Optional[User]:
        """Get the current active user."""
        return self._current_user
    
    def login(self, username: str) -> bool:
        """
        Login a user (simplified - no password for now).
        
        Args:
            username: Username to login
        
        Returns:
            True if login successful, False otherwise
        """
        return self.set_current_user(username)
    
    def logout(self) -> None:
        """Logout current user."""
        if self._current_user:
            logger.info(f"User logged out: {self._current_user.username}")
        self._current_user = None
    
    def check_permission(self, permission: Permission) -> bool:
        """
        Check if current user has a permission.
        
        Args:
            permission: Permission to check
        
        Returns:
            True if current user has permission, False otherwise
        """
        if not self._current_user:
            logger.warning("No current user set, permission check failed")
            return False
        return self._current_user.has_permission(permission)


# Global user manager instance
_user_manager: Optional[UserManager] = None


def get_user_manager() -> UserManager:
    """Get the global user manager instance."""
    global _user_manager
    if _user_manager is None:
        _user_manager = UserManager()
    return _user_manager


def get_current_user() -> Optional[User]:
    """Get the current active user."""
    return get_user_manager().get_current_user()


def require_permission(permission: Permission):
    """
    Decorator to require a permission for a function.
    
    Usage:
        @require_permission(Permission.APPROVE_MANUAL_FIX)
        def approve_fix(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            user_manager = get_user_manager()
            if not user_manager.check_permission(permission):
                raise PermissionError(
                    f"Permission '{permission.value}' required. "
                    f"Current user: {user_manager.get_current_user().username if user_manager.get_current_user() else 'None'}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator

