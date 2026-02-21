"""
Approval Workflow System

Manages approval workflows for manual fixes requiring competent person approval.
JORC/SAMREC compliant approval tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import logging

from .user_auth import User, Permission, get_current_user

logger = logging.getLogger(__name__)


class ApprovalStatus(Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class ApprovalRequest:
    """
    Request for approval of a manual fix.
    
    JORC/SAMREC compliant approval tracking.
    """
    request_id: str
    fix_record_id: str  # Reference to fix log record
    hole_id: str
    table: str
    check: str
    description: str
    before_values: Dict[str, Any]
    proposed_values: Dict[str, Any]
    requested_by: str  # Username
    requested_date: datetime = field(default_factory=datetime.now)
    status: ApprovalStatus = ApprovalStatus.PENDING
    approved_by: Optional[str] = None
    approved_date: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    comments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def approve(self, approver: User, comment: Optional[str] = None) -> None:
        """Approve this request."""
        if not approver.can_approve_fixes():
            raise PermissionError(f"User {approver.username} does not have approval permission")
        
        self.status = ApprovalStatus.APPROVED
        self.approved_by = approver.username
        self.approved_date = datetime.now()
        if comment:
            self.comments.append(f"[{datetime.now().isoformat()}] {approver.username}: {comment}")
        logger.info(f"Approval request {self.request_id} approved by {approver.username}")
    
    def reject(self, approver: User, reason: str) -> None:
        """Reject this request."""
        if not approver.can_approve_fixes():
            raise PermissionError(f"User {approver.username} does not have approval permission")
        
        self.status = ApprovalStatus.REJECTED
        self.approved_by = approver.username
        self.approved_date = datetime.now()
        self.rejection_reason = reason
        self.comments.append(f"[{datetime.now().isoformat()}] {approver.username}: REJECTED - {reason}")
        logger.info(f"Approval request {self.request_id} rejected by {approver.username}: {reason}")
    
    def cancel(self, user: User) -> None:
        """Cancel this request."""
        if user.username != self.requested_by:
            raise PermissionError("Only the requester can cancel a request")
        
        self.status = ApprovalStatus.CANCELLED
        self.comments.append(f"[{datetime.now().isoformat()}] {user.username}: CANCELLED")
        logger.info(f"Approval request {self.request_id} cancelled by {user.username}")


class ApprovalWorkflowManager:
    """
    Manages approval workflows for manual fixes.
    
    Tracks all approval requests and their status.
    """
    
    def __init__(self):
        self.requests: Dict[str, ApprovalRequest] = {}
        self._request_counter = 0
        logger.info("ApprovalWorkflowManager initialized")
    
    def create_request(
        self,
        fix_record_id: str,
        hole_id: str,
        table: str,
        check: str,
        description: str,
        before_values: Dict[str, Any],
        proposed_values: Dict[str, Any],
    ) -> ApprovalRequest:
        """
        Create a new approval request.
        
        Args:
            fix_record_id: Reference to fix log record
            hole_id: Hole ID being fixed
            table: Table name (assays, surveys, etc.)
            check: QC check name
            description: Description of the fix
            before_values: Values before fix
            proposed_values: Proposed values after fix
        
        Returns:
            ApprovalRequest object
        """
        current_user = get_current_user()
        if not current_user:
            raise ValueError("No current user set. Cannot create approval request.")
        
        self._request_counter += 1
        request_id = f"APPROVAL-{self._request_counter:06d}"
        
        request = ApprovalRequest(
            request_id=request_id,
            fix_record_id=fix_record_id,
            hole_id=hole_id,
            table=table,
            check=check,
            description=description,
            before_values=before_values,
            proposed_values=proposed_values,
            requested_by=current_user.username,
        )
        
        self.requests[request_id] = request
        logger.info(f"Created approval request {request_id} for {hole_id} by {current_user.username}")
        return request
    
    def get_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get an approval request by ID."""
        return self.requests.get(request_id)
    
    def get_pending_requests(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return [r for r in self.requests.values() if r.status == ApprovalStatus.PENDING]
    
    def get_requests_for_user(self, username: str) -> List[ApprovalRequest]:
        """Get all requests created by a specific user."""
        return [r for r in self.requests.values() if r.requested_by == username]
    
    def approve_request(self, request_id: str, comment: Optional[str] = None) -> bool:
        """
        Approve a request.
        
        Args:
            request_id: Request ID to approve
            comment: Optional approval comment
        
        Returns:
            True if approved, False if request not found or already processed
        """
        request = self.get_request(request_id)
        if not request:
            logger.warning(f"Approval request {request_id} not found")
            return False
        
        if request.status != ApprovalStatus.PENDING:
            logger.warning(f"Approval request {request_id} is not pending (status: {request.status.value})")
            return False
        
        current_user = get_current_user()
        if not current_user:
            logger.error("No current user set. Cannot approve request.")
            return False
        
        try:
            request.approve(current_user, comment)
            return True
        except PermissionError as e:
            logger.error(f"Permission error approving request {request_id}: {e}")
            return False
    
    def reject_request(self, request_id: str, reason: str) -> bool:
        """
        Reject a request.
        
        Args:
            request_id: Request ID to reject
            reason: Rejection reason
        
        Returns:
            True if rejected, False if request not found or already processed
        """
        request = self.get_request(request_id)
        if not request:
            logger.warning(f"Approval request {request_id} not found")
            return False
        
        if request.status != ApprovalStatus.PENDING:
            logger.warning(f"Approval request {request_id} is not pending (status: {request.status.value})")
            return False
        
        current_user = get_current_user()
        if not current_user:
            logger.error("No current user set. Cannot reject request.")
            return False
        
        try:
            request.reject(current_user, reason)
            return True
        except PermissionError as e:
            logger.error(f"Permission error rejecting request {request_id}: {e}")
            return False
    
    def cancel_request(self, request_id: str) -> bool:
        """
        Cancel a request.
        
        Args:
            request_id: Request ID to cancel
        
        Returns:
            True if cancelled, False if request not found or not cancellable
        """
        request = self.get_request(request_id)
        if not request:
            logger.warning(f"Approval request {request_id} not found")
            return False
        
        if request.status != ApprovalStatus.PENDING:
            logger.warning(f"Approval request {request_id} is not pending (status: {request.status.value})")
            return False
        
        current_user = get_current_user()
        if not current_user:
            logger.error("No current user set. Cannot cancel request.")
            return False
        
        try:
            request.cancel(current_user)
            return True
        except PermissionError as e:
            logger.error(f"Permission error cancelling request {request_id}: {e}")
            return False


# Global approval workflow manager instance
_approval_manager: Optional[ApprovalWorkflowManager] = None


def get_approval_manager() -> ApprovalWorkflowManager:
    """Get the global approval workflow manager instance."""
    global _approval_manager
    if _approval_manager is None:
        _approval_manager = ApprovalWorkflowManager()
    return _approval_manager

