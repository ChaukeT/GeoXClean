"""
DRILLHOLE AUDIT TRAIL (GeoX)

Tracks all changes made to drillhole data for SAMREC/JORC compliance:
- Auto-fixes (from autofix engine)
- Manual edits (from manual edit engine)
- Timestamps, users, before/after values
- Export to PDF/Excel for auditing
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

import pandas as pd


class ChangeType(Enum):
    """Type of change made."""
    AUTO_FIX = "Auto-Fix"
    MANUAL_EDIT = "Manual Edit"
    BATCH_EDIT = "Batch Edit"
    FIND_REPLACE = "Find & Replace"


@dataclass
class AuditEntry:
    """Single audit entry for a change."""
    timestamp: datetime
    change_type: ChangeType
    user: str
    table: str
    hole_id: str
    row_index: int
    column: str
    old_value: Any
    new_value: Any
    rule_code: Optional[str]  # For auto-fixes
    reason: str
    confidence: Optional[float] = None  # For auto-fixes (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "Timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Change Type": self.change_type.value,
            "User": self.user,
            "Table": self.table.upper(),
            "Hole ID": str(self.hole_id),
            "Row Index": self.row_index,
            "Column": self.column,
            "Old Value": self._format_value(self.old_value),
            "New Value": self._format_value(self.new_value),
            "Rule Code": self.rule_code or "",
            "Reason": self.reason,
            "Confidence": f"{self.confidence:.2f}" if self.confidence is not None else "",
        }
    
    def _format_value(self, value: Any) -> str:
        """Format value for display."""
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.6f}".rstrip('0').rstrip('.')
        return str(value)


class AuditTrail:
    """
    Central audit trail manager for all drillhole data changes.
    
    Tracks:
    - All auto-fixes
    - All manual edits
    - Timestamps and users
    - Before/after values
    """
    
    def __init__(self, project_name: str = "Drillhole QC", user: str = "GEOLOGIST"):
        self.project_name = project_name
        self.user = user
        self.entries: List[AuditEntry] = []
        self.session_start = datetime.now()
    
    def add_autofix(
        self,
        table: str,
        rule_code: str,
        hole_id: str,
        row_index: int,
        column: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        confidence: float,
    ):
        """Record an auto-fix action."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            change_type=ChangeType.AUTO_FIX,
            user="SYSTEM",
            table=table,
            hole_id=str(hole_id),
            row_index=row_index,
            column=column,
            old_value=old_value,
            new_value=new_value,
            rule_code=rule_code,
            reason=reason,
            confidence=confidence,
        )
        self.entries.append(entry)
    
    def add_manual_edit(
        self,
        table: str,
        hole_id: str,
        row_index: int,
        column: str,
        old_value: Any,
        new_value: Any,
        reason: str,
        user: Optional[str] = None,
    ):
        """Record a manual edit."""
        entry = AuditEntry(
            timestamp=datetime.now(),
            change_type=ChangeType.MANUAL_EDIT,
            user=user or self.user,
            table=table,
            hole_id=str(hole_id),
            row_index=row_index,
            column=column,
            old_value=old_value,
            new_value=new_value,
            rule_code=None,
            reason=reason,
            confidence=None,
        )
        self.entries.append(entry)
    
    def add_batch_edit(
        self,
        table: str,
        edits: List[Dict[str, Any]],
        reason: str,
        user: Optional[str] = None,
    ):
        """Record a batch edit (multiple rows)."""
        for edit in edits:
            entry = AuditEntry(
                timestamp=datetime.now(),
                change_type=ChangeType.BATCH_EDIT,
                user=user or self.user,
                table=table,
                hole_id=edit.get("hole_id", ""),
                row_index=edit.get("row_index", -1),
                column=edit.get("column", ""),
                old_value=edit.get("old_value"),
                new_value=edit.get("new_value"),
                rule_code=None,
                reason=reason,
                confidence=None,
            )
            self.entries.append(entry)
    
    def add_find_replace(
        self,
        table: str,
        edits: List[Dict[str, Any]],
        find_text: str,
        replace_text: str,
        user: Optional[str] = None,
    ):
        """Record a find & replace operation."""
        for edit in edits:
            entry = AuditEntry(
                timestamp=datetime.now(),
                change_type=ChangeType.FIND_REPLACE,
                user=user or self.user,
                table=table,
                hole_id=edit.get("hole_id", ""),
                row_index=edit.get("row_index", -1),
                column=edit.get("column", ""),
                old_value=edit.get("old_value"),
                new_value=edit.get("new_value"),
                rule_code=None,
                reason=f"Find & Replace: '{find_text}' → '{replace_text}'",
                confidence=None,
            )
            self.entries.append(entry)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.entries:
            return {
                "total_changes": 0,
                "auto_fixes": 0,
                "manual_edits": 0,
                "batch_edits": 0,
                "find_replace": 0,
                "tables_affected": set(),
                "holes_affected": set(),
                "session_duration": "0:00:00",
            }
        
        auto_fixes = sum(1 for e in self.entries if e.change_type == ChangeType.AUTO_FIX)
        manual_edits = sum(1 for e in self.entries if e.change_type == ChangeType.MANUAL_EDIT)
        batch_edits = sum(1 for e in self.entries if e.change_type == ChangeType.BATCH_EDIT)
        find_replace = sum(1 for e in self.entries if e.change_type == ChangeType.FIND_REPLACE)
        
        tables_affected = set(e.table for e in self.entries)
        holes_affected = set(e.hole_id for e in self.entries if e.hole_id)
        
        duration = datetime.now() - self.session_start
        duration_str = str(duration).split('.')[0]  # Remove microseconds
        
        return {
            "total_changes": len(self.entries),
            "auto_fixes": auto_fixes,
            "manual_edits": manual_edits,
            "batch_edits": batch_edits,
            "find_replace": find_replace,
            "tables_affected": tables_affected,
            "holes_affected": holes_affected,
            "session_duration": duration_str,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert audit trail to DataFrame for export."""
        if not self.entries:
            return pd.DataFrame()
        
        data = [entry.to_dict() for entry in self.entries]
        df = pd.DataFrame(data)
        return df
    
    def clear(self):
        """Clear all audit entries."""
        self.entries.clear()
        self.session_start = datetime.now()

