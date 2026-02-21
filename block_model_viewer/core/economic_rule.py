"""
Economic Rule - Centralized Ore/Waste Classification
=====================================================

Prevents rule divergence between Economic tab and Waste/Ore tab.

JORC/SAMREC Compliance:
- Deterministic rule definition
- Hash-based audit trail
- Single source of truth
- Provenance tracking
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class EconomicRule:
    """
    Centralized economic cutoff rule for ore/waste classification.

    This class ensures that cutoff criteria are defined ONCE and referenced
    consistently across all compositing workflows.

    Attributes:
        field: Grade field name (e.g., "Fe", "Au", "Cu")
        operator: Comparison operator (">=", ">", "<", "<=")
        cutoff: Cutoff value (e.g., 62.0 for Fe >= 62%)
        ore_code: Numeric code assigned to ore intervals (default: 1)
        waste_code: Numeric code assigned to waste intervals (default: 0)
        name: Human-readable name for this rule (default: "Default Rule")
        created_at: Timestamp of rule creation (auto-generated)
    """

    field: str
    operator: str
    cutoff: float
    ore_code: int = 1
    waste_code: int = 0
    name: str = "Default Rule"
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Validate rule parameters."""
        if not self.field:
            raise ValueError("Economic rule must specify a field")

        if self.operator not in [">=", ">", "<", "<="]:
            raise ValueError(f"Invalid operator: {self.operator}. Must be >=, >, <, or <=")

        if not isinstance(self.cutoff, (int, float)):
            raise TypeError(f"Cutoff must be numeric, got {type(self.cutoff)}")

    def get_signature(self) -> str:
        """
        Generate deterministic hash signature for audit trail.

        The signature is based on the rule definition (field, operator, cutoff)
        and is used for JORC/SAMREC compliance to verify that datasets were
        generated using consistent criteria.

        Returns:
            12-character hexadecimal hash of the rule

        Example:
            >>> rule = EconomicRule("Fe", ">=", 62.0)
            >>> rule.get_signature()
            'a3f2c1b4e5d6'
        """
        # Only include field, operator, cutoff in signature
        # (ore_code and waste_code are output mapping, not classification logic)
        rule_string = f"{self.field}{self.operator}{self.cutoff}"
        hash_digest = hashlib.sha256(rule_string.encode('utf-8')).hexdigest()
        return hash_digest[:12]  # First 12 chars for readability

    def format_for_display(self) -> str:
        """
        Format rule as human-readable string.

        Returns:
            Formatted rule string (e.g., "Fe >= 62.0")
        """
        return f"{self.field} {self.operator} {self.cutoff}"

    def format_with_codes(self) -> str:
        """
        Format rule with ore/waste code mapping.

        Returns:
            Full rule with codes (e.g., "Fe >= 62.0 → Ore(1)/Waste(0)")
        """
        return f"{self.format_for_display()} → Ore({self.ore_code})/Waste({self.waste_code})"

    def to_dict(self) -> dict:
        """
        Serialize rule to dictionary for storage.

        Returns:
            Dictionary representation with all fields
        """
        return {
            'field': self.field,
            'operator': self.operator,
            'cutoff': self.cutoff,
            'ore_code': self.ore_code,
            'waste_code': self.waste_code,
            'name': self.name,
            'signature': self.get_signature(),
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else str(self.created_at)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EconomicRule':
        """
        Deserialize rule from dictionary.

        Args:
            data: Dictionary with rule fields

        Returns:
            EconomicRule instance
        """
        # Convert created_at from ISO string if present
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except (ValueError, AttributeError):
                created_at = datetime.now()
        elif created_at is None:
            created_at = datetime.now()

        return cls(
            field=data['field'],
            operator=data['operator'],
            cutoff=data['cutoff'],
            ore_code=data.get('ore_code', 1),
            waste_code=data.get('waste_code', 0),
            name=data.get('name', 'Default Rule'),
            created_at=created_at
        )

    def matches_criteria(self, field: str, operator: str, cutoff: float) -> bool:
        """
        Check if given criteria match this rule.

        Args:
            field: Field name to check
            operator: Operator to check
            cutoff: Cutoff value to check

        Returns:
            True if criteria match this rule exactly
        """
        return (
            self.field == field and
            self.operator == operator and
            abs(self.cutoff - cutoff) < 1e-9  # Float comparison tolerance
        )


class EconomicRuleManager:
    """
    Manager for economic rules with DataRegistry integration.

    Provides:
    - Rule validation
    - Mismatch detection
    - Audit logging
    """

    REGISTRY_KEY = "economic_rule"

    @staticmethod
    def validate_rule(rule: EconomicRule) -> tuple[bool, list[str]]:
        """
        Validate economic rule for common issues.

        Args:
            rule: EconomicRule to validate

        Returns:
            (is_valid, warnings_list) tuple
        """
        warnings = []

        # Check for unrealistic cutoff values
        if rule.cutoff < 0:
            warnings.append(f"Negative cutoff ({rule.cutoff}) - verify this is intentional")

        if rule.cutoff > 10000:
            warnings.append(f"Very high cutoff ({rule.cutoff}) - verify units are correct")

        # Check ore/waste codes are different
        if rule.ore_code == rule.waste_code:
            warnings.append(f"Ore code ({rule.ore_code}) equals waste code ({rule.waste_code}) - classification will fail")

        # Check operator makes sense for typical mining scenarios
        if rule.operator in ["<", "<="] and rule.cutoff > 100:
            warnings.append(f"Using {rule.operator} with high cutoff ({rule.cutoff}) - verify this is correct")

        is_valid = len(warnings) == 0
        return is_valid, warnings

    @staticmethod
    def detect_mismatch(
        economic_rule: EconomicRule,
        waste_ore_field: str,
        waste_ore_operator: str,
        waste_ore_cutoff: float
    ) -> Optional[str]:
        """
        Detect rule divergence between Economic tab and Waste/Ore tab.

        Args:
            economic_rule: Rule from Economic tab
            waste_ore_field: Field from Waste/Ore tab
            waste_ore_operator: Operator from Waste/Ore tab
            waste_ore_cutoff: Cutoff from Waste/Ore tab

        Returns:
            Mismatch description if rules differ, None if they match
        """
        if not economic_rule.matches_criteria(waste_ore_field, waste_ore_operator, waste_ore_cutoff):
            return (
                f"RULE MISMATCH DETECTED:\n"
                f"  Economic tab: {economic_rule.format_for_display()}\n"
                f"  Waste/Ore tab: {waste_ore_field} {waste_ore_operator} {waste_ore_cutoff}\n\n"
                f"This violates JORC/SAMREC compliance.\n"
                f"Use 'Sync from Economic Tab' to fix."
            )
        return None

    @staticmethod
    def log_rule_change(rule: EconomicRule, reason: str = "User modification"):
        """
        Log rule changes for audit trail.

        Args:
            rule: The new/modified rule
            reason: Reason for the change
        """
        logger.info(
            f"Economic Rule Changed: {rule.format_with_codes()} "
            f"[Signature: {rule.get_signature()}] - Reason: {reason}"
        )
