"""
Scenario Definition Types (STEP 31)

Configuration objects that describe a full planning scenario.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScenarioID:
    """
    Scenario identifier.
    
    Attributes:
        name: Scenario name
        version: Version string (e.g., "v1", "v2", "latest")
    """
    name: str
    version: str = "v1"
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.name}_{self.version}"
    
    def __hash__(self) -> int:
        """Hash for use in dicts/sets."""
        return hash((self.name, self.version))
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, ScenarioID):
            return False
        return self.name == other.name and self.version == other.version


@dataclass
class ScenarioInputs:
    """
    Input configuration for a planning scenario.
    
    Attributes:
        model_name: Which block model to use
        value_mode: "base" or "geomet"
        value_field: Value field name (e.g., "block_value", "gvalue_PlantA")
        pit_config: Pit optimization configuration (LG/discounted LG options)
        schedule_config: Strategic/tactical/short-term schedule configs
        cutoff_config: Cutoff optimization configuration
        geomet_config: Geometallurgy configuration
        gc_config: Grade Control + digline settings
        risk_config: Price/cost uncertainty, fleet availability
        esg_config: Optional ESG metrics weighting
    """
    model_name: str
    value_mode: str = "base"
    value_field: str = "block_value"
    pit_config: Optional[Dict[str, Any]] = None
    schedule_config: Optional[Dict[str, Any]] = None
    cutoff_config: Optional[Dict[str, Any]] = None
    geomet_config: Optional[Dict[str, Any]] = None
    gc_config: Optional[Dict[str, Any]] = None
    risk_config: Optional[Dict[str, Any]] = None
    esg_config: Optional[Dict[str, Any]] = None
    npvs_config: Optional[Dict[str, Any]] = None  # STEP 32
    
    def __post_init__(self):
        """Validate inputs."""
        if self.value_mode not in ["base", "geomet"]:
            raise ValueError(f"value_mode must be 'base' or 'geomet', got '{self.value_mode}'")


@dataclass
class ScenarioOutputs:
    """
    Output references for a completed scenario.
    
    Attributes:
        irr_result_ref: Reference to IRR results
        schedule_result_ref: Reference to schedule
        pit_shell_ref: Reference to pit shells
        geomet_attrs_ref: Reference to geomet attributes
        gc_model_ref: Reference to GC model
        recon_result_ref: Reference to reconciliation results
        risk_result_ref: Reference to risk/uncertainty results
    """
    irr_result_ref: Optional[str] = None
    schedule_result_ref: Optional[str] = None
    pit_shell_ref: Optional[str] = None
    geomet_attrs_ref: Optional[str] = None
    gc_model_ref: Optional[str] = None
    recon_result_ref: Optional[str] = None
    risk_result_ref: Optional[str] = None


@dataclass
class PlanningScenario:
    """
    Complete planning scenario definition.
    
    Attributes:
        id: ScenarioID
        description: Scenario description
        tags: List of tags for categorization
        inputs: ScenarioInputs
        outputs: ScenarioOutputs (None if not yet run)
        status: Status ("new", "running", "completed", "failed")
        created_at: Creation timestamp
        modified_at: Last modification timestamp
    """
    id: ScenarioID
    description: str = ""
    tags: List[str] = field(default_factory=list)
    inputs: ScenarioInputs = None
    outputs: Optional[ScenarioOutputs] = None
    status: str = "new"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate scenario."""
        if self.status not in ["new", "running", "completed", "failed"]:
            raise ValueError(f"status must be one of ['new', 'running', 'completed', 'failed'], got '{self.status}'")
        
        if self.inputs is None:
            raise ValueError("inputs must be provided")
        
        # Update modified_at if not explicitly set
        if self.modified_at == self.created_at:
            self.modified_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert scenario to dictionary for serialization."""
        return {
            "id": {
                "name": self.id.name,
                "version": self.id.version
            },
            "description": self.description,
            "tags": self.tags,
            "inputs": {
                "model_name": self.inputs.model_name,
                "value_mode": self.inputs.value_mode,
                "value_field": self.inputs.value_field,
                "pit_config": self.inputs.pit_config,
                "schedule_config": self.inputs.schedule_config,
                "cutoff_config": self.inputs.cutoff_config,
                "geomet_config": self.inputs.geomet_config,
                "gc_config": self.inputs.gc_config,
                "risk_config": self.inputs.risk_config,
                "esg_config": self.inputs.esg_config,
                "npvs_config": self.inputs.npvs_config,
            },
            "outputs": {
                "irr_result_ref": self.outputs.irr_result_ref if self.outputs else None,
                "schedule_result_ref": self.outputs.schedule_result_ref if self.outputs else None,
                "pit_shell_ref": self.outputs.pit_shell_ref if self.outputs else None,
                "geomet_attrs_ref": self.outputs.geomet_attrs_ref if self.outputs else None,
                "gc_model_ref": self.outputs.gc_model_ref if self.outputs else None,
                "recon_result_ref": self.outputs.recon_result_ref if self.outputs else None,
                "risk_result_ref": self.outputs.risk_result_ref if self.outputs else None,
            } if self.outputs else None,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningScenario":
        """Create scenario from dictionary."""
        scenario_id = ScenarioID(
            name=data["id"]["name"],
            version=data["id"]["version"]
        )
        
        inputs = ScenarioInputs(**data["inputs"])
        
        outputs = None
        if data.get("outputs"):
            outputs = ScenarioOutputs(**data["outputs"])
        
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        modified_at = datetime.fromisoformat(data["modified_at"]) if isinstance(data["modified_at"], str) else data["modified_at"]
        
        return cls(
            id=scenario_id,
            description=data.get("description", ""),
            tags=data.get("tags", []),
            inputs=inputs,
            outputs=outputs,
            status=data.get("status", "new"),
            created_at=created_at,
            modified_at=modified_at
        )

