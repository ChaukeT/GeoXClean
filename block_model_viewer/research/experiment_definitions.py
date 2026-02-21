"""
Experiment Definitions

Formalize the structure of experiments and scenario grids for research mode.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import itertools
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ExperimentParameter:
    """A parameter to vary in an experiment."""
    name: str  # e.g. "variogram_model", "composite_length", "cutoff"
    values: List[Any]  # discrete set or numeric grid
    type: str  # "categorical" | "numeric"
    notes: str = ""


@dataclass
class ExperimentDefinition:
    """Definition of an experiment with parameters and base configuration."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    domain: str = "geostats"  # "geostats", "planning", "uncertainty", "geotech", etc.
    parameters: List[ExperimentParameter] = field(default_factory=list)
    base_config: Dict[str, Any] = field(default_factory=dict)
    metrics: List[str] = field(default_factory=list)  # e.g. ["rmse_cv", "npv", "irr", "gt_loss"]
    pipelines: List[str] = field(default_factory=list)  # e.g. ["ok", "loocv"], ["pit_optimization", "npv"]


@dataclass
class ExperimentInstance:
    """A single instance of an experiment with specific parameter values."""
    definition_id: str
    index: int
    parameter_values: Dict[str, Any]
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioGrid:
    """A grid of experiment instances generated from an experiment definition."""
    definition: ExperimentDefinition
    instances: List[ExperimentInstance]
    
    @classmethod
    def from_definition(cls, definition: ExperimentDefinition, seed_base: Optional[int] = None) -> "ScenarioGrid":
        """
        Generate a scenario grid from an experiment definition.
        
        Creates a full Cartesian product of all parameter values.
        
        Args:
            definition: ExperimentDefinition instance
            seed_base: Base seed for random number generation (optional)
        
        Returns:
            ScenarioGrid with all combinations
        """
        if not definition.parameters:
            # No parameters: single instance
            instances = [
                ExperimentInstance(
                    definition_id=definition.id,
                    index=0,
                    parameter_values={},
                    seed=seed_base
                )
            ]
            return cls(definition=definition, instances=instances)
        
        # Generate Cartesian product of parameter values
        param_names = [p.name for p in definition.parameters]
        param_value_lists = [p.values for p in definition.parameters]
        
        combinations = list(itertools.product(*param_value_lists))
        
        instances = []
        for idx, combo in enumerate(combinations):
            param_values = dict(zip(param_names, combo))
            
            # Generate seed if base seed provided
            seed = None
            if seed_base is not None:
                seed = seed_base + idx
            
            instance = ExperimentInstance(
                definition_id=definition.id,
                index=idx,
                parameter_values=param_values,
                seed=seed,
                metadata={
                    'parameter_names': param_names,
                    'combination_index': idx
                }
            )
            instances.append(instance)
        
        logger.info(f"Generated scenario grid: {len(instances)} instances from {len(definition.parameters)} parameters")
        
        return cls(definition=definition, instances=instances)
    
    def get_instance(self, index: int) -> Optional[ExperimentInstance]:
        """Get an experiment instance by index."""
        if 0 <= index < len(self.instances):
            return self.instances[index]
        return None
    
    def filter_instances(self, **criteria) -> List[ExperimentInstance]:
        """
        Filter instances by parameter values.
        
        Args:
            **criteria: Parameter name -> value mappings
        
        Returns:
            List of matching instances
        """
        filtered = []
        for instance in self.instances:
            match = True
            for param_name, param_value in criteria.items():
                if instance.parameter_values.get(param_name) != param_value:
                    match = False
                    break
            if match:
                filtered.append(instance)
        return filtered

