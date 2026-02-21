"""
Experiment Runner

Execute experiment instances by calling appropriate pipelines and collecting metrics.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import copy

from .experiment_definitions import ScenarioGrid, ExperimentInstance
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRunResult:
    """Result from running a scenario grid."""
    definition_id: str
    results: List[Dict[str, Any]]  # one per instance
    metrics: List[str]  # metric names
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """
    Execute experiment instances by calling appropriate pipelines.
    
    Maps parameter values to engine configs and runs pipelines via controller.
    """
    
    def __init__(self, controller: Any = None):
        """
        Initialize experiment runner.
        
        Args:
            controller: AppController instance (optional, can use engines directly)
        """
        self.controller = controller
        logger.info("Initialized ExperimentRunner")
    
    def _merge_parameters_into_config(
        self,
        base_config: Dict[str, Any],
        parameter_values: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge parameter values into base config.
        
        Handles nested config structures (e.g. variogram_model.nugget).
        
        Args:
            base_config: Base configuration dict
            parameter_values: Parameter values to merge
        
        Returns:
            Merged configuration dict
        """
        config = copy.deepcopy(base_config)
        
        for param_name, param_value in parameter_values.items():
            # Handle nested paths (e.g. "variogram_model.nugget")
            if '.' in param_name:
                parts = param_name.split('.')
                target = config
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = param_value
            else:
                config[param_name] = param_value
        
        return config
    
    def _run_geostats_pipeline(
        self,
        instance: ExperimentInstance,
        base_config: Dict[str, Any],
        pipelines: List[str]
    ) -> Dict[str, Any]:
        """
        Run geostatistical pipeline (OK/UK/CoK/IK + LOOCV).
        
        Args:
            instance: Experiment instance
            base_config: Base configuration
            pipelines: List of pipelines to run (e.g. ["ok", "loocv"])
        
        Returns:
            Results dict with estimates, samples, metrics
        """
        config = self._merge_parameters_into_config(base_config, instance.parameter_values)
        
        results = {
            'instance_index': instance.index,
            'parameter_values': instance.parameter_values
        }
        
        # Set random seed if provided
        if instance.seed is not None:
            import numpy as np
            np.random.seed(instance.seed)
        
        # Run kriging if requested
        if 'ok' in pipelines or 'uk' in pipelines or 'cok' in pipelines or 'ik' in pipelines:
            # This would call controller methods or engines directly
            # For now, return placeholder
            results['kriging_complete'] = True
        
        # Run LOOCV if requested
        if 'loocv' in pipelines:
            # This would run cross-validation
            # For now, return placeholder
            results['loocv_complete'] = True
        
        return results
    
    def _run_planning_pipeline(
        self,
        instance: ExperimentInstance,
        base_config: Dict[str, Any],
        pipelines: List[str]
    ) -> Dict[str, Any]:
        """
        Run planning pipeline (pit optimization + NPV/IRR).
        
        Args:
            instance: Experiment instance
            base_config: Base configuration
            pipelines: List of pipelines to run (e.g. ["pit_optimization", "npv"])
        
        Returns:
            Results dict with pit, NPV, IRR, metrics
        """
        config = self._merge_parameters_into_config(base_config, instance.parameter_values)
        
        results = {
            'instance_index': instance.index,
            'parameter_values': instance.parameter_values
        }
        
        # Set random seed if provided
        if instance.seed is not None:
            import numpy as np
            np.random.seed(instance.seed)
        
        # Run pit optimization if requested
        if 'pit_optimization' in pipelines or 'pit' in pipelines:
            # This would call pit optimizer
            results['pit_complete'] = True
        
        # Run NPV/IRR if requested
        if 'npv' in pipelines or 'irr' in pipelines:
            # This would compute NPV/IRR
            results['npv_irr_complete'] = True
        
        return results
    
    def _run_uncertainty_pipeline(
        self,
        instance: ExperimentInstance,
        base_config: Dict[str, Any],
        pipelines: List[str]
    ) -> Dict[str, Any]:
        """
        Run uncertainty pipeline (SGSIM + economic uncertainty).
        
        Args:
            instance: Experiment instance
            base_config: Base configuration
            pipelines: List of pipelines to run (e.g. ["sgsim", "economic_uncertainty"])
        
        Returns:
            Results dict with realisations, NPV/IRR distributions, metrics
        """
        config = self._merge_parameters_into_config(base_config, instance.parameter_values)
        
        results = {
            'instance_index': instance.index,
            'parameter_values': instance.parameter_values
        }
        
        # Set random seed if provided
        if instance.seed is not None:
            import numpy as np
            np.random.seed(instance.seed)
        
        # Run SGSIM if requested
        if 'sgsim' in pipelines or 'ik_sgsim' in pipelines or 'cosgsim' in pipelines:
            # This would run simulation
            results['simulation_complete'] = True
        
        # Run economic uncertainty if requested
        if 'economic_uncertainty' in pipelines or 'economic' in pipelines:
            # This would run economic propagation
            results['economic_uncertainty_complete'] = True
        
        return results
    
    def run_instance(
        self,
        instance: ExperimentInstance,
        definition: Any  # ExperimentDefinition
    ) -> Dict[str, Any]:
        """
        Run a single experiment instance.
        
        Args:
            instance: Experiment instance to run
            definition: Experiment definition
        
        Returns:
            Results dict with outputs and metrics
        """
        logger.info(f"Running instance {instance.index} of experiment {definition.id}")
        
        # Determine pipeline based on domain
        domain = definition.domain
        pipelines = definition.pipelines
        
        if domain == 'geostats':
            result = self._run_geostats_pipeline(instance, definition.base_config, pipelines)
        elif domain == 'planning':
            result = self._run_planning_pipeline(instance, definition.base_config, pipelines)
        elif domain == 'uncertainty':
            result = self._run_uncertainty_pipeline(instance, definition.base_config, pipelines)
        else:
            logger.warning(f"Unknown domain: {domain}, using generic pipeline")
            result = {
                'instance_index': instance.index,
                'parameter_values': instance.parameter_values
            }
        
        # Compute metrics if requested
        if definition.metrics:
            context = {
                'samples': result.get('samples'),
                'estimates': result.get('estimates'),
                'original': result.get('original'),
                'estimate': result.get('estimate'),
                'npv_samples': result.get('npv_samples'),
                'irr_samples': result.get('irr_samples'),
                'economic_result': result.get('economic_result'),
                'risk_profile': result.get('risk_profile'),
                'variogram_model': result.get('variogram_model'),
                'gt_curve_ref': result.get('gt_curve_ref'),
                'gt_curve_est': result.get('gt_curve_est')
            }
            
            metrics = compute_metrics(definition.metrics, context)
            result['metrics'] = metrics
        
        return result
    
    def run_scenario_grid(self, grid: ScenarioGrid) -> ExperimentRunResult:
        """
        Run all instances in a scenario grid.
        
        Args:
            grid: ScenarioGrid to run
        
        Returns:
            ExperimentRunResult with all instance results
        """
        logger.info(f"Running scenario grid: {len(grid.instances)} instances")
        
        results = []
        
        for instance in grid.instances:
            try:
                result = self.run_instance(instance, grid.definition)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run instance {instance.index}: {e}", exc_info=True)
                # Add error result
                results.append({
                    'instance_index': instance.index,
                    'parameter_values': instance.parameter_values,
                    'error': str(e),
                    'metrics': {}
                })
        
        logger.info(f"Completed scenario grid: {len(results)}/{len(grid.instances)} instances")
        
        return ExperimentRunResult(
            definition_id=grid.definition.id,
            results=results,
            metrics=grid.definition.metrics,
            metadata={
                'n_instances': len(results),
                'n_successful': sum(1 for r in results if 'error' not in r),
                'definition_name': grid.definition.name
            }
        )


def run_scenario_grid_job(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run scenario grid job (wrapper for controller integration).
    
    Args:
        params: Job parameters dict with:
            - 'grid': ScenarioGrid dict
            - 'controller': AppController instance (optional)
    
    Returns:
        ExperimentRunResult dict
    """
    from .experiment_definitions import ScenarioGrid, ExperimentDefinition
    
    # Reconstruct ScenarioGrid from dict
    grid_dict = params.get('grid')
    if not grid_dict:
        raise ValueError("'grid' parameter required")
    
    # Reconstruct ExperimentDefinition
    def_dict = grid_dict.get('definition', {})
    definition = ExperimentDefinition(
        id=def_dict.get('id', ''),
        name=def_dict.get('name', ''),
        description=def_dict.get('description', ''),
        domain=def_dict.get('domain', 'geostats'),
        parameters=[],  # Would need to reconstruct ExperimentParameter objects
        base_config=def_dict.get('base_config', {}),
        metrics=def_dict.get('metrics', []),
        pipelines=def_dict.get('pipelines', [])
    )
    
    # Reconstruct instances
    instances = []
    for inst_dict in grid_dict.get('instances', []):
        from .experiment_definitions import ExperimentInstance
        instance = ExperimentInstance(
            definition_id=inst_dict.get('definition_id', ''),
            index=inst_dict.get('index', 0),
            parameter_values=inst_dict.get('parameter_values', {}),
            seed=inst_dict.get('seed'),
            metadata=inst_dict.get('metadata', {})
        )
        instances.append(instance)
    
    grid = ScenarioGrid(definition=definition, instances=instances)
    
    # Get controller
    controller = params.get('controller')
    
    # Create runner
    runner = ExperimentRunner(controller=controller)
    
    # Run grid
    result = runner.run_scenario_grid(grid)
    
    # Convert to dict
    return {
        'definition_id': result.definition_id,
        'results': result.results,
        'metrics': result.metrics,
        'metadata': result.metadata
    }

