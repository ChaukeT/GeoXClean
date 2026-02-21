"""
Scenario Store (STEP 31)

In-memory and file-based storage for planning scenarios.
"""

from pathlib import Path
from typing import List, Optional, Dict
import json
import logging
from datetime import datetime

from .scenario_definition import PlanningScenario, ScenarioID

logger = logging.getLogger(__name__)


class ScenarioStore:
    """
    Store and retrieve planning scenarios with JSON serialization.
    """
    
    def __init__(self, base_path: Path):
        """
        Initialize scenario store.
        
        Args:
            base_path: Base directory for scenario storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, PlanningScenario] = {}
        
        logger.info(f"Initialized ScenarioStore at {self.base_path}")
    
    def list_scenarios(self) -> List[PlanningScenario]:
        """
        List all scenarios.
        
        Returns:
            List of PlanningScenario
        """
        scenarios = []
        
        # Load from disk if cache is empty
        if not self._cache:
            self._load_all_from_disk()
        
        scenarios = list(self._cache.values())
        
        # Sort by modified_at (most recent first)
        scenarios.sort(key=lambda s: s.modified_at, reverse=True)
        
        return scenarios
    
    def get(self, name: str, version: str = "latest") -> Optional[PlanningScenario]:
        """
        Get a scenario by name and version.
        
        Args:
            name: Scenario name
            version: Version string ("latest" gets most recent)
            
        Returns:
            PlanningScenario or None if not found
        """
        if version == "latest":
            # Find all versions of this scenario
            scenario_dir = self.base_path / name
            if not scenario_dir.exists():
                return None
            
            versions = []
            for version_file in scenario_dir.glob("*.json"):
                version_name = version_file.stem
                try:
                    scenario = self._load_from_file(version_file)
                    if scenario:
                        versions.append((scenario.modified_at, scenario))
                except Exception as e:
                    logger.warning(f"Failed to load {version_file}: {e}")
            
            if versions:
                versions.sort(key=lambda x: x[0], reverse=True)
                return versions[0][1]
            return None
        
        # Load specific version
        cache_key = f"{name}_{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        scenario_file = self.base_path / name / f"{version}.json"
        if scenario_file.exists():
            scenario = self._load_from_file(scenario_file)
            if scenario:
                self._cache[cache_key] = scenario
                return scenario
        
        return None
    
    def save(self, scenario: PlanningScenario) -> None:
        """
        Save a scenario to disk.
        
        Args:
            scenario: PlanningScenario to save
        """
        scenario_dir = self.base_path / scenario.id.name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        
        scenario_file = scenario_dir / f"{scenario.id.version}.json"
        
        # Update modified_at
        scenario.modified_at = datetime.now()
        
        # Convert to dict and serialize
        scenario_dict = scenario.to_dict()
        
        try:
            with open(scenario_file, 'w') as f:
                json.dump(scenario_dict, f, indent=2, default=str)
            
            # Update cache
            cache_key = f"{scenario.id.name}_{scenario.id.version}"
            self._cache[cache_key] = scenario
            
            logger.info(f"Saved scenario {scenario.id.name} v{scenario.id.version}")
        
        except Exception as e:
            logger.error(f"Failed to save scenario {scenario.id.name}: {e}", exc_info=True)
            raise
    
    def delete(self, name: str, version: str) -> None:
        """
        Delete a scenario.
        
        Args:
            name: Scenario name
            version: Version string
        """
        scenario_file = self.base_path / name / f"{version}.json"
        
        if scenario_file.exists():
            try:
                scenario_file.unlink()
                
                # Remove from cache
                cache_key = f"{name}_{version}"
                if cache_key in self._cache:
                    del self._cache[cache_key]
                
                logger.info(f"Deleted scenario {name} v{version}")
            
            except Exception as e:
                logger.error(f"Failed to delete scenario {name} v{version}: {e}", exc_info=True)
                raise
    
    def _load_all_from_disk(self) -> None:
        """Load all scenarios from disk into cache."""
        if not self.base_path.exists():
            return
        
        for scenario_dir in self.base_path.iterdir():
            if not scenario_dir.is_dir():
                continue
            
            for scenario_file in scenario_dir.glob("*.json"):
                try:
                    scenario = self._load_from_file(scenario_file)
                    if scenario:
                        cache_key = f"{scenario.id.name}_{scenario.id.version}"
                        self._cache[cache_key] = scenario
                except Exception as e:
                    logger.warning(f"Failed to load {scenario_file}: {e}")
    
    def _load_from_file(self, scenario_file: Path) -> Optional[PlanningScenario]:
        """
        Load a scenario from a JSON file.
        
        Args:
            scenario_file: Path to JSON file
            
        Returns:
            PlanningScenario or None if failed
        """
        try:
            with open(scenario_file, 'r') as f:
                data = json.load(f)
            
            scenario = PlanningScenario.from_dict(data)
            return scenario
        
        except Exception as e:
            logger.error(f"Failed to load scenario from {scenario_file}: {e}", exc_info=True)
            return None

