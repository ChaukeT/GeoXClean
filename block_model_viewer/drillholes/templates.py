"""
Export/Import Templates System

Provides standardized templates for data import/export.
Supports custom templates, validation, and format conversion.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from pathlib import Path
from datetime import datetime
import logging
import json
import csv

logger = logging.getLogger(__name__)


class TemplateType(Enum):
    """Types of templates."""
    IMPORT = "import"
    EXPORT = "export"
    VALIDATION = "validation"


@dataclass
class FieldMapping:
    """Mapping between source and target fields."""
    source_field: str
    target_field: str
    data_type: str = "string"
    required: bool = False
    default_value: Any = None
    transform: Optional[Callable[[Any], Any]] = None


@dataclass
class DataTemplate:
    """
    A data template for import/export.
    
    Defines field mappings, validation rules, and format specifications.
    """
    name: str
    template_type: TemplateType
    description: str = ""
    version: str = "1.0"
    created_date: datetime = field(default_factory=datetime.now)
    field_mappings: List[FieldMapping] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    format_spec: Dict[str, Any] = field(default_factory=dict)
    
    def validate_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate data against template rules.
        
        Args:
            data: Data dictionary to validate
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        for mapping in self.field_mappings:
            if mapping.required:
                if mapping.source_field not in data and mapping.target_field not in data:
                    errors.append(f"Required field missing: {mapping.source_field}")
        
        # Check data types
        for mapping in self.field_mappings:
            if mapping.source_field in data:
                value = data[mapping.source_field]
                if value is not None:
                    try:
                        if mapping.data_type == "float":
                            float(value)
                        elif mapping.data_type == "int":
                            int(value)
                        elif mapping.data_type == "date":
                            datetime.fromisoformat(str(value))
                    except (ValueError, TypeError):
                        errors.append(
                            f"Field {mapping.source_field} has invalid type "
                            f"(expected {mapping.data_type})"
                        )
        
        return len(errors) == 0, errors
    
    def transform_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform data using field mappings.
        
        Args:
            data: Source data dictionary
        
        Returns:
            Transformed data dictionary
        """
        transformed = {}
        
        for mapping in self.field_mappings:
            source_value = data.get(mapping.source_field)
            
            if source_value is None and mapping.default_value is not None:
                source_value = mapping.default_value
            
            if source_value is not None:
                # Apply transform if available
                if mapping.transform:
                    try:
                        source_value = mapping.transform(source_value)
                    except Exception as e:
                        logger.warning(f"Transform failed for {mapping.source_field}: {e}")
                
                transformed[mapping.target_field] = source_value
        
        return transformed


class TemplateManager:
    """
    Manages data templates for import/export.
    
    Provides template creation, storage, and application.
    """
    
    def __init__(self):
        self.templates: Dict[str, DataTemplate] = {}
        self._load_default_templates()
        logger.info("TemplateManager initialized")
    
    def _load_default_templates(self):
        """Load default templates."""
        # Default assay import template
        assay_import = DataTemplate(
            name="Assay Import (Standard)",
            template_type=TemplateType.IMPORT,
            description="Standard template for importing assay data",
            field_mappings=[
                FieldMapping("hole_id", "hole_id", "string", required=True),
                FieldMapping("depth_from", "depth_from", "float", required=True),
                FieldMapping("depth_to", "depth_to", "float", required=True),
                FieldMapping("Fe", "Fe", "float", required=False),
                FieldMapping("Au", "Au", "float", required=False),
                FieldMapping("Cu", "Cu", "float", required=False),
            ],
        )
        self.templates[assay_import.name] = assay_import
        
        # Default collar import template
        collar_import = DataTemplate(
            name="Collar Import (Standard)",
            template_type=TemplateType.IMPORT,
            description="Standard template for importing collar data",
            field_mappings=[
                FieldMapping("hole_id", "hole_id", "string", required=True),
                FieldMapping("x", "x", "float", required=True),
                FieldMapping("y", "y", "float", required=True),
                FieldMapping("z", "z", "float", required=True),
                FieldMapping("azimuth", "azimuth", "float", required=False),
                FieldMapping("dip", "dip", "float", required=False),
                FieldMapping("length", "length", "float", required=False),
            ],
        )
        self.templates[collar_import.name] = collar_import
        
        logger.info(f"Loaded {len(self.templates)} default templates")
    
    def add_template(self, template: DataTemplate) -> None:
        """Add a template."""
        self.templates[template.name] = template
        logger.info(f"Added template: {template.name}")
    
    def get_template(self, name: str) -> Optional[DataTemplate]:
        """Get a template by name."""
        return self.templates.get(name)
    
    def list_templates(self, template_type: Optional[TemplateType] = None) -> List[DataTemplate]:
        """List all templates, optionally filtered by type."""
        if template_type:
            return [t for t in self.templates.values() if t.template_type == template_type]
        return list(self.templates.values())
    
    def save_template(self, template: DataTemplate, file_path: Path) -> bool:
        """Save a template to file."""
        try:
            template_dict = {
                "name": template.name,
                "template_type": template.template_type.value,
                "description": template.description,
                "version": template.version,
                "created_date": template.created_date.isoformat(),
                "field_mappings": [
                    {
                        "source_field": m.source_field,
                        "target_field": m.target_field,
                        "data_type": m.data_type,
                        "required": m.required,
                        "default_value": m.default_value,
                    }
                    for m in template.field_mappings
                ],
                "validation_rules": template.validation_rules,
                "format_spec": template.format_spec,
            }
            
            with open(file_path, 'w') as f:
                json.dump(template_dict, f, indent=2)
            
            logger.info(f"Saved template to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving template: {e}", exc_info=True)
            return False
    
    def load_template(self, file_path: Path) -> Optional[DataTemplate]:
        """Load a template from file."""
        try:
            with open(file_path, 'r') as f:
                template_dict = json.load(f)
            
            field_mappings = [
                FieldMapping(
                    source_field=m["source_field"],
                    target_field=m["target_field"],
                    data_type=m.get("data_type", "string"),
                    required=m.get("required", False),
                    default_value=m.get("default_value"),
                )
                for m in template_dict.get("field_mappings", [])
            ]
            
            template = DataTemplate(
                name=template_dict["name"],
                template_type=TemplateType(template_dict["template_type"]),
                description=template_dict.get("description", ""),
                version=template_dict.get("version", "1.0"),
                created_date=datetime.fromisoformat(template_dict.get("created_date", datetime.now().isoformat())),
                field_mappings=field_mappings,
                validation_rules=template_dict.get("validation_rules", {}),
                format_spec=template_dict.get("format_spec", {}),
            )
            
            logger.info(f"Loaded template from {file_path}")
            return template
            
        except Exception as e:
            logger.error(f"Error loading template: {e}", exc_info=True)
            return None
    
    def apply_template(
        self,
        template_name: str,
        data: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Apply a template to transform data.
        
        Args:
            template_name: Name of template to apply
            data: List of data dictionaries to transform
        
        Returns:
            Tuple of (transformed_data, list_of_errors)
        """
        template = self.get_template(template_name)
        if not template:
            return [], [f"Template not found: {template_name}"]
        
        transformed = []
        all_errors = []
        
        for i, row in enumerate(data):
            is_valid, errors = template.validate_data(row)
            if not is_valid:
                all_errors.extend([f"Row {i+1}: {e}" for e in errors])
                continue
            
            transformed_row = template.transform_data(row)
            transformed.append(transformed_row)
        
        logger.info(f"Applied template {template_name} to {len(data)} rows, "
                   f"{len(transformed)} successful, {len(all_errors)} errors")
        
        return transformed, all_errors


# Global template manager instance
_template_manager: Optional[TemplateManager] = None


def get_template_manager() -> TemplateManager:
    """Get the global template manager instance."""
    global _template_manager
    if _template_manager is None:
        _template_manager = TemplateManager()
    return _template_manager

