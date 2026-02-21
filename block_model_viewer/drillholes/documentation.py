"""
Documentation Generator

Provides automated documentation generation for the QC/Editor system.
Generates API docs, user guides, and system documentation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime
from pathlib import Path
import logging
import inspect

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation."""
    API_REFERENCE = "api_reference"
    USER_GUIDE = "user_guide"
    SYSTEM_OVERVIEW = "system_overview"
    COMPONENT_DOCS = "component_docs"


@dataclass
class DocumentationSection:
    """A section of documentation."""
    title: str
    content: str
    subsections: List['DocumentationSection'] = field(default_factory=list)
    code_examples: List[str] = field(default_factory=list)


class DocumentationGenerator:
    """
    Documentation generator.
    
    Generates comprehensive documentation for the system.
    """
    
    def __init__(self):
        logger.info("DocumentationGenerator initialized")
    
    def generate_api_reference(
        self,
        modules: List[str],
        output_path: Path,
    ) -> bool:
        """
        Generate API reference documentation.
        
        Args:
            modules: List of module names to document
            output_path: Path to output file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            content = "# API Reference\n\n"
            content += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for module_name in modules:
                try:
                    module = __import__(module_name, fromlist=[''])
                    content += f"## {module_name}\n\n"
                    
                    # Document classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if obj.__module__ == module_name:
                            content += self._document_class(name, obj)
                    
                    # Document functions
                    for name, obj in inspect.getmembers(module, inspect.isfunction):
                        if obj.__module__ == module_name:
                            content += self._document_function(name, obj)
                    
                    content += "\n"
                    
                except Exception as e:
                    logger.warning(f"Error documenting module {module_name}: {e}")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated API reference: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating API reference: {e}", exc_info=True)
            return False
    
    def _document_class(self, name: str, cls: type) -> str:
        """Document a class."""
        doc = f"### {name}\n\n"
        
        if cls.__doc__:
            doc += f"{cls.__doc__}\n\n"
        
        # Document methods
        methods = inspect.getmembers(cls, predicate=inspect.isfunction)
        if methods:
            doc += "**Methods:**\n\n"
            for method_name, method in methods:
                if not method_name.startswith('_'):
                    doc += f"- `{method_name}()`\n"
                    if method.__doc__:
                        doc += f"  {method.__doc__.split(chr(10))[0]}\n"
            doc += "\n"
        
        return doc
    
    def _document_function(self, name: str, func: callable) -> str:
        """Document a function."""
        doc = f"### {name}\n\n"
        
        if func.__doc__:
            doc += f"{func.__doc__}\n\n"
        
        # Document parameters
        sig = inspect.signature(func)
        if sig.parameters:
            doc += "**Parameters:**\n\n"
            for param_name, param in sig.parameters.items():
                doc += f"- `{param_name}`: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}\n"
            doc += "\n"
        
        return doc
    
    def generate_system_overview(self, output_path: Path) -> bool:
        """
        Generate system overview documentation.
        
        Args:
            output_path: Path to output file
        
        Returns:
            True if successful, False otherwise
        """
        try:
            content = """# QC/Editor System Overview

## Introduction

This is a comprehensive enterprise-grade Quality Control (QC) and Editor system for drillhole data management.

## System Architecture

### Core Components

1. **QC Engine** - Performs quality control checks on drillhole data
2. **Editor** - Provides in-memory editing with undo/redo capabilities
3. **Bulk Fixer** - Applies automated fixes to QC issues
4. **Approval Workflow** - Manages approval processes for manual fixes
5. **Control Sample Management** - Tracks and validates control samples
6. **Analytics Engine** - Provides trend analysis and statistics
7. **Reporting System** - Generates comprehensive reports
8. **Audit Trail** - Tracks all system actions

### Workflow

1. Load drillhole data
2. Run QC analysis
3. Review QC results
4. Apply fixes (auto or manual)
5. Request approvals (if required)
6. Save changes
7. Generate reports

## Features

- Real-time validation
- Bulk operations
- Advanced filtering
- Workflow automation
- Performance monitoring
- Comprehensive audit trail

## Compliance

- JORC compliant
- SAMREC compliant
- Industry standard practices

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Generated system overview: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating system overview: {e}", exc_info=True)
            return False


# Global documentation generator instance
_doc_generator: Optional[DocumentationGenerator] = None


def get_documentation_generator() -> DocumentationGenerator:
    """Get the global documentation generator instance."""
    global _doc_generator
    if _doc_generator is None:
        _doc_generator = DocumentationGenerator()
    return _doc_generator

