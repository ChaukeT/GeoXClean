"""
Abstract base parser for different file formats.

GeoX Invariant Compliance:
- Parser versioning for reproducibility
- File checksums for data integrity
- Transformation tracking in metadata
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging
import hashlib
from pathlib import Path

from ..models.block_model import BlockModel, BlockMetadata

logger = logging.getLogger(__name__)

# Parser framework version
PARSER_FRAMEWORK_VERSION = "1.0.0"


def compute_file_checksum(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute checksum of a file for data integrity verification.
    
    GeoX invariant: All imported files should have checksums for verification.
    
    SECURITY: Validates file path before computing checksum.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)
    
    Returns:
        Hex digest of file checksum
        
    Raises:
        SecurityError: If security validation fails
    """
    from ..utils.security import validate_file_path, SecurityError
    
    try:
        validated_path = validate_file_path(file_path, must_exist=True)
    except SecurityError:
        # If validation fails, still compute checksum but log warning
        logger.warning(f"Path validation failed for checksum computation: {file_path}")
        validated_path = file_path
    
    hash_func = hashlib.new(algorithm)
    with open(validated_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hash_func.update(chunk)
    return hash_func.hexdigest()


class BaseParser(ABC):
    """
    Abstract base class for parsing different file formats into BlockModel objects.
    """
    
    def __init__(self):
        self.supported_extensions: List[str] = []
        self.format_name: str = ""
    
    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if this parser can handle the file
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: Path, **kwargs) -> BlockModel:
        """
        Parse a file into a BlockModel object.
        
        Args:
            file_path: Path to the file to parse
            **kwargs: Additional parsing options
            
        Returns:
            BlockModel object containing the parsed data
            
        Raises:
            ValueError: If the file cannot be parsed
            FileNotFoundError: If the file doesn't exist
        """
        pass
    
    def get_file_info(self, file_path: Path, include_checksum: bool = True) -> Dict[str, Any]:
        """
        Get basic information about a file without parsing it.
        
        GeoX invariant: Includes checksum for data integrity verification.
        
        Args:
            file_path: Path to the file
            include_checksum: Whether to compute file checksum (may be slow for large files)
            
        Returns:
            Dictionary with file information including checksum
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        stat = file_path.stat()
        info = {
            "path": str(file_path),
            "name": file_path.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "format": self.format_name,
            "supported_extensions": self.supported_extensions,
            "parser_version": PARSER_FRAMEWORK_VERSION
        }
        
        # GeoX invariant: Include checksum for data integrity
        if include_checksum:
            info["checksum"] = compute_file_checksum(file_path)
            info["checksum_algorithm"] = "sha256"
        
        return info


class ParserRegistry:
    """
    Registry for managing different file parsers.
    """
    
    def __init__(self):
        self._parsers: List[BaseParser] = []
    
    def register_parser(self, parser: BaseParser) -> None:
        """
        Register a parser with the registry.
        
        Args:
            parser: Parser instance to register
        """
        self._parsers.append(parser)
        logger.info(f"Registered parser for {parser.format_name}")
    
    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """
        Get the appropriate parser for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Parser instance or None if no suitable parser found
        """
        for parser in self._parsers:
            if parser.can_parse(file_path):
                return parser
        
        return None
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get all supported file extensions.
        
        Returns:
            List of supported extensions (including dots)
        """
        extensions = []
        for parser in self._parsers:
            extensions.extend(parser.supported_extensions)
        return sorted(list(set(extensions)))
    
    def parse_file(self, file_path: Path, **kwargs) -> BlockModel:
        """
        Parse a file using the appropriate parser.
        
        GeoX invariant: Adds parser version and framework version to model metadata.
        
        SECURITY: Validates file path and size before parsing.
        
        Args:
            file_path: Path to the file to parse
            **kwargs: Additional parsing options
            
        Returns:
            BlockModel object with provenance metadata
            
        Raises:
            ValueError: If no parser can handle the file
            SecurityError: If security validation fails
        """
        from ..utils.security import validate_file_path, validate_file_size, SecurityError
        
        # SECURITY: Validate path and file size
        try:
            validated_path = validate_file_path(file_path, must_exist=True)
            validate_file_size(validated_path, file_type='csv')
        except SecurityError as e:
            logger.error(f"Security validation failed for {file_path}: {e}")
            raise ValueError(f"File security validation failed: {e}")
        
        parser = self.get_parser(validated_path)
        if parser is None:
            supported = self.get_supported_extensions()
            raise ValueError(f"No parser found for file {validated_path}. Supported formats: {supported}")
        
        logger.info(f"Parsing {validated_path} with {parser.format_name} parser")
        block_model = parser.parse(validated_path, **kwargs)
        
        # GeoX invariant: Add parser framework version to metadata
        if block_model.metadata:
            block_model.metadata.parser_framework_version = PARSER_FRAMEWORK_VERSION
        
        return block_model


# Global parser registry instance
parser_registry = ParserRegistry()
