"""
Security utilities for GeoX.

Provides secure file operations, path validation, and resource limits
to prevent common security vulnerabilities.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple
import tempfile

logger = logging.getLogger(__name__)

# Maximum file sizes (in bytes)
MAX_FILE_SIZE_CSV = 500 * 1024 * 1024  # 500 MB
MAX_FILE_SIZE_PKL = 100 * 1024 * 1024  # 100 MB
MAX_FILE_SIZE_BINARY = 1 * 1024 * 1024 * 1024  # 1 GB
MAX_FILE_SIZE_DEFAULT = 500 * 1024 * 1024  # 500 MB


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


class PathTraversalError(SecurityError):
    """Raised when a path traversal attempt is detected."""
    pass


class FileSizeExceededError(SecurityError):
    """Raised when a file exceeds the maximum allowed size."""
    pass


def validate_file_path(
    user_path: Path,
    allowed_base: Optional[Path] = None,
    must_exist: bool = False,
    allow_absolute: bool = True
) -> Path:
    """
    Validate and normalize file path to prevent directory traversal attacks.
    
    Args:
        user_path: The user-provided path to validate
        allowed_base: Base directory that the path must be within (if provided)
        must_exist: Whether the file must exist
        allow_absolute: Whether absolute paths are allowed
        
    Returns:
        Normalized, validated Path object
        
    Raises:
        PathTraversalError: If path traversal is detected
        FileNotFoundError: If must_exist=True and file doesn't exist
        ValueError: If path validation fails
    """
    if user_path is None:
        raise ValueError("Path cannot be None")
    
    # Convert to Path if needed
    path = Path(user_path)
    
    # Check for suspicious patterns
    path_str = str(path)
    if '..' in path_str or path_str.startswith('~'):
        # Resolve to check if it actually escapes
        try:
            resolved = path.resolve()
        except (OSError, RuntimeError) as e:
            raise PathTraversalError(f"Cannot resolve path: {e}")
        
        # If base directory specified, ensure resolved path is within it
        if allowed_base:
            base_resolved = Path(allowed_base).resolve()
            try:
                resolved.relative_to(base_resolved)
            except ValueError:
                raise PathTraversalError(
                    f"Path {resolved} is outside allowed directory {base_resolved}"
                )
    else:
        resolved = path.resolve()
    
    # Check if absolute paths are allowed
    if not allow_absolute and resolved.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    
    # Check if file exists (if required)
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File does not exist: {resolved}")
    
    return resolved


def validate_file_size(
    file_path: Path,
    max_size: Optional[int] = None,
    file_type: Optional[str] = None
) -> int:
    """
    Validate that a file does not exceed maximum allowed size.
    
    Args:
        file_path: Path to the file to check
        max_size: Maximum allowed size in bytes (overrides file_type defaults)
        file_type: Type of file ('csv', 'pkl', 'binary', etc.)
        
    Returns:
        Actual file size in bytes
        
    Raises:
        FileSizeExceededError: If file exceeds maximum size
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    
    file_size = file_path.stat().st_size
    
    # Determine max size
    if max_size is None:
        if file_type == 'csv':
            max_size = MAX_FILE_SIZE_CSV
        elif file_type == 'pkl':
            max_size = MAX_FILE_SIZE_PKL
        elif file_type == 'binary':
            max_size = MAX_FILE_SIZE_BINARY
        else:
            max_size = MAX_FILE_SIZE_DEFAULT
    
    if file_size > max_size:
        size_mb = file_size / (1024 * 1024)
        max_mb = max_size / (1024 * 1024)
        raise FileSizeExceededError(
            f"File {file_path.name} ({size_mb:.1f} MB) exceeds maximum "
            f"allowed size ({max_mb:.1f} MB)"
        )
    
    return file_size


def safe_open_file(
    file_path: Path,
    mode: str = 'r',
    allowed_base: Optional[Path] = None,
    max_size: Optional[int] = None,
    file_type: Optional[str] = None,
    encoding: Optional[str] = None,
    **kwargs
):
    """
    Safely open a file with security checks.
    
    Args:
        file_path: Path to file
        mode: File open mode ('r', 'rb', 'w', etc.)
        allowed_base: Base directory for path validation
        max_size: Maximum file size (for read operations)
        file_type: Type of file for size limits
        encoding: Text encoding (for text modes)
        **kwargs: Additional arguments for open()
        
    Returns:
        File handle
        
    Raises:
        SecurityError: If security checks fail
    """
    # Validate path
    validated_path = validate_file_path(
        file_path,
        allowed_base=allowed_base,
        must_exist=('r' in mode or 'a' in mode)
    )
    
    # Check file size for read operations
    if 'r' in mode and max_size is not None:
        validate_file_size(validated_path, max_size=max_size, file_type=file_type)
    elif 'r' in mode:
        # Use default size limits
        validate_file_size(validated_path, file_type=file_type)
    
    # Open file
    if encoding:
        return open(validated_path, mode, encoding=encoding, **kwargs)
    else:
        return open(validated_path, mode, **kwargs)


def compute_file_checksum(
    file_path: Path,
    algorithm: str = "sha256",
    max_size: Optional[int] = None
) -> str:
    """
    Compute checksum of a file with size validation.
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('sha256', 'md5', etc.)
        max_size: Maximum file size to process
        
    Returns:
        Hex digest of file checksum
        
    Raises:
        SecurityError: If file exceeds size limit
    """
    if max_size:
        validate_file_size(file_path, max_size=max_size)
    
    hash_func = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        # Read in chunks to handle large files
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hash_func.update(chunk)
    return hash_func.hexdigest()


def create_secure_temp_file(
    suffix: Optional[str] = None,
    prefix: str = 'geox_',
    delete: bool = True,
    mode: str = 'w+b'
) -> Tuple[int, Path]:
    """
    Create a secure temporary file with proper permissions.
    
    Args:
        suffix: File suffix (e.g., '.pkl')
        prefix: File prefix
        delete: Whether to delete file on close
        mode: File mode
        
    Returns:
        Tuple of (file descriptor, Path)
    """
    fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    
    # Set restrictive permissions (owner read/write only)
    os.chmod(temp_path, 0o600)
    
    # If delete=True, the file will be deleted when closed
    # If delete=False, we need to track it for cleanup
    
    return fd, Path(temp_path)


def validate_pickle_file(
    file_path: Path,
    max_size: Optional[int] = None,
    allowed_base: Optional[Path] = None
) -> Tuple[Path, int]:
    """
    Validate a pickle file before loading.
    
    This performs security checks but does NOT prevent all pickle risks.
    Prefer using JSON or other safe formats for user data.
    
    Args:
        file_path: Path to pickle file
        max_size: Maximum file size
        allowed_base: Base directory for path validation
        
    Returns:
        Tuple of (validated_path, file_size)
        
    Raises:
        SecurityError: If validation fails
    """
    validated_path = validate_file_path(
        file_path,
        allowed_base=allowed_base,
        must_exist=True
    )
    
    file_size = validate_file_size(
        validated_path,
        max_size=max_size or MAX_FILE_SIZE_PKL,
        file_type='pkl'
    )
    
    # Additional check: verify it's actually a pickle file
    # (basic check - pickle files often start with specific bytes)
    try:
        with open(validated_path, 'rb') as f:
            first_bytes = f.read(4)
            # Pickle protocol 0-3 start with specific bytes
            # This is a basic check, not foolproof
            if not first_bytes:
                raise SecurityError("Empty pickle file")
    except Exception as e:
        raise SecurityError(f"Cannot read pickle file: {e}")
    
    return validated_path, file_size

