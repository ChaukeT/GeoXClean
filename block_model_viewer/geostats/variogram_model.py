"""
Unified Variogram Model Module - AUTHORITATIVE SOURCE
======================================================

This module provides the CANONICAL variogram model implementations and dataclasses
for the entire GeoX codebase. All estimation engines MUST use these definitions.

AUDIT REQUIREMENT (JORC/SAMREC):
- Single source of truth for variogram functions
- Immutable after fitting unless explicitly re-fitted
- Full lineage tracking from experimental to fitted model

CRITICAL: Do NOT duplicate variogram functions in other modules.
Import from this module instead.

Created: 2025-12-16 as part of variogram subsystem audit remediation.
See: docs/VARIOGRAM_SUBSYSTEM_AUDIT.md
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# VARIOGRAM MODEL TYPES
# =============================================================================

class VariogramModelType(str, Enum):
    """Supported variogram model types."""
    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    POWER = "power"
    NUGGET_ONLY = "nugget"


# =============================================================================
# CANONICAL VARIOGRAM FUNCTIONS - GSLIB Practical Range Convention
# =============================================================================
# ALL functions use signature: func(h, range_, sill, nugget) -> gamma
# This is the GSLIB standard signature. Do NOT change parameter order.

def spherical_model(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
    """
    Spherical variogram model (GSLIB convention).
    
    gamma(h) = c0 + c * (1.5(h/a) - 0.5(h/a)^3) if h < a
             = c0 + c                           otherwise
    
    Parameters
    ----------
    h : np.ndarray
        Lag distances
    range_ : float
        Practical range (distance where sill is reached)
    sill : float
        Total sill = nugget + partial_sill
    nugget : float
        Nugget effect (C0)
    
    Returns
    -------
    np.ndarray
        Semivariance values
    """
    h = np.asarray(h, dtype=float)
    a = max(range_, 1e-12)
    c = max(sill - nugget, 0.0)  # Partial sill
    t = h / a
    gamma = np.where(
        h <= a,
        nugget + c * (1.5 * t - 0.5 * t**3),
        nugget + c,  # = sill
    )
    return gamma


def exponential_model(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
    """
    Exponential variogram model (practical range: 95% sill at h = range_).
    
    gamma(h) = c0 + c * (1 - exp(-3h/a))
    
    Parameters
    ----------
    h : np.ndarray
        Lag distances
    range_ : float
        Practical range (95% of sill)
    sill : float
        Total sill
    nugget : float
        Nugget effect
    
    Returns
    -------
    np.ndarray
        Semivariance values
    """
    h = np.asarray(h, dtype=float)
    a = max(range_, 1e-12)
    c = max(sill - nugget, 0.0)
    return nugget + c * (1.0 - np.exp(-3.0 * h / a))


def gaussian_model(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
    """
    Gaussian variogram model (practical range: 95% sill at h = range_).
    
    gamma(h) = c0 + c * (1 - exp(-3 (h/a)^2))
    
    Parameters
    ----------
    h : np.ndarray
        Lag distances
    range_ : float
        Practical range
    sill : float
        Total sill
    nugget : float
        Nugget effect
    
    Returns
    -------
    np.ndarray
        Semivariance values
    """
    h = np.asarray(h, dtype=float)
    a = max(range_, 1e-12)
    c = max(sill - nugget, 0.0)
    return nugget + c * (1.0 - np.exp(-3.0 * (h / a) ** 2))


def linear_model(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
    """
    Linear variogram model (unbounded, reaches sill at range_).
    
    Parameters
    ----------
    h : np.ndarray
        Lag distances
    range_ : float
        Distance at which model reaches sill
    sill : float
        Sill value
    nugget : float
        Nugget effect
    
    Returns
    -------
    np.ndarray
        Semivariance values
    """
    h = np.asarray(h, dtype=float)
    a = max(range_, 1e-12)
    c = max(sill - nugget, 0.0)
    slope = c / a
    gamma = np.where(
        h <= a,
        nugget + slope * h,
        sill
    )
    return gamma


def nugget_model(h: np.ndarray, range_: float, sill: float, nugget: float = 0.0) -> np.ndarray:
    """
    Pure nugget model (no spatial correlation).
    
    Parameters
    ----------
    h : np.ndarray
        Lag distances
    range_ : float
        Ignored (for API consistency)
    sill : float
        Total sill = nugget for pure nugget model
    nugget : float
        Nugget value
    
    Returns
    -------
    np.ndarray
        Semivariance values (nugget everywhere except h=0)
    """
    h = np.asarray(h, dtype=float)
    gamma = np.where(h > 0, sill, 0.0)
    return gamma


# Model lookup map - CANONICAL reference
MODEL_MAP: Dict[str, Callable] = {
    "spherical": spherical_model,
    "exponential": exponential_model,
    "gaussian": gaussian_model,
    "linear": linear_model,
    "nugget": nugget_model,
}


def get_variogram_function(model_type: str) -> Callable:
    """
    Get canonical variogram function by model type.
    
    Parameters
    ----------
    model_type : str
        Model type name (case-insensitive)
    
    Returns
    -------
    Callable
        Variogram function with signature: func(h, range_, sill, nugget)
    
    Raises
    ------
    ValueError
        If model type is not recognized
    """
    key = model_type.lower().strip()
    if key not in MODEL_MAP:
        valid = ", ".join(MODEL_MAP.keys())
        raise ValueError(f"Unknown variogram model '{model_type}'. Valid types: {valid}")
    return MODEL_MAP[key]


# =============================================================================
# VARIOGRAM STRUCTURE DATACLASS
# =============================================================================

@dataclass
class VariogramStructure:
    """
    Single structure in a (possibly nested) variogram model.
    
    Standard form: γ(h) = contribution * g(h/range)
    where g is the normalized model function.
    """
    model_type: str
    contribution: float  # Partial sill (C_i)
    range_major: float
    range_minor: Optional[float] = None
    range_vertical: Optional[float] = None
    
    def __post_init__(self):
        """Set default ranges for isotropic case."""
        if self.range_minor is None:
            self.range_minor = self.range_major
        if self.range_vertical is None:
            self.range_vertical = self.range_major
        
        # Validation
        if self.contribution < 0:
            raise ValueError(f"Contribution must be >= 0, got {self.contribution}")
        if self.range_major <= 0:
            raise ValueError(f"Range must be > 0, got {self.range_major}")
    
    def evaluate(self, h: np.ndarray, direction: str = "major") -> np.ndarray:
        """
        Evaluate this structure's contribution at distances h.
        
        Parameters
        ----------
        h : np.ndarray
            Distances
        direction : str
            Direction for anisotropic range selection
        
        Returns
        -------
        np.ndarray
            Variogram contribution (without nugget)
        """
        # Select range based on direction
        if direction == "major":
            range_val = self.range_major
        elif direction == "minor":
            range_val = self.range_minor
        elif direction == "vertical":
            range_val = self.range_vertical
        else:
            range_val = self.range_major
        
        # Get model function
        model_func = MODEL_MAP.get(self.model_type, spherical_model)
        
        # Evaluate: contribution is partial sill, so nugget=0 for structure
        return model_func(h, range_val, self.contribution, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export to dictionary."""
        return {
            "model_type": self.model_type,
            "contribution": self.contribution,
            "range_major": self.range_major,
            "range_minor": self.range_minor,
            "range_vertical": self.range_vertical,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariogramStructure":
        """Create from dictionary."""
        return cls(
            model_type=d["model_type"],
            contribution=d["contribution"],
            range_major=d["range_major"],
            range_minor=d.get("range_minor"),
            range_vertical=d.get("range_vertical"),
        )


# =============================================================================
# UNIFIED VARIOGRAM MODEL DATACLASS
# =============================================================================

@dataclass
class VariogramModel:
    """
    Unified variogram model with full lineage tracking.
    
    This is the AUTHORITATIVE representation of a fitted variogram model
    for use in kriging, simulation, and all estimation engines.
    
    Standard form:
    γ(h) = C0 + Σ C_i * g_i(h/a_i)
    
    where C0 = nugget, C_i = partial sills, a_i = ranges, g_i = model functions.
    
    AUDIT REQUIREMENTS:
    - data_hash: SHA-256 hash of source data used for fitting
    - fit_timestamp: When the model was fitted
    - source_dataset_version: Version/ID of source dataset
    - is_immutable: Once True, model cannot be modified
    """
    
    # Core parameters
    nugget: float
    structures: List[VariogramStructure] = field(default_factory=list)
    
    # Anisotropy orientation (degrees)
    azimuth: float = 0.0  # Clockwise from North
    dip: float = 0.0  # Down from horizontal
    plunge: float = 0.0  # Rotation around major axis
    
    # Lineage tracking (AUDIT CRITICAL)
    data_hash: Optional[str] = None
    fit_timestamp: Optional[str] = None
    source_dataset_version: Optional[str] = None
    source_dataset_type: Optional[str] = None  # "composites", "assays", "declustered"
    fitting_method: Optional[str] = None
    
    # Immutability flag
    is_immutable: bool = False
    
    # Validation metadata
    cross_validation_rmse: Optional[float] = None
    fitting_r2: Optional[float] = None
    
    # Model ID for tracking
    model_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate and initialize."""
        if self.nugget < 0:
            raise ValueError(f"Nugget must be >= 0, got {self.nugget}")
        
        # Generate model ID if not provided
        if self.model_id is None:
            self.model_id = self._generate_model_id()
    
    def _generate_model_id(self) -> str:
        """Generate unique model ID based on parameters."""
        content = f"{self.nugget}:{self.total_sill}:{self.n_structures}:{self.primary_range}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
    
    @property
    def total_sill(self) -> float:
        """Total sill (C0 + sum of partial sills)."""
        return self.nugget + sum(s.contribution for s in self.structures)
    
    @property
    def partial_sill(self) -> float:
        """Sum of partial sills (total_sill - nugget)."""
        return sum(s.contribution for s in self.structures)
    
    @property
    def n_structures(self) -> int:
        """Number of nested structures."""
        return len(self.structures)
    
    @property
    def primary_range(self) -> float:
        """Range of the longest-range structure."""
        if not self.structures:
            return 0.0
        return max(s.range_major for s in self.structures)
    
    @property
    def primary_model_type(self) -> str:
        """Model type of the primary (longest-range) structure."""
        if not self.structures:
            return "spherical"
        primary = max(self.structures, key=lambda s: s.range_major)
        return primary.model_type
    
    @property
    def range_major(self) -> float:
        """Major range (alias for primary_range)."""
        return self.primary_range
    
    @property
    def range_minor(self) -> float:
        """Minor range of the primary structure."""
        if not self.structures:
            return 0.0
        primary = max(self.structures, key=lambda s: s.range_major)
        return primary.range_minor or primary.range_major
    
    @property
    def range_vertical(self) -> float:
        """Vertical range of the primary structure."""
        if not self.structures:
            return 0.0
        primary = max(self.structures, key=lambda s: s.range_major)
        return primary.range_vertical or primary.range_major
    
    def evaluate(self, h: np.ndarray, direction: str = "major") -> np.ndarray:
        """
        Evaluate the full variogram model at distances h.
        
        Parameters
        ----------
        h : np.ndarray
            Distances
        direction : str
            Direction for anisotropic evaluation
        
        Returns
        -------
        np.ndarray
            Semivariance values
        """
        h = np.asarray(h, dtype=float)
        result = np.full_like(h, self.nugget)
        
        for struct in self.structures:
            result += struct.evaluate(h, direction)
        
        return result
    
    def covariance(self, h: np.ndarray, direction: str = "major") -> np.ndarray:
        """
        Compute covariance C(h) = sill - gamma(h).
        
        Parameters
        ----------
        h : np.ndarray
            Distances
        direction : str
            Direction for anisotropic evaluation
        
        Returns
        -------
        np.ndarray
            Covariance values
        """
        return self.total_sill - self.evaluate(h, direction)
    
    def validate_for_estimation(self) -> List[str]:
        """
        Validate model for use in kriging/simulation.
        
        Returns
        -------
        List[str]
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check required fields
        if self.nugget < 0:
            errors.append("Nugget cannot be negative")
        
        if not self.structures:
            errors.append("Model must have at least one structure")
        
        for i, struct in enumerate(self.structures):
            if struct.contribution < 0:
                errors.append(f"Structure {i}: contribution cannot be negative")
            if struct.range_major <= 0:
                errors.append(f"Structure {i}: range must be positive")
        
        # Check lineage (audit requirement)
        if self.data_hash is None:
            errors.append("AUDIT: data_hash missing - cannot verify data lineage")
        
        if self.source_dataset_type is None:
            errors.append("AUDIT: source_dataset_type missing - unclear if composites/assays/declustered")
        
        # Check for unreasonable nugget ratio
        if self.total_sill > 0:
            nugget_ratio = self.nugget / self.total_sill
            if nugget_ratio > 0.9:
                errors.append(f"WARNING: Nugget is {nugget_ratio*100:.0f}% of sill - verify this is correct")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export to dictionary for serialization.
        
        This format is used for:
        - Storage in DataRegistry
        - Transmission to kriging/simulation engines
        - Audit trail persistence
        """
        return {
            # Core parameters
            "nugget": self.nugget,
            "total_sill": self.total_sill,
            "structures": [s.to_dict() for s in self.structures],
            
            # Anisotropy
            "azimuth": self.azimuth,
            "dip": self.dip,
            "plunge": self.plunge,
            "range_major": self.range_major,
            "range_minor": self.range_minor,
            "range_vertical": self.range_vertical,
            
            # Lineage (AUDIT CRITICAL)
            "data_hash": self.data_hash,
            "fit_timestamp": self.fit_timestamp,
            "source_dataset_version": self.source_dataset_version,
            "source_dataset_type": self.source_dataset_type,
            "fitting_method": self.fitting_method,
            "is_immutable": self.is_immutable,
            "model_id": self.model_id,
            
            # Validation
            "cross_validation_rmse": self.cross_validation_rmse,
            "fitting_r2": self.fitting_r2,
            
            # Convenience fields for estimation engines
            "model_type": self.primary_model_type,
            "range": self.primary_range,
            "sill": self.total_sill,
        }
    
    def to_kriging_params(self) -> Dict[str, Any]:
        """
        Convert to parameter dict expected by kriging engines.
        
        This provides backward compatibility with existing kriging code
        that expects a simple dict with 'range', 'sill', 'nugget'.
        """
        return {
            "range": self.primary_range,
            "sill": self.total_sill,  # Total sill (nugget + partial)
            "nugget": self.nugget,
            "model_type": self.primary_model_type,
            "anisotropy": {
                "azimuth": self.azimuth,
                "dip": self.dip,
                "major_range": self.range_major,
                "minor_range": self.range_minor,
                "vert_range": self.range_vertical,
            } if self.has_anisotropy else None,
            # Lineage for audit
            "_data_hash": self.data_hash,
            "_model_id": self.model_id,
        }
    
    @property
    def has_anisotropy(self) -> bool:
        """Check if model has geometric anisotropy."""
        if not self.structures:
            return False
        for s in self.structures:
            if s.range_minor != s.range_major or s.range_vertical != s.range_major:
                return True
        return self.azimuth != 0.0 or self.dip != 0.0
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VariogramModel":
        """
        Create VariogramModel from dictionary.
        
        Handles both new format (with structures) and legacy format.
        """
        # Handle structures
        structures = []
        if "structures" in d and d["structures"]:
            for s in d["structures"]:
                structures.append(VariogramStructure.from_dict(s))
        else:
            # Legacy format: single structure
            model_type = d.get("model_type", "spherical")
            sill = d.get("sill", d.get("total_sill", 1.0))
            nugget = d.get("nugget", 0.0)
            range_val = d.get("range", d.get("range_major", 100.0))
            
            # Create single structure with partial sill
            partial_sill = max(sill - nugget, 0.0)
            structures.append(VariogramStructure(
                model_type=model_type,
                contribution=partial_sill,
                range_major=range_val,
                range_minor=d.get("range_minor", range_val),
                range_vertical=d.get("range_vertical", d.get("vert_range", range_val)),
            ))
        
        return cls(
            nugget=d.get("nugget", 0.0),
            structures=structures,
            azimuth=d.get("azimuth", 0.0),
            dip=d.get("dip", 0.0),
            plunge=d.get("plunge", 0.0),
            data_hash=d.get("data_hash"),
            fit_timestamp=d.get("fit_timestamp"),
            source_dataset_version=d.get("source_dataset_version"),
            source_dataset_type=d.get("source_dataset_type"),
            fitting_method=d.get("fitting_method"),
            is_immutable=d.get("is_immutable", False),
            cross_validation_rmse=d.get("cross_validation_rmse"),
            fitting_r2=d.get("fitting_r2"),
            model_id=d.get("model_id"),
        )
    
    @classmethod
    def from_single_fit(
        cls,
        nugget: float,
        sill: float,
        range_: float,
        model_type: str = "spherical",
        data_hash: Optional[str] = None,
        source_dataset_type: Optional[str] = None,
    ) -> "VariogramModel":
        """
        Create model from simple single-structure fit parameters.
        
        This is a convenience constructor for backward compatibility.
        """
        partial_sill = max(sill - nugget, 0.0)
        structure = VariogramStructure(
            model_type=model_type,
            contribution=partial_sill,
            range_major=range_,
        )
        
        return cls(
            nugget=nugget,
            structures=[structure],
            data_hash=data_hash,
            source_dataset_type=source_dataset_type,
            fit_timestamp=datetime.now().isoformat(),
            fitting_method="single_structure_fit",
        )
    
    def lock(self) -> None:
        """
        Lock the model to prevent further modification.
        
        Once locked, the model is considered authoritative and immutable.
        """
        self.is_immutable = True
        logger.info(f"VariogramModel {self.model_id} locked (immutable)")
    
    def copy(self) -> "VariogramModel":
        """Create a mutable copy of this model."""
        d = self.to_dict()
        d["is_immutable"] = False
        d["model_id"] = None  # New copy gets new ID
        return VariogramModel.from_dict(d)


# =============================================================================
# DATA HASH COMPUTATION
# =============================================================================

def compute_data_hash(
    coordinates: np.ndarray,
    values: np.ndarray,
    variable_name: str = "",
    dataset_version: str = "",
) -> str:
    """
    Compute SHA-256 hash of source data for lineage tracking.
    
    Parameters
    ----------
    coordinates : np.ndarray
        (N, 3) array of XYZ coordinates
    values : np.ndarray
        (N,) array of values
    variable_name : str
        Name of the variable
    dataset_version : str
        Dataset version identifier
    
    Returns
    -------
    str
        SHA-256 hash (first 16 characters)
    """
    # Round to avoid floating-point noise
    coords_rounded = np.round(np.asarray(coordinates, dtype=float), decimals=6)
    vals_rounded = np.round(np.asarray(values, dtype=float), decimals=6)
    
    # Create hashable representation
    content = f"v:{variable_name}|ds:{dataset_version}|n:{len(values)}|"
    content += f"coords_sum:{coords_rounded.sum():.6f}|"
    content += f"vals_sum:{vals_rounded.sum():.6f}|"
    content += f"vals_mean:{np.nanmean(vals_rounded):.6f}|"
    content += f"vals_std:{np.nanstd(vals_rounded):.6f}"
    
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_variogram_data_match(
    model: VariogramModel,
    current_data_hash: str,
    raise_on_mismatch: bool = True,
) -> bool:
    """
    Validate that a variogram model matches the current data.

    AUDIT CRITICAL: This check prevents using a variogram fitted to one
    dataset for estimation on a different dataset.

    Parameters
    ----------
    model : VariogramModel
        The variogram model to validate
    current_data_hash : str
        Hash of the current data to be estimated
    raise_on_mismatch : bool
        If True, raise VariogramGateError on mismatch

    Returns
    -------
    bool
        True if model matches data

    Raises
    ------
    VariogramGateError
        If raise_on_mismatch=True and hashes don't match
    """
    if model.data_hash is None:
        logger.warning("AUDIT: Variogram model has no data_hash - cannot verify lineage")
        return True  # Cannot verify, allow to proceed with warning

    if model.data_hash != current_data_hash:
        msg = (
            f"VARIOGRAM LINEAGE VIOLATION: Model was fitted to data with hash "
            f"'{model.data_hash}' but current data has hash '{current_data_hash}'. "
            "This may indicate the variogram was fitted to a different dataset."
        )
        logger.error(msg)
        if raise_on_mismatch:
            from .variogram_gates import VariogramGateError
            raise VariogramGateError(msg)
        return False

    return True


def compute_variogram_signature(variogram_params: Dict[str, Any]) -> str:
    """
    Compute a unique signature for variogram parameters.

    Used to ensure consistency between variogram analysis, OK, and SGSIM.

    Parameters
    ----------
    variogram_params : dict
        Variogram parameters dict with keys: range, sill, nugget, model_type, anisotropy

    Returns
    -------
    str
        SHA-256 signature (first 12 characters)
    """
    # Extract key parameters
    model_type = variogram_params.get('model_type', 'spherical')
    range_val = variogram_params.get('range', 0.0)
    sill = variogram_params.get('sill', 0.0)
    nugget = variogram_params.get('nugget', 0.0)

    # Anisotropy parameters
    aniso = variogram_params.get('anisotropy', {})
    azimuth = aniso.get('azimuth', 0.0) if aniso else 0.0
    dip = aniso.get('dip', 0.0) if aniso else 0.0
    major_range = aniso.get('major_range', range_val) if aniso else range_val
    minor_range = aniso.get('minor_range', range_val) if aniso else range_val
    vert_range = aniso.get('vert_range', range_val) if aniso else range_val

    # Create signature string
    content = (
        f"model:{model_type}|"
        f"nug:{nugget:.6f}|"
        f"sill:{sill:.6f}|"
        f"rng:{range_val:.6f}|"
        f"az:{azimuth:.2f}|"
        f"dip:{dip:.2f}|"
        f"major:{major_range:.6f}|"
        f"minor:{minor_range:.6f}|"
        f"vert:{vert_range:.6f}"
    )

    return hashlib.sha256(content.encode()).hexdigest()[:12]


def validate_variogram_consistency(
    variogram_params: Dict[str, Any],
    reference_signature: str,
    context: str = "estimation",
    raise_on_mismatch: bool = True,
) -> bool:
    """
    Validate that variogram parameters match a reference signature.

    PROFESSIONAL STANDARD: Ensures OK and SGSIM use the same variogram as approved
    in the variogram analysis stage.

    Parameters
    ----------
    variogram_params : dict
        Current variogram parameters
    reference_signature : str
        Reference signature from approved variogram
    context : str
        Context string for error messages (e.g., "Ordinary Kriging")
    raise_on_mismatch : bool
        If True, raise ValueError on mismatch

    Returns
    -------
    bool
        True if signatures match

    Raises
    ------
    ValueError
        If raise_on_mismatch=True and signatures don't match
    """
    current_signature = compute_variogram_signature(variogram_params)

    if current_signature != reference_signature:
        msg = (
            f"VARIOGRAM MISMATCH in {context}: "
            f"Current signature '{current_signature}' does not match "
            f"approved signature '{reference_signature}'. "
            f"You must use the same variogram parameters approved in variogram analysis."
        )
        logger.error(msg)
        if raise_on_mismatch:
            raise ValueError(msg)
        return False

    logger.info(f"✓ Variogram signature validated for {context}: {current_signature}")
    return True


# =============================================================================
# LEGACY COMPATIBILITY ALIASES
# =============================================================================
# These provide backward compatibility with existing code

# Alias for MODEL_MAP
MODEL_FUN = MODEL_MAP

# Alias for spherical_model
spherical_variogram = spherical_model
exponential_variogram = exponential_model
gaussian_variogram = gaussian_model


def fit_variogram_simple(
    distances: np.ndarray,
    gammas: np.ndarray,
    model_type: str = "spherical",
) -> Tuple[float, float, float]:
    """
    Simple variogram fitting - returns (nugget, sill, range).
    
    This is a legacy compatibility wrapper. For new code, use
    the full fitting functions in variogram_functions.py.
    """
    from .variogram_functions import fit_variogram_model
    return fit_variogram_model(distances, gammas, model_type)

