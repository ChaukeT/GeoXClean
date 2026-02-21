"""
Control Sample Management System

Manages CRM (Certified Reference Materials), duplicates, and blanks.
JORC/SAMREC requirement for quality assurance.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ControlSampleType(Enum):
    """Types of control samples."""
    CRM = "crm"  # Certified Reference Material
    DUPLICATE = "duplicate"
    BLANK = "blank"
    STANDARD = "standard"


class ControlSampleStatus(Enum):
    """Status of a control sample."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"


@dataclass
class ControlSample:
    """
    Control sample record.
    
    JORC/SAMREC compliant control sample tracking.
    """
    sample_id: str
    sample_type: ControlSampleType
    hole_id: str
    depth_from: float
    depth_to: float
    element: str  # e.g., "Fe", "Au", "SiO2"
    expected_value: Optional[float] = None  # For CRM
    measured_value: Optional[float] = None
    status: ControlSampleStatus = ControlSampleStatus.PENDING
    z_score: Optional[float] = None  # For CRM (standard deviations from expected)
    rsd_percent: Optional[float] = None  # For duplicates (relative standard deviation)
    inserted_date: datetime = field(default_factory=datetime.now)
    analyzed_date: Optional[datetime] = None
    lab_certificate: Optional[str] = None  # Reference to lab certificate
    comments: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_z_score(self) -> Optional[float]:
        """Calculate Z-score for CRM samples."""
        if self.sample_type != ControlSampleType.CRM:
            return None
        if self.expected_value is None or self.measured_value is None:
            return None
        # Simplified: assume standard deviation from metadata or use default
        std_dev = self.metadata.get("std_dev", 1.0)
        if std_dev == 0:
            return None
        self.z_score = abs((self.measured_value - self.expected_value) / std_dev)
        return self.z_score
    
    def evaluate_crm(self, tolerance_zscore: float = 2.0) -> ControlSampleStatus:
        """
        Evaluate CRM sample against tolerance.
        
        Args:
            tolerance_zscore: Maximum Z-score for passing
        
        Returns:
            ControlSampleStatus
        """
        if self.sample_type != ControlSampleType.CRM:
            return self.status
        
        z_score = self.calculate_z_score()
        if z_score is None:
            self.status = ControlSampleStatus.PENDING
            return self.status
        
        if z_score <= tolerance_zscore:
            self.status = ControlSampleStatus.PASSED
        elif z_score <= tolerance_zscore * 1.5:
            self.status = ControlSampleStatus.WARNING
        else:
            self.status = ControlSampleStatus.FAILED
        
        return self.status
    
    def evaluate_duplicate(self, max_rsd_percent: float = 5.0) -> ControlSampleStatus:
        """
        Evaluate duplicate sample against RSD tolerance.
        
        Args:
            max_rsd_percent: Maximum RSD% for passing
        
        Returns:
            ControlSampleStatus
        """
        if self.sample_type != ControlSampleType.DUPLICATE:
            return self.status
        
        if self.rsd_percent is None:
            self.status = ControlSampleStatus.PENDING
            return self.status
        
        if self.rsd_percent <= max_rsd_percent:
            self.status = ControlSampleStatus.PASSED
        elif self.rsd_percent <= max_rsd_percent * 1.5:
            self.status = ControlSampleStatus.WARNING
        else:
            self.status = ControlSampleStatus.FAILED
        
        return self.status


class ControlSampleManager:
    """
    Manages control samples for quality assurance.
    
    Tracks CRM, duplicates, and blanks according to JORC/SAMREC requirements.
    """
    
    def __init__(self):
        self.samples: Dict[str, ControlSample] = {}
        self._sample_counter = 0
        logger.info("ControlSampleManager initialized")
    
    def add_sample(
        self,
        sample_type: ControlSampleType,
        hole_id: str,
        depth_from: float,
        depth_to: float,
        element: str,
        expected_value: Optional[float] = None,
        measured_value: Optional[float] = None,
    ) -> ControlSample:
        """
        Add a new control sample.
        
        Args:
            sample_type: Type of control sample
            hole_id: Hole ID
            depth_from: Depth from
            depth_to: Depth to
            element: Element name
            expected_value: Expected value (for CRM)
            measured_value: Measured value
        
        Returns:
            ControlSample object
        """
        self._sample_counter += 1
        sample_id = f"CTRL-{self._sample_counter:06d}"
        
        sample = ControlSample(
            sample_id=sample_id,
            sample_type=sample_type,
            hole_id=hole_id,
            depth_from=depth_from,
            depth_to=depth_to,
            element=element,
            expected_value=expected_value,
            measured_value=measured_value,
        )
        
        self.samples[sample_id] = sample
        logger.info(f"Added control sample {sample_id}: {sample_type.value} for {hole_id} ({element})")
        return sample
    
    def get_sample(self, sample_id: str) -> Optional[ControlSample]:
        """Get a control sample by ID."""
        return self.samples.get(sample_id)
    
    def get_samples_for_hole(self, hole_id: str) -> List[ControlSample]:
        """Get all control samples for a specific hole."""
        return [s for s in self.samples.values() if s.hole_id == hole_id]
    
    def get_samples_by_type(self, sample_type: ControlSampleType) -> List[ControlSample]:
        """Get all control samples of a specific type."""
        return [s for s in self.samples.values() if s.sample_type == sample_type]
    
    def get_failed_samples(self) -> List[ControlSample]:
        """Get all failed control samples."""
        return [s for s in self.samples.values() if s.status == ControlSampleStatus.FAILED]
    
    def evaluate_all_samples(
        self,
        crm_tolerance_zscore: float = 2.0,
        duplicate_max_rsd: float = 5.0,
    ) -> Dict[str, int]:
        """
        Evaluate all control samples.
        
        Args:
            crm_tolerance_zscore: Z-score tolerance for CRM
            duplicate_max_rsd: Maximum RSD% for duplicates
        
        Returns:
            Dictionary with counts by status
        """
        results = {
            "passed": 0,
            "failed": 0,
            "warning": 0,
            "pending": 0,
        }
        
        for sample in self.samples.values():
            if sample.sample_type == ControlSampleType.CRM:
                sample.evaluate_crm(crm_tolerance_zscore)
            elif sample.sample_type == ControlSampleType.DUPLICATE:
                sample.evaluate_duplicate(duplicate_max_rsd)
            # Blanks are typically pass/fail based on presence of analyte
            elif sample.sample_type == ControlSampleType.BLANK:
                if sample.measured_value is not None and sample.measured_value > 0:
                    sample.status = ControlSampleStatus.FAILED
                else:
                    sample.status = ControlSampleStatus.PASSED
            
            results[sample.status.value] = results.get(sample.status.value, 0) + 1
        
        logger.info(f"Evaluated {len(self.samples)} control samples: {results}")
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about control samples."""
        stats = {
            "total_samples": len(self.samples),
            "by_type": {},
            "by_status": {},
            "crm_pass_rate": 0.0,
            "duplicate_pass_rate": 0.0,
            "passed": 0,
            "failed": 0,
            "warning": 0,
            "pending": 0,
        }
        
        for sample_type in ControlSampleType:
            samples = self.get_samples_by_type(sample_type)
            stats["by_type"][sample_type.value] = len(samples)
        
        for status in ControlSampleStatus:
            samples = [s for s in self.samples.values() if s.status == status]
            count = len(samples)
            stats["by_status"][status.value] = count
            # Also add to top-level counts for compatibility
            if status.value in ["passed", "failed", "warning", "pending"]:
                stats[status.value] = count
        
        # Calculate pass rates
        crm_samples = self.get_samples_by_type(ControlSampleType.CRM)
        if crm_samples:
            passed_crm = len([s for s in crm_samples if s.status == ControlSampleStatus.PASSED])
            stats["crm_pass_rate"] = passed_crm / len(crm_samples) * 100.0
        
        duplicate_samples = self.get_samples_by_type(ControlSampleType.DUPLICATE)
        if duplicate_samples:
            passed_dup = len([s for s in duplicate_samples if s.status == ControlSampleStatus.PASSED])
            stats["duplicate_pass_rate"] = passed_dup / len(duplicate_samples) * 100.0
        
        return stats


# Global control sample manager instance
_control_sample_manager: Optional[ControlSampleManager] = None


def get_control_sample_manager() -> ControlSampleManager:
    """Get the global control sample manager instance."""
    global _control_sample_manager
    if _control_sample_manager is None:
        _control_sample_manager = ControlSampleManager()
    return _control_sample_manager

