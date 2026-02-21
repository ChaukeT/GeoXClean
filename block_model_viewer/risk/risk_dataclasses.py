"""
Risk data structures for schedule-linked hazard analysis.

Core entities for period-by-period risk profiles and scenario comparison.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np


@dataclass
class PeriodRisk:
    """
    Risk metrics for a single schedule period.
    
    Attributes:
        period_index: Period index (0-based)
        start_time: Period start time (optional)
        end_time: Period end time (optional)
        mined_tonnage: Tonnage mined in this period
        metal: Metal content mined in this period
        seismic_hazard_index: Average seismic hazard index for active zones
        rockburst_risk_index: Average rockburst risk index for active stopes
        slope_risk_index: Average slope risk index for active pit sectors
        combined_risk_score: Combined risk score (weighted combination)
        notes: Additional notes or metadata
    """
    period_index: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    mined_tonnage: float = 0.0
    metal: float = 0.0
    seismic_hazard_index: Optional[float] = None
    rockburst_risk_index: Optional[float] = None
    slope_risk_index: Optional[float] = None
    slope_fos_min: Optional[float] = None  # Minimal FOS across slopes active in this period (STEP 27)
    slope_failure_probability: Optional[float] = None  # Probability of failure for slopes (STEP 27)
    combined_risk_score: Optional[float] = None
    notes: str = ""
    
    def compute_combined_score(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute combined risk score from individual components.
        
        Args:
            weights: Optional weights dict:
                - seismic_weight: Weight for seismic hazard (default: 0.4)
                - rockburst_weight: Weight for rockburst (default: 0.4)
                - slope_weight: Weight for slope risk (default: 0.2)
        
        Returns:
            Combined risk score (0-1)
        """
        if weights is None:
            weights = {
                'seismic_weight': 0.4,
                'rockburst_weight': 0.4,
                'slope_weight': 0.2
            }
        
        components = []
        total_weight = 0.0
        
        if self.seismic_hazard_index is not None:
            components.append(self.seismic_hazard_index * weights.get('seismic_weight', 0.4))
            total_weight += weights.get('seismic_weight', 0.4)
        
        if self.rockburst_risk_index is not None:
            components.append(self.rockburst_risk_index * weights.get('rockburst_weight', 0.4))
            total_weight += weights.get('rockburst_weight', 0.4)
        
        if self.slope_risk_index is not None:
            components.append(self.slope_risk_index / 100.0 * weights.get('slope_weight', 0.2))  # Normalize to 0-1
            total_weight += weights.get('slope_weight', 0.2)
        
        if components and total_weight > 0:
            combined = sum(components) / total_weight
            self.combined_risk_score = combined
            return combined
        
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'period_index': self.period_index,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'mined_tonnage': self.mined_tonnage,
            'metal': self.metal,
            'seismic_hazard_index': self.seismic_hazard_index,
            'rockburst_risk_index': self.rockburst_risk_index,
            'slope_risk_index': self.slope_risk_index,
            'slope_fos_min': self.slope_fos_min,
            'slope_failure_probability': self.slope_failure_probability,
            'combined_risk_score': self.combined_risk_score,
            'notes': self.notes
        }


@dataclass
class ScheduleRiskProfile:
    """
    Complete risk profile for a mine schedule.
    
    Attributes:
        schedule_id: Unique identifier for the schedule
        periods: List of PeriodRisk instances
        summary_stats: Summary statistics dict
        metadata: Additional metadata (schedule type, source, etc.)
    """
    schedule_id: str
    periods: List[PeriodRisk]
    summary_stats: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_summary_stats(self) -> Dict[str, float]:
        """
        Compute summary statistics from periods.
        
        Returns:
            Dict with statistics (mean, max, min risk scores, etc.)
        """
        if not self.periods:
            return {}
        
        # Extract metrics
        combined_scores = [p.combined_risk_score for p in self.periods if p.combined_risk_score is not None]
        seismic_indices = [p.seismic_hazard_index for p in self.periods if p.seismic_hazard_index is not None]
        rockburst_indices = [p.rockburst_risk_index for p in self.periods if p.rockburst_risk_index is not None]
        slope_indices = [p.slope_risk_index for p in self.periods if p.slope_risk_index is not None]
        tonnages = [p.mined_tonnage for p in self.periods]
        
        stats = {
            'n_periods': len(self.periods),
            'total_tonnage': sum(tonnages),
            'total_metal': sum(p.metal for p in self.periods)
        }
        
        if combined_scores:
            stats['combined_risk'] = {
                'mean': float(np.mean(combined_scores)),
                'max': float(np.max(combined_scores)),
                'min': float(np.min(combined_scores)),
                'std': float(np.std(combined_scores))
            }
        
        if seismic_indices:
            stats['seismic_hazard'] = {
                'mean': float(np.mean(seismic_indices)),
                'max': float(np.max(seismic_indices)),
                'min': float(np.min(seismic_indices))
            }
        
        if rockburst_indices:
            stats['rockburst_risk'] = {
                'mean': float(np.mean(rockburst_indices)),
                'max': float(np.max(rockburst_indices)),
                'min': float(np.min(rockburst_indices))
            }
        
        if slope_indices:
            stats['slope_risk'] = {
                'mean': float(np.mean(slope_indices)),
                'max': float(np.max(slope_indices)),
                'min': float(np.min(slope_indices))
            }
        
        self.summary_stats = stats
        return stats
    
    def get_period(self, period_index: int) -> Optional[PeriodRisk]:
        """Get PeriodRisk for a specific period index."""
        for period in self.periods:
            if period.period_index == period_index:
                return period
        return None


@dataclass
class RiskScenarioComparison:
    """
    Comparison of multiple schedule risk profiles.
    
    Attributes:
        base_profile: Base schedule risk profile
        alternative_profiles: List of alternative schedule profiles
        metrics: Comparison metrics dict
    """
    base_profile: ScheduleRiskProfile
    alternative_profiles: List[ScheduleRiskProfile]
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def compute_comparison_metrics(self) -> Dict[str, Any]:
        """
        Compute comparison metrics between base and alternatives.
        
        Returns:
            Dict with delta risk, risk-adjusted metrics, etc.
        """
        if not self.alternative_profiles:
            return {}
        
        comparison = {}
        
        base_stats = self.base_profile.summary_stats
        if not base_stats:
            self.base_profile.compute_summary_stats()
            base_stats = self.base_profile.summary_stats
        
        for i, alt_profile in enumerate(self.alternative_profiles):
            alt_id = alt_profile.schedule_id
            
            if not alt_profile.summary_stats:
                alt_profile.compute_summary_stats()
            alt_stats = alt_profile.summary_stats
            
            # Delta metrics
            delta = {}
            
            if 'combined_risk' in base_stats and 'combined_risk' in alt_stats:
                base_mean = base_stats['combined_risk'].get('mean', 0.0)
                alt_mean = alt_stats['combined_risk'].get('mean', 0.0)
                delta['combined_risk_mean'] = alt_mean - base_mean
                delta['combined_risk_pct_change'] = ((alt_mean - base_mean) / base_mean * 100) if base_mean > 0 else 0.0
            
            if 'combined_risk' in base_stats and 'combined_risk' in alt_stats:
                base_max = base_stats['combined_risk'].get('max', 0.0)
                alt_max = alt_stats['combined_risk'].get('max', 0.0)
                delta['combined_risk_max'] = alt_max - base_max
            
            # Period-by-period comparison
            period_deltas = []
            for period_idx in range(max(len(self.base_profile.periods), len(alt_profile.periods))):
                base_period = self.base_profile.get_period(period_idx)
                alt_period = alt_profile.get_period(period_idx)
                
                if base_period and alt_period:
                    if base_period.combined_risk_score is not None and alt_period.combined_risk_score is not None:
                        period_deltas.append({
                            'period': period_idx,
                            'delta_risk': alt_period.combined_risk_score - base_period.combined_risk_score,
                            'base_risk': base_period.combined_risk_score,
                            'alt_risk': alt_period.combined_risk_score
                        })
            
            comparison[alt_id] = {
                'delta_metrics': delta,
                'period_deltas': period_deltas
            }
        
        self.metrics = comparison
        return comparison

