"""
Ground Control & Geotechnical Analysis Module

Rock mass rating, pillar stability, seismic risk assessment, and support design.

Implements:
- RMR (Rock Mass Rating)
- Q-System (Norwegian Geotechnical Institute)
- GSI (Geological Strength Index)
- Pillar Factor of Safety
- Seismic Probability of Exceedance
- Support selection matrix

Author: BlockModelViewer Team
Date: 2025-11-06
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RockMassClass(Enum):
    """RMR rock mass classification."""
    VERY_GOOD = "Very Good (81-100)"
    GOOD = "Good (61-80)"
    FAIR = "Fair (41-60)"
    POOR = "Poor (21-40)"
    VERY_POOR = "Very Poor (0-20)"


@dataclass
class RockMassProperties:
    """
    Rock mass geotechnical properties.
    
    Attributes:
        ucs: Uniaxial compressive strength (MPa)
        rqd: Rock Quality Designation (%)
        spacing: Joint spacing (m)
        condition: Joint condition rating (0-6)
        groundwater: Groundwater condition rating (0-15)
        orientation: Joint orientation adjustment (-12 to 0)
    """
    ucs: float  # MPa
    rqd: float  # %
    spacing: float  # meters
    condition: int  # 0-6
    groundwater: int  # 0-15
    orientation: int = 0  # -12 to 0


def calculate_rmr(props: RockMassProperties) -> Tuple[float, RockMassClass]:
    """
    Calculate Rock Mass Rating (RMR76/89).
    
    Args:
        props: Rock mass properties
        
    Returns:
        Tuple of (RMR score, classification)
    """
    # 1. UCS strength rating
    if props.ucs > 250:
        ucs_rating = 15
    elif props.ucs > 100:
        ucs_rating = 12
    elif props.ucs > 50:
        ucs_rating = 7
    elif props.ucs > 25:
        ucs_rating = 4
    elif props.ucs > 5:
        ucs_rating = 2
    elif props.ucs > 1:
        ucs_rating = 1
    else:
        ucs_rating = 0
    
    # 2. RQD rating
    if props.rqd >= 90:
        rqd_rating = 20
    elif props.rqd >= 75:
        rqd_rating = 17
    elif props.rqd >= 50:
        rqd_rating = 13
    elif props.rqd >= 25:
        rqd_rating = 8
    else:
        rqd_rating = 3
    
    # 3. Joint spacing rating
    if props.spacing > 2.0:
        spacing_rating = 20
    elif props.spacing > 0.6:
        spacing_rating = 15
    elif props.spacing > 0.2:
        spacing_rating = 10
    elif props.spacing > 0.06:
        spacing_rating = 8
    else:
        spacing_rating = 5
    
    # 4. Joint condition (given)
    condition_rating = props.condition
    
    # 5. Groundwater (given)
    gw_rating = props.groundwater
    
    # 6. Orientation adjustment
    orient_adj = props.orientation
    
    # Total RMR
    rmr = ucs_rating + rqd_rating + spacing_rating + condition_rating + gw_rating + orient_adj
    rmr = max(0, min(100, rmr))  # Clamp to 0-100
    
    # Classification
    if rmr >= 81:
        classification = RockMassClass.VERY_GOOD
    elif rmr >= 61:
        classification = RockMassClass.GOOD
    elif rmr >= 41:
        classification = RockMassClass.FAIR
    elif rmr >= 21:
        classification = RockMassClass.POOR
    else:
        classification = RockMassClass.VERY_POOR
    
    return rmr, classification


def calculate_q_system(rqd: float, jn: float, jr: float, ja: float, jw: float, srf: float) -> float:
    """
    Calculate Q-System (NGI tunneling quality index).
    
    Args:
        rqd: Rock Quality Designation (%)
        jn: Joint set number (0.5-20)
        jr: Joint roughness number (0.5-4)
        ja: Joint alteration number (0.75-20)
        jw: Joint water reduction factor (0.05-1.0)
        srf: Stress reduction factor (0.5-400)
        
    Returns:
        Q value
    """
    q = (rqd / jn) * (jr / ja) * (jw / srf)
    return max(0.001, q)  # Minimum Q value


def calculate_gsi(structure_rating: int, surface_condition: int) -> int:
    """
    Calculate Geological Strength Index (GSI).
    
    Args:
        structure_rating: Rock structure rating (1-5 scale)
        surface_condition: Joint surface condition (1-5 scale)
        
    Returns:
        GSI value (0-100)
    """
    # Simplified GSI estimation
    # In practice, use GSI chart lookup
    gsi = (structure_rating + surface_condition) * 10
    return max(10, min(100, gsi))


def calculate_pillar_fos(pillar_strength_mpa: float, pillar_stress_mpa: float) -> float:
    """
    Calculate pillar Factor of Safety.
    
    Args:
        pillar_strength_mpa: Pillar strength (MPa)
        pillar_stress_mpa: Applied stress (MPa)
        
    Returns:
        Factor of Safety
    """
    if pillar_stress_mpa <= 0:
        return float('inf')
    return pillar_strength_mpa / pillar_stress_mpa


def estimate_pillar_strength(ucs: float, width: float, height: float, k1: float = 0.778, k2: float = 0.222) -> float:
    """
    Estimate pillar strength using Hedley-Grant formula.
    
    Args:
        ucs: Uniaxial compressive strength (MPa)
        width: Pillar width (m)
        height: Pillar height (m)
        k1, k2: Empirical constants
        
    Returns:
        Pillar strength (MPa)
    """
    if height <= 0:
        return ucs
    return ucs * k1 * (width / height) ** k2


def seismic_poe(local_magnitude: float, distance_m: float, ppv_threshold: float = 10.0) -> float:
    """
    Calculate Probability of Exceedance for seismic event.
    
    Args:
        local_magnitude: Local magnitude of seismic event
        distance_m: Distance from source (m)
        ppv_threshold: Peak particle velocity threshold (mm/s)
        
    Returns:
        Probability of exceedance (0-1)
    """
    # Simplified PPV attenuation: PPV = K * (distance)^-α * 10^(β*M)
    K = 1000  # Site constant
    alpha = 1.5
    beta = 0.6
    
    if distance_m <= 0:
        return 1.0
    
    ppv_predicted = K * (distance_m ** -alpha) * (10 ** (beta * local_magnitude))
    
    # Probability model (lognormal distribution approximation)
    sigma = 0.5  # Uncertainty
    z = (np.log(ppv_predicted) - np.log(ppv_threshold)) / sigma
    poe = 1.0 - 0.5 * (1 + np.tanh(z / np.sqrt(2)))
    
    return max(0.0, min(1.0, poe))


def select_support(rmr: float, span: float, excavation_type: str = 'stope') -> Dict[str, str]:
    """
    Select ground support based on RMR and span.
    
    Args:
        rmr: Rock Mass Rating
        span: Excavation span (m)
        excavation_type: Type of excavation
        
    Returns:
        Dictionary with support recommendations
    """
    support = {
        'bolt_type': 'None',
        'bolt_spacing': 'N/A',
        'mesh': 'No',
        'shotcrete': 'No',
        'steel_sets': 'No',
        'stand_up_time': 'Unlimited'
    }
    
    if rmr >= 81:  # Very Good
        support['bolt_type'] = 'Spot bolts'
        support['bolt_spacing'] = '2.5m'
        support['stand_up_time'] = '>1 year for 15m span'
    elif rmr >= 61:  # Good
        support['bolt_type'] = 'Systematic bolts'
        support['bolt_spacing'] = '1.5-2.0m'
        support['mesh'] = 'Optional'
        support['stand_up_time'] = '6 months for 10m span'
    elif rmr >= 41:  # Fair
        support['bolt_type'] = 'Systematic bolts + mesh'
        support['bolt_spacing'] = '1.0-1.5m'
        support['mesh'] = 'Yes'
        support['shotcrete'] = '50-100mm in crown'
        support['stand_up_time'] = '1 week for 5m span'
    elif rmr >= 21:  # Poor
        support['bolt_type'] = 'Systematic bolts + mesh'
        support['bolt_spacing'] = '1.0-1.5m'
        support['mesh'] = 'Yes'
        support['shotcrete'] = '100-150mm in crown and walls'
        support['steel_sets'] = 'Light ribs for span >6m'
        support['stand_up_time'] = '10 hours for 2.5m span'
    else:  # Very Poor
        support['bolt_type'] = 'Systematic bolts + mesh'
        support['bolt_spacing'] = '1.0m'
        support['mesh'] = 'Yes'
        support['shotcrete'] = '150-200mm in crown, walls, and face'
        support['steel_sets'] = 'Heavy ribs <1m spacing'
        support['stand_up_time'] = '30 minutes for 1m span'
    
    return support


def analyze_stope_stability(stope, rock_props: RockMassProperties, stress_mpa: float = 10.0) -> Dict:
    """
    Comprehensive stability analysis for a stope.
    
    Args:
        stope: Stope object
        rock_props: Rock mass properties
        stress_mpa: Applied stress (MPa)
        
    Returns:
        Dictionary with stability analysis results
    """
    # Calculate RMR
    rmr, classification = calculate_rmr(rock_props)
    
    # Calculate Q-system (simplified)
    q = calculate_q_system(
        rqd=rock_props.rqd,
        jn=9.0,  # Moderate jointing
        jr=1.5,  # Moderate roughness
        ja=2.0,  # Slight alteration
        jw=1.0,  # Dry
        srf=2.5  # Medium stress
    )
    
    # Pillar strength and FoS
    crown_pillar_m = stope.geom.get('crown_pillar', 6.0)
    stope_height = stope.geom.get('height', 20.0)
    
    pillar_strength = estimate_pillar_strength(rock_props.ucs, crown_pillar_m, stope_height)
    fos = calculate_pillar_fos(pillar_strength, stress_mpa)
    
    # Support requirements
    span = max(stope.geom.get('length', 0), stope.geom.get('width', 0))
    support = select_support(rmr, span)
    
    # Risk assessment
    if fos < 1.3:
        risk_level = "High"
    elif fos < 1.5:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
    
    return {
        'stope_id': stope.id,
        'rmr': rmr,
        'rmr_class': classification.value,
        'q_system': q,
        'pillar_strength_mpa': pillar_strength,
        'pillar_stress_mpa': stress_mpa,
        'fos': fos,
        'risk_level': risk_level,
        'support': support
    }
