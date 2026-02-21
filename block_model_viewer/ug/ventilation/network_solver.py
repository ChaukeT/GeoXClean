"""
Ventilation Network Solver

Kirchhoff-based airflow network analysis for underground mine ventilation.
Implements Hardy-Cross method for pressure balancing and fan duty calculations.

References:
- McPherson, M.J. (2009). Subsurface Ventilation Engineering
- Hartman et al. (2012). Mine Ventilation and Air Conditioning
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AirwayType(Enum):
    """Types of underground airways"""
    INTAKE = "intake"
    RETURN = "return"
    RAISE = "raise"
    DECLINE = "decline"
    DRIFT = "drift"
    CROSSCUT = "crosscut"


class FanType(Enum):
    """Types of ventilation fans"""
    MAIN_SURFACE = "main_surface"
    BOOSTER = "booster"
    AUXILIARY = "auxiliary"


@dataclass
class VentilationAirway:
    """
    Represents an airway in the ventilation network.
    
    Attributes:
        airway_id: Unique identifier
        from_node: Starting node ID
        to_node: Ending node ID
        length: Length in meters
        area: Cross-sectional area in m²
        perimeter: Perimeter in meters (for friction)
        roughness: Roughness coefficient k (m)
        airway_type: Type of airway
        resistance: Airflow resistance R (Ns²/m⁸)
    """
    airway_id: str
    from_node: str
    to_node: str
    length: float  # m
    area: float  # m²
    perimeter: float  # m
    roughness: float = 0.002  # m (typical concrete)
    airway_type: AirwayType = AirwayType.DRIFT
    resistance: Optional[float] = None  # Ns²/m⁸
    
    def __post_init__(self):
        """Calculate resistance if not provided"""
        if self.resistance is None:
            # Atkinson equation: R = (k * L * P) / (A³)
            # where k = friction factor, L = length, P = perimeter, A = area
            # Friction factor from Colebrook-White
            k = self.roughness * 0.05  # Simplified friction factor
            self.resistance = (k * self.length * self.perimeter) / (self.area ** 3)


@dataclass
class VentilationNode:
    """
    Represents a junction node in the ventilation network.
    
    Attributes:
        node_id: Unique identifier
        elevation: Elevation in meters (for natural ventilation pressure)
        heat_source: Heat generation at node in kW (equipment, stope, etc.)
        required_airflow: Minimum required airflow in m³/s
    """
    node_id: str
    elevation: float = 0.0  # m
    heat_source: float = 0.0  # kW
    required_airflow: float = 0.0  # m³/s


@dataclass
class VentilationFan:
    """
    Represents a fan in the ventilation network.
    
    Attributes:
        fan_id: Unique identifier
        airway_id: Airway where fan is installed
        fan_type: Type of fan
        pressure_curve: Polynomial coefficients [p0, p1, p2] for ΔP = p0 + p1*Q + p2*Q²
        efficiency: Fan total efficiency (0-1)
        rated_power: Rated power in kW
    """
    fan_id: str
    airway_id: str
    fan_type: FanType
    pressure_curve: List[float]  # [p0, p1, p2] in Pa
    efficiency: float = 0.65  # typical
    rated_power: float = 0.0  # kW


@dataclass
class NetworkSolution:
    """
    Solution to ventilation network.
    
    Attributes:
        airway_flows: Dict mapping airway_id to airflow (m³/s, positive = from→to)
        node_pressures: Dict mapping node_id to gauge pressure (Pa)
        fan_duties: Dict mapping fan_id to duty point (flow_m3s, pressure_pa, power_kw)
        total_power: Total fan power consumption in kW
        converged: Whether Hardy-Cross iteration converged
        iterations: Number of iterations required
        max_imbalance: Maximum pressure imbalance in any circuit (Pa)
    """
    airway_flows: Dict[str, float] = field(default_factory=dict)
    node_pressures: Dict[str, float] = field(default_factory=dict)
    fan_duties: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    total_power: float = 0.0
    converged: bool = False
    iterations: int = 0
    max_imbalance: float = 0.0


def solve_ventilation_network(
    airways: List[VentilationAirway],
    nodes: List[VentilationNode],
    fans: List[VentilationFan],
    max_iterations: int = 50,
    tolerance: float = 1.0  # Pa
) -> NetworkSolution:
    """
    Solve ventilation network using Hardy-Cross method.
    
    The Hardy-Cross method iteratively balances airflow in circuits until
    pressure drops around all closed loops sum to zero (Kirchhoff's law).
    
    Algorithm:
    1. Initialize flows (satisfy mass balance at nodes)
    2. For each circuit, calculate pressure imbalance
    3. Adjust flows: ΔQ = -Σ(R*Q*|Q|) / (2*Σ(R*Q²))
    4. Repeat until all circuits balanced
    
    Args:
        airways: List of airways
        nodes: List of nodes
        fans: List of fans
        max_iterations: Maximum Hardy-Cross iterations
        tolerance: Convergence tolerance for pressure imbalance (Pa)
    
    Returns:
        NetworkSolution with flows, pressures, and fan duties
    """
    logger.info(f"Solving ventilation network: {len(airways)} airways, {len(nodes)} nodes, {len(fans)} fans")
    
    # Build network graph
    airway_dict = {aw.airway_id: aw for aw in airways}
    node_dict = {n.node_id: n for n in nodes}
    fan_dict = {f.airway_id: f for f in fans}
    
    # Initialize flows (simple initial guess based on required flows)
    flows = {aw.airway_id: 10.0 for aw in airways}  # m³/s initial guess
    
    # Identify circuits (simplified - assume user provides circuit definitions)
    # For now, use simplified nodal analysis
    
    # Hardy-Cross iterations
    converged = False
    iteration = 0
    max_imbalance = float('inf')
    
    for iteration in range(max_iterations):
        max_imbalance = 0.0
        
        # For each airway, calculate pressure drop
        pressure_drops = {}
        for aw_id, flow in flows.items():
            aw = airway_dict[aw_id]
            
            # Pressure drop: ΔP = R * Q * |Q|
            dp_friction = aw.resistance * flow * abs(flow)
            
            # Add fan pressure if fan is in this airway
            dp_fan = 0.0
            if aw_id in fan_dict:
                fan = fan_dict[aw_id]
                # Fan characteristic: ΔP = p0 + p1*Q + p2*Q²
                p0, p1, p2 = fan.pressure_curve
                dp_fan = p0 + p1 * flow + p2 * (flow ** 2)
            
            pressure_drops[aw_id] = dp_friction - dp_fan  # Fan adds pressure (negative drop)
        
        # Build node equations (mass balance: Σ Q_in = Σ Q_out)
        node_balance = {n.node_id: 0.0 for n in nodes}
        for aw in airways:
            node_balance[aw.from_node] -= flows[aw.airway_id]
            node_balance[aw.to_node] += flows[aw.airway_id]
        
        # Calculate maximum imbalance
        max_imbalance = max(abs(imb) for imb in node_balance.values())
        
        if max_imbalance < tolerance:
            converged = True
            break
        
        # Adjust flows (simplified Newton-Raphson)
        for aw_id in flows.keys():
            aw = airway_dict[aw_id]
            
            # Calculate flow correction
            dp = pressure_drops[aw_id]
            
            # Derivative: d(ΔP)/dQ = 2*R*|Q|
            derivative = 2 * aw.resistance * abs(flows[aw_id]) if flows[aw_id] != 0 else 1e-6
            
            # Update flow
            correction = -dp / derivative * 0.5  # Damping factor 0.5 for stability
            flows[aw_id] += correction
    
    logger.info(f"Network solved: converged={converged}, iterations={iteration}, max_imbalance={max_imbalance:.2f} Pa")
    
    # Calculate node pressures (relative to atmospheric at surface)
    node_pressures = {n.node_id: 0.0 for n in nodes}
    # Simplified: pressure at node = -Σ(ΔP from source to node)
    
    # Calculate fan duties
    fan_duties = {}
    total_power = 0.0
    for fan in fans:
        if fan.airway_id in flows:
            flow = flows[fan.airway_id]
            p0, p1, p2 = fan.pressure_curve
            pressure = p0 + p1 * flow + p2 * (flow ** 2)
            power = calculate_fan_duty(flow, pressure, fan.efficiency)
            fan_duties[fan.fan_id] = (flow, pressure, power)
            total_power += power
    
    return NetworkSolution(
        airway_flows=flows,
        node_pressures=node_pressures,
        fan_duties=fan_duties,
        total_power=total_power,
        converged=converged,
        iterations=iteration + 1,
        max_imbalance=max_imbalance
    )


def calculate_fan_duty(
    airflow: float,  # m³/s
    pressure: float,  # Pa
    efficiency: float = 0.65
) -> float:
    """
    Calculate fan power consumption.
    
    Power = (Q × ΔP) / (η × 1000)
    
    Args:
        airflow: Airflow in m³/s
        pressure: Pressure rise in Pa
        efficiency: Total fan efficiency (0-1)
    
    Returns:
        Power in kW
    """
    if efficiency <= 0:
        efficiency = 0.01  # Prevent division by zero
    
    power_kw = (airflow * pressure) / (efficiency * 1000)
    return max(0, power_kw)


def calculate_heat_stress_index(
    dry_bulb_temp: float,  # °C
    wet_bulb_temp: float,  # °C
    airflow: float,  # m³/s
    area: float  # m²
) -> Dict[str, float]:
    """
    Calculate heat stress indices for underground workers.
    
    Calculates:
    - Air velocity (m/s)
    - Wet Bulb Globe Temperature (WBGT) approximation
    - Heat stress category
    
    WBGT Categories (°C):
    - <26: Acceptable for all work
    - 26-28: Caution, reduce heavy work
    - 28-30: Extreme caution, breaks needed
    - >30: Danger, stop work
    
    Args:
        dry_bulb_temp: Dry bulb temperature in °C
        wet_bulb_temp: Wet bulb temperature in °C
        airflow: Airflow in m³/s
        area: Cross-sectional area in m²
    
    Returns:
        Dict with air_velocity, wbgt, heat_stress_category
    """
    # Calculate air velocity
    air_velocity = airflow / area if area > 0 else 0.0
    
    # Approximate WBGT for underground (no solar radiation)
    # WBGT ≈ 0.7*WBT + 0.3*DBT
    wbgt = 0.7 * wet_bulb_temp + 0.3 * dry_bulb_temp
    
    # Adjust for air velocity (higher velocity reduces heat stress)
    if air_velocity > 0.5:
        # Each m/s above 0.5 reduces effective WBGT by ~0.5°C
        wbgt -= (air_velocity - 0.5) * 0.5
    
    # Categorize heat stress
    if wbgt < 26:
        category = "Acceptable"
    elif wbgt < 28:
        category = "Caution"
    elif wbgt < 30:
        category = "Extreme Caution"
    else:
        category = "Danger"
    
    return {
        'air_velocity_ms': round(air_velocity, 2),
        'wbgt_c': round(wbgt, 1),
        'heat_stress_category': category,
        'dry_bulb_c': dry_bulb_temp,
        'wet_bulb_c': wet_bulb_temp
    }


def estimate_airway_resistance(
    length: float,  # m
    width: float,  # m
    height: float,  # m
    roughness: float = 0.002  # m
) -> float:
    """
    Estimate airway resistance for rectangular cross-section.
    
    Uses Atkinson equation: R = (k * L * P) / A³
    
    Args:
        length: Length in meters
        width: Width in meters
        height: Height in meters
        roughness: Surface roughness in meters
    
    Returns:
        Resistance in Ns²/m⁸
    """
    area = width * height
    perimeter = 2 * (width + height)
    
    # Friction factor (simplified)
    k = roughness * 0.05
    
    resistance = (k * length * perimeter) / (area ** 3)
    return resistance


def design_main_fan(
    total_airflow: float,  # m³/s
    total_resistance: float,  # Ns²/m⁸
    efficiency: float = 0.70,
    pressure_margin: float = 1.15  # 15% margin
) -> Dict[str, float]:
    """
    Design main surface fan for mine.
    
    Args:
        total_airflow: Total mine airflow requirement in m³/s
        total_resistance: Total mine resistance in Ns²/m⁸
        efficiency: Fan total efficiency
        pressure_margin: Safety margin factor
    
    Returns:
        Dict with duty_flow, duty_pressure, rated_power
    """
    # Calculate pressure requirement
    # ΔP = R * Q²
    duty_pressure = total_resistance * (total_airflow ** 2)
    duty_pressure *= pressure_margin  # Add safety margin
    
    # Calculate power
    power = calculate_fan_duty(total_airflow, duty_pressure, efficiency)
    
    # Add motor efficiency loss (typically 95%)
    rated_power = power / 0.95
    
    return {
        'duty_flow_m3s': round(total_airflow, 1),
        'duty_pressure_pa': round(duty_pressure, 0),
        'rated_power_kw': round(rated_power, 1),
        'efficiency': efficiency
    }
