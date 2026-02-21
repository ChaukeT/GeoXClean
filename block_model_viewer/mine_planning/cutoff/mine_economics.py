"""
Multi-Period Mine Economics Engine for JORC/SAMREC Compliant NPV Analysis.

This module provides proper time-phased economic analysis for cutoff optimization:
- Multi-period discounted cash flow (DCF) analysis
- Mine capacity constraints (annual tonnage limits)
- Capital expenditure scheduling (initial + sustaining)
- Tax and royalty calculations
- Proper IRR calculation using scipy
- Payback period computation
- Sensitivity analysis with tornado charts

This is required for audit-grade "NPV-optimised cutoff" claims.

Author: GeoX Mining Software - Mine Economics Engine
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import scipy for IRR calculation
try:
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - IRR calculation will use approximation")


# =============================================================================
# DATA CLASSES
# =============================================================================

class MiningMethod(str, Enum):
    """Mining method types."""
    OPEN_PIT = "open_pit"
    UNDERGROUND = "underground"
    HYBRID = "hybrid"


class ProcessingRoute(str, Enum):
    """Processing route types."""
    CONVENTIONAL = "conventional"
    HEAP_LEACH = "heap_leach"
    DIRECT_SHIPPING = "direct_shipping"
    CUSTOM = "custom"


@dataclass
class MineCapacity:
    """
    Mine capacity constraints.
    
    Attributes:
        annual_ore_capacity: Maximum ore tonnes processed per year
        annual_waste_capacity: Maximum waste tonnes moved per year (optional)
        annual_total_movement: Maximum total material movement per year (optional)
        ramp_up_years: Years to reach full capacity (default 1 = instant)
        ramp_up_profile: Optional custom ramp-up profile (fractions of capacity per year)
    """
    annual_ore_capacity: float  # tonnes/year
    annual_waste_capacity: Optional[float] = None  # tonnes/year
    annual_total_movement: Optional[float] = None  # tonnes/year
    ramp_up_years: int = 1
    ramp_up_profile: Optional[List[float]] = None  # e.g., [0.6, 0.8, 1.0]
    
    def get_capacity_for_year(self, year: int) -> float:
        """Get effective ore capacity for a given year (1-indexed)."""
        if year <= 0:
            return 0.0
        
        if self.ramp_up_profile:
            if year <= len(self.ramp_up_profile):
                return self.annual_ore_capacity * self.ramp_up_profile[year - 1]
            return self.annual_ore_capacity
        
        if year <= self.ramp_up_years:
            # Linear ramp-up
            return self.annual_ore_capacity * (year / self.ramp_up_years)
        return self.annual_ore_capacity


@dataclass
class CapitalExpenditure:
    """
    Capital expenditure schedule.
    
    Attributes:
        initial_capex: Initial capital before production ($)
        sustaining_capex_annual: Annual sustaining capital ($)
        sustaining_capex_per_tonne: Sustaining capital per tonne ore ($/t)
        closure_cost: Mine closure and rehabilitation cost ($)
        working_capital_percentage: Working capital as % of annual revenue
    """
    initial_capex: float = 0.0
    sustaining_capex_annual: float = 0.0
    sustaining_capex_per_tonne: float = 0.0
    closure_cost: float = 0.0
    working_capital_percentage: float = 0.05  # 5% of revenue


@dataclass
class TaxParameters:
    """
    Tax and royalty parameters.
    
    Attributes:
        corporate_tax_rate: Corporate income tax rate (fraction)
        royalty_rate: Royalty rate on revenue (fraction)
        royalty_type: 'revenue' or 'profit' based
        depreciation_years: Years for straight-line depreciation
        tax_loss_carryforward: Allow tax loss carryforward
    """
    corporate_tax_rate: float = 0.30  # 30%
    royalty_rate: float = 0.05  # 5%
    royalty_type: str = "revenue"  # or "profit"
    depreciation_years: int = 10
    tax_loss_carryforward: bool = True


@dataclass
class EconomicParameters:
    """
    Complete economic parameters for mine evaluation.
    
    Attributes:
        metal_price: Metal price per unit ($/unit)
        mining_cost_ore: Mining cost per tonne ore ($/t)
        mining_cost_waste: Mining cost per tonne waste ($/t)
        processing_cost: Processing cost per tonne ore ($/t)
        admin_cost: Administration cost per tonne ore ($/t)
        transport_cost: Transport/logistics cost per tonne ore ($/t)
        recovery_rate: Metallurgical recovery (fraction)
        dilution_factor: Mining dilution (fraction added)
        mining_loss_factor: Mining loss (fraction lost)
        discount_rate: Annual discount rate (fraction)
        inflation_rate: Annual inflation rate (fraction)
        price_escalation: Annual price escalation (fraction)
        cost_escalation: Annual cost escalation (fraction)
    """
    metal_price: float = 50.0
    mining_cost_ore: float = 15.0
    mining_cost_waste: float = 5.0
    processing_cost: float = 25.0
    admin_cost: float = 5.0
    transport_cost: float = 0.0
    recovery_rate: float = 0.85
    dilution_factor: float = 0.05  # 5% dilution
    mining_loss_factor: float = 0.02  # 2% loss
    discount_rate: float = 0.10  # 10%
    inflation_rate: float = 0.02  # 2%
    price_escalation: float = 0.0  # Real terms
    cost_escalation: float = 0.0  # Real terms


@dataclass
class MineEconomicsConfig:
    """
    Complete mine economics configuration.
    """
    economic_params: EconomicParameters = field(default_factory=EconomicParameters)
    capacity: MineCapacity = field(default_factory=lambda: MineCapacity(annual_ore_capacity=10_000_000))
    capex: CapitalExpenditure = field(default_factory=CapitalExpenditure)
    tax: TaxParameters = field(default_factory=TaxParameters)
    mining_method: MiningMethod = MiningMethod.OPEN_PIT
    processing_route: ProcessingRoute = ProcessingRoute.CONVENTIONAL
    project_life_years: Optional[int] = None  # Computed from reserves if None


@dataclass
class AnnualCashFlow:
    """
    Annual cash flow for a single year.
    """
    year: int
    ore_tonnes: float
    waste_tonnes: float = 0.0
    average_grade: float = 0.0
    metal_produced: float = 0.0
    revenue: float = 0.0
    mining_cost: float = 0.0
    processing_cost: float = 0.0
    admin_cost: float = 0.0
    royalty: float = 0.0
    operating_cost_total: float = 0.0
    ebitda: float = 0.0
    depreciation: float = 0.0
    ebit: float = 0.0
    tax: float = 0.0
    net_profit: float = 0.0
    capex: float = 0.0
    working_capital_change: float = 0.0
    free_cash_flow: float = 0.0
    cumulative_cash_flow: float = 0.0
    discounted_cash_flow: float = 0.0


@dataclass
class MineEconomicsResult:
    """
    Complete mine economics result.
    """
    # Summary metrics
    npv: float = 0.0
    irr: Optional[float] = None
    payback_period: Optional[float] = None
    payback_period_discounted: Optional[float] = None
    total_revenue: float = 0.0
    total_operating_cost: float = 0.0
    total_capex: float = 0.0
    total_tax: float = 0.0
    mine_life_years: int = 0
    total_ore_tonnes: float = 0.0
    total_metal_produced: float = 0.0
    average_head_grade: float = 0.0
    average_operating_cost_per_tonne: float = 0.0
    average_all_in_cost_per_unit: float = 0.0
    
    # Annual cash flows
    annual_cash_flows: List[AnnualCashFlow] = field(default_factory=list)
    
    # Configuration used
    config: Optional[MineEconomicsConfig] = None
    cutoff_grade: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# MINE ECONOMICS ENGINE
# =============================================================================

class MineEconomicsEngine:
    """
    Multi-period mine economics engine for proper NPV analysis.
    
    This engine provides audit-grade economic analysis with:
    - Time-phased cash flows with capacity constraints
    - Proper discounting using mid-year convention
    - Capital expenditure scheduling
    - Tax calculations with depreciation
    - IRR calculation using numerical methods
    
    Usage:
        config = MineEconomicsConfig(
            economic_params=EconomicParameters(metal_price=50, ...),
            capacity=MineCapacity(annual_ore_capacity=10_000_000),
            capex=CapitalExpenditure(initial_capex=500_000_000, ...),
            tax=TaxParameters(corporate_tax_rate=0.30, ...)
        )
        engine = MineEconomicsEngine(config)
        result = engine.evaluate_cutoff(gt_curve, cutoff_grade=0.5)
    """
    
    def __init__(self, config: MineEconomicsConfig):
        """
        Initialize the mine economics engine.
        
        Args:
            config: Complete mine economics configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_cutoff(
        self,
        ore_tonnage: float,
        average_grade: float,
        waste_tonnage: float = 0.0,
        cutoff_grade: float = 0.0
    ) -> MineEconomicsResult:
        """
        Evaluate economics for a given cutoff grade.
        
        Args:
            ore_tonnage: Total ore tonnes above cutoff
            average_grade: Average grade above cutoff
            waste_tonnage: Total waste tonnes (optional)
            cutoff_grade: The cutoff grade being evaluated
        
        Returns:
            MineEconomicsResult with full cash flow analysis
        """
        self.logger.info(f"Evaluating cutoff {cutoff_grade:.2f}: "
                        f"{ore_tonnage:,.0f} t @ {average_grade:.3f}")
        
        # Apply mining modifying factors
        econ = self.config.economic_params
        adjusted_tonnage = ore_tonnage * (1 - econ.mining_loss_factor) * (1 + econ.dilution_factor)
        adjusted_grade = average_grade / (1 + econ.dilution_factor)  # Dilution reduces grade
        
        # Determine mine life
        annual_capacity = self.config.capacity.annual_ore_capacity
        mine_life = int(np.ceil(adjusted_tonnage / annual_capacity))
        
        if self.config.project_life_years:
            mine_life = min(mine_life, self.config.project_life_years)
        
        if mine_life == 0:
            return self._empty_result(cutoff_grade)
        
        self.logger.info(f"Mine life: {mine_life} years at {annual_capacity:,.0f} t/year capacity")
        
        # Generate annual cash flows
        annual_flows = self._generate_annual_cash_flows(
            total_ore=adjusted_tonnage,
            average_grade=adjusted_grade,
            total_waste=waste_tonnage,
            mine_life=mine_life
        )
        
        # Calculate summary metrics
        result = self._calculate_summary_metrics(annual_flows, cutoff_grade)
        
        return result
    
    def evaluate_gt_curve(
        self,
        gt_curve: Any,  # GradeTonnageCurve
        waste_tonnage: float = 0.0
    ) -> Dict[float, MineEconomicsResult]:
        """
        Evaluate economics for all cutoffs in a GT curve.
        
        Args:
            gt_curve: GradeTonnageCurve object
            waste_tonnage: Total waste tonnes (constant for now)
        
        Returns:
            Dict mapping cutoff grade to MineEconomicsResult
        """
        results = {}
        
        for point in gt_curve.points:
            result = self.evaluate_cutoff(
                ore_tonnage=point.tonnage,
                average_grade=point.avg_grade,
                waste_tonnage=waste_tonnage,
                cutoff_grade=point.cutoff_grade
            )
            results[point.cutoff_grade] = result
        
        return results
    
    def find_optimal_cutoff(
        self,
        gt_curve: Any,
        waste_tonnage: float = 0.0,
        optimization_target: str = "npv"
    ) -> Tuple[float, MineEconomicsResult]:
        """
        Find the optimal cutoff grade.
        
        Args:
            gt_curve: GradeTonnageCurve object
            waste_tonnage: Total waste tonnes
            optimization_target: "npv", "irr", or "payback"
        
        Returns:
            Tuple of (optimal_cutoff, result_at_optimal)
        """
        results = self.evaluate_gt_curve(gt_curve, waste_tonnage)
        
        if not results:
            return 0.0, self._empty_result(0.0)
        
        # Find optimal based on target
        if optimization_target == "npv":
            optimal = max(results.items(), key=lambda x: x[1].npv)
        elif optimization_target == "irr":
            optimal = max(results.items(), key=lambda x: x[1].irr or 0.0)
        elif optimization_target == "payback":
            # Minimize payback (filter out None)
            valid = [(k, v) for k, v in results.items() if v.payback_period is not None]
            if valid:
                optimal = min(valid, key=lambda x: x[1].payback_period)
            else:
                optimal = max(results.items(), key=lambda x: x[1].npv)
        else:
            optimal = max(results.items(), key=lambda x: x[1].npv)
        
        self.logger.info(f"Optimal cutoff ({optimization_target}): {optimal[0]:.2f} "
                        f"with NPV=${optimal[1].npv:,.0f}")
        
        return optimal
    
    def _generate_annual_cash_flows(
        self,
        total_ore: float,
        average_grade: float,
        total_waste: float,
        mine_life: int
    ) -> List[AnnualCashFlow]:
        """
        Generate annual cash flows for the mine life.
        """
        econ = self.config.economic_params
        cap = self.config.capacity
        capex = self.config.capex
        tax = self.config.tax
        
        flows = []
        remaining_ore = total_ore
        remaining_waste = total_waste
        cumulative_cf = 0.0
        previous_revenue = 0.0
        tax_loss_pool = 0.0
        
        # Year 0: Initial capex
        initial_flow = AnnualCashFlow(
            year=0,
            ore_tonnes=0,
            capex=capex.initial_capex,
            free_cash_flow=-capex.initial_capex,
            cumulative_cash_flow=-capex.initial_capex,
            discounted_cash_flow=-capex.initial_capex
        )
        flows.append(initial_flow)
        cumulative_cf = -capex.initial_capex
        
        # Annual depreciation (straight-line)
        annual_depreciation = capex.initial_capex / tax.depreciation_years if tax.depreciation_years > 0 else 0
        
        for year in range(1, mine_life + 1):
            # Get capacity for this year (ramp-up)
            year_capacity = cap.get_capacity_for_year(year)
            
            # Ore and waste for this year
            ore_this_year = min(remaining_ore, year_capacity)
            remaining_ore -= ore_this_year
            
            waste_this_year = 0.0
            if total_waste > 0 and total_ore > 0:
                waste_ratio = total_waste / total_ore
                waste_this_year = ore_this_year * waste_ratio
                remaining_waste -= waste_this_year
            
            if ore_this_year <= 0:
                break
            
            # Apply price/cost escalation
            price_factor = (1 + econ.price_escalation) ** (year - 1)
            cost_factor = (1 + econ.cost_escalation) ** (year - 1)
            
            metal_price = econ.metal_price * price_factor
            mining_cost_ore = econ.mining_cost_ore * cost_factor
            mining_cost_waste = econ.mining_cost_waste * cost_factor
            processing_cost = econ.processing_cost * cost_factor
            admin_cost = econ.admin_cost * cost_factor
            
            # Metal production
            metal_produced = ore_this_year * average_grade * econ.recovery_rate
            
            # Revenue
            revenue = metal_produced * metal_price
            
            # Operating costs
            mining_cost_total = (ore_this_year * mining_cost_ore + 
                               waste_this_year * mining_cost_waste)
            processing_cost_total = ore_this_year * processing_cost
            admin_cost_total = ore_this_year * admin_cost
            
            # Royalty
            if tax.royalty_type == "revenue":
                royalty = revenue * tax.royalty_rate
            else:
                royalty = 0.0  # Calculated on profit later
            
            operating_cost_total = (mining_cost_total + processing_cost_total + 
                                   admin_cost_total + royalty)
            
            # EBITDA
            ebitda = revenue - operating_cost_total
            
            # Depreciation
            depreciation = annual_depreciation if year <= tax.depreciation_years else 0.0
            
            # EBIT
            ebit = ebitda - depreciation
            
            # Tax
            taxable_income = ebit
            if tax.tax_loss_carryforward and tax_loss_pool > 0:
                if taxable_income > 0:
                    tax_recovery = min(tax_loss_pool, taxable_income)
                    taxable_income -= tax_recovery
                    tax_loss_pool -= tax_recovery
            
            if taxable_income > 0:
                tax_payable = taxable_income * tax.corporate_tax_rate
            else:
                tax_payable = 0.0
                if tax.tax_loss_carryforward:
                    tax_loss_pool += abs(ebit)
            
            # Net profit
            net_profit = ebit - tax_payable
            
            # Capex
            year_capex = (capex.sustaining_capex_annual + 
                         ore_this_year * capex.sustaining_capex_per_tonne)
            
            # Working capital change
            wc_this_year = revenue * capex.working_capital_percentage
            wc_change = wc_this_year - (previous_revenue * capex.working_capital_percentage)
            previous_revenue = revenue
            
            # Free cash flow
            fcf = net_profit + depreciation - year_capex - wc_change
            
            # Cumulative
            cumulative_cf += fcf
            
            # Discounted cash flow (mid-year convention)
            discount_factor = 1.0 / ((1 + econ.discount_rate) ** (year - 0.5))
            dcf = fcf * discount_factor
            
            flow = AnnualCashFlow(
                year=year,
                ore_tonnes=ore_this_year,
                waste_tonnes=waste_this_year,
                average_grade=average_grade,
                metal_produced=metal_produced,
                revenue=revenue,
                mining_cost=mining_cost_total,
                processing_cost=processing_cost_total,
                admin_cost=admin_cost_total,
                royalty=royalty,
                operating_cost_total=operating_cost_total,
                ebitda=ebitda,
                depreciation=depreciation,
                ebit=ebit,
                tax=tax_payable,
                net_profit=net_profit,
                capex=year_capex,
                working_capital_change=wc_change,
                free_cash_flow=fcf,
                cumulative_cash_flow=cumulative_cf,
                discounted_cash_flow=dcf
            )
            flows.append(flow)
        
        # Final year: Closure costs and working capital release
        if len(flows) > 1:
            last_year = flows[-1].year
            wc_release = previous_revenue * capex.working_capital_percentage
            closure_fcf = wc_release - capex.closure_cost
            
            discount_factor = 1.0 / ((1 + econ.discount_rate) ** (last_year + 0.5))
            
            closure_flow = AnnualCashFlow(
                year=last_year + 1,
                ore_tonnes=0,
                capex=capex.closure_cost,
                working_capital_change=-wc_release,
                free_cash_flow=closure_fcf,
                cumulative_cash_flow=cumulative_cf + closure_fcf,
                discounted_cash_flow=closure_fcf * discount_factor
            )
            flows.append(closure_flow)
        
        return flows
    
    def _calculate_summary_metrics(
        self,
        flows: List[AnnualCashFlow],
        cutoff_grade: float
    ) -> MineEconomicsResult:
        """
        Calculate summary metrics from cash flows.
        """
        # NPV (sum of discounted cash flows)
        npv = sum(f.discounted_cash_flow for f in flows)
        
        # Total metrics
        total_revenue = sum(f.revenue for f in flows)
        total_opex = sum(f.operating_cost_total for f in flows)
        total_capex = sum(f.capex for f in flows)
        total_tax = sum(f.tax for f in flows)
        total_ore = sum(f.ore_tonnes for f in flows)
        total_metal = sum(f.metal_produced for f in flows)
        
        # Mine life (years with production)
        production_years = [f for f in flows if f.ore_tonnes > 0]
        mine_life = len(production_years)
        
        # Average metrics
        avg_grade = sum(f.average_grade * f.ore_tonnes for f in flows) / total_ore if total_ore > 0 else 0
        avg_opex_per_tonne = total_opex / total_ore if total_ore > 0 else 0
        avg_aic_per_unit = (total_opex + total_capex) / total_metal if total_metal > 0 else 0
        
        # IRR calculation
        irr = self._calculate_irr(flows)
        
        # Payback periods
        payback = self._calculate_payback(flows, discounted=False)
        payback_discounted = self._calculate_payback(flows, discounted=True)
        
        result = MineEconomicsResult(
            npv=npv,
            irr=irr,
            payback_period=payback,
            payback_period_discounted=payback_discounted,
            total_revenue=total_revenue,
            total_operating_cost=total_opex,
            total_capex=total_capex,
            total_tax=total_tax,
            mine_life_years=mine_life,
            total_ore_tonnes=total_ore,
            total_metal_produced=total_metal,
            average_head_grade=avg_grade,
            average_operating_cost_per_tonne=avg_opex_per_tonne,
            average_all_in_cost_per_unit=avg_aic_per_unit,
            annual_cash_flows=flows,
            config=self.config,
            cutoff_grade=cutoff_grade
        )
        
        return result
    
    def _calculate_irr(self, flows: List[AnnualCashFlow]) -> Optional[float]:
        """
        Calculate Internal Rate of Return using numerical methods.
        """
        cash_flows = [f.free_cash_flow for f in flows]
        years = [f.year for f in flows]
        
        # Check if IRR is calculable
        if all(cf >= 0 for cf in cash_flows) or all(cf <= 0 for cf in cash_flows):
            return None
        
        def npv_at_rate(rate):
            return sum(cf / ((1 + rate) ** y) for cf, y in zip(cash_flows, years))
        
        if SCIPY_AVAILABLE:
            try:
                # Find IRR using Brent's method
                irr = brentq(npv_at_rate, -0.99, 10.0, xtol=1e-6)
                return irr
            except (ValueError, RuntimeError):
                pass
        
        # Fallback: simple iterative search
        for rate in np.linspace(-0.5, 2.0, 500):
            npv_val = npv_at_rate(rate)
            if abs(npv_val) < 1e-6:
                return rate
        
        return None
    
    def _calculate_payback(
        self,
        flows: List[AnnualCashFlow],
        discounted: bool = False
    ) -> Optional[float]:
        """
        Calculate payback period.
        """
        cumulative = 0.0
        
        for i, flow in enumerate(flows):
            cf = flow.discounted_cash_flow if discounted else flow.free_cash_flow
            prev_cumulative = cumulative
            cumulative += cf
            
            if cumulative >= 0 and prev_cumulative < 0:
                # Interpolate within year
                fraction = -prev_cumulative / (cumulative - prev_cumulative)
                return flow.year - 1 + fraction
        
        return None  # Never payback
    
    def _empty_result(self, cutoff_grade: float) -> MineEconomicsResult:
        """Return empty result for zero ore scenarios."""
        return MineEconomicsResult(
            cutoff_grade=cutoff_grade,
            config=self.config,
            metadata={"note": "No ore above cutoff"}
        )


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

class SensitivityAnalyzer:
    """
    Sensitivity analysis for mine economics.
    
    Generates tornado charts and spider diagrams for key parameters.
    """
    
    def __init__(self, base_engine: MineEconomicsEngine):
        """
        Initialize sensitivity analyzer.
        
        Args:
            base_engine: Base MineEconomicsEngine with reference configuration
        """
        self.base_engine = base_engine
        self.logger = logging.getLogger(__name__)
    
    def run_sensitivity(
        self,
        ore_tonnage: float,
        average_grade: float,
        cutoff_grade: float,
        parameters: Optional[List[str]] = None,
        variation_range: float = 0.20  # ±20%
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run sensitivity analysis on key parameters.
        
        Args:
            ore_tonnage: Base ore tonnage
            average_grade: Base average grade
            cutoff_grade: Base cutoff grade
            parameters: Parameters to vary (default: standard set)
            variation_range: Variation range (e.g., 0.20 for ±20%)
        
        Returns:
            Dict mapping parameter name to sensitivity results
        """
        if parameters is None:
            parameters = [
                "metal_price",
                "mining_cost_ore",
                "processing_cost",
                "recovery_rate",
                "discount_rate",
                "annual_ore_capacity",
                "initial_capex"
            ]
        
        # Base case
        base_result = self.base_engine.evaluate_cutoff(
            ore_tonnage, average_grade, 0.0, cutoff_grade
        )
        base_npv = base_result.npv
        
        results = {}
        
        for param in parameters:
            low_npv, high_npv = self._vary_parameter(
                param, ore_tonnage, average_grade, cutoff_grade, variation_range
            )
            
            results[param] = {
                "base_npv": base_npv,
                "low_npv": low_npv,
                "high_npv": high_npv,
                "low_change": low_npv - base_npv,
                "high_change": high_npv - base_npv,
                "swing": high_npv - low_npv,
                "low_pct": (low_npv - base_npv) / abs(base_npv) * 100 if base_npv != 0 else 0,
                "high_pct": (high_npv - base_npv) / abs(base_npv) * 100 if base_npv != 0 else 0,
                "variation_range": variation_range
            }
        
        return results
    
    def _vary_parameter(
        self,
        param: str,
        ore_tonnage: float,
        average_grade: float,
        cutoff_grade: float,
        variation: float
    ) -> Tuple[float, float]:
        """Vary a single parameter and return low/high NPVs."""
        import copy
        
        # Create modified configs
        low_config = copy.deepcopy(self.base_engine.config)
        high_config = copy.deepcopy(self.base_engine.config)
        
        # Modify parameter
        if param in ["metal_price", "mining_cost_ore", "processing_cost", 
                     "admin_cost", "recovery_rate", "discount_rate"]:
            base_val = getattr(low_config.economic_params, param)
            setattr(low_config.economic_params, param, base_val * (1 - variation))
            setattr(high_config.economic_params, param, base_val * (1 + variation))
            
        elif param == "annual_ore_capacity":
            base_val = low_config.capacity.annual_ore_capacity
            low_config.capacity.annual_ore_capacity = base_val * (1 - variation)
            high_config.capacity.annual_ore_capacity = base_val * (1 + variation)
            
        elif param == "initial_capex":
            base_val = low_config.capex.initial_capex
            low_config.capex.initial_capex = base_val * (1 - variation)
            high_config.capex.initial_capex = base_val * (1 + variation)
        
        # Evaluate
        low_engine = MineEconomicsEngine(low_config)
        high_engine = MineEconomicsEngine(high_config)
        
        low_result = low_engine.evaluate_cutoff(ore_tonnage, average_grade, 0.0, cutoff_grade)
        high_result = high_engine.evaluate_cutoff(ore_tonnage, average_grade, 0.0, cutoff_grade)
        
        return low_result.npv, high_result.npv
    
    def generate_tornado_data(
        self,
        sensitivity_results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate tornado chart data sorted by impact.
        
        Args:
            sensitivity_results: Results from run_sensitivity
        
        Returns:
            DataFrame with tornado chart data
        """
        data = []
        for param, vals in sensitivity_results.items():
            data.append({
                "parameter": param,
                "low_change": vals["low_change"],
                "high_change": vals["high_change"],
                "swing": vals["swing"],
                "low_pct": vals["low_pct"],
                "high_pct": vals["high_pct"]
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values("swing", ascending=False)
        
        return df


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_cash_flows_to_csv(result: MineEconomicsResult, filename: str) -> bool:
    """
    Export annual cash flows to CSV.
    """
    try:
        data = []
        for flow in result.annual_cash_flows:
            data.append({
                "year": flow.year,
                "ore_tonnes": flow.ore_tonnes,
                "waste_tonnes": flow.waste_tonnes,
                "average_grade": flow.average_grade,
                "metal_produced": flow.metal_produced,
                "revenue": flow.revenue,
                "mining_cost": flow.mining_cost,
                "processing_cost": flow.processing_cost,
                "admin_cost": flow.admin_cost,
                "royalty": flow.royalty,
                "operating_cost_total": flow.operating_cost_total,
                "ebitda": flow.ebitda,
                "depreciation": flow.depreciation,
                "ebit": flow.ebit,
                "tax": flow.tax,
                "net_profit": flow.net_profit,
                "capex": flow.capex,
                "working_capital_change": flow.working_capital_change,
                "free_cash_flow": flow.free_cash_flow,
                "cumulative_cash_flow": flow.cumulative_cash_flow,
                "discounted_cash_flow": flow.discounted_cash_flow
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        
        # Also export summary
        summary_file = filename.replace(".csv", "_summary.csv")
        summary = {
            "metric": ["NPV", "IRR", "Payback (years)", "Discounted Payback (years)",
                      "Total Revenue", "Total Operating Cost", "Total Capex",
                      "Mine Life (years)", "Total Ore Tonnes", "Total Metal",
                      "Average Grade", "Average OPEX/t", "All-in Cost/unit"],
            "value": [
                result.npv, result.irr, result.payback_period, result.payback_period_discounted,
                result.total_revenue, result.total_operating_cost, result.total_capex,
                result.mine_life_years, result.total_ore_tonnes, result.total_metal_produced,
                result.average_head_grade, result.average_operating_cost_per_tonne,
                result.average_all_in_cost_per_unit
            ]
        }
        pd.DataFrame(summary).to_csv(summary_file, index=False)
        
        logger.info(f"Cash flows exported to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to export cash flows: {e}")
        return False

