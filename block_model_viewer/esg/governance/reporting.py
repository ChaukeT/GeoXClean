"""
ESG Governance & Reporting Module

Generates ESG reports compliant with major frameworks:
- GRI (Global Reporting Initiative)
- ICMM (International Council on Mining and Metals)
- TCFD (Task Force on Climate-related Financial Disclosures)
- SASB (Sustainability Accounting Standards Board)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class ReportingFramework(Enum):
    """Major ESG reporting frameworks"""
    GRI = "gri"  # Global Reporting Initiative
    ICMM = "icmm"  # International Council on Mining and Metals
    TCFD = "tcfd"  # Task Force on Climate-related Financial Disclosures
    SASB = "sasb"  # Sustainability Accounting Standards Board
    CDP = "cdp"  # Carbon Disclosure Project
    DJSI = "djsi"  # Dow Jones Sustainability Index


@dataclass
class ESGMetrics:
    """
    Consolidated ESG metrics for reporting.
    
    Attributes:
        reporting_period: Period identifier
        carbon_metrics: Carbon/energy metrics
        water_metrics: Water management metrics
        waste_metrics: Waste and land metrics
        social_metrics: Social performance metrics
        governance_metrics: Governance indicators
    """
    reporting_period: str
    carbon_metrics: Dict[str, float] = field(default_factory=dict)
    water_metrics: Dict[str, float] = field(default_factory=dict)
    waste_metrics: Dict[str, float] = field(default_factory=dict)
    social_metrics: Dict[str, Any] = field(default_factory=dict)
    governance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """
    Compliance report output.
    
    Attributes:
        framework: Reporting framework used
        report_date: Date of report generation
        metrics: ESG metrics
        disclosures: List of disclosure items
        assurance_level: Assurance level (none, limited, reasonable)
        compliance_score: Overall compliance score (0-100)
        gaps: List of identified gaps
    """
    framework: ReportingFramework
    report_date: datetime
    metrics: ESGMetrics
    disclosures: List[Dict[str, Any]] = field(default_factory=list)
    assurance_level: str = "none"
    compliance_score: float = 0.0
    gaps: List[str] = field(default_factory=list)


def generate_gri_report(
    metrics: ESGMetrics,
    materiality_topics: List[str] = None
) -> ComplianceReport:
    """
    Generate GRI (Global Reporting Initiative) compliant report.
    
    GRI Standards focus on:
    - Universal Standards (GRI 1, 2, 3)
    - Topic Standards (200-400 series)
    - Sector Standards (e.g., Mining and Metals)
    
    Key disclosures for mining:
    - GRI 302: Energy
    - GRI 303: Water and Effluents
    - GRI 305: Emissions
    - GRI 306: Waste
    - GRI 403: Occupational Health and Safety
    - GRI 413: Local Communities
    
    Args:
        metrics: ESG metrics to report
        materiality_topics: Material topics identified through stakeholder engagement
    
    Returns:
        ComplianceReport with GRI disclosures
    """
    logger.info(f"Generating GRI report for period {metrics.reporting_period}")
    
    if materiality_topics is None:
        materiality_topics = [
            "Climate Change",
            "Water Management",
            "Waste & Tailings",
            "Biodiversity",
            "Community Relations",
            "Health & Safety"
        ]
    
    disclosures = []
    
    # GRI 302: Energy
    energy_disclosure = {
        'standard': 'GRI 302-1',
        'title': 'Energy consumption within the organization',
        'data': {
            'total_energy_mwh': metrics.carbon_metrics.get('total_energy_mwh', 0),
            'diesel_energy_mwh': metrics.carbon_metrics.get('diesel_energy_mwh', 0),
            'electricity_mwh': metrics.carbon_metrics.get('electricity_mwh', 0),
            'energy_intensity_mwh_per_t': metrics.carbon_metrics.get('energy_intensity', 0)
        }
    }
    disclosures.append(energy_disclosure)
    
    # GRI 303: Water
    water_disclosure = {
        'standard': 'GRI 303-3',
        'title': 'Water withdrawal',
        'data': {
            'total_water_withdrawal_m3': metrics.water_metrics.get('total_water_use_m3', 0),
            'recycled_water_m3': metrics.water_metrics.get('recycled_water_m3', 0),
            'water_intensity_m3_per_t': metrics.water_metrics.get('water_intensity_m3_per_t', 0),
            'recycling_rate': metrics.water_metrics.get('recycling_rate', 0)
        }
    }
    disclosures.append(water_disclosure)
    
    # GRI 305: Emissions
    emissions_disclosure = {
        'standard': 'GRI 305-1',
        'title': 'Direct (Scope 1) GHG emissions',
        'data': {
            'scope1_t_co2e': metrics.carbon_metrics.get('scope1_t_co2e', 0),
            'scope2_t_co2e': metrics.carbon_metrics.get('scope2_t_co2e', 0),
            'total_co2e_t': metrics.carbon_metrics.get('total_co2e_t', 0),
            'co2e_intensity_kg_per_t': metrics.carbon_metrics.get('co2e_intensity_kg_per_t', 0)
        }
    }
    disclosures.append(emissions_disclosure)
    
    # GRI 306: Waste
    waste_disclosure = {
        'standard': 'GRI 306-3',
        'title': 'Waste generated',
        'data': {
            'waste_rock_t': metrics.waste_metrics.get('waste_generated_t', 0),
            'tailings_t': metrics.waste_metrics.get('tailings_t', 0),
            'hazardous_waste_t': metrics.waste_metrics.get('hazardous_waste_t', 0)
        }
    }
    disclosures.append(waste_disclosure)
    
    # Calculate compliance score
    required_disclosures = 20  # Simplified
    provided_disclosures = len(disclosures)
    compliance_score = min(100, (provided_disclosures / required_disclosures) * 100)
    
    # Identify gaps
    gaps = []
    if metrics.social_metrics.get('safety_metrics') is None:
        gaps.append("GRI 403: Occupational Health and Safety metrics not provided")
    if metrics.social_metrics.get('community_metrics') is None:
        gaps.append("GRI 413: Local Communities metrics not provided")
    
    report = ComplianceReport(
        framework=ReportingFramework.GRI,
        report_date=datetime.now(),
        metrics=metrics,
        disclosures=disclosures,
        assurance_level="none",
        compliance_score=compliance_score,
        gaps=gaps
    )
    
    logger.info(f"GRI report generated: {len(disclosures)} disclosures, {compliance_score:.1f}% compliance")
    return report


def generate_icmm_report(
    metrics: ESGMetrics,
    performance_expectations: List[str] = None
) -> ComplianceReport:
    """
    Generate ICMM (International Council on Mining and Metals) report.
    
    ICMM Mining Principles:
    1. Ethical business
    2. Decision-making
    3. Human rights
    4. Risk management
    5. Health and safety
    6. Environmental performance
    7. Conservation of biodiversity
    8. Responsible production
    9. Social performance
    10. Stakeholder engagement
    
    Args:
        metrics: ESG metrics
        performance_expectations: Specific performance expectations addressed
    
    Returns:
        ComplianceReport with ICMM alignment
    """
    logger.info(f"Generating ICMM report for period {metrics.reporting_period}")
    
    if performance_expectations is None:
        performance_expectations = [
            "Climate Change",
            "Water Stewardship",
            "Tailings Management",
            "Indigenous Peoples",
            "Mine Closure"
        ]
    
    disclosures = []
    
    # Principle 6: Environmental Performance
    env_disclosure = {
        'principle': 'ICMM Principle 6',
        'title': 'Environmental Performance',
        'data': {
            'co2e_intensity': metrics.carbon_metrics.get('co2e_intensity_kg_per_t', 0),
            'water_recycling_rate': metrics.water_metrics.get('recycling_rate', 0),
            'disturbed_area_ha': metrics.waste_metrics.get('disturbed_area_ha', 0),
            'rehabilitated_area_ha': metrics.waste_metrics.get('rehabilitated_area_ha', 0)
        }
    }
    disclosures.append(env_disclosure)
    
    # Principle 7: Biodiversity
    biodiversity_disclosure = {
        'principle': 'ICMM Principle 7',
        'title': 'Conservation of Biodiversity',
        'data': {
            'biodiversity_index': metrics.waste_metrics.get('biodiversity_index', 0),
            'protected_areas_ha': metrics.waste_metrics.get('protected_areas_ha', 0)
        }
    }
    disclosures.append(biodiversity_disclosure)
    
    compliance_score = 70.0  # Simplified scoring
    gaps = ["Full ICMM self-assessment required for complete compliance"]
    
    report = ComplianceReport(
        framework=ReportingFramework.ICMM,
        report_date=datetime.now(),
        metrics=metrics,
        disclosures=disclosures,
        assurance_level="none",
        compliance_score=compliance_score,
        gaps=gaps
    )
    
    logger.info(f"ICMM report generated: {len(disclosures)} disclosures")
    return report


def generate_tcfd_report(
    metrics: ESGMetrics,
    climate_scenarios: List[str] = None
) -> ComplianceReport:
    """
    Generate TCFD (Task Force on Climate-related Financial Disclosures) report.
    
    TCFD Four Pillars:
    1. Governance: Board oversight of climate risks
    2. Strategy: Climate risks and opportunities
    3. Risk Management: Climate risk identification and management
    4. Metrics and Targets: Metrics used to assess climate risks
    
    Args:
        metrics: ESG metrics
        climate_scenarios: Climate scenarios analyzed (e.g., 1.5°C, 2°C, 4°C)
    
    Returns:
        ComplianceReport with TCFD disclosures
    """
    logger.info(f"Generating TCFD report for period {metrics.reporting_period}")
    
    if climate_scenarios is None:
        climate_scenarios = ["2°C Scenario", "4°C Scenario"]
    
    disclosures = []
    
    # Metrics and Targets
    metrics_disclosure = {
        'pillar': 'Metrics and Targets',
        'title': 'GHG emissions and climate metrics',
        'data': {
            'scope1_co2e_t': metrics.carbon_metrics.get('scope1_t_co2e', 0),
            'scope2_co2e_t': metrics.carbon_metrics.get('scope2_t_co2e', 0),
            'scope3_co2e_t': metrics.carbon_metrics.get('scope3_t_co2e', 0),
            'total_co2e_t': metrics.carbon_metrics.get('total_co2e_t', 0),
            'emissions_intensity': metrics.carbon_metrics.get('co2e_intensity_kg_per_t', 0),
            'energy_consumption_mwh': metrics.carbon_metrics.get('total_energy_mwh', 0),
            'renewable_energy_percent': metrics.carbon_metrics.get('renewable_energy_percent', 0)
        }
    }
    disclosures.append(metrics_disclosure)
    
    # Strategy (simplified)
    strategy_disclosure = {
        'pillar': 'Strategy',
        'title': 'Climate-related risks and opportunities',
        'data': {
            'transition_risks': ['Carbon pricing', 'Energy costs', 'Technology transition'],
            'physical_risks': ['Water scarcity', 'Extreme weather', 'Temperature increase'],
            'opportunities': ['Energy efficiency', 'Low-carbon technology', 'Renewable energy'],
            'scenarios_analyzed': climate_scenarios
        }
    }
    disclosures.append(strategy_disclosure)
    
    compliance_score = 60.0  # TCFD requires narrative disclosures
    gaps = [
        "Governance structure for climate oversight not detailed",
        "Climate scenario analysis not quantified",
        "Risk management integration not described"
    ]
    
    report = ComplianceReport(
        framework=ReportingFramework.TCFD,
        report_date=datetime.now(),
        metrics=metrics,
        disclosures=disclosures,
        assurance_level="none",
        compliance_score=compliance_score,
        gaps=gaps
    )
    
    logger.info(f"TCFD report generated: {len(disclosures)} disclosures")
    return report


def generate_sasb_report(
    metrics: ESGMetrics,
    industry: str = "Metals & Mining"
) -> ComplianceReport:
    """
    Generate SASB (Sustainability Accounting Standards Board) report.
    
    SASB Metals & Mining Standard (EM-MM):
    - GHG Emissions (EM-MM-110a.1)
    - Energy Management (EM-MM-130a.1)
    - Water Management (EM-MM-140a.1)
    - Waste & Hazardous Materials (EM-MM-150a.1)
    - Biodiversity Impacts (EM-MM-160a.1)
    - Security, Human Rights & Rights of Indigenous Peoples (EM-MM-210a.1)
    - Community Relations (EM-MM-210b.1)
    - Labor Relations (EM-MM-310a.1)
    - Workforce Health & Safety (EM-MM-320a.1)
    
    Args:
        metrics: ESG metrics
        industry: SASB industry classification
    
    Returns:
        ComplianceReport with SASB metrics
    """
    logger.info(f"Generating SASB report for {industry}, period {metrics.reporting_period}")
    
    disclosures = []
    
    # EM-MM-110a.1: GHG Emissions
    ghg_disclosure = {
        'metric': 'EM-MM-110a.1',
        'title': 'Gross global Scope 1 emissions',
        'unit': 'Metric tons CO2e',
        'value': metrics.carbon_metrics.get('scope1_t_co2e', 0)
    }
    disclosures.append(ghg_disclosure)
    
    # EM-MM-130a.1: Energy Management
    energy_disclosure = {
        'metric': 'EM-MM-130a.1',
        'title': 'Total energy consumed',
        'unit': 'MWh',
        'value': metrics.carbon_metrics.get('total_energy_mwh', 0)
    }
    disclosures.append(energy_disclosure)
    
    # EM-MM-140a.1: Water Management
    water_disclosure = {
        'metric': 'EM-MM-140a.1',
        'title': 'Total fresh water withdrawn',
        'unit': 'm³',
        'value': metrics.water_metrics.get('fresh_water_use_m3', 0)
    }
    disclosures.append(water_disclosure)
    
    # EM-MM-140a.2: Water recycled
    water_recycled_disclosure = {
        'metric': 'EM-MM-140a.2',
        'title': 'Percentage recycled',
        'unit': '%',
        'value': metrics.water_metrics.get('recycling_rate', 0) * 100
    }
    disclosures.append(water_recycled_disclosure)
    
    # EM-MM-150a.1: Waste
    waste_disclosure = {
        'metric': 'EM-MM-150a.1',
        'title': 'Total weight of tailings produced',
        'unit': 'Metric tons',
        'value': metrics.waste_metrics.get('tailings_t', 0)
    }
    disclosures.append(waste_disclosure)
    
    # EM-MM-160a.1: Biodiversity
    biodiversity_disclosure = {
        'metric': 'EM-MM-160a.1',
        'title': 'Area of land disturbed',
        'unit': 'Hectares',
        'value': metrics.waste_metrics.get('disturbed_area_ha', 0)
    }
    disclosures.append(biodiversity_disclosure)
    
    required_metrics = 26  # Full SASB Metals & Mining standard
    provided_metrics = len(disclosures)
    compliance_score = (provided_metrics / required_metrics) * 100
    
    gaps = []
    if metrics.social_metrics.get('safety_metrics') is None:
        gaps.append("EM-MM-320a.1: Workforce Health & Safety metrics not provided")
    if metrics.social_metrics.get('community_metrics') is None:
        gaps.append("EM-MM-210b.1: Community Relations metrics not provided")
    
    report = ComplianceReport(
        framework=ReportingFramework.SASB,
        report_date=datetime.now(),
        metrics=metrics,
        disclosures=disclosures,
        assurance_level="none",
        compliance_score=compliance_score,
        gaps=gaps
    )
    
    logger.info(f"SASB report generated: {len(disclosures)} metrics, {compliance_score:.1f}% compliance")
    return report


def create_audit_trail(
    metrics: ESGMetrics,
    data_sources: Dict[str, str],
    methodology: str = "Activity-based calculation"
) -> Dict[str, Any]:
    """
    Create audit trail for ESG data.
    
    Documents:
    - Data sources
    - Calculation methodology
    - Assumptions
    - Uncertainty
    - Verification status
    
    Args:
        metrics: ESG metrics
        data_sources: Data source documentation
        methodology: Calculation methodology description
    
    Returns:
        Audit trail dictionary
    """
    audit_trail = {
        'reporting_period': metrics.reporting_period,
        'generated_date': datetime.now().isoformat(),
        'methodology': methodology,
        'data_sources': data_sources,
        'metrics_summary': {
            'carbon': len(metrics.carbon_metrics),
            'water': len(metrics.water_metrics),
            'waste': len(metrics.waste_metrics),
            'social': len(metrics.social_metrics),
            'governance': len(metrics.governance_metrics)
        },
        'verification': {
            'status': 'unverified',
            'verifier': None,
            'date': None
        },
        'assumptions': [
            'Emission factors from IPCC 2021',
            'Water recycling rates from operational data',
            'Waste classifications per regulatory requirements'
        ]
    }
    
    return audit_trail


def export_to_json(
    report: ComplianceReport,
    filepath: str
) -> bool:
    """
    Export compliance report to JSON format.
    
    Args:
        report: Compliance report to export
        filepath: Output file path
    
    Returns:
        True if successful
    """
    try:
        report_dict = {
            'framework': report.framework.value,
            'report_date': report.report_date.isoformat(),
            'reporting_period': report.metrics.reporting_period,
            'disclosures': report.disclosures,
            'compliance_score': report.compliance_score,
            'gaps': report.gaps,
            'assurance_level': report.assurance_level
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Report exported to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Failed to export report: {e}")
        return False
