"""
Separated Charts Module

Provides individual chart creation functions for IRR Analysis results,
separating previously combined subplots into standalone exportable charts.
"""

from __future__ import annotations

import logging
import numpy as np
from matplotlib.figure import Figure
from typing import Dict, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


class IRRChartSeparator:
    """
    Separates combined IRR analysis charts into individual exportable figures.
    """
    
    def __init__(self):
        self.figures: Dict[str, Figure] = {}
    
    def create_convergence_charts(self, conv_history: Dict, alpha_target: float) -> Dict[str, Figure]:
        """
        Separate the 2x2 convergence subplot into 4 individual charts.
        
        Args:
            conv_history: Convergence history data
            alpha_target: Target alpha value
        
        Returns:
            Dictionary of {chart_id: figure}
        """
        charts = {}
        
        # Chart 1: IRR Convergence
        fig1 = Figure(figsize=(8, 6))
        ax1 = fig1.add_subplot(111)
        ax1.plot(conv_history['iterations'], np.array(conv_history['r_values']) * 100, 
                marker='o', linewidth=2, markersize=6, color='#2196F3', label='Trial IRR')
        ax1.fill_between(conv_history['iterations'], 
                        np.array(conv_history['r_low_values']) * 100,
                        np.array(conv_history['r_high_values']) * 100,
                        alpha=0.2, color='#2196F3', label='Search Bracket')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Discount Rate (%)', fontsize=12)
        ax1.set_title('IRR Convergence Analysis', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        charts['chart_irr_convergence'] = fig1
        
        # Chart 2: Satisfaction Rate
        fig2 = Figure(figsize=(8, 6))
        ax2 = fig2.add_subplot(111)
        sat_rates = np.array(conv_history['satisfaction_rates']) * 100
        ax2.plot(conv_history['iterations'], sat_rates, 
                marker='s', linewidth=2, markersize=6, color='#4CAF50', label='Satisfaction Rate')
        ax2.axhline(alpha_target * 100, color='red', linestyle='--', linewidth=2, 
                   label=f'Target: {alpha_target * 100:.0f}%')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Satisfaction Rate (%)', fontsize=12)
        ax2.set_title('Scenario Satisfaction Rate Evolution', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        fig2.tight_layout()
        charts['chart_satisfaction_rate'] = fig2
        
        # Chart 3: Mean NPV Evolution
        fig3 = Figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111)
        mean_npvs = np.array(conv_history['mean_npvs']) / 1e6
        ax3.plot(conv_history['iterations'], mean_npvs, 
                marker='D', linewidth=2, markersize=6, color='#FF9800', label='Mean NPV')
        ax3.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Mean NPV ($M)', fontsize=12)
        ax3.set_title('Mean NPV Evolution', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        fig3.tight_layout()
        charts['chart_mean_npv'] = fig3
        
        # Chart 4: Convergence Speed
        fig4 = Figure(figsize=(8, 6))
        ax4 = fig4.add_subplot(111)
        bracket_width = (np.array(conv_history['r_high_values']) - 
                        np.array(conv_history['r_low_values'])) * 100
        ax4.semilogy(conv_history['iterations'], bracket_width, 
                    marker='v', linewidth=2, markersize=6, color='#9C27B0', label='Bracket Width')
        ax4.set_xlabel('Iteration', fontsize=12)
        ax4.set_ylabel('Search Bracket Width (% points, log scale)', fontsize=12)
        ax4.set_title('Convergence Speed Analysis', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        charts['chart_convergence_speed'] = fig4
        
        self.figures.update(charts)
        logger.info(f"Created {len(charts)} individual convergence charts")
        return charts
    
    def create_scenario_distribution_charts(self, npvs: np.ndarray, alpha_target: float) -> Dict[str, Figure]:
        """
        Separate the 2x1 scenario distribution subplot into 2 individual charts.
        
        Args:
            npvs: NPV values array (in dollars, will be converted to millions)
            alpha_target: Target alpha value
        
        Returns:
            Dictionary of {chart_id: figure}
        """
        charts = {}
        npvs_m = npvs / 1e6  # Convert to millions
        
        # Chart 1: NPV Distribution Histogram
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        n, bins, patches = ax1.hist(npvs_m, bins=40, edgecolor='black', alpha=0.7)
        
        # Color code bins
        for i, patch in enumerate(patches):
            if bins[i] < 0:
                patch.set_facecolor('#EF5350')  # Red for negative
            else:
                patch.set_facecolor('#66BB6A')  # Green for positive
        
        # Statistics
        mean_npv = np.mean(npvs_m)
        median_npv = np.median(npvs_m)
        std_npv = np.std(npvs_m)
        p10 = np.percentile(npvs_m, 10)
        p90 = np.percentile(npvs_m, 90)
        
        # Add vertical lines
        ax1.axvline(0, color='black', linestyle='-', linewidth=2, label='Break-even', alpha=0.8)
        ax1.axvline(mean_npv, color='red', linestyle='--', linewidth=2, label=f'Mean: ${mean_npv:.1f}M')
        ax1.axvline(median_npv, color='blue', linestyle='--', linewidth=2, label=f'Median: ${median_npv:.1f}M')
        ax1.axvline(p10, color='orange', linestyle=':', linewidth=1.5, label=f'P10: ${p10:.1f}M', alpha=0.7)
        ax1.axvline(p90, color='purple', linestyle=':', linewidth=1.5, label=f'P90: ${p90:.1f}M', alpha=0.7)
        
        ax1.set_xlabel('NPV ($M)', fontsize=12)
        ax1.set_ylabel('Number of Scenarios', fontsize=12)
        ax1.set_title(f'Scenario NPV Distribution ({len(npvs_m)} scenarios)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Statistics text box
        stats_text = f'μ = ${mean_npv:.2f}M\nσ = ${std_npv:.2f}M\nP(NPV≥0) = {np.sum(npvs_m >= 0)/len(npvs_m)*100:.1f}%'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig1.tight_layout()
        charts['chart_npv_distribution'] = fig1
        
        # Chart 2: Cumulative Distribution Function
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        
        sorted_npvs = np.sort(npvs_m)
        cumulative = np.arange(1, len(sorted_npvs) + 1) / len(sorted_npvs) * 100
        
        ax2.plot(sorted_npvs, cumulative, color='darkblue', linewidth=3, label='CDF')
        ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax2.axhline(50, color='gray', linestyle=':', alpha=0.5, label='Median')
        ax2.axhline(alpha_target * 100, color='green', 
                   linestyle='--', linewidth=2, alpha=0.7, label=f'Target α = {alpha_target*100:.0f}%')
        
        # Grid at key percentiles
        for p in [10, 25, 50, 75, 90]:
            ax2.axhline(p, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
        
        ax2.set_xlabel('NPV ($M)', fontsize=12)
        ax2.set_ylabel('Cumulative Probability (%)', fontsize=12)
        ax2.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 100])
        
        fig2.tight_layout()
        charts['chart_npv_cdf'] = fig2
        
        self.figures.update(charts)
        logger.info(f"Created {len(charts)} individual scenario distribution charts")
        return charts
    
    def create_cashflow_charts(self, cashflows: np.ndarray, discount_rate: float, 
                               npv_details: Optional[Dict] = None) -> Dict[str, Figure]:
        """
        Separate the 3x1 cashflow subplot into 3 individual charts.
        
        Args:
            cashflows: Cash flow values for each period
            discount_rate: Discount rate used
            npv_details: Optional NPV component breakdown
        
        Returns:
            Dictionary of {chart_id: figure}
        """
        charts = {}
        periods = np.arange(len(cashflows))
        
        # Calculate discounted cash flows
        discount_factors = 1.0 / np.power(1.0 + discount_rate, periods)
        discounted_cashflows = cashflows * discount_factors
        cumulative_npv = np.cumsum(discounted_cashflows)
        
        # Chart 1: Period Cash Flows
        fig1 = Figure(figsize=(10, 6))
        ax1 = fig1.add_subplot(111)
        
        width = 0.35
        x = periods
        ax1.bar(x - width/2, cashflows / 1e6, width, label='Undiscounted CF', color='#64B5F6', alpha=0.8)
        ax1.bar(x + width/2, discounted_cashflows / 1e6, width, label='Discounted CF', color='#1976D2', alpha=0.8)
        ax1.axhline(0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Period', fontsize=12)
        ax1.set_ylabel('Cash Flow ($M)', fontsize=12)
        ax1.set_title(f'Period Cash Flows (Discount Rate: {discount_rate*100:.2f}%)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3, axis='y')
        
        fig1.tight_layout()
        charts['chart_period_cashflows'] = fig1
        
        # Chart 2: Cumulative NPV
        fig2 = Figure(figsize=(10, 6))
        ax2 = fig2.add_subplot(111)
        
        ax2.plot(periods, cumulative_npv / 1e6, marker='o', linewidth=2.5, markersize=7, 
                color='#4CAF50', label='Cumulative NPV')
        ax2.fill_between(periods, 0, cumulative_npv / 1e6, alpha=0.3, color='#4CAF50')
        ax2.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Period', fontsize=12)
        ax2.set_ylabel('Cumulative NPV ($M)', fontsize=12)
        ax2.set_title(f'Cumulative NPV Evolution (Final: ${cumulative_npv[-1]/1e6:.2f}M)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        fig2.tight_layout()
        charts['chart_cumulative_npv'] = fig2
        
        # Chart 3: NPV Components
        fig3 = Figure(figsize=(10, 6))
        ax3 = fig3.add_subplot(111)
        
        if npv_details:
            total_npv = npv_details.get('npv', 0) / 1e6
            primary_rev = npv_details.get('primary_revenue', 0) / 1e6
            byproduct_rev = npv_details.get('byproduct_revenue', 0) / 1e6
            opex = npv_details.get('operating_cost', 0) / 1e6
            capex = npv_details.get('capital_cost', 0) / 1e6
            
            categories = ['Primary\nRevenue', 'By-Product\nRevenue', 'Operating\nCost', 'Capital\nCost', 'Net NPV']
            values = [primary_rev, byproduct_rev, -opex, -capex, total_npv]
            colors = ['#66BB6A', '#81C784', '#EF5350', '#E57373', '#2196F3']
        else:
            # Fallback
            positive_cf = np.sum(cashflows[cashflows > 0]) / 1e6
            negative_cf = np.sum(cashflows[cashflows < 0]) / 1e6
            net_cf = positive_cf + negative_cf
            
            categories = ['Total\nRevenue', 'Total\nCosts', 'Net\nCash Flow']
            values = [positive_cf, negative_cf, net_cf]
            colors = ['#66BB6A', '#EF5350', '#2196F3']
        
        bars = ax3.bar(categories, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.axhline(0, color='black', linestyle='-', linewidth=1)
        ax3.set_ylabel('Value ($M)', fontsize=12)
        ax3.set_title('NPV Components Breakdown', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.1f}M',
                    ha='center', va='bottom' if height > 0 else 'top', 
                    fontsize=9, fontweight='bold')
        
        fig3.tight_layout()
        charts['chart_npv_components'] = fig3
        
        self.figures.update(charts)
        logger.info(f"Created {len(charts)} individual cashflow charts")
        return charts
    
    def get_all_figures(self) -> Dict[str, Figure]:
        """Get all created figures."""
        return self.figures
    
    def clear_figures(self):
        """Clear all stored figures."""
        for fig in self.figures.values():
            fig.clear()
        self.figures.clear()
        logger.info("Cleared all separated chart figures")

