"""
Example Usage of Uncertainty Analysis Engine

Demonstrates Monte Carlo simulation, Bootstrap analysis, LHS sampling,
and dashboard generation with synthetic block model data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import uncertainty engine components
from block_model_viewer.uncertainty_engine import (
    MonteCarloSimulator, MonteCarloConfig, SimulationMode,
    ParameterDistribution, DistributionType,
    BootstrapAnalyzer, BootstrapConfig,
    LHSSampler, LHSConfig,
    UncertaintyDashboard, DashboardConfig,
    generate_lhs_samples, bootstrap_ci
)


def create_synthetic_block_model(n_blocks=1000):
    """Create synthetic block model for testing."""
    np.random.seed(42)
    
    # Generate random block positions
    x = np.random.uniform(0, 1000, n_blocks)
    y = np.random.uniform(0, 1000, n_blocks)
    z = np.random.uniform(0, 500, n_blocks)
    
    # Block dimensions
    dx = np.full(n_blocks, 25.0)
    dy = np.full(n_blocks, 25.0)
    dz = np.full(n_blocks, 12.5)
    
    # Tonnage
    volume = dx * dy * dz
    density = np.random.normal(2.7, 0.1, n_blocks)
    tonnage = volume * density
    
    # Grade (with spatial correlation - simple gradient)
    grade_base = 60.0 + 0.01 * x - 0.02 * z + np.random.normal(0, 2, n_blocks)
    grade = np.clip(grade_base, 50, 70)
    
    # Generate grade realizations (for probabilistic analysis)
    n_realizations = 20
    for i in range(n_realizations):
        realization = grade + np.random.normal(0, 1.5, n_blocks)
        df_temp = pd.DataFrame({f'grade_real_{i}': np.clip(realization, 50, 70)})
        if i == 0:
            block_df = pd.DataFrame({
                'x': x, 'y': y, 'z': z,
                'dx': dx, 'dy': dy, 'dz': dz,
                'tonnage': tonnage,
                'grade': grade,
                'density': density
            })
        block_df[f'grade_real_{i}'] = df_temp[f'grade_real_{i}']
    
    return block_df


def example_1_bootstrap_ci():
    """Example 1: Bootstrap confidence intervals for grade statistics."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Bootstrap Confidence Intervals")
    print("="*80)
    
    # Create block model
    block_df = create_synthetic_block_model(1000)
    
    # Configure bootstrap
    config = BootstrapConfig(
        n_iterations=1000,
        confidence_level=0.95,
        method='simple',
        random_seed=42
    )
    
    analyzer = BootstrapAnalyzer(config)
    
    # Analyze grade mean
    grade_data = block_df['grade'].values
    result = analyzer.analyze_statistic(grade_data, np.mean, 'grade_mean')
    
    print("\nGrade Mean Analysis:")
    print(result)
    
    # Analyze multiple statistics
    statistics = {
        'mean_grade': ('grade', np.mean),
        'std_grade': ('grade', np.std),
        'p90_grade': ('grade', lambda x: np.percentile(x, 90)),
        'mean_tonnage': ('tonnage', np.mean)
    }
    
    results = analyzer.analyze_dataframe(block_df, statistics)
    
    print("\nMultiple Statistics:")
    for name, result in results.items():
        print(f"{name}: {result.original_statistic:.2f} "
              f"CI=[{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
    
    # Quick CI function
    ci_lower, ci_upper = bootstrap_ci(grade_data, np.mean, n_iterations=1000)
    print(f"\nQuick CI: [{ci_lower:.2f}, {ci_upper:.2f}]")


def example_2_lhs_sampling():
    """Example 2: Latin Hypercube Sampling for parameter exploration."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Latin Hypercube Sampling")
    print("="*80)
    
    # Define parameter specifications
    param_specs = {
        'price': {
            'type': 'normal',
            'params': {'mean': 100.0, 'std': 10.0}
        },
        'mining_cost': {
            'type': 'uniform',
            'params': {'min': 40.0, 'max': 60.0}
        },
        'recovery': {
            'type': 'triangular',
            'params': {'min': 0.80, 'mode': 0.85, 'max': 0.90}
        },
        'discount_rate': {
            'type': 'uniform',
            'params': {'min': 0.08, 'max': 0.12}
        }
    }
    
    # Generate samples
    samples_df = generate_lhs_samples(
        n_samples=100,
        param_specs=param_specs,
        random_seed=42
    )
    
    print(f"\nGenerated {len(samples_df)} LHS samples")
    print("\nFirst 5 samples:")
    print(samples_df.head())
    
    print("\nSummary statistics:")
    print(samples_df.describe())
    
    # Check correlation (should be near zero for independent sampling)
    print("\nCorrelation matrix (should be near zero):")
    print(samples_df.corr().round(3))
    
    # Export
    output_path = Path('lhs_samples.csv')
    samples_df.to_csv(output_path, index=False)
    print(f"\nExported to: {output_path}")


def example_3_monte_carlo_simple():
    """Example 3: Simple Monte Carlo with fixed schedule."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Monte Carlo Simulation (Fixed Schedule Mode)")
    print("="*80)
    
    # Create block model
    block_df = create_synthetic_block_model(500)
    
    # Create simple fixed schedule (mine blocks in order)
    n_periods = 10
    blocks_per_period = len(block_df) // n_periods
    
    schedule = pd.DataFrame({
        'block_id': block_df.index,
        'period': [i // blocks_per_period for i in range(len(block_df))]
    })
    schedule['period'] = schedule['period'].clip(0, n_periods - 1)
    
    # Define parameter uncertainty
    price_dist = ParameterDistribution(
        name='price',
        distribution=DistributionType.NORMAL,
        base_value=100.0,
        std_dev=15.0
    )
    
    recovery_dist = ParameterDistribution(
        name='recovery',
        distribution=DistributionType.TRIANGULAR,
        base_value=0.85,
        min_value=0.80,
        mode_value=0.85,
        max_value=0.90
    )
    
    discount_dist = ParameterDistribution(
        name='discount_rate',
        distribution=DistributionType.UNIFORM,
        base_value=0.10,
        min_value=0.08,
        max_value=0.12
    )
    
    # Configure Monte Carlo
    config = MonteCarloConfig(
        n_simulations=100,  # Small for demo
        mode=SimulationMode.FIXED_SCHEDULE,
        parameters={
            'price': price_dist,
            'recovery': recovery_dist,
            'discount_rate': discount_dist
        },
        parallel=False,  # Sequential for demo
        random_seed=42,
        track_annual_metrics=True
    )
    
    # Run simulation
    simulator = MonteCarloSimulator(config)
    
    print("\nRunning Monte Carlo simulation...")
    
    def progress_callback(current, total):
        if current % 20 == 0 or current == total:
            print(f"  Progress: {current}/{total} simulations")
    
    results = simulator.run(
        block_model=block_df,
        base_schedule=schedule,
        progress_callback=progress_callback
    )
    
    # Display results
    print(f"\nCompleted {results.n_successful} simulations")
    
    print("\nSummary Statistics:")
    print(results.summary_stats)
    
    print(f"\nRisk Metrics:")
    print(f"  Probability of NPV < 0: {results.probability_of_loss:.1%}")
    if results.value_at_risk:
        print(f"  VaR 95%: ${results.value_at_risk['VaR_95']/1e6:.2f}M")
        print(f"  CVaR 95%: ${results.value_at_risk['CVaR_95']/1e6:.2f}M")
    
    if results.input_output_correlation is not None:
        print("\nInput-Output Correlations:")
        print(results.input_output_correlation.round(3))
    
    return results


def example_4_dashboard_generation(mc_results):
    """Example 4: Generate risk dashboard from Monte Carlo results."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Risk Dashboard Generation")
    print("="*80)
    
    # Create dashboard
    dashboard = UncertaintyDashboard()
    
    # Create summary table
    summary = dashboard.create_summary_table(mc_results)
    print("\nKPI Summary Table:")
    print(summary)
    
    # Generate all dashboard figures
    print("\nGenerating dashboard figures...")
    output_dir = Path('uncertainty_dashboard')
    output_dir.mkdir(exist_ok=True)
    
    figures = dashboard.create_full_dashboard(mc_results, output_dir)
    
    print(f"\nGenerated {len(figures)} figures:")
    for name in figures.keys():
        print(f"  - {name}")
    
    print(f"\nDashboard exported to: {output_dir}")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("UNCERTAINTY ANALYSIS ENGINE - EXAMPLES")
    print("="*80)
    
    # Example 1: Bootstrap CI
    example_1_bootstrap_ci()
    
    # Example 2: LHS Sampling
    example_2_lhs_sampling()
    
    # Example 3: Monte Carlo
    mc_results = example_3_monte_carlo_simple()
    
    # Example 4: Dashboard
    try:
        example_4_dashboard_generation(mc_results)
    except Exception as e:
        print(f"\nDashboard generation skipped (matplotlib may not be available): {e}")
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print("\nCheck output files:")
    print("  - lhs_samples.csv")
    print("  - uncertainty_dashboard/ (if matplotlib available)")


if __name__ == "__main__":
    main()
