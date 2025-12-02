"""
Policy Simulation Module for Government Debt Analysis

This module implements simulation methods to compare policy outcomes
for France against benchmarks from Portugal and Ireland.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class PolicySimulator:
    """
    Simulate and compare policy outcomes for government debt analysis.
    
    Compares France's economic trajectory under different policy scenarios,
    using Portugal and Ireland as benchmark countries.
    """
    
    # Policy intervention parameters based on historical evidence
    POLICY_EFFECTS = {
        'fiscal_consolidation': {
            'debt_reduction_annual': -0.02,  # 2% annual debt-to-GDP reduction
            'unemployment_impact': 0.001,     # Slight initial unemployment increase
            'gdp_impact': -0.005              # Small initial GDP impact
        },
        'structural_reforms': {
            'productivity_gain': 0.01,
            'unemployment_reduction': -0.02,
            'debt_reduction_annual': -0.01
        },
        'pandemic_recovery': {
            'gdp_recovery_rate': 0.03,
            'debt_increase_initial': 0.10,
            'gradual_recovery': 0.02
        }
    }
    
    def __init__(self):
        """Initialize the policy simulator."""
        self.baseline_data = {}
        self.simulation_results = {}
        
    def set_baseline_data(self, 
                         france_data: pd.DataFrame,
                         portugal_data: pd.DataFrame,
                         ireland_data: pd.DataFrame) -> None:
        """
        Set baseline historical data for all countries.
        
        Parameters
        ----------
        france_data : pd.DataFrame
            Historical data for France
        portugal_data : pd.DataFrame
            Historical data for Portugal
        ireland_data : pd.DataFrame
            Historical data for Ireland
        """
        self.baseline_data = {
            'France': france_data,
            'Portugal': portugal_data,
            'Ireland': ireland_data
        }
    
    def simulate_no_intervention(self,
                                base_series: pd.Series,
                                trend_factor: float = 0.02,
                                volatility: float = 0.01,
                                periods: int = 24) -> pd.Series:
        """
        Simulate trajectory without policy intervention.
        
        Parameters
        ----------
        base_series : pd.Series
            Historical baseline data
        trend_factor : float
            Underlying trend factor (positive = increasing)
        volatility : float
            Random volatility factor
        periods : int
            Number of periods to simulate
            
        Returns
        -------
        pd.Series
            Simulated trajectory without intervention
        """
        np.random.seed(42)
        
        last_value = base_series.iloc[-1]
        last_date = base_series.index[-1]
        
        # Generate future dates
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=periods,
                freq='Y'
            )
        else:
            future_dates = range(int(last_value) + 1, int(last_value) + periods + 1)
        
        # Simulate values with trend and volatility
        simulated_values = [last_value]
        for _ in range(periods):
            change = trend_factor + np.random.normal(0, volatility)
            new_value = simulated_values[-1] * (1 + change)
            simulated_values.append(new_value)
        
        return pd.Series(simulated_values[1:], index=future_dates)
    
    def simulate_with_policy(self,
                            base_series: pd.Series,
                            policy_type: str,
                            intensity: float = 1.0,
                            delay: int = 0,
                            periods: int = 24) -> pd.Series:
        """
        Simulate trajectory with policy intervention.
        
        Parameters
        ----------
        base_series : pd.Series
            Historical baseline data
        policy_type : str
            Type of policy intervention
        intensity : float
            Policy intensity multiplier (1.0 = standard effect)
        delay : int
            Periods before policy effects begin
        periods : int
            Number of periods to simulate
            
        Returns
        -------
        pd.Series
            Simulated trajectory with intervention
        """
        if policy_type not in self.POLICY_EFFECTS:
            raise ValueError(f"Unknown policy type: {policy_type}")
            
        np.random.seed(42)
        
        policy_effect = self.POLICY_EFFECTS[policy_type]
        last_value = base_series.iloc[-1]
        last_date = base_series.index[-1]
        
        # Generate future dates
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(
                start=last_date + pd.DateOffset(years=1),
                periods=periods,
                freq='Y'
            )
        else:
            future_dates = range(int(last_value) + 1, int(last_value) + periods + 1)
        
        # Simulate values with policy effects
        simulated_values = [last_value]
        
        # Get the primary effect based on policy type
        if 'debt_reduction_annual' in policy_effect:
            primary_effect = policy_effect['debt_reduction_annual']
        elif 'gdp_recovery_rate' in policy_effect:
            primary_effect = policy_effect['gdp_recovery_rate']
        else:
            primary_effect = -0.01
        
        for i in range(periods):
            # Apply policy effect after delay
            if i >= delay:
                effect = primary_effect * intensity
                # Effects compound over time
                cumulative_factor = 1 + effect * (i - delay + 1) / periods
            else:
                cumulative_factor = 1.0
            
            # Add some volatility
            noise = np.random.normal(0, 0.005)
            new_value = simulated_values[-1] * cumulative_factor * (1 + noise)
            simulated_values.append(new_value)
        
        return pd.Series(simulated_values[1:], index=future_dates)
    
    def benchmark_against_countries(self,
                                   france_actual: pd.Series,
                                   portugal_actual: pd.Series,
                                   ireland_actual: pd.Series,
                                   indicator: str) -> Dict:
        """
        Benchmark France's performance against Portugal and Ireland.
        
        Parameters
        ----------
        france_actual : pd.Series
            France's historical data
        portugal_actual : pd.Series
            Portugal's historical data
        ireland_actual : pd.Series
            Ireland's historical data
        indicator : str
            Name of the indicator being compared
            
        Returns
        -------
        Dict
            Benchmark comparison results
        """
        # Calculate statistics for each country
        stats = {}
        for name, series in [('France', france_actual), 
                            ('Portugal', portugal_actual),
                            ('Ireland', ireland_actual)]:
            stats[name] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'trend': (series.iloc[-1] - series.iloc[0]) / len(series),
                'recent_avg': series.tail(5).mean()
            }
        
        # Calculate relative performance
        france_relative = {
            'vs_portugal': {
                'mean_diff': stats['France']['mean'] - stats['Portugal']['mean'],
                'trend_diff': stats['France']['trend'] - stats['Portugal']['trend'],
                'recent_diff': stats['France']['recent_avg'] - stats['Portugal']['recent_avg']
            },
            'vs_ireland': {
                'mean_diff': stats['France']['mean'] - stats['Ireland']['mean'],
                'trend_diff': stats['France']['trend'] - stats['Ireland']['trend'],
                'recent_diff': stats['France']['recent_avg'] - stats['Ireland']['recent_avg']
            }
        }
        
        return {
            'statistics': stats,
            'france_relative_performance': france_relative,
            'indicator': indicator
        }
    
    def run_policy_scenarios(self,
                            base_series: pd.Series,
                            scenarios: List[Dict],
                            periods: int = 24) -> Dict[str, pd.Series]:
        """
        Run multiple policy scenarios.
        
        Parameters
        ----------
        base_series : pd.Series
            Historical baseline data
        scenarios : List[Dict]
            List of scenario configurations
        periods : int
            Number of periods to simulate
            
        Returns
        -------
        Dict[str, pd.Series]
            Results for each scenario
        """
        results = {
            'baseline': self.simulate_no_intervention(base_series, periods=periods)
        }
        
        for scenario in scenarios:
            name = scenario.get('name', f"scenario_{len(results)}")
            policy_type = scenario.get('policy_type', 'fiscal_consolidation')
            intensity = scenario.get('intensity', 1.0)
            delay = scenario.get('delay', 0)
            
            results[name] = self.simulate_with_policy(
                base_series,
                policy_type,
                intensity,
                delay,
                periods
            )
        
        self.simulation_results = results
        return results
    
    def calculate_policy_impact(self,
                               with_policy: pd.Series,
                               without_policy: pd.Series) -> Dict:
        """
        Calculate the impact of a policy intervention.
        
        Parameters
        ----------
        with_policy : pd.Series
            Simulated values with policy
        without_policy : pd.Series
            Simulated values without policy
            
        Returns
        -------
        Dict
            Impact metrics
        """
        diff = without_policy - with_policy  # Positive = improvement
        
        return {
            'total_impact': diff.sum(),
            'average_impact': diff.mean(),
            'final_impact': diff.iloc[-1],
            'max_impact': diff.max(),
            'impact_trajectory': diff,
            'cumulative_benefit': diff.cumsum().iloc[-1],
            'percentage_improvement': (
                (without_policy.iloc[-1] - with_policy.iloc[-1]) / 
                without_policy.iloc[-1] * 100
            )
        }
    
    def generate_policy_report(self) -> str:
        """
        Generate a comprehensive policy analysis report.
        
        Returns
        -------
        str
            Formatted policy report
        """
        if not self.simulation_results:
            return "No simulation results available. Run simulations first."
        
        report = []
        report.append("=" * 60)
        report.append("FRANCE GOVERNMENT DEBT POLICY ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Compare scenarios
        baseline = self.simulation_results.get('baseline')
        
        for name, series in self.simulation_results.items():
            if name == 'baseline':
                continue
            
            impact = self.calculate_policy_impact(series, baseline)
            
            report.append(f"\n--- {name.upper()} ---")
            report.append(f"Average Impact: {impact['average_impact']:.4f}")
            report.append(f"Final Impact: {impact['final_impact']:.4f}")
            report.append(f"Percentage Improvement: {impact['percentage_improvement']:.2f}%")
            report.append(f"Cumulative Benefit: {impact['cumulative_benefit']:.4f}")
        
        report.append("\n" + "=" * 60)
        report.append("RECOMMENDATIONS")
        report.append("=" * 60)
        report.append("")
        report.append("Based on the simulation results and benchmark comparisons:")
        report.append("1. Implement gradual fiscal consolidation to reduce debt-to-GDP ratio")
        report.append("2. Monitor unemployment effects and adjust policy intensity")
        report.append("3. Leverage structural reforms similar to Ireland's approach")
        report.append("4. Maintain flexibility for pandemic recovery measures")
        
        return "\n".join(report)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for uncertainty analysis in policy outcomes.
    """
    
    def __init__(self, n_simulations: int = 1000):
        """
        Initialize Monte Carlo simulator.
        
        Parameters
        ----------
        n_simulations : int
            Number of simulation runs
        """
        self.n_simulations = n_simulations
        self.results = None
    
    def run_simulation(self,
                      base_value: float,
                      mean_growth: float,
                      volatility: float,
                      periods: int = 24) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        base_value : float
            Starting value
        mean_growth : float
            Mean growth rate
        volatility : float
            Standard deviation of growth
        periods : int
            Number of periods
            
        Returns
        -------
        Dict
            Simulation results with confidence intervals
        """
        all_paths = np.zeros((self.n_simulations, periods))
        
        for sim in range(self.n_simulations):
            values = [base_value]
            for _ in range(periods):
                growth = np.random.normal(mean_growth, volatility)
                values.append(values[-1] * (1 + growth))
            all_paths[sim] = values[1:]
        
        # Calculate statistics
        mean_path = np.mean(all_paths, axis=0)
        std_path = np.std(all_paths, axis=0)
        percentile_5 = np.percentile(all_paths, 5, axis=0)
        percentile_95 = np.percentile(all_paths, 95, axis=0)
        
        self.results = {
            'all_paths': all_paths,
            'mean': mean_path,
            'std': std_path,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'confidence_interval_90': (percentile_5, percentile_95)
        }
        
        return self.results
    
    def get_probability_of_target(self, target: float) -> float:
        """
        Calculate probability of meeting a target.
        
        Parameters
        ----------
        target : float
            Target value
            
        Returns
        -------
        float
            Probability (0-1) of meeting target
        """
        if self.results is None:
            raise ValueError("Run simulation first")
            
        final_values = self.results['all_paths'][:, -1]
        return np.mean(final_values <= target)


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data_preparation import DataLoader
    
    # Load sample data
    loader = DataLoader()
    data = loader.create_sample_data()
    
    # Prepare time series
    france_debt = loader.prepare_time_series(
        data['debt_to_gdp'],
        'Debt_to_GDP',
        'France'
    )
    
    # Run policy simulation
    simulator = PolicySimulator()
    
    scenarios = [
        {'name': 'fiscal_consolidation', 'policy_type': 'fiscal_consolidation', 'intensity': 1.0},
        {'name': 'structural_reforms', 'policy_type': 'structural_reforms', 'intensity': 1.5},
        {'name': 'combined_policy', 'policy_type': 'fiscal_consolidation', 'intensity': 0.5, 'delay': 2}
    ]
    
    results = simulator.run_policy_scenarios(france_debt, scenarios, periods=10)
    
    print("Simulation Results:")
    for name, series in results.items():
        print(f"\n{name}:")
        print(series.tail())
    
    print("\n" + simulator.generate_policy_report())
