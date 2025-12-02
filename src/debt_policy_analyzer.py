"""
France Government Debt Policy Analysis - Complete Pipeline

This module implements the 5-step analysis workflow:
1. Data Preprocessing (2000 onwards, quarterly interpolation, train/test split)
2. Model A: Baseline prediction using Prophet (France status quo)
3. Model B: Policy effect calculation (Portugal/Ireland slope change)
4. Scenario Synthesis: Apply policy effects to baseline
5. Final Visualization: Clean poster-ready graphs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not available. Install with: pip install prophet")


class DebtPolicyAnalyzer:
    """
    Main analyzer class implementing the 5-step workflow for
    France government debt policy analysis.
    """
    
    COUNTRIES = ['France', 'Portugal', 'Ireland']
    
    def __init__(self):
        """Initialize the analyzer."""
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        self.baseline_forecast = None
        self.policy_effect = None
        self.scenario_forecast = None
    
    def _get_value_for_year(self, df: pd.DataFrame, column: str, year: int) -> float:
        """
        Helper method to get a value for a specific year from a dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'Year' column
        column : str
            Column name to extract value from
        year : int
            Target year
            
        Returns
        -------
        float
            Value for the specified year, or last value if year not found
        """
        filtered = df[df['Year'] >= year]
        if len(filtered) > 0:
            return filtered[column].iloc[0]
        return df[column].iloc[-1]
        
    # ==================== STEP 1: DATA PREPROCESSING ====================
    
    def create_sample_debt_data(self) -> pd.DataFrame:
        """
        Create sample debt-to-GDP data for France, Portugal, and Ireland.
        Based on approximate historical trends from 2000-2024.
        """
        np.random.seed(42)
        years = list(range(1990, 2025))
        
        # France: Gradual increase, accelerated after 2008, spike in 2020
        france_base = [35 + i * 0.8 for i in range(len(years))]
        france_noise = np.random.normal(0, 1.5, len(years))
        france = np.array(france_base) + france_noise
        # Financial crisis bump
        for i, y in enumerate(years):
            if 2008 <= y <= 2012:
                france[i] += (y - 2007) * 3
            elif y > 2012:
                france[i] += 15
            # COVID spike
            if y == 2020:
                france[i] += 15
            elif y == 2021:
                france[i] += 12
            elif y >= 2022:
                france[i] += 10
        
        # Portugal: Higher starting, spike during crisis, then recovery after 2014
        portugal_base = [50 + i * 1.0 for i in range(len(years))]
        portugal_noise = np.random.normal(0, 2, len(years))
        portugal = np.array(portugal_base) + portugal_noise
        for i, y in enumerate(years):
            if 2008 <= y <= 2014:
                portugal[i] += (y - 2007) * 5
            elif y > 2014:
                # Austerity effect - debt growth slows then reverses
                portugal[i] += 35 - (y - 2014) * 2.5
            if y == 2020:
                portugal[i] += 15
            elif y == 2021:
                portugal[i] += 10
            elif y >= 2022:
                portugal[i] += 5
        
        # Ireland: Massive crisis spike, then dramatic recovery
        ireland_base = [30 + i * 0.5 for i in range(len(years))]
        ireland_noise = np.random.normal(0, 2, len(years))
        ireland = np.array(ireland_base) + ireland_noise
        for i, y in enumerate(years):
            if 2008 <= y <= 2012:
                ireland[i] += (y - 2007) * 10  # Banking crisis
            elif 2012 < y <= 2014:
                ireland[i] += 50
            elif y > 2014:
                # Strong recovery
                ireland[i] += 50 - (y - 2014) * 5
            if y == 2020:
                ireland[i] += 10
            elif y == 2021:
                ireland[i] += 5
        
        data = pd.DataFrame({
            'Year': years,
            'France': np.clip(france, 30, 130),
            'Portugal': np.clip(portugal, 45, 135),
            'Ireland': np.clip(ireland, 25, 125)
        })
        
        self.raw_data = data
        return data
    
    def preprocess_data(self, 
                       start_year: int = 2000,
                       interpolate_quarterly: bool = True) -> pd.DataFrame:
        """
        Step 1: Data preprocessing.
        - Drop data before start_year
        - Optionally interpolate to quarterly data
        
        Parameters
        ----------
        start_year : int
            First year to include (default 2000)
        interpolate_quarterly : bool
            Whether to interpolate yearly to quarterly data
            
        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        if self.raw_data is None:
            self.create_sample_debt_data()
        
        # Filter from start_year
        data = self.raw_data[self.raw_data['Year'] >= start_year].copy()
        
        if interpolate_quarterly:
            # Convert to quarterly data through interpolation
            quarterly_data = []
            
            for i in range(len(data) - 1):
                year = data.iloc[i]['Year']
                next_year = data.iloc[i + 1]['Year']
                
                for q in range(4):
                    quarter_date = f"{int(year)}-Q{q+1}"
                    row = {'Date': quarter_date, 'Year': year + q * 0.25}
                    
                    for country in self.COUNTRIES:
                        # Linear interpolation between years
                        start_val = data.iloc[i][country]
                        end_val = data.iloc[i + 1][country]
                        row[country] = start_val + (end_val - start_val) * (q / 4)
                    
                    quarterly_data.append(row)
            
            # Add last year's quarters
            last_idx = len(data) - 1
            last_year = data.iloc[last_idx]['Year']
            for q in range(4):
                quarter_date = f"{int(last_year)}-Q{q+1}"
                row = {'Date': quarter_date, 'Year': last_year + q * 0.25}
                for country in self.COUNTRIES:
                    row[country] = data.iloc[last_idx][country]
                quarterly_data.append(row)
            
            data = pd.DataFrame(quarterly_data)
        
        self.processed_data = data
        return data
    
    def split_train_test(self, 
                        train_end_year: int = 2019,
                        test_start_year: int = 2020) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and test sets.
        
        Parameters
        ----------
        train_end_year : int
            Last year for training data
        test_start_year : int
            First year for test data
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Training and test dataframes
        """
        if self.processed_data is None:
            self.preprocess_data()
        
        data = self.processed_data
        
        # Handle both yearly and quarterly data
        if 'Year' in data.columns:
            self.train_data = data[data['Year'] <= train_end_year].copy()
            self.test_data = data[data['Year'] >= test_start_year].copy()
        else:
            self.train_data = data.copy()
            self.test_data = pd.DataFrame()
        
        return self.train_data, self.test_data
    
    # ==================== STEP 2: MODEL A - BASELINE (PROPHET) ====================
    
    def fit_prophet_baseline(self, 
                            country: str = 'France',
                            forecast_years: int = 5) -> pd.DataFrame:
        """
        Step 2: Fit Prophet model for baseline prediction.
        
        Parameters
        ----------
        country : str
            Country to forecast
        forecast_years : int
            Number of years to forecast
            
        Returns
        -------
        pd.DataFrame
            Forecast dataframe with ds, yhat columns
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required. Install with: pip install prophet")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        # Prepare data for Prophet
        data = self.processed_data.copy()
        
        if 'Date' in data.columns:
            # Convert quarterly dates to datetime
            def quarter_to_date(q_str):
                parts = q_str.split('-Q')
                year = int(parts[0])
                quarter = int(parts[1])
                month = (quarter - 1) * 3 + 1
                return pd.Timestamp(year=year, month=month, day=1)
            
            data['ds'] = data['Date'].apply(quarter_to_date)
        else:
            data['ds'] = pd.to_datetime(data['Year'].astype(int), format='%Y')
        
        data['y'] = data[country]
        prophet_data = data[['ds', 'y']].copy()
        
        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_data)
        
        # Create future dataframe
        if 'Date' in self.processed_data.columns:
            # Quarterly forecast
            periods = forecast_years * 4
            future = model.make_future_dataframe(periods=periods, freq='QS')
        else:
            # Yearly forecast
            periods = forecast_years
            future = model.make_future_dataframe(periods=periods, freq='Y')
        
        # Generate forecast
        forecast = model.predict(future)
        
        self.baseline_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        self.baseline_forecast['Year'] = self.baseline_forecast['ds'].dt.year + \
                                         (self.baseline_forecast['ds'].dt.month - 1) / 12
        
        return self.baseline_forecast
    
    # ==================== STEP 3: MODEL B - POLICY EFFECT ====================
    
    def calculate_policy_effect(self,
                               event_year: int = 2011,
                               before_start: int = 2008,
                               before_end: int = 2010,
                               after_start: int = 2012,
                               after_end: int = 2015) -> Dict:
        """
        Step 3: Calculate policy effect using slope change method.
        
        Parameters
        ----------
        event_year : int
            Year of policy intervention (bailout/austerity start)
        before_start, before_end : int
            Crisis period (debt increasing rapidly)
        after_start, after_end : int
            Austerity period (debt stabilizing/decreasing)
            
        Returns
        -------
        Dict
            Policy effect calculations
        """
        if self.raw_data is None:
            self.create_sample_debt_data()
        
        data = self.raw_data.copy()
        
        effects = {}
        
        for country in ['Portugal', 'Ireland']:
            # Before period (crisis)
            before_data = data[(data['Year'] >= before_start) & 
                              (data['Year'] <= before_end)]
            before_values = before_data[country].values
            before_slope = (before_values[-1] - before_values[0]) / (before_end - before_start)
            
            # After period (austerity)
            after_data = data[(data['Year'] >= after_start) & 
                             (data['Year'] <= after_end)]
            after_values = after_data[country].values
            after_slope = (after_values[-1] - after_values[0]) / (after_end - after_start)
            
            # Slope change (effect of austerity)
            slope_change = after_slope - before_slope
            
            effects[country] = {
                'before_slope': before_slope,
                'after_slope': after_slope,
                'slope_change': slope_change
            }
        
        # Average effect across both countries
        avg_effect = (effects['Portugal']['slope_change'] + 
                     effects['Ireland']['slope_change']) / 2
        
        self.policy_effect = {
            'Portugal': effects['Portugal'],
            'Ireland': effects['Ireland'],
            'average_delta': avg_effect,
            'event_year': event_year
        }
        
        return self.policy_effect
    
    # ==================== STEP 4: SCENARIO SYNTHESIS ====================
    
    def synthesize_scenarios(self,
                            scenario_start_year: int = 2025,
                            scenario_end_year: int = 2029) -> pd.DataFrame:
        """
        Step 4: Apply policy effects to baseline prediction.
        
        Parameters
        ----------
        scenario_start_year : int
            First year to apply policy effect
        scenario_end_year : int
            Last year of scenario
            
        Returns
        -------
        pd.DataFrame
            DataFrame with Year, Baseline_Pred, Scenario_Pred columns
        """
        if self.baseline_forecast is None:
            self.fit_prophet_baseline()
        
        if self.policy_effect is None:
            self.calculate_policy_effect()
        
        delta = self.policy_effect['average_delta']
        
        # Filter forecast for scenario period
        scenario_data = self.baseline_forecast[
            (self.baseline_forecast['Year'] >= scenario_start_year) &
            (self.baseline_forecast['Year'] <= scenario_end_year + 1)
        ].copy()
        
        # Calculate scenario values
        scenario_data['Years_Elapsed'] = scenario_data['Year'] - scenario_start_year + 1
        scenario_data['Baseline_Pred'] = scenario_data['yhat']
        scenario_data['Scenario_Pred'] = scenario_data['yhat'] + \
                                         (delta * scenario_data['Years_Elapsed'])
        
        self.scenario_forecast = scenario_data[
            ['ds', 'Year', 'Baseline_Pred', 'Scenario_Pred']
        ].copy()
        
        return self.scenario_forecast
    
    # ==================== STEP 5: FINAL VISUALIZATION ====================
    
    def create_poster_visualization(self,
                                   display_start_year: int = 2015,
                                   display_end_year: int = 2029,
                                   save_path: str = 'debt_policy_forecast.png',
                                   transparent: bool = True) -> plt.Figure:
        """
        Step 5: Create clean poster-ready visualization.
        
        Parameters
        ----------
        display_start_year : int
            First year to display on x-axis
        display_end_year : int
            Last year to display
        save_path : str
            Path to save PNG file
        transparent : bool
            Whether to save with transparent background
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        if self.baseline_forecast is None:
            self.fit_prophet_baseline()
        
        if self.scenario_forecast is None:
            self.synthesize_scenarios()
        
        if self.processed_data is None:
            self.preprocess_data()
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Use English labels for better compatibility
        plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
        
        # Historical data (black dots)
        hist_data = self.processed_data[
            self.processed_data['Year'] >= display_start_year
        ].copy()
        
        if 'Date' in hist_data.columns:
            # Group by year for cleaner display
            hist_yearly = hist_data.groupby(hist_data['Year'].astype(int))['France'].mean()
            ax.scatter(hist_yearly.index, hist_yearly.values, 
                      color='black', s=80, zorder=5, label='Historical Data')
        else:
            ax.scatter(hist_data['Year'], hist_data['France'], 
                      color='black', s=80, zorder=5, label='Historical Data')
        
        # Baseline forecast (red line - status quo)
        baseline_display = self.baseline_forecast[
            (self.baseline_forecast['Year'] >= display_start_year) &
            (self.baseline_forecast['Year'] <= display_end_year)
        ]
        
        ax.plot(baseline_display['Year'], baseline_display['yhat'],
               color='red', linewidth=2.5, linestyle='-',
               label='Baseline (Status Quo)', zorder=3)
        
        # Fill confidence interval for baseline
        ax.fill_between(baseline_display['Year'],
                       baseline_display['yhat_lower'],
                       baseline_display['yhat_upper'],
                       color='red', alpha=0.1)
        
        # Scenario forecast (blue line - with policy)
        scenario_display = self.scenario_forecast[
            self.scenario_forecast['Year'] >= 2025
        ]
        
        ax.plot(scenario_display['Year'], scenario_display['Scenario_Pred'],
               color='blue', linewidth=2.5, linestyle='-',
               label='Austerity Scenario', zorder=4)
        
        # Styling
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Debt-to-GDP Ratio (%)', fontsize=12, fontweight='bold')
        ax.set_title('France Government Debt Forecast: Baseline vs Austerity Policy',
                    fontsize=14, fontweight='bold', pad=20)
        
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set axis limits
        ax.set_xlim(display_start_year - 0.5, display_end_year + 0.5)
        
        # Add vertical line at forecast start
        ax.axvline(x=2024, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(2024.1, ax.get_ylim()[0] + 2, 'Forecast Start', 
               fontsize=9, alpha=0.7)
        
        plt.tight_layout()
        
        # Save with transparency option
        plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                   transparent=transparent, facecolor='white')
        
        return fig
    
    def run_full_analysis(self, 
                         save_visualization: bool = True,
                         output_path: str = 'debt_policy_forecast.png') -> Dict:
        """
        Run the complete 5-step analysis pipeline.
        
        Returns
        -------
        Dict
            Dictionary with all analysis results
        """
        print("=" * 60)
        print("France Government Debt Policy Analysis Pipeline")
        print("=" * 60)
        
        # Step 1: Data Preprocessing
        print("\n📊 Step 1: Data Preprocessing")
        print("-" * 40)
        self.create_sample_debt_data()
        self.preprocess_data(start_year=2000, interpolate_quarterly=True)
        self.split_train_test(train_end_year=2019, test_start_year=2020)
        print(f"  • Raw data: 1990-2024 ({len(self.raw_data)} rows)")
        print(f"  • Processed: 2000-2024, quarterly ({len(self.processed_data)} rows)")
        print(f"  • Train: 2000-2019 ({len(self.train_data)} rows)")
        print(f"  • Test: 2020-2024 ({len(self.test_data)} rows)")
        
        # Step 2: Model A - Baseline Prediction
        print("\n🤖 Step 2: Model A - Baseline Prediction (Prophet)")
        print("-" * 40)
        self.fit_prophet_baseline(country='France', forecast_years=5)
        forecast_2029 = self._get_value_for_year(
            self.baseline_forecast, 'yhat', 2029
        )
        print(f"  • Forecast period: 2025-2029")
        print(f"  • 2029 predicted debt ratio: {forecast_2029:.1f}%")
        
        # Step 3: Model B - Policy Effect
        print("\n📉 Step 3: Model B - Policy Effect Calculation")
        print("-" * 40)
        self.calculate_policy_effect()
        print(f"  • Event baseline: 2011 (bailout/austerity start)")
        print(f"  • Portugal slope change: {self.policy_effect['Portugal']['slope_change']:.2f}%p/year")
        print(f"  • Ireland slope change: {self.policy_effect['Ireland']['slope_change']:.2f}%p/year")
        print(f"  • Average effect (Delta): {self.policy_effect['average_delta']:.2f}%p/year")
        
        # Step 4: Scenario Synthesis
        print("\n🔮 Step 4: Scenario Synthesis")
        print("-" * 40)
        self.synthesize_scenarios(scenario_start_year=2025, scenario_end_year=2029)
        scenario_2029 = self._get_value_for_year(
            self.scenario_forecast, 'Scenario_Pred', 2029
        )
        print(f"  • 2029 Baseline (status quo): {forecast_2029:.1f}%")
        print(f"  • 2029 Austerity scenario: {scenario_2029:.1f}%")
        print(f"  • Expected improvement: {forecast_2029 - scenario_2029:.1f}%p")
        
        # Step 5: Visualization
        print("\n📊 Step 5: Final Visualization")
        print("-" * 40)
        if save_visualization:
            fig = self.create_poster_visualization(
                display_start_year=2015,
                display_end_year=2029,
                save_path=output_path,
                transparent=True
            )
            print(f"  • Save path: {output_path}")
            print(f"  • Background: transparent")
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
        
        return {
            'raw_data': self.raw_data,
            'processed_data': self.processed_data,
            'train_data': self.train_data,
            'test_data': self.test_data,
            'baseline_forecast': self.baseline_forecast,
            'policy_effect': self.policy_effect,
            'scenario_forecast': self.scenario_forecast
        }


def generate_results_dataframe(analyzer: DebtPolicyAnalyzer) -> pd.DataFrame:
    """
    Generate a clean results dataframe for the final presentation.
    
    Parameters
    ----------
    analyzer : DebtPolicyAnalyzer
        Analyzer instance after running analysis
        
    Returns
    -------
    pd.DataFrame
        Clean dataframe with Year, Baseline_Pred, Scenario_Pred
    """
    if analyzer.scenario_forecast is None:
        raise ValueError("Run analysis first")
    
    # Combine historical and forecast data
    hist_data = analyzer.processed_data.copy()
    
    # Get yearly averages for historical
    if 'Date' in hist_data.columns:
        hist_yearly = hist_data.groupby(hist_data['Year'].astype(int)).agg({
            'France': 'mean'
        }).reset_index()
        hist_yearly.columns = ['Year', 'Historical']
    else:
        hist_yearly = hist_data[['Year', 'France']].copy()
        hist_yearly.columns = ['Year', 'Historical']
    
    # Get forecast data
    forecast_yearly = analyzer.baseline_forecast.copy()
    forecast_yearly['Year_Int'] = forecast_yearly['Year'].astype(int)
    forecast_yearly = forecast_yearly.groupby('Year_Int').agg({
        'yhat': 'mean'
    }).reset_index()
    forecast_yearly.columns = ['Year', 'Baseline_Pred']
    
    # Get scenario data
    scenario_yearly = analyzer.scenario_forecast.copy()
    scenario_yearly['Year_Int'] = scenario_yearly['Year'].astype(int)
    scenario_yearly = scenario_yearly.groupby('Year_Int').agg({
        'Baseline_Pred': 'mean',
        'Scenario_Pred': 'mean'
    }).reset_index()
    scenario_yearly.columns = ['Year', 'Baseline_Pred', 'Scenario_Pred']
    
    return scenario_yearly


if __name__ == "__main__":
    # Run the complete analysis
    analyzer = DebtPolicyAnalyzer()
    results = analyzer.run_full_analysis(
        save_visualization=True,
        output_path='debt_policy_forecast.png'
    )
    
    # Generate results dataframe
    results_df = generate_results_dataframe(analyzer)
    print("\n최종 시나리오 비교 데이터프레임:")
    print(results_df.to_string(index=False))
