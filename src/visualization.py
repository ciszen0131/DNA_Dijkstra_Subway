"""
Visualization Module for Government Debt Policy Analysis

This module provides functions to create visualizations for comparing
economic indicators across France, Portugal, and Ireland.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


class EconomicVisualizer:
    """Create visualizations for economic indicator analysis."""
    
    # Color palette for countries
    COLORS = {
        'France': '#0055A4',      # French blue
        'Portugal': '#006600',    # Portuguese green
        'Ireland': '#169B62'      # Irish green
    }
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        style : str
            Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('seaborn-v0_8')
        self.figure_size = (12, 6)
        
    def plot_bond_yields_comparison(self, 
                                    df: pd.DataFrame,
                                    title: str = "10-Year Government Bond Yields Comparison",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot and compare 10-year bond yields across countries.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Date index and country columns
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for country in ['France', 'Portugal', 'Ireland']:
            if country in df.columns:
                ax.plot(df.index, df[country], 
                       label=country, 
                       color=self.COLORS.get(country, None),
                       linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Bond Yield (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add recession periods shading (example: 2008 financial crisis, COVID-19)
        ax.axvspan(pd.Timestamp('2008-01-01'), pd.Timestamp('2009-12-31'), 
                  alpha=0.2, color='gray', label='Financial Crisis')
        ax.axvspan(pd.Timestamp('2020-01-01'), pd.Timestamp('2021-06-30'), 
                  alpha=0.2, color='red', label='COVID-19')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_fiscal_balance_trends(self,
                                   df: pd.DataFrame,
                                   value_col: str = 'Fiscal_Balance',
                                   title: str = "Government Fiscal Balance (% of GDP)",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot fiscal balance trends for each country.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Year, Country, and value columns
        value_col : str
            Column name for the fiscal balance values
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for country in ['France', 'Portugal', 'Ireland']:
            country_data = df[df['Country'] == country].sort_values('Year')
            ax.plot(country_data['Year'], country_data[value_col],
                   label=country,
                   color=self.COLORS.get(country, None),
                   linewidth=2,
                   marker='o',
                   markersize=4)
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.fill_between(df['Year'].unique(), 0, -10, alpha=0.1, color='red')
        ax.fill_between(df['Year'].unique(), 0, 10, alpha=0.1, color='green')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Fiscal Balance (% of GDP)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_debt_to_gdp(self,
                         df: pd.DataFrame,
                         value_col: str = 'Debt_to_GDP',
                         title: str = "Government Debt as % of GDP",
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot debt-to-GDP ratio comparison.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Year, Country, and debt-to-GDP columns
        value_col : str
            Column name for the debt-to-GDP values
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for country in ['France', 'Portugal', 'Ireland']:
            country_data = df[df['Country'] == country].sort_values('Year')
            ax.plot(country_data['Year'], country_data[value_col],
                   label=country,
                   color=self.COLORS.get(country, None),
                   linewidth=2,
                   marker='s',
                   markersize=4)
        
        # Maastricht criteria line (60% debt-to-GDP)
        ax.axhline(y=60, color='red', linestyle='--', linewidth=1.5, 
                  alpha=0.7, label='Maastricht Limit (60%)')
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Debt-to-GDP Ratio (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_unemployment_comparison(self,
                                    df: pd.DataFrame,
                                    value_col: str = 'Unemployment_Rate',
                                    title: str = "Unemployment Rate Comparison",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot unemployment rate comparison across countries.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with Year, Country, and unemployment rate columns
        value_col : str
            Column name for unemployment values
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        for country in ['France', 'Portugal', 'Ireland']:
            country_data = df[df['Country'] == country].sort_values('Year')
            ax.plot(country_data['Year'], country_data[value_col],
                   label=country,
                   color=self.COLORS.get(country, None),
                   linewidth=2,
                   marker='^',
                   markersize=4)
        
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Unemployment Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def create_dashboard(self, 
                        data: Dict[str, pd.DataFrame],
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard with all visualizations.
        
        Parameters
        ----------
        data : Dict[str, pd.DataFrame]
            Dictionary containing all datasets
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Bond Yields
        if 'bond_yields' in data:
            df = data['bond_yields']
            for country in ['France', 'Portugal', 'Ireland']:
                if country in df.columns:
                    axes[0, 0].plot(df.index, df[country],
                                   label=country,
                                   color=self.COLORS.get(country, None),
                                   linewidth=2)
            axes[0, 0].set_title('10-Year Bond Yields', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Yield (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Fiscal Balance
        if 'fiscal' in data:
            df = data['fiscal']
            for country in ['France', 'Portugal', 'Ireland']:
                country_data = df[df['Country'] == country].sort_values('Year')
                axes[0, 1].plot(country_data['Year'], country_data['Fiscal_Balance'],
                               label=country,
                               color=self.COLORS.get(country, None),
                               linewidth=2)
            axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Fiscal Balance (% of GDP)', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Balance (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Debt-to-GDP
        if 'debt_to_gdp' in data:
            df = data['debt_to_gdp']
            for country in ['France', 'Portugal', 'Ireland']:
                country_data = df[df['Country'] == country].sort_values('Year')
                axes[1, 0].plot(country_data['Year'], country_data['Debt_to_GDP'],
                               label=country,
                               color=self.COLORS.get(country, None),
                               linewidth=2)
            axes[1, 0].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='60% Limit')
            axes[1, 0].set_title('Debt-to-GDP Ratio', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Ratio (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Unemployment
        if 'unemployment' in data:
            df = data['unemployment']
            for country in ['France', 'Portugal', 'Ireland']:
                country_data = df[df['Country'] == country].sort_values('Year')
                axes[1, 1].plot(country_data['Year'], country_data['Unemployment_Rate'],
                               label=country,
                               color=self.COLORS.get(country, None),
                               linewidth=2)
            axes[1, 1].set_title('Unemployment Rate', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Rate (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('France Debt Policy Analysis - Country Comparison Dashboard',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_prediction_comparison(self,
                                   actual: pd.Series,
                                   predicted: pd.Series,
                                   forecast: pd.Series,
                                   confidence_interval: Optional[Tuple[pd.Series, pd.Series]] = None,
                                   title: str = "Prediction vs Actual",
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot actual values vs predictions with forecast.
        
        Parameters
        ----------
        actual : pd.Series
            Actual historical values
        predicted : pd.Series
            Predicted values for the historical period
        forecast : pd.Series
            Future forecast values
        confidence_interval : Tuple[pd.Series, pd.Series], optional
            Lower and upper bounds of confidence interval
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        ax.plot(actual.index, actual.values, 'b-', label='Actual', linewidth=2)
        ax.plot(predicted.index, predicted.values, 'g--', label='Fitted', linewidth=2)
        ax.plot(forecast.index, forecast.values, 'r-', label='Forecast', linewidth=2)
        
        if confidence_interval:
            lower, upper = confidence_interval
            ax.fill_between(forecast.index, lower, upper, color='red', alpha=0.2,
                           label='95% CI')
        
        ax.axvline(x=actual.index[-1], color='gray', linestyle=':', alpha=0.7,
                  label='Forecast Start')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig
    
    def plot_policy_scenario_comparison(self,
                                       baseline: pd.Series,
                                       with_policy: pd.Series,
                                       without_policy: pd.Series,
                                       title: str = "Policy Scenario Comparison",
                                       save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot policy scenario comparison.
        
        Parameters
        ----------
        baseline : pd.Series
            Historical baseline values
        with_policy : pd.Series
            Projected values with policy intervention
        without_policy : pd.Series
            Projected values without policy intervention
        title : str
            Plot title
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        ax.plot(baseline.index, baseline.values, 'b-', 
               label='Historical', linewidth=2)
        ax.plot(with_policy.index, with_policy.values, 'g-',
               label='With Policy Intervention', linewidth=2)
        ax.plot(without_policy.index, without_policy.values, 'r--',
               label='Without Policy Intervention', linewidth=2)
        
        # Shade the difference
        ax.fill_between(with_policy.index, 
                       with_policy.values, 
                       without_policy.values,
                       alpha=0.2, color='green',
                       label='Policy Impact')
        
        ax.axvline(x=baseline.index[-1], color='gray', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
        return fig


if __name__ == "__main__":
    # Example usage
    from data_preparation import DataLoader
    
    loader = DataLoader()
    data = loader.create_sample_data()
    
    visualizer = EconomicVisualizer()
    
    # Create dashboard
    fig = visualizer.create_dashboard(data, save_path='dashboard.png')
    plt.show()
