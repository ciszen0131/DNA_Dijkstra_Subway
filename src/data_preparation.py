"""
Data Preparation Module for Government Debt Policy Analysis

This module handles loading, filtering, and preprocessing data from
IMF, OECD, and bond yields datasets for France, Portugal, and Ireland.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class DataLoader:
    """Load and preprocess economic data for analysis."""
    
    # Country codes for the analysis
    COUNTRIES = {
        'France': 'FRA',
        'Portugal': 'PRT',
        'Ireland': 'IRL'
    }
    
    # Key indicators for the analysis
    INDICATORS = {
        'fiscal_balance': 'Net lending/borrowing as % of GDP',
        'unemployment_rate': 'Unemployment rate %',
        'bond_yield_10y': '10-year government bond yield',
        'gdp_growth': 'Real GDP growth rate',
        'debt_to_gdp': 'Government debt as % of GDP',
        'primary_balance': 'Primary balance as % of GDP'
    }
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.data = {}
        
    def load_imf_data(self, filepath: str) -> pd.DataFrame:
        """
        Load IMF World Economic Outlook data.
        
        Parameters
        ----------
        filepath : str
            Path to the IMF data file (CSV or Excel)
            
        Returns
        -------
        pd.DataFrame
            Filtered and processed IMF data
        """
        try:
            if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)
            
            # Filter for our target countries
            country_codes = list(self.COUNTRIES.values())
            if 'ISO' in df.columns:
                df = df[df['ISO'].isin(country_codes)]
            elif 'Country Code' in df.columns:
                df = df[df['Country Code'].isin(country_codes)]
                
            self.data['imf'] = df
            return df
            
        except Exception as e:
            print(f"Error loading IMF data: {e}")
            return pd.DataFrame()
    
    def load_oecd_data(self, filepath: str) -> pd.DataFrame:
        """
        Load OECD economic indicators data.
        
        Parameters
        ----------
        filepath : str
            Path to the OECD data file
            
        Returns
        -------
        pd.DataFrame
            Filtered OECD data for target countries
        """
        try:
            df = pd.read_csv(filepath)
            
            # Filter for target countries
            country_names = list(self.COUNTRIES.keys())
            if 'Country' in df.columns:
                df = df[df['Country'].isin(country_names)]
            elif 'LOCATION' in df.columns:
                df = df[df['LOCATION'].isin(self.COUNTRIES.values())]
                
            self.data['oecd'] = df
            return df
            
        except Exception as e:
            print(f"Error loading OECD data: {e}")
            return pd.DataFrame()
    
    def load_bond_yields(self, filepath: str) -> pd.DataFrame:
        """
        Load 10-year government bond yield data.
        
        Parameters
        ----------
        filepath : str
            Path to the bond yields data file
            
        Returns
        -------
        pd.DataFrame
            Bond yield data for the target countries
        """
        try:
            df = pd.read_csv(filepath, parse_dates=['Date'])
            df.set_index('Date', inplace=True)
            
            # Keep only relevant columns
            target_cols = [col for col in df.columns 
                         if any(country in col for country in self.COUNTRIES.keys())]
            
            if target_cols:
                df = df[target_cols]
                
            self.data['bond_yields'] = df
            return df
            
        except Exception as e:
            print(f"Error loading bond yields data: {e}")
            return pd.DataFrame()
    
    def create_sample_data(self) -> Dict[str, pd.DataFrame]:
        """
        Create sample data for demonstration purposes.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary containing sample datasets
        """
        np.random.seed(42)
        years = range(2000, 2024)
        
        # Sample fiscal balance data (% of GDP)
        fiscal_data = {
            'Year': list(years) * 3,
            'Country': ['France'] * len(years) + ['Portugal'] * len(years) + ['Ireland'] * len(years),
            'Fiscal_Balance': (
                list(np.random.normal(-3.0, 1.5, len(years))) +  # France
                list(np.random.normal(-4.0, 2.0, len(years))) +  # Portugal
                list(np.random.normal(-2.0, 3.0, len(years)))    # Ireland
            )
        }
        fiscal_df = pd.DataFrame(fiscal_data)
        
        # Sample unemployment rate data
        unemployment_data = {
            'Year': list(years) * 3,
            'Country': ['France'] * len(years) + ['Portugal'] * len(years) + ['Ireland'] * len(years),
            'Unemployment_Rate': (
                list(np.clip(np.random.normal(9.0, 1.5, len(years)), 5, 15)) +   # France
                list(np.clip(np.random.normal(10.0, 2.5, len(years)), 4, 18)) +  # Portugal
                list(np.clip(np.random.normal(8.0, 3.5, len(years)), 3, 16))     # Ireland
            )
        }
        unemployment_df = pd.DataFrame(unemployment_data)
        
        # Sample 10-year bond yield data
        dates = pd.date_range(start='2000-01-01', end='2023-12-31', freq='ME')
        bond_data = {
            'Date': dates,
            'France': np.clip(np.random.normal(3.0, 1.5, len(dates)) + 
                             np.sin(np.linspace(0, 4*np.pi, len(dates))), 0, 8),
            'Portugal': np.clip(np.random.normal(4.5, 2.0, len(dates)) + 
                               np.sin(np.linspace(0, 4*np.pi, len(dates))), 0, 15),
            'Ireland': np.clip(np.random.normal(3.5, 2.5, len(dates)) + 
                              np.sin(np.linspace(0, 4*np.pi, len(dates))), 0, 12)
        }
        bond_df = pd.DataFrame(bond_data)
        bond_df.set_index('Date', inplace=True)
        
        # Sample debt to GDP data
        debt_data = {
            'Year': list(years) * 3,
            'Country': ['France'] * len(years) + ['Portugal'] * len(years) + ['Ireland'] * len(years),
            'Debt_to_GDP': (
                list(np.clip(60 + np.cumsum(np.random.normal(2.0, 1.5, len(years))), 50, 120)) +  # France
                list(np.clip(70 + np.cumsum(np.random.normal(2.5, 2.0, len(years))), 50, 140)) +  # Portugal
                list(np.clip(40 + np.cumsum(np.random.normal(1.5, 3.5, len(years))), 25, 130))    # Ireland
            )
        }
        debt_df = pd.DataFrame(debt_data)
        
        self.data['fiscal'] = fiscal_df
        self.data['unemployment'] = unemployment_df
        self.data['bond_yields'] = bond_df
        self.data['debt_to_gdp'] = debt_df
        
        return self.data
    
    def prepare_time_series(self, df: pd.DataFrame, 
                           value_col: str, 
                           country: str,
                           date_col: str = 'Year') -> pd.Series:
        """
        Prepare time series data for a specific country.
        
        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe
        value_col : str
            Column name containing the values
        country : str
            Country name to filter
        date_col : str
            Column name containing dates/years
            
        Returns
        -------
        pd.Series
            Time series data indexed by date
        """
        country_data = df[df['Country'] == country].copy()
        country_data = country_data.sort_values(date_col)
        
        if date_col == 'Year':
            index = pd.to_datetime(country_data[date_col], format='%Y')
        else:
            index = pd.to_datetime(country_data[date_col])
            
        return pd.Series(
            country_data[value_col].values,
            index=index,
            name=f"{country}_{value_col}"
        )
    
    def get_combined_dataset(self) -> pd.DataFrame:
        """
        Combine all datasets into a unified format.
        
        Returns
        -------
        pd.DataFrame
            Combined dataset with all indicators
        """
        if not self.data:
            self.create_sample_data()
            
        combined_list = []
        
        for key, df in self.data.items():
            if 'Country' in df.columns:
                combined_list.append(df)
                
        if combined_list:
            return pd.concat(combined_list, ignore_index=True)
        return pd.DataFrame()


class DataPreprocessor:
    """Preprocess and clean economic data for modeling."""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, 
                             method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        method : str
            Method for handling missing values ('interpolate', 'ffill', 'mean')
            
        Returns
        -------
        pd.DataFrame
            Dataframe with handled missing values
        """
        df = df.copy()
        
        if method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        elif method == 'ffill':
            df = df.ffill().bfill()
        elif method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            
        return df
    
    @staticmethod
    def normalize_data(df: pd.DataFrame, 
                      columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize specified columns using min-max scaling.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to normalize
            
        Returns
        -------
        Tuple[pd.DataFrame, Dict]
            Normalized dataframe and scaling parameters
        """
        df = df.copy()
        scaling_params = {}
        
        for col in columns:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val)
                scaling_params[col] = {'min': min_val, 'max': max_val}
                
        return df, scaling_params
    
    @staticmethod
    def create_lagged_features(df: pd.DataFrame, 
                               columns: List[str], 
                               lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series modeling.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (must be sorted by time)
        columns : List[str]
            Columns to create lags for
        lags : List[int]
            List of lag periods
            
        Returns
        -------
        pd.DataFrame
            Dataframe with lagged features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
                    
        return df
    
    @staticmethod
    def calculate_rolling_statistics(df: pd.DataFrame, 
                                    columns: List[str], 
                                    window: int = 3) -> pd.DataFrame:
        """
        Calculate rolling mean and standard deviation.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        columns : List[str]
            Columns to calculate statistics for
        window : int
            Rolling window size
            
        Returns
        -------
        pd.DataFrame
            Dataframe with rolling statistics
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=window).std()
                
        return df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    sample_data = loader.create_sample_data()
    
    print("Sample Data Created:")
    for key, df in sample_data.items():
        print(f"\n{key}:")
        print(df.head())
