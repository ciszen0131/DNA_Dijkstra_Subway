"""
Time Series Models for Government Debt Policy Analysis

This module implements ARIMA and SARIMAX models for forecasting
economic indicators related to government debt policy.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller, acf, pacf
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Install with: pip install statsmodels")


class ARIMAModel:
    """
    ARIMA model for time series forecasting.
    
    Implements AutoRegressive Integrated Moving Average model
    for economic indicator forecasting.
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA model.
        
        Parameters
        ----------
        order : Tuple[int, int, int]
            (p, d, q) order of the ARIMA model
            p: autoregressive order
            d: differencing order
            q: moving average order
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.residuals = None
        
    def check_stationarity(self, series: pd.Series) -> Dict:
        """
        Check stationarity of the time series using ADF test.
        
        Parameters
        ----------
        series : pd.Series
            Time series data
            
        Returns
        -------
        Dict
            Dictionary with test results
        """
        if not STATSMODELS_AVAILABLE:
            return {"error": "statsmodels not available"}
            
        result = adfuller(series.dropna())
        return {
            'test_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def fit(self, series: pd.Series, 
            auto_order: bool = False) -> 'ARIMAModel':
        """
        Fit the ARIMA model to the data.
        
        Parameters
        ----------
        series : pd.Series
            Time series data to fit
        auto_order : bool
            If True, automatically determine optimal order
            
        Returns
        -------
        ARIMAModel
            Self, for method chaining
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA modeling")
            
        if auto_order:
            self.order = self._find_optimal_order(series)
            
        self.model = ARIMA(series, order=self.order)
        self.fitted_model = self.model.fit()
        self.residuals = self.fitted_model.resid
        
        return self
    
    def _find_optimal_order(self, series: pd.Series,
                           max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using AIC criterion.
        
        Parameters
        ----------
        series : pd.Series
            Time series data
        max_p, max_d, max_q : int
            Maximum values for each parameter
            
        Returns
        -------
        Tuple[int, int, int]
            Optimal (p, d, q) order
        """
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                        
        return best_order
    
    def predict(self, steps: int = 12) -> pd.Series:
        """
        Generate predictions for future periods.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
            
        Returns
        -------
        pd.Series
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
            
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_confidence_interval(self, steps: int = 12, 
                                alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
        """
        Get confidence intervals for the forecast.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        alpha : float
            Significance level (default 0.05 for 95% CI)
            
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Lower and upper bounds of confidence interval
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before getting confidence intervals")
            
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        return conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    
    def get_summary(self) -> str:
        """
        Get model summary statistics.
        
        Returns
        -------
        str
            Model summary
        """
        if self.fitted_model is None:
            return "Model not fitted yet"
        return str(self.fitted_model.summary())
    
    def get_metrics(self) -> Dict:
        """
        Get model performance metrics.
        
        Returns
        -------
        Dict
            Dictionary containing AIC, BIC, and other metrics
        """
        if self.fitted_model is None:
            return {}
            
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'llf': self.fitted_model.llf
        }


class SARIMAXModel:
    """
    SARIMAX model for time series forecasting with seasonality and exogenous variables.
    
    Implements Seasonal AutoRegressive Integrated Moving Average with eXogenous factors
    for economic indicator forecasting that accounts for seasonal patterns and
    external policy interventions.
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        """
        Initialize SARIMAX model.
        
        Parameters
        ----------
        order : Tuple[int, int, int]
            (p, d, q) order for non-seasonal components
        seasonal_order : Tuple[int, int, int, int]
            (P, D, Q, s) order for seasonal components
            s: seasonal period
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.exog_columns = None
        
    def fit(self, series: pd.Series,
            exog: Optional[pd.DataFrame] = None) -> 'SARIMAXModel':
        """
        Fit the SARIMAX model to the data.
        
        Parameters
        ----------
        series : pd.Series
            Endogenous time series data
        exog : pd.DataFrame, optional
            Exogenous variables (policy indicators, etc.)
            
        Returns
        -------
        SARIMAXModel
            Self, for method chaining
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for SARIMAX modeling")
            
        if exog is not None:
            self.exog_columns = exog.columns.tolist()
            
        self.model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            exog=exog,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        self.fitted_model = self.model.fit(disp=False)
        
        return self
    
    def predict(self, steps: int = 12,
                exog_forecast: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate predictions for future periods.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        exog_forecast : pd.DataFrame, optional
            Future values of exogenous variables
            
        Returns
        -------
        pd.Series
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
            
        forecast = self.fitted_model.forecast(steps=steps, exog=exog_forecast)
        return forecast
    
    def get_confidence_interval(self, steps: int = 12,
                                exog_forecast: Optional[pd.DataFrame] = None,
                                alpha: float = 0.05) -> Tuple[pd.Series, pd.Series]:
        """
        Get confidence intervals for the forecast.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        exog_forecast : pd.DataFrame, optional
            Future values of exogenous variables
        alpha : float
            Significance level
            
        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Lower and upper bounds
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
            
        forecast_result = self.fitted_model.get_forecast(steps=steps, exog=exog_forecast)
        conf_int = forecast_result.conf_int(alpha=alpha)
        
        return conf_int.iloc[:, 0], conf_int.iloc[:, 1]
    
    def simulate_policy_intervention(self,
                                    base_series: pd.Series,
                                    intervention_effect: float,
                                    intervention_start: int,
                                    steps: int = 24) -> Dict[str, pd.Series]:
        """
        Simulate the effect of a policy intervention.
        
        Parameters
        ----------
        base_series : pd.Series
            Historical time series
        intervention_effect : float
            Expected effect of the intervention (e.g., -0.02 for 2% reduction)
        intervention_start : int
            Period when intervention begins
        steps : int
            Number of periods to simulate
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary with 'with_policy' and 'without_policy' forecasts
        """
        # Create intervention dummy variable
        intervention = np.zeros(len(base_series) + steps)
        intervention[len(base_series) + intervention_start:] = 1
        
        # Forecast without intervention
        self.fit(base_series)
        forecast_no_policy = self.predict(steps=steps)
        
        # Add intervention effect
        forecast_with_policy = forecast_no_policy.copy()
        intervention_periods = max(0, steps - intervention_start)
        if intervention_periods > 0:
            cumulative_effect = np.cumsum([intervention_effect] * intervention_periods)
            forecast_with_policy.iloc[-intervention_periods:] += cumulative_effect
        
        return {
            'with_policy': forecast_with_policy,
            'without_policy': forecast_no_policy
        }
    
    def get_summary(self) -> str:
        """Get model summary."""
        if self.fitted_model is None:
            return "Model not fitted yet"
        return str(self.fitted_model.summary())
    
    def get_metrics(self) -> Dict:
        """Get model performance metrics."""
        if self.fitted_model is None:
            return {}
            
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'hqic': self.fitted_model.hqic,
            'llf': self.fitted_model.llf
        }


class TimeSeriesAnalyzer:
    """Helper class for time series analysis and diagnostics."""
    
    @staticmethod
    def decompose_series(series: pd.Series, 
                        period: int = 12) -> Dict:
        """
        Decompose time series into trend, seasonal, and residual components.
        
        Parameters
        ----------
        series : pd.Series
            Time series to decompose
        period : int
            Seasonal period
            
        Returns
        -------
        Dict
            Dictionary with trend, seasonal, and residual components
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            result = seasonal_decompose(series, model='additive', period=period)
            
            return {
                'trend': result.trend,
                'seasonal': result.seasonal,
                'residual': result.resid
            }
        except ImportError:
            return {"error": "statsmodels not available"}
    
    @staticmethod
    def calculate_forecast_metrics(actual: pd.Series, 
                                  predicted: pd.Series) -> Dict:
        """
        Calculate forecast accuracy metrics.
        
        Parameters
        ----------
        actual : pd.Series
            Actual values
        predicted : pd.Series
            Predicted values
            
        Returns
        -------
        Dict
            Dictionary containing MAE, MSE, RMSE, and MAPE
        """
        actual = actual.values
        predicted = predicted.values
        
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        
        # MAPE (avoiding division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            mape = np.nan_to_num(mape, nan=0.0, posinf=0.0, neginf=0.0)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape
        }
    
    @staticmethod
    def cross_validate_model(model_class, series: pd.Series,
                            n_splits: int = 5,
                            forecast_horizon: int = 12,
                            **model_params) -> List[Dict]:
        """
        Perform time series cross-validation.
        
        Parameters
        ----------
        model_class : class
            Model class (ARIMAModel or SARIMAXModel)
        series : pd.Series
            Time series data
        n_splits : int
            Number of cross-validation splits
        forecast_horizon : int
            Forecast horizon for each split
        **model_params
            Parameters to pass to the model
            
        Returns
        -------
        List[Dict]
            List of metrics for each split
        """
        results = []
        n = len(series)
        train_size = n - (n_splits * forecast_horizon)
        
        for i in range(n_splits):
            train_end = train_size + (i * forecast_horizon)
            test_end = train_end + forecast_horizon
            
            if test_end > n:
                break
                
            train = series.iloc[:train_end]
            test = series.iloc[train_end:test_end]
            
            model = model_class(**model_params)
            model.fit(train)
            predictions = model.predict(steps=forecast_horizon)
            
            metrics = TimeSeriesAnalyzer.calculate_forecast_metrics(test, predictions)
            results.append(metrics)
        
        return results


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data_preparation import DataLoader
    
    # Load sample data
    loader = DataLoader()
    data = loader.create_sample_data()
    
    # Prepare time series for France
    france_fiscal = loader.prepare_time_series(
        data['fiscal'], 
        'Fiscal_Balance', 
        'France'
    )
    
    # Fit ARIMA model
    arima = ARIMAModel(order=(1, 1, 1))
    arima.fit(france_fiscal)
    
    print("ARIMA Model Summary:")
    print(arima.get_summary())
    
    # Generate forecast
    forecast = arima.predict(steps=5)
    print("\nForecast:")
    print(forecast)
