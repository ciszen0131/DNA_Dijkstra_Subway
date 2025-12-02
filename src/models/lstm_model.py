"""
LSTM Model for Government Debt Policy Analysis

This module implements Long Short-Term Memory neural networks
for time series forecasting of economic indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

from sklearn.preprocessing import MinMaxScaler


class LSTMPredictor:
    """
    LSTM-based predictor for time series forecasting.
    
    Implements a Long Short-Term Memory neural network for
    deep learning-based economic indicator prediction.
    """
    
    def __init__(self, 
                 sequence_length: int = 12,
                 n_features: int = 1,
                 n_units: List[int] = [50, 50],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM predictor.
        
        Parameters
        ----------
        sequence_length : int
            Number of past time steps to use for prediction
        n_features : int
            Number of input features
        n_units : List[int]
            Number of LSTM units in each layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def _build_model(self) -> None:
        """Build the LSTM model architecture."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=self.n_units[0],
            return_sequences=len(self.n_units) > 1,
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.n_units[1:]):
            return_seq = i < len(self.n_units) - 2
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(1))
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Parameters
        ----------
        data : np.ndarray
            Scaled input data
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            X (sequences) and y (targets)
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, series: pd.Series,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.2,
            verbose: int = 0) -> 'LSTMPredictor':
        """
        Fit the LSTM model to the data.
        
        Parameters
        ----------
        series : pd.Series
            Time series data to fit
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        validation_split : float
            Fraction of data to use for validation
        verbose : int
            Verbosity level
            
        Returns
        -------
        LSTMPredictor
            Self, for method chaining
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        # Scale data
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        
        # Build model
        self._build_model()
        
        # Early stopping callback
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        # Store the last sequence for prediction
        self.last_sequence = scaled_data[-self.sequence_length:]
        
        return self
    
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
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            input_seq = current_sequence.reshape((1, self.sequence_length, self.n_features))
            
            # Predict next value
            next_pred = self.model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], [[next_pred]], axis=0)
        
        # Inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return pd.Series(predictions.flatten())
    
    def get_training_history(self) -> Dict:
        """
        Get training history.
        
        Returns
        -------
        Dict
            Dictionary containing training metrics history
        """
        if self.history is None:
            return {}
            
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history.get('val_loss', [])
        }
    
    def evaluate(self, series: pd.Series) -> Dict:
        """
        Evaluate model on test data.
        
        Parameters
        ----------
        series : pd.Series
            Test time series
            
        Returns
        -------
        Dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
            
        # Scale data
        data = series.values.reshape(-1, 1)
        scaled_data = self.scaler.transform(data)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        
        # Evaluate
        loss = self.model.evaluate(X, y, verbose=0)
        
        # Make predictions
        predictions = self.model.predict(X, verbose=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(predictions)
        actual = self.scaler.inverse_transform(y.reshape(-1, 1))
        
        # Calculate metrics
        mae = np.mean(np.abs(actual - predictions))
        rmse = np.sqrt(np.mean((actual - predictions) ** 2))
        
        return {
            'loss': loss,
            'mae': mae,
            'rmse': rmse
        }


class MultivariateLSTM:
    """
    Multivariate LSTM for multi-feature time series forecasting.
    
    This model can incorporate multiple economic indicators
    for more comprehensive predictions.
    """
    
    def __init__(self,
                 sequence_length: int = 12,
                 n_units: List[int] = [64, 32],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize Multivariate LSTM.
        
        Parameters
        ----------
        sequence_length : int
            Number of past time steps
        n_units : List[int]
            LSTM units per layer
        dropout_rate : float
            Dropout rate
        learning_rate : float
            Learning rate
        """
        self.sequence_length = sequence_length
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scalers = {}
        self.feature_names = None
        self.history = None
    
    def _build_model(self, n_features: int, n_outputs: int) -> None:
        """Build multivariate LSTM architecture."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=self.n_units[0],
            return_sequences=len(self.n_units) > 1,
            input_shape=(self.sequence_length, n_features)
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.n_units[1:]):
            return_seq = i < len(self.n_units) - 2
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(n_outputs))
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )
    
    def fit(self, df: pd.DataFrame,
            target_col: str,
            epochs: int = 100,
            batch_size: int = 32,
            validation_split: float = 0.2,
            verbose: int = 0) -> 'MultivariateLSTM':
        """
        Fit the multivariate LSTM model.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with multiple features
        target_col : str
            Name of the target column
        epochs : int
            Training epochs
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        verbose : int
            Verbosity level
            
        Returns
        -------
        MultivariateLSTM
            Self, for method chaining
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM models")
            
        self.feature_names = df.columns.tolist()
        self.target_col = target_col
        
        # Scale each feature
        scaled_data = np.zeros_like(df.values, dtype=np.float32)
        for i, col in enumerate(df.columns):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data[:, i] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            self.scalers[col] = scaler
        
        # Create sequences
        X, y = self._create_sequences(scaled_data, df.columns.get_loc(target_col))
        
        # Build and train model
        n_features = len(self.feature_names)
        self._build_model(n_features, 1)
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        self.last_sequence = scaled_data[-self.sequence_length:]
        
        return self
    
    def _create_sequences(self, data: np.ndarray, 
                         target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, target_idx])
        return np.array(X), np.array(y)
    
    def predict(self, steps: int = 12,
                future_features: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate predictions for future periods.
        
        Parameters
        ----------
        steps : int
            Number of steps to forecast
        future_features : pd.DataFrame, optional
            Known future values of features (if available)
            
        Returns
        -------
        pd.Series
            Forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
            
        predictions = []
        current_sequence = self.last_sequence.copy()
        
        for i in range(steps):
            input_seq = current_sequence.reshape((1, self.sequence_length, len(self.feature_names)))
            next_pred = self.model.predict(input_seq, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence
            if future_features is not None and i < len(future_features):
                new_row = np.zeros(len(self.feature_names))
                for j, col in enumerate(self.feature_names):
                    if col == self.target_col:
                        new_row[j] = next_pred
                    elif col in future_features.columns:
                        val = future_features[col].iloc[i]
                        new_row[j] = self.scalers[col].transform([[val]])[0, 0]
            else:
                new_row = current_sequence[-1].copy()
                target_idx = self.feature_names.index(self.target_col)
                new_row[target_idx] = next_pred
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scalers[self.target_col].inverse_transform(predictions)
        
        return pd.Series(predictions.flatten())


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    from data_preparation import DataLoader
    
    # Load sample data
    loader = DataLoader()
    data = loader.create_sample_data()
    
    # Prepare time series
    france_fiscal = loader.prepare_time_series(
        data['fiscal'],
        'Fiscal_Balance',
        'France'
    )
    
    # Check if TensorFlow is available
    if TF_AVAILABLE:
        # Train LSTM model
        lstm = LSTMPredictor(sequence_length=5, n_units=[32, 16])
        lstm.fit(france_fiscal, epochs=50, verbose=1)
        
        # Generate forecast
        forecast = lstm.predict(steps=5)
        print("LSTM Forecast:")
        print(forecast)
    else:
        print("TensorFlow not available. Skipping LSTM example.")
