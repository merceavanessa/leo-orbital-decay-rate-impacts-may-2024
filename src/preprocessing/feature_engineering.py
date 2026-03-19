import numpy as np
import pandas as pd
from numpy.polynomial import Polynomial

class FeatureEngineer:
    """
    Class for creating features for orbital decay forecasting.
    """
    
    def __init__(self):
        """
        Initialize the FeatureEngineer.
        """
        pass
    
    def add_harmonics(self, df, polynomial_coeffs=None, num_harmonics=4):
        """
        Add harmonic features based on orbital period.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with time index
        polynomial_coeffs : list, optional
            Coefficients for the polynomial that calculates orbital period
            Default is [1.06605722e+02, -7.21846851e-09]
        num_harmonics : int, default=4
            Number of harmonic features to create
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added harmonic features
        """
        df = df.copy()
        
        if polynomial_coeffs is None:
            polynomial_coeffs = [1.06605722e+02, -7.21846851e-09]
            
        p = Polynomial(polynomial_coeffs)
        
        # Calculate orbital period in seconds
        orbital_period = p(df.index.astype(np.int64) / 1e9) * 60
        
        # Calculate time difference in seconds
        dt_seconds = (df.index[1] - df.index[0]).total_seconds()
        
        # Calculate cumulative orbital cycles
        cycles = np.cumsum(1 / orbital_period) * dt_seconds
        
        # Calculate phase angle
        df["phase"] = (2 * np.pi * cycles) % (2 * np.pi)
        
        # Add harmonic features
        for n in range(1, num_harmonics + 1):
            df[f'sin_harmonic_{n}'] = np.sin(n * df["phase"])
            df[f'cos_harmonic_{n}'] = np.cos(n * df["phase"])
            
        return df
    
    def add_rolling_statistics(self, df, column='orbital_decay', windows=None):
        """
        Add rolling statistics for a column.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        column : str, default='orbital_decay'
            Column to calculate rolling statistics for
        windows : list, optional
            List of window sizes in days
            Default is [7, 14, 30]
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added rolling statistics
        """
        df = df.copy()
        
        if windows is None:
            windows = [7, 14, 30]
            
        # Calculate time difference in seconds
        dt_seconds = (df.index[1] - df.index[0]).total_seconds()
        
        for window in windows:
            # Convert window from days to number of samples
            window_samples = int(window * 24 * 60 * (60 // dt_seconds))
            df[f'median_{column}_last_{window}D'] = df[column].rolling(
                window=window_samples, min_periods=1
            ).median()
            
        return df
    
    def make_lags(self, df, lag_config):
        """
        Create lagged features according to configuration.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        lag_config : dict
            Configuration for lag creation
            Format: {column_name: {'lags': [lag1, lag2, ...]}}
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with lagged features
        """
        out = {}
        
        for col, params in lag_config.items():
            n_lags = params.get('lags', [1])
            
            for i in n_lags:
                out[f"{col}_lag_{i}"] = df[col].shift(i)
                
        return pd.DataFrame(out)
    
    def make_multistep_target(self, ts, steps, step_size=1):
        """
        Create multi-step target for forecasting.
        
        Parameters:
        -----------
        ts : pandas.Series
            Time series to forecast
        steps : int
            Number of steps to forecast
        step_size : int, default=1
            Size of each step (to skip intermediate steps)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with multi-step targets
        """
        if step_size == 1:
            return pd.concat(
                {f'y_step_{i + 1}': ts.shift(-i) for i in range(steps)},
                axis=1
            )
        else:
            return pd.concat(
                {f'y_step_{i + 1}': ts.shift(-i) for i in range(0, steps, step_size)},
                axis=1
            )
    
    def prepare_features_and_target(self, df, target_column, lag_config, forecast_horizon_hours, 
                                   target_step_minutes=15, add_harmonics=True, add_rolling_stats=True):
        """
        Complete pipeline for feature engineering.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        target_column : str
            Column to forecast
        lag_config : dict
            Configuration for lag creation
        forecast_horizon_hours : int
            Number of hours to forecast
        target_step_minutes : int, default=15
            Interval between forecast steps in minutes
        add_harmonics : bool, default=True
            Whether to add harmonic features
        add_rolling_stats : bool, default=True
            Whether to add rolling statistics
            
        Returns:
        --------
        tuple
            (X, y) - feature DataFrame and target DataFrame
        """
        processed_df = df.copy()
        
        # Calculate time difference in seconds
        dt_seconds = (processed_df.index[1] - processed_df.index[0]).total_seconds()
        
        # Add harmonics if requested
        if add_harmonics:
            processed_df = self.add_harmonics(processed_df)
            
        # Add rolling statistics if requested
        if add_rolling_stats:
            processed_df = self.add_rolling_statistics(processed_df, column=target_column)
            
        # Create lagged features
        X = self.make_lags(processed_df, lag_config)
        
        # Create multi-step target
        steps = int(forecast_horizon_hours * 60 * (60 // dt_seconds))
        step_size = int(target_step_minutes * (60 // dt_seconds))
        y = self.make_multistep_target(processed_df[[target_column]], steps, step_size)
        
        # Remove rows with NaN values
        na_mask = X.isna().any(axis=1) | y.isna().any(axis=1)
        X = X[~na_mask]
        y = y[~na_mask]
        
        # Rename columns to avoid MultiIndex issues
        X.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in X.columns]
        
        return X, y