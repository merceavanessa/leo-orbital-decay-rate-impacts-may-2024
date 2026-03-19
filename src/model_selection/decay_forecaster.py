import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class DecayForecaster:
    """
    Class for training and evaluating orbital decay forecasting models.
    """
    
    def __init__(self, alpha=0.3):
        """
        Initialize the DecayForecaster.
        
        Parameters:
        -----------
        alpha : float, default=0.3
            Regularization strength for Lasso model
        """
        self.model = None
        self.scaler = StandardScaler()
        self.alpha = alpha
        
    def train(self, X, y, test_size=0.4, random_state=None, shuffle=False):
        """
        Train the decay forecasting model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature DataFrame
        y : pandas.DataFrame
            Target DataFrame
        test_size : float, default=0.4
            Proportion of data to use for testing
        random_state : int, optional
            Random state for reproducibility
        shuffle : bool, default=False
            Whether to shuffle data before splitting
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test, y_fit, y_pred)
        """
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train model
        self.model = Lasso(alpha=self.alpha)
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_fit = pd.DataFrame(
            self.model.predict(X_train_scaled), 
            index=X_train.index, 
            columns=y_train.columns
        )
        y_pred = pd.DataFrame(
            self.model.predict(X_test_scaled), 
            index=X_test.index, 
            columns=y_test.columns
        )
        
        return X_train, X_test, y_train, y_test, y_fit, y_pred
    
    def predict(self, X):
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : pandas.DataFrame
            Feature DataFrame
            
        Returns:
        --------
        pandas.DataFrame
            Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
            
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return pd.DataFrame(predictions, index=X.index, columns=self.model.feature_names_in_)
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance.
        
        Parameters:
        -----------
        y_true : pandas.DataFrame
            True values
        y_pred : pandas.DataFrame
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Overall metrics
        metrics['rmse'] = root_mean_squared_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Per-horizon metrics
        horizon_rmse = []
        horizon_r2 = []
        
        for i in range(y_true.shape[1]):
            horizon_rmse.append(root_mean_squared_error(y_true.iloc[:, i], y_pred.iloc[:, i]))
            horizon_r2.append(r2_score(y_true.iloc[:, i], y_pred.iloc[:, i]))
            
        metrics['horizon_rmse'] = horizon_rmse
        metrics['horizon_r2'] = horizon_r2
        
        return metrics
    
    def compute_prediction_bands(self, y_test, y_pred):
        """
        Compute prediction bands for uncertainty estimation.
        
        Parameters:
        -----------
        y_test : pandas.DataFrame
            True values
        y_pred : pandas.DataFrame
            Predicted values
            
        Returns:
        --------
        tuple
            (band_min, band_max, band_mean, band_sem)
        """
        T, N = y_test.shape
        
        band_min = np.full(T, np.nan)
        band_max = np.full(T, np.nan)
        band_mean = np.full(T, np.nan)
        band_sem = np.full(T, np.nan)
        
        for t in range(T):
            start = max(0, t - N)
            end = t
            preds = []
            for s in range(start, end):
                horizon = t - s - 1
                preds.append(y_pred.iloc[s, horizon])
            if preds:
                preds = np.array(preds)
                sem = preds.std(ddof=1) / np.sqrt(len(preds))
                band_min[t] = np.percentile(preds, 5)
                band_max[t] = np.percentile(preds, 95)
                band_mean[t] = preds.mean()
                band_sem[t] = sem
                
        return band_min, band_max, band_mean, band_sem