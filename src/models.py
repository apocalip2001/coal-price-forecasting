"""
ML models for coal price forecasting.
Includes Ridge, Decision Tree, XGBoost, and LSTM models.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

class BaseModel:
    """Base class for all models."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        raise NotImplementedError
    
    def predict(self, X):
        raise NotImplementedError
    
    def get_params(self):
        return {}

class RidgeModel(BaseModel):
    """Ridge Regression model."""
    
    def __init__(self, alpha=1.0):
        super().__init__("Ridge")
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
    
    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from coefficients."""
        importance = np.abs(self.model.coef_)
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

class DecisionTreeModel(BaseModel):
    """Decision Tree Regressor model."""
    
    def __init__(self, max_depth=5, min_samples_leaf=5, min_samples_split=10):
        super().__init__("DecisionTree")
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            random_state=42
        )
    
    def fit(self, X, y):
        # No scaling needed for tree models
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from tree."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

class XGBoostModel(BaseModel):
    """XGBoost model."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        super().__init__("XGBoost")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbosity=0
        )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance from XGBoost."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)

class LSTMModel(BaseModel):
    """LSTM model for sequence modeling."""
    
    def __init__(self, units=50, dropout=0.2, sequence_length=6, epochs=50, batch_size=16):
        super().__init__("LSTM")
        self.units = units
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
    
    def _build_model(self, n_features):
        """Build LSTM architecture."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        
        model = Sequential([
            LSTM(self.units, input_shape=(self.sequence_length, n_features), return_sequences=True),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _create_sequences(self, X, y=None):
        """Create sequences for LSTM input."""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            if y is not None:
                y_seq.append(y[i + self.sequence_length])
        
        X_seq = np.array(X_seq)
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq
    
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y.values)
        
        # Build and train model
        self.model = self._build_model(X.shape[1])
        self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=0
        )
        self.is_fitted = True
        return self
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_seq = self._create_sequences(X_scaled)
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()

def time_series_cv(model, X, y, n_splits=5):
    """
    Perform time series cross-validation.
    
    Args:
        model: Model instance with fit/predict methods
        X: Features DataFrame
        y: Target Series
        n_splits: Number of CV folds
    
    Returns:
        Dictionary with CV results
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    train_scores = []
    val_scores = []
    predictions = []
    actuals = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Clone model for each fold
        model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
        
        # Fit and predict
        model_clone.fit(X_train, y_train)
        
        train_pred = model_clone.predict(X_train)
        val_pred = model_clone.predict(X_val)
        
        # Calculate RMSE
        train_rmse = np.sqrt(np.mean((y_train - train_pred) ** 2))
        val_rmse = np.sqrt(np.mean((y_val - val_pred) ** 2))
        
        train_scores.append(train_rmse)
        val_scores.append(val_rmse)
        predictions.extend(val_pred)
        actuals.extend(y_val.values)
        
        print(f"Fold {fold + 1}: Train RMSE = {train_rmse:.4f}, Val RMSE = {val_rmse:.4f}")
    
    return {
        'train_rmse_mean': np.mean(train_scores),
        'train_rmse_std': np.std(train_scores),
        'val_rmse_mean': np.mean(val_scores),
        'val_rmse_std': np.std(val_scores),
        'predictions': predictions,
        'actuals': actuals
    }

if __name__ == "__main__":
    # Test models
    print("Testing models...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples), name='target')
    
    # Test each model
    models = [
        RidgeModel(alpha=1.0),
        DecisionTreeModel(max_depth=5),
        XGBoostModel(n_estimators=50),
    ]
    
    for model in models:
        print(f"\n--- {model.name} ---")
        model.fit(X, y)
        preds = model.predict(X)
        rmse = np.sqrt(np.mean((y - preds) ** 2))
        print(f"Training RMSE: {rmse:.4f}")