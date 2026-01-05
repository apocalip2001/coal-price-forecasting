"""
Feature engineering for coal price forecasting.
Creates lagged features, rolling statistics, and technical indicators.
"""

import pandas as pd
import numpy as np

def create_lagged_features(df, columns, lags=[1, 2, 3, 6]):
    """
    Create lagged features for specified columns.
    
    Args:
        df: DataFrame with time series data
        columns: List of column names to create lags for
        lags: List of lag periods (in months)
    
    Returns:
        DataFrame with lagged features added
    """
    result = df.copy()
    
    for col in columns:
        if col in df.columns:
            for lag in lags:
                result[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return result

def create_rolling_features(df, columns, windows=[3, 6, 12]):
    """
    Create rolling mean and volatility features.
    
    Args:
        df: DataFrame with time series data
        columns: List of column names
        windows: List of rolling window sizes (in months)
    
    Returns:
        DataFrame with rolling features added
    """
    result = df.copy()
    
    for col in columns:
        if col in df.columns:
            for window in windows:
                # Rolling mean
                result[f'{col}_ma{window}'] = df[col].rolling(window=window).mean()
                # Rolling volatility (standard deviation)
                result[f'{col}_vol{window}'] = df[col].rolling(window=window).std()
    
    return result

def create_momentum_features(df, columns, periods=[1, 3, 6]):
    """
    Create momentum/rate of change features.
    
    Args:
        df: DataFrame with time series data
        columns: List of column names
        periods: List of periods for momentum calculation
    
    Returns:
        DataFrame with momentum features added
    """
    result = df.copy()
    
    for col in columns:
        if col in df.columns:
            for period in periods:
                result[f'{col}_mom{period}'] = df[col].pct_change(periods=period)
    
    return result

def create_calendar_features(df):
    """
    Create calendar-based features from the index.
    
    Args:
        df: DataFrame with DatetimeIndex
    
    Returns:
        DataFrame with calendar features added
    """
    result = df.copy()
    
    # Month dummies (January = 1, December = 12)
    result['month'] = df.index.month
    
    # Quarter
    result['quarter'] = df.index.quarter
    
    # Year (for trend)
    result['year'] = df.index.year
    
    # Create month dummies
    for month in range(1, 13):
        result[f'month_{month}'] = (df.index.month == month).astype(int)
    
    return result

def create_target_variable(df, target_col, forecast_horizon=1):
    """
    Create the target variable (future return).
    
    Args:
        df: DataFrame with price/return data
        target_col: Column name to use for target
        forecast_horizon: Number of periods ahead to forecast
    
    Returns:
        DataFrame with target variable added
    """
    result = df.copy()
    
    # Future return (what we're trying to predict)
    result['target'] = df[target_col].shift(-forecast_horizon)
    
    # Binary target (1 if positive return, 0 if negative)
    result['target_direction'] = (result['target'] > 0).astype(int)
    
    return result

def build_feature_set(df, target_col, lags=[1, 2, 3, 6], windows=[3, 6, 12]):
    """
    Build complete feature set for modeling.
    
    Args:
        df: Raw DataFrame with price/return data
        target_col: Column to use as target variable
        lags: Lag periods for lagged features
        windows: Window sizes for rolling features
    
    Returns:
        DataFrame with all features and target
    """
    # Get numeric columns for feature engineering
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create features
    result = create_lagged_features(df, numeric_cols, lags)
    result = create_rolling_features(result, numeric_cols, windows)
    result = create_momentum_features(result, numeric_cols, periods=[1, 3])
    result = create_calendar_features(result)
    result = create_target_variable(result, target_col, forecast_horizon=1)
    
    return result

def prepare_model_data(df, target_col='target', drop_na=True):
    """
    Prepare data for modeling by separating features and target.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        drop_na: Whether to drop rows with missing values
    
    Returns:
        X (features), y (target), feature_names
    """
    # Separate target columns
    target_cols = ['target', 'target_direction']
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy() if target_col in df.columns else None
    
    if drop_na:
        # Find rows where both X and y are not null
        valid_idx = X.notna().all(axis=1)
        if y is not None:
            valid_idx = valid_idx & y.notna()
        
        X = X[valid_idx]
        if y is not None:
            y = y[valid_idx]
    
    return X, y, feature_cols

if __name__ == "__main__":
    # Test feature engineering
    print("Testing feature engineering...")
    
    # Create sample data
    dates = pd.date_range(start='2010-01-01', end='2024-12-31', freq='ME')
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'coal_price': 100 + np.cumsum(np.random.randn(len(dates)) * 5),
        'oil_price': 80 + np.cumsum(np.random.randn(len(dates)) * 3),
    }, index=dates)
    
    # Calculate returns
    sample_data['coal_return'] = np.log(sample_data['coal_price'] / sample_data['coal_price'].shift(1))
    
    # Build features
    features = build_feature_set(sample_data, target_col='coal_return')
    
    print(f"\nOriginal shape: {sample_data.shape}")
    print(f"With features shape: {features.shape}")
    print(f"\nFeature columns ({len(features.columns)}):")
    print(features.columns.tolist())