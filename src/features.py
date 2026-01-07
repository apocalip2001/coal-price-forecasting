# Feature engineering module
# Creates lagged features, rolling statistics, and calendar features

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        Columns to create lags for
    lags : list
        List of lag periods (e.g., [1, 2, 5, 10, 21])
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features added
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for lag in lags:
            result[f'{col}_lag{lag}'] = df[col].shift(lag)
    
    return result


def create_rolling_features(df: pd.DataFrame, columns: List[str], 
                           windows: List[int], 
                           functions: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        Columns to create rolling features for
    windows : list
        Window sizes (e.g., [5, 10, 21, 63])
    functions : list
        Rolling functions to apply ('mean', 'std', 'min', 'max', 'sum')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features added
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            rolling = df[col].rolling(window=window)
            
            if 'mean' in functions:
                result[f'{col}_ma{window}'] = rolling.mean()
            if 'std' in functions:
                result[f'{col}_vol{window}'] = rolling.std()
            if 'min' in functions:
                result[f'{col}_min{window}'] = rolling.min()
            if 'max' in functions:
                result[f'{col}_max{window}'] = rolling.max()
            if 'sum' in functions:
                result[f'{col}_sum{window}'] = rolling.sum()
    
    return result


def create_momentum_features(df: pd.DataFrame, columns: List[str], 
                            periods: List[int] = [5, 10, 21]) -> pd.DataFrame:
    """
    Create momentum/change features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        Columns to create momentum features for
    periods : list
        Periods for momentum calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with momentum features added
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for period in periods:
            # Price change over period
            result[f'{col}_mom{period}'] = df[col].diff(period)
            
            # Rate of change
            result[f'{col}_roc{period}'] = df[col].pct_change(period)
    
    return result


def create_volatility_features(df: pd.DataFrame, columns: List[str],
                               windows: List[int] = [5, 10, 21]) -> pd.DataFrame:
    """
    Create volatility-based features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        Columns to create volatility features for
    windows : list
        Windows for volatility calculation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with volatility features added
    """
    result = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        for window in windows:
            # Rolling standard deviation (volatility)
            vol = df[col].rolling(window=window).std()
            result[f'{col}_vol{window}'] = vol
            
            # Normalized volatility (current vol vs longer-term)
            if window < 21:
                long_vol = df[col].rolling(window=63).std()
                result[f'{col}_vol_ratio{window}'] = vol / long_vol
    
    return result


def create_technical_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Create technical analysis indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str
        Column name for price/returns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with technical indicators added
    """
    result = df.copy()
    
    if price_col not in df.columns:
        return result
    
    price = df[price_col]
    
    # Moving average crossovers
    ma5 = price.rolling(5).mean()
    ma21 = price.rolling(21).mean()
    ma63 = price.rolling(63).mean()
    
    result[f'{price_col}_ma_cross_5_21'] = (ma5 > ma21).astype(int)
    result[f'{price_col}_ma_cross_21_63'] = (ma21 > ma63).astype(int)
    
    # Distance from moving averages
    result[f'{price_col}_dist_ma21'] = price - ma21
    result[f'{price_col}_dist_ma63'] = price - ma63
    
    # Bollinger Band position
    ma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20
    result[f'{price_col}_bb_position'] = (price - lower_band) / (upper_band - lower_band)
    
    # RSI (Relative Strength Index)
    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result[f'{price_col}_rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    result[f'{price_col}_macd'] = ema12 - ema26
    result[f'{price_col}_macd_signal'] = result[f'{price_col}_macd'].ewm(span=9, adjust=False).mean()
    
    return result


def create_cross_asset_features(df: pd.DataFrame, 
                                target_col: str,
                                other_cols: List[str],
                                windows: List[int] = [21, 63]) -> pd.DataFrame:
    """
    Create cross-asset correlation features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    target_col : str
        Target column (coal)
    other_cols : list
        Other asset columns
    windows : list
        Windows for rolling correlation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with cross-asset features added
    """
    result = df.copy()
    
    if target_col not in df.columns:
        return result
    
    for col in other_cols:
        if col not in df.columns:
            continue
        for window in windows:
            corr = df[target_col].rolling(window=window).corr(df[col])
            result[f'{target_col}_{col}_corr{window}'] = corr
    
    return result


def create_calendar_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Create calendar-based features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    date_col : str
        Name of date column
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with calendar features added
    """
    result = df.copy()
    
    if date_col not in df.columns:
        return result
    
    dates = pd.to_datetime(df[date_col])
    
    # Day of week (0=Monday, 4=Friday)
    result['day_of_week'] = dates.dt.dayofweek
    
    # Month
    result['month'] = dates.dt.month
    
    # Quarter
    result['quarter'] = dates.dt.quarter
    
    # Week of year
    result['week_of_year'] = dates.dt.isocalendar().week.astype(int)
    
    # Is month end
    result['is_month_end'] = dates.dt.is_month_end.astype(int)
    
    # Is month start
    result['is_month_start'] = dates.dt.is_month_start.astype(int)
    
    return result


def build_feature_set(df: pd.DataFrame, 
                      target_col: str = 'coal_china_yzcm_ret',
                      include_calendar: bool = True,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Build complete feature set for coal price forecasting.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with returns data
    target_col : str
        Target column name
    include_calendar : bool
        Whether to include calendar features
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with all engineered features
    """
    if verbose:
        print("Building feature set...")
        print("=" * 50)
    
    # Get all return columns
    ret_cols = [col for col in df.columns if col.endswith('_ret')]
    
    # Define feature parameters
    lags = [1, 2, 5, 10, 21]
    rolling_windows = [5, 10, 21, 63]
    momentum_periods = [5, 10, 21]
    
    result = df.copy()
    
    # 1. Lag features
    if verbose:
        print("Creating lag features...")
    result = create_lag_features(result, ret_cols, lags)
    
    # 2. Rolling features
    if verbose:
        print("Creating rolling features...")
    result = create_rolling_features(result, ret_cols, rolling_windows, ['mean', 'std'])
    
    # 3. Momentum features
    if verbose:
        print("Creating momentum features...")
    result = create_momentum_features(result, [target_col], momentum_periods)
    
    # 4. Technical indicators for target
    if verbose:
        print("Creating technical indicators...")
    result = create_technical_indicators(result, target_col)
    
    # 5. Cross-asset features
    if verbose:
        print("Creating cross-asset features...")
    other_cols = [col for col in ret_cols if col != target_col]
    result = create_cross_asset_features(result, target_col, other_cols[:5], [21])
    
    # 6. Calendar features
    if include_calendar and 'date' in df.columns:
        if verbose:
            print("Creating calendar features...")
        result = create_calendar_features(result, 'date')
    
    # Drop rows with NaN (from lagging and rolling)
    initial_rows = len(result)
    result = result.dropna()
    dropped_rows = initial_rows - len(result)
    
    if verbose:
        print("=" * 50)
        print(f"Feature engineering complete!")
        print(f"Total features: {len(result.columns)}")
        print(f"Rows dropped (insufficient history): {dropped_rows}")
        print(f"Final dataset: {len(result)} rows")
    
    return result


def get_feature_target_split(df: pd.DataFrame, 
                             target_col: str = 'coal_china_yzcm_ret',
                             exclude_cols: List[str] = ['date']) -> tuple:
    """
    Split DataFrame into features and target.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features and target
    target_col : str
        Name of target column
    exclude_cols : list
        Columns to exclude from features
        
    Returns:
    --------
    tuple
        (X, y) where X is features DataFrame and y is target Series
    """
    feature_cols = [col for col in df.columns 
                   if col != target_col and col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y


if __name__ == "__main__":
    # Example usage
    import data_loader
    
    # Load returns data
    returns = data_loader.load_data('../data/processed/returns.csv')
    
    # Build features
    features_df = build_feature_set(returns)
    
    # Split into X and y
    X, y = get_feature_target_split(features_df)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
