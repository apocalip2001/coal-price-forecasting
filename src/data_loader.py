# Data loading utilities for coal price forecasting
# Downloads and processes market data from Yahoo Finance and FRED

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')


# Asset tickers and their descriptions
ASSETS = {
    # Coal proxy (China Yanzhou Coal Mining)
    'coal_china_yzcm': '1171.HK',
    
    # Energy commodities
    'brent_crude': 'BZ=F',
    'wti_crude': 'CL=F',
    'natural_gas_hh': 'NG=F',
    'gasoline': 'RB=F',
    'heating_oil': 'HO=F',
    
    # Sector ETFs
    'utilities_xlu': 'XLU',
    'clean_energy': 'ICLN',
    
    # Currency pairs
    'usd_index': 'DX-Y.NYB',
    'cny_usd': 'CNY=X',
    'eur_usd': 'EURUSD=X',
    'aud_usd': 'AUDUSD=X',
    
    # Regional/sector indices
    'china_etf_fxi': 'FXI',
    'emerging_mkts': 'EEM',
    'industrials_xli': 'XLI',
    'materials_xlb': 'XLB',
    'steel_slx': 'SLX',
}


def download_market_data(start_date='2015-01-01', end_date=None, save_path=None):
    """
    Download market data for all assets from Yahoo Finance.
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    save_path : str, optional
        Path to save the raw data CSV
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with adjusted close prices for all assets
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading data from {start_date} to {end_date}")
    print("=" * 50)
    
    all_data = {}
    
    for name, ticker in ASSETS.items():
        try:
            print(f"Downloading {name} ({ticker})...", end=" ")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                all_data[name] = data['Adj Close']
                print(f"✓ {len(data)} rows")
            else:
                print("✗ No data")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Combine all data
    df = pd.DataFrame(all_data)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Forward fill missing values (holidays differ across markets)
    df = df.ffill()
    
    # Drop rows where coal price is missing
    df = df.dropna(subset=['coal_china_yzcm'])
    
    print("=" * 50)
    print(f"Final dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path)
        print(f"Saved to {save_path}")
    
    return df


def load_data(filepath):
    """
    Load data from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame with date index
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def calculate_returns(df, method='simple'):
    """
    Calculate returns from price data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    method : str
        'simple' for simple returns, 'log' for log returns
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with returns
    """
    if method == 'simple':
        returns = df.pct_change()
    elif method == 'log':
        returns = np.log(df / df.shift(1))
    else:
        raise ValueError("method must be 'simple' or 'log'")
    
    # Rename columns to indicate returns
    returns.columns = [f"{col}_ret" for col in returns.columns]
    
    return returns


def prepare_dataset(raw_data_path=None, processed_data_path=None):
    """
    Full data preparation pipeline.
    
    Parameters:
    -----------
    raw_data_path : str, optional
        Path to raw price data. If None, downloads fresh data.
    processed_data_path : str, optional
        Path to save processed returns data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with returns for all assets
    """
    # Load or download data
    if raw_data_path and os.path.exists(raw_data_path):
        print(f"Loading data from {raw_data_path}")
        prices = load_data(raw_data_path)
    else:
        prices = download_market_data(save_path=raw_data_path)
    
    # Calculate returns
    returns = calculate_returns(prices, method='simple')
    
    # Drop first row (NaN from pct_change)
    returns = returns.dropna()
    
    # Add date column for convenience
    returns = returns.reset_index()
    returns = returns.rename(columns={'index': 'date'})
    
    if processed_data_path:
        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
        returns.to_csv(processed_data_path, index=False)
        print(f"Saved processed data to {processed_data_path}")
    
    return returns


def get_data_summary(df):
    """
    Print summary statistics for the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to summarize
    """
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    if 'date' in df.columns:
        print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print(f"\nMissing values:\n{df.isnull().sum().sum()} total")
    
    print("\nBasic statistics:")
    print(df.describe().round(4))


if __name__ == "__main__":
    # Example usage
    returns = prepare_dataset(
        raw_data_path='../data/raw/prices.csv',
        processed_data_path='../data/processed/returns.csv'
    )
    get_data_summary(returns)
