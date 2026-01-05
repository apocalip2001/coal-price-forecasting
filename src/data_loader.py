"""
Data loader for coal price forecasting project.
Downloads and processes data from FRED, Yahoo Finance, and other sources.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime
import os

# You'll need a free FRED API key from https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY = "your_fred_api_key_here"  # Replace with your actual key

def get_fred_data(fred, series_dict, start_date, end_date):
    """Download multiple series from FRED."""
    data = {}
    for name, series_id in series_dict.items():
        try:
            data[name] = fred.get_series(series_id, start_date, end_date)
            print(f"Downloaded {name}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
    return pd.DataFrame(data)

def get_yahoo_data(tickers_dict, start_date, end_date):
    """Download multiple series from Yahoo Finance."""
    data = {}
    for name, ticker in tickers_dict.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            data[name] = df['Close']
            print(f"Downloaded {name}")
        except Exception as e:
            print(f"Failed to download {name}: {e}")
    return pd.DataFrame(data)

def load_all_data(start_date="2010-01-01", end_date="2024-12-31", fred_api_key=None):
    """
    Load all data needed for coal price forecasting.
    Returns a DataFrame with monthly frequency.
    """
    if fred_api_key:
        fred = Fred(api_key=fred_api_key)
    else:
        fred = None
        print("Warning: No FRED API key provided. FRED data will be skipped.")
    
    # FRED series
    fred_series = {
        'industrial_production': 'INDPRO',      # Industrial Production Index
        'china_pmi': 'MPMICNMA',                  # China Manufacturing PMI (may not be available)
        'us_coal_production': 'IPG2121S',        # Coal mining industrial production
        'dxy_index': 'DTWEXBGS',                  # Trade Weighted US Dollar Index
    }
    
    # Yahoo Finance tickers
    yahoo_tickers = {
        'brent_crude': 'BZ=F',          # Brent Crude Futures
        'wti_crude': 'CL=F',            # WTI Crude Futures  
        'natural_gas': 'NG=F',          # Natural Gas Futures
        'coal_etf': 'KOL',              # VanEck Coal ETF (proxy for coal prices)
    }
    
    # Download FRED data
    if fred:
        fred_data = get_fred_data(fred, fred_series, start_date, end_date)
    else:
        fred_data = pd.DataFrame()
    
    # Download Yahoo data
    yahoo_data = get_yahoo_data(yahoo_tickers, start_date, end_date)
    
    # Resample to monthly (end of month)
    if not fred_data.empty:
        fred_monthly = fred_data.resample('ME').last()
    else:
        fred_monthly = pd.DataFrame()
        
    yahoo_monthly = yahoo_data.resample('ME').last()
    
    # Combine all data
    if not fred_monthly.empty:
        combined = pd.concat([fred_monthly, yahoo_monthly], axis=1)
    else:
        combined = yahoo_monthly
    
    # Forward fill missing values (up to 3 months)
    combined = combined.ffill(limit=3)
    
    return combined

def calculate_log_returns(df, columns=None):
    """Calculate log returns for specified columns."""
    if columns is None:
        columns = df.columns
    
    returns = pd.DataFrame(index=df.index)
    for col in columns:
        if col in df.columns:
            returns[f'{col}_return'] = np.log(df[col] / df[col].shift(1))
    
    return returns

def save_data(df, filepath):
    """Save DataFrame to CSV."""
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")

if __name__ == "__main__":
    # Test the data loader
    print("Testing data loader...")
    data = load_all_data(start_date="2010-01-01", end_date="2024-12-31")
    print(f"\nData shape: {data.shape}")
    print(f"\nColumns: {data.columns.tolist()}")
    print(f"\nDate range: {data.index.min()} to {data.index.max()}")
    print(f"\nMissing values:\n{data.isnull().sum()}")