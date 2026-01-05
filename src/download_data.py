"""
Download all data for thermal coal price forecasting project.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import os

# Load API key from .env file
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

def download_all_data(start_date="2010-01-01", end_date="2024-12-31"):
    """Download all required data for the project."""
    
    print("="*60)
    print("THERMAL COAL PRICE FORECASTING - DATA DOWNLOAD")
    print("="*60)
    
    # Initialize FRED
    fred = Fred(api_key=FRED_API_KEY)
    
    dataframes = []
    
    # ============================================
    # 1. FRED DATA (Macro & Industrial)
    # ============================================
    print("\n[1/4] Downloading FRED data...")
    
    fred_series = {
        'ind_prod_usa': 'INDPRO',
        'coal_prod_usa': 'IPG2121S',
        'dxy_index': 'DTWEXBGS',
        'cny_usd': 'DEXCHUS',
        'eur_usd': 'DEXUSEU',
        'fed_funds': 'FEDFUNDS',
    }
    
    for name, series_id in fred_series.items():
        try:
            data = fred.get_series(series_id, start_date, end_date)
            if data is not None and len(data) > 0:
                df_temp = pd.DataFrame({name: data})
                dataframes.append(df_temp)
                print(f"  ✓ {name} ({series_id}): {len(data)} obs")
        except Exception as e:
            print(f"  ✗ {name} ({series_id}): {e}")
    
    # ============================================
    # 2. YAHOO FINANCE DATA (Energy & Commodities)
    # ============================================
    print("\n[2/4] Downloading Yahoo Finance data...")
    
    yahoo_tickers = {
        'brent_crude': 'BZ=F',
        'wti_crude': 'CL=F',
        'nat_gas_us': 'NG=F',
        'coal_etf': 'KOL',
    }
    
    for name, ticker in yahoo_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is not None and len(df) > 0:
                # Handle multi-index columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    close_col = ('Close', ticker)
                    if close_col in df.columns:
                        df_temp = pd.DataFrame({name: df[close_col]})
                    else:
                        df_temp = pd.DataFrame({name: df['Close'].iloc[:, 0]})
                else:
                    df_temp = pd.DataFrame({name: df['Close']})
                dataframes.append(df_temp)
                print(f"  ✓ {name} ({ticker}): {len(df_temp)} obs")
        except Exception as e:
            print(f"  ✗ {name} ({ticker}): {e}")
    
    # ============================================
    # 3. COMBINE INTO DATAFRAME
    # ============================================
    print("\n[3/4] Combining data...")
    
    if len(dataframes) == 0:
        print("ERROR: No data was downloaded!")
        return None
    
    # Concatenate all dataframes
    df = pd.concat(dataframes, axis=1)
    
    # Resample to monthly (end of month)
    df_monthly = df.resample('ME').last()
    
    print(f"  Raw data shape: {df.shape}")
    print(f"  Monthly data shape: {df_monthly.shape}")
    
    # ============================================
    # 4. SAVE DATA
    # ============================================
    print("\n[4/4] Saving data...")
    
    # Save raw data
    df_monthly.to_csv('data/raw/market_data_monthly.csv')
    print(f"  ✓ Saved to data/raw/market_data_monthly.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    print(f"\nDate range: {df_monthly.index.min()} to {df_monthly.index.max()}")
    print(f"Total months: {len(df_monthly)}")
    print(f"\nColumns ({len(df_monthly.columns)}):")
    for col in df_monthly.columns:
        non_null = df_monthly[col].notna().sum()
        print(f"  - {col}: {non_null} observations")
    
    print("\nMissing values:")
    print(df_monthly.isnull().sum())
    
    return df_monthly

if __name__ == "__main__":
    df = download_all_data()
    print("\n✓ Data download complete!")