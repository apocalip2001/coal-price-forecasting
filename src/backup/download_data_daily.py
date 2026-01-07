"""
Download DAILY data for thermal coal price forecasting project.
Target: ~2500+ observations (2015-2024) for robust ML modeling.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

def download_daily_data(start_date="2015-01-01", end_date="2024-12-31"):
    """
    Download daily data for thermal coal price forecasting.
    Uses 2015-2024 for better data availability across all series.
    """
    
    print("="*70)
    print("THERMAL COAL PRICE FORECASTING - DAILY DATA DOWNLOAD")
    print("="*70)
    print(f"Period: {start_date} to {end_date}")
    
    dataframes = []
    
    # ================================================================
    # 1. COAL PRICE PROXIES (Target Variable)
    # ================================================================
    print("\n[1/5] Downloading Coal Price Proxies...")
    
    coal_tickers = {
        'coal_etf_kol': 'KOL',           # VanEck Coal ETF
        'coal_china_yzcm': '1898.HK',    # China Yanzhou Coal Mining
        'coal_arch': 'ARCH',              # Arch Resources (US coal)
        'coal_peabody': 'BTU',            # Peabody Energy (US coal)
    }
    
    for name, ticker in coal_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = df['Close'].iloc[:, 0]
                else:
                    close_data = df['Close']
                df_temp = pd.DataFrame({name: close_data})
                df_temp.index = pd.to_datetime(df_temp.index)
                dataframes.append(df_temp)
                print(f"  ✓ {name}: {len(df_temp)} daily obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # ================================================================
    # 2. ENERGY COMPLEX (Oil & Gas)
    # ================================================================
    print("\n[2/5] Downloading Energy Prices...")
    
    energy_tickers = {
        'brent_crude': 'BZ=F',            # Brent Crude Futures
        'wti_crude': 'CL=F',              # WTI Crude Futures
        'natural_gas_hh': 'NG=F',         # Henry Hub Natural Gas
        'gasoline': 'RB=F',               # RBOB Gasoline
        'heating_oil': 'HO=F',            # Heating Oil
    }
    
    for name, ticker in energy_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = df['Close'].iloc[:, 0]
                else:
                    close_data = df['Close']
                df_temp = pd.DataFrame({name: close_data})
                df_temp.index = pd.to_datetime(df_temp.index)
                dataframes.append(df_temp)
                print(f"  ✓ {name}: {len(df_temp)} daily obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # ================================================================
    # 3. CARBON & POWER MARKETS
    # ================================================================
    print("\n[3/5] Downloading Carbon & Power Proxies...")
    
    carbon_tickers = {
        'carbon_etf_krbn': 'KRBN',        # KraneShares Carbon ETF (EU ETS proxy)
        'utilities_xlu': 'XLU',           # Utilities Sector ETF
        'clean_energy': 'ICLN',           # iShares Clean Energy ETF
    }
    
    for name, ticker in carbon_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = df['Close'].iloc[:, 0]
                else:
                    close_data = df['Close']
                df_temp = pd.DataFrame({name: close_data})
                df_temp.index = pd.to_datetime(df_temp.index)
                dataframes.append(df_temp)
                print(f"  ✓ {name}: {len(df_temp)} daily obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # ================================================================
    # 4. FX & MACRO PROXIES
    # ================================================================
    print("\n[4/5] Downloading FX & Macro Proxies...")
    
    fx_tickers = {
        'usd_index': 'DX-Y.NYB',          # US Dollar Index
        'cny_usd': 'CNY=X',               # Chinese Yuan
        'eur_usd': 'EURUSD=X',            # Euro
        'aud_usd': 'AUDUSD=X',            # Australian Dollar (commodity currency)
    }
    
    for name, ticker in fx_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = df['Close'].iloc[:, 0]
                else:
                    close_data = df['Close']
                df_temp = pd.DataFrame({name: close_data})
                df_temp.index = pd.to_datetime(df_temp.index)
                dataframes.append(df_temp)
                print(f"  ✓ {name}: {len(df_temp)} daily obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # ================================================================
    # 5. FREIGHT & INDUSTRIAL DEMAND PROXIES
    # ================================================================
    print("\n[5/5] Downloading Freight & Industrial Proxies...")
    
    industrial_tickers = {
        'bdry_shipping': 'BDRY',          # Dry Bulk Shipping ETF
        'china_etf_fxi': 'FXI',           # iShares China Large-Cap ETF
        'emerging_mkts': 'EEM',           # Emerging Markets ETF
        'industrials_xli': 'XLI',         # Industrial Select Sector
        'materials_xlb': 'XLB',           # Materials Select Sector
        'steel_slx': 'SLX',               # VanEck Steel ETF
    }
    
    for name, ticker in industrial_tickers.items():
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is not None and len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    close_data = df['Close'].iloc[:, 0]
                else:
                    close_data = df['Close']
                df_temp = pd.DataFrame({name: close_data})
                df_temp.index = pd.to_datetime(df_temp.index)
                dataframes.append(df_temp)
                print(f"  ✓ {name}: {len(df_temp)} daily obs")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    
    # ================================================================
    # COMBINE ALL DATA
    # ================================================================
    print("\n" + "="*70)
    print("COMBINING DATA")
    print("="*70)
    
    if len(dataframes) == 0:
        print("ERROR: No data downloaded!")
        return None
    
    # Concatenate all dataframes
    df_combined = pd.concat(dataframes, axis=1)
    df_combined = df_combined.sort_index()
    
    # Remove weekends (keep only business days)
    df_combined = df_combined[df_combined.index.dayofweek < 5]
    
    print(f"\nCombined shape: {df_combined.shape}")
    print(f"Date range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")
    
    # ================================================================
    # DATA QUALITY CHECK
    # ================================================================
    print("\n" + "="*70)
    print("DATA QUALITY SUMMARY")
    print("="*70)
    
    print(f"\n{'Column':<25} {'Obs':>8} {'Missing':>8} {'Missing%':>10} {'Start':>12}")
    print("-"*70)
    
    for col in df_combined.columns:
        obs = df_combined[col].notna().sum()
        missing = df_combined[col].isna().sum()
        missing_pct = (missing / len(df_combined)) * 100
        first_valid = df_combined[col].first_valid_index()
        start_date_col = first_valid.strftime('%Y-%m-%d') if first_valid else 'N/A'
        print(f"{col:<25} {obs:>8} {missing:>8} {missing_pct:>9.1f}% {start_date_col:>12}")
    
    # ================================================================
    # SAVE DATA
    # ================================================================
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    # Save daily data
    df_combined.to_csv('data/raw/daily_market_data.csv')
    print(f"✓ Saved daily data to data/raw/daily_market_data.csv")
    
    # Also create weekly aggregation
    df_weekly = df_combined.resample('W-FRI').last()
    df_weekly.to_csv('data/raw/weekly_market_data.csv')
    print(f"✓ Saved weekly data to data/raw/weekly_market_data.csv")
    
    # Also create monthly aggregation
    df_monthly = df_combined.resample('ME').last()
    df_monthly.to_csv('data/raw/monthly_market_data.csv')
    print(f"✓ Saved monthly data to data/raw/monthly_market_data.csv")
    
    print("\n" + "="*70)
    print(f"DOWNLOAD COMPLETE: {len(df_combined)} daily observations")
    print(f"                   {len(df_combined.columns)} variables")
    print("="*70)
    
    return df_combined

if __name__ == "__main__":
    df = download_daily_data()
