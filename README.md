# Thermal Coal Price Forecasting with Machine Learning

## Project Overview
Machine learning-based forecasting of thermal coal prices using market fundamentals. This project compares four ML approaches: Ridge Regression, Decision Tree, Gradient Boosting, and LSTM neural networks.

## Key Results
| Model | Test RMSE | Directional Accuracy |
|-------|-----------|---------------------|
| Decision Tree | 0.0217 | 52.4% |
| Gradient Boosting | 0.0219 | **54.7%** |
| LSTM | 0.0232 | 51.5% |
| Ridge | 0.0236 | 51.6% |

**Best model:** Gradient Boosting with 54.7% directional accuracy (better than random 50%)

## Data
- **Period:** 2015-2024 (2,541 daily observations)
- **Target:** China Yanzhou Coal Mining (thermal coal proxy)
- **Features:** 151 engineered features from 17 market variables

### Feature Categories
- Energy prices (Brent, WTI, Natural Gas)
- Industrial indicators (Steel, Materials sectors)
- Currency (USD, EUR, AUD, CNY)
- China economic proxies

## Top Predictive Features
1. Steel sector returns (10.1%)
2. Heating oil returns (5.4%)
3. Materials sector returns (4.8%)
4. Coal 63-day momentum (4.0%)
5. Natural gas lagged returns (3.1%)

## Project Structure
