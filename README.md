# Thermal Coal Price Forecasting with Machine Learning

## Project Overview

Machine learning-based forecasting of thermal coal prices using market fundamentals. This project compares four ML approaches: Ridge Regression, Decision Tree, Gradient Boosting, and LSTM neural networks.

## Key Results

| Model | Test RMSE | Directional Accuracy |
|-------|-----------|----------------------|
| Decision Tree | 0.0217 | 52.4% |
| Gradient Boosting | 0.0219 | **54.7%** |
| LSTM | 0.0232 | 51.5% |
| Ridge | 0.0236 | 51.6% |

**Best model:** Gradient Boosting with 54.7% directional accuracy (better than random 50%)

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Option 1: Using pip
```bash
# Clone the repository
git clone https://github.com/apocalip2001/coal-price-forecasting.git
cd coal-price-forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using conda
```bash
# Clone the repository
git clone https://github.com/apocalip2001/coal-price-forecasting.git
cd coal-price-forecasting

# Create conda environment
conda env create -f environment.yml
conda activate coal-forecasting
```

---

## Running the Project

### Run Complete Pipeline
```bash
python main.py
```

### Run Specific Stages
```bash
python main.py --train      # Train models only
python main.py --evaluate   # Evaluate models only
python main.py --predict    # Generate predictions
```

### Custom Random Seed
```bash
python main.py --seed 123   # Use custom seed for reproducibility
```

### Jupyter Notebooks

To explore the analysis interactively:
```bash
jupyter notebook notebooks/
```

---

## Data

- **Period:** 2015-2024 (2,541 daily observations)
- **Target:** China Yanzhou Coal Mining (thermal coal proxy)
- **Features:** 151 engineered features from 17 market variables

### Feature Categories

- Energy prices (Brent, WTI, Natural Gas)
- Industrial indicators (Steel, Materials sectors)
- Currency (USD, EUR, AUD, CNY)
- China economic proxies

---

## Top Predictive Features

1. Steel sector returns (10.1%)
2. Heating oil returns (5.4%)
3. Materials sector returns (4.8%)
4. Coal 63-day momentum (4.0%)
5. Natural gas lagged returns (3.1%)

---

## Project Structure
```
coal-price-forecasting/
├── main.py                 # Entry point - run pipeline
├── requirements.txt        # pip dependencies
├── environment.yml         # conda environment
├── README.md
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned data
├── docs/
│   └── report.pdf          # Methodology report (~10 pages)
├── notebooks/
│   └── *.ipynb             # Jupyter analysis notebooks
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data loading utilities
    ├── features.py         # Feature engineering
    └── models.py           # ML model implementations
```

---

## Reproducibility

All random operations use `random_state=42` by default. Results should be identical across runs on the same hardware.

To verify reproducibility:
```bash
python main.py > run1.log
python main.py > run2.log
diff run1.log run2.log      # Should show no differences
```

---

## Report

See report for the full methodology report including:
- Data description and preprocessing
- Feature engineering approach
- Model architectures and hyperparameters
- Results analysis and discussion

---

## License

MIT License

