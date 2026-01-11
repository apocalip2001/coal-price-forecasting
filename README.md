# Coal Price Forecasting

Machine learning forecasting of thermal coal prices using Ridge Regression, Decision Tree, Gradient Boosting, and LSTM.

## Results

| Model | Test RMSE | Directional Accuracy |
|-------|-----------|----------------------|
| Decision Tree | 0.0217 | 52.4% |
| Gradient Boosting | 0.0219 | **54.7%** |
| LSTM | 0.0232 | 51.5% |
| Ridge | 0.0236 | 51.6% |

## Setup
```bash
# Clone and install
git clone https://github.com/apocalip2001/coal-price-forecasting.git
cd coal-price-forecasting
pip install -r requirements.txt

# Or with conda
conda env create -f environment.yml
conda activate coal-forecasting
```

## Usage
```bash
python main.py              # Run full pipeline
python main.py --train      # Train only
python main.py --evaluate   # Evaluate only
```

## Data

- **Period:** 2015-2024 (2,541 observations)
- **Features:** 151 engineered from 17 market variables
- **Sources:** Energy prices, industrial indicators, currencies

## Reproducibility

All models use `random_state=42`. Run twice and compare:
```bash
python main.py > run1.log && python main.py > run2.log && diff run1.log run2.log
```

## Structure
```
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── environment.yml      # Conda env
├── data/                # Datasets
├── docs/                # Report
├── notebooks/           # Analysis
└── src/                 # Source code
```

## License

MIT
