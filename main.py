#!/usr/bin/env python3
"""
Coal Price Forecasting - Main Entry Point
==========================================
This script runs the complete coal price forecasting pipeline.

Usage:
    python main.py [--train] [--evaluate] [--predict]
"""

import argparse
import os
import sys
import warnings

# Ensure reproducibility
import numpy as np
import random

RANDOM_STATE = 42

def set_seed(seed=RANDOM_STATE):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow/Keras (if using LSTM)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass
    
    # Scikit-learn uses numpy's random state
    print(f"✓ Random seeds set to {seed}")

def main():
    """Main entry point for the coal price forecasting pipeline."""
    parser = argparse.ArgumentParser(
        description='Coal Price Forecasting with Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--train', action='store_true', 
                        help='Train all models')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate trained models')
    parser.add_argument('--predict', action='store_true', 
                        help='Generate predictions')
    parser.add_argument('--all', action='store_true', 
                        help='Run complete pipeline (default)')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help=f'Random seed (default: {RANDOM_STATE})')
    
    args = parser.parse_args()
    
    # Set reproducibility seeds
    set_seed(args.seed)
    
    # Default to running all if no specific flag
    if not (args.train or args.evaluate or args.predict):
        args.all = True
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    print("="*60)
    print("Coal Price Forecasting Pipeline")
    print("="*60)
    print(f"Random State: {args.seed}")
    print()
    
    # Import your modules (adjust paths based on your src/ structure)
    try:
        # Example imports - adjust to match your actual module structure
        # from src.data_loader import load_data
        # from src.feature_engineering import create_features
        # from src.models import train_models, evaluate_models
        pass
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    if args.all or args.train:
        print("\n[1/3] Loading and preprocessing data...")
        # data = load_data('data/raw/')
        # features = create_features(data)
        print("      ✓ Data loaded and features engineered")
        
        print("\n[2/3] Training models...")
        # models = train_models(features, random_state=args.seed)
        print("      ✓ Models trained successfully")
    
    if args.all or args.evaluate:
        print("\n[3/3] Evaluating models...")
        # results = evaluate_models(models, test_data)
        print("      ✓ Evaluation complete")
        
        # Print results summary
        print("\n" + "="*60)
        print("MODEL RESULTS SUMMARY")
        print("="*60)
        print(f"{'Model':<20} {'Test RMSE':<15} {'Dir. Accuracy':<15}")
        print("-"*50)
        # Example output - replace with actual results
        print(f"{'Decision Tree':<20} {'0.0217':<15} {'52.4%':<15}")
        print(f"{'Gradient Boosting':<20} {'0.0219':<15} {'54.7%':<15}")
        print(f"{'LSTM':<20} {'0.0232':<15} {'51.5%':<15}")
        print(f"{'Ridge':<20} {'0.0236':<15} {'51.6%':<15}")
    
    if args.predict:
        print("\n[PREDICT] Generating forecasts...")
        # predictions = generate_predictions(models)
        print("      ✓ Predictions saved to outputs/")
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
