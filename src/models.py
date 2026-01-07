# Model definitions: Ridge, Decision Tree, Gradient Boosting, LSTM
# Includes hyperparameter tuning and cross-validation utilities

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Optional: LSTM support
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be skipped.")


# Model configurations
MODEL_CONFIGS = {
    'ridge': {
        'model': Ridge,
        'params': {
            'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'lasso': {
        'model': Lasso,
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
        }
    },
    'elasticnet': {
        'model': ElasticNet,
        'params': {
            'alpha': [0.01, 0.1, 1.0],
            'l1_ratio': [0.2, 0.5, 0.8]
        }
    },
    'decision_tree': {
        'model': DecisionTreeRegressor,
        'params': {
            'max_depth': [3, 5, 10, 15],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [2, 5, 10]
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingRegressor,
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_samples_split': [5, 10],
            'subsample': [0.8, 1.0]
        }
    },
    'random_forest': {
        'model': RandomForestRegressor,
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [5, 10]
        }
    }
}


def get_walk_forward_splits(X, y, n_splits=5, test_size=None):
    """
    Generate walk-forward cross-validation splits.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
        Target
    n_splits : int
        Number of splits
    test_size : int, optional
        Size of each test set. If None, calculated automatically.
        
    Returns:
    --------
    list of tuples
        List of (train_indices, test_indices) tuples
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    return list(tscv.split(X))


def train_model(X_train, y_train, model_name='gradient_boosting', 
                tune_hyperparams=True, random_state=42):
    """
    Train a model with optional hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    model_name : str
        Name of model to train
    tune_hyperparams : bool
        Whether to perform grid search
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        (trained_model, best_params)
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    if tune_hyperparams:
        # Grid search with time series CV
        tscv = TimeSeriesSplit(n_splits=3)
        
        if 'random_state' in config['model']().get_params():
            base_model = config['model'](random_state=random_state)
        else:
            base_model = config['model']()
        
        grid_search = GridSearchCV(
            base_model,
            config['params'],
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X_train, y_train)
        
        return grid_search.best_estimator_, grid_search.best_params_
    else:
        # Train with default params
        if 'random_state' in config['model']().get_params():
            model = config['model'](random_state=random_state)
        else:
            model = config['model']()
        model.fit(X_train, y_train)
        return model, {}


def train_lstm(X_train, y_train, X_val=None, y_val=None,
               sequence_length=10, epochs=100, batch_size=32):
    """
    Train an LSTM model for time series prediction.
    
    Parameters:
    -----------
    X_train : np.array
        Training features (will be reshaped for LSTM)
    y_train : np.array
        Training target
    X_val : np.array, optional
        Validation features
    y_val : np.array, optional
        Validation target
    sequence_length : int
        Length of input sequences
    epochs : int
        Maximum epochs
    batch_size : int
        Batch size
        
    Returns:
    --------
    tuple
        (model, scaler, history)
    """
    if not LSTM_AVAILABLE:
        raise ImportError("TensorFlow is required for LSTM models")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_train_scaled, y_train, sequence_length)
    
    if X_val is not None and y_val is not None:
        X_val_scaled = scaler.transform(X_val)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, sequence_length)
        validation_data = (X_val_seq, y_val_seq)
    else:
        validation_data = None
    
    # Build model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, X_train.shape[1])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss' if validation_data else 'loss',
                               patience=10, restore_best_weights=True)
    
    # Train
    history = model.fit(
        X_seq, y_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[early_stop],
        verbose=0
    )
    
    return model, scaler, history


def create_sequences(X, y, sequence_length):
    """
    Create sequences for LSTM input.
    
    Parameters:
    -----------
    X : np.array
        Features
    y : np.array
        Target
    sequence_length : int
        Length of sequences
        
    Returns:
    --------
    tuple
        (X_sequences, y_sequences)
    """
    X_seq, y_seq = [], []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


def walk_forward_validation(X, y, model_name='gradient_boosting', 
                           n_splits=5, tune_hyperparams=False,
                           verbose=True):
    """
    Perform walk-forward cross-validation.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_name : str
        Name of model
    n_splits : int
        Number of CV splits
    tune_hyperparams : bool
        Whether to tune hyperparameters
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict
        Results including predictions, metrics, and models
    """
    splits = get_walk_forward_splits(X, y, n_splits)
    
    results = {
        'train_rmse': [],
        'test_rmse': [],
        'train_mae': [],
        'test_mae': [],
        'train_dir_acc': [],
        'test_dir_acc': [],
        'predictions': [],
        'actuals': [],
        'models': [],
        'indices': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"\nFold {fold + 1}/{n_splits}")
            print(f"  Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model, best_params = train_model(X_train, y_train, model_name, 
                                         tune_hyperparams=tune_hyperparams)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Directional accuracy
        train_dir_acc = np.mean((y_train > 0) == (y_train_pred > 0))
        test_dir_acc = np.mean((y_test > 0) == (y_test_pred > 0))
        
        if verbose:
            print(f"  Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
            print(f"  Train Dir Acc: {train_dir_acc:.1%}, Test Dir Acc: {test_dir_acc:.1%}")
        
        # Store results
        results['train_rmse'].append(train_rmse)
        results['test_rmse'].append(test_rmse)
        results['train_mae'].append(train_mae)
        results['test_mae'].append(test_mae)
        results['train_dir_acc'].append(train_dir_acc)
        results['test_dir_acc'].append(test_dir_acc)
        results['predictions'].extend(y_test_pred)
        results['actuals'].extend(y_test.values)
        results['models'].append(model)
        results['indices'].extend(test_idx)
    
    # Summary statistics
    results['mean_train_rmse'] = np.mean(results['train_rmse'])
    results['mean_test_rmse'] = np.mean(results['test_rmse'])
    results['mean_train_dir_acc'] = np.mean(results['train_dir_acc'])
    results['mean_test_dir_acc'] = np.mean(results['test_dir_acc'])
    results['std_test_rmse'] = np.std(results['test_rmse'])
    results['std_test_dir_acc'] = np.std(results['test_dir_acc'])
    
    if verbose:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Mean Test RMSE: {results['mean_test_rmse']:.4f} ± {results['std_test_rmse']:.4f}")
        print(f"Mean Test Dir Acc: {results['mean_test_dir_acc']:.1%} ± {results['std_test_dir_acc']:.1%}")
    
    return results


def compare_models(X, y, model_names=None, n_splits=5, verbose=True):
    """
    Compare multiple models using walk-forward validation.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    model_names : list, optional
        List of model names. If None, uses all available.
    n_splits : int
        Number of CV splits
    verbose : bool
        Print progress
        
    Returns:
    --------
    pd.DataFrame
        Comparison results
    """
    if model_names is None:
        model_names = list(MODEL_CONFIGS.keys())
    
    comparison = []
    
    for model_name in model_names:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training {model_name.upper()}")
            print('='*60)
        
        results = walk_forward_validation(X, y, model_name, n_splits, 
                                         tune_hyperparams=False, verbose=verbose)
        
        comparison.append({
            'model': model_name,
            'mean_rmse': results['mean_test_rmse'],
            'std_rmse': results['std_test_rmse'],
            'mean_dir_acc': results['mean_test_dir_acc'],
            'std_dir_acc': results['std_test_dir_acc'],
            'overfitting_ratio': results['mean_test_rmse'] / results['mean_train_rmse']
        })
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df = comparison_df.sort_values('mean_dir_acc', ascending=False)
    
    if verbose:
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        print(comparison_df.to_string(index=False))
    
    return comparison_df


def save_model(model, filepath, scaler=None):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model : sklearn model or keras model
        Trained model
    filepath : str
        Path to save the model
    scaler : StandardScaler, optional
        Scaler used for preprocessing
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if hasattr(model, 'save'):  # Keras model
        model.save(filepath)
        if scaler:
            joblib.dump(scaler, filepath + '_scaler.pkl')
    else:  # Sklearn model
        joblib.dump(model, filepath)
        if scaler:
            joblib.dump(scaler, filepath.replace('.pkl', '_scaler.pkl'))


def load_model(filepath):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the model file
        
    Returns:
    --------
    model
        Loaded model
    """
    if filepath.endswith('.h5') or os.path.isdir(filepath):
        if LSTM_AVAILABLE:
            return tf.keras.models.load_model(filepath)
        else:
            raise ImportError("TensorFlow required to load LSTM model")
    else:
        return joblib.load(filepath)


if __name__ == "__main__":
    # Example usage
    from features import build_feature_set, get_feature_target_split
    import data_loader
    
    # Load and prepare data
    returns = data_loader.load_data('../data/processed/returns.csv')
    features_df = build_feature_set(returns, verbose=False)
    X, y = get_feature_target_split(features_df)
    
    # Compare models
    comparison = compare_models(X, y, 
                               model_names=['ridge', 'decision_tree', 'gradient_boosting'],
                               n_splits=5)
