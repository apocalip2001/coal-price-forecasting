"""
Evaluation metrics and utilities for coal price forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def regression_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    
    Returns:
        Dictionary with RMSE, MAE, R2
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate classification metrics for direction prediction.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold
    
    Returns:
        Dictionary with accuracy, log_loss
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    
    # Clip probabilities to avoid log(0)
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
    ll = log_loss(y_true, y_pred_proba_clipped)
    
    return {
        'Accuracy': acc,
        'LogLoss': ll
    }

def directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (did we predict up/down correctly?).
    """
    true_direction = (y_true > 0).astype(int)
    pred_direction = (y_pred > 0).astype(int)
    return accuracy_score(true_direction, pred_direction)

def compare_models(results_dict):
    """
    Compare multiple models' results.
    
    Args:
        results_dict: Dictionary of {model_name: metrics_dict}
    
    Returns:
        DataFrame comparing all models
    """
    df = pd.DataFrame(results_dict).T
    df = df.round(4)
    return df

def plot_predictions(y_true, y_pred, title="Predictions vs Actual", save_path=None):
    """
    Plot predicted vs actual values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series plot
    ax1 = axes[0]
    ax1.plot(y_true.values, label='Actual', alpha=0.7)
    ax1.plot(y_pred, label='Predicted', alpha=0.7)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Return')
    ax1.set_title(f'{title} - Time Series')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.set_title(f'{title} - Scatter')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_feature_importance(importance_series, top_n=20, title="Feature Importance", save_path=None):
    """
    Plot feature importance.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = importance_series.head(top_n)
    
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def plot_cv_results(cv_results, title="Cross-Validation Results", save_path=None):
    """
    Plot cross-validation results across models.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(cv_results.keys())
    val_means = [cv_results[m]['val_rmse_mean'] for m in models]
    val_stds = [cv_results[m]['val_rmse_std'] for m in models]
    
    x = np.arange(len(models))
    ax.bar(x, val_means, yerr=val_stds, capsize=5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Validation RMSE')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig

def create_results_summary(model_name, train_metrics, val_metrics, test_metrics=None):
    """
    Create a summary of results for a model.
    """
    summary = {
        'Model': model_name,
        'Train_RMSE': train_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Train_R2': train_metrics['R2'],
        'Val_RMSE': val_metrics['RMSE'],
        'Val_MAE': val_metrics['MAE'],
        'Val_R2': val_metrics['R2'],
    }
    
    if test_metrics:
        summary.update({
            'Test_RMSE': test_metrics['RMSE'],
            'Test_MAE': test_metrics['MAE'],
            'Test_R2': test_metrics['R2'],
        })
    
    return summary

if __name__ == "__main__":
    # Test evaluation functions
    print("Testing evaluation functions...")
    
    np.random.seed(42)
    y_true = pd.Series(np.random.randn(100))
    y_pred = y_true + np.random.randn(100) * 0.5
    
    # Test regression metrics
    metrics = regression_metrics(y_true, y_pred)
    print(f"\nRegression Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test directional accuracy
    dir_acc = directional_accuracy(y_true, y_pred)
    print(f"\nDirectional Accuracy: {dir_acc:.4f}")