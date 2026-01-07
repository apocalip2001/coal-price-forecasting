# Evaluation metrics and visualization utilities
# RMSE, MAE, directional accuracy, and plotting functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'directional_accuracy': np.mean((y_true > 0) == (y_pred > 0)),
        'mean_error': np.mean(y_pred - y_true),
        'std_error': np.std(y_pred - y_true)
    }
    
    return metrics


def calculate_trading_metrics(y_true, y_pred, risk_free_rate=0.02):
    """
    Calculate trading-specific metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual returns
    y_pred : array-like
        Predicted returns
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation
        
    Returns:
    --------
    dict
        Trading metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Strategy: long when predicted positive, cash otherwise
    signals = (y_pred > 0).astype(int)
    strategy_returns = signals * y_true
    
    # Annualization factor (assuming daily returns)
    annual_factor = 252
    
    # Sharpe Ratio
    excess_returns = strategy_returns - risk_free_rate / annual_factor
    sharpe = np.sqrt(annual_factor) * np.mean(excess_returns) / np.std(strategy_returns)
    
    # Sortino Ratio (downside deviation)
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
    sortino = np.sqrt(annual_factor) * np.mean(excess_returns) / downside_std
    
    # Maximum Drawdown
    cumulative = (1 + strategy_returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Win Rate
    wins = np.sum((signals == 1) & (y_true > 0))
    total_trades = np.sum(signals == 1)
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    # Total Return
    total_return = cumulative[-1] - 1
    
    metrics = {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'total_return': total_return,
        'num_trades': total_trades,
        'avg_daily_return': np.mean(strategy_returns),
        'volatility': np.std(strategy_returns) * np.sqrt(annual_factor)
    }
    
    return metrics


def plot_predictions(y_true, y_pred, title='Predictions vs Actual', figsize=(12, 5)):
    """
    Plot predicted vs actual values.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Time series plot
    axes[0].plot(y_true, label='Actual', alpha=0.7)
    axes[0].plot(y_pred, label='Predicted', alpha=0.7)
    axes[0].set_xlabel('Observation')
    axes[0].set_ylabel('Return')
    axes[0].set_title(f'{title} - Time Series')
    axes[0].legend()
    
    # Scatter plot
    axes[1].scatter(y_true, y_pred, alpha=0.5)
    axes[1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 'r--', label='Perfect prediction')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'{title} - Scatter')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, figsize=(10, 8)):
    """
    Plot feature importance from a tree-based model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to show
    figsize : tuple
        Figure size
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True).tail(top_n)
    
    plt.figure(figsize=figsize)
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Features - Gradient Boosting Model')
    plt.tight_layout()
    plt.show()
    
    return importance_df


def plot_permutation_importance(model, X, y, feature_names, top_n=20, 
                                n_repeats=10, figsize=(14, 6)):
    """
    Plot permutation importance.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X : pd.DataFrame or np.array
        Features
    y : pd.Series or np.array
        Target
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to show
    n_repeats : int
        Number of permutation repeats
    figsize : tuple
        Figure size
    """
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X, y, n_repeats=n_repeats, 
                                             random_state=42, scoring='r2')
    
    # Create DataFrame
    perm_df = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Permutation importance with error bars
    axes[0].barh(range(len(perm_df)), perm_df['importance'], 
                 xerr=perm_df['std'], capsize=3)
    axes[0].set_yticks(range(len(perm_df)))
    axes[0].set_yticklabels(perm_df['feature'])
    axes[0].set_xlabel('Mean Importance (Decrease in R²)')
    axes[0].set_title(f'Top {top_n} Features - Permutation Importance')
    axes[0].invert_yaxis()
    
    # Comparison with built-in importance if available
    if hasattr(model, 'feature_importances_'):
        builtin_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        # Merge and compare top features
        compare_df = perm_df.merge(builtin_df, on='feature', suffixes=('_perm', '_builtin'))
        
        x = np.arange(len(compare_df))
        width = 0.35
        
        axes[1].barh(x - width/2, compare_df['importance_builtin'], width, 
                     label='Built-in (Gini)')
        axes[1].barh(x + width/2, compare_df['importance_perm'] * 10, width, 
                     label='Permutation (×10)')
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(compare_df['feature'])
        axes[1].set_xlabel('Importance')
        axes[1].set_title('Built-in vs Permutation Importance')
        axes[1].legend()
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    return perm_df


def plot_equity_curve(y_true, y_pred, initial_capital=10000, figsize=(14, 8)):
    """
    Plot trading strategy equity curves.
    
    Parameters:
    -----------
    y_true : array-like
        Actual returns
    y_pred : array-like
        Predicted returns
    initial_capital : float
        Starting capital
    figsize : tuple
        Figure size
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Strategies
    # Buy and hold
    buyhold_returns = y_true
    buyhold_equity = initial_capital * (1 + buyhold_returns).cumprod()
    
    # Long only (long when predicted positive)
    long_signals = (y_pred > 0).astype(int)
    long_returns = long_signals * y_true
    long_equity = initial_capital * (1 + long_returns).cumprod()
    
    # Long-short
    longshort_signals = np.where(y_pred > 0, 1, -1)
    longshort_returns = longshort_signals * y_true
    longshort_equity = initial_capital * (1 + longshort_returns).cumprod()
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Equity curves
    axes[0, 0].plot(buyhold_equity, label='Buy Hold', linewidth=2)
    axes[0, 0].plot(long_equity, label='Long Only', linewidth=2)
    axes[0, 0].plot(longshort_equity, label='Long Short', linewidth=2)
    axes[0, 0].axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].set_title(f'Equity Curves (Starting Capital: ${initial_capital:,})')
    axes[0, 0].legend()
    
    # Strategy comparison bar chart
    metrics_list = []
    for name, returns in [('Buy Hold', buyhold_returns), 
                          ('Long Only', long_returns), 
                          ('Long Short', longshort_returns)]:
        metrics = calculate_trading_metrics(y_true if name == 'Buy Hold' else returns, 
                                           y_pred)
        metrics['strategy'] = name
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    x = np.arange(3)
    width = 0.25
    
    axes[0, 1].bar(x - width, [metrics_df.loc[i, 'total_return']*100 for i in range(3)], 
                   width, label='Total Return')
    axes[0, 1].bar(x, [metrics_df.loc[i, 'sharpe_ratio'] for i in range(3)], 
                   width, label='Sharpe Ratio')
    axes[0, 1].bar(x + width, [metrics_df.loc[i, 'win_rate']*100 for i in range(3)], 
                   width, label='Win Rate')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(['Buy Hold', 'Long Only', 'Long Short'])
    axes[0, 1].set_title('Strategy Metrics Comparison')
    axes[0, 1].legend()
    
    # Monthly returns for Long Only
    monthly_returns = pd.Series(long_returns).groupby(np.arange(len(long_returns)) // 21).sum()
    colors = ['green' if r > 0 else 'red' for r in monthly_returns]
    axes[1, 0].bar(range(len(monthly_returns)), monthly_returns * 100, color=colors)
    axes[1, 0].set_xlabel('Month')
    axes[1, 0].set_ylabel('Return (%)')
    axes[1, 0].set_title('Monthly Returns - Long Only Strategy (%)')
    
    # Position distribution
    position_counts = pd.Series(long_signals).value_counts()
    axes[1, 1].bar(['Cash (0)', 'Long (1)'], 
                   [position_counts.get(0, 0), position_counts.get(1, 0)],
                   color=['gray', 'green'])
    axes[1, 1].set_ylabel('Number of Days')
    axes[1, 1].set_title('Position Distribution - Long Only Strategy')
    
    plt.tight_layout()
    plt.show()
    
    return metrics_df


def plot_overfitting_analysis(results_dict, figsize=(12, 5)):
    """
    Plot train vs test performance to analyze overfitting.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with model names as keys, each containing 
        'train_rmse' and 'test_rmse' lists
    figsize : tuple
        Figure size
    """
    models = list(results_dict.keys())
    n_models = len(models)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(n_models)
    width = 0.25
    
    train_rmse = [np.mean(results_dict[m]['train_rmse']) for m in models]
    val_rmse = [np.mean(results_dict[m].get('val_rmse', results_dict[m]['test_rmse'])) for m in models]
    test_rmse = [np.mean(results_dict[m]['test_rmse']) for m in models]
    
    ax.bar(x - width, train_rmse, width, label='Train', color='steelblue')
    ax.bar(x, val_rmse, width, label='Validation', color='darkorange')
    ax.bar(x + width, test_rmse, width, label='Test', color='green')
    
    # Add ratio annotations
    for i, m in enumerate(models):
        ratio = test_rmse[i] / train_rmse[i]
        ax.annotate(f'Ratio: {ratio:.2f}', xy=(i, max(train_rmse[i], test_rmse[i])), 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('RMSE')
    ax.set_title('Overfitting Analysis: Train vs Validation vs Test RMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_walk_forward_results(results, model_name='Model', figsize=(14, 5)):
    """
    Plot walk-forward cross-validation results.
    
    Parameters:
    -----------
    results : dict
        Results from walk_forward_validation
    model_name : str
        Name for plot title
    figsize : tuple
        Figure size
    """
    n_folds = len(results['test_rmse'])
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # RMSE by fold
    axes[0].plot(range(1, n_folds + 1), results['train_rmse'], 'o-', label='Train')
    axes[0].plot(range(1, n_folds + 1), results['test_rmse'], 'o-', label='Test')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title(f'Walk-Forward CV: RMSE by Fold')
    axes[0].legend()
    axes[0].set_xticks(range(1, n_folds + 1))
    
    # Directional accuracy by fold
    axes[1].plot(range(1, n_folds + 1), [acc * 100 for acc in results['train_dir_acc']], 
                 'o-', label='Train')
    axes[1].plot(range(1, n_folds + 1), [acc * 100 for acc in results['test_dir_acc']], 
                 'o-', label='Test')
    axes[1].axhline(y=50, color='red', linestyle='--', label='Random (50%)', alpha=0.7)
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Directional Accuracy (%)')
    axes[1].set_title(f'Walk-Forward CV: Directional Accuracy by Fold')
    axes[1].legend()
    axes[1].set_xticks(range(1, n_folds + 1))
    
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plot correlation heatmap.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with numeric columns
    figsize : tuple
        Figure size
    """
    # Select only return columns for cleaner visualization
    ret_cols = [col for col in df.columns if col.endswith('_ret')]
    if len(ret_cols) == 0:
        ret_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    corr_matrix = df[ret_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Matrix: Daily Returns')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def print_model_summary(metrics, model_name='Model'):
    """
    Print formatted model summary.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metrics
    model_name : str
        Name of the model
    """
    print("\n" + "=" * 60)
    print(f"  {model_name.upper()} - PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if 'rmse' in metrics:
        print(f"\n📊 Regression Metrics:")
        print(f"   RMSE:  {metrics['rmse']:.4f}")
        print(f"   MAE:   {metrics['mae']:.4f}")
        print(f"   R²:    {metrics['r2']:.4f}")
    
    if 'directional_accuracy' in metrics:
        print(f"\n🎯 Classification Metrics:")
        print(f"   Directional Accuracy: {metrics['directional_accuracy']:.1%}")
    
    if 'sharpe_ratio' in metrics:
        print(f"\n📈 Trading Metrics:")
        print(f"   Sharpe Ratio:    {metrics['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio:   {metrics['sortino_ratio']:.2f}")
        print(f"   Max Drawdown:    {metrics['max_drawdown']:.1%}")
        print(f"   Win Rate:        {metrics['win_rate']:.1%}")
        print(f"   Total Return:    {metrics['total_return']:.1%}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample data
    n = 500
    y_true = np.random.randn(n) * 0.02
    y_pred = y_true + np.random.randn(n) * 0.01
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    trading_metrics = calculate_trading_metrics(y_true, y_pred)
    
    # Print summary
    all_metrics = {**metrics, **trading_metrics}
    print_model_summary(all_metrics, 'Example Model')
    
    # Plot
    plot_equity_curve(y_true, y_pred)
