import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import datetime


def calculate_metrics(true, pred):
    """Calculate RMSE, MAE, and MAPE metrics"""
    rmse = np.sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    mape = mean_absolute_percentage_error(true, pred) * 100
    return rmse, mae, mape


def create_comprehensive_results_report(model_results_sarima, val_preds_sarima, test_preds_sarima,
                                       model_mlp, val_preds_mlp, test_preds_mlp,
                                       model_xgb, val_preds_xgb, test_preds_xgb,
                                       xgb_results_df, best_xgb_config, data_dict):
    """
    Create a comprehensive Markdown report with all model results
    """
    
    # Get original data for evaluation
    val_orig = data_dict['val_original'][data_dict['val_processed'].index]
    test_orig = data_dict['test_original'][data_dict['test_processed'].index]
    
    # Calculate metrics for each model
    models_metrics = {}
    
    # SARIMA
    if val_preds_sarima is not None and test_preds_sarima is not None:
        val_rmse, val_mae, val_mape = calculate_metrics(val_orig, val_preds_sarima)
        test_rmse, test_mae, test_mape = calculate_metrics(test_orig, test_preds_sarima)
        models_metrics['SARIMA'] = {
            'val': {'rmse': val_rmse, 'mae': val_mae, 'mape': val_mape},
            'test': {'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape}
        }
    
    # MLP
    if val_preds_mlp is not None and test_preds_mlp is not None:
        val_rmse, val_mae, val_mape = calculate_metrics(val_orig, val_preds_mlp)
        test_rmse, test_mae, test_mape = calculate_metrics(test_orig, test_preds_mlp)
        models_metrics['MLP'] = {
            'val': {'rmse': val_rmse, 'mae': val_mae, 'mape': val_mape},
            'test': {'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape}
        }
    
    # XGBoost
    if val_preds_xgb is not None and test_preds_xgb is not None:
        val_rmse, val_mae, val_mape = calculate_metrics(val_orig, val_preds_xgb)
        test_rmse, test_mae, test_mape = calculate_metrics(test_orig, test_preds_xgb)
        models_metrics['XGBoost'] = {
            'val': {'rmse': val_rmse, 'mae': val_mae, 'mape': val_mape},
            'test': {'rmse': test_rmse, 'mae': test_mae, 'mape': test_mape}
        }
    
    # Create Markdown content
    md_content = f"""# Time Series Forecasting Results Report

**Generated on:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview
This report presents the results of a comprehensive time series forecasting analysis on daily activity metrics, comparing three different modeling approaches: SARIMA, Multi-Layer Perceptron (MLP), and XGBoost.

## Dataset Information
- **Source File:** Daily_activity_metrics.csv
- **Target Variable:** Daily calorie expenditure (kcal)
- **Training samples:** {len(data_dict['train_processed'])}
- **Validation samples:** {len(data_dict['val_processed'])}
- **Test samples:** {len(data_dict['test_processed'])}
- **Total observations:** {len(data_dict['train_processed']) + len(data_dict['val_processed']) + len(data_dict['test_processed'])}

## Data Preprocessing Pipeline
- **Transformation Method:** Box-Cox transformation for variance stabilization
- **Filtering:** Kalman filter applied for noise reduction
- **Scaling:** Applied only for MLP model (Standard scaler)
- **Seasonal Pattern:** 30-day periodicity identified and modeled

## Model Configurations

### 1. SARIMA (Seasonal AutoRegressive Integrated Moving Average)
**Model Type:** Statistical time series model
- **ARIMA Order:** (2, 0, 2)
- **Seasonal Order:** (0, 0, 0, 30)
- **Seasonal Period:** 30 days
- **Parameter Selection:** Auto-optimized using AIC criterion
"""

    if 'SARIMA' in models_metrics:
        md_content += f"""
**Performance Metrics:**
| Metric | Validation | Test |
|--------|------------|------|
| RMSE (kcal) | {models_metrics['SARIMA']['val']['rmse']:.4f} | {models_metrics['SARIMA']['test']['rmse']:.4f} |
| MAE (kcal) | {models_metrics['SARIMA']['val']['mae']:.4f} | {models_metrics['SARIMA']['test']['mae']:.4f} |
| MAPE (%) | {models_metrics['SARIMA']['val']['mape']:.2f} | {models_metrics['SARIMA']['test']['mape']:.2f} |
"""
    
    md_content += f"""
### 2. MLP (Multi-Layer Perceptron)
**Model Type:** Neural network regression model
- **Architecture:** Input Layer → Dense(8 units, ReLU) → Dense(1 unit, Linear)
- **Look-back Window:** 30 days
- **Training Epochs:** 100
- **Batch Size:** 16
- **Optimizer:** Adam with default parameters
- **Loss Function:** Mean Squared Error (MSE)
"""

    if 'MLP' in models_metrics:
        md_content += f"""
**Performance Metrics:**
| Metric | Validation | Test |
|--------|------------|------|
| RMSE (kcal) | {models_metrics['MLP']['val']['rmse']:.4f} | {models_metrics['MLP']['test']['rmse']:.4f} |
| MAE (kcal) | {models_metrics['MLP']['val']['mae']:.4f} | {models_metrics['MLP']['test']['mae']:.4f} |
| MAPE (%) | {models_metrics['MLP']['val']['mape']:.2f} | {models_metrics['MLP']['test']['mape']:.2f} |
"""

    md_content += f"""
### 3. XGBoost (Extreme Gradient Boosting)
**Model Type:** Ensemble tree-based regression model
- **Optimal Look-back Window:** {best_xgb_config['look_back']} days
- **Optimal N_estimators:** {best_xgb_config['n_estimators']} trees
- **Hyperparameter Selection:** Grid search with {len(xgb_results_df)} configurations tested
- **Search Space:** Look-back ∈ [7, 14, 21, 30, 45, 60], N_estimators ∈ [10, 15, 25, 35, 40, 50, 75, 100]
"""

    if 'XGBoost' in models_metrics:
        md_content += f"""
**Performance Metrics:**
| Metric | Validation | Test |
|--------|------------|------|
| RMSE (kcal) | {models_metrics['XGBoost']['val']['rmse']:.4f} | {models_metrics['XGBoost']['test']['rmse']:.4f} |
| MAE (kcal) | {models_metrics['XGBoost']['val']['mae']:.4f} | {models_metrics['XGBoost']['test']['mae']:.4f} |
| MAPE (%) | {models_metrics['XGBoost']['val']['mape']:.2f} | {models_metrics['XGBoost']['test']['mape']:.2f} |
"""

    # Model comparison table
    md_content += """
## Comparative Analysis

### Validation Set Performance
| Model | RMSE (kcal) | MAE (kcal) | MAPE (%) | Rank by RMSE |
|-------|-------------|------------|----------|--------------|
"""
    
    # Sort models by validation RMSE for ranking
    val_sorted = sorted(models_metrics.items(), key=lambda x: x[1]['val']['rmse'])
    for rank, (model_name, metrics) in enumerate(val_sorted, 1):
        val_metrics = metrics['val']
        md_content += f"| {model_name} | {val_metrics['rmse']:.4f} | {val_metrics['mae']:.4f} | {val_metrics['mape']:.2f} | {rank} |\n"
    
    md_content += """
### Test Set Performance (Final Evaluation)
| Model | RMSE (kcal) | MAE (kcal) | MAPE (%) | Rank by RMSE |
|-------|-------------|------------|----------|--------------|
"""
    
    # Sort models by test RMSE for ranking
    test_sorted = sorted(models_metrics.items(), key=lambda x: x[1]['test']['rmse'])
    for rank, (model_name, metrics) in enumerate(test_sorted, 1):
        test_metrics = metrics['test']
        md_content += f"| {model_name} | {test_metrics['rmse']:.4f} | {test_metrics['mae']:.4f} | {test_metrics['mape']:.2f} | {rank} |\n"
    
    # Find best model
    best_model = min(models_metrics.items(), key=lambda x: x[1]['test']['rmse'])
    
    md_content += f"""
## Key Results

### 🏆 Best Performing Model
**{best_model[0]}** achieved the lowest test set RMSE of **{best_model[1]['test']['rmse']:.4f} kcal**
"""
    
    # Write to file
    with open('model_results_summary.md', 'w') as f:
        f.write(md_content)

    
    return models_metrics, best_model


# Function to load and analyze results
def analyze_project_results():
    """
    Main function to be called from project.py to generate the comprehensive report
    """
    print("🚀 Starting comprehensive results analysis...")
    print("This function should be called with model results from project.py")
    print("Example usage:")
    print("from results_analysis import create_comprehensive_results_report")
    print("create_comprehensive_results_report(model_results, val_predictions, test_predictions, ...)")


if __name__ == "__main__":
    analyze_project_results()