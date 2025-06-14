# Time Series Forecasting for Daily Activity Metrics 📊

A comprehensive machine learning project that compares three different forecasting approaches (SARIMA, Neural Networks, and XGBoost) for predicting daily calorie expenditure from activity metrics.

## 🎯 Project Overview

This project implements and evaluates multiple time series forecasting models to predict daily calorie expenditure using historical activity data. The analysis includes statistical, neural network, and ensemble methods with comprehensive preprocessing and hyperparameter optimization.

## 🔬 Models Implemented

### 1. **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
- Statistical time series model with seasonal components
- Auto-optimized parameters using AIC criterion
- Handles seasonal patterns with 30-day periodicity

### 2. **MLP** (Multi-Layer Perceptron)
- Neural network with dense layers and ReLU activation
- Configurable look-back window and architecture
- Trained with Adam optimizer and MSE loss

### 3. **XGBoost** (Extreme Gradient Boosting)
- Ensemble tree-based regression model
- Grid search optimization for hyperparameters
- Efficient handling of sequential data patterns

## 🛠️ Features

- **Advanced Preprocessing Pipeline**
  - Box-Cox transformation for variance stabilization
  - Kalman filtering for noise reduction
  - Configurable scaling options
  - Seasonal decomposition analysis

- **Automated Model Selection**
  - Grid search for optimal hyperparameters
  - Cross-validation with proper time series splits
  - Multiple evaluation metrics (RMSE, MAE, MAPE)

- **Comprehensive Evaluation**
  - Performance comparison across models
  - Detailed results reporting
  - Visualization of predictions vs actual values

## 📁 Project Structure

```
op-analytics-steps/
├── README.md                          # This file
├── project.py                         # Main execution script
├── model_results_summary.md          # Detailed results report
├── dataset/
│   └── Daily_activity_metrics.csv    # Input dataset
├── dataset_utils/
│   └── preprocessing.py              # Data preprocessing utilities
├── models/
│   ├── sarima.py                     # SARIMA model implementation
│   ├── mlp_keras.py                  # Neural network model
│   └── xgboost.py                    # XGBoost model implementation
└── results_analysis.py               # Results compilation and reporting
```

## 📊 Results Summary

The project evaluates model performance on both validation and test sets. Here are the key findings:

| Model | Test RMSE (kcal) | Test MAE (kcal) | Test MAPE (%) |
|-------|------------------|-----------------|---------------|
| **XGBoost** | **162.91** | 125.20 | 7.23 |
| SARIMA | 166.28 | 120.87 | 6.78 |
| MLP | 166.87 | 131.34 | 7.69 |

🏆 **Best Performing Model:** XGBoost with optimal configuration of 14-day look-back window and 15 estimators.

For detailed analysis, methodology, and complete results, see: **[Model Results Summary](./model_results_summary.md)**


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
