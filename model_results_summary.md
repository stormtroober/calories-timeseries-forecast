# Time Series Forecasting Results Report

**Generated on:** 2025-06-12 18:06:36

## Project Overview
This report presents the results of a comprehensive time series forecasting analysis on daily activity metrics, comparing three different modeling approaches: SARIMA, Multi-Layer Perceptron (MLP), and XGBoost.

## Dataset Information
- **Source File:** Daily_activity_metrics.csv
- **Target Variable:** Daily calorie expenditure (kcal)
- **Training samples:** 1362
- **Validation samples:** 294
- **Test samples:** 295
- **Total observations:** 1951

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

**Performance Metrics:**
| Metric | Validation | Test |
|--------|------------|------|
| RMSE (kcal) | 133.3726 | 166.2822 |
| MAE (kcal) | 100.2933 | 120.8731 |
| MAPE (%) | 5.68 | 6.78 |

### 2. MLP (Multi-Layer Perceptron)
**Model Type:** Neural network regression model
- **Architecture:** Input Layer → Dense(8 units, ReLU) → Dense(1 unit, Linear)
- **Look-back Window:** 30 days
- **Training Epochs:** 100
- **Batch Size:** 16
- **Optimizer:** Adam with default parameters
- **Loss Function:** Mean Squared Error (MSE)

**Performance Metrics:**
| Metric | Validation | Test |
|--------|------------|------|
| RMSE (kcal) | 137.5815 | 166.8669 |
| MAE (kcal) | 101.8352 | 131.3356 |
| MAPE (%) | 5.71 | 7.69 |

### 3. XGBoost (Extreme Gradient Boosting)
**Model Type:** Ensemble tree-based regression model
- **Optimal Look-back Window:** 14 days
- **Optimal N_estimators:** 15 trees
- **Hyperparameter Selection:** Grid search with 48 configurations tested
- **Search Space:** Look-back ∈ [7, 14, 21, 30, 45, 60], N_estimators ∈ [10, 15, 25, 35, 40, 50, 75, 100]

**Performance Metrics:**
| Metric | Validation | Test |
|--------|------------|------|
| RMSE (kcal) | 132.5582 | 162.9080 |
| MAE (kcal) | 101.7312 | 125.2021 |
| MAPE (%) | 5.80 | 7.23 |

## Comparative Analysis

### Validation Set Performance
| Model | RMSE (kcal) | MAE (kcal) | MAPE (%) | Rank by RMSE |
|-------|-------------|------------|----------|--------------|
| XGBoost | 132.5582 | 101.7312 | 5.80 | 1 |
| SARIMA | 133.3726 | 100.2933 | 5.68 | 2 |
| MLP | 137.5815 | 101.8352 | 5.71 | 3 |

### Test Set Performance (Final Evaluation)
| Model | RMSE (kcal) | MAE (kcal) | MAPE (%) | Rank by RMSE |
|-------|-------------|------------|----------|--------------|
| XGBoost | 162.9080 | 125.2021 | 7.23 | 1 |
| SARIMA | 166.2822 | 120.8731 | 6.78 | 2 |
| MLP | 166.8669 | 131.3356 | 7.69 | 3 |

## Key Results

### 🏆 Best Performing Model
**XGBoost** achieved the lowest test set RMSE of **162.9080 kcal**
