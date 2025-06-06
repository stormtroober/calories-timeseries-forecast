import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from scipy import stats
from pykalman import KalmanFilter
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler

def apply_kalman_filter(data):
    """Apply Kalman filter for smoothing time series data"""
    # Simple Kalman filter for univariate time series
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=data.iloc[0],
                      n_dim_state=1)
    
    # Fit the Kalman filter and get smoothed values
    kf_fitted = kf.em(data.values)
    state_means, _ = kf_fitted.smooth(data.values)
    
    return pd.Series(state_means.flatten(), index=data.index)

def inverse_boxcox_transform(data, lambda_param):
    """Inverse Box-Cox transformation to return to original scale"""
    if lambda_param == 0:
        return np.exp(data)
    else:
        return (data * lambda_param + 1) ** (1 / lambda_param)

def preprocessing(
    df, 
    apply_scaling=False, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=False,
    plot_processed_data=False,
    decomposition_period=30,
    train_frac=0.70,
    val_frac=0.15,
    test_frac=0.15
):
    """
    Improved preprocessing function that returns properly split and processed datasets
    """
    print("=== PREPROCESSING PIPELINE ===")
    
    # 1. Clean initial data
    dataset = df['Calories (kcal)'].copy()
    nan_count = dataset.isnull().sum()
    print(f"Number of NaN values in 'Calories (kcal)': {nan_count}")
    dataset = dataset.dropna()
    
    print(f"Original dataset size: {len(dataset)}")
    
    # 2. Initial split (on original data)
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_original = dataset.iloc[:n_train].copy()
    val_original = dataset.iloc[n_train:n_train + n_val].copy()
    test_original = dataset.iloc[n_train + n_val:].copy()

    print(f"Train size: {len(train_original)} samples")
    print(f"Validation size: {len(val_original)} samples")
    print(f"Test size: {len(test_original)} samples")

    # 3. Statistical tests on training data
    print("\n=== STATISTICAL TESTS (Training Set) ===")
    result = adfuller(train_original)
    adf_stat, p_value = result[0], result[1]
    print(f"ADF Statistic: {adf_stat}")
    print(f"p-value: {p_value}")
    if p_value < 0.05:
        print("✓ Time series appears stationary (p < 0.05)")
    else:
        print("✗ Time series appears non-stationary (p >= 0.05)")

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(train_original)
    print(f"Shapiro-Wilk Statistic: {shapiro_stat}")
    print(f"Shapiro-Wilk p-value: {shapiro_p}")
    if shapiro_p > 0.05:
        print("✓ Data appears to be normally distributed (p > 0.05)")
    else:
        print("✗ Data does not appear to be normally distributed (p <= 0.05)")

    # 4. Outlier removal parameters (learned from training set only)
    print("\n=== OUTLIER REMOVAL ===")
    Q1 = train_original.quantile(0.25)
    Q3 = train_original.quantile(0.75)
    IQR = Q3 - Q1
    multiplier = 3  # Less strict outlier removal

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

    # Apply outlier removal only to training set
    train_mask = (train_original >= lower_bound) & (train_original <= upper_bound)
    
    train_clean = train_original[train_mask].copy()
    val_clean = val_original.copy()  # No outlier removal
    test_clean = test_original.copy()  # No outlier removal
    
    print(f"Train outliers removed: {len(train_original) - len(train_clean)}")
    print(f"Val outliers removed: 0 (no filtering applied)")
    print(f"Test outliers removed: 0 (no filtering applied)")

    # 5. Box-Cox transformation (parameters learned from training)
    boxcox_lambda = None
    if transform_method == "box-cox":
        print("\n=== BOX-COX TRANSFORMATION ===")
        from scipy.stats import boxcox
        
        # Learn lambda from training data only
        train_positive = train_clean + 1e-6 if (train_clean <= 0).any() else train_clean
        train_transformed, lmbda = boxcox(train_positive)
        boxcox_lambda = lmbda
        
        # Apply to all sets using the same lambda
        val_positive = val_clean + 1e-6 if (val_clean <= 0).any() else val_clean
        test_positive = test_clean + 1e-6 if (test_clean <= 0).any() else test_clean
        
        train_transformed = pd.Series(train_transformed, index=train_clean.index)
        val_transformed = pd.Series(boxcox(val_positive, lmbda=lmbda), index=val_clean.index)
        test_transformed = pd.Series(boxcox(test_positive, lmbda=lmbda), index=test_clean.index)
        
        print(f"✓ Applied Box-Cox transformation (lambda={lmbda:.4f})")
        
        # Test normality after Box-Cox
        shapiro_stat_bc, shapiro_p_bc = stats.shapiro(train_transformed)
        print(f"Shapiro-Wilk after Box-Cox: {shapiro_stat_bc:.4f} (p={shapiro_p_bc:.2e})")
    else:
        train_transformed = train_clean.copy()
        val_transformed = val_clean.copy()
        test_transformed = test_clean.copy()

    # 6. Kalman filter smoothing
    if apply_kalman:
        print("\n=== KALMAN FILTER SMOOTHING ===")
        train_processed = apply_kalman_filter(train_transformed)
        val_processed = apply_kalman_filter(val_transformed)
        test_processed = apply_kalman_filter(test_transformed)
        print("✓ Applied Kalman filter for data smoothing")
        
        # Test normality after Kalman
        shapiro_stat_k, shapiro_p_k = stats.shapiro(train_processed)
        print(f"Shapiro-Wilk after Kalman: {shapiro_stat_k:.4f} (p={shapiro_p_k:.2e})")
    else:
        train_processed = train_transformed.copy()
        val_processed = val_transformed.copy()
        test_processed = test_transformed.copy()
        print("Kalman filter not applied")

    # 7. Standardization (optional)
    scaler = None
    if apply_scaling:
        print("\n=== STANDARDIZATION ===")
        scaler = StandardScaler()
        
        # Fit scaler on training data only
        train_final = scaler.fit_transform(train_processed.values.reshape(-1, 1)).flatten()
        val_final = scaler.transform(val_processed.values.reshape(-1, 1)).flatten()
        test_final = scaler.transform(test_processed.values.reshape(-1, 1)).flatten()
        
        train_final = pd.Series(train_final, index=train_processed.index)
        val_final = pd.Series(val_final, index=val_processed.index)
        test_final = pd.Series(test_final, index=test_processed.index)
        
        print("✓ Applied StandardScaler")
    else:
        train_final = train_processed.copy()
        val_final = val_processed.copy()
        test_final = test_processed.copy()
        print("No scaling applied")

    # 8. Final statistics
    print(f"\n=== FINAL DATASET SIZES ===")
    print(f"Train processed: {len(train_final)} samples")
    print(f"Validation processed: {len(val_final)} samples")
    print(f"Test processed: {len(test_final)} samples")
    print(f"Box-Cox lambda: {boxcox_lambda}")

    # 9. Visualization
    if plot_processed_data and transform_method == "box-cox":
        # Plot original scale
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_original.index, train_original, label='Train Original', alpha=0.7, color='blue')
        plt.plot(val_original.index, val_original, label='Val Original', alpha=0.7, color='orange')
        plt.plot(test_original.index, test_original, label='Test Original', alpha=0.7, color='green')
        plt.plot(train_clean.index, train_clean, label='Train Clean', linewidth=2, color='darkblue')
        plt.title('Original Scale Data')
        plt.xlabel('Time')
        plt.ylabel('Calories (kcal)')
        plt.legend()
        plt.grid(True)

        # Plot transformed scale
        plt.subplot(1, 2, 2)
        plt.plot(train_final.index, train_final, label='Train Processed', linewidth=2, color='darkblue')
        plt.plot(val_final.index, val_final, label='Val Processed', linewidth=2, color='darkorange')
        plt.plot(test_final.index, test_final, label='Test Processed', linewidth=2, color='darkgreen')
        plt.title('Processed Data (Box-Cox + Kalman)')
        plt.xlabel('Time')
        plt.ylabel('Transformed Calories')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 10. Seasonal decomposition
    if plot_decomposition and len(train_final) >= 2 * decomposition_period:
        try:
            print(f"\n=== SEASONAL DECOMPOSITION (Period={decomposition_period}) ===")
            decompose_result = seasonal_decompose(train_final, model='additive', period=decomposition_period)
            
            plt.figure(figsize=(12, 10))
            plt.subplot(411)
            plt.plot(decompose_result.observed, label='Observed')
            plt.legend(loc='upper left')
            plt.title(f'Seasonal Decomposition - Training Data (Period={decomposition_period})')
            
            plt.subplot(412)
            plt.plot(decompose_result.trend, label='Trend', color='red')
            plt.legend(loc='upper left')
            
            plt.subplot(413)
            plt.plot(decompose_result.seasonal, label='Seasonality', color='green')
            plt.legend(loc='upper left')
            
            plt.subplot(414)
            plt.plot(decompose_result.resid, label='Residuals', color='orange')
            plt.legend(loc='upper left')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Could not perform seasonal decomposition: {e}")

    # Store preprocessing parameters for inverse transformation
    preprocessing_params = {
        'Q1': Q1, 'Q3': Q3, 'multiplier': multiplier,
        'boxcox_lambda': boxcox_lambda,
        'scaler': scaler,
        'apply_kalman': apply_kalman,
        'apply_scaling': apply_scaling,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }

    return {
        'train_processed': train_final,
        'val_processed': val_final,
        'test_processed': test_final,
        'train_original': train_original,
        'val_original': val_original,
        'test_original': test_original,
        'preprocessing_params': preprocessing_params,
        'boxcox_lambda': boxcox_lambda,
        'scaler': scaler
    }

def inverse_transform_predictions(predictions, preprocessing_params):
    """
    Convert predictions back to original scale
    """
    boxcox_lambda = preprocessing_params['boxcox_lambda']
    scaler = preprocessing_params['scaler']
    apply_scaling = preprocessing_params['apply_scaling']
    
    # Step 1: Inverse scaling
    if apply_scaling and scaler is not None:
        predictions_unscaled = scaler.inverse_transform(predictions.values.reshape(-1, 1)).flatten()
        predictions_unscaled = pd.Series(predictions_unscaled, index=predictions.index)
    else:
        predictions_unscaled = predictions.copy()
    
    # Step 2: Inverse Box-Cox
    if boxcox_lambda is not None:
        predictions_original = inverse_boxcox_transform(predictions_unscaled, boxcox_lambda)
        predictions_original = pd.Series(predictions_original, index=predictions.index)
    else:
        predictions_original = predictions_unscaled.copy()
    
    return predictions_original