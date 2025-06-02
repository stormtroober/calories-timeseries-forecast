import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from scipy import stats
from pykalman import KalmanFilter
from logger import logger
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

def preprocess_validation_test(val_set, test_set, preprocessing_params):
    """
    Apply the same preprocessing steps to validation and test sets
    using parameters learned from the training set
    """
    # Extract parameters from training preprocessing
    Q1, Q3, multiplier, boxcox_lambda, scaler, apply_kalman, apply_scaling = preprocessing_params
    
    # Apply outlier removal with same bounds as training
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Process validation set
    val_outlier_mask = (val_set >= lower_bound) & (val_set <= upper_bound)
    val_clean = val_set[val_outlier_mask]
    
    # Process test set
    test_outlier_mask = (test_set >= lower_bound) & (test_set <= upper_bound)
    test_clean = test_set[test_outlier_mask]
    
    logger.info(f"Validation outliers removed: {len(val_set) - len(val_clean)}")
    logger.info(f"Test outliers removed: {len(test_set) - len(test_clean)}")
    
    # Apply Box-Cox transformation if used in training
    if boxcox_lambda is not None:
        from scipy.stats import boxcox
        val_positive = val_clean + 1e-6 if (val_clean <= 0).any() else val_clean
        test_positive = test_clean + 1e-6 if (test_clean <= 0).any() else test_clean
        
        val_transformed = boxcox(val_positive, lmbda=boxcox_lambda)
        test_transformed = boxcox(test_positive, lmbda=boxcox_lambda)
        
        val_transformed = pd.Series(val_transformed, index=val_clean.index)
        test_transformed = pd.Series(test_transformed, index=test_clean.index)
    else:
        val_transformed = val_clean
        test_transformed = test_clean
    
    # Apply Kalman filter if used in training
    if apply_kalman:
        val_processed = apply_kalman_filter(val_transformed)
        test_processed = apply_kalman_filter(test_transformed)
    else:
        val_processed = val_transformed
        test_processed = test_transformed
    
    # Apply scaling if used in training
    if apply_scaling and scaler is not None:
        val_scaled = scaler.transform(val_processed.values.reshape(-1, 1)).flatten()
        test_scaled = scaler.transform(test_processed.values.reshape(-1, 1)).flatten()
        
        val_final = pd.Series(val_scaled, index=val_processed.index)
        test_final = pd.Series(test_scaled, index=test_processed.index)
    else:
        val_final = val_processed
        test_final = test_processed
    
    return val_final, test_final

def preprocessing(
    df, 
    apply_scaling=False, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=False,
    decomposition_period=30
):
    dataset = df['Calories (kcal)'].copy()
    nan_count = dataset.isnull().sum()
    logger.info(f"Number of NaN values in 'Calories (kcal)': {nan_count}")

    dataset = dataset.dropna()

    # 2. Split into train / validation / test sets (e.g., 70% / 15% / 15%)
    train_frac = 0.70
    val_frac = 0.15
    test_frac = 0.15

    n = len(dataset)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_set = dataset.iloc[:n_train]
    val_set = dataset.iloc[n_train:n_train + n_val]
    test_set = dataset.iloc[n_train + n_val:]

    logger.info(f"Split data: train={len(train_set)} ({train_frac*100:.0f}%), "
                f"val={len(val_set)} ({val_frac*100:.0f}%), "
                f"test={len(test_set)} ({test_frac*100:.0f}%)")

    # 2. Apply statistical tests on original data
    result = adfuller(train_set)
    adf_stat, p_value = result[0], result[1]
    print(f"ADF Statistic: {adf_stat}")
    print(f"p-value: {p_value}")
    if p_value < 0.05:
        print("Time series appears stationary (p < 0.05)")
    else:
        print("Time series appears non-stationary (p >= 0.05)")

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(train_set)
    print(f"Shapiro-Wilk Statistic: {shapiro_stat}")
    print(f"Shapiro-Wilk p-value: {shapiro_p}")
    if shapiro_p > 0.05:
        print("Data appears to be normally distributed (p > 0.05)")
    else:
        print("Data does not appear to be normally distributed (p <= 0.05)")

    # you can then continue processing on train_set / val_set / test_set
    #3. Remove outliers
    Q1 = train_set.quantile(0.2)
    Q3 = train_set.quantile(0.7)
    IQR = Q3 - Q1

    # Use a larger multiplier to be less strict (removes fewer outliers)
    multiplier = 3  # was 1.5

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    # Filter the training set
    outlier_mask = (train_set >= lower_bound) & (train_set <= upper_bound)
    dataset_clean = train_set[outlier_mask]
    
    outliers_removed = len(train_set) - len(dataset_clean)
    logger.info(f"Removed {outliers_removed} outliers using IQR method")
    print(f"Original dataset size: {len(dataset)}")
    print(f"Dataset size after outlier removal: {len(dataset_clean)}")
    print(f"Outliers removed: {outliers_removed} ({outliers_removed/len(dataset)*100:.2f}%)")

    # Optional: Apply only Box-Cox transformation if requested
    boxcox_lambda = None
    if transform_method == "box-cox":
        from scipy.stats import boxcox
        dataset_positive = dataset_clean + 1e-6 if (dataset_clean <= 0).any() else dataset_clean
        transformed, lmbda = boxcox(dataset_positive)
        dataset_transformed = pd.Series(transformed, index=dataset_clean.index)
        boxcox_lambda = lmbda
        print(f"Applied box-cox transformation (lambda={lmbda:.4f})")

        # Repeat Shapiro-Wilk test on Box-Cox–transformed data
        shapiro_stat_bc, shapiro_p_bc = stats.shapiro(dataset_transformed)
        print(f"Shapiro-Wilk Statistic after Box-Cox: {shapiro_stat_bc}")
        print(f"Shapiro-Wilk p-value after Box-Cox: {shapiro_p_bc}")
        if shapiro_p_bc > 0.05:
            print("Transformed data appears to be normally distributed (p > 0.05)")
        else:
            print("Transformed data does not appear to be normally distributed (p <= 0.05)")
    else:
        dataset_transformed = dataset_clean

    # 4. Apply Kalman filter for smoothing (if enabled)
    if apply_kalman:
        dataset_processed = apply_kalman_filter(dataset_transformed)
        print("Applied Kalman filter for data smoothing")

        shapiro_stat_bc, shapiro_p_bc = stats.shapiro(dataset_processed)
        print(f"Shapiro-Wilk Statistic after Box-Cox e Kallman: {shapiro_stat_bc}")
        print(f"Shapiro-Wilk p-value after Box-Cox e Kallman: {shapiro_p_bc}")
        if shapiro_p_bc > 0.05:
            print("Transformed data appears to be normally distributed (p > 0.05)")
        else:
            print("Transformed data does not appear to be normally distributed (p <= 0.05)")
    else:
        dataset_processed = dataset_transformed
        print("Kalman filter not applied")

    # 5. Apply scaling last (if needed)
    scaler = None
    if apply_scaling:
        scaler = StandardScaler()
        dataset_scaled = scaler.fit_transform(dataset_processed.values.reshape(-1, 1)).flatten()
        dataset_final = pd.Series(dataset_scaled, index=dataset_processed.index)
        print("Applied StandardScaler")
    else:
        dataset_final = dataset_processed
        print("No scaling applied")
    
    # Plot the preprocessing steps and final result
    if transform_method == "box-cox":
        # Plot 1: Original, Outlier Removal, Final Output (all in original scale)
        plt.figure(figsize=(12, 6))
        plt.plot(dataset.index, dataset, label='Original (no NaNs)', alpha=0.5)
        plt.plot(dataset_clean.index, dataset_clean, label='After Outlier Removal', alpha=0.7)
        #plt.plot(dataset_final.index, dataset_final, label='Final Output (Box-Cox, Kalman, Scaled)', linestyle='--', linewidth=2)
        plt.title('Preprocessing Result (Original Scale)')
        plt.xlabel('Timestamp')
        plt.ylabel('Calories (kcal)')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot 2: Box-Cox transformed and postprocessing steps (all in Box-Cox scale)
        plt.figure(figsize=(12, 6))
        plt.plot(dataset_transformed.index, dataset_transformed, label='Box-Cox Transformed', alpha=0.7)
        if apply_kalman:
            plt.plot(dataset_processed.index, dataset_processed, label='Smoothed (Kalman Filter)', linewidth=2)
        if apply_scaling:
            plt.plot(dataset_final.index, dataset_final, label='Final Output (Scaled)', linestyle='--', linewidth=2)
        plt.title('Preprocessing Result (Box-Cox Scale)')
        plt.xlabel('Timestamp')
        plt.ylabel('Box-Cox( Calories (kcal) )')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        # Single plot for non-Box-Cox
        plt.figure(figsize=(12, 6))
        plt.plot(dataset.index, dataset, label='Original (no NaNs)', alpha=0.5)
        plt.plot(dataset_clean.index, dataset_clean, label='After Outlier Removal', alpha=0.7)
        if apply_kalman:
            plt.plot(dataset_processed.index, dataset_processed, label='Smoothed (Kalman Filter)', linewidth=2)
            if apply_scaling:
                plt.plot(dataset_final.index, dataset_final, label='Final Output (Smoothed + Scaled)', linestyle='--')
            else:
                plt.plot(dataset_final.index, dataset_final, label='Final Output (Smoothed)', linestyle='--')
        else:
            if apply_scaling:
                plt.plot(dataset_final.index, dataset_final, label='Final Output (Scaled)', linestyle='--')
            else:
                plt.plot(dataset_final.index, dataset_final, label='Final Output', linestyle='--', linewidth=2)
        plt.title('Preprocessing Result')
        plt.xlabel('Timestamp')
        plt.ylabel('Calories (kcal)')
        plt.legend()
        plt.grid(True)
        plt.show()
    

    # Seasonal Decomposition of final processed data
    if plot_decomposition and len(dataset_final) >= 2 * decomposition_period:
        try:
            print(f"\n[Preprocessing] Performing Seasonal Decomposition (period={decomposition_period}) on Final Processed Data:")
            decompose_result = seasonal_decompose(dataset_final, model='additive', period=decomposition_period)
            plt.figure(figsize=(12, 10))
            plt.subplot(411)
            plt.plot(decompose_result.observed, label='Observed')
            plt.legend(loc='upper left')
            plt.title(f'Seasonal Decomposition - Final Processed Data (Period={decomposition_period})')
            plt.subplot(412)
            plt.plot(decompose_result.trend, label='Trend')
            plt.legend(loc='upper left')
            plt.subplot(413)
            plt.plot(decompose_result.seasonal,label='Seasonality')
            plt.legend(loc='upper left')
            plt.subplot(414)
            plt.plot(decompose_result.resid, label='Residuals')
            plt.legend(loc='upper left')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[Preprocessing] Could not perform seasonal decomposition: {e}")
    
    # Store preprocessing parameters for later use
    preprocessing_params = (Q1, Q3, multiplier, boxcox_lambda, scaler, apply_kalman, apply_scaling)
    
    # Also return the original splits for SARIMA
    return dataset_final, scaler, boxcox_lambda, preprocessing_params, train_set, val_set, test_set