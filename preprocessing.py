import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.seasonal import STL
from scipy import stats
from pykalman import KalmanFilter
from logger import logger
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

def preprocessing(df, apply_scaling=False, apply_kalman=False, transform_method=None):
    dataset = df['Calories (kcal)'].copy()
    nan_count = dataset.isnull().sum()
    logger.info(f"Number of NaN values in 'Calories (kcal)': {nan_count}")

    dataset = dataset.dropna()

    # 2. Apply statistical tests on original data
    result = adfuller(dataset)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")

    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(dataset)
    print(f"Shapiro-Wilk Statistic: {shapiro_stat}")
    print(f"Shapiro-Wilk p-value: {shapiro_p}")
    if shapiro_p > 0.05:
        print("Data appears to be normally distributed (p > 0.05)")
    else:
        print("Data does not appear to be normally distributed (p <= 0.05)")

    # 3. Remove outliers
    Q1 = dataset.quantile(0.25)
    Q3 = dataset.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remove outliers
    outlier_mask = (dataset >= lower_bound) & (dataset <= upper_bound)
    dataset_clean = dataset[outlier_mask]
    
    outliers_removed = len(dataset) - len(dataset_clean)
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
    return dataset_final, scaler, boxcox_lambda