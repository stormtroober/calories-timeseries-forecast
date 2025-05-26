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

def preprocessing(df, apply_scaling=False, apply_kalman=False):
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
    Q1 = dataset.quantile(0.2)
    Q3 = dataset.quantile(0.8)
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

    # 4. Apply Kalman filter for smoothing (if enabled)
    if apply_kalman:
        dataset_processed = apply_kalman_filter(dataset_clean)
        print("Applied Kalman filter for data smoothing")
    else:
        dataset_processed = dataset_clean
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
            # Final output is same as cleaned data, so just highlight it
            plt.plot(dataset_final.index, dataset_final, label='Final Output', linestyle='--', linewidth=2)
    
    plt.title('Preprocessing Result')
    plt.xlabel('Timestamp')
    plt.ylabel('Calories (kcal)')
    plt.legend()
    plt.grid(True)
    plt.show()
    return dataset_final, scaler