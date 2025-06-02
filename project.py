import pandas as pd
from preprocessing import preprocessing
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# Load and preprocess data
dataframe = pd.read_csv('./dataset/Daily_activity_metrics.csv')

# Preprocess the data and get all necessary components
train_preprocessed, scaler, boxcox_lambda, preprocessing_params, train_original, val_original, test_original = preprocessing(
    dataframe, 
    apply_scaling=False, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=True,         # Enable decomposition
    decomposition_period=30
)

print(f"Training data preprocessed: {len(train_preprocessed)} samples")
print(f"Original validation set: {len(val_original)} samples")
print(f"Original test set: {len(test_original)} samples")
print(f"Box-Cox lambda: {boxcox_lambda}")



# Placeholder for SARIMA fitting - you would call your sarima fitting function here
# For example:
# if adf_result[1] <= 0.05: # If stationary
#     print("\n--- Proceeding with SARIMA model fitting ---")
#     # model_results = fit_sarima_model_with_data(train_preprocessed, val_original, test_original, preprocessing_params)
#     # print(model_results.summary())
# else:
#     print("\nData is not stationary. Further differencing might be needed for SARIMA.")

print("\n--- End of Analysis ---")