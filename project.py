import pandas as pd
from preprocessing import preprocessing
from sarima import fit_sarima_model, auto_sarima_search
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose

# Load and preprocess data
dataframe = pd.read_csv('./dataset/Daily_activity_metrics.csv')

# Preprocess the data using the improved preprocessing pipeline
data_dict = preprocessing(
    dataframe, 
    apply_scaling=False, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=False,
    decomposition_period=30
)

#Automatic parameter search (uncomment to use)
# print("\n=== AUTOMATIC PARAMETER SEARCH ===")
# best_order, best_seasonal_order, results_df = auto_sarima_search(
#     data_dict, 
#     seasonal_period=30,
#     max_p=2, max_d=1, max_q=2,
#     max_P=1, max_D=1, max_Q=1
# )

# Fit SARIMA model with improved configuration
model_results, val_predictions, test_predictions = fit_sarima_model(
    data_dict,
    seasonal_period=30,
    order=(2, 0, 2),
    seasonal_order=(0, 0, 0, 30)
)