import pandas as pd
from preprocessing import preprocessing
from sarima import fit_sarima_model, auto_sarima_search 
import matplotlib.pyplot as plt
from scipy import stats
from mlp_keras import fit_mlp_model
from mlp_keras import grid_search_mlp

# Load and preprocess data
dataframe = pd.read_csv('./dataset/Daily_activity_metrics.csv')

#Preprocessing for SARIMA model
data_dict = preprocessing(
    dataframe, 
    apply_scaling=False, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=False,
    plot_processed_data=False,
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

model_results, val_predictions, test_predictions = fit_sarima_model(
    data_dict,
    seasonal_period=30,
    order=(2, 0, 2),
    seasonal_order=(0, 0, 0, 30)
)

#Preprocessing for MLP model
data_dict_mlp = preprocessing(
    dataframe, 
    apply_scaling=True, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=False,
    plot_processed_data=False,
    decomposition_period=30
)

# results_df, best_params, best_model = grid_search_mlp(
#     data_dict_mlp,
#     look_back_list,
#     hidden_units_list,
#     epochs_list,
#     batch_size_list
# )
model_mlp, val_preds_mlp, test_preds_mlp = fit_mlp_model(data_dict_mlp,
                                                         look_back=30,
                                                         hidden_units=8,
                                                         epochs=100,
                                                         batch_size=16)


