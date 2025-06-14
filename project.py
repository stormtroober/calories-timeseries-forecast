import pandas as pd
from dataset_utils.preprocessing import preprocessing
from models.sarima import fit_sarima_model, auto_sarima_search 
from models.mlp_keras import fit_mlp_model, grid_search_mlp
from models.xgboost import fit_xgboost_model, grid_search_xgboost


dataframe = pd.read_csv('./dataset/Daily_activity_metrics.csv')
seasonal_period = 30  # Adjust based on your data's seasonality

#Preprocessing without scaling for SARIMA and XGBoost models
data_dict = preprocessing(
    dataframe, 
    apply_scaling=False, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=True,
    plot_processed_data=False,
    decomposition_period=seasonal_period
)

#Preprocessing with scaling for MLP model 
data_dict_mlp = preprocessing(
    dataframe, 
    apply_scaling=True, 
    apply_kalman=True, 
    transform_method='box-cox',
    plot_decomposition=False,
    plot_processed_data=False,
    decomposition_period=seasonal_period
)

model_results, val_predictions, test_predictions = fit_sarima_model(
    data_dict,
    seasonal_period=seasonal_period,
    order=(2, 0, 2),
    seasonal_order=(0, 0, 0, seasonal_period)
)

model_mlp, val_preds_mlp, test_preds_mlp = fit_mlp_model(data_dict_mlp,
                                                         look_back=seasonal_period,
                                                         hidden_units=8,
                                                         epochs=100,
                                                         batch_size=16)


look_back_list = [7, 14, 21, 30, 45, 60]
n_estimators_list = [10, 15, 25, 35, 40, 50, 75, 100]

results_df, best_config, best_model = grid_search_xgboost(
    data_dict, look_back_list, n_estimators_list
)

print(results_df.head())
print(best_config)

model_xgb, val_preds_xgb, test_preds_xgb = fit_xgboost_model(data_dict, look_back=best_config['look_back'], n_estimators=best_config['n_estimators'])

from results_analysis import create_comprehensive_results_report

# Generate comprehensive results report
models_performance, best_model_info = create_comprehensive_results_report(
    model_results, val_predictions, test_predictions,  # SARIMA results
    model_mlp, val_preds_mlp, test_preds_mlp,         # MLP results  
    model_xgb, val_preds_xgb, test_preds_xgb,         # XGBoost results
    results_df, best_config, data_dict                 # Additional data
)