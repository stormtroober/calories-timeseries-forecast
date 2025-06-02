# --- AUTO SARIMA PARAMETER SEARCH ---
# Seasonal period: 30
# Search space: p,d,q ≤ (2,1,2), P,D,Q ≤ (1,1,1)
# Progress: 10/144 (6.9%) - Best AIC so far: -15102.55
# Progress: 20/144 (13.9%) - Best AIC so far: -15678.58
# Progress: 30/144 (20.8%) - Best AIC so far: -16702.66
# Progress: 40/144 (27.8%) - Best AIC so far: -16707.47
# Progress: 50/144 (34.7%) - Best AIC so far: -16781.20
# Progress: 60/144 (41.7%) - Best AIC so far: -16781.20
# Progress: 70/144 (48.6%) - Best AIC so far: -16783.09
# Progress: 80/144 (55.6%) - Best AIC so far: -16783.09
# Progress: 90/144 (62.5%) - Best AIC so far: -16835.02
# Progress: 100/144 (69.4%) - Best AIC so far: -16835.02
# Progress: 110/144 (76.4%) - Best AIC so far: -16835.02
# Progress: 120/144 (83.3%) - Best AIC so far: -16842.31
# Progress: 130/144 (90.3%) - Best AIC so far: -16842.31
# Progress: 140/144 (97.2%) - Best AIC so far: -16842.31

# ✓ Search completed!
# Best SARIMA order: (2, 0, 2)
# Best seasonal order: (0, 0, 0, 30)
# Best AIC: -16842.31

# best_order, best_seasonal_order, results_df = auto_sarima_search(
#     data_dict, 
#     seasonal_period=30,
#     max_p=2, max_d=1, max_q=2,
#     max_P=1, max_D=1, max_Q=1
# )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from dataset_utils.preprocessing import inverse_transform_predictions
import warnings

warnings.filterwarnings("ignore")

def evaluate_model(true, predicted, label_prefix="", scale_name=""):
    """Evaluate model performance with multiple metrics"""
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    mape = mean_absolute_percentage_error(true, predicted) * 100
    
    print(f"{label_prefix}RMSE: {rmse:.4f} {scale_name}")
    print(f"{label_prefix}MAE: {mae:.4f} {scale_name}")
    print(f"{label_prefix}MAPE: {mape:.2f}%")
    
    return rmse, mae, mape

def fit_sarima_model(data_dict, seasonal_period=7, order=(1, 0, 1), seasonal_order=(1, 0, 1, 7)):
    """
    Improved SARIMA model with proper evaluation in original scale
    """
    print("\n" + "="*50)
    print(f"SARIMA MODEL TRAINING (Period={seasonal_period})")
    print("="*50)
    
    # Extract processed data
    train_processed = data_dict['train_processed']
    val_processed = data_dict['val_processed']
    test_processed = data_dict['test_processed']
    
    # Extract original data for evaluation
    train_original = data_dict['train_original']
    val_original = data_dict['val_original'] 
    test_original = data_dict['test_original']
    
    preprocessing_params = data_dict['preprocessing_params']
    
    print(f"Training samples: {len(train_processed)}")
    print(f"Validation samples: {len(val_processed)}")
    print(f"Test samples: {len(test_processed)}")
    print(f"SARIMA order: {order}")
    print(f"Seasonal order: {seasonal_order}")
    
    # 1. Fit SARIMA model on training data
    print("\n--- Training SARIMA Model ---")
    try:
        model = SARIMAX(
            train_processed, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        results = model.fit(disp=False, maxiter=100)
        print("✓ Model training completed successfully")
        print(f"AIC: {results.aic:.2f}")
        print(f"BIC: {results.bic:.2f}")
        print(f"Log-likelihood: {results.llf:.2f}")
        
    except Exception as e:
        print(f"✗ Model training failed: {e}")
        return None, None, None

    # 2. Validation predictions
    try:
        val_pred_processed = results.get_forecast(steps=len(val_processed)).predicted_mean
        val_pred_original = inverse_transform_predictions(val_pred_processed, preprocessing_params)
        
        print("\nValidation Evaluation on Original Scale:")
        # Align indices for evaluation - use only samples that survived preprocessing
        val_original_clean = val_original[val_processed.index]
        evaluate_model(val_original_clean, val_pred_original, "Val ", "(kcal)")
        
    except Exception as e:
        print(f"✗ Validation prediction failed: {e}")
        val_pred_original = None

    # 3. Refit model on train+val for final test predictions
    try:
        train_val_processed = pd.concat([train_processed, val_processed])
        
        final_model = SARIMAX(
            train_val_processed, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False, 
            enforce_invertibility=False
        )
        final_results = final_model.fit(disp=False, maxiter=100)
        
        # Test predictions
        test_pred_processed = final_results.get_forecast(steps=len(test_processed)).predicted_mean
        test_pred_original = inverse_transform_predictions(test_pred_processed, preprocessing_params)
        
        print("\nTest Evaluation on Original Scale:")
        # Align indices for evaluation
        test_original_clean = test_original[test_processed.index]
        evaluate_model(test_original_clean, test_pred_original, "Test ", "(kcal)")
        
    except Exception as e:
        print(f"✗ Test prediction failed: {e}")
        test_pred_original = None
        final_results = results
    
    # Plot 1: Processed scale
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    plt.plot(train_processed.index, train_processed, label="Train", color="blue", alpha=0.7)
    plt.plot(val_processed.index, val_processed, label="Validation", color="orange", alpha=0.8)
    plt.plot(test_processed.index, test_processed, label="Test", color="gray", alpha=0.8)
    
    if val_pred_processed is not None:
        plt.plot(val_pred_processed.index, val_pred_processed, 
                label="Val Prediction", color="green", linewidth=2, linestyle='--')
    if test_pred_original is not None:
        plt.plot(test_pred_processed.index, test_pred_processed, 
                label="Test Prediction", color="red", linewidth=2, linestyle='--')
    
    plt.title(f"SARIMA Forecasting - Processed Scale (Order={order}, Seasonal={seasonal_order})")
    plt.xlabel("Time")
    plt.ylabel("Calories (Transformed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Original scale
    plt.subplot(2, 1, 2)
    plt.plot(train_original.index, train_original, label="Train Original", color="blue", alpha=0.7)
    plt.plot(val_original.index, val_original, label="Validation Original", color="orange", alpha=0.8)
    plt.plot(test_original.index, test_original, label="Test Original", color="gray", alpha=0.8)
    
    if val_pred_original is not None:
        plt.plot(val_pred_original.index, val_pred_original, 
                label="Val Prediction", color="green", linewidth=2, linestyle='--')
    if test_pred_original is not None:
        plt.plot(test_pred_original.index, test_pred_original, 
                label="Test Prediction", color="red", linewidth=2, linestyle='--')
    
    plt.title("SARIMA Forecasting - Original Scale (kcal)")
    plt.xlabel("Time")
    plt.ylabel("Calories (kcal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("SARIMA MODEL ANALYSIS COMPLETED")
    print("="*50)

    return final_results, val_pred_original, test_pred_original

# Auto SARIMA parameter selection
def auto_sarima_search(data_dict, seasonal_period=7, max_p=3, max_d=2, max_q=3, 
                      max_P=2, max_D=1, max_Q=2):
    """
    Automatic SARIMA parameter selection using grid search
    """
    print(f"\n--- AUTO SARIMA PARAMETER SEARCH ---")
    print(f"Seasonal period: {seasonal_period}")
    print(f"Search space: p,d,q ≤ ({max_p},{max_d},{max_q}), P,D,Q ≤ ({max_P},{max_D},{max_Q})")
    
    train_data = data_dict['train_processed']
    val_data = data_dict['val_processed']
    
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    results_log = []
    
    total_combinations = (max_p+1) * (max_d+1) * (max_q+1) * (max_P+1) * (max_D+1) * (max_Q+1)
    current_combo = 0
    
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            current_combo += 1
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, seasonal_period)
                            
                            try:
                                model = SARIMAX(train_data, order=order, seasonal_order=seasonal_order,
                                              enforce_stationarity=False, enforce_invertibility=False)
                                fitted_model = model.fit(disp=False, maxiter=50)
                                
                                aic = fitted_model.aic
                                results_log.append({
                                    'order': order,
                                    'seasonal_order': seasonal_order,
                                    'aic': aic,
                                    'bic': fitted_model.bic
                                })
                                
                                if aic < best_aic:
                                    best_aic = aic
                                    best_order = order
                                    best_seasonal_order = seasonal_order
                                
                                if current_combo % 10 == 0:
                                    print(f"Progress: {current_combo}/{total_combinations} "
                                          f"({100*current_combo/total_combinations:.1f}%) - "
                                          f"Best AIC so far: {best_aic:.2f}")
                                    
                            except Exception as e:
                                continue
    
    print(f"\n✓ Search completed!")
    print(f"Best SARIMA order: {best_order}")
    print(f"Best seasonal order: {best_seasonal_order}")
    print(f"Best AIC: {best_aic:.2f}")
    
    # Show top 5 models
    results_df = pd.DataFrame(results_log).sort_values('aic').head(10)
    print(f"\nTop 10 models by AIC:")
    print(results_df.to_string(index=False))
    
    return best_order, best_seasonal_order, results_df