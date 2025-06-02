import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from preprocessing import inverse_transform_predictions


def evaluate_model(true, predicted, label_prefix="", scale_name=""):
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    mape = mean_absolute_percentage_error(true, predicted) * 100

    print(f"{label_prefix}RMSE: {rmse:.4f} {scale_name}")
    print(f"{label_prefix}MAE: {mae:.4f} {scale_name}")
    print(f"{label_prefix}MAPE: {mape:.2f}%")

    return rmse, mae, mape


def create_sliding_window(series, look_back):
    data = series.values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)


def fit_xgboost_model(data_dict, look_back=30, n_estimators=100):
    print("\n" + "="*50)
    print(f"XGBOOST MODEL TRAINING (look_back={look_back})")
    print("="*50)

    train_proc = data_dict['train_processed']
    val_proc = data_dict['val_processed']
    test_proc = data_dict['test_processed']

    train_orig = data_dict['train_original']
    val_orig = data_dict['val_original']
    test_orig = data_dict['test_original']

    params = data_dict['preprocessing_params']

    print(f"Training samples: {len(train_proc)}")
    print(f"Validation samples: {len(val_proc)}")
    print(f"Test samples: {len(test_proc)}")

    # Create sliding windows for training
    X_train, y_train = create_sliding_window(train_proc, look_back)

    model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Validation prediction (recursive)
    val_preds_proc = []
    input_seq = train_proc.values[-look_back:].tolist()
    for _ in range(len(val_proc)):
        x_input = np.array(input_seq).reshape(1, look_back)
        pred = model.predict(x_input)[0]
        val_preds_proc.append(pred)
        input_seq = input_seq[1:] + [pred]
    val_preds_proc = pd.Series(val_preds_proc, index=val_proc.index)
    val_preds_orig = inverse_transform_predictions(val_preds_proc, params)

    print("\nValidation Evaluation on Original Scale:")
    val_orig_clean = val_orig[val_proc.index]
    evaluate_model(val_orig_clean, val_preds_orig, label_prefix="Val ", scale_name="(kcal)")

    # Refit model on train + val for test prediction
    train_val_proc = pd.concat([train_proc, val_proc])
    X_train_val, y_train_val = create_sliding_window(train_val_proc, look_back)
    model.fit(X_train_val, y_train_val)

    # Test prediction (recursive)
    test_preds_proc = []
    input_seq = train_val_proc.values[-look_back:].tolist()
    for _ in range(len(test_proc)):
        x_input = np.array(input_seq).reshape(1, look_back)
        pred = model.predict(x_input)[0]
        test_preds_proc.append(pred)
        input_seq = input_seq[1:] + [pred]
    test_preds_proc = pd.Series(test_preds_proc, index=test_proc.index)
    test_preds_orig = inverse_transform_predictions(test_preds_proc, params)

    print("\nTest Evaluation on Original Scale:")
    test_orig_clean = test_orig[test_proc.index]
    evaluate_model(test_orig_clean, test_preds_orig, label_prefix="Test ", scale_name="(kcal)")

    # Plot
    plt.figure(figsize=(15, 10))

    # Processed scale
    plt.subplot(2, 1, 1)
    plt.plot(train_proc.index, train_proc, label="Train", color="blue", alpha=0.7)
    plt.plot(val_proc.index, val_proc, label="Validation", color="orange", alpha=0.8)
    plt.plot(test_proc.index, test_proc, label="Test", color="gray", alpha=0.8)
    plt.plot(val_preds_proc.index, val_preds_proc, label="Val Prediction", color="green", linestyle='--')
    plt.plot(test_preds_proc.index, test_preds_proc, label="Test Prediction", color="red", linestyle='--')
    plt.title("XGBoost Forecasting - Processed Scale")
    plt.xlabel("Time")
    plt.ylabel("Calories (Processed)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Original scale
    plt.subplot(2, 1, 2)
    plt.plot(train_orig.index, train_orig, label="Train Original", color="blue", alpha=0.7)
    plt.plot(val_orig.index, val_orig, label="Validation Original", color="orange", alpha=0.8)
    plt.plot(test_orig.index, test_orig, label="Test Original", color="gray", alpha=0.8)
    plt.plot(val_preds_orig.index, val_preds_orig, label="Val Prediction", color="green", linestyle='--')
    plt.plot(test_preds_orig.index, test_preds_orig, label="Test Prediction", color="red", linestyle='--')
    plt.title("XGBoost Forecasting - Original Scale (kcal)")
    plt.xlabel("Time")
    plt.ylabel("Calories (kcal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n" + "="*50)
    print("XGBOOST MODEL ANALYSIS COMPLETED")
    print("="*50)

    return model, val_preds_orig, test_preds_orig


def grid_search_xgboost(data_dict, look_back_list, n_estimators_list):
    """
    Grid search for XGBoost model on validation RMSE
    """
    train_proc = data_dict['train_processed']
    val_proc = data_dict['val_processed']
    train_orig = data_dict['train_original']
    val_orig = data_dict['val_original']
    params = data_dict['preprocessing_params']

    results = []
    best_rmse = np.inf
    best_model = None
    best_config = None

    for look_back in look_back_list:
        X_train, y_train = create_sliding_window(train_proc, look_back)
        for n_estimators in n_estimators_list:
            try:
                print(f"Testing look_back={look_back}, n_estimators={n_estimators}")
                model = XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=42)
                model.fit(X_train, y_train)

                val_preds_proc = []
                input_seq = train_proc.values[-look_back:].tolist()
                for _ in range(len(val_proc)):
                    x_input = np.array(input_seq).reshape(1, look_back)
                    pred = model.predict(x_input)[0]
                    val_preds_proc.append(pred)
                    input_seq = input_seq[1:] + [pred]
                val_preds_proc = pd.Series(val_preds_proc, index=val_proc.index)
                val_preds_orig = inverse_transform_predictions(val_preds_proc, params)

                val_orig_clean = val_orig[val_proc.index]
                rmse, _, _ = evaluate_model(val_orig_clean, val_preds_orig, label_prefix="[Grid] Val ", scale_name="(kcal)")

                results.append({
                    'look_back': look_back,
                    'n_estimators': n_estimators,
                    'val_rmse': rmse
                })

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_config = {
                        'look_back': look_back,
                        'n_estimators': n_estimators
                    }
            except Exception as e:
                print(f"Error with look_back={look_back}, n_estimators={n_estimators}: {e}")

    results_df = pd.DataFrame(results).sort_values('val_rmse').reset_index(drop=True)
    print("\nGrid search completed. Best config:", best_config)
    return results_df, best_config, best_model