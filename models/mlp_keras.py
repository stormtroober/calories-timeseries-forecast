# {'look_back': 30, 'hidden_units': 8, 'epochs': 100, 'batch_size': 16}
# look_back_list = [7, 14, 30]
# hidden_units_list = [8, 16, 32]   # per look_back=7 o 14
# hidden_units_list += [16, 32, 64]
# epochs_list = [20, 50, 100]
# batch_size_list = [16, 32, 64]

# results_df, best_params, best_model = grid_search_mlp(
#     data_dict_mlp,
#     look_back_list,
#     hidden_units_list,
#     epochs_list,
#     batch_size_list
# )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from dataset_utils.preprocessing import inverse_transform_predictions
import time


def evaluate_model(true, predicted, label_prefix="", scale_name=""):
    """Evaluate model performance with multiple metrics"""
    rmse = np.sqrt(mean_squared_error(true, predicted))
    mae = mean_absolute_error(true, predicted)
    mape = mean_absolute_percentage_error(true, predicted) * 100
    
    print(f"{label_prefix}RMSE: {rmse:.4f} {scale_name}")
    print(f"{label_prefix}MAE: {mae:.4f} {scale_name}")
    print(f"{label_prefix}MAPE: {mape:.2f}%")
    
    return rmse, mae, mape


def create_sliding_window(series, look_back):
    """Create input/output pairs using sliding window"""
    data = series.values
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    X = np.array(X)
    y = np.array(y)
    return X, y


def train_and_evaluate(train_proc, val_proc, train_orig, val_orig, look_back, hidden_units,
                       epochs, batch_size, params):
    """
    Train an MLP on train_proc and evaluate on val_proc; return validation RMSE and trained model
    """
    # Prepare training windows
    X_train, y_train = create_sliding_window(train_proc, look_back)

    # Build MLP model
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_dim=look_back))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    
    # Train
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Validation: recursive forecasting
    val_preds_proc = []
    input_seq = train_proc.values[-look_back:].tolist()
    for _ in range(len(val_proc)):
        x_input = np.array(input_seq).reshape(1, look_back)
        pred = model.predict(x_input, verbose=0)[0][0]
        val_preds_proc.append(pred)
        input_seq = input_seq[1:] + [pred]
    val_preds_proc = pd.Series(val_preds_proc, index=val_proc.index)

    # Inverse transform validation predictions
    val_preds_orig = inverse_transform_predictions(val_preds_proc, params)

    # Align cleaned original validation
    val_orig_clean = val_orig[val_proc.index]
    rmse, mae, mape = evaluate_model(val_orig_clean, val_preds_orig, label_prefix="Val ", scale_name="(kcal)")

    return rmse, model


def grid_search_mlp(data_dict, look_back_list, hidden_units_list, epochs_list, batch_size_list):
    """
    Perform grid search over MLP hyperparameters; return DataFrame of results and best params
    """
    train_proc = data_dict['train_processed']
    val_proc = data_dict['val_processed']
    train_orig = data_dict['train_original']
    val_orig = data_dict['val_original']
    params = data_dict['preprocessing_params']

    results = []
    best_rmse = np.inf
    best_params = None
    best_model = None

    # Calculate total iterations for progress tracking
    total_iterations = len(look_back_list) * len(hidden_units_list) * len(epochs_list) * len(batch_size_list)
    current_iteration = 0
    start_time = time.time()
    
    print(f"Starting grid search with {total_iterations} parameter combinations...")
    print("=" * 60)

    for look_back in look_back_list:
        for hidden_units in hidden_units_list:
            for epochs in epochs_list:
                for batch_size in batch_size_list:
                    current_iteration += 1
                    
                    # Calculate progress
                    progress_pct = (current_iteration / total_iterations) * 100
                    elapsed_time = time.time() - start_time
                    
                    if current_iteration > 1:
                        avg_time_per_iter = elapsed_time / (current_iteration - 1)
                        remaining_iters = total_iterations - current_iteration
                        eta_seconds = avg_time_per_iter * remaining_iters
                        eta_minutes = eta_seconds / 60
                        eta_str = f"ETA: {eta_minutes:.1f} min"
                    else:
                        eta_str = "ETA: calculating..."
                    
                    print(f"[{current_iteration}/{total_iterations}] ({progress_pct:.1f}%) {eta_str}")
                    print(f"Testing: look_back={look_back}, hidden_units={hidden_units}, epochs={epochs}, batch_size={batch_size}")
                    
                    try:
                        rmse, model = train_and_evaluate(
                            train_proc, val_proc, train_orig, val_orig,
                            look_back, hidden_units, epochs, batch_size, params
                        )
                        results.append({
                            'look_back': look_back,
                            'hidden_units': hidden_units,
                            'epochs': epochs,
                            'batch_size': batch_size,
                            'val_rmse': rmse
                        })
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_params = {
                                'look_back': look_back,
                                'hidden_units': hidden_units,
                                'epochs': epochs,
                                'batch_size': batch_size
                            }
                            best_model = model
                            print(f"*** New best RMSE: {rmse:.4f} ***")
                        
                        print(f"RMSE: {rmse:.4f}")
                        
                    except Exception as e:
                        print(f"Error for params {look_back},{hidden_units},{epochs},{batch_size}: {e}")
                        continue
                    
                    print("-" * 40)

    total_time = time.time() - start_time
    print("=" * 60)
    print(f"Grid search completed in {total_time/60:.1f} minutes!")
    
    results_df = pd.DataFrame(results).sort_values('val_rmse').reset_index(drop=True)
    print("\nBest params:\n", best_params)
    return results_df, best_params, best_model


def fit_mlp_model(data_dict, look_back=7, hidden_units=8, epochs=50, batch_size=16):
    """
    Fit an MLP using Keras on the processed data and evaluate on validation and test sets
    """
    # Extract processed and original series
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
    print(f"MLP look_back: {look_back}, hidden_units: {hidden_units}, epochs: {epochs}, batch_size: {batch_size}")

    # 1. Prepare training windows
    X_train, y_train = create_sliding_window(train_proc, look_back)

    # 2. Build MLP model
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_dim=look_back))
    model.add(Dense(1))  # output layer
    model.compile(optimizer='adam', loss='mse')
    
    # 3. Train on train set
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    # 4. Validation: recursive forecasting
    val_preds_proc = []
    input_seq = train_proc.values[-look_back:].tolist()
    for _ in range(len(val_proc)):
        x_input = np.array(input_seq).reshape(1, look_back)
        pred = model.predict(x_input, verbose=0)[0][0]
        val_preds_proc.append(pred)
        # slide window
        input_seq = input_seq[1:] + [pred]
    val_preds_proc = pd.Series(val_preds_proc, index=val_proc.index)
    
    # Inverse transform validation predictions
    val_preds_orig = inverse_transform_predictions(val_preds_proc, params)

    print("\nValidation Evaluation on Original Scale:")
    # Align cleaned original validation
    val_orig_clean = val_orig[val_proc.index]
    evaluate_model(val_orig_clean, val_preds_orig, label_prefix="Val ", scale_name="(kcal)")

    # 5. Refit on train + val
    train_val_proc = pd.concat([train_proc, val_proc])
    X_train_val, y_train_val = create_sliding_window(train_val_proc, look_back)

    model_final = Sequential()
    model_final.add(Dense(hidden_units, activation='relu', input_dim=look_back))
    model_final.add(Dense(1))
    model_final.compile(optimizer='adam', loss='mse')
    model_final.fit(X_train_val, y_train_val, epochs=epochs, batch_size=batch_size, verbose=2)

    # 6. Test: recursive forecasting
    test_preds_proc = []
    input_seq = train_val_proc.values[-look_back:].tolist()
    for _ in range(len(test_proc)):
        x_input = np.array(input_seq).reshape(1, look_back)
        pred = model_final.predict(x_input, verbose=0)[0][0]
        test_preds_proc.append(pred)
        input_seq = input_seq[1:] + [pred]
    test_preds_proc = pd.Series(test_preds_proc, index=test_proc.index)
    
    # Inverse transform test predictions
    test_preds_orig = inverse_transform_predictions(test_preds_proc, params)

    print("\nTest Evaluation on Original Scale:")
    test_orig_clean = test_orig[test_proc.index]
    evaluate_model(test_orig_clean, test_preds_orig, label_prefix="Test ", scale_name="(kcal)")

    # 7. Plotting
    plt.figure(figsize=(15, 10))
    # Processed scale
    plt.subplot(2, 1, 1)
    plt.plot(train_proc.index, train_proc, label="Train", color="blue", alpha=0.7)
    plt.plot(val_proc.index, val_proc, label="Validation", color="orange", alpha=0.8)
    plt.plot(test_proc.index, test_proc, label="Test", color="gray", alpha=0.8)
    plt.plot(val_preds_proc.index, val_preds_proc, label="Val Prediction", color="green", linestyle='--')
    plt.plot(test_preds_proc.index, test_preds_proc, label="Test Prediction", color="red", linestyle='--')
    plt.title(f"MLP Forecasting - Processed Scale (look_back={look_back})")
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
    plt.title("MLP Forecasting - Original Scale (kcal)")
    plt.xlabel("Time")
    plt.ylabel("Calories (kcal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return model_final, val_preds_orig, test_preds_orig
