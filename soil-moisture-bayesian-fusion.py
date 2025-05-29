import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
import json
import csv

output_dir = "results"
data_dir = "data"

# ------------------------ User-Defined Variables ------------------------
 
lag = 10  # Number of historical measurements to consider

# ------------------------ Load Data ------------------------
# SMAP_L4 = np.load("SMAP_L4_kendall.npy")  # shape: (samples,)
# AUX_all = np.load("AUX_all_kendall.npy")  # shape: (samples, aux_features)
# MM_all = np.load("MM_all_kendall.npy")        # shape: (samples, 17, 11)

# Define the site name
# site_name = "jr1"
site_name = "jr2"
# site_name = "jr3"
# site_name = "kendall"
#site_name = "lucky_hills"
# site_name = "z1"
# site_name = "z4"


# List of site names
forecast_horizon_list=[1,2,3,4,5]

metrics = {}

for forecast_horizon in forecast_horizon_list:

    
    #site_name="lucky_hills"
    # Dynamically generate file paths using the site_name variable
    
    SMAP_L4 = np.load(f"{data_dir}/SMAP_L4_{site_name}.npy")
    AUX_all = np.load(f"{data_dir}/AUX_all_{site_name}.npy")
    MM_all = np.load(f"{data_dir}/MM_all_{site_name}.npy")
    # SMAP_L4 = np.load(f"SMAP_L4_{site_name}.npy")  # shape: (samples,)
    # AUX_all = np.load(f"AUX_all_{site_name}.npy")  # shape: (samples, aux_features)
    # MM_all = np.load(f"MM_all_{site_name}.npy")    # shape: (samples, 17, 11)
    
    
    
    print("Loaded Data Shapes:")
    print(f"SMAP_L4 shape: {SMAP_L4.shape}")
    print(f"AUX_all shape: {AUX_all.shape}")
    print(f"MM_all shape: {MM_all.shape}")
    
    # Flatten MM_all to 2D for concatenation with AUX_all
    MM_all_flat = MM_all.reshape(MM_all.shape[0], -1)  # shape: (samples, 17*11)
    
    # Combine AUX_all with MM_all
    AUX_all_combined = np.concatenate((AUX_all, MM_all_flat), axis=1)
    print(f"AUX_all_combined shape: {AUX_all_combined.shape}")
    
    # ------------------------ Define Common Start and End Points ------------------------
    start_point = lag  # Retrieval requires `lag` historical measurements
    end_point = len(SMAP_L4) - forecast_horizon  # Forecasting requires `forecast_horizon` future values
    common_start = max(start_point, lag)
    common_end = min(end_point, len(SMAP_L4) - forecast_horizon)
    
    print(f"Common Range: {common_start} to {common_end}")
    
    # Adjust SMAP and auxiliary data based on common range
    SMAP_common = SMAP_L4[common_start:common_end]
    AUX_common = AUX_all_combined[common_start:common_end]
    MM_all_common = MM_all_flat[common_start:common_end,:]
    
    # ------------------------ Split Data ------------------------
    train_ratio = 0.8
    n_samples   = SMAP_common.shape[0]
    train_end   = int(n_samples * train_ratio)
    
    SMAP_train = SMAP_common[:train_end]
    SMAP_val   = SMAP_common[train_end:]
    AUX_train  = AUX_common[:train_end]
    AUX_val    = AUX_common[train_end:]
    # MM_train  = MM_all_common[:train_end]
    # MM_val    = MM_all_common[train_end:]
    
    print("Data Splits:")
    print(f"SMAP_train shape: {SMAP_train.shape}, SMAP_val shape: {SMAP_val.shape}")
    print(f"AUX_train shape: {AUX_train.shape}, AUX_val shape: {AUX_val.shape}")
    # print(f"MM_train shape: {MM_train.shape}, MM_val shape: {MM_val.shape}")
    
    # ------------------------ (Optional) Scale ------------------------
    smap_scaler = MinMaxScaler()
    aux_scaler  = MinMaxScaler()
    
    SMAP_train_norm = smap_scaler.fit_transform(SMAP_train.reshape(-1,1)).flatten()
    SMAP_val_norm   = smap_scaler.transform(SMAP_val.reshape(-1,1)).flatten()
    AUX_train_norm  = aux_scaler.fit_transform(AUX_train)
    AUX_val_norm    = aux_scaler.transform(AUX_val)
    # MM_train_norm   = MM_train
    # MM_val_norm   = MM_val
    print("Data Normalization Complete")
    
    # ------------------------ Retrieval Model ------------------------
    retrieval_ngb = NGBRegressor(
        Dist=Normal,
        n_estimators=500,
        learning_rate=0.01,
        verbose=False,
        random_state=42
    )
    print("Training Retrieval Model...")
    retrieval_ngb.fit(AUX_train_norm, SMAP_train_norm)
    print("Retrieval Model Training Complete")
    
    # ------------------------ Predict Retrieval ------------------------
    pred_retrieval_dist_train = retrieval_ngb.pred_dist(AUX_train_norm)
    y_retrieval_loc_train     = pred_retrieval_dist_train.params['loc']
    # y_retrieval_loc_train     = SMAP_train_norm
    y_retrieval_scale_train   = pred_retrieval_dist_train.params['scale']
    
    pred_retrieval_dist_val = retrieval_ngb.pred_dist(AUX_val_norm)
    y_retrieval_loc_val     = pred_retrieval_dist_val.params['loc']
    # y_retrieval_loc_val     = SMAP_val_norm
    y_retrieval_scale_val   = pred_retrieval_dist_val.params['scale']
    
    print("Retrieval Predictions Complete")
    
    # Adjust retrieval to match SMAP values directly (same time instance)
    SMAP_train_retrieved = smap_scaler.inverse_transform(y_retrieval_loc_train.reshape(-1, 1)).flatten()
    SMAP_val_retrieved = smap_scaler.inverse_transform(y_retrieval_loc_val.reshape(-1, 1)).flatten()
    
    # ------------------------ Forecasting Model ------------------------
    def make_supervised_data_with_aux(smap, aux, lag, forecast_horizon):
        X, y = [], []
        for i in range(lag, len(smap) - forecast_horizon):
            smap_window = smap[i-lag:i]
            aux_current = aux[i]
            features = np.concatenate([smap_window, aux_current])
            X.append(features)
            y.append(smap[i + forecast_horizon])  # Forecast horizon determines target
        return np.array(X), np.array(y)
    
    X_train, y_train = make_supervised_data_with_aux(SMAP_train_norm, AUX_train_norm, lag, forecast_horizon)
    X_val,   y_val   = make_supervised_data_with_aux(SMAP_val_norm,   AUX_val_norm, lag, forecast_horizon)
    
    print("Forecast Data Shapes:")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    
    forecast_ngb = NGBRegressor(
        Dist=Normal,
        n_estimators=500,
        learning_rate=0.01,
        verbose=False,
        random_state=42
    )
    print("Training Forecasting Model...")
    forecast_ngb.fit(X_train, y_train)
    print("Forecasting Model Training Complete")
    
    # ------------------------ Predict Forecast ------------------------
    pred_forecast_dist_train = forecast_ngb.pred_dist(X_train)
    y_forecast_loc_train     = pred_forecast_dist_train.params['loc']
    # y_forecast_loc_train     = y_train
    y_forecast_scale_train   = pred_forecast_dist_train.params['scale']
    
    pred_forecast_dist_val = forecast_ngb.pred_dist(X_val)
    y_forecast_loc_val     = pred_forecast_dist_val.params['loc']
    # y_forecast_loc_val     = y_val
    y_forecast_scale_val   = pred_forecast_dist_val.params['scale']
    
    # Inverse-transform forecasts
    y_train_forecasted = smap_scaler.inverse_transform(y_forecast_loc_train.reshape(-1, 1)).flatten()
    y_val_forecasted = smap_scaler.inverse_transform(y_forecast_loc_val.reshape(-1, 1)).flatten()
    
    print("Forecast Predictions Complete")
    
    # ------------------------ Bayesian Combination ------------------------
    def combine_estimates(loc1, scale1, loc2, scale2):
        combined_var = 1 / (1 / scale1**2 + 1 / scale2**2)
        combined_mean = combined_var * (loc1 / scale1**2 + loc2 / scale2**2)
        combined_std = np.sqrt(combined_var)
        return combined_mean, combined_std
    
    # Combine Retrieval and Forecast for Validation
    # min_length_train = min(len(SMAP_train_retrieved), len(y_train_forecasted))
    combined_loc_train, combined_scale_train = combine_estimates(
        SMAP_train_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)], y_retrieval_scale_train[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)],
        y_train_forecasted, y_forecast_scale_train)
    
    min_length_val = min(len(SMAP_val_retrieved), len(y_val_forecasted))
    combined_loc_val, combined_scale_val = combine_estimates(
        SMAP_val_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)], y_retrieval_scale_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
        y_val_forecasted, y_forecast_scale_val)
    
    # ------------------------ Metrics ------------------------
    print("Metrics:")
    def calculate_metrics(predicted, true):
        mse = np.mean((predicted - true) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predicted - true))
        return mse, rmse, mae
    
    # Retrieval Metrics
    mse_retrieval_train, rmse_retrieval_train, mae_retrieval_train = calculate_metrics(SMAP_train_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)], SMAP_train[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)])
    print(f"Retrieval (Train) - MSE: {mse_retrieval_train:.4f}, RMSE: {rmse_retrieval_train:.4f}, MAE: {mae_retrieval_train:.4f}")
    
    mse_retrieval_val, rmse_retrieval_val, mae_retrieval_val = calculate_metrics(SMAP_val_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)], SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)])
    print(f"Retrieval (Val) - MSE: {mse_retrieval_val:.4f}, RMSE: {rmse_retrieval_val:.4f}, MAE: {mae_retrieval_val:.4f}")
    
    # Forecast Metrics
    mse_forecast_train, rmse_forecast_train, mae_forecast_train = calculate_metrics(y_train_forecasted, SMAP_train[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)])
    print(f"Forecast (Train) - MSE: {mse_forecast_train:.4f}, RMSE: {rmse_forecast_train:.4f}, MAE: {mae_forecast_train:.4f}")
    
    mse_forecast_val, rmse_forecast_val, mae_forecast_val = calculate_metrics(y_val_forecasted, SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)])
    print(f"Forecast (Val) - MSE: {mse_forecast_val:.4f}, RMSE: {rmse_forecast_val:.4f}, MAE: {mae_forecast_val:.4f}")
    
    # Combined Metrics
    mse_combined_train, rmse_combined_train, mae_combined_train = calculate_metrics(combined_loc_train, SMAP_train[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)])
    print(f"Combined (Train) - MSE: {mse_combined_train:.4f}, RMSE: {rmse_combined_train:.4f}, MAE: {mae_combined_train:.4f}")
    
    mse_combined_val, rmse_combined_val, mae_combined_val = calculate_metrics(combined_loc_val, SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_train_retrieved)])
    print(f"Combined (Val) - MSE: {mse_combined_val:.4f}, RMSE: {rmse_combined_val:.4f}, MAE: {mae_combined_val:.4f}")
    
    # ------------------------ Plot ------------------------
        
    sigma_val=1.0
    
    # Retrieval Plot - Validation
    # plt.subplot(3, 1, 1)
    # t_val = np.arange(len(SMAP_val_retrieved))
    t_val_forecast = np.arange(len(y_val_forecasted))
    plt.plot(            SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)], label="SMAP L4 (Val)", color='black',linestyle='--')
    plt.plot( SMAP_val_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)], label="Retrieved", color='green')
    plt.fill_between(t_val_forecast,
                           SMAP_val_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)] - sigma_val * y_retrieval_scale_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
                           SMAP_val_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)] + sigma_val * y_retrieval_scale_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
                       color='green', alpha=0.2, label='±1σ (68.27% CI)')
    # plt.title("Retrieval - Validation")
    plt.xlabel("Time Index")
    plt.ylabel("Soil Moisture")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 0.4)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/retrival_{site_name}.png', format='png', dpi=300)
    plt.show()
    
    
    
    
    plt.plot( SMAP_val[forecast_horizon+lag:forecast_horizon+lag + len(y_val_forecasted)], label="SMAP L4 (Val)", color='black',linestyle='--')
    plt.plot( y_val_forecasted, label="Forecasted", color='red')
    plt.fill_between(t_val_forecast, y_val_forecasted - sigma_val * y_forecast_scale_val[:len(y_val_forecasted)],
                       y_val_forecasted + sigma_val * y_forecast_scale_val[:len(y_val_forecasted)],
                       color='red', alpha=0.2, label='±1σ (68.27% CI)')
    # plt.title("Forecast - Validation")
    plt.xlabel("Time Index")
    plt.ylabel("Soil Moisture")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 0.4)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/forecasting{site_name}.png', format='png', dpi=300)
    plt.show()
    
    
    
    # Combined Plot - Validation
    # plt.subplot(3, 1, 3)
    t_val_combined = np.arange(len(combined_loc_val))
    plt.plot(SMAP_val[forecast_horizon+lag:forecast_horizon+lag + len(y_val_forecasted)], label="SMAP L4 (Val)", color='black',linestyle='--')
    plt.plot(combined_loc_val, label="Combined", color='purple')
    plt.fill_between(t_val_combined, combined_loc_val - sigma_val * combined_scale_val,
                       combined_loc_val + sigma_val * combined_scale_val,
                       color='purple', alpha=0.2, label='±1σ (68.27% CI)')
    # plt.title("Bayesian Combined - Validation")
    plt.xlabel("Time Index")
    plt.ylabel("Soil Moisture")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.0, 0.4)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/combined_{site_name}.png', format='png', dpi=300)
    plt.show()
    

    
    def calculate_unbiased_rmse(predicted, true):
        bias = np.mean(predicted - true)
        unbiased_predictions = predicted - bias
        mse = np.mean((unbiased_predictions - true) ** 2)
        rmse = np.sqrt(mse)
        return rmse

    
    from scipy.stats import norm
    
    def calculate_ece(y_true, mu_pred, sigma_pred, num_bins=10):
        """
        Calculate Expected Calibration Error (ECE).
        
        Parameters:
            y_true (np.ndarray): True values.
            mu_pred (np.ndarray): Predicted means.
            sigma_pred (np.ndarray): Predicted standard deviations.
            num_bins (int): Number of bins for confidence levels (default: 10).
        
        Returns:
            ece (float): Expected Calibration Error.
        """
        n_samples = len(y_true)
        bin_edges = np.linspace(0, 1, num_bins + 1)  # Define bin edges
        ece = 0.0
    
        for i in range(num_bins):
            # Confidence level for the current bin
            lower_bound = bin_edges[i]
            upper_bound = bin_edges[i + 1]
            confidence_level = (lower_bound + upper_bound) / 2
    
            # Calculate z-score for the confidence level
            z_score = norm.ppf(0.5 + confidence_level / 2)
    
            # Confidence intervals
            lower = mu_pred - z_score * sigma_pred
            upper = mu_pred + z_score * sigma_pred
    
            # Identify samples within the confidence interval
            in_interval = (y_true >= lower) & (y_true <= upper)
            empirical_accuracy = np.mean(in_interval)
    
            # Proportion of samples in the current bin
            bin_weight = len(in_interval) / n_samples
    
            # ECE contribution from this bin
            ece += bin_weight * np.abs(confidence_level - empirical_accuracy)
    
        return ece
    
    
    
    
    # ECE for Retrieval (Validation)
    ece_retrieval_val = calculate_ece(
        SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
        SMAP_val_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
        y_retrieval_scale_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)]
    )
    print(f"ECE (Retrieval - Validation): {ece_retrieval_val:.4f}")
    
    # ECE for Forecast (Validation)
    ece_forecast_val = calculate_ece(
        SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
        y_val_forecasted,
        y_forecast_scale_val
    )
    print(f"ECE (Forecast - Validation): {ece_forecast_val:.4f}")
    
    # ECE for Combined (Validation)
    ece_combined_val = calculate_ece(
        SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
        combined_loc_val,
        combined_scale_val
    )
    print(f"ECE (Combined - Validation): {ece_combined_val:.4f}")
    
    # Unbiased RMSE Calculations
    unbiased_rmse_retrieval_val = calculate_unbiased_rmse(
        SMAP_val_retrieved[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)],
        SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)]
    )
    unbiased_rmse_forecast_val = calculate_unbiased_rmse(
        y_val_forecasted,
        SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)]
    )
    unbiased_rmse_combined_val = calculate_unbiased_rmse(
        combined_loc_val,
        SMAP_val[forecast_horizon+lag:forecast_horizon+lag+len(SMAP_val_retrieved)]
    )
    
    print(f"Unbiased RMSE (Retrieval - Validation): {unbiased_rmse_retrieval_val:.4f}")
    print(f"Unbiased RMSE (Forecast - Validation): {unbiased_rmse_forecast_val:.4f}")
    print(f"Unbiased RMSE (Combined - Validation): {unbiased_rmse_combined_val:.4f}")
    
    

    # Store metrics for the current site
    metrics[site_name] = {
        "ECE_Retrieval_Val": ece_retrieval_val,
        "ECE_Forecast_Val": ece_forecast_val,
        "ECE_Combined_Val": ece_combined_val,
        "MSE_Retrieval_Val": mse_retrieval_val,
        "RMSE_Retrieval_Val": rmse_retrieval_val,
        "MAE_Retrieval_Val": mae_retrieval_val,
        "MSE_Forecast_Val": mse_forecast_val,
        "RMSE_Forecast_Val": rmse_forecast_val,
        "MAE_Forecast_Val": mae_forecast_val,
        "MSE_Combined_Val": mse_combined_val,
        "RMSE_Combined_Val": rmse_combined_val,
        "MAE_Combined_Val": mae_combined_val,
    }
    
    metrics[site_name].update({
        "Unbiased_RMSE_Retrieval_Val": unbiased_rmse_retrieval_val,
        "Unbiased_RMSE_Forecast_Val": unbiased_rmse_forecast_val,
        "Unbiased_RMSE_Combined_Val": unbiased_rmse_combined_val,
    })


# Save metrics to JSON file
with open(f"{output_dir}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)



# Save metrics to CSV file
with open(f"{output_dir}/metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    # Write header
    header = ["Site"] + list(next(iter(metrics.values())).keys())
    writer.writerow(header)
    # Write rows
    for site, site_metrics in metrics.items():
        writer.writerow([site] + list(site_metrics.values()))

print("Metrics saved to 'metrics.json' and 'metrics.csv'.")


# Extract site names and ECE metrics for the plot
site_names = list(metrics.keys())
ece_retrieval = [metrics[site]["ECE_Retrieval_Val"] for site in site_names]
ece_forecast = [metrics[site]["ECE_Forecast_Val"] for site in site_names]
ece_combined = [metrics[site]["ECE_Combined_Val"] for site in site_names]

# Bar width and positions
bar_width = 0.2
x = np.arange(len(site_names))

# Plotting the bars
plt.figure(figsize=(12, 6))
plt.bar(x - bar_width, ece_retrieval, width=bar_width, label='ECE Retrieval', color='blue', alpha=0.7)
plt.bar(x, ece_forecast, width=bar_width, label='ECE Forecast', color='green', alpha=0.7)
plt.bar(x + bar_width, ece_combined, width=bar_width, label='ECE Combined', color='orange', alpha=0.7)

# Adding labels, legend, and title
plt.xlabel('Sites', fontsize=12)
plt.ylabel('ECE Values', fontsize=12)
plt.title('ECE Metrics Comparison Across Sites', fontsize=14)
plt.xticks(x, site_names, fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save and show plot
plt.tight_layout()
plt.savefig(f"{output_dir}/ece_metrics_comparison.png", dpi=300)
plt.show()





