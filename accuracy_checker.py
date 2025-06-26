import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os
import logging
import csv
import json
import yaml # Added for loading config
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utils.zabbix_connector import ZabbixConnector


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Define Base Directories (Consistent with your project root) ---
# BASE_DIR is correctly set to the directory where accuracy_checker.py itself resides (the project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODELS_DIR = os.path.join(BASE_DIR, 'models') # Not directly used by accuracy_checker, but good for consistency
COMPARISON_RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'comparison_results')

# Define specific file paths using base directories
PREDICTIONS_FILE = os.path.join(BASE_DIR, "predictions_log.csv")
ACCURACY_PLOTS_DIR = os.path.join(STATIC_DIR, 'accuracy_plots') # Nested directory for plots
COMPARISON_RESULTS_DIR = os.path.join(COMPARISON_RESULTS_BASE_DIR, 'json_reports') # Nested directory for JSONs


# Ensure all necessary output directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True) # Create models dir (used by app.py)
os.makedirs(COMPARISON_RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(ACCURACY_PLOTS_DIR, exist_ok=True)         # Explicitly create nested plots dir
os.makedirs(COMPARISON_RESULTS_DIR, exist_ok=True)    # Explicitly create nested JSONs dir


# Define configuration variables for time windows
COLLECTION_BUFFER_MINUTES = 1 # Buffer time after forecast end to ensure all actual data is collected
HISTORY_WINDOW_HOURS_FOR_PLOT = 6 # How many hours of historical data to show on the plot for context


connector = ZabbixConnector()


def calculate_metrics(actual, predicted):
    """
    Calculate various accuracy metrics between actual and predicted values.

    Args:
        actual (pd.Series or np.array): The true observed values.
        predicted (pd.Series or np.array): The forecasted values.

    Returns:
        dict: A dictionary containing 'rmse', 'mae', 'mape', 'r2', and 'points'.
    """
    # Filter out NaNs from both actual and predicted where they don't align
    # (This assumes the alignment function already handles NaNs from actual_series_raw)
    common_indices = actual.dropna().index.intersection(predicted.dropna().index)
    actual_filtered = actual.loc[common_indices]
    predicted_filtered = predicted.loc[common_indices]

    if actual_filtered.empty:
        return {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'points': 0}

    # Ensure no division by zero for MAPE if actual contains zeros
    # Calculate MAPE only for non-zero actual values
    non_zero_actual_mask = actual_filtered != 0
    if np.any(non_zero_actual_mask):
        mape = np.mean(np.abs((actual_filtered[non_zero_actual_mask] - predicted_filtered[non_zero_actual_mask]) / actual_filtered[non_zero_actual_mask])) * 100
    else:
        mape = 0 # If all actuals are zero, MAPE is conventionally 0 or undefined. Here, 0 for simplicity.


    # R2 can be negative if model is worse than predicting mean.
    # Handle the case where actual values are constant (denominator for R2 becomes zero)
    actual_mean_diff_sq = np.sum((actual_filtered - np.mean(actual_filtered))**2)
    if actual_mean_diff_sq == 0:
        # If actual values are constant, R2 is 1.0 if predictions are also constant and perfect,
        # otherwise it's ill-defined or 0.0 (indicating no better than a mean prediction).
        r2 = 1.0 if np.sum((actual_filtered - predicted_filtered)**2) == 0 else 0.0
    else:
        r2 = 1 - (np.sum((actual_filtered - predicted_filtered)**2) / actual_mean_diff_sq)
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(actual_filtered, predicted_filtered)),
        'mae': mean_absolute_error(actual_filtered, predicted_filtered),
        'mape': mape,
        'r2': r2,
        'points': len(actual_filtered)
    }
    return metrics


def align_timestamps_with_predictions(actual_series, prediction_timestamps, forecast_interval_hours):
    """
    Align actual data timestamps with prediction timestamps using a robust matching approach.
    
    Args:
        actual_series (pd.Series): Time series of actual data with datetime index
        prediction_timestamps (pd.DatetimeIndex): Expected prediction timestamps
        forecast_interval_hours (float): Forecast interval in hours
        
    Returns:
        pd.Series: Actual data aligned with prediction timestamps
    """
    logger.info(f"  Aligning actual data with prediction timestamps...")
    logger.info(f"  Forecast interval: {forecast_interval_hours} hours ({int(forecast_interval_hours*60)} minutes)")
    # logger.info(f"  Prediction timestamps: {prediction_timestamps.tolist()}") # Can be very verbose for many points
    logger.info(f"  Actual data timestamp range: {actual_series.index.min()} to {actual_series.index.max()}")
    
    # Convert forecast interval to minutes for easier handling
    interval_minutes = int(forecast_interval_hours * 60)
    
    # For each prediction timestamp, find the best matching actual data
    aligned_values = []
    matched_timestamps = []
    
    for pred_ts in prediction_timestamps.unique(): # Use .unique() to avoid redundant processing if predictions_df has duplicates
        # Define a search window around the prediction timestamp
        # Look backwards from the prediction timestamp to find data that represents this interval
        window_start = pred_ts - pd.Timedelta(minutes=interval_minutes)
        window_end = pred_ts # The interval is assumed to end at pred_ts
        
        # logger.info(f"    Looking for actual data between {window_start} and {window_end} for prediction at {pred_ts}") # Very verbose
        
        # Find actual data points within this window
        window_mask = (actual_series.index >= window_start) & (actual_series.index <= window_end)
        window_data = actual_series[window_mask]
        
        if not window_data.empty:
            # Take the mean of all data points in this window
            aligned_value = window_data.mean()
            aligned_values.append(aligned_value)
            matched_timestamps.append(pred_ts)
            # logger.info(f"    Found {len(window_data)} data points, mean value: {aligned_value:.4f}") # Very verbose
        else:
            # Fallback: Try a slightly larger window around the prediction timestamp
            # This is useful if Zabbix data collection is not perfectly regular or slightly delayed
            extended_window_margin_minutes = interval_minutes // 2 # e.g., for 2min interval, margin is 1min
            extended_window_start = pred_ts - pd.Timedelta(minutes=interval_minutes + extended_window_margin_minutes)
            extended_window_end = pred_ts + pd.Timedelta(minutes=extended_window_margin_minutes)
            
            extended_mask = (actual_series.index >= extended_window_start) & (actual_series.index <= extended_window_end)
            extended_data = actual_series[extended_mask]
            
            if not extended_data.empty:
                # Find the closest data point in the extended window
                time_diffs = np.abs((extended_data.index - pred_ts).total_seconds())
                closest_idx_in_extended = time_diffs.argmin()
                closest_value = extended_data.iloc[closest_idx_in_extended]
                closest_timestamp_actual = extended_data.index[closest_idx_in_extended]
                
                aligned_values.append(closest_value)
                matched_timestamps.append(pred_ts) # Store with prediction timestamp for alignment
                # logger.info(f"    Used closest data point at {closest_timestamp_actual} (value: {closest_value:.4f}) for prediction at {pred_ts}") # Very verbose
            else:
                logger.warning(f"    No actual data found for prediction timestamp {pred_ts} even with extended window.")
    
    # Create aligned series
    if aligned_values:
        # Sort by timestamp to ensure chronological order
        aligned_series = pd.Series(aligned_values, index=pd.to_datetime(matched_timestamps)).sort_index()
        logger.info(f"  Successfully aligned {len(aligned_series)} data points.")
        return aligned_series
    else:
        logger.warning(f"  No data points could be aligned.")
        return pd.Series(dtype=float)


def check_and_plot_accuracy():
    """
    Main function to read predictions, fetch actual data, calculate accuracy,
    and generate comparison plots.
    """
    
    try:
        # Use csv.reader for initial check to handle headers robustly
        with open(PREDICTIONS_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader) # Read header
            if not header: # If header is empty, file is essentially empty
                raise pd.errors.EmptyDataError("Header missing or file empty")
        
        # Read the predictions log file into a DataFrame
        predictions_df = pd.read_csv(PREDICTIONS_FILE, parse_dates=['prediction_time', 'forecast_timestamp'])
    except pd.errors.EmptyDataError:
        logger.info(f"Predictions log file '{PREDICTIONS_FILE}' is empty. No forecasts to check yet.")
        return
    except FileNotFoundError:
        logger.info(f"Predictions log file '{PREDICTIONS_FILE}' not found. No forecasts to check yet.")
        return
    except Exception as e:
        logger.error(f"Error reading predictions from '{PREDICTIONS_FILE}': {e}", exc_info=True)
        return

    # Group by all identifiers to get unique forecast batches
    unique_forecast_batches = predictions_df[['prediction_time', 'host', 'metric', 'item_key', 'forecast_interval_hours', 'horizon_hours']].drop_duplicates().sort_values(by='prediction_time')

    # Keep track of processed batches to avoid re-processing
    processed_batches_file = os.path.join(ACCURACY_PLOTS_DIR, "processed_batches.txt")
    processed_batches = set()
    if os.path.exists(processed_batches_file):
        with open(processed_batches_file, 'r') as f:
            for line in f:
                processed_batches.add(line.strip())

    for _, batch_info in unique_forecast_batches.iterrows():
        pred_time, host, metric_friendly_name, item_key, forecast_interval_hours, horizon_hours = batch_info.tolist()
        
        # Ensure correct type for forecast_interval_hours and horizon_hours from CSV
        forecast_interval_hours = float(forecast_interval_hours)
        horizon_hours = float(horizon_hours)

        batch_id = f"{host}_{item_key}_{pred_time.isoformat()}"
        if batch_id in processed_batches:
            logger.info(f"  Skipping already processed batch: {batch_id}")
            continue

        logger.info(f"\nProcessing batch: {host}/{metric_friendly_name} (Key: {item_key}) - Forecast made at: {pred_time.strftime('%Y-%m-%d %H:%M:%S')}")

        current_batch_predictions = predictions_df[
            (predictions_df['prediction_time'] == pred_time) &
            (predictions_df['host'] == host) &
            (predictions_df['item_key'] == item_key)
        ].sort_values(by='forecast_timestamp')

        if current_batch_predictions.empty:
            logger.info(f"  No predictions found for this batch. Skipping.")
            continue

        forecast_start_time = current_batch_predictions['forecast_timestamp'].min()
        # Add the forecast interval to include the last interval covered by the forecast point
        forecast_end_time = current_batch_predictions['forecast_timestamp'].max() + timedelta(minutes=int(forecast_interval_hours * 60)) 

        # Check if the forecast period has sufficiently passed to collect actual data
        # Note: If COLLECTION_BUFFER_MINUTES is too large, it might wait a very long time
        if datetime.now() < forecast_end_time + timedelta(minutes=COLLECTION_BUFFER_MINUTES):
            logger.info(f"  Forecast period not yet complete. Horizon ends at {forecast_end_time.strftime('%Y-%m-%d %H:%M')}. Current time: {datetime.now().strftime('%Y-%m-%d %H:%M')}. Skipping for now.")
            continue

        logger.info(f"  Forecast horizon has passed. Fetching actual data for validation and historical context.")

        # Determine the start time for fetching historical data for the plot
        plot_data_start_time = forecast_start_time - timedelta(hours=HISTORY_WINDOW_HOURS_FOR_PLOT)
        
        try:
            # Load configuration (for standalone execution) - Corrected path for root-level accuracy_checker.py
            config_path = os.path.join(BASE_DIR, "configs", "zabbix_config.yaml")
            with open(config_path) as f:
                current_config = yaml.safe_load(f)

            # Get item_id using the item_key
            if host not in current_config["hosts"] or metric_friendly_name not in current_config["hosts"][host]["metrics"]:
                logger.error(f"  Host '{host}' or metric '{metric_friendly_name}' not found in zabbix_config.yaml for item_id lookup.")
                continue # Skip this batch if config is missing

            item_id_zabbix = connector.get_item_id(host, item_key)
            actual_df_raw = connector.fetch_metric(itemid=item_id_zabbix, time_from=plot_data_start_time, time_till=forecast_end_time)
            
            actual_series_raw = actual_df_raw.set_index('timestamp')['value']
            
            if actual_series_raw.empty or len(actual_series_raw) < 1:
                logger.warning(f"  No actual data retrieved for the required period ({plot_data_start_time} to {forecast_end_time}). Skipping accuracy check.")
                continue

        except Exception as e:
            logger.error(f"  Error fetching actual data for {host}/{metric_friendly_name}: {e}", exc_info=True)
            continue

        # Isolate historical data (before forecast starts) for plotting
        historical_series = actual_series_raw[plot_data_start_time : forecast_start_time - pd.Timedelta(seconds=1)]
        
        # Get prediction timestamps for alignment
        prediction_timestamps = current_batch_predictions['forecast_timestamp']
        
        # Use the new alignment function to match actual data with prediction timestamps
        actual_test_series_aligned = align_timestamps_with_predictions(
            actual_series_raw, 
            prediction_timestamps, 
            forecast_interval_hours
        )
        
        if actual_test_series_aligned.empty:
            logger.warning(f"  No actual data could be aligned with prediction timestamps. Skipping accuracy check.")
            
            # Store partial results even if no merge happened, for debugging
            save_comparison_results(host, metric_friendly_name, pred_time, pd.DataFrame(), {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'points': 0})
            continue

        # Convert prediction columns to numeric types
        current_batch_predictions['forecasted_value'] = pd.to_numeric(current_batch_predictions['forecasted_value'])
        current_batch_predictions['lower_bound'] = pd.to_numeric(current_batch_predictions['lower_bound'])
        current_batch_predictions['upper_bound'] = pd.to_numeric(current_batch_predictions['upper_bound'])
        
        # Set indices for merging and sort them
        current_batch_predictions = current_batch_predictions.set_index('forecast_timestamp').sort_index()

        # Merge actual and predicted data based on timestamp
        comparison_df = pd.merge(actual_test_series_aligned.to_frame(name='actual'),
                                 current_batch_predictions[['forecasted_value', 'lower_bound', 'upper_bound']],
                                 left_index=True, right_index=True, how='inner')
        
        if comparison_df.empty:
            logger.warning(f"  No matching data points after merge for {host}/{metric_friendly_name} from {pred_time}. Skipping accuracy plot.")
            logger.warning(f"  Aligned actual indices: {actual_test_series_aligned.index.tolist()}")
            logger.warning(f"  Predicted (batch) indices: {current_batch_predictions.index.tolist()}")
            
            # Store partial results even if no merge happened, for debugging
            save_comparison_results(host, metric_friendly_name, pred_time, comparison_df, {'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'points': 0})
            continue
        
        # Calculate accuracy metrics
        metrics = calculate_metrics(comparison_df['actual'], comparison_df['forecasted_value'])
        
        logger.info(f"  RMSE for forecast made at {pred_time.strftime('%H:%M')}: {metrics['rmse']:.4f}")
        logger.info(f"  MAE for forecast made at {pred_time.strftime('%H:%M')}: {metrics['mae']:.4f}")
        logger.info(f"  Successfully matched {len(comparison_df)} data points for accuracy calculation")

        # Generate and save the comparison plot
        generate_comparison_plot(
            host, metric_friendly_name, pred_time,
            historical_series,
            comparison_df,
            metrics,
            forecast_interval_hours
        )
        
        # Save detailed comparison results to a JSON file
        save_comparison_results(host, metric_friendly_name, pred_time, comparison_df, metrics)
        
        # Mark batch as processed to avoid re-processing in future runs
        with open(processed_batches_file, 'a') as f:
            f.write(batch_id + '\n')


def generate_comparison_plot(host, metric_friendly_name, pred_time, historical_series, comparison_df, metrics, forecast_interval_hours,
                             start_time_for_filename=None, end_time_for_filename=None): # <-- THIS SIGNATURE IS KEY
    """
    Generate and save a plot comparing historical data, actual values, and predicted values
    with confidence intervals.
    """
    # Create figure with subplots for better separation
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 3])
    
    # --- TOP SUBPLOT: Historical Context ---
    if not historical_series.empty:
        # Limit historical data to show more relevant recent context
        recent_history_hours = min(2, HISTORY_WINDOW_HOURS_FOR_PLOT)  # Show max 2 hours of recent history
        forecast_start = comparison_df.index.min() if not comparison_df.empty else historical_series.index.max()
        recent_cutoff = forecast_start - pd.Timedelta(hours=recent_history_hours)
        recent_historical = historical_series[historical_series.index >= recent_cutoff]
        
        ax1.plot(recent_historical.index, recent_historical.values,
                 label='Recent Historical Data', color='steelblue', alpha=0.8, linewidth=1.5)
        
        # Add vertical line showing forecast start
        if not comparison_df.empty:
            ax1.axvline(x=comparison_df.index.min(), color='red', linestyle=':', alpha=0.7, 
                       label='Forecast Start', linewidth=2)
        
        ax1.set_title(f"Recent Historical Context ({recent_history_hours}h before forecast)")
        ax1.set_ylabel("Value")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format x-axis for historical data
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax1.text(0.5, 0.5, 'No Historical Data Available', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14, color='gray')
        ax1.set_title("Historical Context")
    
    # --- BOTTOM SUBPLOT: Forecast Period Comparison ---
    if not comparison_df.empty:
        # Plot actual values (test data) with prominent markers
        ax2.plot(comparison_df.index, comparison_df['actual'],
                 label=f'Actual (Test Data)', color='darkgreen', 
                 marker='o', markersize=10, linewidth=3, alpha=0.9)
        
        # Plot predicted values with different style
        ax2.plot(comparison_df.index, comparison_df['forecasted_value'],
                 label=f'Predicted', color='red', 
                 marker='s', markersize=8, linewidth=2, linestyle='--', alpha=0.9)
        
        # Plot confidence interval
        ax2.fill_between(comparison_df.index,
                        comparison_df['lower_bound'],
                        comparison_df['upper_bound'],
                        color='red', alpha=0.2, label='95% Confidence Interval')
        
        # Add data point annotations for better readability
        if len(comparison_df) <= 30:
            for idx, row in comparison_df.iterrows():
            # Actual value annotation
                ax2.annotate(f'{row["actual"]:.4f}', 
                            (idx, row['actual']), 
                            textcoords="offset points", 
                            xytext=(0,15), ha='center', fontsize=9, 
                            color='darkgreen', weight='bold')
            
            # Predicted value annotation
                ax2.annotate(f'{row["forecasted_value"]:.4f}', 
                            (idx, row['forecasted_value']), 
                            textcoords="offset points", 
                            xytext=(0,-20), ha='center', fontsize=9, 
                            color='red', weight='bold')
        else:
            logger.info(f"  Skipping data point annotations on plot due to too many points ({len(comparison_df)}).")
        
        ax2.set_title(f"Forecast Period: Actual vs Predicted ({int(forecast_interval_hours*60)}min intervals)")
        
        # Format x-axis for forecast period
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=int(forecast_interval_hours*60)))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Set y-axis limits for forecast subplot to focus on relevant range
        forecast_values = pd.concat([
            comparison_df['actual'], 
            comparison_df['forecasted_value'], 
            comparison_df['upper_bound']
        ]).dropna()
        
        if not forecast_values.empty and forecast_values.max() > 0:
            y_min = max(0, forecast_values.min() * 0.8)
            y_max = forecast_values.max() * 1.3
            ax2.set_ylim(y_min, y_max)
        else:
            ax2.set_ylim(0, 0.1)
    else:
        ax2.text(0.5, 0.5, 'No Forecast Data Available for Comparison', 
                ha='center', va='center', transform=ax2.transAxes, 
                fontsize=14, color='gray')
        ax2.set_title("Forecast Period: Actual vs Predicted")
    
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Value")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')
    
    # Add metrics text box to the forecast subplot
    if metrics.get('points', 0) > 0:
        metrics_text = "\n".join([
            f"RMSE: {metrics['rmse']:.4f}",
            f"MAE: {metrics['mae']:.4f}",
            f"MAPE: {metrics['mape']:.1f}%",
            f"R²: {metrics['r2']:.4f}",
            f"Points: {metrics['points']}"
        ])
        ax2.text(0.98, 0.02, metrics_text,
                 transform=ax2.transAxes,
                 verticalalignment='bottom',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'),
                 fontsize=10)
    
    # --- IMPORTANT CHANGE FOR FILENAME GENERATION AND TITLE ---
    # If start_time_for_filename and end_time_for_filename are provided, use them for the filename
    # Otherwise, fall back to the original pred_time for the filename
    if start_time_for_filename and end_time_for_filename:
        filename_suffix = f"_filtered_{start_time_for_filename.strftime('%Y%m%d_%H%M%S')}_{end_time_for_filename.strftime('%Y%m%d_%H%M%S')}.png"
        title_pred_time_info = f"Filtered from {start_time_for_filename.strftime('%Y-%m-%d %H:%M:%S')} to {end_time_for_filename.strftime('%Y-%m-%d %H:%M:%S')}"
    else:
        filename_suffix = f"_{pred_time.strftime('%Y%m%d_%H%M%S')}_accuracy.png"
        title_pred_time_info = f"Prediction made at {pred_time.strftime('%Y-%m-%d %H:%M:%S')}"

    filename = f"{host.replace(' ', '_')}_{metric_friendly_name.replace('[', '_').replace(']', '').replace(',', '_').replace(':', '_').replace('/', '_')}{filename_suffix}"
    plot_path = os.path.join(ACCURACY_PLOTS_DIR, filename) # ACCURACY_PLOTS_DIR is defined in accuracy_checker.py itself
    
    # Main title for entire figure - adjust to use the new title_pred_time_info
    fig.suptitle(f"Forecast Analysis: {host} - {metric_friendly_name}\n"
                 f"{title_pred_time_info}", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Make room for main title
    
    # Save plot
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved improved accuracy plot: {filename}")


def save_comparison_results(host, metric_friendly_name, pred_time, comparison_df, metrics):
    """
    Save detailed comparison results (actual, predicted, bounds, metrics) to a JSON file.

    Args:
        host (str): The host name.
        metric_friendly_name (str): The user-friendly name of the metric.
        pred_time (datetime): The timestamp when the prediction was made.
        comparison_df (pd.DataFrame): DataFrame containing actual, forecasted, lower_bound, and upper_bound for the forecast period.
        metrics (dict): Dictionary of accuracy metrics.
    """
    # Create a copy to avoid modifying the original DataFrame passed into the function
    df_to_serialize = comparison_df.copy()

    # Convert the Timestamp index to string format (ISO 8601 is recommended for JSON)
    # Check if the index is already a DatetimeIndex before mapping
    if isinstance(df_to_serialize.index, pd.DatetimeIndex):
        df_to_serialize.index = df_to_serialize.index.map(lambda x: x.isoformat())

    # Convert DataFrame to a list of dictionaries, including the now-string-formatted index
    comparison_data = df_to_serialize.reset_index().to_dict('records')

    results = {
        'host': host,
        'metric': metric_friendly_name,
        'prediction_time': pred_time.isoformat(), # This was already correctly serialized
        'metrics': metrics,
        'comparison_data': comparison_data
    }
    
    # Sanitize filename
    filename = f"{host.replace(' ', '_')}_{metric_friendly_name.replace('[', '_').replace(']', '').replace(',', '_').replace(':', '_').replace('/', '_')}_{pred_time.strftime('%Y%m%d_%H%M%S')}_results.json"
    with open(os.path.join(COMPARISON_RESULTS_DIR, filename), 'w') as f:
        json.dump(results, f, indent=2) # Use json.dump for proper JSON formatting
    logger.info(f"Saved comparison results: {filename}")


if __name__ == "__main__":
    logger.info("Starting accuracy check and plotting process...")
    check_and_plot_accuracy()
    logger.info("Accuracy check and plotting process completed.")