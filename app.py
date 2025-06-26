from flask import Flask, jsonify, send_from_directory, request, render_template, redirect, url_for
from src.utils.zabbix_connector import ZabbixConnector
from src.models.arima_model import ARIMAForecaster
# --- NEW IMPORT FOR PROPHET ---
from src.models.prophet_model import ProphetForecaster
# --- END NEW IMPORT ---
import pandas as pd
import yaml
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from datetime import datetime, timedelta
import logging
import csv
import numpy as np 

# Import necessary functions from accuracy_checker.py
from accuracy_checker import align_timestamps_with_predictions, calculate_metrics, generate_comparison_plot

# Load environment variables at the very beginning of the script
load_dotenv()

logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
logger = logging.getLogger(__name__)

# --- Define Base Directories for Direct VM Execution ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
COMPARISON_RESULTS_BASE_DIR = os.path.join(BASE_DIR, 'comparison_results')

# Define specific file paths using base directories
PREDICTIONS_FILE = os.path.join(BASE_DIR, "predictions_log.csv")
ACCURACY_PLOTS_DIR = os.path.join(STATIC_DIR, 'accuracy_plots')
COMPARISON_RESULTS_DIR = os.path.join(COMPARISON_RESULTS_BASE_DIR, 'json_reports')

# Ensure all necessary output directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(COMPARISON_RESULTS_BASE_DIR, exist_ok=True)
os.makedirs(ACCURACY_PLOTS_DIR, exist_ok=True)
os.makedirs(COMPARISON_RESULTS_DIR, exist_ok=True)

# --- DATA INTERVAL CONFIGURATION ---
ZABBIX_DATA_INTERVAL_MINUTES = 0.5
FORECAST_INTERVAL_MINUTES = 2
HISTORICAL_DATA_HOURS = 48

# --- MODEL SELECTION CONFIGURATION ---
# Add a new environment variable or config setting to choose the model
# For now, let's use a default here and ensure it's loaded from .env for production
# Options: "ARIMA_ES" (for current Exponential Smoothing fallback) or "PROPHET"
FORECASTING_MODEL_TYPE = os.getenv("FORECASTING_MODEL_TYPE", "PROPHET").upper() 
# Defaulting to PROPHET as that's the desired new direction

# --- Initialize components based on FORECASTING_MODEL_TYPE ---
connector = ZabbixConnector()
forecaster = None # Initialize as None, will be set below
current_model_instance_name = ""

if FORECASTING_MODEL_TYPE == "PROPHET":
    forecaster = ProphetForecaster()
    current_model_instance_name = "ProphetForecaster"
    logger.info("Initialized ProphetForecaster.")
elif FORECASTING_MODEL_TYPE == "ARIMA_ES":
    forecaster = ARIMAForecaster()
    current_model_instance_name = "ARIMAForecaster (Exponential Smoothing Fallback)"
    logger.info("Initialized ARIMAForecaster (Exponential Smoothing Fallback).")
else:
    logger.error(f"Invalid FORECASTING_MODEL_TYPE specified: {FORECASTING_MODEL_TYPE}. Defaulting to Prophet.")
    forecaster = ProphetForecaster()
    current_model_instance_name = "ProphetForecaster (Default Fallback)"

# --- END MODEL SELECTION CONFIGURATION ---

# Load configuration
try:
    with open(os.path.join(BASE_DIR, "configs", "zabbix_config.yaml")) as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error("zabbix_config.yaml not found. Please ensure it's in the configs/ directory.")
    config = {"hosts": {}}
except yaml.YAMLError as e:
    logger.error(f"Error parsing zabbix_config.yaml: {e}")
    config = {"hosts": {}}

# Define prediction storage file headers
PREDICTIONS_HEADERS = ['prediction_time', 'host', 'metric', 'item_key', 'forecast_timestamp', 'forecasted_value', 'lower_bound', 'upper_bound', 'forecast_interval_hours', 'horizon_hours', 'model_type'] # ADDED 'model_type'

# Initialize predictions file with headers if it doesn't exist
if not os.path.exists(PREDICTIONS_FILE):
    try:
        with open(PREDICTIONS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(PREDICTIONS_HEADERS)
        logger.info(f"Initialized {PREDICTIONS_FILE} with headers.")
    except IOError as e:
        logger.error(f"Error initializing {PREDICTIONS_FILE}: {e}")

# Global config for plot (used in get_filtered_graph)
HISTORY_WINDOW_HOURS_FOR_PLOT = 6

@app.route('/')
def home():
    """
    Redirects the root URL to the GUI input form page.
    """
    return redirect(url_for('view_graph_gui'))

@app.route('/static/<path:filename>')
def serve_static(filename):
    """
    Serves static files (e.g., generated PNG plots) from the STATIC_DIR.
    """
    return send_from_directory(STATIC_DIR, filename)

@app.route('/forecast/<host_name>/<metric_friendly_name>/<int:steps_2min>')
def generate_forecast(host_name, metric_friendly_name, steps_2min):
    """
    Endpoint to generate and store forecasts with proper data interval handling.
    Fetches historical data from Zabbix, trains the selected model, generates a forecast,
    stores predictions, and creates a static plot.
    """
    try:
        if host_name not in config.get("hosts", {}):
            return jsonify({"error": f"Invalid host: {host_name}"}), 400
        if metric_friendly_name not in config.get("hosts", {}).get(host_name, {}).get("metrics", {}):
            return jsonify({"error": f"Invalid metric: {metric_friendly_name} for host {host_name}"}), 400
            
        metric_config = config["hosts"][host_name]["metrics"][metric_friendly_name]
        item_key = metric_config["key"]

        item_id = connector.get_item_id(host_name, item_key)
        if not item_id:
            return jsonify({"error": f"Could not find Zabbix item ID for {host_name} - {item_key}"}), 404
        
        app.logger.info(f"Fetching {HISTORICAL_DATA_HOURS} hours of historical data (interval: {ZABBIX_DATA_INTERVAL_MINUTES} min)")
        
        historical_data = connector.fetch_metric(
            itemid=item_id,
            hours=HISTORICAL_DATA_HOURS
        )
        
        if historical_data.empty:
            app.logger.warning(f"No historical data available for {host_name} - {metric_friendly_name}. Check Zabbix data collection.")
            return jsonify({"error": "No historical data available to train the model"}), 400
        
        app.logger.info(f"Retrieved {len(historical_data)} data points for training")
            
        series = historical_data.set_index("timestamp")["value"]
        series.index = pd.to_datetime(series.index)

        # --- DYNAMIC MODEL PATH FOR SAVING ---
        model_name_base = f"{host_name}_{metric_friendly_name.replace('[', '_').replace(']', '').replace(',', '_')}_prod_2min"
        # Adjust model file extension based on the selected forecaster type
        model_extension = "_prophet.pkl" if FORECASTING_MODEL_TYPE == "PROPHET" else "_arima.pkl"
        model_full_path = os.path.join(MODELS_DIR, model_name_base + model_extension)
        # --- END DYNAMIC MODEL PATH ---
        
        app.logger.info(f"Training model with data_interval_minutes={ZABBIX_DATA_INTERVAL_MINUTES} using {current_model_instance_name}")
        model = forecaster.train(
            series, 
            model_full_path, 
            data_interval_minutes=ZABBIX_DATA_INTERVAL_MINUTES
        )
        
        FORECAST_INTERVAL_HOURS = FORECAST_INTERVAL_MINUTES / 60
        FORECAST_HORIZON_HOURS = steps_2min * FORECAST_INTERVAL_HOURS
        
        app.logger.info(f"Generating forecast: {steps_2min} steps for {FORECAST_HORIZON_HOURS:.2f} hours at {FORECAST_INTERVAL_MINUTES}-minute intervals")
        
        forecast_result = forecaster.forecast(
            model, 
            steps=steps_2min,
            forecast_interval_minutes=FORECAST_INTERVAL_MINUTES
        )
        
        # --- PASS MODEL_TYPE TO STORE_PREDICTIONS ---
        store_predictions(
            host_name, 
            metric_friendly_name, 
            item_key, 
            forecast_result, 
            steps_2min, 
            FORECAST_INTERVAL_HOURS, 
            FORECAST_HORIZON_HOURS,
            current_model_instance_name # Pass the model type here
        )
        # --- END PASS MODEL_TYPE ---
        
        plot_url = generate_forecast_plot(
            host_name, 
            metric_friendly_name, 
            series, 
            forecast_result, 
            FORECAST_HORIZON_HOURS, 
            FORECAST_INTERVAL_HOURS
        )
        
        return jsonify({
            "status": "success",
            "host": host_name,
            "metric": metric_friendly_name,
            "forecast": forecast_result['forecast'],
            "timestamps": forecast_result['timestamps'],
            "plot_url": plot_url,
            "model_info": {
                "historical_data_hours": HISTORICAL_DATA_HOURS,
                "data_points_used": len(series),
                "zabbix_data_interval_minutes": ZABBIX_DATA_INTERVAL_MINUTES,
                "forecast_interval_minutes": FORECAST_INTERVAL_MINUTES,
                "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
                "forecasting_model_type": current_model_instance_name # Also include here
            }
        })
        
    except Exception as e:
        app.logger.error(f"Forecast generation error for {host_name}/{metric_friendly_name}: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during forecast generation: {str(e)}"}), 500

# --- UPDATED STORE_PREDICTIONS SIGNATURE ---
def store_predictions(host, metric_friendly_name, item_key, forecast_result, steps_2min, forecast_interval_hours, forecast_horizon_hours, model_type):
# --- END UPDATED SIGNATURE ---
    """
    Stores predictions in the CSV file (predictions_log.csv).
    Stores numerical values as floats, not formatted strings, for better data integrity.
    """
    prediction_time_now = datetime.now()
    
    with open(PREDICTIONS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for i in range(len(forecast_result['forecast'])):
            row = [
                prediction_time_now.isoformat(),
                host,
                metric_friendly_name,
                item_key,
                forecast_result['timestamps'][i],
                forecast_result['forecast'][i],
                forecast_result['lower_bound'][i],
                forecast_result['upper_bound'][i],
                forecast_interval_hours,
                forecast_horizon_hours,
                model_type # ADDED: Store the model type used for this forecast
            ]
            writer.writerow(row)
    app.logger.info(f"Stored {len(forecast_result['forecast'])} predictions to {PREDICTIONS_FILE}")

def generate_forecast_plot(host, metric_friendly_name, historical_series, forecast_result, forecast_horizon_hours, forecast_interval_hours):
    """
    Generates and saves a static PNG plot of the historical data and the new forecast.
    The plot includes the last HISTORY_WINDOW_HOURS_FOR_PLOT hours of historical data for context.
    """
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    
    recent_historical = historical_series.tail(int(HISTORY_WINDOW_HOURS_FOR_PLOT * 60 / ZABBIX_DATA_INTERVAL_MINUTES))
    
    plt.plot(recent_historical.index, recent_historical.values,
            label=f'Recent Historical Data ({HISTORY_WINDOW_HOURS_FOR_PLOT}h)', color='blue', alpha=0.7)
    
    forecast_dates = pd.to_datetime(forecast_result['timestamps'])
    plt.plot(forecast_dates, forecast_result['forecast'],
            label=f'Forecast (Next {int(forecast_horizon_hours*60)} min - {FORECAST_INTERVAL_MINUTES}min intervals)',
            color='red', marker='D', markersize=5)
    
    plt.fill_between(forecast_dates,
                    forecast_result['lower_bound'],
                    forecast_result['upper_bound'],
                    color='pink', alpha=0.3, label='95% CI')
    
    tick_interval_minutes = 30 if forecast_horizon_hours >= 0.5 else 15
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=tick_interval_minutes))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    
    all_values_for_ylim = pd.concat([
        recent_historical, 
        pd.Series(forecast_result['forecast']), 
        pd.Series(forecast_result['upper_bound'])
    ]).dropna()
    max_val = all_values_for_ylim.max() if not all_values_for_ylim.empty else 0.0
    plt.ylim(0, max(max_val * 1.2, 0.5))
    
    # --- ADD MODEL TYPE TO PLOT TITLE ---
    plt.title(f"{host} - {metric_friendly_name} Forecast\n"
              f"Model: {current_model_instance_name}\n" # Added model type
              f"Historical Data Interval: {ZABBIX_DATA_INTERVAL_MINUTES}-min, Forecast Interval: {FORECAST_INTERVAL_MINUTES}-min")
    # --- END ADD MODEL TYPE TO PLOT TITLE ---
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"{host}_{metric_friendly_name.replace('[', '_').replace(']', '').replace(',', '_')}_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_path = os.path.join(ACCURACY_PLOTS_DIR, filename)
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    return f"/static/accuracy_plots/{filename}"

@app.route('/view_accuracy_graph/<host_name>/<metric_friendly_name>/<string:start_time_str>/<string:end_time_str>')
def get_filtered_graph(host_name, metric_friendly_name, start_time_str, end_time_str):
    """
    API Endpoint to provide raw data for a specific time-filtered accuracy graph.
    This endpoint returns JSON data for the interactive Plotly frontend.
    """
    try:
        start_time = datetime.fromisoformat(start_time_str)
        end_time = datetime.fromisoformat(end_time_str)

        if host_name not in config.get("hosts", {}):
            return jsonify({"error": f"Invalid host: {host_name}"}), 400
        if metric_friendly_name not in config.get("hosts", {}).get(host_name, {}).get("metrics", {}):
            return jsonify({"error": f"Invalid metric: {metric_friendly_name} for host {host_name}"}), 400
        
        metric_config = config["hosts"][host_name]["metrics"][metric_friendly_name]
        item_key = metric_config["key"]

        try:
            # --- UPDATED: Ensure 'model_type' column is loaded correctly ---
            predictions_df = pd.read_csv(PREDICTIONS_FILE, 
                                         parse_dates=['prediction_time', 'forecast_timestamp'],
                                         dtype={
                                             'forecasted_value': float,
                                             'lower_bound': float,
                                             'upper_bound': float,
                                             'forecast_interval_hours': float,
                                             'horizon_hours': float,
                                             'model_type': str # Explicitly define as string
                                         })
            # --- END UPDATED ---
        except pd.errors.EmptyDataError:
            app.logger.warning(f"Predictions log file '{PREDICTIONS_FILE}' is empty.")
            return jsonify({"error": "Predictions log file is empty. No data to display."}), 404
        except FileNotFoundError:
            app.logger.error(f"Predictions log file '{PREDICTIONS_FILE}' not found.")
            return jsonify({"error": "Predictions log file not found. Please ensure forecasts have been generated."}), 404
        except Exception as e:
            app.logger.error(f"Error reading predictions from {PREDICTIONS_FILE}: {e}", exc_info=True)
            return jsonify({"error": f"Failed to read predictions data: {str(e)}"}), 500

        # Filter predictions for the specific host and metric
        # No change needed here, as the filtering is by host and metric, not model type
        filtered_predictions = predictions_df[
            (predictions_df['host'] == host_name) &
            (predictions_df['metric'] == metric_friendly_name)
        ].sort_values(by='forecast_timestamp')

        if filtered_predictions.empty:
            app.logger.warning(f"No predictions found for {host_name} - {metric_friendly_name} in {PREDICTIONS_FILE}.")
            return jsonify({"error": "No predictions found for this host/metric in the log file."}), 404

        item_id = connector.get_item_id(host_name, item_key)
        if not item_id:
            return jsonify({"error": f"Could not find Zabbix item ID for {host_name} - {item_key}"}), 404

        overall_data_start = min(filtered_predictions['prediction_time'].min(), start_time) - timedelta(hours=HISTORY_WINDOW_HOURS_FOR_PLOT)
        overall_data_end = max(filtered_predictions['forecast_timestamp'].max(), end_time) + timedelta(minutes=FORECAST_INTERVAL_MINUTES)

        app.logger.info(f"Fetching actual data for {host_name} - {metric_friendly_name} from {overall_data_start} to {overall_data_end}")
        actual_df_raw = connector.fetch_metric(itemid=item_id, time_from=overall_data_start, time_till=overall_data_end)
        
        if 'timestamp' in actual_df_raw.columns:
            actual_df_raw['timestamp'] = pd.to_datetime(actual_df_raw['timestamp'])
            actual_series_raw = actual_df_raw.set_index('timestamp')['value']
        else:
            actual_series_raw = pd.Series([], dtype=float)

        if actual_series_raw.empty:
            app.logger.warning(f"No actual data retrieved from Zabbix for {host_name} - {metric_friendly_name} for the period {overall_data_start} to {overall_data_end}.")
            
            plot_df = filtered_predictions[
                (filtered_predictions['forecast_timestamp'] >= start_time) & # Use forecast_timestamp for filtering
                (filtered_predictions['forecast_timestamp'] <= end_time)
            ]
            # Ensure proper columns exist for graph_data, even if empty
            return jsonify({
                "status": "success",
                "host": host_name,
                "metric": metric_friendly_name,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "data_interval_info": {
                    "zabbix_data_interval_minutes": ZABBIX_DATA_INTERVAL_MINUTES,
                    "forecast_interval_minutes": FORECAST_INTERVAL_MINUTES,
                    "historical_data_hours": HISTORICAL_DATA_HOURS
                },
                "metrics_for_filtered_period": {
                    'rmse': None, 'mae': None, 'mape': None, 'r2': None, 'points': 0
                },
                "graph_data": {
                    "historical_timestamps": [],
                    "historical_values": [],
                    "actual_timestamps": [],
                    "actual_values": [],
                    "forecasted_values": plot_df['forecasted_value'].values.tolist() if not plot_df.empty else [],
                    "lower_bound": plot_df['lower_bound'].tolist() if not plot_df.empty else [],
                    "upper_bound": plot_df['upper_bound'].tolist() if not plot_df.empty else [],
                    "model_types": plot_df['model_type'].tolist() if not plot_df.empty else [] # Return model types for frontend
                }
            }), 200

        forecast_interval_hours = filtered_predictions['forecast_interval_hours'].iloc[0]

        full_aligned_actuals = align_timestamps_with_predictions(
            actual_series_raw,
            filtered_predictions['forecast_timestamp'],
            forecast_interval_hours
        )

        # --- UPDATED MERGE: Include 'model_type' ---
        full_comparison_df = pd.merge(
            full_aligned_actuals.to_frame(name='actual'),
            filtered_predictions.set_index('forecast_timestamp')[['forecasted_value', 'lower_bound', 'upper_bound', 'model_type']], # Include model_type
            left_index=True, right_index=True, how='inner'
        )
        # --- END UPDATED MERGE ---
        
        plot_df = full_comparison_df[
            (full_comparison_df.index >= start_time) &
            (full_comparison_df.index <= end_time)
        ].sort_index()

        historical_end_time_for_plot = plot_df.index.min() if not plot_df.empty else start_time
        plot_historical_df = actual_series_raw[
            (actual_series_raw.index >= historical_end_time_for_plot - timedelta(hours=HISTORY_WINDOW_HOURS_FOR_PLOT)) &
            (actual_series_raw.index < historical_end_time_for_plot)
        ]

        if plot_df.empty and plot_historical_df.empty:
            app.logger.warning(f"No data found for the specified time range {start_time} to {end_time} for plotting (neither forecast nor historical).")
            return jsonify({"error": "No data found for the specified time range."}), 404

        plot_metrics = calculate_metrics(
            plot_df['actual'], 
            plot_df['forecasted_value']
        ) if not plot_df.empty else {
            'rmse': np.nan, 'mae': np.nan, 'mape': np.nan, 'r2': np.nan, 'points': 0
        }
        plot_metrics_json_safe = {k: (v if not pd.isna(v) else None) for k,v in plot_metrics.items()}
        
        return jsonify({
            "status": "success",
            "host": host_name,
            "metric": metric_friendly_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "data_interval_info": {
                "zabbix_data_interval_minutes": ZABBIX_DATA_INTERVAL_MINUTES,
                "forecast_interval_minutes": FORECAST_INTERVAL_MINUTES,
                "historical_data_hours": HISTORICAL_DATA_HOURS
            },
            "metrics_for_filtered_period": plot_metrics_json_safe,
            "graph_data": {
                "historical_timestamps": plot_historical_df.index.strftime('%Y-%m-%dT%H:%M:%S').tolist() if not plot_historical_df.empty else [],
                "historical_values": plot_historical_df.values.tolist() if not plot_historical_df.empty else [],
                "actual_timestamps": plot_df.index.strftime('%Y-%m-%dT%H:%M:%S').tolist() if not plot_df.empty else [],
                "actual_values": plot_df['actual'].values.tolist() if not plot_df.empty else [],
                "forecasted_values": plot_df['forecasted_value'].values.tolist() if not plot_df.empty else [],
                "lower_bound": plot_df['lower_bound'].values.tolist() if not plot_df.empty else [],
                "upper_bound": plot_df['upper_bound'].values.tolist() if not plot_df.empty else [],
                "model_types": plot_df['model_type'].tolist() if not plot_df.empty else [] # ADDED: Return model types for frontend
            }
        }), 200

    except Exception as e:
        app.logger.error(f"Error fetching filtered graph data: {e}", exc_info=True)
        return jsonify({"error": f"Internal server error during data retrieval: {str(e)}"}), 500

@app.route('/view_graph')
def view_graph_gui():
    """Render the input form for interactive graph viewing"""
    error_message = request.args.get('error_message')

    now = datetime.now()
    default_end = now.replace(second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M')
    default_start = (now - timedelta(hours=1)).replace(second=0, microsecond=0).strftime('%Y-%m-%dT%H:%M')
    
    available_hosts = list(config.get("hosts", {}).keys())
    available_metrics = {host: list(config["hosts"][host]["metrics"].keys()) for host in available_hosts}

    return render_template('graph_input.html', 
                          default_start_time=default_start, 
                          default_end_time=default_end,
                          error_message=error_message,
                          data_interval_minutes=ZABBIX_DATA_INTERVAL_MINUTES,
                          forecast_interval_minutes=FORECAST_INTERVAL_MINUTES,
                          available_hosts=available_hosts,
                          available_metrics=available_metrics,
                          forecasting_model_type=current_model_instance_name # ADDED: Pass active model type to GUI
                         )

@app.route('/interactive_graph_display')
def interactive_graph_display():
    """
    Renders the HTML template that will display the interactive Plotly graph.
    It passes the selected parameters from the input form to the template.
    """
    host_name = request.args.get('host')
    metric_friendly_name = request.args.get('metric')
    start_time_str = request.args.get('start_time')
    end_time_str = request.args.get('end_time')

    if not all([host_name, metric_friendly_name, start_time_str, end_time_str]):
        return redirect(url_for('view_graph_gui', 
                               error_message="Please provide all required parameters (Host, Metric, Start Time, End Time)."))

    return render_template(
        'plotly_dashboard.html',
        host=host_name,
        metric=metric_friendly_name,
        start_time=start_time_str,
        end_time=end_time_str,
        data_interval_minutes=ZABBIX_DATA_INTERVAL_MINUTES,
        forecast_interval_minutes=FORECAST_INTERVAL_MINUTES,
        forecasting_model_type=current_model_instance_name # ADDED: Pass active model type to dashboard
    )

# Health check endpoint
@app.route('/health')
def health_check():
    """
    Health check endpoint to verify the application is running and to expose
    key configuration parameters.
    """
    return jsonify({
        "status": "healthy",
        "configuration": {
            "zabbix_data_interval_minutes": ZABBIX_DATA_INTERVAL_MINUTES,
            "forecast_interval_minutes": FORECAST_INTERVAL_MINUTES,
            "historical_data_hours": HISTORICAL_DATA_HOURS,
            "history_window_for_plots_hours": HISTORY_WINDOW_HOURS_FOR_PLOT,
            "predictions_file": PREDICTIONS_FILE,
            "models_directory": MODELS_DIR,
            "active_forecasting_model": current_model_instance_name # ADDED: Expose active model
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)