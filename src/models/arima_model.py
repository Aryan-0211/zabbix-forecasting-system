import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from scipy.stats import boxcox
from scipy.special import inv_boxcox1p # For inverse transformation

# For timeout mechanism (no longer used for auto_arima but kept for structure)
import signal
from contextlib import contextmanager

# For ExponentialSmoothing model
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# Import the result class for ExponentialSmoothing for robust type checking
from statsmodels.tsa.holtwinters.results import HoltWintersResultsWrapper


warnings.filterwarnings('ignore')

# --- Timeout Context Manager (kept for potential future use or other models) ---
class TimeoutException(Exception):
    """Custom exception raised when a block of code times out."""
    pass

@contextmanager
def time_limit(seconds):
    """
    Context manager to enforce a time limit on a block of code.
    Raises TimeoutException if the code takes longer than 'seconds'.
    """
    def signal_handler(signum, frame):
        raise TimeoutException(f"Operation timed out after {seconds} seconds!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds) # Start the alarm
    try:
        yield
    finally:
        signal.alarm(0) # Disable the alarm after the block exits or an exception occurs

class ARIMAForecaster:
    def __init__(self, model_path="models/"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.debug = True
        self.lambda_ = None # To store lambda value from Box-Cox transformation

    def _log(self, message):
        if self.debug:
            print(f"[ARIMAForecaster] {message}")

    def _preprocess_and_downsample(self, series, original_interval_minutes):
        """
        Enhanced preprocessing pipeline:
        1. Drop NaNs.
        2. Downsample to the target forecast interval (2 minutes) if original data is finer.
        3. Apply robust outlier handling (winsorization).
        4. Apply Box-Cox transformation to stabilize variance.
        """
        self._log(f"Original series length: {len(series)}")
        series_clean = series.dropna()
        self._log(f"After removing NaN: {len(series_clean)}")
        
        if len(series_clean) < 50:
            raise ValueError(f"Insufficient data: {len(series_clean)} points (minimum 50 required)")
        
        # 1. Downsample to target forecast interval (2 minutes) if original is finer
        if original_interval_minutes < 2:
            temp_series_for_resample = pd.Series(series_clean.values, index=pd.to_datetime(series_clean.index))
            series_resampled = temp_series_for_resample.resample('2T').mean().dropna()
            self._log(f"Downsampled from {original_interval_minutes}-min to 2-min intervals. New length: {len(series_resampled)}")
            series_clean = series_resampled
        
        if series_clean.empty:
            raise ValueError("Series is empty after downsampling and NaN removal.")

        # 2. Robust outlier handling (winsorization using percentiles)
        q_high = series_clean.quantile(0.98)
        q_low = series_clean.quantile(0.02)
        series_clean = series_clean.clip(lower=q_low, upper=q_high)
        self._log(f"After outlier clipping (2nd-98th percentile): {series_clean.min():.2f} to {series_clean.max():.2f}")
        
        # 3. Stabilize variance with Box-Cox transformation
        series_transformed, self.lambda_ = boxcox(series_clean + 1e-6)
        series_clean = pd.Series(series_transformed, index=series_clean.index)
        
        self._log(f"Data range after Box-Cox transformation: {series_clean.min():.2f} to {series_clean.max():.2f}, Lambda: {self.lambda_:.4f}")
        return series_clean

    # _determine_auto_arima_seasonal_period is kept for reference but no longer directly called in train
    def _determine_auto_arima_seasonal_period(self, n_points, current_interval_minutes):
        """Determines a suitable seasonal period for auto_arima based on simplified logic."""
        daily_points = (24 * 60) // current_interval_minutes
        self._log(f"Seasonality logic: Current interval: {current_interval_minutes} min, Calculated daily points: {daily_points}")

        if n_points < daily_points * 2:
            hourly_points = int(60 / current_interval_minutes)
            if n_points >= 2 * hourly_points:
                self._log(f"Seasonality logic: Limited data, using hourly seasonal period: {hourly_points}")
                return hourly_points
            else:
                self._log(f"Seasonality logic: Very limited data, using a small fixed period (12).")
                return 12
        
        for target_m in [72, 48, 36, 24]:
            if n_points >= 2 * target_m:
                 self._log(f"Seasonality logic: Sufficient data, using target seasonal period: {target_m}")
                 return target_m
        
        hourly_points_fallback = int(60 / current_interval_minutes)
        if n_points >= 2 * hourly_points_fallback:
            self._log(f"Seasonality logic: Fallback to hourly seasonal period: {hourly_points_fallback}")
            return hourly_points_fallback
        
        self._log(f"Seasonality logic: Final fallback to default seasonal period: 12")
        return 12

    def _train_exponential_smoothing(self, series): # Renamed to be specific for ES
        """
        Trains an Exponential Smoothing model.
        Returns a fitted ExponentialSmoothing model object.
        """
        self._log("Attempting to train ExponentialSmoothing model.")
        try:
            # Use 'add' for trend and seasonality, as CPU load often has additive patterns
            # seasonal_periods: default to 24 for daily pattern if data is hourly, or use a reasonable m
            # We determine seasonal_periods based on available data and likely patterns
            seasonal_periods_es = min(len(series)//2, 72) if len(series) > 10 else 1 # max 72 (2.4 hours at 2-min interval)
            
            es_model = ExponentialSmoothing(
                series,
                seasonal_periods=seasonal_periods_es,
                trend='add',
                seasonal='add',
                initialization_method="estimated" # Use estimated initial values
            ).fit()
            self._log("Successfully trained ExponentialSmoothing model.")
            return es_model
        except Exception as es_e:
            self._log(f"ExponentialSmoothing model training failed: {es_e}. This will lead to SimpleMeanPredictor fallback.", exc_info=True)
            # Raise the exception so it's caught by the outer train method, which then falls back
            raise

    def _train_simple_mean_predictor(self, series): # New method for Simple Mean
        """
        Trains a Simple Mean Predictor as a last resort fallback.
        Returns a dummy object that can predict based on a simple average.
        """
        self._log("Attempting to train SimpleMeanPredictor (last resort fallback).")
        class SimpleMeanPredictor:
            def __init__(self, series_data, arima_forecaster_lambda):
                self.series_data = series_data
                self.mean_val = series_data.mean()
                self.last_window_mean = series_data.tail(24).mean() if len(series_data) >= 24 else series_data.mean()
                self.lambda_ = arima_forecaster_lambda
            
            def predict(self, n_periods, return_conf_int=False):
                forecast_vals = np.full(n_periods, self.last_window_mean if not np.isnan(self.last_window_mean) else self.mean_val)
                if return_conf_int:
                    std_dev = np.std(self.series_data)
                    if np.isnan(std_dev) or std_dev == 0 or len(self.series_data) <= 1:
                        std_dev = np.mean(forecast_vals) * 0.05 if np.mean(forecast_vals) > 0 else 0.05
                    lower = np.maximum(forecast_vals - 1.96 * std_dev, 0)
                    upper = forecast_vals + 1.96 * std_dev
                    return forecast_vals, np.array(list(zip(lower, upper)))
                return forecast_vals
            
        return SimpleMeanPredictor(series, self.lambda_)

    def train(self, series, model_file_path, data_interval_minutes=2): 
        """
        Train a forecasting model. This version will prioritize ExponentialSmoothing for stability.
        """
        series_processed = None
        model = None
        
        try:
            self._log(f"Training forecasting model for {model_file_path}")
            
            series_processed = self._preprocess_and_downsample(series, data_interval_minutes) 
            
            # --- MODIFICATION START: Explicitly try ES first, then SimpleMean ---
            # Bypassing auto_arima entirely due to resource exhaustion causing hangs.
            self._log("Bypassing auto_arima due to resource constraints. Attempting ExponentialSmoothing.")
            try:
                model = self._train_exponential_smoothing(series_processed)
            except Exception as es_e:
                self._log(f"ExponentialSmoothing failed to train: {es_e}. Falling back to SimpleMeanPredictor.", exc_info=True)
                model = self._train_simple_mean_predictor(series_processed) # Fallback to SimpleMean
            # --- MODIFICATION END ---
            
            if model is None:
                raise ValueError("Model training failed and no model could be trained.")

            # Store model with metadata, including the Box-Cox lambda_
            model_data = {
                'model': model, # This could be ES or SimpleMeanPredictor
                'trained_at': datetime.now(),
                'data_points': len(series_processed), # Store length of processed data
                'seasonal_period': None, # No auto_arima, so seasonal_period is not explicitly found this way
                'seasonal_enabled': True if isinstance(model, ExponentialSmoothing) and model.seasonal_periods > 1 else False, # Reflect ES seasonality
                'original_data_interval_minutes': data_interval_minutes, # Store original interval for context
                'processed_data_interval_minutes': 2, # Processed data is always 2-min interval
                'preprocessing': 'resample_winsorize_boxcox',
                'boxcox_lambda': self.lambda_ # Store lambda for inverse transformation
            }
            
            with open(model_file_path, "wb") as f:
                pickle.dump(model_data, f)
                
            self._log(f"Model saved to {model_file_path}")
            if isinstance(model, ExponentialSmoothing) or isinstance(model, HoltWintersResultsWrapper):
                self._log(f"Final model type: ExponentialSmoothing (Primary Model Selected)")
            elif hasattr(model, 'predict') and hasattr(model, 'lambda_'):
                self._log(f"Final model type: SimpleMeanPredictor (Fallback Model Selected)")
            else: # Should not happen with current logic
                self._log(f"Final model type: Unknown")
            
            return model

        except Exception as e: # This handles exceptions *outside* the specific model training attempts
            self._log(f"Overall training failed: {str(e)}", exc_info=True)
            raise

    def forecast(self, model, steps=12, forecast_interval_minutes=2): 
        """
        Generate a forecast with confidence intervals.
        Includes inverse Box-Cox transformation to return to original scale.
        """
        try:
            self._log(f"Generating forecast for {steps} steps with {forecast_interval_minutes} min interval")
            
            # Initialize variables
            forecast_transformed = None
            conf_int_transformed = None
            current_lambda = self.lambda_ if self.lambda_ is not None else 1 # Default lambda to 1 if not set

            if isinstance(model, pm.arima.arima.ARIMA): # Kept for backward compatibility if an ARIMA model is loaded
                forecast_transformed, conf_int_transformed = model.predict(n_periods=steps, return_conf_int=True)
            elif isinstance(model, ExponentialSmoothing) or isinstance(model, HoltWintersResultsWrapper):
                start_index = len(model.data.endog)
                end_index = start_index + steps - 1
                forecast_transformed = model.predict(start=start_index, end=end_index)

                es_series_for_std = model.data.endog

                # --- FIX: Ensure std_dev_transformed is not NaN/0 in edge cases, and handle very small values for CI ---
                std_dev_transformed = np.std(es_series_for_std)
                
                # Robust fallback for std_dev calculation if series is constant or too short
                if np.isnan(std_dev_transformed) or std_dev_transformed == 0 or len(es_series_for_std) <= 1:
                    # Use a small percentage of the mean of the forecast itself, or a fixed small value
                    mean_forecast_transformed = np.mean(forecast_transformed) if not np.isnan(np.mean(forecast_transformed)) else 0.0
                    std_dev_transformed = max(mean_forecast_transformed * 0.1, 0.05) # Ensure a minimum positive std dev
                # --- END FIX ---
                
                z_score_95_ci = 1.96 # For 95% CI
                conf_int_lower_transformed = forecast_transformed - z_score_95_ci * std_dev_transformed
                conf_int_upper_transformed = forecast_transformed + z_score_95_ci * std_dev_transformed
                conf_int_transformed = np.array(list(zip(conf_int_lower_transformed, conf_int_upper_transformed)))

            elif hasattr(model, 'predict') and hasattr(model, 'lambda_'): # This indicates SimpleMeanPredictor fallback
                forecast_transformed, conf_int_transformed = model.predict(n_periods=steps, return_conf_int=True)
                current_lambda = model.lambda_ # SimpleMeanPredictor carries the lambda
            else:
                raise TypeError(f"Unsupported model type for forecasting: {type(model)}. Cannot predict.")

            # Apply inverse Box-Cox transformation
            # CRITICAL FIX: Replace NaN values BEFORE inverse Box-Cox if they somehow creep in.
            forecast_transformed = np.nan_to_num(forecast_transformed, nan=0.0) # Replace NaNs with 0.0
            
            # Ensure conf_int_transformed is also numeric. If it became NaN earlier, this is crucial.
            if conf_int_transformed is None or np.any(np.isnan(conf_int_transformed)):
                # If CI is still NaN, create a dummy one based on forecast itself
                std_dev_fallback = (np.mean(forecast_transformed) * 0.1) if np.mean(forecast_transformed) > 0 else 0.05
                conf_int_lower_transformed = np.maximum(forecast_transformed - z_score_95_ci * std_dev_fallback, 0)
                conf_int_upper_transformed = forecast_transformed + z_score_95_ci * std_dev_fallback
                conf_int_transformed = np.array(list(zip(conf_int_lower_transformed, conf_int_upper_transformed)))
            
            forecast = inv_boxcox1p(forecast_transformed, current_lambda) 
            conf_int = inv_boxcox1p(conf_int_transformed, current_lambda) 
            
            # Ensure non-negative values, as CPU load cannot be negative
            forecast = np.maximum(forecast, 0)
            conf_int = np.maximum(conf_int, 0) # This will clamp any negative bounds to 0

            # CRITICAL FIX: Replace any remaining NaNs in the final output arrays
            # This is a last-resort cleanup before returning to avoid 'nan' strings in CSV.
            forecast = np.nan_to_num(forecast, nan=0.0)
            conf_int[:, 0] = np.nan_to_num(conf_int[:, 0], nan=0.0)
            conf_int[:, 1] = np.nan_to_num(conf_int[:, 1], nan=0.0)
            
            result = {
                'forecast': forecast.tolist(),
                'lower_bound': conf_int[:, 0].tolist(),
                'upper_bound': conf_int[:, 1].tolist(),
                'timestamps': self._generate_forecast_timestamps(steps, interval_minutes=forecast_interval_minutes) 
            }
            
            self._log(f"Forecast generated: min={forecast.min():.2f}, max={forecast.max():.2f}")
            return result

        except Exception as e:
            self._log(f"Forecasting failed: {str(e)}")
            raise

    def _generate_forecast_timestamps(self, steps, interval_minutes): 
        """Generate timestamps for forecast periods."""
        now = pd.Timestamp.now()
        # Round up to the next clean interval mark for consistent forecast start times
        next_interval_mark = (now + pd.Timedelta(minutes=interval_minutes)).floor(f'{interval_minutes}min')
        start_time = next_interval_mark.to_pydatetime()

        return [
            (start_time + timedelta(minutes=i*interval_minutes)).isoformat() 
            for i in range(steps)
        ]

    def load_model(self, model_file_path): 
        """Load a previously trained model."""
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"No model found at {model_file_path}")
        with open(model_file_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Load lambda from saved model data when loading the model
        if 'boxcox_lambda' in model_data:
            self.lambda_ = model_data['boxcox_lambda']
            self._log(f"Loaded Box-Cox lambda: {self.lambda_}")
        else:
            self.lambda_ = None # Reset if not found (e.g., older models)
            self._log("Warning: No Box-Cox lambda found in loaded model data.")

        self._log(f"Loaded model from {model_file_path}, trained at {model_data['trained_at']}")
        return model_data['model']

    def get_model_info(self, model_file_path): 
        """Get information about a trained model."""
        if not os.path.exists(model_file_path):
            return None
        with open(model_file_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Ensure series_processed is available if accessing data_points
        # This part might need the series to be passed if not globally available
        data_points_info = model_data['data_points'] if 'data_points' in model_data else 'N/A'
        
        return {
            'trained_at': model_data['trained_at'].isoformat(),
            'data_points': data_points_info, 
            'seasonal_period': model_data.get('seasonal_period', 'N/A'), # Use .get() for robustness
            'seasonal_enabled': model_data.get('seasonal_enabled', False),
            'original_data_interval_minutes': model_data.get('original_data_interval_minutes', 2),
            'processed_data_interval_minutes': model_data.get('processed_data_interval_minutes', 2),
            'preprocessing': model_data.get('preprocessing', 'unknown'),
            'boxcox_lambda': model_data.get('boxcox_lambda'),
            'model_order': str(model_data['model'].order) if hasattr(model_data['model'], 'order') else 'N/A',
            'seasonal_order': str(model_data['model'].seasonal_order) if hasattr(model_data['model'], 'seasonal_order') else 'N/A'
        }
