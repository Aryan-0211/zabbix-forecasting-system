import pandas as pd
from prophet import Prophet
import pickle
import os
import numpy as np
from datetime import datetime, timedelta
import warnings
# Removed boxcox and inv_boxcox1p imports as they are no longer used

warnings.filterwarnings('ignore')

class ProphetForecaster:
    def __init__(self, model_path="models/"):
        self.model_path = model_path
        os.makedirs(model_path, exist_ok=True)
        self.debug = True
        # self.lambda_ = None # Removed as Box-Cox is no longer used

    def _log(self, message):
        if self.debug:
            print(f"[ProphetForecaster] {message}")

    def _preprocess_and_transform(self, series, original_interval_minutes):
        """
        Preprocessing pipeline for Prophet:
        1. Drop NaNs.
        2. Downsample to the target forecast interval (2 minutes) if original data is finer.
        3. Apply robust outlier handling (winsorization).
        4. Convert to Prophet's required DataFrame format ('ds', 'y').
        """
        self._log(f"Original series length: {len(series)}")
        series_clean = series.dropna()
        self._log(f"After removing NaN: {len(series_clean)}")
        
        if len(series_clean) < 50:
            raise ValueError(f"Insufficient data: {len(series_clean)} points (minimum 50 required for Prophet).")
        
        if original_interval_minutes < 2:
            temp_series_for_resample = pd.Series(series_clean.values, index=pd.to_datetime(series_clean.index))
            series_resampled = temp_series_for_resample.resample('2T').mean().dropna()
            self._log(f"Downsampled from {original_interval_minutes}-min to 2-min intervals. New length: {len(series_resampled)}")
            series_clean = series_resampled
        
        if series_clean.empty:
            raise ValueError("Series is empty after downsampling and NaN removal.")

        q_high = series_clean.quantile(0.98)
        q_low = series_clean.quantile(0.02)
        series_clean = series_clean.clip(lower=q_low, upper=q_high)
        self._log(f"After outlier clipping (2nd-98th percentile): {series_clean.min():.2f} to {series_clean.max():.2f}")
        
        # Using original winsorized values directly, no Box-Cox
        df_prophet = pd.DataFrame({
            'ds': series_clean.index,
            'y': np.maximum(series_clean.values, 0.0)
        })
        df_prophet['y_lower'] = 0.0
        df_prophet['y_upper'] = 12.0
        
        
        self._log(f"Data range (raw, winsorized, downsampled): {df_prophet['y'].min():.2f} to {df_prophet['y'].max():.2f}")
        return df_prophet

    def train(self, series, model_file_path, data_interval_minutes=2):
        """
        Train a Prophet model.
        """
        self._log(f"Training Prophet model for {model_file_path}")
        df_prophet = self._preprocess_and_transform(series, data_interval_minutes)

        if df_prophet.empty:
            raise ValueError("Processed data is empty, cannot train Prophet model.")

        model = Prophet(
            seasonality_mode='additive',
            daily_seasonality=False,
            weekly_seasonality=True, # Keep this enabled for more general patterns
            yearly_seasonality=False,
            seasonality_prior_scale=15.0,
            changepoint_prior_scale=0.05
        )
        model.add_seasonality(name='daily_custom', period=1.0, fourier_order=20, prior_scale=15.0) # Period in days
        
        
        # Crucial for non-negative forecasts with Prophet
        #df_prophet['y_lower'] = 0.0
        
        model.fit(df_prophet)
        self._log("Prophet model training completed successfully.")

        # Store model with metadata (updated for no Box-Cox)
        model_data = {
            'model': model,
            'trained_at': datetime.now(),
            'data_points': len(df_prophet),
            'preprocessing': 'resample_winsorize_prophet_format', # Updated description
            'boxcox_lambda': None # Explicitly None as Box-Cox is not used
        }
        
        with open(model_file_path, "wb") as f:
            pickle.dump(model_data, f)
            
        self._log(f"Model saved to {model_file_path}")
        return model

    def forecast(self, model, steps=12, forecast_interval_minutes=2):
        """
        Generate a forecast with confidence intervals using Prophet.
        No inverse Box-Cox transformation needed.
        """
        self._log(f"Generating Prophet forecast for {steps} steps with {forecast_interval_minutes} min interval")
        
        future = model.make_future_dataframe(periods=steps, freq=f'{forecast_interval_minutes}min')
        
        # Crucial for non-negative forecasts with Prophet
        future['y_lower'] = 0.0
        future['y_upper'] = 12.0 

        forecast_df = model.predict(future)
        forecast_future_df = forecast_df.tail(steps)

        forecast = forecast_future_df['yhat'].values # No longer transformed, direct forecast
        lower_bound = forecast_future_df['yhat_lower'].values
        upper_bound = forecast_future_df['yhat_upper'].values
        
        self._log(f"Prophet raw 'yhat' (UNTRANSFORMED forecast) statistics:") # Removed "DIAGNOSTIC"
        self._log(f"  Min: {forecast.min():.4f}")
        self._log(f"  Max: {forecast.max():.4f}")
        self._log(f"  Mean: {forecast.mean():.4f}")

        # Ensure non-negative values (still important for Prophet's yhat_lower)
        forecast = np.maximum(forecast, 0)
        lower_bound = np.maximum(lower_bound, 0)
        upper_bound = np.maximum(upper_bound, 0)

        # Replace any remaining NaNs in the final output arrays
        forecast = np.nan_to_num(forecast, nan=0.0)
        lower_bound = np.nan_to_num(lower_bound, nan=0.0)
        upper_bound = np.nan_to_num(upper_bound, nan=0.0)

        result = {
            'forecast': forecast.tolist(),
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'timestamps': [ts.isoformat() for ts in forecast_future_df['ds']]
        }
        
        self._log(f"Prophet forecast generated: min={forecast.min():.2f}, max={forecast.max():.2f}") # Removed "DIAGNOSTIC"
        return result

    def load_model(self, model_file_path):
        """Load a previously trained Prophet model."""
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"No model found at {model_file_path}")
        with open(model_file_path, "rb") as f:
            model_data = pickle.load(f)
        
        # lambda_ is no longer used, but if it existed in old models, it will be loaded as None
        self._log("Box-Cox lambda not applicable for current Prophet configuration.")

        self._log(f"Loaded Prophet model from {model_file_path}, trained at {model_data['trained_at']}")
        return model_data['model']

    def get_model_info(self, model_file_path):
        """Get information about a trained Prophet model."""
        if not os.path.exists(model_file_path):
            return None
        with open(model_file_path, "rb") as f:
            model_data = pickle.load(f)
        
        model_type = "Prophet"
        
        return {
            'model_type': model_type,
            'trained_at': model_data['trained_at'].isoformat(),
            'data_points': model_data['data_points'],
            'preprocessing': model_data['preprocessing'],
            'boxcox_lambda': model_data.get('boxcox_lambda'), # This will now consistently be None for new models
            'daily_seasonality': model_data['model'].daily_seasonality if hasattr(model_data['model'], 'daily_seasonality') else 'N/A',
            'weekly_seasonality': model_data['model'].weekly_seasonality if hasattr(model_data['model'], 'weekly_seasonality') else 'N/A',
            'yearly_seasonality': model_data['model'].yearly_seasonality if hasattr(model_data['model'], 'yearly_seasonality') else 'N/A',
            'seasonality_mode': model_data['model'].seasonality_mode if hasattr(model_data['model'], 'seasonality_mode') else 'N/A'
        }