# type: ignore
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Weather Dashboard", page_icon="🌡️")

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    df = pd.read_csv("data/engineered_weather.csv", index_col='date', parse_dates=True)
    df = df.sort_index()
    return df

# ==================== LOAD MODELS ====================
@st.cache_resource
def load_models():
    models = {}
    
    # Linear Regression
    try:
        with open("models/linear_regression_model.pkl", "rb") as f:
            lr_data = pickle.load(f)
            models['Linear Regression'] = {
                'model': lr_data['model'],
                'features': lr_data['feature_cols'],
                'metrics': lr_data['test_metrics']
            }
    except Exception as e:
        st.warning(f"⚠️ Linear Regression not loaded: {e}")
    
    # SARIMAX
    try:
        with open("models/sarimax_model_optimized.pkl", "rb") as f:
            sarimax_data = pickle.load(f)
            models['SARIMAX'] = {
                'model': sarimax_data['model_fit'],
                'features': sarimax_data['exog_cols'],
                'metrics': sarimax_data['test_metrics']
            }
    except Exception as e:
        st.warning(f"⚠️ SARIMAX not loaded: {e}")
    
    # LSTM
    try:
        from tensorflow.keras.models import load_model as keras_load
        lstm_model = keras_load("models/lstm_model.h5", compile=False)
        with open("models/lstm_scalers.pkl", "rb") as f:
            scalers = pickle.load(f)
        models['LSTM'] = {
            'model': lstm_model,
            'X_scaler': scalers['X_scaler'],
            'y_scaler': scalers['y_scaler'],
            'lookback': scalers['lookback']
        }
    except Exception as e:
        st.warning(f"⚠️ LSTM not loaded: {e}")
    
    return models

# ==================== FORECAST FUNCTIONS ====================
def forecast_linear(model_data, df, steps):
    """Linear Regression multi-step forecast"""
    model = model_data['model']
    features = model_data['features']
    
    predictions = []
    
    # Get initial values
    last_temps = df['temp_avg'].iloc[-7:].values.tolist()
    
    for i in range(steps):
        # Prepare features for this prediction
        row = {}
        
        # Lag features
        if 'temp_avg_lag1' in features:
            row['temp_avg_lag1'] = last_temps[-1]
        if 'temp_avg_lag2' in features:
            row['temp_avg_lag2'] = last_temps[-2] if len(last_temps) >= 2 else last_temps[-1]
        if 'temp_avg_lag3' in features:
            row['temp_avg_lag3'] = last_temps[-3] if len(last_temps) >= 3 else last_temps[-1]
        if 'temp_avg_lag7' in features:
            row['temp_avg_lag7'] = last_temps[-7] if len(last_temps) >= 7 else last_temps[0]
        
        # Other features
        if 'temp_max_lag1' in features:
            row['temp_max_lag1'] = df['temp_max'].iloc[-1]
        if 'temp_min_lag1' in features:
            row['temp_min_lag1'] = df['temp_min'].iloc[-1]
        
        # Seasonal features for future date
        future_date = pd.Timestamp.now().normalize() + timedelta(days=i+1)
        if 'sin_day' in features:
            row['sin_day'] = np.sin(2 * np.pi * future_date.dayofyear / 365.25)
        if 'cos_day' in features:
            row['cos_day'] = np.cos(2 * np.pi * future_date.dayofyear / 365.25)
        if 'sin_month' in features:
            row['sin_month'] = np.sin(2 * np.pi * future_date.month / 12)
        if 'cos_month' in features:
            row['cos_month'] = np.cos(2 * np.pi * future_date.month / 12)
        if 'temp_range' in features:
            row['temp_range'] = df['temp_range'].iloc[-7:].mean()
        
        # Create DataFrame with correct column order
        current_data = pd.DataFrame([row])[features]
        
        # Predict
        pred = model.predict(current_data)[0]
        pred = float(np.clip(pred, -30, 45))
        predictions.append(pred)
        
        # Update temperature history
        last_temps.append(pred)
        if len(last_temps) > 7:
            last_temps.pop(0)
    
    start_date = pd.Timestamp.now().normalize() + timedelta(days=1)
    dates = pd.date_range(start_date, periods=steps, freq='D')
    
    return pd.DataFrame({'Date': dates, 'Forecast': predictions})

def forecast_sarimax(model_data, df, steps):
    """SARIMAX multi-step forecast with proper lag updates"""
    model = model_data['model']
    exog_cols = model_data['features']
    
    start_date = pd.Timestamp.now().normalize() + timedelta(days=1)
    future_dates = pd.date_range(start_date, periods=steps, freq='D')
    
    predictions = []
    
    # Get initial lag values
    last_temps = df['temp_avg'].iloc[-7:].values.tolist()
    
    for idx, date in enumerate(future_dates):
        row = {}
        
        # Lag features - use predictions as they become available
        if 'temp_avg_lag1' in exog_cols:
            row['temp_avg_lag1'] = predictions[-1] if len(predictions) >= 1 else last_temps[-1]
        
        if 'temp_avg_lag2' in exog_cols:
            row['temp_avg_lag2'] = predictions[-2] if len(predictions) >= 2 else last_temps[-2]
        
        if 'temp_avg_lag3' in exog_cols:
            row['temp_avg_lag3'] = predictions[-3] if len(predictions) >= 3 else last_temps[-3]
        
        if 'temp_avg_lag7' in exog_cols:
            if len(predictions) >= 7:
                row['temp_avg_lag7'] = predictions[-7]
            else:
                row['temp_avg_lag7'] = last_temps[-(7-len(predictions))]
        
        if 'temp_max_lag1' in exog_cols:
            row['temp_max_lag1'] = df['temp_max'].iloc[-1]
        
        if 'temp_min_lag1' in exog_cols:
            row['temp_min_lag1'] = df['temp_min'].iloc[-1]
        
        # Seasonal features
        if 'sin_day' in exog_cols:
            row['sin_day'] = np.sin(2 * np.pi * date.dayofyear / 365.25)
        if 'cos_day' in exog_cols:
            row['cos_day'] = np.cos(2 * np.pi * date.dayofyear / 365.25)
        if 'sin_month' in exog_cols:
            row['sin_month'] = np.sin(2 * np.pi * date.month / 12)
        if 'cos_month' in exog_cols:
            row['cos_month'] = np.cos(2 * np.pi * date.month / 12)
        
        if 'temp_range' in exog_cols:
            row['temp_range'] = df['temp_range'].iloc[-7:].mean()
        
        # Make one-step prediction
        exog_single = pd.DataFrame([row])[exog_cols]
        try:
            pred = model.forecast(steps=1, exog=exog_single.values)[0]
            pred = float(np.clip(pred, -30, 45))
        except Exception as e:
            pred = last_temps[-1]
        
        predictions.append(pred)
    
    return pd.DataFrame({'Date': future_dates, 'Forecast': predictions})

def forecast_lstm(model_data, df, steps):
    """LSTM multi-step forecast"""
    model = model_data['model']
    X_scaler = model_data['X_scaler']
    y_scaler = model_data['y_scaler']
    lookback = model_data['lookback']
    
    # Features used by LSTM
    lstm_features = [
        "temp_avg_lag1", "temp_avg_lag2", "temp_avg_lag3", "temp_avg_lag7",
        "temp_max_lag1", "temp_min_lag1",
        "sin_day", "cos_day", "sin_month", "cos_month", "temp_range"
    ]
    
    # Get recent history
    recent_data = df[lstm_features].iloc[-lookback:].values
    recent_data_scaled = X_scaler.transform(recent_data)
    
    predictions = []
    sequence = recent_data_scaled.copy()
    
    start_date = pd.Timestamp.now().normalize() + timedelta(days=1)
    
    for i in range(steps):
        # Predict next step
        X_input = sequence[-lookback:].reshape(1, lookback, len(lstm_features))
        pred_scaled = model.predict(X_input, verbose=0)[0, 0]
        pred = y_scaler.inverse_transform([[pred_scaled]])[0, 0]
        pred = float(np.clip(pred, -30, 45))
        predictions.append(pred)
        
        # Update sequence for next prediction
        future_date = start_date + timedelta(days=i)
        new_row = np.array([
            pred,  # temp_avg_lag1
            df['temp_avg'].iloc[-1] if i == 0 else predictions[-2] if i >= 2 else pred,
            df['temp_avg'].iloc[-2] if i == 0 else predictions[-3] if i >= 3 else pred,
            df['temp_avg'].iloc[-7] if i < 7 else predictions[-7],
            df['temp_max'].iloc[-1],
            df['temp_min'].iloc[-1],
            np.sin(2 * np.pi * future_date.dayofyear / 365.25),
            np.cos(2 * np.pi * future_date.dayofyear / 365.25),
            np.sin(2 * np.pi * future_date.month / 12),
            np.cos(2 * np.pi * future_date.month / 12),
            df['temp_range'].iloc[-7:].mean()
        ])
        
        new_row_scaled = X_scaler.transform([new_row])
        sequence = np.vstack([sequence[1:], new_row_scaled])
    
    dates = pd.date_range(start_date, periods=steps, freq='D')
    return pd.DataFrame({'Date': dates, 'Forecast': predictions})

# ==================== EXTERNAL API ====================
@st.cache_data(ttl=3600)
def get_open_meteo(lat, lon, steps):
    """Fetch Open-Meteo forecast"""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_mean&forecast_days={min(steps, 16)}&timezone=auto"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        dates = pd.to_datetime(data['daily']['time'])
        temps = data['daily']['temperature_2m_mean']
        
        return pd.DataFrame({'Date': dates, 'Open-Meteo': temps})
    except Exception as e:
        st.error(f"Failed to fetch Open-Meteo: {e}")
        return pd.DataFrame({'Date': [], 'Open-Meteo': []})

# ==================== METRICS ====================
def calculate_metrics(y_true, y_pred):
    """Calculate accuracy metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {'RMSE': np.nan, 'MAE': np.nan, 'SMAPE': np.nan, 'R²': np.nan}
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    denominator = (np.abs(y_true) + np.abs(y_pred))
    denominator[denominator == 0] = 1
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / denominator)
    
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = np.nan
    
    return {'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'SMAPE': round(smape, 2), 'R²': round(r2, 3)}

# ==================== MAIN APP ====================
st.title("🌡️ Weather Forecast Dashboard")
st.markdown("### New York City Temperature Predictions")

# Load data and models
df = load_data()
models = load_models()

if not models:
    st.error("❌ No models loaded! Please train models first.")
    st.info("Run: `py code/train_linear.py` and `py code/train_arima.py`")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    available_models = list(models.keys())
    
    if len(available_models) > 1:
        compare_mode = st.checkbox("Compare All Models", value=True)
        if not compare_mode:
            selected_model = st.selectbox("Select Model", available_models)
    else:
        compare_mode = False
        selected_model = available_models[0]
    
    forecast_days = st.slider("Forecast Days", 1, 14, 7)
    
    show_historical = st.checkbox("Show Historical Data", value=True)
    show_metrics = st.checkbox("Show Accuracy Metrics", value=True)
    
    st.markdown("---")
    st.markdown("**📊 Model Performance:**")
    for model_name in models.keys():
        if 'metrics' in models[model_name]:
            metrics = models[model_name]['metrics']
            st.markdown(f"**{model_name}**")
            st.markdown(f"└ RMSE: {metrics['rmse']:.2f}°C")
            st.markdown(f"└ R²: {metrics['r2']:.3f}")

# Generate forecasts
forecasts = {}

with st.spinner("🔮 Generating forecasts..."):
    if compare_mode:
        for model_name, model_data in models.items():
            try:
                if model_name == 'Linear Regression':
                    forecasts[model_name] = forecast_linear(model_data, df, forecast_days)
                elif model_name == 'SARIMAX':
                    forecasts[model_name] = forecast_sarimax(model_data, df, forecast_days)
                elif model_name == 'LSTM':
                    forecasts[model_name] = forecast_lstm(model_data, df, forecast_days)
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")
    else:
        try:
            model_data = models[selected_model]
            if selected_model == 'Linear Regression':
                forecasts[selected_model] = forecast_linear(model_data, df, forecast_days)
            elif selected_model == 'SARIMAX':
                forecasts[selected_model] = forecast_sarimax(model_data, df, forecast_days)
            elif selected_model == 'LSTM':
                forecasts[selected_model] = forecast_lstm(model_data, df, forecast_days)
        except Exception as e:
            st.error(f"Error: {e}")

# Get Open-Meteo
open_meteo_df = get_open_meteo(40.7128, -74.0060, forecast_days)

# Display forecast tables
st.subheader("📊 Forecast Results")

cols = st.columns(len(forecasts))
for i, (model_name, forecast_df) in enumerate(forecasts.items()):
    with cols[i]:
        st.markdown(f"**{model_name}**")
        display_df = forecast_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
        display_df['Forecast'] = display_df['Forecast'].round(1)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# Visualization
st.subheader("📈 Temperature Forecast")

fig, ax = plt.subplots(figsize=(16, 7))

if show_historical:
    recent_days = 30
    recent_data = df.tail(recent_days)
    ax.plot(recent_data.index, recent_data['temp_avg'], 
            label='Historical', color='black', linewidth=2.5, marker='o', markersize=4, alpha=0.8)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, (model_name, forecast_df) in enumerate(forecasts.items()):
    ax.plot(forecast_df['Date'], forecast_df['Forecast'], 
            label=model_name, linestyle='--', marker='s', 
            linewidth=2.5, markersize=6, color=colors[i % len(colors)])

if not open_meteo_df.empty:
    ax.plot(open_meteo_df['Date'], open_meteo_df['Open-Meteo'], 
            label='Open-Meteo API', linestyle=':', marker='x', 
            linewidth=2.5, markersize=8, color='magenta')

ax.axvline(x=pd.Timestamp.now().normalize(), color='red', linestyle=':', linewidth=2, alpha=0.7, label='Today')
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Temperature (°C)', fontsize=13, fontweight='bold')
ax.set_title('NYC Temperature Forecast', fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

# Metrics vs Open-Meteo
if show_metrics and not open_meteo_df.empty:
    st.subheader("📏 Forecast Accuracy vs Open-Meteo")
    
    metric_cols = st.columns(len(forecasts))
    
    for i, (model_name, forecast_df) in enumerate(forecasts.items()):
        merged = forecast_df.merge(open_meteo_df, on='Date', how='inner')
        
        if not merged.empty:
            metrics = calculate_metrics(merged['Open-Meteo'], merged['Forecast'])
            
            with metric_cols[i]:
                # Determine best metric (lowest RMSE)
                is_best = metrics['RMSE'] == min([calculate_metrics(
                    f.merge(open_meteo_df, on='Date', how='inner')['Open-Meteo'],
                    f.merge(open_meteo_df, on='Date', how='inner')['Forecast']
                )['RMSE'] for _, f in forecasts.items() if not f.merge(open_meteo_df, on='Date', how='inner').empty])
                
                border_color = '#4CAF50' if is_best else '#ddd'
                
                st.markdown(f"""
                <div style='padding:20px; border:2px solid {border_color}; border-radius:10px; background-color:#ffffff; text-align:center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color:#000000; margin-bottom:15px; font-size:20px;'>{model_name} {'🏆' if is_best else ''}</h3>
                    <p style='font-size:20px; color:#000000; margin:10px 0;'><b>RMSE:</b> {metrics['RMSE']}°C</p>
                    <p style='font-size:20px; color:#000000; margin:10px 0;'><b>MAE:</b> {metrics['MAE']}°C</p>
                    <p style='font-size:16px; color:#666; margin:10px 0;'>SMAPE: {metrics['SMAPE']}%</p>
                    <p style='font-size:12px; color:#999; margin-top:15px; font-style:italic;'>vs Open-Meteo Forecast</p>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Location", "New York City")
with col2:
    st.metric("Data Points", f"{len(df):,} days")
with col3:
    st.metric("Last Update", df.index[-1].strftime('%Y-%m-%d'))

st.markdown("""
<div style='text-align:center; color:gray; margin-top:20px;'>
    <p>🌡️ Weather Forecast Dashboard | Data Source: Open-Meteo API | Models: Linear Regression, SARIMAX, LSTM</p>
    <p style='font-size:12px;'>Built with Streamlit • Forecasts are estimates and may differ from actual temperatures</p>
</div>
""", unsafe_allow_html=True)