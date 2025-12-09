from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import pickle
import os

warnings.filterwarnings("ignore")

df = pd.read_csv("data/engineered_weather.csv", index_col='date', parse_dates=True)
df.index.freq = 'D'

exog_cols = [
    "temp_avg_lag1", "temp_avg_lag2", "temp_avg_lag3", "temp_avg_lag7",
    "temp_max_lag1", "temp_min_lag1",
    "sin_day", "cos_day", "sin_month", "cos_month", "temp_range"
]

exog_cols = [col for col in exog_cols if col in df.columns]
df = df.dropna()

test_size = 365
train_df = df.iloc[:-test_size]
test_df = df.iloc[-test_size:]

y_train = train_df["temp_avg"]
exog_train = train_df[exog_cols]
y_test = test_df["temp_avg"]
exog_test = test_df[exog_cols]

model = SARIMAX(
    y_train,
    exog=exog_train,
    order=(2, 0, 2),
    seasonal_order=(1, 0, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False,
    trend='c'
)

model_fit = model.fit(disp=False, maxiter=150, method='lbfgs')

y_pred_train = model_fit.predict(start=y_train.index[0], end=y_train.index[-1], exog=exog_train, dynamic=False)
y_pred_test = model_fit.predict(start=y_test.index[0], end=y_test.index[-1], exog=exog_test, dynamic=False)

test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

os.makedirs('models', exist_ok=True)

model_info = {
    'model_fit': model_fit,
    'exog_cols': exog_cols,
    'test_metrics': {'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}
}

with open('models/sarimax_model_optimized.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print(f"✓ SARIMAX: RMSE={test_rmse:.2f}°C | MAE={test_mae:.2f}°C | R²={test_r2:.4f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Full timeline
ax1.plot(y_train.index, y_train, label="Train", color="black", alpha=0.6)
ax1.plot(y_test.index, y_test, label="Test (Actual)", color="blue", linewidth=2)
ax1.plot(y_pred_train.index, y_pred_train, label="Train Pred", color="green", linestyle="--", alpha=0.5)
ax1.plot(y_pred_test.index, y_pred_test, label="Test Pred", color="red", linestyle="--", linewidth=2)
ax1.axvline(x=y_test.index[0], color='gray', linestyle=':', linewidth=2)
ax1.set_ylabel("Temperature (°C)")
ax1.set_title("SARIMAX: Actual vs Predicted")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Test set zoom
ax2.plot(y_test.index, y_test, label="Actual", color="blue", linewidth=2)
ax2.plot(y_pred_test.index, y_pred_test, label="Predicted", color="red", linestyle="--", linewidth=2)
ax2.fill_between(y_test.index, y_test, y_pred_test, alpha=0.2)
ax2.set_xlabel("Date")
ax2.set_ylabel("Temperature (°C)")
ax2.set_title(f"Test Set (RMSE: {test_rmse:.2f}°C, R²: {test_r2:.3f})")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/sarimax_predictions.png', dpi=150, bbox_inches='tight')
print("✓ Saved: models/sarimax_predictions.png")
plt.show()