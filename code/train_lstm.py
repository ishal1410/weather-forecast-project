# type: ignore
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

df = pd.read_csv("data/engineered_weather.csv", index_col="date", parse_dates=True)

features = [
    "temp_avg_lag1", "temp_avg_lag2", "temp_avg_lag3", "temp_avg_lag7",
    "temp_max_lag1", "temp_min_lag1",
    "sin_day", "cos_day", "sin_month", "cos_month",
    "temp_range", "temp_avg"
]

df_clean = df[[f for f in features if f in df.columns]].dropna()

test_size = 365
train_df = df_clean.iloc[:-test_size]
test_df = df_clean.iloc[-test_size:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(train_df.drop(columns=['temp_avg']))
y_train = scaler_y.fit_transform(train_df[['temp_avg']])

X_test = scaler_X.transform(test_df.drop(columns=['temp_avg']))
y_test = scaler_y.transform(test_df[['temp_avg']])

def create_sequences(X, y, lookback=7):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i, 0])
    return np.array(X_seq), np.array(y_seq)

lookback = 7
X_train_seq, y_train_seq = create_sequences(X_train, y_train, lookback)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, lookback)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(lookback, X_train_seq.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
]

history = model.fit(X_train_seq, y_train_seq, epochs=50, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=0)

y_pred_train = scaler_y.inverse_transform(model.predict(X_train_seq, verbose=0))
y_true_train = scaler_y.inverse_transform(y_train_seq.reshape(-1, 1))

y_pred_test = scaler_y.inverse_transform(model.predict(X_test_seq, verbose=0))
y_true_test = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))

test_rmse = np.sqrt(mean_squared_error(y_true_test, y_pred_test))
test_mae = mean_absolute_error(y_true_test, y_pred_test)
test_r2 = r2_score(y_true_test, y_pred_test)

os.makedirs('models', exist_ok=True)
model.save('models/lstm_model.h5')

with open('models/lstm_scalers.pkl', 'wb') as f:
    pickle.dump({'X_scaler': scaler_X, 'y_scaler': scaler_y, 'lookback': lookback}, f)

print(f"✓ LSTM: RMSE={test_rmse:.2f}°C | MAE={test_mae:.2f}°C | R²={test_r2:.4f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(history.history['loss'], label='Train Loss', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training History')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(y_true_test, label='Actual', linewidth=2)
ax2.plot(y_pred_test, label='Predicted', linestyle='--', linewidth=2)
ax2.set_xlabel('Day')
ax2.set_ylabel('Temperature (°C)')
ax2.set_title(f'Test Predictions (RMSE: {test_rmse:.2f}°C, R²: {test_r2:.3f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('models/lstm_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: models/lstm_results.png")
plt.show()