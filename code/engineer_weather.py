import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/cleaned_weather.csv", index_col='date', parse_dates=True)

# Lag features
for lag in [1, 2, 3, 7]:
    df[f"temp_avg_lag{lag}"] = df["temp_avg"].shift(lag)
    df[f"temp_max_lag{lag}"] = df["temp_max"].shift(lag)
    df[f"temp_min_lag{lag}"] = df["temp_min"].shift(lag)

# Rolling features
for window in [3, 7, 14, 30]:
    df[f"temp_avg_ma{window}"] = df["temp_avg"].rolling(window=window).mean()
    df[f"temp_avg_std{window}"] = df["temp_avg"].rolling(window=window).std()
    df[f"temp_max_rolling{window}"] = df["temp_max"].rolling(window=window).max()
    df[f"temp_min_rolling{window}"] = df["temp_min"].rolling(window=window).min()

# Temperature range
df["temp_range"] = df["temp_max"] - df["temp_min"]
df["temp_range_lag1"] = df["temp_range"].shift(1)
df["temp_range_ma7"] = df["temp_range"].rolling(window=7).mean()

# Rate of change
df["temp_avg_change"] = df["temp_avg"].diff()
df["temp_avg_change_pct"] = df["temp_avg"].pct_change() * 100
df["temp_max_change"] = df["temp_max"].diff()
df["temp_min_change"] = df["temp_min"].diff()

# Seasonal features
df["day_of_year"] = df.index.dayofyear
df["sin_day"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["cos_day"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
df["month"] = df.index.month
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
df["week_of_year"] = df.index.isocalendar().week
df["season"] = df["month"].map({12:0, 1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3})

# Interaction features
df["temp_dev_from_ma7"] = df["temp_avg"] - df["temp_avg_ma7"]
df["temp_dev_from_ma30"] = df["temp_avg"] - df["temp_avg_ma30"]
df["is_extreme_heat"] = (df["temp_max"] > df["temp_max_rolling30"]).astype(int)
df["is_extreme_cold"] = (df["temp_min"] < df["temp_min_rolling30"]).astype(int)

# Target variables
df["target_temp_avg"] = df["temp_avg"]
df["target_temp_avg_smooth"] = df["temp_avg"].rolling(window=7).mean()

# Cleanup
df = df.drop(columns=["day_of_year"], errors='ignore')
df = df.dropna()

df.to_csv("data/engineered_weather.csv")

print(f"✓ Engineered: {len(df)} rows, {len(df.columns)} features")