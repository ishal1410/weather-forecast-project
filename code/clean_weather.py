import pandas as pd
import numpy as np

df = pd.read_csv("data/raw_weather.csv")

date_col = 'time' if 'time' in df.columns else 'date'
df["date"] = pd.to_datetime(df[date_col])
if date_col == 'time':
    df = df.drop(columns=['time'])

df = df.rename(columns={
    'temperature_2m_max': 'temp_max',
    'temperature_2m_min': 'temp_min',
    'temperature_2m_mean': 'temp_avg',
    'precipitation_sum': 'precipitation',
    'rain_sum': 'rain',
    'snowfall_sum': 'snowfall',
    'windspeed_10m_max': 'wind_max'
})

df = df.drop_duplicates(subset='date', keep='first')
df = df.sort_values('date')
df = df.dropna()
df = df.set_index('date')

df.to_csv("data/cleaned_weather.csv")

print(f"✓ Cleaned: {len(df)} days ({df.index.min().date()} to {df.index.max().date()})")