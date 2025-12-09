import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

os.makedirs('data', exist_ok=True)

latitude = 40.7128
longitude = -74.0060
start_date = "2020-01-01"
end_date = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
timezone = "America/New_York"

url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&start_date={start_date}&end_date={end_date}"
    f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
    f"precipitation_sum,rain_sum,snowfall_sum,windspeed_10m_max"
    f"&timezone={timezone}"
)

for attempt in range(3):
    response = requests.get(url)
    if response.status_code == 200:
        break
    time.sleep(2)
else:
    raise Exception("API request failed")

data = response.json()
df_raw = pd.DataFrame(data["daily"])
df_raw.to_csv("data/raw_weather.csv", index=False)

df = df_raw.copy()
df.rename(columns={
    "time": "date",
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_avg",
    "precipitation_sum": "precipitation",
    "rain_sum": "rain",
    "snowfall_sum": "snowfall",
    "windspeed_10m_max": "wind_max"
}, inplace=True)

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_year'] = df['date'].dt.dayofyear
df['temp_range'] = df['temp_max'] - df['temp_min']

df.to_csv("data/cleaned_weather.csv", index=False)

print(f"✓ Done: {len(df)} days ({df['date'].min().date()} to {df['date'].max().date()})")