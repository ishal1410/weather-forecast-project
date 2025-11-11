#  Weather Forecasting with AI

## 📖 Overview
This project uses Linear Regression, ARIMA, and LSTM to forecast short-term temperature trends in New York City (2019–2023). All models are trained on the same engineered dataset and evaluated using RMSE, MAE, R², MAPE, and MSE.

##  Team
1. Saunil Patel  
2. Fenny Patel  
3. Vishal Patel  

---

## 📂 Code Folder name — Weather_Forecast_Project

This folder contains all core scripts for data handling and model training.

### 🧭 Script Overview

| Script Name            | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `fetch_weather.py`     | Downloads NYC weather data (2019–2023) from Open-Meteo API              |
| `clean_weather.py`     | Cleans missing/abnormal values from raw data                            |
| `engineer_weather.py`  | Adds lag features, moving averages, and temporal encodings              |
| `train_linear.py`      | Trains Linear Regression model and prints evaluation metrics            |
| `train_arima.py`       | Trains ARIMA model and prints evaluation metrics                        |
| `train_lstm.py`        | Trains LSTM model and prints evaluation metrics                         |

### 📦 CSV Files

| File Name               | Description                                                             |
|-------------------------|-------------------------------------------------------------------------|
| `raw_weather.csv`       | Raw data pulled from Open-Meteo                                         |
| `cleaned_weather.csv`   | Cleaned version with missing/abnormal values removed                    |
| `engineered_weather.csv`| Final dataset with lag features, moving averages, and date encodings    |

---

## Libraries Used
Data Manipulation
- Pandas and NumPy — for loading, cleaning, and transforming weather data

### Modeling
- Scikit-learn — for Linear Regression and evaluation metrics
- Statsmodels — for ARIMA time series modeling
- TensorFlow/Keras — for building and training the LSTM model

### Visualization
- Matplotlib and Seaborn — for plotting actual vs. predicted temperatures and comparing model metrics

### 🛠️ Installation via Terminal
To install all required libraries, run:
```bash
pip install pandas numpy scikit-learn statsmodels tensorflow matplotlib seaborn
```


##  How to Run

### 1️⃣ Fetch Raw Data
```bash
py code/fetch_weather.py
```
**Output:**
```
Fetched 1461 rows of weather data.
Raw and cleaned weather data saved.
```

### 2️⃣ Clean Data
```bash
py code/clean_weather.py
```
**Output:**
```
✅ Cleaned data saved to data/cleaned_weather.csv
```

### 3️⃣ Engineer Features
```bash
py code/engineer_weather.py
```
**Output:**
```
Engineered features saved to data/engineered_weather.csv
```

---

##  Train Models

### 🔹 Linear Regression
```bash
py code/train_linear.py
```
**Output:**
```
Final Linear Regression Model Trained
MSE: 3.27
RMSE: 1.81
MAE: 1.38
MAPE: 27.85%
R²: 0.94
```

### 🔹 ARIMA
```bash
py code/train_arima.py
```
**Output:**
```
ARIMA Model Trained
MSE: 0.22
RMSE: 0.46
MAE: 0.36
MAPE: 3.88%
R²: 1.00
```

### 🔹 LSTM
```bash
py code/train_lstm.py
```
**Output:**
```
✅ Refined LSTM Model Trained
MSE: 0.75
RMSE: 0.87
MAE: 0.67
MAPE: 14.01%
R²: 0.99
```

Each training script:
- Loads `engineered_weather.csv`
- Splits into train/test (80/20)
- Trains model and prints RMSE, MAE, R², MAPE, MSE
- Displays actual vs. predicted graph

---

Results & Visualization
Open the notebook:
notebooks/result_table.ipynb
then run the cell, it will shows the Compare metrics across models

**Output**
```


| Model             | MSE  | RMSE | MAE  | MAPE (%) | R²   |
|------------------|------|------|------|----------|------|
| Linear Regression| 3.27 | 1.81 | 1.38 | 27.85    | 0.94 |
| ARIMA            | 0.22 | 0.46 | 0.36 | 3.88     | 1.00 |
| Refined LSTM     | 0.78 | 0.80 | 0.60 | 13.38    | 0.99 |
```

**link**

Open-Meteo: “Free weather API with historical and forecast data,” - https://open-meteo.com/

