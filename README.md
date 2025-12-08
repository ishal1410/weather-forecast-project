```markdown
📦 CS666 Final Project Submission – ReadMe

Project Title: Forecasting Weather Temperature Trends Using AI Models  
Course: CS666-003 SEC-1 Final Exam  
Date: December 9, 2025  
Team Members:  
- Vishal Patel  
- Saunil Patel  
- Fenny Patel  

```
📁 Project Structure
```
WETHER_FORECAST_PROJECT/
├── code/
│   ├── fetch_weather.py
│   ├── clean_weather.py
│   ├── engineer_weather.py
│   ├── train_linear.py
│   ├── train_arima.py
│   └── train_lstm.py
├── data/
│   ├── raw_weather.csv
│   ├── cleaned_weather.csv
│   └── engineered_weather.csv
├── models/
│   ├── linear_regression_model.pkl
│   ├── arima_model.pkl
│   ├── sarimax_model.pkl
│   ├── lstm_model.h5
│   └── lstm_scaler.pkl
├── notebooks/
├── app.py
├── requirements.txt
```

---

## How to Run the Project in VS Code

### Option 1: Recommended (Isolated Virtual Environment)

1. **Open Project in VS Code**  
   - File → Open Folder → select `WETHER_FORECAST_PROJECT`.

2. **Create and Activate Virtual Environment**  
   - **Windows**  
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux**  
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Pipeline (optional if models already exist)**  
   ```bash
   python code/fetch_weather.py
   python code/clean_weather.py
   python code/engineer_weather.py
   python code/train_linear.py
   python code/train_arima.py
   python code/train_lstm.py
   ```

5. **Launch Dashboard**  
   ```bash
   streamlit run app.py
   ```

---

### Option 2: Simple (Global Install, No venv)

1. **Install Dependencies Globally**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Pipeline (which we used for our project)**  
   ```bash
   python code/fetch_weather.py
   python code/clean_weather.py
   python code/engineer_weather.py
   python code/train_linear.py
   python code/train_arima.py
   python code/train_lstm.py
   ```

3. **Launch Dashboard**  
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Dashboard Features

- Model selection: Linear Regression, ARIMA, LSTM  
- Forecast horizon: 1-day and 7-day  
- Real-time comparison with Open Meteo API  
- Accuracy metrics: RMSE, MAE, SMAPE  
- CSV export for reproducibility  

---

##  Quick Demo 

If `data/engineered_weather.csv` and all models in `models/` already exist, you can skip training and run directly:

```bash
streamlit run app.py
```

---

