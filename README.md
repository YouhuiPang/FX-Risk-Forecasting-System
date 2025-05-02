
# Quantitative FX Risk Forecasting System 
## ML-Based Volatility Modeling

This project is a web-based FX risk prediction system designed to help users intuitively understand short-term currency volatility. It uses historical exchange rate data and macroeconomic indicators to predict the next 3-day risk level of currency pairs (e.g., USD/CNY) using a machine learning model. The final product is a visual dashboard that requires no financial background to interpret. 

---

## ⚙️ Features

- Real-time risk predictions (Low, Medium, High)
- SHAP-based explanation of key influencing factors
- Past 20-day risk forecast chart
- Optional sentiment analyzer (Beta Not Functioning)
- Fully functional Flask + Tailwind CSS frontend

---

## 🏗️ System Architecture

This project follows a modular structure that separates data processing, model training, prediction logic, and frontend dashboard into clear components. Below is an overview of the system architecture:

### 📁 `scripts/` – Core Logic Modules

- `data_fetcher.py`: Merges raw exchange rate and macroeconomic data into engineered features.
- `feature_engineer.py`: Creates time-series and macro interaction features (e.g. volatility, drawdown).
- `model_trainer.py`: Trains XGBoost models (one per currency pair) with SMOTE+ENN for class balancing.
- `predictor.py`: Loads the latest features and trained model to output risk level predictions and SHAP values.
- `scheduler.py`: Automates daily data update and prediction via `schedule` library.
- `get_exchange_data.py` / `get_macro_data.py`: Fetches raw data from Yahoo Finance and FRED.

### 📁 `models/` – Saved Models

Contains trained models and selected feature sets for each currency pair (`.pkl` files).

### 📁 `data/` – All Data Files

- Raw data: `usd_cny.csv`, `macro.csv`
- Engineered features: `features_*.csv`
- Prediction outputs: `predictions_*.csv`
- SHAP inputs and values: `shap_input_*.csv`, `shap_values_*.npy`

### 📁 `frontend/` – Web Dashboard (Flask + Tailwind CSS)

- `app.py`: Flask backend serving the dashboard and API.
- `templates/index.html`: Main HTML structure styled with Tailwind CSS.
- `static/js/main.js`: Handles real-time rendering of risk levels, forecasts, and sentiment analysis.
- `static/css/style.css`: Optional custom styles.

### 📁 `plots/` and `figures/` – Visualization Assets

- `shap_summary_*.png`: SHAP visualizations for global feature importance.
- `roc_curve.png`, `precision_recall_curve.png`: Performance metrics visualizations.

### 🔁 Workflow Summary

1. **Data Update**: `scheduler.py` triggers data fetching daily.
2. **Data Fetching & Feature Engineering**: `data_fetcher.py` processes new raw data and update latest features.
3. **Prediction**: `predictor.py` loads the latest features and model to generate risk predictions.
4. **Web Display**: `app.py` serves the dashboard at `/`, and `/data` API responds with live predictions and SHAP-based key factor explanations.
5. **Frontend Rendering**: `main.js` dynamically updates charts, risk level indicators, and sentiment fields.

---

## 🔧 Prerequisites

Before installing and running this project, make sure you have Python installed:

- **Python 3.8 or higher**  
  Download from: [https://www.python.org/downloads/](https://www.python.org/downloads/)

- **(Recommended) Virtual Environment**  
  It's best to isolate project dependencies using a virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate      # macOS / Linux
  venv\Scripts\activate       # Windows
  ```

---

## 🚀 Installation

```bash
git clone https://github.com/YouhuiPang/FX-Risk-Forecasting-System.git
cd FX-Risk-Forecasting-System
pip install -r requirements.txt
```

---

## 🔧 Usage

### To start the web dashboard:
```bash
python app.py
```
Then open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

#### To manually update and predict:
```bash
# Step 1: Fetch the latest exchange rate + macroeconomic data
python scripts/data_fetcher.py

# Step 2: Run prediction using the latest features
python scripts/predictor.py
```

## ⏰ Scheduled Prediction (Optional)

To enable scheduled daily automatic forecast and data updates (This is set to update at 00:00):
```bash
python scripts/scheduler.py
```

Or run in the background:
```bash
nohup python scripts/scheduler.py &
```

---

## 💡 Credits

- Exchange rate data from Yahoo Finance (`yfinance`)
- Macroeconomic data from FRED (Federal Reserve API)
- Model: XGBoost classifier with SMOTE + ENN for class imbalance
- Feature explanations via SHAP
- Dashboard styled with Tailwind CSS and Chart.js
