from flask import Flask, jsonify, render_template, request
import os
import pandas as pd

app = Flask(__name__)

# Mock data for fallback
mock_predictions = [
    {
        "risk_prob_low": 0.1,
        "risk_prob_medium": 0.3,
        "risk_prob_high": 0.6,
        "risk_prediction": "HIGH",
        "date": "2025-04-28",
        "sentiment_score": -0.65
    },
    {
        "risk_prob_low": 0.2,
        "risk_prob_medium": 0.5,
        "risk_prob_high": 0.3,
        "risk_prediction": "MEDIUM",
        "date": "2025-04-29",
        "sentiment_score": -0.15
    },
    {
        "risk_prob_low": 0.277733,
        "risk_prob_medium": 0.532206,
        "risk_prob_high": 0.190061,
        "risk_prediction": "MEDIUM",
        "date": "2025-04-30",
        "sentiment_score": 0.45
    }
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def get_data():
    import traceback
    import numpy as np

    pair = request.args.get("pair", "USD/CNY")
    pair_key = pair.lower().replace("/", "_")
    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "data")

    prediction_path = os.path.join(data_dir, f"predictions_{pair_key}.csv")
    shap_path = os.path.join(data_dir, f"shap_values_{pair_key}.npy")
    shap_input_path = os.path.join(data_dir, f"shap_input_{pair_key}.csv")

    try:
        if not os.path.exists(prediction_path):
            raise FileNotFoundError("No real prediction file found.")

        df = pd.read_csv(prediction_path, parse_dates=["date"])
        df = df.sort_values("date")
        today = pd.Timestamp.now().normalize()
        today_row = df[df["date"].dt.normalize() == today]
        if today_row.empty:
            today_row = df.iloc[[-1]]
        latest = today_row.iloc[0]

        risk_index = round(
            latest["risk_prob_low"] * 0 +
            latest["risk_prob_medium"] * 50 +
            latest["risk_prob_high"] * 100
        )

        recent_preds = df.tail(20)[["date", "risk_label", "risk_prob_low", "risk_prob_medium", "risk_prob_high"]]
        df_truth = df[df["risk_level"].notna()].tail(5)[["date", "risk_level"]] if "risk_level" in df.columns else pd.DataFrame()

        feature_name_map = {
            "bollinger_upper": "Upper Bollinger Band",
            "bollinger_lower": "Lower Bollinger Band",
            "bollinger_width": "Bollinger Band Width",
            "rolling_std_5": "5-Day Volatility",
            "rolling_std_20": "20-Day Volatility",
            "rolling_mean_5": "5-Day Moving Average",
            "rolling_mean_20": "20-Day Moving Average",
            "return": "1-Day Exchange Rate Return",
            "close_price": "Current Exchange Rate",
            "drawdown_10": "10-Day Maximum Drawdown",
            "drawdown_speed": "Drawdown Speed",
            "high_low_range_5": "5-Day High-Low Range",
            "fed_funds_rate_x_vol": "Fed Rate & Volatility Interaction",
            "treasury_10y_x_vol": "10Y Treasury & Volatility Interaction",
            "cpi_x_vol": "CPI & Volatility Interaction",
        }

        key_factors = []
        if os.path.exists(shap_path) and os.path.exists(shap_input_path):
            shap_values = np.load(shap_path)
            X = pd.read_csv(shap_input_path)

            if shap_values.shape[0] == len(X):
                class_index = int(latest["risk_prediction"])
                feature_names = X.columns.tolist()

                shap_row = shap_values[-1, :, class_index]
                abs_shap = pd.Series(np.abs(shap_row), index=feature_names)
                key_factors = abs_shap.sort_values(ascending=False).head(3).index.tolist()
                # Replace technical names with readable labels
                pretty_factors = [feature_name_map.get(f, f) for f in key_factors]

        return jsonify({
            "pair": pair,
            "prediction_date": str(latest["date"].date()),
            "risk_label": latest["risk_label"],
            "risk_index": risk_index,
            "risk_probabilities": {
                "low": round(latest["risk_prob_low"], 2),
                "medium": round(latest["risk_prob_medium"], 2),
                "high": round(latest["risk_prob_high"], 2),
            },
            "recent_predictions": recent_preds.to_dict(orient="records"),
            "recent_truth": df_truth.to_dict(orient="records"),
            "key_factors": pretty_factors,
            "sentiment_score": latest.get("sentiment_score", 0)
        })

    except Exception as e:
        print("⚠️ Falling back to mock data due to error:")
        traceback.print_exc()

        latest_prediction = mock_predictions[-1]
        risk_score = round(
            latest_prediction["risk_prob_low"] * 0 +
            latest_prediction["risk_prob_medium"] * 50 +
            latest_prediction["risk_prob_high"] * 100
        )

        return jsonify({
            "risk_index": risk_score,
            "risk_label": latest_prediction["risk_prediction"],
            "prediction_date": latest_prediction["date"],
            "key_factors": ["rolling_std_5", "price_change_1d"],
            "recent_predictions": mock_predictions,
            "sentiment_score": latest_prediction["sentiment_score"]
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
