import os
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
@app.route("/data")
def get_data():
    import traceback

    pair = request.args.get("pair", "USD/CNY")
    pair_key = pair.lower().replace("/", "_")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "data")

    prediction_path = os.path.join(data_dir, f"predictions_{pair_key}.csv")
    shap_path = os.path.join(data_dir, f"shap_values_{pair_key}.npy")
    shap_input_path = os.path.join(data_dir, f"shap_input_{pair_key}.csv")

    print(f"🌀 预测路径: {prediction_path}")
    if not os.path.exists(prediction_path):
        return jsonify({"error": "预测文件不存在"}), 404

    try:
        df = pd.read_csv(prediction_path, parse_dates=["date"])
        print(f"📊 加载 {len(df)} 条记录")

        df = df.sort_values("date")
        today = pd.Timestamp.now().normalize()
        today_row = df[df["date"].dt.normalize() == today]
        if today_row.empty:
            today_row = df.iloc[[-1]]  # fallback
        latest = today_row.iloc[0]

        # 计算风险指数
        risk_index = round(
            latest["risk_prob_low"] * 20 +
            latest["risk_prob_medium"] * 60 +
            latest["risk_prob_high"] * 100
        )

        # 最近预测
        recent_preds = df.tail(2)[["date", "risk_label", "risk_prob_low", "risk_prob_medium", "risk_prob_high"]]

        # 最近真值
        df_truth = df[df["risk_level"].notna()].tail(5)[["date", "risk_level"]] if "risk_level" in df.columns else pd.DataFrame()

        # Key SHAP 特征
        key_factors = []
        if os.path.exists(shap_path) and os.path.exists(shap_input_path):
            import shap
            import numpy as np
            shap_values = np.load(shap_path, allow_pickle=True)
            X = pd.read_csv(shap_input_path)

            shap_values_for_class = shap_values[latest["risk_prediction"]]
            row_idx = today_row.index[0]
            if row_idx < len(shap_values_for_class):
                shap_row = shap_values_for_class[row_idx]
                abs_shap = pd.Series(np.abs(shap_row), index=X.columns)
                key_factors = abs_shap.sort_values(ascending=False).head(3).index.tolist()

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
            "key_factors": key_factors
        })

    except Exception as e:
        print("❌ 错误 traceback:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
