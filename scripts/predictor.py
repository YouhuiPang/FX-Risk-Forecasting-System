import os
import pandas as pd
import datetime
import pickle
import numpy as np
import shap

def load_model_and_features(pair):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root, "models")
    model_path = os.path.join(model_dir, f"risk_model_{pair.lower().replace('/', '_')}_xgb_multi.pkl")
    features_path = os.path.join(model_dir, f"selected_features_{pair.lower().replace('/', '_')}.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)
    return model, selected_features

def add_macro_anomaly_features(X, macro_cols):
    print(f"\n添加宏观异常特征，初始列: {list(X.columns)}")
    X_new = X.copy()
    for col in macro_cols:
        if col not in X_new.columns:
            print(f"⚠️ 宏观指标 {col} 不存在，跳过...")
            continue
        if 'rolling_std_20' in X_new.columns:
            X_new[f'{col}_x_vol'] = X_new[col] * X_new['rolling_std_20']
        mean_val = X_new[col].mean()
        std_val = X_new[col].std()
        if std_val == 0 or pd.isna(std_val):
            z_scores = pd.Series(0, index=X_new.index)
        else:
            z_scores = (X_new[col] - mean_val) / std_val
            X_new[f'{col}_anomaly_weighted'] = X_new[col] * (z_scores.abs() > 2).astype(int) * z_scores.abs()
    return X_new

def predict_risk(pair, features_csv_path, output_csv_path):
    try:
        df = pd.read_csv(features_csv_path, parse_dates=["date"])
    except Exception as e:
        print(f"加载特征数据失败: {e}")
        return

    df = df.sort_values("date")
    today = datetime.datetime.now().date()

    macro_cols = ['fed_funds_rate', 'treasury_10y', 'cpi', 'usd_index']
    df = add_macro_anomaly_features(df, macro_cols)

    model, selected_features = load_model_and_features(pair)
    missing_features = [feat for feat in selected_features if feat not in df.columns]
    if missing_features:
        print(f"缺少特征: {missing_features}")
        return

    X = df[selected_features]
    risk_probs = model.predict_proba(X)
    risk_pred = model.predict(X)
    class_map = dict(zip(model.classes_, ["Low", "Medium", "High"]))

    for i, cls in enumerate(model.classes_):
        df[f"risk_prob_{class_map[cls].lower()}"] = risk_probs[:, i]
    df["risk_prediction"] = risk_pred
    df["risk_label"] = df["risk_prediction"].map(class_map)

    if 'risk_level' not in df.columns:
        print("缺少真实标签 risk_level")
        return

    output_cols = ["date", "risk_label", "risk_prob_low", "risk_prob_medium", "risk_prob_high", "risk_prediction", "risk_level"]
    df[output_cols].to_csv(output_csv_path, index=False)
    print(f"预测结果已保存至: {output_csv_path}")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        pair_key = pair.lower().replace("/", "_")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        np.save(os.path.join(root, "data", f"shap_values_{pair_key}.npy"), shap_values)
        X.to_csv(os.path.join(root, "data", f"shap_input_{pair_key}.csv"), index=False)
        print("✅ SHAP 解释数据已保存")
    except Exception as e:
        print(f"⚠️ 生成 SHAP 数据失败: {e}")

    # 保存历史预测记录
    history_path = os.path.join(root, "data", f"all_predictions_{pair.lower().replace('/', '_')}.csv")
    try:
        df_to_save = df[output_cols].copy()
        df_to_save["prediction_timestamp"] = datetime.datetime.now()

        today = datetime.datetime.now().date()
        recent_dates = set([(today - datetime.timedelta(days=i)) for i in range(3)])

        if os.path.exists(history_path):
            df_old = pd.read_csv(history_path, parse_dates=["date", "prediction_timestamp"])
            known_dates = set(df_old["date"].dt.date)

            is_new_or_recent = df_to_save["date"].dt.date.apply(
                lambda d: d not in known_dates or d in recent_dates
            )
            df_new = df_to_save[is_new_or_recent]

            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        else:
            df_combined = df_to_save

        df_combined.to_csv(history_path, index=False)
        print(f"✅ 历史预测数据已保存至: {history_path}")
    except Exception as e:
        print(f"⚠️ 无法保存历史数据: {e}")

    df_future = df[df["date"].dt.date >= today].copy()
    if df_future.empty:
        df_future = df.tail(1).copy()
        df_future.loc[df_future.index, "date"] = pd.to_datetime(today)

    print("未来预测：")
    print(df_future[["date", "risk_label", "risk_prob_low", "risk_prob_medium", "risk_prob_high"]])
    return df


if __name__ == "__main__":
    pair = "USD/CNY"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_csv_path = os.path.join(root, "data", f"features_{pair.lower().replace('/', '_')}.csv")
    output_csv_path = os.path.join(root, "data", f"predictions_{pair.lower().replace('/', '_')}.csv")

    predict_risk(pair, features_csv_path, output_csv_path)