# scripts/feature_engineer.py
import pandas as pd
import os
import numpy as np
from scipy.stats import genpareto

def load_exchange_data(exchange_df):
    df = exchange_df.copy()
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df["return"] = df["close_price"].pct_change()
    df = df.dropna(subset=["return"])
    return df


def add_technical_indicators(df):
    df["rolling_mean_5"] = df["close_price"].rolling(5, min_periods=1).mean()
    df["rolling_mean_20"] = df["close_price"].rolling(20, min_periods=1).mean()
    df["price_above_ma"] = df["close_price"] > df["rolling_mean_20"]
    df["momentum_10"] = df["close_price"].diff(10)
    df["rolling_std_5"] = df["close_price"].rolling(5, min_periods=1).std()
    df["rolling_std_20"] = df["close_price"].rolling(20, min_periods=1).std()
    df["bollinger_upper"] = df["rolling_mean_20"] + 2 * df["rolling_std_20"]
    df["bollinger_lower"] = df["rolling_mean_20"] - 2 * df["rolling_std_20"]
    df["high_low_range_5"] = df["close_price"].rolling(5, min_periods=1).max() - df["close_price"].rolling(5, min_periods=1).min()
    rolling_max = df["close_price"].rolling(10, min_periods=1).max()
    df["drawdown_10"] = (df["close_price"] - rolling_max) / rolling_max
    df["ma_diff"] = df["rolling_mean_5"] - df["rolling_mean_20"]
    df["return_slope_5"] = df["return"].rolling(5).mean()
    df["drawdown_speed"] = (df["close_price"] - df["close_price"].rolling(5).max()) / 5
    df["bollinger_width"] = df["bollinger_upper"] - df["bollinger_lower"]
    df["volatility_ratio"] = df["rolling_std_5"] / (df["rolling_std_20"] + 1e-6)
    df["range_return_ratio"] = df["high_low_range_5"] / (df["return"].rolling(5).std() + 1e-6)

    delta = df["close_price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)  # 避免除0错误
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df["rsi_slope"] = df["rsi_14"].diff()


    return df

def load_macro_data(macro_df):
    """加载并预处理宏观数据"""
    df = macro_df.copy()
    df = df.sort_values("date").ffill().bfill()  # 填充缺失值
    return df


def merge_features(exchange_df, macro_df, pair, pot_quantile=0.75, evt_quantile=0.95):
    """合并特征并生成多标签 EVT 风险等级（0=低, 1=中, 2=高）"""
    exchange_df = exchange_df.dropna(subset=["date"])
    macro_df = macro_df.dropna(subset=["date"])

    df = pd.merge_asof(
        exchange_df.sort_values("date"),
        macro_df.sort_values("date"),
        on="date",
        direction="backward"
    )

    # Step 1: 生成未来3日最小收益
    df["future_min_return"] = df["return"].rolling(3, min_periods=1).min().shift(-2)

    # Step 2: EVT 拟合高风险阈值
    down_returns = df["future_min_return"][df["future_min_return"] < 0].abs()
    pot_threshold = down_returns.quantile(pot_quantile)
    excesses = down_returns[down_returns > pot_threshold] - pot_threshold

    c, loc, scale = genpareto.fit(excesses)
    n, nu = len(down_returns), len(excesses)
    extreme_threshold = pot_threshold + (scale / c) * ((n / nu * (1 - evt_quantile)) ** (-c) - 1)

    # Step 3: 中风险界限（经验分位数）
    moderate_threshold = down_returns.quantile(0.75)

    # Step 4: 多标签打分（上涨也作为低风险）
    def classify_risk(row):
        x = row["future_min_return"]
        if pd.isna(x):
            return np.nan
        if x >= 0:
            return 0  # Low Risk（上涨或不跌）
        x = abs(x)
        if x > extreme_threshold:
            return 2  # High Risk
        elif x > moderate_threshold:
            return 1  # Medium Risk
        else:
            return 0  # Low Risk（小幅下跌）

    df["risk_level"] = df.apply(classify_risk, axis=1)

    # Debug 输出：阈值 + 标签分布
    print(f"📉 EVT极端风险阈值（q={evt_quantile:.3f}）：{extreme_threshold:.4%}")
    print(f"📊 中风险界限（70%分位）：{moderate_threshold:.4%}")
    print("✅ 各类风险标签分布:")
    print(df["risk_level"].value_counts(dropna=False))

    # Step 5: 清洗数据，保留最后两天用于预测
    if len(df) > 2:
        df_train = df.iloc[:-2].dropna(subset=["future_min_return", "risk_level"])
        df_tail = df.iloc[-2:]
        df_final = pd.concat([df_train, df_tail], axis=0)
    else:
        df_final = df.dropna(subset=["future_min_return", "risk_level"])

    print(f"📦 合并后数据量: {len(df_final)} 行")
    return df_final



def update_features(pair, features_path, exchange_df, macro_df):
    """更新特征数据"""
    # 加载现有特征数据（如果存在）
    if os.path.exists(features_path):
        try:
            features_df = pd.read_csv(features_path, parse_dates=["date"])
        except Exception as e:
            print(f"⚠️ 加载现有特征数据失败: {e}")
            features_df = pd.DataFrame()
    else:
        features_df = pd.DataFrame()

    # 生成新特征
    exchange_df = load_exchange_data(exchange_df)
    exchange_df = add_technical_indicators(exchange_df)
    macro_df = load_macro_data(macro_df)
    new_features = merge_features(exchange_df, macro_df, pair)

    # 追加新特征并去重
    if not new_features.empty:
        print(f"📥 生成新特征数据，行数: {len(new_features)}")
        features_df = pd.concat([features_df, new_features], ignore_index=True)
        features_df = features_df.drop_duplicates(subset=['date'], keep='last')
        features_df = features_df.sort_values("date")
        print(f"📊 合并后数据行数: {len(features_df)}")
        features_df.to_csv(features_path, index=False)
        print(f"✅ 更新特征数据保存至: {features_path}")
    else:
        print("⚠️ 未生成新特征数据，使用现有数据。")

    return features_df

if __name__ == "__main__":
    # 测试模式：加载现有数据并生成特征
    pair = "USD/CNY"
    root = os.path.dirname(os.path.dirname(__file__))
    exchange_data_path = os.path.join(root, "data", "usd_cny.csv")
    macro_data_path = os.path.join(root, "data", "macro.csv")
    features_path = os.path.join(root, "data", f"features_{pair.lower().replace('/', '_')}.csv")

    exchange_df = pd.read_csv(exchange_data_path, parse_dates=["date"])
    macro_df = pd.read_csv(macro_data_path, parse_dates=["date"])
    df_final = update_features(pair, features_path, exchange_df, macro_df)