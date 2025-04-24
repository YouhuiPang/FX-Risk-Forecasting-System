# scripts/data_fetcher.py
import pandas as pd
import os
import datetime
from get_exchange_data import fetch_exchange_rate
from get_macro_data import fetch_macro_data
from feature_engineer import update_features

def fetch_new_data(pair, exchange_data_path, macro_data_path):

    try:
        exchange_df = pd.read_csv(exchange_data_path, parse_dates=["date"])
        current_time = datetime.datetime.now()
        exchange_df = exchange_df[exchange_df['date'] <= current_time]
        exchange_df.to_csv(exchange_data_path, index=False)
        print(f"汇率数据（抓取前）: {len(exchange_df)} 行")
    except Exception:
        exchange_df = pd.DataFrame(columns=["date", "close_price"])
        print("汇率数据文件为空")

    try:
        macro_df = pd.read_csv(macro_data_path, parse_dates=["date"])
        # 删除未来的数据
        macro_df = macro_df[macro_df['date'] <= current_time]
        macro_df.to_csv(macro_data_path, index=False)
        print(f"宏观数据（抓取前）: {len(macro_df)} 行")
    except Exception:
        macro_df = pd.DataFrame(columns=["date", "fed_funds_rate", "treasury_10y", "cpi", "usd_index"])
        print("⚠️ 宏观数据文件为空")

    # 检查数据是否为空
    if exchange_df.empty or macro_df.empty:
        print("⚠️ 数据文件为空，从 2020-01-01 开始抓取所有数据...")
    else:
        last_exchange_date = exchange_df['date'].max()
        last_macro_date = macro_df['date'].max()
        start_date = max(last_exchange_date, last_macro_date)

    end_date = datetime.datetime.now()

    # 检查时间范围（忽略时间部分）
    start_date_date = start_date.date()
    end_date_date = end_date.date()
    if start_date_date > end_date_date:
        print(f"⚠️ 起始日期 {start_date_date} 晚于当前日期 {end_date_date}，可能是系统时间或数据错误，跳过抓取...")
        return exchange_df, macro_df

    # 抓取新数据
    symbol = f"{pair.split('/')[1]}=X"  # 例如 "CNY=X"
    filename = f"{pair.lower().replace('/', '_')}.csv"
    print(f"抓取 {pair} 汇率数据，时间范围: {start_date} 到 {end_date}...")
    fetch_exchange_rate(symbol, filename)  # 移除 period, interval, start_date 参数
    print(f"抓取宏观数据，时间范围: {start_date} 到 {end_date}...")
    fetch_macro_data()  # 移除 start, end_date 参数

    # 重新加载更新后的数据
    exchange_df = pd.read_csv(exchange_data_path, parse_dates=["date"])
    macro_df = pd.read_csv(macro_data_path, parse_dates=["date"])
    print(f"汇率数据（抓取后）: {len(exchange_df)} 行")
    print(f"宏观数据（抓取后）: {len(macro_df)} 行")

    return exchange_df, macro_df

def update_data(pair, exchange_data_path, macro_data_path, features_path, fetch_new=False):
    """更新数据并生成特征"""
    if fetch_new:
        exchange_df, macro_df = fetch_new_data(pair, exchange_data_path, macro_data_path)
    else:
        exchange_df = pd.read_csv(exchange_data_path, parse_dates=["date"])
        macro_df = pd.read_csv(macro_data_path, parse_dates=["date"])

    # 确保 exchange_df 有足够数据
    if len(exchange_df) < 20:
        print(f"⚠️ 汇率数据不足（{len(exchange_df)} 行），需要至少 20 行数据来计算技术指标...")
        return pd.DataFrame()

    # 确保 macro_df 覆盖 exchange_df 的日期范围
    min_exchange_date = exchange_df['date'].min()
    max_exchange_date = exchange_df['date'].max()
    if macro_df['date'].max() < min_exchange_date:
        print(f"⚠️ 宏观数据日期范围（最新 {macro_df['date'].max()}) 未覆盖汇率数据（最早 {min_exchange_date}），无法合并...")
        return pd.DataFrame()

    # 打印日期范围以调试
    print(f"汇率数据日期范围: {min_exchange_date} 到 {max_exchange_date}")
    print(f"宏观数据日期范围: {macro_df['date'].min()} 到 {macro_df['date'].max()}")

    # 调用 feature_engineer.py 的 update_features
    df_final = update_features(pair, features_path, exchange_df, macro_df)
    return df_final


if __name__ == "__main__":
    pair = "USD/CNY"
    root = os.path.dirname(os.path.dirname(__file__))
    filename_base = pair.lower().replace("/", "_")

    exchange_data_path = os.path.join(root, "data", f"{filename_base}.csv")
    macro_data_path = os.path.join(root, "data", "macro.csv")
    features_path = os.path.join(root, "data", f"features_{filename_base}.csv")

    df = update_data(pair, exchange_data_path, macro_data_path, features_path, fetch_new=True)
