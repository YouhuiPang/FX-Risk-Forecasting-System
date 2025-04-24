import yfinance as yf
import pandas as pd
import os
import datetime

def fetch_exchange_rate(symbol="CNY=X", filename="usd_cny.csv"):

    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "data", filename)

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=["date"])
        if not existing.empty:
            start = existing["date"].max() + pd.Timedelta(days=1)
        else:
            existing = pd.DataFrame(columns=["date", "close_price"])
            start = pd.to_datetime("2020-01-01")
    else:
        existing = pd.DataFrame(columns=["date", "close_price"])
        start = pd.to_datetime("2020-01-01")

    end = pd.to_datetime(datetime.datetime.now().date())

    if start > end:
        print("汇率数据已是最新")
        return

    print(f"📥 下载汇率数据：{start.date()} 到 {end.date()}")

    # 下载数据
    data = yf.download(symbol, start=start, end=end, interval="1d", progress=False)
    if data.empty:
        print("⚠️ 无新汇率数据")
        return

    # 清理列名，兼容多层索引
    df = data.reset_index()
    df.columns = [str(col[0]) if isinstance(col, tuple) else str(col) for col in df.columns]
    df = df.rename(columns={"Date": "date", "Close": "close_price"})
    df = df[["date", "close_price"]]
    df["close_price"] = pd.to_numeric(df["close_price"], errors="coerce")
    df = df.dropna(subset=["date", "close_price"])

    # 合并并保存
    combined = pd.concat([existing, df], ignore_index=True)
    combined = combined.drop_duplicates(subset="date").sort_values("date")
    combined.to_csv(path, index=False)
    print(f"✅ 汇率数据已保存：{filename}，共 {len(combined)} 行")

if __name__ == "__main__":
    fetch_exchange_rate()
