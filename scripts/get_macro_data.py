import pandas as pd
import os
import datetime
from fredapi import Fred

FRED_API_KEY = "c217561f5280b51db410bc1f21ef86b5"

def fetch_macro_data():
    root = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(root, "data", "macro.csv")
    fred = Fred(api_key=FRED_API_KEY)

    indicators = {
        "FEDFUNDS": "fed_funds_rate",      # 月度
        "DGS10": "treasury_10y",           # 日度
        "CPIAUCSL": "cpi",                 # 月度
    }

    start = pd.to_datetime("2020-01-01")
    end = pd.to_datetime(datetime.datetime.now().date())
    all_dates = pd.date_range(start=start, end=end, freq="D")
    df_all = pd.DataFrame(index=all_dates)

    for code, name in indicators.items():
        print(f"📥 下载 {name}（{code}）...")
        try:
            series = fred.get_series(code)
            series.name = name
            series = series[(series.index >= start) & (series.index <= end)]
            df = series.to_frame()
            df = df.reindex(all_dates).ffill()
            df_all[name] = df[name]
        except Exception as e:
            print(f"⚠️ {code} 抓取失败：{e}")

    df_all.index.name = "date"
    df_all = df_all.reset_index()
    df_all.to_csv(path, index=False)
    print(f"✅ 宏观数据已保存：{path}（{len(df_all)} 行）")

if __name__ == "__main__":
    fetch_macro_data()
