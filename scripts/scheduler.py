import os
import time
import schedule
import datetime
import pandas as pd



def job_update_and_train(pair):
    """
    每日只更新数据和特征，不重新训练模型。
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pair_key = pair.lower().replace("/", "_")
    exchange_data_path = os.path.join(root, "data", f"{pair_key}.csv")
    macro_data_path = os.path.join(root, "data", "macro.csv")
    features_path = os.path.join(root, "data", f"features_{pair_key}.csv")

    print(f"===== 开始数据更新任务：{pair} =====")
    try:
        from scripts.data_fetcher import update_data
    except Exception as e:
        print(f"导入模块失败: {e}")
        return

    df_final = update_data(pair, exchange_data_path, macro_data_path, features_path, fetch_new=True)
    if df_final.empty:
        print("数据更新失败或无新增数据。")
    else:
        print(f"===== 数据更新任务完成：{pair} =====")


def job_predict(pair):
    """
    定时任务：对指定交易对进行风险预测
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_csv_path = os.path.join(root, "data", f"features_{pair.lower().replace('/', '_')}.csv")
    output_csv_path = os.path.join(root, "data", f"predictions_{pair.lower().replace('/', '_')}.csv")

    print(f"===== 开始风险预测任务：{pair} =====")
    try:
        from scripts.predictor import predict_risk
        predict_risk(pair, features_csv_path, output_csv_path)
    except Exception as e:
        print(f"导入预测模块失败: {e}")
        return

    print(f"===== 风险预测任务完成：{pair} =====")


def update_if_not_updated(pair):
    """
    网页调用时触发的数据检查逻辑，仅更新数据，不训练模型。
    """
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        features_path = os.path.join(root, "data", f"features_{pair.lower().replace('/', '_')}.csv")
        df = pd.read_csv(features_path, parse_dates=["date"])
        df = df.sort_values("date")
        last_date = df["date"].iloc[-1].date()
    except Exception as e:
        print(f"读取特征数据失败: {e}")
        last_date = None

    today = datetime.datetime.now().date()
    if last_date != today:
        print(f"最后更新日期为 {last_date}，今天是 {today}，执行 on-demand 更新。")
        from scripts.data_fetcher import update_data
        exchange_data_path = os.path.join(root, "data", f"{pair.lower().replace('/', '_')}.csv")
        macro_data_path = os.path.join(root, "data", "macro.csv")
        features_path = os.path.join(root, "data", f"features_{pair.lower().replace('/', '_')}.csv")
        update_data(pair, exchange_data_path, macro_data_path, features_path, fetch_new=True)
    else:
        print(f"今日数据已更新（最后更新日期：{last_date}），无需 on-demand 更新。")


def main():
    """
    启动调度器，对多个交易对每日定时更新与预测。
    """
    pairs = ["USD/CNY"]  # 可扩展多个交易对

    for pair in pairs:
        schedule.every().day.at("00:42").do(job_update_and_train, pair=pair)
        schedule.every().day.at("00:43").do(job_predict, pair=pair)

    print("调度器已启动，等待执行任务……")
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()