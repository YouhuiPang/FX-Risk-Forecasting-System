# scripts/model_trainer.py
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold


def clean_data(X):
    print(f"清洗数据前，行数: {len(X)}, 列数: {len(X.columns)}")
    X = X.replace([np.inf, -np.inf], np.nan)
    max_float = np.finfo(np.float64).max
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].where(X[numeric_cols].abs() <= max_float, np.nan)
    key_cols = [col for col in X.columns if col not in ['fed_funds_rate', 'treasury_10y', 'cpi', 'usd_index']]
    X = X.dropna(subset=key_cols)
    print(f"清洗数据后，行数: {len(X)}, 列数: {len(X.columns)}")
    return X


def add_macro_anomaly_features(X, macro_cols):
    print(f"添加宏观异常特征，初始列: {list(X.columns)}")
    X_new = X.copy()
    for col in macro_cols:
        if col not in X_new.columns:
            print(f"宏观指标 {col} 不存在，跳过...")
            continue
        mean_val = X_new[col].mean()
        std_val = X_new[col].std()
        if std_val == 0 or pd.isna(std_val):
            z_scores = pd.Series(0, index=X_new.index)
            print(f"宏观指标 {col} 标准差为0，未计算Z分数")
        else:
            z_scores = (X_new[col] - mean_val) / std_val
            X_new[f'{col}_anomaly_weighted'] = X_new[col] * (z_scores.abs() > 2).astype(int) * z_scores.abs()
            if 'rolling_std_20' in X_new.columns:
                X_new[f'{col}_x_vol'] = X_new[col] * X_new['rolling_std_20']
    print(f"添加宏观异常特征后，列数: {len(X_new.columns)}")
    return X_new


def custom_risk_score(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    recall_high = report.get("2", {}).get("recall", 0)
    recall_medium = report.get("1", {}).get("recall", 0)
    return 0.5 * recall_high + 0.5 * recall_medium

scorer = make_scorer(custom_risk_score, greater_is_better=True)



def train_and_save_model(pair, features_path):
    print(f"\n训练模型：{pair}（多标签 risk_level）")

    # === 加载并预处理数据 ===
    df = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date")
    if len(df) > 2:
        df = df.iloc[:-2]
    X = df[[col for col in df.columns if col not in ["date", "future_min_return", "is_high_risk", "risk_level"]]]
    y = df["risk_level"]
    if y.nunique() < 2:
        print(" 标签不足，无法训练模型"); return

    # 划分数据
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

    # 清洗 & 特征处理
    for name, X_set, y_set in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        X_set = clean_data(X_set)
        X_set = add_macro_anomaly_features(X_set, ['fed_funds_rate', 'treasury_10y', 'cpi', 'usd_index'])
        if name == "train": X_train, y_train = X_set, y_set.loc[X_set.index]
        elif name == "val": X_val, y_val = X_set, y_set.loc[X_set.index]
        else: X_test, y_test = X_set, y_set.loc[X_set.index]

    # 去除常数特征
    constant_cols = X_train.columns[X_train.nunique() <= 1]
    if len(constant_cols) > 0:
        X_train = X_train.drop(columns=constant_cols)
        X_val = X_val.drop(columns=constant_cols)
        X_test = X_test.drop(columns=constant_cols)

    # 特征选择
    selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()].tolist()
    print(f"\n 选择的特征: {selected_features}")
    X_train = X_train[selected_features]
    X_val = X_val[selected_features]
    X_test = X_test[selected_features]

    # 重采样：SMOTE
    smote = SMOTE(sampling_strategy={1: 300, 2: 300}, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f" SMOTE 后: {np.bincount(y_resampled.astype(int))}")

    # 类别权重
    class_weights = {0: 1.0, 1: 4, 2: 2}
    sample_weight = y_resampled.map(class_weights).values

    # ========== Grid Search ==========
    param_grid = {
        "learning_rate": [0.1],
        "max_depth": [8],
        "n_estimators": [700],
        "reg_alpha": [10],
        "reg_lambda": [10],
        "subsample": [0.8],
        "colsample_bytree": [1.0]
    }

    scorer = make_scorer(f1_score, average='weighted')

    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        verbosity=0,
        random_state=42
    )

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scorer,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_resampled, y_resampled, sample_weight=sample_weight)
    best_model = grid.best_estimator_

    print(f"\n 最佳参数: {grid.best_params_}")
    print(f"最佳加权得分: {grid.best_score_:.4f}")

    cv_scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(X_resampled, y_resampled):
        X_tr, X_val_ = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
        y_tr, y_val_ = y_resampled.iloc[train_idx], y_resampled.iloc[val_idx]
        sw_tr = sample_weight[train_idx]

        model = XGBClassifier(**grid.best_params_, objective="multi:softprob", num_class=3)
        model.fit(X_tr, y_tr, sample_weight=sw_tr)

        y_pred_ = model.predict(X_val_)
        score = f1_score(y_val_, y_pred_, average="weighted")
        cv_scores.append(score)

    print(f"手动加权 5折 F1-weighted: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    # ========== Final Evaluation ==========
    y_train_pred = best_model.predict(X_train)
    print("\n📊 训练集分类结果：")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred, target_names=["Low", "Medium", "High"]))

    y_val_pred = best_model.predict(X_val)
    print("\n📊 验证集分类结果：")
    print(confusion_matrix(y_val, y_val_pred))
    print(classification_report(y_val, y_val_pred, target_names=["Low", "Medium", "High"]))

    y_test_pred = best_model.predict(X_test)
    print("\n📊 测试集分类结果：")
    print(confusion_matrix(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, target_names=["Low", "Medium", "High"]))

    # ========== 保存模型 ==========
    root = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(root, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"risk_model_{pair.lower().replace('/', '_')}_xgb_multi.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    feature_path = os.path.join(model_dir, f"selected_features_{pair.lower().replace('/', '_')}.pkl")
    with open(feature_path, "wb") as f:
        pickle.dump(selected_features, f)

    print(f"\n✅ 模型已保存：{model_path}")
    print(f"✅ 特征保存至：{feature_path}")
    return best_model, selected_features


def load_model_and_features(pair):
    """加载模型和特征"""
    root = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(root, "models")
    model_path = os.path.join(model_dir, f"risk_model_{pair.lower().replace('/', '_')}_xgb_new.pkl")
    features_path = os.path.join(model_dir, f"selected_features_{pair.lower().replace('/', '_')}.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)

    return model, selected_features


if __name__ == "__main__":
    pair = "USD/CNY"
    root = os.path.dirname(os.path.dirname(__file__))
    features_path = os.path.join(root, "data", f"features_{pair.lower().replace('/', '_')}.csv")

    # 训练并保存模型
    train_and_save_model(pair, features_path)