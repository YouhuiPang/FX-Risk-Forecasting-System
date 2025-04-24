import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def plot_curves(pred_csv_path, output_dir):
    df = pd.read_csv(pred_csv_path, parse_dates=["date"])

    if 'risk_level' not in df.columns:
        print("缺少真实标签列 risk_level，无法绘图")
        return

    df = df[df["risk_level"].notna()].copy()

    y_true = df["risk_level"]
    y_score = df[["risk_prob_low", "risk_prob_medium", "risk_prob_high"]]

    class_names = ["Low", "Medium", "High"]
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])  # shape: (n_samples, n_classes)

    # ==== ROC Curve ====
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score.iloc[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend()
    plt.grid()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    print(f"✅ ROC 曲线已保存至: {roc_path}")
    plt.close()

    # ==== Precision-Recall Curve ====
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_score.iloc[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_score.iloc[:, i])
        plt.plot(recall, precision, label=f"{class_name} (AP = {ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multi-Class Precision-Recall Curve")
    plt.legend()
    plt.grid()
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_path)
    print(f"✅ PR 曲线已保存至: {pr_path}")
    plt.close()


if __name__ == "__main__":
    pair = "USD/CNY"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pred_csv_path = os.path.join(root, "data", f"predictions_{pair.lower().replace('/', '_')}.csv")
    output_dir = os.path.join(root, "figures")
    os.makedirs(output_dir, exist_ok=True)

    plot_curves(pred_csv_path, output_dir)
