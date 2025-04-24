import os
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


pair_key = "usd_cny"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root, "data")
shap_values_path = os.path.join(data_dir, f"shap_values_{pair_key}.npy")
X_path = os.path.join(data_dir, f"shap_input_{pair_key}.csv")
output_dir = os.path.join(root, "plots")
os.makedirs(output_dir, exist_ok=True)


shap_values_raw = np.load(shap_values_path, allow_pickle=True)  # shape: (n, features, classes)
X = pd.read_csv(X_path)

shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
X = pd.read_csv(X_path)


if isinstance(shap_values, list):
    # 对于多分类：shap_values 是 [class0_array, class1_array, class2_array]
    for i, class_values in enumerate(shap_values):
        if class_values.shape != X.shape:
            raise ValueError(f"Class {i} 的 SHAP 值维度 {class_values.shape} 与特征 X {X.shape} 不一致")
else:
    if shap_values.shape != X.shape:
        raise ValueError(f"SHAP 值维度 {shap_values.shape} 与特征 X {X.shape} 不一致")

class_idx = 1

plt.figure()
shap.summary_plot(shap_values[class_idx], X, show=False, plot_type="bar")
plt.savefig(os.path.join(output_dir, f"shap_summary_bar_class_{class_idx}_{pair_key}.png"))
plt.close()

plt.figure()
shap.summary_plot(shap_values[class_idx], X, show=False)
plt.savefig(os.path.join(output_dir, f"shap_summary_beeswarm_class_{class_idx}_{pair_key}.png"))
plt.close()

print(f"✅ SHAP 解释图已保存至 {output_dir}")
