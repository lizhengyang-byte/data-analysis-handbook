"""
第22章 分类实战：客户流失预测（使用乳腺癌数据集演示完整流程）
运行：python churn_prediction.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve)

# ========== 1. 加载数据（替换为真实客户数据时保持同样接口） ==========
print("=" * 60)
print("1. 加载数据（示例：乳腺癌数据集，类比客户二分类）")
print("=" * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='是否流失')  # 1=良性/留存, 0=恶性/流失

print(f"特征数: {X.shape[1]}, 样本数: {X.shape[0]}")
print(f"类别分布:\n{y.value_counts()}")
print(f"正例占比: {y.mean():.1%}")

# ========== 2. 划分 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 3. 训练模型 ==========
print("\n" + "=" * 60)
print("2. 训练两个模型对比")
print("=" * 60)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# ========== 4. 评估 ==========
for name, model, X_ in [('逻辑回归', lr, X_test_s), ('随机森林', rf, X_test)]:
    y_pred = model.predict(X_)
    y_prob = model.predict_proba(X_)[:, 1]

    print(f"\n--- {name} ---")
    print(f"混淆矩阵:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n分类报告:\n{classification_report(y_test, y_pred,
          target_names=['流失', '留存'])}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ========== 5. 特征重要性 ==========
print("\n" + "=" * 60)
print("3. 随机森林 Top10 重要特征")
print("=" * 60)

importance = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importance.sort_values(ascending=False).head(10)
print(top10.round(4))

# ========== 6. 阈值调整（示例） ==========
print("\n" + "=" * 60)
print("4. 阈值调整示例")
print("=" * 60)

y_prob_rf = rf.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_rf)

# 默认阈值0.5
default_pred = (y_prob_rf >= 0.5).astype(int)
# 高召回阈值0.3（减少漏报流失客户）
recall_pred = (y_prob_rf >= 0.3).astype(int)

print(f"阈值0.5 时的精确率: {(y_test[default_pred==1].mean()):.3f}")
print(f"阈值0.3 时的精确率: {(y_test[recall_pred==1].mean()):.3f}")
print("说明：降低阈值会召回更多正例，但精确率下降")

print("\n✅ 分类实战完成")
