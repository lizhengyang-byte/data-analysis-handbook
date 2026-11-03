"""
项目1：客户流失预测完整流程
运行：python main.py
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

print("=" * 60)
print("项目1：客户流失预测")
print("=" * 60)

# 1. 加载数据（实际项目中替换为用户行为数据）
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='是否流失')

# 2. 划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. 训练
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4. 评估
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("\n模型评估：")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# 5. 输出高风险用户（示例）
X_test_with_prob = X_test.copy()
X_test_with_prob['流失概率'] = y_prob
high_risk = X_test_with_prob.sort_values('流失概率', ascending=False).head(10)
print("\nTop10 高流失风险用户（演示数据）：")
print(high_risk[['流失概率']].round(3))

print("\n✅ 项目1完成")
