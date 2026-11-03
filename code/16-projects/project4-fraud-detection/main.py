"""
项目4：欺诈交易检测
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("项目4：欺诈交易检测")
print("=" * 60)

np.random.seed(42)
n = 1000
# 正常交易
amount = np.random.lognormal(3, 0.5, n)
hour = np.random.normal(14, 3, n).clip(0, 23)

X = np.column_stack([amount, hour])
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

# 孤立森林
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X_s)

scores = model.score_samples(X_s)
predictions = model.predict(X_s)  # -1=异常, 1=正常

df = pd.DataFrame({'金额': amount.round(2), '小时': hour.round(1),
                   '异常分数': scores.round(3), '是否异常': predictions})
anomalies = df[df['是否异常'] == -1]

print(f"\n总交易数: {n}")
print(f"标记为异常: {len(anomalies)} 笔 ({len(anomalies)/n:.1%})")
print("\n可疑交易（分数最低的5笔）：")
print(anomalies.nsmallest(5, '异常分数')[['金额', '小时', '异常分数']].round(2))

print("\n✅ 项目4完成")
