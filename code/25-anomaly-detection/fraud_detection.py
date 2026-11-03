"""
第25章 异常检测：孤立森林识别欺诈交易
运行：python fraud_detection.py
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# ========== 1. 生成合成交易数据 ==========
print("=" * 60)
print("1. 准备数据（合成正常交易+少量欺诈）")
print("=" * 60)

np.random.seed(42)

# 正常交易：金额小、频率正常
n_normal = 900
normal_amount = np.random.lognormal(mean=3, sigma=0.5, size=n_normal)
normal_hour = np.random.normal(14, 3, n_normal).clip(0, 23)
normal_distance = np.random.exponential(2, n_normal)

# 欺诈交易：金额大、深夜、异地
n_fraud = 100
fraud_amount = np.random.lognormal(mean=6, sigma=1, size=n_fraud)
fraud_hour = np.random.choice([2, 3, 4, 22, 23], n_fraud)
fraud_distance = np.random.uniform(50, 500, n_fraud)

X = np.vstack([
    np.column_stack([normal_amount, normal_hour, normal_distance]),
    np.column_stack([fraud_amount, fraud_hour, fraud_distance])
])
y_true = np.array([0] * n_normal + [1] * n_fraud)  # 1=欺诈, 0=正常

df = pd.DataFrame(X, columns=['交易金额', '交易小时', '异地距离'])
print(f"数据集: {df.shape}, 欺诈占比: {y_true.mean():.1%}")
print(df.describe().round(2))

# ========== 2. 标准化 ==========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 3. 孤立森林 ==========
print("\n" + "=" * 60)
print("2. 孤立森林异常检测")
print("=" * 60)

iso = IsolationForest(
    n_estimators=100,
    contamination=0.1,  # 预期异常比例
    random_state=42
)
iso.fit(X_scaled)

# -1 为异常，1 为正常
y_pred_raw = iso.predict(X_scaled)
y_pred = (y_pred_raw == -1).astype(int)

print(classification_report(y_true, y_pred,
                            target_names=['正常', '欺诈']))

# ========== 4. 异常分数分析 ==========
print("\n" + "=" * 60)
print("3. 异常分数分布")
print("=" * 60)

scores = iso.score_samples(X_scaled)
df['异常分数'] = scores
df['真实标签'] = y_true

print("正常交易的异常分数均值:", df[df['真实标签']==0]['异常分数'].mean().round(3))
print("欺诈交易的异常分数均值:", df[df['真实标签']==1]['异常分数'].mean().round(3))

# 分数最低（最异常）的10笔
print("\nTop10 最可疑交易:")
top_suspects = df.nsmallest(10, '异常分数')
print(top_suspects.round(2))

print("\n✅ 异常检测完成")
