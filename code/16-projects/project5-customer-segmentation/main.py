"""
项目5：用户分群（RFM + K-Means）
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("项目5：用户分群")
print("=" * 60)

np.random.seed(42)
n = 500
# 生成RFM数据
recency = np.random.exponential(30, n).clip(1, 180)  # 距今天数
frequency = np.random.poisson(5, n).clip(1, 50)       # 购买次数
monetary = frequency * np.random.uniform(50, 500, n)  # 消费金额

df = pd.DataFrame({
    'Recency': recency.round(0),
    'Frequency': frequency,
    'Monetary': monetary.round(0)
})

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(df)

# K-Means（K=4）
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['群体'] = kmeans.fit_predict(X)

# 各群体特征
profile = df.groupby('群体').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    '群体': 'count'
}).round(1)
profile.columns = ['平均距今天数', '平均购买次数', '平均消费', '用户数']
print("\n各群体画像：")
print(profile)

# 解读（根据RFM经典分群）
print("\n运营建议：")
print("  高价值用户（最近买、频次高、金额大）→ VIP专享、新品优先")
print("  流失风险用户（很久没买、曾经高频）→ 召回优惠券")
print("  一般用户 → 常规促销")
print("  低价值用户 → 低成本触达或不投入")

print("\n✅ 项目5完成")
