# 第15章 机器学习在数据分析中的应用

> 不是所有分析都需要ML。但当数据量大到看不出规律时，ML能帮你发现模式。本章讲分析师最常用的ML方法。

---

## 15.1 什么时候该用ML

### 15.1.1 判断标准

| 情况 | 用什么 |
|------|--------|
| 简单对比、趋势 | Excel / SQL |
| 分组分析、漏斗 | Pandas / SQL |
| 找异常值 | 简单统计规则 |
| 预测未来趋势 | 回归 / 时间序列 |
| 用户分群 | 聚类 |
| 预测是否流失 | 分类 |
| 特征重要性分析 | 随机森林 |

**原则：能用简单方法解决的，不要上ML。**

---

## 15.2 分类：预测用户是否流失

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# 准备数据
features = ['login_days_last_month', 'avg_session_duration', 
            'purchase_count', 'days_since_last_login']
X = df[features]
y = df['is_churned']  # 是否流失

# 划分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 评估
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_prob):.3f}")

# 特征重要性
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

**关键解读：** 哪个特征最重要？比如"days_since_last_login"最重要 → 说明越久没来的用户越容易流失。

---

## 15.3 聚类：用户分群

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 选择特征
features = ['login_freq', 'avg_session_duration', 'spending']
X = df[features]

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means聚类
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# 看每个群体的特征
cluster_profile = df.groupby('cluster')[features].mean()
print(cluster_profile)
```

**怎么选K值（肘部法则）：**

```python
inertias = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 画肘部图，找拐点
import matplotlib.pyplot as plt
plt.plot(range(2, 10), inertias, 'bo-')
plt.xlabel('K值')
plt.ylabel('惯性')
plt.title('肘部法则选K')
plt.show()
```

---

## 15.4 回归：预测销量

```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 准备数据
features = ['price', 'advertising_budget', 'weekend', 'holiday']
X = df[features]
y = df['sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

# 随机森林回归
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# 评估
print(f"线性回归 R²: {r2_score(y_test, y_pred_lr):.3f}")
print(f"随机森林 R²: {r2_score(y_test, y_pred_rf):.3f}")
```

---

## 15.5 时间序列预测

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# 准备时间序列数据
ts = df.set_index('date')['sales'].asfreq('D')

# 训练模型
model = ExponentialSmoothing(ts, seasonal_periods=7, trend='add', seasonal='add')
fitted = model.fit()

# 预测未来30天
forecast = fitted.forecast(30)

print(forecast)
```

---

## 15.6 本章小结

1. **ML是工具不是目的**，能简单解决就不要上ML
2. **分类**用于预测是否流失/是否转化
3. **聚类**用于用户分群
4. **回归**用于预测数值
5. **特征重要性**是分析师最常用的输出——知道哪个因素最重要

> **下一章**：[第16章 真实项目实战案例](16-real-world-case-studies.md) — 四个完整案例，从问题到结论全流程走一遍。
