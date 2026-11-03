# 第21章 监督学习：回归算法

> 预测连续值是分析师最常遇到的建模任务：预测销量、预测房价、预测用户LTV。本章把常用的回归算法从原理到代码全部讲透，最后用一个完整项目串起来。

---

## 21.1 回归问题概述

**回归（Regression）**：预测目标变量是连续数值的问题。

常见的回归场景：
- 电商：预测下个月销售额、预测订单金额
- 房产：预测房价、预测租金
- 金融：预测用户LTV、预测违约损失
- 交通：预测客流量、预测出行时长

**回归 vs 分类的区别**：
- 回归输出：一个具体数值（如房价 = 52.3万）
- 分类输出：一个类别（如房价 = "高"）或概率（如违约概率 = 12%）

参见第19章《机器学习导论》，回顾三大问题类型和建模流程。

### 21.1.1 回归问题的业务价值

回归预测在业务中无处不在，而且往往直接和钱挂钩：

- **销量预测**：告诉供应链备多少货——备少了缺货损失销售额，备多了压库存
- **用户LTV预测**：给每个用户算"这辈子能花多少钱"——决定营销预算怎么分配
- **违约金额预测**：预测每个逾期用户大概会损失多少钱——决定催收策略优先级
- **房价/租金预测**：估值模型，直接影响房产交易定价

**为什么回归比分类更"硬核"？** 因为分类只需要判断"是或否"，而回归要猜一个具体数字，精度要求高得多。

---

## 21.2 线性回归

### 21.2.1 原理

线性回归假设特征和目标之间是线性关系：

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

模型要找一组最优的权重w和截距b，让所有样本的预测值和真实值之间的误差最小。用的方法叫**最小二乘法**：最小化残差平方和。

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# 示例：用面积和房间数预测房价
X = np.array([[50, 1], [70, 2], [90, 2], [120, 3], [150, 3]])
y = np.array([80, 120, 150, 200, 260])  # 单位：万元

model = LinearRegression()
model.fit(X, y)

# 查看系数和截距
print(f"系数: {model.coef_}")      # [0.5, 15.2] → 面积每多1平米，房价涨0.5万；多一个房间，涨15.2万
print(f"截距: {model.intercept_}")  # 约 10.3

# 预测一个新房子：80平米，2个房间
new_house = np.array([[80, 2]])
print(f"预测房价: {model.predict(new_house)[0]:.1f} 万")
```

### 21.2.2 假设条件（Gauss-Markov假设）

线性回归不是随便用的，它有几个假设，违反了效果会打折扣：

1. **线性关系**：特征和目标之间确实是线性的
2. **误差独立**：残差之间不相关（时间序列容易违反）
3. **同方差**：残差的方差不随x变化
4. **正态性**：残差服从正态分布
5. **无多重共线性**：特征之间不要高度相关

**实战中怎么办？**
- 数据量足够大时，假设稍微违反也能用
- 先做EDA（参见第04章探索性数据分析），看看散点图有没有明显的非线性
- 多重共线性严重时，用正则化回归（Ridge/Lasso）

### 21.2.3 系数解释

线性回归最大的优点是**可解释性强**：

```
面积系数 = 0.5  → 控制房间数不变，面积每增加1平米，房价平均涨0.5万
房间数系数 = 15.2 → 控制面积不变，每多一个房间，房价平均涨15.2万
```

这对业务方来说太有价值了——你能直接告诉老板："多一个卧室值15万，多10平米值5万。"

---

## 21.3 多项式回归与过拟合

线性回归只能拟合直线，如果真实关系是曲线呢？

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# 生成一个非线性数据（二次函数）
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X + 3 * X**2 + np.random.randn(50, 1) * 5  # y = 3x² + 2x + 噪声

# 线性回归拟合（效果差）
lr = LinearRegression()
lr.fit(X, y)
y_pred_linear = lr.predict(X)
print(f"线性回归 MSE: {mean_squared_error(y, y_pred_linear):.2f}")

# 多项式回归：把x²也作为特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # 原来的一列x变成了两列：x, x²

lr_poly = LinearRegression()
lr_poly.fit(X_poly, y)
y_pred_poly = lr_poly.predict(X_poly)
print(f"多项式回归 MSE: {mean_squared_error(y, y_pred_poly):.2f}")
```

**过拟合的风险**：多项式次数越高，模型越能"弯"，但也越容易过拟合。

```python
# 试试 degree=15 会发生什么
poly_deg15 = PolynomialFeatures(degree=15, include_bias=False)
X_poly15 = poly_deg15.fit_transform(X)
lr_poly15 = LinearRegression()
lr_poly15.fit(X_poly15, y)
# 训练集MSE非常低，但在新数据上会爆炸——这就是过拟合
```

**经验法则**：
- degree=1 → 直线，欠拟合
- degree=2~3 → 常用，大多数业务关系够用
- degree>5 → 大概率过拟合，除非你有很多数据

---

## 21.4 正则化回归

当特征很多、或者特征之间有共线性时，普通线性回归的系数会很不稳定。正则化通过给系数加"惩罚"，让模型更简单、更泛化。

### 21.4.1 Ridge 回归（L2正则化）

```python
from sklearn.linear_model import Ridge

# alpha 是正则化强度，越大惩罚越重，系数越趋近于0
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)

print(f"Ridge系数: {ridge.coef_}")
# 对比普通线性回归，Ridge的系数会更小、更平滑
```

**特点**：
- 让系数趋近于0但不会变成0（不会删掉任何特征）
- 适合特征都有用、只是要抑制波动的场景
- 对多重共线性特别有效

### 21.4.2 Lasso 回归（L1正则化）

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train, y_train)

print(f"Lasso系数: {lasso.coef_}")
# 很多系数会变成0！这意味着Lasso自动做了特征选择
```

**特点**：
- 会把不重要的特征系数直接压到0（相当于自动特征选择）
- 适合特征很多、你怀疑大部分没用的场景
- 可解释性更强（哪些特征被选中了一目了然）

### 21.4.3 ElasticNet

弹性网 = Ridge + Lasso，兼顾两者：

```python
from sklearn.linear_model import ElasticNet

# l1_ratio=0.5 → 一半L1惩罚，一半L2惩罚
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
elastic.fit(X_train, y_train)
```

### 21.4.4 调参 alpha

```python
from sklearn.model_selection import GridSearchCV

# 网格搜索找最优alpha
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

ridge_cv = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train_scaled, y_train)

print(f"最优alpha: {ridge_cv.best_params_['alpha']}")
print(f"最优分数: {ridge_cv.best_score_:.3f}")
```

**注意**：正则化回归**必须先标准化**！因为不同特征的尺度差异会影响惩罚的公平性。

---

## 21.5 树模型回归

树模型是现在回归任务的主流选择，因为它们不需要标准化、能处理非线性关系、对异常值鲁棒。

### 21.5.1 决策树回归

```python
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)
```

**优点**：可解释性强（可以画出树的决策路径）、不需要标准化
**缺点**：单棵树容易过拟合、不稳定（数据稍微变一点树就不一样）

### 21.5.2 随机森林回归

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,      # 100棵树
    max_depth=10,          # 每棵树最大深度
    min_samples_split=5,  # 节点最少样本数
    random_state=42,
    n_jobs=-1             # 用所有CPU核并行
)
rf.fit(X_train, y_train)

# 特征重要性——分析师最爱的输出
importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)
```

**随机森林为什么比单棵树好？**
- 每棵树随机选一部分特征和样本，避免过拟合
- 多棵树投票，结果更稳定
- 自带特征重要性输出

### 21.5.3 梯度提升回归（XGBoost / LightGBM）

梯度提升树是Kaggle竞赛的大杀器，也是工业界最常用的回归模型。

```python
# LightGBM（比XGBoost更快、内存更小）
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]  # 早停：50轮没提升就停
)

y_pred = model.predict(X_test)
```

**XGBoost vs LightGBM：**

| 对比项 | XGBoost | LightGBM |
|--------|---------|----------|
| 速度 | 中等 | 更快（直方图算法） |
| 内存 | 较高 | 更低 |
| 精度 | 很好 | 很好 |
| 小数据表现 | 更好 | 也不错 |
| 大数据表现 | 好 | 更快 |

**分析师建议**：先用LightGBM起步，速度快、效果好、参数少。如果有特殊需求再上XGBoost。

### 21.5.4 为什么树模型现在这么流行

回顾一下树模型的优势，你就知道为什么它是工业界的首选：

1. **不需要预处理**：不用标准化、不用处理缺失值（树模型自带缺失值处理）、不用管类别编码
2. **能捕捉非线性关系**：自动学到"收入超过1万后，消费意愿反而下降了"这种拐弯的规律
3. **对异常值鲁棒**：极端值只会影响一个分裂点，不会像线性回归那样拉歪整个模型
4. **自带特征重要性**：训练完直接告诉你哪个特征最重要，对业务方太友好了
5. **调参空间大**：想提升效果就调参，总有优化空间

**对比线性模型**：线性模型要做标准化、要检查多重共线性、要处理异常值，而且只能学线性关系。树模型基本就是"扔进去就能用"，效果还更好。

**唯一要注意的是**：树模型不要用在时间序列预测上！树模型不擅长外推——你训练数据里最大的房价是500万，模型预测超过500万的房子时，它只会输出500万这个数字，不会"预测"出600万。时间序列要专门用ARIMA、Prophet或者专门的时序模型。

---

## 21.6 KNN回归

K近邻回归：找最近的K个邻居，取它们的平均值作为预测。

```python
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
# weights='distance' → 距离近的邻居权重更大
knn.fit(X_train_scaled, y_train)  # KNN必须标准化！
```

**优缺点：**
- 优点：简单直观、不需要假设数据分布
- 缺点：预测慢（每个新样本都要算距离）、维度高时效果差、必须标准化

**什么时候用？** 小数据集、特征维度不高、你想做一个baseline模型对比。

---

## 21.7 回归评估指标

### 21.7.1 四大常用指标

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# y_true: 真实值， y_pred: 预测值

# 1. MAE（平均绝对误差）：所有预测值和真实值之差的绝对值的平均
mae = mean_absolute_error(y_true, y_pred)
# 解读：预测房价平均偏差15万

# 2. MSE（均方误差）：误差平方的平均
mse = mean_squared_error(y_true, y_pred)
# 缺点：平方后量纲变了，不好解释

# 3. RMSE（均方根误差）：MSE开根号，和目标同量纲
rmse = np.sqrt(mse)
# 解读：预测房价平均偏差约18万（对大误差更敏感）

# 4. R²（决定系数）：模型解释了多少比例的方差
r2 = r2_score(y_true, y_pred)
# 解读：R²=0.85 → 模型解释了85%的房价波动
# 范围：(-∞, 1]，越接近1越好；小于0说明还不如直接猜均值
```

### 21.7.2 怎么选指标

| 指标 | 特点 | 适用场景 |
|------|------|----------|
| MAE | 对异常值不敏感 | 业务关心平均误差、有离群值 |
| RMSE | 对大误差敏感 | 大误差代价高（比如预测用电量，偏差大了要赔很多钱） |
| R² | 无量纲、好比较 | 不同数据集之间比较模型效果 |
| MAPE | 百分比误差，业务好理解 | 预测值量级变化大时（比如不同城市的销量） |

```python
# MAPE（平均绝对百分比误差）
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 解读：MAPE=12% → 预测平均偏差12%
```

**实战建议**：
- 给业务方汇报：用MAE或MAPE（他们能直接听懂）
- 给技术团队调参：用RMSE（对大误差更敏感）
- 永远同时看多个指标，不要只看一个

---

## 21.8 实战项目：加州房价预测

用 sklearn 自带的加州房价数据集，完整走一遍回归项目流程。

```python
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb

# ========== 1. 加载数据 ==========
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='MedHouseVal')

print(f"数据形状: {X.shape}")
print(f"特征: {data.feature_names}")
print(X.describe())

# ========== 2. 划分训练集和测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 3. 标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 训练多个模型对比 ==========
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (alpha=1)': Ridge(alpha=1.0, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'LightGBM': lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbosity=-1)
}

results = []
for name, model in models.items():
    # 用标准化数据还是原始数据？
    if 'Linear' in name or 'Ridge' in name:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4)
    })

results_df = pd.DataFrame(results).sort_values('R2', ascending=False)
print("\n===== 模型对比 =====")
print(results_df.to_string(index=False))

# ========== 5. 用最好的模型做特征重要性分析 ==========
best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
best_model.fit(X_train, y_train)

importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n===== 特征重要性 =====")
print(importance.to_string(index=False))

# ========== 6. 交叉验证验证稳定性 ==========
cv_scores = cross_val_score(
    best_model, X, y, cv=5, 
    scoring='neg_mean_squared_error'
)
cv_rmse = np.sqrt(-cv_scores)
print(f"\n5折交叉验证 RMSE: {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f}")
```

**预期结果解读：**
- 线性模型 R² 大概在 0.6 左右
- 随机森林 R² 大概在 0.82 左右
- LightGBM R² 大概在 0.84 左右
- 最重要的特征通常是 MedInc（收入中位数）——这很合理，收入越高房价越贵

---

## 21.9 算法选择建议表

| 场景 | 推荐算法 | 理由 |
|------|----------|------|
| 特征少、关系线性、要可解释 | 线性回归 / Ridge | 简单、系数可解释 |
| 特征多、怀疑有冗余特征 | Lasso | 自动做特征选择 |
| 有非线性关系、特征中等数量 | 随机森林 | 不需要调参就能用，效果稳定 |
| 追求最高精度、数据量大 | LightGBM / XGBoost | 竞赛级效果，工业界标配 |
| 数据量小、要做baseline | 线性回归 | 简单快速，跑个基线 |
| 特征维度高、要做特征选择 | Lasso | L1正则化自动筛选 |
| 要向业务方解释模型 | 线性回归 / 决策树 | 可解释性最强 |

**一句话建议**：先用线性回归跑个基线（baseline），然后上随机森林，再追求精度就上LightGBM。不要一上来就上复杂模型，先简单后复杂。

### 21.9.1 调参的基本思路

很多人调参就是瞎试，这里给你一个系统性的思路：

**第一步：先跑默认参数**
- 不要一上来就调参，先用默认值跑一遍，看看基线效果
- 很多时候默认参数就够用了，没必要折腾

**第二步：从影响最大的参数开始调**
- 随机森林：先调 n_estimators（树的数量），再调 max_depth（深度）
- LightGBM：先调 learning_rate（学习率），再调 num_leaves（叶子数），最后调正则化参数

**第三步：用网格搜索或随机搜索**
- 参数少用 GridSearchCV（网格搜索），枚举所有组合
- 参数多用 RandomizedSearchCV（随机搜索），随机采样，效率更高

```python
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

param_dist = {
    'n_estimators': [100, 300, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7, 10],
    'num_leaves': [15, 31, 63],
    'subsample': [0.7, 0.8, 0.9],
}

random_search = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=42, verbosity=-1),
    param_distributions=param_dist,
    n_iter=20,  # 随机试20组
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
random_search.fit(X_train, y_train)

print(f"最优参数: {random_search.best_params_}")
print(f"最优分数: {-random_search.best_score_:.4f}")
```

**调参的坑**：不要在测试集上调参！用验证集（或交叉验证）调参，最后用测试集做最终评估。如果在测试集上调参，相当于偷看了答案，模型上线后效果会大打折扣。

---

## 21.10 本章小结

1. **线性回归**：最基础、最可解释的回归模型，假设特征和目标是线性关系，需要注意多重共线性
2. **多项式回归**：能拟合非线性关系，但次数太高容易过拟合，一般用2-3次
3. **正则化回归**：Ridge（L2）压缩系数、Lasso（L1）自动选特征、ElasticNet两者兼顾，必须先标准化
4. **树模型回归**：随机森林稳定好用、自带特征重要性，LightGBM/XGBoost追求精度时首选
5. **评估指标**：MAE简单好懂、RMSE对大误差敏感、R²看模型解释力，永远看多个指标
6. **实战流程**：加载数据→划分→标准化→多模型对比→选最优→交叉验证→特征重要性分析
7. **算法选择**：先简单后复杂，线性回归做基线，随机森林做主力，LightGBM冲精度

> **下一章：[第22章 监督学习：分类算法](22-supervised-classification.md)** — 预测"会不会"比预测"是多少"更常见。从逻辑回归到SVM，从随机森林到XGBoost分类，还有AUC、混淆矩阵这些你必须搞懂的评估指标。
