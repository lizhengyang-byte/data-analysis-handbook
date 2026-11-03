# 第20章 特征工程实战

> 业界有句话："数据和特征决定了机器学习的上限，模型只是在逼近这个上限。" 好的特征比好的模型重要十倍。本章教你怎么把原始业务数据变成模型能用的高质量特征。

---

## 20.1 什么是特征工程

### 20.1.1 定义

**特征工程**：从原始数据中提取、构造、变换出对预测任务有用的输入变量（特征）的过程。

原始数据通常长这样：
```
user_id | register_time | city | order_count | last_login | gender
  1001  |  2024-01-15   | 成都 |      12     | 2026-11-03 |  男
```

模型不能直接吃这种"脏"数据，需要你加工成：
```
account_age_days | city_tier | order_freq_per_week | days_since_login | gender_male
       658       |     2     |        0.18         |        0         |     1
```

### 20.1.2 为什么重要

```
特征工程的价值 = 信息增量 × 模型可利用性
```

- **信息增量**：这个特征有没有包含预测目标的有用信息？
- **模型可利用性**：模型能不能"看懂"这个特征？（比如树模型能处理类别，但线性回归不能）

**一个残酷的事实**：在 Kaggle 竞赛中，前三名的差距往往不在模型，而在特征工程。一个聪明的特征（比如"用户过去7天的复购率"），比换一个复杂模型的提升还大。

### 20.1.3 特征工程的核心思路

做特征工程时，你脑子里要一直问自己一个问题：

> **这个特征，包含了预测目标的什么信息？**

比如你要预测用户会不会流失：
- "用户过去7天登录了几次" → 包含了"活跃度"信息，越不活跃越容易流失
- "用户上次登录是几天前" → 包含了"沉默时长"信息，越久没来越容易流失
- "用户最近30天消费金额" → 包含了"价值感"信息，高价值用户流失损失更大

**每个特征都应该能回答一个业务问题。** 如果你想不出一个特征和目标有什么关系，那它大概率是个噪声特征，加进去只会让模型变差。

### 20.1.4 特征工程的工作量占比

业界有个经验数据：

```
一个完整的ML项目，时间分配大概是：
  数据获取和清洗：  20%
  特征工程：        40%
  模型训练和调参：  20%
  评估和部署：      20%
```

特征工程占了将近一半的时间，这就是为什么说它是ML的核心竞争力。很多新手花大量时间调模型，却不愿意花时间想"还能加什么特征"，这是本末倒置。

---

## 20.2 数值特征处理

### 20.2.1 标准化 vs 归一化

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

data = np.array([[100], [200], [300], [400], [500]])

# 标准化（Z-Score）：均值为0，标准差为1
# 适用：线性模型、神经网络、SVM、PCA
# 不要求数据有界，对异常值相对稳健
scaler = StandardScaler()
print(scaler.fit_transform(data))
# 输出：[-1.26, -0.63, 0, 0.63, 1.26]

# 归一化（Min-Max）：缩放到[0,1]区间
# 适用：图像处理、神经网络输入
# 对异常值敏感，有明确边界时用
minmax = MinMaxScaler(feature_range=(0, 1))
print(minmax.fit_transform(data))
# 输出：[0, 0.25, 0.5, 0.75, 1.0]
```

**选择建议：**
- 树模型（决策树、随机森林、XGBoost）**不需要标准化**——它们对单调变换不敏感
- 线性回归、逻辑回归、SVM、神经网络 **必须标准化**
- 不知道用什么？先用StandardScaler，不会错

### 20.2.2 对数变换

当数据右偏（大部分值小，少数值很大）时，取对数可以让分布更接近正态：

```python
import pandas as pd
import numpy as np

# 比如用户消费金额，大部分人花几十块，少数人花几万
df['log_amount'] = np.log1p(df['amount'])  # log1p = log(1+x)，处理0值更安全

# 效果：把长尾分布拉平，让模型更容易学到规律
```

**什么时候用对数变换：**
- 收入、消费金额、粉丝数、访问量——这类典型右偏数据
- 你怀疑目标变量与特征之间是指数关系而非线性关系
- 线性回归或逻辑回归的残差分析发现异方差时

### 20.2.3 分箱（Binning）

把连续数值切成区间，变成类别特征。好处：
- 抗异常值（一个极端值不会影响太多）
- 能捕捉非线性关系
- 业务可解释性更强

```python
# 等频分箱（按分位数切，每个区间样本量差不多）
df['age_bin'] = pd.qcut(df['age'], q=5, labels=['极低', '低', '中', '高', '极高'])

# 等宽分箱（按数值区间切）
df['income_bin'] = pd.cut(df['income'], bins=[0, 5000, 10000, 20000, np.inf],
                          labels=['低', '中低', '中高', '高'])

# 树模型自带分裂，一般不用手动分箱；线性模型可以试试
```

### 20.2.4 缺失值处理

```python
import pandas as pd

# 方法1：均值/中位数填充（最简单，适合缺失少的情况）
df['age_filled'] = df['age'].fillna(df['age'].median())  # 中位数比均值更抗异常值

# 方法2：建模填充（用其他特征预测缺失值）
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
df[['age', 'income', 'login_days']] = imputer.fit_transform(df[['age', 'income', 'login_days']])

# 方法3：缺失标记（重要！告诉模型"这个值是缺失的"）
df['age_missing'] = df['age'].isnull().astype(int)  # 先标记
df['age_filled'] = df['age'].fillna(df['age'].median())  # 再填充
```

**缺失值处理的实战建议：**
- 缺失率 < 5%：直接删行或简单填充
- 缺失率 5%-30%：填充 + 加缺失标记列
- 缺失率 > 30%：考虑直接删掉这个特征（信息已经太少了）
- **不要盲目填0**：0可能是有意义的（比如"消费0元"和"消费未知"完全是两回事）

---

## 20.3 类别特征处理

### 20.3.1 编码方法对比

| 方法 | 原理 | 适用场景 | 注意事项 |
|------|------|----------|----------|
| One-Hot | 每个类别变一列0/1 | 低基数类别（<10个类别） | 高基数会爆维度 |
| Label Encoding | 类别→数字 | 树模型 | 线性模型不能用（有大小顺序） |
| Target Encoding | 用目标变量的均值替代类别 | 高基数类别 | 容易过拟合，要交叉验证 |
| Count Encoding | 用类别出现次数替代 | 高基数类别 | 简单但信息有限 |

### 20.3.2 One-Hot Encoding

```python
import pandas as pd

# 假设df['city']有['成都', '北京', '上海']
# One-Hot后变成三列：city_成都, city_北京, city_上海

# 方法1：pandas自带
df_onehot = pd.get_dummies(df, columns=['city', 'gender'], drop_first=True)
# drop_first=True 避免多重共线性（比如知道不是成都也不是北京，就一定是上海）

# 方法2：sklearn（适合Pipeline）
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
city_encoded = encoder.fit_transform(df[['city']])
```

**坑点**：如果测试集中出现了训练集没见过的新类别，One-Hot会报错或变成全0。用 `handle_unknown='ignore'` 可以避免报错。

### 20.3.3 Label Encoding（标签编码）

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])  # 成都=0, 北京=1, 上海=2

# 只给树模型用！因为树模型按阈值分裂，数字大小没有实际意义
# 线性模型用Label Encoding = 灾难，因为它会假设"上海(2) > 北京(1) > 成都(0)"
```

### 20.3.4 Target Encoding（目标编码）

高基数类别（比如用户ID、商品ID，有成千上万个类别）不能One-Hot，这时候用Target Encoding：

```python
# 原理：用每个类别对应的目标变量均值来替代类别
# 比如"用户ID=1001"这个类别，历史上购买率是35%，就用0.35替代

# 必须在交叉验证中做，否则会数据泄露！
from sklearn.model_selection import KFold

def target_encode_smooth(train_df, test_df, col, target, smoothing=10):
    """带平滑的Target Encoding，防止小类别过拟合"""
    # 全局均值
    global_mean = train_df[target].mean()
    
    # 每个类别的统计量
    agg = train_df.groupby(col)[target].agg(['mean', 'count'])
    
    # 平滑：样本少的类别向全局均值靠拢
    smooth_mean = (agg['count'] * agg['mean'] + smoothing * global_mean) / (agg['count'] + smoothing)
    
    # 映射
    train_encoded = train_df[col].map(smooth_mean)
    test_encoded = test_df[col].map(smooth_mean).fillna(global_mean)
    
    return train_encoded, test_encoded
```

**为什么要平滑？** 比如某个类别只出现过1次，而且恰好是正例，那这个类别编码值就是1.0——这完全是噪声，模型会过拟合。平滑后会把这个极端值拉向全局均值。

---

## 20.4 时间特征

时间是业务数据里最有信息量的维度之一。原始的时间戳对模型没用，需要你拆解成各种时间特征。

### 20.4.1 基础时间特征

```python
import pandas as pd

df['datetime'] = pd.to_datetime(df['datetime'])

# 从时间戳提取各种时间维度
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['weekday'] = df['datetime'].dt.dayofweek  # 0=周一, 6=周日
df['hour'] = df['datetime'].dt.hour
df['is_weekend'] = (df['weekday'] >= 5).astype(int)
df['is_month_start'] = df['datetime'].dt.is_month_start.astype(int)
df['is_month_end'] = df['datetime'].dt.is_month_end.astype(int)
df['quarter'] = df['datetime'].dt.quarter
```

### 20.4.2 周期特征（正弦/余弦编码）

时间特征有周期性，直接用数字会有问题：周日(6)和周一(0)在数值上差6，但实际上它们是相邻的。

```python
import numpy as np

# 把小时编码成圆周上的坐标，让23点和0点在空间上接近
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 同理处理星期几
df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
```

**什么时候用？** 线性模型用周期编码效果好。树模型其实不太需要，因为它能自己分裂出"小时>=22或小时<=2"这样的规则。

### 20.4.3 节假日和特殊日期特征

业务数据里，节假日往往是销量爆发的节点，必须单独做特征：

```python
import pandas as pd
import holidays

# 用 holidays 库自动生成中国节假日
cn_holidays = holidays.CN(years=[2024, 2025, 2026])

df['is_holiday'] = df['date'].isin(cn_holidays).astype(int)

# 离节假日还有几天（节前效应：大家提前备货）
df['days_to_holiday'] = df['date'].apply(
    lambda x: min((h - x).days for h in cn_holidays if (h - x).days >= 0) 
    if any((h - x).days >= 0 for h in cn_holidays) else 999
)

# 周末、月末、季末这些特殊日期也可以做特征
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
```

**为什么这个特征重要？** 很多电商业务在节前一周销量会暴涨，节后会暴跌。如果模型不知道"今天离国庆还有3天"，它就无法解释这个波动。

### 20.4.4 滞后特征（Lag Features）

预测"明天销量"时，"昨天销量"肯定是最强的特征之一。

```python
# 假设df按日期排序，是一个时间序列
df = df.sort_values('date')

# 滞后特征：昨天的销量、前天的销量、上周同一天的销量
df['sales_lag_1'] = df['sales'].shift(1)   # 昨天
df['sales_lag_7'] = df['sales'].shift(7)   # 上周同一天
df['sales_lag_30'] = df['sales'].shift(30) # 上个月同一天

# 滚动窗口特征：过去7天的平均销量、最大销量、标准差
df['sales_rolling_7d_mean'] = df['sales'].rolling(window=7).mean()
df['sales_rolling_7d_max'] = df['sales'].rolling(window=7).max()
df['sales_rolling_7d_std'] = df['sales'].rolling(window=7).std()
```

**关键注意**：滞后特征和滚动窗口必须用 `shift(1)` 来避免泄露！
```python
# ❌ 错误：滚动窗口包含了当天的数据（泄露）
df['sales_rolling_7d_mean'] = df['sales'].rolling(window=7).mean()

# ✅ 正确：先shift再rolling，确保只用过去的数据
df['sales_rolling_7d_mean'] = df['sales'].shift(1).rolling(window=7).mean()
```

---

## 20.5 文本特征

文本不能直接喂给模型，需要转成数值。

### 20.5.1 词袋模型（Bag of Words）

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "这个产品很好用",
    "质量很差，不推荐",
    "性价比很高，下次还买",
    "物流太慢了"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())  # 词汇表
print(X.toarray())  # 每句话变成一个词频向量
```

### 20.5.2 TF-IDF

词袋的问题是"的"、"了"这种常见词权重太高。TF-IDF会降低常见词的权重，提高有区分度的词的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
# max_features=1000：只保留最重要的1000个词
# ngram_range=(1,2)：考虑1-gram和2-gram（比如"不好用"是一个2-gram）

X_tfidf = tfidf.fit_transform(corpus)
```

**实战建议：**
- 简单文本分类（情感分析、垃圾邮件）：TF-IDF + 逻辑回归就够用
- 复杂文本：上预训练语言模型（BERT等），那已经超出分析师范畴了
- 先做清洗：去掉停用词、特殊符号、统一大小写

---

## 20.6 特征选择

做了一堆特征，不是越多越好。太多无关特征会：
- 增加计算量
- 引入噪声
- 让模型过拟合

### 20.6.1 过滤法（Filter Method）

用统计指标筛选特征，不依赖模型。

```python
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# 方差分析：选和目标变量最相关的K个特征
selector = SelectKBest(score_func=f_classif, k=10)  # 选top10
X_new = selector.fit_transform(X, y)

# 看看哪些特征被选中了
selected_features = X.columns[selector.get_support()]
print(f"选中的特征: {selected_features.tolist()}")
```

### 20.6.2 包装法（Wrapper Method）

用模型本身来评估特征子集的好坏，最典型的是RFE（递归特征消除）：

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# 用随机森林做基模型，逐步剔除最不重要的特征
model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
X_rfe = rfe.fit_transform(X, y)

print(f"被选中的特征: {X.columns[rfe.support_].tolist()}")
```

### 20.6.3 嵌入法（Embedded Method）

模型训练过程中自动做特征选择，最常用的是树模型的特征重要性：

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 特征重要性
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(20))  # 看top20最重要的特征
# 可以直接剔除重要性接近0的特征
```

**三种方法对比：**

| 方法 | 计算成本 | 效果 | 适用场景 |
|------|----------|------|----------|
| 过滤法 | 最低 | 一般 | 快速筛选、特征特别多时 |
| 包装法 | 最高 | 较好 | 特征数中等、追求最优子集 |
| 嵌入法 | 中等 | 好 | 用树模型时直接用 |

---

## 20.7 用 Pipeline 构建可复用特征工程管线

这是最重要的工程实践。把所有特征处理步骤串成一个Pipeline，避免数据泄露，也方便复用。

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 假设我们有这些特征列
numeric_features = ['age', 'income', 'login_days', 'order_count']
categorical_features = ['city', 'gender', 'membership_level']

# 数值特征处理管线
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 中位数填充
    ('scaler', StandardScaler())                    # 标准化
])

# 类别特征处理管线
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 众数填充
    ('onehot', OneHotEncoder(handle_unknown='ignore'))     # One-Hot
])

# 合并：不同特征用不同处理方式
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 完整管线：预处理 + 模型
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 训练和预测（一行搞定，不会泄露！）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf.fit(X_train, y_train)
print(f"测试集准确率: {clf.score(X_test, y_test):.3f}")

# 保存整个Pipeline，部署时直接用
import joblib
joblib.dump(clf, 'churn_model_pipeline.pkl')
```

**Pipeline的好处：**
1. **彻底避免数据泄露**：所有预处理都在训练集上fit，测试集只transform
2. **可复用**：保存成pkl，部署时直接load用
3. **调参方便**：可以用GridSearchCV对Pipeline里的每一步调参

---

## 20.8 常见坑与注意事项

### 20.8.1 测试集信息泄露

```python
# ❌ 经典错误：在全量数据上做特征工程，再划分
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 这里已经看到了测试集的分布
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

# ✅ 正确：先划分，再在训练集上fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler.fit(X_train)  # 只看训练集
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 20.8.2 类别特征分布不一致

训练集里"城市"只有5个类别，测试集突然冒出一个新城市——模型没见过这个类别，预测就会出错。

**应对方法：**
- One-Hot时设 `handle_unknown='ignore'`，新类别变成全0向量
- 把出现次数太少的类别合并成"其他"类
- 部署后监控类别分布，发现新类别及时重新训练

### 20.8.3 特征之间高度相关

两个特征相关性很高（比如"身高"和"体重"），对线性模型来说是冗余信息，会导致系数不稳定。

**应对方法：**
- 计算特征间相关系数矩阵，去掉相关性 > 0.9 的特征之一
- 用PCA降维（参见第19章无监督学习）
- 树模型对特征相关性不敏感，可以不用管

### 20.8.4 特征太多导致维度灾难

加了500个特征，模型训练慢、效果还不如50个特征——这就是维度灾难。

**应对方法：**
- 先做特征筛选，去掉明显没用的（方差接近0、和目标相关性为0）
- 用嵌入法（树模型特征重要性）自动筛选
- 别急着加特征，先确认每个特征真的有信息量

**经验法则**：不是特征越多越好。50个高质量特征 > 500个垃圾特征。

---

## 20.9 本章小结

1. **特征工程决定上限**：好特征比好模型重要十倍，这是业界共识
2. **数值特征**：标准化给线性模型用，对数变换处理右偏数据，缺失值要加标记列
3. **类别特征**：低基数用One-Hot，树模型用Label，高基数用Target Encoding
4. **时间特征**：拆解年/月/日/星期/小时，加滞后和滚动窗口特征，记得shift防泄露
5. **特征选择**：过滤法快、嵌入法准、包装法好但慢，树模型的特征重要性最实用
6. **Pipeline是工程标配**：把所有预处理串起来，彻底避免数据泄露，方便部署
7. **最大的坑是数据泄露**：永远记住——先划分训练测试，再在训练集上做所有fit操作

> **下一章：[第21章 监督学习：回归算法](21-supervised-regression.md)** — 特征准备好了，用什么模型来做回归预测？本章从线性回归讲到XGBoost，覆盖分析师最常用的回归算法。
