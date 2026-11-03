# 第03章 数据清洗与预处理

> 业界共识：数据分析师 **70%-80% 的时间花在数据清洗上**。本章把所有清洗操作拆解成可复用的代码模板。

---

## 3.1 数据质量的四大问题

拿到一份数据集，先做全面诊断：

```
数据质量问题
├── 缺失值（Missing Values）
├── 异常值（Outliers）
├── 重复值（Duplicates）
└── 格式不一致（Inconsistent Formatting）
```

### 3.1.1 第一步：全面诊断

```python
import pandas as pd
import numpy as np

df = pd.read_csv('raw_data.csv')

# ===== 基本信息 =====
print("=== 数据形状 ===")
print(f"行数: {len(df)}, 列数: {len(df.columns)}")

print("\n=== 前5行 ===")
print(df.head())

print("\n=== 字段类型 ===")
print(df.dtypes)

print("\n=== 缺失值统计 ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失数': missing,
    '缺失率%': missing_pct
})
print(missing_df[missing_df['缺失数'] > 0].sort_values('缺失率%', ascending=False))

print("\n=== 重复行数 ===")
print(f"完全重复行: {df.duplicated().sum()}")

print("\n=== 数值列描述统计 ===")
print(df.describe())

print("\n=== 分类列唯一值 ===")
for col in df.select_dtypes(include='object').columns:
    print(f"\n{col}: {df[col].nunique()}个唯一值")
    print(df[col].value_counts().head(10))
```

---

## 3.2 缺失值处理

### 3.2.1 缺失值的三种类型

| 类型 | 含义 | 例子 | 处理方式 |
|------|------|------|----------|
| MCAR | 完全随机缺失 | 用户不小心没填 | 删除/填充都行 |
| MAR | 随机缺失 | 某类用户更不愿填年龄 | 建模填充 |
| MNAR | 非随机缺失 | 高收入的人故意不填收入 | 谨慎处理，可能有偏差 |

### 3.2.2 删除法

```python
# 1. 删除所有有缺失的行（损失数据多，谨慎用）
df_drop = df.dropna()

# 2. 删除缺失超过阈值的行
# 一行中有超过30%的字段缺失，就删掉
df_drop = df.dropna(thresh=len(df.columns) * 0.7)

# 3. 删除缺失超过阈值的列
# 一列有超过50%的缺失，就删掉这列
df_drop = df.dropna(thresh=len(df) * 0.5, axis=1)

# 4. 只在特定列有缺失才删除
df_drop = df.dropna(subset=['user_id', 'order_time'])
```

**什么时候用删除法：**
- 缺失率很低（<5%）且是随机缺失
- 样本量足够大，删了不影响统计效力
- 关键ID字段缺失（这种行本来就没用）

### 3.2.3 填充法

**数值型填充：**

```python
# 均值填充（适合近似正态分布）
df['age'] = df['age'].fillna(df['age'].mean())

# 中位数填充（适合有偏分布、有异常值）
df['income'] = df['income'].fillna(df['income'].median())

# 众数填充（分类变量）
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

# 前向填充（时间序列：用前一天的值）
df['daily_active_users'] = df['daily_active_users'].ffill()

# 后向填充
df['price'] = df['price'].bfill()

# 指定值填充
df['country'] = df['country'].fillna('Unknown')
df['score'] = df['score'].fillna(0)
```

**分组填充（更聪明的做法）：**

```python
# 不要用全局均值，按组内均值填充
# 比如：按城市填充房价
df['house_price'] = df.groupby('city')['house_price'].transform(
    lambda x: x.fillna(x.median())
)

# 按用户分层填充客单价
df['avg_order_value'] = df.groupby('user_level')['avg_order_value'].transform(
    lambda x: x.fillna(x.median())
)
```

### 3.2.4 模型预测填充

```python
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer

# KNN填充：找相似的样本填充
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[['age', 'income', 'spending']]),
    columns=['age', 'income', 'spending']
)

# 多重插补（MICE）：用回归模型预测缺失值
imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[['age', 'income', 'spending']]),
    columns=['age', 'income', 'spending']
)
```

**注意：**
- 简单分析用均值/中位数就够了
- 建模预测填充会引入偏差，要谨慎
- **最好的做法**：加一个"是否缺失"的标记列，保留缺失这个信息

```python
df['income_is_missing'] = df['income'].isnull().astype(int)
df['income'] = df['income'].fillna(df['income'].median())
```

---

## 3.3 异常值处理

### 3.3.1 三种识别方法

**方法1：箱线图法（IQR）**

```python
def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: 下界={lower:.2f}, 上界={upper:.2f}, 异常值数={len(outliers)}")
    return outliers

# 使用
outliers_income = detect_outliers_iqr(df, 'income')
```

**方法2：Z-Score法**

```python
from scipy import stats

def detect_outliers_zscore(df, col, threshold=3):
    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = df[z_scores > threshold]
    print(f"{col}: Z阈值={threshold}, 异常值数={len(outliers)}")
    return outliers

# 使用
outliers_age = detect_outliers_zscore(df, 'age', threshold=3)
```

**方法3：业务规则法**

```python
# 年龄不可能是负数或超过120
df = df[(df['age'] >= 0) & (df['age'] <= 120)]

# 订单金额不可能是负数（退款除外）
df = df[df['order_amount'] >= 0]

# 转化率不可能超过100%
df = df[(df['conversion_rate'] >= 0) & (df['conversion_rate'] <= 1)]
```

### 3.3.2 异常值处理方法

```python
# 方法1：删除
df_clean = df[(df['income'] >= lower) & (df['income'] <= upper)]

# 方法2：截断（Winsorize）- 把极端值缩到边界
from scipy.stats.mstats import winsorize
df['income_winsorized'] = winsorize(df['income'], limits=[0.01, 0.01])
# 上下各1%的极端值被截断到1%和99%分位数

# 方法3：对数变换（适合有偏分布）
df['log_income'] = np.log1p(df['income'])  # log(1+x)，避免log(0)

# 方法4：分箱
df['income_bin'] = pd.qcut(df['income'], q=5, labels=['很低', '较低', '中等', '较高', '很高'])
```

**异常值处理原则：**
- 先搞清楚异常值是怎么来的：是数据错误还是真的极端值？
- 收入Top 0.1%的富豪不是异常值，是真实存在的
- 年龄=200的肯定是数据错误
- **不要一上来就删异常值，先看一眼这些异常值长什么样**

---

## 3.4 重复值处理

```python
# 检查重复
print(f"完全重复行: {df.duplicated().sum()}")
print(f"user_id重复: {df.duplicated(subset=['user_id']).sum()}")

# 查看哪些重复
dupes = df[df.duplicated(subset=['user_id'], keep=False)]
print(dupes.sort_values('user_id').head(20))

# 删除完全重复行
df = df.drop_duplicates()

# 按关键字段去重，保留最新的一条
df = df.sort_values('update_time').drop_duplicates(
    subset=['user_id'], 
    keep='last'  # 保留最后一条（最新的）
)
```

**去重注意事项：**
- 不要盲目全表去重，有些表本来就允许同一用户多条记录（比如订单表）
- 先搞清楚业务主键是什么
- 去重前后对比数量，差异太大要排查

---

## 3.5 格式统一与类型转换

### 3.5.1 日期时间处理

```python
# 字符串转日期
df['order_time'] = pd.to_datetime(df['order_time'])

# 提取日期特征
df['order_date'] = df['order_time'].dt.date
df['order_year'] = df['order_time'].dt.year
df['order_month'] = df['order_time'].dt.month
df['order_day'] = df['order_time'].dt.day
df['order_hour'] = df['order_time'].dt.hour
df['order_weekday'] = df['order_time'].dt.dayofweek  # 0=周一
df['is_weekend'] = df['order_weekday'].isin([5, 6]).astype(int)

# 时间差
df['delivery_duration'] = df['delivery_time'] - df['order_time']
df['delivery_hours'] = df['delivery_duration'].dt.total_seconds() / 3600
```

### 3.5.2 字符串清洗

```python
# 去除首尾空格
df['name'] = df['name'].str.strip()

# 统一大小写
df['email'] = df['email'].str.lower()

# 提取数字
df['phone'] = df['phone'].str.extract(r'(\d+)')

# 替换特殊字符
df['address'] = df['address'].str.replace(r'[^\w\s]', '', regex=True)

# 拆分字段
df[['province', 'city']] = df['region'].str.split('-', expand=True, n=1)

# 条件替换
df['gender'] = df['gender'].map({'M': '男', 'F': '女', 'male': '男', 'female': '女'})
```

### 3.5.3 类型转换

```python
# 数值转字符串
df['user_id'] = df['user_id'].astype(str)

# 字符串转数值（处理千分位）
df['amount'] = df['amount'].str.replace(',', '').astype(float)

# 分类变量转类别类型
df['city_tier'] = df['city_tier'].astype('category')
```

---

## 3.6 特征工程基础

### 3.6.1 分箱（Binning）

```python
# 等宽分箱
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100], 
                       labels=['少年', '青年', '中年', '老年'])

# 等频分箱
df['income_bin'] = pd.qcut(df['income'], q=4, 
                           labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

### 3.6.2 独热编码

```python
# 独热编码（类别数少的时候用）
df = pd.get_dummies(df, columns=['city_tier'], prefix='city')

# 标签编码（类别多的时候用）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])
```

### 3.6.3 衍生指标

```python
# 客单价
df['aov'] = df['total_amount'] / df['order_count']

# 复购率
df['is_repeat'] = (df['order_count'] > 1).astype(int)

# 时间衰减
from datetime import datetime
latest_date = df['last_order_time'].max()
df['days_since_last_order'] = (latest_date - df['last_order_time']).dt.days
```

---

## 3.7 数据清洗的完整Pipeline

把所有步骤串成一个可复用的函数：

```python
def clean_data(df):
    """数据清洗完整Pipeline"""
    
    # 1. 基本诊断
    print(f"原始数据: {len(df)}行, {len(df.columns)}列")
    
    # 2. 列名标准化
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    # 3. 删除完全重复行
    df = df.drop_duplicates()
    print(f"去重后: {len(df)}行")
    
    # 4. 处理缺失值
    for col in df.columns:
        missing_pct = df[col].isnull().mean()
        if missing_pct > 0.5:
            df = df.drop(columns=[col])
            print(f"删除缺失率>50%的列: {col}")
        elif missing_pct > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna('Unknown')
    
    # 5. 处理异常值（数值列用IQR截断）
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR
        df[col] = df[col].clip(lower, upper)
    
    # 6. 字符串列清洗
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    
    print(f"清洗完成: {len(df)}行, {len(df.columns)}列")
    return df

# 使用
df_clean = clean_data(df_raw)
```

---

## 3.8 本章小结

1. **数据清洗占70%时间**，但它是分析结论可靠的前提
2. **缺失值先看类型再决定**：随机缺失好处理，非随机缺失要谨慎
3. **异常值不要上来就删**：先看是数据错误还是真实极端值
4. **格式统一**是最容易被忽略但最影响后续分析的步骤
5. **把清洗流程写成可复用函数**，不要每次重新写一遍

> **下一章**：[第04章 探索性数据分析（EDA）](04-exploratory-data-analysis.md) — 数据洗干净了，怎么开始分析？单变量、双变量、多变量，一步步带你摸清数据的脾气。
