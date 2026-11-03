# 第04章 探索性数据分析（EDA）

> EDA（Exploratory Data Analysis）是数据分析的核心环节——在正式建模之前，先把数据"摸清楚"。本章教你如何系统性地探索一份数据。

---

## 4.1 EDA的核心思路

EDA不是"画一堆图"，而是带着问题去探索：

```
数据长什么样？（分布）
├── 单变量分析：每个变量单独看
├── 双变量分析：两个变量之间的关系
└── 多变量分析：多个变量同时看
    → 发现规律
    → 发现异常
    → 提出假设
    → 为后续分析指路
```

**EDA的原则：先看整体，再看局部；先看单变量，再看多变量。**

---

## 4.2 单变量分析

### 4.2.1 数值型变量

**描述统计：**

```python
import pandas as pd
import numpy as np

df = pd.read_csv('ecommerce_data.csv')

# 基本统计量
print(df['order_amount'].describe())

# 更详细的统计
from scipy import stats
print(f"偏度: {stats.skew(df['order_amount'])}")
print(f"峰度: {stats.kurtosis(df['order_amount'])}")
print(f"中位数: {df['order_amount'].median()}")
print(f"众数: {df['order_amount'].mode()[0]}")
print(f"P10: {df['order_amount'].quantile(0.1)}")
print(f"P90: {df['order_amount'].quantile(0.9)}")
print(f"P99: {df['order_amount'].quantile(0.99)}")
```

**直方图：**

```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='order_amount', bins=50, kde=True)
plt.title('订单金额分布')
plt.xlabel('金额（元）')
plt.ylabel('频数')
plt.axvline(df['order_amount'].mean(), color='red', linestyle='--', label=f'均值: {df["order_amount"].mean():.1f}')
plt.axvline(df['order_amount'].median(), color='green', linestyle='--', label=f'中位数: {df["order_amount"].median():.1f}')
plt.legend()
plt.show()
```

**看分布的时候要问：**
- 是正态分布还是偏态分布？
- 有几个峰？（单峰/双峰/多峰）
- 有没有长尾？长尾占多少？
- 均值和中位数差多少？（差很多说明有偏）

### 4.2.2 分类型变量

```python
# 频数统计
print(df['city_tier'].value_counts())
print(df['city_tier'].value_counts(normalize=True).round(4) * 100)

# 柱状图
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='city_tier', order=df['city_tier'].value_counts().index)
plt.title('各城市等级用户数')
plt.ylabel('用户数')
plt.show()
```

**分类变量分析要点：**
- 类别数量：太多（>20）需要合并
- 各类占比：有没有极度不均衡的（99% vs 1%）
- 有没有"其他"类占比过大？

---

## 4.3 双变量分析

### 4.3.1 数值 vs 数值

**散点图：**

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='order_amount', alpha=0.3)
plt.title('年龄 vs 订单金额')
plt.show()

# 相关性系数
print(f"Pearson相关系数: {df['age'].corr(df['order_amount']):.3f}")
print(f"Spearman相关系数: {df['age'].corr(df['order_amount'], method='spearman'):.3f}")
```

**相关系数解读：**

| 相关系数 | 强度 | 例子 |
|----------|------|------|
| 0 ~ 0.3 | 弱 | 年龄和消费金额 |
| 0.3 ~ 0.7 | 中等 | 收入和消费金额 |
| 0.7 ~ 1.0 | 强 | 身高和体重 |

**注意：相关不等于因果！** 冰淇淋销量和溺水人数高度相关，但都是因为夏天。

### 4.3.2 数值 vs 分类

```python
# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='city_tier', y='order_amount')
plt.title('各城市等级的订单金额分布')
plt.show()

# 分组统计
print(df.groupby('city_tier')['order_amount'].agg(['mean', 'median', 'std', 'count']))
```

**小提琴图（更直观的分布）：**

```python
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='city_tier', y='order_amount')
plt.title('各城市等级的订单金额分布（小提琴图）')
plt.show()
```

### 4.3.3 分类 vs 分类

```python
# 交叉表
ct = pd.crosstab(df['city_tier'], df['gender'], normalize='index')
print(ct.round(4) * 100)

# 热力图
plt.figure(figsize=(8, 6))
sns.heatmap(ct, annot=True, fmt='.2%', cmap='Blues')
plt.title('城市等级 vs 性别 占比热力图')
plt.show()
```

---

## 4.4 多变量分析

### 4.4.1 相关系数矩阵

```python
# 数值列的相关矩阵
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('变量相关系数矩阵')
plt.tight_layout()
plt.show()
```

**看相关矩阵的要点：**
- 对角线都是1（自己和自己相关）
- 找深色块：哪些变量高度相关？
- 多重共线性：两个自变量高度相关（>0.8），建模时要注意

### 4.4.2 分组对比

```python
# 按多个维度分组
result = df.groupby(['city_tier', 'gender'])['order_amount'].agg(['mean', 'count'])
print(result.round(2))

# 热力图展示
result_pivot = result['mean'].unstack()
plt.figure(figsize=(8, 6))
sns.heatmap(result_pivot, annot=True, fmt='.0f', cmap='YlOrRd')
plt.title('城市等级 × 性别 × 平均订单金额')
plt.show()
```

### 4.4.3 趋势分析

```python
# 按时间看趋势
daily = df.groupby(df['order_date'].dt.date)['order_amount'].sum()

plt.figure(figsize=(14, 6))
plt.plot(daily.index, daily.values)
plt.title('每日GMV趋势')
plt.ylabel('GMV（元）')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 4.5 EDA的实战框架

### 4.5.1 拿到一份新数据的标准操作流程

**第一步：了解数据规模和结构**

```python
print(f"形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"字段类型:\n{df.dtypes}")
```

**第二步：缺失值和重复值检查**

```python
print(f"缺失值:\n{df.isnull().sum()}")
print(f"重复行: {df.duplicated().sum()}")
```

**第三步：单变量分布探索**

```python
# 数值列
for col in df.select_dtypes(include=[np.number]).columns:
    print(f"\n=== {col} ===")
    print(df[col].describe())

# 分类列
for col in df.select_dtypes(include='object').columns:
    print(f"\n=== {col} ===")
    print(f"唯一值数: {df[col].nunique()}")
    print(df[col].value_counts().head(10))
```

**第四步：双变量关系探索**

```python
# 目标变量 vs 其他变量
target = 'order_amount'
for col in df.select_dtypes(include=[np.number]).columns:
    if col != target:
        corr = df[target].corr(df[col])
        print(f"{target} vs {col}: r={corr:.3f}")
```

**第五步：时间趋势（如果有时间字段）**

```python
# 按日/周/月聚合看趋势
df['month'] = df['order_date'].dt.to_period('M')
monthly = df.groupby('month')['order_amount'].agg(['sum', 'count', 'mean'])
print(monthly)
```

### 4.5.2 EDA报告模板

```markdown
# EDA报告：XXX数据集

## 1. 数据概览
- 数据规模：XX行 × XX列
- 时间范围：XXXX-XX-XX 至 XXXX-XX-XX
- 关键字段：XXX, XXX, XXX

## 2. 数据质量
- 缺失率最高的字段：XXX（XX%）
- 重复行：XX行（XX%）
- 异常值：XXX字段有XX个异常值

## 3. 单变量分布
- 数值变量：
  - 订单金额：右偏分布，均值XX，中位数XX
  - 年龄：近似正态，均值XX
- 分类变量：
  - 城市等级：一线XX%，新一线XX%，二线XX%
  - 性别：男XX%，女XX%

## 4. 关键发现
1. 发现1：XXX（有数据支撑）
2. 发现2：XXX
3. 发现3：XXX

## 5. 下一步建议
- 建议1：XXX
- 建议2：XXX
```

---

## 4.6 常见EDA误区

| 误区 | 正确做法 |
|------|----------|
| 上来就画图，没有目的 | 先带着问题探索 |
| 只看均值不看分布 | 同时看直方图/箱线图 |
| 把相关当因果 | 相关只是线索，不是结论 |
| 忽略样本量 | 小样本的波动不要过度解读 |
| 只看整体不看分组 | 分维度看往往有意外发现 |
| 异常值直接删 | 先理解异常值为什么存在 |

---

## 4.7 本章小结

1. **EDA的核心是"带着问题探索"**，不是为了画图而画图
2. **分析顺序**：单变量 → 双变量 → 多变量 → 时间趋势
3. **数值变量看分布**：直方图、箱线图、描述统计
4. **分类变量看占比**：柱状图、交叉表
5. **相关不等于因果**，相关只是发现线索

> **下一章**：[第05章 统计分析与假设检验](05-statistical-analysis.md) — EDA发现了线索，怎么验证它是不是真的？统计推断和假设检验来帮你。
