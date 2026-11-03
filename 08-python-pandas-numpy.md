# 第08章 Python数据分析（Pandas/NumPy）

> SQL负责取数，Pandas负责灵活分析。本章是Pandas实战的核心，从基础操作到性能优化全覆盖。

---

## 8.1 Pandas核心对象

### 8.1.1 Series vs DataFrame

```python
import pandas as pd
import numpy as np

# Series：一维带标签的数组
s = pd.Series([100, 200, 300], index=['苹果', '香蕉', '橙子'])
print(s)
print(s['苹果'])  # 按标签取
print(s[0])       # 按位置取

# DataFrame：二维表格
df = pd.DataFrame({
    '水果': ['苹果', '香蕉', '橙子', '葡萄'],
    '价格': [10, 5, 8, 15],
    '销量': [100, 200, 150, 80]
})
print(df)
print(df.shape)    # (4, 3)
print(df.columns)  # Index(['水果', '价格', '销量'], dtype='object')
print(df.index)   # RangeIndex(start=0, stop=4, step=1)
```

### 8.1.2 读写数据

```python
# 读
df_csv = pd.read_csv('data.csv')
df_excel = pd.read_excel('data.xlsx', sheet_name='Sheet1')
df_json = pd.read_json('data.json')
df_sql = pd.read_sql('SELECT * FROM orders', engine)

# 写
df.to_csv('output.csv', index=False, encoding='utf-8-sig')
df.to_excel('output.xlsx', index=False)
df.to_parquet('output.parquet')  # 推荐，压缩比高、速度快
```

---

## 8.2 数据选择与过滤

### 8.2.1 选择列

```python
# 选一列
df['价格']
df.价格  # 列名是合法变量名时可以用点号

# 选多列
df[['水果', '价格']]

# 按位置选列
df.iloc[:, 0:2]  # 前两列

# 按标签选列
df.loc[:, ['水果', '销量']]
```

### 8.2.2 过滤行

```python
# 条件过滤
df[df['价格'] > 8]
df[(df['价格'] > 8) & (df['销量'] > 100)]
df[(df['价格'] > 8) | (df['销量'] > 100)]

# isin
df[df['水果'].isin(['苹果', '香蕉'])]

# between
df[df['价格'].between(5, 10)]

# query（更简洁）
df.query('价格 > 8 and 销量 > 100')
```

### 8.2.3 loc vs iloc

```python
# loc：按标签（行名、列名）
df.loc[0, '水果']           # 第0行，水果列
df.loc[0:2, ['水果', '价格']]  # 第0到2行（包含2）

# iloc：按位置（第几行、第几列）
df.iloc[0, 0]           # 第0行第0列
df.iloc[0:2, 0:2]       # 第0到1行，第0到1列（不包含2）
```

---

## 8.3 分组聚合（GroupBy）

### 8.3.1 基础GroupBy

```python
# 单字段分组
df.groupby('城市')['订单金额'].mean()

# 多字段分组
df.groupby(['城市', '性别'])['订单金额'].agg(['mean', 'median', 'count', 'sum'])

# 不同字段用不同聚合函数
df.groupby('城市').agg({
    '订单金额': ['mean', 'sum', 'count'],
    '用户ID': 'nunique',
    '订单时间': 'max'
})
```

### 8.3.2 常用聚合函数

| 函数 | 含义 |
|------|------|
| count | 非缺失值数量 |
| nunique | 唯一值数量 |
| sum | 求和 |
| mean | 均值 |
| median | 中位数 |
| std | 标准差 |
| min/max | 最小/最大 |
| quantile(0.25) | 分位数 |
| first/last | 第一个/最后一个 |

### 8.3.3 transform vs apply

```python
# transform：和原DataFrame形状一样，常用于加分组统计列
df['城市平均客单价'] = df.groupby('城市')['订单金额'].transform('mean')
df['城市排名'] = df.groupby('城市')['订单金额'].rank(ascending=False)

# apply：更灵活，返回任意形状
def normalize(x):
    return (x - x.mean()) / x.std()

df['标准化金额'] = df.groupby('城市')['订单金额'].transform(normalize)
```

---

## 8.4 合并与连接

### 8.4.1 Merge（类似SQL JOIN）

```python
# 一对一
pd.merge(users_df, orders_df, on='user_id', how='left')

# 多字段关联
pd.merge(df1, df2, left_on=['city', 'date'], right_on=['城市', '日期'], how='inner')

# 指定后缀
pd.merge(price_df, sales_df, on='product_id', suffixes=('_价格', '_销量'))
```

### 8.4.2 Concat（上下/左右拼接）

```python
# 上下拼接（行增加）
pd.concat([df_jan, df_feb, df_mar], ignore_index=True)

# 左右拼接（列增加）
pd.concat([df_info, df_stats], axis=1)
```

### 8.4.3 Pivot / Melt

```python
# Pivot：宽表变长表
pivot_df = df.pivot(index='日期', columns='城市', values='GMV')
# 行=日期，列=城市，值=GMV

# Melt：宽表变长表
melted = df.melt(id_vars=['日期'], var_name='城市', value_name='GMV')
```

---

## 8.5 时间序列处理

### 8.5.1 日期处理

```python
# 转日期类型
df['日期'] = pd.to_datetime(df['日期'])

# 提取时间特征
df['年'] = df['日期'].dt.year
df['月'] = df['日期'].dt.month
df['日'] = df['日期'].dt.day
df['星期'] = df['日期'].dt.dayofweek  # 0=周一
df['是否周末'] = df['星期'].isin([5, 6]).astype(int)

# 按时间重采样
df.set_index('日期').resample('W')['GMV'].sum()  # 按周聚合
df.set_index('日期').resample('M')['GMV'].mean()  # 按月求均值
```

### 8.5.2 移动窗口

```python
# 7日移动平均
df['GMV_7d_ma'] = df['GMV'].rolling(window=7).mean()

# 累计求和
df['GMV_cum'] = df['GMV'].cumsum()

# 环比增长
df['GMV_mom'] = df['GMV'].pct_change()
```

---

## 8.6 性能优化

### 8.6.1 大数据量技巧

```python
# 1. 指定dtypes减少内存
dtypes = {
    'user_id': 'int32',
    'city': 'category',
    'amount': 'float32'
}
df = pd.read_csv('large.csv', dtype=dtypes)

# 2. 分块读取
chunk_iter = pd.read_csv('huge.csv', chunksize=100000)
results = []
for chunk in chunk_iter:
    results.append(chunk.groupby('city')['amount'].sum())
final = pd.concat(results).groupby(level=0).sum()

# 3. 用向量化操作替代循环
# ❌ 慢
results = []
for idx, row in df.iterrows():
    results.append(row['amount'] * 1.08)

# ✅ 快
df['含税金额'] = df['amount'] * 1.08

# 4. 用query/eval
df = df.query('amount > 100 and city == "北京"')
```

### 8.6.2 常用速度对比

| 操作方式 | 速度 | 建议 |
|----------|------|------|
| iterrows() | 极慢 | 不要用 |
| apply() | 慢 | 小数据量可以 |
| 向量化 | 快 | 优先用 |
| eval()/query() | 很快 | 大数据量用 |
| Polars | 极快 | 超大数据推荐 |

---

## 8.7 本章小结

1. **Pandas三剑客**：选择过滤、分组聚合、合并连接，必须熟练
2. **GroupBy的灵魂**：agg算指标，transform加列
3. **时间序列**：resample重采样、rolling移动窗口
4. **性能优化**：向量化替代循环、指定dtypes、分块读取
5. **数据量大到Pandas扛不住时**：上Polars或者回到SQL/Spark

> **下一章**：[第09章 业务数据分析方法论](09-business-analysis-methodology.md) — 工具都会了，怎么用它解决真实业务问题？
