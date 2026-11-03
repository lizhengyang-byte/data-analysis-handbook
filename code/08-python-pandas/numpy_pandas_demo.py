"""
第08章 Pandas/NumPy 常用操作演示
直接运行：python numpy_pandas_demo.py
"""
import numpy as np
import pandas as pd

# ============ 1. NumPy 基础 ============
print("=" * 50)
print("1. NumPy 基础")
print("=" * 50)

arr = np.array([1, 2, 3, 4, 5])
print(f"数组: {arr}")
print(f"均值: {arr.mean()}, 标准差: {arr.std():.2f}")
print(f"布尔索引 [>3]: {arr[arr > 3]}")

# 二维数组
matrix = np.random.randint(0, 100, size=(3, 4))
print(f"\n3x4 随机矩阵:\n{matrix}")
print(f"按列求和: {matrix.sum(axis=0)}")
print(f"按行均值: {matrix.mean(axis=1)}")


# ============ 2. Pandas 创建与读写 ============
print("\n" + "=" * 50)
print("2. Pandas DataFrame")
print("=" * 50)

df = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '部门': ['技术', '产品', '技术', '运营', '产品'],
    '薪资': [25000, 20000, 30000, 15000, 22000],
    '入职日期': pd.date_range('2023-01-01', periods=5, freq='6ME')
})
print(df)
print(f"\n形状: {df.shape}")
print(f"各部门人数:\n{df['部门'].value_counts()}")


# ============ 3. 选择与过滤 ============
print("\n" + "=" * 50)
print("3. 选择与过滤")
print("=" * 50)

# 条件过滤
tech = df[df['部门'] == '技术']
print(f"技术部员工:\n{tech}")

# 多条件
high_salary = df[(df['薪资'] >= 20000) & (df['部门'] == '技术')]
print(f"\n技术部薪资>=20k:\n{high_salary}")

# loc / iloc
print(f"\nloc 按标签: {df.loc[0, '姓名']}")
print(f"iloc 按位置: {df.iloc[0, 0]}")


# ============ 4. groupby 实战 ============
print("\n" + "=" * 50)
print("4. groupby 聚合")
print("=" * 50)

dept_stats = df.groupby('部门').agg(
    人数=('薪资', 'count'),
    平均薪资=('薪资', 'mean'),
    最高薪资=('薪资', 'max'),
    薪资总额=('薪资', 'sum')
).round(0)
print(dept_stats)

# transform 用法（同部门平均薪资对比）
df['部门平均薪资'] = df.groupby('部门')['薪资'].transform('mean')
df['薪资与部门均值差'] = df['薪资'] - df['部门平均薪资']
print(f"\n新增列:\n{df[['姓名', '部门', '薪资', '部门平均薪资', '薪资与部门均值差']]}")


# ============ 5. 缺失值处理 ============
print("\n" + "=" * 50)
print("5. 缺失值处理")
print("=" * 50)

df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5]
})
print(f"含缺失数据:\n{df_missing}")
print(f"每列缺失数:\n{df_missing.isnull().sum()}")

# 填充
df_filled = df_missing.copy()
df_filled['A'] = df_filled['A'].fillna(df_filled['A'].mean())
df_filled['B'] = df_filled['B'].fillna(df_filled['B'].median())
print(f"\n填充后:\n{df_filled.round(1)}")


# ============ 6. 时间序列 ============
print("\n" + "=" * 50)
print("6. 时间序列")
print("=" * 50)

dates = pd.date_range('2026-01-01', periods=90, freq='D')
ts = pd.Series(np.random.randn(90).cumsum() + 100, index=dates, name='日活')
print(f"前5天:\n{ts.head()}")
print(f"\n月度汇总:")
monthly = ts.resample('ME').agg(['mean', 'min', 'max'])
print(monthly.round(1))


# ============ 7. merge ============
print("\n" + "=" * 50)
print("7. DataFrame merge")
print("=" * 50)

orders = pd.DataFrame({
    '订单ID': [1, 2, 3, 4, 5],
    '用户ID': [101, 102, 101, 103, 102],
    '金额': [99, 299, 199, 49, 599]
})
users = pd.DataFrame({
    '用户ID': [101, 102, 103, 104],
    '城市': ['北京', '上海', '北京', '深圳']
})

merged = orders.merge(users, on='用户ID', how='left')
print(f"订单关联用户:\n{merged}")

# 按城市汇总
city_summary = merged.groupby('城市').agg(
    订单数=('订单ID', 'count'),
    总金额=('金额', 'sum')
)
print(f"\n按城市汇总:\n{city_summary}")

print("\n✅ 所有演示完成")
