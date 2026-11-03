# 第06章 数据可视化

> 一图胜千言。数据分析的最终结论，80%靠可视化来传达。本章教你怎么选图、怎么画图、怎么把图画得让人一眼看懂。

---

## 6.1 图表选型指南

### 6.1.1 按分析目的选图

| 分析目的 | 推荐图表 | 不推荐 |
|----------|----------|--------|
| 看趋势变化 | 折线图 | 饼图 |
| 看大小对比 | 柱状图/条形图 | 饼图（类别多） |
| 看占比构成 | 饼图（≤5类）/ 堆叠柱状图 | 3D饼图 |
| 看分布 | 直方图/箱线图 | 饼图 |
| 看关系 | 散点图/气泡图 | 折线图 |
| 看地理分布 | 地图 | 饼图 |
| 看漏斗 | 漏斗图 | 柱状图 |

### 6.1.2 万能选型矩阵

```
变量类型 × 分析目的
├── 1个变量
│   ├── 数值 → 直方图/箱线图/密度图
│   └── 分类 → 柱状图/饼图
├── 2个变量
│   ├── 数值 vs 数值 → 散点图/折线图
│   ├── 数值 vs 分类 → 柱状图/箱线图
│   └── 分类 vs 分类 → 堆叠柱状图/热力图
└── 3个变量
    ├── 数值 vs 数值 vs 数值 → 气泡图
    └── 数值 vs 分类 vs 分类 → 分组柱状图/折线图
```

---

## 6.2 Matplotlib实战

### 6.2.1 基础设置

```python
import matplotlib.pyplot as plt
import matplotlib as mpl

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False  # 负号显示
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('seaborn-v0_8-whitegrid')
```

### 6.2.2 折线图（趋势）

```python
import pandas as pd
import numpy as np

# 模拟数据
dates = pd.date_range('2026-01-01', periods=30, freq='D')
da1 = np.random.randint(8000, 12000, 30) + np.arange(30) * 50
da2 = np.random.randint(6000, 9000, 30) + np.arange(30) * 30

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates, da1, label='iOS DAU', linewidth=2, color='#FF6B6B')
ax.plot(dates, da2, label='Android DAU', linewidth=2, color='#4ECDC4')

ax.set_title('DAU趋势（2026年1月）', fontsize=14, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('日活用户数')
ax.legend(loc='upper left')

# 标注关键事件
ax.axvline(pd.Timestamp('2026-01-15'), color='gray', linestyle='--', alpha=0.7)
ax.text(pd.Timestamp('2026-01-15'), 12000, '  版本2.0上线', fontsize=10, color='gray')

plt.tight_layout()
plt.show()
```

### 6.2.3 柱状图（对比）

```python
cities = ['一线', '新一线', '二线', '三线', '四线及以下']
users = [4500, 3800, 2800, 1900, 1200]
colors = ['#FF6B6B', '#FFA07A', '#FFD93D', '#6BCB77', '#4D96FF']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(cities, users, color=colors, width=0.6)

# 在柱子上标数值
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 50,
            f'{height:,}', ha='center', va='bottom', fontsize=11)

ax.set_title('各城市等级用户数', fontsize=14, fontweight='bold')
ax.set_ylabel('用户数')
ax.set_ylim(0, 5500)

plt.tight_layout()
plt.show()
```

### 6.2.4 散点图（关系）

```python
age = np.random.normal(30, 8, 200)
income = age * 800 + np.random.normal(0, 2000, 200)

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(age, income, alpha=0.6, s=30, c=income, cmap='viridis')

# 回归线
z = np.polyfit(age, income, 1)
p = np.poly1d(z)
ax.plot(age, p(age), "r--", alpha=0.8, label=f'拟合线: y={z[0]:.0f}x+{z[1]:.0f}')

ax.set_title('年龄 vs 月收入', fontsize=14, fontweight='bold')
ax.set_xlabel('年龄')
ax.set_ylabel('月收入（元）')
ax.legend()
plt.colorbar(scatter, label='收入')

plt.tight_layout()
plt.show()
```

---

## 6.3 Seaborn进阶

### 6.3.1 箱线图+小提琴图

```python
import seaborn as sns

# 模拟数据
np.random.seed(42)
data = {
    'city_tier': np.repeat(['一线', '二线', '三线'], 200),
    'order_amount': np.concatenate([
        np.random.normal(200, 80, 200),
        np.random.normal(150, 60, 200),
        np.random.normal(100, 40, 200)
    ])
}
df = pd.DataFrame(data)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.boxplot(data=df, x='city_tier', y='order_amount', ax=axes[0], palette='Set2')
axes[0].set_title('箱线图：各城市订单金额分布')

sns.violinplot(data=df, x='city_tier', y='order_amount', ax=axes[1], palette='Set2')
axes[1].set_title('小提琴图：各城市订单金额分布')

plt.tight_layout()
plt.show()
```

### 6.3.2 热力图

```python
# 相关系数热力图
np.random.seed(42)
data = np.random.randn(100, 5)
df = pd.DataFrame(data, columns=['DAU', '新增', '留存', 'GMV', '转化率'])

corr = df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('指标相关系数矩阵', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
```

---

## 6.4 配色与设计原则

### 6.4.1 好的配色

```python
# 推荐配色方案
# 1. 单色渐变（适合热力图）
# 2. 分类色（适合柱状图）
# 3. 红蓝对比（适合正负值）

# 常用专业配色
colors_professional = {
    'primary': '#2E86AB',    # 主色：深蓝
    'secondary': '#A23B72',  # 副色：紫红
    'accent': '#F18F01',     # 强调：橙色
    'success': '#3B1F2B',    # 成功：深绿
    'warning': '#C73E1D',    # 警告：红色
}

# 色盲友好配色（Wong, 2011）
colors_colorblind = [
    '#E69F00',  # 橙色
    '#56B4E9',  # 天蓝色
    '#009E73',  # 蓝绿色
    '#F0E442',  # 黄色
    '#0072B2',  # 蓝色
    '#D55E00',  # 橘红色
    '#CC79A7',  # 粉紫色
]
```

### 6.4.2 设计原则

**DO（应该做）：**
- 坐标轴从0开始（柱状图尤其重要）
- 标注关键数字
- 标题写清楚结论，不是"XX趋势"而是"XX指标在9月达到峰值"
- 一张图只讲一个故事
- 用数据墨水比高的图（去掉多余的边框、网格）

**DON'T（不应该做）：**
- 3D图表（扭曲视觉）
- 饼图超过5类
- 彩虹色（不专业）
- 双Y轴（误导读者）
- 花哨的渐变和阴影

---

## 6.5 Plotly交互式图表

```python
import plotly.express as px

# 交互式折线图
df = px.data.gapminder().query("country=='China'")
fig = px.line(df, x='year', y='gdpPercap', title='中国人均GDP变化')
fig.show()

# 交互式散点图（气泡图）
df = px.data.gapminder().query("year==2007")
fig = px.scatter(df, x='gdpPercap', y='lifeExp', 
                 size='pop', color='continent',
                 hover_name='country', size_max=60,
                 title='2007年：人均GDP vs 预期寿命')
fig.show()
```

---

## 6.6 本章小结

1. **先选图再画图**：想清楚要表达什么，再选最合适的图表类型
2. **趋势用折线，对比用柱状，分布用直方图/箱线图，关系用散点图**
3. **好图的标准**：一眼看懂结论，没有多余的装饰
4. **标题要写结论**，不是写"XX趋势"
5. **坐标轴从0开始**，不要用3D和双Y轴

> **下一章**：[第07章 SQL数据分析实战](07-sql-for-data-analysis.md) — 数据分析师的第一生产力：SQL进阶技巧。
