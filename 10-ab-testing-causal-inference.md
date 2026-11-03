# 第10章 A/B测试与因果推断

> 相关不等于因果。A/B测试是证明因果关系的黄金标准。本章从实验设计到结果分析全流程覆盖。

---

## 10.1 A/B测试基础

### 10.1.1 什么是A/B测试

**A/B测试**：把用户随机分成两组，一组用A版本（对照），一组用B版本（实验），对比两组的核心指标差异。

**为什么需要随机分组？**
- 消除用户本身的差异（新老用户、地域分布）
- 消除时间因素的影响
- 保证两组除了实验变量外，其他条件一致

### 10.1.2 A/B测试的标准流程

```
Step 1: 确定假设（H1：新版按钮会提升点击率）
Step 2: 确定核心指标（点击率）
Step 3: 计算需要多少样本量
Step 4: 随机分组，上线实验
Step 5: 跑够时间，收集数据
Step 6: 统计检验，判断是否显著
Step 7: 决策：全量/迭代/放弃
```

---

## 10.2 实验设计

### 10.2.1 确定指标

**核心指标（OEC, Overall Evaluation Criterion）：**
- 1个主要指标：实验成功与否的标准
- 2-3个护栏指标：不能因为提升主要指标而损害其他指标

**例子：**
- 主要指标：按钮点击率
- 护栏指标：页面跳出率（不能升太高）、用户投诉率

### 10.2.2 样本量计算

```python
from statsmodels.stats.power import NormalIndPower
import math

# 参数
baseline_rate = 0.10      # 基准转化率10%
mde = 0.005               # 最小可检测效应：提升0.5个百分点
alpha = 0.05              # 显著性水平
power = 0.8               # 统计效力

# 效应量（Cohen's h）
p1 = baseline_rate
p2 = baseline_rate + mde
h = 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))

analysis = NormalIndPower()
n_per_group = analysis.solve_power(effect_size=h, alpha=alpha, power=power)

print(f"每组需要样本量: {math.ceil(n_per_group):,}")
print(f"总共需要: {math.ceil(n_per_group)*2:,}")
```

**经验值：**
- 检测0.5个百分点的提升（10%→10.5%）：每组约30,000人
- 检测1个百分点的提升（10%→11%）：每组约7,500人
- 检测5个百分点的提升（10%→15%）：每组约300人

### 10.2.3 实验时长

- 至少跑1-2个完整周期（通常1-2周）
- 包含工作日和周末
- 避开节假日、大促等特殊时期

---

## 10.3 结果分析

### 10.3.1 显著性检验

```python
from statsmodels.stats.proportion import proportions_ztest

# A版：980/10000 = 9.8%
# B版：1080/10000 = 10.8%
count = [980, 1080]
nobs = [10000, 10000]

z_stat, p_value = proportions_ztest(count, nobs)

print(f"z统计量: {z_stat:.4f}")
print(f"p值: {p_value:.4f}")
print(f"结论: {'显著' if p_value < 0.05 else '不显著'}")
```

### 10.3.2 效应量与置信区间

```python
from statsmodels.stats.proportion import proportion_confint

p_a = 980/10000
p_b = 1080/10000

ci_a = proportion_confint(980, 10000, alpha=0.05)
ci_b = proportion_confint(1080, 10000, alpha=0.05)

print(f"A版转化率: {p_a:.2%} (95% CI: {ci_a[0]:.2%} ~ {ci_a[1]:.2%})")
print(f"B版转化率: {p_b:.2%} (95% CI: {ci_b[0]:.2%} ~ {ci_b[1]:.2%})")
print(f"提升幅度: {(p_b-p_a)/p_a:.2%}")
```

### 10.3.3 常见陷阱

**1. 偷看（Peeking）**
- 问题：每天看一次结果，看到显著就停
- 后果：假阳性率飙升
- 解决：实验前定好样本量和时间，到了再分析

**2. 辛普森悖论**
- 问题：整体显著，但分人群看不显著
- 解决：做分层分析，但要校正多重检验

**3. novelty effect（新奇效应）**
- 问题：新功能刚上线时大家好奇，效果好，过几天就回落
- 解决：跑足够长的时间

---

## 10.4 因果推断（Observational Study）

当不能做A/B测试时（比如不能随机分配用户），用观察性研究方法。

### 10.4.1 PSM（倾向得分匹配）

**思路：** 给实验组的每个用户，在对照组里找一个特征相似的用户配对。

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

# 计算倾向得分（用户进入实验组的概率）
features = ['age', 'gender', 'city_tier', 'income']
X = df[features]
y = df['is_treatment']  # 是否接受了" treatment"

ps_model = LogisticRegression()
ps_model.fit(X, y)
df['propensity_score'] = ps_model.predict_proba(X)[:, 1]

# 匹配：实验组用户找最接近的对照组用户
treatment = df[df['is_treatment'] == 1]
control = df[df['is_treatment'] == 0]

matched_control = []
for _, t_row in treatment.iterrows():
    # 找倾向得分最接近的对照组用户
    control['distance'] = abs(control['propensity_score'] - t_row['propensity_score'])
    best_match = control.nsmallest(1, 'distance').iloc[0]
    matched_control.append(best_match)

# 对比两组的结果指标
```

### 10.4.2 DID（双重差分）

**思路：** 对比政策前后，实验组和对照组的变化差异。

```
DID = (实验组后 - 实验组前) - (对照组后 - 对照组前)
```

**例子：** 某个城市上线了新功能，对比没上线的城市：
- 上线城市：上线前DAU 10万，上线后12万 → 增长2万
- 没上线城市：同期从8万涨到9万 → 增长1万
- DID效应：2万 - 1万 = 1万（新功能带来的增量）

### 10.4.3 工具变量（IV）

最复杂的方法，用于处理内生性问题。分析师了解概念即可，实际使用需要和统计学家合作。

---

## 10.5 本章小结

1. **A/B测试是因果推断的黄金标准**，能做实验就不要做观察性研究
2. **实验前算样本量**，不要拍脑袋定时间
3. **p值显著还要看效应量**，小提升可能没业务价值
4. **不要偷看结果**，到了预定样本量再分析
5. **不能做实验时**：PSM、DID是常用的替代方案

> **下一章**：[第11章 用户行为分析](11-user-behavior-analysis.md) — 用户在产品里到底干了什么？怎么从行为数据里发现问题？
