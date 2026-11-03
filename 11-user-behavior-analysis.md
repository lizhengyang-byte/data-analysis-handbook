# 第11章 用户行为分析

> 用户打开APP后都干了什么？怎么从海量行为日志里挖出有价值的洞察？本章覆盖用户画像、路径分析、Session分析。

---

## 11.1 用户画像

### 11.1.1 什么是用户画像

**用户画像**：用数据标签描述用户是谁、喜欢什么、行为特征是什么。

**标签体系：**

```
用户标签
├── 基础属性标签
│   ├── 年龄、性别、地域
│   ├── 设备、操作系统
│   └── 注册渠道
├── 行为特征标签
│   ├── 活跃频率（高活/中活/低活/流失）
│   ├── 偏好品类（数码/美妆/服饰）
│   └── 活跃时段（早高峰/晚高峰/深夜）
├── 消费特征标签
│   ├── 消费能力（高/中/低）
│   ├── 客单价区间
│   └── 复购次数
└── 价值标签
    ├── RFM分层
    ├── LTV预估
    └── 流失风险等级
```

### 11.1.2 用户分层

```python
# 按活跃频率分层
def active_level(days_active):
    if days_active >= 20:
        return '高活'
    elif days_active >= 10:
        return '中活'
    elif days_active >= 3:
        return '低活'
    else:
        return '流失'

user_stats['活跃等级'] = user_stats['月活跃天数'].apply(active_level)
```

---

## 11.2 路径分析

### 11.2.1 用户路径还原

把用户的行为按时间串起来，还原完整路径：

```python
# 模拟数据
events = pd.DataFrame({
    'user_id': ['U001', 'U001', 'U001', 'U002', 'U002'],
    'event': ['首页', '商品列表', '商品详情', '首页', '搜索'],
    'timestamp': ['2026-10-01 10:00:00', '2026-10-01 10:01:00', 
                  '2026-10-01 10:03:00', '2026-10-01 11:00:00',
                  '2026-10-01 11:01:00']
})

events['timestamp'] = pd.to_datetime(events['timestamp'])
events = events.sort_values(['user_id', 'timestamp'])

# 拼接路径
events['path'] = events.groupby('user_id')['event'].agg(lambda x: ' → '.join(x))
print(events['path'].unique())
```

### 11.2.2 常见路径模式

- **直接转化路径**：首页 → 商品 → 支付（最短路径）
- **浏览比较路径**：商品A → 商品B → 商品C → 支付
- **搜索路径**：搜索 → 列表 → 详情 → 支付
- **跳出路径**：首页 → 列表 → 离开

---

## 11.3 Session分析

### 11.3.1 什么是Session

**Session**：用户一次连续的访问行为。通常30分钟无操作算作Session结束。

### 11.3.2 Session指标

- Session数：总共有多少次访问
- 平均Session时长：每次访问多久
- 平均页面数：每次访问看几个页面
- 跳出率：只看了一个页面就走的Session占比

```python
# Session切分逻辑
# 按用户排序，计算相邻事件的时间差
events = events.sort_values(['user_id', 'timestamp'])
events['time_diff'] = events.groupby('user_id')['timestamp'].diff().dt.total_seconds()

# 超过30分钟（1800秒）就算新Session
events['new_session'] = (events['time_diff'] > 1800) | (events['time_diff'].isna())
events['session_id'] = events.groupby('user_id')['new_session'].cumsum()
```

---

## 11.4 流失用户分析

### 11.4.1 流失定义

- **流失**：连续N天没有活跃（通常30天）
- **预警**：活跃天数逐渐减少，有流失趋势

### 11.4.2 流失前兆

流失用户在流失前通常有这些行为变化：
- 打开频率下降
- 单次使用时长缩短
- 不再使用核心功能
- 推送点击率下降

---

## 11.5 本章小结

1. **用户画像**是精细化运营的基础，标签体系要分层
2. **路径分析**帮你理解用户怎么逛产品
3. **Session分析**衡量每次访问的质量
4. **流失分析**要找前兆，不要等流失了再召回

> **下一章**：[第12章 销售与营收分析](12-sales-revenue-analysis.md) — GMV怎么拆？客单价怎么涨？复购怎么提升？
