# 第07章 SQL数据分析实战

> SQL是数据分析师的第一生产力。本章从基础到进阶，覆盖窗口函数、CTE、性能优化和高频查询模板。

---

## 7.1 SQL基础回顾

### 7.1.1 数据分析常用SQL语法

```sql
-- 完整SELECT语句结构
SELECT 
    column1,
    COUNT(*) AS cnt,
    SUM(amount) AS total_amount
FROM table_name
WHERE dt = '2026-10-01'          -- 过滤行
  AND status = 'paid'
GROUP BY column1                   -- 分组
HAVING COUNT(*) > 100              -- 过滤分组
ORDER BY total_amount DESC         -- 排序
LIMIT 10;                          -- 限制
```

**执行顺序（重要！）：**
```
FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY → LIMIT
```

**为什么重要：** 你在SELECT里定义的别名，不能在WHERE里用，但可以在ORDER BY里用。

### 7.1.2 JOIN的类型

```sql
-- INNER JOIN：只保留两边都匹配的
SELECT a.user_id, b.order_amount
FROM users a
INNER JOIN orders b ON a.user_id = b.user_id;

-- LEFT JOIN：保留左边全部，右边不匹配的补NULL
SELECT a.user_id, b.order_amount
FROM users a
LEFT JOIN orders b ON a.user_id = b.user_id;

-- FULL OUTER JOIN：两边都保留
SELECT a.user_id, b.order_amount
FROM users a
FULL OUTER JOIN orders b ON a.user_id = b.user_id;
```

**JOIN最常见的坑：**

```sql
-- ❌ 一对多JOIN导致行数膨胀
-- users表1个用户，orders表10个订单
-- JOIN后变成10行，SUM(users.balance)会被放大10倍

-- ✅ 先聚合再JOIN
SELECT 
    a.user_id,
    a.balance,
    COALESCE(b.order_cnt, 0) AS order_cnt,
    COALESCE(b.total_amount, 0) AS total_amount
FROM users a
LEFT JOIN (
    SELECT user_id, 
           COUNT(*) AS order_cnt,
           SUM(amount) AS total_amount
    FROM orders
    WHERE dt = '2026-10-01'
    GROUP BY user_id
) b ON a.user_id = b.user_id;
```

---

## 7.2 窗口函数：分析师的终极武器

### 7.2.1 什么是窗口函数

**普通聚合**：GROUP BY后每组只剩一行
**窗口函数**：每组聚合，但保留每一行

```sql
-- 普通聚合：每个城市一行
SELECT city, AVG(salary) AS avg_salary
FROM employees
GROUP BY city;

-- 窗口函数：每个人一行，同时显示所在城市的平均工资
SELECT 
    name,
    city,
    salary,
    AVG(salary) OVER (PARTITION BY city) AS city_avg_salary
FROM employees;
```

### 7.2.2 常用窗口函数

**1. ROW_NUMBER / RANK / DENSE_RANK**

```sql
-- 每个部门工资排名
SELECT 
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rn,
    RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS rk,
    DENSE_RANK() OVER (PARTITION BY department ORDER BY salary DESC) AS drk
FROM employees;
```

| 函数 | 并列时的行为 | 例子 |
|------|-------------|------|
| ROW_NUMBER | 不并列，强行排序 | 1,2,3,4 |
| RANK | 并列跳号 | 1,1,3,4 |
| DENSE_RANK | 并列不跳号 | 1,1,2,3 |

**2. LAG / LEAD：取前N行/后N行的值**

```sql
-- 计算每日GMV的环比增长
SELECT 
    dt,
    gmv,
    LAG(gmv, 1) OVER (ORDER BY dt) AS prev_day_gmv,
    (gmv - LAG(gmv, 1) OVER (ORDER BY dt)) * 1.0 
        / LAG(gmv, 1) OVER (ORDER BY dt) AS mom_growth
FROM daily_gmv_stats;
```

**3. SUM / AVG OVER：累计计算**

```sql
-- 月度累计GMV
SELECT 
    dt,
    daily_gmv,
    SUM(daily_gmv) OVER (ORDER BY dt) AS cumulative_gmv
FROM daily_gmv_stats
WHERE dt BETWEEN '2026-10-01' AND '2026-10-31';
```

**4. NTILE：分位数**

```sql
-- 用户按消费金额分5档
SELECT 
    user_id,
    total_amount,
    NTILE(5) OVER (ORDER BY total_amount) AS amount_quintile
FROM user_total_spending;
```

### 7.2.3 高频实战模板

**模板1：每组Top N**

```sql
-- 每个城市消费最高的前3个用户
SELECT *
FROM (
    SELECT 
        city,
        user_id,
        total_amount,
        ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_amount DESC) AS rn
    FROM user_spending
) t
WHERE rn <= 3;
```

**模板2：留存计算**

```sql
-- 次日留存
WITH user_dates AS (
    SELECT DISTINCT 
        user_id,
        dt AS active_date
    FROM user_behavior_log
    WHERE dt BETWEEN '2026-10-01' AND '2026-10-07'
)
SELECT 
    a.active_date AS register_date,
    COUNT(DISTINCT a.user_id) AS new_users,
    COUNT(DISTINCT b.user_id) AS retained_next_day,
    COUNT(DISTINCT b.user_id) * 1.0 / COUNT(DISTINCT a.user_id) AS retention_1d
FROM user_dates a
LEFT JOIN user_dates b 
    ON a.user_id = b.user_id 
    AND b.active_date = DATE_ADD(a.active_date, 1)
WHERE a.active_date = '2026-10-01'
GROUP BY a.active_date;
```

**模板3：连续登录天数**

```sql
-- 找出连续登录7天的用户
WITH login_dates AS (
    SELECT DISTINCT user_id, dt
    FROM user_behavior_log
    WHERE dt BETWEEN '2026-10-01' AND '2026-10-31'
),
grouped AS (
    SELECT 
        user_id,
        dt,
        DATE_SUB(dt, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY dt)) AS grp
    FROM login_dates
)
SELECT 
    user_id,
    COUNT(*) AS consecutive_days
FROM grouped
GROUP BY user_id, grp
HAVING COUNT(*) >= 7;
```

---

## 7.3 CTE与子查询

### 7.3.1 WITH语句（CTE）

```sql
-- 复杂查询用WITH拆成步骤
WITH 
-- 第一步：取10月的订单
oct_orders AS (
    SELECT user_id, amount, order_time
    FROM orders
    WHERE dt BETWEEN '2026-10-01' AND '2026-10-31'
      AND status = 'paid'
),
-- 第二步：按用户聚合
user_stats AS (
    SELECT 
        user_id,
        COUNT(*) AS order_cnt,
        SUM(amount) AS total_amount,
        AVG(amount) AS avg_amount
    FROM oct_orders
    GROUP BY user_id
),
-- 第三步：和用户表关联
user_info AS (
    SELECT 
        u.user_id,
        u.city,
        u.register_date,
        s.order_cnt,
        s.total_amount
    FROM users u
    LEFT JOIN user_stats s ON u.user_id = s.user_id
)
SELECT * FROM user_info;
```

**为什么用CTE：**
- 可读性好：一步步来，像编程一样
- 可复用：同一个CTE可以在后面多次引用
- 调试方便：可以单独跑每个CTE看结果

---

## 7.4 SQL性能优化

### 7.4.1 大数据表查询优化

**1. 分区裁剪**

```sql
-- ✅ 指定分区，只扫一个分区
SELECT * FROM dwd_order_detail WHERE dt = '2026-10-01';

-- ❌ 全表扫描，扫所有分区
SELECT * FROM dwd_order_detail WHERE order_time = '2026-10-01';
```

**2. 先过滤再JOIN**

```sql
-- ✅ 先过滤，减少JOIN数据量
SELECT a.user_id, b.amount
FROM (
    SELECT user_id FROM users WHERE register_dt = '2026-10-01'
) a
JOIN (
    SELECT user_id, amount FROM orders WHERE dt = '2026-10-01'
) b ON a.user_id = b.user_id;

-- ❌ 大表直接JOIN
SELECT a.user_id, b.amount
FROM users a JOIN orders b ON a.user_id = b.user_id;
```

**3. 避免SELECT ***

```sql
-- ✅ 只取需要的字段
SELECT user_id, amount, status FROM orders WHERE dt = '2026-10-01';

-- ❌ 取所有字段，浪费内存
SELECT * FROM orders WHERE dt = '2026-10-01';
```

**4. 用EXISTS替代IN**

```sql
-- ✅ 半连接，性能好
SELECT * FROM users u
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.user_id = u.user_id
);

-- ❌ IN大子查询，性能差
SELECT * FROM users u
WHERE u.user_id IN (SELECT user_id FROM orders);
```

### 7.4.2 EXPLAIN分析

```sql
-- 用EXPLAIN看执行计划
EXPLAIN
SELECT * FROM orders WHERE dt = '2026-10-01' AND user_id = 12345;
```

**重点关注：**
- type列：ALL（全表扫描）最差，const/ref最好
- key列：有没有用到索引
- rows列：预估扫描行数，越少越好

---

## 7.5 高频查询模板大全

### 7.5.1 日报模板

```sql
-- 每日核心指标日报
SELECT 
    dt,
    COUNT(DISTINCT user_id) AS dau,
    COUNT(DISTINCT CASE WHEN event = 'app_open' THEN user_id END) AS open_uv,
    COUNT(*) AS pv,
    COUNT(DISTINCT CASE WHEN event = 'pay_success' THEN user_id END) AS pay_uv,
    SUM(CASE WHEN event = 'pay_success' THEN amount ELSE 0 END) AS gmv
FROM dwd_user_behavior_log
WHERE dt = '2026-10-01'
GROUP BY dt;
```

### 7.5.2 同比环比

```sql
-- 同比环比
WITH monthly AS (
    SELECT 
        SUBSTR(dt, 1, 7) AS month,
        SUM(gmv) AS total_gmv
    FROM daily_gmv
    WHERE dt BETWEEN '2025-10-01' AND '2026-10-31'
    GROUP BY SUBSTR(dt, 1, 7)
)
SELECT 
    month,
    total_gmv,
    LAG(total_gmv, 1) OVER (ORDER BY month) AS prev_month_gmv,
    LAG(total_gmv, 12) OVER (ORDER BY month) AS same_month_last_year,
    (total_gmv - LAG(total_gmv, 1) OVER (ORDER BY month)) 
        / LAG(total_gmv, 1) OVER (ORDER BY month) AS mom_growth,
    (total_gmv - LAG(total_gmv, 12) OVER (ORDER BY month)) 
        / LAG(total_gmv, 12) OVER (ORDER BY month) AS yoy_growth
FROM monthly
ORDER BY month;
```

### 7.5.3 RFM用户分层

```sql
-- RFM分析
WITH user_rfm AS (
    SELECT 
        user_id,
        DATEDIFF('2026-10-31', MAX(order_date)) AS recency,
        COUNT(*) AS frequency,
        SUM(amount) AS monetary
    FROM orders
    WHERE order_date >= '2025-10-31'
    GROUP BY user_id
)
SELECT 
    user_id,
    recency,
    frequency,
    monetary,
    -- R评分：越近分越高
    NTILE(5) OVER (ORDER BY recency DESC) AS r_score,
    -- F评分：越多分越高
    NTILE(5) OVER (ORDER BY frequency) AS f_score,
    -- M评分：越多分越高
    NTILE(5) OVER (ORDER BY monetary) AS m_score
FROM user_rfm;
```

---

## 7.6 本章小结

1. **窗口函数是分析师的核心技能**：ROW_NUMBER、LAG/LEAD、累计求和，必须烂熟于心
2. **CTE（WITH）让复杂查询可读**：像编程一样一步步拆
3. **性能优化三要素**：分区裁剪、先过滤再JOIN、不要SELECT *
4. **把高频查询写成模板**，下次直接用
5. **JOIN膨胀是最常见的坑**：一对多关联前先聚合

> **下一章**：[第08章 Python数据分析（Pandas/NumPy）](08-python-pandas-numpy.md) — SQL取完数，用Python做更灵活的分析和建模。
