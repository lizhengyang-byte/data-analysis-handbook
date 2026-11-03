-- ============================================================
-- 第07章 SQL 常用查询模板
-- 适用于 MySQL / PostgreSQL / Hive / ClickHouse 等大多数 SQL 引擎
-- ============================================================

-- ------------------------------------------------------------
-- 1. 留存率计算（次日留存 / 7日留存）
-- ------------------------------------------------------------
-- 假设 user_login 表：user_id, login_date
WITH first_login AS (
    SELECT user_id, MIN(login_date) AS first_date
    FROM user_login
    GROUP BY user_id
),
login_cohort AS (
    SELECT f.user_id,
           f.first_date,
           DATEDIFF(u.login_date, f.first_date) AS day_diff
    FROM first_login f
    JOIN user_login u ON f.user_id = u.user_id
)
SELECT first_date AS 注册日期,
       COUNT(DISTINCT user_id) AS 新增用户,
       COUNT(DISTINCT CASE WHEN day_diff = 1 THEN user_id END) * 1.0
           / COUNT(DISTINCT user_id) AS 次日留存率,
       COUNT(DISTINCT CASE WHEN day_diff = 7 THEN user_id END) * 1.0
           / COUNT(DISTINCT user_id) AS 7日留存率
FROM login_cohort
GROUP BY first_date
ORDER BY first_date;


-- ------------------------------------------------------------
-- 2. 累计求和（窗口函数）
-- ------------------------------------------------------------
SELECT date,
       revenue,
       SUM(revenue) OVER (ORDER BY date
                          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
           AS 累计营收
FROM daily_revenue
ORDER BY date;


-- ------------------------------------------------------------
-- 3. 同比环比
-- ------------------------------------------------------------
SELECT month,
       revenue,
       LAG(revenue, 1) OVER (ORDER BY month) AS 上月营收,
       ROUND((revenue - LAG(revenue, 1) OVER (ORDER BY month))
              / LAG(revenue, 1) OVER (ORDER BY month) * 100, 2) AS 环比增速_pct,
       LAG(revenue, 12) OVER (ORDER BY month) AS 去年同月,
       ROUND((revenue - LAG(revenue, 12) OVER (ORDER BY month))
              / LAG(revenue, 12) OVER (ORDER BY month) * 100, 2) AS 同比增速_pct
FROM monthly_revenue
ORDER BY month;


-- ------------------------------------------------------------
-- 4. 每组 Top N
-- ------------------------------------------------------------
SELECT *
FROM (
    SELECT user_id,
           order_amount,
           ROW_NUMBER() OVER (PARTITION BY user_id
                              ORDER BY order_amount DESC) AS rn
    FROM orders
) t
WHERE rn <= 3;  -- 每个用户金额最高的3笔订单


-- ------------------------------------------------------------
-- 5. 连续登录 N 天问题
-- ------------------------------------------------------------
-- 思路：日期减去行号得到"分组标记"，相同标记的连续日期属于同一连续段
WITH login_flag AS (
    SELECT user_id,
           login_date,
           DATE_SUB(login_date,
                    INTERVAL ROW_NUMBER() OVER (PARTITION BY user_id
                                               ORDER BY login_date) DAY
           ) AS grp
    FROM (SELECT DISTINCT user_id, login_date FROM user_login) t
)
SELECT user_id,
       MIN(login_date) AS 连续开始日期,
       MAX(login_date) AS 连续结束日期,
       COUNT(*) AS 连续天数
FROM login_flag
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;  -- 连续登录至少3天


-- ------------------------------------------------------------
-- 6. 行转列（CASE WHEN 聚合）
-- ------------------------------------------------------------
SELECT user_id,
       SUM(CASE WHEN month = '2026-01' THEN amount ELSE 0 END) AS m1,
       SUM(CASE WHEN month = '2026-02' THEN amount ELSE 0 END) AS m2,
       SUM(CASE WHEN month = '2026-03' THEN amount ELSE 0 END) AS m3
FROM user_monthly_spend
GROUP BY user_id;


-- ------------------------------------------------------------
-- 7. 中位数（PostgreSQL 用 percentile_cont）
-- ------------------------------------------------------------
SELECT department,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS 薪资中位数
FROM employee
GROUP BY department;


-- ------------------------------------------------------------
-- 8. 漏斗转化
-- ------------------------------------------------------------
WITH step1 AS (  -- 访问首页
    SELECT DISTINCT user_id FROM events WHERE event_name = 'view_home'
),
step2 AS (  -- 浏览商品
    SELECT DISTINCT user_id FROM events WHERE event_name = 'view_product'
),
step3 AS (  -- 加入购物车
    SELECT DISTINCT user_id FROM events WHERE event_name = 'add_cart'
),
step4 AS (  -- 完成支付
    SELECT DISTINCT user_id FROM events WHERE event_name = 'purchase'
)
SELECT
    (SELECT COUNT(*) FROM step1) AS 首页访问,
    (SELECT COUNT(*) FROM step2) AS 浏览商品,
    (SELECT COUNT(*) FROM step3) AS 加购物车,
    (SELECT COUNT(*) FROM step4) AS 完成支付,
    ROUND((SELECT COUNT(*) FROM step4) * 1.0
          / (SELECT COUNT(*) FROM step1) * 100, 2) AS 整体转化率_pct;
