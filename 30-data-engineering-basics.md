# 第30章 数据工程基础

> 分析师每天写 SQL，但很少有人系统讲过：数据从业务库到你面前的报表，中间经历了什么。本章讲分析师必须知道的那部分数据工程。

---

## 30.1 数据工程师 vs 数据分析师 vs 数据科学家

很多人搞不清这三个岗位的边界，先一句话定位：

| 角色 | 核心产出 | 关心的问题 |
|---|---|---|
| 数据工程师 | 能查、能算、不丢数据的数据管道 | 数据从哪来、怎么存、怎么调度、怎么不报错 |
| 数据分析师 | 业务洞察、报表、结论 | 业务为什么涨/跌、下一步该干嘛 |
| 数据科学家 | 预测模型、实验设计 | 未来会怎样、怎么优化 |

**类比做饭**：

- 数据工程师：菜市场进货、洗菜、切菜、备料（数据管道）。
- 数据分析师：看菜下饭，告诉你今天该做什么菜（业务分析）。
- 数据科学家：研发新菜谱（建模、实验）。

分析师最容易踩的坑：**拿到一份数据就开跑，从来不问数据是怎么来的、有没有口径问题**。数据工程基础就是帮你看懂上游。

---

## 30.2 数据仓库：OLTP vs OLAP、事实表/维度表

### 30.2.1 OLTP vs OLAP

- **OLTP（Online Transaction Processing）**：业务系统的库，比如 MySQL 的订单表、用户表。设计目标是支持"下单、扣库存"这种高频小事务，表结构高度规范化，避免冗余。
- **OLAP（Online Analytical Processing）**：数据仓库，比如 Hive / ClickHouse / Doris。设计目标是支持"全年销售额按省份、按品类聚合"这种大范围扫描分析。

你在公司写的 99% 的分析 SQL，都是跑在 OLAP 上。

### 30.2.2 事实表与维度表

这是维度建模（Kimball）的核心：

- **事实表（Fact Table）**：存"事件"，比如每一笔订单、每一次点击。通常是长表，一行 = 一个事件，列里是外键（指向维度表）+ 度量值（金额、数量、时长）。
- **维度表（Dimension Table）**：存"描述信息"，比如商品表、用户表、日期表。通常宽而小，一行 = 一个实体。

举例：

```
订单事实表 fact_order
| order_id | user_id | product_id | dt      | amount | quantity |
|----------|---------|------------|---------|--------|----------|
| 10001    | 55      | 23         |2026-11-01| 199.0  | 1        |

用户维度表 dim_user
| user_id | city | gender | register_dt |
|---------|------|--------|-------------|
| 55      | 成都 | 女     | 2024-05-10  |

商品维度表 dim_product
| product_id | category | brand | price |
|----------|----------|-------|-------|
| 23       | 零食     | 三只  | 25    |
```

### 30.2.3 星型模型 vs 雪花模型

- **星型模型**：事实表直接连维度表，维度表不再拆分。最常见，查询简单。
- **雪花模型**：维度表再规范化拆分，比如 `dim_product` 再拆成 `dim_category`。节省存储，但 JOIN 更多，查询更慢。

> 分析师视角：**默认用星型模型**，除非维度表有冗余到必须拆的地步。雪花模型在分析场景里很少见。

---

## 30.3 ETL vs ELT

- **ETL（Extract-Transform-Load）**：先从业务库抽数据 → 在外面（Spark / 脚本）做清洗转换 → 再写进数据仓库。传统数仓（Teradata、Oracle）时代的主流。
- **ELT（Extract-Load-Transform）**：先把原始数据原样抽进仓库（Hive / BigQuery / Snowflake / Doris），然后直接在仓库里用 SQL 做转换。现代云原生数仓的主流。

两者的差别：**转换发生在仓库内还是仓库外**。

| 方式 | 优点 | 缺点 |
|---|---|---|
| ETL | 数据进仓前已规整，仓库干净 | 转换流程不透明，加字段要改管道 |
| ELT | 原始数据保留，灵活，SQL 可追溯 | 仓库承担计算压力，需要好的引擎 |

现在主流是 ELT + dbt（下一节讲），分析师直接写 SQL 做转换。

---

## 30.4 常见工具栈

### 30.4.1 Airflow（调度）

```
# 一个简单的 DAG 描述：每天凌晨跑两个任务
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {'owner': 'analyst',
                'retries': 1,
                'retry_delay': timedelta(minutes=5)}

with DAG('daily_report',
         default_args=default_args,
         schedule_interval='0 2 * * *',   # 每天 02:00
         start_date=datetime(2026,11,1),
         catchup=False) as dag:

    t1 = PythonOperator(task_id='extract_orders',
                         python_callable=extract_orders)
    t2 = PythonOperator(task_id='build_report',
                        python_callable=build_report)
    t1 >> t2
```

Airflow 提供 Web UI：看任务状态、失败重试、任务依赖图、历史运行时间。**它不帮你写数据处理逻辑，只负责"什么时候跑谁、跑挂了怎么办"。**

### 30.4.2 dbt（Data Build Tool）

dbt 让你用 SQL 定义转换逻辑，自动建表、维护依赖、跑测试、出文档。

```sql
-- models/daily_revenue.sql
SELECT dt, SUM(amount) AS revenue
FROM {{ ref('fact_order') }}
WHERE dt >= '2026-01-01'
GROUP BY dt
```

`{{ ref('fact_order') }}` 是 dbt 的引用语法，dbt 自动知道 `daily_revenue` 依赖 `fact_order`，按拓扑顺序执行。

### 30.4.3 Spark（大数据处理）

数据量在亿级以上、或者需要复杂的分布式计算时用。PySpark 写法接近 Pandas：

```python
from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("demo").getOrCreate()
df = spark.read.parquet("s3://data/fact_order/")
df.groupBy("dt").agg(F.sum("amount").alias("revenue")).show()
```

分析师一般不需要写 Spark，知道"Pandas 跑不动时有人用 Spark"就行。

### 30.4.4 ClickHouse / Doris（OLAP 引擎）

列式存储 + 向量化执行，单表亿级数据聚合查询亚秒返回。现在很多公司用 Doris / ClickHouse 做实时报表和用户行为分析的即席查询。

| 工具 | 干什么 |
|---|---|
| Airflow | 调度（什么时候跑） |
| dbt | 转换（怎么算） |
| Spark | 大规模计算（算得动） |
| ClickHouse / Doris | 快速查（查得快） |

---

## 30.5 数据质量：四个维度

数据质量不是"干净"两个字，而是可度量的四个维度：

1. **完整性（Completeness）**：该有的字段有没有缺失。比如 `user_id` 不能空，`amount` 不能 NULL。
2. **一致性（Consistency）**：同一数据在不同表/系统里口径一致。比如订单金额在订单表和支付表里要对得上。
3. **及时性（Timeliness）**：数据该到的时候到了没有。日报早上 8 点前必须就绪。
4. **唯一性（Uniqueness）**：不能重复。订单表里一个 `order_id` 只能出现一次。

### 30.5.1 监控规则示例

```sql
-- 1. 空值率
SELECT
  COUNT_IF(amount IS NULL) * 1.0 / COUNT(*) AS amount_null_rate
FROM fact_order WHERE dt = '2026-11-04';

-- 2. 重复率
SELECT COUNT(*) - COUNT(DISTINCT order_id) AS dup_rows
FROM fact_order WHERE dt = '2026-11-04';

-- 3. 波动检测：今日订单数 vs 过去 7 天同时段均值
SELECT
  today.cnt AS today_cnt,
  avg7.avg_cnt,
  ABS(today.cnt - avg7.avg_cnt) / avg7.avg_cnt AS deviation
FROM (SELECT COUNT(*) cnt FROM fact_order WHERE dt='2026-11-04') today,
     (SELECT AVG(daily_cnt) avg_cnt
      FROM (SELECT dt, COUNT(*) daily_cnt
            FROM fact_order
            WHERE dt BETWEEN '2026-10-28' AND '2026-11-03'
            GROUP BY dt) t) avg7;
-- deviation > 0.3 → 告警
```

这些规则通常沉淀在数据质量平台（Great Expectations / dbt tests / 公司自建），每日自动跑，异常发飞书。

---

## 30.6 分区与分桶：为什么大表要分区

想象一张 10 亿行的订单表。你查"2026-11-04 的订单"，如果没分区，引擎要把 10 亿行全扫一遍。**分区（Partitioning）** 告诉引擎：按 `dt` 把数据切成小目录，查某天只扫那一天的文件。

```sql
-- Hive / Spark 建分区表
CREATE TABLE fact_order (
  order_id   BIGINT,
  user_id    BIGINT,
  amount     DECIMAL(18,2),
  category   STRING
)
PARTITIONED BY (dt STRING)
STORED AS PARQUET;

-- 查询时带分区条件，引擎自动裁剪
SELECT SUM(amount) FROM fact_order
WHERE dt = '2026-11-04';   -- 只扫这一天
```

**分桶（Bucketing）**：在分区内再按某个字段（比如 `user_id`）哈希成 N 份，加速 JOIN 和抽样。分析师层面了解概念即可。

> 经验：日增量超过千万行的表，按天分区；按天还不够小的（比如日志表），再按小时分区。

---

## 30.7 数据建模实战：电商订单分析星型模型

下面是一个可以直接参考的星型模型 DDL（以 Hive / Doris 方言写）：

```sql
-- 1. 事实表：订单
CREATE TABLE fact_order (
  order_id      BIGINT,
  user_id      BIGINT,
  product_id   BIGINT,
  shop_id      BIGINT,
  payment_way  STRING,
  quantity      INT,
  amount        DECIMAL(18,2),
  discount      DECIMAL(18,2),
  pay_amount    DECIMAL(18,2)
)
PARTITIONED BY (dt STRING)
STORED AS PARQUET;

-- 2. 维度表：用户
CREATE TABLE dim_user (
  user_id      BIGINT,
  city         STRING,
  province     STRING,
  gender       STRING,
  age_band     STRING,
  register_dt   STRING,
  user_level   STRING
)
STORED AS PARQUET;

-- 3. 维度表：商品
CREATE TABLE dim_product (
  product_id    BIGINT,
  category_l1   STRING,
  category_l2   STRING,
  brand         STRING,
  price         DECIMAL(18,2)
)
STORED AS PARQUET;

-- 4. 维度表：日期（日历表）
CREATE TABLE dim_date (
  dt            STRING,
  year          INT,
  month         INT,
  day_of_week   INT,
  is_holiday    BOOLEAN
)
STORED AS PARQUET;
```

典型分析查询：

```sql
-- 本月各品类、各城市的 GMV
SELECT
  u.province,
  p.category_l1,
  SUM(o.pay_amount) AS gmv
FROM fact_order o
JOIN dim_user    u ON o.user_id    = u.user_id
JOIN dim_product p ON o.product_id = p.product_id
JOIN dim_date    d ON o.dt         = d.dt
WHERE o.dt BETWEEN '2026-11-01' AND '2026-11-30'
GROUP BY u.province, p.category_l1
ORDER BY gmv DESC;
```

这套结构的好处：

- 事实表只管"发生了什么"，不存冗余描述。
- 维度表改一处（比如用户城市更新），所有分析自动用新值。
- 加新维度（比如加一个"渠道维度"），事实表加一个 `channel_id` 外键即可，不用改老表结构。

---

## 30.8 分析师需要了解的数据工程知识边界

**必须会的**：

- 看得懂星型模型，知道事实表/维度表区别。
- 知道表是按天分区的，写 SQL 必须带 `dt` 过滤。
- 知道数据仓库里的数据是 T+1 的（今天查不到今天的实时数据）。
- 会用 dbt / 调度平台看任务状态、催数据。

**可以不会的**：

- 自己写 Airflow DAG、配 Kafka 集群。
- 调 Spark 内存参数、做 Hive 表优化。
- 设计整个公司的数据架构。

**沟通建议**：发现数据不对，不要直接说"数据错了"，而是带着：① 表名、分区、SQL；② 你预期的值；③ 实际值；④ 怀疑的源头。这样数据工程师能快速定位。

---

## 30.9 常见坑

1. **维度建模混乱**：在事实表里冗余存了用户城市、商品品类，结果城市更新后老数据不更新，报表口径错乱。冗余字段一定要有明确的快照策略（SCD Type 2）。
2. **没有数据血缘**：一个指标"GMV"，到底是订单表算的还是支付表算的？哪几个下游报表在用它？改一个字段影响多少下游？没有血缘文档，改一次怕一次。
3. **SQL 不带分区过滤**：在 10 亿行分区表上跑 `WHERE amount > 100`，引擎全表扫描，把集群资源打满，被 DBA 约谈。**永远先写 `dt` 条件。**
4. **大表 JOIN 没走分区**：两个大表 JOIN，过滤条件里不带分区键，笛卡尔式扫表。
5. **ELT 中间层不清理**：dbt 建了一堆临时表，没人删，仓库越来越大，成本越来越高。
6. **口径不统一**：每个分析师自己写一份"活跃用户"定义，同一份数据两个数字，开会吵架。指标一定要落到指标平台（Metric Store）。
7. **忽视空值与重复**：直接 `SUM(amount)`，没注意 NULL 被忽略、重复订单被算了两次。

---

## 30.10 小结

- 数据工程师备料、分析师做菜、数据科学家研发菜谱。
- OLTP 服务业务、OLAP 服务分析；事实表 + 维度表是数仓的核心结构。
- 现代数仓走 ELT + dbt，转换用 SQL 在仓库内做。
- 工具栈：Airflow 调度、dbt 转换、Spark 算大、ClickHouse/Doris 查快。
- 数据质量看四个维度：完整、一致、及时、唯一。
- 大表必须分区，查询必须带分区过滤。
- 分析师要懂到"能和数据工程师有效对话"为止，不要越界替人家设计架构。

> 下一章预告：后续章节将进入综合实战与职业进阶，把本手册所有章节串成一条完整的数据分析工作流。
> 上一章回顾：[第29章 MLOps与模型部署](29-mlops-deployment.md)。
