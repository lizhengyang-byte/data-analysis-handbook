# 第14章 数据报表与自动化

> 每天重复出数是分析师最大的时间浪费。本章讲怎么搭建报表体系、怎么自动化。

---

## 14.1 报表体系

### 14.1.1 报表分层

| 层级 | 类型 | 频率 | 受众 |
|------|------|------|------|
| L1 | 实时监控看板 | 实时 | 运营/技术 |
| L2 | 日报/周报 | 日/周 | 业务团队 |
| L3 | 月度经营分析 | 月 | 管理层 |
| L4 | 专题分析报告 | 按需 | 决策层 |

### 14.1.2 好报表的标准

- **结论先行**：打开就看到关键数字和结论
- **异常标注**：涨跌超过阈值自动标红/标绿
- **可下钻**：从大盘→渠道→具体原因
- **口径统一**：同一个指标在所有报表里定义一致

---

## 14.2 自动化方案

### 14.2.1 定时任务 + 邮件推送

```python
# 用Python + schedule做定时任务
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def daily_report():
    # 1. 从数据库取数
    # 2. 生成图表
    # 3. 发送邮件
    print("生成日报并发送...")

# 每天早上9点执行
schedule.every().day.at("09:00").do(daily_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### 14.2.2 用Airflow调度

生产环境推荐用Airflow：

```python
# airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'analyst',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'daily_sales_report',
    default_args=default_args,
    schedule_interval='0 9 * * *',  # 每天9点
    start_date=datetime(2026, 10, 1),
    catchup=False,
) as dag:
    
    task1 = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )
    
    task2 = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )
    
    task3 = PythonOperator(
        task_id='send_email',
        python_callable=send_email,
    )
    
    task1 >> task2 >> task3
```

---

## 14.3 BI工具

### 14.3.1 主流BI工具对比

| 工具 | 优点 | 缺点 | 适合 |
|------|------|------|------|
| Tableau | 可视化强、易上手 | 贵 | 大企业 |
| Power BI | 和Excel集成好、性价比高 | 自定义差 | 中小企业 |
| FineBI | 国产、中文友好 | 高级功能少 | 国内企业 |
| Metabase | 开源、简单 | 复杂分析弱 | 技术团队 |
| Superset | 开源、功能全 | 学习曲线陡 | 大数据团队 |

### 14.3.2 做看板的原则

1. **一屏一个主题**：不要把所有指标堆在一个屏
2. **关键数字放最上面**：老板打开第一眼就看到
3. **红绿标注**：涨跌一目了然
4. **加注释**：异常点标注原因

---

## 14.4 本章小结

1. **报表分层**：实时看板、日报月报、专题报告
2. **自动化**把重复劳动解放出来
3. **BI工具**选适合自己团队的，不是越贵越好
4. **好看板**让老板3秒看懂结论

> **下一章**：[第15章 机器学习在数据分析中的应用](15-ml-for-data-analysis.md) — 什么时候该上ML？哪些分析用简单方法就够了？
