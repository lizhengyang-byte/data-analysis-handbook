# 第02章 数据获取与采集

> 巧妇难为无米之炊。数据分析的第一步，永远是把数据拿到手。本章覆盖所有主流数据获取方式：从SQL取数到爬虫、从API到埋点日志。

---

## 2.1 数据来源全景图

企业中数据分析师能接触到的数据源：

```
数据来源
├── 内部数据
│   ├── 业务数据库（MySQL/PostgreSQL/Oracle）
│   ├── 数据仓库（Hive/ClickHouse/MaxCompute）
│   ├── 埋点日志（用户行为日志）
│   ├── 业务后台导出（Excel/CSV）
│   ├── CRM/ERP系统
│   └── 客服/工单系统
└── 外部数据
    ├── 开放API（天气、汇率、股票）
    ├── 网络爬虫（竞品、行业数据）
    ├── 第三方数据平台（QuestMobile、艾瑞）
    ├── 公开数据集（Kaggle、UCI）
    └── 政府/行业报告
```

**分析师90%的时间在和前6种内部数据打交道。**

![数据源全景](assets/images/ch02-data-sources.png)

---

## 2.2 SQL取数：最核心的技能

### 2.2.1 连接数据库

**Python连接各种数据库：**

```python
# MySQL
import pymysql
conn = pymysql.connect(
    host='localhost',
    port=3306,
    user='analyst',
    password='your_password',
    database='ecommerce',
    charset='utf8mb4'
)

# PostgreSQL
import psycopg2
conn = psycopg2.connect(
    host='localhost',
    port=5432,
    user='analyst',
    password='your_password',
    dbname='analytics'
)

# 使用SQLAlchemy（推荐，统一接口）
from sqlalchemy import create_engine
engine = create_engine('mysql+pymysql://analyst:password@localhost:3306/ecommerce?charset=utf8mb4')

# ClickHouse
from clickhouse_driver import Client
client = Client(host='localhost', port=9000, database='default')
```

**读取数据到Pandas：**

```python
import pandas as pd

# 方式1：直接用engine
df = pd.read_sql('SELECT * FROM orders WHERE dt = "2026-10-01"', engine)

# 方式2：参数化查询（防SQL注入）
query = '''
    SELECT user_id, amount, order_time 
    FROM orders 
    WHERE dt BETWEEN %s AND %s 
      AND status = %s
'''
df = pd.read_sql(query, engine, params=('2026-10-01', '2026-10-07', 'paid'))
```

### 2.2.2 取数最佳实践

**1. 永远加LIMIT先看一眼**

```sql
-- 先看10条，确认字段和数据质量
SELECT * FROM orders LIMIT 10;
```

**2. 只取需要的字段，不要SELECT ***

```sql
-- ❌ 慢、占内存
SELECT * FROM orders WHERE dt = '2026-10-01';

-- ✅ 快、省内存
SELECT user_id, amount, status, create_time 
FROM orders 
WHERE dt = '2026-10-01';
```

**3. 分区裁剪（大数据表必看）**

```sql
-- Hive/数仓表通常按dt分区
-- ✅ 指定分区，扫描一个分区
SELECT * FROM dwd_order_detail WHERE dt = '2026-10-01';

-- ❌ 全表扫描，慢到死
SELECT * FROM dwd_order_detail;
```

**4. 大表JOIN注意**

```sql
-- 先过滤再JOIN，减少数据量
-- ✅ 子查询先过滤
SELECT a.user_id, b.amount
FROM (
    SELECT user_id FROM users WHERE register_dt = '2026-10-01'
) a
JOIN orders b ON a.user_id = b.user_id
WHERE b.dt = '2026-10-01';

-- ❌ 大表直接JOIN，容易OOM
SELECT a.user_id, b.amount
FROM users a JOIN orders b ON a.user_id = b.user_id;
```

**5. 取数后校验**

```python
# 取完数据，先做基本校验
print(f"行数: {len(df)}")
print(f"列数: {len(df.columns)}")
print(f"字段类型:\n{df.dtypes}")
print(f"缺失值:\n{df.isnull().sum()}")
print(f"描述统计:\n{df.describe()}")

# 和已知的大盘数字对一下
# 比如这个月GMV应该在1亿左右
print(f"总金额: {df['amount'].sum():,.0f}")
# 如果出来是10亿或者100万，说明取数有问题
```

### 2.2.3 常见取数模板

**模板1：日活用户**

```sql
SELECT 
    dt,
    COUNT(DISTINCT user_id) AS dau
FROM dwd_user_behavior_log
WHERE dt BETWEEN '2026-10-01' AND '2026-10-07'
GROUP BY dt
ORDER BY dt;
```

**模板2：转化率漏斗**

```sql
SELECT 
    dt,
    COUNT(DISTINCT CASE WHEN event = 'view_product' THEN user_id END) AS view_uv,
    COUNT(DISTINCT CASE WHEN event = 'add_cart' THEN user_id END) AS cart_uv,
    COUNT(DISTINCT CASE WHEN event = 'create_order' THEN user_id END) AS order_uv,
    COUNT(DISTINCT CASE WHEN event = 'pay_success' THEN user_id END) AS pay_uv,
    COUNT(DISTINCT CASE WHEN event = 'add_cart' THEN user_id END) * 1.0 
        / COUNT(DISTINCT CASE WHEN event = 'view_product' THEN user_id END) AS view_to_cart,
    COUNT(DISTINCT CASE WHEN event = 'pay_success' THEN user_id END) * 1.0 
        / COUNT(DISTINCT CASE WHEN event = 'create_order' THEN user_id END) AS order_to_pay
FROM dwd_user_behavior_log
WHERE dt = '2026-10-01'
GROUP BY dt;
```

**模板3：留存率**

```sql
SELECT 
    a.dt AS register_dt,
    COUNT(DISTINCT a.user_id) AS new_users,
    COUNT(DISTINCT b.user_id) AS retained_1d,
    COUNT(DISTINCT b.user_id) * 1.0 / COUNT(DISTINCT a.user_id) AS retention_1d
FROM (
    SELECT user_id, dt 
    FROM dwd_user_behavior_log 
    WHERE event = 'app_open' AND dt = '2026-10-01'
) a
LEFT JOIN (
    SELECT DISTINCT user_id 
    FROM dwd_user_behavior_log 
    WHERE dt = '2026-10-02'
) b ON a.user_id = b.user_id
GROUP BY a.dt;
```

---

## 2.3 从Excel/CSV导入数据

### 2.3.1 读取各种格式

```python
import pandas as pd

# CSV（最常用）
df = pd.read_csv('orders.csv')
df = pd.read_csv('orders.csv', encoding='gbk')  # 中文文件常是gbk
df = pd.read_csv('orders.csv', parse_dates=['order_time'])  # 自动解析日期
df = pd.read_csv('orders.csv', dtype={'user_id': str})  # 指定类型

# Excel
df = pd.read_excel('sales_2026.xlsx', sheet_name='Q3')
df = pd.read_excel('sales_2026.xlsx', sheet_name=None)  # 读取所有sheet，返回dict

# TSV
df = pd.read_csv('data.tsv', sep='\t')

# 大文件分块读取
chunk_iter = pd.read_csv('huge_file.csv', chunksize=10000)
for chunk in chunk_iter:
    process(chunk)  # 逐块处理
```

### 2.3.2 常见问题

**问题1：中文乱码**

```python
# 先试utf-8，不行试gbk，再不行试gb18030
df = pd.read_csv('data.csv', encoding='utf-8')
df = pd.read_csv('data.csv', encoding='gbk')
df = pd.read_csv('data.csv', encoding='gb18030')
```

**问题2：数字被读成字符串**

```python
# 查看类型
df['amount'].dtype
# 如果是object，说明有非数字字符

# 转换
df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
# errors='coerce' 会把无法转换的变成NaN
```

**问题3：日期解析错误**

```python
# 手动解析
df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
```

---

## 2.4 API数据获取

### 2.4.1 使用requests调用API

```python
import requests
import pandas as pd

# 基本GET请求
url = 'https://api.example.com/v1/sales'
params = {
    'start_date': '2026-10-01',
    'end_date': '2026-10-07',
    'page': 1,
    'page_size': 100
}
headers = {
    'Authorization': 'Bearer your_token_here',
    'Content-Type': 'application/json'
}

response = requests.get(url, params=params, headers=headers)
data = response.json()

# 检查状态码
if response.status_code == 200:
    df = pd.DataFrame(data['results'])
else:
    print(f"请求失败: {response.status_code}")
```

### 2.4.2 分页拉取

```python
all_data = []
page = 1

while True:
    params = {'page': page, 'page_size': 100}
    resp = requests.get(url, params=params, headers=headers)
    batch = resp.json()['results']
    
    if not batch:  # 没有更多数据了
        break
    
    all_data.extend(batch)
    page += 1
    
    # 礼貌延时，不要被封
    import time
    time.sleep(0.5)

df = pd.DataFrame(all_data)
```

### 2.4.3 常见公共API

| API | 用途 | 网站 |
|-----|------|------|
| 天气 | 历史天气数据 | open-meteo.com |
| 股票 | 股价数据 | akshare/tushare |
| 汇率 | 汇率转换 | exchangerate-api |
| 地图 | 地理编码 | 高德/腾讯地图 |
| 公开数据 | 政府开放数据 | data.gov.cn |

---

## 2.5 网络爬虫基础

### 2.5.1 什么时候需要爬虫

- 竞品价格监控
- 行业报告数据
- 社交媒体数据
- 没有API但需要网页数据的场景

### 2.5.2 基础爬虫（requests + BeautifulSoup）

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://books.toscrape.com/catalogue/page-1.html'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

books = []
for article in soup.find_all('article', class_='product_pod'):
    title = article.h3.a['title']
    price = article.find('p', class_='price_color').text
    rating = article.find('p', class_='star-rating')['class'][1]
    
    books.append({
        'title': title,
        'price': price,
        'rating': rating
    })

df = pd.DataFrame(books)
print(df.head())
```

### 2.5.3 翻页爬取

```python
base_url = 'https://books.toscrape.com/catalogue/page-{}.html'

all_books = []
for page in range(1, 51):  # 爬50页
    url = base_url.format(page)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for article in soup.find_all('article', class_='product_pod'):
            all_books.append({
                'title': article.h3.a['title'],
                'price': article.find('p', class_='price_color').text,
                'page': page
            })
        
        print(f"第{page}页完成，累计{len(all_books)}条")
        time.sleep(1)  # 礼貌爬取
        
    except Exception as e:
        print(f"第{page}页失败: {e}")

df = pd.DataFrame(all_books)
df.to_csv('books.csv', index=False, encoding='utf-8-sig')
```

### 2.5.4 爬虫注意事项

1. **遵守robots.txt**：先看目标网站的 /robots.txt 允许爬什么
2. **控制频率**：加延时，不要给服务器造成压力
3. **不要爬敏感数据**：个人隐私、付费墙内容
4. **User-Agent**：加上浏览器UA，不要用默认的python-requests
5. **异常处理**：网络请求一定会失败，try-except要写好
6. **法律风险**：爬取公开数据一般没问题，商用前确认版权

---

## 2.6 埋点日志与用户行为数据

### 2.6.1 什么是埋点

**埋点**：在APP/网页的特定位置插入代码，记录用户的行为事件。

典型的埋点事件结构：

```json
{
    "event_name": "click_buy_button",
    "user_id": "U12345",
    "device_id": "iPhone15_iOS17",
    "timestamp": "2026-10-01 14:30:25",
    "page": "product_detail",
    "properties": {
        "product_id": "P001",
        "product_price": 99.5,
        "source": "home_page"
    }
}
```

### 2.6.2 埋点日志分析

```sql
-- 典型的埋点日志表结构
-- dwd_user_behavior_log
-- dt (分区) | user_id | event_name | page | properties | client_version | os

-- 1. 统计每个事件的PV/UV
SELECT 
    event_name,
    COUNT(*) AS pv,
    COUNT(DISTINCT user_id) AS uv
FROM dwd_user_behavior_log
WHERE dt = '2026-10-01'
GROUP BY event_name
ORDER BY pv DESC;

-- 2. 页面停留时长（用相邻事件时间差）
WITH user_events AS (
    SELECT 
        user_id,
        page,
        event_time,
        LEAD(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS next_time
    FROM dwd_user_behavior_log
    WHERE dt = '2026-10-01'
)
SELECT 
    page,
    AVG(unix_timestamp(next_time) - unix_timestamp(event_time)) AS avg_stay_sec
FROM user_events
WHERE next_time IS NOT NULL
GROUP BY page;
```

---

## 2.7 第三方数据平台

### 2.7.1 常用数据源

| 平台 | 数据类型 | 获取方式 |
|------|----------|----------|
| 友盟/神策 | APP统计 | 后台导出 / API |
| 百度统计 | 网站流量 | 后台导出 |
| Google Analytics | 网站流量 | API |
| QuestMobile | 行业APP数据 | 报告/PDF |
| 艾瑞咨询 | 行业报告 | 报告/PDF |
| 国家统计局 | 宏观经济 | 网页查询/API |

### 2.7.2 从PDF提取数据

有时候只有PDF报告，需要手动把表格提出来：

```python
import tabula  # 需要安装：pip install tabula-py
import pandas as pd

# 从PDF提取表格
tables = tabula.read_pdf('report.pdf', pages='all')
# 返回DataFrame列表

for i, df in enumerate(tables):
    print(f"第{i+1}个表格:")
    print(df.head())
    df.to_csv(f'table_{i}.csv', index=False)
```

---

## 2.8 数据获取阶段的质量控制

### 2.8.1 取数自检清单

每次取数完成后，必须检查：

- [ ] 行数合理吗？（和上次同期比，突然翻倍/减半都要警惕）
- [ ] 总量对吗？（比如GMV、DAU这种大盘数字，和已知值比对一下）
- [ ] 时间范围对吗？（是不是少取了一天？多取了一天？）
- [ ] 维度齐全吗？（需要的字段都在吗？）
- [ ] 有没有重复数据？（GROUP BY前后的数量对不对）
- [ ] 异常值正常吗？（金额有没有负数？年龄有没有200岁的？）

### 2.8.2 常见取数错误

| 错误类型 | 症状 | 排查方法 |
|----------|------|----------|
| 少取一天 | 总量比预期少1/7 | 检查时间范围，BETWEEN是不是左闭右开 |
| 多表join膨胀 | 行数暴涨 | 检查join键是不是唯一的 |
| 口径不一致 | 同一个指标两个数 | 确认"活跃"的定义，是启动还是有行为 |
| 时区问题 | 凌晨的数据不对 | 确认是UTC还是北京时间 |
| 状态过滤遗漏 | 包含了取消/退款的订单 | 检查WHERE条件的status |

---

## 2.9 本章小结

1. **90%的分析数据来自SQL**，SQL能力是分析师的生命线
2. **取数后必须校验**：行数、总量、分布都要和已知值对比
3. **大表取数必做分区裁剪**，先过滤再JOIN
4. **API和爬虫**是补充手段，记得加延时和异常处理
5. **埋点数据**是用户行为分析的基础，理解事件模型很重要

> **下一章**：[第03章 数据清洗与预处理](03-data-cleaning.md) — 拿到脏数据怎么办？缺失值、异常值、重复值，一步步教你洗干净。
