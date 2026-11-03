# 第29章 MLOps与模型部署

> Notebook 里跑出 90% 准确率只是开始。模型要真的被业务调用、要监控、要能回滚，才算工程化。本章讲从 Notebook 到生产的最短路径。

---

## 29.1 从 Notebook 到生产：为什么 MLOps 重要

Notebook 的特点是**交互式、状态混乱、不可复现**：你 cell 跑过的顺序、变量、当时的库版本，过两周自己都记不清。这对探索没问题，但对"每天给几万个请求做预测"的生产场景就是灾难。

MLOps（Machine Learning Operations）解决三件事：

1. **可复现**：同样的数据 + 代码 + 版本，能跑出同一个模型。
2. **可部署**：模型能被其他系统调用，延迟、吞吐、错误率可控。
3. **可监控**：上线后数据漂移、性能衰减能第一时间发现，而不是等业务方投诉。

分析师在小公司里常常要一个人扛下这三件事；在大公司里，主要和 MLE/平台团队对接。

---

## 29.2 模型持久化：pickle / joblib / ONNX

### 29.2.1 joblib 是 sklearn 生态的默认选择

```python
import joblib

# 训练完保存
joblib.dump(model, "churn_model.joblib")
joblib.dump(scaler, "scaler.joblib")   # 预处理也要一起存！

# 加载
loaded_model = joblib.load("churn_model.joblib")
pred = loaded_model.predict(X_new)
```

> 为什么用 joblib 而不是原生 pickle？因为 sklearn 里大量是 numpy 数组，joblib 序列化效率更高、文件更小。

### 29.2.2 为什么预处理也要存

常见错误：只存模型，不存 scaler / encoder。线上新来一个用户，你忘了把它做同样的标准化，模型直接给出离谱结果。**任何在训练集上 fit 过的对象，都要和模型一起保存。**

### 29.2.3 ONNX：跨语言、跨框架

如果需要在 C# / Java / 移动端跑模型，或者要极致推理性能，把模型转成 ONNX 格式：

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(model, initial_types=initial_type)
with open("churn_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

日常小项目用 joblib 足够，ONNX 是有跨平台需求时才上。

---

## 29.3 部署方式：批处理、REST API、实时预测

### 29.3.1 批处理预测（Offline Batch）

最常见、最稳。每天凌晨跑一次，把预测结果写回数据库，业务系统直接读：

```python
# 每天凌晨由 Airflow / crontab 触发
import pandas as pd, joblib

model = joblib.load("churn_model.joblib")
new_users = pd.read_sql("SELECT * FROM users WHERE dt = '2026-11-04'", conn)
features = new_users[['login_days_last_month', 'avg_session_duration',
                      'purchase_count', 'days_since_last_login']]
new_users['churn_prob'] = model.predict_proba(features)[:, 1]
new_users.to_sql("user_churn_score", conn, if_exists='append', index=False)
```

**优点**：延迟不敏感、出问题可重跑、不影响线上实时链路。
**适合**：风控打分、推荐召回、每日报表。

### 29.3.2 REST API（FastAPI / Flask）

请求来了现场算预测，毫秒级返回。后面 29.7 给完整模板。

### 29.3.3 实时预测（Streaming）

Kafka 来一条用户行为事件 → 特征实时拼装 → 模型推理 → 结果写回在线存储（Redis / HBase）。这是真正的 MLE / 实时工程师的地盘，分析师了解即可。

| 方式 | 延迟 | 复杂度 | 典型场景 |
|---|---|---|---|
| 批处理 | 小时~天 | 低 | 日报、每周打分 |
| REST API | 几十~几百 ms | 中 | 下单风控、实时推荐 |
| 流式 | 毫秒 | 高 | 广告竞价、反欺诈 |

---

## 29.4 模型监控：数据漂移、概念漂移、性能衰减

模型上线不是终点。**三个月后线上的数据分布变了，模型分数会悄悄掉下来。**

### 29.4.1 两种漂移

- **数据漂移（Data Drift）**：输入特征分布变了。比如以前用户平均消费 200 元，现在变成 50 元，`avg_spending` 这个特征的均值和方差都偏移了。
- **概念漂移（Concept Drift）**：输入和标签的关系变了。比如疫情后"30 天没登录"不再等同于"会流失"，用户行为模式本身变了。

### 29.4.2 监控什么

```python
# 1. 特征分布：监控 PSI / KS 检验，超阈值告警
from scipy.stats import ks_2samp

def drift_report(train_df, live_df, col):
    stat, p = ks_2samp(train_df[col], live_df[col])
    return {'column': col, 'ks_stat': stat, 'p_value': p,
            'drift': p < 0.01}

# 2. 预测分布：线上 churn_prob 的均值、分位数有没有突变
# 3. 真实标签回流：过几周拿到真实是否流失，回看模型 AUC
```

### 29.4.3 性能衰减的应对

- 阈值告警：AUC 从 0.85 掉到 0.75，自动发邮件/飞书告警。
- 定期再训练：每周/每月用最新数据重训一次。
- 影子模式：新模型先和老模型并行跑一段时间，对比差异，再切流量。

---

## 29.5 版本管理：DVC、MLflow 简介

### 29.5.1 为什么需要

- 数据版本：训练数据改了一行，模型就不一样，怎么知道哪个模型对应哪个数据？
- 模型版本：上线后出问题，要能一键回滚到上一个版本。
- 实验追踪：你调了 50 次参数，怎么记得住哪个最好？

### 29.5.2 DVC

把数据和模型文件当成代码一样做版本管理，但不放进 Git（太大）。配合 Git remote 存数据快照，`dvc push` / `dvc pull`。

### 29.5.3 MLflow

```python
import mlflow
import mlflow.sklearn

with mlflow.start_run():
    mlflow.log_param("n_estimators", 200)
    mlflow.log_metric("auc", 0.852)
    mlflow.sklearn.log_model(model, "model")
```

MLflow 自带 Web UI，能看到所有实验的参数、指标、模型文件，一键注册到"模型仓库"，线上直接拉指定版本。

> 小团队起步：先写一个 `experiments.csv` 手记录参数和指标，也比什么都没有强。MLflow 是数据量上来后再上。

---

## 29.6 工程化最佳实践

### 29.6.1 配置管理

不要把数据库地址、API key、文件路径硬编码在代码里：

```python
# config.yaml
model:
  path: models/churn_v2.joblib
feature_cols:
  - login_days_last_month
  - avg_session_duration
```

用 `pydantic` 或 `yaml` 读进来，环境差异（开发/生产）用环境变量覆盖。

### 29.6.2 日志

```python
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

log.info("loaded model from %s", model_path)
log.warning("feature X has 15% missing values")
log.error("prediction failed for user_id=%s", uid)
```

线上出问题第一件事就是看日志，没有日志等于盲人摸象。

### 29.6.3 单元测试

至少测：模型能加载、输入形状对、输出在合理范围、空值/异常值能优雅处理。

```python
def test_predict_returns_prob():
    model = joblib.load("models/churn_v2.joblib")
    sample = pd.DataFrame([{...一个正常用户...}])
    p = model.predict_proba(sample)[0, 1]
    assert 0 <= p <= 1
```

### 29.6.4 CI/CD 简介

GitHub Actions / GitLab CI：代码 push 后自动跑测试 → 通过后构建镜像 → 部署到测试环境 → 人工审核 → 上生产。分析师不一定要自己写 CI，但要知道这个流程存在。

---

## 29.7 实战：joblib 保存 + FastAPI 包装 `/predict`

### 29.7.1 训练并保存（train.py）

```python
import pandas as pd, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv("user_churn.csv")
FEATURES = ['login_days_last_month', 'avg_session_duration',
            'purchase_count', 'days_since_last_login']
X, y = df[FEATURES], df['is_churned']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(X_tr)
X_tr_s = scaler.transform(X_tr)

model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
model.fit(X_tr_s, y_tr)
print("test AUC:", roc_auc_score(y_te, model.predict_proba(scaler.transform(X_te))[:,1]))

joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(FEATURES, "features.joblib")
```

### 29.7.2 服务化（app.py）

```python
# pip install fastapi uvicorn pydantic
import joblib, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Churn Prediction API")

model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")
FEATURES = joblib.load("features.joblib")

class UserFeatures(BaseModel):
    login_days_last_month: float
    avg_session_duration: float
    purchase_count: float
    days_since_last_login: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(feat: UserFeatures):
    X = np.array([[getattr(feat, c) for c in FEATURES]])
    X_s = scaler.transform(X)
    prob = float(model.predict_proba(X_s)[0, 1])
    return {"churn_probability": prob,
            "risk": "high" if prob > 0.5 else "low"}
```

启动：

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

测试：

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"login_days_last_month":2,"avg_session_duration":30,"purchase_count":0,"days_since_last_login":45}'
```

自动文档：访问 `http://localhost:8000/docs`，FastAPI 自动生成 Swagger UI，业务方可以直接在页面上点按钮测试。

---

## 29.8 数据分析工程师 vs MLE 的分工边界

| 工作 | 分析师 | MLE |
|---|---|---|
| 探索数据、出业务洞察 | ✅ 主战场 | 偶尔 |
| 建 baseline 模型、调参 | ✅ | ✅ |
| 写 FastAPI 服务、上 K8s | 小公司自己干 | ✅ 主战场 |
| 模型监控、漂移告警 | 提需求 | ✅ |
| 特征平台、在线特征存储 | 了解即可 | ✅ |
| 大规模分布式训练 | 不碰 | ✅ |

> 边界不是固定的。小公司分析师常常要把整链路打通；大公司里，你只要把模型训练好、把指标和监控需求讲清楚，剩下交给 MLE。

---

## 29.9 常见坑

1. **训练/推理环境不一致**：本机 sklearn 1.0 训练，线上 sklearn 1.3 加载，模型直接报错或结果不同。一定要 `pip freeze > requirements.txt`，线上用同一份。
2. **依赖地狱**：`requirements.txt` 里不写版本号，三个月后装出完全不同的环境。**所有依赖都锁版本**。
3. **只存模型不存预处理**：前面讲过，线上没做 scaler，预测全错。
4. **没有监控就上线**：三个月后业务方发现数据不对，模型早就漂了。
5. **FastAPI 里每次请求都 joblib.load 模型**：模型应该在服务启动时加载一次，请求里直接用，否则每个请求都卡几百毫秒。
6. **把训练数据直接暴露在 API 里**：/predict 接口不能返回训练集里其他用户的信息，做越权和注入校验。

---

## 29.10 小结

- 部署三选一：批处理最稳，REST 最灵活，流式最复杂。
- joblib 存模型 + 预处理对象；ONNX 跨语言再用。
- 上线后必须监控数据漂移、概念漂移、性能衰减。
- MLflow / DVC 解决"哪个模型对应哪份数据"的可复现问题。
- 配置、日志、测试、CI/CD 是工程化的四大件。
- FastAPI + joblib 是分析师能独立搞定的最短路径。

> 下一章：[第30章 数据工程基础](30-data-engineering-basics.md)——分析师要懂多少数据工程，才不会被数据工程师看不起？
> 上一章回顾：[第28章 模型评估与调优](28-model-evaluation-tuning.md)。
