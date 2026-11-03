# 第28章 模型评估与调优

> 训练一个模型只需要三行代码，把模型调到能用、敢上线、能解释清楚，才是真正的功夫。本章讲评估、交叉验证、调参、过拟合诊断与模型解释。

---

## 28.1 为什么评估比模型选择更重要

很多新手的习惯是：拿到数据 → 跑一下 `RandomForestClassifier` → 看到准确率 85% 就觉得完事了。但真正决定模型能不能上线的，不是你用了哪个模型，而是：

1. **这个 85% 是真实能力，还是数据泄露带来的假象？**
2. **在新数据上还能保持 85% 吗？**
3. **错的样本错在哪类？代价是什么？**
4. **模型做决定的依据，业务方敢相信吗？**

> 一句话：模型选择决定上限，评估与调优决定下限。一个评估严谨的逻辑回归，往往比一个评估草率的深度神经网络更可靠。

分析师在业务场景里，目标不是刷榜，而是**得出一个可信、可解释、可复用的结论**。

---

## 28.2 交叉验证：K-Fold、StratifiedKFold、时序CV

### 28.2.1 为什么不能只用一次 train_test_split

单次 80/20 划分有运气成分：刚好抽到的测试集简单，分数就虚高；测试集难，分数就虚低。**交叉验证（Cross-Validation, CV）** 通过多次划分取平均，让评估更稳定。

### 28.2.2 K-Fold

把数据切成 K 份（常用 5 或 10），轮流用其中 1 份做验证、其余做训练，重复 K 次：

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
# 输出类似：CV AUC: 0.842 ± 0.018
```

`shuffle=True` 很重要，否则原始数据顺序会影响切分。

### 28.2.3 StratifiedKFold：分类任务必用

分类任务里，如果类别不平衡（比如流失用户只占 5%），普通 K-Fold 可能某一折里正样本极少。**StratifiedKFold** 保证每一折里正负样本比例与整体一致：

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
```

> 经验法则：分类任务一律用 StratifiedKFold，回归任务用普通 KFold。

### 28.2.4 TimeSeriesSplit：时序数据绝不能随机切

金融、库存、用户行为预测这类带时间维度的数据，**随机打乱再切分就是数据泄露**——你用未来的数据预测过去，分数虚高，上线就崩。

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
# 每折：训练集 = 前 N 段，验证集 = 紧接着的下一段
# 永远不会用"后面的数据"训练、"前面的数据"验证
for train_idx, val_idx in tscv.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

| 数据类型 | 推荐 CV |
|---|---|
| 普通表格、类别平衡 | KFold |
| 普通表格、类别不平衡 | StratifiedKFold |
| 时间序列 | TimeSeriesSplit |
| 分组数据（同一用户多条记录） | GroupKFold |

---

## 28.3 超参数调优

### 28.3.1 GridSearchCV：穷举

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_leaf': [1, 5, 10],
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)
best_model = grid.best_estimator_
```

缺点：组合数爆炸（3×4×3 = 36 组，每组 5 折 = 180 次训练）。

### 28.3.2 RandomizedSearchCV：随机采样

先定义参数分布，固定采样次数（比如 30 次），适合参数空间大的时候：

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': [3, 5, 10, None],
    'min_samples_leaf': randint(1, 20),
}

rand_search = RandomizedSearchCV(model, param_dist, n_iter=30,
                                 cv=5, scoring='roc_auc',
                                 random_state=42, n_jobs=-1)
rand_search.fit(X_train, y_train)
```

### 28.3.3 贝叶斯优化（Optuna 简介）

Grid / Random 都是"盲调"，不利用之前试过的结果。**贝叶斯优化**会根据历史结果建模，智能地在"未知区域探索"和"已知好区域附近挖掘"之间权衡。

```python
# pip install optuna
import optuna
from sklearn.metrics import roc_auc_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
    }
    m = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(m, X_train, y_train, cv=3, scoring='roc_auc')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print(study.best_params, study.best_value)
```

> 经验：小项目 RandomizedSearch 够用；大项目、训练成本高时用 Optuna 这类贝叶斯优化，性价比更高。

---

## 28.4 学习曲线与验证曲线：诊断过拟合/欠拟合

### 28.4.1 学习曲线：看数据量与拟合程度

```python
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring='roc_auc', n_jobs=-1
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='val')
plt.xlabel('training samples'); plt.ylabel('AUC'); plt.legend()
plt.title('Learning Curve')
plt.show()
```

**怎么读这张图：**

| 曲线形态 | 含义 | 对策 |
|---|---|---|
| 训练高分，验证低分，差距大 | 过拟合 | 加正则、减复杂度、加数据 |
| 训练低分，验证也低分 | 欠拟合 | 换更复杂模型、加特征 |
| 训练验证都低且差距小 | 模型容量不够 / 噪声太大 | 重新审视特征与目标 |
| 验证曲线还在上升 | 加数据有用 | 去攒更多数据 |

### 28.4.2 验证曲线：看单个超参数

```python
from sklearn.model_selection import validation_curve

param_range = [3, 5, 10, 15, 20]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(), X_train, y_train,
    param_name='max_depth', param_range=param_range,
    cv=5, scoring='roc_auc'
)
```

横轴是 `max_depth`，纵轴是分数，一眼看出哪个深度开始过拟合。

---

## 28.5 偏差-方差权衡

这是所有 ML 问题的根。

- **偏差（Bias）**：模型太简单，连训练集都拟合不好 → 欠拟合。
- **方差（Variance）**：模型太复杂，对训练集噪声敏感 → 过拟合。
- **不可消除误差**：数据本身噪声，再强的模型也救不了。

```
总误差 = 偏差² + 方差 + 噪声
```

调参本质就是在偏差和方差之间找平衡点：

- 降低偏差：增加模型复杂度（更深的树、更多的树、更高阶的多项式）
- 降低方差：正则化（剪枝、`min_samples_leaf`）、更多数据、集成

---

## 28.6 模型解释：特征重要性、SHAP、LIME 简介

业务方永远会问一句话：**"你这个预测凭什么？"**

### 28.6.1 自带特征重要性

```python
import pandas as pd
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print(imp.head(10))
```

简单但粗糙，只反映"全局平均"，不告诉你单个样本为什么被这么判。

### 28.6.2 SHAP：给每一次预测打分

```python
# pip install shap
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 全局：特征对整体预测的贡献
shap.summary_plot(shap_values, X_test)

# 单样本：为什么这一条被判为"高流失风险"
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```

SHAP 的好处是**每个特征的贡献可加、有方向**，能直接拿去和业务方对话："这个用户被判流失，70% 是因为他 30 天没登录。"

### 28.6.3 LIME：局部可解释

LIME 思路：在单个样本周围做微小扰动，看预测怎么变，从而反推这个样本附近模型怎么决策。适合解释任意黑盒模型，但稳定性不如 SHAP。

> 原则：**能解释的模型才敢上线。** 尤其在金融、医疗、风控场景，解释性是合规要求。

---

## 28.7 基线模型（Dummy）为什么必须先建

直接上复杂模型之前，先跑一个"傻基线"：

```python
from sklearn.dummy import DummyClassifier, DummyRegressor

# 分类：永远预测最多的那个类
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print(dummy.score(X_test, y_test))  # 比如 95% 准确率——但这只是"全猜多数类"
```

这个 95% 看起来很高，但毫无价值。它的作用是**告诉你：一个不做任何思考的模型能拿多少分**。你的 RandomForest 必须显著超过这个基线，否则就是在做无用功。

常见的简单基线：

- 回归：永远预测均值 / 中位数
- 分类：永远预测多数类 / 随机预测
- 时序：上一期的值（naive forecast）

---

## 28.8 实战：RandomForest 完整 CV + 调参 + 学习曲线

下面是一个可以直接套用的模板：

```python
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV,
                                     learning_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import randint

# 1. 数据
df = pd.read_csv("user_churn.csv")
features = ['login_days_last_month', 'avg_session_duration',
            'purchase_count', 'days_since_last_login']
X, y = df[features], df['is_churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 2. 基线
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
print(f"Dummy AUC: {roc_auc_score(y_test, dummy.predict_proba(X_test)[:,1]):.3f}")

# 3. 随机调参
param_dist = {
    'n_estimators': randint(80, 250),
    'max_depth': [3, 5, 8, 12, None],
    'min_samples_leaf': randint(1, 15),
}
rf = RandomForestClassifier(random_state=42, class_weight='balanced')
search = RandomizedSearchCV(rf, param_dist, n_iter=20,
                            cv=StratifiedKFold(5, shuffle=True, random_state=42),
                            scoring='roc_auc', n_jobs=-1, random_state=42)
search.fit(X_train, y_train)
best = search.best_estimator_
print("best params:", search.best_params_, "best CV:", search.best_score_.round(3))

# 4. 测试集终验（只跑一次！）
y_prob = best.predict_proba(X_test)[:, 1]
print("Test AUC:", roc_auc_score(y_test, y_prob).round(3))
print(classification_report(y_test, best.predict(X_test)))

# 5. 学习曲线
ts, tr, va = learning_curve(best, X_train, y_train, cv=5,
                            train_sizes=np.linspace(0.1, 1, 8),
                            scoring='roc_auc', n_jobs=-1)
plt.plot(ts, tr.mean(1), label='train'); plt.plot(ts, va.mean(1), label='val')
plt.legend(); plt.title('Learning Curve'); plt.show()
```

---

## 28.9 常见坑

1. **测试集被用于调参**：在测试集上反复试参数，直到分数好看——这不是评估，是作弊。正确做法：训练集 → 调参 → 验证集挑最优 → 测试集只跑一次。
2. **数据泄露**：把全量数据做了 `fit(StandardScaler)` 再切分，等于测试集的统计量跑进了训练流程。**所有预处理都要在训练集上 fit，再 transform 到验证/测试集。**
3. **多次使用测试集**：每次实验完都看一眼测试集分数再改方案，测试集就被"间接训练"了。测试集应该是"最终期末考"，一学期只考一次。
4. **类别不平衡只看准确率**：99% 准确率可能只是因为模型全猜多数类。一定要看 AUC、Precision/Recall、F1。
5. **时序数据随机切分**：前面讲过，必然泄露。
6. **CV 时忘记分组**：同一用户多条记录，可能同时出现在训练和验证集，相当于泄题。

---

## 28.10 小结

- 评估体系比模型选择更重要：先有基线，再谈提升。
- 分类用 StratifiedKFold，时序用 TimeSeriesSplit，随机切分不是万能。
- 调参从 RandomizedSearch 起步，复杂场景上 Optuna。
- 学习曲线告诉你"加数据还是加复杂度"，验证曲线告诉你"哪个参数在过拟合"。
- SHAP 让你能向业务方解释每一个预测。
- **测试集只用一次**，预处理不要泄露，CV 不要乱切。

> 下一章：[第29章 MLOps与模型部署](29-mlops-deployment.md)——模型训练完了，怎么真正跑起来给业务用？
> 上一章回顾：[第15章 机器学习在数据分析中的应用](15-ml-for-data-analysis.md)。
