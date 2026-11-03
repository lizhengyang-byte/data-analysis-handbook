# 第27章 推荐系统基础

> 打开淘宝、抖音、Netflix，背后都是同一类问题：在海量物品里，给这个用户挑出他最可能喜欢的那几个。本章讲清楚推荐系统的主流思路。

---

## 27.1 推荐系统是什么，为什么重要

用户面对 10 万件商品，不可能挨个翻。推荐系统的目标就是**过滤**：把 10 万缩到 10 个，主动推到首页。

业务价值：
- **电商**：推荐位贡献 20%~30% 的 GMV 很常见。
- **内容/视频**：停留时长、留存几乎全靠推荐流。
- **音乐/资讯**：长尾物品的曝光主要靠推荐，而非搜索。

对分析师来说，理解推荐系统意味着你能看懂"为什么这个转化漏斗长这样"、"为什么这个品类突然涨了"。

---

## 27.2 推荐的三大流派

| 流派 | 核心思路 | 代表 |
|------|---------|------|
| 协同过滤 CF | "和你相似的人喜欢什么" / "和这件商品相似的商品你看过" | UserCF / ItemCF |
| 基于内容 Content-based | 根据物品属性和用户画像匹配 | 标签+向量 |
| 混合推荐 Hybrid | CF + 内容 + 上下文 + 排序模型 | 工业界主流 |

实战里几乎都是混合的：召回多路 → 粗排 → 精排 → 重排。本章聚焦最经典的协同过滤。

---

## 27.3 Popularity Baseline：先把最蠢的做出来

在写任何花哨模型之前，先上一个**热门推荐**作为 baseline。它往往就是新业务冷启动的全部。

```python
import pandas as pd

# df: [user, item, rating]
popular = (df.groupby('item')['user'].nunique()
             .sort_values(ascending=False))
top_items = popular.head(50).index.tolist()

# 所有新用户默认推 top_items
```

**为什么要先做它**：
1. 上线快，当天就能有推荐位；
2. 后面所有模型都要和它比——**连热门推荐都打不过的模型就是白做**。

---

## 27.4 协同过滤：UserCF 与 ItemCF

### 27.4.1 余弦相似度

两个用户（或两件物品）的相似度，用共现评分向量的余弦夹角衡量：

```python
import numpy as np

def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
```

### 27.4.2 User-based CF

"和我品味像的人买了什么，我也可能喜欢。"

```python
# R: 稀疏的 user-item 评分矩阵（行=用户，列=物品，未评分为 0）
def user_cf_recommend(user_id, R, k=20, top_n=10):
    sims = R @ R[user_id] / (
        np.linalg.norm(R, axis=1) * np.linalg.norm(R[user_id]) + 1e-9)
    top_k_users = np.argsort(-sims)[:k]
    scores = R[top_k_users].mean(axis=0)         # 用相似用户的平均分打分
    already = set(np.where(R[user_id] > 0)[0])
    recs = [i for i in np.argsort(-scores) if i not in already][:top_n]
    return recs
```

**问题**：用户数远多于物品数时，用户相似度矩阵巨大且稀疏；新用户一来就没招。

### 27.4.3 Item-based CF

"你看过 A，而和 A 像的是 B，那给你推 B。"电商、视频网站主流用这个，因为物品数量相对稳定、物品相似度可以离线算好。

```python
def item_cf_scores(R):
    # 物品相似度矩阵：item x item
    norm = np.linalg.norm(R, axis=0, keepdims=True) + 1e-9
    item_sim = (R.T @ R) / (norm.T @ norm)
    np.fill_diagonal(item_sim, 0)
    return item_sim

def recommend(user_vec, item_sim, top_n=10):
    scores = item_sim @ user_vec        # 该用户对每件物品的兴趣分
    seen   = np.where(user_vec > 0)[0]
    scores[seen] = -np.inf
    return np.argsort(-scores)[:top_n]
```

**UserCF vs ItemCF 怎么选**：
- 用户少、物品多（社交新闻）→ UserCF；
- 物品稳定、用户海量（电商、视频）→ ItemCF；
- 想看"为什么推这个"，ItemCF 更可解释（"因为你买过 X"）。

---

## 27.5 矩阵分解：SVD 与 FunkSVD

协同过滤在稀疏矩阵上效果会掉——大部分用户只点过几件物品。矩阵分解的思路：把超大的 user-item 矩阵 R 拆成两个小矩阵 U（用户向量）和 V（物品向量），让 `R ≈ U @ V.T`。

```python
from sklearn.decomposition import TruncatedSVD

# R 是稀疏评分矩阵
svd = TruncatedSVD(n_components=32, random_state=42)
U = svd.fit_transform(R)       # 每个用户 32 维向量
V = svd.components_.T          # 每个物品 32 维向量
R_hat = U @ V.T                # 补全后的预测评分
```

**FunkSVD**（隐语义模型）是工业版：只在"有评分的位置"上做梯度下降，优化 `(r_ui - u_i·v_j)² + 正则`，能处理超大稀疏矩阵。包上 `implicit` 或 `surprise` 就能用：

```python
# pip install surprise
from surprise import SVD, Dataset, Reader
data = Dataset.load_from_df(df[['user','item','rating']], Reader(rating_scale=(1,5)))
trainset = data.build_full_trainset()
algo = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
algo.fit(trainset)
pred = algo.predict('user_42', 'item_7')   # 预测评分
```

好处：向量空间里"相似用户""相似物品"都能用距离表示，还能接上深度学习。

---

## 27.6 怎么评估推荐系统

### 27.6.1 离线指标

- **Precision@K**：你推的 K 个里，用户真喜欢了几个？
- **Recall@K**：用户所有喜欢的东西里，有多少被你推到了？
- **Coverage（覆盖率）**：被推荐过的物品占总物品比例。太低 = 只推爆款。
- **Novelty（新颖度）**：推的是不是用户从没听过的东西。

```python
def precision_recall_at_k(recs, held_out, k=10):
    recs = recs[:k]
    hit  = len(set(recs) & set(held_out))
    return hit/k, hit/len(held_out)
```

做法：把每个用户最近 N 条交互切出来当测试集，用前面的数据训练，看预测能不能命中。

### 27.6.2 在线 A/B

离线指标涨了 ≠ 业务涨了。最终要看 **CTR、停留时长、转化率、GMV、留存**，做严格的 A/B 测试（见 [第10章 A/B 测试](10-ab-testing-causal-inference.md)）。

---

## 27.7 冷启动：新用户/新物品怎么办

协同过滤的命门——没有历史行为就没东西可算。

| 冷启动类型 | 常见解法 |
|-----------|---------|
| 新用户 | 先上 popularity 热门；让你选几个兴趣标签再推；用注册信息（地域、设备）做粗画像 |
| 新物品 | 用物品内容属性（类别、价格、品牌）算相似度，先分给"喜欢同类"的用户；新物品期给流量扶持 |
| 新业务 | 干脆先做编辑精选 + 热门榜单，攒到一定数据量再切模型 |

经验法则：**前 2 周老老实实热门+人工运营，别迷信算法**。

---

## 27.8 实战：ItemCF 电影推荐

下面这段用合成的用户-电影评分矩阵，跑出一个能直接看的推荐列表：

```python
import numpy as np
import pandas as pd

# 1. 构造用户-物品评分矩阵（5 个用户 × 6 部电影）
# 行=用户 u1..u5, 列=电影 m1..m6, 0 表示没看过
R = np.array([
    [5, 3, 0, 1, 4, 0],
    [4, 0, 0, 1, 0, 0],
    [1, 1, 0, 5, 0, 5],
    [0, 0, 5, 0, 0, 2],
    [0, 2, 4, 0, 1, 0],
], dtype=float)
movies = ['加勒比海盗', '玩具总动员', '蝙蝠侠', '阿甘正传', '泰坦尼克号', '黑客帝国']

# 2. 算物品相似度（余弦）
norm = np.linalg.norm(R, axis=0, keepdims=True) + 1e-9
item_sim = (R.T @ R) / (norm.T @ norm)
np.fill_diagonal(item_sim, 0)

# 3. 给 u1（R[0]）推荐他没看过的电影
u_vec = R[0]
scores = item_sim @ u_vec
unseen = np.where(u_vec == 0)[0]
ranked = sorted(unseen, key=lambda i: -scores[i])
print("用户1的推荐：")
for i in ranked:
    print(f"  {movies[i]:<8}  score={scores[i]:.3f}")
```

预期输出：用户 1 看过《加勒比海盗》《玩具总动员》《阿甘正传》《泰坦尼克号》，没看过的《蝙蝠侠》《黑客帝国》会按相似度打分排出来。把 `R` 换成你真实的用户行为矩阵（哪怕是点击/收藏的 0/1 矩阵），这套代码就能直接用。

---

## 27.9 常见坑

1. **数据稀疏**。100 万用户 × 1000 万物品，非零项可能不到 0.01%。直接算相似度噪声巨大，先做交互过滤（过滤掉交互 <5 的用户、曝光 <50 的物品），再上矩阵分解。
2. **马太效应**。热门物品被反复推，越来越热；长尾物品永远没曝光。要在重排阶段做**去重、打散、多样性惩罚**。
3. **过滤气泡**。用户越被推相似内容，越只产生相似行为，推荐越窄。定期注入探索（exploration）流量。
4. **离线涨、在线跌**。离线指标只看"能不能命中历史点击"，忽略了惊喜度、时长、GMV。一定要上 A/B。
5. **新物品/新用户不处理**。冷启动阶段全给热门，新内容永远起不来。要给新品"流量扶持位"。
6. **把点击等同于喜欢**。标题党点了不代表喜欢。负反馈（划走、秒关、点踩）也要进模型。
7. **直接上线不设 baseline**。没和"热门推荐"对比就上深度学习，往往白忙活一周。

---

## 27.10 本章小结

- 推荐三件套：**Popularity baseline → 协同过滤 → 矩阵分解**，一步一步往上叠。
- UserCF 适合用户少、社交属性强；ItemCF 适合电商、视频这种物品相对稳定的场景。
- 矩阵分解（SVD/FunkSVD）把稀疏交互压成低维向量，是现代深度推荐的基础。
- 离线看 **Precision/Recall@K、覆盖率**，线上看 **A/B 的业务指标**。
- 冷启动靠热门+运营+内容画像，别指望 CF 一上来就灵。

> **上一章**：[第26章 NLP 在数据分析中的应用](26-nlp-for-analytics.md) — 把文本变成可分析的向量。
>
> 至此本手册进阶篇结束。回到 [目录](SUMMARY.md) 挑下一个主题继续。
