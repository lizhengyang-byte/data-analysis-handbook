# 第26章 NLP 在数据分析中的应用

> 业务里 80% 的信息其实藏在文本里——客服工单、评论、聊天记录。本章讲分析师怎么把这些"非结构化文字"变成能统计、能建模的数字。

---

## 26.1 NLP 能解决什么分析问题

| 业务问题 | NLP 做法 |
|---------|---------|
| 产品评论是好评还是差评 | 情感分类 |
| 客服工单在投诉物流还是质量 | 文本多分类 |
| 用户最近在抱怨什么 | 关键词提取 + 聚类 |
| 几千条反馈归成几个主题 | 主题模型 LDA |
| 相似工单自动合并/推荐历史方案 | 文本向量相似度 |

分析师不需要自己训 BERT，**能把现成工具串成一条可用的流水线**就够了。

---

## 26.2 文本预处理：中文比英文麻烦

机器不认识汉字，得先把句子切成词。

```python
import jieba

text = "这个手机续航真的强，但是拍照一般般，屏幕还行。"
words = jieba.lcut(text)
print(words)
# ['这个', '手机', '续航', '真的', '强', '，', '但是', '拍照', '一般般', '，', '屏幕', '还行', '。']
```

接着去停用词（"的、了、是、但是"这类没信息量的词）和标点：

```python
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

zh_stop = {'的', '了', '是', '我', '你', '他', '但是', '就', '都', '也', '在', '和', '，', '。', '！', '？'}

def clean(text):
    tokens = jieba.lcut(text)
    return [t for t in tokens if t.strip() and t not in zh_stop]

print(clean(text))
# ['手机', '续航', '真的', '强', '拍照', '一般般', '屏幕', '还行']
```

> **坑**：中文没有空格，分词错了后面全错。行业词、品牌名要加自定义词典：`jieba.add_word('华为Mate60')`。

---

## 26.3 词袋与 TF-IDF

### 26.3.1 CountVectorizer：词袋

把文档变成"每个词出现几次"的向量。

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["手机 续航 强", "手机 拍照 差", "屏幕 清晰 拍照"]
cv = CountVectorizer()
X = cv.fit_transform(docs)
print(cv.get_feature_names_out())
# ['手机', '屏幕', '拍照', '清晰', '续航', '强', '差']
print(X.toarray())
```

### 26.3.2 TF-IDF：更聪明的权重

"的、是、在"在每篇文档里都出现，应该降权。TF-IDF = 词频 × 逆文档频率，让"在少数文档里高频出现"的词权重更高。

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(tokenizer=lambda s: s.split(), token_pattern=None)
X = tfidf.fit_transform(docs)
```

中文场景一般要自己分词后再喂进去（见上一节的 `clean`），不要让 TfidfVectorizer 按空格切。

---

## 26.4 文本分类：情感分析

拿到一堆评论，要自动标正/负面。经典基线：**TF-IDF + 逻辑回归**，几十行代码就能跑，工业界至今仍在用。

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# docs: list[str], labels: 1=正面, 0=负面
X_train, X_test, y_train, y_test = train_test_split(
    docs, labels, test_size=0.2, random_state=42, stratify=labels)

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=lambda s: clean(s), token_pattern=None,
                              ngram_range=(1, 2), min_df=2)),
    ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
])
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
```

`ngram_range=(1,2)` 加入二元组（"续航 强""拍照 差"），能抓住否定和转折。`class_weight='balanced'` 应对正负样本不均。

---

## 26.5 关键词提取

### 26.5.1 TF-IDF 直接抽

一篇文档里 TF-IDF 分数最高的词，就是它的关键词。

```python
def top_terms(doc, tfidf, vectorizer, k=5):
    v = vectorizer.transform([doc])
    scores = zip(vectorizer.get_feature_names_out(), v.toarray()[0])
    return sorted(scores, key=lambda x: -x[1])[:k]
```

### 26.5.2 TextRank / YAKE

更适合"单篇文本"抽关键词，不依赖语料库：

- **TextRank**：把词看成节点，共现关系连成图，跑 PageRank。
- **YAKE**：基于统计特征（大小写、位置、频率）打分，无需训练。

```python
# pip install jieba yake
import yake
kw = yake.KeywordExtractor(lan='zh', n=2, top=5)
print(kw.extract_keywords("这个手机续航真的强但是拍照一般般"))
```

---

## 26.6 主题模型：LSA 与 LDA

几千条反馈不想逐条看，想自动归成几个主题。

- **LSA（潜在语义分析）**：对 TF-IDF 矩阵做 SVD 降维，每个"主题"是一组词的组合。
- **LDA（隐含狄利克雷分布）**：假设每篇文档由若干主题按比例混合而成，每个主题又是一组词的概率分布。

```python
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(tokenizer=lambda s: clean(s), token_pattern=None, min_df=5)
X = cv.fit_transform(corpus)

lda = LatentDirichletAllocation(n_components=5, random_state=42)
doc_topic = lda.fit_transform(X)   # 每篇文档在 5 个主题上的分布

# 看每个主题的 top 词
words = cv.get_feature_names_out()
for i, topic in enumerate(lda.components_):
    top = [words[j] for j in topic.argsort()[-10:][::-1]]
    print(f"主题{i}: {top}")
```

**怎么挑 K**：没有金标准，先跑 5/10/20 看主题词是否"像人话"，能解释就停。

---

## 26.7 现代方法：词向量与预训练模型

- **Word2Vec / FastText**：把每个词映射成 200 维向量，语义相近的词向量也近（"手机"≈"电话"）。可以直接做相似度检索。
- **BERT 类预训练模型**：把整句话编码成向量，情感分类、语义相似度、问答都靠它。开源生态里有 `transformers`、`sentence-transformers`，几行代码拿到一个比 TF-IDF 强一大截的句子向量。

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
emb = model.encode(["手机续航很强", "电池非常耐用"])
# emb[0] 与 emb[1] 的余弦相似度会很高
```

分析师路线建议：**先用 TF-IDF + 逻辑回归跑通基线，再考虑上 BERT**。基线能解决 70% 的问题，剩下 30% 才值得烧 GPU。

---

## 26.8 实战：电商评论情感分析完整流程

下面这段从造数据到评估一气呵成，可直接运行：

```python
import jieba
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 合成评论数据（实际项目里换成 df['comment']）
pos = [
    "手机续航真的很给力，用一天都不用充电",
    "拍照效果超出预期，夜景也清晰",
    "物流很快，包装完好，客服态度好",
    "屏幕流畅，颜值高，非常满意",
    "系统很顺滑，性价比超高，推荐购买",
] * 40
neg = [
    "拍照模糊，夜景根本没法看",
    "电池太差了，半天就没电",
    "物流很慢，客服一直不回复",
    "屏幕有坏点，申请退款被拒",
    "系统卡顿，经常死机，非常后悔",
] * 40

docs  = pos + neg
y     = [1]*len(pos) + [0]*len(neg)

# 2. 预处理
zh_stop = {'的','了','是','我','你','他','但是','就','都','也','在','和','很','非常'}
def clean(t):
    return [w for w in jieba.lcut(t) if w.strip() and w not in zh_stop]

# 3. 训练 + 评估
X_tr, X_te, y_tr, y_te = train_test_split(docs, y, test_size=0.2,
                                          random_state=42, stratify=y)
pipe = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=clean, token_pattern=None,
                              ngram_range=(1,2), min_df=1)),
    ('clf',   LogisticRegression(max_iter=1000)),
])
pipe.fit(X_tr, y_tr)
print(classification_report(y_te, pipe.predict(X_te), digits=3))

# 4. 预测新评论
new = ["屏幕摔一下就碎了，质量堪忧", "续航不错，办公够用"]
print(pipe.predict(new))      # [0 1]
```

跑出来 F1 一般在 0.95 上下——因为合成数据太干净。真实业务里正负样本噪声大、类别不平衡，F1 能到 0.8 就已经很有用了。

---

## 26.9 常见坑

1. **中文分词错**。行业黑话、品牌名、新词必须加词典；新一批评论进来后要检查未登录词。
2. **停用词表不全**。网上下的通用停用词表里没有"您好""亲""谢谢"这种客服高频废话，要自己补。
3. **类别不平衡**。10000 条评论里只有 200 条差评，模型会偷懒全猜好评。用 `class_weight='balanced'` 或过采样（见 [第15章 ML](15-ml-for-data-analysis.md)）。
4. **min_df/max_df 没调**。词表太大 → 过拟合；太小 → 丢信息。一般 `min_df=2~5`、`max_df=0.9`。
5. **直接上 BERT 而没有基线**。先跑 TF-IDF 逻辑回归，搞清楚误差长什么样，再决定要不要升级模型。
6. **把"未登录词/错别字/谐音梗"当干净文本**。电商评论里"yyds""绝绝子""鲨疯了"满天飞，不做清洗会严重偏。
7. **只看准确率**。情感分析是不平衡分类，看 PR/F1，别信 accuracy。

---

## 26.10 本章小结

- NLP 分析流水线：**清洗 → 分词 → 向量化（TF-IDF）→ 分类/聚类/关键词**。
- 中文一定要用 **jieba**，并自定义词典、停用词表。
- 文本分类基线：**TF-IDF + 逻辑回归**，几行代码、易解释、易上线。
- 关键词用 TF-IDF/YAKE；反馈归类用 LDA；追求精度再上 **sentence-transformers**。
- 类别不平衡是常态，看 F1/PR-AUC，别被 accuracy 骗。

> **下一章**：[第27章 推荐系统基础](27-recommender-systems.md) — 用户为什么"猜你喜欢"，协同过滤和矩阵分解是怎么回事。
