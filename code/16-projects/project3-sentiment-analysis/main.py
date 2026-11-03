"""
项目3：评论情感分析
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("=" * 60)
print("项目3：评论情感分析")
print("=" * 60)

# 训练数据
train_data = [
    ("质量很好，推荐", 1), ("物流快，满意", 1), ("客服好，五星", 1),
    ("性价比高，值", 1), ("漂亮，做工好", 1), ("和描述一致", 1),
    ("第二次买了", 1), ("惊喜，点赞", 1), ("产品过硬", 1),
    ("质量差，不推荐", 0), ("物流慢，包装破", 0), ("客服差", 0),
    ("贵，不值", 0), ("难用", 0), ("有划痕", 0),
    ("和描述不符", 0), ("售后差", 0), ("质量一般", 0),
]
df = pd.DataFrame(train_data, columns=['评论', '标签'])

# 向量化
vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), max_features=300)
X = vec.fit_transform(df['评论'])
clf = LogisticRegression()
clf.fit(X, df['标签'])

# 新评论批量预测
new_comments = [
    "这个产品真的很好用，已经推荐给朋友了",
    "太差了，用了一次就坏了，再也不买",
    "发货速度还行，东西一般般",
    "客服态度很好，解决了我的问题",
    "包装简陋，而且有异味",
]
X_new = vec.transform(new_comments)
probs = clf.predict_proba(X_new)[:, 1]

print("\n预测结果：")
for c, p in zip(new_comments, probs):
    label = "正面" if p > 0.5 else "负面"
    print(f"  [{label} {p:.2f}] {c}")

print(f"\n正面评论占比: {(probs > 0.5).mean():.0%}")
print("\n✅ 项目3完成")
