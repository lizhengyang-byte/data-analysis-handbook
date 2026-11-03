"""
第26章 NLP 情感分析：电商评论正负分类
运行：python sentiment_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ========== 1. 准备标注评论数据（示例） ==========
print("=" * 60)
print("1. 准备电商评论数据")
print("=" * 60)

reviews = [
    # 正面
    ("东西很好用，质量超出预期，推荐购买", 1),
    ("物流很快，包装精美，下次还会再来", 1),
    ("客服态度很好，问题解决得很快，满意", 1),
    ("性价比很高，这个价位能买到这样的产品很值", 1),
    ("用了一周感觉不错，功能齐全，操作简单", 1),
    ("外观漂亮，做工精细，朋友都说好看", 1),
    ("和描述的一样，没有色差，尺寸合适", 1),
    ("第二次购买了，一如既往的好，五星好评", 1),
    ("产品质量过硬，发货速度快，非常满意", 1),
    ("收到货很惊喜，比想象中好，点赞", 1),
    # 负面
    ("质量很差，用了两天就坏了，不推荐", 0),
    ("物流太慢了，等了一周才到，包装还破了", 0),
    ("客服态度差，问了半天才回复，不满意", 0),
    ("价格偏贵，性价比低，不如实体店", 0),
    ("操作复杂，说明书不清，不好用", 0),
    ("外观有划痕，做工粗糙，失望", 0),
    ("和描述不符，有色差，尺寸也不对", 0),
    ("第一次买就出问题，售后还不给退", 0),
    ("产品质量一般，不值这个价，不推荐", 0),
    ("收到就有故障，退货流程麻烦，差评", 0),
    # 中性/混合
    ("一般般吧，能用但没有特别惊喜", 1),
    ("价格还行，质量中规中矩", 1),
    ("发货速度一般，产品本身还行", 1),
    ("不算特别好，但也没什么大问题", 1),
    ("凑活用吧，这个价格不能要求太多", 0),
]

df = pd.DataFrame(reviews, columns=['评论', '标签'])
print(f"评论数: {len(df)}, 正面: {(df['标签']==1).sum()}, 负面: {(df['标签']==0).sum()}")

# ========== 2. TF-IDF 向量化 ==========
print("\n" + "=" * 60)
print("2. TF-IDF 特征提取")
print("=" * 60)

# 中文分词简单版（按字符n-gram，无需jieba也能跑）
# 实际项目中建议用 jieba 分词后再向量化
vectorizer = TfidfVectorizer(
    analyzer='char',        # 按字符（中文场景）
    ngram_range=(1, 2),      # 1-2元字符组合
    max_features=500
)

X = vectorizer.fit_transform(df['评论'])
y = df['标签']
print(f"TF-IDF 矩阵形状: {X.shape}")
print(f"部分特征词: {vectorizer.get_feature_names_out()[:20]}")

# ========== 3. 训练分类器 ==========
print("\n" + "=" * 60)
print("3. 训练情感分类器")
print("=" * 60)

# 小数据集，直接用全部训练演示（实际项目要划分）
clf = LogisticRegression(C=1.0, random_state=42)
clf.fit(X, y)

# 用训练集自带数据评估（演示用，实际要留出测试集）
y_pred = clf.predict(X)
print(classification_report(y, y_pred, target_names=['负面', '正面']))

# ========== 4. 预测新评论 ==========
print("\n" + "=" * 60)
print("4. 对新评论预测")
print("=" * 60)

new_reviews = [
    "产品很好，物流也快，五星",
    "垃圾产品，用一次就坏了",
    "还行吧，一般般",
    "客服态度超级差，再也不买了",
    "非常满意，推荐给朋友了",
]

X_new = vectorizer.transform(new_reviews)
probs = clf.predict_proba(X_new)[:, 1]

for review, prob in zip(new_reviews, probs):
    sentiment = "😊 正面" if prob > 0.6 else ("😐 中性" if prob > 0.4 else "😞 负面")
    print(f"[{sentiment} ({prob:.2f})] {review}")

# ========== 5. 关键词提取（TF-IDF 权重） ==========
print("\n" + "=" * 60)
print("5. 正面评论的关键词")
print("=" * 60)

positive_comments = df[df['标签'] == 1]['评论'].tolist()
tfidf_pos = vectorizer.transform(positive_comments)
mean_weights = np.asarray(tfidf_pos.mean(axis=0)).ravel()
feature_names = vectorizer.get_feature_names_out()

top_idx = mean_weights.argsort()[-10:][::-1]
for idx in top_idx:
    print(f"  '{feature_names[idx]}': {mean_weights[idx]:.4f}")

print("\n✅ NLP 情感分析完成")
