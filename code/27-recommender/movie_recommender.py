"""
第27章 推荐系统：基于 ItemCF 的电影推荐
运行：python movie_recommender.py
"""
import numpy as np
import pandas as pd

# ========== 1. 构造用户-电影评分矩阵 ==========
print("=" * 60)
print("1. 构造评分矩阵")
print("=" * 60)

# 5部电影
movies = ['科幻A', '爱情B', '动作C', '动画D', '恐怖E']
# 8个用户的评分（0表示未评分）
ratings_dict = {
    '用户1': [5, 3, 0, 4, 0],
    '用户2': [4, 0, 5, 0, 2],
    '用户3': [0, 5, 2, 0, 1],
    '用户4': [3, 4, 0, 5, 0],
    '用户5': [0, 0, 4, 0, 5],
    '用户6': [5, 0, 5, 3, 0],
    '用户7': [0, 4, 0, 4, 2],
    '用户8': [4, 3, 0, 0, 4],
}

R = pd.DataFrame(ratings_dict, index=movies).T
print("评分矩阵（0=未看过）:")
print(R)

# ========== 2. 计算电影相似度（余弦相似度） ==========
print("\n" + "=" * 60)
print("2. 电影间相似度（余弦）")
print("=" * 60)

def cosine_similarity_matrix(matrix):
    """计算物品间余弦相似度"""
    # matrix: 用户 x 物品 → 转置为 物品 x 用户
    item_matrix = matrix.values.T
    norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # 避免除零
    normalized = item_matrix / norms
    similarity = normalized @ normalized.T
    return pd.DataFrame(similarity, index=matrix.columns, columns=matrix.columns)

item_sim = cosine_similarity_matrix(R)
print(item_sim.round(3))

# ========== 3. 为用户1做推荐 ==========
print("\n" + "=" * 60)
print("3. 为用户1推荐电影")
print("=" * 60)

target_user = '用户1'
user_ratings = R.loc[target_user]
print(f"{target_user} 的评分: {dict(user_ratings)}")

# 对每部未看过的电影，计算预测评分
predictions = {}
for movie in movies:
    if user_ratings[movie] > 0:
        continue  # 看过的不推荐

    # 找到用户评过分的电影
    rated = user_ratings[user_ratings > 0].index
    # 加权平均：相似度 * 用户评分
    sims = item_sim.loc[movie, rated]
    scores = user_ratings[rated]

    pred = (sims * scores).sum() / sims.sum() if sims.sum() > 0 else 0
    predictions[movie] = round(pred, 2)

print(f"\n预测评分（未看过的电影）:")
for movie, score in sorted(predictions.items(), key=lambda x: -x[1]):
    print(f"  {movie}: {score}")

recommended = max(predictions, key=predictions.get)
print(f"\n🎬 推荐给 {target_user}: {recommended}（预测评分 {predictions[recommended]}）")

# ========== 4. 所有用户推荐列表 ==========
print("\n" + "=" * 60)
print("4. 所有用户 Top1 推荐")
print("=" * 60)

for user in R.index:
    user_ratings = R.loc[user]
    predictions = {}
    for movie in movies:
        if user_ratings[movie] > 0:
            continue
        rated = user_ratings[user_ratings > 0].index
        sims = item_sim.loc[movie, rated]
        scores = user_ratings[rated]
        pred = (sims * scores).sum() / sims.sum() if sims.sum() > 0 else 0
        predictions[movie] = pred
    if predictions:
        top = max(predictions, key=predictions.get)
        print(f"  {user} → 推荐 {top}（预测 {predictions[top]:.2f}）")

print("\n✅ 推荐系统演示完成")
