"""
第21章 回归实战：加州房价预测
运行：python housing_price_prediction.py
依赖：pip install scikit-learn pandas matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载加州房价数据集")
print("=" * 60)

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='房价中位数（10万美元）')

print(f"数据集形状: {X.shape}")
print(f"特征列表: {list(X.columns)}")
print(f"\n数据概览:\n{X.describe().round(2)}")

# ========== 2. 划分训练集/测试集 ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n训练集: {X_train.shape}, 测试集: {X_test.shape}")

# ========== 3. 特征标准化 ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 4. 模型训练与对比 ==========
print("\n" + "=" * 60)
print("2. 三个模型对比")
print("=" * 60)

models = {
    '线性回归': LinearRegression(),
    'Ridge回归': Ridge(alpha=1.0),
    '随机森林': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

results = []
for name, model in models.items():
    # 线性模型用标准化后的数据，树模型用原始数据
    if name in ['线性回归', 'Ridge回归']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 5折交叉验证
    cv_X = X_train_scaled if name != '随机森林' else X_train
    cv_scores = cross_val_score(model, cv_X, y_train, cv=5,
                                scoring='neg_mean_absolute_error')

    results.append({
        '模型': name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R²': round(r2, 4),
        'CV_MAE均值': round(-cv_scores.mean(), 4)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# ========== 5. 随机森林特征重要性 ==========
print("\n" + "=" * 60)
print("3. 随机森林特征重要性")
print("=" * 60)

best_model = models['随机森林']
importance = pd.Series(best_model.feature_importances_,
                       index=X.columns).sort_values(ascending=False)
print(importance.round(4))

# ========== 6. 预测示例 ==========
print("\n" + "=" * 60)
print("4. 预测示例")
print("=" * 60)

sample = X_test.iloc[:3]
pred = best_model.predict(sample)
for i in range(3):
    actual = y_test.iloc[i]
    print(f"样本{i+1}: 预测房价={pred[i]:.3f}（10万美元）, "
          f"实际房价={actual:.3f}, 误差={pred[i]-actual:+.3f}")

print("\n✅ 回归实战完成")
