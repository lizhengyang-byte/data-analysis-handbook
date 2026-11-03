"""
第24章 时间序列预测：用合成销售数据演示
运行：python sales_forecast.py
依赖：pip install statsmodels scikit-learn pandas matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面环境
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ========== 1. 生成合成月度销售数据（24个月） ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=36, freq='ME')
# 趋势 + 季节性 + 噪声
trend = np.linspace(100, 160, 36)
seasonal = 20 * np.sin(2 * np.pi * np.arange(36) / 12)
noise = np.random.normal(0, 5, 36)
sales = trend + seasonal + noise

df = pd.DataFrame({'月份': dates, '销售额': sales.round(1)})
df.set_index('月份', inplace=True)
print(df.tail(12))

# ========== 2. STL 分解 ==========
print("\n" + "=" * 60)
print("2. STL 趋势/季节分解")
print("=" * 60)

stl = STL(df['销售额'], period=12, robust=True)
result = stl.fit()
print(f"趋势分量均值: {result.trend.mean():.1f}")
print(f"季节分量范围: [{result.seasonal.min():.1f}, {result.seasonal.max():.1f}]")
print(f"残差分量标准差: {result.resid.std():.1f}")

# ========== 3. 划分训练/测试 ==========
train = df.iloc[:-6]
test = df.iloc[-6:]
print(f"\n训练集: {len(train)} 个月, 测试集: {len(test)} 个月")

# ========== 4. Holt-Winters 预测 ==========
print("\n" + "=" * 60)
print("3. Holt-Winters 预测")
print("=" * 60)

model = ExponentialSmoothing(
    train['销售额'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()

forecast = model.forecast(len(test))
results = pd.DataFrame({
    '实际值': test['销售额'],
    '预测值': forecast.round(1)
})
print(results)

mae = mean_absolute_error(test['销售额'], forecast)
rmse = np.sqrt(mean_squared_error(test['销售额'], forecast))
print(f"\nMAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ========== 5. 下6个月预测 ==========
print("\n" + "=" * 60)
print("4. 未来6个月预测")
print("=" * 60)

model_full = ExponentialSmoothing(
    df['销售额'],
    trend='add',
    seasonal='add',
    seasonal_periods=12
).fit()
future = model_full.forecast(6)
future_dates = pd.date_range(start=df.index[-1], periods=7, freq='ME')[1:]
future_df = pd.DataFrame({'预测销售额': future.round(1)}, index=future_dates)
print(future_df)

# 保存图
fig, ax = plt.subplots(figsize=(10, 5))
df['销售额'].plot(ax=ax, label='历史销售', marker='o')
future.plot(ax=ax, label='未来预测', marker='s', linestyle='--')
ax.set_title('月度销售额预测')
ax.set_ylabel('销售额（万元）')
ax.legend()
plt.tight_layout()
plt.savefig('sales_forecast.png', dpi=100)
print("\n📊 预测图已保存为 sales_forecast.png")

print("\n✅ 时间序列预测完成")
