"""
项目2：销售预测完整流程
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("=" * 60)
print("项目2：销售预测")
print("=" * 60)

# 1. 生成36个月历史数据
np.random.seed(42)
dates = pd.date_range('2024-01', periods=36, freq='ME')
trend = np.linspace(100, 180, 36)
seasonal = 15 * np.sin(2 * np.pi * np.arange(36) / 12)
sales = trend + seasonal + np.random.normal(0, 4, 36)

df = pd.DataFrame({'销售额': sales.round(1)}, index=dates)
print(f"历史数据：{len(df)} 个月")
print(df.tail(6))

# 2. Holt-Winters 预测
model = ExponentialSmoothing(df['销售额'], trend='add',
                              seasonal='add', seasonal_periods=12).fit()
forecast = model.forecast(6)
future_dates = pd.date_range(start=dates[-1], periods=7, freq='ME')[1:]

result = pd.DataFrame({
    '预测销售额': forecast.round(1),
    '预测下限': (forecast - 10).round(1),
    '预测上限': (forecast + 10).round(1)
}, index=future_dates)

print("\n未来6个月预测：")
print(result)

print("\n✅ 项目2完成")
