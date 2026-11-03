"""
项目6：A/B测试分析
"""
import numpy as np
import pandas as pd
from scipy import stats

print("=" * 60)
print("项目6：A/B测试分析")
print("=" * 60)

np.random.seed(42)

# 模拟实验数据
n_control = 10000
n_treatment = 10000

# 对照组转化率 5%，实验组 5.8%
control_conv = np.random.binomial(1, 0.05, n_control)
treatment_conv = np.random.binomial(1, 0.058, n_treatment)

print(f"对照组：{n_control} 人，转化率 {control_conv.mean():.2%}")
print(f"实验组：{n_treatment} 人，转化率 {treatment_conv.mean():.2%}")

# 两比例Z检验
p1 = control_conv.mean()
p2 = treatment_conv.mean()
p_pooled = (control_conv.sum() + treatment_conv.sum()) / (n_control + n_treatment)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_control + 1/n_treatment))
z = (p2 - p1) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

lift = (p2 - p1) / p1

print(f"\n增量：{lift:.2%}")
print(f"Z值：{z:.3f}")
print(f"p值：{p_value:.4f}")

if p_value < 0.05 and lift > 0:
    print("\n✅ 结论：实验组显著优于对照组，建议全量发布")
elif p_value < 0.05 and lift < 0:
    print("\n❌ 结论：实验组显著更差，不要发布")
else:
    print("\n⏳ 结论：差异不显著，继续观察或扩大样本")

print("\n✅ 项目6完成")
