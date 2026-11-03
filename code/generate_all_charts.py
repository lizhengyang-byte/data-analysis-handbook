"""
一键生成所有手册配图
运行：python code/generate_all_charts.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 全局配置 ==========
OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images')
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

BLUE = '#2E86AB'
RED = '#FF6B6B'
GREEN = '#06D6A0'
YELLOW = '#FFD166'
PURPLE = '#9B5DE5'
GRAY = '#6C757D'
ORANGE = '#F8961E'
PALETTE = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, GRAY]
sns.set_palette(PALETTE)
np.random.seed(42)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✓ {name}')

# 导入各部分
sys.path.insert(0, os.path.dirname(__file__))

# Part 1: ch01-12
from generate_charts_part1 import (
    ch01_mece_tree, ch01_aarrr_funnel, ch02_data_sources,
    ch03_missing_heatmap, ch03_outlier_boxplot,
    ch04_histogram, ch04_boxplot_city, ch04_scatter_correlation, ch04_correlation_matrix,
    ch05_normal_dist, ch05_pvalue, ch05_confidence_interval,
    ch06_good_bad, ch06_color_palette, ch07_sql_order,
    ch09_funnel, ch09_rfm_matrix, ch09_retention_curve,
    ch10_ab_flow, ch10_sample_size, ch12_gmv_waterfall, ch12_yoy_mom,
)

# Part 2: ch16-31
import generate_charts_part2 as p2
p2.setup(OUT, BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, GRAY, PALETTE)

# Part 3: ch32
import generate_charts_part3 as p3

print('开始生成图表...\n')

# === ch01-12 ===
print('--- ch01-12 ---')
ch01_mece_tree()
ch01_aarrr_funnel()
ch02_data_sources()
ch03_missing_heatmap()
ch03_outlier_boxplot()
ch04_histogram()
ch04_boxplot_city()
ch04_scatter_correlation()
ch04_correlation_matrix()
ch05_normal_dist()
ch05_pvalue()
ch05_confidence_interval()
ch06_good_bad()
ch06_color_palette()
ch07_sql_order()
ch09_funnel()
ch09_rfm_matrix()
ch09_retention_curve()
ch10_ab_flow()
ch10_sample_size()
ch12_gmv_waterfall()
ch12_yoy_mom()

# === ch16-31 ===
print('\n--- ch16-31 ---')
p2.ch16_project1_roc()
p2.ch16_project2_forecast()
p2.ch16_project3_sentiment()
p2.ch16_project4_fraud()
p2.ch16_project5_segmentation()
p2.ch16_project6_ab()
p2.ch19_ml_timeline()
p2.ch19_algorithm_map()
p2.ch21_regression_fit()
p2.ch21_residual_plot()
p2.ch22_roc_curve()
p2.ch22_confusion_matrix()
p2.ch23_kmeans()
p2.ch23_pca()
p2.ch24_decomposition()
p2.ch24_forecast_vs_actual()
p2.ch25_anomaly()
p2.ch31_salary()

# === ch32 ===
print('\n--- ch32 ---')
p3.gen_all(OUT)

# 统计
count = len([f for f in os.listdir(OUT) if f.endswith('.png')])
print(f'\n✅ 完成！共生成 {count} 张图表，保存在 {OUT}')
