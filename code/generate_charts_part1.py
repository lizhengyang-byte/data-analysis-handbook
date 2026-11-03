"""
一键生成手册所有配图
运行：python code/generate_all_charts.py
输出：assets/images/*.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns

# ========== 全局配置 ==========
OUT = os.path.join(os.path.dirname(__file__), '..', 'assets', 'images')
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

# ============================================================
# 第01章
# ============================================================
def ch01_mece_tree():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    boxes = [
        (5, 5.3, 'GMV'),
        (2, 3.5, '付费用户数'), (8, 3.5, '客单价'),
        (0.8, 1.5, '新付费\n用户'), (3.2, 1.5, '老付费\n用户'),
        (6.8, 1.5, '购买频次'), (9.2, 1.5, '件单价'),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch((x-0.9, y-0.35), 1.8, 0.7,
            boxstyle="round,pad=0.1", fc=BLUE, ec='white', lw=2, alpha=0.9))
        ax.text(x, y, t, ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    # 连线
    for (x1, y1), (x2, y2) in [((5,4.95),(2,3.85)), ((5,4.95),(8,3.85)),
                                ((2,3.15),(0.8,1.85)), ((2,3.15),(3.2,1.85)),
                                ((8,3.15),(6.8,1.85)), ((8,3.15),(9.2,1.85))]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-', color=GRAY, lw=1.5))
    ax.set_title('MECE拆解示例：GMV', fontsize=13, fontweight='bold')
    save(fig, 'ch01-mece-tree.png')

def ch01_aarrr_funnel():
    stages = ['获取\nAcquisition', '激活\nActivation', '留存\nRetention',
              '变现\nRevenue', '推荐\nReferral']
    values = [10000, 6500, 3500, 1500, 400]
    colors = [BLUE, '#3FA7D6', GREEN, YELLOW, RED]
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = range(len(stages))
    max_v = max(values)
    for i, (stage, v, c) in enumerate(zip(stages, values, colors)):
        width = v / max_v * 8
        left = (8 - width) / 2
        ax.barh(i, width, left=left, height=0.7, color=c, alpha=0.85)
        ax.text(4, i, f'{stage}  {v:,}', ha='center', va='center',
                color='white', fontweight='bold', fontsize=10)
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_xlim(-0.5, 8.5); ax.invert_yaxis()
    ax.set_title('AARRR海盗模型漏斗', fontsize=13, fontweight='bold')
    save(fig, 'ch01-aarrr-funnel.png')

# ============================================================
# 第02章
# ============================================================
def ch02_data_sources():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    sources = [
        (2, 5.5, '业务数据库\nMySQL/PG', BLUE),
        (5, 5.5, '日志文件\nNginx/App', GREEN),
        (8, 5.5, '第三方API\n微信/支付', YELLOW),
        (2, 3, '爬虫数据\nPython/Scrapy', PURPLE),
        (5, 3, '问卷/调研\nExcel导入', ORANGE),
        (8, 3, '数据仓库\nHive/CH', RED),
    ]
    for x, y, t, c in sources:
        ax.add_patch(FancyBboxPatch((x-1.2, y-0.6), 2.4, 1.2,
            boxstyle="round,pad=0.1", fc=c, ec='white', alpha=0.85))
        ax.text(x, y, t, ha='center', va='center', color='white',
                fontsize=9, fontweight='bold')
    ax.add_patch(FancyBboxPatch((3.5, 0.5), 3, 1,
        boxstyle="round,pad=0.1", fc='#333', ec='white'))
    ax.text(5, 1, '数据落地层\nCSV / DataFrame', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')
    for x, y, _, _ in sources:
        ax.annotate('', xy=(5, 1.5), xytext=(x, y-0.6),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))
    ax.set_title('数据分析数据源全景', fontsize=13, fontweight='bold')
    save(fig, 'ch02-data-sources.png')

# ============================================================
# 第03章
# ============================================================
def ch03_missing_heatmap():
    np.random.seed(42)
    data = np.random.choice([0, np.nan], size=(20, 8), p=[0.85, 0.15])
    df = pd.DataFrame(data, columns=[f'特征{i+1}' for i in range(8)])
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.isna(), cbar=False, cmap=['#2E86AB', '#FF6B6B'], ax=ax)
    ax.set_title('缺失值分布热力图（蓝色=有值，红色=缺失）', fontsize=12)
    ax.set_xlabel(''); ax.set_ylabel('样本行')
    save(fig, 'ch03-missing-heatmap.png')

def ch03_outlier_boxplot():
    np.random.seed(42)
    normal = np.random.normal(100, 15, 100)
    outliers = np.array([180, 195, 220, -20, -10])
    data = np.concatenate([normal, outliers])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(data, vert=True, patch_artist=True,
               boxprops=dict(facecolor=BLUE, alpha=0.5),
               flierprops=dict(marker='o', markerfacecolor=RED, markersize=8))
    ax.set_xticklabels(['订单金额'])
    ax.set_title('箱线图识别异常值（红点=异常）', fontsize=12)
    ax.set_ylabel('金额（元）')
    save(fig, 'ch03-outlier-boxplot.png')

# ============================================================
# 第04章
# ============================================================
def ch04_histogram():
    np.random.seed(42)
    data = np.random.lognormal(5, 0.5, 500)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(data, bins=40, color=BLUE, alpha=0.7, edgecolor='white')
    ax.set_title('订单金额分布直方图', fontsize=12)
    ax.set_xlabel('订单金额（元）'); ax.set_ylabel('频数')
    ax.axvline(data.mean(), color=RED, linestyle='--', label=f'均值 {data.mean():.0f}')
    ax.legend()
    save(fig, 'ch04-histogram.png')

def ch04_boxplot_city():
    np.random.seed(42)
    cities = ['北京', '上海', '广州', '深圳', '成都']
    data = [np.random.normal(m, 20, 100) for m in [200, 250, 180, 220, 150]]
    fig, ax = plt.subplots(figsize=(8, 4))
    bp = ax.boxplot(data, tick_labels=cities, patch_artist=True)
    for patch, c in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_title('各城市订单金额箱线图', fontsize=12)
    ax.set_ylabel('订单金额（元）')
    save(fig, 'ch04-boxplot-city.png')

def ch04_scatter_correlation():
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    y = 0.7 * x + np.random.normal(0, 0.5, 200)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.6, color=BLUE, s=30)
    z = np.polyfit(x, y, 1)
    ax.plot(x, np.polyval(z, x), color=RED, lw=2, label=f'r={np.corrcoef(x,y)[0,1]:.2f}')
    ax.set_xlabel('广告投入（万元）'); ax.set_ylabel('销售额（万元）')
    ax.set_title('广告投入 vs 销售额散点图', fontsize=12)
    ax.legend()
    save(fig, 'ch04-scatter-correlation.png')

def ch04_correlation_matrix():
    np.random.seed(42)
    cols = ['DAU', '新增', '留存', '转化率', '客单价', 'GMV']
    data = np.random.randn(100, 6)
    corr = np.corrcoef(data.T)
    # 让GMV和DAU强相关
    corr[0, 5] = corr[5, 0] = 0.85
    corr[3, 5] = corr[5, 3] = 0.72
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=cols, yticklabels=cols, ax=ax)
    ax.set_title('特征相关系数矩阵', fontsize=12)
    save(fig, 'ch04-correlation-matrix.png')

# ============================================================
# 第05章
# ============================================================
def ch05_normal_dist():
    x = np.linspace(-4, 4, 200)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color=BLUE, lw=2)
    ax.fill_between(x, y, alpha=0.3, color=BLUE)
    ax.axvline(0, color=RED, linestyle='--', label='均值=0')
    ax.set_title('标准正态分布 N(0,1)', fontsize=12)
    ax.set_xlabel('z 值'); ax.set_ylabel('概率密度')
    ax.legend()
    save(fig, 'ch05-normal-distribution.png')

def ch05_pvalue():
    x = np.linspace(-4, 4, 300)
    y = (1/np.sqrt(2*np.pi)) * np.exp(-x**2/2)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color=BLUE, lw=2)
    # 右侧拒绝域
    mask = x > 1.96
    ax.fill_between(x[mask], y[mask], color=RED, alpha=0.7, label='拒绝域 α=0.025')
    mask2 = x < -1.96
    ax.fill_between(x[mask2], y[mask2], color=RED, alpha=0.7)
    ax.axvline(1.96, color=RED, linestyle=':')
    ax.axvline(-1.96, color=RED, linestyle=':')
    ax.set_title('p值与拒绝域示意（双侧检验）', fontsize=12)
    ax.legend()
    save(fig, 'ch05-pvalue-illustration.png')

def ch05_confidence_interval():
    np.random.seed(42)
    groups = ['A组', 'B组', 'C组', 'D组', 'E组']
    means = [50, 55, 52, 60, 58]
    cis = [(48,52), (53,57), (50,54), (57,63), (55,61)]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(groups))
    ax.errorbar(x, means, yerr=[[m-l for m,(l,h) in zip(means,cis)],
                                [h-m for m,(l,h) in zip(means,cis)]],
                fmt='o', color=BLUE, ecolor=GRAY, capsize=8, markersize=10)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.axhline(55, color=RED, linestyle='--', label='基准值 55')
    ax.set_title('各组均值与95%置信区间', fontsize=12)
    ax.legend()
    save(fig, 'ch05-confidence-interval.png')

# ============================================================
# 第06章
# ============================================================
def ch06_good_bad():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 坏图：3D饼图感、颜色杂
    axes[0].pie([30, 25, 20, 15, 10], labels=['A','B','C','D','E'],
                colors=PALETTE, autopct='%1.0f%%')
    axes[0].set_title('✗ 坏图：饼图过多切片、无排序', fontsize=11, color=RED)
    # 好图：排序横向条形图
    vals = [30, 25, 20, 15, 10]
    labels = ['A类', 'B类', 'C类', 'D类', 'E类']
    axes[1].barh(labels, vals, color=BLUE, alpha=0.8)
    axes[1].set_title('✓ 好图：排序条形图，一眼看大小', fontsize=11, color=GREEN)
    plt.tight_layout()
    save(fig, 'ch06-good-bad-chart.png')

def ch06_color_palette():
    fig, ax = plt.subplots(figsize=(8, 2))
    for i, c in enumerate(PALETTE):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
        ax.text(i+0.5, 0.5, c, ha='center', va='center', color='white', fontsize=9)
    ax.set_xlim(0, len(PALETTE)); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_title('推荐配色方案', fontsize=12)
    save(fig, 'ch06-color-palette.png')

# ============================================================
# 第07章
# ============================================================
def ch07_sql_order():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis('off')
    steps = [('FROM / JOIN', 1), ('WHERE', 2), ('GROUP BY', 3),
             ('HAVING', 4), ('SELECT', 5), ('ORDER BY', 6), ('LIMIT', 7)]
    for i, (t, y) in enumerate(steps):
        c = BLUE if i % 2 == 0 else GREEN
        ax.add_patch(FancyBboxPatch((2, y-0.35), 6, 0.7,
            boxstyle="round,pad=0.1", fc=c, alpha=0.85))
        ax.text(5, y, t, ha='center', va='center', color='white', fontweight='bold')
    ax.set_title('SQL执行顺序（从上到下）', fontsize=13)
    save(fig, 'ch07-sql-execution-order.png')

# ============================================================
# 第09章
# ============================================================
def ch09_funnel():
    stages = ['访问首页', '浏览商品', '加入购物车', '提交订单', '支付成功']
    vals = [10000, 5000, 2000, 1200, 1000]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [BLUE, '#3FA7D6', GREEN, YELLOW, RED]
    for i, (s, v, c) in enumerate(zip(stages, vals, colors)):
        width = v / max(vals) * 8
        left = (8 - width) / 2
        ax.barh(i, width, left=left, height=0.7, color=c, alpha=0.85)
        rate = f'{v/vals[0]*100:.0f}%' if i == 0 else f'{v/vals[i-1]*100:.0f}%转化'
        ax.text(4, i, f'{s}  {v:,}  ({rate})', ha='center', va='center',
                color='white', fontweight='bold')
    ax.invert_yaxis(); ax.axis('off')
    ax.set_title('电商转化漏斗', fontsize=13, fontweight='bold')
    save(fig, 'ch09-funnel.png')

def ch09_rfm_matrix():
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    # 象限分隔
    ax.axhline(5, color=GRAY, lw=1)
    ax.axvline(5, color=GRAY, lw=1)
    quads = [(2.5, 7.5, '重要价值客户', GREEN), (7.5, 7.5, '重要保持客户', BLUE),
             (2.5, 2.5, '一般价值客户', YELLOW), (7.5, 2.5, '一般挽留客户', RED)]
    for x, y, t, c in quads:
        ax.add_patch(plt.Rectangle((x-2.2, y-2.2), 4.4, 4.4, color=c, alpha=0.3))
        ax.text(x, y, t, ha='center', va='center', fontsize=12, fontweight='bold')
    ax.set_xlabel('R（最近购买）→ 越右越近', fontsize=11)
    ax.set_ylabel('F+M（频次+金额）→ 越上越高', fontsize=11)
    ax.set_title('RFM客户分群矩阵', fontsize=13, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, 'ch09-rfm-matrix.png')

def ch09_retention_curve():
    cohorts = ['新用户', '活跃用户', 'VIP']
    x = np.arange(0, 30)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (c, label) in enumerate(zip([BLUE, GREEN, RED], cohorts)):
        y = 100 * np.exp(-x * (0.05 + i*0.04))
        ax.plot(x, y, color=c, label=label, lw=2)
    ax.set_xlabel('注册后天数'); ax.set_ylabel('留存率（%）')
    ax.set_title('用户留存曲线衰减', fontsize=12)
    ax.legend()
    save(fig, 'ch09-retention-curve.png')

# ============================================================
# 第10章
# ============================================================
def ch10_ab_flow():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    steps = ['确定假设', '计算样本量', '分流实验', '收集数据', '显著性检验', '决策上线']
    for i, s in enumerate(steps):
        x = 0.5 + i * 1.6
        ax.add_patch(FancyBboxPatch((x-0.7, 1), 1.4, 1,
            boxstyle="round,pad=0.1", fc=BLUE if i%2==0 else GREEN, alpha=0.85))
        ax.text(x, 1.5, s, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if i < len(steps) - 1:
            ax.annotate('', xy=(x+0.85, 1.5), xytext=(x+0.7, 1.5),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=2))
    ax.set_title('A/B测试标准流程', fontsize=13)
    save(fig, 'ch10-ab-test-flow.png')

def ch10_sample_size():
    p1 = 0.05
    p2_range = np.linspace(0.05, 0.08, 100)  # MDE
    z = 1.96 + 0.84
    n = 2 * (z**2) * 0.05 * 0.95 / (p2_range - p1)**2
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot((p2_range - p1) * 100, n, color=BLUE, lw=2)
    ax.set_xlabel('MDE 最小可检测效应（百分点）')
    ax.set_ylabel('每组所需样本量')
    ax.set_title('MDE与样本量关系（基线转化率5%）', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    save(fig, 'ch10-sample-size.png')

# ============================================================
# 第12章
# ============================================================
def ch12_gmv_waterfall():
    labels = ['上期GMV', '新用户下降', '老用户下降', '客单价提升', '本期GMV']
    values = [1000, -150, -80, 120, 890]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    cumulative = 0
    for i, (l, v) in enumerate(zip(labels, values)):
        if i in [0, len(labels)-1]:
            ax.bar(i, v, color=BLUE if i==0 else GREEN, alpha=0.8)
            cumulative = v
        else:
            color = RED if v < 0 else GREEN
            bottom = cumulative + v if v < 0 else cumulative
            ax.bar(i, abs(v), bottom=bottom, color=color, alpha=0.8)
            cumulative += v
        ax.text(i, v if i in [0,len(labels)-1] else cumulative + (5 if v>0 else -20),
                f'{v:+}' if i not in [0,len(labels)-1] else str(v),
                ha='center', fontsize=9)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('GMV变动瀑布图（万元）', fontsize=12)
    save(fig, 'ch12-gmv-waterfall.png')

def ch12_yoy_mom():
    months = ['1月','2月','3月','4月','5月','6月']
    revenue = [800, 750, 900, 950, 1000, 1100]
    yoy = [5, 8, 12, 10, 15, 18]
    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(months, revenue, color=BLUE, alpha=0.7, label='营收（万元）')
    ax1.set_ylabel('营收（万元）', color=BLUE)
    ax2 = ax1.twinx()
    ax2.plot(months, yoy, color=RED, marker='o', lw=2, label='同比增速（%）')
    ax2.set_ylabel('同比增速（%）', color=RED)
    ax1.set_title('月度营收与同比增速', fontsize=12)
    save(fig, 'ch12-yoy-mom.png')

print('Part 1 (ch01-12) defined.')
