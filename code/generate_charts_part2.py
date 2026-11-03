"""
图表生成 Part 2：ch16-31 配图
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc

# 复用全局配置（由主脚本导入）
def setup(OUT, BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, GRAY, PALETTE):
    globals().update(locals())

def save(fig, name):
    path = f'{OUT}/{name}'
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f'  ✓ {name}')

# ============================================================
# 第16章 项目案例图
# ============================================================
def ch16_project1_roc():
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 500)
    # 三个模型的预测分数
    fpr1, tpr1, _ = roc_curve(y_true, np.random.normal(0.7, 0.2, 500) * y_true + np.random.normal(0.3, 0.1, 500))
    fpr2, tpr2, _ = roc_curve(y_true, np.random.normal(0.6, 0.25, 500))
    fpr3, tpr3, _ = roc_curve(y_true, np.random.normal(0.5, 0.3, 500))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr1, tpr1, color=BLUE, lw=2, label=f'随机森林 (AUC={auc(fpr1,tpr1):.2f})')
    ax.plot(fpr2, tpr2, color=GREEN, lw=2, label=f'逻辑回归 (AUC={auc(fpr2,tpr2):.2f})')
    ax.plot(fpr3, tpr3, color=GRAY, lw=2, linestyle='--', label=f'基线 (AUC={auc(fpr3,tpr3):.2f})')
    ax.plot([0,1],[0,1],'k:', alpha=0.3)
    ax.set_xlabel('假阳性率'); ax.set_ylabel('真阳性率')
    ax.set_title('项目1：客户流失预测ROC曲线', fontsize=12)
    ax.legend()
    save(fig, 'ch16-project1-churn-roc.png')

def ch16_project2_forecast():
    np.random.seed(42)
    dates = pd.date_range('2025-01', periods=30, freq='ME')
    trend = np.linspace(100, 150, 30)
    seasonal = 10 * np.sin(2*np.pi*np.arange(30)/12)
    actual = trend + seasonal + np.random.normal(0, 3, 30)
    forecast = trend[24:] + seasonal[24:] + np.random.normal(0, 2, 6)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(dates[:24], actual[:24], 'o-', color=BLUE, label='历史销售')
    ax.plot(dates[24:], actual[24:], 's-', color=GREEN, label='实际值')
    ax.plot(dates[24:], forecast, '^--', color=RED, label='预测值')
    ax.axvline(dates[24], color=GRAY, linestyle=':')
    ax.set_title('项目2：销售预测 vs 实际', fontsize=12)
    ax.legend()
    plt.xticks(rotation=45)
    save(fig, 'ch16-project2-forecast.png')

def ch16_project3_sentiment():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ['正面', '中性', '负面']
    sizes = [65, 20, 15]
    colors = [GREEN, YELLOW, RED]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
    axes[0].set_title('评论情感分布')
    # 关键词频率
    words = ['质量好','物流快','客服好','性价比','包装好','质量差','物流慢','客服差']
    counts = [45, 38, 30, 25, 20, 28, 22, 18]
    colors_bar = [GREEN if c > counts[4] else RED for c in counts]
    axes[1].barh(words, counts, color=colors_bar)
    axes[1].set_title('关键词出现频次')
    plt.tight_layout()
    save(fig, 'ch16-project3-sentiment.png')

def ch16_project4_fraud():
    np.random.seed(42)
    normal = np.random.multivariate_normal([100, 14], [[50,0],[0,2]], 800)
    fraud = np.random.multivariate_normal([500, 3], [[100,0],[0,1]], 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(normal[:,0], normal[:,1], alpha=0.5, color=BLUE, s=20, label='正常交易')
    ax.scatter(fraud[:,0], fraud[:,1], alpha=0.8, color=RED, s=40, marker='x', label='欺诈交易')
    ax.set_xlabel('交易金额（元）'); ax.set_ylabel('交易小时')
    ax.set_title('项目4：欺诈检测散点图', fontsize=12)
    ax.legend()
    save(fig, 'ch16-project4-fraud-scatter.png')

def ch16_project5_segmentation():
    X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.5)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(4):
        ax.scatter(X[y==i,0], X[y==i,1], s=30, alpha=0.7, label=f'群体{i+1}')
    ax.set_title('项目5：客户分群（K-Means）', fontsize=12)
    ax.legend()
    save(fig, 'ch16-project5-segmentation.png')

def ch16_project6_ab():
    groups = ['对照组', '实验组']
    conv = [5.0, 5.8]
    ci = [0.3, 0.3]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(groups, conv, yerr=ci, capsize=10, color=[GRAY, BLUE], alpha=0.8)
    ax.set_ylabel('转化率（%）')
    ax.set_title('项目6：A/B测试结果（p<0.05）', fontsize=12)
    ax.set_ylim(0, 8)
    for bar, v in zip(bars, conv):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.3, f'{v}%', ha='center', fontweight='bold')
    save(fig, 'ch16-project6-ab-result.png')

# ============================================================
# 第19章
# ============================================================
def ch19_ml_timeline():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    steps = ['业务理解', '数据准备', '特征工程', '模型训练', '评估调优', '部署监控']
    descs = ['明确问题\n定义目标', '清洗合并\n探索分析', '构造特征\n选择子集', '选模型\n训练拟合', '交叉验证\n调参', '上线API\n漂移监控']
    for i, (s, d) in enumerate(zip(steps, descs)):
        x = 0.7 + i * 1.55
        ax.add_patch(plt.Rectangle((x-0.6, 1.2), 1.2, 1, color=BLUE, alpha=0.85))
        ax.text(x, 1.9, s, ha='center', va='center', color='white', fontweight='bold', fontsize=9)
        ax.text(x, 0.6, d, ha='center', va='center', fontsize=8, color=GRAY)
        if i < len(steps)-1:
            ax.annotate('', xy=(x+0.95, 1.7), xytext=(x+0.6, 1.7),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))
    ax.set_title('机器学习标准建模流程（CRISP-DM）', fontsize=13, fontweight='bold')
    save(fig, 'ch19-ml-timeline.png')

def ch19_algorithm_map():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off')
    nodes = [
        (5, 7.3, '有标签？', BLUE),
        (2.5, 5.5, '连续值？', GREEN),
        (7.5, 5.5, '离散值？', YELLOW),
        (1, 3.5, '线性回归\nRidge/Lasso', '#888'),
        (4, 3.5, '树模型\nRF/XGBoost', '#888'),
        (6.5, 3.5, '逻辑回归\nSVM', '#888'),
        (9, 3.5, '树模型\nRF/GBDT', '#888'),
        (5, 1.5, '无标签 → K-Means / DBSCAN / PCA', PURPLE),
    ]
    for x, y, t, c in nodes:
        ax.add_patch(plt.Rectangle((x-1, y-0.4), 2, 0.8, color=c, alpha=0.85))
        ax.text(x, y, t, ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    edges = [((5,6.9),(2.5,5.9)), ((5,6.9),(7.5,5.9)),
             ((2.5,5.1),(1,3.9)), ((2.5,5.1),(4,3.9)),
             ((7.5,5.1),(6.5,3.9)), ((7.5,5.1),(9,3.9)),
             ((5,6.9),(5,1.9))]
    for (x1,y1),(x2,y2) in edges:
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))
    ax.set_title('算法选择决策图', fontsize=13, fontweight='bold')
    save(fig, 'ch19-algorithm-map.png')

# ============================================================
# 第21章
# ============================================================
def ch21_regression_fit():
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + 3 + np.random.normal(0, 3, 50)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, color=BLUE, label='样本')
    z = np.polyfit(x, y, 1)
    ax.plot(x, np.polyval(z, x), color=RED, lw=2, label=f'拟合线 y={z[0]:.1f}x+{z[1]:.1f}')
    ax.set_xlabel('面积（平米）'); ax.set_ylabel('房价（万元）')
    ax.set_title('线性回归拟合示例', fontsize=12)
    ax.legend()
    save(fig, 'ch21-regression-fit.png')

def ch21_residual_plot():
    np.random.seed(42)
    x = np.linspace(0, 10, 50)
    y = 2.5 * x + 3 + np.random.normal(0, 3, 50)
    z = np.polyfit(x, y, 1)
    resid = y - np.polyval(z, x)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, resid, color=PURPLE, alpha=0.7)
    ax.axhline(0, color=RED, linestyle='--')
    ax.set_xlabel('拟合值'); ax.set_ylabel('残差')
    ax.set_title('残差图（应随机分布在0附近）', fontsize=12)
    save(fig, 'ch21-residual-plot.png')

# ============================================================
# 第22章
# ============================================================
def ch22_roc_curve():
    np.random.seed(42)
    y = np.random.binomial(1, 0.4, 300)
    fig, ax = plt.subplots(figsize=(6, 5))
    for color, name, sep in [(BLUE, 'RF', 0.6), (GREEN, 'GBDT', 0.5), (RED, 'LR', 0.3)]:
        score = np.random.normal(sep*y, 0.3)
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} AUC={auc(fpr,tpr):.2f}')
    ax.plot([0,1],[0,1],'k:',alpha=0.3)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('多模型ROC对比', fontsize=12)
    ax.legend()
    save(fig, 'ch22-roc-curve.png')

def ch22_confusion_matrix():
    cm = np.array([[85, 15], [10, 90]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测流失', '预测留存'],
                yticklabels=['实际流失', '实际留存'], ax=ax)
    ax.set_title('混淆矩阵示例', fontsize=12)
    save(fig, 'ch22-confusion-matrix.png')

# ============================================================
# 第23章
# ============================================================
def ch23_kmeans():
    X, y = make_blobs(n_samples=300, centers=4, random_state=42, cluster_std=1.2)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i in range(4):
        ax.scatter(X[y==i,0], X[y==i,1], s=25, alpha=0.7, label=f'K={i+1}组')
    centers = np.array([X[y==i].mean(axis=0) for i in range(4)])
    ax.scatter(centers[:,0], centers[:,1], marker='*', s=300, c='gold',
               edgecolors='black', label='中心点')
    ax.set_title('K-Means聚类效果', fontsize=12)
    ax.legend()
    save(fig, 'ch23-kmeans-cluster.png')

def ch23_pca():
    np.random.seed(42)
    X = np.random.multivariate_normal([0,0], [[3,2],[2,2]], 300)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.6, color=PURPLE, s=25)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)')
    ax.set_title('PCA降维可视化', fontsize=12)
    save(fig, 'ch23-pca-scatter.png')

# ============================================================
# 第24章
# ============================================================
def ch24_decomposition():
    np.random.seed(42)
    t = np.arange(60)
    trend = np.linspace(100, 140, 60)
    seasonal = 10 * np.sin(2*np.pi*t/12)
    resid = np.random.normal(0, 3, 60)
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(t, trend+seasonal+resid, color=BLUE); axes[0].set_title('原始序列', fontsize=10)
    axes[1].plot(t, trend, color=GREEN); axes[1].set_title('趋势分量', fontsize=10)
    axes[2].plot(t, seasonal, color=RED); axes[2].set_title('季节分量', fontsize=10)
    axes[3].plot(t, resid, color=GRAY); axes[3].set_title('残差分量', fontsize=10)
    plt.tight_layout()
    save(fig, 'ch24-ts-decomposition.png')

def ch24_forecast_vs_actual():
    np.random.seed(42)
    t = np.arange(36)
    actual = 100 + 2*t + 8*np.sin(2*np.pi*t/12) + np.random.normal(0,3,36)
    forecast = 100 + 2*t[30:] + 8*np.sin(2*np.pi*t[30:]/12)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t[:30], actual[:30], 'o-', color=BLUE, label='历史')
    ax.plot(t[30:], actual[30:], 's-', color=GREEN, label='实际')
    ax.plot(t[30:], forecast, '^--', color=RED, label='预测')
    ax.axvline(30, color=GRAY, linestyle=':')
    ax.set_title('时序预测 vs 实际', fontsize=12)
    ax.legend()
    save(fig, 'ch24-forecast-vs-actual.png')

# ============================================================
# 第25章
# ============================================================
def ch25_anomaly():
    np.random.seed(42)
    normal = np.random.normal(0, 1, (500, 2))
    anomaly = np.random.uniform(-4, 4, (20, 2))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(normal[:,0], normal[:,1], alpha=0.5, color=BLUE, s=20, label='正常')
    ax.scatter(anomaly[:,0], anomaly[:,1], alpha=0.9, color=RED, s=60, marker='X', label='异常')
    ax.set_title('孤立森林异常检测结果', fontsize=12)
    ax.legend()
    save(fig, 'ch25-anomaly-scatter.png')

# ============================================================
# 第31章
# ============================================================
def ch31_salary():
    roles = ['实习生', '初级', '中级', '高级', '专家', '总监']
    low = [4, 8, 15, 25, 40, 60]
    high = [6, 15, 25, 40, 70, 120]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(roles))
    ax.bar(x, np.array(high)-np.array(low), bottom=low, color=BLUE, alpha=0.7)
    ax.plot(x, [(l+h)/2 for l,h in zip(low,high)], 'ro-', label='中位数')
    ax.set_xticks(x); ax.set_xticklabels(roles)
    ax.set_ylabel('月薪（K）')
    ax.set_title('数据分析师薪资范围参考（一线城市）', fontsize=12)
    ax.legend()
    save(fig, 'ch31-salary-range.png')

print('Part 2 functions defined.')
