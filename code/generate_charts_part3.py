"""
图表生成 Part 3：第32章绘图教程配图（20张）
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def save(fig, name, OUT):
    path = f'{OUT}/{name}'
    fig.savefig(path, bbox_inches='tight', facecolor='white', dpi=150)
    plt.close(fig)
    print(f'  ✓ {name}')

BLUE = '#2E86AB'; RED = '#FF6B6B'; GREEN = '#06D6A0'
YELLOW = '#FFD166'; PURPLE = '#9B5DE5'; ORANGE = '#F8961E'; GRAY = '#6C757D'
PALETTE = [BLUE, RED, GREEN, YELLOW, PURPLE, ORANGE, GRAY]

def gen_all(OUT):
    np.random.seed(42)

    # 1. anatomy
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([1,2,3,4], [1,4,2,5], 'o-', color=BLUE)
    ax.set_title('Figure标题', fontsize=14)
    ax.set_xlabel('X轴标签', fontsize=11)
    ax.set_ylabel('Y轴标签', fontsize=11)
    ax.annotate('图例', xy=(2,4), xytext=(3,4.5),
                arrowprops=dict(arrowstyle='->'))
    save(fig, 'ch32-anatomy.png', OUT)

    # 2. line-basic
    x = np.linspace(0, 10, 50)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, np.sin(x), color=BLUE, lw=2)
    ax.set_title('基础折线图'); ax.set_xlabel('X'); ax.set_ylabel('sin(x)')
    save(fig, 'ch32-line-basic.png', OUT)

    # 3. line-multi
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, np.sin(x), color=BLUE, label='sin(x)', lw=2)
    ax.plot(x, np.cos(x), color=RED, label='cos(x)', lw=2)
    ax.plot(x, np.sin(x)*np.cos(x), color=GREEN, label='sin*cos', lw=2)
    ax.set_title('多系列折线图'); ax.legend()
    save(fig, 'ch32-line-multi.png', OUT)

    # 4. line-annotate
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.sin(x)
    ax.plot(x, y, color=BLUE, lw=2)
    peak_idx = np.argmax(y)
    ax.annotate(f'峰值 ({x[peak_idx]:.1f}, {y[peak_idx]:.2f})',
                xy=(x[peak_idx], y[peak_idx]),
                xytext=(x[peak_idx]+1, y[peak_idx]-0.3),
                arrowprops=dict(arrowstyle='->', color=RED),
                fontsize=10, color=RED)
    ax.set_title('带标注的折线图')
    save(fig, 'ch32-line-annotate.png', OUT)

    # 5. bar-basic
    cats = ['A','B','C','D','E']
    vals = [23, 45, 56, 34, 67]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cats, vals, color=PALETTE)
    ax.set_title('基础柱状图'); ax.set_ylabel('数值')
    save(fig, 'ch32-bar-basic.png', OUT)

    # 6. bar-stacked
    fig, ax = plt.subplots(figsize=(7, 4))
    v1 = [20, 35, 30, 25, 40]
    v2 = [15, 25, 20, 30, 20]
    ax.bar(cats, v1, color=BLUE, label='系列1')
    ax.bar(cats, v2, bottom=v1, color=RED, label='系列2')
    ax.set_title('堆叠柱状图'); ax.legend()
    save(fig, 'ch32-bar-stacked.png', OUT)

    # 7. bar-horizontal
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(cats, vals, color=GREEN)
    ax.set_title('横向条形图')
    save(fig, 'ch32-bar-horizontal.png', OUT)

    # 8. scatter-basic
    sx = np.random.normal(0, 1, 100)
    sy = 2*sx + np.random.normal(0, 0.5, 100)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sx, sy, alpha=0.6, color=BLUE, s=40)
    ax.set_title('基础散点图')
    save(fig, 'ch32-scatter-basic.png', OUT)

    # 9. scatter-bubble
    fig, ax = plt.subplots(figsize=(7, 5))
    sizes = np.random.uniform(50, 400, 100)
    colors = np.random.uniform(0, 100, 100)
    sc = ax.scatter(sx, sy, s=sizes, c=colors, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='颜色值')
    ax.set_title('气泡图（大小+颜色双编码）')
    save(fig, 'ch32-scatter-bubble.png', OUT)

    # 10. histogram-basic
    data = np.random.normal(0, 1, 500)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data, bins=30, color=BLUE, alpha=0.7, edgecolor='white')
    ax.set_title('基础直方图')
    save(fig, 'ch32-histogram-basic.png', OUT)

    # 11. boxplot-basic
    data_box = [np.random.normal(0, std, 100) for std in [1, 1.5, 2, 2.5]]
    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data_box, tick_labels=['组1','组2','组3','组4'], patch_artist=True)
    for patch, c in zip(bp['boxes'], PALETTE):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_title('基础箱线图')
    save(fig, 'ch32-boxplot-basic.png', OUT)

    # 12. heatmap-basic
    corr = np.random.uniform(-1, 1, (6, 6))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    ax.set_title('基础热力图')
    save(fig, 'ch32-heatmap-basic.png', OUT)

    # 13. subplot-layout
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0,0].plot(x, np.sin(x), color=BLUE); axes[0,0].set_title('折线图')
    axes[0,1].bar(cats, vals, color=RED); axes[0,1].set_title('柱状图')
    axes[1,0].scatter(sx, sy, color=GREEN, s=20); axes[1,0].set_title('散点图')
    axes[1,1].hist(data, bins=20, color=PURPLE); axes[1,1].set_title('直方图')
    plt.tight_layout()
    save(fig, 'ch32-subplot-layout.png', OUT)

    # 14. seaborn-scatter
    df_sns = pd.DataFrame({'X': sx, 'Y': sy,
                           '类别': np.random.choice(['A','B','C'], 100)})
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df_sns, x='X', y='Y', hue='类别',
                    style='类别', s=80, ax=ax)
    ax.set_title('Seaborn散点图（按类别着色）')
    save(fig, 'ch32-seaborn-scatter.png', OUT)

    # 15. seaborn-box
    df_box = pd.DataFrame({
        '值': np.concatenate([np.random.normal(m, 1, 50) for m in [0, 2, 4]]),
        '类别': np.concatenate([[c]*50 for c in ['甲','乙','丙']])
    })
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df_box, x='类别', y='值', palette='Set2', ax=ax)
    ax.set_title('Seaborn箱线图')
    save(fig, 'ch32-seaborn-box.png', OUT)

    # 16. seaborn-heatmap
    pivot = np.random.uniform(10, 100, (5, 5))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                xticklabels=['1月','2月','3月','4月','5月'],
                yticklabels=['产品A','B','C','D','E'])
    ax.set_title('Seaborn热力图')
    save(fig, 'ch32-seaborn-heatmap.png', OUT)

    # 17. style-template
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # 默认风格
    axes[0].plot(x, np.sin(x), label='sin')
    axes[0].plot(x, np.cos(x), label='cos')
    axes[0].set_title('默认风格（matplotlib default）'); axes[0].legend()
    # 自定义风格
    axes[1].plot(x, np.sin(x), color=BLUE, lw=2.5, label='sin')
    axes[1].plot(x, np.cos(x), color=RED, lw=2.5, label='cos')
    axes[1].set_title('统一模板风格'); axes[1].legend()
    axes[1].grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    save(fig, 'ch32-style-template.png', OUT)

    # 18. bad-chart-3d (模拟3D饼图反例)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].pie([30, 25, 20, 15, 10], labels=['A','B','C','D','E'],
                autopct='%1.0f%%', colors=PALETTE)
    axes[0].set_title('✗ 3D饼图：角度扭曲，难比较', fontsize=11, color=RED)
    # 好图
    axes[1].barh(['A','B','C','D','E'], [30,25,20,15,10], color=BLUE)
    axes[1].set_title('✓ 横向条形图：精确比较', fontsize=11, color=GREEN)
    plt.tight_layout()
    save(fig, 'ch32-bad-chart-3d.png', OUT)

    # 19. bad-chart-axis
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(['A','B'], [100, 105], color=BLUE)
    axes[0].set_ylim(95, 108)
    axes[0].set_title('✗ Y轴不从0开始：夸大差异', fontsize=11, color=RED)
    axes[1].bar(['A','B'], [100, 105], color=BLUE)
    axes[1].set_ylim(0, 115)
    axes[1].set_title('✓ Y轴从0开始：真实比例', fontsize=11, color=GREEN)
    plt.tight_layout()
    save(fig, 'ch32-bad-chart-axis.png', OUT)

    # 20. gallery
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    plot_funcs = [
        lambda ax: ax.plot(x, np.sin(x), color=BLUE),
        lambda ax: ax.bar(cats, vals, color=RED),
        lambda ax: ax.scatter(sx, sy, s=15, color=GREEN),
        lambda ax: ax.hist(data, bins=20, color=PURPLE),
        lambda ax: ax.boxplot(data_box, patch_artist=True),
        lambda ax: ax.pie([30,25,20,25], colors=PALETTE),
        lambda ax: ax.plot(x, np.cumsum(np.random.randn(50)), color=ORANGE),
        lambda ax: ax.fill_between(x, np.sin(x), color=YELLOW, alpha=0.5),
    ]
    titles = ['折线', '柱状', '散点', '直方图', '箱线', '饼图', '面积', '填充']
    for i, (fn, t) in enumerate(zip(plot_funcs, titles)):
        ax = fig.add_subplot(gs[i//4, i%4])
        fn(ax); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Python 图表类型 Gallery', fontsize=14, fontweight='bold')
    save(fig, 'ch32-gallery.png', OUT)

    print(f'  ch32: 20 charts done.')
