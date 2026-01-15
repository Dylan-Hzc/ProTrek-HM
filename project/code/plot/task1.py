import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Iterable
import matplotlib.ticker as ticker

# 1. 设置输出路径
output_dir = r"D:\Data\experiments\2025\12\24\step1"
os.makedirs(output_dir, exist_ok=True)

# 2. 读取数据
df = pd.read_csv(r"D:\Data\experiments\2025\12\24\step1\step1.csv")

# 【强制修正1】在筛选前就确保 blocks 是字符串，防止后续类型混乱
df['blocks'] = df['blocks'].astype(str)
df_plot = df[df['blocks'] == '0'].copy()

# 【强制修正2】确保 mutation_rate 绝对是字符串 (Categorical)，解决"全蓝"问题
df_plot['mutation_rate'] = df_plot['mutation_rate'].astype(str)

def _slugify(text: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in str(text))

# 【强制修正3】通过全局配置强制加粗，防止被 seaborn 样式覆盖
# 这种方式比单独在 label 里设置更强硬
sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    'font.weight': 'bold',          # 全局字体加粗
    'axes.labelweight': 'bold',     # 坐标轴标签加粗
    'axes.titleweight': 'bold',     # 标题加粗
    'axes.titlesize': 18,
    'axes.labelsize': 16,           # 稍微调大字号
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

datasets: Iterable[str] = sorted(df_plot['dataset'].unique())

# --- 图 1: Recall (Recall@10 vs k) ---
for ds in datasets:
    data_ds = df_plot[df_plot['dataset'] == ds].copy()
    if data_ds.empty:
        continue

    data_ds = data_ds.sort_values('k')

    plt.figure(figsize=(8, 5))
    
    # 【强制修正4】使用 'tab10' 调色板，确保颜色差异巨大（不再全是蓝色）
    ax = sns.lineplot(
        data=data_ds,
        x='k', y='recall@10',
        hue='mutation_rate', style='mutation_rate',
        marker='o', linewidth=2.5, markersize=8,
        palette='tab10' 
    )
    
    # 再次显式设置加粗 (双重保险)
    ax.set_xlabel('k', fontweight='bold', fontsize=16)
    ax.set_ylabel('Recall@10', fontweight='bold', fontsize=16)
    
    # 强制 X 轴只显示整数刻度
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    try:
        ymax = data_ds['recall@10'].max()
        ymin = data_ds['recall@10'].min()
        margin = max(0.02, (ymax - ymin) * 0.05)
        ax.set_ylim(max(0, ymin - margin), min(1.0, ymax + margin))
    except Exception:
        pass
        
    ax.legend(title='mutation_rate', frameon=False)
    sns.despine()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"fig1_recall_k_{_slugify(ds)}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# --- 图 2: Efficiency (Candidates & Time) ---
for ds in datasets:
    data_ds = df_plot[df_plot['dataset'] == ds]
    if data_ds.empty:
        continue

    data_agg = (
        data_ds.groupby('k')[['avg_candidates', 'avg_time_ms']]
        .mean()
        .reset_index()
        .sort_values('k')
    )

    fig, ax_left = plt.subplots(figsize=(8.8, 5.2))
    c1, c2 = 'tab:blue', 'tab:red'

    line1, = ax_left.plot(
        data_agg['k'], data_agg['avg_candidates'],
        color=c1, marker='s', linewidth=2.0, markersize=7, label='Avg Candidates'
    )
    
    # 显式加粗
    ax_left.set_xlabel('k', fontweight='bold', fontsize=16)
    ax_left.set_ylabel('Avg Candidates', color=c1, fontweight='bold', fontsize=16)
    
    ax_left.tick_params(axis='y', labelcolor=c1)
    ax_left.grid(False)

    # 强制 X 轴只显示整数刻度
    ax_left.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax_right = ax_left.twinx()
    line2, = ax_right.plot(
        data_agg['k'], data_agg['avg_time_ms'],
        color=c2, marker='^', linestyle='--', linewidth=2.0, markersize=7, label='Avg Time Per Query (ms)'
    )
    
    # 显式加粗
    ax_right.set_ylabel('Avg Time Per Query (ms)', color=c2, fontweight='bold', fontsize=16)
    
    ax_right.tick_params(axis='y', labelcolor=c2)
    ax_right.grid(False)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax_left.legend(lines, labels, frameon=False, loc='best')

    sns.despine(ax=ax_left, right=True)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"fig2_efficiency_k_{_slugify(ds)}.png")
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {save_path}")

# --- 图 3: Index Size ---
for ds in datasets:
    data_ds = (
        df_plot[df_plot['dataset'] == ds][['k', 'index_size_keys']]
        .drop_duplicates()
        .sort_values('k')
    )
    if data_ds.empty:
        continue

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=data_ds, x='k', y='index_size_keys',
        palette='Blues_d', edgecolor='black'
    )
    
    # 显式加粗
    ax.set_xlabel('k', fontweight='bold', fontsize=16)
    ax.set_ylabel('Index Size (keys)', fontweight='bold', fontsize=16)
    
    sns.despine()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"fig3_index_size_{_slugify(ds)}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {save_path}")