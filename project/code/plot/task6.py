import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 配置与数据读取 (保持不变)
# ==========================================
base_dir_step6 = r"D:\Data\experiments\2025\12\24\step6\unreviewed"
file_path_step6 = os.path.join(base_dir_step6, "step6.csv")

base_dir_step5 = r"D:\Data\experiments\2025\12\24\step5\unreviewed"
file_path_step5 = os.path.join(base_dir_step5, "step5.csv")

output_dir = r"D:\Data\experiments\2025\12\24\step6"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    print("Reading and processing data...")
    # Step 6
    df6 = pd.read_csv(file_path_step6, sep=None, engine='python')
    df6.columns = df6.columns.str.strip()
    if 'seed_pattern' not in df6.columns:
        if 'k' in df6.columns:
            df6['seed_pattern'] = df6['k'].apply(lambda x: f"w={int(x)}")
        else:
            exit()
    df6['blocks'] = df6['blocks'].astype(str).replace({'0.0': '0', '0': '0'})
    df6 = df6[df6['blocks'] == '0'].copy()
    df6['seed_pattern'] = df6['seed_pattern'].astype(str)
    df6['Method'] = df6['seed_pattern'].apply(lambda x: f"Spaced (w={x.split('=')[-1] if '=' in x else x})")
    df6['Type'] = 'Step 6'
    df6['Category'] = 'Spaced' # 用于配色分组

    # Step 5
    df5 = pd.read_csv(file_path_step5, sep=None, engine='python')
    df5.columns = df5.columns.str.strip()
    df5['blocks'] = df5['blocks'].astype(str).replace({'0.0': '0', '0': '0'})
    df5 = df5[df5['blocks'] == '0'].copy()
    df5['Method'] = df5['k'].apply(lambda x: f"Contiguous (k={int(x)})")
    df5['Type'] = 'Step 5'
    df5['Category'] = 'Contiguous'
    df5['seed_pattern'] = df5['k'].apply(lambda x: '1' * int(x) if pd.notnull(x) else '')

    # Merge
    df_all = pd.concat([df5, df6], ignore_index=True)
    df_high_mut = df_all[np.isclose(df_all['mutation_rate'], 0.3)].copy()

except Exception as e:
    print(f"Error: {e}")
    exit()

# ==========================================
# 2. 全局绘图样式 (极简风格)
# ==========================================
sns.set_theme(style="white", font_scale=1.3) # 使用纯白背景，字体加大
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# --- 高级配色方案 ---
# Contiguous (基准): 灰黑色系
# Spaced (实验): 鲜艳对比色 (Teal, Gold, Coral)
unique_methods = sorted(df_all['Method'].unique())
# 短标签，缓解 X 轴拥挤
def _short_name(m: str) -> str:
    if m.startswith('Contiguous'):
        # Contiguous (k=5) -> Contig k=5
        inside = m[m.find('(')+1:m.find(')')] if '(' in m and ')' in m else ''
        return f"Contig {inside.replace('k=','k=')}".strip()
    if m.startswith('Spaced'):
        # Spaced (w=5) -> Spaced w=5
        inside = m[m.find('(')+1:m.find(')')] if '(' in m and ')' in m else ''
        return f"Spaced {inside.replace('w=','w=')}".strip()
    return m
df_all['MethodShort'] = df_all['Method'].map(_short_name)
palette = {}
markers = {}
dashes = {}

colors_spaced = ['#009E73', '#E69F00', '#CC79A7', '#56B4E9'] # 科研常用色盲友好色
color_idx = 0

for m in unique_methods:
    if "Contiguous" in m:
        palette[m] = '#333333'  # 深灰
        markers[m] = 'o'
        dashes[m] = (3, 2)      # 虚线
    else:
        palette[m] = colors_spaced[color_idx % len(colors_spaced)]
        color_idx += 1
        markers[m] = 'D'        # 菱形
        dashes[m] = ""          # 实线

# ==========================================
# 图 1: Recall Trend (去噪版)
# ==========================================
def plot_recall_clean():
    if df_high_mut.empty: return
    
    plt.figure(figsize=(9.5, 6.2))
    ax = plt.gca()
    
    # 1) 先画线（无阴影、无 marker），保证清晰
    sns.lineplot(
        data=df_high_mut,
        x='sw_top_n', y='recall@10', hue='Method', style='Method',
        palette=palette, dashes=dashes,
        linewidth=2.6, marker=None, errorbar=None, ax=ax
    )
    
    # 2) 再叠加散点，用点大小表达 Top-N，避免文字重叠
    sc = sns.scatterplot(
        data=df_high_mut,
        x='sw_top_n', y='recall@10', hue='Method', style='Method',
        palette=palette, markers=markers,
        size='sw_top_n', sizes=(40, 160), alpha=0.9,
        edgecolor='white', linewidth=0.8, legend=False, ax=ax
    )
    
    plt.title('Recall Robustness (Mutation=0.3)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('SW Rerank Limit (Top-N)', fontsize=14, fontweight='bold')
    plt.ylabel('Recall@10', fontsize=14, fontweight='bold')
    
    # 网格线淡化
    plt.grid(True, linestyle='--', alpha=0.4)
    
    # 图例：仅显示方法图例；额外添加 Top-N 尺寸注释
    handles, labels = ax.get_legend_handles_labels()
    # 去重并仅保留 Method 项（Seaborn 可能混入重复）
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l in seen or l in ['sw_top_n']:
            continue
        seen.add(l)
        uniq.append((h, l))
    if uniq:
        ax.legend([h for h, _ in uniq], [l for _, l in uniq],
                  bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title='Method')
    ax.text(1.02, 0.0, 'Marker size ~ Top-N', transform=ax.transAxes, fontsize=11)
    
    # X轴刻度整数化
    ticks = sorted(df_high_mut['sw_top_n'].unique())
    plt.xticks(ticks)
    ax.set_xlim(min(ticks), max(ticks))
    ax.set_ylim(0.0, 1.0)
    
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_spaced_recall_high_mutation_clean.png"), dpi=300)
    print("Saved clean Recall plot.")

# ==========================================
# 图 2: Trade-off (气泡图版)
# ==========================================
def plot_tradeoff_clean():
    plt.figure(figsize=(10, 7))
    data = df_high_mut if not df_high_mut.empty else df_all
    
    # 关键修改：用 size 代表 sw_top_n，不再写满屏幕的文字
    ax = sns.scatterplot(
        data=data,
        x='avg_time_ms', y='recall@10', 
        hue='Method', style='Method', size='sw_top_n',
        palette=palette, markers=markers,
        sizes=(80, 360), # 气泡大小范围，稍微收敛
        alpha=0.85, edgecolor='white', linewidth=1.5
    )
    
    plt.xscale('log')
    plt.title('Accuracy vs. Speed Trade-off (Mutation=0.3)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Avg Time per Query (ms) [Log]', fontsize=14, fontweight='bold')
    plt.ylabel('Recall@10', fontsize=14, fontweight='bold')
    
    # 网格
    plt.grid(True, which="major", linestyle='-', alpha=0.3)
    plt.grid(True, which="minor", linestyle=':', alpha=0.2)
    
    # 图例：仅方法放外侧；尺寸说明作为第二行文字
    handles, labels = ax.get_legend_handles_labels()
    uniq = []
    seen = set()
    for h, l in zip(handles, labels):
        if l in seen or l in ['sw_top_n']:
            continue
        seen.add(l)
        uniq.append((h, l))
    if uniq:
        ax.legend([h for h, _ in uniq], [l for _, l in uniq],
                  bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title='Method')
    ax.text(1.02, 0.0, 'Marker size ~ Top-N', transform=ax.transAxes, fontsize=11)
    
    # 标注最佳点 (Pareto Frontier)，只标注最右上角的点
    best_point = data.loc[data['recall@10'].idxmax()]
    plt.text(
        best_point['avg_time_ms'], best_point['recall@10'] + 0.005, 
        f"Best: {best_point['Method'].split('(')[0]}", 
        fontsize=10, fontweight='bold', color=palette[best_point['Method']],
        ha='center', va='bottom'
    )

    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_tradeoff_clean.png"), dpi=300)
    print("Saved clean Tradeoff plot.")

# ==========================================
# 图 3: Overhead (双轴优化版)
# ==========================================
def plot_overhead_clean():
    # 使用短标签并排序：先 Contiguous 再 Spaced，提升可读性
    df_all_sorted = df_all.copy()
    df_all_sorted['MethodShort'] = df_all_sorted['Method'].map(_short_name)
    df_all_sorted['CategoryOrder'] = np.where(df_all_sorted['Category'] == 'Contiguous', 0, 1)
    df_avg = df_all_sorted.groupby(['CategoryOrder', 'Method', 'MethodShort'])[['avg_candidates', 'avg_time_ms']].mean().reset_index().sort_values(['CategoryOrder', 'MethodShort'])
    
    # 为了配色好看，我们需要手动给 Bar 分配颜色
    # Contiguous 用灰色，Spaced 用彩色
    bar_colors = [palette[m] for m in df_avg['Method']]
    
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    # 1. 柱状图 (Candidates)
    # 使用 alpha 让柱子稍微淡一点，突出折线
    # 调整柱颜色：Contiguous 统一浅灰，Spaced 使用彩色
    bar_palette = {m: ('#9E9E9E' if 'Contiguous' in m else palette[m]) for m in df_avg['Method']}
    bars = sns.barplot(
        data=df_avg, x='MethodShort', y='avg_candidates', 
        palette=bar_palette, ax=ax1, alpha=0.65, edgecolor='black', linewidth=1.0
    )
    
    ax1.set_ylabel('Avg Candidates (Bars)', fontsize=14, fontweight='bold', color='#555555')
    ax1.tick_params(axis='y', colors='#555555')
    ax1.set_xlabel('')
    
    # 减少杂乱：去除柱上数值，避免标签拥挤

    # 2. 折线图 (Time) - 使用极其醒目的颜色 (如深红) 代表时间代价
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df_avg, x='MethodShort', y='avg_time_ms', 
        color='#D55E00', marker='o', markersize=12, linewidth=4, 
        ax=ax2, label='Total Runtime', sort=False
    )
    
    ax2.set_ylabel('Total Runtime (ms)', fontsize=14, fontweight='bold', color='#D55E00')
    ax2.tick_params(axis='y', colors='#D55E00')
    ax2.spines['right'].set_color('#D55E00')
    ax2.spines['right'].set_linewidth(2)
    
    plt.title('Overhead Analysis: Search Space vs. Actual Time', fontsize=16, fontweight='bold', pad=20)
    
    # X轴标签优化：短标签 + 旋转 + 紧凑间距
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=15, ha='right', fontsize=11)
    
    # 手动添加 Legend
    # 创建一个 dummy line 用于图例显示 Runtime
    import matplotlib.lines as mlines
    line_patch = mlines.Line2D([], [], color='#D55E00', marker='o', markersize=10, linewidth=3, label='Runtime (ms)')
    bar_patch = mlines.Line2D([], [], color='#9E9E9E', marker='s', linestyle='None', markersize=10, alpha=0.8, label='Candidates')
    
    plt.legend(handles=[bar_patch, line_patch], loc='upper left', frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_overhead_clean.png"), dpi=300)
    print("Saved clean Overhead plot.")

# ==========================================
# 执行
# ==========================================
plot_recall_clean()
plot_tradeoff_clean()
plot_overhead_clean()

print("\nAll plots optimized for aesthetics.")
