import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# ==========================================
# 1. 配置与数据读取
# ==========================================
base_dir = r"D:\Data\experiments\2025\12\24\step4"
file_path = os.path.join(base_dir, "compare.xlsx")

if not os.path.exists(base_dir):
    os.makedirs(base_dir)

def get_group_label(row):
    val_str = str(row['(dropoffv,top_n_candidate,cap_hit)']).lower()
    if 'baseline' in val_str:
        return 'Baseline'
    else:
        try:
            clean_str = val_str.replace('(', '').replace(')', '').replace(' ', '')
            d, n, c = clean_str.split(',')
            return f"D={d}, N={n}, C={c}"
        except:
            return val_str

try:
    print(f"Reading data from: {file_path}")
    df = pd.read_excel(file_path)
    
    # 数据预处理
    df['Group'] = df.apply(get_group_label, axis=1)
    df['k_label'] = df['k'].apply(lambda x: f"k={x}")
    df['mutation_pct'] = df['mutation_rate'] * 100 
    
    # 关键组标签
    lbl_fast = "D=30, N=1000, C=1000"
    lbl_acc  = "D=50, N=5000, C=2000"
    lbl_base = "Baseline"
    
except Exception as e:
    print(f"Error: {e}")
    exit()

# ==========================================
# 2. 样式定义
# ==========================================
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 定义颜色与样式
styles = {
    lbl_base: {'color': 'black',   'lw': 2.0, 'ls': '--', 'marker': 'o'},
    lbl_fast: {'color': '#e74c3c', 'lw': 2.0, 'ls': '-',  'marker': '^'}, # Red
    lbl_acc:  {'color': '#2980b9', 'lw': 2.0, 'ls': '-',  'marker': 's'}, # Blue
    'other':  {'color': '#cccccc', 'lw': 1.0, 'ls': '-',  'marker': None} # Grey
}

def create_legend_handles():
    handles = []
    labels = []
    
    # 1. Baseline
    h1 = mlines.Line2D([], [], color='black', ls='--', lw=2, marker='o', markersize=6, label='Baseline')
    handles.append(h1); labels.append('Baseline')
    
    # 2. Fastest
    h2 = mlines.Line2D([], [], color='#e74c3c', ls='-', lw=2, marker='^', markersize=6, label='Numba (Fastest)')
    handles.append(h2); labels.append('Numba (Fastest)')
    
    # 3. Best Recall
    h3 = mlines.Line2D([], [], color='#2980b9', ls='-', lw=2, marker='s', markersize=6, label='Numba (Best Recall)')
    handles.append(h3); labels.append('Numba (Best Recall)')
    
    # 4. Others
    h4 = mlines.Line2D([], [], color='#cccccc', ls='-', lw=1, label='Numba (Others)')
    handles.append(h4); labels.append('Numba (Others)')
    
    # 5. Band Description (Optional, visual only)
    h5 = mpatches.Patch(color='gray', alpha=0.2, label='Length Variation (SD)')
    handles.append(h5); labels.append('Length Variance')

    return handles, labels

# ==========================================
# 3. 绘制带波动范围的图表
# ==========================================
def plot_stress_test_with_bands():
    k_values = sorted(df['k'].unique())
    fig, axes = plt.subplots(3, len(k_values), figsize=(4.5 * len(k_values), 10), sharex=True)
    
    if len(k_values) == 1: axes = np.array([axes]).T

    # 配置：行指标、Y轴标题、是否画波动带(bands)、是否对数轴
    rows_config = [
        {'y_col': 'recall@10',   'ylabel': 'Recall@10',       'bands': True,  'log': False, 'ylim': (0, 1.05)},
        {'y_col': 'precision@5', 'ylabel': 'Precision@5',     'bands': True,  'log': False, 'ylim': None},
        {'y_col': 'avg_time_ms', 'ylabel': 'Time (ms) [Log]', 'bands': False, 'log': True,  'ylim': None} 
    ]

    for row_idx, config in enumerate(rows_config):
        y_col = config['y_col']
        use_bands = config['bands']
        
        # 决定 errorbar 参数：
        # 'sd' = 标准差 (Standard Deviation)，展示数据的离散程度
        # None = 只画均值
        err_style = 'sd' if use_bands else None
        
        for col_idx, k in enumerate(k_values):
            ax = axes[row_idx, col_idx]
            subset = df[df['k'] == k]
            
            # --- 1. 画背景线 (Others) ---
            # 背景线我们不画波动带，只画均值，避免混乱
            others = subset[~subset['Group'].isin([lbl_base, lbl_fast, lbl_acc])]
            # 使用 seaborn lineplot 自动聚合 length 维度求均值
            if not others.empty:
                sns.lineplot(
                    data=others, x='mutation_pct', y=y_col, hue='Group',
                    palette={g: styles['other']['color'] for g in others['Group'].unique()},
                    legend=False, ax=ax, errorbar=None, lw=1, alpha=0.5, zorder=1
                )

            # --- 2. 画重点线 (Focus) ---
            # 分别画三条线，以便控制颜色和图层
            for g in [lbl_acc, lbl_fast, lbl_base]: # 绘制顺序：蓝->红->黑(最上层)
                group_data = subset[subset['Group'] == g]
                if group_data.empty: continue
                
                s = styles[g]
                
                # 使用 Seaborn 绘制，自动处理 length 的聚合
                # errorbar='sd' 会自动绘制阴影带
                sns.lineplot(
                    data=group_data, x='mutation_pct', y=y_col,
                    color=s['color'], linestyle=s['ls'], marker=s['marker'], markersize=7,
                    errorbar=err_style, # 这里控制是否显示波动带
                    ax=ax, label=g, zorder=10, linewidth=2.5,
                    err_kws={'alpha': 0.2} # 阴影透明度
                )
                
            # 清除 sns 自动生成的图例 (我们最后统一加)
            if ax.get_legend(): ax.get_legend().remove()

            # --- 3. 轴修饰 ---
            if row_idx == 0:
                ax.set_title(f"k = {k}", fontsize=14, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(config['ylabel'], fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel("")
            
            if row_idx == 2:
                ax.set_xlabel("Mutation Rate (%)", fontsize=12, fontweight='bold')
            
            if config['log']:
                ax.set_yscale('log')
            if config['ylim']:
                ax.set_ylim(config['ylim'])
                
            ax.grid(True, alpha=0.3, which='both', ls='--')

    # 添加统一图例
    handles, labels = create_legend_handles()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=5, frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10) # 底部留白给图例
    
    save_path = os.path.join(base_dir, "combined_stress_test_with_bands.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")

# 执行
plot_stress_test_with_bands()