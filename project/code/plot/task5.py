import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# ==========================================
# 1. 配置与数据读取
# ==========================================
base_dir_step5 = r"D:\Data\experiments\2025\12\24\step5\unreviewed"
file_path_step5 = os.path.join(base_dir_step5, "step5.csv")

base_dir_step3 = r"D:\Data\experiments\2025\12\24\step3\unreviewed"
file_path_step3 = os.path.join(base_dir_step3, "step3.csv")

output_dir = r"D:\Data\experiments\2025\12\24\step5"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

BLOCK_ORDER = ['0', '(1, 3)', '(3, 5)']

try:
    print("Reading data...")
    # Step 5
    df5 = pd.read_csv(file_path_step5)
    df5 = df5[df5['mutation_rate'] == 0].copy()
    df5['Method'] = df5['sw_top_n'].apply(lambda x: f"SW Top-{int(x)}")
    df5['Type'] = 'Step 5'

    # Step 3
    df3 = pd.read_csv(file_path_step3)
    df3 = df3[df3['mutation_rate'] == 0].copy()
    df3['Method'] = "Ungapped (Baseline)"
    df3['Type'] = 'Step 3'
    df3['sw_top_n'] = np.nan

    df_all = pd.concat([df3, df5], ignore_index=True)
    
    # 清洗 blocks
    df_all['blocks'] = df_all['blocks'].astype(str).replace({'0.0': '0', 0: '0'})
    
    print("Blocks found:", df_all['blocks'].unique())
    print("Methods found:", df_all['Method'].unique())

except Exception as e:
    print(f"Error reading files: {e}")
    exit()

# ==========================================
# 2. 样式定义：黑色基准 + 蓝色渐变
# ==========================================
sns.set_theme(style="ticks", font_scale=1.2)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.grid'] = False  # <--- 修改：关闭网格线
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.linewidth'] = 1.2

# --- 关键修改：定义颜色板 ---
palette = {
    "Ungapped (Baseline)": "#000000",  # 黑色
    "SW Top-100": "#08519C",           # 浅蓝 (最宽泛)
    "SW Top-50":  "#3182BD",           # 中蓝
    "SW Top-20":  "#6BAED6"            # 深蓝 (最精细/最强)
}

# 定义线型和标记
styles = {
    "Ungapped (Baseline)": (3, 1.5),   # 虚线
    "SW Top-100": "",                  # 实线
    "SW Top-50":  "",                  # 实线
    "SW Top-20":  ""                   # 实线
}

markers = {
    "Ungapped (Baseline)": 'o',        # 圆点
    "SW Top-100": '^',                 # 三角
    "SW Top-50":  's',                 # 方块
    "SW Top-20":  'D'                  # 菱形
}

# 定义绘图顺序 (图例顺序)
hue_order = ["Ungapped (Baseline)", "SW Top-100", "SW Top-50", "SW Top-20"]

# 确保所有出现的 label 都在 palette 中 (防止数据只有部分 Top-N 导致报错)
available_methods = df_all['Method'].unique()
for m in available_methods:
    if m not in palette:
        palette[m] = 'gray'
        styles[m] = ""
        markers[m] = 'x'

# ==========================================
# 3. 绘图函数 (折线图)
# ==========================================
def plot_metric_vs_blocks(y_col, ylabel, title_text, filename):
    plt.figure(figsize=(8, 6))
    
    # 过滤掉 hue_order 中不存在于数据里的标签
    current_order = [m for m in hue_order if m in available_methods]
    
    sns.lineplot(
        data=df_all, 
        x='blocks', 
        y=y_col, 
        hue='Method', 
        style='Method',
        palette=palette,
        markers=markers,
        dashes=styles,
        linewidth=2.5,
        markersize=9,
        hue_order=current_order
    )
    
    # 修正 X 轴显示
    existing_blocks = [b for b in BLOCK_ORDER if b in df_all['blocks'].unique()]
    plt.xticks(ticks=range(len(existing_blocks)), labels=existing_blocks)
    
    # plt.title(title_text, fontsize=14, fontweight='bold', pad=15) # <--- 修改：移除 Title
    plt.xlabel("Indel Block Size (Gap Length)", fontsize=12, fontweight='bold')
    plt.ylabel(ylabel, fontsize=12, fontweight='bold')
    
    # 图例设置
    plt.legend(title=None, fontsize=10, frameon=True, edgecolor='gray')
    
    sns.despine()
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {filename}")

# ==========================================
# 4. 绘图函数 (柱状图)
# ==========================================
def plot_cost_comparison():
    df_cost = df_all.groupby(['Method', 'Type'])[['avg_time_ms', 'static_index_MB']].mean().reset_index()
    
    # 过滤并确定顺序
    current_order = [m for m in hue_order if m in df_cost['Method'].unique()]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- Subplot 1: Time ---
    sns.barplot(
        data=df_cost, x='Method', y='avg_time_ms', 
        hue='Method', palette=palette, ax=axes[0], edgecolor='black', 
        linewidth=1.2, legend=False, order=current_order
    )
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')
        
    # axes[0].set_title("Average Runtime per Query", fontweight='bold') # <--- 修改：移除 Title
    axes[0].set_ylabel("Time (ms)", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', rotation=20)
    
    # --- Subplot 2: Memory ---
    sns.barplot(
        data=df_cost, x='Method', y='static_index_MB', 
        hue='Method', palette=palette, ax=axes[1], edgecolor='black', 
        linewidth=1.2, legend=False, order=current_order
    )
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.0f', padding=3, fontsize=10, fontweight='bold')
        
    # axes[1].set_title("Static Index & Cache Size", fontweight='bold') # <--- 修改：移除 Title
    axes[1].set_ylabel("Size (MB)", fontweight='bold')
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', rotation=20)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "4_cost_comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: 4_cost_comparison.png")

# ==========================================
# 执行
# ==========================================
# title 参数传进去但不使用了
plot_metric_vs_blocks('recall@10', 'Recall@10', 'Recall Robustness', '1_recall_comparison.png')
plot_metric_vs_blocks('precision@10', 'Precision@10', 'Precision Robustness', '2_precision_comparison.png')
plot_metric_vs_blocks('mrr', 'Mean Reciprocal Rank (MRR)', 'Ranking Quality', '3_mrr_comparison.png')
plot_cost_comparison()

print("\nAll plots generated successfully in:", output_dir)