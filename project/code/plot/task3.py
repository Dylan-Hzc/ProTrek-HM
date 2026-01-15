import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. 配置路径
# 输入文件路径
path_step2 = r"D:\Data\experiments\2025\12\24\step2\unreviewed\step2.csv"
path_step3 = r"D:\Data\experiments\2025\12\24\step3\unreviewed\step3.csv"

# 输出目录
output_dir = r"D:\Data\experiments\2025\12\24\step3"

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 加载数据
df_step2 = pd.read_csv(path_step2, dtype={'blocks': str})
df_step3 = pd.read_csv(path_step3, dtype={'blocks': str})

# 过滤数据
df_step2 = df_step2[df_step2['dataset'] == 'unreviewed'].copy()
df_step3 = df_step3[df_step3['dataset'] == 'unreviewed'].copy()

# 3. 准备 MRR 对比数据
df_step2['step'] = 'Step 2'
df_step3['step'] = 'Step 3'
cols_mrr = ['k', 'mutation_rate', 'blocks', 'length', 'mrr', 'step']
df_mrr = pd.concat([df_step2[cols_mrr], df_step3[cols_mrr]])

# 4. 绘图 1: MRR 对比 (Step 2 vs Step 3)
sns.set_theme(style="whitegrid")
g1 = sns.relplot(
    data=df_mrr,
    x="length", y="mrr",
    hue="step", style="blocks",
    col="mutation_rate", row="k",
    kind="line", markers=True,
    height=3, aspect=1.5,
    facet_kws={'margin_titles': True}
)
g1.fig.suptitle('', y=1.02)

# --- 修改开始：加粗 X 轴和 Y 轴标签 ---
for ax in g1.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
# --- 修改结束 ---

# 保存图片 1
save_path_1 = os.path.join(output_dir, 'mrr_comparison.png')
plt.savefig(save_path_1, bbox_inches='tight', dpi=300)
print(f"Saved: {save_path_1}")
plt.close() # 关闭以释放内存

# 5. 准备 Step 3 时间分析数据
df_time = df_step3.melt(
    id_vars=['k', 'mutation_rate', 'blocks', 'length'],
    value_vars=['avg_time_ms', 'avg_extend_time_ms'],
    var_name='Time Metric', value_name='Time (ms)'
)

# 6. 绘图 2: Step 3 时间分析
g2 = sns.relplot(
    data=df_time,
    x="length", y="Time (ms)",
    hue="Time Metric", style="blocks",
    col="mutation_rate", row="k",
    kind="line", markers=True,
    height=3, aspect=1.5,
    facet_kws={'margin_titles': True}
)
g2.fig.suptitle('', y=1.02)

# --- 修改开始：加粗 X 轴和 Y 轴标签 ---
for ax in g2.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
    ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
# --- 修改结束 ---

# 保存图片 2
save_path_2 = os.path.join(output_dir, 'step3_time_analysis.png')
plt.savefig(save_path_2, bbox_inches='tight', dpi=300)
print(f"Saved: {save_path_2}")
plt.close()