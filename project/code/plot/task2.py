import os
import itertools
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- 配置区域 ---
# 请确认以下路径是否需要微调
OUTPUT_DIR = r"D:\Data\experiments\2025\12\24\step2"
STEP1_PATH = r"D:\Data\experiments\2025\12\24\step1\unreviewed\step1.csv"
STEP2_PATH = r"D:\Data\experiments\2025\12\24\step2\unreviewed\step2.csv"

# 过滤参数
DATASET_TARGET = 'unreviewed'
K_SET: List[int] = [4, 5, 6]
MUT_ALLOWED: List[float] = [0.0, 0.1]

# 绘图样式
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
sns.set_context('paper', font_scale=1.2)


def ensure_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_and_clean_data() -> pd.DataFrame:
    """加载数据，合并 Step1/2，并执行严格过滤"""
    
    def _read(path: str, label: str) -> pd.DataFrame:
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        df['source'] = label
        return df

    # 1. 加载
    df = pd.concat([
        _read(STEP1_PATH, 'Step 1'),
        _read(STEP2_PATH, 'Step 2')
    ], ignore_index=True, sort=False)
    
    if df.empty: return df

    # 2. 过滤 Dataset (忽略大小写)
    if 'dataset' in df.columns:
        df = df[df['dataset'].astype(str).str.lower() == DATASET_TARGET.lower()].copy()

    # 3. 过滤 K
    df['k'] = pd.to_numeric(df['k'], errors='coerce')
    df = df[df['k'].isin(K_SET)]

    # 4. 过滤 Mutation Rate
    if 'mutation_rate' in df.columns:
        df['mutation_rate'] = pd.to_numeric(df['mutation_rate'], errors='coerce')
        mask = df['mutation_rate'].apply(lambda x: any(np.isclose(x, m) for m in MUT_ALLOWED))
        df = df[mask].copy()
        df['mutation_str'] = df['mutation_rate'].apply(lambda x: f"{x:g}")

    # 5. 清洗 Blocks (核心逻辑)
    if 'blocks' in df.columns:
        def classify_block(val):
            s = str(val).replace(" ", "") # 去除空格干扰
            if s in ['0', '0.0']: 
                return 'Block 0'
            # 兼容 (1,3), 1,3, (1, 3) 等多种写法
            if any(x in s for x in ['(1,3)', '1,3']):
                return 'Block (1, 3)'
            return 'Other'

        df['blocks_group'] = df['blocks'].apply(classify_block)
        df = df[df['blocks_group'].isin(['Block 0', 'Block (1, 3)'])]

    return df


def get_padded_stats(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """计算统计量，并强制填充缺失组合 (Padding)，确保绘图对齐"""
    if metric not in df.columns:
        print(f"Skipping {metric}: column missing.")
        return pd.DataFrame()

    group_cols = ['k', 'blocks_group', 'mutation_str', 'source']
    
    # 基础聚合
    stats = df.groupby(group_cols)[metric].agg(['mean', 'std', 'count']).reset_index()
    stats['ci'] = 1.96 * (stats['std'] / np.sqrt(stats['count'])).fillna(0)

    # 强制补全所有组合 (Cartesian Product)
    # 即使 Step 2 缺少某些 K 或 Block，也会生成 NaN 行，保证图表占位正确
    full_idx = pd.MultiIndex.from_product([
        sorted(K_SET),
        ['Block 0', 'Block (1, 3)'],
        [f"{x:g}" for x in MUT_ALLOWED],
        ['Step 1', 'Step 2']
    ], names=group_cols)

    stats = stats.set_index(group_cols).reindex(full_idx).reset_index()
    
    # 设定 Categorical 顺序
    stats['mutation_str'] = pd.Categorical(stats['mutation_str'], categories=[f"{x:g}" for x in MUT_ALLOWED], ordered=True)
    stats['source'] = pd.Categorical(stats['source'], categories=['Step 1', 'Step 2'], ordered=True)
    
    return stats


def plot_combined_grid(df: pd.DataFrame):
    """绘制 4x3 布局对比图"""
    metrics = ['recall@1', 'recall@10']
    block_groups = ['Block 0', 'Block (1, 3)']
    k_values = sorted(K_SET)
    
    n_rows = len(metrics) * len(block_groups) # 4行
    n_cols = len(k_values)                    # 3列
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.2 * n_rows), sharey='row')
    
    # 颜色：Step 1(蓝), Step 2(红)
    palette = {'Step 1': '#1f77b4', 'Step 2': '#d62728'}
    
    row_idx = 0
    for metric in metrics:
        stats_data = get_padded_stats(df, metric)
        if stats_data.empty: 
            row_idx += len(block_groups)
            continue

        for block_group in block_groups:
            for col_idx, k in enumerate(k_values):
                ax = axes[row_idx, col_idx]
                
                # 提取子集
                subset = stats_data[
                    (stats_data['k'] == k) & 
                    (stats_data['blocks_group'] == block_group)
                ].copy()

                # 1. 绘制柱状图 (Hue Order 强制固定)
                sns.barplot(
                    data=subset, x='mutation_str', y='mean', hue='source',
                    hue_order=['Step 1', 'Step 2'],
                    palette=palette, edgecolor='black', ax=ax, zorder=3
                )

                # 2. 绘制误差棒 (基于坐标计算)
                # 两个柱子，Seaborn默认每个宽0.4。中心偏移量：Step 1 (-0.2), Step 2 (+0.2)
                offsets = {'Step 1': -0.2, 'Step 2': 0.2}
                x_labels = [f"{x:g}" for x in MUT_ALLOWED]

                for _, row in subset.iterrows():
                    if pd.isna(row['mean']): continue
                    try:
                        x_base = x_labels.index(str(row['mutation_str']))
                        x_pos = x_base + offsets[row['source']]
                        ax.errorbar(
                            x=x_pos, y=row['mean'], yerr=row['ci'],
                            fmt='none', c='black', capsize=4, elinewidth=1.5, zorder=4
                        )
                    except ValueError: pass

                # 格式化
                ax.set_ylim(0, 1.05)
                ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)
                ax.set_xlabel('')
                ax.set_ylabel('')
                
                # 标签设置
                metric_name = "Recall@1" if metric == 'recall@1' else "Recall@10"
                if row_idx == 0:
                    ax.set_title(f"k = {k}", fontsize=14, fontweight='bold', pad=10)
                if col_idx == 0:
                    ax.set_ylabel(f"{metric_name}\n({block_group})", fontsize=12, fontweight='bold')
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Mutation Rate", fontsize=11, fontweight='bold')
                if ax.get_legend(): ax.get_legend().remove()

            row_idx += 1

    # 全局图例
    handles = [
        Patch(facecolor=palette['Step 1'], edgecolor='black', label='Step 1'),
        Patch(facecolor=palette['Step 2'], edgecolor='black', label='Step 2')
    ]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    out_file = os.path.join(OUTPUT_DIR, "combined_performance_comparison.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Figure saved: {out_file}")


def export_tables(df: pd.DataFrame):
    """生成 Candidates 和 Time 的统计表"""
    out_path = os.path.join(OUTPUT_DIR, "efficiency_summary.xlsx")
    
    with pd.ExcelWriter(out_path, engine='xlsxwriter') as writer:
        for metric in ['avg_candidates', 'avg_time_ms']:
            stats = get_padded_stats(df, metric)
            if stats.empty: continue
            
            # 1. 透视表 (便于查看对比)
            pivot = stats.pivot_table(
                index=['k', 'blocks_group', 'mutation_str'], 
                columns='source', values='mean'
            )
            pivot.to_excel(writer, sheet_name=f"{metric}_pivot")
            
            # 2. 原始统计 (去除空行)
            raw = stats.dropna(subset=['mean'])
            raw.to_excel(writer, sheet_name=f"{metric}_raw", index=False)
            
    print(f"Table saved: {out_path}")


def main():
    ensure_dir(OUTPUT_DIR)
    print("Loading data...")
    df = load_and_clean_data()
    
    if df.empty:
        print("Error: No data found after filtering. Check file paths and content.")
        return

    print("Generating Plots...")
    plot_combined_grid(df)
    
    print("Exporting Tables...")
    export_tables(df)
    print("Done.")

if __name__ == '__main__':
    main()