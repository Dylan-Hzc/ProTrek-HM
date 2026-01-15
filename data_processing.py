import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import difflib
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
import os

# ================= 1. 全局配置区域 =================
DATA_PATH = 'dataset/protrek_data.tsv'  # 原始数据路径

# 输出文件路径
OUTPUT_HARD_ALL = 'dataset/hard_negatives_all.csv'
OUTPUT_HARD_TRAIN = 'dataset/hard_negatives_train.csv'
OUTPUT_HARD_TEST = 'dataset/hard_negatives_test.csv'
OUTPUT_EASY_FILE = 'dataset/easy_samples_all.csv'
OUTPUT_VIZ_IMG = 'dataset/hard_negative_examples.png'
# 挖掘阈值参数
TEXT_SIM_THRESHOLD = 0.5
MIN_SEQ_SIMILARITY = 0.7
MAX_SEQ_SIMILARITY = 0.99999999

# 采样数量
EASY_SAMPLE_LIMIT = 1000
SPLIT_TEST_SIZE = 0.2
RANDOM_SEED = 42

# =================================================

# 设置随机种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def get_seq_similarity(seq1, seq2):
    if seq1 == seq2: return 1.0
    return difflib.SequenceMatcher(None, seq1, seq2).ratio()


# ================= 2. 核心可视化模块 (极致美化版) =================

def find_mutation_site(seq1, seq2, window=20):
    """
    找到突变位点并截取上下文 (窗口稍大一点，展示更多背景)
    """
    min_len = min(len(seq1), len(seq2))
    diff_idx = -1
    for i in range(min_len):
        if seq1[i] != seq2[i]:
            diff_idx = i
            break
    if diff_idx == -1: diff_idx = min_len

    start = max(0, diff_idx - window)
    end = min(max(len(seq1), len(seq2)), diff_idx + window)

    # 截取片段
    sub_a = seq1[start:end]
    sub_b = seq2[start:end]

    # 计算相对突变位置
    rel_idx = diff_idx - start

    # 构造一致性行 (Consensus Line)
    consensus = []
    for c1, c2 in zip(sub_a, sub_b):
        if c1 == c2:
            consensus.append('|')
        else:
            consensus.append('*')  # Mutation marker
    consensus = "".join(consensus)

    return sub_a, sub_b, consensus, rel_idx, start


def visualize_single_best_sample(pairs, filename):
    if len(pairs) == 0: return
    print(f"[-] 正在生成可视化图表: {filename} ...")

    # 1. 挑选Sequence Similarity 最高，且不是 1.0
    best_sample = max(pairs, key=lambda x: x['seq_similarity'])

    # 准备数据
    seq_a, seq_b = best_sample['anchor_seq'], best_sample['hard_neg_seq']
    desc_a, desc_b = best_sample['anchor_text'], best_sample['hard_neg_text']
    sim_score = best_sample['seq_similarity']

    sub_a, sub_b, consensus, rel_idx, start_pos = find_mutation_site(seq_a, seq_b, window=22)

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_axis_off()  # 隐藏坐标轴

    # A. 顶部标题栏
    ax.text(50, 95, "Hard Negative Mining: The 'Twin' Protein Challenge",
            ha='center', va='center', fontsize=18, fontweight='bold', color='#2c3e50')
    ax.text(50, 89, f"Sequence Identity: {sim_score:.2%} | Single Point Mutation Detected",
            ha='center', va='center', fontsize=12, color='#7f8c8d')

    # 分割线
    ax.plot([10, 90], [84, 84], color='#bdc3c7', linewidth=1)

    # B. 文本描述对比区域 (Card Style)
    # 定义文本框样式
    box_style_a = dict(boxstyle='round,pad=0.5', facecolor='#e8f6f3', edgecolor='#1abc9c', alpha=0.8)
    box_style_b = dict(boxstyle='round,pad=0.5', facecolor='#fdedec', edgecolor='#e74c3c', alpha=0.8)

    # 自动换行
    wrapper = textwrap.TextWrapper(width=80)
    wrapped_a = "\n".join(wrapper.wrap(desc_a))
    wrapped_b = "\n".join(wrapper.wrap(desc_b))

    # 绘制 Anchor 描述
    ax.text(10, 75, "Anchor Protein (Positive Query)", fontsize=11, fontweight='bold', color='#16a085')
    ax.text(10, 68, wrapped_a, fontsize=10, color='#2c3e50', bbox=box_style_a, va='top', family='sans-serif')

    # 绘制 HardNeg 描述
    # 动态计算 Y 轴位置，防止文本重叠
    offset_y = len(wrapped_a.split('\n')) * 3
    neg_title_y = 68 - offset_y - 5
    neg_desc_y = neg_title_y - 7

    ax.text(10, neg_title_y, "Hard Negative Protein (Functionally Distinct)", fontsize=11, fontweight='bold',
            color='#c0392b')
    ax.text(10, neg_desc_y, wrapped_b, fontsize=10, color='#2c3e50', bbox=box_style_b, va='top', family='sans-serif')

    # C. 序列比对区域 (Alignment View)
    align_y_center = 20

    # 背景装饰条
    rect = patches.Rectangle((5, 5), 90, 30, linewidth=0, facecolor='#f4f6f7', zorder=0, alpha=0.5)
    ax.add_patch(rect)
    ax.text(10, 30, "Local Sequence Alignment (Context Window)", fontsize=10, fontweight='bold', color='#34495e')

    # 字体配置
    mono_font = {'family': 'monospace', 'size': 14}
    char_spacing = 1.8  # 字符间距因子
    start_x = 12

    # 绘制：位置标尺
    # ax.text(start_x, align_y_center + 8, f"Pos {start_pos}...", **mono_font, color='gray', fontsize=10)

    # 绘制高亮光柱 (Highlight Column)
    # 计算突变点在图上的 x 坐标
    mut_x = start_x + rel_idx * char_spacing
    # 画一个贯穿上下的红色高亮条
    highlighter = patches.Rectangle((mut_x - 0.6, align_y_center - 6), 1.2, 14,
                                    facecolor='#fadbd8', edgecolor='none', zorder=1)
    ax.add_patch(highlighter)

    # 逐字符绘制
    for i, (char_a, char_b, mark) in enumerate(zip(sub_a, sub_b, consensus)):
        cur_x = start_x + i * char_spacing

        is_mut = (mark == '*')

        # 1. Anchor Seq
        color_a = 'black' if not is_mut else '#c0392b'
        weight = 'normal' if not is_mut else 'bold'
        ax.text(cur_x, align_y_center + 3, char_a, ha='center', color=color_a, fontweight=weight, **mono_font, zorder=2)

        # 2. Consensus Symbols
        mark_color = '#bdc3c7' if not is_mut else '#e74c3c'
        mark_char = '|' if not is_mut else '≠'
        ax.text(cur_x, align_y_center, mark_char, ha='center', color=mark_color, fontweight='bold', fontsize=12,
                family='monospace', zorder=2)

        # 3. HardNeg Seq
        color_b = 'black' if not is_mut else '#c0392b'
        ax.text(cur_x, align_y_center - 3, char_b, ha='center', color=color_b, fontweight=weight, **mono_font, zorder=2)

    # 标注序列名称
    ax.text(start_x - 2, align_y_center + 3, "Seq A:", ha='right', fontsize=12, fontweight='bold', color='#16a085')
    ax.text(start_x - 2, align_y_center - 3, "Seq B:", ha='right', fontsize=12, fontweight='bold', color='#c0392b')

    # 标注 Mutation
    ax.annotate('Mutation Site', xy=(mut_x, align_y_center - 6), xytext=(mut_x, align_y_center - 12),
                arrowprops=dict(facecolor='#e74c3c', shrink=0.05, width=1.5, headwidth=8),
                ha='center', fontsize=10, color='#c0392b', fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# ================= 3. 主逻辑 =================

def run_data_pipeline():
    print(f"[-] [Step 1] 读取数据: {DATA_PATH} ...")
    try:
        df = pd.read_csv(DATA_PATH, sep='\t')
    except FileNotFoundError:
        print(f"错误：找不到文件 {DATA_PATH}")
        return

    # 全局去重
    df = df.drop_duplicates(subset=['protein_sequence', 'description']).reset_index(drop=True)
    print(f"    有效数据量: {len(df)}")

    # --- A. 挖掘困难样本 ---
    print("[-] [Step 2] 开始挖掘困难样本 (Hard Negatives)...")
    grouped = df.groupby('foldseek_seq')
    hard_negative_pairs = []
    hard_indices_set = set()

    for foldseek_seq, group in tqdm(grouped, desc="Processing Clusters"):
        if len(group) < 2: continue

        descriptions = group['description'].tolist()
        seqs = group['protein_sequence'].tolist()
        indices = group.index.tolist()

        try:
            tfidf = TfidfVectorizer().fit_transform(descriptions)
            text_sim_matrix = cosine_similarity(tfidf)
        except ValueError:
            continue

        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                if text_sim_matrix[i, j] > TEXT_SIM_THRESHOLD: continue
                if abs(len(seqs[i]) - len(seqs[j])) / max(len(seqs[i]), len(seqs[j])) > 0.2: continue

                seq_sim = get_seq_similarity(seqs[i], seqs[j])
                if MIN_SEQ_SIMILARITY <= seq_sim <= MAX_SEQ_SIMILARITY:
                    hard_negative_pairs.append({
                        'anchor_seq': seqs[i],
                        'anchor_text': descriptions[i],
                        'hard_neg_seq': seqs[j],
                        'hard_neg_text': descriptions[j],
                        'seq_similarity': seq_sim,
                        'type': 'hard'
                    })
                    hard_indices_set.add(indices[i])
                    hard_indices_set.add(indices[j])

    print(f"    [√] 挖掘完成！共找到 {len(hard_negative_pairs)} 对困难样本。")

    if len(hard_negative_pairs) == 0:
        print("[!] 未挖掘到数据，流程终止。")
        return

    # --- B. 生成可视化报告 ---
    print("[-] [Step 3] 生成美观可视化对比图...")
    visualize_single_best_sample(hard_negative_pairs, OUTPUT_VIZ_IMG)

    # --- C. 数据切分 ---
    print("[-] [Step 4] 执行数据集切分...")
    df_hard = pd.DataFrame(hard_negative_pairs)
    train_df, test_df = train_test_split(df_hard, test_size=SPLIT_TEST_SIZE, random_state=RANDOM_SEED)

    df_hard.to_csv(OUTPUT_HARD_ALL, index=False)
    train_df.to_csv(OUTPUT_HARD_TRAIN, index=False)
    test_df.to_csv(OUTPUT_HARD_TEST, index=False)

    print(f"    Train: {len(train_df)} | Test: {len(test_df)}")

    # --- D. 简单样本 ---
    print("[-] [Step 5] 收集简单样本...")
    all_indices = set(df.index.tolist())
    remaining_indices = list(all_indices - hard_indices_set)
    easy_samples = []

    if len(remaining_indices) > 0:
        sample_size = min(len(remaining_indices), EASY_SAMPLE_LIMIT)
        sampled_indices = random.sample(remaining_indices, sample_size)

        for idx in tqdm(sampled_indices, desc="Collecting Easy Samples"):
            row = df.iloc[idx]
            rand_idx = random.choice(remaining_indices)
            while rand_idx == idx: rand_idx = random.choice(remaining_indices)
            rand_row = df.iloc[rand_idx]

            easy_samples.append({
                'anchor_seq': row['protein_sequence'],
                'anchor_text': row['description'],
                'hard_neg_seq': rand_row['protein_sequence'],
                'hard_neg_text': rand_row['description'],
                'seq_similarity': 0.0,
                'type': 'easy'
            })

    if len(easy_samples) > 0:
        pd.DataFrame(easy_samples).to_csv(OUTPUT_EASY_FILE, index=False)

    print("\n" + "=" * 50)
    print("流水线执行完毕！")
    print(f"1. 训练数据: {OUTPUT_HARD_TRAIN}")
    print(f"2. 评估数据: {OUTPUT_HARD_TEST} & {OUTPUT_EASY_FILE}")
    print(f"3. 报告插图: {OUTPUT_VIZ_IMG} (已极致美化)")
    print("=" * 50)


if __name__ == "__main__":
    run_data_pipeline()