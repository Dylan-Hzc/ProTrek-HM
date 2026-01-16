import difflib
import os
import random
import textwrap

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ================= 1. 全局配置区域 =================
DATA_PATH = "dataset/protrek_data.tsv"  # 原始数据路径

# 输出文件路径
OUTPUT_HARD_ALL = "dataset/hard_negatives_all.csv"
OUTPUT_HARD_TRAIN = "dataset/hard_negatives_train.csv"
OUTPUT_HARD_TEST = "dataset/hard_negatives_test.csv"
OUTPUT_EASY_FILE = "dataset/easy_samples_all.csv"
OUTPUT_VIZ_IMG = "evaluation_results/hard_negative_examples.png"
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
    if seq1 == seq2:
        return 1.0
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
    if diff_idx == -1:
        diff_idx = min_len

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
            consensus.append("|")
        else:
            consensus.append("*")  # Mutation marker
    consensus = "".join(consensus)

    return sub_a, sub_b, consensus, rel_idx, start


def visualize_single_best_sample(pairs, filename):
    if len(pairs) == 0:
        return
    print(f"[-] 正在生成可视化图表: {filename} ...")

    # 1. 挑选Sequence Similarity 最高，且不是 1.0
    best_sample = max(pairs, key=lambda x: x["seq_similarity"])

    # 准备数据
    seq_a, seq_b = best_sample["anchor_seq"], best_sample["hard_neg_seq"]
    desc_a, desc_b = best_sample["anchor_text"], best_sample["hard_neg_text"]
    sim_score = best_sample["seq_similarity"]

    sub_a, sub_b, _, _, _ = find_mutation_site(seq_a, seq_b, window=25)

    # --- 开始绘图 ---
    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(12, 4))

    # 隐藏网格和轴
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # 标题
    plt.title(
        "Sequence Mismatch Map: High-Similarity Hard Negative",
        fontsize=14,
        pad=30,
        fontweight="bold",
        color="#2c3e50",
    )

    # 文字描述：使用 textwrap 优雅处理
    wrapper = textwrap.TextWrapper(width=100)
    desc_text = f"Anchor: {desc_a}\nNegative: {desc_b}"
    wrapped_lines = wrapper.wrap(desc_text)
    full_desc = "\n".join(wrapped_lines)

    ax.text(
        0.5,
        -0.15,
        full_desc,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#7f8c8d",
        style="italic",
        linespacing=1.5,
    )

    # 序列对齐绘制
    char_spacing = 0.85 / (len(sub_a) + 1)
    start_x = 0.08

    y_anchor = 0.75
    y_neg = 0.35

    # 绘制标签
    ax.text(
        start_x - 0.02,
        y_anchor,
        "Anchor  ",
        ha="right",
        va="center",
        fontweight="bold",
        color="#2c3e50",
        fontsize=11,
    )
    ax.text(
        start_x - 0.02,
        y_neg,
        "Negative",
        ha="right",
        va="center",
        fontweight="bold",
        color="#2c3e50",
        fontsize=11,
    )

    # 逐字符绘制
    mono_font = {"family": "monospace", "size": 13}
    for i, (char_a, char_b) in enumerate(zip(sub_a, sub_b)):
        cur_x = start_x + i * char_spacing
        is_mut = char_a != char_b

        # 突变红色，其余灰色
        char_color = "#E74C3C" if is_mut else "#95A5A6"
        char_weight = "bold" if is_mut else "normal"

        # 绘制字符
        ax.text(
            cur_x,
            y_anchor,
            char_a,
            ha="center",
            va="center",
            color=char_color,
            fontweight=char_weight,
            **mono_font,
        )
        ax.text(
            cur_x,
            y_neg,
            char_b,
            ha="center",
            va="center",
            color=char_color,
            fontweight=char_weight,
            **mono_font,
        )

        if is_mut:
            # 突出显示突变位点
            rect = patches.Rectangle(
                (cur_x - char_spacing / 2, y_neg - 0.1),
                char_spacing,
                0.6,
                linewidth=0,
                facecolor="#E74C3C",
                alpha=0.1,
                zorder=0,
            )
            ax.add_patch(rect)
            ax.text(
                cur_x,
                (y_anchor + y_neg) / 2,
                "!",
                ha="center",
                va="center",
                color="#E74C3C",
                fontweight="bold",
                fontsize=10,
            )

    # 显示相似度
    ax.text(
        0.95,
        0.95,
        f"Sequence Identity: {sim_score:.2%}",
        transform=ax.transAxes,
        ha="right",
        fontsize=10,
        fontweight="bold",
        color="#34495e",
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


# ================= 3. 主逻辑 =================


def run_data_pipeline():
    print(f"[-] [Step 1] 读取数据: {DATA_PATH} ...")
    try:
        df = pd.read_csv(DATA_PATH, sep="\t")
    except FileNotFoundError:
        print(f"错误：找不到文件 {DATA_PATH}")
        return

    # 全局去重
    df = df.drop_duplicates(subset=["protein_sequence", "description"]).reset_index(
        drop=True
    )
    print(f"    有效数据量: {len(df)}")

    # --- A. 挖掘困难样本 ---
    print("[-] [Step 2] 开始挖掘困难样本 (Hard Negatives)...")
    grouped = df.groupby("foldseek_seq")
    hard_negative_pairs = []
    hard_indices_set = set()

    for foldseek_seq, group in tqdm(grouped, desc="Processing Clusters"):
        if len(group) < 2:
            continue

        descriptions = group["description"].tolist()
        seqs = group["protein_sequence"].tolist()
        indices = group.index.tolist()

        try:
            tfidf = TfidfVectorizer().fit_transform(descriptions)
            text_sim_matrix = cosine_similarity(tfidf)
        except ValueError:
            continue

        n = len(group)
        for i in range(n):
            for j in range(i + 1, n):
                if text_sim_matrix[i, j] > TEXT_SIM_THRESHOLD:
                    continue
                if (
                    abs(len(seqs[i]) - len(seqs[j])) / max(len(seqs[i]), len(seqs[j]))
                    > 0.2
                ):
                    continue

                seq_sim = get_seq_similarity(seqs[i], seqs[j])
                if MIN_SEQ_SIMILARITY <= seq_sim <= MAX_SEQ_SIMILARITY:
                    hard_negative_pairs.append(
                        {
                            "anchor_seq": seqs[i],
                            "anchor_text": descriptions[i],
                            "hard_neg_seq": seqs[j],
                            "hard_neg_text": descriptions[j],
                            "seq_similarity": seq_sim,
                            "type": "hard",
                        }
                    )
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
    train_df, test_df = train_test_split(
        df_hard, test_size=SPLIT_TEST_SIZE, random_state=RANDOM_SEED
    )

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
            while rand_idx == idx:
                rand_idx = random.choice(remaining_indices)
            rand_row = df.iloc[rand_idx]

            easy_samples.append(
                {
                    "anchor_seq": row["protein_sequence"],
                    "anchor_text": row["description"],
                    "hard_neg_seq": rand_row["protein_sequence"],
                    "hard_neg_text": rand_row["description"],
                    "seq_similarity": 0.0,
                    "type": "easy",
                }
            )

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
