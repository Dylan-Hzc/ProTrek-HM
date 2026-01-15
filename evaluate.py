import gc
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", context="talk", font_scale=1.0)

# 蓝色(正), 红色(负), 灰色(背景)
COLORS = {
    "pos": "#2E86C1",  # 强蓝色
    "neg": "#C0392B",  # 深红色
    "bg": "#D5D8DC",  # 浅灰色
    "groups": ["#1ABC9C", "#9B59B6", "#F39C12"],  # Top3 组的专属色 (绿, 紫, 橙)
}

# ================= ⚙️ 路径与配置 =================
HARD_TEST_FILE = "dataset/hard_negatives_test.csv"
EASY_TEST_FILE = "dataset/easy_samples_all.csv"
SAVE_DIR = "evaluation_results"  # 新目录，区分之前的

BASELINE_CHECKPOINT = "weights/ProTrek_35M/ProTrek_35M.pt"
FINETUNED_CHECKPOINT = "weights/ProTrek_35M/protrek_finetuned_epoch30.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CONFIG = {
    "protein_config": "weights/ProTrek_35M/esm2_t12_35M_UR50D",
    "text_config": "weights/ProTrek_35M/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "weights/ProTrek_35M/foldseek_t12_35M",
}

sys.path.append("./ProTrek")
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# ================= 🛠️ 核心工具函数 =================


def load_model(checkpoint_path):
    config = BASE_CONFIG.copy()
    config["from_checkpoint"] = checkpoint_path
    logger.info(f"Loading: {checkpoint_path}")
    return ProTrekTrimodalModel(**config).eval().to(DEVICE)


def get_scores_and_embeddings(model, df):
    """
    计算所有样本的分数，并返回 Embeddings 以便后续筛选
    """
    pos_scores = []
    neg_scores = []
    embeddings = {"anchor": [], "text": [], "hard_neg": []}

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
            a_emb = model.get_protein_repr([row["anchor_seq"]])
            t_emb = model.get_text_repr([row["anchor_text"]])
            n_emb = model.get_protein_repr([row["hard_neg_seq"]])

            # 转换为 Numpy
            a_np = a_emb.cpu().numpy()
            t_np = t_emb.cpu().numpy()
            n_np = n_emb.cpu().numpy()

            # 计算分数
            s_pos = np.sum(a_np * t_np)
            s_neg = np.sum(n_np * t_np)

            pos_scores.append(s_pos)
            neg_scores.append(s_neg)

            embeddings["anchor"].append(a_np)
            embeddings["text"].append(t_np)
            embeddings["hard_neg"].append(n_np)

    return np.array(pos_scores), np.array(neg_scores), embeddings


# ================= 📊 绘图函数 =================


def plot_bar_chart_pro(acc_data):
    """
    图 1: 准确率对比 (极简风)
    """
    logger.info("Plotting Pro Bar Chart...")
    plt.figure(figsize=(7, 6))

    df = pd.DataFrame(acc_data)

    # 绘制柱状图
    ax = sns.barplot(
        data=df,
        x="Dataset",
        y="Accuracy",
        hue="Model",
        palette=[COLORS["bg"], COLORS["pos"]],
        edgecolor=".2",
    )

    sns.despine(top=True, right=True)
    plt.ylim(0, 110)
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.xlabel("")
    plt.title("Model Performance Improvement", fontweight="bold", pad=20)
    plt.legend(frameon=False, loc="upper left", markerscale=0.5, fontsize="x-small")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # 标注数值
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "1_accuracy_comparison.png"), dpi=300)
    plt.close()


def plot_kde_pro(base_scores, ft_scores):
    """
    图 2: 分数分布
    """
    logger.info("Plotting Pro KDE...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Helper to plot one subplot
    def draw_kde(ax, pos, neg, title):
        sns.kdeplot(
            pos,
            ax=ax,
            fill=True,
            color=COLORS["pos"],
            alpha=0.3,
            linewidth=2,
            label="Positive Pair",
        )
        sns.kdeplot(
            neg,
            ax=ax,
            fill=True,
            color=COLORS["neg"],
            alpha=0.3,
            linewidth=2,
            label="Negative Pair",
        )
        ax.set_title(title, fontweight="bold", fontsize=14)
        ax.set_xlabel("Similarity Score")
        ax.grid(linestyle=":", alpha=0.6)
        sns.despine(ax=ax)

    draw_kde(
        axes[0], base_scores["pos"], base_scores["neg"], "Baseline Model (Confused)"
    )
    draw_kde(axes[1], ft_scores["pos"], ft_scores["neg"], "Finetuned Model (Separated)")

    axes[0].legend(frameon=False, loc="upper left")

    plt.suptitle(
        "Score Distribution Shift on Hard Negatives", fontweight="bold", y=0.97
    )
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "2_score_distribution.png"), dpi=300)
    plt.close()


def plot_focused_tsne(base_emb, ft_emb, top_indices):
    logger.info("Plotting Focused t-SNE...")

    # 准备背景数据
    n_bg = 50
    bg_indices = np.random.choice(len(base_emb["text"]), n_bg, replace=False)

    # 准备 Top 3 数据
    target_indices = top_indices

    # 合并索引
    all_indices = list(set(list(bg_indices) + list(target_indices)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    for i, (emb_dict, title) in enumerate(
        [(base_emb, "Baseline Space"), (ft_emb, "Finetuned Space")]
    ):
        ax = axes[i]

        # 1. 收集这一轮要画的所有向量
        vectors = []
        labels = []  # 用来标记属于哪个组

        # 结构：[Text_1...Text_N, Anchor_1...Anchor_N, Neg_1...Neg_N]
        for idx in all_indices:
            vectors.append(emb_dict["text"][idx].flatten())
            vectors.append(emb_dict["anchor"][idx].flatten())
            vectors.append(emb_dict["hard_neg"][idx].flatten())

        vectors = np.array(vectors)

        # 2. 降维 (PCA 初始化 + t-SNE 微调，保证结构稳定)
        reducer = TSNE(
            n_components=2, perplexity=10, random_state=42, init="pca", learning_rate=50
        )
        embedded = reducer.fit_transform(vectors)

        # 3. 绘图
        n_samples = len(all_indices)

        # A. 先画背景 (灰色，透明)
        # 坐标映射: index j 对应 embedded 中的 [3*j, 3*j+1, 3*j+2]
        for j, list_idx in enumerate(all_indices):
            if list_idx in target_indices:
                continue  # 跳过 Top 3，最后画

            t_xy = embedded[3 * j]
            a_xy = embedded[3 * j + 1]
            n_xy = embedded[3 * j + 2]

            # 画点
            ax.scatter(t_xy[0], t_xy[1], c=COLORS["bg"], s=30, alpha=0.3)
            ax.scatter(a_xy[0], a_xy[1], c=COLORS["bg"], s=30, alpha=0.3)
            ax.scatter(n_xy[0], n_xy[1], c=COLORS["bg"], s=30, alpha=0.3)

        # B. 再画 Top 3 (高亮，带连线)
        legend_elements = []  # 手动图例

        for k, target_idx in enumerate(target_indices):
            # 找到 target_idx 在 all_indices 中的位置
            j = all_indices.index(target_idx)

            t_xy = embedded[3 * j]
            a_xy = embedded[3 * j + 1]
            n_xy = embedded[3 * j + 2]

            group_color = COLORS["groups"][k]

            # 画连线 (虚线)
            ax.plot(
                [t_xy[0], a_xy[0]],
                [t_xy[1], a_xy[1]],
                color=group_color,
                linestyle="-",
                alpha=0.6,
                linewidth=1.5,
            )
            ax.plot(
                [t_xy[0], n_xy[0]],
                [t_xy[1], n_xy[1]],
                color=group_color,
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
            )

            # 画点 (Text=星形, Anchor=圆形, Neg=叉形)
            ax.scatter(
                t_xy[0],
                t_xy[1],
                color=group_color,
                marker="*",
                s=300,
                edgecolor="white",
                label="Text" if k == 0 else "",
                zorder=10,
            )
            ax.scatter(
                a_xy[0],
                a_xy[1],
                color=group_color,
                marker="o",
                s=150,
                edgecolor="white",
                label="Anchor" if k == 0 else "",
                zorder=10,
            )
            ax.scatter(
                n_xy[0],
                n_xy[1],
                color=group_color,
                marker="X",
                s=150,
                edgecolor="white",
                label="Hard Neg" if k == 0 else "",
                zorder=10,
            )

            # 标注组号
            ax.text(
                t_xy[0],
                t_xy[1] + 0.5,
                f"Case {k + 1}",
                fontsize=12,
                fontweight="bold",
                color=group_color,
                ha="center",
            )

        ax.set_title(title, fontweight="bold", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        sns.despine(left=True, bottom=True)

        if i == 0:
            # 自定义图例
            from matplotlib.lines import Line2D

            custom_lines = [
                Line2D([0], [0], color="gray", linestyle="-"),
                Line2D([0], [0], color="gray", linestyle="--"),
            ]
            ax.legend(
                custom_lines,
                ["Text-Anchor (Should be Close)", "Text-Negative (Should be Far)"],
                loc="lower left",
                fontsize=10,
            )

    plt.suptitle(
        "Embedding Space Evolution (Focus on Top-3 Improved Cases)",
        fontweight="bold",
        y=0.95,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "3_tsne_top3_focus.png"), dpi=300)
    plt.close()


# ================= 🚀 主程序 =================


def main():
    logger.info(">>> Starting Pro Evaluation Pipeline <<<")

    # 1. 准备数据
    if not os.path.exists(HARD_TEST_FILE):
        logger.error("Data file not found!")
        return

    df_hard = pd.read_csv(HARD_TEST_FILE)
    df_easy = pd.read_csv(EASY_TEST_FILE)

    # 2. 运行 Baseline
    model_base = load_model(BASELINE_CHECKPOINT)

    # 计算 Hard 的分数和 Embedding (用于 t-SNE)
    h_base_pos, h_base_neg, h_base_emb = get_scores_and_embeddings(model_base, df_hard)
    # 计算 Easy 的分数 (只用于 Bar Chart)
    e_base_pos, e_base_neg, _ = get_scores_and_embeddings(model_base, df_easy)

    del model_base
    torch.cuda.empty_cache()

    # 3. 运行 Finetuned
    model_ft = load_model(FINETUNED_CHECKPOINT)

    h_ft_pos, h_ft_neg, h_ft_emb = get_scores_and_embeddings(model_ft, df_hard)
    e_ft_pos, e_ft_neg, _ = get_scores_and_embeddings(model_ft, df_easy)

    del model_ft
    torch.cuda.empty_cache()

    # 4. 计算准确率
    def calc_acc(p, n):
        return np.mean(p > n) * 100

    results = [
        {
            "Model": "Baseline",
            "Dataset": "Hard Samples",
            "Accuracy": calc_acc(h_base_pos, h_base_neg),
        },
        {
            "Model": "Finetuned",
            "Dataset": "Hard Samples",
            "Accuracy": calc_acc(h_ft_pos, h_ft_neg),
        },
        {
            "Model": "Baseline",
            "Dataset": "Easy Samples",
            "Accuracy": calc_acc(e_base_pos, e_base_neg),
        },
        {
            "Model": "Finetuned",
            "Dataset": "Easy Samples",
            "Accuracy": calc_acc(e_ft_pos, e_ft_neg),
        },
    ]

    margin_base = h_base_pos - h_base_neg
    margin_ft = h_ft_pos - h_ft_neg
    improvement = margin_ft - margin_base

    # 获取前3名的索引
    top_3_indices = np.argsort(improvement)[-3:]
    logger.info(f"Top 3 Improved Indices: {top_3_indices}")

    logger.info("--- Generating Plots ---")

    plot_bar_chart_pro(results)

    # 仅使用 Hard 样本画分布图
    score_data_base = {"pos": h_base_pos, "neg": h_base_neg}
    score_data_ft = {"pos": h_ft_pos, "neg": h_ft_neg}
    plot_kde_pro(score_data_base, score_data_ft)

    # 画聚焦版 t-SNE
    plot_focused_tsne(h_base_emb, h_ft_emb, top_3_indices)

    logger.info(f"Done! Check results in {SAVE_DIR}")


if __name__ == "__main__":
    main()
