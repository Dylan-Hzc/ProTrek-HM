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
sns.set_theme(style="whitegrid", context="paper")

# 更新为更专业、对比度更高的配色
COLORS = {
    "pos": "#48C9B0",  # 湖绿色 (Soft Green/Blue)
    "neg": "#E74C3C",  # 珊瑚红 (Red/Orange)
    "bg": "#BDC3C7",  # 灰色基调
    "accent": "#2E86C1",  # 强调蓝
    "groups": sns.color_palette("Set2", 3).as_hex(),
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
    图 1: 准确率对比 (Grouped Bar Chart with soft colors)
    """
    logger.info("Plotting Pro Bar Chart...")
    plt.figure(figsize=(8, 6))

    df = pd.DataFrame(acc_data)

    # 绘制分组柱状图
    ax = sns.barplot(
        data=df,
        x="Dataset",
        y="Accuracy",
        hue="Model",
        palette="Set2",
        edgecolor=".2",
        linewidth=1,
        alpha=0.9,
    )

    sns.despine()
    plt.ylim(0, 115)
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    plt.xlabel("")
    plt.title(
        "Classification Accuracy Comparison", fontsize=14, fontweight="bold", pad=20
    )
    plt.legend(frameon=True, loc="upper left", fontsize="small", title="Model Type")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    # 标注数值
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1f%%",
            padding=5,
            fontsize=10,
            fontweight="bold",
            color="#2c3e50",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "1_accuracy_comparison.png"), dpi=300)
    plt.close()


def plot_kde_pro(base_scores, ft_scores):
    """
    图 2: 分数分布 (KDE with fill=True)
    """
    logger.info("Plotting Pro KDE...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    def draw_kde(ax, pos, neg, title):
        sns.kdeplot(
            pos,
            ax=ax,
            fill=True,
            color=COLORS["pos"],
            alpha=0.4,
            linewidth=2.5,
            label="Positive Pairs",
        )
        sns.kdeplot(
            neg,
            ax=ax,
            fill=True,
            color=COLORS["neg"],
            alpha=0.4,
            linewidth=2.5,
            label="Negative Pairs",
        )
        ax.set_title(title, fontweight="bold", fontsize=13, pad=15)
        ax.set_xlabel("Similarity Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        sns.despine(ax=ax)

    draw_kde(
        axes[0],
        base_scores["pos"],
        base_scores["neg"],
        "Baseline (ProTrek-35M Zero-shot)",
    )
    draw_kde(
        axes[1],
        ft_scores["pos"],
        ft_scores["neg"],
        "Finetuned (Optimized Embedding Space)",
    )

    axes[0].legend(frameon=True, loc="upper right", fontsize="small")

    plt.suptitle(
        "Contrastive Score Distribution: Overcoming Hard Negatives",
        fontweight="bold",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, "2_score_distribution.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()


def plot_focused_tsne(base_emb, ft_emb, top_indices):
    logger.info("Plotting Focused t-SNE...")

    # 准备数据
    n_bg = 50
    bg_indices = np.random.choice(len(base_emb["text"]), n_bg, replace=False)
    target_indices = top_indices
    all_indices = list(set(list(bg_indices) + list(target_indices)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    for i, (emb_dict, title) in enumerate(
        [(base_emb, "Baseline Latent Space"), (ft_emb, "Finetuned Latent Space")]
    ):
        ax = axes[i]
        vectors = []
        for idx in all_indices:
            vectors.append(emb_dict["text"][idx].flatten())
            vectors.append(emb_dict["anchor"][idx].flatten())
            vectors.append(emb_dict["hard_neg"][idx].flatten())
        vectors = np.array(vectors)

        reducer = TSNE(
            n_components=2, perplexity=10, random_state=42, init="pca", learning_rate=50
        )
        embedded = reducer.fit_transform(vectors)

        # 1. 绘制背景点 (极简，低透明度)
        for j, list_idx in enumerate(all_indices):
            if list_idx in target_indices:
                continue
            ax.scatter(
                embedded[3 * j : 3 * j + 3, 0],
                embedded[3 * j : 3 * j + 3, 1],
                c="#D5D8DC",
                s=20,
                alpha=0.15,
                zorder=1,
            )

        # 2. 绘制 Top 3 重点样本
        for k, target_idx in enumerate(target_indices):
            j = all_indices.index(target_idx)
            t_xy, a_xy, n_xy = embedded[3 * j], embedded[3 * j + 1], embedded[3 * j + 2]
            group_color = COLORS["groups"][k]

            # 绘制细灰色连线 (Anchor -> Text, Anchor -> Negative)
            ax.plot(
                [a_xy[0], t_xy[0]],
                [a_xy[1], t_xy[1]],
                color="#BDC3C7",
                linestyle="-",
                linewidth=0.8,
                alpha=0.5,
                zorder=2,
            )
            ax.plot(
                [a_xy[0], n_xy[0]],
                [a_xy[1], n_xy[1]],
                color="#BDC3C7",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                zorder=2,
            )

            # 绘制样本点 (缩小 s, 增加 alpha)
            ax.scatter(
                t_xy[0],
                t_xy[1],
                color=group_color,
                marker="*",
                s=180,
                alpha=0.7,
                zorder=10,
                label="Text" if k == 0 else "",
            )
            ax.scatter(
                a_xy[0],
                a_xy[1],
                color=group_color,
                marker="o",
                s=100,
                alpha=0.7,
                zorder=10,
                label="Anchor" if k == 0 else "",
            )
            ax.scatter(
                n_xy[0],
                n_xy[1],
                color=group_color,
                marker="X",
                s=100,
                alpha=0.7,
                zorder=10,
                label="Hard Neg" if k == 0 else "",
            )

        ax.set_title(title, fontweight="bold", fontsize=15, pad=15)
        ax.set_xticks([])
        ax.set_yticks([])

        # 移除边框 (Minimalist)
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.suptitle(
        "Embedding Evolution: Anchor-Centric Visualization",
        fontweight="bold",
        fontsize=18,
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(SAVE_DIR, "3_tsne_top3_focus.png"), dpi=300, bbox_inches="tight"
    )
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
