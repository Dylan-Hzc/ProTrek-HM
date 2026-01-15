import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
import os
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

BATCH_SIZE = 24
LEARNING_RATE = 1e-5  # 全量微调建议使用较小的学习率
EPOCHS = 30  # 训练轮数
MARGIN = 0.2  # Triplet Loss 边界
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 路径配置
HARD_NEG_FILE = 'dataset/hard_negatives_train.csv'  # 挖掘出的困难样本文件
SAVE_DIR = "weights/ProTrek_35M"  # 保存权重的目录

# 模型配置
MODEL_CONFIG = {
    "protein_config": "weights/ProTrek_35M/esm2_t12_35M_UR50D",
    "text_config": "weights/ProTrek_35M/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "weights/ProTrek_35M/foldseek_t12_35M",
    "from_checkpoint": "weights/ProTrek_35M/ProTrek_35M.pt"
}

sys.path.append("./ProTrek")
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ================= 1. 数据集定义 =================
class HardNegativeDataset(Dataset):
    def __init__(self, csv_file):
        logger.info(f"Loading dataset from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        self.df = self.df.dropna(subset=['anchor_seq', 'anchor_text', 'hard_neg_seq']).reset_index(drop=True)
        logger.info(f"Dataset loaded. Total samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row['anchor_seq'], row['anchor_text'], row['hard_neg_seq']


# ================= 2. 核心训练逻辑 =================
def train():
    # 0. 准备环境
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    if not os.path.exists(HARD_NEG_FILE):
        raise FileNotFoundError(f"找不到挖掘数据: {HARD_NEG_FILE}，请先运行 Step 1 代码！")

    # 1. 加载数据
    dataset = HardNegativeDataset(HARD_NEG_FILE)
    # num_workers 加速数据预处理
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

    # 2. 初始化模型
    logger.info("Initializing ProTrek Model...")
    model = ProTrekTrimodalModel(**MODEL_CONFIG).to(DEVICE)
    model.train()  # 开启训练模式 (启用 Dropout 等)

    # 3. 验证 Key 结构
    logger.info("Verifying model architecture for saving...")
    if hasattr(model, 'model') and isinstance(model.model, nn.Module):
        logger.info("Detected 'model.model' structure (ModuleList). Will save this to match evaluate.py.")
        # 打印一下第一个 key 看看是否是数字开头
        sample_key = list(model.model.state_dict().keys())[0]
        logger.info(f"Sample Key: {sample_key} (Expect digit-based key like '1.model...')")
    else:
        logger.warning("Warning: 'model.model' not found. This might cause loading issues in evaluate.py!")

    # 4. 优化器 (全量微调)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 用于记录 Loss
    loss_history = []

    # 5. 训练循环
    logger.info(f"Start Training for {EPOCHS} epochs on {DEVICE}...")

    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch in pbar:
            anchor_seqs, texts, neg_seqs = batch
            # 转为 list 供 tokenizer 使用
            anchor_seqs = list(anchor_seqs)
            texts = list(texts)
            neg_seqs = list(neg_seqs)

            optimizer.zero_grad()

            # --- Forward Pass ---
            # 直接调用官方 API，确保预处理流程一致
            anchor_emb = model.get_protein_repr(anchor_seqs)  # [Batch, 1024]
            text_emb = model.get_text_repr(texts)  # [Batch, 1024]
            neg_emb = model.get_protein_repr(neg_seqs)  # [Batch, 1024]

            # --- Triplet Loss Calculation ---
            # 使用点积 (Dot Product)
            # 正样本得分 (Anchor <-> Text)
            pos_score = torch.sum(anchor_emb * text_emb, dim=1)

            # 负样本得分 (Hard Negative Seq <-> Text)
            neg_score = torch.sum(neg_emb * text_emb, dim=1)

            # Margin Loss: Max(0, neg - pos + margin)
            loss = torch.clamp(neg_score - pos_score + MARGIN, min=0).mean()

            # --- Backward Pass ---
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # --- Epoch 结束 ---
        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)  # 记录 Loss
        logger.info(f"Epoch {epoch + 1} Completed. Avg Loss: {avg_loss:.4f}")

        save_path = os.path.join(SAVE_DIR, f"protrek_finetuned_epoch{epoch + 1}.pt")

        # 确定要保存的状态字典
        if hasattr(model, 'model'):
            model_state = model.model.state_dict()
        else:
            model_state = model.state_dict()

        save_dict = {
            "model": model_state,
            "config": MODEL_CONFIG,
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict()
        }

        if epoch % 5 == 4:
            torch.save(save_dict, save_path)
            logger.info(f"Checkpoint saved to {save_path}")

    logger.info("Training Finished.")

    # ================= 6. 绘制 Loss 曲线 =================
    logger.info("Plotting loss curve...")
    try:
        plt.switch_backend('Agg')

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', linestyle='-', color='b', label='Train Loss')
        plt.title('Training Loss Curve (Fine-tuning)')
        plt.xlabel('Epoch')
        plt.ylabel('Triplet Margin Loss')
        plt.grid(True)
        plt.legend()

        # 保存图片
        plot_path = os.path.join(SAVE_DIR, "loss_curve.png")
        plt.savefig(plot_path)
        logger.info(f"Loss curve saved to: {plot_path}")

        # 保存 CSV 数据
        df_loss = pd.DataFrame({'epoch': range(1, EPOCHS + 1), 'loss': loss_history})
        csv_path = os.path.join(SAVE_DIR, "loss_history.csv")
        df_loss.to_csv(csv_path, index=False)
        logger.info(f"Loss data saved to: {csv_path}")

    except Exception as e:
        logger.error(f"Failed to plot loss curve: {e}")


if __name__ == "__main__":
    train()