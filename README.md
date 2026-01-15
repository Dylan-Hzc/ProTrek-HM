# ProTrek-HM: Optimized Protein-Text Retrieval via Hard Negative Mining

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)

This repository is a **fine-tuned extension** of the original [ProTrek](https://github.com/westlake-repl/ProTrek) model.

While the original ProTrek excels at general protein-text retrieval, it struggles with **"Twin Proteins"**—variants with high sequence homology (>90%) but distinct functions. **ProTrek-HM (Hard Mining)** introduces a contrastive learning pipeline with Triplet Margin Loss to solve this fine-grained distinction challenge.

## 🚀 Key Improvements

| Metric | Original Baseline | **ProTrek-HM (Ours)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Hard Sample Accuracy** | 56.2% | **90.5%** | **+34.3%** 🔺 |
| Easy Sample Accuracy | 98.1% | 90.4% | -7.7% |
| Semantic Separation | Low | High | See Visualization |

> **Core Logic:** We strictly mined "Anchor-Positive-HardNegative" triplets where negatives have high sequence similarity but distinct textual semantics.

## 📊 Performance Visualization

### 1. The Challenge: "Twin" Proteins
We identified hard negatives where a single point mutation leads to a functional shift, yet standard models fail to distinguish them.
![Hard Negative Example](dataset/hard_negative_examples.png)
*(Note: Visualizes the sequence alignment and semantic gap)*

### 2. Quantitative Improvement
Fine-tuning significantly boosts performance on hard samples, effectively solving the "confusion" problem.
![Accuracy Comparison](evaluation_results/1_accuracy_comparison.png)

### 3. Embedding Space Evolution (t-SNE)
**Left (Baseline):** Hard negatives cluster closely with anchors (Confusion).  
**Right (Ours):** The model learns to push hard negatives away while keeping anchors and text aligned.
![t-SNE Visualization](evaluation_results/3_tsne_top3_focus.png)

### 4. Discriminative Power
The score distribution shows a clear separation between positive and negative pairs after fine-tuning.
![Score Distribution](evaluation_results/2_score_distribution.png)

## 🛠️ Installation & Usage

### Prerequisites

**1. Environment Setup**
```bash
conda env create -f environment.yml
conda activate protrek
```

**2. Download Original Weights**
Since we fine-tune based on the original ProTrek, please download the base model (e.g., `ProTrek_650M`) and place it in the `weights/` directory.
* **[Insert Link to Original ProTrek Weights Here]**

---

### 1. Hard Negative Mining (Data Generation)
We generate the training triplets (Anchor-Positive-HardNegative) from the original dataset.
* **Input:** The original ProTrek dataset (`dataset/protrek_data.tsv`)
* **Output:** A CSV file containing mined hard negatives.

```bash
# This script uses the base model to calculate semantic differences
# Note: Ensure your data_processing.py accepts these arguments
python data_processing.py \
  --input dataset/protrek_data.tsv \
  --output dataset/hard_negatives_train.csv \
  --model_path weights/ProTrek_650M
```

### 2. Fine-tuning
Train the model using the mined hard triplets and Triplet Margin Loss.

```bash
python model_training.py \
  --train_data dataset/hard_negatives_train.csv \
  --base_model weights/ProTrek_650M \
  --epochs 30 \
  --lr 1e-5 \
  --save_dir weights/ProTrek-HM
```

*(Check `evaluation_results/loss_curve.png` for training progress)*

### 3. Evaluation & Visualization
After training, run the evaluation script to reproduce the paper's figures (Accuracy, t-SNE, Score Distribution).

```bash
python evaluate.py \
  --model_path weights/ProTrek-HM/final_checkpoint.pt \
  --test_data dataset/hard_negatives_test.csv \
  --output_dir evaluation_results/
```

## 📂 Repository Structure

```text
├── model/                  # Core ProTrek architecture
├── model_training.py       # Fine-tuning script with Triplet Loss
├── data_processing.py      # Hard negative mining logic
├── evaluate.py             # Evaluation & plotting scripts
├── evaluation_results/     # Generated plots (Accuracy, t-SNE, etc.)
├── example/                # Sample data
└── environment.yml         # Dependencies
```
