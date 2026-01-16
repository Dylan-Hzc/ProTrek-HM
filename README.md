# ProTrek-HM: Optimized Protein-Text Retrieval via Hard Negative Mining

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](./LICENSE)

This repository is a **fine-tuned extension** of the original [ProTrek](https://github.com/westlake-repl/ProTrek) model.

While the original ProTrek excels at general protein-text retrieval, it struggles with **"Twin Proteins"**—variants with high sequence homology (>90%) but distinct functions. **ProTrek-HM (Hard Mining)** introduces a contrastive learning pipeline with Triplet Margin Loss to solve this fine-grained distinction challenge.

## 🚀 Key Improvements

| Metric | Original Baseline | **ProTrek-HM (Ours)** | Improvement |
| :--- | :---: | :---: | :---: |
| **Hard Sample Accuracy** | 56.2% | **89.8%** | **+33.6%** 🔺 |
| Easy Sample Accuracy | 98.1% | 91.0% | -7.1% |
| Semantic Separation | Low | High | See Visualization |

> **Core Logic:** We strictly mined "Anchor-Positive-HardNegative" triplets where negatives have high sequence similarity but distinct textual semantics.

---

## 🛠️ Step-by-Step Reproduction Guide

To fully reproduce our results, please follow the 4 steps below strictly.

### 📦 Step 1: Environment & Verification

**1. Install Dependencies**
Follow the instruction from the original ProTrek repository to set up the Conda environment.
```bash
conda env create -f environment.yml
conda activate protrek
```

**2. Verify Installation (Crucial)**
We use **ProTrek_35M** in this project (different from the default 650M). Please ensure you have downloaded the 35M weights into the `weights/` directory.

Run the following Python script to verify the model loads correctly and produces embeddings:

```python
import torch
from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel
from utils.foldseek_util import get_struc_seq

# 1. Load configuration (Using ProTrek_35M)
# Note: Paths updated from 650M to 35M versions
config = {
    "protein_config": "weights/ProTrek_35M/esm2_t12_35M_UR50D",
    "text_config": "weights/ProTrek_35M/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "structure_config": "weights/ProTrek_35M/foldseek_t12_35M",
    "from_checkpoint": "weights/ProTrek_35M/ProTrek_35M.pt"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {device}...")
model = ProTrekTrimodalModel(**config).eval().to(device)

# 2. Load example data
pdb_path = "example/8ac8.cif"
# Ensure the bin/foldseek path is correct for your environment
seqs = get_struc_seq("bin/foldseek", pdb_path, ["A"])["A"]
aa_seq = seqs[0]
foldseek_seq = seqs[1].lower()
text = "Replication initiator in the monomeric form, and autogenous repressor in the dimeric form."

# 3. Inference
with torch.no_grad():
    # Obtain embeddings
    seq_embedding = model.get_protein_repr([aa_seq])
    print("Protein sequence embedding shape:", seq_embedding.shape)
    
    struc_embedding = model.get_structure_repr([foldseek_seq])
    print("Protein structure embedding shape:", struc_embedding.shape)
    
    text_embedding = model.get_text_repr([text])
    print("Text embedding shape:", text_embedding.shape)
    
    # Calculate Similarity Scores
    seq_struc_score = seq_embedding @ struc_embedding.T / model.temperature
    print("Similarity score between protein sequence and structure:", seq_struc_score.item())

    seq_text_score = seq_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein sequence and text:", seq_text_score.item())
    
    struc_text_score = struc_embedding @ text_embedding.T / model.temperature
    print("Similarity score between protein structure and text:", struc_text_score.item())
```

**Expected Output:**
```text
Protein sequence embedding shape: torch.Size([1, 1024])
Protein structure embedding shape: torch.Size([1, 1024])
Text embedding shape: torch.Size([1, 1024])
Similarity score between protein sequence and structure: 28.506...
Similarity score between protein sequence and text: 17.842...
Similarity score between protein structure and text: 11.866...
```

---

### 📥 Step 2: Prepare Codebase
Ensure the core scripts are placed in the root of the repository (parallel to `model/` folder):
* `data_processing.py`
* `model_training.py`
* `evaluate.py`

---

### 🗃️ Step 3: Prepare Data
1. Download the original **ProTrek_data**.
2. Place the `protrek_data.tsv` file in the root directory (or parallel path).
3. Ensure the script arguments in Step 4 point to the correct location of this file.

---

### 🚀 Step 4: Run the Full Pipeline

Run the scripts in the following sequence.

**1. Generate Hard Negatives**
```bash
python data_processing.py \
  --input protrek_data.tsv \
  --output dataset/hard_negatives_train.csv \
  --model_path weights/ProTrek_35M/ProTrek_35M.pt
```
> **Output:** You will see `hard_negative_examples.png` in the `evaluation_results/` directory.

**2. Model Fine-tuning**
```bash
python model_training.py \
  --train_data dataset/hard_negatives_train.csv \
  --base_model weights/ProTrek_35M/ProTrek_35M.pt \
  --epochs 30 \
  --save_dir weights/ProTrek_35M
```
> **Output:** You will see `loss_curve.png` generated in `weights/ProTrek_35M/`.

**3. Evaluation**
```bash
python evaluate.py \
  --model_path weights/ProTrek_35M/protrek_finetuned_epoch30.pt \
  --test_data dataset/hard_negatives_test.csv \
  --output_dir evaluation_results/
```
> **Output:** The following plots will be generated in `evaluation_results/`:
> * `1_accuracy_comparison.png`
> * `2_score_distribution.png`
> * `3_tsne_top3_focus.png`

---

## 📊 Results Gallery

**1. Accuracy Comparison**
Fine-tuning significantly boosts performance on hard samples.
![Accuracy](evaluation_results/1_accuracy_comparison.png)

**2. Embedding Space Evolution (t-SNE)**
**Left (Baseline):** Hard negatives cluster closely with anchors.
**Right (Ours):** The model learns to push hard negatives away.
![t-SNE](evaluation_results/3_tsne_top3_focus.png)

**3. Discriminative Power**
Clear separation between positive and negative pairs.
![Score Dist](evaluation_results/2_score_distribution.png)

## 📂 Repository Structure

```text
├── model/                  # Core ProTrek architecture
├── model_training.py       # Fine-tuning script with Triplet Loss
├── data_processing.py      # Hard negative mining logic
├── evaluate.py             # Evaluation & plotting scripts
├── evaluation_results/     # Generated plots (Accuracy, t-SNE, etc.)
├── example/                # Sample data
├── weights/                # Model checkpoints (ProTrek_35M)
└── environment.yml         # Dependencies
```