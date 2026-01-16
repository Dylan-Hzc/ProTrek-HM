# An improved ProTrek search engine

**Project:** Optimization Strategies for ProTrek search engine

**Students:** Zhangchi Huang, Hongyuan Chen

**GitHub:** https://github.com/Dylan-Hzc/ProTrek-HM.git

**Date:** 2026-01-15

## 1. Overview

State-of-the-art protein language models (pLMs) like ProTrek utilize tri-modal contrastive learning to align sequence, structure, and text. However, standard InfoNCE pre-training often suffers from the **"Granularity Gap,"** where functionally distinct but structurally homologous proteins collapse into identical embedding vectors. This project addresses this limitation by implementing a fine-tuning regime centered on **Hard Negative Mining** and **Triplet Margin Loss**. By explicitly identifying "twin-like" proteins—those with high sequence identity but divergent functional descriptions—and enforcing a geometric margin in the embedding space, we significantly improve the model's discriminative precision. Our fine-tuned ProTrek-35M model achieves a Pairwise Ranking Accuracy of 89.78% on hard samples, compared to the baseline's 56.20%.

## 2. Background

#### 2.1 The ProTrek Architecture

Traditional bioinformatics retrieval tools (such as BLAST and HMMER) rely on sequence alignment and evolutionary conservation. While they excel at identifying homologs, they struggle to capture the semantic connection between "user intent" and the "biological object."

ProTrek stands at the forefront of this technological shift, employing a tri-modal alignment architecture:

- Sequence Encoding: Based on ESM-2 (Transformer), capturing the physicochemical rules of amino acids.
- Structure Encoding: Utilizes Foldseek (3Di alphabet) to discretize 3D coordinates into structural tokens.
- Text Encoding: Uses PubMedBERT to comprehend specialized biomedical semantics.
- Alignment Mechanism: Employs InfoNCE Loss for contrastive learning to project all three modalities into a shared latent space.

Although ProTrek accelerates large-scale retrieval by 30-60x, it suffers from **The Granularity Gap.** Specifically, the model fails to distinguish between proteins with high sequence similarity but distinct functions (e.g., wild-type GFP vs. a non-fluorescent point mutant). In the vector space, their cosine similarity often exceeds 0.99, leading to "Embedding Collapse."

#### 2.2 Survey of SOTA Algorithms

To identify an optimization path, we surveyed retrieval algorithms across different levels:

- **Traditional Algorithms (BLAST/MMseqs2):** High precision based on local alignment, but computational costs grow linearly with database size, making them unsuitable for billion-scale data.
- **Dual Encoders (ProTrek/ESM):** Extremely fast (via ANN Search), but suffer from the aforementioned granularity bottleneck.
- **Late Interaction Models (ColBERT):** Preserve token-level interactions to solve local alignment issues, but incur prohibitive storage costs (Exabyte scale), making them suitable only for re-ranking.
- **Hard Negative Contrastive Learning (CLEAN/Protein-Vec):** Introducing "Hard Negatives" forces the model to learn subtle features, representing a key direction for resolving granularity issues.

## 3. Method

### 3.1 Hard Negative Mining Strategy

We constructed a specialized dataset to force the model to recognize functional divergence amidst sequence similarity. Negatives were mined from `protrek_data.tsv` using the following tiered logic:

1. **Cluster Grouping:** Examples were grouped by `foldseek_seq` as a structure-based proxy.
2. **"Twin" Selection Criteria:** Within clusters, we selected protein pairs $(P_{anchor}, P_{neg})$ satisfying:
   - **High Sequence Similarity:** $0.7 \le \text{SequenceMatcher Ratio} \le 0.9999$
   - **Low Text Similarity:** $\text{TF-IDF Cosine Similarity} < 0.5$

This strict filtering yields pairs that look identical to a sequence encoder but perform different biological functions (e.g., a kinase vs. an inactive mutant).

**Figure 1 (Hard negative example mined from the dataset).**

![](evaluation_results/hard_negative_examples.png)

Interpretation:
- The visualization highlights a high-identity pair and pinpoints a single-point mutation.
- The text descriptions differ, demonstrating why these pairs are hard: small sequence changes can correspond to large functional changes.

### 3.2 Fine-tuning Objective (Triplet Margin Loss)

For each mined sample we form a triplet:

- **Anchor protein sequence**: `anchor_seq`
- **Anchor text description**: `anchor_text`
- **Hard negative protein sequence**: `hard_neg_seq`

Unlike InfoNCE, which optimizes against a batch of random negatives, we employ **Triplet Margin Loss** to enforce a specific geometric constraint. 

For each triplet tuple $( \text{Anchor Protein}, \text{Positive Text}, \text{Hard Negative Protein} )$, we calculate:

- **Positive Score:** $s_{pos} = \langle f_{prot}(anchor\_seq), f_{text}(anchor\_text) \rangle$
- **Negative Score:** $s_{neg} = \langle f_{prot}(hard\_neg\_seq), f_{text}(anchor\_text) \rangle$

The objective function is minimized:

$$\mathcal{L} = \max(0,\; s_{neg} - s_{pos} + m)$$

Where the margin $m = 0.2$. This forces the model to push the hard negative away from the text by a distinct margin, ensuring $s_{pos} \ge s_{neg} + m$.

## 4. Experimental Setup

### 4.1 Data Splits

We evaluated the model on two distinct sets to measure both improvement in resolution and retention of general knowledge.

| **Dataset Type**           | **Filename**               | **Row Count** | **Description**                                  |
| -------------------------- | -------------------------- | ------------- | ------------------------------------------------ |
| **Hard Negatives (Train)** | `hard_negatives_train.csv` | 545           | Used for optimization.                           |
| **Hard Negatives (Test)**  | `hard_negatives_test.csv`  | 137           | The "Twin" pairs; the primary evaluation target. |
| **Easy Samples (All)**     | `easy_samples_all.csv`     | 1000          | Random non-hard samples to check generalization. |

### 4.2 Training Configuration

- Baseline Checkpoint: `weights/ProTrek_35M/ProTrek_35M.pt`
- Optimizer: AdamW (Weight decay: $1e^{-4}$)
- Hyperparameters:
  - **Batch size: 4** (Adjusted from 16 to fit 24GB GPU VRAM)
  - Learning Rate: $1e^{-5}$
  - Epochs: 30
  - Margin ($m$): 0.2
  - **Optimization: Mixed Precision (AMP) and Gradient Accumulation** (Used to mitigate OOM during full fine-tuning of tri-modal components).
- Hardware: NVIDIA GPU (24GB VRAM).

## 5. Evaluation

### 5.1 Evaluation datasets

We evaluate on two datasets:

- **Hard samples**: the difficult mined pairs.
- **Easy samples**: randomly sampled non-hard examples for sanity/generalization.

### 5.2 Metrics

Since the model outputs real-valued similarity scores, we evaluated performance using both ranking and classification metrics:

**Pairwise Ranking Accuracy**

Defined as the probability that the correct protein scores higher than the hard negative:

$$
\text{PairwiseAcc} = \Pr(s_{pos} > s_{neg})
$$
It measures how often the positive pair outranks its matched hard negative.

**Precision, Recall & F1 Score**

We evaluate performance by treating the retrieval task as a binary classification problem. Each similarity score is treated as a labeled instance:

- Label 1 (Positive): Corresponds to the positive pair score, $s_{pos}$.
- Label 0 (Negative): Corresponds to the negative pair score, $s_{neg}$.

For a decision threshold $\tau$:

- Predict Positive if $s \ge \tau$
- Predict Negative if $s < \tau$

We report the standard classification metrics:
$$
\text{Precision} = \frac{TP}{TP+FP} \quad \quad \text{Recall} = \frac{TP}{TP+FN}
$$

$$
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$



*Note: The threshold $\tau$ is determined dynamically by sweeping through all unique scores in the dataset and selecting the value that maximizes the F1 score.*

## 6. Results & Discussion

### 6.1 Quantitative Comparison

The fine-tuning process yielded substantial improvements in the model's ability to resolve hard negatives.

| **Dataset**       | **Model**      | **PairwiseAcc (↑)** | **Precision (↑)** | **Recall (↑)** | **F1 (↑)** | **Best τ** |
| ----------------- | -------------- | ------------------- | ----------------- | -------------- | ---------- | ---------- |
| **Hard (n=137)**  | **Baseline**   | 0.5620              | 0.5018            | 1.0000         | 0.6683     | 0.1963     |
| **Hard (n=137)**  | **Fine-tuned** | **0.8978**          | **0.7643**        | **0.7810**     | **0.7726** | 0.1402     |
| **Easy (n=1000)** | **Baseline**   | 0.9810              | 0.9949            | 0.9690         | 0.9818     | 0.1406     |
| **Easy (n=1000)** | **Fine-tuned** | 0.9100              | 0.8480            | 0.8650         | 0.8564     | 0.1665     |

Key takeaways:
- **Resolution of the "Identity Trap":** The Baseline model performed near random guessing (Acc ~0.56) on Hard samples, confirming it cannot distinguish between high-identity sequences. The **Fine-tuned model improved this to ~0.90**, demonstrating it successfully learned to utilize the "non-identical" residue information to separate functions.
- **Generalization Trade-off:** We observed a slight performance regression on "Easy" samples (0.98 $\to$ 0.91). This suggests the model may be over-specializing to difficult cases. Future work could employ a mixed-batch training strategy (Hard + Random Negatives) to maintain the global structure while refining local resolution.

### 6.2 Accuracy bar chart (pairwise ranking accuracy)

<img src="evaluation_results/1_accuracy_comparison.png" style="zoom: 25%;" />

Interpretation:
- The finetuned model significantly improves on the **Hard Samples** bar.
- The Easy Samples bar can decrease if fine-tuning focuses only on hard negatives.

## 7. Training Details

The training loop logs an epoch-level loss history and saves a loss curve.

**Figure (Training loss curve).**

<img src="weights/ProTrek_35M/loss_curve.png" style="zoom: 67%;" />

Observations:

The loss decreased from 0.1916 to 0.0067 over 30 epochs with steady convergence. No divergence was observed, confirming that a learning rate of $1e^{-5}$ and margin of 0.2 are stable for this architecture.

## 8. Visual Analysis

### 8.1 t-SNE: embedding space evolution (Top-3 improved cases)

The visualization focuses on the **Top 3 Improved Indices: [77, 50, 133]**, which showed the highest increase in cosine similarity between anchor protein and positive text after fine-tuning.

![](evaluation_results/3_tsne_top3_focus.png)

The t-SNE plots reveal that the fine-tuned space physically moves the anchor protein embedding closer to its corresponding text embedding, while pushing the hard negative "twin" to a more distant region of the vector space, effectively breaking the sequence identity correlation.

### 8.2 Score distribution shift

![](evaluation_results/2_score_distribution.png)

- **Baseline:** The distributions of Positive scores ($s_{pos}$) and Hard Negative scores ($s_{neg}$) overlap almost completely, indicating “confused”.
- **Fine-tuned:** The distributions separate, with $s_{pos}$ shifting right and $s_{neg}$ shifting left, creating the desired margin.

## 9. Contributions

- Zhangchi Huang (Implementation Lead & Methodology):
  - Core Implementation: Constructed the hard negative mining pipeline (data preprocessing) and implemented the Triplet Margin Loss fine-tuning framework.
  - Experimentation: Executed the training loops, debugged the evaluation scripts, and optimized hyperparameters (learning rate, margin).
  - Writing: Authored the Methodology sections, defining the technical scope and architectural details of the project.
- Hongyuan Chen (Research & Analysis):
  - SOTA Survey & Motivation: Conducted the literature survey on state-of-the-art algorithms (e.g., ColBERT, Protein-Vec) and formulated the theoretical background regarding the "Granularity Gap" .
  - Analysis & Writing: Performed the detailed interpretation of experimental metrics (Precision/Recall/F1), authored the other sections, and refined the overall logical flow of the report.

---

