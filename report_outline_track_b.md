# Technical Report Outline: ColabProTrek-HM

## Abstract

- Write a one-paragraph summary.
- Identify ProTrek as the selected protein language model.
- Explain the ColabPLM adaptation.
- Mention the hard-negative fine-tuned ProTrek-HM extension.
- Summarize the main smoke/fine-tuned comparison result.

## 1. Introduction

- Introduce protein language models and multimodal retrieval.
- Explain why sequence-text protein retrieval matters.
- State the course Track B goal.
- Summarize the project contribution: a runnable Colab wrapper and notebook for ProTrek-HM.

## 2. Background: ProTrek

- Describe the sequence encoder.
- Describe the structure encoder.
- Describe the text encoder.
- Explain contrastive alignment across modalities.
- Explain why the sequence-text path is used as the default Colab demo.
- Note the optional structure modality.

## 3. Model Architecture

- `ProTrekTrimodalModel`
- `ProteinEncoder`
- `TextEncoder`
- `StructureEncoder`
- Projection into a shared representation space.
- Normalized embeddings.
- Cosine similarity and optional temperature scaling.

## 4. ProTrek-HM: Hard Negative Mining

- Motivate hard negatives for fine-grained protein-text retrieval.
- Define the anchor sequence.
- Define the positive text.
- Define the hard negative sequence.
- List the data columns: `anchor_seq`, `anchor_text`, `hard_neg_seq`, `hard_neg_text`, `seq_similarity`, `type`.
- Explain why easy negatives are insufficient.
- Avoid overclaiming beyond a course-scale experiment.

## 5. Fine-tuning Objective

- Describe the triplet-style ranking loss.
- Define positive score versus negative score.
- Explain the margin.
- Describe checkpoint saving.
- Explain why an inference-only checkpoint is exported for Colab.

## 6. ColabPLM Implementation

- `ColabProTrekHM` wrapper.
- Local path resolution.
- CPU/GPU automatic device selection.
- Baseline and optional fine-tuned checkpoint loading.
- Sequence embedding method.
- Text embedding method.
- Similarity matrix.
- Pair scoring.
- Explain why FAISS, demo server, and training are excluded from the default path.

## 7. Colab Notebook Workflow

- Runtime check.
- Clone or locate repo.
- Dependency install.
- Weight setup.
- Dry-run path resolution.
- Baseline model loading.
- Toy similarity matrix.
- Optional fine-tuned comparison.
- Optional mini hard-negative CSV evaluation.
- Saved outputs.

## 8. Experiments

- Local baseline smoke test result.
- Local fine-tuned comparison result.
- Optional mini CSV evaluation.
- Screenshots to include.
- Runtime environment.

Observed local values:

- baseline pair scores: `[0.0192, -0.0292]`
- fine-tuned pair scores: `[0.1910, -0.1382]`
- toy similarity matrix shape: `[2, 2]`
- checkpoint export: 1.8G training checkpoint to approximately 678M inference-only checkpoint
- note that toy examples only validate pipeline execution

## 9. Results and Discussion

- The pipeline successfully loads ProTrek-35M.
- The wrapper computes sequence-text similarity.
- The fine-tuned checkpoint changes pair scores in the expected direction on the toy demo.
- The hard-negative design targets more discriminative retrieval.
- Distinguish engineering validation from biological validation.

## 10. Reproducibility

- Repository branch.
- Colab notebook link.
- `requirements_colab.txt`.
- Model assets.
- Fine-tuned checkpoint hosting path.
- Commands for smoke test.
- Random seed if applicable.
- Exact conda/local environment for local run.

## 11. Limitations

- This is not a state-of-the-art claim.
- Fine-tuned checkpoint hosting is required.
- ProTrek-650M is excluded by default.
- FAISS index retrieval is excluded by default.
- Foldseek structure pipeline is optional.
- The default demo is a small toy demo.
- CPU-only runtime has speed limitations.

## 12. Conclusion

- Track B deliverable achieved.
- ProTrek adapted into a ColabPLM-style workflow.
- ProTrek-HM adds a hard-negative retrieval extension.
- Future work: fully hosted Colab checkpoint, structure modality, larger evaluation.

## Appendix A: Screenshot Checklist

- Runtime / GPU check.
- Dependency installation.
- Weight setup.
- Wrapper path resolution.
- Baseline model loaded.
- Toy similarity matrix output.
- Optional fine-tuned comparison.
- Final saved outputs.

## Appendix B: Commands

Local dry-run:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test_protrek_hm.py --model-dir weights/ProTrek_35M --checkpoint weights/ProTrek_35M/ProTrek_35M.pt --device auto --dry-run
```

Baseline smoke test:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test_protrek_hm.py --model-dir weights/ProTrek_35M --checkpoint weights/ProTrek_35M/ProTrek_35M.pt --device auto --batch-size 1 --max-seq-len 128
```

Fine-tuned smoke test:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test_protrek_hm.py --model-dir weights/ProTrek_35M --checkpoint weights/ProTrek_35M/ProTrek_35M.pt --finetuned-checkpoint weights/ProTrek_35M/protrek_hm_35m_inference_only.pt --device auto --batch-size 1 --max-seq-len 128
```

Checkpoint export:

```text
python scripts/export_inference_checkpoint.py --input weights/ProTrek_35M/protrek_finetuned_epoch30.pt --output weights/ProTrek_35M/protrek_hm_35m_inference_only.pt
```
