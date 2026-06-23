# ColabProTrek-HM: A ColabPLM for Hard-Negative Protein-Text Retrieval

## 1. Project Summary

This is a Track B final project deliverable. It adapts ProTrek into a ColabPLM-style notebook and lightweight wrapper that can run protein sequence to natural-language function retrieval in Google Colab or a local Python environment.

The default path uses ProTrek-35M and computes sequence-text similarity. The project also includes an optional hard-negative-mined fine-tuned checkpoint path for ProTrek-HM comparison, using an inference-only checkpoint exported from the local training checkpoint.

## 2. Why ProTrek

ProTrek is a tri-modal protein representation model for sequence, structure, and text. This project uses the sequence-text path as the default Colab demo because it is the smallest reliable path for a course runtime while still demonstrating multimodal protein retrieval.

Structure/Foldseek support is optional. The wrapper can embed Foldseek sequences directly, but it does not require a Foldseek binary or PDB preprocessing in the default Colab workflow.

## 3. What is ProTrek-HM

HM means Hard Negative Mining. The existing local project constructed hard negative triplets where the negative protein sequence is similar to the anchor sequence but has different text semantics.

Fine-tuning used:

- anchor protein sequence
- positive function text
- hard negative protein sequence

The objective increased separation between true sequence-text pairs and hard negative sequence-text pairs. This README does not claim state-of-the-art performance; it documents a reproducible course implementation and a Colab-friendly adaptation.

## 4. Repository Additions

| File | Purpose |
|---|---|
| `protrek_hm_colab.py` | Lightweight `ColabProTrekHM` wrapper for path resolution, model loading, embeddings, similarity matrices, and pair scores. |
| `scripts/smoke_test_protrek_hm.py` | Minimal baseline and optional fine-tuned sequence-text smoke test. |
| `scripts/export_inference_checkpoint.py` | Exports a training checkpoint to an inference-only checkpoint with optimizer state removed. |
| `colab/ColabProTrek_HM.ipynb` | Runnable Colab notebook scaffold for Track B. |
| `requirements_colab.txt` | Minimal Colab dependency list, excluding PyTorch and non-default retrieval/demo dependencies. |
| `README_ColabProTrek_HM.md` | This usage and delivery guide. |
| `report_outline_track_b.md` | Technical report outline for the final write-up. |

## 5. Default Runtime Path

- Model: ProTrek-35M
- Task: sequence-text similarity
- Required assets:
  - `weights/ProTrek_35M/ProTrek_35M.pt`
  - `weights/ProTrek_35M/esm2_t12_35M_UR50D/`
  - `weights/ProTrek_35M/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/`
  - `weights/ProTrek_35M/foldseek_t12_35M/`
- Optional fine-tuned checkpoint:
  - `weights/ProTrek_35M/protrek_hm_35m_inference_only.pt`

Hosted optional fine-tuned checkpoint:

```text
https://github.com/Dylan-Hzc/ProTrek-HM/releases/download/colabprotrek-hm-v0.1/protrek_hm_35m_inference_only.pt
```

This file is an inference-only ProTrek-HM 35M checkpoint exported from the original training checkpoint. It is intentionally hosted as a release asset instead of being committed to Git. The baseline-only path does not require this file; the optional fine-tuned comparison uses it when available.

## 6. Local Quick Start

Activate the local environment:

```text
conda activate protrek
```

Dry-run path resolution without loading a checkpoint:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test_protrek_hm.py --model-dir weights/ProTrek_35M --checkpoint weights/ProTrek_35M/ProTrek_35M.pt --device auto --dry-run
```

Run the baseline ProTrek-35M toy smoke test:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test_protrek_hm.py --model-dir weights/ProTrek_35M --checkpoint weights/ProTrek_35M/ProTrek_35M.pt --device auto --batch-size 1 --max-seq-len 128
```

Run optional baseline versus fine-tuned comparison:

```text
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/smoke_test_protrek_hm.py --model-dir weights/ProTrek_35M --checkpoint weights/ProTrek_35M/ProTrek_35M.pt --finetuned-checkpoint weights/ProTrek_35M/protrek_hm_35m_inference_only.pt --device auto --batch-size 1 --max-seq-len 128
```

## 7. Exporting an Inference-only Checkpoint

The original training checkpoint can contain optimizer state and other training metadata. The inference-only checkpoint is smaller and more Colab-friendly. It keeps the loader-compatible top-level `model` key, plus `metadata` and optional `config`, while removing optimizer state.

Export command:

```text
python scripts/export_inference_checkpoint.py --input weights/ProTrek_35M/protrek_finetuned_epoch30.pt --output weights/ProTrek_35M/protrek_hm_35m_inference_only.pt
```

The local export converted a roughly 1.8G training checkpoint into a roughly 678M inference-only checkpoint.

## 8. Google Colab Usage

1. Open `colab/ColabProTrek_HM.ipynb`.
2. Use a GPU runtime if available.
3. If the branch is not pushed, update `BRANCH` in the notebook after pushing or merging.
4. Let the notebook download baseline ProTrek-35M from Hugging Face, or provide the baseline files manually.
5. The notebook includes the hosted GitHub Release URL for the optional fine-tuned inference-only checkpoint.
6. If you use another checkpoint host, update `FINETUNED_CKPT_URL` or `GOOGLE_DRIVE_FINETUNED_PATH` in the notebook.
7. Run baseline-only if the fine-tuned checkpoint is absent.

## 9. What is Deliberately Not Default

The default Colab path does not use:

- ProTrek-650M
- SwissProt FAISS index
- Foldseek binary pipeline
- full training
- full `evaluate.py`
- demo server

These are excluded because of model size, Colab memory/runtime constraints, reproducibility, and course demo scope. The goal is a minimal, understandable, runnable sequence-text retrieval workflow.

## 10. Expected Screenshots

Capture:

- runtime/GPU check
- dependency installation
- weight setup
- wrapper path resolution
- baseline model loaded
- toy similarity matrix
- optional fine-tuned comparison
- saved outputs

## 11. Known Limitations

- Toy examples validate pipeline execution; they are not biological conclusions.
- The fine-tuned checkpoint is not in Git because `weights/` is gitignored.
- Structure retrieval requires Foldseek preprocessing and is optional.
- CPU-only runtime may be slow.
- ProTrek-35M is the default demo model.
- ProTrek-650M and the FAISS index are excluded from the default Colab path.

## 12. Track B Deliverable Mapping

| Track B requirement | Project artifact |
|---|---|
| Technical report | `report_outline_track_b.md`, to be expanded into the final report. |
| Successful execution screenshots | Notebook screenshot checklist and smoke test outputs. |
| Complete runnable Colab notebook link | `colab/ColabProTrek_HM.ipynb`, to be opened and shared through Colab. |
| Code adaptations | `protrek_hm_colab.py`, `scripts/smoke_test_protrek_hm.py`, `scripts/export_inference_checkpoint.py`. |
| Optional model checkpoint hosting | Upload `protrek_hm_35m_inference_only.pt` to Google Drive, GitHub Release, or Hugging Face. |
