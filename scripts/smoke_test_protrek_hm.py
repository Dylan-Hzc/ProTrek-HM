#!/usr/bin/env python
"""Minimal smoke test for the ColabProTrekHM wrapper."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from protrek_hm_colab import ColabProTrekHM


TOY_SEQUENCES = [
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNY",
    "GAVLILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIH",
]

TOY_TEXTS = [
    "DNA-binding protein involved in transcription regulation",
    "Membrane transporter involved in ion transport",
]

REQUIRED_SAMPLE_COLUMNS = ["anchor_seq", "anchor_text", "hard_neg_seq"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small sequence-text smoke test for ProTrek-HM."
    )
    parser.add_argument("--model-dir", default="weights/ProTrek_35M")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--finetuned-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve paths and print wrapper metadata without loading model weights.",
    )
    parser.add_argument(
        "--scale-by-temperature",
        action="store_true",
        help="Report ProTrek matching scores instead of cosine similarities.",
    )
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--sample-csv", default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def truncate_sequence(seq: str, max_len: int) -> str:
    if max_len <= 0:
        raise ValueError("--max-seq-len must be a positive integer.")
    return seq[:max_len]


def tensor_to_list(tensor: torch.Tensor, digits: int = 4) -> list[Any]:
    rounded = torch.round(tensor.detach().cpu() * (10**digits)) / (10**digits)
    return rounded.tolist()


def format_matrix(tensor: torch.Tensor) -> str:
    return json.dumps(tensor_to_list(tensor), ensure_ascii=False)


def format_vector(tensor: torch.Tensor) -> str:
    return json.dumps(tensor_to_list(tensor), ensure_ascii=False)


def make_wrapper(args: argparse.Namespace, finetuned_checkpoint: str | None = None) -> ColabProTrekHM:
    return ColabProTrekHM(
        model_dir=args.model_dir,
        checkpoint_path=args.checkpoint,
        finetuned_checkpoint_path=finetuned_checkpoint,
        device=args.device,
        batch_size=args.batch_size,
        scale_by_temperature=args.scale_by_temperature,
    )


def run_sample_csv_eval(
    wrapper: ColabProTrekHM,
    csv_path: str,
    num_samples: int,
    max_seq_len: int,
) -> dict[str, Any]:
    import pandas as pd

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Sample CSV not found: {path}")

    df = pd.read_csv(path, nrows=num_samples)
    missing = [col for col in REQUIRED_SAMPLE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Sample CSV is missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_SAMPLE_COLUMNS).head(num_samples)
    positive_scores = []
    negative_scores = []
    correct = []

    for _, row in df.iterrows():
        anchor_seq = truncate_sequence(str(row["anchor_seq"]), max_seq_len)
        anchor_text = str(row["anchor_text"])
        hard_neg_seq = truncate_sequence(str(row["hard_neg_seq"]), max_seq_len)

        pos = wrapper.score_pairs([anchor_seq], [anchor_text]).item()
        neg = wrapper.score_pairs([hard_neg_seq], [anchor_text]).item()
        positive_scores.append(pos)
        negative_scores.append(neg)
        correct.append(pos > neg)

    accuracy = float(sum(correct) / len(correct)) if correct else 0.0
    return {
        "csv_path": str(path),
        "num_rows": int(len(df)),
        "accuracy": accuracy,
        "positive_scores": [round(float(value), 4) for value in positive_scores],
        "negative_scores": [round(float(value), 4) for value in negative_scores],
    }


def run_wrapper_once(
    label: str,
    args: argparse.Namespace,
    finetuned_checkpoint: str | None = None,
) -> tuple[dict[str, Any], ColabProTrekHM]:
    wrapper = make_wrapper(args, finetuned_checkpoint=finetuned_checkpoint)
    load_start = time.time()
    wrapper.load()

    toy_sequences = [truncate_sequence(seq, args.max_seq_len) for seq in TOY_SEQUENCES]
    toy_texts = list(TOY_TEXTS)

    similarity = wrapper.similarity_matrix(sequences=toy_sequences, texts=toy_texts)
    pair_scores = wrapper.score_pairs(toy_sequences, toy_texts)
    elapsed = time.time() - load_start

    description = wrapper.describe()
    result: dict[str, Any] = {
        "label": label,
        "active_checkpoint_path": description["active_checkpoint_path"],
        "device": description["device"],
        "sequence_count": len(toy_sequences),
        "text_count": len(toy_texts),
        "similarity_matrix_shape": list(similarity.shape),
        "similarity_matrix": tensor_to_list(similarity),
        "pair_scores": tensor_to_list(pair_scores),
        "elapsed_seconds": elapsed,
    }

    if args.sample_csv:
        result["sample_csv_metrics"] = run_sample_csv_eval(
            wrapper,
            args.sample_csv,
            args.num_samples,
            args.max_seq_len,
        )

    if not args.quiet:
        print(f"[{label}] active checkpoint: {result['active_checkpoint_path']}")
        print(f"[{label}] device: {result['device']}")
        print(f"[{label}] sequence count: {result['sequence_count']}")
        print(f"[{label}] text count: {result['text_count']}")
        print(f"[{label}] similarity matrix shape: {result['similarity_matrix_shape']}")
        print(f"[{label}] similarity matrix: {format_matrix(similarity)}")
        print(f"[{label}] pair scores: {format_vector(pair_scores)}")
        print(f"[{label}] elapsed seconds: {elapsed:.2f}")
        if "sample_csv_metrics" in result:
            metrics = result["sample_csv_metrics"]
            print(f"[{label}] sample csv rows: {metrics['num_rows']}")
            print(f"[{label}] sample csv mini accuracy: {metrics['accuracy']:.4f}")

    return result, wrapper


def save_json(result: dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)


def build_base_result(args: argparse.Namespace, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "model_dir": args.model_dir,
        "checkpoint": args.checkpoint,
        "finetuned_checkpoint": args.finetuned_checkpoint,
        "device": args.device,
        "dry_run": args.dry_run,
        "elapsed_seconds": elapsed_seconds,
        "toy_sequences": [truncate_sequence(seq, args.max_seq_len) for seq in TOY_SEQUENCES],
        "toy_texts": list(TOY_TEXTS),
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be a positive integer.")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be a positive integer.")
    if args.finetuned_checkpoint and not Path(args.finetuned_checkpoint).is_file():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint does not exist: {args.finetuned_checkpoint}"
        )


def main() -> int:
    args = parse_args()
    started = time.time()

    try:
        validate_args(args)
        result = build_base_result(args, elapsed_seconds=0.0)

        if args.dry_run:
            wrapper = make_wrapper(args)
            description = wrapper.describe()
            paths = wrapper.resolve_paths()
            result["describe"] = description
            result["resolved_paths"] = {key: str(value) for key, value in paths.items()}
            result["elapsed_seconds"] = time.time() - started
            if not args.quiet:
                print("Dry run: model weights were not loaded.")
                print("Describe:")
                print(json.dumps(description, indent=2, ensure_ascii=False))
                print("Resolved paths:")
                print(json.dumps(result["resolved_paths"], indent=2, ensure_ascii=False))
            if args.output_json:
                save_json(result, args.output_json)
            return 0

        baseline_result, baseline_wrapper = run_wrapper_once("baseline", args)
        result["baseline_similarity_matrix"] = baseline_result["similarity_matrix"]
        result["baseline_pair_scores"] = baseline_result["pair_scores"]
        result["baseline"] = baseline_result

        del baseline_wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if args.finetuned_checkpoint:
            finetuned_result, finetuned_wrapper = run_wrapper_once(
                "finetuned",
                args,
                finetuned_checkpoint=args.finetuned_checkpoint,
            )
            result["finetuned_similarity_matrix"] = finetuned_result["similarity_matrix"]
            result["finetuned_pair_scores"] = finetuned_result["pair_scores"]
            result["finetuned"] = finetuned_result

            if not args.quiet:
                print("Baseline vs fine-tuned pair scores:")
                print(
                    json.dumps(
                        {
                            "baseline": result["baseline_pair_scores"],
                            "finetuned": result["finetuned_pair_scores"],
                        },
                        ensure_ascii=False,
                    )
                )

            del finetuned_wrapper
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result["elapsed_seconds"] = time.time() - started
        if args.output_json:
            save_json(result, args.output_json)
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
