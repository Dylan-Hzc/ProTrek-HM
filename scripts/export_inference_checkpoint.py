#!/usr/bin/env python
"""Export a training checkpoint as an inference-only ProTrek checkpoint."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


CANDIDATE_MODEL_KEYS = ("model", "state_dict", "model_state_dict", "net", "module")
CREATED_BY = "scripts/export_inference_checkpoint.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a checkpoint to an inference-only format compatible with ProTrek."
    )
    parser.add_argument("--input", required=True, help="Input checkpoint path.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect the input checkpoint without writing an output file.",
    )
    parser.add_argument(
        "--strip-module-prefix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Strip a leading 'module.' prefix from state_dict keys.",
    )
    parser.add_argument(
        "--model-key",
        default="auto",
        help="Checkpoint key containing the model state_dict, or 'auto'.",
    )
    parser.add_argument(
        "--keep-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep top-level 'config' from the source checkpoint when present.",
    )
    return parser.parse_args()


def checkpoint_keys(checkpoint: Any) -> list[str]:
    if not isinstance(checkpoint, Mapping):
        return []
    return [str(key) for key in checkpoint.keys()]


def is_tensor_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False

    values = list(value.values())
    tensor_count = sum(isinstance(item, torch.Tensor) for item in values)
    return tensor_count > 0 and tensor_count / len(values) >= 0.5


def find_model_state_dict(checkpoint: Any, model_key: str) -> tuple[Mapping[str, Any], str]:
    if model_key != "auto":
        if not isinstance(checkpoint, Mapping):
            raise ValueError("--model-key was provided, but checkpoint is not a dict.")
        if model_key not in checkpoint:
            raise ValueError(f"Model key '{model_key}' was not found in checkpoint.")
        state_dict = checkpoint[model_key]
        if not is_tensor_state_dict(state_dict):
            raise ValueError(f"Checkpoint key '{model_key}' is not a tensor state_dict.")
        return state_dict, model_key

    if isinstance(checkpoint, Mapping):
        for key in CANDIDATE_MODEL_KEYS:
            if key in checkpoint and is_tensor_state_dict(checkpoint[key]):
                return checkpoint[key], key

        if is_tensor_state_dict(checkpoint):
            return checkpoint, "plain_state_dict"

    raise ValueError(
        "Could not find a supported model state_dict. Expected one of: "
        + ", ".join(CANDIDATE_MODEL_KEYS)
        + ", or a plain tensor state_dict."
    )


def strip_module_prefix(key: str) -> str:
    prefix = "module."
    if key.startswith(prefix):
        return key[len(prefix) :]
    return key


def normalize_state_dict(
    state_dict: Mapping[str, Any], strip_prefix: bool
) -> tuple[OrderedDict[str, torch.Tensor], int, int]:
    normalized: OrderedDict[str, torch.Tensor] = OrderedDict()
    stripped_count = 0
    dropped_non_tensor = 0

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            dropped_non_tensor += 1
            continue

        new_key = str(key)
        if strip_prefix:
            stripped_key = strip_module_prefix(new_key)
            if stripped_key != new_key:
                stripped_count += 1
            new_key = stripped_key

        normalized[new_key] = value.detach().cpu()

    if not normalized:
        raise ValueError("The selected state_dict did not contain any tensor values.")

    return normalized, stripped_count, dropped_non_tensor


def tensor_stats(state_dict: Mapping[str, torch.Tensor]) -> dict[str, int]:
    num_tensors = 0
    num_parameters = 0
    tensor_size_bytes = 0

    for value in state_dict.values():
        num_tensors += 1
        num_parameters += value.numel()
        tensor_size_bytes += value.numel() * value.element_size()

    return {
        "num_tensors": num_tensors,
        "num_parameters": num_parameters,
        "tensor_size_bytes": tensor_size_bytes,
    }


def load_checkpoint(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Input checkpoint does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input checkpoint is not a file: {path}")
    return torch.load(path, map_location="cpu")


def build_output_checkpoint(
    checkpoint: Any,
    model_state: OrderedDict[str, torch.Tensor],
    metadata: dict[str, Any],
    keep_config: bool,
) -> dict[str, Any]:
    output = {
        "model": model_state,
        "metadata": metadata,
    }
    if keep_config and isinstance(checkpoint, Mapping) and "config" in checkpoint:
        output["config"] = checkpoint["config"]
    return output


def atomic_torch_save(obj: Any, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output checkpoint already exists: {output_path}. Use --overwrite to replace it."
        )

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=str(output_path.parent),
            delete=False,
        ) as tmp_file:
            tmp_name = tmp_file.name

        torch.save(obj, tmp_name)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output checkpoint already exists: {output_path}. Use --overwrite to replace it."
            )
        os.replace(tmp_name, output_path)
        tmp_name = None

    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def print_summary(metadata: dict[str, Any], output_size_bytes: int | None) -> None:
    print("Checkpoint export summary")
    print(f"  source_path: {metadata['source_path']}")
    print(f"  source_top_level_keys: {metadata['source_top_level_keys']}")
    print(f"  model_key_used: {metadata['model_key_used']}")
    print(f"  num_tensors: {metadata['num_tensors']}")
    print(f"  num_parameters: {metadata['num_parameters']}")
    print(f"  tensor_size_bytes: {metadata['tensor_size_bytes']}")
    print(f"  input_size_bytes: {metadata['input_size_bytes']}")
    print(f"  stripped_module_prefix_count: {metadata['stripped_module_prefix_count']}")
    print(f"  dropped_non_tensor_entries: {metadata['dropped_non_tensor_entries']}")
    if output_size_bytes is not None:
        print(f"  output_size_bytes: {output_size_bytes}")
    else:
        print("  output_size_bytes: inspect-only")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    checkpoint = load_checkpoint(input_path)
    source_keys = checkpoint_keys(checkpoint)
    raw_state_dict, model_key_used = find_model_state_dict(checkpoint, args.model_key)
    model_state, stripped_count, dropped_non_tensor = normalize_state_dict(
        raw_state_dict,
        strip_prefix=args.strip_module_prefix,
    )

    stats = tensor_stats(model_state)
    metadata = {
        "source_path": str(input_path),
        "source_top_level_keys": source_keys,
        "model_key_used": model_key_used,
        "num_tensors": stats["num_tensors"],
        "num_parameters": stats["num_parameters"],
        "tensor_size_bytes": stats["tensor_size_bytes"],
        "input_size_bytes": input_path.stat().st_size,
        "created_by": CREATED_BY,
        "stripped_module_prefix_count": stripped_count,
        "dropped_non_tensor_entries": dropped_non_tensor,
        "strip_module_prefix": bool(args.strip_module_prefix),
    }

    if args.inspect_only:
        print_summary(metadata, output_size_bytes=None)
        print("Inspect-only mode: no output checkpoint was written.")
        return

    output_checkpoint = build_output_checkpoint(
        checkpoint,
        model_state,
        metadata,
        keep_config=args.keep_config,
    )
    atomic_torch_save(output_checkpoint, output_path, overwrite=args.overwrite)
    output_size = output_path.stat().st_size
    print_summary(metadata, output_size_bytes=output_size)
    print(f"Saved inference-only checkpoint: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
