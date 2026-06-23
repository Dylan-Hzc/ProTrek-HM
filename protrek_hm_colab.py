"""Colab-friendly wrapper for ProTrek-HM inference.

The default model is local ProTrek-35M, with protein sequence <-> text
retrieval as the minimal use case. Structure/Foldseek embeddings are optional:
this wrapper accepts Foldseek sequences directly and does not require a
Foldseek binary for the baseline Colab demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class ColabProTrekHM:
    """Lightweight inference wrapper around ProTrekTrimodalModel."""

    def __init__(
        self,
        model_dir: str = "weights/ProTrek_35M",
        checkpoint_path: str | None = None,
        finetuned_checkpoint_path: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
        repr_dim: int = 1024,
        temperature: float = 0.07,
        include_structure_encoder: bool | str = "auto",
        scale_by_temperature: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")

        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
        self.finetuned_checkpoint_path = finetuned_checkpoint_path
        self.device_spec = device
        self.device = self._resolve_device()
        self.batch_size = batch_size
        self.repr_dim = repr_dim
        self.temperature = temperature
        self.include_structure_encoder = include_structure_encoder
        self.scale_by_temperature = scale_by_temperature

        self.model: Any | None = None
        self.loaded = False
        self.paths: dict[str, Any] | None = None

    def _resolve_device(self) -> torch.device:
        requested = str(self.device_spec).lower()
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda" or requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is not available. Use device='auto' or device='cpu'."
                )
            return torch.device(requested)
        raise ValueError(f"Unsupported device value: {self.device_spec!r}")

    @staticmethod
    def _first_existing_dir(model_dir: Path, exact_name: str, patterns: list[str]) -> Path | None:
        exact_path = model_dir / exact_name
        if exact_path.is_dir():
            return exact_path

        for pattern in patterns:
            matches = sorted(path for path in model_dir.glob(pattern) if path.is_dir())
            if matches:
                return matches[0]
        return None

    def _resolve_structure_mode(self, structure_config: Path | None) -> tuple[bool, Path | None]:
        mode = self.include_structure_encoder
        if mode == "auto":
            return structure_config is not None, structure_config
        if mode is True:
            if structure_config is None:
                raise FileNotFoundError(
                    "include_structure_encoder=True but no local foldseek config directory was found."
                )
            return True, structure_config
        if mode is False:
            return False, None
        raise ValueError("include_structure_encoder must be True, False, or 'auto'.")

    def resolve_paths(self) -> dict[str, Any]:
        """Resolve local model paths without loading model weights."""
        model_dir = Path(self.model_dir).expanduser()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

        checkpoint_path = (
            Path(self.checkpoint_path).expanduser()
            if self.checkpoint_path is not None
            else model_dir / "ProTrek_35M.pt"
        )
        active_checkpoint_path = (
            Path(self.finetuned_checkpoint_path).expanduser()
            if self.finetuned_checkpoint_path is not None
            else checkpoint_path
        )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found: {checkpoint_path}")
        if not active_checkpoint_path.is_file():
            raise FileNotFoundError(f"Active checkpoint not found: {active_checkpoint_path}")

        protein_config = self._first_existing_dir(
            model_dir,
            "esm2_t12_35M_UR50D",
            ["esm2*"],
        )
        text_config = self._first_existing_dir(
            model_dir,
            "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
            ["BiomedNLP*", "*BERT*"],
        )
        structure_config = self._first_existing_dir(
            model_dir,
            "foldseek_t12_35M",
            ["foldseek*"],
        )

        if protein_config is None:
            raise FileNotFoundError(
                f"Protein config directory not found under {model_dir}. Expected esm2_t12_35M_UR50D or esm2*."
            )
        if text_config is None:
            raise FileNotFoundError(
                f"Text config directory not found under {model_dir}. Expected PubMedBERT/BERT local config."
            )

        include_structure, structure_config = self._resolve_structure_mode(structure_config)

        self.paths = {
            "model_dir": model_dir,
            "checkpoint_path": checkpoint_path,
            "active_checkpoint_path": active_checkpoint_path,
            "protein_config": protein_config,
            "text_config": text_config,
            "structure_config": structure_config,
            "include_structure_encoder": include_structure,
        }
        return dict(self.paths)

    def load(self) -> "ColabProTrekHM":
        """Instantiate ProTrekTrimodalModel and load the selected checkpoint."""
        from model.ProTrek.protrek_trimodal_model import ProTrekTrimodalModel

        paths = self.resolve_paths()
        model_config = {
            "protein_config": str(paths["protein_config"]),
            "text_config": str(paths["text_config"]),
            "structure_config": (
                str(paths["structure_config"]) if paths["structure_config"] is not None else None
            ),
            "repr_dim": self.repr_dim,
            "temperature": self.temperature,
            "load_protein_pretrained": False,
            "load_text_pretrained": False,
            "use_mlm_loss": False,
            "use_zlpr_loss": False,
            "use_saprot": False,
            "gradient_checkpointing": False,
            "from_checkpoint": str(paths["active_checkpoint_path"]),
        }

        model = ProTrekTrimodalModel(**model_config)
        model.eval()
        model.to(self.device)

        self.model = model
        self.loaded = True
        return self

    def _ensure_loaded(self) -> Any:
        if self.model is None or not self.loaded:
            raise RuntimeError("Model is not loaded. Call load() before embedding inputs.")
        return self.model

    @staticmethod
    def _validate_string_list(name: str, values: list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(values, str) or not isinstance(values, (list, tuple)):
            raise TypeError(f"{name} must be a non-empty list or tuple of strings.")
        if not values:
            raise ValueError(f"{name} must not be empty.")
        if not all(isinstance(item, str) for item in values):
            raise TypeError(f"Every item in {name} must be a string.")
        return list(values)

    @staticmethod
    def _to_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected a torch.Tensor.")
        return tensor.detach().cpu()

    @staticmethod
    def _maybe_numpy(tensor: torch.Tensor, return_numpy: bool):
        return tensor.numpy() if return_numpy else tensor

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        return batch_size

    def embed_sequences(
        self,
        sequences: list[str],
        batch_size: int | None = None,
        return_numpy: bool = False,
    ):
        model = self._ensure_loaded()
        sequences = self._validate_string_list("sequences", sequences)
        active_batch_size = self._validate_batch_size(batch_size or self.batch_size)
        with torch.no_grad():
            embeddings = model.get_protein_repr(sequences, batch_size=active_batch_size)
        embeddings = self._to_cpu_tensor(embeddings)
        return self._maybe_numpy(embeddings, return_numpy)

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
        return_numpy: bool = False,
    ):
        model = self._ensure_loaded()
        texts = self._validate_string_list("texts", texts)
        active_batch_size = self._validate_batch_size(batch_size or self.batch_size)
        with torch.no_grad():
            embeddings = model.get_text_repr(texts, batch_size=active_batch_size)
        embeddings = self._to_cpu_tensor(embeddings)
        return self._maybe_numpy(embeddings, return_numpy)

    def embed_structures(
        self,
        foldseek_sequences: list[str],
        batch_size: int | None = None,
        return_numpy: bool = False,
    ):
        model = self._ensure_loaded()
        foldseek_sequences = self._validate_string_list(
            "foldseek_sequences", foldseek_sequences
        )
        if not hasattr(model, "structure_encoder"):
            raise RuntimeError(
                "Structure encoder is not available. Initialize with include_structure_encoder=True "
                "and provide a local foldseek config directory."
            )
        active_batch_size = self._validate_batch_size(batch_size or self.batch_size)
        with torch.no_grad():
            embeddings = model.get_structure_repr(
                foldseek_sequences,
                batch_size=active_batch_size,
            )
        embeddings = self._to_cpu_tensor(embeddings)
        return self._maybe_numpy(embeddings, return_numpy)

    @staticmethod
    def _as_2d_cpu_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
        tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a 2D tensor or array.")
        return tensor.float()

    def _temperature_scale(self) -> torch.Tensor:
        model = self._ensure_loaded()
        temperature = model.temperature.detach().cpu()
        return temperature.float()

    def similarity_matrix(
        self,
        sequences: list[str] | None = None,
        texts: list[str] | None = None,
        sequence_embeddings: torch.Tensor | None = None,
        text_embeddings: torch.Tensor | None = None,
        return_numpy: bool = False,
    ):
        if sequences is not None and sequence_embeddings is not None:
            raise ValueError("Provide either sequences or sequence_embeddings, not both.")
        if texts is not None and text_embeddings is not None:
            raise ValueError("Provide either texts or text_embeddings, not both.")

        if sequence_embeddings is None:
            if sequences is None:
                raise ValueError("Provide sequences or sequence_embeddings.")
            sequence_embeddings = self.embed_sequences(sequences)
        if text_embeddings is None:
            if texts is None:
                raise ValueError("Provide texts or text_embeddings.")
            text_embeddings = self.embed_texts(texts)

        seq_emb = self._as_2d_cpu_tensor("sequence_embeddings", sequence_embeddings)
        txt_emb = self._as_2d_cpu_tensor("text_embeddings", text_embeddings)
        scores = seq_emb @ txt_emb.T

        if self.scale_by_temperature:
            scores = scores / self._temperature_scale()

        return self._maybe_numpy(scores, return_numpy)

    def score_pairs(
        self,
        sequences: list[str],
        texts: list[str],
        return_numpy: bool = False,
    ):
        sequences = self._validate_string_list("sequences", sequences)
        texts = self._validate_string_list("texts", texts)
        if len(sequences) != len(texts):
            raise ValueError("sequences and texts must have the same length.")

        seq_emb = self.embed_sequences(sequences)
        txt_emb = self.embed_texts(texts)
        scores = (seq_emb * txt_emb).sum(dim=1)

        if self.scale_by_temperature:
            scores = scores / self._temperature_scale()

        return self._maybe_numpy(scores, return_numpy)

    def describe(self) -> dict[str, Any]:
        """Return resolved local paths and runtime settings without loading weights."""
        paths = self.resolve_paths()
        return {
            "model_dir": str(paths["model_dir"]),
            "checkpoint_path": str(paths["checkpoint_path"]),
            "active_checkpoint_path": str(paths["active_checkpoint_path"]),
            "protein_config": str(paths["protein_config"]),
            "text_config": str(paths["text_config"]),
            "structure_config": (
                str(paths["structure_config"]) if paths["structure_config"] is not None else None
            ),
            "include_structure_encoder": paths["include_structure_encoder"],
            "device": str(self.device),
            "batch_size": self.batch_size,
            "loaded": self.loaded,
            "scale_by_temperature": self.scale_by_temperature,
        }
