"""Utilities for loading and using the local Alibaba-NLP/gte-large-en-v1.5 embedder."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Sequence, Union, cast

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@lru_cache(maxsize=4)
def _load_sentence_transformer(model_name: str, device: str, trust_remote_code: bool) -> SentenceTransformer:
    """Cache SentenceTransformer instances per (model, device, trust flag)."""
    return SentenceTransformer(model_name, device=device, trust_remote_code=trust_remote_code)


class LocalEmbeddingClient:
    """Thin wrapper around the SentenceTransformer embedder."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        prefer_gpu: bool | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
        trust_remote_code: bool | None = None,
    ) -> None:
        self.model_name = model_name or _get_env("LOCAL_EMBED_MODEL", DEFAULT_MODEL)

        if prefer_gpu is None:
            prefer_gpu = _bool_env("LOCAL_EMBED_USE_GPU", True)

        if prefer_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if batch_size is None:
            batch_env = _get_env("LOCAL_EMBED_BATCH_SIZE", "8")
            try:
                batch_size = int(batch_env)
            except (TypeError, ValueError):
                batch_size = 8
        self.batch_size = max(1, batch_size)

        if normalize_embeddings is None:
            normalize_embeddings = _bool_env("LOCAL_EMBED_NORMALIZE", True)
        self.normalize_embeddings = normalize_embeddings

        if trust_remote_code is None:
            trust_remote_code = _bool_env("LOCAL_EMBED_TRUST_REMOTE_CODE", True)
        self.trust_remote_code = trust_remote_code

        self._model = _load_sentence_transformer(self.model_name, self.device, self.trust_remote_code)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: Union[str, Sequence[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            batch: List[str] = [texts]
        else:
            batch = list(texts)

        if not batch:
            return []

        embeddings = self._model.encode(
            batch,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )

        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()  # type: ignore[return-value]

        # Fallback in case encode returns an iterable of tensors
        return [
            vector.detach().cpu().numpy().tolist() if hasattr(vector, "detach") else list(vector)  # type: ignore[misc]
            for vector in cast(Iterable, embeddings)
        ]


__all__ = ["LocalEmbeddingClient", "DEFAULT_MODEL"]
