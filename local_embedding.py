"""Utilities for loading and using local embedding models with query/passage prefix support.

Supported prefix conventions (auto-detected from model name):
  - E5 family (intfloat/e5-*): "query: " / "passage: "
  - Other models: no prefix (override via constructor args if needed)
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Sequence, Union, cast

import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "Alibaba-NLP/gte-large-en-v1.5"


def _detect_prefixes(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) for known model families."""
    name_lower = model_name.lower()
    # E5 family requires "query: " / "passage: " prefixes.
    # https://huggingface.co/intfloat/e5-large-v2#faq
    if "/e5-" in name_lower or "-e5-" in name_lower or name_lower.startswith("e5-"):
        return "query: ", "passage: "
    return "", ""


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
    """Thin wrapper around the SentenceTransformer embedder with prefix support.

    Models like E5 require specific prefixes on queries vs passages for optimal
    accuracy.  This client auto-detects the convention from the model name and
    exposes ``embed_query`` / ``embed_passages`` helpers that prepend them
    automatically.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        prefer_gpu: bool | None = None,
        batch_size: int | None = None,
        normalize_embeddings: bool | None = None,
        trust_remote_code: bool | None = None,
        query_prefix: str | None = None,
        passage_prefix: str | None = None,
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

        detected_q, detected_p = _detect_prefixes(self.model_name)
        self.query_prefix = query_prefix if query_prefix is not None else detected_q
        self.passage_prefix = passage_prefix if passage_prefix is not None else detected_p

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    def embed(self, texts: Union[str, Sequence[str]], *, prefix: str = "") -> List[List[float]]:
        """Embed one or more texts, optionally prepending *prefix* to each."""
        if isinstance(texts, str):
            batch: List[str] = [texts]
        else:
            batch = list(texts)

        if not batch:
            return []

        if prefix:
            batch = [prefix + t for t in batch]

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

    # ------------------------------------------------------------------
    # Convenience methods with automatic prefix handling
    # ------------------------------------------------------------------

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string with the model's query prefix."""
        results = self.embed(text, prefix=self.query_prefix)
        if not results:
            raise RuntimeError("Embedding returned no vector for query")
        return results[0]

    def embed_passages(self, texts: Union[str, Sequence[str]]) -> List[List[float]]:
        """Embed one or more passages/documents with the model's passage prefix."""
        return self.embed(texts, prefix=self.passage_prefix)


__all__ = ["LocalEmbeddingClient", "DEFAULT_MODEL"]
