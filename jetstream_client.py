"""Utility helpers for interacting with the Jetstream inference service."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Union

from openai import OpenAI

DEFAULT_BASE_URL = "https://llm.jetstream-cloud.org/llama-4-scout/v1"
DEFAULT_MODEL = "llama-4-scout"
DEFAULT_EMBED_MODEL = "Alibaba-NLP/gte-large-en-v1.5"

JsonType = Union[Dict[str, Any], List[Any]]


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


class JetstreamInferenceClient:
    """Thin wrapper around the Jetstream OpenAI-compatible API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = (base_url or _get_env("JETSTREAM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or _get_env("JETSTREAM_MODEL", DEFAULT_MODEL)
        key = api_key if api_key is not None else _get_env("JETSTREAM_API_KEY")
        # The Jetstream direct endpoints ignore the token, but the OpenAI client expects something non-empty.
        self.api_key = key if key else "EMPTY"
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=timeout)
        embed_base_url = _get_env("JETSTREAM_EMBED_BASE_URL")
        if embed_base_url:
            self.embed_client = OpenAI(base_url=embed_base_url.rstrip("/"), api_key=self.api_key, timeout=timeout)
        else:
            self.embed_client = self.client
        self.embed_model = _get_env("JETSTREAM_EMBED_MODEL", DEFAULT_EMBED_MODEL)

    def chat_completion(self, messages: Sequence[Dict[str, Any]], **kwargs: Any) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - network/transport errors
            raise RuntimeError(f"Jetstream inference request failed: {exc}") from exc

        if not response.choices:
            raise RuntimeError("Jetstream inference request returned no choices")

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Jetstream inference request returned empty content")

        return content.strip()

    def chat_completion_json(self, messages: Sequence[Dict[str, Any]], **kwargs: Any) -> JsonType:
        text = self.chat_completion(messages, **kwargs)
        payload = _extract_json_payload(text)
        if payload is None:
            raise RuntimeError("Jetstream response did not contain valid JSON payload")
        return payload

    def embed(
        self,
        inputs: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> List[List[float]]:
        if isinstance(inputs, str):
            batched_inputs = [inputs]
        else:
            batched_inputs = list(inputs)

        if not batched_inputs:
            return []

        target_model = model or self.embed_model or DEFAULT_EMBED_MODEL
        client = self.embed_client

        try:
            response = client.embeddings.create(
                model=target_model,
                input=batched_inputs,
                **kwargs,
            )
        except Exception as exc:  # pragma: no cover - network/transport errors
            raise RuntimeError(f"Jetstream embedding request failed: {exc}") from exc

        data = getattr(response, "data", None)
        if not data:
            raise RuntimeError("Jetstream embedding request returned no data")

        # Ensure embeddings are returned in input order
        data_sorted = sorted(data, key=lambda item: getattr(item, "index", 0))
        return [list(getattr(item, "embedding", [])) for item in data_sorted]


def _extract_json_payload(text: str) -> Optional[JsonType]:
    """Extract the first JSON object or array from a block of text."""
    text = text.strip()
    if not text:
        return None

    # Fast path: try to parse the entire response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    first_brace = text.find("{")
    first_bracket = text.find("[")

    start = None
    end_char = None

    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start = first_brace
        end_char = "}"
    elif first_bracket != -1:
        start = first_bracket
        end_char = "]"

    if start is None or end_char is None:
        return None

    end = text.rfind(end_char)
    if end == -1 or end <= start:
        return None

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None
