"""High-level chat orchestration built on dot-slash learn primitives."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

try:  # When running as package
    from .llm import LLMQuerySystem
except ImportError:  # When module is executed from repo root
    from llm import LLMQuerySystem

logger = logging.getLogger(__name__)

HUMAN_ESCALATION_MESSAGE = (
    "I couldn't locate supporting course materials. Please contact your instructor or TA for additional guidance."
)
IRRELEVANT_MESSAGE = (
    "I can only help with questions about this course. Please ask about course topics, assignments, or materials."
)


class ChatService:
    """Wraps LLMQuerySystem to provide a conversational endpoint."""

    def __init__(
        self,
        llm_factory: Callable[[str, bool], LLMQuerySystem],
        *,
        summary_threshold: int = 6,
    ) -> None:
        self._llm_factory = llm_factory
        self._summary_threshold = max(3, summary_threshold)

    def handle_chat(
        self,
        *,
        question: str,
        history: Optional[List[Dict[str, Any]]],
        class_name: Optional[str],
        collection_name: str,
        language: Optional[str] = "en",
        enable_guardrails: bool = True,
        summarize_history: bool = True,
    ) -> Dict[str, Any]:
        system = self._llm_factory(collection_name, enable_guardrails)
        normalized_history = self._normalize_history(history)

        is_relevant, relevance_reason = self._check_relevance(
            system.jetstream_client,
            question,
            class_name,
        )

        if not is_relevant:
            return {
                "answer": IRRELEVANT_MESSAGE,
                "engine_results": [],
                "human_required": False,
                "summary": None,
                "relevance": {
                    "is_relevant": False,
                    "explanation": relevance_reason,
                },
            }

        summary_text: Optional[str] = None
        prompt_history = normalized_history

        if summarize_history and len(normalized_history) >= self._summary_threshold:
            summary_text = self._summarize_history(system.jetstream_client, normalized_history)
            if summary_text:
                prompt_history = [
                    {
                        "role": "system",
                        "content": f"Conversation summary: {summary_text}",
                    }
                ]

        # Use a lightweight reranker (small candidate set) to limit LLM calls.
        documents = system.query_engine.search(
            query=question,
            top_k=3,
            use_reranker=True,
            rerank_candidates=12,
            stage1_top_k=5,
            min_score=7.0,
            verbose=False,
        )
        engine_results = self._format_engine_results(documents)

        if not documents:
            return {
                "answer": HUMAN_ESCALATION_MESSAGE,
                "engine_results": engine_results,
                "human_required": True,
                "summary": summary_text,
                "relevance": {
                    "is_relevant": True,
                    "explanation": relevance_reason,
                },
            }

        answer = system.query(
            question,
            show_context=False,
            max_length=1024,
            conversation_history=prompt_history,
            precomputed_documents=documents,
            response_language=language,
        )

        return {
            "answer": answer,
            "engine_results": engine_results,
            "human_required": False,
            "summary": summary_text,
            "relevance": {
                "is_relevant": True,
                "explanation": relevance_reason,
            },
        }

    def _normalize_history(self, history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
        if not history:
            return []
        normalized: List[Dict[str, str]] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            role = (entry.get("role") or "user").strip().lower()
            if role not in {"user", "assistant", "system"}:
                role = "user"
            content = (entry.get("content") or "").strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        # Keep only the most recent turns to limit prompt size
        return normalized[-12:]

    def _check_relevance(self, client, question: str, class_name: Optional[str]) -> Tuple[bool, str]:
        course_label = class_name or "this course"
        messages = [
            {
                "role": "system",
                "content": (
                    "You determine whether a student's question is related to the provided course description. "
                    "Respond with JSON using keys 'relevant' (true/false) and 'explanation'."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Course description: {course_label}\n"
                    f"Question: {question}\n"
                    "Is the question relevant?"
                ),
            },
        ]
        try:
            payload = client.chat_completion_json(
                messages,
                max_tokens=150,
                temperature=0,
            )
            return bool(payload.get("relevant", True)), str(payload.get("explanation", ""))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Question relevance check failed: %s", exc)
            return True, "Relevance check unavailable."

    def _summarize_history(self, client, history: List[Dict[str, str]]) -> Optional[str]:
        if not history:
            return None
        transcript = "\n".join(f"{turn['role'].title()}: {turn['content']}" for turn in history)
        messages = [
            {
                "role": "system",
                "content": "Summarize the following student and assistant dialogue in 2-3 sentences for context.",
            },
            {"role": "user", "content": transcript},
        ]
        try:
            summary = client.chat_completion(
                messages,
                max_tokens=200,
                temperature=0.3,
                top_p=0.9,
            )
            return summary.strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Conversation summarization failed: %s", exc)
            return None

    def _format_engine_results(self, documents: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for result, score in documents:
            payload = result.get("payload", {}) if isinstance(result, dict) else {}
            formatted.append(
                {
                    "file_path": payload.get("file_path", "N/A"),
                    "course": payload.get("course", "N/A"),
                    "file_type": payload.get("file_type", "N/A"),
                    "skills": payload.get("skills", []),
                    "relevance_score": score,
                    "content_preview": payload.get("content_preview", ""),
                    "chunk_index": payload.get("chunk_index"),
                    "total_chunks": payload.get("total_chunks"),
                }
            )
        return formatted
