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
GENERAL_KNOWLEDGE_DISCLAIMER = (
    "\n\n---\n*Note: This answer is based on general knowledge, not your course materials. "
    "It may not reflect your instructor's specific expectations. Please verify with course resources or your instructor.*"
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
        fallback_behavior: str = "both",
    ) -> Dict[str, Any]:
        system = self._llm_factory(collection_name, enable_guardrails)
        normalized_history = self._normalize_history(history)

        # Relevance check — currently bypassed (always treated as relevant)
        # to allow full fallback/escalation flow. TODO: revisit thresholds.
        is_relevant, relevance_reason = True, "Relevance check bypassed for testing."

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

        documents = system.query_engine.search(
            query=question,
            top_k=3,
            use_reranker=True,
            rerank_candidates=20,
            stage1_top_k=5,
            min_score=4.5,
            verbose=False,
        )
        engine_results = self._format_engine_results(documents)

        if not documents:
            logger.warning(
                "Chat search returned 0 documents for q=%r collection=%s "
                "(all candidates scored below min_score threshold)",
                question,
                collection_name,
            )

        answer = system.query(
            question,
            show_context=False,
            max_length=1024,
            conversation_history=prompt_history,
            precomputed_documents=documents,
            response_language=language,
        )

        human_required = False
        suggested_post: Optional[str] = None
        if not documents:
            use_general = fallback_behavior in ("general_knowledge", "both")
            use_escalation = fallback_behavior in ("escalate", "both")
            if use_general:
                answer += GENERAL_KNOWLEDGE_DISCLAIMER
            if use_escalation:
                human_required = True
                suggested_post = self._generate_escalation_post(
                    system.jetstream_client,
                    question,
                    answer,
                    class_name,
                )

        return {
            "answer": answer,
            "engine_results": engine_results,
            "human_required": human_required,
            "fallback_behavior": fallback_behavior if not documents else None,
            "suggested_post": suggested_post,
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

    def _generate_escalation_post(
        self,
        client: Any,
        question: str,
        ai_answer: str,
        class_name: Optional[str],
    ) -> Optional[str]:
        """Ask the LLM to draft a discussion board post for the student."""
        course_label = class_name or "this course"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are helping a student compose an anonymous discussion board post for their course. "
                    "The AI assistant was unable to find relevant course materials to answer their question. "
                    "Draft a clear, concise post that:\n"
                    "1. States the student's question clearly\n"
                    "2. Mentions that the AI assistant could not find an answer in the course materials\n"
                    "3. Asks the instructor or classmates for help\n"
                    "Keep the tone polite and academic. Do NOT include the AI's answer. "
                    "Do NOT reveal the student's identity. Write in first person as the student. "
                    "Keep it under 150 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Course: {course_label}\n"
                    f"My question: {question}\n\n"
                    "Please draft a discussion board post for me."
                ),
            },
        ]
        try:
            post = client.chat_completion(
                messages,
                max_tokens=300,
                temperature=0.4,
                top_p=0.9,
            )
            return post.strip()
        except Exception as exc:
            logger.warning("Escalation post generation failed: %s", exc)
            # Fall back to a simple formatted version
            return (
                f"Hi everyone,\n\n"
                f"I had a question about {course_label} that the AI assistant couldn't find in the course materials:\n\n"
                f"**{question}**\n\n"
                f"Could an instructor or classmate help me with this? Thank you!"
            )

    def _format_engine_results(self, documents: List[Tuple[Dict[str, Any], float]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for result, score in documents:
            payload = result.get("payload", {}) if isinstance(result, dict) else {}
            file_path = payload.get("file_path", "N/A")

            # Derive a human-readable title from the file path
            title = payload.get("title")
            if not title and file_path and file_path != "N/A":
                import os
                title = os.path.splitext(os.path.basename(file_path))[0].replace("-", " ").replace("_", " ").title()

            # Determine source type from file_type or file extension
            file_type = payload.get("file_type", "N/A")
            source_type = file_type
            if source_type == "N/A" and file_path != "N/A":
                import os
                ext = os.path.splitext(file_path)[1].lower().lstrip(".")
                type_map = {"pdf": "PDF", "txt": "Text", "md": "Markdown", "html": "Page", "docx": "Document"}
                source_type = type_map.get(ext, ext.upper() if ext else "N/A")

            # Extended content preview (up to 200 chars)
            content_preview = payload.get("content_preview", "")
            if isinstance(content_preview, str) and len(content_preview) > 200:
                content_preview = content_preview[:200] + "..."

            formatted.append(
                {
                    "file_path": file_path,
                    "title": title or file_path,
                    "course": payload.get("course", "N/A"),
                    "file_type": file_type,
                    "source_type": source_type,
                    "module_name": payload.get("module_name", None),
                    "skills": payload.get("skills", []),
                    "relevance_score": score,
                    "content_preview": content_preview,
                    "chunk_index": payload.get("chunk_index"),
                    "total_chunks": payload.get("total_chunks"),
                }
            )
        return formatted
