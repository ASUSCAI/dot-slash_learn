'''
Reranker Module for RAG Results

Uses Jetstream-hosted Llama 4 Scout to rerank documents retrieved from vector search.
This provides more accurate relevance scoring by considering the full context
of both the query and document together without requiring local transformer models.

This is meant to be imported and used as a module in other scripts.

Usage:
    from reranker import DocumentReranker

    reranker = DocumentReranker()

    # results should be a list of dicts with 'payload' key
    reranked_results = reranker.rerank(query, results)

    # Returns list of tuples: (result, rerank_score), sorted by score descending
    for result, score in reranked_results:
        print(f"Score: {score:.4f}")
        print(f"File: {result['payload']['file_path']}")
'''

import logging
from typing import List, Dict, Any, Tuple, Optional

from jetstream_client import JetstreamInferenceClient, _get_env

logger = logging.getLogger(__name__)


# Default batch size for batch prompting (10 is a safe value that balances API call reduction with accuracy)
DEFAULT_BATCH_SIZE = 10

class DocumentReranker:
    """Jetstream-based reranker for search results."""

    def __init__(
        self,
        *,
        verbose: bool = True,
        jetstream_client: Optional[JetstreamInferenceClient] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.verbose = verbose
        self.batch_size = batch_size
        if jetstream_client is not None:
            self.jetstream_client = jetstream_client
        else:
            self.jetstream_client = JetstreamInferenceClient(
                base_url=_get_env('JETSTREAM_RERANK_BASE_URL') or _get_env('JETSTREAM_BASE_URL'),
                model=_get_env('JETSTREAM_RERANK_MODEL') or _get_env('JETSTREAM_MODEL') or 'llama-4-scout',
                api_key=_get_env('JETSTREAM_RERANK_API_KEY') or _get_env('JETSTREAM_API_KEY'),
                timeout=int(_get_env('JETSTREAM_RERANK_TIMEOUT', _get_env('JETSTREAM_TIMEOUT') or '120')),
            )

        if self.verbose:
            print(f'Reranker mode: Jetstream batch scoring (batch_size={self.batch_size})')

    def compute_rerank_score(self, query: str, document: str) -> float:
        """
        Compute reranking score for a single query-document pair.
        Higher scores indicate better relevance.

        NOTE: For efficiency, prefer using compute_batch_scores() which processes
        multiple documents in a single API call.

        Args:
            query: The search query
            document: The document text to score

        Returns:
            Relevance score (higher = more relevant)
        """
        scores = self.compute_batch_scores(query, [document])
        return scores[0] if scores else 0.0

    def compute_batch_scores(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute reranking scores for multiple documents in a single API call.
        References:
        - https://aclanthology.org/2023.emnlp-industry.74 (Batch Prompting paper)
        - https://latitude-blog.ghost.io/blog/scaling-llms-with-batch-processing-ultimate-guide/

        Args:
            query: The search query
            documents: List of document texts to score

        Returns:
            List of relevance scores (higher = more relevant), in same order as input
        """
        if not documents:
            return []

        # Truncate documents to reasonable length for the prompt
        doc_excerpts = []
        for doc in documents:
            excerpt = doc[:1500] + '...' if len(doc) > 1500 else doc
            doc_excerpts.append(excerpt)

        # Build the batch prompt with numbered documents
        docs_formatted = []
        for i, excerpt in enumerate(doc_excerpts):
            docs_formatted.append(f"[DOCUMENT {i}]\n{excerpt}\n[/DOCUMENT {i}]")

        documents_text = "\n\n".join(docs_formatted)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a relevance scoring assistant. "
                    "Score each document's relevance to the query from 0 to 10. "
                    "Reply with ONLY a JSON object — no explanation, no markdown, no reasoning. "
                    "Format: {\"scores\": [{\"doc_id\": 0, \"score\": 8.5}, {\"doc_id\": 1, \"score\": 3.2}]}"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments to score:\n\n{documents_text}",
            },
        ]

        # Step 1: Get raw text from LLM so we can log it before JSON parsing
        try:
            max_tokens = max(200, len(documents) * 40)
            raw_text = self.jetstream_client.chat_completion(messages, max_tokens=max_tokens, temperature=0)
        except RuntimeError as exc:
            logger.error("Batch reranker LLM call failed for %d docs: %s", len(documents), exc)
            return [0.0] * len(documents)

        logger.info("Reranker raw LLM response (%d docs): %s", len(documents), raw_text[:500])

        # Step 2: Parse JSON from the raw text
        from jetstream_client import _extract_json_payload
        assessment = _extract_json_payload(raw_text)

        if assessment is None:
            logger.error(
                "Reranker JSON parse failed for %d docs. Full raw text:\n%s",
                len(documents), raw_text,
            )
            return [0.0] * len(documents)

        logger.info("Reranker parsed JSON type=%s keys=%s",
                     type(assessment).__name__,
                     list(assessment.keys()) if isinstance(assessment, dict) else f"list[{len(assessment)}]")

        scores = [0.0] * len(documents)

        # Normalize response shape: the LLM may return
        #   {"scores": [...]}, [{"doc_id":0,"score":8.5},...], or [8.5, 3.2, ...]
        if isinstance(assessment, list):
            scores_list = assessment
        elif isinstance(assessment, dict):
            scores_list = assessment.get('scores', assessment.get('results', []))
        else:
            scores_list = []

        if not isinstance(scores_list, list) or not scores_list:
            logger.warning(
                "Reranker could not locate scores array in response for %d docs. Parsed payload: %s",
                len(documents), assessment,
            )
            return scores

        first = scores_list[0]
        logger.info("Reranker scores_list format: len=%d first_item_type=%s first_item=%s",
                     len(scores_list), type(first).__name__, str(first)[:200])

        # Flat-number array: [8.5, 3.2, ...]
        if not isinstance(first, dict):
            for idx, val in enumerate(scores_list):
                if idx < len(documents):
                    try:
                        scores[idx] = max(0.0, min(float(val), 10.0))
                    except (TypeError, ValueError):
                        continue
            logger.info("Reranker flat-array scores: %s", scores)
            return scores

        # Object array: [{"doc_id": 0, "score": 8.5}, ...]
        logger.info("Reranker object-array keys in first item: %s", list(first.keys()))
        for item in scores_list:
            if not isinstance(item, dict):
                continue
            doc_id = next((item[k] for k in ('doc_id', 'id', 'document_id') if item.get(k) is not None), None)
            score = next((item[k] for k in ('score', 'relevance', 'relevance_score') if item.get(k) is not None), None)
            if doc_id is not None and score is not None:
                try:
                    idx = int(doc_id)
                    if 0 <= idx < len(documents):
                        scores[idx] = max(0.0, min(float(score), 10.0))
                except (TypeError, ValueError):
                    continue
            else:
                logger.warning("Reranker could not extract doc_id/score from item: %s", item)

        logger.info("Reranker final scores: %s", scores)

        if all(s == 0.0 for s in scores):
            logger.warning(
                "Reranker returned all-zero scores for %d docs. Full parsed response: %s",
                len(documents), assessment,
            )

        return scores

    def rerank(self, query: str, results: List[Dict[str, Any]], min_score: float = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rerank search results using the reranker model.

        Args:
            query: The search query
            results: List of search results. Each result should be a dict with either:
                     - 'payload' key containing document metadata, OR
                     - document metadata directly in the dict
            min_score: Minimum reranker score threshold. Documents below this score are filtered out.

        Returns:
            List of tuples: (result, rerank_score), sorted by rerank_score descending
        """
        if not results:
            return []

        num_batches = (len(results) + self.batch_size - 1) // self.batch_size
        if self.verbose:
            print(f'Reranking {len(results)} results in {num_batches} batch(es) of up to {self.batch_size}...')

        # Build document contexts for all results
        doc_contexts = []
        for result in results:
            # Handle both formats: result might have 'payload' key or be the payload itself
            if 'payload' in result:
                payload = result['payload']
            else:
                payload = result

            # Build document context from metadata and content
            doc_context = f"Course: {payload.get('course', '')}\n"
            doc_context += f"Type: {payload.get('file_type', '')}\n"

            if payload.get('title'):
                doc_context += f"Title: {payload.get('title', '')}\n"
            if payload.get('description'):
                doc_context += f"Description: {payload.get('description', '')}\n"

            document_text = payload.get('full_content', '') or payload.get('content_preview', '')
            doc_context += f"\nContent:\n{document_text}"

            doc_contexts.append(doc_context)

        # Process in batches
        all_scores = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(results))
            batch_contexts = doc_contexts[start_idx:end_idx]

            if self.verbose:
                print(f'  Batch {batch_idx + 1}/{num_batches}: scoring documents {start_idx + 1}-{end_idx}...')

            batch_scores = self.compute_batch_scores(query, batch_contexts)
            all_scores.extend(batch_scores)

            if self.verbose:
                # Show individual scores for this batch
                for i, score in enumerate(batch_scores):
                    print(f'    [{start_idx + i + 1}/{len(results)}] Score: {score:.4f}')

        # Combine results with scores
        reranked = list(zip(results, all_scores))

        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum score if specified
        if min_score is not None:
            original_count = len(reranked)
            filtered_out = [(r, s) for r, s in reranked if s < min_score]
            reranked = [(r, s) for r, s in reranked if s >= min_score]
            if filtered_out:
                dropped = original_count - len(reranked)
                scores_summary = ", ".join(
                    f"{r.get('payload', r).get('file_path', '?').split('/')[-1]}={s:.1f}"
                    for r, s in filtered_out[:5]
                )
                logger.info(
                    "Reranker filtered %d/%d docs below min_score=%.1f (top dropped: %s)",
                    dropped, original_count, min_score, scores_summary,
                )
                if self.verbose:
                    print(f'  Filtered out {dropped} documents below threshold {min_score:.1f}')

        if self.verbose:
            print()

        return reranked

    def offload_to_cpu(self):
        """Move the reranker model to CPU to free VRAM"""
        if self.verbose:
            print('Remote reranker is stateless; offload_to_cpu is a no-op')
