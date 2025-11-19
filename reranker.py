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

from typing import List, Dict, Any, Tuple, Optional

from jetstream_client import JetstreamInferenceClient, _get_env


class DocumentReranker:
    """Jetstream-based reranker for search results."""

    def __init__(
        self,
        *,
        verbose: bool = True,
        jetstream_client: Optional[JetstreamInferenceClient] = None,
    ) -> None:
        self.verbose = verbose
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
            print('Reranker mode: Jetstream remote scoring')

    def compute_rerank_score(self, query: str, document: str) -> float:
        """
        Compute reranking score for a query-document pair.
        Higher scores indicate better relevance.

        Args:
            query: The search query
            document: The document text to score

        Returns:
            Relevance score (higher = more relevant)
        """
        return self._compute_remote_score(query, document)

    def _compute_remote_score(self, query: str, document: str) -> float:
        """Use Jetstream LLM to estimate a relevance score."""
        doc_excerpt = document
        if len(doc_excerpt) > 2000:
            doc_excerpt = doc_excerpt[:2000] + '...'

        messages = [
            {
                "role": "system",
                "content": (
                    "You score how relevant a document chunk is to a user query for an educational RAG system. "
                    "Respond with JSON containing: score (float from 0 to 10) and rationale (string)."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocument:\n{doc_excerpt}",
            },
        ]

        try:
            assessment = self.jetstream_client.chat_completion_json(messages, max_tokens=200, temperature=0)
        except RuntimeError as exc:
            if self.verbose:
                print(f'  ⚠ Remote reranker error: {exc}')
            return 0.0

        score = assessment.get('score')
        try:
            return max(0.0, min(float(score), 10.0))
        except (TypeError, ValueError):
            return 0.0

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
        if self.verbose:
            print(f'Reranking {len(results)} results...')

        reranked = []

        for idx, result in enumerate(results):
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

            document_text = payload.get('content_preview', '')
            doc_context += f"\nContent:\n{document_text}"

            # Compute reranking score
            rerank_score = self.compute_rerank_score(query, doc_context)

            reranked.append((result, rerank_score))

            if self.verbose:
                print(f'  [{idx+1}/{len(results)}] Score: {rerank_score:.4f}')

        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Filter by minimum score if specified
        if min_score is not None:
            original_count = len(reranked)
            reranked = [(result, score) for result, score in reranked if score >= min_score]
            if self.verbose and original_count > len(reranked):
                print(f'  Filtered out {original_count - len(reranked)} documents below threshold {min_score:.1f}')

        if self.verbose:
            print()

        return reranked

    def offload_to_cpu(self):
        """Move the reranker model to CPU to free VRAM"""
        if self.verbose:
            print('Remote reranker is stateless; offload_to_cpu is a no-op')
