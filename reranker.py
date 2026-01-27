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


# Default batch size for batch prompting (10 is a safe value that balances API call reduction with accuracy)
DEFAULT_BATCH_SIZE = 10

class DocumentReranker:
    """Jetstream-based reranker for search results with batch prompting optimization."""

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
                    "You are a relevance scoring assistant for an educational RAG system. "
                    "You will be given a query and multiple documents. Score each document's relevance "
                    "to the query on a scale from 0 to 10 (10 = highly relevant, 0 = not relevant). "
                    "Respond with a JSON object containing a 'scores' array with one object per document. "
                    "Each object should have 'doc_id' (integer) and 'score' (float 0-10). "
                    "Example response: {\"scores\": [{\"doc_id\": 0, \"score\": 8.5}, {\"doc_id\": 1, \"score\": 3.2}]}"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments to score:\n\n{documents_text}",
            },
        ]

        try:
            # Increase max_tokens based on number of documents (roughly 30 tokens per doc score)
            max_tokens = max(200, len(documents) * 40)
            assessment = self.jetstream_client.chat_completion_json(messages, max_tokens=max_tokens, temperature=0)
        except RuntimeError as exc:
            if self.verbose:
                print(f'  ⚠ Batch reranker error: {exc}')
            # Return zeros for all documents on error
            return [0.0] * len(documents)

        # Parse the batch response
        scores = [0.0] * len(documents)  # Default to 0 for any missing scores

        scores_list = assessment.get('scores', [])
        if isinstance(scores_list, list):
            for item in scores_list:
                if isinstance(item, dict):
                    doc_id = item.get('doc_id')
                    score = item.get('score')
                    if doc_id is not None and score is not None:
                        try:
                            idx = int(doc_id)
                            if 0 <= idx < len(documents):
                                scores[idx] = max(0.0, min(float(score), 10.0))
                        except (TypeError, ValueError):
                            continue

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

            document_text = payload.get('content_preview', '')
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
