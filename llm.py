'''
LLM Query System with RAG

Main entry point for querying course materials with LLM assistance.
Performs two-stage reranking RAG and delegates generation to Jetstream inference service models.

Usage:
    python llm.py "What is dynamic programming?"
    python llm.py "Explain Newton's laws" --collection physics_materials
'''

import argparse
import os
from typing import List, Dict, Any, Tuple, Optional

from guardrails import SafetyGuardrails
from jetstream_client import JetstreamInferenceClient, _get_env
from query import QueryEngine


class LLMQuerySystem:

    def __init__(
        self,
        collection_name: str = 'cs_materials',
        qdrant_host: str = 'localhost',
        qdrant_port: int = 6333,
        enable_guardrails: bool = True,
        rewrite_base_url: Optional[str] = None,
        rewrite_model: Optional[str] = None,
        answer_base_url: Optional[str] = None,
        answer_model: Optional[str] = None,
        api_key: Optional[str] = None,
        request_timeout: int = 120,
    ):
        """
        Initialize the LLM query system with RAG and Jetstream inference service.

        Args:
            collection_name: Qdrant collection to search
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port
            enable_guardrails: Whether to enable safety guardrails
            rewrite_base_url: Base URL for the Jetstream model used in query rewriting
            rewrite_model: Model ID for query rewriting
            answer_base_url: Base URL for the Jetstream model used in answer generation
            answer_model: Model ID for answer generation
            api_key: API token if connecting via the Open WebUI proxy (optional)
            request_timeout: Timeout (seconds) for Jetstream API requests
        """
        base_url = (
            answer_base_url
            or rewrite_base_url
            or _get_env('JETSTREAM_BASE_URL')
            or 'https://llm.jetstream-cloud.org/llama-4-scout/v1'
        )
        model = (
            answer_model
            or rewrite_model
            or _get_env('JETSTREAM_MODEL')
            or 'llama-4-scout'
        )
        client_api_key = api_key or _get_env('JETSTREAM_API_KEY')

        print('\n' + '='*80)
        print('CONFIGURING JETSTREAM INFERENCE CLIENT')
        print('='*80)
        self.jetstream_client = JetstreamInferenceClient(
            base_url=base_url,
            model=model,
            api_key=client_api_key,
            timeout=request_timeout,
        )
        print(f'Jetstream model: {self.jetstream_client.model} @ {self.jetstream_client.base_url}\n')

        # Initialize RAG query engine
        print('='*80)
        print('INITIALIZING RAG SYSTEM')
        print('='*80)
        self.query_engine = QueryEngine(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name,
            jetstream_client=self.jetstream_client,
        )

        # Initialize guardrails (on CPU to avoid large local allocations)
        self.enable_guardrails = enable_guardrails
        self.guardrails = None
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('INITIALIZING GUARDRAILS')
            print('='*80)
            self.guardrails = SafetyGuardrails(verbose=True, jetstream_client=self.jetstream_client)

    def _rewrite_query(self, original_query: str) -> str:
        """
        Rewrite the user query to be more detailed and contextually rich for RAG.

        This helps improve similarity search by expanding simple questions into
        more comprehensive queries that match better with course material embeddings.

        Args:
            original_query: The user's original question

        Returns:
            Rewritten query optimized for RAG retrieval
        """
        system_prompt = (
            "You optimize user queries for a Retrieval-Augmented Generation (RAG) system "
            "covering computer science course materials. Expand short questions into detailed, "
            "technical language that improves vector similarity search. Respond with a single "
            "concise line that keeps the original intent."
        )
        user_prompt = (
            f"Original question: \"{original_query}\"\n"
            "Include relevant CS terminology, synonyms, and related concepts. Return only the rewritten query."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            rewritten = self.jetstream_client.chat_completion(
                messages,
                max_tokens=120,
                temperature=0.5,
                top_p=0.9,
            )
        except RuntimeError as exc:
            print(f'⚠ Query rewriting failed ({exc}); using original question.')
            return original_query

        # Remove any common prefixes the model might add
        prefixes_to_remove = ['Output:', 'Expanded query:', 'Answer:', 'A:', 'Q:']
        for prefix in prefixes_to_remove:
            if rewritten.startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()

        # Take only the first paragraph/line if multiple were generated
        if '\n' in rewritten:
            rewritten = rewritten.split('\n')[0].strip()

        # If rewriting failed or produced garbage, use original query
        if len(rewritten) < 3 or len(rewritten) > 500:
            return original_query

        return rewritten

    def _generate_response(self, prompt: str, max_tokens: int) -> str:
        """Call the Jetstream inference service to generate a response."""
        max_tokens = max(32, min(max_tokens, 4096))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful teaching assistant for computer science courses. "
                    "Use the provided context when available. If the context is missing, answer "
                    "from general knowledge and be clear about any limitations."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        return self.jetstream_client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )

    def format_context(self, documents: List[Tuple[Dict[str, Any], float]]) -> str:
        """
        Format retrieved documents as context for the LLM.

        Args:
            documents: List of (result_dict, score) tuples

        Returns:
            Formatted context string
        """
        context_parts = []

        for idx, (result, score) in enumerate(documents, 1):
            payload = result['payload']

            # Extract key information
            file_path = payload.get('file_path', 'N/A')
            file_type = payload.get('file_type', 'N/A')
            course = payload.get('course', 'N/A')

            # Get content (prefer full_content if available, otherwise use content_preview)
            content = payload.get('full_content', payload.get('content_preview', ''))

            # Format document
            doc_str = f"Document {idx} (Relevance: {score:.4f})\n"
            doc_str += f"Source: {file_path}\n"
            doc_str += f"Course: {course}\n"
            doc_str += f"Type: {file_type}\n"
            doc_str += f"Content:\n{content}\n"

            context_parts.append(doc_str)

        return '\n' + ('='*80) + '\n\n'.join(context_parts) + '\n' + ('='*80)

    def query(self, user_query: str, show_context: bool = True, max_length: int = 2048):
        """Query the Jetstream-backed LLM with optional RAG context."""
        # GUARDRAIL CHECKPOINT 1: Validate input query
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('GUARDRAIL CHECK: Validating Input Query')
            print('='*80)

            is_safe, reason = self.guardrails.validate_input(user_query)

            if not is_safe:
                print(f'\n✗ QUERY BLOCKED: {reason}')
                print('='*80)
                return f"I cannot process this query. Reason: {reason}"

            print('='*80)

        # QUERY REWRITING: Expand query for better RAG retrieval
        print('\n' + '='*80)
        print('QUERY REWRITING: Expanding query for RAG')
        print('='*80)
        print(f'Original query: "{user_query}"')

        rewritten_query = self._rewrite_query(user_query)
        print(f'Rewritten query: "{rewritten_query}"')
        print('='*80)

        # Stage 1 & 2: Perform RAG with two-stage reranking using rewritten query
        print('\n' + '='*80)
        print('RETRIEVING RELEVANT DOCUMENTS')
        print('='*80)

        documents = self.query_engine.search(
            query=rewritten_query,
            top_k=3,
            use_reranker=True,
            rerank_candidates=50,
            stage1_top_k=7,
            min_score=7.0,
            verbose=True
        )

        use_rag = len(documents) > 0
        context = ''
        context_texts: List[str] = []

        if use_rag:
            context = self.format_context(documents)
            if show_context:
                print('\n' + '='*80)
                print('RETRIEVED DOCUMENTS (Context for LLM)')
                print('='*80)
                print(context)

            for result, _ in documents:
                payload = result['payload']
                context_texts.append(payload.get('full_content', payload.get('content_preview', '')))
        else:
            print('\n' + '='*80)
            print('⚠ NO DOCUMENTS PASSED THRESHOLD - LLM will answer without RAG')
            print('='*80)

        if use_rag:
            prompt = f"""Based on the following course materials, please answer the question. Use the information from the documents to provide a comprehensive and accurate answer.

{context}

Question: {user_query}

Answer:"""
        else:
            prompt = f"""Please answer the following question to the best of your knowledge. Note that no relevant course materials were found with high enough confidence, so provide a general answer based on your training.

Question: {user_query}

Answer:"""

        # Generate response via Jetstream inference service
        print('\n' + '='*80)
        print('GENERATING RESPONSE VIA JETSTREAM INFERENCE SERVICE')
        print('='*80 + '\n')

        try:
            response = self._generate_response(prompt, max_length)
        except RuntimeError as exc:
            print(f'\n✗ RESPONSE GENERATION FAILED: {exc}')
            print('='*80)
            return f"I could not contact the Jetstream inference service. Error: {exc}"

        response = response.strip()
        if response.lower().startswith('answer:'):
            response = response.split(':', 1)[-1].strip()

        # GUARDRAIL CHECKPOINT 2: Validate output response
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('GUARDRAIL CHECK: Validating Output Response')
            print('='*80)

            is_safe, reason, sanitized_response = self.guardrails.validate_output(
                query=user_query,
                response=response,
                context=context_texts if use_rag else None,
                check_hallucination=use_rag,
                check_pii=True
            )

            if not is_safe:
                print(f'\n✗ RESPONSE BLOCKED: {reason}')
                print('='*80)
                return f"I generated a response, but it failed safety validation. Reason: {reason}"

            if sanitized_response:
                response = sanitized_response

            print('='*80)

        return response


def main():
    parser = argparse.ArgumentParser(description='Query course materials with LLM assistance')
    parser.add_argument(
        'query',
        type=str,
        help='The question to ask'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='cs_materials',
        help='Qdrant collection to search (default: cs_materials)'
    )
    parser.add_argument(
        '--no-context',
        action='store_true',
        help='Hide the retrieved documents (only show final answer)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum length of generated response (default: 2048)'
    )
    parser.add_argument(
        '--no-guardrails',
        action='store_true',
        help='Disable safety guardrails (not recommended)'
    )
    parser.add_argument(
        '--rewrite-base-url',
        type=str,
        default=os.environ.get('JETSTREAM_REWRITE_BASE_URL'),
        help='Override Jetstream base URL for query rewriting (default: https://llm.jetstream-cloud.org/llama-4-scout/v1)'
    )
    parser.add_argument(
        '--rewrite-model',
        type=str,
        default=os.environ.get('JETSTREAM_REWRITE_MODEL'),
        help='Override Jetstream model ID for query rewriting (default: llama-4-scout)'
    )
    parser.add_argument(
        '--answer-base-url',
        type=str,
        default=os.environ.get('JETSTREAM_ANSWER_BASE_URL'),
        help='Override Jetstream base URL for answer generation (default: https://llm.jetstream-cloud.org/gpt-oss-120b/v1)'
    )
    parser.add_argument(
        '--answer-model',
        type=str,
        default=os.environ.get('JETSTREAM_ANSWER_MODEL'),
        help='Override Jetstream model ID for answer generation (default: gpt-oss-120b)'
    )
    parser.add_argument(
        '--jetstream-api-key',
        type=str,
        default=os.environ.get('JETSTREAM_API_KEY'),
        help='Jetstream API token (required when using the Open WebUI proxy)'
    )
    parser.add_argument(
        '--jetstream-timeout',
        type=int,
        default=int(os.environ.get('JETSTREAM_TIMEOUT', '120')),
        help='Timeout in seconds for Jetstream inference requests (default: 120)'
    )

    args = parser.parse_args()

    # Initialize system
    system = LLMQuerySystem(
        collection_name=args.collection,
        enable_guardrails=not args.no_guardrails,
        rewrite_base_url=args.rewrite_base_url,
        rewrite_model=args.rewrite_model,
        answer_base_url=args.answer_base_url,
        answer_model=args.answer_model,
        api_key=args.jetstream_api_key,
        request_timeout=args.jetstream_timeout,
    )

    # Query
    response = system.query(
        args.query,
        show_context=not args.no_context,
        max_length=args.max_length
    )

    # Print final answer
    print('='*80)
    print('FINAL ANSWER')
    print('='*80)
    print(response)
    print('\n' + '='*80)


if __name__ == '__main__':
    main()
