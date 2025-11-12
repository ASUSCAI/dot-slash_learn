'''
LLM Query System with RAG

Main entry point for querying course materials with LLM assistance.
Performs two-stage reranking RAG and passes context to Qwen3-VL-4B-Instruct.

Usage:
    python llm.py "What is dynamic programming?"
    python llm.py "Explain Newton's laws" --collection physics_materials
'''

import argparse
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from query import QueryEngine
from guardrails import SafetyGuardrails


class LLMQuerySystem:

    def __init__(self, collection_name: str = 'cs_materials', qdrant_host: str = 'localhost', qdrant_port: int = 6333, enable_guardrails: bool = True):
        """
        Initialize the LLM query system with RAG.

        Args:
            collection_name: Qdrant collection to search
            qdrant_host: Qdrant host address
            qdrant_port: Qdrant port
            enable_guardrails: Whether to enable safety guardrails
        """
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')

        # Initialize RAG query engine
        print('\n' + '='*80)
        print('INITIALIZING RAG SYSTEM')
        print('='*80)
        self.query_engine = QueryEngine(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name
        )

        # Initialize guardrails (on CPU to save VRAM)
        self.enable_guardrails = enable_guardrails
        self.guardrails = None
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('INITIALIZING GUARDRAILS')
            print('='*80)
            self.guardrails = SafetyGuardrails(device='cpu', verbose=True)

        # Don't load LLM yet - we'll lazy load it after RAG to save VRAM
        self.llm = None
        self.processor = None
        print('\nLLM will be loaded after RAG completes (lazy loading to save VRAM)\n')

    def _load_llm(self):
        """Lazy load the LLM model when needed"""
        if self.llm is None:
            print('\n' + '='*80)
            print('LOADING LLM: Qwen3-VL-4B-Instruct')
            print('='*80)
            self.processor = AutoProcessor.from_pretrained('Qwen/Qwen3-VL-4B-Instruct')
            self.llm = AutoModelForImageTextToText.from_pretrained(
                'Qwen/Qwen3-VL-4B-Instruct',
                dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map='auto'
            )
            self.llm.eval()
            print(f'LLM loaded successfully\n')

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
        """
        Query the LLM with RAG context.

        Args:
            user_query: The user's question
            show_context: Whether to print the retrieved documents (for debugging)
            max_length: Maximum length of generated response

        Returns:
            LLM response string
        """
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

        # Stage 1 & 2: Perform RAG with two-stage reranking
        print('\n' + '='*80)
        print('RETRIEVING RELEVANT DOCUMENTS')
        print('='*80)

        documents = self.query_engine.search(
            query=user_query,
            top_k=3,               # Final number of documents to pass to LLM
            use_reranker=True,
            rerank_candidates=50,  # Retrieve 50 candidates from vector search
            stage1_top_k=7,        # Keep top 7 after first rerank, then expand with neighbors
            verbose=True
        )

        if not documents:
            return "Sorry, I couldn't find any relevant documents to answer your question."

        # Offload RAG models to CPU to free VRAM for LLM
        print('\n' + '='*80)
        print('FREEING VRAM: Moving RAG models to CPU...')
        print('='*80)
        self.query_engine.model.to('cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('VRAM freed\n')

        # Format context
        context = self.format_context(documents)

        # Show retrieved documents for debugging
        if show_context:
            print('\n' + '='*80)
            print('RETRIEVED DOCUMENTS (Context for LLM)')
            print('='*80)
            print(context)

        # Now load LLM after RAG models are offloaded
        self._load_llm()

        # Build prompt
        prompt = f"""Based on the following course materials, please answer the question. Use the information from the documents to provide a comprehensive and accurate answer.

{context}

Question: {user_query}

Answer:"""

        # Generate response
        print('\n' + '='*80)
        print('GENERATING RESPONSE')
        print('='*80 + '\n')

        # Format as conversation for the vision-language model
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Apply chat template and tokenize
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=text, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract just the answer part (after "Answer:")
        if "Answer:" in response:
            response = response.split("Answer:")[-1].strip()

        # GUARDRAIL CHECKPOINT 2: Validate output response
        if self.enable_guardrails:
            print('\n' + '='*80)
            print('GUARDRAIL CHECK: Validating Output Response')
            print('='*80)

            # Extract content from documents for grounding check
            context_texts = []
            for result, score in documents:
                payload = result['payload']
                content = payload.get('full_content', payload.get('content_preview', ''))
                context_texts.append(content)

            is_safe, reason, sanitized_response = self.guardrails.validate_output(
                query=user_query,
                response=response,
                context=context_texts,
                check_hallucination=True,
                check_pii=True
            )

            if not is_safe:
                print(f'\n✗ RESPONSE BLOCKED: {reason}')
                print('='*80)
                return f"I generated a response, but it failed safety validation. Reason: {reason}"

            # Use sanitized response if PII was redacted
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

    args = parser.parse_args()

    # Initialize system
    system = LLMQuerySystem(
        collection_name=args.collection,
        enable_guardrails=not args.no_guardrails
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
