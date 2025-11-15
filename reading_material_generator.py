'''
Reading Material Generator

Generates educational reading material in markdown format based on course materials and skills.

Pipeline:
1. Filter Qdrant chunks by skill
2. Generate structured markdown reading material using LLM
3. Format with proper markdown syntax (headings, lists, code blocks, etc.)

Usage:
    from reading_material_generator import ReadingMaterialGenerator

    generator = ReadingMaterialGenerator(qdrant_client, collection_name, embedder)
    markdown = await generator.generate_reading_material(learning_objective, skill, style)
'''

import torch
import asyncio
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class ReadingMaterialGenerator:

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embedder):
        """
        Initialize ReadingMaterialGenerator.

        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of the Qdrant collection
            embedder: CourseEmbedder instance (for LLM access)
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedder = embedder

    async def generate_reading_material(
        self,
        learning_objective: str,
        skill: str,
        mastery_level: str = "beginner"
    ) -> str:
        """
        Generate reading material in markdown format.

        Args:
            learning_objective: The learning objective description
            skill: The skill to generate material for
            mastery_level: Learner's mastery level - 'beginner', 'intermediate', or 'advanced'

        Returns:
            Markdown formatted reading material
        """
        print('='*80)
        print('READING MATERIAL GENERATION PIPELINE')
        print('='*80)
        print(f'Learning Objective: {learning_objective}')
        print(f'Skill: {skill}')
        print(f'Mastery Level: {mastery_level}')

        # VRAM OPTIMIZATION: Offload embedding model to free VRAM
        print('\n' + '='*80)
        print('FREEING VRAM: Offloading embedding model to CPU...')
        print('='*80)
        self._offload_embedding_model()

        # Retrieve chunks by skill
        print('\n' + '='*80)
        print('RETRIEVING COURSE MATERIALS')
        print('='*80)

        chunks = self._retrieve_chunks_by_skill(skill)

        if not chunks:
            print(f'\nWarning: No chunks found for skill "{skill}"')
            # Restore embedding model before returning
            self._restore_embedding_model()
            return f"# {skill}\n\nNo course materials found for this skill."

        print(f'\nRetrieved {len(chunks)} chunks for skill "{skill}"')

        # Generate reading material
        print('\n' + '='*80)
        print('GENERATING READING MATERIAL')
        print('='*80)

        markdown_content = self._generate_content_with_llm(
            chunks, learning_objective, skill, mastery_level
        )

        print('\n' + '='*80)
        print('READING MATERIAL GENERATION COMPLETE')
        print('='*80)

        # VRAM CLEANUP: Offload LLM and restore embedding model
        print('\n' + '='*80)
        print('CLEANUP: Offloading LLM and restoring embedding model...')
        print('='*80)
        self._cleanup_vram()

        return markdown_content

    def _retrieve_chunks_by_skill(self, skill: str) -> List[Dict[str, Any]]:
        """
        Retrieve chunks from Qdrant that contain the specified skill.

        Args:
            skill: The skill to filter by

        Returns:
            List of chunk payloads
        """
        try:
            print(f'\nSearching for chunks with skill: "{skill}"')

            results, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="skills",
                            match=MatchValue(value=skill)
                        )
                    ]
                ),
                limit=100,  # Retrieve up to 100 chunks
                with_payload=True,
                with_vectors=False
            )

            chunks = [point.payload for point in results]
            print(f'  Found {len(chunks)} chunks')

            return chunks

        except Exception as e:
            print(f'Error retrieving chunks: {e}')
            return []

    def _format_chunks_for_context(
        self,
        chunks: List[Dict[str, Any]],
        skill: str
    ) -> str:
        """
        Format chunks into context string for LLM.

        Args:
            chunks: List of chunk payloads
            skill: Skill to emphasize in context

        Returns:
            Formatted context string
        """
        context_parts = []

        # Use more chunks for reading material (up to 30)
        chunks_to_use = chunks[:30]

        for idx, chunk in enumerate(chunks_to_use, 1):
            file_path = chunk.get('file_path', 'N/A')
            content = chunk.get('full_content', chunk.get('content_preview', ''))

            # Truncate very long chunks
            if len(content) > 2000:
                content = content[:2000] + "..."

            context_parts.append(f"[Chunk {idx}]\nSource: {file_path}\nContent: {content}\n")

        context = "\n".join(context_parts)
        context = f"Focus on content related to: {skill}\n\n{context}"

        return context

    def _generate_content_with_llm(
        self,
        chunks: List[Dict[str, Any]],
        learning_objective: str,
        skill: str,
        mastery_level: str
    ) -> str:
        """
        Generate markdown reading material using LLM.

        Args:
            chunks: Retrieved chunks
            learning_objective: Learning objective
            skill: Target skill
            mastery_level: Learner's mastery level - 'beginner', 'intermediate', or 'advanced'

        Returns:
            Markdown formatted content
        """
        # Define mastery-level specific guidance
        if mastery_level.lower() == "beginner":
            mastery_guidance = """BEGINNER LEVEL - Simple and Accessible:

**Language Requirements:**
- Use simple, clear language that anyone can understand
- AVOID technical jargon, idioms, and complex terminology
- When technical terms are unavoidable, provide inline definitions using this format:
  **term** (*definition in simple words*)
  Example: "A **variable** (*a named storage location for data*) is like a labeled box"
- Break down complex ideas into small, digestible steps
- Use analogies and real-world comparisons to explain concepts

**Structure and Content:**
- Start with the absolute basics - assume no prior knowledge
- Include MANY practical examples with detailed explanations
- Provide step-by-step walkthroughs for any processes
- Use numbered lists for sequential steps
- Include "What this means" or "Why this matters" sections
- Add plenty of code examples with line-by-line explanations

**Tone:**
- Friendly, encouraging, and conversational
- Patient and thorough - don't skip "obvious" details
- Use "you" to address the reader directly"""

        elif mastery_level.lower() == "intermediate":
            mastery_guidance = """INTERMEDIATE LEVEL - Technical with Accessibility:

**Language Requirements:**
- Balance technical accuracy with clarity
- Use technical terminology but explain it when first introduced using:
  **term** (*brief technical definition*)
- Assume basic familiarity but don't assume expertise
- You can use some industry terms, but clarify specialized concepts

**Structure and Content:**
- Build on foundational knowledge - quick review of basics is okay
- Mix theory with practical examples
- Include both "how" and "why" explanations
- Show common patterns and best practices
- Include edge cases and gotchas
- Compare different approaches where relevant
- Moderate amount of code examples with focused explanations

**Tone:**
- Professional but approachable
- Assume the reader is motivated to learn deeper concepts
- Bridge the gap between beginner understanding and advanced expertise"""

        else:  # advanced
            mastery_guidance = """ADVANCED LEVEL - Full Technical Rigor:

**Language Requirements:**
- Use precise technical terminology throughout
- Industry-standard language and formal definitions
- Assume familiarity with related concepts
- No need for basic explanations or inline definitions
- Can use idioms, design patterns, and advanced concepts freely

**Structure and Content:**
- Focus on deep technical understanding and nuances
- Emphasize theoretical foundations and formal principles
- Discuss performance implications, trade-offs, and optimization
- Cover edge cases, limitations, and advanced use cases
- Reference related concepts and interconnections
- Include complex examples that demonstrate mastery
- Discuss real-world applications in production systems

**Tone:**
- Professional, academic, and authoritative
- Assume reader has strong foundation and wants expert-level insight
- Dense, information-rich content - efficiency over hand-holding"""

        # Combine chunks into context
        context = self._format_chunks_for_context(chunks, skill)

        prompt = f"""Based on the course materials below, create comprehensive reading material about "{skill}" in markdown format.

Learning Objective: {learning_objective}
Target Skill: {skill}
Mastery Level: {mastery_level.upper()}

{mastery_guidance}

Course Materials:
{context}

Create well-structured educational reading material following the mastery level guidelines above with these requirements:

1. **Structure**: Use proper markdown formatting with clear sections:
   - Start with a main heading (# {skill})
   - Use subheadings (##, ###) for major topics and subtopics
   - BEGINNER: More sections with smaller chunks (Introduction, What is..., Why it matters, Basic Examples, Step-by-Step Guide, Common Mistakes, Summary)
   - INTERMEDIATE: Balanced sections (Overview, Core Concepts, Practical Applications, Best Practices, Common Patterns, Summary)
   - ADVANCED: Dense sections (Introduction, Theoretical Foundation, Implementation Details, Advanced Techniques, Trade-offs and Optimization, Conclusion)

2. **Content Depth**:
   - Base content directly on the course materials provided
   - BEGINNER: Start from absolute basics, many examples, step-by-step walkthroughs
   - INTERMEDIATE: Build on fundamentals, mix of theory and practice, include patterns and edge cases
   - ADVANCED: Deep technical detail, performance considerations, production concerns, advanced use cases
   - Use bullet points and numbered lists where appropriate
   - Include code blocks with proper syntax highlighting when showing code

3. **Terminology Handling**:
   - BEGINNER: Use **bold term** (*simple definition in italics*) format for ALL technical terms
   - INTERMEDIATE: Use **bold term** (*technical definition*) for specialized concepts only
   - ADVANCED: Use technical terms freely without inline definitions

4. **Markdown Elements**:
   - Headers: #, ##, ###
   - Lists: -, *, numbered lists
   - Code blocks: ```language ... ```
   - Inline code: `code`
   - Bold: **text**
   - Italic: *text*
   - Blockquotes: > for important notes or quotes

5. **Length and Comprehensiveness**:
   - BEGINNER: 1000-1800 words - detailed and thorough with lots of explanation
   - INTERMEDIATE: 800-1500 words - balanced depth covering the skill completely
   - ADVANCED: 600-1200 words - dense, information-rich content

6. **Educational Value**: Ensure the content completely covers "{skill}" at the appropriate mastery level

Generate the markdown content now:"""

        # Load LLM and generate
        self.embedder._load_llm()

        print('\nCalling LLM to generate reading material...')
        response = self._call_llm_sync(prompt, max_tokens=2500)

        return response.strip()

    def _call_llm_sync(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Synchronous LLM call.

        Args:
            prompt: The prompt to send to LLM
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response text
        """
        # Ensure LLM is loaded
        self.embedder._load_llm()

        messages = [{"role": "user", "content": prompt}]
        text = self.embedder.llm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.embedder.llm_processor(text=text, return_tensors='pt')
        inputs = {k: v.to(self.embedder.llm_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.embedder.llm.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        response = self.embedder.llm_processor.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response.strip()

    def _offload_embedding_model(self):
        """Offload embedding model to CPU to free VRAM for LLM"""
        try:
            if hasattr(self.embedder, 'model') and self.embedder.model is not None:
                print('  Offloading embedding model to CPU...')
                self.embedder.model = self.embedder.model.cpu()
                torch.cuda.empty_cache()
                print('  ✓ Embedding model offloaded')
        except Exception as e:
            print(f'  Warning: Could not offload embedding model: {e}')

    def _restore_embedding_model(self):
        """Restore embedding model to GPU"""
        try:
            if hasattr(self.embedder, 'model') and self.embedder.model is not None:
                print('  Restoring embedding model to GPU...')
                self.embedder.model = self.embedder.model.to(self.embedder.device)
                print('  ✓ Embedding model restored')
        except Exception as e:
            print(f'  Warning: Could not restore embedding model: {e}')

    def _cleanup_vram(self):
        """Cleanup VRAM after generation"""
        try:
            print('  Offloading LLM to CPU...')
            if hasattr(self.embedder, 'llm') and self.embedder.llm is not None:
                self.embedder.llm = self.embedder.llm.cpu()
                torch.cuda.empty_cache()
                print('  ✓ LLM offloaded')

            # Restore embedding model
            self._restore_embedding_model()

            torch.cuda.empty_cache()
            print('  ✓ VRAM cleanup complete')

        except Exception as e:
            print(f'  Warning: VRAM cleanup encountered issue: {e}')
