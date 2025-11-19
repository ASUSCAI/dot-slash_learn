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
        print('LOCAL EMBEDDING: Alibaba gte-large runs locally; model stays in memory for speed.')
        print('='*80)

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
        print('CLEANUP: Local embedder remains loaded; no teardown performed.')
        print('='*80)

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
            mastery_guidance = """BEGINNER LEVEL - Simple and Accessible

**Language Requirements:**
- Use plain, common words and short sentences.
- Introduce every technical term with the format **term** (*clear definition*).
- Avoid idioms; if one appears in the source, explain it inline.

**Structure and Content:**
- Assume the reader has no prior knowledge.
- Provide frequent, concrete examples drawn from the materials.
- Include numbered, step-by-step instructions where the source describes a process.
- Highlight "Why it matters" and "Common pitfalls" sections using headings.

**Tone:**
- Friendly, encouraging, direct, and second-person where appropriate ("You can...").
- Keep paragraphs short and focused on one idea."""

        elif mastery_level.lower() == "intermediate":
            mastery_guidance = """INTERMEDIATE LEVEL - Technical with Accessibility

**Language Requirements:**
- Balance precise terminology with quick reminders using **term** (*brief definition*).
- Maintain active voice and concrete verbs.
- Keep sentences lean; remove fillers and nominalizations.

**Structure and Content:**
- Connect concepts to previously learned fundamentals before extending them.
- Combine theory and application: clarify "How it works" and "Why it matters".
- Call out patterns, comparisons, and edge cases using bullet lists.
- Provide focused code or data examples only when they support the explanation.

**Tone:**
- Professional yet approachable.
- Emphasize how each detail helps the practitioner deepen skill confidence."""

        else:  # advanced
            mastery_guidance = """ADVANCED LEVEL - Expert Depth with Clarity

**Language Requirements:**
- Use exact technical terms while keeping sentences direct.
- When a niche concept appears for the first time, pair it with a concise parenthetical reminder.
- Avoid rhetorical questions and idioms; state implications plainly.

**Structure and Content:**
- Focus on theoretical nuance, trade-offs, and performance considerations.
- Organize dense information into short subsections and lists.
- Discuss real-world applications, limitations, and advanced techniques drawn from the materials.
- Highlight decision criteria or comparison tables when helpful.

**Tone:**
- Authoritative and efficient, prioritizing actionable insight.
- Assume strong background knowledge but still signal why each detail matters."""

        # Combine chunks into context
        context = self._format_chunks_for_context(chunks, skill)

        prompt = f"""Based on the course materials below, create concise, learner-friendly reading material about "{skill}" in markdown format.

Learning Objective: {learning_objective}
Target Skill: {skill}
Mastery Level: {mastery_level.upper()}

{mastery_guidance}

Course Materials:
{context}

Follow these non-negotiable writing rules:

1. **Overall structure and organization**
    - Keep the total length between 500 and 900 words regardless of mastery level.
    - Start with a level-one heading `# {skill}`.
    - Immediately add an **advance organizer** titled "## You will learn" with exactly 2-3 bullet points previewing the key takeaways.
    - Use short sections with descriptive micro-headings such as "## What it covers", "## Why it matters", "## How it works", "## Try it step-by-step", "## Common pitfalls", and "## What changed next" when relevant.
    - End with a final section titled "## Key takeaway" containing one plain-language sentence summarizing the most important point.

2. **Plain language pass**
    - Use active voice and subject-verb-object order.
    - Prefer concrete verbs instead of nominalizations (write "explain" not "offer an explanation").
    - Replace idioms or metaphors with direct wording, or explain them inline (for example, "spread quickly (people often say 'like wildfire' to mean very fast)").
    - Keep sentences short; avoid center-embedded clauses and rhetorical questions.
    - Do not use the em dash character (Unicode U+2014); rely on commas, parentheses, or separate sentences instead.
    - Never include quizzes, comprehension questions, or prompts that ask the reader anything.

3. **Chunking and signaling**
    - Break paragraphs into compact chunks (3-4 sentences maximum).
    - Use bullet or numbered lists for sequences, comparisons, or steps.
    - Provide clear transitions that explain why each section matters before diving into detail.

4. **Referent clarity and consistent naming**
    - Repeat the specific noun instead of relying on ambiguous pronouns (say "the equilibrium constant" instead of "it" when readers could be confused).
    - Use the same name for a concept throughout; do not alternate between synonyms once a term is chosen.

5. **Vocabulary scaffolds**
    - Define each technical term at first use with a short parenthetical explanation: **term** (simple definition).
    - Pair specialized language with a familiar synonym when precision matters: "photosynthesis (how plants make food)".

6. **Information ordering and fidelity to sources**
    - Lead with the main point of each section before supporting details.
    - Base all claims on the provided course materials. Do not invent new facts.
    - Use lists instead of dense prose whenever you describe steps or categories.

7. **Formatting requirements**
    - Use Markdown headings (#, ##, ###), bullet lists, numbered lists, bold, and italics appropriately.
    - Include code blocks only if the source materials contain runnable examples; otherwise prefer prose explanations.

Generate the markdown content now without adding any questions or quiz items."""

        # Load LLM and generate
        self.embedder._load_llm()

        print('\nCalling LLM to generate reading material...')
        response = self._call_llm_sync(prompt, max_tokens=1200)

        return response.strip()

    def _call_llm_sync(self, prompt: str, max_tokens: int = 1200) -> str:
        """
        Synchronous LLM call.

        Args:
            prompt: The prompt to send to LLM
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response text
        """
        return self.embedder.generate_text(
            prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
        )

    def _offload_embedding_model(self):
        """Compatibility no-op for legacy GPU workflow."""
        print('  Local embedder stays loaded; offload is not implemented.')

    def _restore_embedding_model(self):
        """Compatibility no-op for legacy GPU workflow."""
        print('  Local embedder was not offloaded; nothing to restore.')

    def _cleanup_vram(self):
        """Compatibility no-op for legacy GPU workflow."""
        print('  Jetstream LLM stays remote; no VRAM cleanup required.')
