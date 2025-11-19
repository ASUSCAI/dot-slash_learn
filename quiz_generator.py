'''
Quiz Generator

Generates quiz questions with explanations based on course materials and skills.

Pipeline:
1. Filter Qdrant chunks by skill
2. Generate questions using LLM (80% MCQ, 20% True/False)
3. Parallel generation of explanations for each option

Usage:
    from quiz_generator import QuizGenerator

    generator = QuizGenerator(qdrant_client, collection_name, embedder)
    questions = generator.generate_quiz(learning_objective, skill, num_questions)
'''

import json
import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable, Tuple
from concurrent.futures import ThreadPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue


class QuizGenerator:

    def __init__(self, qdrant_client: QdrantClient, collection_name: str, embedder):
        """
        Initialize QuizGenerator.

        Args:
            qdrant_client: Qdrant client instance
            collection_name: Name of the Qdrant collection
            embedder: CourseEmbedder instance (for LLM access)
        """
        self.client = qdrant_client
        self.collection_name = collection_name
        self.embedder = embedder
        self.executor = ThreadPoolExecutor(max_workers=4)  # Reduced to 4 workers to save VRAM

    async def generate_quiz(
        self,
        learning_objective: str,
        skill: str,
        num_easy: int = 0,
        num_medium: int = 0,
        num_hard: int = 0,
        question_style: str = "concrete",
        question_callback: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate quiz questions with explanations at different difficulty levels.

        VRAM Management Strategy:
        1. Offload embedding model to CPU before loading LLM
        2. Load LLM only when needed
        3. Keep LLM loaded for all question/explanation generation
        4. Offload LLM to CPU after completion

        Args:
            learning_objective: The learning objective description
            skill: The skill to test
            num_easy: Number of easy questions to generate
            num_medium: Number of medium difficulty questions to generate
            num_hard: Number of hard questions to generate
            question_style: Explanation style - 'concrete' (example-driven) or 'abstract' (technical)

        Returns:
            List of question dictionaries with explanations and difficulty levels
        """
        total_questions = num_easy + num_medium + num_hard

        print('='*80)
        print('QUIZ GENERATION PIPELINE')
        print('='*80)
        print(f'Learning Objective: {learning_objective}')
        print(f'Skill: {skill}')
        print(f'Total Questions: {total_questions}')
        print(f'  - Easy: {num_easy}')
        print(f'  - Medium: {num_medium}')
        print(f'  - Hard: {num_hard}')

        # VRAM OPTIMIZATION: Offload embedding model to free VRAM
        print('\n' + '='*80)
        print('LOCAL EMBEDDING: Alibaba gte-large runs locally; model stays resident for faster generation.')
        print('='*80)

        # Phase 1: Retrieve chunks and lazily generate questions
        print('\n' + '='*80)
        print('PHASE 1: Retrieving chunks and lazily generating questions')
        print('='*80)

        chunks = self._retrieve_chunks_by_skill(skill)

        if not chunks:
            print(f'\nWarning: No chunks found for skill "{skill}"')
            return []

        print(f'\nRetrieved {len(chunks)} chunks for skill "{skill}"')

        # Prepare shared context once
        context = self._format_chunks_for_context(chunks, skill)

        # Calculate total MCQ/T-F split once for all questions
        # Then distribute across difficulty levels
        total_mcq = int(total_questions * 0.8)
        total_tf = total_questions - total_mcq

        # Ensure at least 1 of each if total >= 2
        if total_questions >= 2 and total_mcq == 0:
            total_mcq = 1
            total_tf = total_questions - 1
        elif total_questions >= 2 and total_tf == 0:
            total_tf = 1
            total_mcq = total_questions - 1

        print(f'\nTotal distribution: {total_mcq} MCQ, {total_tf} True/False')

        # Distribute MCQ and T/F questions across difficulty levels proportionally
        difficulty_distribution = []

        if num_easy > 0:
            easy_mcq = round(total_mcq * (num_easy / total_questions))
            easy_tf = num_easy - easy_mcq
            difficulty_distribution.append(('easy', num_easy, easy_mcq, easy_tf))

        if num_medium > 0:
            medium_mcq = round(total_mcq * (num_medium / total_questions))
            medium_tf = num_medium - medium_mcq
            difficulty_distribution.append(('medium', num_medium, medium_mcq, medium_tf))

        if num_hard > 0:
            hard_mcq = round(total_mcq * (num_hard / total_questions))
            hard_tf = num_hard - hard_mcq
            difficulty_distribution.append(('hard', num_hard, hard_mcq, hard_tf))

        # Adjust for rounding errors - ensure we hit exact totals
        actual_mcq = sum(mcq for _, _, mcq, _ in difficulty_distribution)
        actual_tf = sum(tf for _, _, _, tf in difficulty_distribution)

        if actual_mcq != total_mcq:
            diff = total_mcq - actual_mcq
            # Add/subtract from largest group
            largest_idx = max(range(len(difficulty_distribution)),
                            key=lambda i: difficulty_distribution[i][1])
            diff_name, diff_total, diff_mcq, diff_tf = difficulty_distribution[largest_idx]
            difficulty_distribution[largest_idx] = (diff_name, diff_total, diff_mcq + diff, diff_tf - diff)

        # Generate questions lazily and attach explanations before moving on
        final_questions: List[Dict[str, Any]] = []
        question_counter = 0

        async def _handle_question(raw_question: Optional[Dict[str, Any]]) -> None:
            nonlocal question_counter
            if not raw_question:
                return

            question_counter += 1
            enriched = await self._build_question_with_explanations(
                question_data=raw_question,
                context=context,
                skill=skill,
                question_style=question_style,
                question_number=question_counter
            )

            if question_callback is not None:
                await question_callback(enriched)

            final_questions.append(enriched)

        for difficulty, total, num_mcq_diff, num_tf_diff in difficulty_distribution:
            if num_mcq_diff > 0:
                print(f'\nGenerating {num_mcq_diff} MCQ questions at {difficulty.upper()} difficulty')
            for idx in range(num_mcq_diff):
                print(f'  -> Generating MCQ {idx + 1} of {num_mcq_diff} ({difficulty})')
                raw_question = self._generate_single_question_with_llm(
                    context=context,
                    learning_objective=learning_objective,
                    skill=skill,
                    difficulty=difficulty,
                    question_type='mcq'
                )
                await _handle_question(raw_question)

            if num_tf_diff > 0:
                print(f'\nGenerating {num_tf_diff} True/False questions at {difficulty.upper()} difficulty')
            for idx in range(num_tf_diff):
                print(f'  -> Generating True/False {idx + 1} of {num_tf_diff} ({difficulty})')
                raw_question = self._generate_single_question_with_llm(
                    context=context,
                    learning_objective=learning_objective,
                    skill=skill,
                    difficulty=difficulty,
                    question_type='true_false'
                )
                await _handle_question(raw_question)

        if not final_questions:
            print('\nWarning: Failed to generate any complete questions')
            self._restore_embedding_model()
            return []

        print('\n' + '='*80)
        print(f'QUIZ GENERATION COMPLETE: {len(final_questions)} questions (lazy loading)')
        print('='*80)

        # VRAM CLEANUP: Offload LLM and restore embedding model
        print('\n' + '='*80)
        print('CLEANUP: Local embedder remains loaded; no additional teardown performed.')
        print('='*80)

        return final_questions

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
        skill: Optional[str] = None
    ) -> str:
        """
        Format chunks into context string for LLM.

        Args:
            chunks: List of chunk payloads
            skill: Optional skill to emphasize in context

        Returns:
            Formatted context string
        """
        context_parts = []

        # Limit to first 20 chunks to avoid token limits
        chunks_to_use = chunks[:20]

        for idx, chunk in enumerate(chunks_to_use, 1):
            file_path = chunk.get('file_path', 'N/A')
            content = chunk.get('full_content', chunk.get('content_preview', ''))

            # Truncate very long chunks
            if len(content) > 1500:
                content = content[:1500] + "..."

            context_parts.append(f"[Chunk {idx}]\nSource: {file_path}\nContent: {content}\n")

        context = "\n".join(context_parts)

        if skill:
            context = f"Focus on content related to: {skill}\n\n{context}"

        return context

    async def _build_question_with_explanations(
        self,
        *,
        question_data: Dict[str, Any],
        context: str,
        skill: str,
        question_style: str,
        question_number: int
    ) -> Dict[str, Any]:
        """Attach explanations to a single question before generating the next."""

        tasks = []

        for option in question_data['options']:
            option_id = option['id']
            is_correct = (option_id == question_data['correct_answer'])
            tasks.append(
                self._generate_single_explanation(
                    question=question_data['question'],
                    option_id=option_id,
                    option_text=option['text'],
                    is_correct=is_correct,
                    context=context,
                    skill=skill,
                    question_style=question_style
                )
            )

        explanations = await asyncio.gather(*tasks)

        options_with_explanations = []
        for option, explanation in zip(question_data['options'], explanations):
            options_with_explanations.append({
                'id': option['id'],
                'text': option['text'],
                'explanation': explanation,
                'is_correct': option['id'] == question_data['correct_answer']
            })

        return {
            'id': f'q{question_number}',
            'type': question_data['type'],
            'question': question_data['question'],
            'options': options_with_explanations,
            'skill': question_data.get('skill', skill),
            'difficulty': question_data.get('difficulty', 'medium')
        }

    async def iter_quiz_questions(
        self,
        learning_objective: str,
        skill: str,
        num_easy: int = 0,
        num_medium: int = 0,
        num_hard: int = 0,
        question_style: str = "concrete"
    ):
        """Yield quiz questions one at a time as they are generated."""

        event_queue: "asyncio.Queue[Tuple[str, Optional[Dict[str, Any]]]]" = asyncio.Queue()

        async def _callback(question: Dict[str, Any]) -> None:
            await event_queue.put(("question", question))

        async def _run_generation() -> None:
            try:
                await self.generate_quiz(
                    learning_objective=learning_objective,
                    skill=skill,
                    num_easy=num_easy,
                    num_medium=num_medium,
                    num_hard=num_hard,
                    question_style=question_style,
                    question_callback=_callback
                )
                await event_queue.put(("complete", None))
            except Exception as exc:
                await event_queue.put(("error", exc))

        runner_task = asyncio.create_task(_run_generation())

        try:
            while True:
                event_type, payload = await event_queue.get()
                if event_type == "question":
                    if payload is not None:
                        yield payload
                elif event_type == "complete":
                    break
                elif event_type == "error":
                    if isinstance(payload, Exception):
                        raise payload
                    raise RuntimeError("Quiz generation failed unexpectedly")
        finally:
            await asyncio.gather(runner_task, return_exceptions=True)

    def _generate_single_question_with_llm(
        self,
        *,
        context: str,
        learning_objective: str,
        skill: str,
        difficulty: str,
        question_type: str
    ) -> Optional[Dict[str, Any]]:
        """Generate a single quiz question via LLM to reduce latency spikes."""

        difficulty_guides = {
            'easy': 'Easy questions should test basic recall and understanding of fundamental concepts. Use straightforward language and test direct knowledge from the materials.',
            'medium': 'Medium questions should require applying concepts and making connections. Test understanding beyond simple recall.',
            'hard': 'Hard questions should involve complex analysis, synthesis of multiple concepts, or subtle edge cases. These should challenge deep understanding.'
        }

        question_type = question_type.lower()

        if question_type == 'mcq':
            type_instructions = (
                'Generate exactly one multiple-choice question with four answer options labeled "A", "B", "C", and "D".'
            )
            format_instructions = (
                '[{"id": "A", "text": "Option text"}, {"id": "B", "text": "Option text"}, {"id": "C", "text": "Option text"}, {"id": "D", "text": "Option text"}]'
            )
            type_value = 'mcq'
        elif question_type == 'true_false':
            type_instructions = 'Generate exactly one true/false question with options for "true" and "false".'
            format_instructions = (
                '[{"id": "true", "text": "True"}, {"id": "false", "text": "False"}]'
            )
            type_value = 'true_false'
        else:
            print(f'  Warning: Unsupported question type "{question_type}"')
            return None

        prompt = f"""Based on the course materials below, generate one quiz question to test understanding of "{skill}".

Learning Objective: {learning_objective}
Target Skill: {skill}
Difficulty Level: {difficulty.upper()}

{difficulty_guides[difficulty]}

Course Materials:
{context}

{type_instructions}

Requirements:
1. Focus on testing understanding of "{skill}".
2. Base the question directly on the course materials provided.
3. Make the question clear and unambiguous.
4. Ensure only one correct answer.
5. Make incorrect options plausible but clearly wrong.
6. Keep answer options short (5-10 words each).
7. Do not include any additional commentary or explanation.

Return ONLY a valid JSON object in this exact format (no additional text):
{{
  "type": "{type_value}",
  "question": "Question text",
  "options": {format_instructions},
  "correct_answer": "ID of correct option"
}}

JSON Object:"""

        print('    Calling LLM for single question...')
        response = self._call_llm_sync(prompt, max_tokens=600)

        try:
            response = response.strip()

            # Attempt to locate JSON object in response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx == -1 or end_idx <= start_idx:
                print('    Warning: Could not locate JSON object in LLM response')
                print(f'    Response: {response[:400]}...')
                return None

            json_str = response[start_idx:end_idx]
            parsed = json.loads(json_str)

            if isinstance(parsed, list):
                if not parsed:
                    print('    Warning: LLM returned an empty list')
                    return None
                parsed = parsed[0]

            if not isinstance(parsed, dict):
                print('    Warning: Parsed JSON is not an object')
                return None

            if parsed.get('type') != type_value:
                print('    Warning: LLM returned unexpected question type')
                return None

            parsed['difficulty'] = difficulty
            return parsed

        except json.JSONDecodeError as exc:
            print(f'    Warning: Failed to parse JSON from LLM response: {exc}')
            print(f'    Response: {response[:400]}...')
            return None

    async def _generate_single_explanation(
        self,
        question: str,
        option_id: str,
        option_text: str,
        is_correct: bool,
        context: str,
        skill: str,
        question_style: str = "concrete"
    ) -> str:
        """
        Generate explanation for a single option (separate LLM conversation).

        Args:
            question: The question text
            option_id: Option identifier (A, B, C, D, true, false)
            option_text: Option text
            is_correct: Whether this option is correct
            context: Course material context
            skill: Target skill
            question_style: Explanation style - 'concrete' or 'abstract'

        Returns:
            Explanation text
        """
        correctness = "correct" if is_correct else "incorrect"

        # Define style-specific guidance
        if question_style.lower() == "concrete":
            style_guidance = """Use concrete examples and real-world scenarios to illustrate your explanation.
Provide specific instances and practical demonstrations of the concept.
Use story-based explanations with tangible examples that students can relate to.
Avoid overly technical jargon - focus on making the concept accessible through examples."""
        else:  # abstract
            style_guidance = """Use precise technical terminology and formal definitions.
Focus on theoretical concepts and formal explanations.
Emphasize the technical aspects and academic understanding of the concept.
Use proper terminology from the field and reference formal principles."""

        prompt = f"""Based on the course materials about "{skill}", explain why this answer is {correctness}.

Course Materials:
{context}

Question: {question}
Selected Answer ({option_id}): {option_text}

Provide a single crisp sentence (no more than 18 words) explaining why this answer is {correctness}.
Focus on educational value; highlight the key idea a student must know about "{skill}".
Reference the most relevant detail from the course materials when possible.

{style_guidance}

Explanation:"""

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        explanation = await loop.run_in_executor(
            self.executor,
            self._call_llm_sync,
            prompt,
            200  # max_tokens
        )

        return explanation.strip()

    def _call_llm_sync(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Synchronous LLM call for thread pool executor.

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
        print('  Local embedder remains in memory; offload is not implemented.')

    def _restore_embedding_model(self):
        """Compatibility no-op for legacy GPU workflow."""
        print('  Local embedder was never offloaded; nothing to restore.')

    def _cleanup_vram(self):
        """Compatibility no-op for legacy GPU workflow."""
        print('  Jetstream LLM stays remote; no VRAM cleanup required locally.')

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
