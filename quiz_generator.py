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
import torch
import asyncio
from typing import List, Dict, Any, Optional
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
        question_style: str = "concrete"
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
        print('FREEING VRAM: Offloading embedding model to CPU...')
        print('='*80)
        self._offload_embedding_model()

        # Phase 1: Retrieve chunks and generate questions
        print('\n' + '='*80)
        print('PHASE 1: Retrieving chunks and generating questions')
        print('='*80)

        chunks = self._retrieve_chunks_by_skill(skill)

        if not chunks:
            print(f'\nWarning: No chunks found for skill "{skill}"')
            return []

        print(f'\nRetrieved {len(chunks)} chunks for skill "{skill}"')

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

        # Generate questions for each difficulty level with specific MCQ/T-F counts
        all_questions_data = []

        for difficulty, total, num_mcq_diff, num_tf_diff in difficulty_distribution:
            questions = self._generate_questions_with_llm(
                chunks, learning_objective, skill,
                num_mcq=num_mcq_diff,
                num_tf=num_tf_diff,
                difficulty=difficulty
            )
            all_questions_data.extend(questions)

        if not all_questions_data:
            print('\nWarning: Failed to generate questions')
            # Restore embedding model before returning
            self._restore_embedding_model()
            return []

        print(f'\nGenerated {len(all_questions_data)} questions total')

        # Phase 2: Generate explanations in parallel
        print('\n' + '='*80)
        print('PHASE 2: Generating explanations for all options (parallel)')
        print('='*80)

        questions_with_explanations = await self._generate_all_explanations(
            all_questions_data, chunks, skill, question_style
        )

        print('\n' + '='*80)
        print(f'QUIZ GENERATION COMPLETE: {len(questions_with_explanations)} questions')
        print('='*80)

        # VRAM CLEANUP: Offload LLM and restore embedding model
        print('\n' + '='*80)
        print('CLEANUP: Offloading LLM and restoring embedding model...')
        print('='*80)
        self._cleanup_vram()

        return questions_with_explanations

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

    def _generate_questions_with_llm(
        self,
        chunks: List[Dict[str, Any]],
        learning_objective: str,
        skill: str,
        num_mcq: int,
        num_tf: int,
        difficulty: str = 'medium'
    ) -> List[Dict[str, Any]]:
        """
        Generate questions using LLM (Phase 1).

        Args:
            chunks: Retrieved chunks
            learning_objective: Learning objective
            skill: Target skill
            num_mcq: Number of MCQ questions to generate
            num_tf: Number of True/False questions to generate
            difficulty: Difficulty level ('easy', 'medium', or 'hard')

        Returns:
            List of question data (without explanations yet)
        """
        num_questions = num_mcq + num_tf

        # Define difficulty guidance for LLM
        difficulty_guides = {
            'easy': 'Easy questions should test basic recall and understanding of fundamental concepts. Use straightforward language and test direct knowledge from the materials.',
            'medium': 'Medium questions should require applying concepts and making connections. Test understanding beyond simple recall.',
            'hard': 'Hard questions should involve complex analysis, synthesis of multiple concepts, or subtle edge cases. These should challenge deep understanding.'
        }

        print(f'\nGenerating {num_mcq} MCQ and {num_tf} True/False questions at {difficulty.upper()} difficulty')

        # Combine chunks into context
        context = self._format_chunks_for_context(chunks, skill)

        prompt = f"""Based on the course materials below, generate {num_questions} quiz questions to test understanding of "{skill}".

Learning Objective: {learning_objective}
Target Skill: {skill}
Difficulty Level: {difficulty.upper()}

{difficulty_guides[difficulty]}

Course Materials:
{context}

Generate exactly:
- {num_mcq} Multiple Choice Questions (4 options each: A, B, C, D)
- {num_tf} True/False Questions

Requirements for each question:
1. Focus on testing understanding of "{skill}"
2. Base questions directly on the course materials provided
3. Make questions clear and unambiguous
4. Ensure only one correct answer per question
5. Make incorrect options plausible but clearly wrong
6. **IMPORTANT**: Adjust question complexity to match the {difficulty.upper()} difficulty level
7. **CRITICAL**: Keep answer options SHORT and CONCISE (5-10 words maximum per option)
   - Use brief, direct phrases
   - Avoid lengthy explanations in options
   - Get straight to the point

Return ONLY a valid JSON array in this exact format (no additional text):
[
  {{
    "type": "mcq",
    "question": "What is the primary purpose of dynamic programming?",
    "options": [
      {{"id": "A", "text": "Random problem solving"}},
      {{"id": "B", "text": "Optimize recursion with memoization"}},
      {{"id": "C", "text": "Obfuscate code"}},
      {{"id": "D", "text": "Eliminate all loops"}}
    ],
    "correct_answer": "B"
  }},
  {{
    "type": "true_false",
    "question": "Dynamic programming requires memoization in all cases.",
    "options": [
      {{"id": "true", "text": "True"}},
      {{"id": "false", "text": "False"}}
    ],
    "correct_answer": "false"
  }}
]

JSON Array:"""

        # Load LLM and generate
        self.embedder._load_llm()

        print('\nCalling LLM to generate questions...')
        response = self._call_llm_sync(prompt, max_tokens=1500)

        # Parse JSON response
        try:
            # Extract JSON array from response
            response = response.strip()

            # Find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                questions = json.loads(json_str)

                # Add difficulty level to each question
                for question in questions:
                    question['difficulty'] = difficulty

                print(f'\nSuccessfully parsed {len(questions)} {difficulty} questions from LLM response')
                return questions
            else:
                print(f'\nWarning: Could not find JSON array in LLM response')
                print(f'Response: {response[:500]}...')
                return []

        except json.JSONDecodeError as e:
            print(f'\nWarning: Failed to parse JSON from LLM response: {e}')
            print(f'Response: {response[:500]}...')
            return []

    async def _generate_all_explanations(
        self,
        questions_data: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]],
        skill: str,
        question_style: str = "concrete"
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for all options in parallel (Phase 2).

        Args:
            questions_data: Questions without explanations
            chunks: Retrieved chunks for context
            skill: Target skill
            question_style: Explanation style - 'concrete' or 'abstract'

        Returns:
            Complete questions with explanations
        """
        context = self._format_chunks_for_context(chunks, skill)

        # Create tasks for all option explanations
        tasks = []
        task_metadata = []  # Track which question/option each task corresponds to

        for q_idx, question in enumerate(questions_data):
            for option in question['options']:
                is_correct = (option['id'] == question['correct_answer'])

                task = self._generate_single_explanation(
                    question=question['question'],
                    option_id=option['id'],
                    option_text=option['text'],
                    is_correct=is_correct,
                    context=context,
                    skill=skill,
                    question_style=question_style
                )
                tasks.append(task)
                task_metadata.append({
                    'q_idx': q_idx,
                    'option_id': option['id']
                })

        print(f'\nGenerating {len(tasks)} explanations in parallel...')

        # Run all tasks in parallel
        explanations = await asyncio.gather(*tasks)

        print(f'Generated {len(explanations)} explanations')

        # Map explanations back to questions
        return self._combine_explanations_with_questions(
            questions_data, explanations, task_metadata
        )

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

Provide a clear, concise explanation (2-3 sentences) for why this answer is {correctness}.
Focus on educational value - help the student understand the concept of "{skill}".
Reference specific details from the course materials when possible.

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

    def _combine_explanations_with_questions(
        self,
        questions_data: List[Dict[str, Any]],
        explanations: List[str],
        task_metadata: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine generated explanations back into question structure.

        Args:
            questions_data: Questions without explanations
            explanations: Generated explanations
            task_metadata: Metadata mapping explanations to questions/options

        Returns:
            Complete questions with explanations
        """
        # Build explanation map: {q_idx: {option_id: explanation}}
        explanation_map = {}
        for metadata, explanation in zip(task_metadata, explanations):
            q_idx = metadata['q_idx']
            option_id = metadata['option_id']

            if q_idx not in explanation_map:
                explanation_map[q_idx] = {}

            explanation_map[q_idx][option_id] = explanation

        # Build final questions list
        final_questions = []

        for q_idx, question_data in enumerate(questions_data):
            # Build options with explanations and correctness
            options = []
            for option in question_data['options']:
                option_id = option['id']
                explanation = explanation_map.get(q_idx, {}).get(option_id, '')
                is_correct = (option_id == question_data['correct_answer'])

                options.append({
                    'id': option_id,
                    'text': option['text'],
                    'explanation': explanation,
                    'is_correct': is_correct
                })

            final_questions.append({
                'id': f'q{q_idx + 1}',
                'type': question_data['type'],
                'question': question_data['question'],
                'options': options,
                'skill': question_data.get('skill', ''),
                'difficulty': question_data.get('difficulty', 'medium')
            })

        return final_questions

    def _offload_embedding_model(self):
        """Offload embedding model to CPU to free VRAM for LLM"""
        try:
            if hasattr(self.embedder, 'model') and self.embedder.model is not None:
                print('  Offloading embedding model to CPU...')
                self.embedder.model.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print('  ✓ Embedding model offloaded, VRAM freed')
        except Exception as e:
            print(f'  Warning: Could not offload embedding model: {e}')

    def _restore_embedding_model(self):
        """Restore embedding model back to GPU"""
        try:
            if hasattr(self.embedder, 'model') and self.embedder.model is not None:
                print('  Restoring embedding model to GPU...')
                self.embedder.model.to(self.embedder.device)
                print('  ✓ Embedding model restored to GPU')
        except Exception as e:
            print(f'  Warning: Could not restore embedding model: {e}')

    def _cleanup_vram(self):
        """Cleanup VRAM by offloading LLM and restoring embedding model"""
        try:
            # Offload LLM to CPU
            if self.embedder.llm is not None:
                print('  Offloading LLM to CPU...')
                self.embedder.llm.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print('  ✓ LLM offloaded to CPU')

            # Restore embedding model to GPU
            self._restore_embedding_model()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print('  ✓ VRAM cleanup complete')

        except Exception as e:
            print(f'  Warning: VRAM cleanup encountered issue: {e}')

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
