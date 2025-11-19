'''
Course Material Embedder

Embeds course materials (PDFs, code files, etc.) into Qdrant vector database
with group-based metadata for retrieval. Reads group definitions from JSON files
and creates embeddings using a local Alibaba-NLP/gte-large-en-v1.5 SentenceTransformer.

Usage:
    from course_embedder import CourseEmbedder

    embedder = CourseEmbedder(
        qdrant_host = 'localhost',
        qdrant_port = 6333,
        collection_name = 'course_materials',
        exclude_extensions=['.py', '.txt']
    )

    embedder.embed_course(
        mappings_json_path = 'data/computer_science/6_0001/groups.json',
        base_path = '.'
    )
'''

import os
import json
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import PyPDF2
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from jetstream_client import JetstreamInferenceClient, _get_env
from local_embedding import LocalEmbeddingClient


class CourseEmbedder:

    def __init__(self, qdrant_host: str = 'localhost', qdrant_port: int = 6333, collection_name: str = 'course_materials', exclude_extensions: List[str] = None, exclude_filenames: List[str] = None, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize CourseEmbedder with semantic chunking support.

        Args:
            chunk_size: Target words per chunk (default: 500)
            chunk_overlap: Word overlap between chunks (default: 100)
        """
        self.exclude_extensions = [ext.lower() for ext in (exclude_extensions or [])]
        self.exclude_filenames = set(exclude_filenames or [])
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        base_url = _get_env('JETSTREAM_BASE_URL')
        embed_api_key = _get_env('JETSTREAM_EMBED_API_KEY') or _get_env('JETSTREAM_API_KEY')
        chat_model = _get_env('JETSTREAM_MODEL')

        self.jetstream_client = JetstreamInferenceClient(
            base_url=base_url,
            model=chat_model,
            api_key=embed_api_key,
        )

        self.embedding_client = LocalEmbeddingClient()
        self.embedding_model_name = self.embedding_client.model_name
        self.vector_size = self.embedding_client.dimension

        print(
            'Local embedding model: '
            f"{self.embedding_model_name} (dim={self.vector_size}, device={self.embedding_client.device})"
        )

        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self._collection_lock = threading.Lock()

        self._ensure_collection_exists()

        # Compatibility placeholders for previous GPU-based pipeline
        self.model = self.embedding_client
        self.device = self.embedding_client.device

        # LLM toggles for downstream utilities
        self.llm = None
        self.llm_processor = None
        self.use_remote_llm = True
        self.use_remote_embedding = False
        self._llm_ready = False

    def _load_llm(self):
        """Lazy load the LLM model for skill extraction and classification"""
        if self.use_remote_llm and not getattr(self, '_llm_ready', False):
            print('\nUsing Jetstream-hosted LLM for skill extraction and content generation (no local weights to load).\n')
            self._llm_ready = True

    def generate_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 600,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate text using the shared Jetstream client."""
        self._load_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior teaching assistant who writes concise, factual educational content. "
                    "Follow the instructions exactly and do not include safety warnings unless requested."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        return self.jetstream_client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _extract_skills(self, combined_text: str, learning_objective: str, max_skills: int = 6) -> List[str]:
        """
        Extract high-level skills from course materials using LLM.

        Args:
            combined_text: Combined text from all documents
            learning_objective: The learning objective for these materials
            max_skills: Maximum number of skills to extract (default: 6)

        Returns:
            List of skill names
        """
        self._load_llm()

        if len(combined_text) > 15000:
            combined_text = combined_text[:15000] + "\n\n[...truncated for length...]"

        system_prompt = (
            "You read course materials and extract the key skills students will practice. "
            "Return only a JSON array of concise skill names ordered by importance."
        )

        user_prompt = (
            f"Learning objective: {learning_objective}\n\n"
            f"List up to {max_skills} distinct skills that are directly taught or required. "
            "Avoid generic abilities and focus on concrete technical competencies.\n\n"
            f"Course materials:\n{combined_text}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.jetstream_client.chat_completion_json(
                messages,
                max_tokens=256,
                temperature=0.2,
                top_p=0.9,
            )
        except RuntimeError as exc:
            print(f"Warning: Skill extraction request failed ({exc})")
            return []

        if isinstance(response, list):
            skills = response
        elif isinstance(response, dict):
            skills = response.get('skills', [])
        else:
            skills = []

        cleaned = []
        for skill in skills:
            if not isinstance(skill, str):
                continue
            candidate = skill.strip()
            if candidate and candidate.lower() not in {s.lower() for s in cleaned}:
                cleaned.append(candidate)
            if len(cleaned) >= max_skills:
                break

        return cleaned

    def _classify_chunk(self, chunk: str, skills: List[str]) -> List[str]:
        """
        Classify a chunk into one or more skills.

        Args:
            chunk: The text chunk to classify
            skills: List of available skills

        Returns:
            List of skill names that apply to this chunk
        """
        if not skills:
            return []

        self._load_llm()

        limited_chunk = chunk
        if len(limited_chunk) > 4000:
            limited_chunk = limited_chunk[:4000] + "..."

        system_prompt = (
            "Classify the text chunk into any matching skills. Respond with a JSON array containing only the skill names."
        )
        user_prompt = (
            "Available skills: " + json.dumps(skills) + "\n\n"
            "Text chunk:\n" + limited_chunk + "\n\n"
            "Return [] if none apply."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.jetstream_client.chat_completion_json(
                messages,
                max_tokens=120,
                temperature=0.1,
                top_p=0.9,
            )
        except RuntimeError as exc:
            print(f"Warning: Chunk classification failed ({exc})")
            return []

        classified_skills: List[str] = []
        if isinstance(response, list):
            classified_skills = [s for s in response if isinstance(s, str)]
        elif isinstance(response, dict):
            classified_skills = [s for s in response.get('skills', []) if isinstance(s, str)]

        valid_set = {skill.lower(): skill for skill in skills}
        filtered: List[str] = []
        for skill in classified_skills:
            normalized = skill.strip().lower()
            if normalized in valid_set:
                canonical = valid_set[normalized]
                if canonical not in filtered:
                    filtered.append(canonical)

        return filtered

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text() + '\n'
                return text.strip()
        except Exception as e:
            print(f'Error extracting PDF {pdf_path}: {e}')
            return ''

    def extract_text_from_code(self, code_path: str) -> str:
        try:
            with open(code_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            print(f'Error reading code file {code_path}: {e}')
            return ''

    def extract_text_from_file(self, file_path: str) -> str:
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.py'):
            return self.extract_text_from_code(file_path)
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                print(f'Error reading file {file_path}: {e}')
                return ''

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks using sentence boundaries.

        Strategy:
        1. Split by sentences (using periods, exclamation marks, question marks)
        2. Group sentences into chunks targeting self.chunk_size words
        3. Overlap chunks by self.chunk_overlap words for context continuity

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        words = text.split()
        if not words:
            return []

        chunks: List[str] = []
        chunk_words: List[str] = []

        for word in words:
            chunk_words.append(word)
            if len(chunk_words) >= self.chunk_size:
                chunks.append(' '.join(chunk_words))
                if self.chunk_overlap > 0:
                    chunk_words = chunk_words[-self.chunk_overlap:]
                else:
                    chunk_words = []

        if chunk_words:
            chunks.append(' '.join(chunk_words))

        return chunks

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            print('Warning: Empty text provided for embedding')
            return [0.0] * self.vector_size

        try:
            embeddings = self.embedding_client.embed(text)
        except Exception as exc:
            raise RuntimeError(f'Local embedding failed: {exc}') from exc

        if not embeddings:
            raise RuntimeError('Local embedder returned no vector')

        vector = embeddings[0]
        if len(vector) != self.vector_size:
            print(
                f'Warning: embedding dimension mismatch (expected {self.vector_size}, got {len(vector)}); '
                'updating cached dimension.'
            )
            self.vector_size = len(vector)

        return vector

    def load_mappings_from_json(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r') as f:
            return json.load(f)

    def embed_course(self, mappings_json_path: str, base_path: str = "."):
        groups = self.load_mappings_from_json(mappings_json_path)

        points = []
        point_id = self.get_next_point_id()

        for group_id, group_data in groups.items():
            if group_id.startswith('_TEMPLATE'):
                continue

            sibling_paths = [f['path'] for f in group_data['files']]

            for file_info in group_data['files']:
                file_path = os.path.join(base_path, file_info['path'])
                filename = os.path.basename(file_path)

                file_ext = os.path.splitext(file_path)[1].lower()
                if file_ext in self.exclude_extensions:
                    print(f'- SKIPPED (extension {file_ext}): {filename}')
                    continue

                if filename in self.exclude_filenames:
                    print(f'- SKIPPED (filename): {filename}')
                    continue

                print(f"- Embedding {file_info['type']}: {filename}")

                text_content = self.extract_text_from_file(file_path)

                if not text_content.strip():
                    print(f'    WARNING: No content extracted!')
                    continue

                print(f'    Content length: {len(text_content)} characters')

                # Chunk the document
                chunks = self.chunk_text(text_content)
                print(f'    Split into {len(chunks)} chunks')

                # Create a point for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    vector = self.embed_text(chunk)

                    payload = {
                        'file_path': file_info['path'],
                        'file_type': file_info['type'],
                        'description': file_info.get('description', ''),

                        'group_id': group_id,
                        'group_type': group_data['group_type'],
                        'course': group_data['course'],

                        'sibling_files': sibling_paths,
                        'sibling_count': len(sibling_paths),

                        # Chunk metadata
                        'chunk_index': chunk_idx,
                        'total_chunks': len(chunks),
                        'is_chunked': True,

                        # Store full chunk content for context
                        'full_content': chunk,
                        'content_preview': chunk[:300],
                        'content_length': len(chunk)
                    }

                    if group_data['group_type'] == 'lecture':
                        payload['lecture_number'] = group_data.get('lecture_number')
                        payload['title'] = group_data.get('title', '')
                    elif group_data['group_type'] == 'assignment':
                        payload['assignment_number'] = group_data.get('assignment_number')
                        payload['title'] = group_data.get('title', '')

                    point = PointStruct(id = point_id, vector = vector, payload = payload)

                    points.append(point)
                    point_id += 1

        if points:
            self._ensure_collection_exists()
            print(f'\n Uploading {len(points)} points to Qdrant...')
            batch_size = 10
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f'  Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}')

            print(f'Successfully embedded {len(points)} documents!')
        else:
            print('No points to upload!')

    def get_next_point_id(self) -> int:
        self._ensure_collection_exists()
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            return points_count
        except:
            return 0

    def _ensure_collection_exists(self) -> None:
        """Ensure the target collection exists before performing Qdrant operations."""
        with self._collection_lock:
            info = None
            try:
                info = self.client.get_collection(self.collection_name)
            except Exception:
                info = None

            if info is not None:
                existing_size = None
                try:
                    vectors_config = getattr(getattr(info, 'config', None), 'params', None)
                    if vectors_config is not None:
                        vectors = getattr(vectors_config, 'vectors', None)
                        if vectors is not None:
                            existing_size = getattr(vectors, 'size', None)
                except Exception:  # pragma: no cover - defensive guard against API changes
                    existing_size = None

                if existing_size is not None and existing_size != self.vector_size:
                    points_count = getattr(info, 'points_count', None)
                    if not points_count:
                        print(
                            f'Recreating empty collection {self.collection_name}: '
                            f'expected vector size {existing_size}, current embedder outputs {self.vector_size}.'
                        )
                        self.client.recreate_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                        )
                        print(f'Collection {self.collection_name} recreated with size {self.vector_size}')
                        return

                    raise RuntimeError(
                        f'Collection {self.collection_name} expects vector size {existing_size}, '
                        f'but local embeddings produce {self.vector_size}. '
                        'Create a new collection or reconfigure the existing one before embedding.'
                    )
                return

            print(f'Creating collection {self.collection_name}...')
            try:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
                )
                print(f'Collection {self.collection_name} created successfully')
            except Exception as exc:
                # Another worker may have created the collection after the initial check.
                try:
                    self.client.get_collection(self.collection_name)
                    print(f'Collection {self.collection_name} became available')
                except Exception:
                    raise exc

    # ============================================================================
    # NEW API-based embedding methods
    # ============================================================================

    def embed_files(
        self,
        file_paths: List[str],
        learning_objective: str,
        course: str = None,
        group_type: str = None,
        group_id: str = None,
        module_name: Optional[str] = None,
        module_item_ids: Optional[List[str]] = None,
        file_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        **extra_metadata
    ) -> Dict[str, Any]:
        """
        Embed a list of files into the collection with skill-based classification.

        Pipeline:
        1. Extract text from all files
        2. Use LLM to extract 5-6 high-level skills from combined content
        3. Chunk all documents
        4. Parallel processing:
           - Classify each chunk into skills (multithreaded LLM calls)
           - Generate embeddings for each chunk (GPU batched)
        5. Upload to Qdrant with skill metadata

        Args:
            file_paths: List of absolute file paths to embed
            learning_objective: Description of what these files teach
            course: Optional course identifier (e.g., "6_0001")
            group_type: Optional group type (e.g., "lecture", "assignment")
            group_id: Optional group identifier
            **extra_metadata: Additional metadata to store with chunks

        Returns:
            Dict with success status, number of chunks created, skills extracted, and any errors
        """
        errors = []

        # ===================================================================
        # STEP 1: Extract text from all files
        # ===================================================================
        print('='*80)
        print('STEP 1: Extracting text from files')
        print('='*80)

        file_contents = []  # List of (file_path, text_content, file_type)

        for file_path in file_paths:
            filename = os.path.basename(file_path)

            # Check if file exists
            if not os.path.exists(file_path):
                errors.append(f"File not found: {file_path}")
                continue

            print(f'- Reading: {filename}')

            # Extract text from file
            text_content = self.extract_text_from_file(file_path)

            if not text_content.strip():
                errors.append(f"No content extracted from: {file_path}")
                print(f'    WARNING: No content extracted!')
                continue

            print(f'    Content length: {len(text_content)} characters')

            # Detect file type from extension
            file_ext = os.path.splitext(file_path)[1].lower()
            file_type = 'code' if file_ext in ['.py', '.js', '.java', '.cpp', '.c'] else \
                       'pdf' if file_ext == '.pdf' else 'text'

            file_contents.append((file_path, text_content, file_type))

        if not file_contents:
            return {
                'success': False,
                'total_chunks': 0,
                'files_processed': 0,
                'errors': errors,
                'skills': []
            }

        # ===================================================================
        # STEP 2: Extract skills from combined content
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 2: Extracting skills from course materials')
        print('='*80)

        # Combine all text content
        combined_text = "\n\n".join([content for _, content, _ in file_contents])
        print(f'Combined content length: {len(combined_text)} characters')

        # Extract skills using LLM
        skills = self._extract_skills(combined_text, learning_objective, max_skills=6)

        if skills:
            print(f'\nExtracted {len(skills)} skills:')
            for i, skill in enumerate(skills, 1):
                print(f'  {i}. {skill}')
        else:
            print('\nWarning: No skills extracted. Proceeding without skill classification.')

        # ===================================================================
        # STEP 3: Chunk all documents
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 3: Chunking documents')
        print('='*80)

        all_chunks = []  # List of (file_path, chunk_text, chunk_idx, total_chunks, file_type)

        for file_path, text_content, file_type in file_contents:
            filename = os.path.basename(file_path)
            print(f'- Chunking: {filename}')

            chunks = self.chunk_text(text_content)
            print(f'    Split into {len(chunks)} chunks')

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append((file_path, chunk, chunk_idx, len(chunks), file_type))

        print(f'\nTotal chunks: {len(all_chunks)}')

        # ===================================================================
        # STEP 4: Parallel classification and embedding
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 4: Classifying chunks and generating embeddings (parallel)')
        print('='*80)

        chunk_skills_map = {}  # Map chunk index to skills
        chunk_embeddings_map = {}  # Map chunk index to embedding vector

        # Function to classify a single chunk
        def classify_chunk_worker(idx_and_chunk):
            idx, (file_path, chunk, chunk_idx, total_chunks, file_type) = idx_and_chunk
            chunk_skills = self._classify_chunk(chunk, skills) if skills else []
            return idx, chunk_skills

        # Parallel classification using ThreadPoolExecutor
        if skills:
            print(f'\nClassifying {len(all_chunks)} chunks into skills...')
            with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust workers based on available resources
                future_to_idx = {
                    executor.submit(classify_chunk_worker, (idx, chunk_data)): idx
                    for idx, chunk_data in enumerate(all_chunks)
                }

                completed = 0
                for future in as_completed(future_to_idx):
                    idx, chunk_skills = future.result()
                    chunk_skills_map[idx] = chunk_skills
                    completed += 1
                    if completed % 10 == 0 or completed == len(all_chunks):
                        print(f'  Classified {completed}/{len(all_chunks)} chunks')
        else:
            print('\nSkipping classification (no skills extracted)')
            for idx in range(len(all_chunks)):
                chunk_skills_map[idx] = []

        # Generate embeddings (can be done in parallel with classification, but we'll do it after for simplicity)
        print(f'\nGenerating embeddings for {len(all_chunks)} chunks...')
        for idx, (file_path, chunk, chunk_idx, total_chunks, file_type) in enumerate(all_chunks):
            vector = self.embed_text(chunk)
            chunk_embeddings_map[idx] = vector
            if (idx + 1) % 10 == 0 or (idx + 1) == len(all_chunks):
                print(f'  Embedded {idx + 1}/{len(all_chunks)} chunks')

        # ===================================================================
        # STEP 5: Create points and upload to Qdrant
        # ===================================================================
        print('\n' + '='*80)
        print('STEP 5: Uploading to Qdrant')
        print('='*80)

        points = []
        point_id = self.get_next_point_id()

        for idx, (file_path, chunk, chunk_idx, total_chunks, file_type) in enumerate(all_chunks):
            vector = chunk_embeddings_map[idx]
            chunk_skills = chunk_skills_map[idx]

            # Build metadata payload
            payload = {
                'file_path': file_path,
                'file_type': file_type,
                'learning_objective': learning_objective,
                'skills': chunk_skills,  # Add skills to metadata

                # Chunk metadata
                'chunk_index': chunk_idx,
                'total_chunks': total_chunks,
                'is_chunked': True,

                # Store full chunk content for context
                'full_content': chunk,
                'content_preview': chunk[:300],
                'content_length': len(chunk)
            }

            # Add optional fields
            if course:
                payload['course'] = course
            if group_type:
                payload['group_type'] = group_type
            if group_id:
                payload['group_id'] = group_id
            if module_name:
                payload['module_name'] = module_name
            if module_item_ids:
                payload['module_item_ids'] = module_item_ids

            metadata_for_file = None
            if file_metadata:
                metadata_for_file = file_metadata.get(file_path) or file_metadata.get(str(file_path))
            if metadata_for_file:
                payload.update(metadata_for_file)

            # Add any extra metadata
            payload.update(extra_metadata)

            point = PointStruct(id=point_id, vector=vector, payload=payload)
            points.append(point)
            point_id += 1

        # Upload to Qdrant in batches
        if points:
            self._ensure_collection_exists()
            print(f'\nUploading {len(points)} points to Qdrant...')
            batch_size = 10
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                print(f'  Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}')

            print(f'\n✓ Successfully embedded {len(points)} chunks from {len(file_contents)} files!')

        # Offload LLM to free memory
        return {
            'success': len(points) > 0,
            'total_chunks': len(points),
            'files_processed': len(file_contents),
            'errors': errors,
            'skills': skills
        }

    def collect_module_status(
        self,
        course: Optional[str] = None,
        group_type: Optional[str] = "module",
    ) -> List[Dict[str, Any]]:
        """Return aggregated status for embedded modules."""
        self._ensure_collection_exists()
        module_map: Dict[str, Dict[str, Any]] = {}
        items_key = "items"
        offset = None

        filter_conditions = []
        if course:
            filter_conditions.append(
                FieldCondition(key="course", match=MatchValue(value=course))
            )
        if group_type:
            filter_conditions.append(
                FieldCondition(key="group_type", match=MatchValue(value=group_type))
            )

        if filter_conditions:
            scroll_filter = Filter(must=filter_conditions)
        else:
            scroll_filter = None

        while True:
            scroll_kwargs: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "limit": 200,
                "with_payload": True,
                "with_vectors": False,
            }
            if scroll_filter is not None:
                scroll_kwargs["scroll_filter"] = scroll_filter
            if offset is not None:
                scroll_kwargs["offset"] = offset

            records, offset = self.client.scroll(**scroll_kwargs)
            if not records and offset is None:
                break

            for record in records or []:
                payload = record.payload or {}
                group_id = payload.get("group_id")
                if not group_id:
                    continue

                group_id_str = str(group_id)
                module_entry = module_map.setdefault(
                    group_id_str,
                    {
                        "group_id": group_id_str,
                        "module_name": payload.get("module_name"),
                        "learning_objective": payload.get("learning_objective"),
                        "file_paths": set(),
                        "absolute_file_paths": set(),
                        "skills": set(),
                        "module_item_ids": set(),
                        items_key: {},
                        "chunk_count": 0,
                        "ingested_at": payload.get("ingested_at"),
                    },
                )

                module_entry["chunk_count"] += 1

                requested_path = payload.get("requested_path") or payload.get("file_path")
                absolute_path = payload.get("file_path")
                if requested_path:
                    module_entry["file_paths"].add(str(requested_path))
                if absolute_path:
                    module_entry["absolute_file_paths"].add(str(absolute_path))

                for skill in payload.get("skills") or []:
                    module_entry["skills"].add(skill)

                module_item_id = payload.get("module_item_id")
                if module_item_id is not None:
                    item_id_str = str(module_item_id)
                    module_entry["module_item_ids"].add(item_id_str)
                    items_map: Dict[str, Dict[str, Any]] = module_entry[items_key]
                    item_entry = items_map.setdefault(
                        item_id_str,
                        {
                            "module_item_id": item_id_str,
                            "file_paths": set(),
                            "absolute_file_paths": set(),
                            "skills": set(),
                            "chunk_count": 0,
                        },
                    )
                    item_entry["chunk_count"] += 1
                    if requested_path:
                        item_entry["file_paths"].add(str(requested_path))
                    if absolute_path:
                        item_entry["absolute_file_paths"].add(str(absolute_path))
                    for skill in payload.get("skills") or []:
                        item_entry["skills"].add(skill)

                ingested_at = payload.get("ingested_at")
                if ingested_at:
                    current = module_entry.get("ingested_at")
                    if current is None or str(ingested_at) > str(current):
                        module_entry["ingested_at"] = ingested_at

            if offset is None:
                break

        results: List[Dict[str, Any]] = []
        for module_entry in module_map.values():
            items_map = module_entry.pop(items_key)
            module_entry["file_paths"] = sorted(module_entry["file_paths"])
            module_entry["absolute_file_paths"] = sorted(module_entry["absolute_file_paths"])
            module_entry["skills"] = sorted(module_entry["skills"])
            module_entry["module_item_ids"] = sorted(module_entry["module_item_ids"])
            module_entry["file_count"] = len(module_entry["file_paths"])

            items_list: List[Dict[str, Any]] = []
            for item_entry in items_map.values():
                items_list.append(
                    {
                        "module_item_id": item_entry["module_item_id"],
                        "file_paths": sorted(item_entry["file_paths"]),
                        "absolute_file_paths": sorted(item_entry["absolute_file_paths"]),
                        "skills": sorted(item_entry["skills"]),
                        "file_count": len(item_entry["file_paths"]),
                        "chunk_count": item_entry["chunk_count"],
                    }
                )

            items_list.sort(key=lambda entry: entry["module_item_id"] or "")
            module_entry["items"] = items_list
            results.append(module_entry)

        results.sort(key=lambda entry: entry.get("module_name") or entry["group_id"])
        return results

    def list_learning_objectives(
        self,
        course: Optional[str] = None,
        group_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return unique learning objectives stored in the collection."""
        self._ensure_collection_exists()

        objective_map: Dict[str, Dict[str, Any]] = {}
        offset = None

        filter_conditions = []
        if course:
            filter_conditions.append(
                FieldCondition(key="course", match=MatchValue(value=course))
            )
        if group_type:
            filter_conditions.append(
                FieldCondition(key="group_type", match=MatchValue(value=group_type))
            )

        scroll_filter = Filter(must=filter_conditions) if filter_conditions else None

        while True:
            scroll_kwargs: Dict[str, Any] = {
                "collection_name": self.collection_name,
                "limit": 200,
                "with_payload": True,
                "with_vectors": False,
            }
            if scroll_filter is not None:
                scroll_kwargs["scroll_filter"] = scroll_filter
            if offset is not None:
                scroll_kwargs["offset"] = offset

            records, offset = self.client.scroll(**scroll_kwargs)
            if not records and offset is None:
                break

            for record in records or []:
                payload = record.payload or {}
                raw_objective = payload.get("learning_objective")
                if raw_objective is None:
                    continue

                title = str(raw_objective).strip()
                if not title:
                    continue

                normalized = title.casefold()
                entry = objective_map.setdefault(
                    normalized,
                    {
                        "title": title,
                        "occurrence_count": 0,
                        "courses": set(),
                        "group_types": set(),
                        "sample_module_names": set(),
                        "last_ingested_at": None,
                    },
                )

                entry["occurrence_count"] += 1

                course_value = payload.get("course")
                if course_value:
                    entry["courses"].add(str(course_value))

                group_value = payload.get("group_type")
                if group_value:
                    entry["group_types"].add(str(group_value))

                module_name = payload.get("module_name")
                if isinstance(module_name, str):
                    stripped_module = module_name.strip()
                    if stripped_module:
                        entry["sample_module_names"].add(stripped_module)

                ingested_at = payload.get("ingested_at")
                if ingested_at is not None:
                    ingested_str = str(ingested_at)
                    current_best = entry["last_ingested_at"]
                    if current_best is None or ingested_str > current_best:
                        entry["last_ingested_at"] = ingested_str

            if offset is None:
                break

        results: List[Dict[str, Any]] = []
        for objective_entry in objective_map.values():
            sample_modules = sorted(objective_entry["sample_module_names"])
            if len(sample_modules) > 10:
                sample_modules = sample_modules[:10]
            results.append(
                {
                    "title": objective_entry["title"],
                    "occurrence_count": objective_entry["occurrence_count"],
                    "courses": sorted(objective_entry["courses"]),
                    "group_types": sorted(objective_entry["group_types"]),
                    "sample_module_names": sample_modules,
                    "last_ingested_at": objective_entry["last_ingested_at"],
                }
            )

        results.sort(key=lambda entry: entry["title"].casefold())
        return results

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """
        Delete all chunks associated with a specific file from the collection.

        Args:
            file_path: Exact file path to match and delete

        Returns:
            Dict with success status and number of chunks deleted
        """
        self._ensure_collection_exists()
        try:
            # Search for all points with matching file_path
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path)
                        )
                    ]
                ),
                limit=10000  # Adjust if you expect more chunks per file
            )

            points_to_delete = search_result[0]
            point_ids = [point.id for point in points_to_delete]

            if not point_ids:
                return {
                    'success': True,
                    'chunks_deleted': 0,
                    'message': f'No chunks found for file: {file_path}'
                }

            # Delete the points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids
            )

            print(f'Deleted {len(point_ids)} chunks for file: {file_path}')

            return {
                'success': True,
                'chunks_deleted': len(point_ids),
                'message': f'Successfully deleted {len(point_ids)} chunks'
            }

        except Exception as e:
            print(f'Error deleting file {file_path}: {e}')
            return {
                'success': False,
                'chunks_deleted': 0,
                'error': str(e)
            }


# ============================================================================
# OLD JSON-based embedding logic (commented out for API-based system)
# ============================================================================
# if __name__ == '__main__':
#     embedder = CourseEmbedder(
#         qdrant_host = 'localhost',
#         qdrant_port = 6333,
#         collection_name = 'course_materials',
#         exclude_extensions=['.py', '.txt']
#     )
#
#     embedder.embed_course(
#         mappings_json_path = 'data/computer_science/6_0001/groups.json',
#         base_path = '.'
#     )
