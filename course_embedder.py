'''
Course Material Embedder

Embeds course materials (PDFs, code files, etc.) into Qdrant vector database
with group-based metadata for retrieval. Reads group definitions from JSON files
and creates embeddings using Qwen3-Embedding-4B model.

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
import torch
import PyPDF2
import re
from pathlib import Path
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
from qdrant_client.models import Distance, VectorParams, PointStruct


class CourseEmbedder:

    def __init__(self, qdrant_host: str = 'localhost', qdrant_port: int = 6333, collection_name: str = 'course_materials', exclude_extensions: List[str] = None, exclude_filenames: List[str] = None, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize CourseEmbedder with semantic chunking support.

        Args:
            chunk_size: Target tokens per chunk (default: 500 ~= 700-800 words)
            chunk_overlap: Token overlap between chunks (default: 100 ~= 140 words)
        """
        self.exclude_extensions = exclude_extensions or []
        self.exclude_filenames = exclude_filenames or []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        if self.device.type == 'cuda':
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-4B')
        self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-4B')
        self.model.to(self.device)
        self.model.eval()

        self.client = QdrantClient(host = qdrant_host, port = qdrant_port)
        self.collection_name = collection_name

        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name = collection_name,
                vectors_config = VectorParams(size = 2560, distance = Distance.COSINE)
            )

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
        2. Group sentences into chunks targeting self.chunk_size tokens
        3. Overlap chunks by self.chunk_overlap tokens for context continuity

        Returns:
            List of text chunks
        """
        if not text.strip():
            return []

        # Split into sentences (simple but effective regex)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            # Tokenize to count tokens
            sentence_tokens = len(self.tokenizer.encode(sentence, add_special_tokens=False))

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))

                # Start new chunk with overlap
                # Keep sentences from the end totaling ~chunk_overlap tokens
                overlap_chunk = []
                overlap_tokens = 0
                for sent in reversed(current_chunk):
                    sent_tokens = len(self.tokenizer.encode(sent, add_special_tokens=False))
                    if overlap_tokens + sent_tokens > self.chunk_overlap:
                        break
                    overlap_chunk.insert(0, sent)
                    overlap_tokens += sent_tokens

                current_chunk = overlap_chunk
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            print('Warning: Empty text provided for embedding')
            return [0.0] * 2560

        inputs = self.tokenizer(text, return_tensors = 'pt', truncation = True, max_length = 8192, padding = True)

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        return embedding.cpu().numpy().tolist()

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

                file_ext = os.path.splitext(file_path)[1]
                if file_ext in self.exclude_extensions:
                    print(f'- SKIPPED (extension {file_ext}): {filename}')
                    continue

                if filename in self.exclude_filenames:
                    print(f'- SKIPPED (filename): {filename}')
                    continue

                print(f'- Embedding {file_info['type']}: {filename}')

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
        try:
            collection_info = self.client.get_collection(self.collection_name)
            points_count = collection_info.points_count
            return points_count
        except:
            return 0


if __name__ == '__main__':
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
