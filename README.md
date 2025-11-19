# dot-slash_learn

RAG-LLM Query API for course materials.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set Qdrant connection (optional, defaults to localhost:6333)
export QDRANT_HOST=localhost
export QDRANT_PORT=6333
```

## Run

```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### Embeddings & Jetstream Toggles

Embeddings now use the lighter-weight [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) sentence transformer. Override the model by exporting `EMBEDDING_MODEL_NAME` before starting the API.

Jetstream still handles guardrails, reranking, and text generation. If you set `JETSTREAM_DISABLE_LOCAL_MODELS=1`, embeddings continue to run locally (Jetstream2 does not expose an embedding service) while other components switch to Jetstream-hosted models. Fine-grained controls remain available:

- `JETSTREAM_REMOTE_RERANKER`
- `JETSTREAM_REMOTE_LLM`
- `JETSTREAM_GUARD_MODEL` / related base-url overrides

## Usage

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is dynamic programming?",
    "collection_name": "cs_materials"
  }'
```

**Python:**

```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={"query": "What is recursion?", "collection_name": "cs_materials"}
)

print(response.json()["answer"])
```

## API

**POST /api/v1/query**

```json
{
  "query": "string (required)",
  "collection_name": "string (default: cs_materials)",
  "show_context": "boolean (default: false)",
  "max_length": "integer (default: 2048)",
  "enable_guardrails": "boolean (default: true)"
}
```

**GET /health**

Check if API and Qdrant are connected.
