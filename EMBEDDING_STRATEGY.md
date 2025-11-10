# Embedding Strategy for Multi-Subject Course Materials

## Overview

This document outlines the strategy for embedding course materials across 7 subjects into separate vector databases.

## Subject Collections

Create **7 separate collections** in Qdrant:

| Subject | Collection Name | Data Path |
|---------|----------------|-----------|
| Computer Science | `cs_materials` | `data/computer_science/` |
| Physics | `physics_materials` | `data/physics/` |
| Chemistry | `chemistry_materials` | `data/chemistry/` |
| Mathematics | `mathematics_materials` | `data/mathematics/` |
| Biology | `biology_materials` | `data/biology/` |
| Electrical Engineering | `ee_materials` | `data/electrical_engineering/` |
| Mechanical Engineering | `me_materials` | `data/mechanical_engineering/` |

## Why Separate Collections?

✅ **Pros:**
- Clean separation by subject
- Easier to manage and update individual subjects
- Can search within specific subject when user specifies
- Better organization and debugging
- Independent scaling per subject

❌ **Cons:**
- Need to know subject beforehand (but likely known from user query)
- Multiple embedding operations (one per subject)

## Embedding Script

Create a script to embed all subjects:

```python
# embed_all_subjects.py
from course_embedder import CourseEmbedder

SUBJECTS = [
    ("cs_materials", "data/computer_science/6_0001/groups.json"),
    ("physics_materials", "data/physics/groups.json"),
    ("chemistry_materials", "data/chemistry/groups.json"),
    ("mathematics_materials", "data/mathematics/groups.json"),
    ("biology_materials", "data/biology/groups.json"),
    ("ee_materials", "data/electrical_engineering/groups.json"),
    ("me_materials", "data/mechanical_engineering/groups.json"),
]

for collection_name, groups_path in SUBJECTS:
    print(f"\n{'='*60}")
    print(f"Embedding {collection_name}...")
    print(f"{'='*60}\n")

    embedder = CourseEmbedder(
        qdrant_host="localhost",
        qdrant_port=6333,
        collection_name=collection_name,
        exclude_extensions=['.py', '.txt', '.js', '.java', '.cpp']
    )

    embedder.embed_course(
        mappings_json_path=groups_path,
        base_path="."
    )

    print(f"✓ Completed {collection_name}\n")

print("\n" + "="*60)
print("ALL SUBJECTS EMBEDDED SUCCESSFULLY!")
print("="*60)
```

## Query Strategy

### Option 1: Subject-Specific Query (Recommended)

When user specifies subject in query:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

def query_subject(question, subject="cs"):
    """Query specific subject collection"""
    collection_map = {
        "cs": "cs_materials",
        "physics": "physics_materials",
        "chemistry": "chemistry_materials",
        "math": "mathematics_materials",
        "biology": "biology_materials",
        "ee": "ee_materials",
        "me": "me_materials"
    }

    collection_name = collection_map.get(subject, "cs_materials")

    # Embed question
    query_vector = embedder.embed_text(question)

    # Search specific collection
    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=1
    )

    return results
```

### Option 2: Multi-Subject Query

Search across multiple subjects if unclear:

```python
def query_all_subjects(question, top_k=1):
    """Query all subject collections and return best match"""
    query_vector = embedder.embed_text(question)

    all_results = []

    for collection in ["cs_materials", "physics_materials", ...]:
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k
        )

        for result in results:
            result.payload["_source_collection"] = collection
            all_results.append(result)

    # Sort by score
    all_results.sort(key=lambda x: x.score, reverse=True)

    return all_results[:top_k]
```

## Retrieval with Group Context

After finding the top match, retrieve all siblings:

```python
def get_group_context(client, collection_name, top_result):
    """Get all files in the same group as top result"""
    group_id = top_result.payload["group_id"]

    # Fetch all siblings
    siblings, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "group_id", "match": {"value": group_id}}
            ]
        },
        limit=100
    )

    # Load file contents
    context_files = []
    for sibling in siblings:
        file_path = sibling.payload["file_path"]
        # Load actual file content here
        content = load_file(file_path)
        context_files.append({
            "type": sibling.payload["file_type"],
            "path": file_path,
            "content": content
        })

    return context_files
```

## Storage Estimates

Assuming:
- Average course: 20 lectures + 10 assignments = 30 groups
- Average group: 2 files (transcript + slides) = 60 documents per course
- Vector size: 2560 floats × 4 bytes = 10KB per document
- Metadata: ~2KB per document

**Per course:** 60 docs × 12KB = ~720KB
**Per subject:** ~5 courses = 3.6MB
**Total (7 subjects):** ~25MB

Very manageable! Even with 100 courses across all subjects, you'd only use ~250MB.

## Workflow Summary

1. **Prepare groups.json** for each subject
2. **Run embedding script** for all subjects
3. **Verify embeddings** using visualization or table export
4. **Query** by subject or across all subjects
5. **Retrieve group context** for top match
6. **Pass to LLM** for answer generation

## Monitoring Collections

Check collection status:

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

collections = client.get_collections()
for collection in collections.collections:
    info = client.get_collection(collection.name)
    print(f"{collection.name}: {info.points_count} documents")
```

## Next Steps

1. Create `groups.json` files for each subject
2. Set up Qdrant (Docker recommended)
3. Run embedding script
4. Visualize to verify structure
5. Build query interface
