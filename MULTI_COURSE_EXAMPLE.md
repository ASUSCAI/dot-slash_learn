# Multi-Course Embedding Example

## How It Works

The updated [embed_all_subjects.py](embed_all_subjects.py:1) now supports embedding **multiple courses per subject** into the same collection.

## Configuration

Each subject in `SUBJECT_CONFIG` now has a `groups_paths` list (instead of single `groups_path`):

```python
SUBJECT_CONFIG = {
    'cs': {
        'collection': 'cs_materials',
        'groups_paths': [  # LIST of groups.json files
            'data/computer_science/6_0001/groups.json',
            'data/computer_science/6_006/groups.json',
            'data/computer_science/6_046J/groups.json',
            'data/computer_science/6_S096/groups.json',
        ],
        'display_name': 'Computer Science'
    }
}
```

## What Happens When You Run It

```bash
python embed_all_subjects.py --subjects cs
```

**Output:**
```
======================================================================
Embedding Computer Science (cs)
======================================================================

Found 4 course(s) to embed:
  - data/computer_science/6_0001/groups.json
  - data/computer_science/6_006/groups.json
  - data/computer_science/6_046J/groups.json
  - data/computer_science/6_S096/groups.json

  Embedding course: 6_0001 (groups.json)
  - Embedding transcript: lecture1_transcript.pdf
  - Embedding notes: lecture1_slides.pdf
  ...
  ✓ Completed 6_0001

  Embedding course: 6_006 (groups.json)
  - Embedding transcript: lec1_transcript.pdf
  ...
  ✓ Completed 6_006

  Embedding course: 6_046J (groups.json)
  ...
  ✓ Completed 6_046J

  Embedding course: 6_S096 (groups.json)
  ...
  ✓ Completed 6_S096

✓ Successfully embedded 4 course(s) for Computer Science
```

## Result

All 4 courses are now in **one collection** (`cs_materials`):

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)
info = client.get_collection("cs_materials")

print(f"cs_materials has {info.points_count} documents")
# cs_materials has 240 documents (60 per course × 4 courses)
```

## Benefits

1. **Single search** across all CS courses
2. **Easier querying** - don't need to know which specific course
3. **Better organization** - subject-level collections
4. **Flexible** - can add/remove courses easily

## Example Queries

### Query Across All CS Courses

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Search across ALL CS courses at once
results = client.search(
    collection_name="cs_materials",
    query_vector=query_vector,
    limit=5
)

# Results might come from any of the 4 courses
for result in results:
    print(f"Course: {result.payload['course']}")
    print(f"Group: {result.payload['group_id']}")
    print(f"File: {result.payload['file_path']}")
    print(f"Score: {result.score}\n")
```

### Query Specific Course Within Subject

```python
# Only search 6.0001 materials
results = client.search(
    collection_name="cs_materials",
    query_vector=query_vector,
    query_filter={
        "must": [
            {"key": "course", "match": {"value": "6_0001"}}
        ]
    },
    limit=5
)
```

## Adding New Courses

To add a new course to Computer Science:

1. **Create the groups.json** file:
   ```
   data/computer_science/6_042/groups.json
   ```

2. **Update SUBJECT_CONFIG**:
   ```python
   'cs': {
       'collection': 'cs_materials',
       'groups_paths': [
           'data/computer_science/6_0001/groups.json',
           'data/computer_science/6_006/groups.json',
           'data/computer_science/6_046J/groups.json',
           'data/computer_science/6_S096/groups.json',
           'data/computer_science/6_042/groups.json',  # NEW!
       ],
       'display_name': 'Computer Science'
   }
   ```

3. **Re-run embedding**:
   ```bash
   python embed_all_subjects.py --subjects cs
   ```

The new course will be added to the existing `cs_materials` collection!

## Backward Compatibility

The code still supports the old single-path format:

```python
# Old format (still works)
'biology': {
    'collection': 'biology_materials',
    'groups_path': 'data/biology/groups.json',  # Single path
    'display_name': 'Biology'
}

# New format (recommended)
'biology': {
    'collection': 'biology_materials',
    'groups_paths': [                            # List of paths
        'data/biology/7_012/groups.json',
        'data/biology/7_014/groups.json',
    ],
    'display_name': 'Biology'
}
```

## Directory Structure Example

```
data/
├── computer_science/
│   ├── 6_0001/
│   │   └── groups.json          ← Course 1
│   ├── 6_006/
│   │   └── groups.json          ← Course 2
│   ├── 6_046J/
│   │   └── groups.json          ← Course 3
│   └── 6_S096/
│       └── groups.json          ← Course 4
│
└── physics/
    ├── 8_01/
    │   └── groups.json          ← Physics Course 1
    └── 8_02/
        └── groups.json          ← Physics Course 2
```

All CS courses → `cs_materials` collection
All Physics courses → `physics_materials` collection

## Metadata Structure

Each embedded document includes the course identifier:

```json
{
  "file_path": "data/computer_science/6_0001/lectures/lec1.pdf",
  "file_type": "transcript",
  "group_id": "6_0001_lecture_1",
  "group_type": "lecture",
  "course": "6_0001",              ← Course identifier
  "subject": "computer_science",   ← Inferred from path
  "sibling_files": [...],
  ...
}
```

This allows you to:
- Search across all courses in a subject
- Filter by specific course if needed
- Track which course a document belongs to