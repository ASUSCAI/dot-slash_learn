# Course Material Groups - Quick Reference

## Overview

This system allows you to group related course materials (lectures, assignments, etc.) so that when one file is retrieved via RAG, all related files in the same group are provided to the LLM.

## File Structure

```
data/
├── groups_template.json          # Template with examples
└── computer_science/
    └── 6_0001/
        └── groups.json            # Course-specific groups
```

## Creating Groups

### 1. Copy Template

Use the patterns from `groups_template.json` to create groups.

### 2. Group ID Naming Convention

- **Lectures**: `{course}_lecture_{number}`
  - Example: `6_0001_lecture_1`

- **Assignments**: `{course}_ps{number}` or `{course}_{assignment_name}`
  - Example: `6_0001_ps0`, `6_0001_ps1`

- **Reference Materials**: `{course}_reference_{name}`
  - Example: `6_0001_reference_style_guide`

### 3. File Types

Common file types to use in the `type` field:

- **Lectures**:
  - `transcript` - Lecture transcript PDF
  - `notes` - Lecture slides/notes PDF
  - `code` - Code examples (.py, .js, etc.)

- **Assignments**:
  - `assignment` - Main assignment instructions
  - `supporting` - Supporting documents
  - `test` - Test files
  - `solution` - Solution files (if applicable)

- **Reference**:
  - `reference` - Style guides, additional resources

### 4. Required Fields

```json
{
  "group_id": {
    "group_type": "lecture|assignment|reference",  // REQUIRED
    "course": "COURSE_CODE",                       // REQUIRED
    "files": [                                      // REQUIRED
      {
        "path": "relative/path/to/file",           // REQUIRED
        "type": "file_type",                       // REQUIRED
        "description": "Optional description"      // OPTIONAL
      }
    ]
  }
}
```

### 5. Optional Fields

Add these for better organization:

- **Lectures**: `lecture_number`, `title`
- **Assignments**: `assignment_number`, `title`
- Any other metadata you find useful

## Example: Adding a New Lecture

```json
{
  "6_0001_lecture_13": {
    "group_type": "lecture",
    "course": "6_0001",
    "lecture_number": 13,
    "title": "Searching and Sorting",
    "files": [
      {
        "path": "data/computer_science/6_0001/lectures/lec13_transcript.pdf",
        "type": "transcript",
        "description": "Lecture 13 transcript"
      },
      {
        "path": "data/computer_science/6_0001/lectures/lec13_slides.pdf",
        "type": "notes",
        "description": "Lecture 13 slides"
      },
      {
        "path": "data/computer_science/6_0001/lectures/lec13_code.py",
        "type": "code",
        "description": "Sorting algorithms examples"
      }
    ]
  }
}
```

## Example: Assignment with Multiple Files

```json
{
  "6_0001_ps2": {
    "group_type": "assignment",
    "course": "6_0001",
    "assignment_number": 2,
    "title": "Problem Set 2 - Hangman",
    "files": [
      {
        "path": "data/computer_science/6_0001/assignments/ps2_instructions.pdf",
        "type": "assignment",
        "description": "Main instructions"
      },
      {
        "path": "data/computer_science/6_0001/assignments/ps2_starter.py",
        "type": "supporting",
        "description": "Starter code"
      },
      {
        "path": "data/computer_science/6_0001/assignments/ps2_tests.py",
        "type": "test",
        "description": "Test cases"
      },
      {
        "path": "data/computer_science/6_0001/assignments/words.txt",
        "type": "supporting",
        "description": "Word list for hangman"
      }
    ]
  }
}
```

## Setting Up Qdrant

### Option 1: Using Docker (Recommended)

**Step 1: Install Docker**
- Mac: Download from [docker.com](https://www.docker.com/products/docker-desktop)
- Linux: `sudo apt-get install docker.io` or `brew install docker`

**Step 2: Start Qdrant**
```bash
# Pull and run Qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# Verify it's running
curl http://localhost:6333
```

**Step 3: Check Qdrant Dashboard**
- Open browser: http://localhost:6333/dashboard
- You should see the Qdrant web UI

### Option 2: Using Qdrant Cloud (Free Tier)

**Step 1: Create Account**
1. Go to [cloud.qdrant.io](https://cloud.qdrant.io)
2. Sign up for free account (1GB free tier)
3. Create a new cluster

**Step 2: Get API Key**
1. Copy your cluster URL (e.g., `https://xyz.cloud.qdrant.io`)
2. Copy your API key

**Step 3: Update Code**
```python
from qdrant_client import QdrantClient

# Use cloud credentials
embedder = CourseEmbedder(
    qdrant_host="https://xyz.cloud.qdrant.io",
    qdrant_port=6333,
    collection_name="course_materials"
)

# Or use API key
client = QdrantClient(
    url="https://xyz.cloud.qdrant.io",
    api_key="your-api-key-here"
)
```

### Option 3: Local Installation (No Docker)

```bash
# Download and extract
wget https://github.com/qdrant/qdrant/releases/download/v1.7.4/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar -xvf qdrant-x86_64-unknown-linux-gnu.tar.gz

# Run Qdrant
./qdrant
```

## Using the Embedding Script

### 1. Install Dependencies

```bash
pip install qdrant-client transformers torch PyPDF2
```

### 2. Verify Qdrant is Running

```bash
# Test connection
curl http://localhost:6333/collections
```

### 3. Embed Your Course

#### Single Course
```python
from course_embedder import CourseEmbedder

embedder = CourseEmbedder(
    qdrant_host="localhost",
    qdrant_port=6333,
    collection_name="cs_materials",
    exclude_extensions=['.py', '.txt']  # Skip code files
)

embedder.embed_course(
    mappings_json_path="data/computer_science/6_0001/groups.json",
    base_path="."
)
```

#### Multiple Subjects (Separate Collections)
```python
from course_embedder import CourseEmbedder

# Physics materials
physics_embedder = CourseEmbedder(
    collection_name="physics_materials",
    exclude_extensions=['.py', '.txt']
)
physics_embedder.embed_course("data/physics/groups.json")

# Chemistry materials
chemistry_embedder = CourseEmbedder(
    collection_name="chemistry_materials",
    exclude_extensions=['.py', '.txt']
)
chemistry_embedder.embed_course("data/chemistry/groups.json")

# Computer Science materials
cs_embedder = CourseEmbedder(
    collection_name="cs_materials",
    exclude_extensions=['.py', '.txt']
)
cs_embedder.embed_course("data/computer_science/6_0001/groups.json")

# Math, Biology, EE, ME... (follow same pattern)
```

**Why separate collections?**
- Clean separation by subject
- Easier to manage and update
- Can search within specific subject when needed
- Better organization

### 4. Verify Embeddings

```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Check collection info
info = client.get_collection("cs_materials")
print(f"Points in collection: {info.points_count}")

# List all collections
collections = client.get_collections()
for collection in collections.collections:
    print(f"- {collection.name}: {collection.points_count} points")
```

## Tips

1. **Keep paths relative** to your project root for portability
2. **Use consistent naming** for group_ids across courses
3. **Add descriptions** to help you remember what each file contains
4. **Test incrementally** - start with one group, embed it, query it, then add more
5. **Multiple groups.json files** - You can have one per course or combine them

## Metadata Available in Qdrant

When you query, each result contains:

```python
result.payload = {
    "file_path": "...",
    "file_type": "transcript|notes|code|assignment|...",
    "group_id": "6_0001_lecture_1",
    "group_type": "lecture|assignment|reference",
    "course": "6_0001",
    "sibling_files": ["path1", "path2", ...],  # All files in this group
    "content_preview": "First 300 chars...",
    "content_length": 12345,
    # Plus any custom fields you added to groups.json
}
```

## Troubleshooting

**Empty content extracted from PDF?**
- Try a different PDF library (pdfplumber, pymupdf)
- Check if PDF is scanned (needs OCR)

**File not found?**
- Check that paths are relative to `base_path`
- Verify file exists at the specified location

**No results from query?**
- Check collection has documents: `client.get_collection("course_materials")`
- Verify embeddings uploaded successfully
- Try a simpler query first

## Visualizing Embeddings

### 1. View as Table

You can export embeddings to a pandas DataFrame and view as table:

```python
from qdrant_client import QdrantClient
import pandas as pd

client = QdrantClient(host="localhost", port=6333)

# Get all points from collection
points, _ = client.scroll(
    collection_name="cs_materials",
    limit=1000,  # Adjust based on collection size
    with_payload=True,
    with_vectors=False  # Set True if you want to see vectors
)

# Convert to table format
data = []
for point in points:
    data.append({
        "id": point.id,
        "file_path": point.payload.get("file_path"),
        "file_type": point.payload.get("file_type"),
        "group_id": point.payload.get("group_id"),
        "course": point.payload.get("course"),
        "content_length": point.payload.get("content_length"),
    })

df = pd.DataFrame(data)
print(df)

# Save to CSV
df.to_csv("embeddings_table.csv", index=False)
```

### 2. Visualize in 2D/3D

Since embeddings are 2560-dimensional, we need dimensionality reduction (UMAP or t-SNE):

```bash
pip install umap-learn plotly pandas scikit-learn
```

**Create visualization script:**

```python
from qdrant_client import QdrantClient
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP
from sklearn.manifold import TSNE

client = QdrantClient(host="localhost", port=6333)

# Fetch all points WITH vectors
points, _ = client.scroll(
    collection_name="cs_materials",
    limit=1000,
    with_payload=True,
    with_vectors=True  # Important!
)

# Extract vectors and metadata
vectors = np.array([point.vector for point in points])
ids = [point.id for point in points]
group_ids = [point.payload.get("group_id", "unknown") for point in points]
file_types = [point.payload.get("file_type", "unknown") for point in points]
file_paths = [point.payload.get("file_path", "").split("/")[-1] for point in points]

# Option 1: UMAP (recommended - preserves global structure)
print("Running UMAP dimensionality reduction...")
reducer = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
embeddings_3d = reducer.fit_transform(vectors)

# Option 2: t-SNE (alternative)
# reducer = TSNE(n_components=3, random_state=42, perplexity=30)
# embeddings_3d = reducer.fit_transform(vectors)

# Create DataFrame
df = pd.DataFrame({
    'x': embeddings_3d[:, 0],
    'y': embeddings_3d[:, 1],
    'z': embeddings_3d[:, 2],
    'id': ids,
    'group_id': group_ids,
    'file_type': file_types,
    'filename': file_paths
})

# 3D Interactive Plot
fig = px.scatter_3d(
    df,
    x='x', y='y', z='z',
    color='group_id',
    symbol='file_type',
    hover_data=['id', 'filename'],
    title='Course Materials Embedding Space (3D)',
    labels={'group_id': 'Group', 'file_type': 'Type'}
)

fig.update_traces(marker=dict(size=5))
fig.update_layout(height=800)
fig.show()

# Save as HTML
fig.write_html("embeddings_3d.html")
print("Saved to embeddings_3d.html")

# 2D Plot (using only first 2 dimensions)
fig_2d = px.scatter(
    df,
    x='x', y='y',
    color='group_id',
    symbol='file_type',
    hover_data=['id', 'filename'],
    title='Course Materials Embedding Space (2D)',
    labels={'group_id': 'Group', 'file_type': 'Type'}
)

fig_2d.update_traces(marker=dict(size=8))
fig_2d.update_layout(height=700, width=1000)
fig_2d.write_html("embeddings_2d.html")
print("Saved to embeddings_2d.html")
```

**What you'll see:**
- Each **point** = one embedded document
- **Colors** = different groups (lectures, assignments)
- **Shapes** = different file types (transcript, notes, etc.)
- **Clusters** = similar content groups together
- **Hover** = see document ID and filename

### 3. Using Qdrant Dashboard

Qdrant has a built-in dashboard for basic visualization:

1. Open http://localhost:6333/dashboard
2. Select your collection
3. View:
   - Collection statistics
   - Point count
   - Vector dimensions
   - Storage info
4. Search and inspect individual points

### 4. Advanced: Atlas by Nomic

For large-scale interactive visualization:

```bash
pip install nomic
```

```python
import nomic
from nomic import atlas
import numpy as np

# Login to Nomic (free account)
nomic.login()

# Prepare data
data = [{
    "id": point.id,
    "text": point.payload.get("content_preview", ""),
    "group_id": point.payload.get("group_id"),
    "file_type": point.payload.get("file_type"),
} for point in points]

vectors = np.array([point.vector for point in points])

# Create interactive map
project = atlas.map_embeddings(
    embeddings=vectors,
    data=data,
    id_field="id",
    name="Course Materials Embeddings"
)

print(f"View your map at: {project.maps[0].url}")
```

This creates a beautiful interactive map where you can explore clusters and search through embeddings visually.
