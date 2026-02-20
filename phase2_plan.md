# Phase 2 Implementation Plan: `semantic_tree.py`

## Context

Phase 2 of the HCE roadmap: build the **Semantic Tree** (Merkle/Aggregation Tree) for long-term episodic memory, and the **Hierarchical Relevance Search (HRS)** algorithm for retrieval. This is the second of three data structures powering the HCE.

## File to create

`semantic_tree.py` (~300 lines)

## Dependencies

- Standard library: `dataclasses`, `typing`, `json`, `pathlib`, `math`, `hashlib`, `time`, `re`, `collections`
- No new external dependencies (vectors handled with pure Python + math)

## Components

### 1. Utility functions

- `cosine_similarity(a, b) -> float` — dot product / (norm_a * norm_b)
- `default_summarizer(texts) -> str` — extractive: first sentence of each text, truncated
- `default_vectorizer(text) -> list[float]` — feature hashing (hashing trick) into 256-dim vector, normalized

### 2. `SearchResult` dataclass

Structured result from HRS: `node_id`, `score`, `content`, `metadata`.

### 3. `TreeNode` dataclass

- `node_id: str`
- `content: str` — raw text for leaves, summary for internal nodes
- `vector: list[float]` — embedding of content
- `children_ids: list[str]`
- `parent_id: str | None`
- `is_leaf: bool`
- `metadata: dict`
- `timestamp: float`

### 4. `SemanticTree` class

| Method | Purpose |
|---|---|
| `__init__(summarizer, vectorizer, branch_factor)` | Pluggable summarizer/vectorizer, default branch_factor=4 |
| `add_interaction(content, metadata) -> str` | Add leaf node, trigger tree rebuild |
| `get_node(node_id) -> TreeNode or None` | Retrieve node by ID |
| `get_leaves() -> list[TreeNode]` | All leaf nodes in insertion order |
| `leaf_count` / `node_count` | Properties |
| `_rebuild_tree()` | Bottom-up: group leaves into parents, recurse |
| `save(path)` / `load(path)` | JSON serialization |

### 5. `hierarchical_relevance_search()` standalone function

```python
hierarchical_relevance_search(
    tree: SemanticTree,
    query: str,
    threshold: float = 0.1,
    top_k: int = 5,
) -> list[SearchResult]
```

**Algorithm:**
1. Vectorize the query using the tree's vectorizer
2. Start at root; compute similarity with each child
3. Prune children below threshold
4. Recurse into surviving children (sorted by similarity, best-first)
5. At leaves, collect candidates
6. Return top-k by score

### 6. Key design decisions

- **Feature hashing** for default vectorizer — no vocabulary needed, fixed 256 dimensions, works incrementally
- **Pluggable** summarizer/vectorizer — swap in LLM summarization or real embeddings later
- **Bottom-up rebuild** on each insertion — simple and correct; optimize to incremental updates if needed
- **Standalone HRS** function — same pattern as `spreading_activation`, enables algorithm swapping

## Verification

1. `python -c "from semantic_tree import SemanticTree, hierarchical_relevance_search"` — import check
2. Smoke test: add 10 interactions, search for a related query, verify results contain relevant leaves
3. Save/load round-trip
4. Verify pruning: search with high threshold should return fewer results
