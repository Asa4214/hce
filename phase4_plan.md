# Phase 4 Implementation Plan: HCE Pipeline

## Context

Phase 4 wraps everything into a middleware pipeline. It builds the remaining data structure (Sliding Focus Buffer), the remaining algorithm (Context Budgeting), and the HCEPipeline class that orchestrates all components.

## File to create

`hce_pipeline.py` (~350 lines)

## Dependencies

- `hce_core`: EntityGraph, NodeType, EdgeType, spreading_activation
- `semantic_tree`: SemanticTree, hierarchical_relevance_search
- `entity_extractor`: EntityExtractor
- Standard library: `collections.deque`, `dataclasses`, `time`, `json`, `pathlib`

## Components

### 1. `FocusEntry` dataclass
- `role: str` — "user" or "assistant"
- `content: str`
- `timestamp: float`

### 2. `SlidingFocusBuffer` class (Circular Buffer — 3rd data structure)
The short-term memory component from the architecture.

| Method | Purpose |
|---|---|
| `__init__(max_size)` | Create buffer with max N entries (default 10) |
| `add(role, content)` | Push entry, oldest drops off when full |
| `get_recent(n)` | Get last n entries (or all if n is None) |
| `clear()` | Empty the buffer |
| `size` property | Current entry count |
| `to_text()` | Format buffer as readable text for LLM context |

### 3. `ContextCandidate` dataclass
- `content: str`
- `source: str` — "graph", "tree", or "buffer"
- `utility: float`
- `token_cost: int`
- `metadata: dict`

### 4. `estimate_tokens(text) -> int`
Simple heuristic: `len(text.split()) * 4 // 3` (~1.33 tokens per word).
Pluggable — users can replace with tiktoken.

### 5. `context_budgeting(candidates, budget) -> list[ContextCandidate]`
Standalone function. Greedy knapsack:
1. Compute ratio = utility / token_cost for each candidate
2. Sort by ratio descending
3. Greedily pack until budget exhausted
4. Return selected candidates

### 6. `HCEPipeline` class — the main middleware

| Method | Purpose |
|---|---|
| `__init__(max_focus, context_budget, graph, tree)` | Initialize with optional pre-built components |
| `retrieve_context(query) -> str` | Run all retrieval algorithms, budget, format context block |
| `update(user_query, ai_response)` | Store interaction in all three structures |
| `build_prompt(user_query) -> str` | Combine [Context Block] + [Focus Buffer] + [User Query] |
| `wrap_chat(chat_func) -> Callable` | Decorator: intercept → enrich → call LLM → update |
| `save(directory)` / `load(directory)` | Persist all state to a directory |

### Retrieval flow (inside `retrieve_context`):
1. Extract entities from query → seed nodes for spreading activation
2. Run `spreading_activation(graph, seeds)` → graph candidates
3. Run `hierarchical_relevance_search(tree, query)` → tree candidates
4. Collect focus buffer entries → buffer candidates
5. Merge all into `ContextCandidate` list
6. Run `context_budgeting(candidates, budget)` → selected candidates
7. Format selected candidates into a context block string

### Update flow (inside `update`):
1. Add user query + AI response to focus buffer
2. Store combined interaction in semantic tree
3. Extract entities from both query and response, update graph

## Verification

1. Import check
2. End-to-end: create pipeline, process several queries, verify context retrieval improves
3. Context budgeting: verify budget is respected
4. wrap_chat: verify decorator enriches prompts and stores responses
5. Save/load round-trip
