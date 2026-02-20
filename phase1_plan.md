# Phase 1 Implementation Plan: `hce_core.py`

## Context

The HCE project needs its first implementation file. Phase 1 of the roadmap calls for the `EntityGraph` (property graph using networkx) and the **Spreading Activation** algorithm. This is the foundational data structure and algorithm that all subsequent phases build on.

## File to create

`hce_core.py` (~250 lines)

## Dependencies

- `networkx` (only external dependency)
- Standard library: `enum`, `dataclasses`, `typing`, `json`, `pathlib`

## Components

### 1. Enums: `NodeType` and `EdgeType`

From the architecture plan:
- `NodeType`: FILE, FUNCTION, CONCEPT, PERSON, EVENT
- `EdgeType`: IMPORTS, CALLS, RELATES_TO, PART_OF

### 2. `ActivationResult` dataclass

Structured result from spreading activation: `node_id`, `score`, `node_type`, `label`, `metadata`.

### 3. `EntityGraph` class (wraps `nx.MultiDiGraph`)

Using `MultiDiGraph` so two nodes can have multiple edge types (e.g., file A both IMPORTS and RELATES_TO file B).

**Methods:**

| Method | Purpose |
|---|---|
| `add_node(node_id, node_type, label, metadata)` | Add typed node; merges metadata if exists, raises if type conflicts |
| `add_edge(source_id, target_id, edge_type, weight, metadata)` | Add typed weighted edge; raises KeyError if nodes missing |
| `get_node(node_id)` | Return node attrs dict or None |
| `get_neighbors(node_id, edge_type, direction)` | Get neighbors with optional type/direction filter |
| `get_nodes_by_type(node_type)` | List node IDs of a given type |
| `has_node(node_id)` | Existence check |
| `remove_node(node_id)` | Remove node + edges |
| `remove_edge(source_id, target_id)` | Remove edge |
| `find_nodes(label_contains, node_type, metadata_filter)` | Search with AND-combined filters |
| `subgraph(node_ids)` | Extract subgraph as new EntityGraph |
| `save(path)` / `load(path)` | JSON serialization |
| `node_count` / `edge_count` | Properties |

### 4. `spreading_activation()` standalone function

```python
spreading_activation(
    graph: EntityGraph,
    seed_nodes: dict[str, float],
    decay: float = 0.5,
    max_iterations: int = 3,
    min_activation: float = 0.01,
    top_n: int = 10
) -> list[ActivationResult]
```

**Algorithm:**
1. Initialize activation from `seed_nodes` dict (node_id -> energy)
2. For each iteration: propagate `energy * decay * edge_weight` to all neighbors (both directions), accumulate additively
3. Prune nodes below `min_activation` threshold
4. Return top-N sorted by score descending

Standalone (not a method) so alternative algorithms can be added later without modifying EntityGraph.

### 5. Key design decisions

- **MultiDiGraph** over DiGraph — supports multiple edge types between same node pair
- **JSON** serialization over pickle — human-readable, debuggable
- **Additive** activation — nodes connected via multiple paths accumulate more energy
- **Bidirectional** propagation — associative recall is symmetric
- **String node IDs** — Phase 3 will use file paths / entity names as natural IDs

## Verification

1. `pip install networkx` (if not already installed)
2. `python -c "from hce_core import EntityGraph, spreading_activation, NodeType, EdgeType"` — import check
3. Quick smoke test in Python REPL:
   - Create graph, add nodes/edges, run spreading_activation, verify results are sorted and seed nodes appear
   - Test save/load round-trip
