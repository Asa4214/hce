# Scalability Guide

HCE is designed as a local-first middleware memory system. This document covers what it handles well today, where the limits are, and what changes are needed to scale further.

## Current Capacity

| Dimension | Comfortable Range | Why |
|-----------|------------------|-----|
| Graph size | Up to ~10K nodes | NetworkX in-memory graph with O(1) node lookup |
| Conversations | Hundreds of interactions | Semantic tree search is O(log n) via branch pruning |
| Retrieval speed | Sub-second for local use | Spreading activation bounded at 3 iterations, top-15 results |
| Concurrent sessions | Multiple Claude Code / Copilot CLI windows | FileLock (shared/exclusive) prevents corruption |
| Codebase indexing | Small to medium projects | 8 languages supported; Python via AST, others via regex |

For a **single developer** using HCE with Claude Code or Copilot CLI on their own projects, the current implementation handles typical workloads without issues.

## Known Bottlenecks

### 1. Full tree rebuild on every interaction — O(n)

**File:** `semantic_tree.py` — `_rebuild_tree()`

Every call to `add_interaction()` destroys the entire internal tree and rebuilds it bottom-up from all leaves. At 100 interactions this takes milliseconds. At 10,000+ it becomes noticeably slow.

**Impact:** Write latency grows linearly with stored interactions.

**Fix:** Incremental insertion — add the new leaf and update only the path from that leaf to the root, leaving the rest of the tree untouched. Complexity drops from O(n) to O(log n) per insert.

### 2. JSON serialization — full state loaded/saved every time

**Files:** `hce_core.py`, `semantic_tree.py`, `hce_pipeline.py` — `save()` / `load()`

Every save serializes the entire graph and tree to JSON. Every load deserializes all of it back into memory. At 300 nodes this is fast. At 100K nodes with 500K edges, the JSON files can exceed 50MB, making every save/load operation expensive.

**Impact:** I/O latency grows linearly with total state size. Memory usage equals full state size at all times.

**Fix:** Replace JSON files with SQLite (or another embedded database). This enables partial reads, indexed queries, and incremental writes without loading the entire state into memory.

### 3. Bag-of-words vectorizer — no semantic understanding

**File:** `semantic_tree.py` — `_default_vectorizer()`

The current vectorizer uses feature hashing (bag-of-words). It matches on exact word overlap only:

- "car" and "automobile" — no match
- "deploy" and "ship to production" — no match
- "error handling" and "exception management" — no match

**Impact:** Retrieval quality degrades as content grows, because more relevant memories are missed by keyword-only matching.

**Fix:** Use semantic embeddings via sentence-transformers (local) or an embedding API (hosted). This enables meaning-based matching regardless of exact wording.

### 4. Single-machine, single-user architecture

**File:** `hce_mcp_server.py` — `STATE_DIR = Path.home() / ".hce_state"`

All state is stored in one directory (`~/.hce_state/`) with no user isolation. Two users on the same machine would share the same memory. There's no authentication, no user separation, and no remote access.

**Impact:** Cannot be deployed as a shared service or multi-tenant system.

**Fix:** Per-user state directories (keyed by user ID or project), or a server-based architecture with an API layer and user authentication.

### 5. No incremental codebase re-indexing

**File:** `hce_mcp_server.py` — `hce_crawl_project()`

Re-crawling a project removes all code-structural edges (IMPORTS, CALLS, PART_OF) for every node that appears in the crawl, then re-adds them from scratch. Changed one file? The entire codebase is re-parsed.

**Impact:** Re-indexing time is proportional to total project size, not to the size of the change.

**Fix:** Diff-based crawling — track file modification timestamps, only re-parse files that changed since the last crawl, and merge just those results into the graph.

### 6. Extractive summarizer — no abstractive understanding

**File:** `semantic_tree.py` — `_default_summarizer()`

The summarizer picks the first sentence from each child node's content. It doesn't understand or compress meaning — it just truncates.

**Impact:** Internal tree nodes (summaries) may not accurately represent their children, leading to suboptimal branch pruning during hierarchical search.

**Fix:** Use an LLM-based summarizer that generates true abstractive summaries of grouped interactions.

## Scaling Tiers

| Tier | Target User | Interactions | Graph Size | Changes Needed |
|------|------------|-------------|------------|----------------|
| **Current** | Single developer, local | Up to ~500 | Up to ~10K nodes | None — works as-is |
| **Power user** | Single developer, heavy use | Up to ~5,000 | Up to ~100K nodes | Incremental tree insert, SQLite storage |
| **Team** | Small team, shared server | Up to ~50,000 | Up to ~1M nodes | Real embeddings, database backend, diff-based crawl, user isolation |
| **Production SaaS** | Thousands of users | Millions | Millions of nodes | PostgreSQL + pgvector, streaming updates, API layer, authentication, rate limiting |

## Recommended Upgrade Path

If you want to scale HCE beyond single-user local use, here's the priority order:

### Priority 1: SQLite backend
Replace JSON file persistence with SQLite. This solves serialization overhead, enables partial reads/writes, and provides a foundation for indexed queries. The `save()`/`load()` interface stays the same — only the implementation changes.

### Priority 2: Semantic embeddings
Swap the bag-of-words vectorizer for sentence-transformers (e.g., `all-MiniLM-L6-v2`). This dramatically improves retrieval quality and is a drop-in replacement — the vectorizer is already a pluggable function passed to `SemanticTree.__init__()`.

### Priority 3: Incremental tree insertion
Replace the full `_rebuild_tree()` with incremental insertion. Add new leaves and update only the affected path to root. This changes write complexity from O(n) to O(log n).

### Priority 4: Diff-based crawling
Track file modification times in the entity graph. On re-crawl, only re-parse changed files and merge their results. This makes re-indexing proportional to the change size, not the project size.

### Priority 5: Multi-user support
Add per-user state isolation (separate state directories or database schemas), an API layer for remote access, and authentication. This is only needed if HCE is deployed as a shared service.

## Architecture Advantage

The core architecture — three parallel retrieval structures with knapsack budgeting — scales conceptually to any size. The bottlenecks listed above are all **implementation choices** (JSON vs. database, full-rebuild vs. incremental, bag-of-words vs. embeddings) that can be swapped without changing the overall design.

The `HCEPipeline` orchestrator, the `retrieve_context()` flow, the `context_budgeting()` algorithm, and the MCP server interface all remain unchanged regardless of which backend or vectorizer is used.
