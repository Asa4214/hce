# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The **Holographic Context Engine (HCE)** is a middleware memory system for LLMs. It replaces linear conversation history with three parallel data structures — an Entity Graph, a Semantic Tree, and a Focus Buffer — to retrieve only the most relevant context within a token budget.

All 4 implementation phases are complete. 194 tests passing.

## Architecture

Three parallel data structures:

- **Entity Graph** (`hce_core.py`) — Property graph via `networkx.MultiDiGraph`. Node types: File, Function, Concept, Person, Event. Edge types: Imports, Calls, Relates_To, Part_Of. Retrieved via **Spreading Activation**.
- **Semantic Tree** (`semantic_tree.py`) — Aggregation tree. Leaves = raw turns, internal nodes = vectorized summaries. Retrieved via **Hierarchical Relevance Search** (HRS) with branch pruning.
- **Focus Buffer** (`hce_pipeline.py: SlidingFocusBuffer`) — Circular buffer (deque) of the last N turns. Retrieved by recency.

Orchestrated by **HCEPipeline** (`hce_pipeline.py`), which runs all three retrievals, applies **Context Budgeting** (greedy knapsack by `Utility / Token_Cost`), and formats the result.

Cross-cutting concerns:

- **Concurrent Access** (`hce_core.py: FileLock`) — POSIX advisory locking via `fcntl.flock()` protects `~/.hce_state/` from corruption when multiple MCP server instances run concurrently. Shared locks for reads, exclusive locks for writes, 10s timeout.
- **Multi-Language Indexing** (`project_crawler.py`) — Python uses full AST parsing; Java, JS/TS, Go, Rust, C/C++, and Ruby use regex-based parsers. All produce a common `ParseResult` dataclass that the main loop converts into graph nodes/edges.

## File Map

| File | Purpose |
|------|---------|
| `hce_core.py` | EntityGraph + Spreading Activation + FileLock |
| `semantic_tree.py` | SemanticTree + HRS + vectorizer/summarizer |
| `entity_extractor.py` | Regex/heuristic NER → graph updates |
| `project_crawler.py` | Multi-language codebase indexer (Python/Java/JS/TS/Go/Rust/C/C++/Ruby) |
| `hce_pipeline.py` | Pipeline + FocusBuffer + ContextBudgeting |
| `hce_mcp_server.py` | MCP server (6 tools) for Claude Code |
| `test_*.py` | Test suites for each module |

## Commands

```bash
# Run all tests
pytest

# Run a specific test file
pytest test_hce_core.py -v

# Install dependencies (editable mode)
pip install -e ".[dev]"
```

## Code Conventions

- All modules use `from __future__ import annotations`
- Type hints on all public functions
- Docstrings on all public classes and methods
- Enums for node/edge types (`NodeType`, `EdgeType`) — never raw strings
- `save()`/`load()` pattern for persistence (JSON-based)
- Tests follow `Test<ClassName>` naming with `test_<behavior>` methods

## Key Patterns to Follow

- **EntityGraph access:** Always use `add_node()`, `add_edge()`, `get_node()`, etc. — never access `_graph` directly
- **Imports:** Modules import from each other (`from hce_core import EntityGraph`), not via a package init
- **Serialization:** All structures serialize to JSON via `save(path)` / `load(path)` classmethods
- **Pipeline state:** `HCEPipeline.save(dir)` writes three files: `entity_graph.json`, `semantic_tree.json`, `pipeline_state.json`
- **FileLock usage:** Always wrap `save()`/`load()` with `FileLock(STATE_DIR / ".lock")` — exclusive for writes, `shared=True` for reads
- **Adding a new language parser:** Add a `_parse_<lang>(source, file_id) -> ParseResult` function, then register it in `_LANGUAGE_MAP` by extension
- **ParseResult convention:** All language parsers return `ParseResult(definitions, imports, calls)` — never write to the graph directly from a parser

## Known Limitations (as of v0.2.0)

- Vectorizer is bag-of-words feature hashing (no semantic embeddings)
- Summarizer is extractive (first sentences), not LLM-based
- NER is regex/heuristic, not ML
- Full tree rebuild on every add_interaction()
- Non-Python language parsers are regex-based (no semantic understanding, may miss complex patterns)
- FileLock uses POSIX `fcntl.flock()` — not available on Windows without adaptation

## HCE MCP Integration — Automatic Behavior Rules

The HCE MCP server (`hce_mcp_server.py`) provides persistent memory across sessions. **You MUST follow these rules automatically without the user asking:**

### On session start
1. Call `hce_status` to check current memory state.
2. If the entity graph has 0 nodes, call `hce_crawl_project` on this project root to index the codebase.

### Before answering substantive questions
- Call `hce_retrieve_context` with the user's query before answering any question about:
  - The codebase (architecture, how something works, where something is)
  - Previous conversations or decisions
  - Anything that might benefit from prior context
- You do NOT need to retrieve context for simple commands like "fix this typo" or "run tests".

### After important exchanges
- Call `hce_store_interaction` after any exchange that involves:
  - Architectural decisions or design discussions
  - Bug investigations and their resolutions
  - New feature implementations or significant code changes
  - Explanations of how something works
- Summarize concisely when storing — don't store verbatim multi-page responses.
- Do NOT store trivial exchanges (greetings, simple file reads, typo fixes).

### When the user asks to index/learn/crawl a project
- Call `hce_crawl_project` with the path they specify (or the project root if unspecified).

### General
- Never ask the user "should I store this in HCE?" — use your judgment based on the rules above.
- Never ask the user "should I check HCE for context?" — just do it when relevant.
- The status block in tool responses gives you memory stats — you don't need to relay these to the user unless they ask about HCE state.
