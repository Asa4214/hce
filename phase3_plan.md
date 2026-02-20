# Phase 3 Implementation Plan: Project Crawler & Entity Extractor

## Context

Phase 3 builds two ingestion pipelines that populate the EntityGraph (Phase 1) and SemanticTree (Phase 2):
1. **Project Crawler** — walks a Python codebase and extracts code structure into the EntityGraph
2. **Entity Extractor** — processes conversation text via NER and builds associative memory

## Files to create

- `project_crawler.py` (~200 lines)
- `entity_extractor.py` (~200 lines)

## Dependencies

- Standard library only: `ast`, `pathlib`, `re`, `fnmatch`, `dataclasses`
- Uses `hce_core.EntityGraph`, `NodeType`, `EdgeType` from Phase 1

## Component 1: Project Crawler (`project_crawler.py`)

### Purpose
Crawl a Python project directory, parse source files with `ast`, and build an EntityGraph representing code structure.

### Components

**`CrawlConfig` dataclass:**
- `extensions: list[str]` — file extensions to process (default: `[".py"]`)
- `ignore_patterns: list[str]` — glob patterns to skip (default: `["__pycache__", ".git", ".venv", "node_modules"]`)

**`crawl_project(root_path, config) -> EntityGraph`:**
- Walk directory tree, filter files
- For each Python file: parse AST, extract entities
- Return populated EntityGraph

**AST extraction:**
- FILE nodes: one per source file (node_id = relative path)
- FUNCTION nodes: functions and classes (node_id = `file_path::func_name`)
- PART_OF edges: function → file
- IMPORTS edges: file → imported module/file
- CALLS edges: function → called function (best-effort name matching)

## Component 2: Entity Extractor (`entity_extractor.py`)

### Purpose
Extract named entities from conversation text and maintain an EntityGraph of people, concepts, and events.

### Components

**`Entity` dataclass:**
- `text: str`, `entity_type: NodeType`, `start: int`, `end: int`

**`EntityExtractor` class:**
- `__init__(ner_func=None)` — pluggable NER, defaults to regex-based
- `extract(text) -> list[Entity]` — extract entities from text
- `update_graph(text, graph, interaction_id) -> list[str]` — extract entities, add to graph, create edges

**Default NER (`_default_ner`):**
- PERSON: Title Case name sequences (2+ capitalized words not at sentence start), common name patterns
- CONCEPT: Capitalized single words or quoted terms, repeated nouns
- EVENT: Temporal phrases ("meeting on Monday", "trip to X")

**Graph construction:**
- Each extracted entity becomes a node (deduplication by normalized text)
- Entities co-occurring in the same message get RELATES_TO edges
- If `interaction_id` is provided, entities get RELATES_TO edges to that interaction

## Verification

1. `crawl_project(".")` on the HCE project itself — should find hce_core.py, semantic_tree.py, etc.
2. Entity extraction on sample sentences — should find people, concepts
3. Graph integration: extract → verify nodes and edges exist in graph
