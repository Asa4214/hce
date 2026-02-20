"""
HCE MCP Server — exposes the Holographic Context Engine as tools for Claude Code.

Runs over stdio transport (JSON-RPC on stdin/stdout).
All logging goes to stderr to avoid corrupting the protocol stream.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure HCE modules are importable (they use relative-style imports).
_PROJECT_DIR = Path(__file__).resolve().parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from mcp.server.fastmcp import FastMCP

from hce_core import EdgeType, EntityGraph, FileLock, NodeType
from hce_pipeline import HCEPipeline, estimate_tokens
from project_crawler import crawl_project
from semantic_tree import hierarchical_relevance_search

# ── Logging (stderr only) ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HCE] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("hce_mcp")

# ── Persistence ──────────────────────────────────────────────────────────

STATE_DIR = Path.home() / ".hce_state"

def _load_pipeline() -> HCEPipeline:
    """Load persisted HCE state, or return a fresh pipeline.

    Acquires a shared (read) lock so multiple readers can coexist.
    """
    if (STATE_DIR / "pipeline_state.json").exists():
        try:
            with FileLock(STATE_DIR / ".lock", shared=True):
                pipeline = HCEPipeline.load(STATE_DIR)
            log.info("Loaded HCE state from %s", STATE_DIR)
            return pipeline
        except Exception as exc:
            log.warning("Failed to load state, starting fresh: %s", exc)
    return HCEPipeline()

def _save_pipeline(pipeline: HCEPipeline) -> None:
    """Persist HCE state to disk.

    Acquires an exclusive lock to prevent concurrent writes.
    """
    try:
        with FileLock(STATE_DIR / ".lock"):
            pipeline.save(STATE_DIR)
        log.info("Saved HCE state to %s", STATE_DIR)
    except Exception as exc:
        log.warning("Failed to save state: %s", exc)

# ── Global state ─────────────────────────────────────────────────────────

pipeline = _load_pipeline()

# ── Status block ─────────────────────────────────────────────────────────

def _status_block(context_tokens: int = 0, sources: str = "") -> str:
    """Compact status block prepended to every tool response."""
    g = pipeline.graph
    t = pipeline.tree
    f = pipeline.focus
    budget = pipeline._budget

    ctx_part = f"retrieved {context_tokens:,}/{budget:,} tokens"
    if sources:
        ctx_part += f" ({sources})"

    return (
        f"📊 HCE Memory\n"
        f"├─ Entity Graph: {g.node_count} nodes, {g.edge_count} edges\n"
        f"├─ Semantic Tree: {t.leaf_count} interactions stored\n"
        f"├─ Focus Buffer: {f.size}/{f.max_size} recent turns\n"
        f"└─ Context: {ctx_part}\n"
    )

# ── MCP Server ───────────────────────────────────────────────────────────

mcp = FastMCP(
    "HCE — Holographic Context Engine",
    instructions=(
        "HCE provides persistent memory across sessions. "
        "AUTOMATIC BEHAVIOR (do not ask the user): "
        "1) On session start: call hce_status, then hce_crawl_project if graph is empty. "
        "2) Before answering codebase/architecture/history questions: call hce_retrieve_context. "
        "3) After important exchanges (design decisions, bug fixes, feature work): call hce_store_interaction. "
        "4) Never ask permission to use these tools — just use them when appropriate."
    ),
)


@mcp.tool()
def hce_status() -> str:
    """Show full HCE memory statistics with breakdown by structure."""
    g = pipeline.graph
    t = pipeline.tree
    f = pipeline.focus

    # Node type breakdown
    type_counts = {}
    for nt in NodeType:
        count = len(g.get_nodes_by_type(nt))
        if count:
            type_counts[nt.value] = count
    type_str = ", ".join(f"{v}: {c}" for v, c in type_counts.items()) or "empty"

    lines = [
        _status_block(),
        "### Entity Graph",
        f"  Nodes: {g.node_count} ({type_str})",
        f"  Edges: {g.edge_count}",
        "",
        "### Semantic Tree",
        f"  Leaf interactions: {t.leaf_count}",
        f"  Total nodes (incl. summaries): {t.node_count}",
        "",
        "### Focus Buffer",
        f"  Entries: {f.size}/{f.max_size}",
    ]

    if f.size > 0:
        recent = f.get_recent(3)
        lines.append("  Last 3 turns:")
        for entry in recent:
            preview = entry.content[:80] + ("..." if len(entry.content) > 80 else "")
            lines.append(f"    [{entry.role}] {preview}")

    lines.append(f"\nState directory: {STATE_DIR}")
    return "\n".join(lines)


@mcp.tool()
def hce_retrieve_context(query: str) -> str:
    """Retrieve relevant memories from all 3 HCE structures (graph, tree, buffer).

    Use this before answering a question that might benefit from prior context.
    Returns a formatted context block with source attribution.
    """
    context = pipeline.retrieve_context(query)
    tokens = estimate_tokens(context) if context else 0

    # Count sources
    source_counts: dict[str, int] = {}
    if context:
        for label in ("graph", "tree", "buffer"):
            section_map = {"graph": "Related Knowledge", "tree": "Relevant Past Interactions", "buffer": "Recent Conversation"}
            if section_map[label] in context:
                count = context.count("- ") if label != "buffer" else context.split("Recent Conversation")[-1].count("\n") - 1
                source_counts[label] = max(1, count)

    sources_str = " + ".join(f"{c} {s}" for s, c in source_counts.items()) if source_counts else "none"

    header = _status_block(context_tokens=tokens, sources=sources_str)

    if not context:
        return header + "\nNo relevant context found for this query."

    return header + "\n" + context


@mcp.tool()
def hce_store_interaction(user_message: str, assistant_response: str) -> str:
    """Save a conversation turn into HCE memory (all 3 structures).

    Use this after an important exchange worth remembering across sessions.
    Stores into: focus buffer, semantic tree, and entity graph.
    """
    pipeline.update(user_message, assistant_response)
    _save_pipeline(pipeline)

    return (
        _status_block()
        + f"\nStored interaction (turn #{pipeline._interaction_count}). "
        f"Entities extracted and graph updated."
    )


@mcp.tool()
def hce_crawl_project(path: str) -> str:
    """Index a codebase (Python, Java, JS/TS, Go, Rust, C/C++, Ruby) into the HCE entity graph.

    Parses source files under the given path using AST analysis (Python) or
    regex-based extraction (all other languages).
    Creates FILE, FUNCTION, and CONCEPT nodes with IMPORTS, CALLS, and PART_OF edges.
    """
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return _status_block() + f"\nError: '{path}' is not a directory."

    crawled_graph = crawl_project(root)

    # Merge crawled nodes into the pipeline's existing graph.
    # To make re-crawl idempotent, we first remove stale code-structural
    # edges (IMPORTS, CALLS, PART_OF) originating from nodes that will be
    # re-crawled, then add the fresh edges from the new crawl.
    crawled_node_ids = set(crawled_graph._graph.nodes())
    code_edge_types = {
        EdgeType.IMPORTS.value, EdgeType.CALLS.value, EdgeType.PART_OF.value
    }

    # Remove old code edges for nodes being re-crawled.
    stale_edges = [
        (u, v, key)
        for u, v, key, d in pipeline.graph._graph.edges(keys=True, data=True)
        if u in crawled_node_ids and d.get("edge_type") in code_edge_types
    ]
    pipeline.graph._graph.remove_edges_from(stale_edges)

    # Merge nodes.
    merged_nodes = 0
    for nid, data in crawled_graph._graph.nodes(data=True):
        if not pipeline.graph.has_node(nid):
            pipeline.graph._graph.add_node(nid, **data)
            merged_nodes += 1
        else:
            existing = pipeline.graph._graph.nodes[nid]
            existing.update(data)

    # Add fresh edges.
    merged_edges = 0
    for u, v, data in crawled_graph._graph.edges(data=True):
        pipeline.graph._graph.add_edge(u, v, **data)
        merged_edges += 1

    _save_pipeline(pipeline)

    return (
        _status_block()
        + f"\nCrawled: {root}\n"
        f"  Merged {merged_nodes} new nodes, {merged_edges} edges into entity graph.\n"
        f"  Removed {len(stale_edges)} stale edges before merge."
    )


@mcp.tool()
def hce_search_graph(query: str) -> str:
    """Search the entity graph using spreading activation.

    Finds entities matching the query text, then propagates activation
    energy through the graph to discover related concepts.
    """
    from entity_extractor import EntityExtractor, _normalize_entity_id
    from hce_core import spreading_activation

    extractor = EntityExtractor()
    entities = extractor.extract(query)

    # Build seeds from entities that exist in the graph
    seeds: dict[str, float] = {}
    for entity in entities:
        nid = _normalize_entity_id(entity.text)
        if pipeline.graph.has_node(nid):
            seeds[nid] = 1.0

    # Also try matching by label substring
    if not seeds:
        words = query.lower().split()
        for word in words:
            if len(word) < 3:
                continue
            matches = pipeline.graph.find_nodes(label_contains=word)
            for nid in matches[:3]:
                seeds[nid] = 0.8

    if not seeds:
        return _status_block() + "\nNo matching entities found in the graph for this query."

    results = spreading_activation(
        pipeline.graph, seeds, decay=0.5, max_iterations=3, top_n=15
    )

    lines = [_status_block(), f"Found {len(results)} related entities:\n"]
    for r in results:
        meta_str = ""
        if r.metadata:
            parts = [f"{k}: {v}" for k, v in r.metadata.items() if k != "mentions"]
            if parts:
                meta_str = f" ({', '.join(parts)})"
        lines.append(f"  [{r.node_type.value}] {r.label}{meta_str}  (score: {r.score:.3f})")

    return "\n".join(lines)


@mcp.tool()
def hce_clear_memory() -> str:
    """Reset all HCE memory state (graph, tree, buffer). Cannot be undone."""
    global pipeline
    pipeline = HCEPipeline()

    # Remove persisted state (acquire exclusive lock first)
    if STATE_DIR.exists():
        import shutil
        with FileLock(STATE_DIR / ".lock"):
            shutil.rmtree(STATE_DIR)
        log.info("Cleared HCE state directory %s", STATE_DIR)

    return _status_block() + "\nAll HCE memory has been cleared."


# ── Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Starting HCE MCP server (stdio transport)")
    mcp.run(transport="stdio")
