"""
HCE Phase 4: Pipeline — ties EntityGraph, SemanticTree, and Focus Buffer
together with Context Budgeting into a middleware wrapper for LLM chat.

Data flow:
  User Query → extract entities → query Graph + Tree + Buffer
  → Context Budgeting → build enriched prompt → LLM → store response
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from hce_core import EdgeType, EntityGraph, NodeType, atomic_write, spreading_activation

log = logging.getLogger(__name__)
from entity_extractor import EntityExtractor
from semantic_tree import SemanticTree, hierarchical_relevance_search


# ── Sliding Focus Buffer (Short-Term Memory) ─────────────────────────────

@dataclass
class FocusEntry:
    role: str  # "user" or "assistant"
    content: str
    timestamp: float = 0.0


class SlidingFocusBuffer:
    """Circular buffer for the last N interaction turns."""

    def __init__(self, max_size: int = 10) -> None:
        self._buffer: deque[FocusEntry] = deque(maxlen=max_size)
        self._max_size = max_size

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def max_size(self) -> int:
        return self._max_size

    def add(self, role: str, content: str) -> None:
        """Push an entry. Oldest entry drops when buffer is full."""
        self._buffer.append(
            FocusEntry(role=role, content=content, timestamp=time.time())
        )

    def get_recent(self, n: int | None = None) -> list[FocusEntry]:
        """Return the last *n* entries (or all if n is None)."""
        entries = list(self._buffer)
        if n is not None:
            entries = entries[-n:]
        return entries

    def clear(self) -> None:
        self._buffer.clear()

    def to_text(self) -> str:
        """Format the buffer contents as readable text for LLM context."""
        if not self._buffer:
            return ""
        lines: list[str] = []
        for entry in self._buffer:
            prefix = "User" if entry.role == "user" else "Assistant"
            lines.append(f"{prefix}: {entry.content}")
        return "\n".join(lines)

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize buffer entries to plain dicts."""
        return [
            {"role": e.role, "content": e.content, "timestamp": e.timestamp}
            for e in self._buffer
        ]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]], max_size: int = 10) -> SlidingFocusBuffer:
        """Restore buffer from serialized dicts."""
        buf = cls(max_size=max_size)
        for entry in data:
            fe = FocusEntry(
                role=entry["role"],
                content=entry["content"],
                timestamp=entry.get("timestamp", 0.0),
            )
            buf._buffer.append(fe)
        return buf


# ── Context Budgeting (Knapsack) ─────────────────────────────────────────

@dataclass
class ContextCandidate:
    content: str
    source: str  # "graph", "tree", or "buffer"
    utility: float
    token_cost: int
    metadata: dict[str, Any] = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """Estimate token count from text. ~1.33 tokens per word."""
    words = len(text.split())
    return max(1, words * 4 // 3)


def context_budgeting(
    candidates: list[ContextCandidate],
    budget: int = 4000,
) -> list[ContextCandidate]:
    """Greedy knapsack: select candidates by utility/cost ratio within budget.

    1. Compute ratio for each candidate.
    2. Sort by ratio descending.
    3. Greedily pack until budget exhausted.
    """
    if not candidates:
        return []

    # Sort by efficiency (utility per token), breaking ties by raw utility
    ranked = sorted(
        candidates,
        key=lambda c: (
            c.utility / max(1, c.token_cost),
            c.utility,
        ),
        reverse=True,
    )

    selected: list[ContextCandidate] = []
    remaining = budget

    for candidate in ranked:
        if candidate.token_cost <= remaining:
            selected.append(candidate)
            remaining -= candidate.token_cost

    return selected


# ── HCE Pipeline ─────────────────────────────────────────────────────────

class HCEPipeline:
    """Middleware that sits between the user and the LLM.

    Orchestrates EntityGraph (associative memory), SemanticTree (episodic
    memory), and SlidingFocusBuffer (short-term memory) to build
    context-enriched prompts.
    """

    def __init__(
        self,
        max_focus_entries: int = 10,
        context_budget: int = 4000,
        graph: EntityGraph | None = None,
        tree: SemanticTree | None = None,
        token_estimator: Callable[[str], int] | None = None,
    ) -> None:
        self.graph = graph or EntityGraph()
        self.tree = tree or SemanticTree()
        self.focus = SlidingFocusBuffer(max_focus_entries)
        self._extractor = EntityExtractor()
        self._budget = context_budget
        self._estimate_tokens = token_estimator or estimate_tokens
        self._interaction_count = 0

    # ── Core retrieval ───────────────────────────────────────────────

    def retrieve_context(self, query: str) -> str:
        """Run all retrieval algorithms, budget the results, and return
        a formatted context block."""
        candidates: list[ContextCandidate] = []

        # 1. Graph retrieval via spreading activation
        graph_candidates = self._retrieve_from_graph(query)
        candidates.extend(graph_candidates)

        # 2. Tree retrieval via HRS
        tree_candidates = self._retrieve_from_tree(query)
        candidates.extend(tree_candidates)

        # 3. Focus buffer (recent conversation)
        buffer_candidates = self._retrieve_from_buffer()
        candidates.extend(buffer_candidates)

        log.debug("Retrieval candidates: %d graph, %d tree, %d buffer",
                  len(graph_candidates), len(tree_candidates), len(buffer_candidates))

        # 4. Context budgeting
        selected = context_budgeting(candidates, self._budget)
        total_tokens = sum(c.token_cost for c in selected)
        log.info("Context budgeting: %d/%d candidates selected, %d/%d tokens used",
                 len(selected), len(candidates), total_tokens, self._budget)

        # 5. Format into a context block
        return self._format_context_block(selected)

    def _retrieve_from_graph(self, query: str) -> list[ContextCandidate]:
        """Extract entities from query, use as seeds for spreading activation."""
        entities = self._extractor.extract(query)
        seeds: dict[str, float] = {}
        for entity in entities:
            from entity_extractor import _normalize_entity_id
            nid = _normalize_entity_id(entity.text)
            if self.graph.has_node(nid):
                seeds[nid] = 1.0

        if not seeds:
            return []

        results = spreading_activation(
            self.graph, seeds, decay=0.5, max_iterations=3, top_n=10
        )

        candidates: list[ContextCandidate] = []
        for result in results:
            # Skip seed nodes themselves (the query already contains this info)
            if result.node_id in seeds:
                continue
            content = f"[{result.node_type.value}] {result.label}"
            if result.metadata:
                detail = ", ".join(
                    f"{k}: {v}" for k, v in result.metadata.items()
                    if k != "mentions"
                )
                if detail:
                    content += f" ({detail})"
            candidates.append(
                ContextCandidate(
                    content=content,
                    source="graph",
                    utility=result.score,
                    token_cost=self._estimate_tokens(content),
                    metadata={"node_id": result.node_id},
                )
            )
        return candidates

    def _retrieve_from_tree(self, query: str) -> list[ContextCandidate]:
        """Search the SemanticTree for relevant past interactions."""
        if self.tree.leaf_count == 0:
            return []

        results = hierarchical_relevance_search(
            self.tree, query, threshold=0.05, top_k=5
        )

        candidates: list[ContextCandidate] = []
        for result in results:
            candidates.append(
                ContextCandidate(
                    content=result.content,
                    source="tree",
                    utility=result.score,
                    token_cost=self._estimate_tokens(result.content),
                    metadata=result.metadata,
                )
            )
        return candidates

    def _retrieve_from_buffer(self) -> list[ContextCandidate]:
        """Convert recent focus buffer entries into candidates."""
        entries = self.focus.get_recent()
        if not entries:
            return []

        # Recent entries get higher utility (recency weighting)
        candidates: list[ContextCandidate] = []
        n = len(entries)
        for i, entry in enumerate(entries):
            recency = (i + 1) / n  # 0..1, most recent = 1.0
            text = f"{entry.role}: {entry.content}"
            candidates.append(
                ContextCandidate(
                    content=text,
                    source="buffer",
                    utility=0.5 + 0.5 * recency,  # range: 0.5 to 1.0
                    token_cost=self._estimate_tokens(text),
                )
            )
        return candidates

    def _format_context_block(self, candidates: list[ContextCandidate]) -> str:
        """Format selected candidates into a structured context block."""
        if not candidates:
            return ""

        # Group by source
        by_source: dict[str, list[ContextCandidate]] = {}
        for c in candidates:
            by_source.setdefault(c.source, []).append(c)

        sections: list[str] = []

        if "tree" in by_source:
            items = by_source["tree"]
            lines = [f"- {c.content}" for c in items]
            sections.append("## Relevant Past Interactions\n" + "\n".join(lines))

        if "graph" in by_source:
            items = by_source["graph"]
            lines = [f"- {c.content}" for c in items]
            sections.append("## Related Knowledge\n" + "\n".join(lines))

        if "buffer" in by_source:
            items = by_source["buffer"]
            lines = [c.content for c in items]
            sections.append("## Recent Conversation\n" + "\n".join(lines))

        return "\n\n".join(sections)

    # ── Update (store interactions) ──────────────────────────────────

    def update(self, user_query: str, ai_response: str) -> None:
        """Store an interaction into all three memory structures."""
        self._interaction_count += 1
        interaction_text = f"User: {user_query}\nAssistant: {ai_response}"
        interaction_id = f"interaction_{self._interaction_count}"

        # 1. Focus buffer
        self.focus.add("user", user_query)
        self.focus.add("assistant", ai_response)

        # 2. Semantic tree
        self.tree.add_interaction(
            interaction_text,
            metadata={"turn": self._interaction_count},
        )

        # 3. Entity graph — extract from both query and response
        self._extractor.update_graph(user_query, self.graph)
        self._extractor.update_graph(ai_response, self.graph)

        log.info("Stored interaction #%d (graph: %d nodes, tree: %d leaves, buffer: %d/%d)",
                 self._interaction_count, self.graph.node_count,
                 self.tree.leaf_count, self.focus.size, self.focus.max_size)

    # ── Prompt building ──────────────────────────────────────────────

    def build_prompt(self, user_query: str) -> str:
        """Build the full prompt: [Context Block] + [User Query].

        The context block includes relevant past interactions, related
        knowledge, and recent conversation — all within the token budget.
        """
        context = self.retrieve_context(user_query)

        parts: list[str] = []
        if context:
            parts.append(context)
        parts.append(f"## Current Query\n{user_query}")

        return "\n\n".join(parts)

    # ── Chat wrapper ─────────────────────────────────────────────────

    def wrap_chat(
        self, chat_func: Callable[[str], str]
    ) -> Callable[[str], str]:
        """Wrap a chat function with HCE context management.

        The wrapped function:
        1. Builds an enriched prompt with retrieved context.
        2. Calls the original chat function.
        3. Stores both query and response in HCE memory.
        4. Returns the original response.
        """
        @wraps(chat_func)
        def wrapped(user_query: str) -> str:
            prompt = self.build_prompt(user_query)
            response = chat_func(prompt)
            self.update(user_query, response)
            return response

        return wrapped

    # ── Serialization ────────────────────────────────────────────────

    _SCHEMA_VERSION = 1

    def save(self, directory: str | Path) -> None:
        """Persist all HCE state to a directory."""
        dirpath = Path(directory)
        dirpath.mkdir(parents=True, exist_ok=True)

        self.graph.save(dirpath / "entity_graph.json")
        self.tree.save(dirpath / "semantic_tree.json")

        state = {
            "schema_version": self._SCHEMA_VERSION,
            "interaction_count": self._interaction_count,
            "context_budget": self._budget,
            "max_focus_entries": self.focus.max_size,
            "focus_buffer": self.focus.to_list(),
        }
        atomic_write(dirpath / "pipeline_state.json", json.dumps(state, indent=2))
        log.info("Saved HCE pipeline state v%d to %s",
                 self._SCHEMA_VERSION, dirpath)

    @classmethod
    def load(cls, directory: str | Path) -> HCEPipeline:
        """Restore HCE pipeline from a saved directory."""
        dirpath = Path(directory)

        state = json.loads((dirpath / "pipeline_state.json").read_text())
        version = state.get("schema_version", 1)
        if version > cls._SCHEMA_VERSION:
            raise ValueError(
                f"Pipeline state in {dirpath} has schema version {version}, "
                f"but this code only supports up to version {cls._SCHEMA_VERSION}. "
                f"Please upgrade HCE."
            )
        graph = EntityGraph.load(dirpath / "entity_graph.json")
        tree = SemanticTree.load(dirpath / "semantic_tree.json")
        focus = SlidingFocusBuffer.from_list(
            state["focus_buffer"],
            max_size=state["max_focus_entries"],
        )

        pipeline = cls(
            max_focus_entries=state["max_focus_entries"],
            context_budget=state["context_budget"],
            graph=graph,
            tree=tree,
        )
        pipeline.focus = focus
        pipeline._interaction_count = state["interaction_count"]

        log.info("Loaded HCE pipeline v%d from %s (%d interactions)",
                 version, dirpath, pipeline._interaction_count)
        return pipeline
