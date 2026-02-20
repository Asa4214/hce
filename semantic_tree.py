"""
HCE Phase 2: SemanticTree and Hierarchical Relevance Search.

Provides a Merkle/Aggregation tree for long-term episodic memory.
Leaves are raw interaction turns; internal nodes are vectorized summaries.
HRS performs recursive vector similarity with early branch pruning.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hce_core import atomic_write

log = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────────────────

Summarizer = Callable[[list[str]], str]
Vectorizer = Callable[[str], list[float]]

_VECTOR_DIM = 256


# ── Utility functions ────────────────────────────────────────────────────

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased."""
    return re.findall(r"[a-z0-9]+", text.lower())


def default_vectorizer(text: str) -> list[float]:
    """Feature-hashing vectorizer: hash tokens into a fixed-size vector."""
    vec = [0.0] * _VECTOR_DIM
    tokens = _tokenize(text)
    if not tokens:
        return vec
    counts = Counter(tokens)
    for token, count in counts.items():
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        idx = h % _VECTOR_DIM
        sign = 1.0 if (h // _VECTOR_DIM) % 2 == 0 else -1.0
        vec[idx] += sign * count
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def make_embedding_vectorizer(
    model_name: str = "all-MiniLM-L6-v2",
) -> Vectorizer:
    """Create a vectorizer backed by sentence-transformers.

    The model is loaded lazily on first call, not at import time.
    Requires ``pip install sentence-transformers``.
    """
    _model = None

    def _vectorize(text: str) -> list[float]:
        nonlocal _model
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embedding vectorizer. "
                    "Install it with: pip install 'hce[embeddings]'"
                )
            _model = SentenceTransformer(model_name)
            log.info("Loaded embedding model '%s' (dim=%d)",
                     model_name, _model.get_sentence_embedding_dimension())
        return _model.encode(text).tolist()

    return _vectorize


def get_default_vectorizer() -> Vectorizer:
    """Return the best available vectorizer.

    Uses sentence-transformers if installed, otherwise falls back to
    the feature-hashing vectorizer.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        log.info("sentence-transformers available — using embedding vectorizer")
        return make_embedding_vectorizer()
    except ImportError:
        log.info("sentence-transformers not installed — using feature-hash vectorizer")
        return default_vectorizer


def default_summarizer(texts: list[str]) -> str:
    """Extractive summarizer: first sentence of each text, truncated."""
    sentences = []
    for text in texts:
        first = text.split(".")[0].strip()
        if first:
            sentences.append(first[:200])
    return ". ".join(sentences) + "." if sentences else ""


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class TreeNode:
    node_id: str
    content: str
    vector: list[float]
    children_ids: list[str] = field(default_factory=list)
    parent_id: str | None = None
    is_leaf: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class SearchResult:
    node_id: str
    score: float
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Semantic Tree ────────────────────────────────────────────────────────

class SemanticTree:
    """Merkle/Aggregation tree for long-term episodic memory."""

    def __init__(
        self,
        summarizer: Summarizer | None = None,
        vectorizer: Vectorizer | None = None,
        branch_factor: int = 4,
    ) -> None:
        if branch_factor < 2:
            raise ValueError(f"branch_factor must be >= 2, got {branch_factor}")
        self._nodes: dict[str, TreeNode] = {}
        self._leaf_ids: list[str] = []  # insertion order
        self._root_id: str | None = None
        self._summarizer = summarizer or default_summarizer
        self._vectorizer = vectorizer or get_default_vectorizer()
        self._branch_factor = branch_factor
        self._next_id = 0
        self._dirty = False  # True when leaves added but tree not yet rebuilt

    # ── Properties ───────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        self._ensure_built()
        return len(self._nodes)

    @property
    def leaf_count(self) -> int:
        return len(self._leaf_ids)

    @property
    def root_id(self) -> str | None:
        self._ensure_built()
        return self._root_id

    # ── ID generation ────────────────────────────────────────────────

    def _gen_id(self, prefix: str = "node") -> str:
        nid = f"{prefix}_{self._next_id}"
        self._next_id += 1
        return nid

    # ── Public API ───────────────────────────────────────────────────

    def add_interaction(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a raw interaction as a leaf node.

        The internal tree structure is rebuilt lazily — only when the tree
        is searched or saved, not on every insert.  This makes consecutive
        inserts O(1) each instead of O(n).
        """
        node_id = self._gen_id("leaf")
        vector = self._vectorizer(content)
        node = TreeNode(
            node_id=node_id,
            content=content,
            vector=vector,
            is_leaf=True,
            metadata=metadata or {},
            timestamp=time.time(),
        )
        self._nodes[node_id] = node
        self._leaf_ids.append(node_id)
        self._dirty = True
        log.info("Added interaction %s (leaf %d, rebuild deferred)",
                 node_id, self.leaf_count)
        return node_id

    def _ensure_built(self) -> None:
        """Rebuild the tree if it has been modified since the last build."""
        if self._dirty:
            self._rebuild_tree()
            self._dirty = False
            log.debug("Lazy rebuild complete: %d nodes", self.node_count)

    def get_node(self, node_id: str) -> TreeNode | None:
        return self._nodes.get(node_id)

    def get_leaves(self) -> list[TreeNode]:
        """Return all leaf nodes in insertion order."""
        return [self._nodes[lid] for lid in self._leaf_ids]

    # ── Tree construction ────────────────────────────────────────────

    def _rebuild_tree(self) -> None:
        """Build the aggregation tree bottom-up from leaves."""
        # Clear all internal nodes
        internal_ids = [
            nid for nid, node in self._nodes.items() if not node.is_leaf
        ]
        for nid in internal_ids:
            del self._nodes[nid]

        # Reset parent pointers on leaves
        for lid in self._leaf_ids:
            self._nodes[lid].parent_id = None

        if not self._leaf_ids:
            self._root_id = None
            return

        if len(self._leaf_ids) == 1:
            self._root_id = self._leaf_ids[0]
            return

        # Build bottom-up: group current level into parents
        current_level = list(self._leaf_ids)

        while len(current_level) > 1:
            next_level: list[str] = []

            for i in range(0, len(current_level), self._branch_factor):
                group = current_level[i : i + self._branch_factor]

                if len(group) == 1 and len(current_level) > 1:
                    # Single leftover node — just promote it
                    next_level.append(group[0])
                    continue

                # Create internal node summarizing this group
                child_contents = [self._nodes[cid].content for cid in group]
                summary = self._summarizer(child_contents)
                vector = self._vectorizer(summary)
                parent_id = self._gen_id("internal")

                parent = TreeNode(
                    node_id=parent_id,
                    content=summary,
                    vector=vector,
                    children_ids=list(group),
                    is_leaf=False,
                    timestamp=max(self._nodes[cid].timestamp for cid in group),
                )
                self._nodes[parent_id] = parent

                for cid in group:
                    self._nodes[cid].parent_id = parent_id

                next_level.append(parent_id)

            current_level = next_level

        self._root_id = current_level[0]

    # ── Serialization ────────────────────────────────────────────────

    _SCHEMA_VERSION = 1

    def save(self, path: str | Path) -> None:
        """Serialize tree to JSON."""
        self._ensure_built()
        data = {
            "schema_version": self._SCHEMA_VERSION,
            "branch_factor": self._branch_factor,
            "next_id": self._next_id,
            "leaf_ids": self._leaf_ids,
            "root_id": self._root_id,
            "nodes": {
                nid: {
                    "node_id": node.node_id,
                    "content": node.content,
                    "vector": node.vector,
                    "children_ids": node.children_ids,
                    "parent_id": node.parent_id,
                    "is_leaf": node.is_leaf,
                    "metadata": node.metadata,
                    "timestamp": node.timestamp,
                }
                for nid, node in self._nodes.items()
            },
        }
        atomic_write(path, json.dumps(data, indent=2))
        log.info("Saved SemanticTree v%d (%d leaves, %d total nodes) to %s",
                 self._SCHEMA_VERSION, self.leaf_count, self.node_count, path)

    @classmethod
    def load(cls, path: str | Path) -> SemanticTree:
        """Deserialize tree from JSON."""
        raw = json.loads(Path(path).read_text())
        version = raw.get("schema_version", 1)
        if version > cls._SCHEMA_VERSION:
            raise ValueError(
                f"SemanticTree file {path} has schema version {version}, "
                f"but this code only supports up to version {cls._SCHEMA_VERSION}. "
                f"Please upgrade HCE."
            )
        tree = cls(branch_factor=raw["branch_factor"])
        tree._next_id = raw["next_id"]
        tree._leaf_ids = raw["leaf_ids"]
        tree._root_id = raw["root_id"]
        for nid, ndata in raw["nodes"].items():
            tree._nodes[nid] = TreeNode(**ndata)
        log.info("Loaded SemanticTree v%d (%d leaves, %d total nodes) from %s",
                 version, tree.leaf_count, tree.node_count, path)
        return tree


# ── Hierarchical Relevance Search ────────────────────────────────────────

def hierarchical_relevance_search(
    tree: SemanticTree,
    query: str,
    threshold: float = 0.1,
    top_k: int = 5,
) -> list[SearchResult]:
    """Search the SemanticTree using recursive vector similarity with pruning.

    1. Vectorize the query using the tree's vectorizer.
    2. Starting from root, compute similarity with each child.
    3. Prune children below threshold.
    4. Recurse into surviving children (best-first).
    5. At leaves, collect candidates.
    6. Return top-k by score.
    """
    if tree.root_id is None:
        return []

    query_vector = tree._vectorizer(query)
    candidates: list[SearchResult] = []

    def _search_node(node_id: str, inherited_score: float) -> None:
        node = tree.get_node(node_id)
        if node is None:
            return

        sim = cosine_similarity(query_vector, node.vector)
        # Combine: direct similarity weighted with inherited context
        score = sim * 0.7 + inherited_score * 0.3

        if node.is_leaf:
            candidates.append(
                SearchResult(
                    node_id=node.node_id,
                    score=score,
                    content=node.content,
                    metadata=node.metadata,
                )
            )
            return

        # Score and sort children, prune below threshold
        child_scores: list[tuple[str, float]] = []
        for cid in node.children_ids:
            child = tree.get_node(cid)
            if child is None:
                continue
            child_sim = cosine_similarity(query_vector, child.vector)
            if child_sim >= threshold:
                child_scores.append((cid, child_sim))

        # Best-first traversal
        child_scores.sort(key=lambda x: x[1], reverse=True)

        for cid, child_sim in child_scores:
            _search_node(cid, score)

    _search_node(tree.root_id, 0.0)

    candidates.sort(key=lambda r: r.score, reverse=True)
    top = candidates[:top_k]
    log.debug("HRS: %d candidates found, returning top %d (threshold=%.2f)",
              len(candidates), len(top), threshold)
    return top
