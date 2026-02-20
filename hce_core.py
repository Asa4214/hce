"""
HCE Phase 1: EntityGraph and Spreading Activation.

Provides a typed property graph (wrapping networkx.MultiDiGraph) and
a spreading-activation retrieval algorithm for associative recall.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx

log = logging.getLogger(__name__)


def atomic_write(path: str | Path, content: str) -> None:
    """Write *content* to *path* atomically.

    Writes to a temporary file in the same directory, then uses
    ``os.replace`` (atomic on POSIX) to swap it into place.  This
    guarantees the target file is never left in a half-written state.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode())
        os.fsync(fd)
        os.close(fd)
        closed = True
        os.replace(tmp, target)
    except BaseException:
        if not closed:
            os.close(fd)
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ── File Locking ─────────────────────────────────────────────────────────

class FileLock:
    """POSIX advisory file lock using ``fcntl.flock()``.

    Supports exclusive (write) and shared (read) modes with a configurable
    timeout.  Designed to protect ``~/.hce_state/`` from concurrent access
    by multiple MCP server instances.

    Usage::

        with FileLock(STATE_DIR / ".lock"):
            # exclusive access
            pipeline.save(STATE_DIR)

        with FileLock(STATE_DIR / ".lock", shared=True):
            # shared (read) access
            pipeline = HCEPipeline.load(STATE_DIR)
    """

    def __init__(
        self,
        lock_path: str | Path,
        shared: bool = False,
        timeout: float = 10.0,
    ) -> None:
        self._lock_path = Path(lock_path)
        self._shared = shared
        self._timeout = timeout
        self._fd: int | None = None

    def __enter__(self) -> FileLock:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)

        op = fcntl.LOCK_SH if self._shared else fcntl.LOCK_EX
        deadline = time.monotonic() + self._timeout

        while True:
            try:
                fcntl.flock(self._fd, op | fcntl.LOCK_NB)
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(self._fd)
                    self._fd = None
                    mode = "shared" if self._shared else "exclusive"
                    raise TimeoutError(
                        f"Could not acquire {mode} lock on "
                        f"{self._lock_path} within {self._timeout}s"
                    )
                time.sleep(0.05)

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None


# ── Enums ────────────────────────────────────────────────────────────────

class NodeType(Enum):
    FILE = "file"
    FUNCTION = "function"
    CONCEPT = "concept"
    PERSON = "person"
    EVENT = "event"


class EdgeType(Enum):
    IMPORTS = "imports"
    CALLS = "calls"
    RELATES_TO = "relates_to"
    PART_OF = "part_of"


# ── Activation Result ────────────────────────────────────────────────────

@dataclass
class ActivationResult:
    node_id: str
    score: float
    node_type: NodeType
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Entity Graph ─────────────────────────────────────────────────────────

class EntityGraph:
    """Typed property graph backed by networkx.MultiDiGraph."""

    def __init__(self) -> None:
        self._graph = nx.MultiDiGraph()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ── Node operations ──────────────────────────────────────────────

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        label: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a typed node. Merges metadata if node exists; raises if type conflicts."""
        if self._graph.has_node(node_id):
            existing = self._graph.nodes[node_id]
            if existing["node_type"] != node_type.value:
                raise ValueError(
                    f"Node '{node_id}' already exists with type "
                    f"'{existing['node_type']}', cannot change to '{node_type.value}'"
                )
            existing["label"] = label
            if metadata:
                existing.setdefault("metadata", {}).update(metadata)
        else:
            self._graph.add_node(
                node_id,
                node_type=node_type.value,
                label=label,
                metadata=metadata or {},
            )

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return node attribute dict or None if not found."""
        if not self._graph.has_node(node_id):
            return None
        return dict(self._graph.nodes[node_id])

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges."""
        self._graph.remove_node(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> list[str]:
        """Return node IDs matching the given type."""
        return [
            nid
            for nid, data in self._graph.nodes(data=True)
            if data.get("node_type") == node_type.value
        ]

    def find_nodes(
        self,
        label_contains: str | None = None,
        node_type: NodeType | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[str]:
        """Search nodes with AND-combined filters."""
        results: list[str] = []
        for nid, data in self._graph.nodes(data=True):
            if node_type and data.get("node_type") != node_type.value:
                continue
            if label_contains and label_contains.lower() not in data.get("label", "").lower():
                continue
            if metadata_filter:
                meta = data.get("metadata", {})
                if not all(meta.get(k) == v for k, v in metadata_filter.items()):
                    continue
            results.append(nid)
        return results

    # ── Edge operations ──────────────────────────────────────────────

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a typed, weighted edge. Raises KeyError if either node is missing."""
        for nid in (source_id, target_id):
            if not self._graph.has_node(nid):
                raise KeyError(f"Node '{nid}' does not exist")
        self._graph.add_edge(
            source_id,
            target_id,
            edge_type=edge_type.value,
            weight=weight,
            metadata=metadata or {},
        )

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove all edges between source and target."""
        while self._graph.has_edge(source_id, target_id):
            keys = list(self._graph[source_id][target_id])
            self._graph.remove_edge(source_id, target_id, key=keys[0])

    def get_neighbors(
        self,
        node_id: str,
        edge_type: EdgeType | None = None,
        direction: str = "both",
    ) -> list[tuple[str, dict[str, Any]]]:
        """Get neighbors with optional type/direction filter.

        Returns list of (neighbor_id, edge_data) tuples.
        direction: "out", "in", or "both".
        """
        neighbors: list[tuple[str, dict[str, Any]]] = []
        seen: set[tuple[str, str, int]] = set()

        if direction in ("out", "both"):
            for _, target, key, data in self._graph.out_edges(node_id, data=True, keys=True):
                if edge_type and data.get("edge_type") != edge_type.value:
                    continue
                ident = (target, data.get("edge_type", ""), key)
                if ident not in seen:
                    seen.add(ident)
                    neighbors.append((target, dict(data)))

        if direction in ("in", "both"):
            for source, _, key, data in self._graph.in_edges(node_id, data=True, keys=True):
                if edge_type and data.get("edge_type") != edge_type.value:
                    continue
                ident = (source, data.get("edge_type", ""), key)
                if ident not in seen:
                    seen.add(ident)
                    neighbors.append((source, dict(data)))

        return neighbors

    # ── Subgraph ─────────────────────────────────────────────────────

    def subgraph(self, node_ids: list[str]) -> EntityGraph:
        """Extract a subgraph as a new EntityGraph."""
        sub = EntityGraph()
        sub._graph = self._graph.subgraph(node_ids).copy()
        return sub

    # ── Serialization ────────────────────────────────────────────────

    _SCHEMA_VERSION = 1

    def save(self, path: str | Path) -> None:
        """Serialize graph to JSON."""
        data = {
            "schema_version": self._SCHEMA_VERSION,
            "nodes": [
                {"id": nid, **attrs}
                for nid, attrs in self._graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, **d}
                for u, v, d in self._graph.edges(data=True)
            ],
        }
        atomic_write(path, json.dumps(data, indent=2))
        log.info("Saved EntityGraph v%d (%d nodes, %d edges) to %s",
                 self._SCHEMA_VERSION, self.node_count, self.edge_count, path)

    @classmethod
    def load(cls, path: str | Path) -> EntityGraph:
        """Deserialize graph from JSON."""
        raw = json.loads(Path(path).read_text())
        version = raw.get("schema_version", 1)
        if version > cls._SCHEMA_VERSION:
            raise ValueError(
                f"EntityGraph file {path} has schema version {version}, "
                f"but this code only supports up to version {cls._SCHEMA_VERSION}. "
                f"Please upgrade HCE."
            )
        graph = cls()
        for node in raw["nodes"]:
            nid = node.pop("id")
            graph._graph.add_node(nid, **node)
        for edge in raw["edges"]:
            src = edge.pop("source")
            tgt = edge.pop("target")
            graph._graph.add_edge(src, tgt, **edge)
        log.info("Loaded EntityGraph v%d (%d nodes, %d edges) from %s",
                 version, graph.node_count, graph.edge_count, path)
        return graph


# ── Spreading Activation ─────────────────────────────────────────────────

def spreading_activation(
    graph: EntityGraph,
    seed_nodes: dict[str, float],
    decay: float = 0.5,
    max_iterations: int = 3,
    min_activation: float = 0.01,
    top_n: int = 10,
) -> list[ActivationResult]:
    """Run spreading activation over the EntityGraph.

    1. Initialize activation energy from seed_nodes (node_id -> energy).
    2. Each iteration: propagate energy * decay * edge_weight to neighbors
       (both directions), accumulating additively.
    3. Prune nodes below min_activation.
    4. Return top_n results sorted by score descending.
    """
    # Validate seeds exist
    for nid in seed_nodes:
        if not graph.has_node(nid):
            raise KeyError(f"Seed node '{nid}' does not exist in graph")

    log.debug("Spreading activation: %d seeds, decay=%.2f, max_iter=%d",
              len(seed_nodes), decay, max_iterations)
    activation: dict[str, float] = dict(seed_nodes)

    for _ in range(max_iterations):
        new_activation: dict[str, float] = {}

        for nid, energy in activation.items():
            if energy < min_activation:
                continue

            neighbors = graph.get_neighbors(nid, direction="both")
            for neighbor_id, edge_data in neighbors:
                weight = edge_data.get("weight", 1.0)
                spread = energy * decay * weight
                new_activation[neighbor_id] = (
                    new_activation.get(neighbor_id, 0.0) + spread
                )

        # Merge: keep the max of old activation and new propagated energy
        for nid, energy in new_activation.items():
            activation[nid] = max(activation.get(nid, 0.0), energy)

        # Prune
        activation = {
            nid: e for nid, e in activation.items() if e >= min_activation
        }

    # Build results
    results: list[ActivationResult] = []
    for nid, score in activation.items():
        node_data = graph.get_node(nid)
        if node_data is None:
            continue
        results.append(
            ActivationResult(
                node_id=nid,
                score=score,
                node_type=NodeType(node_data["node_type"]),
                label=node_data.get("label", ""),
                metadata=node_data.get("metadata", {}),
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    top = results[:top_n]
    log.debug("Spreading activation: %d active nodes, returning top %d",
              len(results), len(top))
    return top
