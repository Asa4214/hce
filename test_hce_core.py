"""Tests for hce_core: EntityGraph and spreading_activation."""

import json
import tempfile
from pathlib import Path

import pytest

from hce_core import (
    ActivationResult,
    EdgeType,
    EntityGraph,
    NodeType,
    spreading_activation,
)


# ── EntityGraph: Node operations ─────────────────────────────────────────

class TestEntityGraphNodes:
    def test_add_and_get_node(self):
        g = EntityGraph()
        g.add_node("f1", NodeType.FILE, "main.py", {"lang": "python"})
        node = g.get_node("f1")
        assert node is not None
        assert node["node_type"] == "file"
        assert node["label"] == "main.py"
        assert node["metadata"]["lang"] == "python"

    def test_add_node_merge_metadata(self):
        g = EntityGraph()
        g.add_node("f1", NodeType.FILE, "main.py", {"lang": "python"})
        g.add_node("f1", NodeType.FILE, "main.py (updated)", {"version": "2"})
        node = g.get_node("f1")
        assert node["label"] == "main.py (updated)"
        assert node["metadata"]["lang"] == "python"
        assert node["metadata"]["version"] == "2"

    def test_add_node_type_conflict_raises(self):
        g = EntityGraph()
        g.add_node("x", NodeType.FILE, "a.py")
        with pytest.raises(ValueError, match="cannot change"):
            g.add_node("x", NodeType.FUNCTION, "func_a")

    def test_get_nonexistent_node_returns_none(self):
        g = EntityGraph()
        assert g.get_node("nope") is None

    def test_has_node(self):
        g = EntityGraph()
        g.add_node("a", NodeType.CONCEPT, "testing")
        assert g.has_node("a")
        assert not g.has_node("b")

    def test_remove_node(self):
        g = EntityGraph()
        g.add_node("a", NodeType.CONCEPT, "test")
        g.remove_node("a")
        assert not g.has_node("a")
        assert g.node_count == 0

    def test_node_count(self):
        g = EntityGraph()
        assert g.node_count == 0
        g.add_node("a", NodeType.FILE, "a.py")
        g.add_node("b", NodeType.FILE, "b.py")
        assert g.node_count == 2

    def test_get_nodes_by_type(self):
        g = EntityGraph()
        g.add_node("f1", NodeType.FILE, "a.py")
        g.add_node("f2", NodeType.FILE, "b.py")
        g.add_node("c1", NodeType.CONCEPT, "auth")
        assert sorted(g.get_nodes_by_type(NodeType.FILE)) == ["f1", "f2"]
        assert g.get_nodes_by_type(NodeType.CONCEPT) == ["c1"]
        assert g.get_nodes_by_type(NodeType.PERSON) == []

    def test_find_nodes_label_contains(self):
        g = EntityGraph()
        g.add_node("f1", NodeType.FILE, "login.py")
        g.add_node("f2", NodeType.FILE, "logout.py")
        g.add_node("f3", NodeType.FILE, "main.py")
        assert sorted(g.find_nodes(label_contains="log")) == ["f1", "f2"]

    def test_find_nodes_combined_filters(self):
        g = EntityGraph()
        g.add_node("f1", NodeType.FILE, "auth.py", {"lang": "python"})
        g.add_node("f2", NodeType.FUNCTION, "authenticate", {"lang": "python"})
        g.add_node("f3", NodeType.FILE, "auth.js", {"lang": "js"})
        result = g.find_nodes(
            label_contains="auth",
            node_type=NodeType.FILE,
            metadata_filter={"lang": "python"},
        )
        assert result == ["f1"]


# ── EntityGraph: Edge operations ─────────────────────────────────────────

class TestEntityGraphEdges:
    def _make_graph(self) -> EntityGraph:
        g = EntityGraph()
        g.add_node("a", NodeType.FILE, "a.py")
        g.add_node("b", NodeType.FILE, "b.py")
        g.add_node("c", NodeType.FUNCTION, "func_c")
        return g

    def test_add_and_count_edges(self):
        g = self._make_graph()
        g.add_edge("a", "b", EdgeType.IMPORTS)
        assert g.edge_count == 1

    def test_add_edge_missing_node_raises(self):
        g = self._make_graph()
        with pytest.raises(KeyError, match="does not exist"):
            g.add_edge("a", "missing", EdgeType.IMPORTS)

    def test_multiple_edge_types_between_same_pair(self):
        g = self._make_graph()
        g.add_edge("a", "b", EdgeType.IMPORTS)
        g.add_edge("a", "b", EdgeType.RELATES_TO)
        assert g.edge_count == 2

    def test_remove_edge(self):
        g = self._make_graph()
        g.add_edge("a", "b", EdgeType.IMPORTS)
        g.add_edge("a", "b", EdgeType.RELATES_TO)
        g.remove_edge("a", "b")
        assert g.edge_count == 0

    def test_get_neighbors_both_directions(self):
        g = self._make_graph()
        g.add_edge("a", "b", EdgeType.IMPORTS)
        g.add_edge("c", "a", EdgeType.CALLS)
        neighbors = g.get_neighbors("a", direction="both")
        neighbor_ids = {n[0] for n in neighbors}
        assert neighbor_ids == {"b", "c"}

    def test_get_neighbors_out_only(self):
        g = self._make_graph()
        g.add_edge("a", "b", EdgeType.IMPORTS)
        g.add_edge("c", "a", EdgeType.CALLS)
        out = g.get_neighbors("a", direction="out")
        assert [n[0] for n in out] == ["b"]

    def test_get_neighbors_in_only(self):
        g = self._make_graph()
        g.add_edge("a", "b", EdgeType.IMPORTS)
        g.add_edge("c", "a", EdgeType.CALLS)
        inn = g.get_neighbors("a", direction="in")
        assert [n[0] for n in inn] == ["c"]

    def test_get_neighbors_filter_by_edge_type(self):
        g = self._make_graph()
        g.add_edge("a", "b", EdgeType.IMPORTS)
        g.add_edge("a", "c", EdgeType.CALLS)
        imports_only = g.get_neighbors("a", edge_type=EdgeType.IMPORTS)
        assert len(imports_only) == 1
        assert imports_only[0][0] == "b"


# ── EntityGraph: Subgraph ────────────────────────────────────────────────

class TestEntityGraphSubgraph:
    def test_subgraph_preserves_nodes_and_edges(self):
        g = EntityGraph()
        g.add_node("a", NodeType.FILE, "a.py")
        g.add_node("b", NodeType.FILE, "b.py")
        g.add_node("c", NodeType.FILE, "c.py")
        g.add_edge("a", "b", EdgeType.IMPORTS)
        g.add_edge("b", "c", EdgeType.IMPORTS)
        sub = g.subgraph(["a", "b"])
        assert sub.node_count == 2
        assert sub.edge_count == 1
        assert sub.has_node("a")
        assert not sub.has_node("c")


# ── EntityGraph: Serialization ───────────────────────────────────────────

class TestEntityGraphSerialization:
    def test_save_and_load_roundtrip(self):
        g = EntityGraph()
        g.add_node("f1", NodeType.FILE, "main.py", {"lang": "python"})
        g.add_node("f2", NodeType.FUNCTION, "main", {"lines": 10})
        g.add_edge("f1", "f2", EdgeType.PART_OF, weight=0.8)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        g.save(path)

        # Verify JSON is human-readable
        raw = json.loads(Path(path).read_text())
        assert "nodes" in raw
        assert "edges" in raw

        loaded = EntityGraph.load(path)
        assert loaded.node_count == 2
        assert loaded.edge_count == 1
        node = loaded.get_node("f1")
        assert node["label"] == "main.py"

        Path(path).unlink()


# ── Spreading Activation ─────────────────────────────────────────────────

class TestSpreadingActivation:
    def _build_chain(self) -> EntityGraph:
        """A -> B -> C -> D (linear chain)."""
        g = EntityGraph()
        for nid, label in [("A", "alpha"), ("B", "beta"), ("C", "gamma"), ("D", "delta")]:
            g.add_node(nid, NodeType.CONCEPT, label)
        g.add_edge("A", "B", EdgeType.RELATES_TO, weight=1.0)
        g.add_edge("B", "C", EdgeType.RELATES_TO, weight=1.0)
        g.add_edge("C", "D", EdgeType.RELATES_TO, weight=1.0)
        return g

    def test_seed_node_appears_in_results(self):
        g = self._build_chain()
        results = spreading_activation(g, {"A": 1.0})
        ids = [r.node_id for r in results]
        assert "A" in ids

    def test_results_sorted_descending(self):
        g = self._build_chain()
        results = spreading_activation(g, {"A": 1.0})
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_nearby_nodes_score_higher_than_distant(self):
        g = self._build_chain()
        results = spreading_activation(g, {"A": 1.0})
        score_map = {r.node_id: r.score for r in results}
        assert score_map["A"] >= score_map.get("B", 0)
        assert score_map.get("B", 0) >= score_map.get("C", 0)

    def test_energy_decays_with_distance(self):
        g = self._build_chain()
        results = spreading_activation(g, {"A": 1.0}, decay=0.5, max_iterations=3)
        score_map = {r.node_id: r.score for r in results}
        # B gets ~0.5 from A directly, plus echo energy from C->B in later iterations
        assert 0.4 <= score_map.get("B", 0) <= 0.7

    def test_multiple_seeds(self):
        g = self._build_chain()
        results = spreading_activation(g, {"A": 1.0, "D": 1.0})
        score_map = {r.node_id: r.score for r in results}
        # Middle nodes should get energy from both sides
        assert score_map.get("B", 0) > 0
        assert score_map.get("C", 0) > 0

    def test_top_n_limits_results(self):
        g = self._build_chain()
        results = spreading_activation(g, {"A": 1.0}, top_n=2)
        assert len(results) <= 2

    def test_missing_seed_raises(self):
        g = self._build_chain()
        with pytest.raises(KeyError, match="Seed node"):
            spreading_activation(g, {"NOPE": 1.0})

    def test_edge_weight_affects_propagation(self):
        g = EntityGraph()
        g.add_node("X", NodeType.CONCEPT, "center")
        g.add_node("Y", NodeType.CONCEPT, "strong")
        g.add_node("Z", NodeType.CONCEPT, "weak")
        g.add_edge("X", "Y", EdgeType.RELATES_TO, weight=1.0)
        g.add_edge("X", "Z", EdgeType.RELATES_TO, weight=0.1)
        results = spreading_activation(g, {"X": 1.0}, decay=0.5, max_iterations=1)
        score_map = {r.node_id: r.score for r in results}
        assert score_map.get("Y", 0) > score_map.get("Z", 0)

    def test_multipath_accumulation(self):
        """Node reachable via two paths should accumulate more energy."""
        g = EntityGraph()
        for nid in ["S", "M1", "M2", "T"]:
            g.add_node(nid, NodeType.CONCEPT, nid)
        g.add_edge("S", "M1", EdgeType.RELATES_TO)
        g.add_edge("S", "M2", EdgeType.RELATES_TO)
        g.add_edge("M1", "T", EdgeType.RELATES_TO)
        g.add_edge("M2", "T", EdgeType.RELATES_TO)
        results = spreading_activation(g, {"S": 1.0}, decay=0.5, max_iterations=2)
        score_map = {r.node_id: r.score for r in results}
        # T gets energy from two paths: S->M1->T and S->M2->T
        # Each M gets 0.5, then T gets 0.5*0.5 = 0.25 from each = 0.5 total
        # T matches M1 despite being one hop further — multipath accumulation works
        assert score_map.get("T", 0) >= score_map.get("M1", 0)

    def test_result_has_correct_types(self):
        g = self._build_chain()
        results = spreading_activation(g, {"A": 1.0})
        for r in results:
            assert isinstance(r, ActivationResult)
            assert isinstance(r.node_type, NodeType)
            assert isinstance(r.score, float)
