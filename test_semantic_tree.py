"""Tests for semantic_tree: SemanticTree and Hierarchical Relevance Search."""

import json
import tempfile
from pathlib import Path

import pytest

from semantic_tree import (
    SearchResult,
    SemanticTree,
    TreeNode,
    cosine_similarity,
    default_summarizer,
    default_vectorizer,
    hierarchical_relevance_search,
)


# ── Utility functions ────────────────────────────────────────────────────

class TestCosineSimlarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert cosine_similarity(a, b) == 0.0


class TestDefaultVectorizer:
    def test_returns_correct_dimension(self):
        vec = default_vectorizer("hello world")
        assert len(vec) == 256

    def test_normalized_to_unit_length(self):
        import math
        vec = default_vectorizer("this is a test sentence for vectorization")
        norm = math.sqrt(sum(x * x for x in vec))
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_empty_string_returns_zero_vector(self):
        vec = default_vectorizer("")
        assert all(x == 0.0 for x in vec)

    def test_similar_texts_have_higher_similarity(self):
        v1 = default_vectorizer("python programming language")
        v2 = default_vectorizer("python coding language")
        v3 = default_vectorizer("banana fruit yellow tropical")
        sim_related = cosine_similarity(v1, v2)
        sim_unrelated = cosine_similarity(v1, v3)
        assert sim_related > sim_unrelated

    def test_identical_text_similarity_is_one(self):
        v = default_vectorizer("exact same text")
        assert cosine_similarity(v, v) == pytest.approx(1.0)


class TestDefaultSummarizer:
    def test_extracts_first_sentences(self):
        texts = [
            "First doc here. More details follow.",
            "Second doc content. Extra info.",
        ]
        summary = default_summarizer(texts)
        assert "First doc here" in summary
        assert "Second doc content" in summary

    def test_empty_input(self):
        assert default_summarizer([]) == ""

    def test_truncates_long_sentences(self):
        long_text = "x" * 500 + ". rest."
        summary = default_summarizer([long_text])
        assert len(summary) <= 210  # 200 char truncation + ". "


# ── TreeNode ─────────────────────────────────────────────────────────────

class TestTreeNode:
    def test_dataclass_creation(self):
        node = TreeNode(
            node_id="test_0",
            content="hello",
            vector=[0.1, 0.2],
            is_leaf=True,
        )
        assert node.node_id == "test_0"
        assert node.is_leaf
        assert node.children_ids == []
        assert node.parent_id is None


# ── SemanticTree ─────────────────────────────────────────────────────────

class TestSemanticTree:
    def test_empty_tree(self):
        tree = SemanticTree()
        assert tree.node_count == 0
        assert tree.leaf_count == 0
        assert tree.root_id is None

    def test_add_single_interaction(self):
        tree = SemanticTree()
        lid = tree.add_interaction("User asked about Python")
        assert tree.leaf_count == 1
        assert tree.root_id == lid
        node = tree.get_node(lid)
        assert node is not None
        assert node.is_leaf
        assert "Python" in node.content

    def test_add_multiple_interactions(self):
        tree = SemanticTree(branch_factor=2)
        ids = []
        for i in range(4):
            ids.append(tree.add_interaction(f"Interaction {i}"))
        assert tree.leaf_count == 4
        # With 4 leaves and branch_factor=2: 2 internal + 1 root = 7 nodes
        # Actually: 4 leaves -> 2 parents -> 1 root = 7 total
        assert tree.node_count == 7

    def test_tree_structure_with_branch_factor_2(self):
        tree = SemanticTree(branch_factor=2)
        for i in range(4):
            tree.add_interaction(f"Message {i}")
        # Root should be internal with 2 children
        root = tree.get_node(tree.root_id)
        assert root is not None
        assert not root.is_leaf
        assert len(root.children_ids) == 2

    def test_get_leaves_preserves_order(self):
        tree = SemanticTree()
        contents = ["First", "Second", "Third"]
        for c in contents:
            tree.add_interaction(c)
        leaves = tree.get_leaves()
        assert [l.content for l in leaves] == contents

    def test_metadata_stored_on_leaf(self):
        tree = SemanticTree()
        lid = tree.add_interaction("test", metadata={"source": "user"})
        node = tree.get_node(lid)
        assert node.metadata["source"] == "user"

    def test_nonexistent_node_returns_none(self):
        tree = SemanticTree()
        assert tree.get_node("fake_id") is None

    def test_internal_nodes_have_summaries(self):
        tree = SemanticTree(branch_factor=2)
        tree.add_interaction("Python is a programming language")
        tree.add_interaction("JavaScript runs in browsers")
        # With 2 leaves and branch_factor=2, root is an internal node
        root = tree.get_node(tree.root_id)
        assert not root.is_leaf
        assert len(root.content) > 0  # has a summary
        assert len(root.vector) > 0  # has a vector (dim varies by vectorizer)

    def test_odd_number_of_leaves(self):
        tree = SemanticTree(branch_factor=2)
        for i in range(3):
            tree.add_interaction(f"Message {i}")
        # 3 leaves with bf=2: leaves 0,1 -> parent; leaf 2 promoted
        # Then parent + leaf_2 -> root
        assert tree.leaf_count == 3
        root = tree.get_node(tree.root_id)
        assert not root.is_leaf


# ── Serialization ────────────────────────────────────────────────────────

class TestSemanticTreeSerialization:
    def test_save_and_load_roundtrip(self):
        tree = SemanticTree(branch_factor=2)
        tree.add_interaction("First message", metadata={"turn": 1})
        tree.add_interaction("Second message", metadata={"turn": 2})
        tree.add_interaction("Third message", metadata={"turn": 3})

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        tree.save(path)

        # Verify JSON is readable
        raw = json.loads(Path(path).read_text())
        assert "nodes" in raw
        assert "leaf_ids" in raw

        loaded = SemanticTree.load(path)
        assert loaded.leaf_count == 3
        assert loaded.node_count == tree.node_count
        leaves = loaded.get_leaves()
        assert leaves[0].content == "First message"
        assert leaves[0].metadata["turn"] == 1

        Path(path).unlink()

    def test_load_preserves_tree_structure(self):
        tree = SemanticTree(branch_factor=2)
        for i in range(4):
            tree.add_interaction(f"Turn {i}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        tree.save(path)
        loaded = SemanticTree.load(path)

        root = loaded.get_node(loaded.root_id)
        assert root is not None
        assert not root.is_leaf
        assert len(root.children_ids) == 2

        Path(path).unlink()


# ── Hierarchical Relevance Search ────────────────────────────────────────

class TestHierarchicalRelevanceSearch:
    def _build_tree(self) -> SemanticTree:
        tree = SemanticTree(branch_factor=2)
        tree.add_interaction("Python is great for data science and machine learning")
        tree.add_interaction("JavaScript is used for web development and frontend")
        tree.add_interaction("Python has libraries like pandas and numpy for data analysis")
        tree.add_interaction("CSS styles web pages with colors and layouts")
        tree.add_interaction("Machine learning models use training data to make predictions")
        tree.add_interaction("HTML provides the structure of web pages")
        return tree

    def test_empty_tree_returns_empty(self):
        tree = SemanticTree()
        results = hierarchical_relevance_search(tree, "anything")
        assert results == []

    def test_returns_search_results(self):
        tree = self._build_tree()
        results = hierarchical_relevance_search(tree, "Python data science")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_sorted_by_score_descending(self):
        tree = self._build_tree()
        results = hierarchical_relevance_search(tree, "web development")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_results_rank_higher(self):
        tree = self._build_tree()
        results = hierarchical_relevance_search(
            tree, "Python data science machine learning", threshold=0.0
        )
        # Python/ML related content should rank above CSS/HTML
        if len(results) >= 2:
            top_content = results[0].content.lower()
            assert "python" in top_content or "machine learning" in top_content or "data" in top_content

    def test_top_k_limits_results(self):
        tree = self._build_tree()
        results = hierarchical_relevance_search(tree, "programming", top_k=2)
        assert len(results) <= 2

    def test_high_threshold_prunes_results(self):
        tree = self._build_tree()
        all_results = hierarchical_relevance_search(tree, "test", threshold=0.0)
        strict_results = hierarchical_relevance_search(tree, "test", threshold=0.9)
        assert len(strict_results) <= len(all_results)

    def test_single_leaf_tree(self):
        tree = SemanticTree()
        tree.add_interaction("The only interaction about quantum computing")
        results = hierarchical_relevance_search(tree, "quantum computing")
        assert len(results) == 1
        assert "quantum" in results[0].content

    def test_results_contain_metadata(self):
        tree = SemanticTree()
        tree.add_interaction("test content", metadata={"turn": 42})
        results = hierarchical_relevance_search(tree, "test content", threshold=0.0)
        assert results[0].metadata["turn"] == 42

    def test_custom_vectorizer(self):
        """Tree works with a custom vectorizer."""
        call_count = [0]

        def counting_vectorizer(text: str) -> list[float]:
            call_count[0] += 1
            return default_vectorizer(text)

        tree = SemanticTree(vectorizer=counting_vectorizer, branch_factor=2)
        tree.add_interaction("test one")
        tree.add_interaction("test two")
        before = call_count[0]
        hierarchical_relevance_search(tree, "test")
        assert call_count[0] > before  # vectorizer was called for query
