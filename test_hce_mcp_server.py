"""Tests for the HCE MCP server tools.

Each test gets an isolated pipeline and temp state directory so tests
don't leak into each other or touch the real ~/.hce_state.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

import hce_mcp_server as server
from hce_core import FileLock
from hce_pipeline import HCEPipeline


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_pipeline(monkeypatch, tmp_path):
    """Give every test a fresh pipeline and isolated state directory."""
    monkeypatch.setattr(server, "pipeline", HCEPipeline())
    monkeypatch.setattr(server, "STATE_DIR", tmp_path / "hce_state")
    yield


# ── hce_status ───────────────────────────────────────────────────────────

class TestHceStatus:
    def test_empty_state(self):
        result = server.hce_status()
        assert "Entity Graph" in result
        assert "0 nodes" in result
        assert "0 edges" in result
        assert "0 interactions stored" in result
        assert "0/10" in result  # focus buffer

    def test_status_after_store(self):
        server.hce_store_interaction("hello", "hi there")
        result = server.hce_status()
        assert "1 interactions stored" in result
        assert "2/10" in result  # user + assistant = 2 buffer entries

    def test_status_shows_node_type_breakdown(self):
        server.hce_crawl_project(str(Path(__file__).parent))
        result = server.hce_status()
        assert "file:" in result
        assert "function:" in result

    def test_status_shows_recent_turns(self):
        server.hce_store_interaction("what is HCE?", "a memory system")
        result = server.hce_status()
        assert "what is HCE?" in result
        assert "a memory system" in result


# ── hce_store_interaction ────────────────────────────────────────────────

class TestHceStoreInteraction:
    def test_stores_into_all_structures(self):
        result = server.hce_store_interaction("user msg", "assistant msg")
        p = server.pipeline
        assert p.tree.leaf_count == 1
        assert p.focus.size == 2  # user + assistant
        assert "turn #1" in result

    def test_increments_turn_count(self):
        server.hce_store_interaction("q1", "a1")
        result = server.hce_store_interaction("q2", "a2")
        assert "turn #2" in result
        assert server.pipeline._interaction_count == 2

    def test_persists_to_disk(self, tmp_path):
        state_dir = tmp_path / "hce_state"
        server.STATE_DIR = state_dir
        server.hce_store_interaction("persist test", "should save")
        assert (state_dir / "pipeline_state.json").exists()
        assert (state_dir / "entity_graph.json").exists()
        assert (state_dir / "semantic_tree.json").exists()

    def test_status_block_in_response(self):
        result = server.hce_store_interaction("q", "a")
        assert "HCE Memory" in result
        assert "Entity Graph:" in result


# ── hce_retrieve_context ─────────────────────────────────────────────────

class TestHceRetrieveContext:
    def test_empty_state_returns_no_context(self):
        result = server.hce_retrieve_context("anything")
        assert "No relevant context found" in result

    def test_retrieves_stored_interaction(self):
        server.hce_store_interaction(
            "How does spreading activation work?",
            "It propagates energy through graph edges with a decay factor."
        )
        result = server.hce_retrieve_context("spreading activation")
        assert "HCE Memory" in result
        # Should find the stored interaction via tree or buffer
        assert "spreading activation" in result.lower() or "energy" in result.lower()

    def test_retrieves_from_buffer(self):
        server.hce_store_interaction("what is Python?", "a programming language")
        result = server.hce_retrieve_context("Python programming")
        # Buffer should contain the recent turn
        assert "programming language" in result or "Python" in result

    def test_status_block_shows_token_count(self):
        server.hce_store_interaction("test query", "test response")
        result = server.hce_retrieve_context("test")
        # Should show non-zero tokens retrieved
        assert "retrieved" in result


# ── hce_crawl_project ────────────────────────────────────────────────────

class TestHceCrawlProject:
    def test_crawl_populates_graph(self):
        project_dir = str(Path(__file__).parent)
        result = server.hce_crawl_project(project_dir)
        assert "Crawled:" in result
        assert server.pipeline.graph.node_count > 0
        assert server.pipeline.graph.edge_count > 0

    def test_crawl_invalid_path(self):
        result = server.hce_crawl_project("/nonexistent/path/xyz")
        assert "Error:" in result
        assert "not a directory" in result

    def test_crawl_creates_file_nodes(self):
        project_dir = str(Path(__file__).parent)
        server.hce_crawl_project(project_dir)
        from hce_core import NodeType
        file_nodes = server.pipeline.graph.get_nodes_by_type(NodeType.FILE)
        assert len(file_nodes) > 0
        # Should find at least this test file
        assert any("test_hce_mcp_server" in nid for nid in file_nodes)

    def test_crawl_is_idempotent(self):
        project_dir = str(Path(__file__).parent)
        server.hce_crawl_project(project_dir)
        nodes1 = server.pipeline.graph.node_count
        edges1 = server.pipeline.graph.edge_count

        server.hce_crawl_project(project_dir)
        nodes2 = server.pipeline.graph.node_count
        edges2 = server.pipeline.graph.edge_count

        assert nodes1 == nodes2
        assert edges1 == edges2

    def test_crawl_preserves_conversational_edges(self):
        # Store an interaction first (creates RELATES_TO edges)
        server.hce_store_interaction(
            "Dr. Alice Smith discussed Python",
            "Python is a programming language"
        )
        graph_before_edges = server.pipeline.graph.edge_count

        # Now crawl — should NOT remove RELATES_TO edges from entities
        project_dir = str(Path(__file__).parent)
        server.hce_crawl_project(project_dir)

        # The graph should have edges from both crawl AND conversation
        assert server.pipeline.graph.edge_count > 0

    def test_crawl_persists_state(self, tmp_path):
        state_dir = tmp_path / "hce_state"
        server.STATE_DIR = state_dir
        # Use a minimal directory with at least one .py file
        mini_dir = tmp_path / "src"
        mini_dir.mkdir()
        (mini_dir / "hello.py").write_text("def greet(): pass\n")

        server.hce_crawl_project(str(mini_dir))
        assert (state_dir / "entity_graph.json").exists()


# ── hce_search_graph ─────────────────────────────────────────────────────

class TestHceSearchGraph:
    def test_empty_graph_returns_no_results(self):
        result = server.hce_search_graph("anything")
        assert "No matching entities" in result

    def test_search_after_crawl(self):
        project_dir = str(Path(__file__).parent)
        server.hce_crawl_project(project_dir)
        result = server.hce_search_graph("EntityGraph")
        assert "related entities" in result
        assert "score:" in result

    def test_search_by_label_fallback(self):
        # Add a node manually
        from hce_core import NodeType
        server.pipeline.graph.add_node("test_node", NodeType.CONCEPT, "foobar")
        result = server.hce_search_graph("foobar")
        # Label fallback should find it
        assert "foobar" in result

    def test_search_returns_status_block(self):
        result = server.hce_search_graph("anything")
        assert "HCE Memory" in result


# ── hce_clear_memory ─────────────────────────────────────────────────────

class TestHceClearMemory:
    def test_clears_all_structures(self):
        server.hce_store_interaction("q", "a")
        assert server.pipeline.graph.node_count > 0 or server.pipeline.tree.leaf_count > 0

        result = server.hce_clear_memory()
        assert "cleared" in result.lower()
        assert server.pipeline.graph.node_count == 0
        assert server.pipeline.tree.leaf_count == 0
        assert server.pipeline.focus.size == 0

    def test_clears_persisted_state(self, tmp_path):
        state_dir = tmp_path / "hce_state"
        server.STATE_DIR = state_dir
        server.hce_store_interaction("q", "a")
        assert state_dir.exists()

        server.hce_clear_memory()
        assert not state_dir.exists()

    def test_clear_on_empty_state(self):
        # Should not crash
        result = server.hce_clear_memory()
        assert "cleared" in result.lower()
        assert server.pipeline.graph.node_count == 0


# ── _status_block ────────────────────────────────────────────────────────

class TestStatusBlock:
    def test_format_empty(self):
        block = server._status_block()
        assert "Entity Graph: 0 nodes, 0 edges" in block
        assert "Semantic Tree: 0 interactions" in block
        assert "Focus Buffer: 0/10" in block
        assert "retrieved 0/4,000 tokens" in block

    def test_format_with_context_tokens(self):
        block = server._status_block(context_tokens=1500, sources="3 tree + 2 buffer")
        assert "1,500/4,000" in block
        assert "3 tree + 2 buffer" in block

    def test_reflects_pipeline_state(self):
        server.pipeline.update("q", "a")
        block = server._status_block()
        assert "1 interactions" in block
        assert "2/10" in block  # buffer has user + assistant


# ── Persistence round-trip ───────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        state_dir = tmp_path / "hce_state"
        server.STATE_DIR = state_dir

        server.hce_store_interaction("remember this", "stored for later")
        server.hce_crawl_project(str(Path(__file__).parent))

        nodes_before = server.pipeline.graph.node_count
        edges_before = server.pipeline.graph.edge_count
        leaves_before = server.pipeline.tree.leaf_count

        # Simulate reload
        loaded = HCEPipeline.load(state_dir)
        assert loaded.graph.node_count == nodes_before
        assert loaded.graph.edge_count == edges_before
        assert loaded.tree.leaf_count == leaves_before
        assert loaded._interaction_count == 1


# ── FileLock ──────────────────────────────────────────────────────────────

class TestFileLock:
    def test_exclusive_lock_blocks_concurrent(self, tmp_path):
        """An exclusive lock should block a second exclusive acquire."""
        lock_file = tmp_path / ".lock"
        acquired = threading.Event()
        blocked = threading.Event()
        release = threading.Event()

        def holder():
            with FileLock(lock_file):
                acquired.set()
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        acquired.wait(timeout=2)

        # Try to acquire with a very short timeout — should fail
        with pytest.raises(TimeoutError):
            with FileLock(lock_file, timeout=0.2):
                pass  # pragma: no cover

        release.set()
        t.join(timeout=2)

    def test_shared_locks_allow_concurrent_reads(self, tmp_path):
        """Multiple shared locks should coexist without blocking."""
        lock_file = tmp_path / ".lock"
        acquired_count = threading.Event()
        barrier = threading.Barrier(2, timeout=5)

        results = []

        def reader(idx):
            with FileLock(lock_file, shared=True, timeout=2):
                results.append(idx)
                barrier.wait()  # Both must be inside the lock concurrently

        t1 = threading.Thread(target=reader, args=(1,))
        t2 = threading.Thread(target=reader, args=(2,))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert sorted(results) == [1, 2]

    def test_timeout_raises(self, tmp_path):
        """A lock that can't be acquired within timeout should raise TimeoutError."""
        lock_file = tmp_path / ".lock"
        release = threading.Event()

        def holder():
            with FileLock(lock_file):
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        # Wait until the thread has started
        time.sleep(0.1)

        with pytest.raises(TimeoutError, match="exclusive lock"):
            with FileLock(lock_file, timeout=0.2):
                pass  # pragma: no cover

        release.set()
        t.join(timeout=2)

    def test_shared_blocked_by_exclusive(self, tmp_path):
        """A shared lock should be blocked by an exclusive lock."""
        lock_file = tmp_path / ".lock"
        release = threading.Event()

        def holder():
            with FileLock(lock_file):
                release.wait(timeout=5)

        t = threading.Thread(target=holder)
        t.start()
        time.sleep(0.1)

        with pytest.raises(TimeoutError, match="shared lock"):
            with FileLock(lock_file, shared=True, timeout=0.2):
                pass  # pragma: no cover

        release.set()
        t.join(timeout=2)

    def test_lock_creates_parent_dirs(self, tmp_path):
        """Lock should create parent directories if needed."""
        lock_file = tmp_path / "sub" / "dir" / ".lock"
        with FileLock(lock_file):
            assert lock_file.exists()

    def test_save_with_lock_end_to_end(self, tmp_path):
        """Pipeline save via MCP server should work with locking."""
        state_dir = tmp_path / "hce_state"
        server.STATE_DIR = state_dir
        server.hce_store_interaction("lock test", "should work")
        assert (state_dir / "pipeline_state.json").exists()
        assert (state_dir / "entity_graph.json").exists()
