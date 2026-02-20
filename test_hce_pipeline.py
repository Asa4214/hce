"""Tests for hce_pipeline: SlidingFocusBuffer, context_budgeting, and HCEPipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from hce_core import EntityGraph, NodeType, EdgeType
from semantic_tree import SemanticTree
from hce_pipeline import (
    ContextCandidate,
    FocusEntry,
    HCEPipeline,
    SlidingFocusBuffer,
    context_budgeting,
    estimate_tokens,
)


# ── SlidingFocusBuffer ──────────────────────────────────────────────────

class TestSlidingFocusBuffer:
    def test_empty_buffer(self):
        buf = SlidingFocusBuffer(max_size=5)
        assert buf.size == 0
        assert buf.get_recent() == []
        assert buf.to_text() == ""

    def test_add_and_size(self):
        buf = SlidingFocusBuffer(max_size=5)
        buf.add("user", "hello")
        buf.add("assistant", "hi there")
        assert buf.size == 2

    def test_circular_eviction(self):
        buf = SlidingFocusBuffer(max_size=3)
        buf.add("user", "msg1")
        buf.add("assistant", "resp1")
        buf.add("user", "msg2")
        buf.add("assistant", "resp2")  # should evict msg1
        assert buf.size == 3
        recent = buf.get_recent()
        assert recent[0].content == "resp1"  # oldest remaining
        assert recent[-1].content == "resp2"

    def test_get_recent_n(self):
        buf = SlidingFocusBuffer(max_size=10)
        for i in range(5):
            buf.add("user", f"msg{i}")
        recent = buf.get_recent(2)
        assert len(recent) == 2
        assert recent[-1].content == "msg4"

    def test_clear(self):
        buf = SlidingFocusBuffer(max_size=5)
        buf.add("user", "hello")
        buf.clear()
        assert buf.size == 0

    def test_to_text_format(self):
        buf = SlidingFocusBuffer()
        buf.add("user", "What is Python?")
        buf.add("assistant", "Python is a programming language.")
        text = buf.to_text()
        assert "User: What is Python?" in text
        assert "Assistant: Python is a programming language." in text

    def test_serialization_roundtrip(self):
        buf = SlidingFocusBuffer(max_size=5)
        buf.add("user", "hello")
        buf.add("assistant", "world")
        data = buf.to_list()
        restored = SlidingFocusBuffer.from_list(data, max_size=5)
        assert restored.size == 2
        entries = restored.get_recent()
        assert entries[0].content == "hello"
        assert entries[1].content == "world"

    def test_max_size_property(self):
        buf = SlidingFocusBuffer(max_size=7)
        assert buf.max_size == 7


# ── estimate_tokens ──────────────────────────────────────────────────────

class TestEstimateTokens:
    def test_empty_string(self):
        assert estimate_tokens("") == 1  # min 1

    def test_simple_text(self):
        tokens = estimate_tokens("hello world foo bar")
        assert tokens > 0
        assert tokens >= 4  # at least word count

    def test_longer_text(self):
        text = " ".join(["word"] * 100)
        tokens = estimate_tokens(text)
        assert 100 <= tokens <= 200


# ── context_budgeting ────────────────────────────────────────────────────

class TestContextBudgeting:
    def test_empty_candidates(self):
        assert context_budgeting([], budget=100) == []

    def test_all_fit_within_budget(self):
        candidates = [
            ContextCandidate("a", "tree", utility=1.0, token_cost=10),
            ContextCandidate("b", "graph", utility=0.8, token_cost=10),
        ]
        selected = context_budgeting(candidates, budget=100)
        assert len(selected) == 2

    def test_budget_respected(self):
        candidates = [
            ContextCandidate("big", "tree", utility=1.0, token_cost=50),
            ContextCandidate("small", "graph", utility=0.9, token_cost=10),
            ContextCandidate("medium", "buffer", utility=0.5, token_cost=30),
        ]
        selected = context_budgeting(candidates, budget=40)
        total_cost = sum(c.token_cost for c in selected)
        assert total_cost <= 40

    def test_high_efficiency_preferred(self):
        candidates = [
            ContextCandidate("cheap-good", "tree", utility=1.0, token_cost=5),
            ContextCandidate("expensive-ok", "graph", utility=1.0, token_cost=50),
        ]
        selected = context_budgeting(candidates, budget=50)
        # cheap-good has ratio 0.2, expensive-ok has ratio 0.02
        # cheap-good should be selected first
        assert selected[0].content == "cheap-good"

    def test_zero_budget_returns_nothing(self):
        candidates = [
            ContextCandidate("a", "tree", utility=1.0, token_cost=10),
        ]
        selected = context_budgeting(candidates, budget=0)
        assert len(selected) == 0


# ── HCEPipeline ─────────────────────────────────────────────────────────

class TestHCEPipelineBasic:
    def test_init_creates_empty_structures(self):
        p = HCEPipeline()
        assert p.graph.node_count == 0
        assert p.tree.leaf_count == 0
        assert p.focus.size == 0

    def test_init_with_existing_graph(self):
        g = EntityGraph()
        g.add_node("test", NodeType.CONCEPT, "test concept")
        p = HCEPipeline(graph=g)
        assert p.graph.node_count == 1

    def test_update_populates_all_structures(self):
        p = HCEPipeline()
        p.update("Tell me about Python", "Python is a programming language.")
        # Focus buffer should have 2 entries (user + assistant)
        assert p.focus.size == 2
        # Semantic tree should have 1 leaf
        assert p.tree.leaf_count == 1

    def test_multiple_updates(self):
        p = HCEPipeline()
        p.update("What is Python?", "Python is a language.")
        p.update("What about JavaScript?", "JavaScript runs in browsers.")
        assert p.focus.size == 4
        assert p.tree.leaf_count == 2


class TestHCEPipelineRetrieval:
    def _build_pipeline(self) -> HCEPipeline:
        p = HCEPipeline(context_budget=2000)
        p.update(
            "Alice told me about machine learning",
            "Machine learning uses data to build predictive models.",
        )
        p.update(
            "Bob explained neural networks",
            "Neural networks are inspired by the human brain.",
        )
        p.update(
            "I need to learn Python for data science",
            "Python is great for data science with libraries like pandas.",
        )
        return p

    def test_retrieve_context_returns_string(self):
        p = self._build_pipeline()
        ctx = p.retrieve_context("Tell me about machine learning")
        assert isinstance(ctx, str)

    def test_retrieve_context_not_empty_for_related_query(self):
        p = self._build_pipeline()
        ctx = p.retrieve_context("What is machine learning?")
        assert len(ctx) > 0

    def test_build_prompt_includes_query(self):
        p = self._build_pipeline()
        prompt = p.build_prompt("What is Python?")
        assert "What is Python?" in prompt

    def test_build_prompt_includes_context_sections(self):
        p = self._build_pipeline()
        prompt = p.build_prompt("Tell me about data science")
        # Should have at least the query section
        assert "## Current Query" in prompt


class TestHCEPipelineWrapChat:
    def test_wrap_chat_calls_original(self):
        p = HCEPipeline()
        call_log: list[str] = []

        def mock_chat(prompt: str) -> str:
            call_log.append(prompt)
            return "I am a mock response."

        wrapped = p.wrap_chat(mock_chat)
        response = wrapped("Hello, AI!")
        assert response == "I am a mock response."
        assert len(call_log) == 1

    def test_wrap_chat_enriches_prompt(self):
        p = HCEPipeline()
        # Add some history first
        p.update("I like Python", "Python is great!")
        received_prompts: list[str] = []

        def mock_chat(prompt: str) -> str:
            received_prompts.append(prompt)
            return "OK"

        wrapped = p.wrap_chat(mock_chat)
        wrapped("Tell me about Python")
        # The prompt should be enriched, not just the raw query
        assert len(received_prompts[0]) > len("Tell me about Python")

    def test_wrap_chat_stores_response(self):
        p = HCEPipeline()

        def mock_chat(prompt: str) -> str:
            return "This is the response."

        wrapped = p.wrap_chat(mock_chat)
        wrapped("Test query")
        # After the call, focus buffer should be updated
        assert p.focus.size == 2  # user + assistant
        # Semantic tree should have the interaction
        assert p.tree.leaf_count == 1

    def test_wrap_chat_context_improves_over_time(self):
        p = HCEPipeline()
        prompts: list[str] = []

        def mock_chat(prompt: str) -> str:
            prompts.append(prompt)
            return "Some response about the topic."

        wrapped = p.wrap_chat(mock_chat)

        # First call — no history
        wrapped("What is Python?")
        first_prompt_len = len(prompts[0])

        # Second call — now there's history
        wrapped("Tell me more about Python")
        second_prompt_len = len(prompts[1])

        # Second prompt should be longer (enriched with context)
        assert second_prompt_len > first_prompt_len


class TestHCEPipelineSerialization:
    def test_save_and_load_roundtrip(self):
        p = HCEPipeline(max_focus_entries=5, context_budget=3000)
        p.update("Hello", "Hi there!")
        p.update("How are you?", "I'm doing well.")

        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(tmpdir)

            # Verify files exist
            assert (Path(tmpdir) / "entity_graph.json").exists()
            assert (Path(tmpdir) / "semantic_tree.json").exists()
            assert (Path(tmpdir) / "pipeline_state.json").exists()

            loaded = HCEPipeline.load(tmpdir)
            assert loaded.focus.size == 4
            assert loaded.tree.leaf_count == 2
            assert loaded._interaction_count == 2
            assert loaded._budget == 3000

    def test_load_preserves_focus_buffer(self):
        p = HCEPipeline()
        p.update("Question", "Answer")

        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(tmpdir)
            loaded = HCEPipeline.load(tmpdir)
            entries = loaded.focus.get_recent()
            assert entries[0].content == "Question"
            assert entries[1].content == "Answer"


# ── Integration test ─────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_conversation_flow(self):
        """Simulate a multi-turn conversation through the pipeline."""
        p = HCEPipeline(context_budget=2000)
        responses: list[str] = []

        def echo_chat(prompt: str) -> str:
            resp = f"Echo: received {len(prompt)} chars"
            responses.append(resp)
            return resp

        chat = p.wrap_chat(echo_chat)

        # Turn 1: introduce a topic
        chat("Alice is a data scientist who works with Python")
        assert p.focus.size == 2
        assert p.tree.leaf_count == 1

        # Turn 2: related topic
        chat("Bob is a web developer using JavaScript")
        assert p.focus.size == 4
        assert p.tree.leaf_count == 2

        # Turn 3: query that should retrieve relevant context
        chat("What programming languages do Alice and Bob use?")
        assert p.focus.size == 6
        assert p.tree.leaf_count == 3

        # Verify all 3 responses were generated
        assert len(responses) == 3

        # The third prompt should be the richest (most history to draw from)
        # We can verify by checking the focus buffer
        entries = p.focus.get_recent()
        assert any("Alice" in e.content for e in entries)
        assert any("Bob" in e.content for e in entries)
