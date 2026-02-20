"""Tests for the Entity Extractor (HCE Phase 3, Component 2)."""

from __future__ import annotations

import pytest

from hce_core import EntityGraph, NodeType, EdgeType
from entity_extractor import (
    Entity,
    EntityExtractor,
    _default_ner,
    _normalize_entity_id,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _entity_texts(entities: list[Entity]) -> list[str]:
    """Return a list of lowered entity texts for easy assertions."""
    return [e.text.lower() for e in entities]


def _entity_types(entities: list[Entity]) -> dict[str, NodeType]:
    """Map lowered entity text -> NodeType."""
    return {e.text.lower(): e.entity_type for e in entities}


# ── _normalize_entity_id ────────────────────────────────────────────────────

class TestNormalizeEntityId:
    def test_basic(self):
        assert _normalize_entity_id("John Smith") == "john_smith"

    def test_strips_whitespace(self):
        assert _normalize_entity_id("  Hello World  ") == "hello_world"

    def test_collapses_internal_spaces(self):
        assert _normalize_entity_id("foo   bar") == "foo_bar"


# ── PERSON extraction ──────────────────────────────────────────────────────

class TestPersonExtraction:
    def test_basic_person(self):
        entities = _default_ner("I met John Smith at the conference")
        persons = [e for e in entities if e.entity_type == NodeType.PERSON]
        texts = [e.text for e in persons]
        assert "John Smith" in texts

    def test_title_prefix(self):
        entities = _default_ner("Please ask Dr. Alice Brown about it")
        persons = [e for e in entities if e.entity_type == NodeType.PERSON]
        texts = [e.text for e in persons]
        assert any("Alice Brown" in t for t in texts)

    def test_three_word_name(self):
        entities = _default_ner("I know Mary Jane Watson personally")
        persons = [e for e in entities if e.entity_type == NodeType.PERSON]
        texts = [e.text for e in persons]
        assert any("Mary Jane Watson" in t for t in texts)

    def test_sentence_start_filter(self):
        """Common sentence-start phrases should not become PERSON entities."""
        entities = _default_ner("The quick brown fox jumps over the lazy dog.")
        persons = [e for e in entities if e.entity_type == NodeType.PERSON]
        assert len(persons) == 0


# ── CONCEPT extraction ─────────────────────────────────────────────────────

class TestConceptExtraction:
    def test_quoted_concept(self):
        entities = _default_ner("We discussed the concept of 'machine learning'")
        concepts = [e for e in entities if e.entity_type == NodeType.CONCEPT]
        texts = [e.text.lower() for e in concepts]
        assert "machine learning" in texts

    def test_double_quoted(self):
        entities = _default_ner('The idea of "neural networks" is fascinating')
        concepts = [e for e in entities if e.entity_type == NodeType.CONCEPT]
        texts = [e.text.lower() for e in concepts]
        assert "neural networks" in texts

    def test_signal_preceded_concept(self):
        entities = _default_ner("She talked about Python for hours")
        concepts = [e for e in entities if e.entity_type == NodeType.CONCEPT]
        texts = [e.text for e in concepts]
        assert "Python" in texts

    def test_repeated_capitalized_word(self):
        entities = _default_ner(
            "Rust is a language. Rust has a borrow checker."
        )
        concepts = [e for e in entities if e.entity_type == NodeType.CONCEPT]
        texts = [e.text for e in concepts]
        assert "Rust" in texts

    def test_long_quoted_string_ignored(self):
        """Quoted strings >= 50 chars should not become concepts."""
        long_quote = "a" * 55
        entities = _default_ner(f'He said "{long_quote}" in the meeting')
        concepts = [e for e in entities if e.entity_type == NodeType.CONCEPT]
        quoted_long = [e for e in concepts if e.text == long_quote]
        assert len(quoted_long) == 0


# ── EVENT extraction ───────────────────────────────────────────────────────

class TestEventExtraction:
    def test_meeting_on_day(self):
        entities = _default_ner("meeting on Monday at the office")
        events = [e for e in entities if e.entity_type == NodeType.EVENT]
        assert len(events) >= 1
        texts = [e.text.lower() for e in events]
        assert any("monday" in t for t in texts)

    def test_gerund_to_place(self):
        entities = _default_ner("We are traveling to Japan next week")
        events = [e for e in entities if e.entity_type == NodeType.EVENT]
        texts = [e.text.lower() for e in events]
        assert any("japan" in t for t in texts)

    def test_meeting_at_place(self):
        entities = _default_ner("meeting at Google was productive")
        events = [e for e in entities if e.entity_type == NodeType.EVENT]
        texts = [e.text.lower() for e in events]
        assert any("google" in t for t in texts)


# ── EntityExtractor.extract (deduplication) ────────────────────────────────

class TestExtractDeduplication:
    def test_dedup_same_entity(self):
        """Same entity mentioned twice should appear once in extract output."""
        extractor = EntityExtractor()
        text = "John Smith met Alice Brown. Later John Smith called Alice Brown."
        entities = extractor.extract(text)
        ids = [_normalize_entity_id(e.text) for e in entities]
        assert ids.count("john_smith") == 1
        assert ids.count("alice_brown") == 1


# ── EntityExtractor.update_graph ───────────────────────────────────────────

class TestUpdateGraph:
    def test_adds_nodes(self):
        extractor = EntityExtractor()
        graph = EntityGraph()
        text = "I met John Smith at the conference"
        node_ids = extractor.update_graph(text, graph)
        assert len(node_ids) >= 1
        assert graph.has_node("john_smith")

    def test_multiple_entities_create_relates_to_edges(self):
        """All co-occurring entities should be linked by RELATES_TO."""
        extractor = EntityExtractor()
        graph = EntityGraph()
        text = "Dr. Alice Brown discussed 'quantum computing' with Bob Carter"
        node_ids = extractor.update_graph(text, graph)
        # Should have at least 2 nodes
        assert len(node_ids) >= 2
        # Check RELATES_TO edges exist between at least one pair
        assert graph.edge_count >= 1

    def test_dedup_increments_mentions(self):
        """Re-extracting the same entity should increment its mentions count."""
        extractor = EntityExtractor()
        graph = EntityGraph()
        text1 = "John Smith is here"
        text2 = "John Smith left the building"
        extractor.update_graph(text1, graph)
        extractor.update_graph(text2, graph)
        node = graph.get_node("john_smith")
        assert node is not None
        assert node["metadata"]["mentions"] == 2

    def test_interaction_id_creates_links(self):
        """When interaction_id exists in graph, entities link to it."""
        extractor = EntityExtractor()
        graph = EntityGraph()
        # Pre-create an interaction node
        graph.add_node("turn_42", NodeType.EVENT, label="Turn 42")
        text = "John Smith talked about 'deep learning'"
        node_ids = extractor.update_graph(text, graph, interaction_id="turn_42")
        # Each entity should have an edge to turn_42
        for nid in node_ids:
            neighbors = graph.get_neighbors(nid, edge_type=EdgeType.RELATES_TO, direction="out")
            neighbor_ids = [n for n, _ in neighbors]
            assert "turn_42" in neighbor_ids, (
                f"Node {nid} should link to interaction turn_42"
            )

    def test_interaction_id_missing_no_error(self):
        """If interaction_id is not in graph, no error and no link."""
        extractor = EntityExtractor()
        graph = EntityGraph()
        text = "John Smith is here"
        # Should not raise
        node_ids = extractor.update_graph(text, graph, interaction_id="nonexistent")
        assert len(node_ids) >= 1


# ── Custom NER function ────────────────────────────────────────────────────

class TestCustomNer:
    def test_custom_ner_is_used(self):
        """A user-provided NER function replaces the default."""

        def my_ner(text: str) -> list[Entity]:
            return [Entity("custom_entity", NodeType.CONCEPT, 0, len(text))]

        extractor = EntityExtractor(ner_func=my_ner)
        entities = extractor.extract("any text at all")
        assert len(entities) == 1
        assert entities[0].text == "custom_entity"
        assert entities[0].entity_type == NodeType.CONCEPT


# ── Empty text ──────────────────────────────────────────────────────────────

class TestEmptyText:
    def test_empty_string(self):
        extractor = EntityExtractor()
        entities = extractor.extract("")
        assert entities == []

    def test_no_entities_in_lowercase(self):
        extractor = EntityExtractor()
        entities = extractor.extract("nothing special here at all")
        # Should not crash; may return empty or very few items
        assert isinstance(entities, list)


# ── Co-occurrence edges ────────────────────────────────────────────────────

class TestCoOccurrence:
    def test_alice_bob_python(self):
        """'Alice told Bob about Python' should yield edges between all pairs."""
        extractor = EntityExtractor()
        graph = EntityGraph()
        text = "Alice Carter told Bob Davis about Python"
        node_ids = extractor.update_graph(text, graph)

        # We expect at least Alice Carter, Bob Davis, and Python
        assert len(node_ids) >= 3

        # Verify edges exist between all extracted pairs
        for i, a in enumerate(node_ids):
            for b in node_ids[i + 1:]:
                if a == b:
                    continue
                # Check at least one direction
                neighbors_out = graph.get_neighbors(
                    a, edge_type=EdgeType.RELATES_TO, direction="out"
                )
                neighbors_in = graph.get_neighbors(
                    a, edge_type=EdgeType.RELATES_TO, direction="in"
                )
                connected_ids = {n for n, _ in neighbors_out} | {n for n, _ in neighbors_in}
                assert b in connected_ids, (
                    f"Expected RELATES_TO edge between {a} and {b}"
                )
