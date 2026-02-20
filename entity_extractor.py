"""
HCE Phase 3, Component 2: Entity Extractor.

Extracts named entities (PERSON, CONCEPT, EVENT) from conversation text
and populates the EntityGraph with nodes and co-occurrence edges.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from hce_core import NodeType, EdgeType, EntityGraph

log = logging.getLogger(__name__)


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """A named entity extracted from text."""

    text: str
    entity_type: NodeType
    start: int
    end: int


# ── Helpers ─────────────────────────────────────────────────────────────────

def _normalize_entity_id(text: str) -> str:
    """Convert entity text to a stable node ID.

    Lowercase, strip whitespace, replace internal whitespace runs with
    a single underscore.
    """
    return re.sub(r"\s+", "_", text.strip().lower())


# ── Default NER (regex / heuristic) ────────────────────────────────────────

# Days and months used by EVENT detection
_DAYS = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "Saturday", "Sunday",
)
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

_DAY_PAT = "|".join(_DAYS)
_MONTH_PAT = "|".join(_MONTHS)

# Common title prefixes for PERSON detection
_TITLE_PREFIX = r"(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+"

# Words that commonly precede a CONCEPT
_CONCEPT_SIGNALS = (
    r"about", r"called", r"named", r"the\s+concept\s+of",
    r"the\s+topic\s+of", r"regarding",
)
_CONCEPT_SIGNAL_PAT = "|".join(_CONCEPT_SIGNALS)

# Sentence boundary: period followed by one or more spaces
_SENTENCE_START_RE = re.compile(r"\.\s+$")


def _is_sentence_start(text: str, match_start: int) -> bool:
    """Return True if *match_start* sits right after a sentence boundary or
    at position 0 (beginning of text)."""
    if match_start == 0:
        return True
    prefix = text[:match_start]
    return bool(_SENTENCE_START_RE.search(prefix))


def _default_ner(text: str) -> list[Entity]:
    """Simple regex/heuristic NER that requires no ML models.

    Detects PERSON, CONCEPT, and EVENT entities.
    """
    entities: list[Entity] = []

    # ── PERSON detection ────────────────────────────────────────────

    # 1a. Title prefix names: "Dr. Alice Smith"
    for m in re.finditer(
        rf"({_TITLE_PREFIX}[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text
    ):
        entities.append(Entity(m.group(1), NodeType.PERSON, m.start(), m.end()))

    # Already-captured spans (to avoid duplicates from titled names)
    _person_spans: set[tuple[int, int]] = {(e.start, e.end) for e in entities}

    # 1b. Two-or-more consecutive capitalized words
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text):
        span = (m.start(), m.end())
        # Skip if already captured via title prefix
        if any(s <= span[0] and span[1] <= e for s, e in _person_spans):
            continue
        # Filter sentence-start false positives
        if _is_sentence_start(text, m.start()):
            # Allow only if at least the second word is also capitalized
            # (which it always is by the regex), BUT reject common
            # non-name patterns by checking the match is not a known
            # sentence opener.  We keep it simple: if the first word is
            # a common non-name word, skip.
            first_word = m.group(1).split()[0]
            _COMMON_WORDS = {
                "The", "This", "That", "These", "Those", "There", "Here",
                "What", "When", "Where", "Which", "While", "Who", "How",
                "Some", "Many", "Most", "Each", "Every", "All", "Any",
                "Our", "Your", "His", "Her", "Its", "Their", "My",
                "But", "And", "For", "Nor", "Yet", "Not", "Now",
                "After", "Before", "During", "Since", "Until",
                "Once", "Then", "Also", "Just", "Even",
            }
            if first_word in _COMMON_WORDS:
                continue
        entities.append(Entity(m.group(1), NodeType.PERSON, m.start(), m.end()))

    # ── CONCEPT detection ───────────────────────────────────────────

    # 2a. Quoted terms (double or single quotes), short phrases < 50 chars
    for m in re.finditer(r"""(?:["'\u2018\u2019\u201c\u201d])([^"'\u2018\u2019\u201c\u201d]{1,49})(?:["'\u2018\u2019\u201c\u201d])""", text):
        entities.append(Entity(m.group(1), NodeType.CONCEPT, m.start(), m.end()))

    # 2b. Signal-preceded concepts: "about Python", "the concept of gravity"
    #     The signal words are matched case-insensitively via (?i:...),
    #     but the captured concept must start with an uppercase letter.
    for m in re.finditer(
        rf"(?i:(?:{_CONCEPT_SIGNAL_PAT}))\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        text,
    ):
        candidate = m.group(1)
        # Only accept if the candidate starts with uppercase
        if candidate[0].isupper():
            start = m.start(1)
            end = m.end(1)
            entities.append(Entity(candidate, NodeType.CONCEPT, start, end))

    # 2c. Single capitalized words appearing more than once
    single_caps: dict[str, list[re.Match]] = {}
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\b", text):
        word = m.group(1)
        single_caps.setdefault(word, []).append(m)
    # Filter: must appear > 1 time and not already captured
    existing_texts_lower = {e.text.lower() for e in entities}
    for word, matches in single_caps.items():
        if len(matches) > 1 and word.lower() not in existing_texts_lower:
            # Use first occurrence
            m = matches[0]
            entities.append(Entity(word, NodeType.CONCEPT, m.start(), m.end()))

    # ── EVENT detection ─────────────────────────────────────────────

    # 3a. Temporal: "<noun/phrase> on <day/month/date>"
    for m in re.finditer(
        rf"\b(\w+(?:\s+\w+){{0,3}})\s+on\s+({_DAY_PAT}|{_MONTH_PAT}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?)\b",
        text,
    ):
        full = m.group(0)
        entities.append(Entity(full, NodeType.EVENT, m.start(), m.end()))

    # 3b. Temporal: "<noun> at/in <time/place>"
    for m in re.finditer(
        rf"\b(\w+(?:\s+\w+){{0,2}})\s+(?:at|in)\s+({_MONTH_PAT}|\d{{1,2}}:\d{{2}}(?:\s*[APap][Mm])?)\b",
        text,
    ):
        full = m.group(0)
        entities.append(Entity(full, NodeType.EVENT, m.start(), m.end()))

    # 3c. Gerund + to/at/in + Place (capitalized)
    for m in re.finditer(
        r"\b([A-Za-z]+ing)\s+(?:to|at|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        text,
    ):
        full = m.group(0)
        entities.append(Entity(full, NodeType.EVENT, m.start(), m.end()))

    return entities


# ── Entity Extractor ────────────────────────────────────────────────────────

class EntityExtractor:
    """Extract named entities from text and populate an EntityGraph."""

    def __init__(
        self,
        ner_func: Callable[[str], list[Entity]] | None = None,
    ) -> None:
        self._ner = ner_func if ner_func is not None else _default_ner

    # ── Public API ──────────────────────────────────────────────────

    def extract(self, text: str) -> list[Entity]:
        """Extract entities from *text*, deduplicated by normalized text.

        When multiple entities share the same normalized form, the first
        occurrence (by start offset) is kept.
        """
        raw = self._ner(text)
        seen: set[str] = set()
        deduped: list[Entity] = []
        for entity in sorted(raw, key=lambda e: e.start):
            key = _normalize_entity_id(entity.text)
            if key not in seen:
                seen.add(key)
                deduped.append(entity)
        log.debug("Extracted %d entities (%d raw, %d after dedup)",
                  len(deduped), len(raw), len(deduped))
        return deduped

    def update_graph(
        self,
        text: str,
        graph: EntityGraph,
        interaction_id: str | None = None,
    ) -> list[str]:
        """Extract entities, add/update graph nodes, and wire co-occurrence edges.

        For every pair of entities extracted from the same *text*, a
        ``RELATES_TO`` edge is created (if not already present).

        If *interaction_id* is given **and** already exists in the graph,
        each extracted entity is also linked to it via a ``RELATES_TO`` edge.

        Returns the list of node IDs that were added or updated.
        """
        entities = self.extract(text)
        node_ids: list[str] = []

        for entity in entities:
            nid = _normalize_entity_id(entity.text)
            node_ids.append(nid)

            if graph.has_node(nid):
                # Increment mention counter
                existing = graph.get_node(nid)
                mentions = existing.get("metadata", {}).get("mentions", 0) + 1
                graph.add_node(
                    nid,
                    node_type=entity.entity_type,
                    label=entity.text,
                    metadata={"mentions": mentions},
                )
            else:
                graph.add_node(
                    nid,
                    node_type=entity.entity_type,
                    label=entity.text,
                    metadata={"mentions": 1},
                )

        # Co-occurrence edges between all entity pairs
        for i, a_id in enumerate(node_ids):
            for b_id in node_ids[i + 1:]:
                if a_id == b_id:
                    continue
                # Add edge only if not already present
                if not _has_edge(graph, a_id, b_id, EdgeType.RELATES_TO):
                    graph.add_edge(
                        a_id, b_id,
                        edge_type=EdgeType.RELATES_TO,
                        weight=1.0,
                    )

        # Link to interaction node if it exists
        if interaction_id and graph.has_node(interaction_id):
            for nid in node_ids:
                if not _has_edge(graph, nid, interaction_id, EdgeType.RELATES_TO):
                    graph.add_edge(
                        nid, interaction_id,
                        edge_type=EdgeType.RELATES_TO,
                        weight=1.0,
                    )

        log.debug("Updated graph with %d entities from text (%d chars)",
                  len(node_ids), len(text))
        return node_ids


# ── Private helpers ─────────────────────────────────────────────────────────

def _has_edge(
    graph: EntityGraph,
    source: str,
    target: str,
    edge_type: EdgeType,
) -> bool:
    """Check whether an edge of the given type already exists between two nodes."""
    neighbors = graph.get_neighbors(source, edge_type=edge_type, direction="out")
    return any(nid == target for nid, _ in neighbors)
