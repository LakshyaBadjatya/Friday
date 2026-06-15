"""Knowledge graph / entity cards (Tier 2): a tiny personal entity graph.

This package owns FRIDAY's lightweight knowledge graph — the people, projects,
organizations, and things the owner talks about, plus the relations between them
— and the *entity card* view that answers "what do you know about X?" by stitching
an entity together with its relations and the long-term facts that mention it.

It reuses existing infrastructure only — the typed
:class:`~friday.providers.llm.LLMProvider` boundary for one NON-FATAL extraction
pass and the Phase-4 SQLite path (``memory_db_path``) for a sibling store — and is
off by default behind ``FRIDAY_ENABLE_KNOWLEDGE_GRAPH``. Extraction is non-fatal:
any provider or parse error yields an empty result, never raising.

The public surface is the typed :class:`~friday.graph.store.Entity` /
:class:`~friday.graph.store.Relation` models, the
:class:`~friday.graph.store.SQLiteGraphStore` adapter, and the
:class:`~friday.graph.extractor.EntityExtractor` pipeline.
"""

from __future__ import annotations

from friday.graph.extractor import EntityExtractor
from friday.graph.store import Entity, Relation, SQLiteGraphStore

__all__ = [
    "Entity",
    "EntityExtractor",
    "Relation",
    "SQLiteGraphStore",
]
