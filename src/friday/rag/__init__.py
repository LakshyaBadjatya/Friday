"""Personal RAG: ingest a file/note so it becomes answerable with citations.

This package layers a thin *ingestion* seam on top of the Phase-4 retrieval
stack — it adds no new retrieval logic. A document is chunked and each chunk is
written to the existing :class:`~friday.memory.vector.VectorStore`; a single
"ingested <source>" fact is recorded in the
:class:`~friday.memory.long_term.LongTermStore` so the document is listable and
forgettable. Because the unchanged :class:`~friday.agents.knowledge.KnowledgeAgent`
already retrieves from that same vector store, an ingested note is immediately
answerable (with citations) through the normal ``/chat`` knowledge path.

The whole feature is gated behind ``FRIDAY_ENABLE_RAG`` (default off): when the
flag is off the ``/rag`` routes are ``404`` and nothing in this package runs.
PDF reading is optional and lazy — ``pypdf`` is imported only when a ``.pdf`` is
ingested and is *not* a project dependency.
"""

from __future__ import annotations

from friday.rag.ingest import DocumentIngestor, IngestResult, chunk_text

__all__ = ["DocumentIngestor", "IngestResult", "chunk_text"]
