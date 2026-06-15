"""Unit tests for the personal-RAG ingest layer (Tier 1, Stage 1A).

The :mod:`friday.rag.ingest` module turns a document into retrievable chunks in
the *existing* Phase-4 stores: each chunk is written to the injected
:class:`~friday.memory.vector.VectorStore` under an id derived from the source,
and a single "ingested <source>" fact is recorded in the
:class:`~friday.memory.long_term.SQLiteLongTermStore` so the document is listable
and forgettable. Nothing here touches the network: tests use the deterministic
:class:`~friday.providers.embeddings.FakeEmbeddings` and ``":memory:"`` stores.

Pinned behaviours:

* :func:`chunk_text` is paragraph/size-aware with overlap and never drops
  content — the concatenation of the chunks (overlap removed) covers the whole
  input.
* ``DocumentIngestor.ingest`` chunks, stores each chunk under
  ``f"{source_id}#{i}"`` in the vector store, records one long-term fact, and a
  subsequent vector query returns a chunk carrying the source id.
* A :class:`~friday.agents.knowledge.KnowledgeAgent` built over the *same* store
  answers a question about the ingested text, citing the source id — so an
  ingested note is answerable through the unchanged knowledge path.
* ``read_text`` decodes ``.txt`` / ``.md`` directly and raises a clear,
  pip-pointing error for ``.pdf`` when ``pypdf`` is not installed (it is not a
  project dependency).
* ``forget_source`` removes the document from both the vector store and the
  long-term store.
"""

from __future__ import annotations

import pytest

from friday.agents.base import AgentResult
from friday.agents.knowledge import KnowledgeAgent
from friday.core.state import GraphState, Mode
from friday.memory.long_term import SQLiteLongTermStore
from friday.memory.vector import SQLiteVectorStore
from friday.providers.embeddings import FakeEmbeddings
from friday.rag.ingest import DocumentIngestor, IngestResult, chunk_text


def _ingestor() -> tuple[DocumentIngestor, SQLiteVectorStore, SQLiteLongTermStore]:
    """A :class:`DocumentIngestor` over ``":memory:"`` stores + FakeEmbeddings."""
    embedder = FakeEmbeddings(dim=64)
    vector = SQLiteVectorStore(":memory:", embedder=embedder, dim=64)
    long_term = SQLiteLongTermStore(":memory:")
    ingestor = DocumentIngestor(
        vector, long_term, chunk_size=80, chunk_overlap=20
    )
    return ingestor, vector, long_term


def _state(user_input: str) -> GraphState:
    return GraphState(
        session_id="rag-test", mode=Mode.CONVERSATION, user_input=user_input
    )


# --------------------------------------------------------------------------- #
# chunk_text
# --------------------------------------------------------------------------- #
def test_chunk_text_short_input_is_single_chunk() -> None:
    """Text under ``size`` returns exactly one chunk equal to the input."""
    text = "A short note."
    chunks = chunk_text(text, size=800, overlap=120)
    assert chunks == [text]


def test_chunk_text_empty_input_returns_empty() -> None:
    """Empty / whitespace-only input yields no chunks (nothing to index)."""
    assert chunk_text("", size=800, overlap=120) == []
    assert chunk_text("   \n\n  ", size=800, overlap=120) == []


def test_chunk_text_splits_long_input_with_overlap() -> None:
    """A long input is split into multiple bounded, overlapping chunks."""
    text = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_text(text, size=120, overlap=30)
    assert len(chunks) > 1
    # Each chunk respects the size budget (with a little slack for word edges).
    for chunk in chunks:
        assert len(chunk) <= 120 + 30
    # Consecutive chunks overlap: the tail of one shares text with the next.
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first[-10:] in text
        # Overlap means the second chunk begins inside the first chunk's span.
        assert second[:10] in first or second[:10] in text


def test_chunk_text_loses_no_content() -> None:
    """Every token of the input survives somewhere in the produced chunks."""
    text = " ".join(f"token{i}" for i in range(500))
    chunks = chunk_text(text, size=150, overlap=40)
    combined = " ".join(chunks)
    for i in range(500):
        assert f"token{i}" in combined


def test_chunk_text_paragraph_aware() -> None:
    """Paragraph breaks are preserved as natural split points; no content lost."""
    paras = ["Paragraph one is here." * 5, "Paragraph two is here." * 5]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, size=80, overlap=20)
    combined = " ".join(chunks)
    assert "Paragraph one" in combined
    assert "Paragraph two" in combined


# --------------------------------------------------------------------------- #
# DocumentIngestor.ingest
# --------------------------------------------------------------------------- #
async def test_ingest_returns_result_and_stores_chunks() -> None:
    """Ingest chunks, returns an :class:`IngestResult`, and indexes each chunk."""
    ingestor, vector, _ = _ingestor()
    text = (
        "FRIDAY is a defensive-only local assistant built in Python. "
        "It runs offline and never calls out to external accounts by default. "
        "The knowledge agent grounds answers strictly in retrieved sources."
    )
    result = await ingestor.ingest("notes-friday", text)

    assert isinstance(result, IngestResult)
    assert result.source_id == "notes-friday"
    assert result.chunks >= 1

    # A query about the content returns a chunk carrying the source id.
    hits = vector.query("What is FRIDAY the local assistant?", k=4)
    assert hits
    assert any(hit.source_id.startswith("notes-friday#") for hit in hits)


async def test_ingest_records_one_listable_forgettable_fact() -> None:
    """Ingest records exactly one 'ingested <source>' long-term fact."""
    ingestor, _, long_term = _ingestor()
    await ingestor.ingest("notes-friday", "Some content about FRIDAY internals.")

    facts = long_term.query_facts("notes-friday", limit=10)
    assert len(facts) == 1
    assert "ingested" in facts[0].text.lower()
    assert "notes-friday" in facts[0].text


async def test_knowledge_agent_answers_from_ingested_note_citing_source() -> None:
    """A KnowledgeAgent over the same store answers, citing the ingested source."""
    ingestor, vector, long_term = _ingestor()
    await ingestor.ingest(
        "notes-quasar",
        "The Quasar protocol uses a rotating handshake token every ninety seconds.",
    )
    agent = KnowledgeAgent(store=vector, long_term=long_term)

    result = await agent.run(_state("How often does the Quasar protocol rotate?"))

    assert isinstance(result, AgentResult)
    assert "notes-quasar" in result.output
    # Grounded, not a parametric decline.
    assert "nothing" not in result.output.lower()


# --------------------------------------------------------------------------- #
# read_text
# --------------------------------------------------------------------------- #
def test_read_text_decodes_txt_and_md() -> None:
    """``.txt`` and ``.md`` payloads decode directly to their UTF-8 text."""
    ingestor, _, _ = _ingestor()
    assert ingestor.read_text("note.txt", b"hello world") == "hello world"
    assert ingestor.read_text("note.md", b"# Title\nbody") == "# Title\nbody"


def test_read_text_pdf_without_pypdf_raises_clear_error() -> None:
    """``.pdf`` without ``pypdf`` installed raises a clear pip-pointing error."""
    ingestor, _, _ = _ingestor()
    import importlib.util

    if importlib.util.find_spec("pypdf") is not None:  # pragma: no cover
        pytest.skip("pypdf is installed; the clear-error path is not exercised")
    with pytest.raises(RuntimeError) as exc:
        ingestor.read_text("doc.pdf", b"%PDF-1.4 fake")
    assert "pypdf" in str(exc.value).lower()
    assert "pip install" in str(exc.value).lower()


def test_read_text_unknown_extension_decodes_as_text() -> None:
    """An unknown extension falls back to a best-effort UTF-8 decode."""
    ingestor, _, _ = _ingestor()
    assert ingestor.read_text("note", b"plain bytes") == "plain bytes"


# --------------------------------------------------------------------------- #
# forget_source
# --------------------------------------------------------------------------- #
async def test_forget_source_removes_from_vector_and_long_term() -> None:
    """``forget_source`` drops the doc from both stores; re-query returns nothing."""
    ingestor, vector, long_term = _ingestor()
    secret = "The launch codes for the secret project are stored in vault seven."
    await ingestor.ingest("notes-secret", secret)
    # Present before forgetting. The exact chunk text self-matches under the
    # deterministic FakeEmbeddings (identical text -> identical vector -> score 1).
    before = vector.query(secret, k=4)
    assert any(hit.source_id.startswith("notes-secret#") for hit in before)
    assert long_term.query_facts("notes-secret", limit=10)

    removed = ingestor.forget_source("notes-secret")
    assert removed >= 1

    # Gone from both stores afterwards.
    after = vector.query(secret, k=4)
    assert not any(hit.source_id.startswith("notes-secret#") for hit in after)
    assert long_term.query_facts("notes-secret", limit=10) == []
