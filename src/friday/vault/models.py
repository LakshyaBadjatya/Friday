"""Vault domain models — the shapes every other vault module speaks in.

Pure pydantic v2 with no behaviour beyond one privacy predicate. Timestamps are
ISO-8601 UTC strings, matching :mod:`friday.study.store` and
:mod:`friday.journal.store`.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class Privacy(StrEnum):
    """Who may see an item, and whether it may reach a model provider."""

    PRIVATE = "private"
    SHARED = "shared"
    LOCKED = "locked"


class CaptureSource(StrEnum):
    """Where a capture came from."""

    CAMERA = "camera"
    SCREEN = "screen"
    SHARE = "share"
    WEB = "web"
    DESKTOP = "desktop"
    SCAN = "scan"


class ItemStatus(StrEnum):
    """Lifecycle of an item, from signed intent to processed result."""

    PENDING = "pending"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class CloudinaryAsset(BaseModel):
    """The identity of one uploaded asset, as Cloudinary reports it."""

    public_id: str
    version: int
    format: str
    bytes: int
    width: int = 0
    height: int = 0
    resource_type: str = "image"


class Derived(BaseModel):
    """Downscaled working copy and thumbnail public ids."""

    work_public_id: str = ""
    thumb_public_id: str = ""


class Classification(BaseModel):
    """What FRIDAY decided this capture is."""

    kind: str = "unknown"
    subject: str = ""
    chapter: str = ""
    tags: list[str] = Field(default_factory=list)


class Hashes(BaseModel):
    """Content hashes: exact dedupe and perceptual board-frame stitching."""

    sha256: str = ""
    phash: str = ""


class Item(BaseModel):
    """One capture in the vault."""

    id: str
    owner_uid: str
    space: str = "private"
    privacy: Privacy = Privacy.PRIVATE
    source: CaptureSource
    status: ItemStatus = ItemStatus.PENDING
    cloudinary: CloudinaryAsset | None = None
    derived: Derived = Field(default_factory=Derived)
    ocr_text: str = ""
    ocr_engine: str = ""
    caption: str = ""
    classification: Classification = Field(default_factory=Classification)
    hashes: Hashes = Field(default_factory=Hashes)
    solve_id: str | None = None
    note_ids: list[str] = Field(default_factory=list)
    created_at: str
    processed_at: str | None = None
    device_id: str = ""

    def may_leave_for_model(self) -> bool:
        """Whether these bytes may be sent to a model provider.

        Locked items never may — the pipeline consults this before every
        provider call, so the guarantee lives in one place.
        """
        return self.privacy is not Privacy.LOCKED


class Draft(BaseModel):
    """One operator's attempt at a solution."""

    operator: str
    model_id: str = ""
    steps: list[str] = Field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    latency_ms: int = 0


class Verification(BaseModel):
    """The independent check on the drafts' arithmetic."""

    engine: str = "sympy"
    ok: bool = False
    detail: str = ""


class Consensus(BaseModel):
    """What the panel settled on, and how contested it was."""

    final_answer: str = ""
    agreement: str = "0/0"
    judged: bool = False


class Solve(BaseModel):
    """The full record of solving one problem — drafts, check, and dissent."""

    id: str
    item_ids: list[str]
    subject: str = ""
    statement: str = ""
    latex: str = ""
    drafts: list[Draft] = Field(default_factory=list)
    verification: Verification = Field(default_factory=Verification)
    consensus: Consensus = Field(default_factory=Consensus)
    dissent: list[str] = Field(default_factory=list)
    created_at: str = ""


class Note(BaseModel):
    """A written note over one or more captures."""

    id: str
    owner_uid: str
    space: str = "private"
    title: str = ""
    subject: str = ""
    chapter: str = ""
    markdown: str = ""
    item_ids: list[str] = Field(default_factory=list)
    flashcard_ids: list[int] = Field(default_factory=list)
    rag_source_id: str = ""
    exported_pdf_public_id: str | None = None
    created_at: str = ""


class GradedQuestion(BaseModel):
    """One question's marks in a graded exam."""

    q: str
    marks_awarded: float = 0.0
    marks_total: float = 0.0
    feedback: str = ""


class ExamSession(BaseModel):
    """A timed paper: the question images, your answers, and the marking."""

    id: str
    owner_uid: str
    paper_item_ids: list[str] = Field(default_factory=list)
    answer_item_ids: list[str] = Field(default_factory=list)
    started_at: str = ""
    duration_s: int = 0
    status: str = "open"
    grading: list[GradedQuestion] = Field(default_factory=list)
    total: float = 0.0
