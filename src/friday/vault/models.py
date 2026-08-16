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
    secure_url: str = ""


class DerivedAssets(BaseModel):
    """The downscaled working copy and thumbnail derived from the original upload."""

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
    derived: DerivedAssets = Field(default_factory=DerivedAssets)
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
    equation: str = ""
    """The decisive step of the solution as a single-variable equation in
    plain ASCII SymPy can parse (e.g. ``"2*x + 4 = 10"``), or ``""`` when the
    problem does not reduce to one. Requested from the operator itself, so
    verifying against it is weaker than a truly independent derivation — see
    :mod:`friday.vault.solver` for the honest limitation."""
    confidence: float = 0.0
    latency_ms: int = 0


class VerificationStatus(StrEnum):
    """What the independent check actually established.

    ``ok: bool`` used to conflate "re-derived and disagreed" with "could not
    be checked at all" — both fell to ``ok=False``, so a chapter of correct
    but unverifiable answers (word problems, chemistry, prose) looked
    indistinguishable from a chapter that was genuinely being gotten wrong.
    This status makes the three outcomes distinct.
    """

    VERIFIED = "verified"
    """Re-derived independently and agreed with the chosen draft."""

    REFUTED = "refuted"
    """Re-derived independently and DISAGREED with the chosen draft."""

    NOT_VERIFIABLE = "not_verifiable"
    """Could not be checked at all — no equation, unparseable, multi-variable,
    or no solution. This is not a verdict on correctness either way."""


class Verification(BaseModel):
    """The independent check on the drafts' arithmetic."""

    engine: str = "sympy"
    status: VerificationStatus = VerificationStatus.NOT_VERIFIABLE
    detail: str = ""

    @property
    def ok(self) -> bool:
        """Read-only: ``True`` only when the check ran and agreed.

        Kept so existing readers that only care about "did it check out"
        do not need to learn the three-way status. It is deliberately not a
        settable field — the status is the source of truth, ``ok`` is a view
        onto it.
        """
        return self.status is VerificationStatus.VERIFIED


class Consensus(BaseModel):
    """What the panel settled on, and how contested it was.

    ``agreement`` is always ``"<votes>/<drafts>"`` — the count of drafts that
    agreed with ``final_answer`` over the total number of drafts — since later
    code parses and displays it in that fixed format.
    """

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
    created_at: str


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
    created_at: str


class GradedQuestion(BaseModel):
    """One question's marks in a graded exam."""

    q: str
    marks_awarded: float = 0.0
    marks_total: float = 0.0
    feedback: str = ""


class ExamSessionStatus(StrEnum):
    """Lifecycle of a timed exam session."""

    OPEN = "open"
    GRADED = "graded"


class ExamSession(BaseModel):
    """A timed paper: the question images, your answers, and the marking."""

    id: str
    owner_uid: str
    paper_item_ids: list[str] = Field(default_factory=list)
    answer_item_ids: list[str] = Field(default_factory=list)
    started_at: str
    duration_s: int = 0
    status: ExamSessionStatus = ExamSessionStatus.OPEN
    grading: list[GradedQuestion] = Field(default_factory=list)
    total: float = 0.0
