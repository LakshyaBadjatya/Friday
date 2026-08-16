"""The capture-processing pipeline: image bytes in, a filed (and maybe solved) item out.

The user photographs a homework problem, the phone uploads it to Cloudinary, and
the upload route verifies and commits the asset onto the :class:`~friday.vault.models.Item`.
:class:`Pipeline` is what happens next: fetch the bytes back → OCR them locally →
classify what the capture is → if it is a solvable problem, run the always-full
ensemble → persist.

**The rule that dominates the ordering: a LOCKED item never reaches a model
provider.** OCR runs locally for every capture, but classification and solving
are skipped entirely when the item is locked. The check is a single call to
:meth:`~friday.vault.models.Item.may_leave_for_model`, so the guarantee lives in
one place and cannot drift — see that method's docstring. Get this wrong and
biometric-locked material is sent to Gemini.

Each stage's result is persisted as it completes (not only once at the very
end), so a process crash partway through the chain still leaves the earlier
stages' work on disk rather than losing the capture back to whatever state it
was in before :meth:`Pipeline.process` started.

Two failure modes are deliberately survived rather than propagated, because
"the photo is filed" is a stronger guarantee than "the photo was solved":

* **OCR (or the bytes fetch) raises** — logged, the item keeps whatever
  ``ocr_text`` it already had (empty, on a first pass), and processing
  continues to classification/solving/READY exactly as if OCR had returned "".
* **The solver raises** — logged, the item is filed with its classification
  and no ``solve_id``, and still reaches READY. A solver outage or an
  unexpected bug in the ensemble must not cost the user their photo; they can
  always ask FRIDAY to solve it again later. This mirrors the exhausted-quota
  path just below it, which reaches READY unsolved for the same reason.

:data:`ItemStatus.FAILED` is defined on the lifecycle enum but is never set by
this pipeline (see the module-level test suite for the explicit assertion) —
every failure mode this pipeline can hit degrades to "filed, unsolved" rather
than "failed", by design. It remains available for a future stage (e.g. a
fetch that 404s because the Cloudinary asset was deleted out from under us)
that genuinely has nothing to file.

``fetch_bytes`` is injected on purpose: this module has no opinion on how the
bytes come back from Cloudinary, and tests pass a deterministic stub. **A
production wiring must not default to ``lambda public_id: b""``** — that was
the plan's draft placeholder, and it would "succeed" while silently OCR'ing
nothing on every capture. The working value to fetch from is the item's own
``item.cloudinary.secure_url`` (stored at commit time, see the upload/commit
route) — whoever wires the production :class:`Pipeline` should build
``fetch_bytes`` around an HTTP GET of that URL, not around ``public_id`` alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from friday.perception.ocr import OCRProvider
from friday.providers.llm import LLMProvider
from friday.vault.index import VaultIndex
from friday.vault.models import ItemStatus
from friday.vault.organizer import Organizer
from friday.vault.quota import QuotaGuard
from friday.vault.solver import Solver

logger = logging.getLogger("friday.vault.pipeline")

#: Classification kinds worth spending the ensemble on. Everything else
#: (document, receipt, screenshot, photo, board, unknown) is filed with its
#: classification and left alone — there is nothing there for the solver to
#: solve. Kept as a module constant (rather than inlined) so the "what counts
#: as a problem" decision is visible and grep-able in one place.
_SOLVABLE_KINDS = frozenset({"problem"})


class Pipeline:
    """Runs one capture from committed upload through to a filed, maybe-solved item.

    Args:
        index: Where items and solves are persisted.
        ocr: Reads text out of image bytes. Runs for every capture, locked or not.
        organizer_llm: The model used to classify a capture. Wrapped in a fresh
            :class:`~friday.vault.organizer.Organizer` here so callers only ever
            hand the pipeline a bare provider.
        solver: The pre-built ensemble solver (already carries its operator
            roster) — run only for solvable, unlocked, under-quota captures.
        quota: Bounds how many solves may run today; storage quota is enforced
            earlier, at upload time, and is not this pipeline's concern.
        fetch_bytes: Resolves a committed Cloudinary ``public_id`` to the raw
            image bytes. See the module docstring for the production wiring
            this must NOT be (``lambda public_id: b""``).
    """

    def __init__(
        self,
        *,
        index: VaultIndex,
        ocr: OCRProvider,
        organizer_llm: LLMProvider,
        solver: Solver,
        quota: QuotaGuard,
        fetch_bytes: Callable[[str], bytes],
    ) -> None:
        self._index = index
        self._ocr = ocr
        self._organizer = Organizer(organizer_llm)
        self._solver = solver
        self._quota = quota
        self._fetch_bytes = fetch_bytes

    def today(self) -> str:
        """The ``YYYY-MM-DD`` (UTC) key the daily solve cap is bucketed on."""
        return datetime.now(UTC).strftime("%Y-%m-%d")

    async def process(self, owner_uid: str, item_id: str) -> None:
        """Run the whole chain for one item: OCR, classify, maybe solve, file.

        Returns quietly (does nothing) when the item does not exist or has no
        committed Cloudinary asset yet — there is nothing to process, and that
        is not this method's error to raise.
        """
        item = self._index.get_item(owner_uid, item_id)
        if item is None or item.cloudinary is None:
            return

        item.status = ItemStatus.PROCESSING
        self._index.put_item(item)

        try:
            image_bytes = self._fetch_bytes(item.cloudinary.public_id)
            item.ocr_text = await self._ocr.read(image_bytes)
            item.ocr_engine = type(self._ocr).__name__
        except Exception:  # noqa: BLE001 - a fetch/OCR failure must not lose the capture
            logger.warning(
                "vault pipeline: OCR failed for item %s, filing with empty text",
                item_id,
                exc_info=True,
            )
        self._index.put_item(item)

        if item.may_leave_for_model():
            item.classification = await self._organizer.classify(
                ocr_text=item.ocr_text, caption=item.caption
            )
            self._index.put_item(item)

            if item.classification.kind in _SOLVABLE_KINDS and self._quota.may_solve(self.today()):
                try:
                    solve = await self._solver.solve(item_ids=[item.id], ocr_text=item.ocr_text)
                except Exception:  # noqa: BLE001 - a solver outage must not lose the capture
                    logger.warning(
                        "vault pipeline: solve failed for item %s, filing unsolved",
                        item_id,
                        exc_info=True,
                    )
                else:
                    self._index.put_solve(solve)
                    self._quota.record_solve(self.today())
                    item.solve_id = solve.id
                    self._index.put_item(item)

        item.status = ItemStatus.READY
        item.processed_at = datetime.now(UTC).isoformat()
        self._index.put_item(item)
