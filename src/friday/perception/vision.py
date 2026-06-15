"""Object-detection boundary, deterministic fake, and lazy real adapter.

* :class:`Detection` — the pydantic v2 detection result (label/confidence/bbox).
* :class:`VisionProvider` — the runtime-checkable ``detect`` protocol.
* :class:`FakeVision` — a deterministic provider returning scripted detections,
  so tests run with zero heavy libraries and no models.
* :class:`YoloVision` — the real adapter that lazy-imports ``cv2`` + ``ultralytics``
  inside its methods and raises a clear error (with a ``make install-perception``
  hint) when the backend is absent.

No heavy perception library is imported at module top level, so importing this
module never requires ``opencv-python``/``ultralytics`` and ``uv sync`` stays
unaffected.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from friday.errors import ProviderError

_INSTALL_HINT = (
    "opencv-python / ultralytics are not installed. Perception extras are "
    "optional and excluded from the uv lock; install them with "
    "`make install-perception`."
)


class Detection(BaseModel):
    """A single detected object in an image.

    Attributes:
        label: The detected class label (e.g. ``"person"``).
        confidence: Detection confidence in ``[0.0, 1.0]``.
        bbox: Bounding box as ``(x1, y1, x2, y2)`` in pixel coordinates.
    """

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[float, float, float, float]


@runtime_checkable
class VisionProvider(Protocol):
    """Contract detecting objects in an ``image`` (encoded image bytes)."""

    async def detect(self, image: bytes) -> list[Detection]:
        """Detect objects in ``image`` and return the detections.

        Args:
            image: Encoded image bytes (e.g. PNG/JPEG).

        Returns:
            The list of :class:`Detection` for this image (possibly empty).
        """
        ...


class FakeVision:
    """A deterministic :class:`VisionProvider` for tests.

    Returns its scripted ``detections`` for every call (a copy, so callers may
    mutate the result without affecting the script). The default script is a
    single high-confidence ``"person"`` so the offline path always exercises a
    non-empty detection list.
    """

    def __init__(self, detections: list[Detection] | None = None) -> None:
        """Create the fake provider.

        Args:
            detections: The detections returned by :meth:`detect`; defaults to a
                single scripted ``"person"`` detection.
        """
        self.detections: list[Detection] = (
            list(detections)
            if detections is not None
            else [Detection(label="person", confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0))]
        )

    async def detect(self, image: bytes) -> list[Detection]:
        return [d.model_copy() for d in self.detections]


class YoloVision:
    """Real :class:`VisionProvider` backed by ``ultralytics`` YOLO (lazy).

    The heavy ``cv2`` + ``ultralytics`` imports happen inside the methods (model
    load in ``__init__``), so importing this module never requires the backend.
    When the backend is missing, a :class:`friday.errors.ProviderError` is raised
    with a ``make install-perception`` hint.
    """

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25) -> None:
        """Construct the detector, loading the YOLO model.

        Args:
            model_name: The ultralytics model weights to load.
            confidence: Minimum confidence threshold for a kept detection.

        Raises:
            ProviderError: If ``cv2``/``ultralytics`` are not installed.
        """
        self.model_name = model_name
        self.confidence = confidence
        try:
            # Optional perception backend: excluded from the uv lock, so mypy has
            # no stub for it; lazily imported here and guarded by the ImportError.
            # Imported as plain modules (not ``from ultralytics import YOLO``) so
            # the whole-statement ``# type: ignore`` stays on one line ruff won't
            # wrap.
            import cv2  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: F401, PLC0415
            import ultralytics  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
            raise ProviderError(_INSTALL_HINT) from exc
        self._model = ultralytics.YOLO(model_name)

    async def detect(self, image: bytes) -> list[Detection]:
        """Decode ``image`` and run YOLO, returning detections above threshold."""
        import cv2  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415
        import numpy as np  # type: ignore[import-not-found, import-untyped, unused-ignore]  # noqa: PLC0415

        buffer = np.frombuffer(image, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        results = self._model(frame, conf=self.confidence)
        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        label=str(names[cls]),
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                    )
                )
        return detections
