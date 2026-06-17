# © Lakshya Badjatya — Author
"""Text-to-image seam — a prompt in, an image (base64) out.

A runtime-checkable :class:`ImageGenerator` protocol with a dependency-free
:class:`FakeImageGenerator` (a deterministic SVG placeholder that embeds the
prompt — visibly a stand-in, perfect for offline tests and demos) and a lazily
imported :class:`DiffusersImageGenerator` real backend. :func:`build_image_generator`
returns ``None`` unless ``enable_imagegen`` is on and degrades to the fake when
``diffusers`` is absent, so the flag is safe on a machine without a GPU stack.

Output is a uniform :class:`GeneratedImage` (``media_type`` + base64 ``data``) so
the route never has to special-case the backend's image format.
"""

from __future__ import annotations

import base64
import logging
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from friday.config import Settings

logger = logging.getLogger("friday.multimodal.imagegen")


class GeneratedImage(BaseModel):
    """A generated image: the prompt, its MIME type, and base64-encoded bytes."""

    prompt: str
    media_type: str
    data_base64: str


@runtime_checkable
class ImageGenerator(Protocol):
    """Anything that renders a text prompt to an image."""

    def generate(self, prompt: str) -> GeneratedImage:
        """Render ``prompt`` and return a :class:`GeneratedImage`."""
        ...


def _svg_placeholder(prompt: str) -> str:
    """A small, deterministic SVG that displays the (escaped, clipped) prompt."""
    clipped = (prompt or "").strip()[:120]
    safe = (
        clipped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512">'
        '<rect width="512" height="512" fill="#060910"/>'
        '<circle cx="256" cy="220" r="120" fill="none" stroke="#4fe3ff" '
        'stroke-width="6"/>'
        '<text x="256" y="430" fill="#9fb3cc" font-family="monospace" '
        f'font-size="18" text-anchor="middle">{safe}</text>'
        "</svg>"
    )


class FakeImageGenerator:
    """A deterministic generator that returns an SVG placeholder (no deps)."""

    def generate(self, prompt: str) -> GeneratedImage:
        """Return an SVG placeholder embedding the prompt."""
        svg = _svg_placeholder(prompt)
        data = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        return GeneratedImage(
            prompt=prompt, media_type="image/svg+xml", data_base64=data
        )


class DiffusersImageGenerator:
    """Real text-to-image over ``diffusers`` (lazy-imported in :meth:`generate`)."""

    def __init__(self, model: str = "stabilityai/sd-turbo") -> None:
        self._model = model

    def generate(self, prompt: str) -> GeneratedImage:
        """Run the diffusers pipeline and return a PNG image (base64)."""
        import io  # noqa: PLC0415

        pipeline = _load_diffusers(self._model)
        image = pipeline(prompt).images[0]
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        return GeneratedImage(prompt=prompt, media_type="image/png", data_base64=data)


def _load_diffusers(model: str) -> Any:
    """Lazy-load a diffusers pipeline; raises ImportError when not installed."""
    import diffusers  # type: ignore[import-not-found]  # noqa: PLC0415

    return diffusers.AutoPipelineForText2Image.from_pretrained(model)


def _diffusers_available() -> bool:
    """Whether ``diffusers`` can be imported (no model download)."""
    try:
        import diffusers  # noqa: F401,PLC0415
    except ImportError:
        return False
    return True


def build_image_generator(settings: Settings) -> ImageGenerator | None:
    """Return a generator when ``enable_imagegen`` is on, else ``None``.

    Real ``diffusers`` backend when installed, otherwise the deterministic SVG
    fake (logged), so the flag is safe without a heavy image stack.
    """
    if not settings.enable_imagegen:
        return None
    if _diffusers_available():
        return DiffusersImageGenerator()
    logger.warning(
        "enable_imagegen is set but diffusers is not installed; using the "
        "deterministic SVG FakeImageGenerator (install image extras for real gen)"
    )
    return FakeImageGenerator()
