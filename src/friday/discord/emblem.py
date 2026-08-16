"""An arc reactor per operator, drawn rather than downloaded.

The roster needed faces. A Discord webhook takes an ``avatar_url`` and nothing
else — no file upload, no data URI — so each operator needs a real image at a
real public address, which is what the route in ``routes_discord`` exists for.

The reference sheet supplied was a watermarked stock product listing, so these
are drawn from scratch in the same visual language instead: concentric rings, a
segmented bezel, the inverted triangle in the middle, a lit core. Original
geometry, no licence attached to it, and it costs about a millisecond.

Each operator gets its own hue, its own segment count and its own rotation, so
they are told apart at 40 pixels in a message list — which is the only size
anybody will ever see them at. Drawing rather than shipping eight PNGs also
means the repository stays free of binaries and a new operator needs one line
here rather than a trip through an image editor.
"""

from __future__ import annotations

import io
import math
import os
from typing import Any

#: ``name -> (rim, core, segments, tilt)``. The hue carries the operator's
#: character — EDITH's warning red, GECKO's money green, FORGE's hot metal —
#: while the segment count and tilt give each ring a different silhouette so
#: they stay distinguishable when the colour is too small to read.
_MARKS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int], int, float]] = {
    "FRIDAY":   ((255, 138, 40),  (255, 214, 160), 12, 0.0),
    "JARVIS":   ((90, 190, 255),  (220, 245, 255), 18, 10.0),
    "EDITH":    ((235, 64, 68),   (255, 176, 176), 10, 18.0),
    "ORACLE":   ((150, 110, 255), (214, 196, 255), 14, 12.0),
    "GECKO":    ((60, 210, 130),  (176, 255, 214), 8, 24.0),
    "KAREN":    ((255, 190, 60),  (255, 232, 176), 11, 6.0),
    "VERONICA": ((255, 100, 190), (255, 190, 228), 13, 30.0),
    "JOCASTA":  ((70, 205, 220),  (180, 244, 250), 9, 15.0),
    "VISION":   ((250, 220, 90),  (255, 244, 190), 16, 21.0),
    "FORGE":    ((255, 130, 50),  (255, 200, 150), 7, 27.0),
}

#: Rendered at 256 and left for Discord to downscale. Drawing straight at the
#: display size gives visibly ragged curves; supersampling is cheaper than
#: antialiasing circles by hand.
_SIZE = 256
_SUPER = 4

_BACKPLATE = (11, 14, 19, 255)

_CACHE: dict[str, bytes] = {}


def known(name: str) -> bool:
    """Whether an emblem exists for this name."""
    return (name or "").strip().upper() in _MARKS


def render(name: str) -> bytes | None:
    """PNG bytes for an operator's emblem, or ``None`` if the name is unknown.

    Cached: the drawing is deterministic and Discord refetches avatars often
    enough that redrawing each time would be waste for no gain.
    """
    key = (name or "").strip().upper()
    if key not in _MARKS:
        return None
    cached = _CACHE.get(key)
    if cached is None:
        cached = _draw(key)
        _CACHE[key] = cached
    return cached


def _draw(key: str) -> bytes:
    """Draw one reactor at 4x and downsample.

    Pillow is imported here rather than at module scope on purpose. This module
    is reached from the Discord gateway's import chain, and a test exists
    specifically to keep heavy libraries out of it — an emblem is drawn nine
    times in the life of the process, so paying for the import at module load
    would be the wrong trade even without that test.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415
    from PIL.Image import Resampling  # noqa: PLC0415

    rim, core, segments, tilt = _MARKS[key]
    size = _SIZE * _SUPER
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    pen = ImageDraw.Draw(image)
    mid = size / 2

    # The dark disc everything sits on, inset slightly so the rim stroke is not
    # clipped by the edge of the canvas.
    pen.ellipse(_box(mid, size * 0.47), fill=_BACKPLATE)
    pen.ellipse(_box(mid, size * 0.47), outline=(*rim, 255), width=int(size * 0.022))

    # The segmented bezel: short spokes between two radii, which reads as "arc
    # reactor" more than any other single element of the design.
    inner, outer = size * 0.30, size * 0.425
    for step in range(segments):
        angle = math.radians(tilt + step * 360.0 / segments)
        pen.line(
            [
                (mid + inner * math.cos(angle), mid + inner * math.sin(angle)),
                (mid + outer * math.cos(angle), mid + outer * math.sin(angle)),
            ],
            fill=(*rim, 210),
            width=int(size * 0.026),
        )

    pen.ellipse(_box(mid, size * 0.285), outline=(*rim, 255), width=int(size * 0.018))

    # The inverted triangle, tilted with the bezel so the whole mark turns
    # together instead of looking like two stacked designs.
    points = []
    for step in range(3):
        angle = math.radians(tilt + 90 + step * 120)
        points.append(
            (mid + size * 0.225 * math.cos(angle), mid + size * 0.225 * math.sin(angle))
        )
    pen.polygon(points, outline=(*core, 255), width=int(size * 0.016))

    # The lit core, as shrinking discs — a cheap glow that survives being
    # downscaled to 40 pixels, which a real blur does not. The outer halo takes
    # the rim colour rather than the pale core one: a light tint at low alpha
    # over a near-black plate averages out to grey, which made every reactor
    # look like it had a smudge in the middle regardless of its hue.
    for radius, alpha, tint in (
        (0.150, 70, rim), (0.115, 130, rim), (0.085, 200, core), (0.060, 255, core)
    ):
        pen.ellipse(_box(mid, size * radius), fill=(*tint, alpha))
    pen.ellipse(_box(mid, size * 0.032), fill=(255, 255, 255, 255))

    small = image.resize((_SIZE, _SIZE), Resampling.LANCZOS)
    buffer = io.BytesIO()
    small.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _box(mid: float, radius: float) -> tuple[float, float, float, float]:
    """A bounding box centred on the canvas — the only kind used here."""
    return (mid - radius, mid - radius, mid + radius, mid + radius)


def public_base() -> str:
    """Where this instance is reachable from the internet, or ``""``.

    Render exports its own external URL, so there is nothing to configure in
    the normal case. ``FRIDAY_PUBLIC_URL`` overrides it for anywhere else.
    """
    return (
        os.environ.get("FRIDAY_PUBLIC_URL")
        or os.environ.get("RENDER_EXTERNAL_URL")
        or ""
    ).strip()


def avatar_url(base: Any, name: str) -> str:
    """Public URL for an operator's emblem, or ``""`` when there is no base.

    Discord fetches this itself, so it has to be reachable from the internet
    rather than from inside the process. With no public base URL the caller
    simply posts without an avatar — a plainer message, not a failure.
    """
    root = str(base or "").strip().rstrip("/")
    key = (name or "").strip().upper()
    if not root or key not in _MARKS:
        return ""
    return f"{root}/discord/emblem/{key.lower()}.png"
