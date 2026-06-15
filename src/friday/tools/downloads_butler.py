"""Downloads "butler": organize a folder into category subfolders by extension.

:class:`DownloadsButlerTool` tidies a single root directory (typically the
owner's *Downloads* folder) by sorting its top-level files into category
subfolders (``images/``, ``docs/``, ``archives/``, ``code/`` ...) chosen from
each file's extension. It is the only tool in slice ``tools-b`` that mutates the
filesystem, so it is ``side_effecting=True`` and ``idempotent=False`` — the
registry confirm-step (build-spec §12) gates a real (non-dry-run) move.

Safety contract:

* **Dry-run by default.** ``ButlerArgs.dry_run`` defaults to ``True``: the tool
  computes and returns the *plan* (which file would go to which category folder)
  and moves NOTHING. A caller must pass ``dry_run=False`` *and* clear the
  confirm-step to actually move files.
* **Path-confined to ``root``.** Every source and destination is resolved and
  verified to live under the (resolved) ``root``. A ``root`` that escapes via
  ``..`` or a symlink, or any planned destination that would land outside
  ``root``, yields a typed refusal (``path_escape``) and moves nothing.
* **Top level only.** Only the immediate files in ``root`` are organized; the
  category subfolders themselves (and any other existing directories) are left
  untouched, so re-running is safe and a second dry-run shows an empty plan.

The categories map is injectable so callers can extend it; the default covers
the common desktop file kinds. Nothing here imports application config — the
root and category map arrive as parameters (dependency injection).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from friday.tools.base import ToolError, ToolResult

logger = logging.getLogger("friday.tools.downloads_butler")

#: Default extension -> category mapping. Extensions are compared lower-cased and
#: WITHOUT the leading dot. Anything unmatched falls into :data:`OTHER_CATEGORY`.
DEFAULT_CATEGORIES: dict[str, str] = {
    # images
    "jpg": "images",
    "jpeg": "images",
    "png": "images",
    "gif": "images",
    "webp": "images",
    "bmp": "images",
    "svg": "images",
    "heic": "images",
    "tiff": "images",
    # documents
    "pdf": "docs",
    "doc": "docs",
    "docx": "docs",
    "odt": "docs",
    "rtf": "docs",
    "txt": "docs",
    "md": "docs",
    "xls": "docs",
    "xlsx": "docs",
    "csv": "docs",
    "ppt": "docs",
    "pptx": "docs",
    # archives
    "zip": "archives",
    "tar": "archives",
    "gz": "archives",
    "tgz": "archives",
    "bz2": "archives",
    "xz": "archives",
    "7z": "archives",
    "rar": "archives",
    # code
    "py": "code",
    "js": "code",
    "ts": "code",
    "tsx": "code",
    "jsx": "code",
    "json": "code",
    "yaml": "code",
    "yml": "code",
    "toml": "code",
    "sh": "code",
    "rs": "code",
    "go": "code",
    "c": "code",
    "h": "code",
    "cpp": "code",
    "java": "code",
    # audio / video
    "mp3": "audio",
    "wav": "audio",
    "flac": "audio",
    "m4a": "audio",
    "ogg": "audio",
    "mp4": "video",
    "mkv": "video",
    "mov": "video",
    "avi": "video",
    "webm": "video",
}

#: Category for files whose extension is not in the map (or that have none).
OTHER_CATEGORY = "other"


class ButlerArgs(BaseModel):
    """Arguments for :class:`DownloadsButlerTool`.

    ``root`` is the directory to organize. ``dry_run`` (default ``True``) returns
    the plan without moving anything; pass ``dry_run=False`` to actually move
    files (and the registry confirm-step must also be cleared).
    """

    root: str = Field(min_length=1)
    dry_run: bool = True


class PlannedMove(BaseModel):
    """One planned (or executed) relocation in the butler's plan."""

    source: str
    destination: str
    category: str


def _category_for(name: str, categories: Mapping[str, str]) -> str:
    """Return the category for a file ``name`` using ``categories`` (or OTHER)."""
    suffix = Path(name).suffix.lstrip(".").lower()
    if not suffix:
        return OTHER_CATEGORY
    return categories.get(suffix, OTHER_CATEGORY)


def _is_within(child: Path, parent: Path) -> bool:
    """True iff resolved ``child`` is ``parent`` or lives beneath it."""
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


class DownloadsButlerTool:
    """Organize ``root`` into category subfolders by file extension.

    Args:
        categories: Extension (lower-case, no dot) -> category-folder mapping.
            Defaults to :data:`DEFAULT_CATEGORIES`. Injected so callers can
            extend/replace the taxonomy without editing this module.
    """

    name = "downloads_butler"
    description = (
        "Organize files in a folder into category subfolders by extension "
        "(images/docs/archives/code/...). Dry-run by default."
    )
    args_model = ButlerArgs
    required_permission = "files"
    idempotent = False
    side_effecting = True

    def __init__(
        self, *, categories: Mapping[str, str] | None = None
    ) -> None:
        self._categories: Mapping[str, str] = (
            dict(DEFAULT_CATEGORIES) if categories is None else dict(categories)
        )

    async def __call__(self, args: Any) -> ToolResult:
        """Plan (and, unless ``dry_run``, perform) the organization of ``root``."""
        # ``args`` arrives validated from the registry; coerce defensively so the
        # tool is also safe to call directly with a raw mapping.
        if not isinstance(args, ButlerArgs):
            args = ButlerArgs.model_validate(args)

        root = Path(args.root).expanduser()
        try:
            resolved_root = root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            logger.info("butler: root %r not resolvable: %s", args.root, exc)
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="root_not_found",
                    message=f"root {args.root!r} does not exist or is unreadable",
                    retriable=False,
                ),
            )

        if not resolved_root.is_dir():
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="root_not_dir",
                    message=f"root {args.root!r} is not a directory",
                    retriable=False,
                ),
            )

        plan, escape = self._build_plan(resolved_root)
        if escape is not None:
            logger.warning("butler: planned path escape: %s", escape)
            return ToolResult(
                ok=False,
                data={},
                error=ToolError(
                    code="path_escape",
                    message=escape,
                    retriable=False,
                ),
            )

        moved: list[PlannedMove] = []
        if not args.dry_run:
            moved = self._execute(resolved_root, plan)

        return ToolResult(
            ok=True,
            data={
                "dry_run": args.dry_run,
                "root": str(resolved_root),
                "planned": [m.model_dump() for m in plan],
                "moved": [m.model_dump() for m in moved],
                "count": len(plan),
            },
            error=None,
        )

    def _build_plan(
        self, resolved_root: Path
    ) -> tuple[list[PlannedMove], str | None]:
        """Compute the move plan for top-level files; flag any path escape.

        Returns ``(plan, None)`` normally, or ``([], reason)`` if any planned
        destination would land outside ``resolved_root`` (defence in depth — the
        destinations are derived from names under ``root`` so this should not
        happen, but a hostile filename is refused rather than risked).
        """
        plan: list[PlannedMove] = []
        for entry in sorted(resolved_root.iterdir(), key=lambda p: p.name):
            # Organize only top-level *files*; leave subdirectories (including the
            # category folders themselves) untouched so re-runs are idempotent.
            if not entry.is_file():
                continue
            category = _category_for(entry.name, self._categories)
            destination = resolved_root / category / entry.name
            if not _is_within(destination.resolve(), resolved_root):
                return [], (
                    f"planned destination for {entry.name!r} escapes root"
                )
            plan.append(
                PlannedMove(
                    source=str(entry),
                    destination=str(destination),
                    category=category,
                )
            )
        return plan, None

    def _execute(
        self, resolved_root: Path, plan: list[PlannedMove]
    ) -> list[PlannedMove]:
        """Create category folders and move each planned file; return what moved.

        A destination that already exists is skipped (left in place) so the
        butler never clobbers an existing file — it reports only what it actually
        relocated.
        """
        moved: list[PlannedMove] = []
        for item in plan:
            dest = Path(item.destination)
            # Re-confirm confinement immediately before the mutation.
            if not _is_within(dest.parent.resolve(), resolved_root):
                logger.warning("butler: skipping out-of-root destination %s", dest)
                continue
            if dest.exists():
                logger.info("butler: destination exists, skipping %s", dest)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(item.source, str(dest))
            moved.append(item)
        return moved
