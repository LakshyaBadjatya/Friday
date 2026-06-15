"""Unit tests for :class:`friday.tools.downloads_butler.DownloadsButlerTool`.

Fully offline: every test runs against a ``tmp_path`` directory, so no real
Downloads folder is touched. The tool is dry-run by default and path-confined to
its ``root``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from friday.tools.base import ToolResult
from friday.tools.downloads_butler import (
    DEFAULT_CATEGORIES,
    ButlerArgs,
    DownloadsButlerTool,
)


def _touch(directory: Path, name: str, body: str = "x") -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


# -- attributes / args --------------------------------------------------- #


def test_butler_tool_attrs() -> None:
    tool = DownloadsButlerTool()
    assert tool.name == "downloads_butler"
    assert tool.side_effecting is True
    assert tool.idempotent is False
    assert tool.required_permission == "files"
    assert tool.args_model is ButlerArgs


def test_butler_args_dry_run_defaults_true() -> None:
    args = ButlerArgs(root="/some/where")
    assert args.dry_run is True


def test_butler_args_rejects_empty_root() -> None:
    with pytest.raises(ValueError):
        ButlerArgs(root="")


# -- dry-run planning ---------------------------------------------------- #


async def test_dry_run_returns_plan_without_moving(tmp_path: Path) -> None:
    img = _touch(tmp_path, "photo.JPG")
    doc = _touch(tmp_path, "report.pdf")
    code = _touch(tmp_path, "script.py")

    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(tmp_path), dry_run=True))

    assert isinstance(result, ToolResult)
    assert result.ok is True
    assert result.data["dry_run"] is True
    assert result.data["moved"] == []
    assert result.data["count"] == 3

    # Files are untouched on disk.
    assert img.exists()
    assert doc.exists()
    assert code.exists()
    assert not (tmp_path / "images").exists()

    by_cat = {p["source"]: p["category"] for p in result.data["planned"]}
    assert by_cat[str(img)] == "images"
    assert by_cat[str(doc)] == "docs"
    assert by_cat[str(code)] == "code"


async def test_extension_match_is_case_insensitive(tmp_path: Path) -> None:
    _touch(tmp_path, "ARCHIVE.ZIP")
    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(tmp_path), dry_run=True))
    assert result.data["planned"][0]["category"] == "archives"


async def test_unknown_and_extensionless_go_to_other(tmp_path: Path) -> None:
    _touch(tmp_path, "mystery.qux")
    _touch(tmp_path, "READMEnoext")
    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(tmp_path), dry_run=True))
    cats = sorted(p["category"] for p in result.data["planned"])
    assert cats == ["other", "other"]


# -- real move ----------------------------------------------------------- #


async def test_non_dry_run_moves_files_into_category_folders(
    tmp_path: Path,
) -> None:
    img = _touch(tmp_path, "photo.png")
    doc = _touch(tmp_path, "notes.txt")

    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(tmp_path), dry_run=False))

    assert result.ok is True
    assert result.data["dry_run"] is False
    assert result.data["count"] == 2
    assert len(result.data["moved"]) == 2

    # Originals are gone; files now live under category folders.
    assert not img.exists()
    assert not doc.exists()
    assert (tmp_path / "images" / "photo.png").exists()
    assert (tmp_path / "docs" / "notes.txt").exists()


async def test_re_running_after_move_is_idempotent_empty_plan(
    tmp_path: Path,
) -> None:
    _touch(tmp_path, "photo.png")
    tool = DownloadsButlerTool()
    await tool(ButlerArgs(root=str(tmp_path), dry_run=False))

    # Second pass: the only top-level entries are now the category dirs, which
    # are skipped — so the plan is empty.
    again = await tool(ButlerArgs(root=str(tmp_path), dry_run=True))
    assert again.ok is True
    assert again.data["count"] == 0
    assert again.data["planned"] == []


async def test_existing_destination_is_not_clobbered(tmp_path: Path) -> None:
    _touch(tmp_path, "photo.png", body="new")
    images = tmp_path / "images"
    images.mkdir()
    existing = _touch(images, "photo.png", body="original")

    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(tmp_path), dry_run=False))

    # The pre-existing destination keeps its original contents; nothing moved.
    assert existing.read_text(encoding="utf-8") == "original"
    assert result.data["moved"] == []
    # The top-level source is left in place since it could not be moved.
    assert (tmp_path / "photo.png").exists()


async def test_subdirectories_are_left_untouched(tmp_path: Path) -> None:
    sub = tmp_path / "existing_dir"
    sub.mkdir()
    _touch(sub, "inner.pdf")
    _touch(tmp_path, "top.pdf")

    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(tmp_path), dry_run=True))

    sources = [p["source"] for p in result.data["planned"]]
    assert str(tmp_path / "top.pdf") in sources
    # The file nested in a subdirectory is NOT part of the plan.
    assert str(sub / "inner.pdf") not in sources


# -- path confinement ---------------------------------------------------- #


async def test_missing_root_returns_typed_error(tmp_path: Path) -> None:
    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(tmp_path / "nope")))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "root_not_found"
    assert result.error.retriable is False


async def test_root_that_is_a_file_returns_typed_error(tmp_path: Path) -> None:
    f = _touch(tmp_path, "iamafile.txt")
    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(f)))
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "root_not_dir"


async def test_traversal_in_root_is_resolved_and_confined(tmp_path: Path) -> None:
    # A root expressed with ``..`` still resolves to a real dir under tmp_path;
    # the plan's destinations all stay within the resolved root.
    real = tmp_path / "downloads"
    real.mkdir()
    _touch(real, "a.png")
    sneaky = tmp_path / "downloads" / ".." / "downloads"

    tool = DownloadsButlerTool()
    result = await tool(ButlerArgs(root=str(sneaky), dry_run=True))

    assert result.ok is True
    assert result.data["root"] == str(real.resolve())
    for planned in result.data["planned"]:
        assert planned["destination"].startswith(str(real.resolve()))


# -- custom category injection ------------------------------------------- #


async def test_injected_categories_override_default(tmp_path: Path) -> None:
    _touch(tmp_path, "thing.dat")
    tool = DownloadsButlerTool(categories={"dat": "datasets"})
    result = await tool(ButlerArgs(root=str(tmp_path), dry_run=True))
    assert result.data["planned"][0]["category"] == "datasets"


def test_default_categories_cover_expected_buckets() -> None:
    buckets = set(DEFAULT_CATEGORIES.values())
    assert {"images", "docs", "archives", "code"} <= buckets
