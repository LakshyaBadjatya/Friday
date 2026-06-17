# © Lakshya Badjatya — Author
"""Validity tests for the browser-extension surface (MV3 manifest + popup)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

_EXT_DIR = Path(__file__).resolve().parents[2] / "src" / "friday" / "browser_ext"


def test_manifest_is_valid_mv3() -> None:
    manifest = json.loads((_EXT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["manifest_version"] == 3
    assert manifest["name"]
    assert manifest["version"]
    # The popup the toolbar action opens must exist on disk.
    popup = manifest["action"]["default_popup"]
    assert (_EXT_DIR / popup).is_file()


def test_popup_html_loads_popup_js() -> None:
    html = (_EXT_DIR / "popup.html").read_text(encoding="utf-8")
    assert "popup.js" in html


def test_popup_js_posts_to_chat() -> None:
    js = (_EXT_DIR / "popup.js").read_text(encoding="utf-8")
    assert "/chat" in js
    assert "session_id" in js


def test_popup_js_is_valid_javascript() -> None:
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        pytest.skip("node is not available")
    result = subprocess.run(
        [node, "--check", str(_EXT_DIR / "popup.js")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
