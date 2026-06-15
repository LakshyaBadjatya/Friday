"""Architecture guard tests.

Enforces the provider-abstraction principle (design §2.2 / spec §1.2, §9.1): no
LLM SDK may be imported anywhere under ``src/friday/agents`` or
``src/friday/core``. Business logic depends only on the ``LLMProvider``
abstraction; ``providers/llm.py`` is the single module permitted to import an
LLM SDK. This is grep-enforced so a leak fails the build, not a code review.
"""

from __future__ import annotations

import pathlib
import re

# Matches a top-level (possibly indented) ``import openai`` /
# ``from openai import ...`` / ``from openai.foo import ...`` and the equivalent
# for anthropic and google.generativeai.
_BANNED_IMPORT = re.compile(
    r"^\s*(?:import|from)\s+(?:openai|anthropic|google\.generativeai)\b",
    re.MULTILINE,
)

_GUARDED_ROOTS = (
    pathlib.Path("src/friday/agents"),
    pathlib.Path("src/friday/core"),
)


def _project_root() -> pathlib.Path:
    # tests/unit/test_architecture.py -> repo root is two parents up from tests/.
    return pathlib.Path(__file__).resolve().parents[2]


def test_no_llm_sdk_in_business_logic() -> None:
    root = _project_root()
    checked = 0
    for rel_root in _GUARDED_ROOTS:
        guarded = root / rel_root
        if not guarded.exists():
            continue
        for py_file in guarded.rglob("*.py"):
            checked += 1
            source = py_file.read_text(encoding="utf-8")
            match = _BANNED_IMPORT.search(source)
            assert match is None, (
                f"LLM SDK import leaked into {py_file}: {match.group(0).strip()!r}. "
                "Business logic must depend on the LLMProvider abstraction; only "
                "src/friday/providers/llm.py may import an LLM SDK."
            )
    # Guard against a silent no-op if the layout ever changes: at least the
    # __init__.py modules under each root should have been scanned.
    assert checked >= len(_GUARDED_ROOTS), (
        "architecture guard scanned no files; the agents/core package layout "
        "may have moved"
    )
