"""Discover and load owner-supplied tool plugins into the tool registry.

A *plugin* is a single ``.py`` file in the plugins directory that defines a
module-level ``def get_tools() -> list[Tool]`` returning instances satisfying the
:class:`~friday.tools.base.Tool` protocol (``name`` / ``description`` /
``args_model`` / ``required_permission`` / ``idempotent`` / ``side_effecting`` /
``__call__``). See ``examples/plugins/README.md`` for the convention and the
trusted-code caveat.

Two public entry points:

* :func:`discover_plugins` — load every ``*.py`` in the directory (skipping
  dunder files and ``README``), call its ``get_tools()``, and report a
  :class:`PluginInfo` per file. Any import / attribute / call error is *captured*
  into ``PluginInfo.error`` and discovery continues — this function never raises
  on a bad plugin, so one broken file can never crash startup.
* :func:`load_into` — discover, then register each loaded plugin's tools into a
  :class:`~friday.tools.registry.ToolRegistry`. A plugin tool whose ``name``
  collides with an already-registered (built-in) tool is **rejected**: the
  collision is recorded as that plugin's ``error`` and the built-in is left
  intact (never overwritten). Returns the :class:`PluginInfo` list.

**Trust model.** Plugins are arbitrary local Python the owner drops in (like a
shell rc); loading one executes its top-level code. The whole surface is off
behind ``FRIDAY_ENABLE_PLUGINS`` and every loaded tool still flows through the
registry's permission + confirm-step gates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from pydantic import BaseModel

from friday.logging import get_logger
from friday.tools.base import Tool
from friday.tools.registry import ToolRegistry

logger = get_logger("friday.plugins.loader")


class PluginInfo(BaseModel):
    """The outcome of loading one plugin file.

    ``name`` is the module stem (the file name without ``.py``). ``path`` is the
    absolute path the plugin was loaded from. ``tools`` lists the names of the
    tools the plugin contributed — for :func:`discover_plugins` this is every
    tool ``get_tools()`` returned; for :func:`load_into` it is the subset that
    was actually registered (a built-in collision drops the colliding name and
    records it in ``error``). ``error`` is ``None`` on success, otherwise a
    human-readable description of the import / attribute / call / collision
    failure.
    """

    name: str
    path: str
    tools: list[str] = []
    error: str | None = None


def _candidate_files(plugins_dir: str) -> list[Path]:
    """Return the ``*.py`` plugin files in ``plugins_dir`` (sorted, filtered).

    Skips dunder files (``__init__.py`` / ``__main__.py`` / any ``__*``) and any
    ``README`` stem so docs dropped beside plugins are ignored. A missing or
    non-directory path yields an empty list — an absent plugins dir is not an
    error, it simply means there are no plugins.
    """
    base = Path(plugins_dir)
    if not base.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(base.glob("*.py")):
        stem = path.stem
        if stem.startswith("__"):
            continue
        if stem.lower() == "readme":
            continue
        files.append(path)
    return files


def _load_tools_from_file(path: Path) -> list[Tool]:
    """Import ``path`` as a module and return the tools its ``get_tools()`` yields.

    Loads the file via :func:`importlib.util.spec_from_file_location` under a
    namespaced module name, calls module-level ``get_tools()``, and validates the
    result is a list/tuple of objects structurally satisfying the :class:`Tool`
    protocol. Any failure raises — the caller (:func:`discover_plugins`) is
    responsible for capturing it into a :class:`PluginInfo`.
    """
    module_name = f"friday_plugins.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so a plugin that introspects ``sys.modules`` (or uses
    # dataclasses/pickle, which look the module up by name) resolves correctly.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Don't leave a half-initialized module lingering in sys.modules.
        sys.modules.pop(module_name, None)
        raise

    get_tools = getattr(module, "get_tools", None)
    if get_tools is None:
        raise AttributeError(
            f"plugin {path.name!r} defines no module-level get_tools()"
        )
    if not callable(get_tools):
        raise TypeError(f"plugin {path.name!r} get_tools is not callable")

    raw = get_tools()
    if not isinstance(raw, list | tuple):
        raise TypeError(
            f"plugin {path.name!r} get_tools() must return a list of tools, "
            f"got {type(raw).__name__}"
        )

    tools: list[Tool] = []
    for item in raw:
        if not isinstance(item, Tool):
            raise TypeError(
                f"plugin {path.name!r} get_tools() returned an object that is "
                f"not a Tool: {type(item).__name__}"
            )
        tools.append(item)
    return tools


def discover_plugins(plugins_dir: str) -> list[PluginInfo]:
    """Discover every plugin in ``plugins_dir``, never raising on a bad one.

    For each ``*.py`` file (dunder files and ``README`` skipped), import it and
    call ``get_tools()``, collecting the returned tool names into a
    :class:`PluginInfo`. ANY import / attribute / call / type error is captured
    into ``PluginInfo.error`` (and logged) and discovery moves on to the next
    file, so a single broken plugin can never crash discovery or startup. The
    returned list contains one entry per candidate file — loaded *and* errored —
    in sorted filename order.
    """
    infos: list[PluginInfo] = []
    for path in _candidate_files(plugins_dir):
        info = PluginInfo(name=path.stem, path=str(path.resolve()))
        try:
            tools = _load_tools_from_file(path)
        except BaseException as exc:  # noqa: BLE001 - isolation is the whole point
            # Capture *anything* (including SyntaxError raised on exec, and even
            # a plugin that raises BaseException) so one bad file is isolated and
            # never propagates out of discovery.
            info.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "plugin failed to load",
                extra={"plugin": info.name, "path": info.path, "error": info.error},
            )
        else:
            info.tools = [tool.name for tool in tools]
            logger.info(
                "plugin loaded",
                extra={"plugin": info.name, "tools": info.tools},
            )
        infos.append(info)
    return infos


def load_into(registry: ToolRegistry, plugins_dir: str) -> list[PluginInfo]:
    """Discover plugins and register their tools into ``registry``.

    Re-runs discovery and, for each successfully-loaded plugin, registers its
    tools into ``registry`` — but **rejects** a tool whose ``name`` is already
    registered (a built-in). Because ``app.py`` registers all built-in tools
    *first*, this guarantees a plugin can never shadow a built-in: the collision
    is recorded on that plugin's ``error`` and the colliding name is dropped from
    its ``tools`` (the built-in entry is left untouched). A plugin that already
    failed discovery is passed through unchanged (no tools to register).

    Returns the :class:`PluginInfo` list (loaded + errored), where each loaded
    entry's ``tools`` reflects what was *actually* registered.
    """
    infos: list[PluginInfo] = []
    for path in _candidate_files(plugins_dir):
        info = PluginInfo(name=path.stem, path=str(path.resolve()))
        try:
            tools = _load_tools_from_file(path)
        except BaseException as exc:  # noqa: BLE001 - isolation is the whole point
            info.error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "plugin failed to load",
                extra={"plugin": info.name, "path": info.path, "error": info.error},
            )
            infos.append(info)
            continue

        registered: list[str] = []
        collisions: list[str] = []
        for tool in tools:
            if _is_registered(registry, tool.name):
                collisions.append(tool.name)
                logger.warning(
                    "plugin tool rejected: name collides with a built-in",
                    extra={"plugin": info.name, "tool": tool.name},
                )
                continue
            registry.register(tool)
            registered.append(tool.name)
        info.tools = registered
        if collisions:
            joined = ", ".join(repr(name) for name in collisions)
            info.error = (
                f"rejected tool(s) {joined}: name already registered "
                "(built-in tools win collisions)"
            )
        infos.append(info)
    return infos


def _is_registered(registry: ToolRegistry, name: str) -> bool:
    """Whether ``name`` is already registered, via the registry's public ``get``.

    Uses the public :meth:`ToolRegistry.get` (which raises :class:`KeyError` for
    an unknown name) rather than reaching into registry internals, so the
    collision check stays correct as the registry evolves.
    """
    try:
        registry.get(name)
    except KeyError:
        return False
    return True
