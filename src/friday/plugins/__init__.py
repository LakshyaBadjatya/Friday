"""Plugin / extension system (Tier 2; default off).

This package lets the owner drop a Python file into a ``plugins/`` directory that
exposes FRIDAY tools; the loader discovers each file, calls its ``get_tools()``
convention, and registers the returned :class:`~friday.tools.base.Tool` instances
into the shared :class:`~friday.tools.registry.ToolRegistry` at startup. Plugin
tools then go through the *same* permission gating + confirm-step as built-in
tools — there is no separate execution path.

Plugins are **trusted local code** the owner installs (arbitrary Python by
design, like a shell rc file); the whole surface is off behind
``FRIDAY_ENABLE_PLUGINS``. A broken plugin is captured and skipped, never
crashing startup, and a built-in tool always wins a name collision.

The public surface is :class:`~friday.plugins.loader.PluginInfo` plus the
:func:`~friday.plugins.loader.discover_plugins` /
:func:`~friday.plugins.loader.load_into` functions.
"""

from __future__ import annotations

from friday.plugins.loader import PluginInfo, discover_plugins, load_into

__all__ = ["PluginInfo", "discover_plugins", "load_into"]
