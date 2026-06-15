# FRIDAY plugins

FRIDAY can be extended with **plugins**: plain Python files you drop into a
plugins directory that contribute extra tools. At startup FRIDAY discovers each
file, loads it, and registers the tools it exposes into the same tool registry
the built-in tools use — so your plugin tools get the **same permission gating
and confirm-step** as everything else. There is no separate execution path.

The whole feature is **off by default** behind `FRIDAY_ENABLE_PLUGINS`.

## Enabling plugins

```bash
export FRIDAY_ENABLE_PLUGINS=true
# Optional — defaults to "plugins" (relative to the process working directory).
export FRIDAY_PLUGINS_DIR=plugins
```

Put your plugin files in that directory, then start FRIDAY. You can see what
loaded (and what failed, and why) at `GET /plugins`:

```bash
curl localhost:8000/plugins
# [{"name": "hello_plugin", "path": ".../plugins/hello_plugin.py",
#   "tools": ["dice_roll"], "error": null}]
```

When `FRIDAY_ENABLE_PLUGINS` is off, no plugins are loaded and `GET /plugins`
returns `404` — the feature simply does not exist for callers.

## The `get_tools()` convention

A plugin is a single `*.py` file in the plugins directory that defines a
**module-level** function:

```python
def get_tools() -> list[Tool]:
    ...
```

It returns a list of objects that satisfy the `friday.tools.base.Tool` protocol.
Each tool is a small class exposing:

| attribute             | meaning                                                       |
| --------------------- | ------------------------------------------------------------- |
| `name`                | unique tool name (the registry key)                           |
| `description`         | one-line description the LLM sees                             |
| `args_model`          | a pydantic `BaseModel` subclass validating the call arguments |
| `required_permission` | the permission string the registry gates on                   |
| `idempotent`          | `True` if re-running with the same args is safe               |
| `side_effecting`      | `True` if it reaches the outside world (gated by confirm-step) |
| `async __call__(self, args)` | runs the tool, returning a `ToolResult`                |

A side-effecting, non-idempotent tool is automatically held at the registry's
**confirm-step** until the caller confirms — you get that for free.

The discovery rules:

- Only `*.py` files are loaded. Dunder files (`__init__.py`, `__main__.py`, any
  `__*`) and `README` are skipped.
- The loader calls `get_tools()` and registers each returned tool.
- A tool whose `name` **collides with a built-in** tool is **rejected** — the
  built-in always wins, and the collision is reported in that plugin's `error`
  (the built-in is never overwritten). Choose a distinctive `name`.

See [`hello_plugin.py`](./hello_plugin.py) for a complete, deterministic
example (`dice_roll`).

## Failure isolation

Loading is **resilient**: if a plugin has a syntax error, a missing
`get_tools()`, or `get_tools()` raises, that plugin is skipped — its `error` is
recorded and surfaced at `GET /plugins`, and **every other plugin still loads**.
One broken plugin can never crash startup.

## ⚠️ Trusted code only

A plugin is **arbitrary Python that FRIDAY executes** when it loads the file —
exactly like a shell `rc` file or a `git` hook. Loading a plugin runs its
top-level code with the same privileges as FRIDAY itself.

**Only install plugins you wrote or fully trust.** Never drop a plugin from an
untrusted source into your plugins directory. This is by design — plugins are an
owner-level extension mechanism, not a sandbox.
