# © Lakshya Badjatya — Author
"""Integration tests for the ``/models`` API + the ``/chat`` model override.

Fully OFFLINE and DETERMINISTIC: no network, no key. The app is built on the
``fake`` provider (so it constructs no real gateway), and the enabled tests
*inject* a real :class:`~friday.models.gateway.ModelGateway` over scripted
:class:`~friday.providers.llm.FakeLLM` providers (plus a
:class:`~friday.models.catalog.ModelCatalog`) onto ``app.state`` before hitting
the routes. The single source of model truth is the real catalog/gateway logic;
only the providers are faked.

Covered:
* ``GET /models`` lists the available models + the active id.
* ``POST /models/active`` switches the active model and rejects an unknown id (404).
* ``POST /models/compare`` returns one ``CompareResult`` per requested model
  (and a judged ``best`` when asked).
* ``/chat`` with a ``model`` override routes that one turn through the chosen
  model (and restores the gateway's active model afterwards).
* Every ``/models`` route is ``404`` on the fake build (no gateway wired).
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from friday.app import create_app
from friday.core.orchestrator import Orchestrator
from friday.memory.short_term import ShortTermMemory
from friday.models.catalog import ModelCatalog, ModelInfo
from friday.models.gateway import ModelGateway
from friday.providers.llm import FakeLLM, LLMProvider, LLMResponse, Usage
from friday.tools.registry import ToolRegistry
from friday.tools.web_search import WebSearchTool

PERSONA_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "friday" / "persona" / "friday.md"
)

# Two catalogued models served by two fake providers — enough to exercise list /
# switch / compare without any real provider.
_CATALOG: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="openrouter:google/gemma-4-31b-it:free",
        provider="openrouter",
        model="google/gemma-4-31b-it:free",
        label="Gemma 4 31B IT",
        free=True,
    ),
    ModelInfo(
        id="opencode:mimo-v2.5-free",
        provider="opencode",
        model="mimo-v2.5-free",
        label="MiMo v2.5",
        free=True,
    ),
)

_DEFAULT_ACTIVE = "openrouter:google/gemma-4-31b-it:free"


def _resp(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], usage=Usage())


def _make_gateway(
    *,
    openrouter: list[LLMResponse] | None = None,
    opencode: list[LLMResponse] | None = None,
    default_model_id: str = _DEFAULT_ACTIVE,
) -> tuple[ModelGateway, ModelCatalog]:
    """A real gateway over two scripted fake providers + a real catalog."""
    catalog = ModelCatalog(
        available_providers={"openrouter", "opencode"}, catalog=_CATALOG
    )
    providers: dict[str, LLMProvider] = {
        "openrouter": FakeLLM(responses=openrouter or []),
        "opencode": FakeLLM(responses=opencode or []),
    }
    gateway = ModelGateway(
        providers, catalog, default_model_id=default_model_id
    )
    return gateway, catalog


def _inject_gateway(app: object, gateway: ModelGateway, catalog: ModelCatalog) -> None:
    """Stash a gateway + catalog onto app.state (the enabled-build shape)."""
    app.state.gateway = gateway  # type: ignore[attr-defined]
    app.state.model_catalog = catalog  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# No gateway (fake build) -> every /models route is 404
# --------------------------------------------------------------------------- #
def test_models_routes_404_without_gateway() -> None:
    """On the fake build no gateway is wired, so every /models route is 404."""
    app = create_app()
    with TestClient(app) as client:
        # Fake build: build_runtime wired no gateway.
        assert client.app.state.gateway is None
        assert client.app.state.model_catalog is None
        listing = client.get("/models")
        active = client.post(
            "/models/active", json={"model_id": "openrouter:google/gemma-4-31b-it:free"}
        )
        compare = client.post("/models/compare", json={"prompt": "hi"})
    assert listing.status_code == 404
    assert active.status_code == 404
    assert compare.status_code == 404


# --------------------------------------------------------------------------- #
# GET /models lists the available models + the active id
# --------------------------------------------------------------------------- #
def test_list_models_returns_models_and_active() -> None:
    app = create_app()
    gateway, catalog = _make_gateway()
    with TestClient(app) as client:
        _inject_gateway(client.app, gateway, catalog)
        resp = client.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] == _DEFAULT_ACTIVE
    ids = [m["id"] for m in body["models"]]
    assert ids == [
        "openrouter:google/gemma-4-31b-it:free",
        "opencode:mimo-v2.5-free",
    ]
    # Each entry carries the full ModelInfo shape.
    first = body["models"][0]
    assert first["provider"] == "openrouter"
    assert first["model"] == "google/gemma-4-31b-it:free"
    assert first["label"] == "Gemma 4 31B IT"
    assert first["free"] is True


# --------------------------------------------------------------------------- #
# POST /models/active switches the active model (and rejects unknown ids)
# --------------------------------------------------------------------------- #
def test_set_active_switches_model() -> None:
    app = create_app()
    gateway, catalog = _make_gateway()
    with TestClient(app) as client:
        _inject_gateway(client.app, gateway, catalog)
        resp = client.post(
            "/models/active", json={"model_id": "opencode:mimo-v2.5-free"}
        )
        # The switch is reflected on a subsequent listing.
        listing = client.get("/models")
    assert resp.status_code == 200
    assert resp.json() == {"active": "opencode:mimo-v2.5-free"}
    assert listing.json()["active"] == "opencode:mimo-v2.5-free"
    assert gateway.active_model_id == "opencode:mimo-v2.5-free"


def test_set_active_unknown_id_is_404() -> None:
    app = create_app()
    gateway, catalog = _make_gateway()
    with TestClient(app) as client:
        _inject_gateway(client.app, gateway, catalog)
        resp = client.post(
            "/models/active", json={"model_id": "openrouter:does-not-exist"}
        )
        # The active model is unchanged after a rejected switch.
        listing = client.get("/models")
    assert resp.status_code == 404
    assert listing.json()["active"] == _DEFAULT_ACTIVE
    assert gateway.active_model_id == _DEFAULT_ACTIVE


# --------------------------------------------------------------------------- #
# POST /models/compare returns one CompareResult per requested model
# --------------------------------------------------------------------------- #
def test_compare_returns_one_result_per_model() -> None:
    app = create_app()
    gateway, catalog = _make_gateway(
        openrouter=[_resp("gemma says hi")],
        opencode=[_resp("mimo says hi")],
    )
    with TestClient(app) as client:
        _inject_gateway(client.app, gateway, catalog)
        resp = client.post(
            "/models/compare",
            json={
                "prompt": "say hi",
                "models": [
                    "openrouter:google/gemma-4-31b-it:free",
                    "opencode:mimo-v2.5-free",
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["best"] is None  # judge not requested
    results = body["results"]
    assert len(results) == 2
    by_id = {r["model_id"]: r for r in results}
    assert by_id["openrouter:google/gemma-4-31b-it:free"]["text"] == "gemma says hi"
    assert by_id["openrouter:google/gemma-4-31b-it:free"]["ok"] is True
    assert by_id["opencode:mimo-v2.5-free"]["text"] == "mimo says hi"
    assert by_id["opencode:mimo-v2.5-free"]["ok"] is True


def test_compare_with_judge_returns_best() -> None:
    app = create_app()
    # Two candidate answers, then the judge (active model = openrouter) names one.
    gateway, catalog = _make_gateway(
        openrouter=[_resp("gemma answer"), _resp("[opencode:mimo-v2.5-free]")],
        opencode=[_resp("mimo answer")],
    )
    with TestClient(app) as client:
        _inject_gateway(client.app, gateway, catalog)
        resp = client.post(
            "/models/compare",
            json={
                "prompt": "which is best?",
                "models": [
                    "openrouter:google/gemma-4-31b-it:free",
                    "opencode:mimo-v2.5-free",
                ],
                "judge": True,
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 2
    assert body["best"] == "opencode:mimo-v2.5-free"


# --------------------------------------------------------------------------- #
# /chat with a "model" override routes that turn to the chosen model
# --------------------------------------------------------------------------- #
def _orchestrator_over(gateway: ModelGateway) -> Orchestrator:
    """A real orchestrator whose LLM is the gateway (a drop-in LLMProvider)."""
    registry = ToolRegistry()
    registry.register(WebSearchTool())
    return Orchestrator(
        llm=gateway,
        registry=registry,
        memory=ShortTermMemory(),
        persona_path=PERSONA_PATH,
    )


def test_chat_model_override_routes_that_turn() -> None:
    """A ``model`` override sends the turn to the chosen provider, then restores."""
    app = create_app()
    # The default active model (openrouter) would script "gemma reply"; the
    # override must instead pull from the opencode provider for this one turn.
    gateway, catalog = _make_gateway(
        openrouter=[_resp("gemma reply")],
        opencode=[_resp("Routed to MiMo, Boss.")],
    )
    orchestrator = _orchestrator_over(gateway)
    with TestClient(app) as client:
        _inject_gateway(client.app, gateway, catalog)
        client.app.state.orchestrator = orchestrator
        resp = client.post(
            "/chat",
            json={
                "session_id": "models-1",
                "text": "what's 2+2",
                "model": "opencode:mimo-v2.5-free",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Routed to MiMo, Boss."
    # The override is a single-turn switch: the active model is restored.
    assert gateway.active_model_id == _DEFAULT_ACTIVE


def test_chat_without_override_uses_active_model() -> None:
    """Omitting ``model`` keeps the default path (the gateway's active model)."""
    app = create_app()
    gateway, catalog = _make_gateway(
        openrouter=[_resp("Four, Boss.")],
        opencode=[_resp("should not be used")],
    )
    orchestrator = _orchestrator_over(gateway)
    with TestClient(app) as client:
        _inject_gateway(client.app, gateway, catalog)
        client.app.state.orchestrator = orchestrator
        resp = client.post(
            "/chat", json={"session_id": "models-2", "text": "what's 2+2"}
        )
    assert resp.status_code == 200
    assert resp.json()["text"] == "Four, Boss."
    assert gateway.active_model_id == _DEFAULT_ACTIVE
