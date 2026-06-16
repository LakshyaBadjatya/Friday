# © Lakshya Badjatya — Author
"""Unit tests for the model catalog (``friday.models.catalog``).

The catalog is a pure, offline lookup over the verified default models: it filters
the listing to the providers actually available at runtime and resolves a
``provider:model`` id to a :class:`ModelInfo`. No network, no LLM SDK.
"""

from __future__ import annotations

from friday.models.catalog import DEFAULT_CATALOG, ModelCatalog, ModelInfo


def test_model_info_id_is_provider_colon_model() -> None:
    info = ModelInfo(
        id="openrouter:google/gemma-4-31b-it:free",
        provider="openrouter",
        model="google/gemma-4-31b-it:free",
        label="Gemma 4 31B",
        free=True,
    )
    assert info.id == f"{info.provider}:{info.model}"


def test_default_catalog_ids_match_provider_colon_model() -> None:
    assert DEFAULT_CATALOG, "default catalog must not be empty"
    for info in DEFAULT_CATALOG:
        assert info.id == f"{info.provider}:{info.model}"


def test_default_catalog_includes_verified_models() -> None:
    ids = {info.id for info in DEFAULT_CATALOG}
    assert "openrouter:google/gemma-4-31b-it:free" in ids
    assert "openrouter:openai/gpt-oss-20b:free" in ids
    assert "opencode:mimo-v2.5-free" in ids
    assert "nvidia:meta/llama-3.1-8b-instruct" in ids


def test_default_catalog_all_free() -> None:
    # All verified models are free-tier (incl. the NVIDIA fallback).
    for info in DEFAULT_CATALOG:
        assert info.free is True


def test_catalog_filters_by_available_providers() -> None:
    catalog = ModelCatalog(available_providers={"openrouter"})
    listed = catalog.list_models()
    assert listed, "expected at least one openrouter model"
    assert all(info.provider == "openrouter" for info in listed)
    # opencode / nvidia are filtered out.
    providers = {info.provider for info in listed}
    assert providers == {"openrouter"}


def test_catalog_lists_multiple_available_providers() -> None:
    catalog = ModelCatalog(available_providers={"openrouter", "opencode"})
    providers = {info.provider for info in catalog.list_models()}
    assert providers == {"openrouter", "opencode"}


def test_catalog_empty_when_no_providers_available() -> None:
    catalog = ModelCatalog(available_providers=set())
    assert catalog.list_models() == []


def test_catalog_get_resolves_known_id() -> None:
    catalog = ModelCatalog(available_providers={"openrouter"})
    info = catalog.get("openrouter:google/gemma-4-31b-it:free")
    assert info is not None
    assert info.provider == "openrouter"
    assert info.model == "google/gemma-4-31b-it:free"


def test_catalog_get_resolves_id_even_for_unavailable_provider() -> None:
    # ``get`` resolves from the full catalog regardless of availability so the
    # gateway can still describe / attempt a configured-but-unlisted model.
    catalog = ModelCatalog(available_providers={"openrouter"})
    info = catalog.get("opencode:mimo-v2.5-free")
    assert info is not None
    assert info.provider == "opencode"


def test_catalog_get_unknown_id_is_none() -> None:
    catalog = ModelCatalog(available_providers={"openrouter"})
    assert catalog.get("nope:does-not-exist") is None


def test_catalog_ids_are_only_available_providers() -> None:
    catalog = ModelCatalog(available_providers={"opencode"})
    ids = catalog.ids()
    assert all(i.startswith("opencode:") for i in ids)
    assert "opencode:mimo-v2.5-free" in ids
