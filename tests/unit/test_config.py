from pydantic import SecretStr

from friday.config import Settings, get_settings


def test_defaults_safe() -> None:
    s = Settings(_env_file=None)
    assert s.llm_provider == "fake"
    assert s.route_min_confidence == 0.55
    assert s.enable_voice is False
    assert s.owner_address == "Boss"


def test_secret_not_in_repr() -> None:
    s = Settings(_env_file=None, nvidia_api_key="nvapi-secret")
    assert "nvapi-secret" not in repr(s)
    assert "nvapi-secret" not in str(s)
    assert isinstance(s.nvidia_api_key, SecretStr)
    assert s.nvidia_api_key.get_secret_value() == "nvapi-secret"


def test_default_secret_is_none() -> None:
    s = Settings(_env_file=None)
    assert s.nvidia_api_key is None


def test_friday_prefixed_alias_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRIDAY_LLM_PROVIDER", "nvidia")
    monkeypatch.setenv("FRIDAY_ROUTE_MIN_CONFIDENCE", "0.8")
    monkeypatch.setenv("FRIDAY_ENABLE_VOICE", "true")
    s = Settings(_env_file=None)
    assert s.llm_provider == "nvidia"
    assert s.route_min_confidence == 0.8
    assert s.enable_voice is True


def test_nvidia_key_from_unprefixed_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-from-env")
    s = Settings(_env_file=None)
    assert isinstance(s.nvidia_api_key, SecretStr)
    assert s.nvidia_api_key.get_secret_value() == "nvapi-from-env"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_phase2_field_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.device_allowlist == []
    assert s.alert_rate_limit_seconds == 300.0
    assert s.alert_dedupe is True


def test_device_allowlist_comma_split_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRIDAY_DEVICE_ALLOWLIST", "light.kitchen, switch.fan ,plug.1")
    s = Settings(_env_file=None)
    assert s.device_allowlist == ["light.kitchen", "switch.fan", "plug.1"]


def test_device_allowlist_empty_env_is_empty_list(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRIDAY_DEVICE_ALLOWLIST", "")
    s = Settings(_env_file=None)
    assert s.device_allowlist == []


def test_alerting_fields_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRIDAY_ALERT_RATE_LIMIT_SECONDS", "60")
    monkeypatch.setenv("FRIDAY_ALERT_DEDUPE", "false")
    s = Settings(_env_file=None)
    assert s.alert_rate_limit_seconds == 60.0
    assert s.alert_dedupe is False


def test_gemini_fallback_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.llm_fallback_provider == "none"
    assert s.gemini_api_key is None
    assert s.gemini_base_url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert s.gemini_model == "gemini-2.0-flash"


def test_gemini_key_is_secret_and_not_in_repr() -> None:
    s = Settings(_env_file=None, gemini_api_key="gemini-secret")
    assert isinstance(s.gemini_api_key, SecretStr)
    assert "gemini-secret" not in repr(s)
    assert "gemini-secret" not in str(s)
    assert s.gemini_api_key.get_secret_value() == "gemini-secret"


def test_gemini_key_from_unprefixed_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-from-env")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    monkeypatch.setenv(
        "GEMINI_BASE_URL", "https://example.test/v1beta/openai/"
    )
    s = Settings(_env_file=None)
    assert isinstance(s.gemini_api_key, SecretStr)
    assert s.gemini_api_key.get_secret_value() == "gemini-from-env"
    assert s.gemini_model == "gemini-2.0-flash-exp"
    assert s.gemini_base_url == "https://example.test/v1beta/openai/"


def test_llm_fallback_provider_from_prefixed_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", "gemini")
    s = Settings(_env_file=None)
    assert s.llm_fallback_provider == "gemini"


# --------------------------------------------------------------------------- #
# OpenRouter / OpenCode + model catalog config
# --------------------------------------------------------------------------- #
def test_openrouter_opencode_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.openrouter_api_key is None
    assert s.opencode_api_key is None
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert s.opencode_base_url == "https://opencode.ai/zen/v1"


def test_openrouter_key_is_secret_and_from_unprefixed_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")
    s = Settings(_env_file=None)
    assert isinstance(s.openrouter_api_key, SecretStr)
    assert "sk-or-from-env" not in repr(s)
    assert "sk-or-from-env" not in str(s)
    assert s.openrouter_api_key.get_secret_value() == "sk-or-from-env"


def test_opencode_key_is_secret_and_from_unprefixed_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("OPENCODE_API_KEY", "oc-from-env")
    s = Settings(_env_file=None)
    assert isinstance(s.opencode_api_key, SecretStr)
    assert "oc-from-env" not in repr(s)
    assert s.opencode_api_key.get_secret_value() == "oc-from-env"


def test_default_model_id_default() -> None:
    s = Settings(_env_file=None)
    assert s.default_model_id == "openrouter:google/gemma-4-31b-it:free"


def test_compare_model_ids_default() -> None:
    s = Settings(_env_file=None)
    assert s.compare_model_ids == [
        "openrouter:openai/gpt-oss-20b:free",
        "openrouter:google/gemma-4-31b-it:free",
        "opencode:mimo-v2.5-free",
        "nvidia:meta/llama-3.1-8b-instruct",
    ]


def test_compare_model_ids_comma_split_from_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv(
        "FRIDAY_COMPARE_MODEL_IDS",
        "openrouter:a:free, opencode:b , nvidia:c",
    )
    s = Settings(_env_file=None)
    assert s.compare_model_ids == [
        "openrouter:a:free",
        "opencode:b",
        "nvidia:c",
    ]


def test_compare_model_ids_empty_env_is_empty_list(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRIDAY_COMPARE_MODEL_IDS", "")
    s = Settings(_env_file=None)
    assert s.compare_model_ids == []


def test_llm_provider_accepts_openrouter_opencode_gateway(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for value in ("openrouter", "opencode", "gateway"):
        monkeypatch.setenv("FRIDAY_LLM_PROVIDER", value)
        s = Settings(_env_file=None)
        assert s.llm_provider == value


def test_llm_fallback_provider_accepts_new_values(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for value in ("openrouter", "opencode", "gateway"):
        monkeypatch.setenv("FRIDAY_LLM_FALLBACK_PROVIDER", value)
        s = Settings(_env_file=None)
        assert s.llm_fallback_provider == value
