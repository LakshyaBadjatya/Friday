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
