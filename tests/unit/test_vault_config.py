"""Unit tests for the vault settings block."""

from __future__ import annotations

from friday.config import Settings


def _settings(**kw: object) -> Settings:
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


def test_vault_is_off_by_default() -> None:
    assert _settings().enable_vault is False


def test_vault_defaults() -> None:
    s = _settings()
    assert s.cloudinary_cloud_name == ""
    assert s.vault_index_backend == "firestore"
    assert s.vault_sqlite_path == "data/vault.db"
    assert s.vault_quota_gb == 25.0
    assert s.vault_daily_solve_cap == 200
    assert s.vault_solver_operators == "VISION,ORACLE,GECKO"
    assert s.vault_shared_space_id == ""


def test_cloudinary_keys_default_to_none() -> None:
    s = _settings()
    assert s.cloudinary_api_key is None
    assert s.cloudinary_api_secret is None


def test_cloudinary_secret_is_not_in_repr() -> None:
    s = _settings(cloudinary_api_secret="topsecret")
    assert "topsecret" not in repr(s)
    assert s.cloudinary_api_secret is not None
    assert s.cloudinary_api_secret.get_secret_value() == "topsecret"


def test_vault_settings_from_friday_prefixed_env(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FRIDAY_ENABLE_VAULT", "true")
    monkeypatch.setenv("FRIDAY_VAULT_INDEX_BACKEND", "sqlite")
    monkeypatch.setenv("FRIDAY_VAULT_QUOTA_GB", "50")
    monkeypatch.setenv("FRIDAY_VAULT_DAILY_SOLVE_CAP", "500")
    s = _settings()
    assert s.enable_vault is True
    assert s.vault_index_backend == "sqlite"
    assert s.vault_quota_gb == 50.0
    assert s.vault_daily_solve_cap == 500
