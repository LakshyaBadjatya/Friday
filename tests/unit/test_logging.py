import json

from friday.logging import bind_correlation_id, configure_logging, get_logger


def test_json_and_redaction(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(json_logs=True, level="INFO")
    bind_correlation_id("abc-123")
    get_logger("t").info("hello", extra={"nvidia_api_key": "nvapi-x", "n": 1})
    line = capsys.readouterr().err.strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["correlation_id"] == "abc-123"
    assert rec["nvidia_api_key"] == "***REDACTED***"
    assert rec["n"] == 1
    assert rec["message"] == "hello"
    assert rec["level"] == "INFO"


def test_redacts_varied_sensitive_keys(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(json_logs=True, level="INFO")
    bind_correlation_id("cid-2")
    get_logger("t").info(
        "redact",
        extra={
            "Authorization": "Bearer xyz",
            "user_token": "tok",
            "db_password": "pw",
            "client_secret": "sk",
            "plain": "visible",
        },
    )
    line = capsys.readouterr().err.strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["Authorization"] == "***REDACTED***"
    assert rec["user_token"] == "***REDACTED***"
    assert rec["db_password"] == "***REDACTED***"
    assert rec["client_secret"] == "***REDACTED***"
    assert rec["plain"] == "visible"


def test_correlation_id_default_when_unbound(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(json_logs=True, level="INFO")
    bind_correlation_id(None)
    get_logger("t").info("no-cid")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    rec = json.loads(line)
    assert rec["correlation_id"] is None


def test_plain_text_logging(capsys) -> None:  # type: ignore[no-untyped-def]
    configure_logging(json_logs=False, level="INFO")
    bind_correlation_id("plain-cid")
    get_logger("t").info("plain message")
    err = capsys.readouterr().err
    assert "plain message" in err
