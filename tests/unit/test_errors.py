from friday.errors import (
    FridayError,
    PermissionError,
    ProviderError,
    RoutingError,
    ToolError,
)


def test_hierarchy() -> None:
    for cls in (ProviderError, ToolError, PermissionError, RoutingError):
        assert issubclass(cls, FridayError)


def test_carries_message() -> None:
    err = ProviderError("boom")
    assert "boom" in str(err)
