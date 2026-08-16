"""A machine dialling in, being asked something, and answering.

The parts of this that matter are the refusals. A socket that runs jobs on
somebody's laptop is the one genuinely dangerous surface in the system, and the
tests below pin down that it stays shut: no token configured means no link, a
wrong token means no link, and a machine that is not there is refused
immediately rather than having work queued for it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from friday.config import get_settings
from friday.link.protocol import Hello, Job, JobKind, Result
from friday.link.registry import Link, LinkRegistry, registry_of


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("FRIDAY_LINK_TOKEN", "s3cret-token")
    monkeypatch.setenv("FRIDAY_AUTH_ENABLED", "false")
    get_settings.cache_clear()
    from friday.app import create_app  # noqa: PLC0415

    with TestClient(create_app()) as running:
        yield running
    get_settings.cache_clear()


def test_a_relay_with_the_token_is_registered(client: TestClient) -> None:
    with client.websocket_connect("/link/connect?token=s3cret-token") as socket:
        socket.send_text(
            Hello(machine="thinkpad", platform="Linux", commands=["uptime"])
            .model_dump_json()
        )
        registry = registry_of(client.app)
        for _ in range(50):
            if registry.machines():
                break
            socket.send_text(json.dumps({"beat": 1}))
        assert "thinkpad" in registry.machines()


def test_no_token_configured_means_the_door_does_not_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unset secret must refuse every relay, not accept all of them."""
    monkeypatch.delenv("FRIDAY_LINK_TOKEN", raising=False)
    monkeypatch.setenv("FRIDAY_AUTH_ENABLED", "false")
    get_settings.cache_clear()
    from friday.app import create_app  # noqa: PLC0415

    with TestClient(create_app()) as running, pytest.raises(Exception):  # noqa: B017, PT011
        with running.websocket_connect("/link/connect?token=anything") as socket:
            socket.send_text(Hello(machine="x").model_dump_json())
            socket.receive_text()
    get_settings.cache_clear()


def test_the_wrong_token_is_refused(client: TestClient) -> None:
    with pytest.raises(Exception):  # noqa: B017, PT011
        with client.websocket_connect("/link/connect?token=not-it") as socket:
            socket.send_text(Hello(machine="x").model_dump_json())
            socket.receive_text()


@pytest.mark.anyio
async def test_a_job_reaches_the_machine_and_the_answer_comes_back() -> None:
    """The registry matches replies by id, not by arrival order."""

    class Socket:
        def __init__(self) -> None:
            self.sent: list[Job] = []

        async def send_text(self, raw: str) -> None:
            self.sent.append(Job.model_validate_json(raw))

    socket = Socket()
    link = Link("thinkpad", socket, ["uptime"], "Linux")
    registry = LinkRegistry()
    registry.attach(link)

    async def answer_when_asked() -> None:
        for _ in range(100):
            if socket.sent:
                link.deliver(
                    Result(id=socket.sent[0].id, ok=True, data={"host": "thinkpad"})
                )
                return
            await asyncio.sleep(0.01)

    asyncio.get_running_loop().create_task(answer_when_asked())
    result = await registry.run(JobKind.SECURITY_SCAN, timeout=5.0)
    assert result.ok
    assert result.data["host"] == "thinkpad"


@pytest.mark.anyio
async def test_asking_a_machine_that_is_not_there_says_so_immediately() -> None:
    """Never queued. A job delivered in six hours answers a forgotten question."""
    registry = LinkRegistry()
    result = await registry.run(JobKind.SECURITY_SCAN, machine="thinkpad", timeout=1.0)
    assert result.ok is False
    assert "isn't linked" in result.error
    assert "relay" in result.error          # and says how to fix it


@pytest.mark.anyio
async def test_a_disconnect_mid_job_fails_the_job_rather_than_hanging() -> None:
    class Socket:
        async def send_text(self, raw: str) -> None:
            return None

    link = Link("thinkpad", Socket(), [], "")
    registry = LinkRegistry()
    registry.attach(link)

    async def yank() -> None:
        await asyncio.sleep(0.05)
        registry.detach("thinkpad", link)

    asyncio.get_running_loop().create_task(yank())
    result = await registry.run(JobKind.IDENTIFY, timeout=5.0)
    assert result.ok is False
    assert "disconnected" in result.error
