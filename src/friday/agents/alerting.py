"""The alerting agent: deduplicated, rate-limited owner notifications.

:class:`AlertingAgent` is the FRIDAY specialist for "alert / notify / escalate /
warn" turns (build-spec section 4 / 9.7). It implements the
:class:`~friday.agents.base.Agent` protocol (``name="alerting"``,
``allowed_tools={"notify"}``) and exists to do one thing well: turn an alert
request into a single ``notify`` send while collapsing noisy duplicates.

**Dedupe + rate-limit.** A flood of *identical* alerts inside
``settings.alert_rate_limit_seconds`` must not page the owner five times — they
collapse to exactly ONE send. The agent keeps an in-memory map from *alert
identity* (channel + target + subject + body) to the timestamp of the last send
of that identity. A new alert is dispatched only when its identity has never
been seen, or when the configured window has fully elapsed since its last send;
otherwise it is suppressed. A genuinely *distinct* alert is never deduped
against a different one, so it always gets its own send (subject to its own
window).

**Time is injected, never read from the wall clock.** "Now" comes from a
``clock`` callable handed in at construction. Tests drive that clock by hand, so
the windowing is fully deterministic and offline; production wiring passes
``time.monotonic``/``time.time``.

**The notify tool is side-effecting and non-idempotent**, so the registry
confirm-step (build-spec section 12) would otherwise block it. The alerting
agent is the explicit decision *to* notify, so once it has decided an alert is
not a duplicate it dispatches with ``confirmed=True`` — the dedupe/rate-limit
gate *is* the deliberation the confirm-step protects against.

This module imports no LLM SDK and touches no network — it depends only on the
:class:`~friday.tools.registry.ToolRegistry` abstraction — so it keeps
``friday.agents`` clean for the SDK-isolation guard (grep-enforced by
``tests/unit/test_architecture.py``).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from friday.agents.base import AgentResult
from friday.config import Settings, get_settings
from friday.core.state import GraphState
from friday.errors import PermissionError
from friday.providers.llm import ToolCall
from friday.tools.registry import ToolRegistry

logger = logging.getLogger("friday.agents.alerting")

# The single tool this agent is permitted to reach.
_ALLOWED_TOOLS: frozenset[str] = frozenset({"notify"})

# An alert's identity: the tuple that decides whether two alerts are "the same"
# for dedupe purposes. Two alerts collapse iff every field matches.
_AlertIdentity = tuple[str, str, str, str]

# A monotonic-ish clock: a zero-arg callable returning the current time in
# seconds as a float. Injected so tests are deterministic.
Clock = Callable[[], float]


class AlertingAgent:
    """Dispatch alerts through ``notify`` with dedupe + rate-limiting.

    Args:
        registry: The tool registry the agent dispatches ``notify`` through. The
            allow-list is enforced by the registry, so the agent can only reach
            the tools it declares.
        clock: A zero-arg callable returning "now" in seconds (a float). Injected
            so the rate-limit window is deterministic in tests; pass
            ``time.monotonic`` (or ``time.time``) in production.
        settings: Application settings supplying ``alert_rate_limit_seconds`` (the
            dedupe window) and ``alert_dedupe`` (the on/off flag). Defaults to the
            process-wide :func:`~friday.config.get_settings`.
    """

    name = "alerting"
    allowed_tools = _ALLOWED_TOOLS

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        clock: Clock,
        settings: Settings | None = None,
    ) -> None:
        self._registry = registry
        self._clock = clock
        self._settings: Settings = settings if settings is not None else get_settings()
        # identity -> timestamp of the last actually-dispatched send for it.
        self._last_sent: dict[_AlertIdentity, float] = {}

    # -- alert extraction --------------------------------------------------- #
    @staticmethod
    def _alert_from_state(state: GraphState) -> dict[str, str] | None:
        """Pull the alert the orchestrator staged in ``scratchpad['alert']``.

        Returns the alert as a ``str -> str`` mapping, or ``None`` if no alert
        was staged (a graceful no-send rather than a crash). Required fields are
        coerced to strings; missing optional fields default to empty strings and
        are validated downstream by the notify tool's ``args_model``.
        """
        raw = state.scratchpad.get("alert")
        if not isinstance(raw, dict):
            return None
        return {
            "channel": str(raw.get("channel", "")),
            "target": str(raw.get("target", "")),
            "subject": str(raw.get("subject", "")),
            "body": str(raw.get("body", "")),
        }

    @staticmethod
    def _identity(alert: dict[str, str]) -> _AlertIdentity:
        """The dedupe key: an alert is "the same" iff every field matches."""
        return (
            alert["channel"],
            alert["target"],
            alert["subject"],
            alert["body"],
        )

    # -- dedupe / rate-limit gate ------------------------------------------- #
    def _is_duplicate(self, identity: _AlertIdentity, now: float) -> bool:
        """Whether this identity was sent within the rate-limit window of ``now``.

        Returns ``False`` (i.e. allow the send) when dedupe is disabled, when the
        identity has never been sent, or when the window has fully elapsed since
        its last send. Otherwise the alert is a suppressible duplicate.
        """
        if not self._settings.alert_dedupe:
            return False
        last = self._last_sent.get(identity)
        if last is None:
            return False
        window = self._settings.alert_rate_limit_seconds
        return (now - last) < window

    def _evict_stale(self, now: float) -> None:
        """Drop dedupe entries whose window has fully elapsed.

        Once ``now - ts >= window`` an entry can no longer suppress anything, so
        keeping it only leaks memory. Sweeping here bounds ``_last_sent`` to roughly
        the identities seen within one window rather than growing without limit
        over a long-lived process handling many distinct alerts.
        """
        if not self._settings.alert_dedupe:
            return
        window = self._settings.alert_rate_limit_seconds
        if window <= 0:
            return
        stale = [key for key, ts in self._last_sent.items() if (now - ts) >= window]
        for key in stale:
            del self._last_sent[key]

    # -- dispatch ----------------------------------------------------------- #
    async def _dispatch(self, alert: dict[str, str]) -> ToolCall | None:
        """Send the alert via ``notify`` with the confirm-step satisfied.

        Returns the issued :class:`ToolCall` on a successful send, or ``None`` if
        the registry refused (permission/validation/handled tool failure). The
        agent owns the decision to notify, so it passes ``confirmed=True`` to
        clear the registry's side-effecting confirm-step.
        """
        raw_args: dict[str, object] = dict(alert)
        call = ToolCall(
            id=f"call_{uuid.uuid4().hex}", name="notify", arguments=raw_args
        )
        try:
            result = await self._registry.execute(
                "notify",
                raw_args,
                allowed_tools=self.allowed_tools,
                confirmed=True,
            )
        except PermissionError as exc:  # pragma: no cover - defensive
            logger.warning("alerting denied notify: %s", exc)
            return None

        if not result.ok:
            detail = result.error.message if result.error is not None else "unknown"
            logger.warning("alerting notify failed: %s", detail)
            return None
        return call

    # -- public entrypoint -------------------------------------------------- #
    async def run(self, state: GraphState) -> AgentResult:
        """Dispatch the staged alert unless it is a within-window duplicate.

        Returns an :class:`AgentResult` whose ``tool_calls_made`` carries the
        ``notify`` call when an alert was actually sent, and is empty when the
        alert was suppressed as a duplicate (or no alert was staged). ``output``
        is a short human summary of what happened.
        """
        alert = self._alert_from_state(state)
        if alert is None:
            return AgentResult(
                output="No alert was provided, so there is nothing to send.",
                tool_calls_made=[],
                confidence=1.0,
            )

        identity = self._identity(alert)
        now = self._clock()

        if self._is_duplicate(identity, now):
            logger.info(
                "alerting suppressed duplicate channel=%s target=%s subject=%r",
                alert["channel"],
                alert["target"],
                alert["subject"],
            )
            return AgentResult(
                output=(
                    f"Suppressed a duplicate alert {alert['subject']!r}: an "
                    "identical one was already sent inside the rate-limit window."
                ),
                tool_calls_made=[],
                confidence=1.0,
            )

        call = await self._dispatch(alert)
        if call is None:
            return AgentResult(
                output=(
                    f"Tried to send alert {alert['subject']!r} but the notify "
                    "tool rejected it."
                ),
                tool_calls_made=[],
                confidence=0.5,
            )

        # Record the send time so future identical alerts are deduped within the
        # window. Only a *successful* dispatch updates the clock. Evict elapsed
        # entries first so the dedupe map stays bounded.
        self._evict_stale(now)
        self._last_sent[identity] = now
        return AgentResult(
            output=(
                f"Sent alert {alert['subject']!r} via {alert['channel']} to "
                f"{alert['target']}."
            ),
            tool_calls_made=[call],
            confidence=1.0,
        )
